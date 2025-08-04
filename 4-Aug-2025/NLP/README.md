# Beyond Fixed: Variable-Length Denoising for Diffusion Large Language Models 

**Authors**: Jinsong Li, Xiaoyi Dong, Yuhang Zang, Yuhang Cao, Jiaqi Wang, Dahua Lin  

**Link**: [PDF](https://arxiv.org/pdf/2508.00819)  

**Abstract**: Diffusion Large Language Models (DLLMs) are emerging as a powerful alternative to the dominant Autoregressive Large Language Models, offering efficient parallel generation and capable global context modeling. However, the practical application of DLLMs is hindered by a critical architectural constraint: the need for a statically predefined generation length. This static length allocation leads to a problematic trade-off: insufficient lengths cripple performance on complex tasks, while excessive lengths incur significant computational overhead and sometimes result in performance degradation. While the inference framework is rigid, we observe that the model itself possesses internal signals that correlate with the optimal response length for a given task. To bridge this gap, we leverage these latent signals and introduce DAEDAL, a novel training-free denoising strategy that enables Dynamic Adaptive Length Expansion for Diffusion Large Language Models. DAEDAL operates in two phases: 1) Before the denoising process, DAEDAL starts from a short initial length and iteratively expands it to a coarse task-appropriate length, guided by a sequence completion metric. 2) During the denoising process, DAEDAL dynamically intervenes by pinpointing and expanding insufficient generation regions through mask token insertion, ensuring the final output is fully developed. Extensive experiments on DLLMs demonstrate that DAEDAL achieves performance comparable, and in some cases superior, to meticulously tuned fixed-length baselines, while simultaneously enhancing computational efficiency by achieving a higher effective token ratio. By resolving the static length constraint, DAEDAL unlocks new potential for DLLMs, bridging a critical gap with their Autoregressive counterparts and paving the way for more efficient and capable generation. 

---
# Do They Understand Them? An Updated Evaluation on Nonbinary Pronoun Handling in Large Language Models 

**Authors**: Xushuo Tang, Yi Ding, Zhengyi Yang, Yin Chen, Yongrui Gu, Wenke Yang, Mingchen Ju, Xin Cao, Yongfei Liu, Wenjie Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.00788)  

**Abstract**: Large language models (LLMs) are increasingly deployed in sensitive contexts where fairness and inclusivity are critical. Pronoun usage, especially concerning gender-neutral and neopronouns, remains a key challenge for responsible AI. Prior work, such as the MISGENDERED benchmark, revealed significant limitations in earlier LLMs' handling of inclusive pronouns, but was constrained to outdated models and limited evaluations. In this study, we introduce MISGENDERED+, an extended and updated benchmark for evaluating LLMs' pronoun fidelity. We benchmark five representative LLMs, GPT-4o, Claude 4, DeepSeek-V3, Qwen Turbo, and Qwen2.5, across zero-shot, few-shot, and gender identity inference. Our results show notable improvements compared with previous studies, especially in binary and gender-neutral pronoun accuracy. However, accuracy on neopronouns and reverse inference tasks remains inconsistent, underscoring persistent gaps in identity-sensitive reasoning. We discuss implications, model-specific observations, and avenues for future inclusive AI research. 

---
# ITUNLP at SemEval-2025 Task 8: Question-Answering over Tabular Data: A Zero-Shot Approach using LLM-Driven Code Generation 

**Authors**: Atakan Site, Emre Hakan Erdemir, Gülşen Eryiğit  

**Link**: [PDF](https://arxiv.org/pdf/2508.00762)  

**Abstract**: This paper presents our system for SemEval-2025 Task 8: DataBench, Question-Answering over Tabular Data. The primary objective of this task is to perform question answering on given tabular datasets from diverse domains under two subtasks: DataBench QA (Subtask I) and DataBench Lite QA (Subtask II). To tackle both subtasks, we developed a zero-shot solution with a particular emphasis on leveraging Large Language Model (LLM)-based code generation. Specifically, we propose a Python code generation framework utilizing state-of-the-art open-source LLMs to generate executable Pandas code via optimized prompting strategies. Our experiments reveal that different LLMs exhibit varying levels of effectiveness in Python code generation. Additionally, results show that Python code generation achieves superior performance in tabular question answering compared to alternative approaches. Although our ranking among zero-shot systems is unknown at the time of this paper's submission, our system achieved eighth place in Subtask I and sixth place in Subtask~II among the 30 systems that outperformed the baseline in the open-source models category. 

---
# MMBERT: Scaled Mixture-of-Experts Multimodal BERT for Robust Chinese Hate Speech Detection under Cloaking Perturbations 

**Authors**: Qiyao Xue, Yuchen Dou, Ryan Shi, Xiang Lorraine Li, Wei Gao  

**Link**: [PDF](https://arxiv.org/pdf/2508.00760)  

**Abstract**: Hate speech detection on Chinese social networks presents distinct challenges, particularly due to the widespread use of cloaking techniques designed to evade conventional text-based detection systems. Although large language models (LLMs) have recently improved hate speech detection capabilities, the majority of existing work has concentrated on English datasets, with limited attention given to multimodal strategies in the Chinese context. In this study, we propose MMBERT, a novel BERT-based multimodal framework that integrates textual, speech, and visual modalities through a Mixture-of-Experts (MoE) architecture. To address the instability associated with directly integrating MoE into BERT-based models, we develop a progressive three-stage training paradigm. MMBERT incorporates modality-specific experts, a shared self-attention mechanism, and a router-based expert allocation strategy to enhance robustness against adversarial perturbations. Empirical results in several Chinese hate speech datasets show that MMBERT significantly surpasses fine-tuned BERT-based encoder models, fine-tuned LLMs, and LLMs utilizing in-context learning approaches. 

---
# GLiDRE: Generalist Lightweight model for Document-level Relation Extraction 

**Authors**: Robin Armingaud, Romaric Besançon  

**Link**: [PDF](https://arxiv.org/pdf/2508.00757)  

**Abstract**: Relation Extraction (RE) is a fundamental task in Natural Language Processing, and its document-level variant poses significant challenges, due to the need to model complex interactions between entities across sentences. Current approaches, largely based on the ATLOP architecture, are commonly evaluated on benchmarks like DocRED and Re-DocRED. However, their performance in zero-shot or few-shot settings remains largely underexplored due to the task's complexity. Recently, the GLiNER model has shown that a compact NER model can outperform much larger Large Language Models. With a similar motivation, we introduce GLiDRE, a new model for document-level relation extraction that builds on the key ideas of GliNER. We benchmark GLiDRE against state-of-the-art models across various data settings on the Re-DocRED dataset. Our results demonstrate that GLiDRE achieves state-of-the-art performance in few-shot scenarios. Our code is publicly available. 

---
# Agentic large language models improve retrieval-based radiology question answering 

**Authors**: Sebastian Wind, Jeta Sopa, Daniel Truhn, Mahshad Lotfinia, Tri-Thien Nguyen, Keno Bressem, Lisa Adams, Mirabela Rusu, Harald Köstler, Gerhard Wellein, Andreas Maier, Soroosh Tayebi Arasteh  

**Link**: [PDF](https://arxiv.org/pdf/2508.00743)  

**Abstract**: Clinical decision-making in radiology increasingly benefits from artificial intelligence (AI), particularly through large language models (LLMs). However, traditional retrieval-augmented generation (RAG) systems for radiology question answering (QA) typically rely on single-step retrieval, limiting their ability to handle complex clinical reasoning tasks. Here we propose an agentic RAG framework enabling LLMs to autonomously decompose radiology questions, iteratively retrieve targeted clinical evidence from Radiopaedia, and dynamically synthesize evidence-based responses. We evaluated 24 LLMs spanning diverse architectures, parameter scales (0.5B to >670B), and training paradigms (general-purpose, reasoning-optimized, clinically fine-tuned), using 104 expert-curated radiology questions from previously established RSNA-RadioQA and ExtendedQA datasets. Agentic retrieval significantly improved mean diagnostic accuracy over zero-shot prompting (73% vs. 64%; P<0.001) and conventional online RAG (73% vs. 68%; P<0.001). The greatest gains occurred in mid-sized models (e.g., Mistral Large improved from 72% to 81%) and small-scale models (e.g., Qwen 2.5-7B improved from 55% to 71%), while very large models (>200B parameters) demonstrated minimal changes (<2% improvement). Additionally, agentic retrieval reduced hallucinations (mean 9.4%) and retrieved clinically relevant context in 46% of cases, substantially aiding factual grounding. Even clinically fine-tuned models exhibited meaningful improvements (e.g., MedGemma-27B improved from 71% to 81%), indicating complementary roles of retrieval and fine-tuning. These results highlight the potential of agentic frameworks to enhance factuality and diagnostic accuracy in radiology QA, particularly among mid-sized LLMs, warranting future studies to validate their clinical utility. 

---
# Applying Psychometrics to Large Language Model Simulated Populations: Recreating the HEXACO Personality Inventory Experiment with Generative Agents 

**Authors**: Sarah Mercer, Daniel P. Martin, Phil Swatton  

**Link**: [PDF](https://arxiv.org/pdf/2508.00742)  

**Abstract**: Generative agents powered by Large Language Models demonstrate human-like characteristics through sophisticated natural language interactions. Their ability to assume roles and personalities based on predefined character biographies has positioned them as cost-effective substitutes for human participants in social science research. This paper explores the validity of such persona-based agents in representing human populations; we recreate the HEXACO personality inventory experiment by surveying 310 GPT-4 powered agents, conducting factor analysis on their responses, and comparing these results to the original findings presented by Ashton, Lee, & Goldberg in 2004. Our results found 1) a coherent and reliable personality structure was recoverable from the agents' responses demonstrating partial alignment to the HEXACO framework. 2) the derived personality dimensions were consistent and reliable within GPT-4, when coupled with a sufficiently curated population, and 3) cross-model analysis revealed variability in personality profiling, suggesting model-specific biases and limitations. We discuss the practical considerations and challenges encountered during the experiment. This study contributes to the ongoing discourse on the potential benefits and limitations of using generative agents in social science research and provides useful guidance on designing consistent and representative agent personas to maximise coverage and representation of human personality traits. 

---
# Out-of-Context Abduction: LLMs Make Inferences About Procedural Data Leveraging Declarative Facts in Earlier Training Data 

**Authors**: Sohaib Imran, Rob Lamb, Peter M. Atkinson  

**Link**: [PDF](https://arxiv.org/pdf/2508.00741)  

**Abstract**: Large language models (LLMs) are trained on large corpora, yet it is unclear whether they can reason about the information present within their training data. We design experiments to study out-of-context abduction in LLMs, the ability to infer the most plausible explanations for observations using relevant facts present in training data. We train treatment LLMs on names and behavior descriptions of fictitious chatbots, but not on examples of dialogue with the chatbots. We find that OpenAI's GPT 4o LLM can correctly infer at least one chatbot's name after observing example responses characteristic of that chatbot. We also find that previously training GPT 4o on descriptions of a chatbot's behavior allows it to display behaviors more characteristic of the chatbot when iteratively trained to display such behaviors. Our results have implications for situational awareness in LLMs and, therefore, for AI safety. 

---
# Dynamically Adaptive Reasoning via LLM-Guided MCTS for Efficient and Context-Aware KGQA 

**Authors**: Yingxu Wang, Shiqi Fan, Mengzhu Wang, Siwei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.00719)  

**Abstract**: Knowledge Graph Question Answering (KGQA) aims to interpret natural language queries and perform structured reasoning over knowledge graphs by leveraging their relational and semantic structures to retrieve accurate answers. Recent KGQA methods primarily follow either retrieve-then-reason paradigm, relying on GNNs or heuristic rules for static paths extraction, or dynamic path generation strategies that use large language models (LLMs) with prompting to jointly perform retrieval and reasoning. However, the former suffers from limited adaptability due to static path extraction and lack of contextual refinement, while the latter incurs high computational costs and struggles with accurate path evaluation due to reliance on fixed scoring functions and extensive LLM calls. To address these issues, this paper proposes Dynamically Adaptive MCTS-based Reasoning (DAMR), a novel framework that integrates symbolic search with adaptive path evaluation for efficient and context-aware KGQA. DAMR employs a Monte Carlo Tree Search (MCTS) backbone guided by an LLM-based planner, which selects top-$k$ relevant relations at each step to reduce search space. To improve path evaluation accuracy, we introduce a lightweight Transformer-based scorer that performs context-aware plausibility estimation by jointly encoding the question and relation sequence through cross-attention, enabling the model to capture fine-grained semantic shifts during multi-hop reasoning. Furthermore, to alleviate the scarcity of high-quality supervision, DAMR incorporates a dynamic pseudo-path refinement mechanism that periodically generates training signals from partial paths explored during search, allowing the scorer to continuously adapt to the evolving distribution of reasoning trajectories. Extensive experiments on multiple KGQA benchmarks show that DAMR significantly outperforms state-of-the-art methods. 

---
# NyayaRAG: Realistic Legal Judgment Prediction with RAG under the Indian Common Law System 

**Authors**: Shubham Kumar Nigam, Balaramamahanthi Deepak Patnaik, Shivam Mishra, Ajay Varghese Thomas, Noel Shallum, Kripabandhu Ghosh, Arnab Bhattacharya  

**Link**: [PDF](https://arxiv.org/pdf/2508.00709)  

**Abstract**: Legal Judgment Prediction (LJP) has emerged as a key area in AI for law, aiming to automate judicial outcome forecasting and enhance interpretability in legal reasoning. While previous approaches in the Indian context have relied on internal case content such as facts, issues, and reasoning, they often overlook a core element of common law systems, which is reliance on statutory provisions and judicial precedents. In this work, we propose NyayaRAG, a Retrieval-Augmented Generation (RAG) framework that simulates realistic courtroom scenarios by providing models with factual case descriptions, relevant legal statutes, and semantically retrieved prior cases. NyayaRAG evaluates the effectiveness of these combined inputs in predicting court decisions and generating legal explanations using a domain-specific pipeline tailored to the Indian legal system. We assess performance across various input configurations using both standard lexical and semantic metrics as well as LLM-based evaluators such as G-Eval. Our results show that augmenting factual inputs with structured legal knowledge significantly improves both predictive accuracy and explanation quality. 

---
# Better Call Claude: Can LLMs Detect Changes of Writing Style? 

**Authors**: Johannes Römisch, Svetlana Gorovaia, Mariia Halchynska, Gleb Schmidt, Ivan P. Yamshchikov  

**Link**: [PDF](https://arxiv.org/pdf/2508.00680)  

**Abstract**: This article explores the zero-shot performance of state-of-the-art large language models (LLMs) on one of the most challenging tasks in authorship analysis: sentence-level style change detection. Benchmarking four LLMs on the official PAN~2024 and 2025 "Multi-Author Writing Style Analysis" datasets, we present several observations. First, state-of-the-art generative models are sensitive to variations in writing style - even at the granular level of individual sentences. Second, their accuracy establishes a challenging baseline for the task, outperforming suggested baselines of the PAN competition. Finally, we explore the influence of semantics on model predictions and present evidence suggesting that the latest generation of LLMs may be more sensitive to content-independent and purely stylistic signals than previously reported. 

---
# Segment First, Retrieve Better: Realistic Legal Search via Rhetorical Role-Based Queries 

**Authors**: Shubham Kumar Nigam, Tanmay Dubey, Noel Shallum, Arnab Bhattacharya  

**Link**: [PDF](https://arxiv.org/pdf/2508.00679)  

**Abstract**: Legal precedent retrieval is a cornerstone of the common law system, governed by the principle of stare decisis, which demands consistency in judicial decisions. However, the growing complexity and volume of legal documents challenge traditional retrieval methods. TraceRetriever mirrors real-world legal search by operating with limited case information, extracting only rhetorically significant segments instead of requiring complete documents. Our pipeline integrates BM25, Vector Database, and Cross-Encoder models, combining initial results through Reciprocal Rank Fusion before final re-ranking. Rhetorical annotations are generated using a Hierarchical BiLSTM CRF classifier trained on Indian judgments. Evaluated on IL-PCR and COLIEE 2025 datasets, TraceRetriever addresses growing document volume challenges while aligning with practical search constraints, reliable and scalable foundation for precedent retrieval enhancing legal research when only partial case knowledge is available. 

---
# Team "better_call_claude": Style Change Detection using a Sequential Sentence Pair Classifier 

**Authors**: Gleb Schmidt, Johannes Römisch, Mariia Halchynska, Svetlana Gorovaia, Ivan P. Yamshchikov  

**Link**: [PDF](https://arxiv.org/pdf/2508.00675)  

**Abstract**: Style change detection - identifying the points in a document where writing style shifts - remains one of the most important and challenging problems in computational authorship analysis. At PAN 2025, the shared task challenges participants to detect style switches at the most fine-grained level: individual sentences. The task spans three datasets, each designed with controlled and increasing thematic variety within documents. We propose to address this problem by modeling the content of each problem instance - that is, a series of sentences - as a whole, using a Sequential Sentence Pair Classifier (SSPC). The architecture leverages a pre-trained language model (PLM) to obtain representations of individual sentences, which are then fed into a bidirectional LSTM (BiLSTM) to contextualize them within the document. The BiLSTM-produced vectors of adjacent sentences are concatenated and passed to a multi-layer perceptron for prediction per adjacency. Building on the work of previous PAN participants classical text segmentation, the approach is relatively conservative and lightweight. Nevertheless, it proves effective in leveraging contextual information and addressing what is arguably the most challenging aspect of this year's shared task: the notorious problem of "stylistically shallow", short sentences that are prevalent in the proposed benchmark data. Evaluated on the official PAN-2025 test datasets, the model achieves strong macro-F1 scores of 0.923, 0.828, and 0.724 on the EASY, MEDIUM, and HARD data, respectively, outperforming not only the official random baselines but also a much more challenging one: claude-3.7-sonnet's zero-shot performance. 

---
# MELAC: Massive Evaluation of Large Language Models with Alignment of Culture in Persian Language 

**Authors**: Farhan Farsi, Farnaz Aghababaloo, Shahriar Shariati Motlagh, Parsa Ghofrani, MohammadAli SadraeiJavaheri, Shayan Bali, Amirhossein Shabani, Farbod Bijary, Ghazal Zamaninejad, AmirMohammad Salehoof, Saeedeh Momtazi  

**Link**: [PDF](https://arxiv.org/pdf/2508.00673)  

**Abstract**: As large language models (LLMs) become increasingly embedded in our daily lives, evaluating their quality and reliability across diverse contexts has become essential. While comprehensive benchmarks exist for assessing LLM performance in English, there remains a significant gap in evaluation resources for other languages. Moreover, because most LLMs are trained primarily on data rooted in European and American cultures, they often lack familiarity with non-Western cultural contexts. To address this limitation, our study focuses on the Persian language and Iranian culture. We introduce 19 new evaluation datasets specifically designed to assess LLMs on topics such as Iranian law, Persian grammar, Persian idioms, and university entrance exams. Using these datasets, we benchmarked 41 prominent LLMs, aiming to bridge the existing cultural and linguistic evaluation gap in the field. 

---
# Medical Reasoning in the Era of LLMs: A Systematic Review of Enhancement Techniques and Applications 

**Authors**: Wenxuan Wang, Zizhan Ma, Meidan Ding, Shiyi Zheng, Shengyuan Liu, Jie Liu, Jiaming Ji, Wenting Chen, Xiang Li, Linlin Shen, Yixuan Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2508.00669)  

**Abstract**: The proliferation of Large Language Models (LLMs) in medicine has enabled impressive capabilities, yet a critical gap remains in their ability to perform systematic, transparent, and verifiable reasoning, a cornerstone of clinical practice. This has catalyzed a shift from single-step answer generation to the development of LLMs explicitly designed for medical reasoning. This paper provides the first systematic review of this emerging field. We propose a taxonomy of reasoning enhancement techniques, categorized into training-time strategies (e.g., supervised fine-tuning, reinforcement learning) and test-time mechanisms (e.g., prompt engineering, multi-agent systems). We analyze how these techniques are applied across different data modalities (text, image, code) and in key clinical applications such as diagnosis, education, and treatment planning. Furthermore, we survey the evolution of evaluation benchmarks from simple accuracy metrics to sophisticated assessments of reasoning quality and visual interpretability. Based on an analysis of 60 seminal studies from 2022-2025, we conclude by identifying critical challenges, including the faithfulness-plausibility gap and the need for native multimodal reasoning, and outlining future directions toward building efficient, robust, and sociotechnically responsible medical AI. 

---
# DACTYL: Diverse Adversarial Corpus of Texts Yielded from Large Language Models 

**Authors**: Shantanu Thorat, Andrew Caines  

**Link**: [PDF](https://arxiv.org/pdf/2508.00619)  

**Abstract**: Existing AIG (AI-generated) text detectors struggle in real-world settings despite succeeding in internal testing, suggesting that they may not be robust enough. We rigorously examine the machine-learning procedure to build these detectors to address this. Most current AIG text detection datasets focus on zero-shot generations, but little work has been done on few-shot or one-shot generations, where LLMs are given human texts as an example. In response, we introduce the Diverse Adversarial Corpus of Texts Yielded from Language models (DACTYL), a challenging AIG text detection dataset focusing on one-shot/few-shot generations. We also include texts from domain-specific continued-pre-trained (CPT) language models, where we fully train all parameters using a memory-efficient optimization approach. Many existing AIG text detectors struggle significantly on our dataset, indicating a potential vulnerability to one-shot/few-shot and CPT-generated texts. We also train our own classifiers using two approaches: standard binary cross-entropy (BCE) optimization and a more recent approach, deep X-risk optimization (DXO). While BCE-trained classifiers marginally outperform DXO classifiers on the DACTYL test set, the latter excels on out-of-distribution (OOD) texts. In our mock deployment scenario in student essay detection with an OOD student essay dataset, the best DXO classifier outscored the best BCE-trained classifier by 50.56 macro-F1 score points at the lowest false positive rates for both. Our results indicate that DXO classifiers generalize better without overfitting to the test set. Our experiments highlight several areas of improvement for AIG text detectors. 

---
# Prompting Science Report 3: I'll pay you or I'll kill you -- but will you care? 

**Authors**: Lennart Meincke, Ethan Mollick, Lilach Mollick, Dan Shapiro  

**Link**: [PDF](https://arxiv.org/pdf/2508.00614)  

**Abstract**: This is the third in a series of short reports that seek to help business, education, and policy leaders understand the technical details of working with AI through rigorous testing. In this report, we investigate two commonly held prompting beliefs: a) offering to tip the AI model and b) threatening the AI model. Tipping was a commonly shared tactic for improving AI performance and threats have been endorsed by Google Founder Sergey Brin (All-In, May 2025, 8:20) who observed that 'models tend to do better if you threaten them,' a claim we subject to empirical testing here. We evaluate model performance on GPQA (Rein et al. 2024) and MMLU-Pro (Wang et al. 2024).
We demonstrate two things:
- Threatening or tipping a model generally has no significant effect on benchmark performance.
- Prompt variations can significantly affect performance on a per-question level. However, it is hard to know in advance whether a particular prompting approach will help or harm the LLM's ability to answer any particular question.
Taken together, this suggests that simple prompting variations might not be as effective as previously assumed, especially for difficult problems. However, as reported previously (Meincke et al. 2025a), prompting approaches can yield significantly different results for individual questions. 

---
# GHTM: A Graph based Hybrid Topic Modeling Approach in Low-Resource Bengali Language 

**Authors**: Farhana Haque, Md. Abdur Rahman, Sumon Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2508.00605)  

**Abstract**: Topic modeling is a Natural Language Processing (NLP) technique that is used to identify latent themes and extract topics from text corpora by grouping similar documents based on their most significant keywords. Although widely researched in English, topic modeling remains understudied in Bengali due to its morphological complexity, lack of adequate resources and initiatives. In this contribution, a novel Graph Convolutional Network (GCN) based model called GHTM (Graph-Based Hybrid Topic Model) is proposed. This model represents input vectors of documents as nodes in the graph, which GCN uses to produce semantically rich embeddings. The embeddings are then decomposed using Non-negative Matrix Factorization (NMF) to get the topical representations of the underlying themes of the text corpus. This study compares the proposed model against a wide range of Bengali topic modeling techniques, from traditional methods such as LDA, LSA, and NMF to contemporary frameworks such as BERTopic and Top2Vec on three Bengali datasets. The experimental results demonstrate the effectiveness of the proposed model by outperforming other models in topic coherence and diversity. In addition, we introduce a novel Bengali dataset called "NCTBText" sourced from Bengali textbook materials to enrich and diversify the predominantly newspaper-centric Bengali corpora. 

---
# A Context-Aware Dual-Metric Framework for Confidence Estimation in Large Language Models 

**Authors**: Mingruo Yuan, Shuyi Zhang, Ben Kao  

**Link**: [PDF](https://arxiv.org/pdf/2508.00600)  

**Abstract**: Accurate confidence estimation is essential for trustworthy large language models (LLMs) systems, as it empowers the user to determine when to trust outputs and enables reliable deployment in safety-critical applications. Current confidence estimation methods for LLMs neglect the relevance between responses and contextual information, a crucial factor in output quality evaluation, particularly in scenarios where background knowledge is provided. To bridge this gap, we propose CRUX (Context-aware entropy Reduction and Unified consistency eXamination), the first framework that integrates context faithfulness and consistency for confidence estimation via two novel metrics. First, contextual entropy reduction represents data uncertainty with the information gain through contrastive sampling with and without context. Second, unified consistency examination captures potential model uncertainty through the global consistency of the generated answers with and without context. Experiments across three benchmark datasets (CoQA, SQuAD, QuAC) and two domain-specific datasets (BioASQ, EduQG) demonstrate CRUX's effectiveness, achieving the highest AUROC than existing baselines. 

---
# SynAdapt: Learning Adaptive Reasoning in Large Language Models via Synthetic Continuous Chain-of-Thought 

**Authors**: Jianwei Wang, Ziming Wu, Fuming Lai, Shaobing Lian, Ziqian Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2508.00574)  

**Abstract**: While Chain-of-Thought (CoT) reasoning improves model performance, it incurs significant time costs due to the generation of discrete CoT tokens (DCoT). Continuous CoT (CCoT) offers a more efficient alternative, but existing CCoT methods are hampered by indirect fine-tuning, limited alignment, or inconsistent targets. To overcome these limitations, we propose \textit{SynAdapt}, an innovative efficient reasoning framework. Specifically, \textit{SynAdapt} generates the synthetic CCoT to serve as a precise and effective alignment target for LLMs. This synthetic CCoT explicitly guides the LLM to learn CCoT and derive accurate answers directly. Furthermore, relying solely on CCoT is insufficient for solving hard questions. To address this, \textit{SynAdapt} integrates a difficulty classifier that leverages both question context and CCoT to identify hard questions. CCoT can effectively help identify hard questions after some brief reasoning. We then adaptively prompt the LLM to re-think these hard questions for improved performance. Extensive experimental results across various benchmarks from different difficulty levels strongly demonstrate the effectiveness of our method, achieving the best accuracy-efficiency trade-off. 

---
# PaPaformer: Language Model from Pre-trained Paraller Paths 

**Authors**: Joonas Tapaninaho, Mourad Oussala  

**Link**: [PDF](https://arxiv.org/pdf/2508.00544)  

**Abstract**: The training of modern large-language models requires an increasingly amount of computation power and time. Even smaller variants, such as small-language models (SLMs), take several days to train in the best-case scenarios, often requiring multiple GPUs. This paper explores methods to train and evaluate decoder-only transformer-based language models in hours instead of days/weeks. We introduces \textit{PaPaformer}, a decoder-only transformer architecture variant, whose lower-dimensional parallel paths are combined into larger model. The paper shows that these lower-dimensional paths can be trained individually with different types of training data and then combined into one larger model. This method gives the option to reduce the total number of model parameters and the training time with increasing performance. Moreover, the use of parallel path structure opens interesting possibilities to customize paths to accommodate specific task requirements. 

---
# The Prosody of Emojis 

**Authors**: Giulio Zhou, Tsz Kin Lam, Alexandra Birch, Barry Haddow  

**Link**: [PDF](https://arxiv.org/pdf/2508.00537)  

**Abstract**: Prosodic features such as pitch, timing, and intonation are central to spoken communication, conveying emotion, intent, and discourse structure. In text-based settings, where these cues are absent, emojis act as visual surrogates that add affective and pragmatic nuance. This study examines how emojis influence prosodic realisation in speech and how listeners interpret prosodic cues to recover emoji meanings. Unlike previous work, we directly link prosody and emoji by analysing actual human speech data, collected through structured but open-ended production and perception tasks. This provides empirical evidence of how emoji semantics shape spoken delivery and perception. Results show that speakers adapt their prosody based on emoji cues, listeners can often identify the intended emoji from prosodic variation alone, and greater semantic differences between emojis correspond to increased prosodic divergence. These findings suggest that emojis can act as meaningful carriers of prosodic intent, offering insight into their communicative role in digitally mediated contexts. 

---
# EFlat-LoRA: Efficiently Seeking Flat Minima for Better Generalization in Fine-Tuning Large Language Models and Beyond 

**Authors**: Jiaxin Deng, Qingcheng Zhu, Junbiao Pang, Linlin Yang, Zhongqian Fu, Baochang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.00522)  

**Abstract**: Little research explores the correlation between the expressive ability and generalization ability of the low-rank adaptation (LoRA). Sharpness-Aware Minimization (SAM) improves model generalization for both Convolutional Neural Networks (CNNs) and Transformers by encouraging convergence to locally flat minima. However, the connection between sharpness and generalization has not been fully explored for LoRA due to the lack of tools to either empirically seek flat minima or develop theoretical methods. In this work, we propose Flat-LoRA and its efficient version i.e., EFlat-LoRA, to seek flat minima for LoRA. Concretely, we theoretically demonstrate that perturbations in the full parameter space can be transferred to the low-rank subspace. This approach eliminates the potential interference introduced by perturbations across multiple matrices in the low-rank subspace. Our extensive experiments on large language models and vision-language models demonstrate that EFlat-LoRA achieves optimize efficiency comparable to that of LoRA while simultaneously attaining comparable or even better performance. For example, on the GLUE dataset with RoBERTa-large, EFlat-LoRA outperforms LoRA and full fine-tuning by 1.0% and 0.5% on average, respectively. On vision-language models e.g., Qwen-VL-Chat shows performance improvements of 1.5% and 1.0% on SQA and VizWiz datasets, respectively. These empirical results also verify that the generalization of LoRA is closely related to sharpness, which is omitted by previous methods. 

---
# The Missing Parts: Augmenting Fact Verification with Half-Truth Detection 

**Authors**: Yixuan Tang, Jincheng Wang, Anthony K.H. Tung  

**Link**: [PDF](https://arxiv.org/pdf/2508.00489)  

**Abstract**: Fact verification systems typically assess whether a claim is supported by retrieved evidence, assuming that truthfulness depends solely on what is stated. However, many real-world claims are half-truths, factually correct yet misleading due to the omission of critical context. Existing models struggle with such cases, as they are not designed to reason about what is left unsaid. We introduce the task of half-truth detection, and propose PolitiFact-Hidden, a new benchmark with 15k political claims annotated with sentence-level evidence alignment and inferred claim intent. To address this challenge, we present TRACER, a modular re-assessment framework that identifies omission-based misinformation by aligning evidence, inferring implied intent, and estimating the causal impact of hidden content. TRACER can be integrated into existing fact-checking pipelines and consistently improves performance across multiple strong baselines. Notably, it boosts Half-True classification F1 by up to 16 points, highlighting the importance of modeling omissions for trustworthy fact verification. 

---
# GETALP@AutoMin 2025: Leveraging RAG to Answer Questions based on Meeting Transcripts 

**Authors**: Jeongwoo Kang, Markarit Vartampetian, Felix Herron, Yongxin Zhou, Diandra Fabre, Gabriela Gonzalez-Saez  

**Link**: [PDF](https://arxiv.org/pdf/2508.00476)  

**Abstract**: This paper documents GETALP's submission to the Third Run of the Automatic Minuting Shared Task at SIGDial 2025. We participated in Task B: question-answering based on meeting transcripts. Our method is based on a retrieval augmented generation (RAG) system and Abstract Meaning Representations (AMR). We propose three systems combining these two approaches. Our results show that incorporating AMR leads to high-quality responses for approximately 35% of the questions and provides notable improvements in answering questions that involve distinguishing between different participants (e.g., who questions). 

---
# Learning an Efficient Multi-Turn Dialogue Evaluator from Multiple Judges 

**Authors**: Yuqi Tang, Kehua Feng, Yunfeng Wang, Zhiwen Chen, Chengfei Lv, Gang Yu, Qiang Zhang, Keyan Ding  

**Link**: [PDF](https://arxiv.org/pdf/2508.00454)  

**Abstract**: Evaluating the conversational abilities of large language models (LLMs) remains a challenging task. Current mainstream approaches primarily rely on the ``LLM-as-a-judge" paradigm, where an LLM is prompted to serve as an evaluator to assess dialogue quality. However, such methods often suffer from various biases, which undermine the reliability and consistency of the evaluation results. To mitigate these biases, recent methods employ multiple LLMs as judges and aggregate their judgments to select the optimal assessment. Although effective, this multi-judge approach incurs significant computational overhead during inference. In this paper, we propose an efficient multi-turn dialogue evaluator that captures the collective wisdom of multiple LLM judges by aggregating their preference knowledge into a single model. Our approach preserves the advantages of diverse multi-judge feedback while drastically reducing the evaluation cost, enabling fast and flexible dialogue quality assessment. Extensive experiments on seven single rating and pairwise comparison dialogue evaluation benchmarks demonstrate that our method outperforms existing baselines across diverse scenarios, showcasing its efficiency and robustness. 

---
# ReaGAN: Node-as-Agent-Reasoning Graph Agentic Network 

**Authors**: Minghao Guo, Xi Zhu, Jingyuan Huang, Kai Mei, Yongfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.00429)  

**Abstract**: Graph Neural Networks (GNNs) have achieved remarkable success in graph-based learning by propagating information among neighbor nodes via predefined aggregation mechanisms. However, such fixed schemes often suffer from two key limitations. First, they cannot handle the imbalance in node informativeness -- some nodes are rich in information, while others remain sparse. Second, predefined message passing primarily leverages local structural similarity while ignoring global semantic relationships across the graph, limiting the model's ability to capture distant but relevant information. We propose Retrieval-augmented Graph Agentic Network (ReaGAN), an agent-based framework that empowers each node with autonomous, node-level decision-making. Each node acts as an agent that independently plans its next action based on its internal memory, enabling node-level planning and adaptive message propagation. Additionally, retrieval-augmented generation (RAG) allows nodes to access semantically relevant content and build global relationships in the graph. ReaGAN achieves competitive performance under few-shot in-context settings using a frozen LLM backbone without fine-tuning, showcasing the potential of agentic planning and local-global retrieval in graph learning. 

---
# Combining Discrete Wavelet and Cosine Transforms for Efficient Sentence Embedding 

**Authors**: Rana Salama, Abdou Youssef, Mona Diab  

**Link**: [PDF](https://arxiv.org/pdf/2508.00420)  

**Abstract**: Wavelets have emerged as a cutting edge technology in a number of fields. Concrete results of their application in Image and Signal processing suggest that wavelets can be effectively applied to Natural Language Processing (NLP) tasks that capture a variety of linguistic properties. In this paper, we leverage the power of applying Discrete Wavelet Transforms (DWT) to word and sentence embeddings. We first evaluate, intrinsically and extrinsically, how wavelets can effectively be used to consolidate important information in a word vector while reducing its dimensionality. We further combine DWT with Discrete Cosine Transform (DCT) to propose a non-parameterized model that compresses a sentence with a dense amount of information in a fixed size vector based on locally varying word features. We show the efficacy of the proposed paradigm on downstream applications models yielding comparable and even superior (in some tasks) results to original embeddings. 

---
# SA-GCS: Semantic-Aware Gaussian Curriculum Scheduling for UAV Vision-Language Navigation 

**Authors**: Hengxing Cai, Jinhan Dong, Yijie Rao, Jingcheng Deng, Jingjun Tan, Qien Chen, Haidong Wang, Zhen Wang, Shiyu Huang, Agachai Sumalee, Renxin Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2508.00390)  

**Abstract**: Unmanned Aerial Vehicle (UAV) Vision-Language Navigation (VLN) aims to enable agents to accurately localize targets and plan flight paths in complex environments based on natural language instructions, with broad applications in intelligent inspection, disaster rescue, and urban monitoring. Recent progress in Vision-Language Models (VLMs) has provided strong semantic understanding for this task, while reinforcement learning (RL) has emerged as a promising post-training strategy to further improve generalization. However, existing RL methods often suffer from inefficient use of training data, slow convergence, and insufficient consideration of the difficulty variation among training samples, which limits further performance improvement. To address these challenges, we propose \textbf{Semantic-Aware Gaussian Curriculum Scheduling (SA-GCS)}, a novel training framework that systematically integrates Curriculum Learning (CL) into RL. SA-GCS employs a Semantic-Aware Difficulty Estimator (SA-DE) to quantify the complexity of training samples and a Gaussian Curriculum Scheduler (GCS) to dynamically adjust the sampling distribution, enabling a smooth progression from easy to challenging tasks. This design significantly improves training efficiency, accelerates convergence, and enhances overall model performance. Extensive experiments on the CityNav benchmark demonstrate that SA-GCS consistently outperforms strong baselines across all metrics, achieves faster and more stable convergence, and generalizes well across models of different scales, highlighting its robustness and scalability. The implementation of our approach is publicly available. 

---
# Multi-Layer Attention is the Amplifier of Demonstration Effectiveness 

**Authors**: Dingzirui Wang, Xuangliang Zhang, Keyan Xu, Qingfu Zhu, Wanxiang Che, Yang Deng  

**Link**: [PDF](https://arxiv.org/pdf/2508.00385)  

**Abstract**: Numerous studies have investigated the underlying mechanisms of in-context learning (ICL) effectiveness to inspire the design of related methods. However, existing work predominantly assumes the effectiveness of the demonstrations provided within ICL, while many research indicates that not all demonstrations are effective, failing to yielding any performance improvement during ICL. Therefore, in this paper, we investigate the reasons behind demonstration ineffectiveness. Our analysis is based on gradient flow and linear self-attention models. By setting the gradient flow to zero, we deduce that a demonstration becomes ineffective if its information has either been learned by the model or is irrelevant to the user query. Furthermore, we demonstrate that in multi-layer models, the disparity in effectiveness among demonstrations is amplified with layer increasing, causing the model to focus more on effective ones. Considering that current demonstration selection methods primarily focus on the relevance to the user query while overlooking the information that the model has already assimilated, we propose a novel method called GradS, which leverages gradient flow for demonstration selection. We use the magnitude of the gradient flow of the demonstration with respect to a given user query as the criterion, thereby ensuring the effectiveness of the chosen ones. We validate our derivation and GradS on four prominent LLMs across five mainstream datasets. The experimental results confirm that the disparity in effectiveness among demonstrations is magnified as the model layer increases, substantiating our derivations. Moreover, GradS achieves a relative improvement of $6.8\%$ on average over the strongest baselines, demonstrating its effectiveness. 

---
# EdgeInfinite-Instruct: Bridging SFT-Based Optimization and NPU-Level Efficiency for Edge Devices 

**Authors**: Jiyu Chen, Poh Seng Lim, Shuang Peng, Daxiong Luo, JungHau Foo, Yap Deep, Timothy Lee Jun Jie, Kelvin Teh Kae Wen, Fan Yang, Danyu Feng, Hao-Yun Chen, Peng-Wen Chen, Fangyuan Li, Xiaoxin Chen, Wong Wai Mun  

**Link**: [PDF](https://arxiv.org/pdf/2508.00370)  

**Abstract**: Deploying Transformer-based large language models (LLMs) on resource-constrained edge devices for long-sequence tasks remains challenging due to the quadratic time complexity of self-attention and growing Key-Value (KV) cache demands. While existing KV cache optimizations improve memory efficiency, they often fail to reduce time to first token (TTFT) and may degrade performance through token pruning. Alternative sequence modeling architectures address some of these limitations, but typically require full retraining and lack infrastructure support. EdgeInfinite offers an efficient solution by fine-tuning only a small subset of parameters, maintaining quality while reducing both computational and memory costs, including improved TTFT. However, its instruction-following ability is limited, and it lacks mobile-specific optimizations. To address these issues, we propose EdgeInfinite-Instruct, which introduces a Segmented Supervised Fine-Tuning (S-SFT) strategy tailored to long-sequence tasks such as summarization and question answering. We further optimized EdgeInfinite-Instruct for efficient deployment on edge NPUs by employing fine-grained post-training quantization (PTQ) to reduce computational demands while maintaining accuracy, and by implementing a fixed-shape computation graph that balances memory usage and on-device efficiency through scenario-specific customization of input token and cache sizes. Experiments on long-context benchmarks and real-world mobile tasks show that our approach improves domain-specific performance while maintaining efficiency on NPU-accelerated edge devices. 

---
# Lucy: edgerunning agentic web search on mobile with machine generated task vectors 

**Authors**: Alan Dao, Dinh Bach Vu, Alex Nguyen, Norapat Buppodom  

**Link**: [PDF](https://arxiv.org/pdf/2508.00360)  

**Abstract**: Small language models (SLMs) are inherently limited in knowledge-intensive tasks due to their constrained capacity. While test-time computation offers a path to enhanced performance, most approaches treat reasoning as a fixed or heuristic process. In this work, we propose a new paradigm: viewing the model's internal reasoning, delimited by <think> and </think> tags, as a dynamic task vector machine. Rather than treating the content inside these tags as a mere trace of thought, we interpret the generation process itself as a mechanism through which the model \textbf{constructs and refines its own task vectors} on the fly. We developed a method to optimize this dynamic task vector machine through RLVR and successfully trained an agentic web-search model. We present Lucy, a 1.7B-parameter SLM that leverages this dynamic reasoning mechanism with MCP integration to achieve 78.3% accuracy on the SimpleQA benchmark, performing on par with much larger models such as DeepSeek-V3. This demonstrates that small models can rival large ones when equipped with structured, self-constructed task reasoning. 

---
# PilotRL: Training Language Model Agents via Global Planning-Guided Progressive Reinforcement Learning 

**Authors**: Keer Lu, Chong Chen, Bin Cui, Huang Leng, Wentao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.00344)  

**Abstract**: Large Language Models (LLMs) have shown remarkable advancements in tackling agent-oriented tasks. Despite their potential, existing work faces challenges when deploying LLMs in agent-based environments. The widely adopted agent paradigm ReAct centers on integrating single-step reasoning with immediate action execution, which limits its effectiveness in complex tasks requiring long-term strategic planning. Furthermore, the coordination between the planner and executor during problem-solving is also a critical factor to consider in agent design. Additionally, current approaches predominantly rely on supervised fine-tuning, which often leads models to memorize established task completion trajectories, thereby restricting their generalization ability when confronted with novel problem contexts. To address these challenges, we introduce an adaptive global plan-based agent paradigm AdaPlan, aiming to synergize high-level explicit guidance with execution to support effective long-horizon decision-making. Based on the proposed paradigm, we further put forward PilotRL, a global planning-guided training framework for LLM agents driven by progressive reinforcement learning. We first develop the model's ability to follow explicit guidance from global plans when addressing agent tasks. Subsequently, based on this foundation, we focus on optimizing the quality of generated plans. Finally, we conduct joint optimization of the model's planning and execution coordination. Experiments indicate that PilotRL could achieve state-of-the-art performances, with LLaMA3.1-8B-Instruct + PilotRL surpassing closed-sourced GPT-4o by 3.60%, while showing a more substantial gain of 55.78% comparing to GPT-4o-mini at a comparable parameter scale. 

---
# Improving Multimodal Contrastive Learning of Sentence Embeddings with Object-Phrase Alignment 

**Authors**: Kaiyan Zhao, Zhongtao Miao, Yoshimasa Tsuruoka  

**Link**: [PDF](https://arxiv.org/pdf/2508.00332)  

**Abstract**: Multimodal sentence embedding models typically leverage image-caption pairs in addition to textual data during training. However, such pairs often contain noise, including redundant or irrelevant information on either the image or caption side. To mitigate this issue, we propose MCSEO, a method that enhances multimodal sentence embeddings by incorporating fine-grained object-phrase alignment alongside traditional image-caption alignment. Specifically, MCSEO utilizes existing segmentation and object detection models to extract accurate object-phrase pairs, which are then used to optimize a contrastive learning objective tailored to object-phrase correspondence. Experimental results on semantic textual similarity (STS) tasks across different backbone models demonstrate that MCSEO consistently outperforms strong baselines, highlighting the significance of precise object-phrase alignment in multimodal representation learning. 

---
# Systematic Evaluation of Optimization Techniques for Long-Context Language Models 

**Authors**: Ammar Ahmed, Sheng Di, Franck Cappello, Zirui Liu, Jingoo Han, Ali Anwar  

**Link**: [PDF](https://arxiv.org/pdf/2508.00305)  

**Abstract**: Large language models (LLMs) excel across diverse natural language processing tasks but face resource demands and limited context windows. Although techniques like pruning, quantization, and token dropping can mitigate these issues, their efficacy in long-context scenarios and system evaluation remains underexplored. This paper systematically benchmarks these optimizations, characterizing memory usage, latency, and throughput, and studies how these methods impact the quality of text generation. We first analyze individual optimization methods for two LLM architectures supporting long context and then systematically evaluate combinations of these techniques to assess how this deeper analysis impacts performance metrics. We subsequently study the scalability of individual optimization methods on a larger variant with 70 billion-parameter model. Our novel insights reveal that naive combination inference optimization algorithms can adversely affect larger models due to compounded approximation errors, as compared to their smaller counterparts. Experiments show that relying solely on F1 obscures these effects by hiding precision-recall trade-offs in question answering tasks. By integrating system-level profiling with task-specific insights, this study helps LLM practitioners and researchers explore and balance efficiency, accuracy, and scalability across tasks and hardware configurations. 

---
# Integrating clinical reasoning into large language model-based diagnosis through etiology-aware attention steering 

**Authors**: Peixian Li, Yu Tian, Ruiqi Tu, Chengkai Wu, Jingjing Ren, Jingsong Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.00285)  

**Abstract**: Objective: Large Language Models (LLMs) demonstrate significant capabilities in medical text understanding and generation. However, their diagnostic reliability in complex clinical scenarios remains limited. This study aims to enhance LLMs' diagnostic accuracy and clinical reasoning ability. Method: We propose an Etiology-Aware Attention Steering Framework to integrate structured clinical reasoning into LLM-based diagnosis. Specifically, we first construct Clinical Reasoning Scaffolding (CRS) based on authoritative clinical guidelines for three representative acute abdominal emergencies: acute appendicitis, acute pancreatitis, and acute cholecystitis. Next, we develop the Etiology-Aware Head Identification algorithm to pinpoint attention heads crucial for the model's etiology reasoning. To ensure reliable clinical reasoning alignment, we introduce the Reasoning-Guided Parameter-Efficient Fine-tuning that embeds etiological reasoning cues into input representations and steers the selected Etiology-Aware Heads toward critical information through a Reasoning-Guided Loss function. Result: On the Consistent Diagnosis Cohort, our framework improves average diagnostic accuracy by 15.65% and boosts the average Reasoning Focus Score by 31.6% over baselines. External validation on the Discrepant Diagnosis Cohort further confirms its effectiveness in enhancing diagnostic accuracy. Further assessments via Reasoning Attention Frequency indicate that our models exhibit enhanced reliability when faced with real-world complex scenarios. Conclusion: This study presents a practical and effective approach to enhance clinical reasoning in LLM-based diagnosis. By aligning model attention with structured CRS, the proposed framework offers a promising paradigm for building more interpretable and reliable AI diagnostic systems in complex clinical settings. 

---
# Model Misalignment and Language Change: Traces of AI-Associated Language in Unscripted Spoken English 

**Authors**: Bryce Anderson, Riley Galpin, Tom S. Juzek  

**Link**: [PDF](https://arxiv.org/pdf/2508.00238)  

**Abstract**: In recent years, written language, particularly in science and education, has undergone remarkable shifts in word usage. These changes are widely attributed to the growing influence of Large Language Models (LLMs), which frequently rely on a distinct lexical style. Divergences between model output and target audience norms can be viewed as a form of misalignment. While these shifts are often linked to using Artificial Intelligence (AI) directly as a tool to generate text, it remains unclear whether the changes reflect broader changes in the human language system itself. To explore this question, we constructed a dataset of 22.1 million words from unscripted spoken language drawn from conversational science and technology podcasts. We analyzed lexical trends before and after ChatGPT's release in 2022, focusing on commonly LLM-associated words. Our results show a moderate yet significant increase in the usage of these words post-2022, suggesting a convergence between human word choices and LLM-associated patterns. In contrast, baseline synonym words exhibit no significant directional shift. Given the short time frame and the number of words affected, this may indicate the onset of a remarkable shift in language use. Whether this represents natural language change or a novel shift driven by AI exposure remains an open question. Similarly, although the shifts may stem from broader adoption patterns, it may also be that upstream training misalignments ultimately contribute to changes in human language use. These findings parallel ethical concerns that misaligned models may shape social and moral beliefs. 

---
# Semantic Compression for Word and Sentence Embeddings using Discrete Wavelet Transform 

**Authors**: Rana Aref Salama, Abdou Youssef, Mona Diab  

**Link**: [PDF](https://arxiv.org/pdf/2508.00220)  

**Abstract**: Wavelet transforms, a powerful mathematical tool, have been widely used in different domains, including Signal and Image processing, to unravel intricate patterns, enhance data representation, and extract meaningful features from data. Tangible results from their application suggest that Wavelet transforms can be applied to NLP capturing a variety of linguistic and semantic properties. In this paper, we empirically leverage the application of Discrete Wavelet Transforms (DWT) to word and sentence embeddings. We aim to showcase the capabilities of DWT in analyzing embedding representations at different levels of resolution and compressing them while maintaining their overall quality. We assess the effectiveness of DWT embeddings on semantic similarity tasks to show how DWT can be used to consolidate important semantic information in an embedding vector. We show the efficacy of the proposed paradigm using different embedding models, including large language models, on downstream tasks. Our results show that DWT can reduce the dimensionality of embeddings by 50-93% with almost no change in performance for semantic similarity tasks, while achieving superior accuracy in most downstream tasks. Our findings pave the way for applying DWT to improve NLP applications. 

---
# Tabular Data Understanding with LLMs: A Survey of Recent Advances and Challenges 

**Authors**: Xiaofeng Wu, Alan Ritter, Wei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2508.00217)  

**Abstract**: Tables have gained significant attention in large language models (LLMs) and multimodal large language models (MLLMs) due to their complex and flexible structure. Unlike linear text inputs, tables are two-dimensional, encompassing formats that range from well-structured database tables to complex, multi-layered spreadsheets, each with different purposes. This diversity in format and purpose has led to the development of specialized methods and tasks, instead of universal approaches, making navigation of table understanding tasks challenging. To address these challenges, this paper introduces key concepts through a taxonomy of tabular input representations and an introduction of table understanding tasks. We highlight several critical gaps in the field that indicate the need for further research: (1) the predominance of retrieval-focused tasks that require minimal reasoning beyond mathematical and logical operations; (2) significant challenges faced by models when processing complex table structures, large-scale tables, length context, or multi-table scenarios; and (3) the limited generalization of models across different tabular representations and formats. 

---
# Comparison of Large Language Models for Deployment Requirements 

**Authors**: Alper Yaman, Jannik Schwab, Christof Nitsche, Abhirup Sinha, Marco Huber  

**Link**: [PDF](https://arxiv.org/pdf/2508.00185)  

**Abstract**: Large Language Models (LLMs), such as Generative Pre-trained Transformers (GPTs) are revolutionizing the generation of human-like text, producing contextually relevant and syntactically correct content. Despite challenges like biases and hallucinations, these Artificial Intelligence (AI) models excel in tasks, such as content creation, translation, and code generation. Fine-tuning and novel architectures, such as Mixture of Experts (MoE), address these issues. Over the past two years, numerous open-source foundational and fine-tuned models have been introduced, complicating the selection of the optimal LLM for researchers and companies regarding licensing and hardware requirements. To navigate the rapidly evolving LLM landscape and facilitate LLM selection, we present a comparative list of foundational and domain-specific models, focusing on features, such as release year, licensing, and hardware requirements. This list is published on GitLab and will be continuously updated. 

---
# Is neural semantic parsing good at ellipsis resolution, or isn't it? 

**Authors**: Xiao Zhang, Johan bos  

**Link**: [PDF](https://arxiv.org/pdf/2508.00121)  

**Abstract**: Neural semantic parsers have shown good overall performance for a variety of linguistic phenomena, reaching semantic matching scores of more than 90%. But how do such parsers perform on strongly context-sensitive phenomena, where large pieces of semantic information need to be duplicated to form a meaningful semantic representation? A case in point is English verb phrase ellipsis, a construct where entire verb phrases can be abbreviated by a single auxiliary verb. Are the otherwise known as powerful semantic parsers able to deal with ellipsis or aren't they? We constructed a corpus of 120 cases of ellipsis with their fully resolved meaning representation and used this as a challenge set for a large battery of neural semantic parsers. Although these parsers performed very well on the standard test set, they failed in the instances with ellipsis. Data augmentation 

---
# FACTORY: A Challenging Human-Verified Prompt Set for Long-Form Factuality 

**Authors**: Mingda Chen, Yang Li, Xilun Chen, Adina Williams, Gargi Ghosh, Scott Yih  

**Link**: [PDF](https://arxiv.org/pdf/2508.00109)  

**Abstract**: Long-form factuality evaluation assesses the ability of models to generate accurate, comprehensive responses to short prompts. Existing benchmarks often lack human verification, leading to potential quality issues. To address this limitation, we introduce FACTORY, a large-scale, human-verified prompt set. Developed using a model-in-the-loop approach and refined by humans, FACTORY includes challenging prompts that are fact-seeking, answerable, and unambiguous. We conduct human evaluations on 6 state-of-the-art language models using FACTORY and existing datasets. Our results show that FACTORY is a challenging benchmark: approximately 40% of the claims made in the responses of SOTA models are not factual, compared to only 10% for other datasets. Our analysis identifies the strengths of FACTORY over prior benchmarks, emphasizing its reliability and the necessity for models to reason across long-tailed facts. 

---
# Semiotic Complexity and Its Epistemological Implications for Modeling Culture 

**Authors**: Zachary K. Stine, James E. Deitrick  

**Link**: [PDF](https://arxiv.org/pdf/2508.00095)  

**Abstract**: Greater theorizing of methods in the computational humanities is needed for epistemological and interpretive clarity, and therefore the maturation of the field. In this paper, we frame such modeling work as engaging in translation work from a cultural, linguistic domain into a computational, mathematical domain, and back again. Translators benefit from articulating the theory of their translation process, and so do computational humanists in their work -- to ensure internal consistency, avoid subtle yet consequential translation errors, and facilitate interpretive transparency. Our contribution in this paper is to lay out a particularly consequential dimension of the lack of theorizing and the sorts of translation errors that emerge in our modeling practices as a result. Along these lines we introduce the idea of semiotic complexity as the degree to which the meaning of some text may vary across interpretive lenses, and make the case that dominant modeling practices -- especially around evaluation -- commit a translation error by treating semiotically complex data as semiotically simple when it seems epistemologically convenient by conferring superficial clarity. We then lay out several recommendations for researchers to better account for these epistemological issues in their own work. 

---
# Do LLMs produce texts with "human-like" lexical diversity? 

**Authors**: Kelly Kendro, Jeffrey Maloney, Scott Jarvis  

**Link**: [PDF](https://arxiv.org/pdf/2508.00086)  

**Abstract**: The degree to which LLMs produce writing that is truly human-like remains unclear despite the extensive empirical attention that this question has received. The present study addresses this question from the perspective of lexical diversity. Specifically, the study investigates patterns of lexical diversity in LLM-generated texts from four ChatGPT models (-3.5, -4, -o4 mini, and -4.5) in comparison with texts written by L1 and L2 English participants (n = 240) across four education levels. Six dimensions of lexical diversity were measured in each text: volume, abundance, variety-repetition, evenness, disparity, and dispersion. Results from one-way MANOVAs, one-way ANOVAS, and Support Vector Machines revealed that the LLM-generated texts differed significantly from human-written texts for each variable, with ChatGPT-o4 mini and -4.5 differing the most. Within these two groups, ChatGPT-4.5 demonstrated higher levels of lexical diversity despite producing fewer tokens. The human writers' lexical diversity did not differ across subgroups (i.e., education, language status). Altogether, the results indicate that LLMs do not produce human-like texts in relation to lexical diversity, and the newer LLMs produce less human-like texts than older models. We discuss the implications of these results for language pedagogy and related applications. 

---
# PhysicsEval: Inference-Time Techniques to Improve the Reasoning Proficiency of Large Language Models on Physics Problems 

**Authors**: Oshayer Siddique, J. M Areeb Uzair Alam, Md Jobayer Rahman Rafy, Syed Rifat Raiyan, Hasan Mahmud, Md Kamrul Hasan  

**Link**: [PDF](https://arxiv.org/pdf/2508.00079)  

**Abstract**: The discipline of physics stands as a cornerstone of human intellect, driving the evolution of technology and deepening our understanding of the fundamental principles of the cosmos. Contemporary literature includes some works centered on the task of solving physics problems - a crucial domain of natural language reasoning. In this paper, we evaluate the performance of frontier LLMs in solving physics problems, both mathematical and descriptive. We also employ a plethora of inference-time techniques and agentic frameworks to improve the performance of the models. This includes the verification of proposed solutions in a cumulative fashion by other, smaller LLM agents, and we perform a comparative analysis of the performance that the techniques entail. There are significant improvements when the multi-agent framework is applied to problems that the models initially perform poorly on. Furthermore, we introduce a new evaluation benchmark for physics problems, ${\rm P{\small HYSICS}E{\small VAL}}$, consisting of 19,609 problems sourced from various physics textbooks and their corresponding correct solutions scraped from physics forums and educational websites. Our code and data are publicly available at this https URL. 

---
# Classification of Psychiatry Clinical Notes by Diagnosis: A Deep Learning and Machine Learning Approach 

**Authors**: Sergio Rubio-Martín, María Teresa García-Ordás, Antonio Serrano-García, Clara Margarita Franch-Pato, Arturo Crespo-Álvaro, José Alberto Benítez-Andrades  

**Link**: [PDF](https://arxiv.org/pdf/2508.00695)  

**Abstract**: The classification of clinical notes into specific diagnostic categories is critical in healthcare, especially for mental health conditions like Anxiety and Adjustment Disorder. In this study, we compare the performance of various Artificial Intelligence models, including both traditional Machine Learning approaches (Random Forest, Support Vector Machine, K-nearest neighbors, Decision Tree, and eXtreme Gradient Boost) and Deep Learning models (DistilBERT and SciBERT), to classify clinical notes into these two diagnoses. Additionally, we implemented three oversampling strategies: No Oversampling, Random Oversampling, and Synthetic Minority Oversampling Technique (SMOTE), to assess their impact on model performance. Hyperparameter tuning was also applied to optimize model accuracy. Our results indicate that oversampling techniques had minimal impact on model performance overall. The only exception was SMOTE, which showed a positive effect specifically with BERT-based models. However, hyperparameter optimization significantly improved accuracy across the models, enhancing their ability to generalize and perform on the dataset. The Decision Tree and eXtreme Gradient Boost models achieved the highest accuracy among machine learning approaches, both reaching 96%, while the DistilBERT and SciBERT models also attained 96% accuracy in the deep learning category. These findings underscore the importance of hyperparameter tuning in maximizing model performance. This study contributes to the ongoing research on AI-assisted diagnostic tools in mental health by providing insights into the efficacy of different model architectures and data balancing methods. 

---
# Demo: TOSense -- What Did You Just Agree to? 

**Authors**: Xinzhang Chen, Hassan Ali, Arash Shaghaghi, Salil S. Kanhere, Sanjay Jha  

**Link**: [PDF](https://arxiv.org/pdf/2508.00659)  

**Abstract**: Online services often require users to agree to lengthy and obscure Terms of Service (ToS), leading to information asymmetry and legal risks. This paper proposes TOSense-a Chrome extension that allows users to ask questions about ToS in natural language and get concise answers in real time. The system combines (i) a crawler "tos-crawl" that automatically extracts ToS content, and (ii) a lightweight large language model pipeline: MiniLM for semantic retrieval and BART-encoder for answer relevance verification. To avoid expensive manual annotation, we present a novel Question Answering Evaluation Pipeline (QEP) that generates synthetic questions and verifies the correctness of answers using clustered topic matching. Experiments on five major platforms, Apple, Google, X (formerly Twitter), Microsoft, and Netflix, show the effectiveness of TOSense (with up to 44.5% accuracy) across varying number of topic clusters. During the demonstration, we will showcase TOSense in action. Attendees will be able to experience seamless extraction, interactive question answering, and instant indexing of new sites. 

---
# Context-based Motion Retrieval using Open Vocabulary Methods for Autonomous Driving 

**Authors**: Stefan Englmeier, Max A. Büttner, Katharina Winter, Fabian B. Flohr  

**Link**: [PDF](https://arxiv.org/pdf/2508.00589)  

**Abstract**: Autonomous driving systems must operate reliably in safety-critical scenarios, particularly those involving unusual or complex behavior by Vulnerable Road Users (VRUs). Identifying these edge cases in driving datasets is essential for robust evaluation and generalization, but retrieving such rare human behavior scenarios within the long tail of large-scale datasets is challenging. To support targeted evaluation of autonomous driving systems in diverse, human-centered scenarios, we propose a novel context-aware motion retrieval framework. Our method combines Skinned Multi-Person Linear (SMPL)-based motion sequences and corresponding video frames before encoding them into a shared multimodal embedding space aligned with natural language. Our approach enables the scalable retrieval of human behavior and their context through text queries. This work also introduces our dataset WayMoCo, an extension of the Waymo Open Dataset. It contains automatically labeled motion and scene context descriptions derived from generated pseudo-ground-truth SMPL sequences and corresponding image data. Our approach outperforms state-of-the-art models by up to 27.5% accuracy in motion-context retrieval, when evaluated on the WayMoCo dataset. 

---
# Activation-Guided Local Editing for Jailbreaking Attacks 

**Authors**: Jiecong Wang, Haoran Li, Hao Peng, Ziqian Zeng, Zihao Wang, Haohua Du, Zhengtao Yu  

**Link**: [PDF](https://arxiv.org/pdf/2508.00555)  

**Abstract**: Jailbreaking is an essential adversarial technique for red-teaming these models to uncover and patch security flaws. However, existing jailbreak methods face significant drawbacks. Token-level jailbreak attacks often produce incoherent or unreadable inputs and exhibit poor transferability, while prompt-level attacks lack scalability and rely heavily on manual effort and human ingenuity. We propose a concise and effective two-stage framework that combines the advantages of these approaches. The first stage performs a scenario-based generation of context and rephrases the original malicious query to obscure its harmful intent. The second stage then utilizes information from the model's hidden states to guide fine-grained edits, effectively steering the model's internal representation of the input from a malicious toward a benign one. Extensive experiments demonstrate that this method achieves state-of-the-art Attack Success Rate, with gains of up to 37.74% over the strongest baseline, and exhibits excellent transferability to black-box models. Our analysis further demonstrates that AGILE maintains substantial effectiveness against prominent defense mechanisms, highlighting the limitations of current safeguards and providing valuable insights for future defense development. Our code is available at this https URL. 

---
# ContestTrade: A Multi-Agent Trading System Based on Internal Contest Mechanism 

**Authors**: Li Zhao, Rui Sun, Zuoyou Jiang, Bo Yang, Yuxiao Bai, Mengting Chen, Xinyang Wang, Jing Li, Zuo Bai  

**Link**: [PDF](https://arxiv.org/pdf/2508.00554)  

**Abstract**: In financial trading, large language model (LLM)-based agents demonstrate significant potential. However, the high sensitivity to market noise undermines the performance of LLM-based trading systems. To address this limitation, we propose a novel multi-agent system featuring an internal competitive mechanism inspired by modern corporate management structures. The system consists of two specialized teams: (1) Data Team - responsible for processing and condensing massive market data into diversified text factors, ensuring they fit the model's constrained context. (2) Research Team - tasked with making parallelized multipath trading decisions based on deep research methods. The core innovation lies in implementing a real-time evaluation and ranking mechanism within each team, driven by authentic market feedback. Each agent's performance undergoes continuous scoring and ranking, with only outputs from top-performing agents being adopted. The design enables the system to adaptively adjust to dynamic environment, enhances robustness against market noise and ultimately delivers superior trading performance. Experimental results demonstrate that our proposed system significantly outperforms prevailing multiagent systems and traditional quantitative investment methods across diverse evaluation metrics. 

---
# Towards a unified framework for programming paradigms: A systematic review of classification formalisms and methodological foundations 

**Authors**: Mikel Vandeloise  

**Link**: [PDF](https://arxiv.org/pdf/2508.00534)  

**Abstract**: The rise of multi-paradigm languages challenges traditional classification methods, leading to practical software engineering issues like interoperability defects. This systematic literature review (SLR) maps the formal foundations of programming paradigms. Our objective is twofold: (1) to assess the state of the art of classification formalisms and their limitations, and (2) to identify the conceptual primitives and mathematical frameworks for a more powerful, reconstructive approach.
Based on a synthesis of 74 primary studies, we find that existing taxonomies lack conceptual granularity, a unified formal basis, and struggle with hybrid languages. In response, our analysis reveals a strong convergence toward a compositional reconstruction of paradigms. This approach identifies a minimal set of orthogonal, atomic primitives and leverages mathematical frameworks, predominantly Type theory, Category theory and Unifying Theories of Programming (UTP), to formally guarantee their compositional properties.
We conclude that the literature reflects a significant intellectual shift away from classification towards these promising formal, reconstructive frameworks. This review provides a map of this evolution and proposes a research agenda for their unification. 

---
# Fine-grained Spatiotemporal Grounding on Egocentric Videos 

**Authors**: Shuo Liang, Yiwu Zhong, Zi-Yuan Hu, Yeyao Tao, Liwei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.00518)  

**Abstract**: Spatiotemporal video grounding aims to localize target entities in videos based on textual queries. While existing research has made significant progress in exocentric videos, the egocentric setting remains relatively underexplored, despite its growing importance in applications such as augmented reality and robotics. In this work, we conduct a systematic analysis of the discrepancies between egocentric and exocentric videos, revealing key challenges such as shorter object durations, sparser trajectories, smaller object sizes, and larger positional shifts. To address these challenges, we introduce EgoMask, the first pixel-level benchmark for fine-grained spatiotemporal grounding in egocentric videos. It is constructed by our proposed automatic annotation pipeline, which annotates referring expressions and object masks across short-, medium-, and long-term videos. Additionally, we create EgoMask-Train, a large-scale training dataset to facilitate model development. Experiments demonstrate that the state-of-the-art spatiotemporal grounding models perform poorly on our benchmark EgoMask, but fine-tuning on EgoMask-Train yields significant improvements, while preserving performance on exocentric datasets. Our work thus provides essential resources and insights for advancing egocentric video understanding. Our code is available at this https URL . 

---
# Cognitive Kernel-Pro: A Framework for Deep Research Agents and Agent Foundation Models Training 

**Authors**: Tianqing Fang, Zhisong Zhang, Xiaoyang Wang, Rui Wang, Can Qin, Yuxuan Wan, Jun-Yu Ma, Ce Zhang, Jiaqi Chen, Xiyun Li, Hongming Zhang, Haitao Mi, Dong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2508.00414)  

**Abstract**: General AI Agents are increasingly recognized as foundational frameworks for the next generation of artificial intelligence, enabling complex reasoning, web interaction, coding, and autonomous research capabilities. However, current agent systems are either closed-source or heavily reliant on a variety of paid APIs and proprietary tools, limiting accessibility and reproducibility for the research community. In this work, we present \textbf{Cognitive Kernel-Pro}, a fully open-source and (to the maximum extent) free multi-module agent framework designed to democratize the development and evaluation of advanced AI agents. Within Cognitive Kernel-Pro, we systematically investigate the curation of high-quality training data for Agent Foundation Models, focusing on the construction of queries, trajectories, and verifiable answers across four key domains: web, file, code, and general reasoning. Furthermore, we explore novel strategies for agent test-time reflection and voting to enhance agent robustness and performance. We evaluate Cognitive Kernel-Pro on GAIA, achieving state-of-the-art results among open-source and free agents. Notably, our 8B-parameter open-source model surpasses previous leading systems such as WebDancer and WebSailor, establishing a new performance standard for accessible, high-capability AI agents. Code is available at this https URL 

---
# Benchmarking LLMs for Unit Test Generation from Real-World Functions 

**Authors**: Dong Huang, Jie M. Zhang, Mark Harman, Qianru Zhang, Mingzhe Du, See-Kiong Ng  

**Link**: [PDF](https://arxiv.org/pdf/2508.00408)  

**Abstract**: Recently, large language models (LLMs) have shown great promise in automating unit test generation, significantly reducing the manual effort required by developers. To effectively evaluate the capabilities of LLMs in this domain, it is crucial to have a well-designed benchmark that accurately reflects real-world scenarios and mitigates common pitfalls. Existing LLM test generation benchmarks are limited by two critical drawbacks: data contamination and structurally simple function code. As a result, we often cannot rely on the validity of scientific conclusions drawn from empirical studies using these limited benchmarks. The empirical evidence presented may be biased due to contamination and may fail to generalize beyond toy programs due to structural simplicity.
To address these problems, we introduce ULT (UnLeakedTestbench), a new benchmark specifically designed for function-level unit test generation from real-world Python functions. ULT is constructed through a multi-stage curation process that ensures high cyclomatic complexity and mitigates test case contamination. With 3,909 carefully selected function-level tasks, ULT provides a more realistic and challenging evaluation of LLMs' test generation capabilities. We also provide PLT (PreLeakedTestbench), a pair benchmark of ULT with leaked tests designed to enable a controlled analysis of memorization versus reasoning in test generation. Our evaluation results demonstrate that ULT is significantly more challenging. For example, test cases generated by LLMs only achieve 41.32\%, 45.10\%, 30.22\%, and 40.21\% for accuracy, statement coverage, branch coverage, and mutation score on average for all LLMs, respectively. These results are substantially lower than the corresponding metrics on TestEval (91.79\%, 92.18\%, 82.04\%, and 49.69\%) and PLT (47.07\%, 55.13\%, 40.07\%, and 50.80\%). 

---
# R1-ACT: Efficient Reasoning Model Safety Alignment by Activating Safety Knowledge 

**Authors**: Yeonjun In, Wonjoong Kim, Sangwu Park, Chanyoung Park  

**Link**: [PDF](https://arxiv.org/pdf/2508.00324)  

**Abstract**: Although large reasoning models (LRMs) have demonstrated impressive capabilities on complex tasks, recent studies reveal that these models frequently fulfill harmful user instructions, raising significant safety concerns. In this paper, we investigate the underlying cause of LRM safety risks and find that models already possess sufficient safety knowledge but fail to activate it during reasoning. Based on this insight, we propose R1-Act, a simple and efficient post-training method that explicitly triggers safety knowledge through a structured reasoning process. R1-Act achieves strong safety improvements while preserving reasoning performance, outperforming prior alignment methods. Notably, it requires only 1,000 training examples and 90 minutes of training on a single RTX A6000 GPU. Extensive experiments across multiple LRM backbones and sizes demonstrate the robustness, scalability, and practical efficiency of our approach. 

---
# Mind the Gap: The Divergence Between Human and LLM-Generated Tasks 

**Authors**: Yi-Long Lu, Jiajun Song, Chunhui Zhang, Wei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.00282)  

**Abstract**: Humans constantly generate a diverse range of tasks guided by internal motivations. While generative agents powered by large language models (LLMs) aim to simulate this complex behavior, it remains uncertain whether they operate on similar cognitive principles. To address this, we conducted a task-generation experiment comparing human responses with those of an LLM agent (GPT-4o). We find that human task generation is consistently influenced by psychological drivers, including personal values (e.g., Openness to Change) and cognitive style. Even when these psychological drivers are explicitly provided to the LLM, it fails to reflect the corresponding behavioral patterns. They produce tasks that are markedly less social, less physical, and thematically biased toward abstraction. Interestingly, while the LLM's tasks were perceived as more fun and novel, this highlights a disconnect between its linguistic proficiency and its capacity to generate human-like, embodied this http URL conclude that there is a core gap between the value-driven, embodied nature of human cognition and the statistical patterns of LLMs, highlighting the necessity of incorporating intrinsic motivation and physical grounding into the design of more human-aligned agents. 

---
# MetaAgent: Toward Self-Evolving Agent via Tool Meta-Learning 

**Authors**: Hongjin Qian, Zheng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.00271)  

**Abstract**: In this work, we propose MetaAgent, an agentic paradigm inspired by the principle of learning-by-doing, where expertise is developed through hands-on practice and continual self-improvement. MetaAgent starts with a minimal workflow, equipped only with basic reasoning and adaptive help-seeking abilities. When a knowledge gap is encountered, MetaAgent generates natural language help requests, which are routed to the most suitable external tool by a dedicated tool router. As MetaAgent solves tasks, it continually conducts self-reflection and answer verification, distilling actionable experience into concise texts that are dynamically incorporated into future task contexts. Besides, MetaAgent autonomously builds in-house tools and a persistent knowledge base by organizing its tool-use history, further enhancing its ability to retrieve and integrate relevant information We term this continual, data-driven process as \textit{meta tool learning}, through which MetaAgent incrementally refines its reasoning and tool-use strategies, without changing model parameters or requiring further post-training. Evaluated on challenging knowledge discovery benchmarks, including GAIA, WebWalkerQA, and BrowseCamp, MetaAgent consistently outperforms workflow-based baselines and matches or exceeds end-to-end trained agents, demonstrating the promise of self-evolving agentic systems for robust, general-purpose knowledge discovery. We provide our source codes in this https URL. 

---
# Towards Higher Effective Rank in Parameter-efficient Fine-tuning using Khatri--Rao Product 

**Authors**: Paul Albert, Frederic Z. Zhang, Hemanth Saratchandran, Anton van den Hengel, Ehsan Abbasnejad  

**Link**: [PDF](https://arxiv.org/pdf/2508.00230)  

**Abstract**: Parameter-efficient fine-tuning (PEFT) has become a standard approach for adapting large pre-trained models. Amongst PEFT methods, low-rank adaptation (LoRA) has achieved notable success. However, recent studies have highlighted its limitations compared against full-rank alternatives, particularly when applied to multimodal and large language models. In this work, we present a quantitative comparison amongst full-rank and low-rank PEFT methods using a synthetic matrix approximation benchmark with controlled spectral properties. Our results confirm that LoRA struggles to approximate matrices with relatively flat spectrums or high frequency components -- signs of high effective ranks. To this end, we introduce KRAdapter, a novel PEFT algorithm that leverages the Khatri-Rao product to produce weight updates, which, by construction, tends to produce matrix product with a high effective rank. We demonstrate performance gains with KRAdapter on vision-language models up to 1B parameters and on large language models up to 8B parameters, particularly on unseen common-sense reasoning tasks. In addition, KRAdapter maintains the memory and compute efficiency of LoRA, making it a practical and robust alternative to fine-tune billion-scale parameter models. 

---
# RL-PLUS: Countering Capability Boundary Collapse of LLMs in Reinforcement Learning with Hybrid-policy Optimization 

**Authors**: Yihong Dong, Xue Jiang, Yongding Tao, Huanyu Liu, Kechi Zhang, Lili Mou, Rongyu Cao, Yingwei Ma, Jue Chen, Binhua Li, Zhi Jin, Fei Huang, Yongbin Li, Ge Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.00222)  

**Abstract**: Reinforcement Learning with Verifiable Reward (RLVR) has significantly advanced the complex reasoning abilities of Large Language Models (LLMs). However, it struggles to break through the inherent capability boundaries of the base LLM, due to its inherently on-policy strategy with LLM's immense action space and sparse reward. Further, RLVR can lead to the capability boundary collapse, narrowing the LLM's problem-solving scope. To address this problem, we propose RL-PLUS, a novel approach that synergizes internal exploitation (i.e., Thinking) with external data (i.e., Learning) to achieve stronger reasoning capabilities and surpass the boundaries of base models. RL-PLUS integrates two core components: Multiple Importance Sampling to address for distributional mismatch from external data, and an Exploration-Based Advantage Function to guide the model towards high-value, unexplored reasoning paths. We provide both theoretical analysis and extensive experiments to demonstrate the superiority and generalizability of our approach. The results show that RL-PLUS achieves state-of-the-art performance compared with existing RLVR methods on six math reasoning benchmarks and exhibits superior performance on six out-of-distribution reasoning tasks. It also achieves consistent and significant gains across diverse model families, with average relative improvements ranging from 21.1\% to 69.2\%. Moreover, Pass@k curves across multiple benchmarks indicate that RL-PLUS effectively resolves the capability boundary collapse problem. 

---
# On the Risk of Misleading Reports: Diagnosing Textual Biases in Multimodal Clinical AI 

**Authors**: David Restrepo, Ira Ktena, Maria Vakalopoulou, Stergios Christodoulidis, Enzo Ferrante  

**Link**: [PDF](https://arxiv.org/pdf/2508.00171)  

**Abstract**: Clinical decision-making relies on the integrated analysis of medical images and the associated clinical reports. While Vision-Language Models (VLMs) can offer a unified framework for such tasks, they can exhibit strong biases toward one modality, frequently overlooking critical visual cues in favor of textual information. In this work, we introduce Selective Modality Shifting (SMS), a perturbation-based approach to quantify a model's reliance on each modality in binary classification tasks. By systematically swapping images or text between samples with opposing labels, we expose modality-specific biases. We assess six open-source VLMs-four generalist models and two fine-tuned for medical data-on two medical imaging datasets with distinct modalities: MIMIC-CXR (chest X-ray) and FairVLMed (scanning laser ophthalmoscopy). By assessing model performance and the calibration of every model in both unperturbed and perturbed settings, we reveal a marked dependency on text input, which persists despite the presence of complementary visual information. We also perform a qualitative attention-based analysis which further confirms that image content is often overshadowed by text details. Our findings highlight the importance of designing and evaluating multimodal medical models that genuinely integrate visual and textual cues, rather than relying on single-modality signals. 

---
# Watch the Weights: Unsupervised monitoring and control of fine-tuned LLMs 

**Authors**: Ziqian Zhong, Aditi Raghunathan  

**Link**: [PDF](https://arxiv.org/pdf/2508.00161)  

**Abstract**: The releases of powerful open-weight large language models (LLMs) are often not accompanied by access to their full training data. Existing interpretability methods, particularly those based on activations, often require or assume distributionally similar data. This is a significant limitation when detecting and defending against novel potential threats like backdoors, which are by definition out-of-distribution.
In this work, we introduce a new method for understanding, monitoring and controlling fine-tuned LLMs that interprets weights, rather than activations, thereby side stepping the need for data that is distributionally similar to the unknown training data. We demonstrate that the top singular vectors of the weight difference between a fine-tuned model and its base model correspond to newly acquired behaviors. By monitoring the cosine similarity of activations along these directions, we can detect salient behaviors introduced during fine-tuning with high precision.
For backdoored models that bypasses safety mechanisms when a secret trigger is present, our method stops up to 100% of attacks with a false positive rate below 1.2%. For models that have undergone unlearning, we detect inference on erased topics with accuracy up to 95.42% and can even steer the model to recover "unlearned" information. Besides monitoring, our method also shows potential for pre-deployment model auditing: by analyzing commercial instruction-tuned models (OLMo, Llama, Qwen), we are able to uncover model-specific fine-tuning focus including marketing strategies and Midjourney prompt generation.
Our implementation can be found at this https URL. 

---
# A Survey on Code Generation with LLM-based Agents 

**Authors**: Yihong Dong, Xue Jiang, Jiaru Qian, Tian Wang, Kechi Zhang, Zhi Jin, Ge Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.00083)  

**Abstract**: Code generation agents powered by large language models (LLMs) are revolutionizing the software development paradigm. Distinct from previous code generation techniques, code generation agents are characterized by three core features. 1) Autonomy: the ability to independently manage the entire workflow, from task decomposition to coding and debugging. 2) Expanded task scope: capabilities that extend beyond generating code snippets to encompass the full software development lifecycle (SDLC). 3) Enhancement of engineering practicality: a shift in research emphasis from algorithmic innovation toward practical engineering challenges, such as system reliability, process management, and tool integration. This domain has recently witnessed rapid development and an explosion in research, demonstrating significant application potential. This paper presents a systematic survey of the field of LLM-based code generation agents. We trace the technology's developmental trajectory from its inception and systematically categorize its core techniques, including both single-agent and multi-agent architectures. Furthermore, this survey details the applications of LLM-based agents across the full SDLC, summarizes mainstream evaluation benchmarks and metrics, and catalogs representative tools. Finally, by analyzing the primary challenges, we identify and propose several foundational, long-term research directions for the future work of the field. 

---
# GPT-4.1 Sets the Standard in Automated Experiment Design Using Novel Python Libraries 

**Authors**: Nuno Fachada, Daniel Fernandes, Carlos M. Fernandes, Bruno D. Ferreira-Saraiva, João P. Matos-Carvalho  

**Link**: [PDF](https://arxiv.org/pdf/2508.00033)  

**Abstract**: Large Language Models (LLMs) have advanced rapidly as tools for automating code generation in scientific research, yet their ability to interpret and use unfamiliar Python APIs for complex computational experiments remains poorly characterized. This study systematically benchmarks a selection of state-of-the-art LLMs in generating functional Python code for two increasingly challenging scenarios: conversational data analysis with the \textit{ParShift} library, and synthetic data generation and clustering using \textit{pyclugen} and \textit{scikit-learn}. Both experiments use structured, zero-shot prompts specifying detailed requirements but omitting in-context examples. Model outputs are evaluated quantitatively for functional correctness and prompt compliance over multiple runs, and qualitatively by analyzing the errors produced when code execution fails. Results show that only a small subset of models consistently generate correct, executable code, with GPT-4.1 standing out as the only model to always succeed in both tasks. In addition to benchmarking LLM performance, this approach helps identify shortcomings in third-party libraries, such as unclear documentation or obscure implementation bugs. Overall, these findings highlight current limitations of LLMs for end-to-end scientific automation and emphasize the need for careful prompt design, comprehensive library documentation, and continued advances in language model capabilities. 

---
# Scalable Spectrum Availability Prediction using a Markov Chain Framework and ITU-R Propagation Models 

**Authors**: Abir Ray  

**Link**: [PDF](https://arxiv.org/pdf/2508.00028)  

**Abstract**: Spectrum resources are often underutilized across time and space, motivating dynamic spectrum access strategies that allow secondary users to exploit unused frequencies. A key challenge is predicting when and where spectrum will be available (i.e., unused by primary licensed users) in order to enable proactive and interference-free access. This paper proposes a scalable framework for spectrum availability prediction that combines a two-state Markov chain model of primary user activity with high-fidelity propagation models from the ITU-R (specifically Recommendations P.528 and P.2108). The Markov chain captures temporal occupancy patterns, while the propagation models incorporate path loss and clutter effects to determine if primary signals exceed interference thresholds at secondary user locations. By integrating these components, the proposed method can predict spectrum opportunities both in time and space with improved accuracy. We develop the system model and algorithm for the approach, analyze its scalability and computational efficiency, and discuss assumptions, limitations, and potential applications. The framework is flexible and can be adapted to various frequency bands and scenarios. The results and analysis show that the proposed approach can effectively identify available spectrum with low computational cost, making it suitable for real-time spectrum management in cognitive radio networks and other dynamic spectrum sharing systems. 

---
