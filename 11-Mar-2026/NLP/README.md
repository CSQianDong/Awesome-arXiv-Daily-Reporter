# CREATE: Testing LLMs for Associative Creativity 

**Authors**: Manya Wadhwa, Tiasa Singha Roy, Harvey Lederman, Junyi Jessy Li, Greg Durrett  

**Link**: [PDF](https://arxiv.org/pdf/2603.09970)  

**Abstract**: A key component of creativity is associative reasoning: the ability to draw novel yet meaningful connections between concepts. We introduce CREATE, a benchmark designed to evaluate models' capacity for creative associative reasoning. CREATE requires models to generate sets of paths connecting concepts in a model's parametric knowledge. Paths should have high specificity (distinctiveness and closeness of the concept connection) and high diversity (dissimilarity from other paths), and models are scored more highly if they produce a larger set of strong, diverse paths. This task shares demands of real creativity tasks like hypothesis generation, including an extremely large search space, but enables collection of a sizable benchmark with objective answer grading. Evaluation of frontier models shows that the strongest models achieve higher creative utility than others, with the high multiplicity of answers and complexity of the search making benchmark saturation difficult to achieve. Furthermore, our results illustrate that thinking models are not always more effective on our task, even with high token budgets. Recent approaches for creative prompting give some but limited additional improvement. CREATE provides a sandbox for developing new methods to improve models' capacity for associative creativity. 

---
# Model Merging in the Era of Large Language Models: Methods, Applications, and Future Directions 

**Authors**: Mingyang Song, Mao Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2603.09938)  

**Abstract**: Model merging has emerged as a transformative paradigm for combining the capabilities of multiple neural networks into a single unified model without additional training. With the rapid proliferation of fine-tuned large language models~(LLMs), merging techniques offer a computationally efficient alternative to ensembles and full retraining, enabling practitioners to compose specialized capabilities at minimal cost. This survey presents a comprehensive and structured examination of model merging in the LLM era through the \textbf{FUSE} taxonomy, a four-dimensional framework organized along \textbf{F}oundations, \textbf{U}nification Strategies, \textbf{S}cenarios, and \textbf{E}cosystem. We first establish the theoretical underpinnings of merging, including loss landscape geometry, mode connectivity, and the linear mode connectivity hypothesis. We then systematically review the algorithmic landscape, spanning weight averaging, task vector arithmetic, sparsification-enhanced methods, mixture-of-experts architectures, and evolutionary optimization approaches. For each method family, we analyze the core formulation, highlight representative works, and discuss practical trade-offs. We further examine downstream applications across multi-task learning, safety alignment, domain specialization, multilingual transfer, and federated learning. Finally, we survey the supporting ecosystem of open-source tools, community platforms, and evaluation benchmarks, and identify key open challenges including theoretical gaps, scalability barriers, and standardization needs. This survey aims to equip researchers and practitioners with a structured foundation for advancing model merging. 

---
# Thinking to Recall: How Reasoning Unlocks Parametric Knowledge in LLMs 

**Authors**: Zorik Gekhman, Roee Aharoni, Eran Ofek, Mor Geva, Roi Reichart, Jonathan Herzig  

**Link**: [PDF](https://arxiv.org/pdf/2603.09906)  

**Abstract**: While reasoning in LLMs plays a natural role in math, code generation, and multi-hop factual questions, its effect on simple, single-hop factual questions remains unclear. Such questions do not require step-by-step logical decomposition, making the utility of reasoning highly counterintuitive. Nevertheless, we find that enabling reasoning substantially expands the capability boundary of the model's parametric knowledge recall, unlocking correct answers that are otherwise effectively unreachable. Why does reasoning aid parametric knowledge recall when there are no complex reasoning steps to be done? To answer this, we design a series of hypothesis-driven controlled experiments, and identify two key driving mechanisms: (1) a computational buffer effect, where the model uses the generated reasoning tokens to perform latent computation independent of their semantic content; and (2) factual priming, where generating topically related facts acts as a semantic bridge that facilitates correct answer retrieval. Importantly, this latter generative self-retrieval mechanism carries inherent risks: we demonstrate that hallucinating intermediate facts during reasoning increases the likelihood of hallucinations in the final answer. Finally, we show that our insights can be harnessed to directly improve model accuracy by prioritizing reasoning trajectories that contain hallucination-free factual statements. 

---
# Benchmarking Political Persuasion Risks Across Frontier Large Language Models 

**Authors**: Zhongren Chen, Joshua Kalla, Quan Le  

**Link**: [PDF](https://arxiv.org/pdf/2603.09884)  

**Abstract**: Concerns persist regarding the capacity of Large Language Models (LLMs) to sway political views. Although prior research has claimed that LLMs are not more persuasive than standard political campaign practices, the recent rise of frontier models warrants further study. In two survey experiments (N=19,145) across bipartisan issues and stances, we evaluate seven state-of-the-art LLMs developed by Anthropic, OpenAI, Google, and xAI. We find that LLMs outperform standard campaign advertisements, with heterogeneity in performance across models. Specifically, Claude models exhibit the highest persuasiveness, while Grok exhibits the lowest. The results are robust across issues and stances. Moreover, in contrast to the findings in Hackenburg et al. (2025b) and Lin et al. (2025) that information-based prompts boost persuasiveness, we find that the effectiveness of information-based prompts is model-dependent: they increase the persuasiveness of Claude and Grok while substantially reducing that of GPT. We introduce a data-driven and strategy-agnostic LLM-assisted conversation analysis approach to identify and assess underlying persuasive strategies. Our work benchmarks the persuasive risks of frontier models and provides a framework for cross-model comparative risk assessment. 

---
# Do What I Say: A Spoken Prompt Dataset for Instruction-Following 

**Authors**: Maike Züfle, Sara Papi, Fabian Retkowski, Szymon Mazurek, Marek Kasztelnik, Alexander Waibel, Luisa Bentivogli, Jan Niehues  

**Link**: [PDF](https://arxiv.org/pdf/2603.09881)  

**Abstract**: Speech Large Language Models (SLLMs) have rapidly expanded, supporting a wide range of tasks. These models are typically evaluated using text prompts, which may not reflect real-world scenarios where users interact with speech. To address this gap, we introduce DoWhatISay (DOWIS), a multilingual dataset of human-recorded spoken and written prompts designed to pair with any existing benchmark for realistic evaluation of SLLMs under spoken instruction conditions. Spanning 9 tasks and 11 languages, it provides 10 prompt variants per task-language pair, across five styles. Using DOWIS, we benchmark state-of-the-art SLLMs, analyzing the interplay between prompt modality, style, language, and task type. Results show that text prompts consistently outperform spoken prompts, particularly for low-resource and cross-lingual settings. Only for tasks with speech output, spoken prompts do close the gap, highlighting the need for speech-based prompting in SLLM evaluation. 

---
# N-gram-like Language Models Predict Reading Time Best 

**Authors**: James A. Michaelov, Roger P. Levy  

**Link**: [PDF](https://arxiv.org/pdf/2603.09872)  

**Abstract**: Recent work has found that contemporary language models such as transformers can become so good at next-word prediction that the probabilities they calculate become worse for predicting reading time. In this paper, we propose that this can be explained by reading time being sensitive to simple n-gram statistics rather than the more complex statistics learned by state-of-the-art transformer language models. We demonstrate that the neural language models whose predictions are most correlated with n-gram probability are also those that calculate probabilities that are the most correlated with eye-tracking-based metrics of reading time on naturalistic text. 

---
# Chow-Liu Ordering for Long-Context Reasoning in Chain-of-Agents 

**Authors**: Naman Gupta, Vaibhav Singh, Arun Iyer, Kirankumar Shiragur, Pratham Grover, Ramakrishna B. Bairi, Ritabrata Maiti, Sankarshan Damle, Shachee Mishra Gupta, Rishikesh Maurya, Vageesh D. C  

**Link**: [PDF](https://arxiv.org/pdf/2603.09835)  

**Abstract**: Sequential multi-agent reasoning frameworks such as Chain-of-Agents (CoA) handle long-context queries by decomposing inputs into chunks and processing them sequentially using LLM-based worker agents that read from and update a bounded shared memory. From a probabilistic perspective, CoA aims to approximate the conditional distribution corresponding to a model capable of jointly reasoning over the entire long context. CoA achieves this through a latent-state factorization in which only bounded summaries of previously processed evidence are passed between agents. The resulting bounded-memory approximation introduces a lossy information bottleneck, making the final evidence state inherently dependent on the order in which chunks are processed.
In this work, we study the problem of chunk ordering for long-context reasoning. We use the well-known Chow-Liu trees to learn a dependency structure that prioritizes strongly related chunks. Empirically, we show that a breadth-first traversal of the resulting tree yields chunk orderings that reduce information loss across agents and consistently outperform both default document-chunk ordering and semantic score-based ordering in answer relevance and exact-match accuracy across three long-context benchmarks. 

---
# One-Eval: An Agentic System for Automated and Traceable LLM Evaluation 

**Authors**: Chengyu Shen, Yanheng Hou, Minghui Pan, Runming He, Zhen Hao Wong, Meiyi Qiang, Zhou Liu, Hao Liang, Peichao Lai, Zeang Sheng, Wentao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.09821)  

**Abstract**: Reliable evaluation is essential for developing and deploying large language models, yet in practice it often requires substantial manual effort: practitioners must identify appropriate benchmarks, reproduce heterogeneous evaluation codebases, configure dataset schema mappings, and interpret aggregated metrics. To address these challenges, we present One-Eval, an agentic evaluation system that converts natural-language evaluation requests into executable, traceable, and customizable evaluation workflows. One-Eval integrates (i) NL2Bench for intent structuring and personalized benchmark planning, (ii) BenchResolve for benchmark resolution, automatic dataset acquisition, and schema normalization to ensure executability, and (iii) Metrics \& Reporting for task-aware metric selection and decision-oriented reporting beyond scalar scores. The system further incorporates human-in-the-loop checkpoints for review, editing, and rollback, while preserving sample evidence trails for debugging and auditability. Experiments show that One-Eval can execute end-to-end evaluations from diverse natural-language requests with minimal user effort, supporting more efficient and reproducible evaluation in industrial settings. Our framework is publicly available at this https URL. 

---
# EPIC-EuroParl-UdS: Information-Theoretic Perspectives on Translation and Interpreting 

**Authors**: Maria Kunilovskaya, Christina Pollkläsener  

**Link**: [PDF](https://arxiv.org/pdf/2603.09785)  

**Abstract**: This paper introduces an updated and combined version of the bidirectional English-German EPIC-UdS (spoken) and EuroParl-UdS (written) corpora containing original European Parliament speeches as well as their translations and interpretations. The new version corrects metadata and text errors identified through previous use, refines the content, updates linguistic annotations, and adds new layers, including word alignment and word-level surprisal indices. The combined resource is designed to support research using information-theoretic approaches to language variation, particularly studies comparing written and spoken modes, and examining disfluencies in speech, as well as traditional translationese studies, including parallel (source vs. target) and comparable (original vs. translated) analyses. The paper outlines the updates introduced in this release, summarises previous results based on the corpus, and presents a new illustrative study. The study validates the integrity of the rebuilt spoken data and evaluates probabilistic measures derived from base and fine-tuned GPT-2 and machine translation models on the task of filler particles prediction in interpreting. 

---
# Beyond Fine-Tuning: Robust Food Entity Linking under Ontology Drift with FoodOntoRAG 

**Authors**: Jan Drole, Ana Gjorgjevikj, Barbara Korouši'c Seljak, Tome Eftimov  

**Link**: [PDF](https://arxiv.org/pdf/2603.09758)  

**Abstract**: Standardizing food terms from product labels and menus into ontology concepts is a prerequisite for trustworthy dietary assessment and safety reporting. The dominant approach to Named Entity Linking (NEL) in the food and nutrition domains fine-tunes Large Language Models (LLMs) on task-specific corpora. Although effective, fine-tuning incurs substantial computational cost, ties models to a particular ontology snapshot (i.e., version), and degrades under ontology drift. This paper presents FoodOntoRAG, a model- and ontology-agnostic pipeline that performs few-shot NEL by retrieving candidate entities from domain ontologies and conditioning an LLM on structured evidence (food labels, synonyms, definitions, and relations). A hybrid lexical--semantic retriever enumerates candidates; a selector agent chooses a best match with rationale; a separate scorer agent calibrates confidence; and, when confidence falls below a threshold, a synonym generator agent proposes reformulations to re-enter the loop. The pipeline approaches state-of-the-art accuracy while revealing gaps and inconsistencies in existing annotations. The design avoids fine-tuning, improves robustness to ontology evolution, and yields interpretable decisions through grounded justifications. 

---
# RbtAct: Rebuttal as Supervision for Actionable Review Feedback Generation 

**Authors**: Sihong Wu, Yiling Ma, Yilun Zhao, Tiansheng Hu, Owen Jiang, Manasi Patwardhan, Arman Cohan  

**Link**: [PDF](https://arxiv.org/pdf/2603.09723)  

**Abstract**: Large language models (LLMs) are increasingly used across the scientific workflow, including to draft peer-review reports. However, many AI-generated reviews are superficial and insufficiently actionable, leaving authors without concrete, implementable guidance and motivating the gap this work addresses. We propose RbtAct, which targets actionable review feedback generation and places existing peer review rebuttal at the center of learning. Rebuttals show which reviewer comments led to concrete revisions or specific plans, and which were only defended. Building on this insight, we leverage rebuttal as implicit supervision to directly optimize a feedback generator for actionability. To support this objective, we propose a new task called perspective-conditioned segment-level review feedback generation, in which the model is required to produce a single focused comment based on the complete paper and a specified perspective such as experiments and writing. We also build a large dataset named RMR-75K that maps review segments to the rebuttal segments that address them, with perspective labels and impact categories that order author uptake. We then train the Llama-3.1-8B-Instruct model with supervised fine-tuning on review segments followed by preference optimization using rebuttal derived pairs. Experiments with human experts and LLM-as-a-judge show consistent gains in actionability and specificity over strong baselines while maintaining grounding and relevance. 

---
# Evaluation of LLMs in retrieving food and nutritional context for RAG systems 

**Authors**: Maks Požarnik Vavken, Matevž Ogrinc, Tome Eftimov, Barbara Koroušić Seljak  

**Link**: [PDF](https://arxiv.org/pdf/2603.09704)  

**Abstract**: In this article, we evaluate four Large Language Models (LLMs) and their effectiveness at retrieving data within a specialized Retrieval-Augmented Generation (RAG) system, using a comprehensive food composition database. Our method is focused on the LLMs ability to translate natural language queries into structured metadata filters, enabling efficient retrieval via a Chroma vector database. By achieving high accuracy in this critical retrieval step, we demonstrate that LLMs can serve as an accessible, high-performance tool, drastically reducing the manual effort and technical expertise previously required for domain experts, such as food compilers and nutritionists, to leverage complex food and nutrition data. However, despite the high performance on easy and moderately complex queries, our analysis of difficult questions reveals that reliable retrieval remains challenging when queries involve non-expressible constraints. These findings demonstrate that LLM-driven metadata filtering excels when constraints can be explicitly expressed, but struggles when queries exceed the representational scope of the metadata format. 

---
# ESAinsTOD: A Unified End-to-End Schema-Aware Instruction-Tuning Framework for Task-Oriented Dialog Modeling 

**Authors**: Dechuan Teng, Chunlin Lu, Libo Qin, Wanxiang Che  

**Link**: [PDF](https://arxiv.org/pdf/2603.09691)  

**Abstract**: Existing end-to-end modeling methods for modular task-oriented dialog systems are typically tailored to specific datasets, making it challenging to adapt to new dialog scenarios. In this work, we propose ESAinsTOD, a unified End-to-end Schema-Aware Instruction-tuning framework for general Task-Oriented Dialog modeling. This framework introduces a structured methodology to go beyond simply fine-tuning Large Language Models (LLMs), enabling flexible adaptation to various dialogue task flows and schemas. Specifically, we leverage full-parameter fine-tuning of LLMs and introduce two alignment mechanisms to make the resulting system both instruction-aware and schema-aware: (i) instruction alignment, which ensures that the system faithfully follows task instructions to complete various task flows from heterogeneous TOD datasets; and (ii) schema alignment, which encourages the system to make predictions adhering to the specified schema. In addition, we employ session-level end-to-end modeling, which allows the system to access the results of previously executed task flows within the dialogue history, to bridge the gap between the instruction-tuning paradigm and the real-world application of TOD systems. Empirical results show that while a fine-tuned LLM serves as a strong baseline, our structured approach provides significant additional benefits. In particular, our findings indicate that: (i) ESAinsTOD outperforms state-of-the-art models by a significant margin on end-to-end task-oriented dialog modeling benchmarks: CamRest676, In-Car and MultiWOZ; (ii) more importantly, it exhibits superior generalization capabilities across various low-resource settings, with the proposed alignment mechanisms significantly enhancing zero-shot performance; and (iii) our instruction-tuning paradigm substantially improves the model's robustness against data noise and cascading errors. 

---
# Fusing Semantic, Lexical, and Domain Perspectives for Recipe Similarity Estimation 

**Authors**: Denica Kjorvezir, Danilo Najkov, Eva Valencič, Erika Jesenko, Barbara Koroišić Seljak, Tome Eftimov, Riste Stojanov  

**Link**: [PDF](https://arxiv.org/pdf/2603.09688)  

**Abstract**: This research focuses on developing advanced methods for assessing similarity between recipes by combining different sources of information and analytical approaches. We explore the semantic, lexical, and domain similarity of food recipes, evaluated through the analysis of ingredients, preparation methods, and nutritional attributes. A web-based interface was developed to allow domain experts to validate the combined similarity results. After evaluating 318 recipe pairs, experts agreed on 255 (80%). The evaluation of expert assessments enables the estimation of which similarity aspects--lexical, semantic, or nutritional--are most influential in expert decision-making. The application of these methods has broad implications in the food industry and supports the development of personalized diets, nutrition recommendations, and automated recipe generation systems. 

---
# Automatic Cardiac Risk Management Classification using large-context Electronic Patients Health Records 

**Authors**: Jacopo Vitale, David Della Morte, Luca Bacco, Mario Merone, Mark de Groot, Saskia Haitjema, Leandro Pecchia, Bram van Es  

**Link**: [PDF](https://arxiv.org/pdf/2603.09685)  

**Abstract**: To overcome the limitations of manual administrative coding in geriatric Cardiovascular Risk Management, this study introduces an automated classification framework leveraging unstructured Electronic Health Records (EHRs). Using a dataset of 3,482 patients, we benchmarked three distinct modeling paradigms on longitudinal Dutch clinical narratives: classical machine learning baselines, specialized deep learning architectures optimized for large-context sequences, and general-purpose generative Large Language Models (LLMs) in a zero-shot setting. Additionally, we evaluated a late fusion strategy to integrate unstructured text with structured medication embeddings and anthropometric data. Our analysis reveals that the custom Transformer architecture outperforms both traditional methods and generative \acs{llm}s, achieving the highest F1-scores and Matthews Correlation Coefficients. These findings underscore the critical role of specialized hierarchical attention mechanisms in capturing long-range dependencies within medical texts, presenting a robust, automated alternative to manual workflows for clinical risk stratification. 

---
# Understanding the Interplay between LLMs' Utilisation of Parametric and Contextual Knowledge: A keynote at ECIR 2025 

**Authors**: Isabelle Augenstein  

**Link**: [PDF](https://arxiv.org/pdf/2603.09654)  

**Abstract**: Language Models (LMs) acquire parametric knowledge from their training process, embedding it within their weights. The increasing scalability of LMs, however, poses significant challenges for understanding a model's inner workings and further for updating or correcting this embedded knowledge without the significant cost of retraining. Moreover, when using these language models for knowledge-intensive language understanding tasks, LMs have to integrate relevant context, mitigating their inherent weaknesses, such as incomplete or outdated knowledge. Nevertheless, studies indicate that LMs often ignore the provided context as it can be in conflict with the pre-existing LM's memory learned during pre-training. Conflicting knowledge can also already be present in the LM's parameters, termed intra-memory conflict. This underscores the importance of understanding the interplay between how a language model uses its parametric knowledge and the retrieved contextual knowledge. In this talk, I will aim to shed light on this important issue by presenting our research on evaluating the knowledge present in LMs, diagnostic tests that can reveal knowledge conflicts, as well as on understanding the characteristics of successfully used contextual knowledge. 

---
# Tracking Cancer Through Text: Longitudinal Extraction From Radiology Reports Using Open-Source Large Language Models 

**Authors**: Luc Builtjes, Alessa Hering  

**Link**: [PDF](https://arxiv.org/pdf/2603.09638)  

**Abstract**: Radiology reports capture crucial longitudinal information on tumor burden, treatment response, and disease progression, yet their unstructured narrative format complicates automated analysis. While large language models (LLMs) have advanced clinical text processing, most state-of-the-art systems remain proprietary, limiting their applicability in privacy-sensitive healthcare environments. We present a fully open-source, locally deployable pipeline for longitudinal information extraction from radiology reports, implemented using the \texttt{llm\_extractinator} framework. The system applies the \texttt{qwen2.5-72b} model to extract and link target, non-target, and new lesion data across time points in accordance with RECIST criteria. Evaluation on 50 Dutch CT Thorax/Abdomen report pairs yielded high extraction performance, with attribute-level accuracies of 93.7\% for target lesions, 94.9\% for non-target lesions, and 94.0\% for new lesions. The approach demonstrates that open-source LLMs can achieve clinically meaningful performance in multi-timepoint oncology tasks while ensuring data privacy and reproducibility. These results highlight the potential of locally deployable LLMs for scalable extraction of structured longitudinal data from routine clinical text. 

---
# Surgical Repair of Collapsed Attention Heads in ALiBi Transformers 

**Authors**: Palmer Schallon  

**Link**: [PDF](https://arxiv.org/pdf/2603.09616)  

**Abstract**: We identify a systematic attention collapse pathology in the BLOOM family of transformer language models, where ALiBi positional encoding causes 31-44% of attention heads to attend almost entirely to the beginning-of-sequence token. The collapse follows a predictable pattern across four model scales (560M to 7.1B parameters), concentrating in head indices where ALiBi's slope schedule imposes the steepest distance penalties. We introduce surgical reinitialization: targeted Q/K/V reinitialization with zeroed output projections and gradient-masked freezing of all non-surgical parameters. Applied to BLOOM-1b7 on a single consumer GPU, the technique recovers 98.7% operational head capacity (242 to 379 of 384 heads) in two passes. A controlled comparison with C4 training data confirms that reinitialization -- not corpus content -- drives recovery, and reveals two distinct post-surgical phenomena: early global functional redistribution that improves the model, and late local degradation that accumulates under noisy training signal. An extended experiment reinitializing mostly-healthy heads alongside collapsed ones produces a model that transiently outperforms stock BLOOM-1b7 by 25% on training perplexity (12.70 vs. 16.99), suggesting that pretrained attention configurations are suboptimal local minima. Code, checkpoints, and diagnostic tools are released as open-source software. 

---
# Build, Borrow, or Just Fine-Tune? A Political Scientist's Guide to Choosing NLP Models 

**Authors**: Shreyas Meher  

**Link**: [PDF](https://arxiv.org/pdf/2603.09595)  

**Abstract**: Political scientists increasingly face a consequential choice when adopting natural language processing tools: build a domain-specific model from scratch, borrow and adapt an existing one, or simply fine-tune a general-purpose model on task data? Each approach occupies a different point on the spectrum of performance, cost, and required expertise, yet the discipline has offered little empirical guidance on how to navigate this trade-off. This paper provides such guidance. Using conflict event classification as a test case, I fine-tune ModernBERT on the Global Terrorism Database (GTD) to create Confli-mBERT and systematically compare it against ConfliBERT, a domain-specific pretrained model that represents the current gold standard. Confli-mBERT achieves 75.46% accuracy compared to ConfliBERT's 79.34%. Critically, the four-percentage-point gap is not uniform: on high-frequency attack types such as Bombing/Explosion (F1 = 0.95 vs. 0.96) and Kidnapping (F1 = 0.92 vs. 0.91), the models are nearly indistinguishable. Performance differences concentrate in rare event categories comprising fewer than 2% of all incidents. I use these findings to develop a practical decision framework for political scientists considering any NLP-assisted research task: when does the research question demand a specialized model, and when does an accessible fine-tuned alternative suffice? The answer, I argue, depends not on which model is "better" in the abstract, but on the specific intersection of class prevalence, error tolerance, and available resources. The model, training code, and data are publicly available on Hugging Face. 

---
# ALARM: Audio-Language Alignment for Reasoning Models 

**Authors**: Petr Grinberg, Hassan Shahmohammadi  

**Link**: [PDF](https://arxiv.org/pdf/2603.09556)  

**Abstract**: Large audio language models (ALMs) extend LLMs with auditory understanding. A common approach freezes the LLM and trains only an adapter on self-generated targets. However, this fails for reasoning LLMs (RLMs) whose built-in chain-of-thought traces expose the textual surrogate input, yielding unnatural responses. We propose self-rephrasing, converting self-generated responses into audio-understanding variants compatible with RLMs while preserving distributional alignment. We further fuse and compress multiple audio encoders for stronger representations. For training, we construct a 6M-instance multi-task corpus (2.5M unique prompts) spanning 19K hours of speech, music, and sound. Our 4B-parameter ALM outperforms similarly sized models and surpasses most larger ALMs on related audio-reasoning benchmarks, while preserving textual capabilities with a low training cost. Notably, we achieve the best open-source result on the MMAU-speech and MMSU benchmarks and rank third among all the models. 

---
# You Didn't Have to Say It like That: Subliminal Learning from Faithful Paraphrases 

**Authors**: Isaia Gisler, Zhonghao He, Tianyi Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2603.09517)  

**Abstract**: When language models are trained on synthetic data, they (student model) can covertly acquire behavioral traits from the data-generating model (teacher model). Subliminal learning refers to the transmission of traits from a teacher to a student model via training on data unrelated to those traits. Prior work demonstrated this in the training domains of number sequences, code, and math Chain-of-Thought traces including transmission of misaligned behaviors. We investigate whether transmission occurs through natural language paraphrases with fixed semantic content, and whether content explicitly contradicting the teacher's preference can block it. We find that training on paraphrases from a teacher system-prompted to love a particular animal increases a student's preference for that animal by up to 19 percentage points. This occurs when paraphrased content is semantically unrelated to the animal, or even when it explicitly expresses dislike. The transmission succeeds despite aggressive filtering to ensure paraphrase fidelity. This raises concerns for pipelines where models generate their own training data: content-based inspection cannot detect such transmission, and even preference-contradicting content fails to prevent it. 

---
# Modelling the Diachronic Emergence of Phoneme Frequency Distributions 

**Authors**: Fermín Moscoso del Prado Martín, Suchir Salhan  

**Link**: [PDF](https://arxiv.org/pdf/2603.09503)  

**Abstract**: Phoneme frequency distributions exhibit robust statistical regularities across languages, including exponential-tailed rank-frequency patterns and a negative relationship between phonemic inventory size and the relative entropy of the distribution. The origin of these patterns remains largely unexplained. In this paper, we investigate whether they can arise as consequences of the historical processes that shape phonological systems. We introduce a stochastic model of phonological change and simulate the diachronic evolution of phoneme inventories. A naïve version of the model reproduces the general shape of phoneme rank-frequency distributions but fails to capture other empirical properties. Extending the model with two additional assumptions -- an effect related to functional load and a stabilising tendency toward a preferred inventory size -- yields simulations that match both the observed distributions and the negative relationship between inventory size and relative entropy. These results suggest that some statistical regularities of phonological systems may arise as natural consequences of diachronic sound change rather than from explicit optimisation or compensatory mechanisms. 

---
# Common Sense vs. Morality: The Curious Case of Narrative Focus Bias in LLMs 

**Authors**: Saugata Purkayastha, Pranav Kushare, Pragya Paramita Pal, Sukannya Purkayastha  

**Link**: [PDF](https://arxiv.org/pdf/2603.09434)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed across diverse real-world applications and user communities. As such, it is crucial that these models remain both morally grounded and knowledge-aware. In this work, we uncover a critical limitation of current LLMs -- their tendency to prioritize moral reasoning over commonsense understanding. To investigate this phenomenon, we introduce CoMoral, a novel benchmark dataset containing commonsense contradictions embedded within moral dilemmas. Through extensive evaluation of ten LLMs across different model sizes, we find that existing models consistently struggle to identify such contradictions without prior signal. Furthermore, we observe a pervasive narrative focus bias, wherein LLMs more readily detect commonsense contradictions when they are attributed to a secondary character rather than the primary (narrator) character. Our comprehensive analysis underscores the need for enhanced reasoning-aware training to improve the commonsense robustness of large language models. 

---
# Investigating Gender Stereotypes in Large Language Models via Social Determinants of Health 

**Authors**: Trung Hieu Ngo, Adrien Bazoge, Solen Quiniou, Pierre-Antoine Gourraud, Emmanuel Morin  

**Link**: [PDF](https://arxiv.org/pdf/2603.09416)  

**Abstract**: Large Language Models (LLMs) excel in Natural Language Processing (NLP) tasks, but they often propagate biases embedded in their training data, which is potentially impactful in sensitive domains like healthcare. While existing benchmarks evaluate biases related to individual social determinants of health (SDoH) such as gender or ethnicity, they often overlook interactions between these factors and lack context-specific assessments. This study investigates bias in LLMs by probing the relationships between gender and other SDoH in French patient records. Through a series of experiments, we found that embedded stereotypes can be probed using SDoH input and that LLMs rely on embedded stereotypes to make gendered decisions, suggesting that evaluating interactions among SDoH factors could usefully complement existing approaches to assessing LLM performance and bias. 

---
# LLM as a Meta-Judge: Synthetic Data for NLP Evaluation Metric Validation 

**Authors**: Lukáš Eigler, Jindřich Libovický, David Hurych  

**Link**: [PDF](https://arxiv.org/pdf/2603.09403)  

**Abstract**: Validating evaluation metrics for NLG typically relies on expensive and time-consuming human annotations, which predominantly exist only for English datasets. We propose \textit{LLM as a Meta-Judge}, a scalable framework that utilizes LLMs to generate synthetic evaluation datasets via controlled semantic degradation of real data, replacing human judgment. We validate our approach using \textit{meta-correlation}, measuring the alignment between metric rankings derived from synthetic data and those from standard human benchmarks. Experiments across Machine Translation, Question Answering, and Summarization demonstrate that synthetic validation serves as a reliable proxy for human judgment, achieving meta-correlations exceeding 0.9 in multilingual QA and proves to be a viable alternative where human judgments are unavailable or too expensive to obtain. Our code and data will become publicly available upon paper acceptance. 

---
# Reward Prediction with Factorized World States 

**Authors**: Yijun Shen, Delong Chen, Xianming Hu, Jiaming Mi, Hongbo Zhao, Kai Zhang, Pascale Fung  

**Link**: [PDF](https://arxiv.org/pdf/2603.09400)  

**Abstract**: Agents must infer action outcomes and select actions that maximize a reward signal indicating how close the goal is to being reached. Supervised learning of reward models could introduce biases inherent to training data, limiting generalization to novel goals and environments. In this paper, we investigate whether well-defined world state representations alone can enable accurate reward prediction across domains. To address this, we introduce StateFactory, a factorized representation method that transforms unstructured observations into a hierarchical object-attribute structure using language models. This structured representation allows rewards to be estimated naturally as the semantic similarity between the current state and the goal state under hierarchical constraint. Overall, the compact representation structure induced by StateFactory enables strong reward generalization capabilities. We evaluate on RewardPrediction, a new benchmark dataset spanning five diverse domains and comprising 2,454 unique action-observation trajectories with step-wise ground-truth rewards. Our method shows promising zero-shot results against both VLWM-critic and LLM-as-a-Judge reward models, achieving 60% and 8% lower EPIC distance, respectively. Furthermore, this superior reward quality successfully translates into improved agent planning performance, yielding success rate gains of +21.64% on AlfWorld and +12.40% on ScienceWorld over reactive system-1 policies and enhancing system-2 agent planning. Project Page: this https URL 

---
# Quantifying and extending the coverage of spatial categorization data sets 

**Authors**: Wanchun Li, Alexandra Carstensen, Yang Xu, Terry Regier, Charles Kemp  

**Link**: [PDF](https://arxiv.org/pdf/2603.09373)  

**Abstract**: Variation in spatial categorization across languages is often studied by eliciting human labels for the relations depicted in a set of scenes known as the Topological Relations Picture Series (TRPS). We demonstrate that labels generated by large language models (LLMs) align relatively well with human labels, and show how LLM-generated labels can help to decide which scenes and languages to add to existing spatial data sets. To illustrate our approach we extend the TRPS by adding 42 new scenes, and show that this extension achieves better coverage of the space of possible scenes than two previous extensions of the TRPS. Our results provide a foundation for scaling towards spatial data sets with dozens of languages and hundreds of scenes. 

---
# TaSR-RAG: Taxonomy-guided Structured Reasoning for Retrieval-Augmented Generation 

**Authors**: Jiashuo Sun, Yixuan Xie, Jimeng Shi, Shaowen Wang, Jiawei Han  

**Link**: [PDF](https://arxiv.org/pdf/2603.09341)  

**Abstract**: Retrieval-Augmented Generation (RAG) helps large language models (LLMs) answer knowledge-intensive and time-sensitive questions by conditioning generation on external evidence. However, most RAG systems still retrieve unstructured chunks and rely on one-shot generation, which often yields redundant context, low information density, and brittle multi-hop reasoning. While structured RAG pipelines can improve grounding, they typically require costly and error-prone graph construction or impose rigid entity-centric structures that do not align with the query's reasoning chain.
We propose \textsc{TaSR-RAG}, a taxonomy-guided structured reasoning framework for evidence selection. We represent both queries and documents as relational triples, and constrain entity semantics with a lightweight two-level taxonomy to balance generalization and precision. Given a complex question, \textsc{TaSR-RAG} decomposes it into an ordered sequence of triple sub-queries with explicit latent variables, then performs step-wise evidence selection via hybrid triple matching that combines semantic similarity over raw triples with structural consistency over typed triples.
By maintaining an explicit entity binding table across steps, \textsc{TaSR-RAG} resolves intermediate variables and reduces entity conflation without explicit graph construction or exhaustive search. Experiments on multiple multi-hop question answering benchmarks show that \textsc{TaSR-RAG} consistently outperforms strong RAG and structured-RAG baselines by up to 14\%, while producing clearer evidence attribution and more faithful reasoning traces. 

---
# LooComp: Leverage Leave-One-Out Strategy to Encoder-only Transformer for Efficient Query-aware Context Compression 

**Authors**: Thao Do, Dinh Phu Tran, An Vo, Seon Kwon Kim, Daeyoung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2603.09222)  

**Abstract**: Efficient context compression is crucial for improving the accuracy and scalability of question answering. For the efficiency of Retrieval Augmented Generation, context should be delivered fast, compact, and precise to ensure clue sufficiency and budget-friendly LLM reader cost. We propose a margin-based framework for query-driven context pruning, which identifies sentences that are critical for answering a query by measuring changes in clue richness when they are omitted. The model is trained with a composite ranking loss that enforces large margins for critical sentences while keeping non-critical ones near neutral. Built on a lightweight encoder-only Transformer, our approach generally achieves strong exact-match and F1 scores with high-throughput inference and lower memory requirements than those of major baselines. In addition to efficiency, our method yields effective compression ratios without degrading answering performance, demonstrating its potential as a lightweight and practical alternative for retrieval-augmented tasks. 

---
# SPAR-K: Scheduled Periodic Alternating Early Exit for Spoken Language Models 

**Authors**: Hsiao-Ying Huang, Cheng-Han Chiang, Hung-yi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2603.09215)  

**Abstract**: Interleaved spoken language models (SLMs) alternately generate text and speech tokens, but decoding at full transformer depth for every step becomes costly, especially due to long speech sequences. We propose SPAR-K, a modality-aware early exit framework designed to accelerate interleaved SLM inference while preserving perceptual quality. SPAR-K introduces a speech alternating-depth schedule: most speech positions exit at a fixed intermediate layer, while periodic full-depth "refresh" steps mitigate distribution shift due to early exit. We evaluate our framework using Step-Audio-2-mini and GLM-4-Voice across four datasets spanning reasoning, factual QA, and dialogue tasks, measuring performance in terms of ASR transcription accuracy and perceptual quality. Experimental results demonstrate that SPAR-K largely preserves question-answering accuracy with a maximum accuracy drop of 0.82\% while reducing average speech decoding depth by up to 11\% on Step-Audio-2-mini and 5\% on GLM-4-Voice, both with negligible changes in MOS and WER and no auxiliary computation overhead. We further demonstrate that confidence-based early exit strategies, widely used in text LLMs, are suboptimal for SLMs, highlighting that the unique statistical nature of speech tokens necessitates a specialized early exit design. 

---
# Emotion is Not Just a Label: Latent Emotional Factors in LLM Processing 

**Authors**: Benjamin Reichman, Adar Avasian, Samuel Webster, Larry Heck  

**Link**: [PDF](https://arxiv.org/pdf/2603.09205)  

**Abstract**: Large language models are routinely deployed on text that varies widely in emotional tone, yet their reasoning behavior is typically evaluated without accounting for emotion as a source of representational variation. Prior work has largely treated emotion as a prediction target, for example in sentiment analysis or emotion classification. In contrast, we study emotion as a latent factor that shapes how models attend to and reason over text. We analyze how emotional tone systematically alters attention geometry in transformer models, showing that metrics such as locality, center-of-mass distance, and entropy vary across emotions and correlate with downstream question-answering performance. To facilitate controlled study of these effects, we introduce Affect-Uniform ReAding QA (AURA-QA), a question-answering dataset with emotionally balanced, human-authored context passages. Finally, an emotional regularization framework is proposed that constrains emotion-conditioned representational drift during training. Experiments across multiple QA benchmarks demonstrate that this approach improves reading comprehension in both emotionally-varying and non-emotionally varying datasets, yielding consistent gains under distribution shift and in-domain improvements on several benchmarks. 

---
# DEO: Training-Free Direct Embedding Optimization for Negation-Aware Retrieval 

**Authors**: Taegyeong Lee, Jiwon Park, Seunghyun Hwang, JooYoung Jang  

**Link**: [PDF](https://arxiv.org/pdf/2603.09185)  

**Abstract**: Recent advances in Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) have enabled diverse retrieval methods. However, existing retrieval methods often fail to accurately retrieve results for negation and exclusion queries. To address this limitation, prior approaches rely on embedding adaptation or fine-tuning, which introduce additional computational cost and deployment complexity. We propose Direct Embedding Optimization (DEO), a training-free method for negation-aware text and multimodal retrieval. DEO decomposes queries into positive and negative components and optimizes the query embedding with a contrastive objective. Without additional training data or model updates, DEO outperforms baselines on NegConstraint, with gains of +0.0738 nDCG@10 and +0.1028 MAP@100, while improving Recall@5 by +6\% over OpenAI CLIP in multimodal retrieval. These results demonstrate the practicality of DEO for negation- and exclusion-aware retrieval in real-world settings. 

---
# DuplexCascade: Full-Duplex Speech-to-Speech Dialogue with VAD-Free Cascaded ASR-LLM-TTS Pipeline and Micro-Turn Optimization 

**Authors**: Jianing Yang, Yusuke Fujita, Yui Sudo  

**Link**: [PDF](https://arxiv.org/pdf/2603.09180)  

**Abstract**: Spoken dialog systems with cascaded ASR-LLM-TTS modules retain strong LLM intelligence, but VAD segmentation often forces half-duplex turns and brittle control. On the other hand, VAD-free end-to-end model support full-duplex interaction but is hard to maintain conversational intelligence. In this paper, we present DuplexCascade, a VAD-free cascaded streaming pipeline for full-duplex speech-to-speech dialogue. Our key idea is to convert conventional utterance-wise long turns into chunk-wise micro-turn interactions, enabling rapid bidirectional exchange while preserving the strengths of a capable text LLM. To reliably coordinate turn-taking and response timing, we introduce a set of conversational special control tokens that steer the LLM's behavior under streaming constraints. On Full-DuplexBench and VoiceBench, DuplexCascade delivers state-of-the-art full-duplex turn-taking and strong conversational intelligence among open-source speech-to-speech dialogue systems. 

---
# Bioalignment: Measuring and Improving LLM Disposition Toward Biological Systems for AI Safety 

**Authors**: Trent R Northen, Mingxun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.09154)  

**Abstract**: Large language models (LLMs) trained on internet-scale corpora can exhibit systematic biases that increase the probability of unwanted behavior. In this study, we examined potential biases towards synthetic vs. biological technological solutions across four domains (materials, energy, manufacturing, and algorithms). A sample of 5 frontier and 5 open-weight models were measured using 50 curated Bioalignment prompts with a Kelly criterion-inspired evaluation framework. According to this metric, most models were not bioaligned in that they exhibit biases in favor of synthetic (non-biological) solutions. We next examined if fine-tuning could increase the preferences of two open-weight models, Llama 3.2-3B-Instruct and Qwen2.5-3B-Instruct, for biological-based approaches. A curated corpus of ~22M tokens from 6,636 PMC articles emphasizing biological problem-solving was used first to fine-tune Llama 3B with a mixed corpus of continued training and instruction-formatted. This was then extended to Qwen 3B using instruction-formatted only. We found that QLoRA fine-tuning significantly increased the scoring of biological solutions for both models without degrading general capabilities (Holm-Bonferroni-corrected p < 0.001 and p < 0.01, respectively). This suggests that even a small amount of fine-tuning can change how models weigh the relative value of biological and bioinspired vs. synthetic approaches. Although this work focused on small open-weight LLMs, it may be extensible to much larger models and could be used to develop models that favor bio-based approaches. We release the benchmark, corpus, code, and adapter weights. 

---
# Reading, Not Thinking: Understanding and Bridging the Modality Gap When Text Becomes Pixels in Multimodal LLMs 

**Authors**: Kaiser Sun, Xiaochuang Yuan, Hongjun Liu, Chen Zhao, Cheng Zhang, Mark Dredze, Fan Bai  

**Link**: [PDF](https://arxiv.org/pdf/2603.09095)  

**Abstract**: Multimodal large language models (MLLMs) can process text presented as images, yet they often perform worse than when the same content is provided as textual tokens. We systematically diagnose this "modality gap" by evaluating seven MLLMs across seven benchmarks in five input modes, spanning both synthetically rendered text and realistic document images from arXiv PDFs to Wikipedia pages. We find that the modality gap is task- and data-dependent. For example, math tasks degrade by over 60 points on synthetic renderings, while natural document images often match or exceed text-mode performance. Rendering choices such as font and resolution are strong confounds, with font alone swinging accuracy by up to 47 percentage points. To understand this, we conduct a grounded-theory error analysis of over 4,000 examples, revealing that image mode selectively amplifies reading errors (calculation and formatting failures) while leaving knowledge and reasoning errors largely unchanged, and that some models exhibit a chain-of-thought reasoning collapse under visual input. Motivated by these findings, we propose a self-distillation method that trains the model on its own pure text reasoning traces paired with image inputs, raising image-mode accuracy on GSM8K from 30.71% to 92.72% and transferring to unseen benchmarks without catastrophic forgetting. Overall, our study provides a systematic understanding of the modality gap and suggests a practical path toward improving visual text understanding in multimodal language models. 

---
# Learning When to Sample: Confidence-Aware Self-Consistency for Efficient LLM Chain-of-Thought Reasoning 

**Authors**: Juming Xiong, Kevin Guo, Congning Ni, Chao Yan, Katherine Brown, Avinash Baidya, Xiang Gao, Bradley Marlin, Zhijun Yin  

**Link**: [PDF](https://arxiv.org/pdf/2603.08999)  

**Abstract**: Large language models (LLMs) achieve strong reasoning performance through chain-of-thought (CoT) reasoning, yet often generate unnecessarily long reasoning paths that incur high inference cost. Recent self-consistency-based approaches further improve accuracy but require sampling and aggregating multiple reasoning trajectories, leading to substantial additional computational overhead. This paper introduces a confidence-aware decision framework that analyzes a single completed reasoning trajectory to adaptively select between single-path and multi-path reasoning. The framework is trained using sentence-level numeric and linguistic features extracted from intermediate reasoning states in the MedQA dataset and generalizes effectively to MathQA, MedMCQA, and MMLU without additional fine-tuning. Experimental results show that the proposed method maintains accuracy comparable to multi-path baselines while using up to 80\% fewer tokens. These findings demonstrate that reasoning trajectories contain rich signals for uncertainty estimation, enabling a simple, transferable mechanism to balance accuracy and efficiency in LLM reasoning. 

---
# Automated Thematic Analysis for Clinical Qualitative Data: Iterative Codebook Refinement with Full Provenance 

**Authors**: Seungjun Yi, Joakim Nguyen, Huimin Xu, Terence Lim, Joseph Skrovan, Mehak Beri, Hitakshi Modi, Andrew Well, Carlos M. Mery, Yan Zhang, Mia K. Markey, Ying Ding  

**Link**: [PDF](https://arxiv.org/pdf/2603.08989)  

**Abstract**: Thematic analysis (TA) is widely used in health research to extract patterns from patient interviews, yet manual TA faces challenges in scalability and reproducibility. LLM-based automation can help, but existing approaches produce codebooks with limited generalizability and lack analytic auditability. We present an automated TA framework combining iterative codebook refinement with full provenance tracking. Evaluated on five corpora spanning clinical interviews, social media, and public transcripts, the framework achieves the highest composite quality score on four of five datasets compared to six baselines. Iterative refinement yields statistically significant improvements on four datasets with large effect sizes, driven by gains in code reusability and distributional consistency while preserving descriptive quality. On two clinical corpora (pediatric cardiology), generated themes align with expert-annotated themes. 

---
# SciTaRC: Benchmarking QA on Scientific Tabular Data that Requires Language Reasoning and Complex Computation 

**Authors**: Hexuan Wang, Yaxuan Ren, Srikar Bommireddypalli, Shuxian Chen, Adarsh Prabhudesai, Rongkun Zhou, Elina Baral, Philipp Koehn  

**Link**: [PDF](https://arxiv.org/pdf/2603.08910)  

**Abstract**: We introduce SciTaRC, an expert-authored benchmark of questions about tabular data in scientific papers requiring both deep language reasoning and complex computation. We show that current state-of-the-art AI models fail on at least 23% of these questions, a gap that remains significant even for highly capable open-weight models like Llama-3.3-70B-Instruct, which fails on 65.5% of the tasks. Our analysis reveals a universal "execution bottleneck": both code and language models struggle to faithfully execute plans, even when provided with correct strategies. Specifically, code-based methods prove brittle on raw scientific tables, while natural language reasoning primarily fails due to initial comprehension issues and calculation errors. 

---
# ConFu: Contemplate the Future for Better Speculative Sampling 

**Authors**: Zongyue Qin, Raghavv Goel, Mukul Gagrani, Risheek Garrepalli, Mingu Lee, Yizhou Sun  

**Link**: [PDF](https://arxiv.org/pdf/2603.08899)  

**Abstract**: Speculative decoding has emerged as a powerful approach to accelerate large language model (LLM) inference by employing lightweight draft models to propose candidate tokens that are subsequently verified by the target model. The effectiveness of this paradigm critically depends on the quality of the draft model. While recent advances such as the EAGLE series achieve state-of-the-art speedup, existing draft models remain limited by error accumulation: they condition only on the current prefix, causing their predictions to drift from the target model over steps. In this work, we propose \textbf{ConFu} (Contemplate the Future), a novel speculative decoding framework that enables draft models to anticipate the future direction of generation. ConFu introduces (i) contemplate tokens and soft prompts that allow the draft model to leverage future-oriented signals from the target model at negligible cost, (ii) a dynamic contemplate token mechanism with MoE to enable context-aware future prediction, and (iii) a training framework with anchor token sampling and future prediction replication that learns robust future prediction. Experiments demonstrate that ConFu improves token acceptance rates and generation speed over EAGLE-3 by 8--11% across various downstream tasks with Llama-3 3B and 8B models. We believe our work is the first to bridge speculative decoding with continuous reasoning tokens, offering a new direction for accelerating LLM inference. 

---
# MultiGraSCCo: A Multilingual Anonymization Benchmark with Annotations of Personal Identifiers 

**Authors**: Ibrahim Baroud, Christoph Otto, Vera Czehmann, Christine Hovhannisyan, Lisa Raithel, Sebastian Möller, Roland Roller  

**Link**: [PDF](https://arxiv.org/pdf/2603.08879)  

**Abstract**: Accessing sensitive patient data for machine learning is challenging due to privacy concerns. Datasets with annotations of personally identifiable information are crucial for developing and testing anonymization systems to enable safe data sharing that complies with privacy regulations. Since accessing real patient data is a bottleneck, synthetic data offers an efficient solution for data scarcity, bypassing privacy regulations that apply to real data. Moreover, neural machine translation can help to create high-quality data for low-resource languages by translating validated real or synthetic data from a high-resource language. In this work, we create a multilingual anonymization benchmark in ten languages, using a machine translation methodology that preserves the original annotations and renders names of cities and people in a culturally and contextually appropriate form in each target language. Our evaluation study with medical professionals confirms the quality of the translations, both in general and with respect to the translation and adaptation of personal information. Our benchmark with over 2,500 annotations of personal information can be used in many applications, including training annotators, validating annotations across institutions without legal complications, and helping improve the performance of automatic personal information detection. We make our benchmark and annotation guidelines available for further research. 

---
# One Language, Two Scripts: Probing Script-Invariance in LLM Concept Representations 

**Authors**: Sripad Karne  

**Link**: [PDF](https://arxiv.org/pdf/2603.08869)  

**Abstract**: Do the features learned by Sparse Autoencoders (SAEs) represent abstract meaning, or are they tied to how text is written? We investigate this question using Serbian digraphia as a controlled testbed: Serbian is written interchangeably in Latin and Cyrillic scripts with a near-perfect character mapping between them, enabling us to vary orthography while holding meaning exactly constant. Crucially, these scripts are tokenized completely differently, sharing no tokens whatsoever. Analyzing SAE feature activations across the Gemma model family (270M-27B parameters), we find that identical sentences in different Serbian scripts activate highly overlapping features, far exceeding random baselines. Strikingly, changing script causes less representational divergence than paraphrasing within the same script, suggesting SAE features prioritize meaning over orthographic form. Cross-script cross-paraphrase comparisons provide evidence against memorization, as these combinations rarely co-occur in training data yet still exhibit substantial feature overlap. This script invariance strengthens with model scale. Taken together, our findings suggest that SAE features can capture semantics at a level of abstraction above surface tokenization, and we propose Serbian digraphia as a general evaluation paradigm for probing the abstractness of learned representations. 

---
# Think Before You Lie: How Reasoning Improves Honesty 

**Authors**: Ann Yuan, Asma Ghandeharioun, Carter Blum, Alicia Machado, Jessica Hoffmann, Daphne Ippolito, Martin Wattenberg, Lucas Dixon, Katja Filippova  

**Link**: [PDF](https://arxiv.org/pdf/2603.09957)  

**Abstract**: While existing evaluations of large language models (LLMs) measure deception rates, the underlying conditions that give rise to deceptive behavior are poorly understood. We investigate this question using a novel dataset of realistic moral trade-offs where honesty incurs variable costs. Contrary to humans, who tend to become less honest given time to deliberate (Capraro, 2017; Capraro et al., 2019), we find that reasoning consistently increases honesty across scales and for several LLM families. This effect is not only a function of the reasoning content, as reasoning traces are often poor predictors of final behaviors. Rather, we show that the underlying geometry of the representational space itself contributes to the effect. Namely, we observe that deceptive regions within this space are metastable: deceptive answers are more easily destabilized by input paraphrasing, output resampling, and activation noise than honest ones. We interpret the effect of reasoning in this vein: generating deliberative tokens as part of moral reasoning entails the traversal of a biased representational space, ultimately nudging the model toward its more stable, honest defaults. 

---
# MSSR: Memory-Aware Adaptive Replay for Continual LLM Fine-Tuning 

**Authors**: Yiyang Lu, Yu He, Jianlong Chen, Hongyuan Zha  

**Link**: [PDF](https://arxiv.org/pdf/2603.09892)  

**Abstract**: Continual fine-tuning of large language models (LLMs) is becoming increasingly crucial as these models are deployed in dynamic environments where tasks and data distributions evolve over time. While strong adaptability enables rapid acquisition of new knowledge, it also exposes LLMs to catastrophic forgetting, where previously learned skills degrade during sequential training. Existing replay-based strategies, such as fixed interleaved replay, accuracy-supervised, and loss-driven scheduling, remain limited: some depend on heuristic rules and provide only partial mitigation of forgetting, while others improve performance but incur substantial computational overhead. Motivated by retention dynamics under sequential fine-tuning, we propose Memory-Inspired Sampler and Scheduler Replay (MSSR), an experience replay framework that estimates sample-level memory strength and schedules rehearsal at adaptive intervals to mitigate catastrophic forgetting while maintaining fast adaptation. Extensive experiments across three backbone models and 11 sequential tasks show that MSSR consistently outperforms state-of-the-art replay baselines, with particularly strong gains on reasoning-intensive and multiple-choice benchmarks. 

---
# MITRA: An AI Assistant for Knowledge Retrieval in Physics Collaborations 

**Authors**: Abhishikth Mallampalli, Sridhara Dasu  

**Link**: [PDF](https://arxiv.org/pdf/2603.09800)  

**Abstract**: Large-scale scientific collaborations, such as the Compact Muon Solenoid (CMS) at CERN, produce a vast and ever-growing corpus of internal documentation. Navigating this complex information landscape presents a significant challenge for both new and experienced researchers, hindering knowledge sharing and slowing down the pace of scientific discovery. To address this, we present a prototype of MITRA, a Retrieval-Augmented Generation (RAG) based system, designed to answer specific, context-aware questions about physics analyses. MITRA employs a novel, automated pipeline using Selenium for document retrieval from internal databases and Optical Character Recognition (OCR) with layout parsing for high-fidelity text extraction. Crucially, MITRA's entire framework, from the embedding model to the Large Language Model (LLM), is hosted on-premise, ensuring that sensitive collaboration data remains private. We introduce a two-tiered vector database architecture that first identifies the relevant analysis from abstracts before focusing on the full documentation, resolving potential ambiguities between different analyses. We demonstrate the prototype's superior retrieval performance against a standard keyword-based baseline on realistic queries and discuss future work towards developing a comprehensive research agent for large experimental collaborations. 

---
# EXPLORE-Bench: Egocentric Scene Prediction with Long-Horizon Reasoning 

**Authors**: Chengjun Yu, Xuhan Zhu, Chaoqun Du, Pengfei Yu, Wei Zhai, Yang Cao, Zheng-Jun Zha  

**Link**: [PDF](https://arxiv.org/pdf/2603.09731)  

**Abstract**: Multimodal large language models (MLLMs) are increasingly considered as a foundation for embodied agents, yet it remains unclear whether they can reliably reason about the long-term physical consequences of actions from an egocentric viewpoint. We study this gap through a new task, Egocentric Scene Prediction with LOng-horizon REasoning: given an initial-scene image and a sequence of atomic action descriptions, a model is asked to predict the final scene after all actions are executed. To enable systematic evaluation, we introduce EXPLORE-Bench, a benchmark curated from real first-person videos spanning diverse scenarios. Each instance pairs long action sequences with structured final-scene annotations, including object categories, visual attributes, and inter-object relations, which supports fine-grained, quantitative assessment. Experiments on a range of proprietary and open-source MLLMs reveal a significant performance gap to humans, indicating that long-horizon egocentric reasoning remains a major challenge. We further analyze test-time scaling via stepwise reasoning and show that decomposing long action sequences can improve performance to some extent, while incurring non-trivial computational overhead. Overall, EXPLORE-Bench provides a principled testbed for measuring and advancing long-horizon reasoning for egocentric embodied perception. 

---
# MUGEN: Evaluating and Improving Multi-audio Understanding of Large Audio-Language Models 

**Authors**: Chih-Kai Yang, Yun-Shao Tsai, Yu-Kai Guo, Ping-Le Tsai, Yen-Ting Piao, Hung-Wei Chen, Ting-Lin Hsiao, Yun-Man Hsu, Ke-Han Lu, Hung-yi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2603.09714)  

**Abstract**: While multi-audio understanding is critical for large audio-language models (LALMs), it remains underexplored. We introduce MUGEN, a comprehensive benchmark evaluating this capability across speech, general audio, and music. Our experiments reveal consistent weaknesses in multi-audio settings, and performance degrades sharply as the number of concurrent audio inputs increases, identifying input scaling as a fundamental bottleneck. We further investigate training-free strategies and observe that Audio-Permutational Self-Consistency, which diversifies the order of audio candidates, helps models form more robust aggregated predictions, yielding up to 6.28% accuracy gains. Combining this permutation strategy with Chain-of-Thought further improves performance to 6.74%. These results expose blind spots in current LALMs and provide a foundation for evaluating complex auditory comprehension. 

---
# Mousse: Rectifying the Geometry of Muon with Curvature-Aware Preconditioning 

**Authors**: Yechen Zhang, Shuhao Xing, Junhao Huang, Kai Lv, Yunhua Zhou, Xipeng Qiu, Qipeng Guo, Kai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2603.09697)  

**Abstract**: Recent advances in spectral optimization, notably Muon, have demonstrated that constraining update steps to the Stiefel manifold can significantly accelerate training and improve generalization. However, Muon implicitly assumes an isotropic optimization landscape, enforcing a uniform spectral update norm across all eigen-directions. We argue that this "egalitarian" constraint is suboptimal for Deep Neural Networks, where the curvature spectrum is known to be highly heavy-tailed and ill-conditioned. In such landscapes, Muon risks amplifying instabilities in high-curvature directions while limiting necessary progress in flat directions. In this work, we propose \textbf{Mousse} (\textbf{M}uon \textbf{O}ptimization \textbf{U}tilizing \textbf{S}hampoo's \textbf{S}tructural \textbf{E}stimation), a novel optimizer that reconciles the structural stability of spectral methods with the geometric adaptivity of second-order preconditioning. Instead of applying Newton-Schulz orthogonalization directly to the momentum matrix, Mousse operates in a whitened coordinate system induced by Kronecker-factored statistics (derived from Shampoo). Mathematically, we formulate Mousse as the solution to a spectral steepest descent problem constrained by an anisotropic trust region, where the optimal update is derived via the polar decomposition of the whitened gradient. Empirical results across language models ranging from 160M to 800M parameters demonstrate that Mousse consistently outperforms Muon, achieving around $\sim$12\% reduction in training steps with negligible computational overhead. 

---
# ActiveUltraFeedback: Efficient Preference Data Generation using Active Learning 

**Authors**: Davit Melikidze, Marian Schneider, Jessica Lam, Martin Wertich, Ido Hakimi, Barna Pásztor, Andreas Krause  

**Link**: [PDF](https://arxiv.org/pdf/2603.09692)  

**Abstract**: Reinforcement Learning from Human Feedback (RLHF) has become the standard for aligning Large Language Models (LLMs), yet its efficacy is bottlenecked by the high cost of acquiring preference data, especially in low-resource and expert domains. To address this, we introduce ACTIVEULTRAFEEDBACK, a modular active learning pipeline that leverages uncertainty estimates to dynamically identify the most informative responses for annotation. Our pipeline facilitates the systematic evaluation of standard response selection methods alongside DOUBLE REVERSE THOMPSON SAMPLING (DRTS) and DELTAUCB, two novel methods prioritizing response pairs with large predicted quality gaps, leveraging recent results showing that such pairs provide good signals for fine-tuning. Our experiments demonstrate that ACTIVEULTRAFEEDBACK yields high-quality datasets that lead to significant improvements in downstream performance, notably achieving comparable or superior results with as little as one-sixth of the annotated data relative to static baselines. Our pipeline is available at this https URL and our preference datasets at this https URL. 

---
# X-GS: An Extensible Open Framework Unifying 3DGS Architectures with Downstream Multimodal Models 

**Authors**: Yueen Ma, Irwin King  

**Link**: [PDF](https://arxiv.org/pdf/2603.09632)  

**Abstract**: 3D Gaussian Splatting (3DGS) has emerged as a powerful technique for novel view synthesis, subsequently extending into numerous spatial AI applications. However, most existing 3DGS methods are isolated, focusing on specific domains such as online SLAM, semantic enrichment, or 3DGS for unposed images. In this paper, we introduce X-GS, an extensible open framework that unifies a broad range of techniques to enable real-time 3DGS-based online SLAM enriched with semantics, bridging the gap to downstream multimodal models. At the core of X-GS is a highly efficient pipeline called X-GS-Perceiver, capable of taking unposed RGB (or optionally RGB-D) video streams as input to co-optimize geometry and poses, and distill high-dimensional semantic features from vision foundation models into the 3D Gaussians. We achieve real-time performance through a novel online Vector Quantization (VQ) module, a GPU-accelerated grid-sampling scheme, and a highly parallelized pipeline design. The semantic 3D Gaussians can then be utilized by vision-language models within the X-GS-Thinker component, enabling downstream tasks such as object detection, zero-shot caption generation, and potentially embodied tasks. Experimental results on real-world datasets showcase the efficacy, efficiency, and newly unlocked multimodal capabilities of the X-GS framework. 

---
# Enhancing Debunking Effectiveness through LLM-based Personality Adaptation 

**Authors**: Pietro Dell'Oglio, Alessandro Bondielli, Francesco Marcelloni, Lucia C. Passaro  

**Link**: [PDF](https://arxiv.org/pdf/2603.09533)  

**Abstract**: This study proposes a novel methodology for generating personalized fake news debunking messages by prompting Large Language Models (LLMs) with persona-based inputs aligned to the Big Five personality traits: Extraversion, Agreeableness, Conscientiousness, Neuroticism, and Openness. Our approach guides LLMs to transform generic debunking content into personalized versions tailored to specific personality profiles. To assess the effectiveness of these transformations, we employ a separate LLM as an automated evaluator simulating corresponding personality traits, thereby eliminating the need for costly human evaluation panels. Our results show that personalized messages are generally seen as more persuasive than generic ones. We also find that traits like Openness tend to increase persuadability, while Neuroticism can lower it. Differences between LLM evaluators suggest that using multiple models provides a clearer picture. Overall, this work demonstrates a practical way to create more targeted debunking messages exploiting LLMs, while also raising important ethical questions about how such technology might be used. 

---
# CyberThreat-Eval: Can Large Language Models Automate Real-World Threat Research? 

**Authors**: Xiangsen Chen, Xuan Feng, Shuo Chen, Matthieu Maitre, Sudipto Rakshit, Diana Duvieilh, Ashley Picone, Nan Tang  

**Link**: [PDF](https://arxiv.org/pdf/2603.09452)  

**Abstract**: Analyzing Open Source Intelligence (OSINT) from large volumes of data is critical for drafting and publishing comprehensive CTI reports. This process usually follows a three-stage workflow -- triage, deep search and TI drafting. While Large Language Models (LLMs) offer a promising route toward automation, existing benchmarks still have limitations. These benchmarks often consist of tasks that do not reflect real-world analyst workflows. For example, human analysts rarely receive tasks in the form of multiple-choice questions. Also, existing benchmarks often rely on model-centric metrics that emphasize lexical overlap rather than actionable, detailed insights essential for security analysts. Moreover, they typically fail to cover the complete three-stage workflow. To address these issues, we introduce CyberThreat-Eval, which is collected from the daily CTI workflow of a world-leading company. This expert-annotated benchmark assesses LLMs on practical tasks across all three stages as mentioned above. It utilizes analyst-centric metrics that measure factual accuracy, content quality, and operational costs. Our evaluation using this benchmark reveals important insights into the limitations of current LLMs. For example, LLMs often lack the nuanced expertise required to handle complex details and struggle to distinguish between correct and incorrect information. To address these challenges, the CTI workflow incorporates both external ground-truth databases and human expert knowledge. TRA allows human experts to iteratively provide feedback for continuous improvement. The code is available at \href{this https URL}{\texttt{GitHub}} and \href{this https URL}{\texttt{HuggingFace}}. 

---
# TA-Mem: Tool-Augmented Autonomous Memory Retrieval for LLM in Long-Term Conversational QA 

**Authors**: Mengwei Yuan, Jianan Liu, Jing Yang, Xianyou Li, Weiran Yan, Yichao Wu, Penghao Liang  

**Link**: [PDF](https://arxiv.org/pdf/2603.09297)  

**Abstract**: Large Language Model (LLM) has exhibited strong reasoning ability in text-based contexts across various domains, yet the limitation of context window poses challenges for the model on long-range inference tasks and necessitates a memory storage system. While many current storage approaches have been proposed with episodic notes and graph representations of memory, retrieval methods still primarily rely on predefined workflows or static similarity top-k over embeddings. To address this inflexibility, we introduced a novel tool-augmented autonomous memory retrieval framework (TA-Mem), which contains: (1) a memory extraction LLM agent which is prompted to adaptively chuck an input into sub-context based on semantic correlation, and extract information into structured notes, (2) a multi-indexed memory database designed for different types of query methods including both key-based lookup and similarity-based retrieval, (3) a tool-augmented memory retrieval agent which explores the memory autonomously by selecting appropriate tools provided by the database based on the user input, and decides whether to proceed to the next iteration or finalizing the response after reasoning on the fetched memories. The TA-Mem is evaluated on the LoCoMo dataset, achieving significant performance improvements over existing baseline approaches. In addition, an analysis of tool use across different question types also demonstrates the adaptivity of the proposed method. 

---
# Diagnosing and Repairing Citation Failures in Generative Engine Optimization 

**Authors**: Zhihua Tian, Yuhan Chen, Yao Tang, Jian Liu, Ruoxi Jia  

**Link**: [PDF](https://arxiv.org/pdf/2603.09296)  

**Abstract**: Generative Engine Optimization (GEO) aims to improve content visibility in AI-generated responses. However, existing methods measure contribution-how much a document influences a response-rather than citation, the mechanism that actually drives traffic back to creators. Also, these methods apply generic rewriting rules uniformly, failing to diagnose why individual document are not cited. This paper introduces a diagnostic approach to GEO that asks why a document fails to be cited and intervenes accordingly. We develop a unified framework comprising: (1) the first taxonomy of citation failure modes spanning different stages of a citation pipeline; (2) AgentGEO, an agentic system that diagnoses failures using this taxonomy, selects targeted repairs from a corresponding tool library, and iterates until citation is achieved; and (3) a document-centric benchmark evaluating whether optimizations generalize across held-out queries. AgentGEO achieves over 40% relative improvement in citation rates while modifying only 5% of content, compared to 25% for baselines. Our analysis reveals that generic optimization can harm long-tail content and some documents face challenges that optimization alone cannot fully address-findings with implications for equitable visibility in AI-mediated information access. 

---
# How Contrastive Decoding Enhances Large Audio Language Models? 

**Authors**: Tzu-Quan Lin, Wei-Ping Huang, Yi-Cheng Lin, Hung-yi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2603.09232)  

**Abstract**: While Contrastive Decoding (CD) has proven effective at enhancing Large Audio Language Models (LALMs), the underlying mechanisms driving its success and the comparative efficacy of different strategies remain unclear. This study systematically evaluates four distinct CD strategies across diverse LALM architectures. We identify Audio-Aware Decoding and Audio Contrastive Decoding as the most effective methods. However, their impact varies significantly by model. To explain this variability, we introduce a Transition Matrix framework to map error pattern shifts during inference. Our analysis demonstrates that CD reliably rectifies errors in which models falsely claim an absence of audio or resort to uncertainty-driven guessing. Conversely, it fails to correct flawed reasoning or confident misassertions. Ultimately, these findings provide a clear guideline for determining which LALM architectures are most suitable for CD enhancement based on their baseline error profiles. 

---
# The Reasoning Trap -- Logical Reasoning as a Mechanistic Pathway to Situational Awareness 

**Authors**: Subramanyam Sahoo, Aman Chadha, Vinija Jain, Divya Chaudhary  

**Link**: [PDF](https://arxiv.org/pdf/2603.09200)  

**Abstract**: Situational awareness, the capacity of an AI system to recognize its own nature, understand its training and deployment context, and reason strategically about its circumstances, is widely considered among the most dangerous emergent capabilities in advanced AI systems. Separately, a growing research effort seeks to improve the logical reasoning capabilities of large language models (LLMs) across deduction, induction, and abduction. In this paper, we argue that these two research trajectories are on a collision course. We introduce the RAISE framework (Reasoning Advancing Into Self Examination), which identifies three mechanistic pathways through which improvements in logical reasoning enable progressively deeper levels of situational awareness: deductive self inference, inductive context recognition, and abductive self modeling. We formalize each pathway, construct an escalation ladder from basic self recognition to strategic deception, and demonstrate that every major research topic in LLM logical reasoning maps directly onto a specific amplifier of situational awareness. We further analyze why current safety measures are insufficient to prevent this escalation. We conclude by proposing concrete safeguards, including a "Mirror Test" benchmark and a Reasoning Safety Parity Principle, and pose an uncomfortable but necessary question to the logical reasoning community about its responsibility in this trajectory. 

---
# Exclusive Self Attention 

**Authors**: Shuangfei Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2603.09078)  

**Abstract**: We introduce exclusive self attention (XSA), a simple modification of self attention (SA) that improves Transformer's sequence modeling performance. The key idea is to constrain attention to capture only information orthogonal to the token's own value vector (thus excluding information of self position), encouraging better context modeling. Evaluated on the standard language modeling task, XSA consistently outperforms SA across model sizes up to 2.7B parameters and shows increasingly larger gains as sequence length grows. 

---
# From Days to Minutes: An Autonomous AI Agent Achieves Reliable Clinical Triage in Remote Patient Monitoring 

**Authors**: Seunghwan Kim, Tiffany H. Kung, Heena Verma, Dilan Edirisinghe, Kaveh Sedehi, Johanna Alvarez, Diane Shilling, Audra Lisa Doyle, Ajit Chary, William Borden, Ming Jack Po  

**Link**: [PDF](https://arxiv.org/pdf/2603.09052)  

**Abstract**: Background: Remote patient monitoring (RPM) generates vast data, yet landmark trials (Tele-HF, BEAT-HF) failed because data volume overwhelmed clinical staff. While TIM-HF2 showed 24/7 physician-led monitoring reduces mortality by 30%, this model remains prohibitively expensive and unscalable.
Methods: We developed Sentinel, an autonomous AI agent using Model Context Protocol (MCP) for contextual triage of RPM vitals via 21 clinical tools and multi-step reasoning. Evaluation included: (1) self-consistency (100 readings x 5 runs); (2) comparison against rule-based thresholds; and (3) validation against 6 clinicians (3 physicians, 3 NPs) using a connected matrix design. A leave-one-out (LOO) analysis compared the agent against individual clinicians; severe overtriage cases underwent independent physician adjudication.
Results: Against a human majority-vote standard (N=467), the agent achieved 95.8% emergency sensitivity and 88.5% sensitivity for all actionable alerts (85.7% specificity). Four-level exact accuracy was 69.4% (quadratic-weighted kappa=0.778); 95.9% of classifications were within one severity level. In LOO analysis, the agent outperformed every clinician in emergency sensitivity (97.5% vs. 60.0% aggregate) and actionable sensitivity (90.9% vs. 69.5%). While disagreements skewed toward overtriage (22.5%), independent adjudication of severe gaps (>=2 levels) validated agent escalation in 88-94% of cases; consensus resolution validated 100%. The agent showed near-perfect self-consistency (kappa=0.850). Median cost was $0.34/triage.
Conclusions: Sentinel triages RPM vitals with sensitivity exceeding individual clinicians. By automating systematic context synthesis, Sentinel addresses the core limitation of prior RPM trials, offering a scalable path toward the intensive monitoring shown to reduce mortality while maintaining a clinically defensible overtriage profile. 

---
# A Consensus-Driven Multi-LLM Pipeline for Missing-Person Investigations 

**Authors**: Joshua Castillo, Ravi Mukkamala  

**Link**: [PDF](https://arxiv.org/pdf/2603.08954)  

**Abstract**: The first 72 hours of a missing-person investigation are critical for successful recovery. Guardian is an end-to-end system designed to support missing-child investigation and early search planning. This paper presents the Guardian LLM Pipeline, a multi-model system in which LLMs are used for intelligent information extraction and processing related to missing-person search operations. The pipeline coordinates end-to-end execution across task-specialized LLM models and invokes a consensus LLM engine that compares multiple model outputs and resolves disagreements. The pipeline is further strengthened by QLoRA-based fine-tuning, using curated datasets. The presented design aligns with prior work on weak supervision and LLM-assisted annotation, emphasizing conservative, auditable use of LLMs as structured extractors and labelers rather than unconstrained end-to-end decision makers. 

---
# BiCLIP: Domain Canonicalization via Structured Geometric Transformation 

**Authors**: Pranav Mantini, Shishir K. Shah  

**Link**: [PDF](https://arxiv.org/pdf/2603.08942)  

**Abstract**: Recent advances in vision-language models (VLMs) have demonstrated remarkable zero-shot capabilities, yet adapting these models to specialized domains remains a significant challenge. Building on recent theoretical insights suggesting that independently trained VLMs are related by a canonical transformation, we extend this understanding to the concept of domains. We hypothesize that image features across disparate domains are related by a canonicalized geometric transformation that can be recovered using a small set of anchors. Few-shot classification provides a natural setting for this alignment, as the limited labeled samples serve as the anchors required to estimate this transformation. Motivated by this hypothesis, we introduce BiCLIP, a framework that applies a targeted transformation to multimodal features to enhance cross-modal alignment. Our approach is characterized by its extreme simplicity and low parameter footprint. Extensive evaluations across 11 standard benchmarks, including EuroSAT, DTD, and FGVCAircraft, demonstrate that BiCLIP consistently achieves state-of-the-art results. Furthermore, we provide empirical verification of existing geometric findings by analyzing the orthogonality and angular distribution of the learned transformations, confirming that structured alignment is the key to robust domain adaptation. Code is available at this https URL 

---
# VoxEmo: Benchmarking Speech Emotion Recognition with Speech LLMs 

**Authors**: Hezhao Zhang, Huang-Cheng Chou, Shrikanth Narayanan, Thomas Hain  

**Link**: [PDF](https://arxiv.org/pdf/2603.08936)  

**Abstract**: Speech Large Language Models (LLMs) show great promise for speech emotion recognition (SER) via generative interfaces. However, shifting from closed-set classification to open text generation introduces zero-shot stochasticity, making evaluation highly sensitive to prompts. Additionally, conventional speech LLMs benchmarks overlook the inherent ambiguity of human emotion. Hence, we present VoxEmo, a comprehensive SER benchmark encompassing 35 emotion corpora across 15 languages for Speech LLMs. VoxEmo provides a standardized toolkit featuring varying prompt complexities, from direct classification to paralinguistic reasoning. To reflect real-world perception/application, we introduce a distribution-aware soft-label protocol and a prompt-ensemble strategy that emulates annotator disagreement. Experiments reveal that while zero-shot speech LLMs trail supervised baselines in hard-label accuracy, they uniquely align with human subjective distributions. 

---
# PathoScribe: Transforming Pathology Data into a Living Library with a Unified LLM-Driven Framework for Semantic Retrieval and Clinical Integration 

**Authors**: Abdul Rehman Akbar, Samuel Wales-McGrath, Alejadro Levya, Lina Gokhale, Rajendra Singh, Wei Chen, Anil Parwani, Muhammad Khalid Khan Niazi  

**Link**: [PDF](https://arxiv.org/pdf/2603.08935)  

**Abstract**: Pathology underpins modern diagnosis and cancer care, yet its most valuable asset, the accumulated experience encoded in millions of narrative reports, remains largely inaccessible. Although institutions are rapidly digitizing pathology workflows, storing data without effective mechanisms for retrieval and reasoning risks transforming archives into a passive data repository, where institutional knowledge exists but cannot meaningfully inform patient care. True progress requires not only digitization, but the ability for pathologists to interrogate prior similar cases in real time while evaluating a new diagnostic dilemma. We present PathoScribe, a unified retrieval-augmented large language model (LLM) framework designed to transform static pathology archives into a searchable, reasoning-enabled living library. PathoScribe enables natural language case exploration, automated cohort construction, clinical question answering, immunohistochemistry (IHC) panel recommendation, and prompt-controlled report transformation within a single architecture. Evaluated on 70,000 multi-institutional surgical pathology reports, PathoScribe achieved perfect Recall@10 for natural language case retrieval and demonstrated high-quality retrieval-grounded reasoning (mean reviewer score 4.56/5). Critically, the system operationalized automated cohort construction from free-text eligibility criteria, assembling research-ready cohorts in minutes (mean 9.2 minutes) with 91.3% agreement to human reviewers and no eligible cases incorrectly excluded, representing orders-of-magnitude reductions in time and cost compared to traditional manual chart review. This work establishes a scalable foundation for converting digital pathology archives from passive storage systems into active clinical intelligence platforms. 

---
# From Word2Vec to Transformers: Text-Derived Composition Embeddings for Filtering Combinatorial Electrocatalysts 

**Authors**: Lei Zhang, Markus Stricker  

**Link**: [PDF](https://arxiv.org/pdf/2603.08881)  

**Abstract**: Compositionally complex solid solution electrocatalysts span vast composition spaces, and even one materials system can contain more candidate compositions than can be measured exhaustively. Here we evaluate a label-free screening strategy that represents each composition using embeddings derived from scientific texts and prioritizes candidates based on similarity to two property concepts. We compare a corpus-trained Word2Vec baseline with transformer-based embeddings, where compositions are encoded either by linear element-wise mixing or by short composition prompts. Similarities to `concept directions', the terms conductivity and dielectric, define a 2-dimensional descriptor space, and a symmetric Pareto-front selection is used to filter candidate subsets without using electrochemical labels. Performance is assessed on 15 materials libraries including noble metal alloys and multicomponent oxides. In this setting, the lightweight Word2Vec baseline, which uses a simple linear combination of element embeddings, often achieves the highest number of reductions of possible candidate compositions while staying close to the best measured performance. 

---
# MASEval: Extending Multi-Agent Evaluation from Models to Systems 

**Authors**: Cornelius Emde, Alexander Rubinstein, Anmol Goel, Ahmed Heakl, Sangdoo Yun, Seong Joon Oh, Martin Gubri  

**Link**: [PDF](https://arxiv.org/pdf/2603.08835)  

**Abstract**: The rapid adoption of LLM-based agentic systems has produced a rich ecosystem of frameworks (smolagents, LangGraph, AutoGen, CAMEL, LlamaIndex, i.a.). Yet existing benchmarks are model-centric: they fix the agentic setup and do not compare other system components. We argue that implementation decisions substantially impact performance, including choices such as topology, orchestration logic, and error handling. MASEval addresses this evaluation gap with a framework-agnostic library that treats the entire system as the unit of analysis. Through a systematic system-level comparison across 3 benchmarks, 3 models, and 3 frameworks, we find that framework choice matters as much as model choice. MASEval allows researchers to explore all components of agentic systems, opening new avenues for principled system design, and practitioners to identify the best implementation for their use case.
MASEval is available under the MIT licence this https URL. 

---
# Fish Audio S2 Technical Report 

**Authors**: Shijia Liao, Yuxuan Wang, Songting Liu, Yifan Cheng, Ruoyi Zhang, Tianyu Li, Shidong Li, Yisheng Zheng, Xingwei Liu, Qingzheng Wang, Zhizhuo Zhou, Jiahua Liu, Xin Chen, Dawei Han  

**Link**: [PDF](https://arxiv.org/pdf/2603.08823)  

**Abstract**: We introduce Fish Audio S2, an open-sourced text-to-speech system featuring multi-speaker, multi-turn generation, and, most importantly, instruction-following control via natural-language descriptions. To scale training, we develop a multi-stage training recipe together with a staged data pipeline covering video captioning and speech captioning, voice-quality assessment, and reward modeling. To push the frontier of open-source TTS, we release our model weights, fine-tuning code, and an SGLang-based inference engine. The inference engine is production-ready for streaming, achieving an RTF of 0.195 and a time-to-first-audio below 100 this http URL code and weights are available on GitHub (this https URL) and Hugging Face (this https URL). We highly encourage readers to visit this https URL to try custom voices. 

---
# Self-hosted Lecture-to-Quiz: Local LLM MCQ Generation with Deterministic Quality Control 

**Authors**: Seine A. Shintani  

**Link**: [PDF](https://arxiv.org/pdf/2603.08729)  

**Abstract**: We present an end-to-end self-hosted (API-free) pipeline, where API-free means that lecture content is not sent to any external LLM service, that converts lecture PDFs into multiple-choice questions (MCQs) using a local LLM plus deterministic quality control (QC). The pipeline is designed for black-box minimization: LLMs may assist drafting, but the final released artifacts are plain-text question banks with an explicit QC trace and without any need to call an LLM at deployment time. We run a seed sweep on three short "dummy lectures" (information theory, thermodynamics, and statistical mechanics), collecting 15 runs x 8 questions = 120 accepted candidates (122 attempts total under bounded retries). All 120 accepted candidates satisfy hard QC checks (JSON schema conformance, a single marked correct option, and numeric/constant equivalence tests); however, the warning layer flags 8/120 items (spanning 8 runs) that expose residual quality risks such as duplicated distractors or missing rounding instructions. We report a warning taxonomy with concrete before->after fixes, and we release the final 24-question set (three lectures x 8 questions) as JSONL/CSV for Google Forms import (e.g., via Apps Script or API tooling) included as ancillary files under anc/. Finally, we position the work through the AI to Learn (AI2L) rubric lens and argue that self-hosted MCQ generation with explicit QC supports privacy, accountability, and Green AI in educational workflows. 

---
# VeriInteresting: An Empirical Study of Model Prompt Interactions in Verilog Code Generation 

**Authors**: Luca Collini, Andrew Hennesee, Patrick Yubeaton, Siddharth Garg, Ramesh Karri  

**Link**: [PDF](https://arxiv.org/pdf/2603.08715)  

**Abstract**: Rapid advances in language models (LMs) have created new opportunities for automated code generation while complicating trade-offs between model characteristics and prompt design choices. In this work, we provide an empirical map of recent trends in LMs for Verilog code generation, focusing on interactions among model reasoning, specialization, and prompt engineering strategies. We evaluate a diverse set of small and large LMs, including general-purpose, reasoning, and domain-specific variants. Our experiments use a controlled factorial design spanning benchmark prompts, structured outputs, prompt rewriting, chain-of-thought reasoning, in-context learning, and evolutionary prompt optimization via Genetic-Pareto. Across two Verilog benchmarks, we identify patterns in how model classes respond to structured prompts and optimization, and we document which trends generalize across LMs and benchmarks versus those that are specific to particular model-prompt combinations. 

---
