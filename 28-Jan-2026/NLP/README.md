# Evaluation of Oncotimia: An LLM based system for supporting tumour boards 

**Authors**: Luis Lorenzo, Marcos Montana-Mendez, Sergio Figueiras, Miguel Boubeta, Cristobal Bernardo-Castineira  

**Link**: [PDF](https://arxiv.org/pdf/2601.19899)  

**Abstract**: Multidisciplinary tumour boards (MDTBs) play a central role in oncology decision-making but require manual processes and structuring large volumes of heterogeneous clinical information, resulting in a substantial documentation burden. In this work, we present ONCOTIMIA, a modular and secure clinical tool designed to integrate generative artificial intelligence (GenAI) into oncology workflows and evaluate its application to the automatic completion of lung cancer tumour board forms using large language models (LLMs). The system combines a multi-layer data lake, hybrid relational and vector storage, retrieval-augmented generation (RAG) and a rule-driven adaptive form model to transform unstructured clinical documentation into structured and standardised tumour board records. We assess the performance of six LLMs deployed through AWS Bedrock on ten lung cancer cases, measuring both completion form accuracy and end-to-end latency. The results demonstrate high performance across models, with the best performing configuration achieving an 80% of correct field completion and clinically acceptable response time for most LLMs. Larger and more recent models exhibit best accuracies without incurring prohibitive latency. These findings provide empirical evidence that LLM- assisted autocompletion form is technically feasible and operationally viable in multidisciplinary lung cancer workflows and support its potential to significantly reduce documentation burden while preserving data quality. 

---
# Reflective Translation: Improving Low-Resource Machine Translation via Structured Self-Reflection 

**Authors**: Nicholas Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2601.19871)  

**Abstract**: Low-resource languages such as isiZulu and isiXhosa face persistent challenges in machine translation due to limited parallel data and linguistic resources. Recent advances in large language models suggest that self-reflection, prompting a model to critique and revise its own outputs, can improve reasoning quality and factual consistency. Building on this idea, this paper introduces Reflective Translation, a prompt-based framework in which a model generates an initial translation, produces a structured self-critique, and then uses this reflection to generate a refined translation. The approach is evaluated on English-isiZulu and English-isiXhosa translation using OPUS-100 and NTREX-African, across multiple prompting strategies and confidence thresholds. Results show consistent improvements in both BLEU and COMET scores between first- and second-pass translations, with average gains of up to +0.22 BLEU and +0.18 COMET. Statistical significance testing using paired nonparametric tests confirms that these improvements are robust. The proposed method is model-agnostic, requires no fine-tuning, and introduces a reflection-augmented dataset that can support future supervised or analysis-driven work. These findings demonstrate that structured self-reflection is a practical and effective mechanism for improving translation quality in low-resource settings. 

---
# Identifying and Transferring Reasoning-Critical Neurons: Improving LLM Inference Reliability via Activation Steering 

**Authors**: Fangan Dong, Zuming Yan, Xuri Ge, Zhiwei Xu, Mengqi Zhang, Xuanang Chen, Ben He, Xin Xin, Zhumin Chen, Ying Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2601.19847)  

**Abstract**: Despite the strong reasoning capabilities of recent large language models (LLMs), achieving reliable performance on challenging tasks often requires post-training or computationally expensive sampling strategies, limiting their practical efficiency. In this work, we first show that a small subset of neurons in LLMs exhibits strong predictive correlations with reasoning correctness. Based on this observation, we propose AdaRAS (Adaptive Reasoning Activation Steering), a lightweight test-time framework that improves reasoning reliability by selectively intervening on neuron activations. AdaRAS identifies Reasoning-Critical Neurons (RCNs) via a polarity-aware mean-difference criterion and adaptively steers their activations during inference, enhancing incorrect reasoning traces while avoiding degradation on already-correct cases. Experiments on 10 mathematics and coding benchmarks demonstrate consistent improvements, including over 13% gains on AIME-24 and AIME-25. Moreover, AdaRAS exhibits strong transferability across datasets and scalability to stronger models, outperforming post-training methods without additional training or sampling cost. 

---
# When Iterative RAG Beats Ideal Evidence: A Diagnostic Study in Scientific Multi-hop Question Answering 

**Authors**: Mahdi Astaraki, Mohammad Arshi Saloot, Ali Shiraee Kasmaee, Hamidreza Mahyar, Soheila Samiee  

**Link**: [PDF](https://arxiv.org/pdf/2601.19827)  

**Abstract**: Retrieval-Augmented Generation (RAG) extends large language models (LLMs) beyond parametric knowledge, yet it is unclear when iterative retrieval-reasoning loops meaningfully outperform static RAG, particularly in scientific domains with multi-hop reasoning, sparse domain knowledge, and heterogeneous evidence. We provide the first controlled, mechanism-level diagnostic study of whether synchronized iterative retrieval and reasoning can surpass an idealized static upper bound (Gold Context) RAG. We benchmark eleven state-of-the-art LLMs under three regimes: (i) No Context, measuring reliance on parametric memory; (ii) Gold Context, where all oracle evidence is supplied at once; and (iii) Iterative RAG, a training-free controller that alternates retrieval, hypothesis refinement, and evidence-aware stopping. Using the chemistry-focused ChemKGMultiHopQA dataset, we isolate questions requiring genuine retrieval and analyze behavior with diagnostics spanning retrieval coverage gaps, anchor-carry drop, query quality, composition fidelity, and control calibration. Across models, Iterative RAG consistently outperforms Gold Context, with gains up to 25.6 percentage points, especially for non-reasoning fine-tuned models. Staged retrieval reduces late-hop failures, mitigates context overload, and enables dynamic correction of early hypothesis drift, but remaining failure modes include incomplete hop coverage, distractor latch trajectories, early stopping miscalibration, and high composition failure rates even with perfect retrieval. Overall, staged retrieval is often more influential than the mere presence of ideal evidence; we provide practical guidance for deploying and diagnosing RAG systems in specialized scientific settings and a foundation for more reliable, controllable iterative retrieval-reasoning frameworks. 

---
# Zero-Shot Stance Detection in the Wild: Dynamic Target Generation and Multi-Target Adaptation 

**Authors**: Aohua Li, Yuanshuo Zhang, Ge Gao, Bo Chen, Xiaobing Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2601.19802)  

**Abstract**: Current stance detection research typically relies on predicting stance based on given targets and text. However, in real-world social media scenarios, targets are neither predefined nor static but rather complex and dynamic. To address this challenge, we propose a novel task: zero-shot stance detection in the wild with Dynamic Target Generation and Multi-Target Adaptation (DGTA), which aims to automatically identify multiple target-stance pairs from text without prior target knowledge. We construct a Chinese social media stance detection dataset and design multi-dimensional evaluation metrics. We explore both integrated and two-stage fine-tuning strategies for large language models (LLMs) and evaluate various baseline models. Experimental results demonstrate that fine-tuned LLMs achieve superior performance on this task: the two-stage fine-tuned Qwen2.5-7B attains the highest comprehensive target recognition score of 66.99%, while the integrated fine-tuned DeepSeek-R1-Distill-Qwen-7B achieves a stance detection F1 score of 79.26%. 

---
# LVLMs and Humans Ground Differently in Referential Communication 

**Authors**: Peter Zeng, Weiling Li, Amie Paige, Zhengxiang Wang, Panagiotis Kaliosis, Dimitris Samaras, Gregory Zelinsky, Susan Brennan, Owen Rambow  

**Link**: [PDF](https://arxiv.org/pdf/2601.19792)  

**Abstract**: For generative AI agents to partner effectively with human users, the ability to accurately predict human intent is critical. But this ability to collaborate remains limited by a critical deficit: an inability to model common ground. Here, we present a referential communication experiment with a factorial design involving director-matcher pairs (human-human, human-AI, AI-human, and AI-AI) that interact with multiple turns in repeated rounds to match pictures of objects not associated with any obvious lexicalized labels. We release the online pipeline for data collection, the tools and analyses for accuracy, efficiency, and lexical overlap, and a corpus of 356 dialogues (89 pairs over 4 rounds each) that unmasks LVLMs' limitations in interactively resolving referring expressions, a crucial skill that underlies human language use. 

---
# Strong Reasoning Isn't Enough: Evaluating Evidence Elicitation in Interactive Diagnosis 

**Authors**: Zhuohan Long, Zhijie Bao, Zhongyu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2601.19773)  

**Abstract**: Interactive medical consultation requires an agent to proactively elicit missing clinical evidence under uncertainty. Yet existing evaluations largely remain static or outcome-centric, neglecting the evidence-gathering process. In this work, we propose an interactive evaluation framework that explicitly models the consultation process using a simulated patient and a \rev{simulated reporter} grounded in atomic evidences. Based on this representation, we introduce Information Coverage Rate (ICR) to quantify how completely an agent uncovers necessary evidence during interaction. To support systematic study, we build EviMed, an evidence-based benchmark spanning diverse conditions from common complaints to rare diseases, and evaluate 10 models with varying reasoning abilities. We find that strong diagnostic reasoning does not guarantee effective information collection, and this insufficiency acts as a primary bottleneck limiting performance in interactive settings. To address this, we propose REFINE, a strategy that leverages diagnostic verification to guide the agent in proactively resolving uncertainties. Extensive experiments demonstrate that REFINE consistently outperforms baselines across diverse datasets and facilitates effective model collaboration, enabling smaller agents to achieve superior performance under strong reasoning supervision. Our code can be found at this https URL . 

---
# TokenSeek: Memory Efficient Fine Tuning via Instance-Aware Token Ditching 

**Authors**: Runjia Zeng, Qifan Wang, Qiang Guan, Ruixiang Tang, Lifu Huang, Zhenting Wang, Xueling Zhang, Cheng Han, Dongfang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2601.19739)  

**Abstract**: Fine tuning has been regarded as a de facto approach for adapting large language models (LLMs) to downstream tasks, but the high training memory consumption inherited from LLMs makes this process inefficient. Among existing memory efficient approaches, activation-related optimization has proven particularly effective, as activations consistently dominate overall memory consumption. Although prior arts offer various activation optimization strategies, their data-agnostic nature ultimately results in ineffective and unstable fine tuning. In this paper, we propose TokenSeek, a universal plugin solution for various transformer-based models through instance-aware token seeking and ditching, achieving significant fine-tuning memory savings (e.g., requiring only 14.8% of the memory on Llama3.2 1B) with on-par or even better performance. Furthermore, our interpretable token seeking process reveals the underlying reasons for its effectiveness, offering valuable insights for future research on token efficiency. Homepage: this https URL 

---
# Component-Level Lesioning of Language Models Reveals Clinically Aligned Aphasia Phenotypes 

**Authors**: Yifan Wang, Jichen Zheng, Jingyuan Sun, Yunhao Zhang, Chunyu Ye, Jixing Li, Chengqing Zong, Shaonan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.19723)  

**Abstract**: Large language models (LLMs) increasingly exhibit human-like linguistic behaviors and internal representations that they could serve as computational simulators of language cognition. We ask whether LLMs can be systematically manipulated to reproduce language-production impairments characteristic of aphasia following focal brain lesions. Such models could provide scalable proxies for testing rehabilitation hypotheses, and offer a controlled framework for probing the functional organization of language. We introduce a clinically grounded, component-level framework that simulates aphasia by selectively perturbing functional components in LLMs, and apply it to both modular Mixture-of-Experts models and dense Transformers using a unified intervention interface. Our pipeline (i) identifies subtype-linked components for Broca's and Wernicke's aphasia, (ii) interprets these components with linguistic probing tasks, and (iii) induces graded impairments by progressively perturbing the top-k subtype-linked components, evaluating outcomes with Western Aphasia Battery (WAB) subtests summarized by Aphasia Quotient (AQ). Across architectures and lesioning strategies, subtype-targeted perturbations yield more systematic, aphasia-like regressions than size-matched random perturbations, and MoE modularity supports more localized and interpretable phenotype-to-component mappings. These findings suggest that modular LLMs, combined with clinically informed component perturbations, provide a promising platform for simulating aphasic language production and studying how distinct language functions degrade under targeted disruptions. 

---
# SynCABEL: Synthetic Contextualized Augmentation for Biomedical Entity Linking 

**Authors**: Adam Remaki, Christel Gérardin, Eulàlia Farré-Maduell, Martin Krallinger, Xavier Tannier  

**Link**: [PDF](https://arxiv.org/pdf/2601.19667)  

**Abstract**: We present SynCABEL (Synthetic Contextualized Augmentation for Biomedical Entity Linking), a framework that addresses a central bottleneck in supervised biomedical entity linking (BEL): the scarcity of expert-annotated training data. SynCABEL leverages large language models to generate context-rich synthetic training examples for all candidate concepts in a target knowledge base, providing broad supervision without manual annotation. We demonstrate that SynCABEL, when combined with decoder-only models and guided inference establish new state-of-the-art results across three widely used multilingual benchmarks: MedMentions for English, QUAERO for French, and SPACCC for Spanish. Evaluating data efficiency, we show that SynCABEL reaches the performance of full human supervision using up to 60% less annotated data, substantially reducing reliance on labor-intensive and costly expert labeling. Finally, acknowledging that standard evaluation based on exact code matching often underestimates clinically valid predictions due to ontology redundancy, we introduce an LLM-as-a-judge protocol. This analysis reveals that SynCABEL significantly improves the rate of clinically valid predictions. Our synthetic datasets, models, and code are released to support reproducibility and future research. 

---
# One Token Is Enough: Improving Diffusion Language Models with a Sink Token 

**Authors**: Zihou Zhang, Zheyong Xie, Li Zhong, Haifeng Liu, Shaosheng Cao  

**Link**: [PDF](https://arxiv.org/pdf/2601.19657)  

**Abstract**: Diffusion Language Models (DLMs) have emerged as a compelling alternative to autoregressive approaches, enabling parallel text generation with competitive performance. Despite these advantages, there is a critical instability in DLMs: the moving sink phenomenon. Our analysis indicates that sink tokens exhibit low-norm representations in the Transformer's value space, and that the moving sink phenomenon serves as a protective mechanism in DLMs to prevent excessive information mixing. However, their unpredictable positions across diffusion steps undermine inference robustness. To resolve this, we propose a simple but effective extra sink token implemented via a modified attention mask. Specifically, we introduce a special token constrained to attend solely to itself, while remaining globally visible to all other tokens. Experimental results demonstrate that introducing a single extra token stabilizes attention sinks, substantially improving model performance. Crucially, further analysis confirms that the effectiveness of this token is independent of its position and characterized by negligible semantic content, validating its role as a robust and dedicated structural sink. 

---
# RATE: Reviewer Profiling and Annotation-free Training for Expertise Ranking in Peer Review Systems 

**Authors**: Weicong Liu, Zixuan Yang, Yibo Zhao, Xiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.19637)  

**Abstract**: Reviewer assignment is increasingly critical yet challenging in the LLM era, where rapid topic shifts render many pre-2023 benchmarks outdated and where proxy signals poorly reflect true reviewer familiarity. We address this evaluation bottleneck by introducing LR-bench, a high-fidelity, up-to-date benchmark curated from 2024-2025 AI/NLP manuscripts with five-level self-assessed familiarity ratings collected via a large-scale email survey, yielding 1055 expert-annotated paper-reviewer-score annotations. We further propose RATE, a reviewer-centric ranking framework that distills each reviewer's recent publications into compact keyword-based profiles and fine-tunes an embedding model with weak preference supervision constructed from heuristic retrieval signals, enabling matching each manuscript against a reviewer profile directly. Across LR-bench and the CMU gold-standard dataset, our approach consistently achieves state-of-the-art performance, outperforming strong embedding baselines by a clear margin. We release LR-bench at this https URL, and a GitHub repository at this https URL. 

---
# Up to 36x Speedup: Mask-based Parallel Inference Paradigm for Key Information Extraction in MLLMs 

**Authors**: Xinzhong Wang, Ya Guo, Jing Li, Huan Chen, Yi Tu, Yijie Hong, Gongshen Liu, Huijia Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2601.19613)  

**Abstract**: Key Information Extraction (KIE) from visually-rich documents (VrDs) is a critical task, for which recent Large Language Models (LLMs) and Multi-Modal Large Language Models (MLLMs) have demonstrated strong potential. However, their reliance on autoregressive inference, which generates outputs sequentially, creates a significant efficiency bottleneck, especially as KIE tasks often involve extracting multiple, semantically independent fields. To overcome this limitation, we introduce PIP: a Parallel Inference Paradigm for KIE. Our approach reformulates the problem by using "[mask]" tokens as placeholders for all target values, enabling their simultaneous generation in a single forward pass. To facilitate this paradigm, we develop a tailored mask pre-training strategy and construct large-scale supervised datasets. Experimental results show that our PIP-models achieve a 5-36x inference speedup with negligible performance degradation compared to traditional autoregressive base models. By substantially improving efficiency while maintaining high accuracy, PIP paves the way for scalable and practical real-world KIE solutions. 

---
# Decompose-and-Formalise: Recursively Verifiable Natural Language Inference 

**Authors**: Xin Quan, Marco Valentino, Louise A. Dennis, André Freitas  

**Link**: [PDF](https://arxiv.org/pdf/2601.19605)  

**Abstract**: Recent work has shown that integrating large language models (LLMs) with theorem provers (TPs) in neuro-symbolic pipelines helps with entailment verification and proof-guided refinement of explanations for natural language inference (NLI). However, scaling such refinement to naturalistic NLI remains difficult: long, syntactically rich inputs and deep multi-step arguments amplify autoformalisation errors, where a single local mismatch can invalidate the proof. Moreover, current methods often handle failures via costly global regeneration due to the difficulty of localising the responsible span or step from prover diagnostics. Aiming to address these problems, we propose a decompose-and-formalise framework that (i) decomposes premise-hypothesis pairs into an entailment tree of atomic steps, (ii) verifies the tree bottom-up to isolate failures to specific nodes, and (iii) performs local diagnostic-guided refinement instead of regenerating the whole explanation. Moreover, to improve faithfulness of autoformalisation, we introduce $\theta$-substitution in an event-based logical form to enforce consistent argument-role bindings. Across a range of reasoning tasks using five LLM backbones, our method achieves the highest explanation verification rates, improving over the state-of-the-art by 26.2%, 21.7%, 21.6% and 48.9%, while reducing refinement iterations and runtime and preserving strong NLI accuracy. 

---
# Yunque DeepResearch Technical Report 

**Authors**: Yuxuan Cai, Xinyi Lai, Peng Yuan, Weiting Liu, Huajian Li, Mingda Li, Xinghua Wang, Shengxie Zheng, Yanchao Hao, Yuyang Yin, Zheng Wei  

**Link**: [PDF](https://arxiv.org/pdf/2601.19578)  

**Abstract**: Deep research has emerged as a transformative capability for autonomous agents, empowering Large Language Models to navigate complex, open-ended tasks. However, realizing its full potential is hindered by critical limitations, including escalating contextual noise in long-horizon tasks, fragility leading to cascading errors, and a lack of modular extensibility. To address these challenges, we introduce Yunque DeepResearch, a hierarchical, modular, and robust framework. The architecture is characterized by three key components: (1) a centralized Multi-Agent Orchestration System that routes subtasks to an Atomic Capability Pool of tools and specialized sub-agents; (2) a Dynamic Context Management mechanism that structures completed sub-goals into semantic summaries to mitigate information overload; and (3) a proactive Supervisor Module that ensures resilience through active anomaly detection and context pruning. Yunque DeepResearch achieves state-of-the-art performance across a range of agentic deep research benchmarks, including GAIA, BrowseComp, BrowseComp-ZH, and Humanity's Last Exam. We open-source the framework, reproducible implementations, and application cases to empower the community. 

---
# Automated Safety Benchmarking: A Multi-agent Pipeline for LVLMs 

**Authors**: Xiangyang Zhu, Yuan Tian, Zicheng Zhang, Qi Jia, Chunyi Li, Renrui Zhang, Heng Li, Zongrui Wang, Wei Sun  

**Link**: [PDF](https://arxiv.org/pdf/2601.19507)  

**Abstract**: Large vision-language models (LVLMs) exhibit remarkable capabilities in cross-modal tasks but face significant safety challenges, which undermine their reliability in real-world applications. Efforts have been made to build LVLM safety evaluation benchmarks to uncover their vulnerability. However, existing benchmarks are hindered by their labor-intensive construction process, static complexity, and limited discriminative power. Thus, they may fail to keep pace with rapidly evolving models and emerging risks. To address these limitations, we propose VLSafetyBencher, the first automated system for LVLM safety benchmarking. VLSafetyBencher introduces four collaborative agents: Data Preprocessing, Generation, Augmentation, and Selection agents to construct and select high-quality samples. Experiments validates that VLSafetyBencher can construct high-quality safety benchmarks within one week at a minimal cost. The generated benchmark effectively distinguish safety, with a safety rate disparity of 70% between the most and least safe models. 

---
# GradPruner: Gradient-Guided Layer Pruning Enabling Efficient Fine-Tuning and Inference for LLMs 

**Authors**: Wei Huang, Anda Cheng, Yinggui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.19503)  

**Abstract**: Fine-tuning Large Language Models (LLMs) with downstream data is often considered time-consuming and expensive. Structured pruning methods are primarily employed to improve the inference efficiency of pre-trained models. Meanwhile, they often require additional time and memory for training, knowledge distillation, structure search, and other strategies, making efficient model fine-tuning challenging to achieve. To simultaneously enhance the training and inference efficiency of downstream task fine-tuning, we introduce GradPruner, which can prune layers of LLMs guided by gradients in the early stages of fine-tuning. GradPruner uses the cumulative gradients of each parameter during the initial phase of fine-tuning to compute the Initial Gradient Information Accumulation Matrix (IGIA-Matrix) to assess the importance of layers and perform pruning. We sparsify the pruned layers based on the IGIA-Matrix and merge them with the remaining layers. Only elements with the same sign are merged to reduce interference from sign variations. We conducted extensive experiments on two LLMs across eight downstream datasets. Including medical, financial, and general benchmark tasks. The results demonstrate that GradPruner has achieved a parameter reduction of 40% with only a 0.99% decrease in accuracy. Our code is publicly available. 

---
# ClaimPT: A Portuguese Dataset of Annotated Claims in News Articles 

**Authors**: Ricardo Campos, Raquel Sequeira, Sara Nerea, Inês Cantante, Diogo Folques, Luís Filipe Cunha, João Canavilhas, António Branco, Alípio Jorge, Sérgio Nunes, Nuno Guimarães, Purificação Silvano  

**Link**: [PDF](https://arxiv.org/pdf/2601.19490)  

**Abstract**: Fact-checking remains a demanding and time-consuming task, still largely dependent on manual verification and unable to match the rapid spread of misinformation online. This is particularly important because debunking false information typically takes longer to reach consumers than the misinformation itself; accelerating corrections through automation can therefore help counter it more effectively. Although many organizations perform manual fact-checking, this approach is difficult to scale given the growing volume of digital content. These limitations have motivated interest in automating fact-checking, where identifying claims is a crucial first step. However, progress has been uneven across languages, with English dominating due to abundant annotated data. Portuguese, like other languages, still lacks accessible, licensed datasets, limiting research, NLP developments and applications. In this paper, we introduce ClaimPT, a dataset of European Portuguese news articles annotated for factual claims, comprising 1,308 articles and 6,875 individual annotations. Unlike most existing resources based on social media or parliamentary transcripts, ClaimPT focuses on journalistic content, collected through a partnership with LUSA, the Portuguese News Agency. To ensure annotation quality, two trained annotators labeled each article, with a curator validating all annotations according to a newly proposed scheme. We also provide baseline models for claim detection, establishing initial benchmarks and enabling future NLP and IR applications. By releasing ClaimPT, we aim to advance research on low-resource fact-checking and enhance understanding of misinformation in news media. 

---
# Dynamic Multi-Expert Projectors with Stabilized Routing for Multilingual Speech Recognition 

**Authors**: Isha Pandey, Ashish Mittal, Vartul Bahuguna, Ganesh Ramakrishnan  

**Link**: [PDF](https://arxiv.org/pdf/2601.19451)  

**Abstract**: Recent advances in LLM-based ASR connect frozen speech encoders with Large Language Models (LLMs) via lightweight projectors. While effective in monolingual settings, a single projector struggles to capture the diverse acoustic-to-semantic mappings required for multilingual ASR. To address this, we propose SMEAR-MoE, a stabilized Mixture-of-Experts projector that ensures dense gradient flow to all experts, preventing expert collapse while enabling cross-lingual sharing. We systematically compare monolithic, static multi-projector, and dynamic MoE designs across four Indic languages (Hindi, Marathi, Tamil, Telugu). Our SMEAR-MoE achieves strong performance, delivering upto a 7.6% relative WER reduction over the single-projector baseline, while maintaining comparable runtime efficiency. Analysis of expert routing further shows linguistically meaningful specialization, with related languages sharing experts. These results demonstrate that stable multi-expert projectors are key to scalable and robust multilingual ASR. 

---
# KG-CRAFT: Knowledge Graph-based Contrastive Reasoning with LLMs for Enhancing Automated Fact-checking 

**Authors**: Vítor N. Lourenço, Aline Paes, Tillman Weyde, Audrey Depeige, Mohnish Dubey  

**Link**: [PDF](https://arxiv.org/pdf/2601.19447)  

**Abstract**: Claim verification is a core component of automated fact-checking systems, aimed at determining the truthfulness of a statement by assessing it against reliable evidence sources such as documents or knowledge bases. This work presents KG-CRAFT, a method that improves automatic claim verification by leveraging large language models (LLMs) augmented with contrastive questions grounded in a knowledge graph. KG-CRAFT first constructs a knowledge graph from claims and associated reports, then formulates contextually relevant contrastive questions based on the knowledge graph structure. These questions guide the distillation of evidence-based reports, which are synthesised into a concise summary that is used for veracity assessment by LLMs. Extensive evaluations on two real-world datasets (LIAR-RAW and RAWFC) demonstrate that our method achieves a new state-of-the-art in predictive performance. Comprehensive analyses validate in detail the effectiveness of our knowledge graph-based contrastive reasoning approach in improving LLMs' fact-checking capabilities. 

---
# Do LLMs Truly Benefit from Longer Context in Automatic Post-Editing? 

**Authors**: Ahrii Kim, Seong-heum Kim  

**Link**: [PDF](https://arxiv.org/pdf/2601.19410)  

**Abstract**: Automatic post-editing (APE) aims to refine machine translations by correcting residual errors. Although recent large language models (LLMs) demonstrate strong translation capabilities, their effectiveness for APE--especially under document-level context--remains insufficiently understood. We present a systematic comparison of proprietary and open-weight LLMs under a naive document-level prompting setup, analyzing APE quality, contextual behavior, robustness, and efficiency.
Our results show that proprietary LLMs achieve near human-level APE quality even with simple one-shot prompting, regardless of whether document context is provided. While these models exhibit higher robustness to data poisoning attacks than open-weight counterparts, this robustness also reveals a limitation: they largely fail to exploit document-level context for contextual error correction. Furthermore, standard automatic metrics do not reliably reflect these qualitative improvements, highlighting the continued necessity of human evaluation. Despite their strong performance, the substantial cost and latency overheads of proprietary LLMs render them impractical for real-world APE deployment. Overall, our findings elucidate both the promise and current limitations of LLM-based document-aware APE, and point toward the need for more efficient long-context modeling approaches for translation refinement. 

---
# Binary Token-Level Classification with DeBERTa for All-Type MWE Identification: A Lightweight Approach with Linguistic Enhancement 

**Authors**: Diego Rossini, Lonneke van der Plas  

**Link**: [PDF](https://arxiv.org/pdf/2601.19360)  

**Abstract**: We present a comprehensive approach for multiword expression (MWE) identification that combines binary token-level classification, linguistic feature integration, and data augmentation. Our DeBERTa-v3-large model achieves 69.8% F1 on the CoAM dataset, surpassing the best results (Qwen-72B, 57.8% F1) on this dataset by 12 points while using 165x fewer parameters. We achieve this performance by (1) reformulating detection as binary token-level START/END/INSIDE classification rather than span-based prediction, (2) incorporating NP chunking and dependency features that help discontinuous and NOUN-type MWEs identification, and (3) applying oversampling that addresses severe class imbalance in the training data. We confirm the generalization of our method on the STREUSLE dataset, achieving 78.9% F1. These results demonstrate that carefully designed smaller models can substantially outperform LLMs on structured NLP tasks, with important implications for resource-constrained deployments. 

---
# Cross-Examination Framework: A Task-Agnostic Diagnostic for Information Fidelity in Text-to-Text Generation 

**Authors**: Tathagata Raha, Clement Christophe, Nada Saadi, Hamza A Javed, Marco AF Pimentel, Ronnie Rajan, Praveenkumar Kanithi  

**Link**: [PDF](https://arxiv.org/pdf/2601.19350)  

**Abstract**: Traditional metrics like BLEU and BERTScore fail to capture semantic fidelity in generative text-to-text tasks. We adapt the Cross-Examination Framework (CEF) for a reference-free, multi-dimensional evaluation by treating the source and candidate as independent knowledge bases. CEF generates verifiable questions from each text and performs a cross-examination to derive three interpretable scores: Coverage, Conformity, and Consistency. Validated across translation, summarization and clinical note-generation, our framework identifies critical errors, such as content omissions and factual contradictions, missed by standard metrics. A key contribution is a systematic robustness analysis to select a stable judge model. Crucially, the strong correlation between our reference-free and with-reference modes validates CEF's reliability without gold references. Furthermore, human expert validation demonstrates that CEF mismatching questions align with meaning-altering semantic errors higher than with non-semantic errors, particularly excelling at identifying entity-based and relational distortions. 

---
# When Benchmarks Leak: Inference-Time Decontamination for LLMs 

**Authors**: Jianzhe Chai, Yu Zhe, Jun Sakuma  

**Link**: [PDF](https://arxiv.org/pdf/2601.19334)  

**Abstract**: Benchmark-based evaluation is the de facto standard for comparing large language models (LLMs). However, its reliability is increasingly threatened by test set contamination, where test samples or their close variants leak into training data and artificially inflate reported performance. To address this issue, prior work has explored two main lines of mitigation. One line attempts to identify and remove contaminated benchmark items before evaluation, but this inevitably alters the evaluation set itself and becomes unreliable when contamination is moderate or severe. The other line preserves the benchmark and instead suppresses contaminated behavior at evaluation time; however, such interventions often interfere with normal inference and lead to noticeable performance degradation on clean inputs. We propose DeconIEP, a decontamination framework that operates entirely during evaluation by applying small, bounded perturbations in the input embedding space. Guided by a relatively less-contaminated reference model, DeconIEP learns an instance-adaptive perturbation generator that steers the evaluated model away from memorization-driven shortcut pathways. Across multiple open-weight LLMs and benchmarks, extensive empirical results show that DeconIEP achieves strong decontamination effectiveness while incurring only minimal degradation in benign utility. 

---
# Formula-One Prompting: Adaptive Reasoning Through Equations For Applied Mathematics 

**Authors**: Natapong Nitarach, Pittawat Taveekitworachai, Kunat Pipatanakul  

**Link**: [PDF](https://arxiv.org/pdf/2601.19302)  

**Abstract**: Prompting techniques such as Chain-of-Thought (CoT) and Program-of-Thought (PoT) improve LLM mathematical reasoning by structuring intermediate steps in natural language or code. However, applied mathematics problems in domains like finance, physics, and cryptography often require recalling or deriving governing equations, a step that current approaches do not explicitly leverage. We propose Formula-One Prompting (F-1), a two-phase approach that uses mathematical equations as an intermediate representation before adaptive solving. F-1 first formulates governing equations from problem descriptions, then selects a solving strategy among CoT, PoT, or direct computation based on the generated equations, all within a single LLM call. Results across five models and four benchmarks show F-1 outperforms CoT by +5.76% and PoT by +8.42% on average. Crucially, gains are largest in applied domains: +13.30% on FinanceMath over CoT, and within OlympiadBench, larger gains on physics (+2.55%) than pure math (+0.44%). This demonstrates that F-1 is more effective than CoT in applied mathematics problems. 

---
# MetaGen: Self-Evolving Roles and Topologies for Multi-Agent LLM Reasoning 

**Authors**: Yimeng Wang, Jiaxing Zhao, Hongbin Xie, Hexing Ma, Yuzhen Lei, Shuangxue Liu, Xuan Song, Zichen Zhang, Haoran Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.19290)  

**Abstract**: Large language models are increasingly deployed as multi-agent systems, where specialized roles communicate and collaborate through structured interactions to solve complex tasks that often exceed the capacity of a single agent. However, most existing systems still rely on a fixed role library and an execution-frozen interaction topology, a rigid design choice that frequently leads to task mismatch, prevents timely adaptation when new evidence emerges during reasoning, and further inflates inference cost. We introduce MetaGen, a training-free framework that adapts both the role space and the collaboration topology at inference time, without updating base model weights. MetaGen generates and rewrites query-conditioned role specifications to maintain a controllable dynamic role pool, then instantiates a constrained execution graph around a minimal backbone. During execution, it iteratively updates role prompts and adjusts structural decisions using lightweight feedback signals. Experiments on code generation and multi-step reasoning benchmarks show that MetaGen improves the accuracy and cost tradeoff over strong multi-agent baselines. 

---
# ReToP: Learning to Rewrite Electronic Health Records for Clinical Prediction 

**Authors**: Jesus Lovon-Melgarejo, Jose G. Moreno, Christine Damase-Michel, Lynda Tamine  

**Link**: [PDF](https://arxiv.org/pdf/2601.19286)  

**Abstract**: Electronic Health Records (EHRs) provide crucial information for clinical decision-making. However, their high-dimensionality, heterogeneity, and sparsity make clinical prediction challenging. Large Language Models (LLMs) allowed progress towards addressing this challenge by leveraging parametric medical knowledge to enhance EHR data for clinical prediction tasks. Despite the significant achievements made so far, most of the existing approaches are fundamentally task-agnostic in the sense that they deploy LLMs as EHR encoders or EHR completion modules without fully integrating signals from the prediction tasks. This naturally hinders task performance accuracy. In this work, we propose Rewrite-To-Predict (ReToP), an LLM-based framework that addresses this limitation through an end-to-end training of an EHR rewriter and a clinical predictor. To cope with the lack of EHR rewrite training data, we generate synthetic pseudo-labels using clinical-driven feature selection strategies to create diverse patient rewrites for fine-tuning the EHR rewriter. ReToP aligns the rewriter with prediction objectives using a novel Classifier Supervised Contribution (CSC) score that enables the EHR rewriter to generate clinically relevant rewrites that directly enhance prediction. Our ReToP framework surpasses strong baseline models across three clinical tasks on MIMIC-IV. Moreover, the analysis of ReToP shows its generalizability to unseen datasets and tasks with minimal fine-tuning while preserving faithful rewrites and emphasizing task-relevant predictive features. 

---
# DART: Diffusion-Inspired Speculative Decoding for Fast LLM Inference 

**Authors**: Fuliang Liu, Xue Li, Ketai Zhao, Yinxi Gao, Ziyan Zhou, Zhonghui Zhang, Zhibin Wang, Wanchun Dou, Sheng Zhong, Chen Tian  

**Link**: [PDF](https://arxiv.org/pdf/2601.19278)  

**Abstract**: Speculative decoding is an effective and lossless approach for accelerating LLM inference. However, existing widely adopted model-based draft designs, such as EAGLE3, improve accuracy at the cost of multi-step autoregressive inference, resulting in high drafting latency and ultimately rendering the drafting stage itself a performance bottleneck. Inspired by diffusion-based large language models (dLLMs), we propose DART, which leverages parallel generation to reduce drafting latency. DART predicts logits for multiple future masked positions in parallel within a single forward pass based on hidden states of the target model, thereby eliminating autoregressive rollouts in the draft model while preserving a lightweight design. Based on these parallel logit predictions, we further introduce an efficient tree pruning algorithm that constructs high-quality draft token trees with N-gram-enforced semantic continuity. DART substantially reduces draft-stage overhead while preserving high draft accuracy, leading to significantly improved end-to-end decoding speed. Experimental results demonstrate that DART achieves a 2.03x--3.44x wall-clock time speedup across multiple datasets, surpassing EAGLE3 by 30% on average and offering a practical speculative decoding framework. Code is released at this https URL. 

---
# Riddle Quest : The Enigma of Words 

**Authors**: Niharika Sri Parasa, Chaitali Diwan, Srinath Srinivasa  

**Link**: [PDF](https://arxiv.org/pdf/2601.19273)  

**Abstract**: Riddles are concise linguistic puzzles that describe an object or idea through indirect, figurative, or playful clues. They are a longstanding form of creative expression, requiring the solver to interpret hints, recognize patterns, and draw inferences to identify the answers. In this work, we introduce a simple pipeline for creating and evaluating analogy-based riddles. The system includes a triples creator that builds structured facts about a concept, a semantic mapper that selects attributes useful for analogy, a stylized generator that turns them into riddle clues, and a validator that collects all possible answers the riddle could point to. We use this validator to study whether large language models can recover the full answer set for different riddle types. Our case study shows that while models often guess the main intended answer, they frequently miss other valid interpretations. This highlights the value of riddles as a lightweight tool for examining reasoning coverage and ambiguity handling in language models. 

---
# DiaDem: Advancing Dialogue Descriptions in Audiovisual Video Captioning for Multimodal Large Language Models 

**Authors**: Xinlong Chen, Weihong Lin, Jingyun Hua, Linli Yao, Yue Ding, Bozhou Li, Bohan Zeng, Yang Shi, Qiang Liu, Yuanxing Zhang, Pengfei Wan, Liang Wang, Tieniu Tan  

**Link**: [PDF](https://arxiv.org/pdf/2601.19267)  

**Abstract**: Accurate dialogue description in audiovisual video captioning is crucial for downstream understanding and generation tasks. However, existing models generally struggle to produce faithful dialogue descriptions within audiovisual captions. To mitigate this limitation, we propose DiaDem, a powerful audiovisual video captioning model capable of generating captions with more precise dialogue descriptions while maintaining strong overall performance. We first synthesize a high-quality dataset for SFT, then employ a difficulty-partitioned two-stage GRPO strategy to further enhance dialogue descriptions. To enable systematic evaluation of dialogue description capabilities, we introduce DiaDemBench, a comprehensive benchmark designed to evaluate models across diverse dialogue scenarios, emphasizing both speaker attribution accuracy and utterance transcription fidelity in audiovisual captions. Extensive experiments on DiaDemBench reveal even commercial models still exhibit substantial room for improvement in dialogue-aware captioning. Notably, DiaDem not only outperforms the Gemini series in dialogue description accuracy but also achieves competitive performance on general audiovisual captioning benchmarks, demonstrating its overall effectiveness. 

---
# RPO-RAG: Aligning Small LLMs with Relation-aware Preference Optimization for Knowledge Graph Question Answering 

**Authors**: Kaehyun Um, KyuHwan Yeom, Haerim Yang, Minyoung Choi, Hyeongjun Yang, Kyong-Ho Lee  

**Link**: [PDF](https://arxiv.org/pdf/2601.19225)  

**Abstract**: Large Language Models (LLMs) have recently demonstrated remarkable reasoning abilities, yet hallucinate on knowledge-intensive tasks. Retrieval-augmented generation (RAG) mitigates this issue by grounding answers in external sources, e.g., knowledge graphs (KGs). However, existing KG-based RAG approaches rely on semantics-unaware path sampling and are weakly aligned with KG reasoning objectives, which limits further accuracy gains. They also feed retrieved paths directly into the reasoner without organizing them into answer-centered reasoning paths, hindering small LLMs' ability to leverage the retrieved knowledge. Furthermore, prior works predominantly rely on large LLMs (e.g., ChatGPT/GPT-4) or assume backbones above 7B parameters, leaving sub-7B models underexplored. We address this gap with RPO-RAG, the first KG-based RAG framework specifically designed for small LLMs, to the best of our knowledge. RPO-RAG introduces three key innovations: (1) a query-path semantic sampling strategy that provides informative supervisory signals; (2) a relation-aware preference optimization that aligns training with intermediate KG reasoning signals (e.g., relation); and (3) an answer-centered prompt design that organizes entities and reasoning paths in an interpretable format. Extensive experiments on two benchmark Knowledge Graph Question Answering (KGQA) datasets, WebQSP and CWQ, demonstrate that RPO-RAG effectively bridges the performance gap between small and large language models. On WebQSP, it improves F1 by up to 8.8%, reflecting enhanced answer precision, while on CWQ it achieves new state-of-the-art results among models under 8B parameters in both Hit and F1. Overall, RPO-RAG substantially improves the reasoning capability of small LLMs, even under 3B parameters-highlighting their potential for resource-efficient and practical on-device KGQA applications. 

---
# DREAMSTATE: Diffusing States and Parameters for Recurrent Large Language Models 

**Authors**: Liu Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2601.19221)  

**Abstract**: Modern Recurrent Neural Networks (RNNs), such as RWKV, are distinguished by their powerful short-range modeling capabilities and efficient fixed-size states, which constitute a core advantage over standard Transformers. However, there is a significant lack of research into their internal state as an editable knowledge representation. To fill this gap, we first explore the representational properties of the RWKV state by proposing the DREAMSTATE framework. This framework utilizes a conditional Diffusion Transformer (DiT) to directly model the probability manifold of the state, enabling its generation and editing. The structural nature of this representation is validated through t-SNE visualizations and controlled generation experiments. After successfully uncovering and modeling the state's representational potential, we further propose a novel hybrid architecture that combines the local advantages of RNNs with global context adaptability. This architecture features a parallel DiT that processes a variable-length global context to dynamically generate and adjust the core recurrent module's WKV parameters, transforming the fixed recurrence mechanism into a context-aware dynamic function. Experiments demonstrate that this hybrid model can be trained stably via a multi-objective loss, validating its design feasibility. Our work not only opens a new research direction for RNN state representation but also provides a concrete architectural reference for future model design. The code is publicly available at: this https URL. 

---
# A Hybrid Supervised-LLM Pipeline for Actionable Suggestion Mining in Unstructured Customer Reviews 

**Authors**: Aakash Trivedi, Aniket Upadhyay, Pratik Narang, Dhruv Kumar, Praveen Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2601.19214)  

**Abstract**: Extracting actionable suggestions from customer reviews is essential for operational decision-making, yet these directives are often embedded within mixed-intent, unstructured text. Existing approaches either classify suggestion-bearing sentences or generate high-level summaries, but rarely isolate the precise improvement instructions businesses need. We evaluate a hybrid pipeline combining a high-recall RoBERTa classifier trained with a precision-recall surrogate to reduce unrecoverable false negatives with a controlled, instruction-tuned LLM for suggestion extraction, categorization, clustering, and summarization. Across real-world hospitality and food datasets, the hybrid system outperforms prompt-only, rule-based, and classifier-only baselines in extraction accuracy and cluster coherence. Human evaluations further confirm that the resulting suggestions and summaries are clear, faithful, and interpretable. Overall, our results show that hybrid reasoning architectures achieve meaningful improvements fine-grained actionable suggestion mining while highlighting challenges in domain adaptation and efficient local deployment. 

---
# How Do Transformers Learn to Associate Tokens: Gradient Leading Terms Bring Mechanistic Interpretability 

**Authors**: Shawn Im, Changdae Oh, Zhen Fang, Sharon Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.19208)  

**Abstract**: Semantic associations such as the link between "bird" and "flew" are foundational for language modeling as they enable models to go beyond memorization and instead generalize and generate coherent text. Understanding how these associations are learned and represented in language models is essential for connecting deep learning with linguistic theory and developing a mechanistic foundation for large language models. In this work, we analyze how these associations emerge from natural language data in attention-based language models through the lens of training dynamics. By leveraging a leading-term approximation of the gradients, we develop closed-form expressions for the weights at early stages of training that explain how semantic associations first take shape. Through our analysis, we reveal that each set of weights of the transformer has closed-form expressions as simple compositions of three basis functions (bigram, token-interchangeability, and context mappings), reflecting the statistics of the text corpus and uncovering how each component of the transformer captures semantic associations based on these compositions. Experiments on real-world LLMs demonstrate that our theoretical weight characterizations closely match the learned weights, and qualitative analyses further show how our theorem shines light on interpreting the learned associations in transformers. 

---
# Do Images Speak Louder than Words? Investigating the Effect of Textual Misinformation in VLMs 

**Authors**: Chi Zhang, Wenxuan Ding, Jiale Liu, Mingrui Wu, Qingyun Wu, Ray Mooney  

**Link**: [PDF](https://arxiv.org/pdf/2601.19202)  

**Abstract**: Vision-Language Models (VLMs) have shown strong multimodal reasoning capabilities on Visual-Question-Answering (VQA) benchmarks. However, their robustness against textual misinformation remains under-explored. While existing research has studied the effect of misinformation in text-only domains, it is not clear how VLMs arbitrate between contradictory information from different modalities. To bridge the gap, we first propose the CONTEXT-VQA (i.e., Conflicting Text) dataset, consisting of image-question pairs together with systematically generated persuasive prompts that deliberately conflict with visual evidence. Then, a thorough evaluation framework is designed and executed to benchmark the susceptibility of various models to these conflicting multimodal inputs. Comprehensive experiments over 11 state-of-the-art VLMs reveal that these models are indeed vulnerable to misleading textual prompts, often overriding clear visual evidence in favor of the conflicting text, and show an average performance drop of over 48.2% after only one round of persuasive conversation. Our findings highlight a critical limitation in current VLMs and underscore the need for improved robustness against textual manipulation. 

---
# Transparency-First Medical Language Models: Datasheets, Model Cards, and End-to-End Data Provenance for Clinical NLP 

**Authors**: Olaf Yunus Laitinen Imanov, Taner Yilmaz, Ayse Tuba Tugrul, Melike Nesrin Zaman, Ozkan Gunalp, Duygu Erisken, Sila Burde Dulger, Rana Irem Turhan, Izzet Ozdemir, Derya Umut Kulali, Ozan Akbulut, Harun Demircioglu, Hasan Basri Kara, Berfin Tavan  

**Link**: [PDF](https://arxiv.org/pdf/2601.19191)  

**Abstract**: We introduce TeMLM, a set of transparency-first release artifacts for clinical language models. TeMLM unifies provenance, data transparency, modeling transparency, and governance into a single, machine-checkable release bundle. We define an artifact suite (TeMLM-Card, TeMLM-Datasheet, TeMLM-Provenance) and a lightweight conformance checklist for repeatable auditing. We instantiate the artifacts on Technetium-I, a large-scale synthetic clinical NLP dataset with 498,000 notes, 7.74M PHI entity annotations across 10 types, and ICD-9-CM diagnosis labels, and report reference results for ProtactiniumBERT (about 100 million parameters) on PHI de-identification (token classification) and top-50 ICD-9 code extraction (multi-label classification). We emphasize that synthetic benchmarks are valuable for tooling and process validation, but models should be validated on real clinical data prior to deployment. 

---
# Leveraging Sentence-oriented Augmentation and Transformer-Based Architecture for Vietnamese-Bahnaric Translation 

**Authors**: Tan Sang Nguyen, Quoc Nguyen Pham, Tho Quan  

**Link**: [PDF](https://arxiv.org/pdf/2601.19124)  

**Abstract**: The Bahnar people, an ethnic minority in Vietnam with a rich ancestral heritage, possess a language of immense cultural and historical significance. The government places a strong emphasis on preserving and promoting the Bahnaric language by making it accessible online and encouraging communication across generations. Recent advancements in artificial intelligence, such as Neural Machine Translation (NMT), have brought about a transformation in translation by improving accuracy and fluency. This, in turn, contributes to the revival of the language through educational efforts, communication, and documentation. Specifically, NMT is pivotal in enhancing accessibility for Bahnaric speakers, making information and content more readily available. Nevertheless, the translation of Vietnamese into Bahnaric faces practical challenges due to resource constraints, especially given the limited resources available for the Bahnaric language. To address this, we employ state-of-the-art techniques in NMT along with two augmentation strategies for domain-specific Vietnamese-Bahnaric translation task. Importantly, both approaches are flexible and can be used with various neural machine translation models. Additionally, they do not require complex data preprocessing steps, the training of additional systems, or the acquisition of extra data beyond the existing training parallel corpora. 

---
# PsyProbe: Proactive and Interpretable Dialogue through User State Modeling for Exploratory Counseling 

**Authors**: Sohhyung Park, Hyunji Kang, Sungzoon Cho, Dongil Kim  

**Link**: [PDF](https://arxiv.org/pdf/2601.19096)  

**Abstract**: Recent advances in large language models have enabled mental health dialogue systems, yet existing approaches remain predominantly reactive, lacking systematic user state modeling for proactive therapeutic exploration. We introduce PsyProbe, a dialogue system designed for the exploration phase of counseling that systematically tracks user psychological states through the PPPPPI framework (Presenting, Predisposing, Precipitating, Perpetuating, Protective, Impact) augmented with cognitive error detection. PsyProbe combines State Builder for extracting structured psychological profiles, Memory Construction for tracking information gaps, Strategy Planner for Motivational Interviewing behavioral codes, and Response Generator with Question Ideation and Critic/Revision modules to generate contextually appropriate, proactive questions. We evaluate PsyProbe with 27 participants in real-world Korean counseling scenarios, including automatic evaluation across ablation modes, user evaluation, and expert evaluation by a certified counselor. The full PsyProbe model consistently outperforms baseline and ablation modes in automatic evaluation. User evaluation demonstrates significantly increased engagement intention and improved naturalness compared to baseline. Expert evaluation shows that PsyProbe substantially improves core issue understanding and achieves question rates comparable to professional counselors, validating the effectiveness of systematic state modeling and proactive questioning for therapeutic exploration. 

---
# Optimizing Conversational Quality in Spoken Dialogue Systems with Reinforcement Learning from AI Feedback 

**Authors**: Siddhant Arora, Jinchuan Tian, Jiatong Shi, Hayato Futami, Yosuke Kashiwagi, Emiru Tsunoo, Shinji Watanabe  

**Link**: [PDF](https://arxiv.org/pdf/2601.19063)  

**Abstract**: Reinforcement learning from human or AI feedback (RLHF/RLAIF) for speech-in/speech-out dialogue systems (SDS) remains underexplored, with prior work largely limited to single semantic rewards applied at the utterance level. Such setups overlook the multi-dimensional and multi-modal nature of conversational quality, which encompasses semantic coherence, audio naturalness, speaker consistency, emotion alignment, and turn-taking behavior. Moreover, they are fundamentally mismatched with duplex spoken dialogue systems that generate responses incrementally, where agents must make decisions based on partial utterances. We address these limitations with the first multi-reward RLAIF framework for SDS, combining semantic, audio-quality, and emotion-consistency rewards. To align utterance-level preferences with incremental, blockwise decoding in duplex models, we apply turn-level preference sampling and aggregate per-block log-probabilities within a single DPO objective. We present the first systematic study of preference learning for improving SDS quality in both multi-turn Chain-of-Thought and blockwise duplex models, and release a multi-reward DPO dataset to support reproducible research. Experiments show that single-reward RLAIF selectively improves its targeted metric, while joint multi-reward training yields consistent gains across semantic quality and audio naturalness. These results highlight the importance of holistic, multi-reward alignment for practical conversational SDS. 

---
# FROST: Filtering Reasoning Outliers with Attention for Efficient Reasoning 

**Authors**: Haozheng Luo, Zhuolin Jiang, Md Zahid Hasan, Yan Chen, Soumalya Sarkar  

**Link**: [PDF](https://arxiv.org/pdf/2601.19001)  

**Abstract**: We propose FROST, an attention-aware method for efficient reasoning. Unlike traditional approaches, FROST leverages attention weights to prune uncritical reasoning paths, yielding shorter and more reliable reasoning trajectories. Methodologically, we introduce the concept of reasoning outliers and design an attention-based mechanism to remove them. Theoretically, FROST preserves and enhances the model's reasoning capacity while eliminating outliers at the sentence level. Empirically, we validate FROST on four benchmarks using two strong reasoning models (Phi-4-Reasoning and GPT-OSS-20B), outperforming state-of-the-art methods such as TALE and ThinkLess. Notably, FROST achieves an average 69.68% reduction in token usage and a 26.70% improvement in accuracy over the base model. Furthermore, in evaluations of attention outlier metrics, FROST reduces the maximum infinity norm by 15.97% and the average kurtosis by 91.09% compared to the base model. Code is available at this https URL 

---
# Malicious Repurposing of Open Science Artefacts by Using Large Language Models 

**Authors**: Zahra Hashemi, Zhiqiang Zhong, Jun Pang, Wei Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2601.18998)  

**Abstract**: The rapid evolution of large language models (LLMs) has fuelled enthusiasm about their role in advancing scientific discovery, with studies exploring LLMs that autonomously generate and evaluate novel research ideas. However, little attention has been given to the possibility that such models could be exploited to produce harmful research by repurposing open science artefacts for malicious ends. We fill the gap by introducing an end-to-end pipeline that first bypasses LLM safeguards through persuasion-based jailbreaking, then reinterprets NLP papers to identify and repurpose their artefacts (datasets, methods, and tools) by exploiting their vulnerabilities, and finally assesses the safety of these proposals using our evaluation framework across three dimensions: harmfulness, feasibility of misuse, and soundness of technicality. Overall, our findings demonstrate that LLMs can generate harmful proposals by repurposing ethically designed open artefacts; however, we find that LLMs acting as evaluators strongly disagree with one another on evaluation outcomes: GPT-4.1 assigns higher scores (indicating greater potential harms, higher soundness and feasibility of misuse), Gemini-2.5-pro is markedly stricter, and Grok-3 falls between these extremes. This indicates that LLMs cannot yet serve as reliable judges in a malicious evaluation setup, making human evaluation essential for credible dual-use risk assessment. 

---
# LLMs versus the Halting Problem: Revisiting Program Termination Prediction 

**Authors**: Oren Sultan, Jordi Armengol-Estape, Pascal Kesseli, Julien Vanegue, Dafna Shahaf, Yossi Adi, Peter O'Hearn  

**Link**: [PDF](https://arxiv.org/pdf/2601.18987)  

**Abstract**: Determining whether a program terminates is a central problem in computer science. Turing's foundational result established the Halting Problem as undecidable, showing that no algorithm can universally determine termination for all programs and inputs. Consequently, automatic verification tools approximate termination, sometimes failing to prove or disprove; these tools rely on problem-specific architectures and abstractions, and are usually tied to particular programming languages. Recent success and progress in large language models (LLMs) raises the following question: can LLMs reliably predict program termination? In this work, we evaluate LLMs on a diverse set of C programs from the Termination category of the International Competition on Software Verification (SV-Comp) 2025. Our results suggest that LLMs perform remarkably well at predicting program termination, where GPT-5 and Claude Sonnet-4.5 would rank just behind the top-ranked tool (using test-time-scaling), and Code World Model (CWM) would place just behind the second-ranked tool. While LLMs are effective at predicting program termination, they often fail to provide a valid witness as a proof. Moreover, LLMs performance drops as program length increases. We hope these insights motivate further research into program termination and the broader potential of LLMs for reasoning about undecidable problems. 

---
# BabyReasoningBench: Generating Developmentally-Inspired Reasoning Tasks for Evaluating Baby Language Models 

**Authors**: Kaustubh D. Dhole  

**Link**: [PDF](https://arxiv.org/pdf/2601.18933)  

**Abstract**: Traditional evaluations of reasoning capabilities of language models are dominated by adult-centric benchmarks that presuppose broad world knowledge, complex instruction following, and mature pragmatic competence. These assumptions are mismatched to baby language models trained on developmentally plausible input such as child-directed speech and early-childhood narratives, and they obscure which reasoning abilities (if any) emerge under such constraints. We introduce BabyReasoningBench, a GPT-5.2 generated benchmark of 19 reasoning tasks grounded in classic paradigms from developmental psychology, spanning theory of mind, analogical and relational reasoning, causal inference and intervention selection, and core reasoning primitives that are known to be confounded by memory and pragmatics. We find that two GPT-2 based baby language models (pretrained on 10M and 100M of child-directed speech text) show overall low but uneven performance, with dissociations across task families: scaling improves several causal and physical reasoning tasks, while belief attribution and pragmatics-sensitive tasks remain challenging. BabyReasoningBench provides a developmentally grounded lens for analyzing what kinds of reasoning are supported by child-like training distributions, and for testing mechanistic hypotheses about how such abilities emerge. 

---
# Flatter Tokens are More Valuable for Speculative Draft Model Training 

**Authors**: Jiaming Fan, Daming Cao, Xiangzhong Luo, Jiale Fu, Chonghan Liu, Xu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2601.18902)  

**Abstract**: Speculative Decoding (SD) is a key technique for accelerating Large Language Model (LLM) inference, but it typically requires training a draft model on a large dataset. We approach this problem from a data-centric perspective, finding that not all training samples contribute equally to the SD acceptance rate. Specifically, our theoretical analysis and empirical validation reveals that tokens inducing flatter predictive distributions from the target model are more valuable than those yielding sharply peaked distributions. Based on this insight, we propose flatness, a new metric to quantify this property, and develop the Sample-level-flatness-based Dataset Distillation (SFDD) approach, which filters the training data to retain only the most valuable samples. Experiments on the EAGLE framework demonstrate that SFDD can achieve over 2$\times$ training speedup using only 50% of the data, while keeping the final model's inference speedup within 4% of the full-dataset baseline. This work introduces an effective, data-centric approach that substantially improves the training efficiency for Speculative Decoding. Our code is available at this https URL. 

---
# Self-Aware Knowledge Probing: Evaluating Language Models' Relational Knowledge through Confidence Calibration 

**Authors**: Christopher Kissling, Elena Merdjanovska, Alan Akbik  

**Link**: [PDF](https://arxiv.org/pdf/2601.18901)  

**Abstract**: Knowledge probing quantifies how much relational knowledge a language model (LM) has acquired during pre-training. Existing knowledge probes evaluate model capabilities through metrics like prediction accuracy and precision. Such evaluations fail to account for the model's reliability, reflected in the calibration of its confidence scores. In this paper, we propose a novel calibration probing framework for relational knowledge, covering three modalities of model confidence: (1) intrinsic confidence, (2) structural consistency and (3) semantic grounding. Our extensive analysis of ten causal and six masked language models reveals that most models, especially those pre-trained with the masking objective, are overconfident. The best-calibrated scores come from confidence estimates that account for inconsistencies due to statement rephrasing. Moreover, even the largest pre-trained models fail to encode the semantics of linguistic confidence expressions accurately. 

---
# Language Family Matters: Evaluating LLM-Based ASR Across Linguistic Boundaries 

**Authors**: Yuchen Zhang, Ravi Shekhar, Haralambos Mouratidis  

**Link**: [PDF](https://arxiv.org/pdf/2601.18899)  

**Abstract**: Large Language Model (LLM)-powered Automatic Speech Recognition (ASR) systems achieve strong performance with limited resources by linking a frozen speech encoder to a pretrained LLM via a lightweight connector. Prior work trains a separate connector per language, overlooking linguistic relatedness. We propose an efficient and novel connector-sharing strategy based on linguistic family membership, enabling one connector per family, and empirically validate its effectiveness across two multilingual LLMs and two real-world corpora spanning curated and crowd-sourced speech. Our results show that family-based connectors reduce parameter count while improving generalization across domains, offering a practical and scalable strategy for multilingual ASR deployment. 

---
# Post-LayerNorm Is Back: Stable, ExpressivE, and Deep 

**Authors**: Chen Chen, Lai Wei  

**Link**: [PDF](https://arxiv.org/pdf/2601.19895)  

**Abstract**: Large language model (LLM) scaling is hitting a wall. Widening models yields diminishing returns, and extending context length does not improve fundamental expressivity. In contrast, depth scaling offers theoretically superior expressivity, yet current Transformer architectures struggle to train reliably at extreme depths. We revisit the Post-LayerNorm (Post-LN) formulation, whose instability at scale caused its replacement by Pre-LN in modern LLMs. We show that the central failure mode of Post-LN arises from the ResNet-style residual pathway, which introduces gradient vanishing in deep networks. We present Keel, a Post-LN Transformer that replaces this residual path with a Highway-style connection. This modification preserves the gradient flow through the residual branch, preventing signal vanishing from the top layers to the bottom. Unlike prior methods, Keel enables stable training at extreme depths without requiring specialized initialization or complex optimization tricks. Keel trains robustly at depths exceeding 1000 layers and consistently improves perplexity and depth-scaling characteristics over Pre-LN. These findings indicate that Post-LN, when paired with a Highway-style connection, provides a simple and effective foundation for building deeply scalable LLMs, opening the possibility for future infinite-depth architectures. 

---
# Neural Neural Scaling Laws 

**Authors**: Michael Y. Hu, Jane Pan, Ayush Rajesh Jhaveri, Nicholas Lourie, Kyunghyun Cho  

**Link**: [PDF](https://arxiv.org/pdf/2601.19831)  

**Abstract**: Neural scaling laws predict how language model performance improves with increased compute. While aggregate metrics like validation loss can follow smooth power-law curves, individual downstream tasks exhibit diverse scaling behaviors: some improve monotonically, others plateau, and some even degrade with scale. We argue that predicting downstream performance from validation perplexity suffers from two limitations: averaging token-level losses obscures signal, and no simple parametric family can capture the full spectrum of scaling behaviors. To address this, we propose Neural Neural Scaling Laws (NeuNeu), a neural network that frames scaling law prediction as time-series extrapolation. NeuNeu combines temporal context from observed accuracy trajectories with token-level validation losses, learning to predict future performance without assuming any bottleneck or functional form. Trained entirely on open-source model checkpoints from HuggingFace, NeuNeu achieves 2.04% mean absolute error in predicting model accuracy on 66 downstream tasks -- a 38% reduction compared to logistic scaling laws (3.29% MAE). Furthermore, NeuNeu generalizes zero-shot to unseen model families, parameter counts, and downstream tasks. Our work suggests that predicting downstream scaling laws directly from data outperforms parametric alternatives. 

---
# Rethinking Discrete Speech Representation Tokens for Accent Generation 

**Authors**: Jinzuomu Zhong, Yi Wang, Korin Richmond, Peter Bell  

**Link**: [PDF](https://arxiv.org/pdf/2601.19786)  

**Abstract**: Discrete Speech Representation Tokens (DSRTs) have become a foundational component in speech generation. While prior work has extensively studied phonetic and speaker information in DSRTs, how accent information is encoded in DSRTs remains largely unexplored. In this paper, we present the first systematic investigation of accent information in DSRTs. We propose a unified evaluation framework that measures both accessibility of accent information via a novel Accent ABX task and recoverability via cross-accent Voice Conversion (VC) resynthesis. Using this framework, we analyse DSRTs derived from a variety of speech encoders. Our results reveal that accent information is substantially reduced when ASR supervision is used to fine-tune the encoder, but cannot be effectively disentangled from phonetic and speaker information through naive codebook size reduction. Based on these findings, we propose new content-only and content-accent DSRTs that significantly outperform existing designs in controllable accent generation. Our work highlights the importance of accent-aware evaluation and provides practical guidance for designing DSRTs for accent-controlled speech generation. 

---
# RvB: Automating AI System Hardening via Iterative Red-Blue Games 

**Authors**: Lige Huang, Zicheng Liu, Jie Zhang, Lewen Yan, Dongrui Liu, Jing Shao  

**Link**: [PDF](https://arxiv.org/pdf/2601.19726)  

**Abstract**: The dual offensive and defensive utility of Large Language Models (LLMs) highlights a critical gap in AI security: the lack of unified frameworks for dynamic, iterative adversarial adaptation hardening. To bridge this gap, we propose the Red Team vs. Blue Team (RvB) framework, formulated as a training-free, sequential, imperfect-information game. In this process, the Red Team exposes vulnerabilities, driving the Blue Team to learning effective solutions without parameter updates. We validate our framework across two challenging domains: dynamic code hardening against CVEs and guardrail optimization against jailbreaks. Our empirical results show that this interaction compels the Blue Team to learn fundamental defensive principles, leading to robust remediations that are not merely overfitted to specific exploits. RvB achieves Defense Success Rates of 90\% and 45\% across the respective tasks while maintaining near 0\% False Positive Rates, significantly surpassing baselines. This work establishes the iterative adversarial interaction framework as a practical paradigm that automates the continuous hardening of AI systems. 

---
# Explicit Multi-head Attention for Inter-head Interaction in Large Language Models 

**Authors**: Runyu Peng, Yunhua Zhou, Demin Song, Kai Lv, Bo Wang, Qipeng Guo, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2601.19611)  

**Abstract**: In large language models built upon the Transformer architecture, recent studies have shown that inter-head interaction can enhance attention performance. Motivated by this, we propose Multi-head Explicit Attention (MEA), a simple yet effective attention variant that explicitly models cross-head interaction. MEA consists of two key components: a Head-level Linear Composition (HLC) module that separately applies learnable linear combinations to the key and value vectors across heads, thereby enabling rich inter-head communication; and a head-level Group Normalization layer that aligns the statistical properties of the recombined heads. MEA shows strong robustness in pretraining, which allows the use of larger learning rates that lead to faster convergence, ultimately resulting in lower validation loss and improved performance across a range of tasks. Furthermore, we explore the parameter efficiency of MEA by reducing the number of attention heads and leveraging HLC to reconstruct them using low-rank "virtual heads". This enables a practical key-value cache compression strategy that reduces KV-cache memory usage by 50% with negligible performance loss on knowledge-intensive and scientific reasoning tasks, and only a 3.59% accuracy drop for Olympiad-level mathematical benchmarks. 

---
# Benchmarks Saturate When The Model Gets Smarter Than The Judge 

**Authors**: Marthe Ballon, Andres Algaba, Brecht Verbeken, Vincent Ginis  

**Link**: [PDF](https://arxiv.org/pdf/2601.19532)  

**Abstract**: Benchmarks are important tools to track progress in the development of Large Language Models (LLMs), yet inaccuracies in datasets and evaluation methods consistently undermine their effectiveness. Here, we present Omni-MATH-2, a manually revised version of the Omni-MATH dataset comprising a clean, exact-answer subset ($n{=}4181$) and a tagged, non-standard subset ($n{=}247$). Each problem was audited to ensure LaTeX compilability, solvability and verifiability, which involved adding missing figures or information, labeling problems requiring a proof, estimation or image, and removing clutter. This process significantly reduces dataset-induced noise, thereby providing a more precise assessment of model performance. The annotated dataset also allows us to evaluate judge-induced noise by comparing GPT-5 mini with the original Omni-Judge, revealing substantial discrepancies between judges on both the clean and tagged problem subsets. Expert annotations reveal that Omni-Judge is wrong in $96.4\%$ of the judge disagreements, indicating its inability to differentiate between models' abilities, even well before saturation of the benchmark occurs. As problems become more challenging, we find that increasingly competent judges become essential in order to prevent judge errors from masking genuine differences between models. Finally, neither judge identifies the present failure modes for the subset of tagged problems, demonstrating that dataset quality and judge reliability are both critical to develop accurate benchmarks of model performance. 

---
# Enhancing Academic Paper Recommendations Using Fine-Grained Knowledge Entities and Multifaceted Document Embeddings 

**Authors**: Haixu Xi, Heng Zhang, Chengzhi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.19513)  

**Abstract**: In the era of explosive growth in academic literature, the burden of literature review on scholars are increasing. Proactively recommending academic papers that align with scholars' literature needs in the research process has become one of the crucial pathways to enhance research efficiency and stimulate innovative thinking. Current academic paper recommendation systems primarily focus on broad and coarse-grained suggestions based on general topic or field similarities. While these systems effectively identify related literature, they fall short in addressing scholars' more specific and fine-grained needs, such as locating papers that utilize particular research methods, or tackle distinct research tasks within the same topic. To meet the diverse and specific literature needs of scholars in the research process, this paper proposes a novel academic paper recommendation method. This approach embeds multidimensional information by integrating new types of fine-grained knowledge entities, title and abstract of document, and citation data. Recommendations are then generated by calculating the similarity between combined paper vectors. The proposed recommendation method was evaluated using the STM-KG dataset, a knowledge graph that incorporates scientific concepts derived from papers across ten distinct domains. The experimental results indicate that our method outperforms baseline models, achieving an average precision of 27.3% among the top 50 recommendations. This represents an improvement of 6.7% over existing approaches. 

---
# ALRM: Agentic LLM for Robotic Manipulation 

**Authors**: Vitor Gaboardi dos Santos, Ibrahim Khadraoui, Ibrahim Farhat, Hamza Yous, Samy Teffahi, Hakim Hacid  

**Link**: [PDF](https://arxiv.org/pdf/2601.19510)  

**Abstract**: Large Language Models (LLMs) have recently empowered agentic frameworks to exhibit advanced reasoning and planning capabilities. However, their integration in robotic control pipelines remains limited in two aspects: (1) prior \ac{llm}-based approaches often lack modular, agentic execution mechanisms, limiting their ability to plan, reflect on outcomes, and revise actions in a closed-loop manner; and (2) existing benchmarks for manipulation tasks focus on low-level control and do not systematically evaluate multistep reasoning and linguistic variation. In this paper, we propose Agentic LLM for Robot Manipulation (ALRM), an LLM-driven agentic framework for robotic manipulation. ALRM integrates policy generation with agentic execution through a ReAct-style reasoning loop, supporting two complementary modes: Code-asPolicy (CaP) for direct executable control code generation, and Tool-as-Policy (TaP) for iterative planning and tool-based action execution. To enable systematic evaluation, we also introduce a novel simulation benchmark comprising 56 tasks across multiple environments, capturing linguistically diverse instructions. Experiments with ten LLMs demonstrate that ALRM provides a scalable, interpretable, and modular approach for bridging natural language reasoning with reliable robotic execution. Results reveal Claude-4.1-Opus as the top closed-source model and Falcon-H1-7B as the top open-source model under CaP. 

---
# Ad Insertion in LLM-Generated Responses 

**Authors**: Shengwei Xu, Zhaohua Chen, Xiaotie Deng, Zhiyi Huang, Grant Schoenebeck  

**Link**: [PDF](https://arxiv.org/pdf/2601.19435)  

**Abstract**: Sustainable monetization of Large Language Models (LLMs) remains a critical open challenge. Traditional search advertising, which relies on static keywords, fails to capture the fleeting, context-dependent user intents--the specific information, goods, or services a user seeks--embedded in conversational flows. Beyond the standard goal of social welfare maximization, effective LLM advertising imposes additional requirements on contextual coherence (ensuring ads align semantically with transient user intents) and computational efficiency (avoiding user interaction latency), as well as adherence to ethical and regulatory standards, including preserving privacy and ensuring explicit ad disclosure. Although various recent solutions have explored bidding on token-level and query-level, both categories of approaches generally fail to holistically satisfy this multifaceted set of constraints.
We propose a practical framework that resolves these tensions through two decoupling strategies. First, we decouple ad insertion from response generation to ensure safety and explicit disclosure. Second, we decouple bidding from specific user queries by using ``genres'' (high-level semantic clusters) as a proxy. This allows advertisers to bid on stable categories rather than sensitive real-time response, reducing computational burden and privacy risks. We demonstrate that applying the VCG auction mechanism to this genre-based framework yields approximately dominant strategy incentive compatibility (DSIC) and individual rationality (IR), as well as approximately optimal social welfare, while maintaining high computational efficiency. Finally, we introduce an "LLM-as-a-Judge" metric to estimate contextual coherence. Our experiments show that this metric correlates strongly with human ratings (Spearman's $\rho\approx 0.66$), outperforming 80% of individual human evaluators. 

---
# Group Distributionally Robust Optimization-Driven Reinforcement Learning for LLM Reasoning 

**Authors**: Kishan Panaganti, Zhenwen Liang, Wenhao Yu, Haitao Mi, Dong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2601.19280)  

**Abstract**: Recent progress in Large Language Model (LLM) reasoning is increasingly driven by the refinement of post-training loss functions and alignment strategies. However, standard Reinforcement Learning (RL) paradigms like Group Relative Policy Optimization (GRPO) remain constrained by static uniformity: uniform prompt sampling and a fixed number of rollouts per prompt. For heterogeneous, heavy-tailed reasoning data, this creates structural inefficiencies that waste compute on already-solved patterns while under-training the long tail of hard problems. To address this, we propose Multi-Adversary Group Distributionally Robust Optimization (GDRO), an optimization-first framework that moves beyond uniform reasoning models by dynamically adapting the training distribution.
We introduce an Online Difficulty Classifier that partitions prompts into dynamic pass@k difficulty groups. We then propose two independent GDRO games for post-training: (1) Prompt-GDRO, which employs an EMA-debiased multiplicative-weights bandit sampler to target the intensive difficulty margin and upweight persistently hard groups without frequency bias; and (2) Rollout-GDRO, which uses a shadow-price controller to reallocate rollouts across groups, maximizing gradient variance reduction on hard tasks under a fixed mean budget (compute-neutral). We provide no-regret guarantees for both controllers and additionally a variance-proxy analysis motivating a square-root optimal rollout allocation for Rollout-GDRO. We validate our framework on the DAPO 14.1k dataset using Qwen3-Base models. Prompt-GDRO and Rollout-GDRO achieve average relative gains of +10.6% and +10.1%, respectively, in pass@8 accuracy across 1.7B, 4B, and 8B scales compared to the GRPO baseline. Qualitative analysis shows an emergent curriculum: the adversaries shift resources to the evolving reasoning frontier, enhancing the reasoning model's performance. 

---
# More at Stake: How Payoff and Language Shape LLM Agent Strategies in Cooperation Dilemmas 

**Authors**: Trung-Kiet Huynh, Dao-Sy Duy-Minh, Thanh-Bang Cao, Phong-Hao Le, Hong-Dan Nguyen, Nguyen Lam Phu Quy, Minh-Luan Nguyen-Vo, Hong-Phat Pham, Pham Phu Hoa, Thien-Kim Than, Chi-Nguyen Tran, Huy Tran, Gia-Thoai Tran-Le, Alessio Buscemi, Le Hong Trang, Anh Han  

**Link**: [PDF](https://arxiv.org/pdf/2601.19082)  

**Abstract**: As LLMs increasingly act as autonomous agents in interactive and multi-agent settings, understanding their strategic behavior is critical for safety, coordination, and AI-driven social and economic systems. We investigate how payoff magnitude and linguistic context shape LLM strategies in repeated social dilemmas, using a payoff-scaled Prisoner's Dilemma to isolate sensitivity to incentive strength. Across models and languages, we observe consistent behavioral patterns, including incentive-sensitive conditional strategies and cross-linguistic divergence. To interpret these dynamics, we train supervised classifiers on canonical repeated-game strategies and apply them to LLM decisions, revealing systematic, model- and language-dependent behavioral intentions, with linguistic framing sometimes matching or exceeding architectural effects. Our results provide a unified framework for auditing LLMs as strategic agents and highlight cooperation biases with direct implications for AI governance and multi-agent system design. 

---
# Who's in Charge? Disempowerment Patterns in Real-World LLM Usage 

**Authors**: Mrinank Sharma, Miles McCain, Raymond Douglas, David Duvenaud  

**Link**: [PDF](https://arxiv.org/pdf/2601.19062)  

**Abstract**: Although AI assistants are now deeply embedded in society, there has been limited empirical study of how their usage affects human empowerment. We present the first large-scale empirical analysis of disempowerment patterns in real-world AI assistant interactions, analyzing 1.5 million consumer this http URL conversations using a privacy-preserving approach. We focus on situational disempowerment potential, which occurs when AI assistant interactions risk leading users to form distorted perceptions of reality, make inauthentic value judgments, or act in ways misaligned with their values. Quantitatively, we find that severe forms of disempowerment potential occur in fewer than one in a thousand conversations, though rates are substantially higher in personal domains like relationships and lifestyle. Qualitatively, we uncover several concerning patterns, such as validation of persecution narratives and grandiose identities with emphatic sycophantic language, definitive moral judgments about third parties, and complete scripting of value-laden personal communications that users appear to implement verbatim. Analysis of historical trends reveals an increase in the prevalence of disempowerment potential over time. We also find that interactions with greater disempowerment potential receive higher user approval ratings, possibly suggesting a tension between short-term user preferences and long-term human empowerment. Our findings highlight the need for AI systems designed to robustly support human autonomy and flourishing. 

---
# Principled Fine-tuning of LLMs from User-Edits: A Medley of Preference, Supervision, and Reward 

**Authors**: Dipendra Misra, Aldo Pacchiano, Ta-Chung Chi, Ge Gao  

**Link**: [PDF](https://arxiv.org/pdf/2601.19055)  

**Abstract**: We study how to fine-tune LLMs using user-edit deployment data consisting of a set of context, an agent's response, and user edits. This deployment data is naturally generated by users in applications such as LLMs-based writing assistants and coding agents. The _natural_ origin of user edits makes it a desired source for adapting and personalizing LLMs. In this setup, there emerges a unification of various feedback types namely preferences, supervised labels, and cost that are typically studied separately in the literature. In this paper, we initiate the theoretical investigation of learning from user edits. We first derive bounds for learning algorithms that learn from each of these feedback types. We prove that these algorithms have different trade-offs depending upon the user, data distribution, and model class. We then propose a simple ensembling procedure to jointly learn from these feedback types. On two domains adapted from Gao et al. 2024, we show our ensembling procedure outperforms these methods that learn from individual feedback. Further, we show that our proposed procedure can robustly adapt to different user-edit distributions at test time. 

---
# Is Finer Better? The Limits of Microscaling Formats in Large Language Models 

**Authors**: Andrea Fasoli, Monodeep Kar, Chi-Chun Liu, Swagath Venkataramani, Viji Srinivasan, Leland Chang, Naigang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.19026)  

**Abstract**: Microscaling data formats leverage per-block tensor quantization to enable aggressive model compression with limited loss in accuracy. Unlocking their potential for efficient training and inference necessitates hardware-friendly implementations that handle matrix multiplications in a native format and adopt efficient error-mitigation strategies. Herein, we report the emergence of a surprising behavior associated with microscaling quantization, whereas the output of a quantized model degrades as block size is decreased below a given threshold. This behavior clashes with the expectation that a smaller block size should allow for a better representation of the tensor elements. We investigate this phenomenon both experimentally and theoretically, decoupling the sources of quantization error behind it. Experimentally, we analyze the distributions of several Large Language Models and identify the conditions driving the anomalous behavior. Theoretically, we lay down a framework showing remarkable agreement with experimental data from pretrained model distributions and ideal ones. Overall, we show that the anomaly is driven by the interplay between narrow tensor distributions and the limited dynamic range of the quantized scales. Based on these insights, we propose the use of FP8 unsigned E5M3 (UE5M3) as a novel hardware-friendly format for the scales in FP4 microscaling data types. We demonstrate that UE5M3 achieves comparable performance to the conventional FP8 unsigned E4M3 scales while obviating the need of global scaling operations on weights and activations. 

---
# Save the Good Prefix: Precise Error Penalization via Process-Supervised RL to Enhance LLM Reasoning 

**Authors**: Haolin Liu, Dian Yu, Sidi Lu, Yujun Zhou, Rui Liu, Zhenwen Liang, Haitao Mi, Chen-Yu Wei, Dong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2601.18984)  

**Abstract**: Reinforcement learning (RL) has emerged as a powerful framework for improving the reasoning capabilities of large language models (LLMs). However, most existing RL approaches rely on sparse outcome rewards, which fail to credit correct intermediate steps in partially successful solutions. Process reward models (PRMs) offer fine-grained step-level supervision, but their scores are often noisy and difficult to evaluate. As a result, recent PRM benchmarks focus on a more objective capability: detecting the first incorrect step in a reasoning path. However, this evaluation target is misaligned with how PRMs are typically used in RL, where their step-wise scores are treated as raw rewards to maximize. To bridge this gap, we propose Verifiable Prefix Policy Optimization (VPPO), which uses PRMs only to localize the first error during RL. Given an incorrect rollout, VPPO partitions the trajectory into a verified correct prefix and an erroneous suffix based on the first error, rewarding the former while applying targeted penalties only after the detected mistake. This design yields stable, interpretable learning signals and improves credit assignment. Across multiple reasoning benchmarks, VPPO consistently outperforms sparse-reward RL and prior PRM-guided baselines on both Pass@1 and Pass@K. 

---
# Intent2QoS: Language Model-Driven Automation of Traffic Shaping Configurations 

**Authors**: Sudipta Acharya, Burak Kantarci  

**Link**: [PDF](https://arxiv.org/pdf/2601.18974)  

**Abstract**: Traffic shaping and Quality of Service (QoS) enforcement are critical for managing bandwidth, latency, and fairness in networks. These tasks often rely on low-level traffic control settings, which require manual setup and technical expertise. This paper presents an automated framework that converts high-level traffic shaping intents in natural or declarative language into valid and correct traffic control rules. To the best of our knowledge, we present the first end-to-end pipeline that ties intent translation in a queuing-theoretic semantic model and, with a rule-based critic, yields deployable Linux traffic control configuration sets. The framework has three steps: (1) a queuing simulation with priority scheduling and Active Queue Management (AQM) builds a semantic model; (2) a language model, using this semantic model and a traffic profile, generates sub-intents and configuration rules; and (3) a rule-based critic checks and adjusts the rules for correctness and policy compliance. We evaluate multiple language models by generating traffic control commands from business intents that comply with relevant standards for traffic control protocols. Experimental results on 100 intents show significant gains, with LLaMA3 reaching 0.88 semantic similarity and 0.87 semantic coverage, outperforming other models by over 30\. A thorough sensitivity study demonstrates that AQM-guided prompting reduces variability threefold compared to zero-shot baselines. 

---
# SICL-AT: Another way to adapt Auditory LLM to low-resource task 

**Authors**: Haolong Zheng, Siyin Wang, Zengrui Jin, Mark Hasegawa-Johnson  

**Link**: [PDF](https://arxiv.org/pdf/2601.18904)  

**Abstract**: Auditory Large Language Models (LLMs) have demonstrated strong performance across a wide range of speech and audio understanding tasks. Nevertheless, they often struggle when applied to low-resource or unfamiliar tasks. In case of labeled in-domain data is scarce or mismatched to the true test distribution, direct fine-tuning can be brittle. In-Context Learning (ICL) provides a training-free, inference-time solution by adapting auditory LLMs through conditioning on a few in-domain demonstrations. In this work, we first show that \emph{Vanilla ICL}, improves zero-shot performance across diverse speech and audio tasks for selected models which suggest this ICL adaptation capability can be generalized to multimodal setting. Building on this, we propose \textbf{Speech In-Context Learning Adaptation Training (SICL-AT)}, a post-training recipe utilizes only high resource speech data intending to strengthen model's in-context learning capability. The enhancement can generalize to audio understanding/reasoning task. Experiments indicate our proposed method consistently outperforms direct fine-tuning in low-resource scenario. 

---
# XProvence: Zero-Cost Multilingual Context Pruning for Retrieval-Augmented Generation 

**Authors**: Youssef Mohamed, Mohamed Elhoseiny, Thibault Formal, Nadezhda Chirkova  

**Link**: [PDF](https://arxiv.org/pdf/2601.18886)  

**Abstract**: This paper introduces XProvence, a multilingual zero-cost context pruning model for retrieval-augmented generation (RAG), trained on 16 languages and supporting 100+ languages through effective cross-lingual transfer. Motivated by the growing use of RAG systems across diverse languages, we explore several strategies to generalize the Provence framework-which first integrated efficient zero-cost context pruning directly into the re-ranking model-beyond English. Across four multilingual question answering benchmarks, we show how XProvence can prune RAG contexts with minimal-to-no performance degradation and outperforms strong baselines. Our model is available at this https URL. 

---
