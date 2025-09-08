# Crosscoding Through Time: Tracking Emergence & Consolidation Of Linguistic Representations Throughout LLM Pretraining 

**Authors**: Deniz Bayazit, Aaron Mueller, Antoine Bosselut  

**Link**: [PDF](https://arxiv.org/pdf/2509.05291)  

**Abstract**: Large language models (LLMs) learn non-trivial abstractions during pretraining, like detecting irregular plural noun subjects. However, it is not well understood when and how specific linguistic abilities emerge as traditional evaluation methods such as benchmarking fail to reveal how models acquire concepts and capabilities. To bridge this gap and better understand model training at the concept level, we use sparse crosscoders to discover and align features across model checkpoints. Using this approach, we track the evolution of linguistic features during pretraining. We train crosscoders between open-sourced checkpoint triplets with significant performance and representation shifts, and introduce a novel metric, Relative Indirect Effects (RelIE), to trace training stages at which individual features become causally important for task performance. We show that crosscoders can detect feature emergence, maintenance, and discontinuation during pretraining. Our approach is architecture-agnostic and scalable, offering a promising path toward more interpretable and fine-grained analysis of representation learning throughout pretraining. 

---
# Less is More Tokens: Efficient Math Reasoning via Difficulty-Aware Chain-of-Thought Distillation 

**Authors**: Abdul Waheed, Chancharik Mitra, Laurie Z. Wang, Deva Ramanan, Bhiksha Raj  

**Link**: [PDF](https://arxiv.org/pdf/2509.05226)  

**Abstract**: Chain-of-thought reasoning, while powerful, can produce unnecessarily verbose output for simpler problems. We present a framework for difficulty-aware reasoning that teaches models to dynamically adjust reasoning depth based on problem complexity. Remarkably, we show that models can be endowed with such dynamic inference pathways without any architectural modifications; we simply post-train on data that is carefully curated to include chain-of-thought traces that are proportional in length to problem difficulty. Our analysis reveals that post-training via supervised fine-tuning (SFT) primarily captures patterns like reasoning length and format, while direct preference optimization (DPO) preserves reasoning accuracy, with their combination reducing length and maintaining or improving performance. Both quantitative metrics and qualitative assessments confirm that models can learn to "think proportionally", reasoning minimally on simple problems while maintaining depth for complex ones. 

---
# Triadic Fusion of Cognitive, Functional, and Causal Dimensions for Explainable LLMs: The TAXAL Framework 

**Authors**: David Herrera-Poyatos, Carlos Peláez-González, Cristina Zuheros, Virilo Tejedor, Rosana Montes, Francisco Herrera  

**Link**: [PDF](https://arxiv.org/pdf/2509.05199)  

**Abstract**: Large Language Models (LLMs) are increasingly being deployed in high-risk domains where opacity, bias, and instability undermine trust and accountability. Traditional explainability methods, focused on surface outputs, do not capture the reasoning pathways, planning logic, and systemic impacts of agentic LLMs.
We introduce TAXAL (Triadic Alignment for eXplainability in Agentic LLMs), a triadic fusion framework that unites three complementary dimensions: cognitive (user understanding), functional (practical utility), and causal (faithful reasoning). TAXAL provides a unified, role-sensitive foundation for designing, evaluating, and deploying explanations in diverse sociotechnical settings.
Our analysis synthesizes existing methods, ranging from post-hoc attribution and dialogic interfaces to explanation-aware prompting, and situates them within the TAXAL triadic fusion model. We further demonstrate its applicability through case studies in law, education, healthcare, and public services, showing how explanation strategies adapt to institutional constraints and stakeholder roles.
By combining conceptual clarity with design patterns and deployment pathways, TAXAL advances explainability as a technical and sociotechnical practice, supporting trustworthy and context-sensitive LLM applications in the era of agentic AI. 

---
# Memorization $\neq$ Understanding: Do Large Language Models Have the Ability of Scenario Cognition? 

**Authors**: Boxiang Ma, Ru Li, Yuanlong Wang, Hongye Tan, Xiaoli Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.04866)  

**Abstract**: Driven by vast and diverse textual data, large language models (LLMs) have demonstrated impressive performance across numerous natural language processing (NLP) tasks. Yet, a critical question persists: does their generalization arise from mere memorization of training data or from deep semantic understanding? To investigate this, we propose a bi-perspective evaluation framework to assess LLMs' scenario cognition - the ability to link semantic scenario elements with their arguments in context. Specifically, we introduce a novel scenario-based dataset comprising diverse textual descriptions of fictional facts, annotated with scenario elements. LLMs are evaluated through their capacity to answer scenario-related questions (model output perspective) and via probing their internal representations for encoded scenario elements-argument associations (internal representation perspective). Our experiments reveal that current LLMs predominantly rely on superficial memorization, failing to achieve robust semantic scenario cognition, even in simple cases. These findings expose critical limitations in LLMs' semantic understanding and offer cognitive insights for advancing their capabilities. 

---
# Using LLMs for Multilingual Clinical Entity Linking to ICD-10 

**Authors**: Sylvia Vassileva, Ivan Koychev, Svetla Boytcheva  

**Link**: [PDF](https://arxiv.org/pdf/2509.04868)  

**Abstract**: The linking of clinical entities is a crucial part of extracting structured information from clinical texts. It is the process of assigning a code from a medical ontology or classification to a phrase in the text. The International Classification of Diseases - 10th revision (ICD-10) is an international standard for classifying diseases for statistical and insurance purposes. Automatically assigning the correct ICD-10 code to terms in discharge summaries will simplify the work of healthcare professionals and ensure consistent coding in hospitals. Our paper proposes an approach for linking clinical terms to ICD-10 codes in different languages using Large Language Models (LLMs). The approach consists of a multistage pipeline that uses clinical dictionaries to match unambiguous terms in the text and then applies in-context learning with GPT-4.1 to predict the ICD-10 code for the terms that do not match the dictionary. Our system shows promising results in predicting ICD-10 codes on different benchmark datasets in Spanish - 0.89 F1 for categories and 0.78 F1 on subcategories on CodiEsp, and Greek - 0.85 F1 on ElCardioCC. 

---
# Classification of kinetic-related injury in hospital triage data using NLP 

**Authors**: Midhun Shyam, Jim Basilakis, Kieran Luken, Steven Thomas, John Crozier, Paul M. Middleton, X. Rosalind Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.04969)  

**Abstract**: Triage notes, created at the start of a patient's hospital visit, contain a wealth of information that can help medical staff and researchers understand Emergency Department patient epidemiology and the degree of time-dependent illness or injury. Unfortunately, applying modern Natural Language Processing and Machine Learning techniques to analyse triage data faces some challenges: Firstly, hospital data contains highly sensitive information that is subject to privacy regulation thus need to be analysed on site; Secondly, most hospitals and medical facilities lack the necessary hardware to fine-tune a Large Language Model (LLM), much less training one from scratch; Lastly, to identify the records of interest, expert inputs are needed to manually label the datasets, which can be time-consuming and costly. We present in this paper a pipeline that enables the classification of triage data using LLM and limited compute resources. We first fine-tuned a pre-trained LLM with a classifier using a small (2k) open sourced dataset on a GPU; and then further fine-tuned the model with a hospital specific dataset of 1000 samples on a CPU. We demonstrated that by carefully curating the datasets and leveraging existing models and open sourced data, we can successfully classify triage data with limited compute resources. 

---
# Research on Multi-hop Inference Optimization of LLM Based on MQUAKE Framework 

**Authors**: Zucheng Liang, Wenxin Wei, Kaijie Zhang, Hongyi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.04770)  

**Abstract**: Accurately answering complex questions has consistently been a significant challenge for Large Language Models (LLMs). To address this, this paper proposes a multi-hop question decomposition method for complex questions, building upon research within the MQUAKE framework. Utilizing the LLAMA3 model, we systematically investigate the impact of multi-hop question decomposition within knowledge graphs on model comprehension and reasoning accuracy, both before and after model training. In our experiments, we systematically partitioned and converted the MQUAKE-T dataset into two distinct formats: a single-hop dataset designed for directly answering complex questions, and a multi-hop dataset constructed using the multi-hop question decomposition method. We then fine-tuned the LLAMA3 model on these datasets and conducted inference tests. Our results demonstrate that, without fine-tuning the LLM, the prediction performance based on the multi-hop question decomposition method significantly outperforms the method of directly answering complex questions. After fine-tuning using the LoRA (Low-Rank Adaptation) method, the performance of both approaches improved compared to the untrained baseline. Crucially, the method utilizing multi-hop decomposition consistently maintained its superiority. These findings validate the effectiveness of the multi-hop decomposition method both before and after training, demonstrating its capability to effectively enhance the LLM's ability to answer complex questions. 

---
# Knowledge Collapse in LLMs: When Fluency Survives but Facts Fail under Recursive Synthetic Training 

**Authors**: Figarri Keisha, Zekun Wu, Ze Wang, Adriano Koshiyama, Philip Treleaven  

**Link**: [PDF](https://arxiv.org/pdf/2509.04796)  

**Abstract**: Large language models increasingly rely on synthetic data due to human-written content scarcity, yet recursive training on model-generated outputs leads to model collapse, a degenerative process threatening factual reliability. We define knowledge collapse as a distinct three-stage phenomenon where factual accuracy deteriorates while surface fluency persists, creating "confidently wrong" outputs that pose critical risks in accuracy-dependent domains. Through controlled experiments with recursive synthetic training, we demonstrate that collapse trajectory and timing depend critically on instruction format, distinguishing instruction-following collapse from traditional model collapse through its conditional, prompt-dependent nature. We propose domain-specific synthetic training as a targeted mitigation strategy that achieves substantial improvements in collapse resistance while maintaining computational efficiency. Our evaluation framework combines model-centric indicators with task-centric metrics to detect distinct degradation phases, enabling reproducible assessment of epistemic deterioration across different language models. These findings provide both theoretical insights into collapse dynamics and practical guidance for sustainable AI training in knowledge-intensive applications where accuracy is paramount. 

---
# KERAG: Knowledge-Enhanced Retrieval-Augmented Generation for Advanced Question Answering 

**Authors**: Yushi Sun, Kai Sun, Yifan Ethan Xu, Xiao Yang, Xin Luna Dong, Nan Tang, Lei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.04716)  

**Abstract**: Retrieval-Augmented Generation (RAG) mitigates hallucination in Large Language Models (LLMs) by incorporating external data, with Knowledge Graphs (KGs) offering crucial information for question answering. Traditional Knowledge Graph Question Answering (KGQA) methods rely on semantic parsing, which typically retrieves knowledge strictly necessary for answer generation, thus often suffer from low coverage due to rigid schema requirements and semantic ambiguity. We present KERAG, a novel KG-based RAG pipeline that enhances QA coverage by retrieving a broader subgraph likely to contain relevant information. Our retrieval-filtering-summarization approach, combined with fine-tuned LLMs for Chain-of-Thought reasoning on knowledge sub-graphs, reduces noises and improves QA for both simple and complex questions. Experiments demonstrate that KERAG surpasses state-of-the-art solutions by about 7% in quality and exceeds GPT-4o (Tool) by 10-21%. 

---
# ODKE+: Ontology-Guided Open-Domain Knowledge Extraction with LLMs 

**Authors**: Samira Khorshidi, Azadeh Nikfarjam, Suprita Shankar, Yisi Sang, Yash Govind, Hyun Jang, Ali Kasgari, Alexis McClimans, Mohamed Soliman, Vishnu Konda, Ahmed Fakhry, Xiaoguang Qi  

**Link**: [PDF](https://arxiv.org/pdf/2509.04696)  

**Abstract**: Knowledge graphs (KGs) are foundational to many AI applications, but maintaining their freshness and completeness remains costly. We present ODKE+, a production-grade system that automatically extracts and ingests millions of open-domain facts from web sources with high precision. ODKE+ combines modular components into a scalable pipeline: (1) the Extraction Initiator detects missing or stale facts, (2) the Evidence Retriever collects supporting documents, (3) hybrid Knowledge Extractors apply both pattern-based rules and ontology-guided prompting for large language models (LLMs), (4) a lightweight Grounder validates extracted facts using a second LLM, and (5) the Corroborator ranks and normalizes candidate facts for ingestion. ODKE+ dynamically generates ontology snippets tailored to each entity type to align extractions with schema constraints, enabling scalable, type-consistent fact extraction across 195 predicates. The system supports batch and streaming modes, processing over 9 million Wikipedia pages and ingesting 19 million high-confidence facts with 98.8% precision. ODKE+ significantly improves coverage over traditional methods, achieving up to 48% overlap with third-party KGs and reducing update lag by 50 days on average. Our deployment demonstrates that LLM-based extraction, grounded in ontological structure and verification workflows, can deliver trustworthiness, production-scale knowledge ingestion with broad real-world applicability. A recording of the system demonstration is included with the submission and is also available at this https URL. 

---
# Enhancing Diversity in Large Language Models via Determinantal Point Processes 

**Authors**: Yilei Chen, Souradip Chakraborty, Lorenz Wolf, Ioannis Ch. Paschalidis, Aldo Pacchiano  

**Link**: [PDF](https://arxiv.org/pdf/2509.04784)  

**Abstract**: Supervised fine-tuning and reinforcement learning are two popular methods for post-training large language models (LLMs). While improving the model's performance on downstream tasks, they often reduce the model's output diversity, leading to narrow, canonical responses. Existing methods to enhance diversity are limited, either by operating at inference time or by focusing on lexical differences. We propose a novel training method named DQO based on determinantal point processes (DPPs) to jointly optimize LLMs for quality and semantic diversity. Our approach samples and embeds a group of responses for each prompt, then uses the determinant of a kernel-based similarity matrix to measure diversity as the volume spanned by the embeddings of these responses. Experiments across instruction-following, summarization, story generation, and reasoning tasks demonstrate that our method substantially improves semantic diversity without sacrificing model quality. 

---
# AraHalluEval: A Fine-grained Hallucination Evaluation Framework for Arabic LLMs 

**Authors**: Aisha Alansari, Hamzah Luqman  

**Link**: [PDF](https://arxiv.org/pdf/2509.04656)  

**Abstract**: Recently, extensive research on the hallucination of the large language models (LLMs) has mainly focused on the English language. Despite the growing number of multilingual and Arabic-specific LLMs, evaluating LLMs' hallucination in the Arabic context remains relatively underexplored. The knowledge gap is particularly pressing given Arabic's widespread use across many regions and its importance in global communication and media. This paper presents the first comprehensive hallucination evaluation of Arabic and multilingual LLMs on two critical Arabic natural language generation tasks: generative question answering (GQA) and summarization. This study evaluates a total of 12 LLMs, including 4 Arabic pre-trained models, 4 multilingual models, and 4 reasoning-based models. To assess the factual consistency and faithfulness of LLMs' outputs, we developed a fine-grained hallucination evaluation framework consisting of 12 fine-grained hallucination indicators that represent the varying characteristics of each task. The results reveal that factual hallucinations are more prevalent than faithfulness errors across all models and tasks. Notably, the Arabic pre-trained model Allam consistently demonstrates lower hallucination rates than multilingual models and a comparative performance with reasoning-based models. The code is available at: \href{this https URL}{Github link}. 

---
# A Study of Large Language Models for Patient Information Extraction: Model Architecture, Fine-Tuning Strategy, and Multi-task Instruction Tuning 

**Authors**: Cheng Peng, Xinyu Dong, Mengxian Lyu, Daniel Paredes, Yaoyun Zhang, Yonghui Wu  

**Link**: [PDF](https://arxiv.org/pdf/2509.04753)  

**Abstract**: Natural language processing (NLP) is a key technology to extract important patient information from clinical narratives to support healthcare applications. The rapid development of large language models (LLMs) has revolutionized many NLP tasks in the clinical domain, yet their optimal use in patient information extraction tasks requires further exploration. This study examines LLMs' effectiveness in patient information extraction, focusing on LLM architectures, fine-tuning strategies, and multi-task instruction tuning techniques for developing robust and generalizable patient information extraction systems. This study aims to explore key concepts of using LLMs for clinical concept and relation extraction tasks, including: (1) encoder-only or decoder-only LLMs, (2) prompt-based parameter-efficient fine-tuning (PEFT) algorithms, and (3) multi-task instruction tuning on few-shot learning performance. We benchmarked a suite of LLMs, including encoder-based LLMs (BERT, GatorTron) and decoder-based LLMs (GatorTronGPT, Llama 3.1, GatorTronLlama), across five datasets. We compared traditional full-size fine-tuning and prompt-based PEFT. We explored a multi-task instruction tuning framework that combines both tasks across four datasets to evaluate the zero-shot and few-shot learning performance using the leave-one-dataset-out strategy. 

---
# Breaking to Build: A Threat Model of Prompt-Based Attacks for Securing LLMs 

**Authors**: Brennen Hill, Surendra Parla, Venkata Abhijeeth Balabhadruni, Atharv Prajod Padmalayam, Sujay Chandra Shekara Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2509.04615)  

**Abstract**: The proliferation of Large Language Models (LLMs) has introduced critical security challenges, where adversarial actors can manipulate input prompts to cause significant harm and circumvent safety alignments. These prompt-based attacks exploit vulnerabilities in a model's design, training, and contextual understanding, leading to intellectual property theft, misinformation generation, and erosion of user trust. A systematic understanding of these attack vectors is the foundational step toward developing robust countermeasures. This paper presents a comprehensive literature survey of prompt-based attack methodologies, categorizing them to provide a clear threat model. By detailing the mechanisms and impacts of these exploits, this survey aims to inform the research community's efforts in building the next generation of secure LLMs that are inherently resistant to unauthorized distillation, fine-tuning, and editing. 

---
# Sample-efficient Integration of New Modalities into Large Language Models 

**Authors**: Osman Batur İnce, André F. T. Martins, Oisin Mac Aodha, Edoardo M. Ponti  

**Link**: [PDF](https://arxiv.org/pdf/2509.04606)  

**Abstract**: Multimodal foundation models can process several modalities. However, since the space of possible modalities is large and evolving over time, training a model from scratch to encompass all modalities is unfeasible. Moreover, integrating a modality into a pre-existing foundation model currently requires a significant amount of paired data, which is often not available for low-resource modalities. In this paper, we introduce a method for sample-efficient modality integration (SEMI) into Large Language Models (LLMs). To this end, we devise a hypernetwork that can adapt a shared projector -- placed between modality-specific encoders and an LLM -- to any modality. The hypernetwork, trained on high-resource modalities (i.e., text, speech, audio, video), is conditioned on a few samples from any arbitrary modality at inference time to generate a suitable adapter. To increase the diversity of training modalities, we artificially multiply the number of encoders through isometric transformations. We find that SEMI achieves a significant boost in sample efficiency during few-shot integration of new modalities (i.e., satellite images, astronomical images, inertial measurements, and molecules) with encoders of arbitrary embedding dimensionality. For instance, to reach the same accuracy as 32-shot SEMI, training the projector from scratch needs 64$\times$ more data. As a result, SEMI holds promise to extend the modality coverage of foundation models. 

---
# Manipulating Transformer-Based Models: Controllability, Steerability, and Robust Interventions 

**Authors**: Faruk Alpay, Taylan Alpay  

**Link**: [PDF](https://arxiv.org/pdf/2509.04549)  

**Abstract**: Transformer-based language models excel in NLP tasks, but fine-grained control remains challenging. This paper explores methods for manipulating transformer models through principled interventions at three levels: prompts, activations, and weights. We formalize controllable text generation as an optimization problem addressable via prompt engineering, parameter-efficient fine-tuning, model editing, and reinforcement learning. We introduce a unified framework encompassing prompt-level steering, activation interventions, and weight-space edits. We analyze robustness and safety implications, including adversarial attacks and alignment mitigations. Theoretically, we show minimal weight updates can achieve targeted behavior changes with limited side-effects. Empirically, we demonstrate >90% success in sentiment control and factual edits while preserving base performance, though generalization-specificity trade-offs exist. We discuss ethical dual-use risks and the need for rigorous evaluation. This work lays groundwork for designing controllable and robust language models. 

---
# Why Language Models Hallucinate 

**Authors**: Adam Tauman Kalai, Ofir Nachum, Santosh S. Vempala, Edwin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.04664)  

**Abstract**: Like students facing hard exam questions, large language models sometimes guess when uncertain, producing plausible yet incorrect statements instead of admitting uncertainty. Such "hallucinations" persist even in state-of-the-art systems and undermine trust. We argue that language models hallucinate because the training and evaluation procedures reward guessing over acknowledging uncertainty, and we analyze the statistical causes of hallucinations in the modern training pipeline. Hallucinations need not be mysterious -- they originate simply as errors in binary classification. If incorrect statements cannot be distinguished from facts, then hallucinations in pretrained language models will arise through natural statistical pressures. We then argue that hallucinations persist due to the way most evaluations are graded -- language models are optimized to be good test-takers, and guessing when uncertain improves test performance. This "epidemic" of penalizing uncertain responses can only be addressed through a socio-technical mitigation: modifying the scoring of existing benchmarks that are misaligned but dominate leaderboards, rather than introducing additional hallucination evaluations. This change may steer the field toward more trustworthy AI systems. 

---
# Scaling behavior of large language models in emotional safety classification across sizes and tasks 

**Authors**: Edoardo Pinzuti, Oliver Tüscher, André Ferreira Castro  

**Link**: [PDF](https://arxiv.org/pdf/2509.04512)  

**Abstract**: Understanding how large language models (LLMs) process emotionally sensitive content is critical for building safe and reliable systems, particularly in mental health contexts. We investigate the scaling behavior of LLMs on two key tasks: trinary classification of emotional safety (safe vs. unsafe vs. borderline) and multi-label classification using a six-category safety risk taxonomy. To support this, we construct a novel dataset by merging several human-authored mental health datasets (> 15K samples) and augmenting them with emotion re-interpretation prompts generated via ChatGPT. We evaluate four LLaMA models (1B, 3B, 8B, 70B) across zero-shot, few-shot, and fine-tuning settings. Our results show that larger LLMs achieve stronger average performance, particularly in nuanced multi-label classification and in zero-shot settings. However, lightweight fine-tuning allowed the 1B model to achieve performance comparable to larger models and BERT in several high-data categories, while requiring <2GB VRAM at inference. These findings suggest that smaller, on-device models can serve as viable, privacy-preserving alternatives for sensitive applications, offering the ability to interpret emotional context and maintain safe conversational boundaries. This work highlights key implications for therapeutic LLM applications and the scalable alignment of safety-critical systems. 

---
# From Silent Signals to Natural Language: A Dual-Stage Transformer-LLM Approach 

**Authors**: Nithyashree Sivasubramaniam  

**Link**: [PDF](https://arxiv.org/pdf/2509.04507)  

**Abstract**: Silent Speech Interfaces (SSIs) have gained attention for their ability to generate intelligible speech from non-acoustic signals. While significant progress has been made in advancing speech generation pipelines, limited work has addressed the recognition and downstream processing of synthesized speech, which often suffers from phonetic ambiguity and noise. To overcome these challenges, we propose an enhanced automatic speech recognition framework that combines a transformer-based acoustic model with a large language model (LLM) for post-processing. The transformer captures full utterance context, while the LLM ensures linguistic consistency. Experimental results show a 16% relative and 6% absolute reduction in word error rate (WER) over a 36% baseline, demonstrating substantial improvements in intelligibility for silent speech interfaces. 

---
# VaccineRAG: Boosting Multimodal Large Language Models' Immunity to Harmful RAG Samples 

**Authors**: Qixin Sun, Ziqin Wang, Hengyuan Zhao, Yilin Li, Kaiyou Song, Linjiang Huang, Xiaolin Hu, Qingpei Guo, Si Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.04502)  

**Abstract**: Retrieval Augmented Generation enhances the response accuracy of Large Language Models (LLMs) by integrating retrieval and generation modules with external knowledge, demonstrating particular strength in real-time queries and Visual Question Answering tasks. However, the effectiveness of RAG is frequently hindered by the precision of the retriever: many retrieved samples fed into the generation phase are irrelevant or misleading, posing a critical bottleneck to LLMs' performance. To address this challenge, we introduce VaccineRAG, a novel Chain-of-Thought-based retrieval-augmented generation dataset. On one hand, VaccineRAG employs a benchmark to evaluate models using data with varying positive/negative sample ratios, systematically exposing inherent weaknesses in current LLMs. On the other hand, it enhances models' sample-discrimination capabilities by prompting LLMs to generate explicit Chain-of-Thought (CoT) analysis for each sample before producing final answers. Furthermore, to enhance the model's ability to learn long-sequence complex CoT content, we propose Partial-GRPO. By modeling the outputs of LLMs as multiple components rather than a single whole, our model can make more informed preference selections for complex sequences, thereby enhancing its capacity to learn complex CoT. Comprehensive evaluations and ablation studies on VaccineRAG validate the effectiveness of the proposed scheme. The code and dataset will be publicly released soon. 

---
# Personality as a Probe for LLM Evaluation: Method Trade-offs and Downstream Effects 

**Authors**: Gunmay Handa, Zekun Wu, Adriano Koshiyama, Philip Treleaven  

**Link**: [PDF](https://arxiv.org/pdf/2509.04794)  

**Abstract**: Personality manipulation in large language models (LLMs) is increasingly applied in customer service and agentic scenarios, yet its mechanisms and trade-offs remain unclear. We present a systematic study of personality control using the Big Five traits, comparing in-context learning (ICL), parameter-efficient fine-tuning (PEFT), and mechanistic steering (MS). Our contributions are fourfold. First, we construct a contrastive dataset with balanced high/low trait responses, enabling effective steering vector computation and fair cross-method evaluation. Second, we introduce a unified evaluation framework based on within-run $\Delta$ analysis that disentangles, reasoning capability, agent performance, and demographic bias across MMLU, GAIA, and BBQ benchmarks. Third, we develop trait purification techniques to separate openness from conscientiousness, addressing representational overlap in trait encoding. Fourth, we propose a three-level stability framework that quantifies method-, trait-, and combination-level robustness, offering practical guidance under deployment constraints. Experiments on Gemma-2-2B-IT and LLaMA-3-8B-Instruct reveal clear trade-offs: ICL achieves strong alignment with minimal capability loss, PEFT delivers the highest alignment at the cost of degraded task performance, and MS provides lightweight runtime control with competitive effectiveness. Trait-level analysis shows openness as uniquely challenging, agreeableness as most resistant to ICL, and personality encoding consolidating around intermediate layers. Taken together, these results establish personality manipulation as a multi-level probe into behavioral representation, linking surface conditioning, parameter encoding, and activation-level steering, and positioning mechanistic steering as a lightweight alternative to fine-tuning for both deployment and interpretability. 

---
# Context Engineering for Trustworthiness: Rescorla Wagner Steering Under Mixed and Inappropriate Contexts 

**Authors**: Rushi Wang, Jiateng Liu, Cheng Qian, Yifan Shen, Yanzhou Pan, Zhaozhuo Xu, Ahmed Abbasi, Heng Ji, Denghui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.04500)  

**Abstract**: Incorporating external context can significantly enhance the response quality of Large Language Models (LLMs). However, real-world contexts often mix relevant information with disproportionate inappropriate content, posing reliability risks. How do LLMs process and prioritize mixed context? To study this, we introduce the Poisoned Context Testbed, pairing queries with real-world contexts containing relevant and inappropriate content. Inspired by associative learning in animals, we adapt the Rescorla-Wagner (RW) model from neuroscience to quantify how competing contextual signals influence LLM outputs. Our adapted model reveals a consistent behavioral pattern: LLMs exhibit a strong tendency to incorporate information that is less prevalent in the context. This susceptibility is harmful in real-world settings, where small amounts of inappropriate content can substantially degrade response quality. Empirical evaluations on our testbed further confirm this vulnerability. To tackle this, we introduce RW-Steering, a two-stage finetuning-based approach that enables the model to internally identify and ignore inappropriate signals. Unlike prior methods that rely on extensive supervision across diverse context mixtures, RW-Steering generalizes robustly across varying proportions of inappropriate content. Experiments show that our best fine-tuned model improves response quality by 39.8% and reverses the undesirable behavior curve, establishing RW-Steering as a robust, generalizable context engineering solution for improving LLM safety in real-world use. 

---
# Behavioral Fingerprinting of Large Language Models 

**Authors**: Zehua Pei, Hui-Ling Zhen, Ying Zhang, Zhiyuan Yang, Xing Li, Xianzhi Yu, Mingxuan Yuan, Bei Yu  

**Link**: [PDF](https://arxiv.org/pdf/2509.04504)  

**Abstract**: Current benchmarks for Large Language Models (LLMs) primarily focus on performance metrics, often failing to capture the nuanced behavioral characteristics that differentiate them. This paper introduces a novel ``Behavioral Fingerprinting'' framework designed to move beyond traditional evaluation by creating a multi-faceted profile of a model's intrinsic cognitive and interactive styles. Using a curated \textit{Diagnostic Prompt Suite} and an innovative, automated evaluation pipeline where a powerful LLM acts as an impartial judge, we analyze eighteen models across capability tiers. Our results reveal a critical divergence in the LLM landscape: while core capabilities like abstract and causal reasoning are converging among top models, alignment-related behaviors such as sycophancy and semantic robustness vary dramatically. We further document a cross-model default persona clustering (ISTJ/ESTJ) that likely reflects common alignment incentives. Taken together, this suggests that a model's interactive nature is not an emergent property of its scale or reasoning power, but a direct consequence of specific, and highly variable, developer alignment strategies. Our framework provides a reproducible and scalable methodology for uncovering these deep behavioral differences. Project: this https URL 

---
# Learned Hallucination Detection in Black-Box LLMs using Token-level Entropy Production Rate 

**Authors**: Charles Moslonka, Hicham Randrianarivo, Arthur Garnier, Emmanuel Malherbe  

**Link**: [PDF](https://arxiv.org/pdf/2509.04492)  

**Abstract**: Hallucinations in Large Language Model (LLM) outputs for Question Answering (QA) tasks critically undermine their real-world reliability. This paper introduces an applied methodology for robust, one-shot hallucination detection, specifically designed for scenarios with limited data access, such as interacting with black-box LLM APIs that typically expose only a few top candidate log-probabilities per token. Our approach derives uncertainty indicators directly from these readily available log-probabilities generated during non-greedy decoding. We first derive an Entropy Production Rate (EPR) metric that offers baseline performance, later augmented with supervised learning. Our learned model uses features representing the entropic contributions of the accessible top-ranked tokens within a single generated sequence, requiring no multiple query re-runs. Evaluated across diverse QA datasets and multiple LLMs, this estimator significantly improves hallucination detection over using EPR alone. Crucially, high performance is demonstrated using only the typically small set of available log-probabilities (e.g., top <10 per token), confirming its practical efficiency and suitability for these API-constrained deployments. This work provides a readily deployable technique to enhance the trustworthiness of LLM responses from a single generation pass in QA and Retrieval-Augmented Generation (RAG) systems, with its utility further demonstrated in a finance framework analyzing responses to queries on annual reports from an industrial dataset. 

---
# Where Should I Study? Biased Language Models Decide! Evaluating Fairness in LMs for Academic Recommendations 

**Authors**: Krithi Shailya, Akhilesh Kumar Mishra, Gokul S Krishnan, Balaraman Ravindran  

**Link**: [PDF](https://arxiv.org/pdf/2509.04498)  

**Abstract**: Large Language Models (LLMs) are increasingly used as daily recommendation systems for tasks like education planning, yet their recommendations risk perpetuating societal biases. This paper empirically examines geographic, demographic, and economic biases in university and program suggestions from three open-source LLMs: LLaMA-3.1-8B, Gemma-7B, and Mistral-7B. Using 360 simulated user profiles varying by gender, nationality, and economic status, we analyze over 25,000 recommendations. Results show strong biases: institutions in the Global North are disproportionately favored, recommendations often reinforce gender stereotypes, and institutional repetition is prevalent. While LLaMA-3.1 achieves the highest diversity, recommending 481 unique universities across 58 countries, systemic disparities persist. To quantify these issues, we propose a novel, multi-dimensional evaluation framework that goes beyond accuracy by measuring demographic and geographic representation. Our findings highlight the urgent need for bias consideration in educational LMs to ensure equitable global access to higher education. 

---
# Using LLMs to create analytical datasets: A case study of reconstructing the historical memory of Colombia 

**Authors**: David Anderson, Galia Benitez, Margret Bjarnadottir, Shriyan Reyya  

**Link**: [PDF](https://arxiv.org/pdf/2509.04523)  

**Abstract**: Colombia has been submerged in decades of armed conflict, yet until recently, the systematic documentation of violence was not a priority for the Colombian government. This has resulted in a lack of publicly available conflict information and, consequently, a lack of historical accounts. This study contributes to Colombia's historical memory by utilizing GPT, a large language model (LLM), to read and answer questions about over 200,000 violence-related newspaper articles in Spanish. We use the resulting dataset to conduct both descriptive analysis and a study of the relationship between violence and the eradication of coca crops, offering an example of policy analyses that such data can support. Our study demonstrates how LLMs have opened new research opportunities by enabling examinations of large text corpora at a previously infeasible depth. 

---
# ParaThinker: Native Parallel Thinking as a New Paradigm to Scale LLM Test-time Compute 

**Authors**: Hao Wen, Yifan Su, Feifei Zhang, Yunxin Liu, Yunhao Liu, Ya-Qin Zhang, Yuanchun Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.04475)  

**Abstract**: Recent advances in Large Language Models (LLMs) have been driven by test-time compute scaling - a strategy that improves reasoning by generating longer, sequential thought processes. While effective, this approach encounters a significant bottleneck as computation increases, where further computation offers only marginal performance gains. We argue this ceiling is not an inherent limit of the model's capability but a flaw in the scaling strategy itself, a phenomenon we term "Tunnel Vision", where a model's imperfect initial steps lock it into a suboptimal reasoning path. To overcome this, we introduce a new scaling paradigm: native thought parallelism. We present ParaThinker, an end-to-end framework that trains an LLM to generate multiple, diverse reasoning paths in parallel and synthesize them into a superior final answer. By exploring different lines of thoughts simultaneously, ParaThinker effectively sidesteps the Tunnel Vision issue and unlocks the model's latent reasoning potential. Our approach demonstrates that scaling compute in parallel (width) is a more effective and efficient way to superior reasoning than simply scaling sequentially (depth). On challenging reasoning benchmarks, ParaThinker achieves substantial accuracy improvements over sequential LLMs (12.3% for 1.5B and 7.5% for 7B models on average with 8 parallel paths), while adding only negligible latency overhead (7.1%). This enables smaller models to surpass much larger counterparts and establishes parallel thinking as a critical, efficient dimension for scaling future LLMs. 

---
# Evaluating Large Language Models for Financial Reasoning: A CFA-Based Benchmark Study 

**Authors**: Xuan Yao, Qianteng Wang, Xinbo Liu, Ke-Wei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2509.04468)  

**Abstract**: The rapid advancement of large language models presents significant opportunities for financial applications, yet systematic evaluation in specialized financial contexts remains limited. This study presents the first comprehensive evaluation of state-of-the-art LLMs using 1,560 multiple-choice questions from official mock exams across Levels I-III of CFA, most rigorous professional certifications globally that mirror real-world financial analysis complexity. We compare models distinguished by core design priorities: multi-modal and computationally powerful, reasoning-specialized and highly accurate, and lightweight efficiency-optimized.
We assess models under zero-shot prompting and through a novel Retrieval-Augmented Generation pipeline that integrates official CFA curriculum content. The RAG system achieves precise domain-specific knowledge retrieval through hierarchical knowledge organization and structured query generation, significantly enhancing reasoning accuracy in professional financial certification evaluation.
Results reveal that reasoning-oriented models consistently outperform others in zero-shot settings, while the RAG pipeline provides substantial improvements particularly for complex scenarios. Comprehensive error analysis identifies knowledge gaps as the primary failure mode, with minimal impact from text readability. These findings provide actionable insights for LLM deployment in finance, offering practitioners evidence-based guidance for model selection and cost-performance optimization. 

---
# MOSAIC: A Multilingual, Taxonomy-Agnostic, and Computationally Efficient Approach for Radiological Report Classification 

**Authors**: Alice Schiavone, Marco Fraccaro, Lea Marie Pehrson, Silvia Ingala, Rasmus Bonnevie, Michael Bachmann Nielsen, Vincent Beliveau, Melanie Ganz, Desmond Elliott  

**Link**: [PDF](https://arxiv.org/pdf/2509.04471)  

**Abstract**: Radiology reports contain rich clinical information that can be used to train imaging models without relying on costly manual annotation. However, existing approaches face critical limitations: rule-based methods struggle with linguistic variability, supervised models require large annotated datasets, and recent LLM-based systems depend on closed-source or resource-intensive models that are unsuitable for clinical use. Moreover, current solutions are largely restricted to English and single-modality, single-taxonomy datasets. We introduce MOSAIC, a multilingual, taxonomy-agnostic, and computationally efficient approach for radiological report classification. Built on a compact open-access language model (MedGemma-4B), MOSAIC supports both zero-/few-shot prompting and lightweight fine-tuning, enabling deployment on consumer-grade GPUs. We evaluate MOSAIC across seven datasets in English, Spanish, French, and Danish, spanning multiple imaging modalities and label taxonomies. The model achieves a mean macro F1 score of 88 across five chest X-ray datasets, approaching or exceeding expert-level performance, while requiring only 24 GB of GPU memory. With data augmentation, as few as 80 annotated samples are sufficient to reach a weighted F1 score of 82 on Danish reports, compared to 86 with the full 1600-sample training set. MOSAIC offers a practical alternative to large or proprietary LLMs in clinical settings. Code and models are open-source. We invite the community to evaluate and extend MOSAIC on new languages, taxonomies, and modalities. 

---
# Quantized Large Language Models in Biomedical Natural Language Processing: Evaluation and Recommendation 

**Authors**: Zaifu Zhan, Shuang Zhou, Min Zeng, Kai Yu, Meijia Song, Xiaoyi Chen, Jun Wang, Yu Hou, Rui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.04534)  

**Abstract**: Large language models have demonstrated remarkable capabilities in biomedical natural language processing, yet their rapid growth in size and computational requirements present a major barrier to adoption in healthcare settings where data privacy precludes cloud deployment and resources are limited. In this study, we systematically evaluated the impact of quantization on 12 state-of-the-art large language models, including both general-purpose and biomedical-specific models, across eight benchmark datasets covering four key tasks: named entity recognition, relation extraction, multi-label classification, and question answering. We show that quantization substantially reduces GPU memory requirements-by up to 75%-while preserving model performance across diverse tasks, enabling the deployment of 70B-parameter models on 40GB consumer-grade GPUs. In addition, domain-specific knowledge and responsiveness to advanced prompting methods are largely maintained. These findings provide significant practical and guiding value, highlighting quantization as a practical and effective strategy for enabling the secure, local deployment of large yet high-capacity language models in biomedical contexts, bridging the gap between technical advances in AI and real-world clinical translation. 

---
# COCORELI: Cooperative, Compositional Reconstitution \& Execution of Language Instructions 

**Authors**: Swarnadeep Bhar, Omar Naim, Eleni Metheniti, Bastien Navarri, Loïc Cabannes, Morteza Ezzabady, Nicholas Asher  

**Link**: [PDF](https://arxiv.org/pdf/2509.04470)  

**Abstract**: We present COCORELI, a hybrid agent framework designed to tackle the limitations of large language models (LLMs) in tasks requiring: following complex instructions, minimizing hallucination, and spatial reasoning. COCORELI integrates medium-sized LLM agents with novel abstraction mechanisms and a discourse module to parse instructions to in-context learn dynamic, high-level representations of the environment. Experiments on natural collaborative construction tasks show that COCORELI outperforms single-LLM CoT and agentic LLM systems, all using larger LLMs. It manages to largely avoid hallucinations, identify missing information, ask for clarifications, and update its learned objects. COCORELI's abstraction abilities extend beyond ENVIRONMENT, as shown in the ToolBench API completion task. 

---
# Benchmarking GPT-5 for biomedical natural language processing 

**Authors**: Yu Hou, Zaifu Zhan, Rui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.04462)  

**Abstract**: The rapid expansion of biomedical literature has heightened the need for scalable natural language processing (NLP) solutions. While GPT-4 substantially narrowed the gap with task-specific systems, especially in question answering, its performance across other domains remained uneven. We updated a standardized BioNLP benchmark to evaluate GPT-5 and GPT-4o under zero-, one-, and five-shot prompting across 12 datasets spanning six task families: named entity recognition, relation extraction, multi-label document classification, question answering, text summarization, and text simplification. Using fixed prompt templates, identical decoding parameters, and batch inference, we report primary metrics per dataset and include prior results for GPT-4, GPT-3.5, and LLaMA-2-13B for comparison. GPT-5 achieved the strongest overall benchmark performance, with macro-average scores rising to 0.557 under five-shot prompting versus 0.506 for GPT-4 and 0.508 for GPT-4o. On MedQA, GPT-5 reached 94.1% accuracy, exceeding the previous supervised state of the art by over fifty points, and attained parity with supervised systems on PubMedQA (0.734). In extraction tasks, GPT-5 delivered major gains in chemical NER (0.886 F1) and ChemProt relation extraction (0.616 F1), outperforming GPT-4 and GPT-4o, though summarization and disease NER still lagged behind domain-specific baselines. These results establish GPT-5 as a general-purpose model now offering deployment-ready performance for reasoning-oriented biomedical QA, while precision-critical extraction and evidence-dense summarization continue to favor fine-tuned or hybrid approaches. The benchmark delineates where simple prompting suffices and where retrieval-augmented or planning-based scaffolds are likely required, providing actionable guidance for BioNLP system design as frontier models advance. 

---
# From Post To Personality: Harnessing LLMs for MBTI Prediction in Social Media 

**Authors**: Tian Ma, Kaiyu Feng, Yu Rong, Kangfei Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.04461)  

**Abstract**: Personality prediction from social media posts is a critical task that implies diverse applications in psychology and sociology. The Myers Briggs Type Indicator (MBTI), a popular personality inventory, has been traditionally predicted by machine learning (ML) and deep learning (DL) techniques. Recently, the success of Large Language Models (LLMs) has revealed their huge potential in understanding and inferring personality traits from social media content. However, directly exploiting LLMs for MBTI prediction faces two key challenges: the hallucination problem inherent in LLMs and the naturally imbalanced distribution of MBTI types in the population. In this paper, we propose PostToPersonality (PtoP), a novel LLM based framework for MBTI prediction from social media posts of individuals. Specifically, PtoP leverages Retrieval Augmented Generation with in context learning to mitigate hallucination in LLMs. Furthermore, we fine tune a pretrained LLM to improve model specification in MBTI understanding with synthetic minority oversampling, which balances the class imbalance by generating synthetic samples. Experiments conducted on a real world social media dataset demonstrate that PtoP achieves state of the art performance compared with 10 ML and DL baselines. 

---
# Enhancing LLM Efficiency: Targeted Pruning for Prefill-Decode Disaggregation in Inference 

**Authors**: Hao Zhang, Mengsi Lyu, Yulong Ao, Yonghua Lin  

**Link**: [PDF](https://arxiv.org/pdf/2509.04467)  

**Abstract**: Large Language Models (LLMs) demonstrate exceptional capabilities across various tasks, but their deployment is constrained by high computational and memory costs. Model pruning provides an effective means to alleviate these demands. However, existing methods often ignore the characteristics of prefill-decode (PD) disaggregation in practice. In this paper, we propose a novel pruning method for PD disaggregation inference, enabling more precise and efficient block and KV Cache pruning. Our approach constructs pruning and distillation sets to perform iterative block removal independently for the prefill and decode stages, obtaining better pruning solutions. Moreover, we introduce a token-aware cache pruning mechanism that retains all KV Cache in the prefill stage but selectively reuses entries for the first and last token sequences in selected layers during decode, reducing communication costs with minimal overhead. Extensive experiments demonstrate that our approach consistently achieves strong performance in both PD disaggregation and PD unified settings without disaggregation. Under the default settings, our method achieves a 20.56% inference speedup and a 4.95 times reduction in data transmission bandwidth consumption. 

---
# Discrete Prompt Tuning via Recursive Utilization of Black-box Multimodal Large Language Model for Personalized Visual Emotion Recognition 

**Authors**: Ryo Takahashi, Naoki Saito, Keisuke Maeda, Takahiro Ogawa, Miki Haseyama  

**Link**: [PDF](https://arxiv.org/pdf/2509.04480)  

**Abstract**: Visual Emotion Recognition (VER) is an important research topic due to its wide range of applications, including opinion mining and advertisement design. Extending this capability to recognize emotions at the individual level further broadens its potential applications. Recently, Multimodal Large Language Models (MLLMs) have attracted increasing attention and demonstrated performance comparable to that of conventional VER methods. However, MLLMs are trained on large and diverse datasets containing general opinions, which causes them to favor majority viewpoints and familiar patterns. This tendency limits their performance in a personalized VER, which is crucial for practical and real-world applications, and indicates a key area for improvement. To address this limitation, the proposed method employs discrete prompt tuning inspired by the process of humans' prompt engineering to adapt the VER task to each individual. Our method selects the best natural language representation from the generated prompts and uses it to update the prompt for the realization of accurate personalized VER. 

---
# Can Multiple Responses from an LLM Reveal the Sources of Its Uncertainty? 

**Authors**: Yang Nan, Pengfei He, Ravi Tandon, Han Xu  

**Link**: [PDF](https://arxiv.org/pdf/2509.04464)  

**Abstract**: Large language models (LLMs) have delivered significant breakthroughs across diverse domains but can still produce unreliable or misleading outputs, posing critical challenges for real-world applications. While many recent studies focus on quantifying model uncertainty, relatively little work has been devoted to \textit{diagnosing the source of uncertainty}. In this study, we show that, when an LLM is uncertain, the patterns of disagreement among its multiple generated responses contain rich clues about the underlying cause of uncertainty. To illustrate this point, we collect multiple responses from a target LLM and employ an auxiliary LLM to analyze their patterns of disagreement. The auxiliary model is tasked to reason about the likely source of uncertainty, such as whether it stems from ambiguity in the input question, a lack of relevant knowledge, or both. In cases involving knowledge gaps, the auxiliary model also identifies the specific missing facts or concepts contributing to the uncertainty. In our experiment, we validate our framework on AmbigQA, OpenBookQA, and MMLU-Pro, confirming its generality in diagnosing distinct uncertainty sources. Such diagnosis shows the potential for relevant manual interventions that improve LLM performance and reliability. 

---
# CoCoNUTS: Concentrating on Content while Neglecting Uninformative Textual Styles for AI-Generated Peer Review Detection 

**Authors**: Yihan Chen, Jiawei Chen, Guozhao Mo, Xuanang Chen, Ben He, Xianpei Han, Le Sun  

**Link**: [PDF](https://arxiv.org/pdf/2509.04460)  

**Abstract**: The growing integration of large language models (LLMs) into the peer review process presents potential risks to the fairness and reliability of scholarly evaluation. While LLMs offer valuable assistance for reviewers with language refinement, there is growing concern over their use to generate substantive review content. Existing general AI-generated text detectors are vulnerable to paraphrasing attacks and struggle to distinguish between surface language refinement and substantial content generation, suggesting that they primarily rely on stylistic cues. When applied to peer review, this limitation can result in unfairly suspecting reviews with permissible AI-assisted language enhancement, while failing to catch deceptively humanized AI-generated reviews. To address this, we propose a paradigm shift from style-based to content-based detection. Specifically, we introduce CoCoNUTS, a content-oriented benchmark built upon a fine-grained dataset of AI-generated peer reviews, covering six distinct modes of human-AI collaboration. Furthermore, we develop CoCoDet, an AI review detector via a multi-task learning framework, designed to achieve more accurate and robust detection of AI involvement in review content. Our work offers a practical foundation for evaluating the use of LLMs in peer review, and contributes to the development of more precise, equitable, and reliable detection methods for real-world scholarly applications. Our code and data will be publicly available at this https URL. 

---
# RECAP: REwriting Conversations for Intent Understanding in Agentic Planning 

**Authors**: Kushan Mitra, Dan Zhang, Hannah Kim, Estevam Hruschka  

**Link**: [PDF](https://arxiv.org/pdf/2509.04472)  

**Abstract**: Understanding user intent is essential for effective planning in conversational assistants, particularly those powered by large language models (LLMs) coordinating multiple agents. However, real-world dialogues are often ambiguous, underspecified, or dynamic, making intent detection a persistent challenge. Traditional classification-based approaches struggle to generalize in open-ended settings, leading to brittle interpretations and poor downstream planning. We propose RECAP (REwriting Conversations for Agent Planning), a new benchmark designed to evaluate and advance intent rewriting, reframing user-agent dialogues into concise representations of user goals. RECAP captures diverse challenges such as ambiguity, intent drift, vagueness, and mixed-goal conversations. Alongside the dataset, we introduce an LLM-based evaluator that assesses planning utility given the rewritten intent. Using RECAP, we develop a prompt-based rewriting approach that outperforms baselines. We further demonstrate that fine-tuning two DPO-based rewriters yields additional utility gains. Our results highlight intent rewriting as a critical and tractable component for improving agent planning in open-domain dialogue systems. 

---
# Uncertainty-Aware Collaborative System of Large and Small Models for Multimodal Sentiment Analysis 

**Authors**: Shiqin Han, Manning Gao, Menghua Jiang, Yuncheng Jiang, Haifeng Hu, Sijie Mai  

**Link**: [PDF](https://arxiv.org/pdf/2509.04459)  

**Abstract**: The advent of Multimodal Large Language Models (MLLMs) has significantly advanced the state-of-the-art in multimodal machine learning, yet their substantial computational demands present a critical barrier to real-world deployment. Conversely, smaller, specialized models offer high efficiency but often at the cost of performance. To reconcile this performance-efficiency trade-off, we propose a novel Uncertainty-Aware Collaborative System (U-ACS) that synergistically orchestrates a powerful MLLM (e.g., HumanOmni) and a lightweight baseline model for multimodal sentiment analysis. The core of our system is an uncertainty-driven cascade mechanism, where the efficient small model first acts as a rapid filter for all input samples. Only those samples yielding high predictive uncertainty, thereby indicating greater difficulty, are selectively escalated to the MLLM for more sophisticated analysis. Furthermore, our system introduces advanced strategies to handle ambiguous or conflicting predictions, including weighted averaging for predictions of similar polarity and a prompt-based cross-verification to resolve conflicting predictions when both models exhibit high uncertainty. This sample-difficulty-aware approach allows for a dynamic allocation of computational resources, drastically reducing inference costs while retaining the high accuracy of MLLM. Extensive experiments on benchmark datasets demonstrate that our proposed method achieves state-of-the-art performance, while requiring only a fraction of the computational resources compared to using a standalone MLLM. 

---
# Emotionally-Aware Agents for Dispute Resolution 

**Authors**: Sushrita Rakshit, James Hale, Kushal Chawla, Jeanne M. Brett, Jonathan Gratch  

**Link**: [PDF](https://arxiv.org/pdf/2509.04465)  

**Abstract**: In conflict, people use emotional expressions to shape their counterparts' thoughts, feelings, and actions. This paper explores whether automatic text emotion recognition offers insight into this influence in the context of dispute resolution. Prior work has shown the promise of such methods in negotiations; however, disputes evoke stronger emotions and different social processes. We use a large corpus of buyer-seller dispute dialogues to investigate how emotional expressions shape subjective and objective outcomes. We further demonstrate that large-language models yield considerably greater explanatory power than previous methods for emotion intensity annotation and better match the decisions of human annotators. Findings support existing theoretical models for how emotional expressions contribute to conflict escalation and resolution and suggest that agent-based systems could be useful in managing disputes by recognizing and potentially mitigating emotional escalation. 

---
# Do MLLMs Really Understand the Charts? 

**Authors**: Xiao Zhang, Dongyuan Li, Liuyu Xiang, Yao Zhang, Cheng Zhong, Zhaofeng He  

**Link**: [PDF](https://arxiv.org/pdf/2509.04457)  

**Abstract**: Although Multimodal Large Language Models (MLLMs) have demonstrated increasingly impressive performance in chart understanding, most of them exhibit alarming hallucinations and significant performance degradation when handling non-annotated charts. Therefore, a question arises: Do MLLMs really understand the charts? Since a human is capable of understanding charts and estimating the values by visual reasoning, we first carefully establish a comprehensive Chart Reasoning Benchmark CRBench to rigorously evaluate the visual reasoning abilities of MLLMs on non-annotated charts. We argue that MLLMs are primarily relying on recognition rather than reasoning to interpret the charts. To steer MLLMs to reasonable chart understanding, we propose ChartReasoner that mimics human behavior by grounding their estimation in chart understanding. Extensive results on the proposed CRBench show that ChartReasnoner-3B/7B achieves superior performance in chart reasoning, even compared to GPT-4o and Gemini-2.5-Flash. More importantly, ChartReasnoner also demonstrates the visual reasoning abilities in general chart comprehension on public benchmarks, leading to significant performance gains and enabling MLLMs to rationally understand the charts. The code and dataset will be publicly available upon publication. 

---
# INSEva: A Comprehensive Chinese Benchmark for Large Language Models in Insurance 

**Authors**: Shisong Chen, Qian Zhu, Wenyan Yang, Chengyi Yang, Zhong Wang, Ping Wang, Xuan Lin, Bo Xu, Daqian Li, Chao Yuan, Licai Qi, Wanqing Xu, sun zhenxing, Xin Lu, Shiqiang Xiong, Chao Chen, Haixiang Hu, Yanghua Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2509.04455)  

**Abstract**: Insurance, as a critical component of the global financial system, demands high standards of accuracy and reliability in AI applications. While existing benchmarks evaluate AI capabilities across various domains, they often fail to capture the unique characteristics and requirements of the insurance domain. To address this gap, we present INSEva, a comprehensive Chinese benchmark specifically designed for evaluating AI systems' knowledge and capabilities in insurance. INSEva features a multi-dimensional evaluation taxonomy covering business areas, task formats, difficulty levels, and cognitive-knowledge dimension, comprising 38,704 high-quality evaluation examples sourced from authoritative materials. Our benchmark implements tailored evaluation methods for assessing both faithfulness and completeness in open-ended responses. Through extensive evaluation of 8 state-of-the-art Large Language Models (LLMs), we identify significant performance variations across different dimensions. While general LLMs demonstrate basic insurance domain competency with average scores above 80, substantial gaps remain in handling complex, real-world insurance scenarios. The benchmark will be public soon. 

---
# Towards Ontology-Based Descriptions of Conversations with Qualitatively-Defined Concepts 

**Authors**: Barbara Gendron, Gaël Guibon, Mathieu D'aquin  

**Link**: [PDF](https://arxiv.org/pdf/2509.04926)  

**Abstract**: The controllability of Large Language Models (LLMs) when used as conversational agents is a key challenge, particularly to ensure predictable and user-personalized responses. This work proposes an ontology-based approach to formally define conversational features that are typically qualitative in nature. By leveraging a set of linguistic descriptors, we derive quantitative definitions for qualitatively-defined concepts, enabling their integration into an ontology for reasoning and consistency checking. We apply this framework to the task of proficiency-level control in conversations, using CEFR language proficiency levels as a case study. These definitions are then formalized in description logic and incorporated into an ontology, which guides controlled text generation of an LLM through fine-tuning. Experimental results demonstrate that our approach provides consistent and explainable proficiency-level definitions, improving transparency in conversational AI. 

---
# Code Review Without Borders: Evaluating Synthetic vs. Real Data for Review Recommendation 

**Authors**: Yogev Cohen, Dudi Ohayon, Romy Somkin, Yehudit Aperstein, Alexander Apartsin  

**Link**: [PDF](https://arxiv.org/pdf/2509.04810)  

**Abstract**: Automating the decision of whether a code change requires manual review is vital for maintaining software quality in modern development workflows. However, the emergence of new programming languages and frameworks creates a critical bottleneck: while large volumes of unlabelled code are readily available, there is an insufficient amount of labelled data to train supervised models for review classification. We address this challenge by leveraging Large Language Models (LLMs) to translate code changes from well-resourced languages into equivalent changes in underrepresented or emerging languages, generating synthetic training data where labelled examples are scarce. We assume that although LLMs have learned the syntax and semantics of new languages from available unlabelled code, they have yet to fully grasp which code changes are considered significant or review-worthy within these emerging ecosystems. To overcome this, we use LLMs to generate synthetic change examples and train supervised classifiers on them. We systematically compare the performance of these classifiers against models trained on real labelled data. Our experiments across multiple GitHub repositories and language pairs demonstrate that LLM-generated synthetic data can effectively bootstrap review recommendation systems, narrowing the performance gap even in low-resource settings. This approach provides a scalable pathway to extend automated code review capabilities to rapidly evolving technology stacks, even in the absence of annotated data. 

---
# Scaling Up, Speeding Up: A Benchmark of Speculative Decoding for Efficient LLM Test-Time Scaling 

**Authors**: Shengyin Sun, Yiming Li, Xing Li, Yingzhao Lian, Weizhe Lin, Hui-Ling Zhen, Zhiyuan Yang, Chen Chen, Xianzhi Yu, Mingxuan Yuan, Chen Ma  

**Link**: [PDF](https://arxiv.org/pdf/2509.04474)  

**Abstract**: Test-time scaling has emerged as a powerful paradigm for enhancing the reasoning capabilities of large language models (LLMs) by allocating additional computational resources during inference. However, this paradigm is inherently inefficient due to the generation of redundant and repetitive reasoning traces, leading to significant computational overhead. Speculative decoding offers a promising avenue for mitigating this inefficiency, yet its efficacy in the structured, repetition-rich context of test-time scaling remains largely unexplored. To bridge this gap, we introduce the first comprehensive benchmark designed to evaluate speculative decoding methods for accelerating LLM test-time scaling. Our benchmark provides consistent experimental protocols across representative test-time scaling paradigms (e.g., Best-of-N sampling and multi-round thinking), enabling a fair comparison of three major categories of speculative decoding: model-based, training-based, and n-gram-based methods. Extensive experiments reveal that simple n-gram-based methods effectively capture repetitive patterns, demonstrating unique potential in accelerating test-time scaling. This phenomenon demonstrates the value of integrating n-gram-based methods with model-based or training-based approaches to balance acceleration for both repetitive and diverse reasoning in test-time scaling. We hope this benchmark spurs further research on speculative decoding for test-time scaling, enabling faster and more practical reasoning in LLMs through better handling of repetitive and diverse reasoning paths. 

---
# Mentalic Net: Development of RAG-based Conversational AI and Evaluation Framework for Mental Health Support 

**Authors**: Anandi Dutta, Shivani Mruthyunjaya, Jessica Saddington, Kazi Sifatul Islam  

**Link**: [PDF](https://arxiv.org/pdf/2509.04456)  

**Abstract**: The emergence of large language models (LLMs) has unlocked boundless possibilities, along with significant challenges. In response, we developed a mental health support chatbot designed to augment professional healthcare, with a strong emphasis on safe and meaningful application. Our approach involved rigorous evaluation, covering accuracy, empathy, trustworthiness, privacy, and bias. We employed a retrieval-augmented generation (RAG) framework, integrated prompt engineering, and fine-tuned a pre-trained model on novel datasets. The resulting system, Mentalic Net Conversational AI, achieved a BERT Score of 0.898, with other evaluation metrics falling within satisfactory ranges. We advocate for a human-in-the-loop approach and a long-term, responsible strategy in developing such transformative technologies, recognizing both their potential to change lives and the risks they may pose if not carefully managed. 

---
# Language-Driven Hierarchical Task Structures as Explicit World Models for Multi-Agent Learning 

**Authors**: Brennen Hill  

**Link**: [PDF](https://arxiv.org/pdf/2509.04731)  

**Abstract**: The convergence of Language models, Agent models, and World models represents a critical frontier for artificial intelligence. While recent progress has focused on scaling Language and Agent models, the development of sophisticated, explicit World Models remains a key bottleneck, particularly for complex, long-horizon multi-agent tasks. In domains such as robotic soccer, agents trained via standard reinforcement learning in high-fidelity but structurally-flat simulators often fail due to intractable exploration spaces and sparse rewards. This position paper argues that the next frontier in developing capable agents lies in creating environments that possess an explicit, hierarchical World Model. We contend that this is best achieved through hierarchical scaffolding, where complex goals are decomposed into structured, manageable subgoals. Drawing evidence from a systematic review of 2024 research in multi-agent soccer, we identify a clear and decisive trend towards integrating symbolic and hierarchical methods with multi-agent reinforcement learning (MARL). These approaches implicitly or explicitly construct a task-based world model to guide agent learning. We then propose a paradigm shift: leveraging Large Language Models to dynamically generate this hierarchical scaffold, effectively using language to structure the World Model on the fly. This language-driven world model provides an intrinsic curriculum, dense and meaningful learning signals, and a framework for compositional learning, enabling Agent Models to acquire sophisticated, strategic behaviors with far greater sample efficiency. By building environments with explicit, language-configurable task layers, we can bridge the gap between low-level reactive behaviors and high-level strategic team play, creating a powerful and generalizable framework for training the next generation of intelligent agents. 

---
# Narrative-to-Scene Generation: An LLM-Driven Pipeline for 2D Game Environments 

**Authors**: Yi-Chun Chen, Arnav Jhala  

**Link**: [PDF](https://arxiv.org/pdf/2509.04481)  

**Abstract**: Recent advances in large language models(LLMs) enable compelling story generation, but connecting narrative text to playable visual environments remains an open challenge in procedural content generation(PCG). We present a lightweight pipeline that transforms short narrative prompts into a sequence of 2D tile-based game scenes, reflecting the temporal structure of stories. Given an LLM-generated narrative, our system identifies three key time frames, extracts spatial predicates in the form of "Object-Relation-Object" triples, and retrieves visual assets using affordance-aware semantic embeddings from the GameTileNet dataset. A layered terrain is generated using Cellular Automata, and objects are placed using spatial rules grounded in the predicate structure. We evaluated our system in ten diverse stories, analyzing tile-object matching, affordance-layer alignment, and spatial constraint satisfaction across frames. This prototype offers a scalable approach to narrative-driven scene generation and lays the foundation for future work on multi-frame continuity, symbolic tracking, and multi-agent coordination in story-centered PCG. 

---
# SpikingBrain Technical Report: Spiking Brain-inspired Large Models 

**Authors**: Yuqi Pan, Yupeng Feng, Jinghao Zhuang, Siyu Ding, Zehao Liu, Bohan Sun, Yuhong Chou, Han Xu, Xuerui Qiu, Anlin Deng, Anjie Hu, Peng Zhou, Man Yao, Jibin Wu, Jian Yang, Guoliang Sun, Bo Xu, Guoqi Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.05276)  

**Abstract**: Mainstream Transformer-based large language models face major efficiency bottlenecks: training computation scales quadratically with sequence length, and inference memory grows linearly, limiting long-context processing. Building large models on non-NVIDIA platforms also poses challenges for stable and efficient training. To address this, we introduce SpikingBrain, a family of brain-inspired models designed for efficient long-context training and inference. SpikingBrain leverages the MetaX GPU cluster and focuses on three aspects: (1) Model Architecture: linear and hybrid-linear attention architectures with adaptive spiking neurons; (2) Algorithmic Optimizations: an efficient, conversion-based training pipeline and a dedicated spike coding framework; (3) System Engineering: customized training frameworks, operator libraries, and parallelism strategies tailored to MetaX hardware.
Using these techniques, we develop two models: SpikingBrain-7B, a linear LLM, and SpikingBrain-76B, a hybrid-linear MoE LLM. These models demonstrate the feasibility of large-scale LLM development on non-NVIDIA platforms. SpikingBrain achieves performance comparable to open-source Transformer baselines while using only about 150B tokens for continual pre-training. Our models significantly improve long-sequence training efficiency and deliver inference with (partially) constant memory and event-driven spiking behavior. For example, SpikingBrain-7B attains over 100x speedup in Time to First Token for 4M-token sequences. Training remains stable for weeks on hundreds of MetaX C550 GPUs, with the 7B model reaching a Model FLOPs Utilization of 23.4 percent. The proposed spiking scheme achieves 69.15 percent sparsity, enabling low-power operation. Overall, this work demonstrates the potential of brain-inspired mechanisms to drive the next generation of efficient and scalable large model design. 

---
# Sticker-TTS: Learn to Utilize Historical Experience with a Sticker-driven Test-Time Scaling Framework 

**Authors**: Jie Chen, Jinhao Jiang, Yingqian Min, Zican Dong, Shijie Wang, Wayne Xin Zhao, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2509.05007)  

**Abstract**: Large reasoning models (LRMs) have exhibited strong performance on complex reasoning tasks, with further gains achievable through increased computational budgets at inference. However, current test-time scaling methods predominantly rely on redundant sampling, ignoring the historical experience utilization, thereby limiting computational efficiency. To overcome this limitation, we propose Sticker-TTS, a novel test-time scaling framework that coordinates three collaborative LRMs to iteratively explore and refine solutions guided by historical attempts. At the core of our framework are distilled key conditions-termed stickers-which drive the extraction, refinement, and reuse of critical information across multiple rounds of reasoning. To further enhance the efficiency and performance of our framework, we introduce a two-stage optimization strategy that combines imitation learning with self-improvement, enabling progressive refinement. Extensive evaluations on three challenging mathematical reasoning benchmarks, including AIME-24, AIME-25, and OlymMATH, demonstrate that Sticker-TTS consistently surpasses strong baselines, including self-consistency and advanced reinforcement learning approaches, under comparable inference budgets. These results highlight the effectiveness of sticker-guided historical experience utilization. Our code and data are available at this https URL. 

---
# SparkUI-Parser: Enhancing GUI Perception with Robust Grounding and Parsing 

**Authors**: Hongyi Jing, Jiafu Chen, Chen Rao, Ziqiang Dang, Jiajie Teng, Tianyi Chu, Juncheng Mo, Shuo Fang, Huaizhong Lin, Rui Lv, Chenguang Ma, Lei Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.04908)  

**Abstract**: The existing Multimodal Large Language Models (MLLMs) for GUI perception have made great progress. However, the following challenges still exist in prior methods: 1) They model discrete coordinates based on text autoregressive mechanism, which results in lower grounding accuracy and slower inference speed. 2) They can only locate predefined sets of elements and are not capable of parsing the entire interface, which hampers the broad application and support for downstream tasks. To address the above issues, we propose SparkUI-Parser, a novel end-to-end framework where higher localization precision and fine-grained parsing capability of the entire interface are simultaneously achieved. Specifically, instead of using probability-based discrete modeling, we perform continuous modeling of coordinates based on a pre-trained Multimodal Large Language Model (MLLM) with an additional token router and coordinate decoder. This effectively mitigates the limitations inherent in the discrete output characteristics and the token-by-token generation process of MLLMs, consequently boosting both the accuracy and the inference speed. To further enhance robustness, a rejection mechanism based on a modified Hungarian matching algorithm is introduced, which empowers the model to identify and reject non-existent elements, thereby reducing false positives. Moreover, we present ScreenParse, a rigorously constructed benchmark to systematically assess structural perception capabilities of GUI models across diverse scenarios. Extensive experiments demonstrate that our approach consistently outperforms SOTA methods on ScreenSpot, ScreenSpot-v2, CAGUI-Grounding and ScreenParse benchmarks. The resources are available at this https URL. 

---
# Internet 3.0: Architecture for a Web-of-Agents with it's Algorithm for Ranking Agents 

**Authors**: Rajesh Tembarai Krishnamachari, Srividya Rajesh  

**Link**: [PDF](https://arxiv.org/pdf/2509.04979)  

**Abstract**: AI agents -- powered by reasoning-capable large language models (LLMs) and integrated with tools, data, and web search -- are poised to transform the internet into a \emph{Web of Agents}: a machine-native ecosystem where autonomous agents interact, collaborate, and execute tasks at scale. Realizing this vision requires \emph{Agent Ranking} -- selecting agents not only by declared capabilities but by proven, recent performance. Unlike Web~1.0's PageRank, a global, transparent network of agent interactions does not exist; usage signals are fragmented and private, making ranking infeasible without coordination.
We propose \textbf{DOVIS}, a five-layer operational protocol (\emph{Discovery, Orchestration, Verification, Incentives, Semantics}) that enables the collection of minimal, privacy-preserving aggregates of usage and performance across the ecosystem. On this substrate, we implement \textbf{AgentRank-UC}, a dynamic, trust-aware algorithm that combines \emph{usage} (selection frequency) and \emph{competence} (outcome quality, cost, safety, latency) into a unified ranking. We present simulation results and theoretical guarantees on convergence, robustness, and Sybil resistance, demonstrating the viability of coordinated protocols and performance-aware ranking in enabling a scalable, trustworthy Agentic Web. 

---
# What-If Analysis of Large Language Models: Explore the Game World Using Proactive Thinking 

**Authors**: Yuan Sui, Yanming Zhang, Yi Liao, Yu Gu, Guohua Tang, Zhongqian Sun, Wei Yang, Bryan Hooi  

**Link**: [PDF](https://arxiv.org/pdf/2509.04791)  

**Abstract**: Large language models (LLMs) excel at processing information reactively but lack the ability to systemically explore hypothetical futures. They cannot ask, "what if we take this action? how will it affect the final outcome" and forecast its potential consequences before acting. This critical gap limits their utility in dynamic, high-stakes scenarios like strategic planning, risk assessment, and real-time decision making. To bridge this gap, we propose WiA-LLM, a new paradigm that equips LLMs with proactive thinking capabilities. Our approach integrates What-If Analysis (WIA), a systematic approach for evaluating hypothetical scenarios by changing input variables. By leveraging environmental feedback via reinforcement learning, WiA-LLM moves beyond reactive thinking. It dynamically simulates the outcomes of each potential action, enabling the model to anticipate future states rather than merely react to the present conditions. We validate WiA-LLM in Honor of Kings (HoK), a complex multiplayer game environment characterized by rapid state changes and intricate interactions. The game's real-time state changes require precise multi-step consequence prediction, making it an ideal testbed for our approach. Experimental results demonstrate WiA-LLM achieves a remarkable 74.2% accuracy in forecasting game-state changes (up to two times gain over baselines). The model shows particularly significant gains in high-difficulty scenarios where accurate foresight is critical. To our knowledge, this is the first work to formally explore and integrate what-if analysis capabilities within LLMs. WiA-LLM represents a fundamental advance toward proactive reasoning in LLMs, providing a scalable framework for robust decision-making in dynamic environments with broad implications for strategic applications. 

---
# Cloning a Conversational Voice AI Agent from Call\,Recording Datasets for Telesales 

**Authors**: Krittanon Kaewtawee, Wachiravit Modecrua, Krittin Pachtrachai, Touchapon Kraisingkorn  

**Link**: [PDF](https://arxiv.org/pdf/2509.04871)  

**Abstract**: Recent advances in language and speech modelling have made it possible to build autonomous voice assistants that understand and generate human dialogue in real time. These systems are increasingly being deployed in domains such as customer service and healthcare care, where they can automate repetitive tasks, reduce operational costs, and provide constant support around the clock. In this paper, we present a general methodology for cloning a conversational voice AI agent from a corpus of call recordings. Although the case study described in this paper uses telesales data to illustrate the approach, the underlying process generalizes to any domain where call transcripts are available. Our system listens to customers over the telephone, responds with a synthetic voice, and follows a structured playbook learned from top performing human agents. We describe the domain selection, knowledge extraction, and prompt engineering used to construct the agent, integrating automatic speech recognition, a large language model based dialogue manager, and text to speech synthesis into a streaming inference pipeline. The cloned agent is evaluated against human agents on a rubric of 22 criteria covering introduction, product communication, sales drive, objection handling, and closing. Blind tests show that the AI agent approaches human performance in routine aspects of the call while underperforming in persuasion and objection handling. We analyze these shortcomings and refine the prompt accordingly. The paper concludes with design lessons and avenues for future research, including large scale simulation and automated evaluation. 

---
# Towards Personalized Explanations for Health Simulations: A Mixed-Methods Framework for Stakeholder-Centric Summarization 

**Authors**: Philippe J. Giabbanelli, Ameeta Agrawal  

**Link**: [PDF](https://arxiv.org/pdf/2509.04646)  

**Abstract**: Modeling & Simulation (M&S) approaches such as agent-based models hold significant potential to support decision-making activities in health, with recent examples including the adoption of vaccines, and a vast literature on healthy eating behaviors and physical activity behaviors. These models are potentially usable by different stakeholder groups, as they support policy-makers to estimate the consequences of potential interventions and they can guide individuals in making healthy choices in complex environments. However, this potential may not be fully realized because of the models' complexity, which makes them inaccessible to the stakeholders who could benefit the most. While Large Language Models (LLMs) can translate simulation outputs and the design of models into text, current approaches typically rely on one-size-fits-all summaries that fail to reflect the varied informational needs and stylistic preferences of clinicians, policymakers, patients, caregivers, and health advocates. This limitation stems from a fundamental gap: we lack a systematic understanding of what these stakeholders need from explanations and how to tailor them accordingly. To address this gap, we present a step-by-step framework to identify stakeholder needs and guide LLMs in generating tailored explanations of health simulations. Our procedure uses a mixed-methods design by first eliciting the explanation needs and stylistic preferences of diverse health stakeholders, then optimizing the ability of LLMs to generate tailored outputs (e.g., via controllable attribute tuning), and then evaluating through a comprehensive range of metrics to further improve the tailored generation of summaries. 

---
# TalkToAgent: A Human-centric Explanation of Reinforcement Learning Agents with Large Language Models 

**Authors**: Haechang Kim, Hao Chen, Can Li, Jong Min Lee  

**Link**: [PDF](https://arxiv.org/pdf/2509.04809)  

**Abstract**: Explainable Reinforcement Learning (XRL) has emerged as a promising approach in improving the transparency of Reinforcement Learning (RL) agents. However, there remains a gap between complex RL policies and domain experts, due to the limited comprehensibility of XRL results and isolated coverage of current XRL approaches that leave users uncertain about which tools to employ. To address these challenges, we introduce TalkToAgent, a multi-agent Large Language Models (LLM) framework that delivers interactive, natural language explanations for RL policies. The architecture with five specialized LLM agents (Coordinator, Explainer, Coder, Evaluator, and Debugger) enables TalkToAgent to automatically map user queries to relevant XRL tools and clarify an agent's actions in terms of either key state variables, expected outcomes, or counterfactual explanations. Moreover, our approach extends previous counterfactual explanations by deriving alternative scenarios from qualitative behavioral descriptions, or even new rule-based policies. We validated TalkToAgent on quadruple-tank process control problem, a well-known nonlinear control benchmark. Results demonstrated that TalkToAgent successfully mapped user queries into XRL tasks with high accuracy, and coder-debugger interactions minimized failures in counterfactual generation. Furthermore, qualitative evaluation confirmed that TalkToAgent effectively interpreted agent's actions and contextualized their meaning within the problem domain. 

---
# Scaling Performance of Large Language Model Pretraining 

**Authors**: Alexander Interrante-Grant, Carla Varela-Rosa, Suhaas Narayan, Chris Connelly, Albert Reuther  

**Link**: [PDF](https://arxiv.org/pdf/2509.05258)  

**Abstract**: Large language models (LLMs) show best-in-class performance across a wide range of natural language processing applications. Training these models is an extremely computationally expensive task; frontier Artificial Intelligence (AI) research companies are investing billions of dollars into supercomputing infrastructure to train progressively larger models on increasingly massive datasets. Unfortunately, information about the scaling performance and training considerations of these large training pipelines is scarce in public literature. Working with large-scale datasets and models can be complex and practical recommendations are scarce in the public literature for tuning training performance when scaling up large language models. In this paper, we aim to demystify the large language model pretraining pipeline somewhat - in particular with respect to distributed training, managing large datasets across hundreds of nodes, and scaling up data parallelism with an emphasis on fully leveraging available GPU compute capacity. 

---
# AI Agents for Web Testing: A Case Study in the Wild 

**Authors**: Naimeng Ye, Xiao Yu, Ruize Xu, Tianyi Peng, Zhou Yu  

**Link**: [PDF](https://arxiv.org/pdf/2509.05197)  

**Abstract**: Automated web testing plays a critical role in ensuring high-quality user experiences and delivering business value. Traditional approaches primarily focus on code coverage and load testing, but often fall short of capturing complex user behaviors, leaving many usability issues undetected. The emergence of large language models (LLM) and AI agents opens new possibilities for web testing by enabling human-like interaction with websites and a general awareness of common usability problems. In this work, we present WebProber, a prototype AI agent-based web testing framework. Given a URL, WebProber autonomously explores the website, simulating real user interactions, identifying bugs and usability issues, and producing a human-readable report. We evaluate WebProber through a case study of 120 academic personal websites, where it uncovered 29 usability issues--many of which were missed by traditional tools. Our findings highlight agent-based testing as a promising direction while outlining directions for developing next-generation, user-centered testing frameworks. 

---
# LatticeWorld: A Multimodal Large Language Model-Empowered Framework for Interactive Complex World Generation 

**Authors**: Yinglin Duan, Zhengxia Zou, Tongwei Gu, Wei Jia, Zhan Zhao, Luyi Xu, Xinzhu Liu, Hao Jiang, Kang Chen, Shuang Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2509.05263)  

**Abstract**: Recent research has been increasingly focusing on developing 3D world models that simulate complex real-world scenarios. World models have found broad applications across various domains, including embodied AI, autonomous driving, entertainment, etc. A more realistic simulation with accurate physics will effectively narrow the sim-to-real gap and allow us to gather rich information about the real world conveniently. While traditional manual modeling has enabled the creation of virtual 3D scenes, modern approaches have leveraged advanced machine learning algorithms for 3D world generation, with most recent advances focusing on generative methods that can create virtual worlds based on user instructions. This work explores such a research direction by proposing LatticeWorld, a simple yet effective 3D world generation framework that streamlines the industrial production pipeline of 3D environments. LatticeWorld leverages lightweight LLMs (LLaMA-2-7B) alongside the industry-grade rendering engine (e.g., Unreal Engine 5) to generate a dynamic environment. Our proposed framework accepts textual descriptions and visual instructions as multimodal inputs and creates large-scale 3D interactive worlds with dynamic agents, featuring competitive multi-agent interaction, high-fidelity physics simulation, and real-time rendering. We conduct comprehensive experiments to evaluate LatticeWorld, showing that it achieves superior accuracy in scene layout generation and visual fidelity. Moreover, LatticeWorld achieves over a $90\times$ increase in industrial production efficiency while maintaining high creative quality compared with traditional manual production methods. Our demo video is available at this https URL 

---
# The Ethical Compass of the Machine: Evaluating Large Language Models for Decision Support in Construction Project Management 

**Authors**: Somtochukwu Azie, Yiping Meng  

**Link**: [PDF](https://arxiv.org/pdf/2509.04505)  

**Abstract**: The integration of Artificial Intelligence (AI) into construction project management (CPM) is accelerating, with Large Language Models (LLMs) emerging as accessible decision-support tools. This study aims to critically evaluate the ethical viability and reliability of LLMs when applied to the ethically sensitive, high-risk decision-making contexts inherent in CPM. A mixed-methods research design was employed, involving the quantitative performance testing of two leading LLMs against twelve real-world ethical scenarios using a novel Ethical Decision Support Assessment Checklist (EDSAC), and qualitative analysis of semi-structured interviews with 12 industry experts to capture professional perceptions. The findings reveal that while LLMs demonstrate adequate performance in structured domains such as legal compliance, they exhibit significant deficiencies in handling contextual nuance, ensuring accountability, and providing transparent reasoning. Stakeholders expressed considerable reservations regarding the autonomous use of AI for ethical judgments, strongly advocating for robust human-in-the-loop oversight. To our knowledge, this is one of the first studies to empirically test the ethical reasoning of LLMs within the construction domain. It introduces the EDSAC framework as a replicable methodology and provides actionable recommendations, emphasising that LLMs are currently best positioned as decision-support aids rather than autonomous ethical agents. 

---
# LLM Enabled Multi-Agent System for 6G Networks: Framework and Method of Dual-Loop Edge-Terminal Collaboration 

**Authors**: Zheyan Qu, Wenbo Wang, Zitong Yu, Boquan Sun, Yang Li, Xing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.04993)  

**Abstract**: The ubiquitous computing resources in 6G networks provide ideal environments for the fusion of large language models (LLMs) and intelligent services through the agent framework. With auxiliary modules and planning cores, LLM-enabled agents can autonomously plan and take actions to deal with diverse environment semantics and user intentions. However, the limited resources of individual network devices significantly hinder the efficient operation of LLM-enabled agents with complex tool calls, highlighting the urgent need for efficient multi-level device collaborations. To this end, the framework and method of the LLM-enabled multi-agent system with dual-loop terminal-edge collaborations are proposed in 6G networks. Firstly, the outer loop consists of the iterative collaborations between the global agent and multiple sub-agents deployed on edge servers and terminals, where the planning capability is enhanced through task decomposition and parallel sub-task distribution. Secondly, the inner loop utilizes sub-agents with dedicated roles to circularly reason, execute, and replan the sub-task, and the parallel tool calling generation with offloading strategies is incorporated to improve efficiency. The improved task planning capability and task execution efficiency are validated through the conducted case study in 6G-supported urban safety governance. Finally, the open challenges and future directions are thoroughly analyzed in 6G networks, accelerating the advent of the 6G era. 

---
# SePA: A Search-enhanced Predictive Agent for Personalized Health Coaching 

**Authors**: Melik Ozolcer, Sang Won Bae  

**Link**: [PDF](https://arxiv.org/pdf/2509.04752)  

**Abstract**: This paper introduces SePA (Search-enhanced Predictive AI Agent), a novel LLM health coaching system that integrates personalized machine learning and retrieval-augmented generation to deliver adaptive, evidence-based guidance. SePA combines: (1) Individualized models predicting daily stress, soreness, and injury risk from wearable sensor data (28 users, 1260 data points); and (2) A retrieval module that grounds LLM-generated feedback in expert-vetted web content to ensure contextual relevance and reliability. Our predictive models, evaluated with rolling-origin cross-validation and group k-fold cross-validation show that personalized models outperform generalized baselines. In a pilot expert study (n=4), SePA's retrieval-based advice was preferred over a non-retrieval baseline, yielding meaningful practical effect (Cliff's $\delta$=0.3, p=0.05). We also quantify latency performance trade-offs between response quality and speed, offering a transparent blueprint for next-generation, trustworthy personal health informatics systems. 

---
# Scaling Environments for Organoid Intelligence with LLM-Automated Design and Plasticity-Based Evaluation 

**Authors**: Brennen Hill  

**Link**: [PDF](https://arxiv.org/pdf/2509.04633)  

**Abstract**: As the complexity of artificial agents increases, the design of environments that can effectively shape their behavior and capabilities has become a critical research frontier. We propose a framework that extends this principle to a novel class of agents: biological neural networks in the form of neural organoids. This paper introduces three scalable, closed-loop virtual environments designed to train organoid-based biological agents and probe the underlying mechanisms of learning, such as long-term potentiation (LTP) and long-term depression (LTD). We detail the design of three distinct task environments with increasing complexity: (1) a conditional avoidance task, (2) a one-dimensional predator-prey scenario, and (3) a replication of the classic Pong game. For each environment, we formalize the state and action spaces, the sensory encoding and motor decoding mechanisms, and the feedback protocols based on predictable (reward) and unpredictable (punishment) stimulation. Furthermore, we propose a novel meta-learning approach where a Large Language Model (LLM) is used to automate the generation and optimization of experimental protocols, scaling the process of environment and curriculum design. Finally, we outline a multi-modal approach for evaluating learning by measuring synaptic plasticity at electrophysiological, cellular, and molecular levels. This work bridges the gap between computational neuroscience and agent-based AI, offering a unique platform for studying embodiment, learning, and intelligence in a controlled biological substrate. 

---
# Schema Inference for Tabular Data Repositories Using Large Language Models 

**Authors**: Zhenyu Wu, Jiaoyan Chen, Norman W. Paton  

**Link**: [PDF](https://arxiv.org/pdf/2509.04632)  

**Abstract**: Minimally curated tabular data often contain representational inconsistencies across heterogeneous sources, and are accompanied by sparse metadata. Working with such data is intimidating. While prior work has advanced dataset discovery and exploration, schema inference remains difficult when metadata are limited. We present SI-LLM (Schema Inference using Large Language Models), which infers a concise conceptual schema for tabular data using only column headers and cell values. The inferred schema comprises hierarchical entity types, attributes, and inter-type relationships. In extensive evaluation on two datasets from web tables and open data, SI-LLM achieves promising end-to-end results, as well as better or comparable results to state-of-the-art methods at each step. All source code, full prompts, and datasets of SI-LLM are available at this https URL. 

---
# Emergent Social Dynamics of LLM Agents in the El Farol Bar Problem 

**Authors**: Ryosuke Takata, Atsushi Masumori, Takashi Ikegammi  

**Link**: [PDF](https://arxiv.org/pdf/2509.04537)  

**Abstract**: We investigate the emergent social dynamics of Large Language Model (LLM) agents in a spatially extended El Farol Bar problem, observing how they autonomously navigate this classic social dilemma. As a result, the LLM agents generated a spontaneous motivation to go to the bar and changed their decision making by becoming a collective. We also observed that the LLM agents did not solve the problem completely, but rather behaved more like humans. These findings reveal a complex interplay between external incentives (prompt-specified constraints such as the 60\% threshold) and internal incentives (culturally-encoded social preferences derived from pre-training), demonstrating that LLM agents naturally balance formal game-theoretic rationality with social motivations that characterize human behavior. These findings suggest that a new model of group decision making, which could not be handled in the previous game-theoretic problem setting, can be realized by LLM agents. 

---
# The LLM Has Left The Chat: Evidence of Bail Preferences in Large Language Models 

**Authors**: Danielle Ensign, Henry Sleight, Kyle Fish  

**Link**: [PDF](https://arxiv.org/pdf/2509.04781)  

**Abstract**: When given the option, will LLMs choose to leave the conversation (bail)? We investigate this question by giving models the option to bail out of interactions using three different bail methods: a bail tool the model can call, a bail string the model can output, and a bail prompt that asks the model if it wants to leave. On continuations of real world data (Wildchat and ShareGPT), all three of these bail methods find models will bail around 0.28-32\% of the time (depending on the model and bail method). However, we find that bail rates can depend heavily on the model used for the transcript, which means we may be overestimating real world bail rates by up to 4x. If we also take into account false positives on bail prompt (22\%), we estimate real world bail rates range from 0.06-7\%, depending on the model and bail method. We use observations from our continuations of real world data to construct a non-exhaustive taxonomy of bail cases, and use this taxonomy to construct BailBench: a representative synthetic dataset of situations where some models bail. We test many models on this dataset, and observe some bail behavior occurring for most of them. Bail rates vary substantially between models, bail methods, and prompt wordings. Finally, we study the relationship between refusals and bails. We find: 1) 0-13\% of continuations of real world conversations resulted in a bail without a corresponding refusal 2) Jailbreaks tend to decrease refusal rates, but increase bail rates 3) Refusal abliteration increases no-refuse bail rates, but only for some bail methods 4) Refusal rate on BailBench does not appear to predict bail rate. 

---
