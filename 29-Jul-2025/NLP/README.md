# Multi-Agent-as-Judge: Aligning LLM-Agent-Based Automated Evaluation with Multi-Dimensional Human Evaluation 

**Authors**: Jiaju Chen, Yuxuan Lu, Xiaojie Wang, Huimin Zeng, Jing Huang, Jiri Gesi, Ying Xu, Bingsheng Yao, Dakuo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.21028)  

**Abstract**: Nearly all human work is collaborative; thus, the evaluation of real-world NLP applications often requires multiple dimensions that align with diverse human perspectives. As real human evaluator resources are often scarce and costly, the emerging "LLM-as-a-judge" paradigm sheds light on a promising approach to leverage LLM agents to believably simulate human evaluators. Yet, to date, existing LLM-as-a-judge approaches face two limitations: persona descriptions of agents are often arbitrarily designed, and the frameworks are not generalizable to other tasks. To address these challenges, we propose MAJ-EVAL, a Multi-Agent-as-Judge evaluation framework that can automatically construct multiple evaluator personas with distinct dimensions from relevant text documents (e.g., research papers), instantiate LLM agents with the personas, and engage in-group debates with multi-agents to Generate multi-dimensional feedback. Our evaluation experiments in both the educational and medical domains demonstrate that MAJ-EVAL can generate evaluation results that better align with human experts' ratings compared with conventional automated evaluation metrics and existing LLM-as-a-judge methods. 

---
# Memorization in Fine-Tuned Large Language Models 

**Authors**: Danil Savine, Muni Sreenivas Pydi, Jamal Atif, Olivier Cappé  

**Link**: [PDF](https://arxiv.org/pdf/2507.21009)  

**Abstract**: This study investigates the mechanisms and factors influencing memorization in fine-tuned large language models (LLMs), with a focus on the medical domain due to its privacy-sensitive nature. We examine how different aspects of the fine-tuning process affect a model's propensity to memorize training data, using the PHEE dataset of pharmacovigilance events.
Our research employs two main approaches: a membership inference attack to detect memorized data, and a generation task with prompted prefixes to assess verbatim reproduction. We analyze the impact of adapting different weight matrices in the transformer architecture, the relationship between perplexity and memorization, and the effect of increasing the rank in low-rank adaptation (LoRA) fine-tuning.
Key findings include: (1) Value and Output matrices contribute more significantly to memorization compared to Query and Key matrices; (2) Lower perplexity in the fine-tuned model correlates with increased memorization; (3) Higher LoRA ranks lead to increased memorization, but with diminishing returns at higher ranks.
These results provide insights into the trade-offs between model performance and privacy risks in fine-tuned LLMs. Our findings have implications for developing more effective and responsible strategies for adapting large language models while managing data privacy concerns. 

---
# Mind the Gap: Conformative Decoding to Improve Output Diversity of Instruction-Tuned Large Language Models 

**Authors**: Max Peeperkorn, Tom Kouwenhoven, Dan Brown, Anna Jordanous  

**Link**: [PDF](https://arxiv.org/pdf/2507.20956)  

**Abstract**: Instruction-tuning large language models (LLMs) reduces the diversity of their outputs, which has implications for many tasks, particularly for creative tasks. This paper investigates the ``diversity gap'' for a writing prompt narrative generation task. This gap emerges as measured by current diversity metrics for various open-weight and open-source LLMs. The results show significant decreases in diversity due to instruction-tuning. We explore the diversity loss at each fine-tuning stage for the OLMo and OLMo 2 models to further understand how output diversity is affected. The results indicate that DPO has the most substantial impact on diversity. Motivated by these findings, we present a new decoding strategy, conformative decoding, which guides an instruct model using its more diverse base model to reintroduce output diversity. We show that conformative decoding typically increases diversity and even maintains or improves quality. 

---
# FRED: Financial Retrieval-Enhanced Detection and Editing of Hallucinations in Language Models 

**Authors**: Likun Tan, Kuan-Wei Huang, Kevin Wu  

**Link**: [PDF](https://arxiv.org/pdf/2507.20930)  

**Abstract**: Hallucinations in large language models pose a critical challenge for applications requiring factual reliability, particularly in high-stakes domains such as finance. This work presents an effective approach for detecting and editing factually incorrect content in model-generated responses based on the provided context. Given a user-defined domain-specific error taxonomy, we construct a synthetic dataset by inserting tagged errors into financial question-answering corpora and then fine-tune four language models, Phi-4, Phi-4-mini, Qwen3-4B, and Qwen3-14B, to detect and edit these factual inaccuracies. Our best-performing model, fine-tuned Phi-4, achieves an 8% improvement in binary F1 score and a 30% gain in overall detection performance compared to OpenAI-o3. Notably, our fine-tuned Phi-4-mini model, despite having only 4 billion parameters, maintains competitive performance with just a 2% drop in binary detection and a 0.1% decline in overall detection compared to OpenAI-o3. Our work provides a practical solution for detecting and editing factual inconsistencies in financial text generation while introducing a generalizable framework that can enhance the trustworthiness and alignment of large language models across diverse applications beyond finance. Our code and data are available at this https URL. 

---
# FHSTP@EXIST 2025 Benchmark: Sexism Detection with Transparent Speech Concept Bottleneck Models 

**Authors**: Roberto Labadie-Tamayo, Adrian Jaques Böck, Djordje Slijepčević, Xihui Chen, Andreas Babic, Matthias Zeppelzauer  

**Link**: [PDF](https://arxiv.org/pdf/2507.20924)  

**Abstract**: Sexism has become widespread on social media and in online conversation. To help address this issue, the fifth Sexism Identification in Social Networks (EXIST) challenge is initiated at CLEF 2025. Among this year's international benchmarks, we concentrate on solving the first task aiming to identify and classify sexism in social media textual posts. In this paper, we describe our solutions and report results for three subtasks: Subtask 1.1 - Sexism Identification in Tweets, Subtask 1.2 - Source Intention in Tweets, and Subtask 1.3 - Sexism Categorization in Tweets. We implement three models to address each subtask which constitute three individual runs: Speech Concept Bottleneck Model (SCBM), Speech Concept Bottleneck Model with Transformer (SCBMT), and a fine-tuned XLM-RoBERTa transformer model. SCBM uses descriptive adjectives as human-interpretable bottleneck concepts. SCBM leverages large language models (LLMs) to encode input texts into a human-interpretable representation of adjectives, then used to train a lightweight classifier for downstream tasks. SCBMT extends SCBM by fusing adjective-based representation with contextual embeddings from transformers to balance interpretability and classification performance. Beyond competitive results, these two models offer fine-grained explanations at both instance (local) and class (global) levels. We also investigate how additional metadata, e.g., annotators' demographic profiles, can be leveraged. For Subtask 1.1, XLM-RoBERTa, fine-tuned on provided data augmented with prior datasets, ranks 6th for English and Spanish and 4th for English in the Soft-Soft evaluation. Our SCBMT achieves 7th for English and Spanish and 6th for Spanish. 

---
# MediQAl: A French Medical Question Answering Dataset for Knowledge and Reasoning Evaluation 

**Authors**: Adrien Bazoge  

**Link**: [PDF](https://arxiv.org/pdf/2507.20917)  

**Abstract**: This work introduces MediQAl, a French medical question answering dataset designed to evaluate the capabilities of language models in factual medical recall and reasoning over real-world clinical scenarios. MediQAl contains 32,603 questions sourced from French medical examinations across 41 medical subjects. The dataset includes three tasks: (i) Multiple-Choice Question with Unique answer, (ii) Multiple-Choice Question with Multiple answer, and (iii) Open-Ended Question with Short-Answer. Each question is labeled as Understanding or Reasoning, enabling a detailed analysis of models' cognitive capabilities. We validate the MediQAl dataset through extensive evaluation with 14 large language models, including recent reasoning-augmented models, and observe a significant performance gap between factual recall and reasoning tasks. Our evaluation provides a comprehensive benchmark for assessing language models' performance on French medical question answering, addressing a crucial gap in multilingual resources for the medical domain. 

---
# Soft Injection of Task Embeddings Outperforms Prompt-Based In-Context Learning 

**Authors**: Jungwon Park, Wonjong Rhee  

**Link**: [PDF](https://arxiv.org/pdf/2507.20906)  

**Abstract**: In-Context Learning (ICL) enables Large Language Models (LLMs) to perform tasks by conditioning on input-output examples in the prompt, without requiring any update in model parameters. While widely adopted, it remains unclear whether prompting with multiple examples is the most effective and efficient way to convey task information. In this work, we propose Soft Injection of task embeddings. The task embeddings are constructed only once using few-shot ICL prompts and repeatedly used during inference. Soft injection is performed by softly mixing task embeddings with attention head activations using pre-optimized mixing parameters, referred to as soft head-selection parameters. This method not only allows a desired task to be performed without in-prompt demonstrations but also significantly outperforms existing ICL approaches while reducing memory usage and compute cost at inference time. An extensive evaluation is performed across 57 tasks and 12 LLMs, spanning four model families of sizes from 4B to 70B. Averaged across 57 tasks, our method outperforms 10-shot ICL by 10.1%-13.9% across 12 LLMs. Additional analyses show that our method also serves as an insightful tool for analyzing task-relevant roles of attention heads, revealing that task-relevant head positions selected by our method transfer across similar tasks but not across dissimilar ones -- underscoring the task-specific nature of head functionality. Our soft injection method opens a new paradigm for reducing prompt length and improving task performance by shifting task conditioning from the prompt space to the activation space. 

---
# Leveraging Open-Source Large Language Models for Clinical Information Extraction in Resource-Constrained Settings 

**Authors**: Luc Builtjes, Joeran Bosma, Mathias Prokop, Bram van Ginneken, Alessa Hering  

**Link**: [PDF](https://arxiv.org/pdf/2507.20859)  

**Abstract**: Medical reports contain rich clinical information but are often unstructured and written in domain-specific language, posing challenges for information extraction. While proprietary large language models (LLMs) have shown promise in clinical natural language processing, their lack of transparency and data privacy concerns limit their utility in healthcare. This study therefore evaluates nine open-source generative LLMs on the DRAGON benchmark, which includes 28 clinical information extraction tasks in Dutch. We developed \texttt{llm\_extractinator}, a publicly available framework for information extraction using open-source generative LLMs, and used it to assess model performance in a zero-shot setting. Several 14 billion parameter models, Phi-4-14B, Qwen-2.5-14B, and DeepSeek-R1-14B, achieved competitive results, while the bigger Llama-3.3-70B model achieved slightly higher performance at greater computational cost. Translation to English prior to inference consistently degraded performance, highlighting the need of native-language processing. These findings demonstrate that open-source LLMs, when used with our framework, offer effective, scalable, and privacy-conscious solutions for clinical information extraction in low-resource settings. 

---
# A survey of diversity quantification in natural language processing: The why, what, where and how 

**Authors**: Louis Estève, Marie-Catherine de Marneffe, Nurit Melnik, Agata Savary, Olha Kanishcheva  

**Link**: [PDF](https://arxiv.org/pdf/2507.20858)  

**Abstract**: The concept of diversity has received increased consideration in Natural Language Processing (NLP) in recent years. This is due to various motivations like promoting and inclusion, approximating human linguistic behavior, and increasing systems' performance. Diversity has however often been addressed in an ad hoc manner in NLP, and with few explicit links to other domains where this notion is better theorized. We survey articles in the ACL Anthology from the past 6 years, with "diversity" or "diverse" in their title. We find a wide range of settings in which diversity is quantified, often highly specialized and using inconsistent terminology. We put forward a unified taxonomy of why, what on, where, and how diversity is measured in NLP. Diversity measures are cast upon a unified framework from ecology and economy (Stirling, 2007) with 3 dimensions of diversity: variety, balance and disparity. We discuss the trends which emerge due to this systematized approach. We believe that this study paves the way towards a better formalization of diversity in NLP, which should bring a better understanding of this notion and a better comparability between various approaches. 

---
# Latent Inter-User Difference Modeling for LLM Personalization 

**Authors**: Yilun Qiu, Tianhao Shi, Xiaoyan Zhao, Fengbin Zhu, Yang Zhang, Fuli Feng  

**Link**: [PDF](https://arxiv.org/pdf/2507.20849)  

**Abstract**: Large language models (LLMs) are increasingly integrated into users' daily lives, leading to a growing demand for personalized outputs. Previous work focuses on leveraging a user's own history, overlooking inter-user differences that are crucial for effective personalization. While recent work has attempted to model such differences, the reliance on language-based prompts often hampers the effective extraction of meaningful distinctions. To address these issues, we propose Difference-aware Embedding-based Personalization (DEP), a framework that models inter-user differences in the latent space instead of relying on language prompts. DEP constructs soft prompts by contrasting a user's embedding with those of peers who engaged with similar content, highlighting relative behavioral signals. A sparse autoencoder then filters and compresses both user-specific and difference-aware embeddings, preserving only task-relevant features before injecting them into a frozen LLM. Experiments on personalized review generation show that DEP consistently outperforms baseline methods across multiple metrics. Our code is available at this https URL. 

---
# Automating Thematic Review of Prevention of Future Deaths Reports: Replicating the ONS Child Suicide Study using Large Language Models 

**Authors**: Sam Osian, Arpan Dutta, Sahil Bhandari, Iain E. Buchan, Dan W. Joyce  

**Link**: [PDF](https://arxiv.org/pdf/2507.20786)  

**Abstract**: Prevention of Future Deaths (PFD) reports, issued by coroners in England and Wales, flag systemic hazards that may lead to further loss of life. Analysis of these reports has previously been constrained by the manual effort required to identify and code relevant cases. In 2025, the Office for National Statistics (ONS) published a national thematic review of child-suicide PFD reports ($\leq$ 18 years), identifying 37 cases from January 2015 to November 2023 - a process based entirely on manual curation and coding. We evaluated whether a fully automated, open source "text-to-table" language-model pipeline (PFD Toolkit) could reproduce the ONS's identification and thematic analysis of child-suicide PFD reports, and assessed gains in efficiency and reliability. All 4,249 PFD reports published from July 2013 to November 2023 were processed via PFD Toolkit's large language model pipelines. Automated screening identified cases where the coroner attributed death to suicide in individuals aged 18 or younger, and eligible reports were coded for recipient category and 23 concern sub-themes, replicating the ONS coding frame. PFD Toolkit identified 72 child-suicide PFD reports - almost twice the ONS count. Three blinded clinicians adjudicated a stratified sample of 144 reports to validate the child-suicide screening. Against the post-consensus clinical annotations, the LLM-based workflow showed substantial to almost-perfect agreement (Cohen's $\kappa$ = 0.82, 95% CI: 0.66-0.98, raw agreement = 91%). The end-to-end script runtime was 8m 16s, transforming a process that previously took months into one that can be completed in minutes. This demonstrates that automated LLM analysis can reliably and efficiently replicate manual thematic reviews of coronial data, enabling scalable, reproducible, and timely insights for public health and safety. The PFD Toolkit is openly available for future research. 

---
# On The Role of Pretrained Language Models in General-Purpose Text Embeddings: A Survey 

**Authors**: Meishan Zhang, Xin Zhang, Xinping Zhao, Shouzheng Huang, Baotian Hu, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.20783)  

**Abstract**: Text embeddings have attracted growing interest due to their effectiveness across a wide range of natural language processing (NLP) tasks, such as retrieval, classification, clustering, bitext mining, and summarization. With the emergence of pretrained language models (PLMs), general-purpose text embeddings (GPTE) have gained significant traction for their ability to produce rich, transferable representations. The general architecture of GPTE typically leverages PLMs to derive dense text representations, which are then optimized through contrastive learning on large-scale pairwise datasets. In this survey, we provide a comprehensive overview of GPTE in the era of PLMs, focusing on the roles PLMs play in driving its development. We first examine the fundamental architecture and describe the basic roles of PLMs in GPTE, i.e., embedding extraction, expressivity enhancement, training strategies, learning objectives, and data construction. Then, we describe advanced roles enabled by PLMs, such as multilingual support, multimodal integration, code understanding, and scenario-specific adaptation. Finally, we highlight potential future research directions that move beyond traditional improvement goals, including ranking integration, safety considerations, bias mitigation, structural information incorporation, and the cognitive extension of embeddings. This survey aims to serve as a valuable reference for both newcomers and established researchers seeking to understand the current state and future potential of GPTE. 

---
# Multilingual Self-Taught Faithfulness Evaluators 

**Authors**: Carlo Alfano, Aymen Al Marjani, Zeno Jonke, Amin Mantrach, Saab Mansour, Marcello Federico  

**Link**: [PDF](https://arxiv.org/pdf/2507.20752)  

**Abstract**: The growing use of large language models (LLMs) has increased the need for automatic evaluation systems, particularly to address the challenge of information hallucination. Although existing faithfulness evaluation approaches have shown promise, they are predominantly English-focused and often require expensive human-labeled training data for fine-tuning specialized models. As LLMs see increased adoption in multilingual contexts, there is a need for accurate faithfulness evaluators that can operate across languages without extensive labeled data. This paper presents Self-Taught Evaluators for Multilingual Faithfulness, a framework that learns exclusively from synthetic multilingual summarization data while leveraging cross-lingual transfer learning. Through experiments comparing language-specific and mixed-language fine-tuning approaches, we demonstrate a consistent relationship between an LLM's general language capabilities and its performance in language-specific evaluation tasks. Our framework shows improvements over existing baselines, including state-of-the-art English evaluators and machine translation-based approaches. 

---
# Investigating Structural Pruning and Recovery Techniques for Compressing Multimodal Large Language Models: An Empirical Study 

**Authors**: Yiran Huang, Lukas Thede, Massimiliano Mancini, Wenjia Xu, Zeynep Akata  

**Link**: [PDF](https://arxiv.org/pdf/2507.20749)  

**Abstract**: While Multimodal Large Language Models (MLLMs) demonstrate impressive capabilities, their substantial computational and memory requirements pose significant barriers to practical deployment. Current parameter reduction techniques primarily involve training MLLMs from Small Language Models (SLMs), but these methods offer limited flexibility and remain computationally intensive. To address this gap, we propose to directly compress existing MLLMs through structural pruning combined with efficient recovery training. Specifically, we investigate two structural pruning paradigms--layerwise and widthwise pruning--applied to the language model backbone of MLLMs, alongside supervised finetuning and knowledge distillation. Additionally, we assess the feasibility of conducting recovery training with only a small fraction of the available data. Our results show that widthwise pruning generally maintains better performance in low-resource scenarios with limited computational resources or insufficient finetuning data. As for the recovery training, finetuning only the multimodal projector is sufficient at small compression levels (< 20%). Furthermore, a combination of supervised finetuning and hidden-state distillation yields optimal recovery across various pruning levels. Notably, effective recovery can be achieved with as little as 5% of the original training data, while retaining over 95% of the original performance. Through empirical study on two representative MLLMs, i.e., LLaVA-v1.5-7B and Bunny-v1.0-3B, this study offers actionable insights for practitioners aiming to compress MLLMs effectively without extensive computation resources or sufficient data. 

---
# Text2VLM: Adapting Text-Only Datasets to Evaluate Alignment Training in Visual Language Models 

**Authors**: Gabriel Downer, Sean Craven, Damian Ruck, Jake Thomas  

**Link**: [PDF](https://arxiv.org/pdf/2507.20704)  

**Abstract**: The increasing integration of Visual Language Models (VLMs) into AI systems necessitates robust model alignment, especially when handling multimodal content that combines text and images. Existing evaluation datasets heavily lean towards text-only prompts, leaving visual vulnerabilities under evaluated. To address this gap, we propose \textbf{Text2VLM}, a novel multi-stage pipeline that adapts text-only datasets into multimodal formats, specifically designed to evaluate the resilience of VLMs against typographic prompt injection attacks. The Text2VLM pipeline identifies harmful content in the original text and converts it into a typographic image, creating a multimodal prompt for VLMs. Also, our evaluation of open-source VLMs highlights their increased susceptibility to prompt injection when visual inputs are introduced, revealing critical weaknesses in the current models' alignment. This is in addition to a significant performance gap compared to closed-source frontier models. We validate Text2VLM through human evaluations, ensuring the alignment of extracted salient concepts; text summarization and output classification align with human expectations. Text2VLM provides a scalable tool for comprehensive safety assessment, contributing to the development of more robust safety mechanisms for VLMs. By enhancing the evaluation of multimodal vulnerabilities, Text2VLM plays a role in advancing the safe deployment of VLMs in diverse, real-world applications. 

---
# When Scale Meets Diversity: Evaluating Language Models on Fine-Grained Multilingual Claim Verification 

**Authors**: Hanna Shcharbakova, Tatiana Anikina, Natalia Skachkova, Josef van Genabith  

**Link**: [PDF](https://arxiv.org/pdf/2507.20700)  

**Abstract**: The rapid spread of multilingual misinformation requires robust automated fact verification systems capable of handling fine-grained veracity assessments across diverse languages. While large language models have shown remarkable capabilities across many NLP tasks, their effectiveness for multilingual claim verification with nuanced classification schemes remains understudied. We conduct a comprehensive evaluation of five state-of-the-art language models on the X-Fact dataset, which spans 25 languages with seven distinct veracity categories. Our experiments compare small language models (encoder-based XLM-R and mT5) with recent decoder-only LLMs (Llama 3.1, Qwen 2.5, Mistral Nemo) using both prompting and fine-tuning approaches. Surprisingly, we find that XLM-R (270M parameters) substantially outperforms all tested LLMs (7-12B parameters), achieving 57.7% macro-F1 compared to the best LLM performance of 16.9%. This represents a 15.8% improvement over the previous state-of-the-art (41.9%), establishing new performance benchmarks for multilingual fact verification. Our analysis reveals problematic patterns in LLM behavior, including systematic difficulties in leveraging evidence and pronounced biases toward frequent categories in imbalanced data settings. These findings suggest that for fine-grained multilingual fact verification, smaller specialized models may be more effective than general-purpose large models, with important implications for practical deployment of fact-checking systems. 

---
# Geometric-Mean Policy Optimization 

**Authors**: Yuzhong Zhao, Yue Liu, Junpeng Liu, Jingye Chen, Xun Wu, Yaru Hao, Tengchao Lv, Shaohan Huang, Lei Cui, Qixiang Ye, Fang Wan, Furu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2507.20673)  

**Abstract**: Recent advancements, such as Group Relative Policy Optimization (GRPO), have enhanced the reasoning capabilities of large language models by optimizing the arithmetic mean of token-level rewards. However, GRPO suffers from unstable policy updates when processing tokens with outlier importance-weighted rewards, which manifests as extreme importance sampling ratios during training, i.e., the ratio between the sampling probabilities assigned to a token by the current and old policies. In this work, we propose Geometric-Mean Policy Optimization (GMPO), a stabilized variant of GRPO. Instead of optimizing the arithmetic mean, GMPO maximizes the geometric mean of token-level rewards, which is inherently less sensitive to outliers and maintains a more stable range of importance sampling ratio. In addition, we provide comprehensive theoretical and experimental analysis to justify the design and stability benefits of GMPO. Beyond improved stability, GMPO-7B outperforms GRPO by an average of 4.1% on multiple mathematical benchmarks and 1.4% on multimodal reasoning benchmark, including AIME24, AMC, MATH500, OlympiadBench, Minerva, and Geometry3K. Code is available at this https URL. 

---
# Ontology-Enhanced Knowledge Graph Completion using Large Language Models 

**Authors**: Wenbin Guo, Xin Wang, Jiaoyan Chen, Zhao Li, Zirui Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.20643)  

**Abstract**: Large Language Models (LLMs) have been extensively adopted in Knowledge Graph Completion (KGC), showcasing significant research advancements. However, as black-box models driven by deep neural architectures, current LLM-based KGC methods rely on implicit knowledge representation with parallel propagation of erroneous knowledge, thereby hindering their ability to produce conclusive and decisive reasoning outcomes. We aim to integrate neural-perceptual structural information with ontological knowledge, leveraging the powerful capabilities of LLMs to achieve a deeper understanding of the intrinsic logic of the knowledge. We propose an ontology enhanced KGC method using LLMs -- OL-KGC. It first leverages neural perceptual mechanisms to effectively embed structural information into the textual space, and then uses an automated extraction algorithm to retrieve ontological knowledge from the knowledge graphs (KGs) that needs to be completed, which is further transformed into a textual format comprehensible to LLMs for providing logic guidance. We conducted extensive experiments on three widely-used benchmarks -- FB15K-237, UMLS and WN18RR. The experimental results demonstrate that OL-KGC significantly outperforms existing mainstream KGC methods across multiple evaluation metrics, achieving state-of-the-art performance. 

---
# Before the Outrage: Challenges and Advances in Predicting Online Antisocial Behavior 

**Authors**: Anaïs Ollagnier  

**Link**: [PDF](https://arxiv.org/pdf/2507.20614)  

**Abstract**: Antisocial behavior (ASB) on social media-including hate speech, harassment, and trolling-poses growing challenges for platform safety and societal wellbeing. While prior work has primarily focused on detecting harmful content after it appears, predictive approaches aim to forecast future harmful behaviors-such as hate speech propagation, conversation derailment, or user recidivism-before they fully unfold. Despite increasing interest, the field remains fragmented, lacking a unified taxonomy or clear synthesis of existing methods. This paper presents a systematic review of over 49 studies on ASB prediction, offering a structured taxonomy of five core task types: early harm detection, harm emergence prediction, harm propagation prediction, behavioral risk prediction, and proactive moderation support. We analyze how these tasks differ by temporal framing, prediction granularity, and operational goals. In addition, we examine trends in modeling techniques-from classical machine learning to pre-trained language models-and assess the influence of dataset characteristics on task feasibility and generalization. Our review highlights methodological challenges, such as dataset scarcity, temporal drift, and limited benchmarks, while outlining emerging research directions including multilingual modeling, cross-platform generalization, and human-in-the-loop systems. By organizing the field around a coherent framework, this survey aims to guide future work toward more robust and socially responsible ASB prediction. 

---
# ZSE-Cap: A Zero-Shot Ensemble for Image Retrieval and Prompt-Guided Captioning 

**Authors**: Duc-Tai Dinh, Duc Anh Khoa Dinh  

**Link**: [PDF](https://arxiv.org/pdf/2507.20564)  

**Abstract**: We present ZSE-Cap (Zero-Shot Ensemble for Captioning), our 4th place system in Event-Enriched Image Analysis (EVENTA) shared task on article-grounded image retrieval and captioning. Our zero-shot approach requires no finetuning on the competition's data. For retrieval, we ensemble similarity scores from CLIP, SigLIP, and DINOv2. For captioning, we leverage a carefully engineered prompt to guide the Gemma 3 model, enabling it to link high-level events from the article to the visual content in the image. Our system achieved a final score of 0.42002, securing a top-4 position on the private test set, demonstrating the effectiveness of combining foundation models through ensembling and prompting. Our code is available at this https URL. 

---
# Enhancing Hallucination Detection via Future Context 

**Authors**: Joosung Lee, Cheonbok Park, Hwiyeol Jo, Jeonghoon Kim, Joonsuk Park, Kang Min Yoo  

**Link**: [PDF](https://arxiv.org/pdf/2507.20546)  

**Abstract**: Large Language Models (LLMs) are widely used to generate plausible text on online platforms, without revealing the generation process. As users increasingly encounter such black-box outputs, detecting hallucinations has become a critical challenge. To address this challenge, we focus on developing a hallucination detection framework for black-box generators. Motivated by the observation that hallucinations, once introduced, tend to persist, we sample future contexts. The sampled future contexts provide valuable clues for hallucination detection and can be effectively integrated with various sampling-based methods. We extensively demonstrate performance improvements across multiple methods using our proposed sampling approach. 

---
# Dialogues of Dissent: Thematic and Rhetorical Dimensions of Hate and Counter-Hate Speech in Social Media Conversations 

**Authors**: Effi Levi, Gal Ron, Odelia Oshri, Shaul R. Shenhav  

**Link**: [PDF](https://arxiv.org/pdf/2507.20528)  

**Abstract**: We introduce a novel multi-labeled scheme for joint annotation of hate and counter-hate speech in social media conversations, categorizing hate and counter-hate messages into thematic and rhetorical dimensions. The thematic categories outline different discursive aspects of each type of speech, while the rhetorical dimension captures how hate and counter messages are communicated, drawing on Aristotle's Logos, Ethos and Pathos. We annotate a sample of 92 conversations, consisting of 720 tweets, and conduct statistical analyses, incorporating public metrics, to explore patterns of interaction between the thematic and rhetorical dimensions within and between hate and counter-hate speech. Our findings provide insights into the spread of hate messages on social media, the strategies used to counter them, and their potential impact on online behavior. 

---
# SAND-Math: Using LLMs to Generate Novel, Difficult and Useful Mathematics Questions and Answers 

**Authors**: Chaitanya Manem, Pratik Prabhanjan Brahma, Prakamya Mishra, Zicheng Liu, Emad Barsoum  

**Link**: [PDF](https://arxiv.org/pdf/2507.20527)  

**Abstract**: The demand for Large Language Models (LLMs) capable of sophisticated mathematical reasoning is growing across industries. However, the development of performant mathematical LLMs is critically bottlenecked by the scarcity of difficult, novel training data. We introduce \textbf{SAND-Math} (Synthetic Augmented Novel and Difficult Mathematics problems and solutions), a pipeline that addresses this by first generating high-quality problems from scratch and then systematically elevating their complexity via a new \textbf{Difficulty Hiking} step. We demonstrate the effectiveness of our approach through two key findings. First, augmenting a strong baseline with SAND-Math data significantly boosts performance, outperforming the next-best synthetic dataset by \textbf{$\uparrow$ 17.85 absolute points} on the AIME25 benchmark. Second, in a dedicated ablation study, we show our Difficulty Hiking process is highly effective: by increasing average problem difficulty from 5.02 to 5.98, this step lifts AIME25 performance from 46.38\% to 49.23\%. The full generation pipeline, final dataset, and a fine-tuned model form a practical and scalable toolkit for building more capable and efficient mathematical reasoning LLMs. SAND-Math dataset is released here: \href{this https URL}{this https URL} 

---
# AQUA: A Large Language Model for Aquaculture & Fisheries 

**Authors**: Praneeth Narisetty, Uday Kumar Reddy Kattamanchi, Lohit Akshant Nimma, Sri Ram Kaushik Karnati, Shiva Nagendra Babu Kore, Mounika Golamari, Tejashree Nageshreddy  

**Link**: [PDF](https://arxiv.org/pdf/2507.20520)  

**Abstract**: Aquaculture plays a vital role in global food security and coastal economies by providing sustainable protein sources. As the industry expands to meet rising demand, it faces growing challenges such as disease outbreaks, inefficient feeding practices, rising labor costs, logistical inefficiencies, and critical hatchery issues, including high mortality rates and poor water quality control. Although artificial intelligence has made significant progress, existing machine learning methods fall short of addressing the domain-specific complexities of aquaculture. To bridge this gap, we introduce AQUA, the first large language model (LLM) tailored for aquaculture, designed to support farmers, researchers, and industry practitioners. Central to this effort is AQUADAPT (Data Acquisition, Processing and Tuning), an Agentic Framework for generating and refining high-quality synthetic data using a combination of expert knowledge, largescale language models, and automated evaluation techniques. Our work lays the foundation for LLM-driven innovations in aquaculture research, advisory systems, and decision-making tools. 

---
# Speaking in Words, Thinking in Logic: A Dual-Process Framework in QA Systems 

**Authors**: Tuan Bui, Trong Le, Phat Thai, Sang Nguyen, Minh Hua, Ngan Pham, Thang Bui, Tho Quan  

**Link**: [PDF](https://arxiv.org/pdf/2507.20491)  

**Abstract**: Recent advances in large language models (LLMs) have significantly enhanced question-answering (QA) capabilities, particularly in open-domain contexts. However, in closed-domain scenarios such as education, healthcare, and law, users demand not only accurate answers but also transparent reasoning and explainable decision-making processes. While neural-symbolic (NeSy) frameworks have emerged as a promising solution, leveraging LLMs for natural language understanding and symbolic systems for formal reasoning, existing approaches often rely on large-scale models and exhibit inefficiencies in translating natural language into formal logic representations.
To address these limitations, we introduce Text-JEPA (Text-based Joint-Embedding Predictive Architecture), a lightweight yet effective framework for converting natural language into first-order logic (NL2FOL). Drawing inspiration from dual-system cognitive theory, Text-JEPA emulates System 1 by efficiently generating logic representations, while the Z3 solver operates as System 2, enabling robust logical inference. To rigorously evaluate the NL2FOL-to-reasoning pipeline, we propose a comprehensive evaluation framework comprising three custom metrics: conversion score, reasoning score, and Spearman rho score, which collectively capture the quality of logical translation and its downstream impact on reasoning accuracy.
Empirical results on domain-specific datasets demonstrate that Text-JEPA achieves competitive performance with significantly lower computational overhead compared to larger LLM-based systems. Our findings highlight the potential of structured, interpretable reasoning frameworks for building efficient and explainable QA systems in specialized domains. 

---
# CodeNER: Code Prompting for Named Entity Recognition 

**Authors**: Sungwoo Han, Hyeyeon Kim, Jingun Kwon, Hidetaka Kamigaito, Manabu Okumura  

**Link**: [PDF](https://arxiv.org/pdf/2507.20423)  

**Abstract**: Recent studies have explored various approaches for treating candidate named entity spans as both source and target sequences in named entity recognition (NER) by leveraging large language models (LLMs). Although previous approaches have successfully generated candidate named entity spans with suitable labels, they rely solely on input context information when using LLMs, particularly, ChatGPT. However, NER inherently requires capturing detailed labeling requirements with input context information. To address this issue, we propose a novel method that leverages code-based prompting to improve the capabilities of LLMs in understanding and performing NER. By embedding code within prompts, we provide detailed BIO schema instructions for labeling, thereby exploiting the ability of LLMs to comprehend long-range scopes in programming languages. Experimental results demonstrate that the proposed code-based prompting method outperforms conventional text-based prompting on ten benchmarks across English, Arabic, Finnish, Danish, and German datasets, indicating the effectiveness of explicitly structuring NER instructions. We also verify that combining the proposed code-based prompting method with the chain-of-thought prompting further improves performance. 

---
# Survey of NLU Benchmarks Diagnosing Linguistic Phenomena: Why not Standardize Diagnostics Benchmarks? 

**Authors**: Khloud AL Jallad, Nada Ghneim, Ghaida Rebdawi  

**Link**: [PDF](https://arxiv.org/pdf/2507.20419)  

**Abstract**: Natural Language Understanding (NLU) is a basic task in Natural Language Processing (NLP). The evaluation of NLU capabilities has become a trending research topic that attracts researchers in the last few years, resulting in the development of numerous benchmarks. These benchmarks include various tasks and datasets in order to evaluate the results of pretrained models via public leaderboards. Notably, several benchmarks contain diagnostics datasets designed for investigation and fine-grained error analysis across a wide range of linguistic phenomena. This survey provides a comprehensive review of available English, Arabic, and Multilingual NLU benchmarks, with a particular emphasis on their diagnostics datasets and the linguistic phenomena they covered. We present a detailed comparison and analysis of these benchmarks, highlighting their strengths and limitations in evaluating NLU tasks and providing in-depth error analysis. When highlighting the gaps in the state-of-the-art, we noted that there is no naming convention for macro and micro categories or even a standard set of linguistic phenomena that should be covered. Consequently, we formulated a research question regarding the evaluation metrics of the evaluation diagnostics benchmarks: "Why do not we have an evaluation standard for the NLU evaluation diagnostics benchmarks?" similar to ISO standard in industry. We conducted a deep analysis and comparisons of the covered linguistic phenomena in order to support experts in building a global hierarchy for linguistic phenomena in future. We think that having evaluation metrics for diagnostics evaluation could be valuable to gain more insights when comparing the results of the studied models on different diagnostics benchmarks. 

---
# CONCAP: Seeing Beyond English with Concepts Retrieval-Augmented Captioning 

**Authors**: George Ibrahim, Rita Ramos, Yova Kementchedjhieva  

**Link**: [PDF](https://arxiv.org/pdf/2507.20411)  

**Abstract**: Multilingual vision-language models have made significant strides in image captioning, yet they still lag behind their English counterparts due to limited multilingual training data and costly large-scale model parameterization. Retrieval-augmented generation (RAG) offers a promising alternative by conditioning caption generation on retrieved examples in the target language, reducing the need for extensive multilingual training. However, multilingual RAG captioning models often depend on retrieved captions translated from English, which can introduce mismatches and linguistic biases relative to the source language. We introduce CONCAP, a multilingual image captioning model that integrates retrieved captions with image-specific concepts, enhancing the contextualization of the input image and grounding the captioning process across different languages. Experiments on the XM3600 dataset indicate that CONCAP enables strong performance on low- and mid-resource languages, with highly reduced data requirements. Our findings highlight the effectiveness of concept-aware retrieval augmentation in bridging multilingual performance gaps. 

---
# Cognitive Chain-of-Thought: Structured Multimodal Reasoning about Social Situations 

**Authors**: Eunkyu Park, Wesley Hanwen Deng, Gunhee Kim, Motahhare Eslami, Maarten Sap  

**Link**: [PDF](https://arxiv.org/pdf/2507.20409)  

**Abstract**: Chain-of-Thought (CoT) prompting helps models think step by step. But what happens when they must see, understand, and judge-all at once? In visual tasks grounded in social context, where bridging perception with norm-grounded judgments is essential, flat CoT often breaks down. We introduce Cognitive Chain-of-Thought (CoCoT), a prompting strategy that scaffolds VLM reasoning through three cognitively inspired stages: perception, situation, and norm. Our experiments show that, across multiple multimodal benchmarks (including intent disambiguation, commonsense reasoning, and safety), CoCoT consistently outperforms CoT and direct prompting (+8\% on average). Our findings demonstrate that cognitively grounded reasoning stages enhance interpretability and social awareness in VLMs, paving the way for safer and more reliable multimodal systems. 

---
# Length Representations in Large Language Models 

**Authors**: Sangjun Moon, Dasom Choi, Jingun Kwon, Hidetaka Kamigaito, Manabu Okumura  

**Link**: [PDF](https://arxiv.org/pdf/2507.20398)  

**Abstract**: Large language models (LLMs) have shown remarkable capabilities across various tasks, that are learned from massive amounts of text-based data. Although LLMs can control output sequence length, particularly in instruction-based settings, the internal mechanisms behind this control have been unexplored yet. In this study, we provide empirical evidence on how output sequence length information is encoded within the internal representations in LLMs. In particular, our findings show that multi-head attention mechanisms are critical in determining output sequence length, which can be adjusted in a disentangled manner. By scaling specific hidden units within the model, we can control the output sequence length without losing the informativeness of the generated text, thereby indicating that length information is partially disentangled from semantic information. Moreover, some hidden units become increasingly active as prompts become more length-specific, thus reflecting the model's internal awareness of this attribute. Our findings suggest that LLMs have learned robust and adaptable internal mechanisms for controlling output length without any external control. 

---
# RMTBench: Benchmarking LLMs Through Multi-Turn User-Centric Role-Playing 

**Authors**: Hao Xiang, Tianyi Tang, Yang Su, Bowen Yu, An Yang, Fei Huang, Yichang Zhang, Yaojie Lu, Hongyu Lin, Xianpei Han, Jingren Zhou, Junyang Lin, Le Sun  

**Link**: [PDF](https://arxiv.org/pdf/2507.20352)  

**Abstract**: Recent advancements in Large Language Models (LLMs) have shown outstanding potential for role-playing applications. Evaluating these capabilities is becoming crucial yet remains challenging. Existing benchmarks mostly adopt a \textbf{character-centric} approach, simplify user-character interactions to isolated Q&A tasks, and fail to reflect real-world applications. To address this limitation, we introduce RMTBench, a comprehensive \textbf{user-centric} bilingual role-playing benchmark featuring 80 diverse characters and over 8,000 dialogue rounds. RMTBench includes custom characters with detailed backgrounds and abstract characters defined by simple traits, enabling evaluation across various user scenarios. Our benchmark constructs dialogues based on explicit user motivations rather than character descriptions, ensuring alignment with practical user applications. Furthermore, we construct an authentic multi-turn dialogue simulation mechanism. With carefully selected evaluation dimensions and LLM-based scoring, this mechanism captures the complex intention of conversations between the user and the character. By shifting focus from character background to user intention fulfillment, RMTBench bridges the gap between academic evaluation and practical deployment requirements, offering a more effective framework for assessing role-playing capabilities in LLMs. All code and datasets will be released soon. 

---
# DYNARTmo: A Dynamic Articulatory Model for Visualization of Speech Movement Patterns 

**Authors**: Bernd J. Kröger  

**Link**: [PDF](https://arxiv.org/pdf/2507.20343)  

**Abstract**: We present DYNARTmo, a dynamic articulatory model designed to visualize speech articulation processes in a two-dimensional midsagittal plane. The model builds upon the UK-DYNAMO framework and integrates principles of articulatory underspecification, segmental and gestural control, and coarticulation. DYNARTmo simulates six key articulators based on ten continuous and six discrete control parameters, allowing for the generation of both vocalic and consonantal articulatory configurations. The current implementation is embedded in a web-based application (SpeechArticulationTrainer) that includes sagittal, glottal, and palatal views, making it suitable for use in phonetics education and speech therapy. While this paper focuses on the static modeling aspects, future work will address dynamic movement generation and integration with articulatory-acoustic modules. 

---
# Advancing Dialectal Arabic to Modern Standard Arabic Machine Translation 

**Authors**: Abdullah Alabdullah, Lifeng Han, Chenghua Lin  

**Link**: [PDF](https://arxiv.org/pdf/2507.20301)  

**Abstract**: Dialectal Arabic (DA) poses a persistent challenge for natural language processing (NLP), as most everyday communication in the Arab world occurs in dialects that diverge significantly from Modern Standard Arabic (MSA). This linguistic divide limits access to digital services and educational resources and impedes progress in Arabic machine translation. This paper presents two core contributions to advancing DA-MSA translation for the Levantine, Egyptian, and Gulf dialects, particularly in low-resource and computationally constrained settings: a comprehensive evaluation of training-free prompting techniques, and the development of a resource-efficient fine-tuning pipeline. Our evaluation of prompting strategies across six large language models (LLMs) found that few-shot prompting consistently outperformed zero-shot, chain-of-thought, and our proposed Ara-TEaR method. GPT-4o achieved the highest performance across all prompting settings. For fine-tuning, a quantized Gemma2-9B model achieved a CHrF++ score of 49.88, outperforming zero-shot GPT-4o (44.58). Joint multi-dialect trained models outperformed single-dialect counterparts by over 10% CHrF++, and 4-bit quantization reduced memory usage by 60% with less than 1% performance loss. The results and insights of our experiments offer a practical blueprint for improving dialectal inclusion in Arabic NLP, showing that high-quality DA-MSA machine translation is achievable even with limited resources and paving the way for more inclusive language technologies. 

---
# What Language(s) Does Aya-23 Think In? How Multilinguality Affects Internal Language Representations 

**Authors**: Katharina Trinley, Toshiki Nakai, Tatiana Anikina, Tanja Baeumel  

**Link**: [PDF](https://arxiv.org/pdf/2507.20279)  

**Abstract**: Large language models (LLMs) excel at multilingual tasks, yet their internal language processing remains poorly understood. We analyze how Aya-23-8B, a decoder-only LLM trained on balanced multilingual data, handles code-mixed, cloze, and translation tasks compared to predominantly monolingual models like Llama 3 and Chinese-LLaMA-2. Using logit lens and neuron specialization analyses, we find: (1) Aya-23 activates typologically related language representations during translation, unlike English-centric models that rely on a single pivot language; (2) code-mixed neuron activation patterns vary with mixing rates and are shaped more by the base language than the mixed-in one; and (3) Aya-23's languagespecific neurons for code-mixed inputs concentrate in final layers, diverging from prior findings on decoder-only models. Neuron overlap analysis further shows that script similarity and typological relations impact processing across model types. These findings reveal how multilingual training shapes LLM internals and inform future cross-lingual transfer research. 

---
# MoL-RL: Distilling Multi-Step Environmental Feedback into LLMs for Feedback-Independent Reasoning 

**Authors**: Kang Yang, Jingxue Chen, Qingkun Tang, Tianxiang Zhang, Qianchun Lu  

**Link**: [PDF](https://arxiv.org/pdf/2507.20278)  

**Abstract**: Large language models (LLMs) face significant challenges in effectively leveraging sequential environmental feedback (EF) signals, such as natural language evaluations, for feedback-independent chain-of-thought (CoT) reasoning. Existing approaches either convert EF into scalar rewards, losing rich contextual information, or employ refinement datasets, failing to exploit the multi-step and discrete nature of EF interactions. To address these limitations, we propose MoL-RL, a novel training paradigm that integrates multi-step EF signals into LLMs through a dual-objective optimization framework. Our method combines MoL (Mixture-of-Losses) continual training, which decouples domain-specific EF signals (optimized via cross-entropy loss) and general language capabilities (preserved via Kullback-Leibler divergence), with GRPO-based post-training to distill sequential EF interactions into single-step inferences. This synergy enables robust feedback-independent reasoning without relying on external feedback loops. Experimental results on mathematical reasoning (MATH-500, AIME24/AIME25) and code generation (CodeAgent-Test) benchmarks demonstrate that MoL-RL achieves state-of-the-art performance with the Qwen3-8B model, while maintaining strong generalization across model scales (Qwen3-4B). This work provides a promising approach for leveraging multi-step textual feedback to enhance LLMs' reasoning capabilities in diverse domains. 

---
# EMBRACE: Shaping Inclusive Opinion Representation by Aligning Implicit Conversations with Social Norms 

**Authors**: Abeer Aldayel, Areej Alokaili  

**Link**: [PDF](https://arxiv.org/pdf/2507.20264)  

**Abstract**: Shaping inclusive representations that embrace diversity and ensure fair participation and reflections of values is at the core of many conversation-based models. However, many existing methods rely on surface inclusion using mention of user demographics or behavioral attributes of social groups. Such methods overlook the nuanced, implicit expression of opinion embedded in conversations. Furthermore, the over-reliance on overt cues can exacerbate misalignment and reinforce harmful or stereotypical representations in model outputs. Thus, we took a step back and recognized that equitable inclusion needs to account for the implicit expression of opinion and use the stance of responses to validate the normative alignment. This study aims to evaluate how opinions are represented in NLP or computational models by introducing an alignment evaluation framework that foregrounds implicit, often overlooked conversations and evaluates the normative social views and discourse. Our approach models the stance of responses as a proxy for the underlying opinion, enabling a considerate and reflective representation of diverse social viewpoints. We evaluate the framework using both (i) positive-unlabeled (PU) online learning with base classifiers, and (ii) instruction-tuned language models to assess post-training alignment. Through this, we provide a lens on how implicit opinions are (mis)represented and offer a pathway toward more inclusive model behavior. 

---
# Post-Completion Learning for Language Models 

**Authors**: Xiang Fei, Siqi Wang, Shu Wei, Yuxiang Nie, Wei Shi, Hao Feng, Can Huang  

**Link**: [PDF](https://arxiv.org/pdf/2507.20252)  

**Abstract**: Current language model training paradigms typically terminate learning upon reaching the end-of-sequence (<eos>}) token, overlooking the potential learning opportunities in the post-completion space. We propose Post-Completion Learning (PCL), a novel training framework that systematically utilizes the sequence space after model output completion, to enhance both the reasoning and self-evaluation abilities. PCL enables models to continue generating self-assessments and reward predictions during training, while maintaining efficient inference by stopping at the completion point.
To fully utilize this post-completion space, we design a white-box reinforcement learning method: let the model evaluate the output content according to the reward rules, then calculate and align the score with the reward functions for supervision. We implement dual-track SFT to optimize both reasoning and evaluation capabilities, and mixed it with RL training to achieve multi-objective hybrid optimization.
Experimental results on different datasets and models demonstrate consistent improvements over traditional SFT and RL methods. Our method provides a new technical path for language model training that enhances output quality while preserving deployment efficiency. 

---
# Modeling Professionalism in Expert Questioning through Linguistic Differentiation 

**Authors**: Giulia D'Agostino, Chung-Chi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.20249)  

**Abstract**: Professionalism is a crucial yet underexplored dimension of expert communication, particularly in high-stakes domains like finance. This paper investigates how linguistic features can be leveraged to model and evaluate professionalism in expert questioning. We introduce a novel annotation framework to quantify structural and pragmatic elements in financial analyst questions, such as discourse regulators, prefaces, and request types. Using both human-authored and large language model (LLM)-generated questions, we construct two datasets: one annotated for perceived professionalism and one labeled by question origin. We show that the same linguistic features correlate strongly with both human judgments and authorship origin, suggesting a shared stylistic foundation. Furthermore, a classifier trained solely on these interpretable features outperforms gemini-2.0 and SVM baselines in distinguishing expert-authored questions. Our findings demonstrate that professionalism is a learnable, domain-general construct that can be captured through linguistically grounded modeling. 

---
# Reframe Your Life Story: Interactive Narrative Therapist and Innovative Moment Assessment with Large Language Models 

**Authors**: Yi Feng, Jiaqi Wang, Wenxuan Zhang, Zhuang Chen, Yutong Shen, Xiyao Xiao, Minlie Huang, Liping Jing, Jian Yu  

**Link**: [PDF](https://arxiv.org/pdf/2507.20241)  

**Abstract**: Recent progress in large language models (LLMs) has opened new possibilities for mental health support, yet current approaches lack realism in simulating specialized psychotherapy and fail to capture therapeutic progression over time. Narrative therapy, which helps individuals transform problematic life stories into empowering alternatives, remains underutilized due to limited access and social stigma. We address these limitations through a comprehensive framework with two core components. First, INT (Interactive Narrative Therapist) simulates expert narrative therapists by planning therapeutic stages, guiding reflection levels, and generating contextually appropriate expert-like responses. Second, IMA (Innovative Moment Assessment) provides a therapy-centric evaluation method that quantifies effectiveness by tracking "Innovative Moments" (IMs), critical narrative shifts in client speech signaling therapy progress. Experimental results on 260 simulated clients and 230 human participants reveal that INT consistently outperforms standard LLMs in therapeutic quality and depth. We further demonstrate the effectiveness of INT in synthesizing high-quality support conversations to facilitate social applications. 

---
# Co-NAML-LSTUR: A Combined Model with Attentive Multi-View Learning and Long- and Short-term User Representations for News Recommendation 

**Authors**: Minh Hoang Nguyen, Thuat Thien Nguyen, Minh Nhat Ta  

**Link**: [PDF](https://arxiv.org/pdf/2507.20210)  

**Abstract**: News recommendation systems play a vital role in mitigating information overload by delivering personalized news content. A central challenge is to effectively model both multi-view news representations and the dynamic nature of user interests, which often span both short- and long-term preferences. Existing methods typically rely on single-view features of news articles (e.g., titles or categories) or fail to comprehensively capture user preferences across time scales. In this work, we propose Co-NAML-LSTUR, a hybrid news recommendation framework that integrates NAML for attentive multi-view news modeling and LSTUR for capturing both long- and short-term user representations. Our model also incorporates BERT-based word embeddings to enhance semantic feature extraction. We evaluate Co-NAML-LSTUR on two widely used benchmarks, MIND-small and MIND-large. Experimental results show that Co-NAML-LSTUR achieves substantial improvements over most state-of-the-art baselines on MIND-small and MIND-large, respectively. These results demonstrate the effectiveness of combining multi-view news representations with dual-scale user modeling. The implementation of our model is publicly available at this https URL. 

---
# IQ Test for LLMs: An Evaluation Framework for Uncovering Core Skills in LLMs 

**Authors**: Aviya Maimon, Amir DN Cohen, Gal Vishne, Shauli Ravfogel, Reut Tsarfaty  

**Link**: [PDF](https://arxiv.org/pdf/2507.20208)  

**Abstract**: Current evaluations of large language models (LLMs) rely on benchmark scores, but it is difficult to interpret what these individual scores reveal about a model's overall skills. Specifically, as a community we lack understanding of how tasks relate to one another, what they measure in common, how they differ, or which ones are redundant. As a result, models are often assessed via a single score averaged across benchmarks, an approach that fails to capture the models' wholistic strengths and limitations. Here, we propose a new evaluation paradigm that uses factor analysis to identify latent skills driving performance across benchmarks. We apply this method to a comprehensive new leaderboard showcasing the performance of 60 LLMs on 44 tasks, and identify a small set of latent skills that largely explain performance. Finally, we turn these insights into practical tools that identify redundant tasks, aid in model selection, and profile models along each latent skill. 

---
# Diversity-Enhanced Reasoning for Subjective Questions 

**Authors**: Yumeng Wang, Zhiyuan Fan, Jiayu Liu, Yi R. Fung  

**Link**: [PDF](https://arxiv.org/pdf/2507.20187)  

**Abstract**: Large reasoning models (LRM) with long chain-of-thought (CoT) capabilities have shown strong performance on objective tasks, such as math reasoning and coding. However, their effectiveness on subjective questions that may have different responses from different perspectives is still limited by a tendency towards homogeneous reasoning, introduced by the reliance on a single ground truth in supervised fine-tuning and verifiable reward in reinforcement learning. Motivated by the finding that increasing role perspectives consistently improves performance, we propose MultiRole-R1, a diversity-enhanced framework with multiple role perspectives, to improve the accuracy and diversity in subjective reasoning tasks. MultiRole-R1 features an unsupervised data construction pipeline that generates reasoning chains that incorporate diverse role perspectives. We further employ reinforcement learning via Group Relative Policy Optimization (GRPO) with reward shaping, by taking diversity as a reward signal in addition to the verifiable reward. With specially designed reward functions, we successfully promote perspective diversity and lexical diversity, uncovering a positive relation between reasoning diversity and accuracy. Our experiment on six benchmarks demonstrates MultiRole-R1's effectiveness and generalizability in enhancing both subjective and objective reasoning, showcasing the potential of diversity-enhanced training in LRMs. 

---
# SessionIntentBench: A Multi-task Inter-session Intention-shift Modeling Benchmark for E-commerce Customer Behavior Understanding 

**Authors**: Yuqi Yang, Weiqi Wang, Baixuan Xu, Wei Fan, Qing Zong, Chunkit Chan, Zheye Deng, Xin Liu, Yifan Gao, Changlong Yu, Chen Luo, Yang Li, Zheng Li, Qingyu Yin, Bing Yin, Yangqiu Song  

**Link**: [PDF](https://arxiv.org/pdf/2507.20185)  

**Abstract**: Session history is a common way of recording user interacting behaviors throughout a browsing activity with multiple products. For example, if an user clicks a product webpage and then leaves, it might because there are certain features that don't satisfy the user, which serve as an important indicator of on-the-spot user preferences. However, all prior works fail to capture and model customer intention effectively because insufficient information exploitation and only apparent information like descriptions and titles are used. There is also a lack of data and corresponding benchmark for explicitly modeling intention in E-commerce product purchase sessions. To address these issues, we introduce the concept of an intention tree and propose a dataset curation pipeline. Together, we construct a sibling multimodal benchmark, SessionIntentBench, that evaluates L(V)LMs' capability on understanding inter-session intention shift with four subtasks. With 1,952,177 intention entries, 1,132,145 session intention trajectories, and 13,003,664 available tasks mined using 10,905 sessions, we provide a scalable way to exploit the existing session data for customer intention understanding. We conduct human annotations to collect ground-truth label for a subset of collected data to form an evaluation gold set. Extensive experiments on the annotated data further confirm that current L(V)LMs fail to capture and utilize the intention across the complex session setting. Further analysis show injecting intention enhances LLMs' performances. 

---
# SGPO: Self-Generated Preference Optimization based on Self-Improver 

**Authors**: Hyeonji Lee, Daejin Jo, Seohwan Yun, Sungwoong Kim  

**Link**: [PDF](https://arxiv.org/pdf/2507.20181)  

**Abstract**: Large language models (LLMs), despite their extensive pretraining on diverse datasets, require effective alignment to human preferences for practical and reliable deployment. Conventional alignment methods typically employ off-policy learning and depend on human-annotated datasets, which limits their broad applicability and introduces distribution shift issues during training. To address these challenges, we propose Self-Generated Preference Optimization based on Self-Improver (SGPO), an innovative alignment framework that leverages an on-policy self-improving mechanism. Specifically, the improver refines responses from a policy model to self-generate preference data for direct preference optimization (DPO) of the policy model. Here, the improver and policy are unified into a single model, and in order to generate higher-quality preference data, this self-improver learns to make incremental yet discernible improvements to the current responses by referencing supervised fine-tuning outputs. Experimental results on AlpacaEval 2.0 and Arena-Hard show that the proposed SGPO significantly improves performance over DPO and baseline self-improving methods without using external preference data. 

---
# Goal Alignment in LLM-Based User Simulators for Conversational AI 

**Authors**: Shuhaib Mehri, Xiaocheng Yang, Takyoung Kim, Gokhan Tur, Shikib Mehri, Dilek Hakkani-Tür  

**Link**: [PDF](https://arxiv.org/pdf/2507.20152)  

**Abstract**: User simulators are essential to conversational AI, enabling scalable agent development and evaluation through simulated interactions. While current Large Language Models (LLMs) have advanced user simulation capabilities, we reveal that they struggle to consistently demonstrate goal-oriented behavior across multi-turn conversations--a critical limitation that compromises their reliability in downstream applications. We introduce User Goal State Tracking (UGST), a novel framework that tracks user goal progression throughout conversations. Leveraging UGST, we present a three-stage methodology for developing user simulators that can autonomously track goal progression and reason to generate goal-aligned responses. Moreover, we establish comprehensive evaluation metrics for measuring goal alignment in user simulators, and demonstrate that our approach yields substantial improvements across two benchmarks (MultiWOZ 2.4 and {\tau}-Bench). Our contributions address a critical gap in conversational AI and establish UGST as an essential framework for developing goal-aligned user simulators. 

---
# Multi-Agent Interactive Question Generation Framework for Long Document Understanding 

**Authors**: Kesen Wang, Daulet Toibazar, Abdulrahman Alfulayt, Abdulaziz S. Albadawi, Ranya A. Alkahtani, Asma A. Ibrahim, Haneen A. Alhomoud, Sherif Mohamed, Pedro J. Moreno  

**Link**: [PDF](https://arxiv.org/pdf/2507.20145)  

**Abstract**: Document Understanding (DU) in long-contextual scenarios with complex layouts remains a significant challenge in vision-language research. Although Large Vision-Language Models (LVLMs) excel at short-context DU tasks, their performance declines in long-context settings. A key limitation is the scarcity of fine-grained training data, particularly for low-resource languages such as Arabic. Existing state-of-the-art techniques rely heavily on human annotation, which is costly and inefficient. We propose a fully automated, multi-agent interactive framework to generate long-context questions efficiently. Our approach efficiently generates high-quality single- and multi-page questions for extensive English and Arabic documents, covering hundreds of pages across diverse domains. This facilitates the development of LVLMs with enhanced long-context understanding ability. Experimental results in this work have shown that our generated English and Arabic questions (\textbf{AraEngLongBench}) are quite challenging to major open- and close-source LVLMs. The code and data proposed in this work can be found in this https URL. Sample Question and Answer (QA) pairs and structured system prompts can be found in the Appendix. 

---
# Multi-Stage Verification-Centric Framework for Mitigating Hallucination in Multi-Modal RAG 

**Authors**: Baiyu Chen, Wilson Wongso, Xiaoqian Hu, Yue Tan, Flora Salim  

**Link**: [PDF](https://arxiv.org/pdf/2507.20136)  

**Abstract**: This paper presents the technical solution developed by team CRUISE for the KDD Cup 2025 Meta Comprehensive RAG Benchmark for Multi-modal, Multi-turn (CRAG-MM) challenge. The challenge aims to address a critical limitation of modern Vision Language Models (VLMs): their propensity to hallucinate, especially when faced with egocentric imagery, long-tail entities, and complex, multi-hop questions. This issue is particularly problematic in real-world applications where users pose fact-seeking queries that demand high factual accuracy across diverse modalities. To tackle this, we propose a robust, multi-stage framework that prioritizes factual accuracy and truthfulness over completeness. Our solution integrates a lightweight query router for efficiency, a query-aware retrieval and summarization pipeline, a dual-pathways generation and a post-hoc verification. This conservative strategy is designed to minimize hallucinations, which incur a severe penalty in the competition's scoring metric. Our approach achieved 3rd place in Task 1, demonstrating the effectiveness of prioritizing answer reliability in complex multi-modal RAG systems. Our implementation is available at this https URL . 

---
# Sem-DPO: Mitigating Semantic Inconsistency in Preference Optimization for Prompt Engineering 

**Authors**: Anas Mohamed, Azal Ahmad Khan, Xinran Wang, Ahmad Faraz Khan, Shuwen Ge, Saman Bahzad Khan, Ayaan Ahmad, Ali Anwar  

**Link**: [PDF](https://arxiv.org/pdf/2507.20133)  

**Abstract**: Generative AI can now synthesize strikingly realistic images from text, yet output quality remains highly sensitive to how prompts are phrased. Direct Preference Optimization (DPO) offers a lightweight, off-policy alternative to RL for automatic prompt engineering, but its token-level regularization leaves semantic inconsistency unchecked as prompts that win higher preference scores can still drift away from the user's intended meaning.
We introduce Sem-DPO, a variant of DPO that preserves semantic consistency yet retains its simplicity and efficiency. Sem-DPO scales the DPO loss by an exponential weight proportional to the cosine distance between the original prompt and winning candidate in embedding space, softly down-weighting training signals that would otherwise reward semantically mismatched prompts. We provide the first analytical bound on semantic drift for preference-tuned prompt generators, showing that Sem-DPO keeps learned prompts within a provably bounded neighborhood of the original text. On three standard text-to-image prompt-optimization benchmarks and two language models, Sem-DPO achieves 8-12% higher CLIP similarity and 5-9% higher human-preference scores (HPSv2.1, PickScore) than DPO, while also outperforming state-of-the-art baselines. These findings suggest that strong flat baselines augmented with semantic weighting should become the new standard for prompt-optimization studies and lay the groundwork for broader, semantics-aware preference optimization in language models. 

---
# AI-Driven Generation of Old English: A Framework for Low-Resource Languages 

**Authors**: Rodrigo Gabriel Salazar Alva, Matías Nuñez, Cristian López, Javier Martín Arista  

**Link**: [PDF](https://arxiv.org/pdf/2507.20111)  

**Abstract**: Preserving ancient languages is essential for understanding humanity's cultural and linguistic heritage, yet Old English remains critically under-resourced, limiting its accessibility to modern natural language processing (NLP) techniques. We present a scalable framework that uses advanced large language models (LLMs) to generate high-quality Old English texts, addressing this gap. Our approach combines parameter-efficient fine-tuning (Low-Rank Adaptation, LoRA), data augmentation via backtranslation, and a dual-agent pipeline that separates the tasks of content generation (in English) and translation (into Old English). Evaluation with automated metrics (BLEU, METEOR, and CHRF) shows significant improvements over baseline models, with BLEU scores increasing from 26 to over 65 for English-to-Old English translation. Expert human assessment also confirms high grammatical accuracy and stylistic fidelity in the generated texts. Beyond expanding the Old English corpus, our method offers a practical blueprint for revitalizing other endangered languages, effectively uniting AI innovation with the goals of cultural preservation. 

---
# ProsodyLM: Uncovering the Emerging Prosody Processing Capabilities in Speech Language Models 

**Authors**: Kaizhi Qian, Xulin Fan, Junrui Ni, Slava Shechtman, Mark Hasegawa-Johnson, Chuang Gan, Yang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.20091)  

**Abstract**: Speech language models refer to language models with speech processing and understanding capabilities. One key desirable capability for speech language models is the ability to capture the intricate interdependency between content and prosody. The existing mainstream paradigm of training speech language models, which converts speech into discrete tokens before feeding them into LLMs, is sub-optimal in learning prosody information -- we find that the resulting LLMs do not exhibit obvious emerging prosody processing capabilities via pre-training alone. To overcome this, we propose ProsodyLM, which introduces a simple tokenization scheme amenable to learning prosody. Each speech utterance is first transcribed into text, followed by a sequence of word-level prosody tokens. Compared with conventional speech tokenization schemes, the proposed tokenization scheme retains more complete prosody information, and is more understandable to text-based LLMs. We find that ProsodyLM can learn surprisingly diverse emerging prosody processing capabilities through pre-training alone, ranging from harnessing the prosody nuances in generated speech, such as contrastive focus, understanding emotion and stress in an utterance, to maintaining prosody consistency in long contexts. 

---
# RAG in the Wild: On the (In)effectiveness of LLMs with Mixture-of-Knowledge Retrieval Augmentation 

**Authors**: Ran Xu, Yuchen Zhuang, Yue Yu, Haoyu Wang, Wenqi Shi, Carl Yang  

**Link**: [PDF](https://arxiv.org/pdf/2507.20059)  

**Abstract**: Retrieval-augmented generation (RAG) enhances large language models (LLMs) by integrating external knowledge retrieved at inference time. While RAG demonstrates strong performance on benchmarks largely derived from general-domain corpora like Wikipedia, its effectiveness under realistic, diverse retrieval scenarios remains underexplored. We evaluated RAG systems using MassiveDS, a large-scale datastore with mixture of knowledge, and identified critical limitations: retrieval mainly benefits smaller models, rerankers add minimal value, and no single retrieval source consistently excels. Moreover, current LLMs struggle to route queries across heterogeneous knowledge sources. These findings highlight the need for adaptive retrieval strategies before deploying RAG in real-world settings. Our code and data can be found at this https URL. 

---
# A Tensor-Based Compiler and a Runtime for Neuron-Level DNN Certifier Specifications 

**Authors**: Avaljot Singh, Yamin Chandini Sarita, Aditya Mishra, Ishaan Goyal, Gagandeep Singh, Charith Mendis  

**Link**: [PDF](https://arxiv.org/pdf/2507.20055)  

**Abstract**: The uninterpretability of DNNs has led to the adoption of abstract interpretation-based certification as a practical means to establish trust in real-world systems that rely on DNNs. However, the current landscape supports only a limited set of certifiers, and developing new ones or modifying existing ones for different applications remains difficult. This is because the mathematical design of certifiers is expressed at the neuron level, while their implementations are optimized and executed at the tensor level. This mismatch creates a semantic gap between design and implementation, making manual bridging both complex and expertise-intensive -- requiring deep knowledge in formal methods, high-performance computing, etc.
We propose a compiler framework that automatically translates neuron-level specifications of DNN certifiers into tensor-based, layer-level implementations. This is enabled by two key innovations: a novel stack-based intermediate representation (IR) and a shape analysis that infers the implicit tensor operations needed to simulate the neuron-level semantics. During lifting, the shape analysis creates tensors in the minimal shape required to perform the corresponding operations. The IR also enables domain-specific optimizations as rewrites. At runtime, the resulting tensor computations exhibit sparsity tied to the DNN architecture. This sparsity does not align well with existing formats. To address this, we introduce g-BCSR, a double-compression format that represents tensors as collections of blocks of varying sizes, each possibly internally sparse.
Using our compiler and g-BCSR, we make it easy to develop new certifiers and analyze their utility across diverse DNNs. Despite its flexibility, the compiler achieves performance comparable to hand-optimized implementations. 

---
# Infogen: Generating Complex Statistical Infographics from Documents 

**Authors**: Akash Ghosh, Aparna Garimella, Pritika Ramu, Sambaran Bandyopadhyay, Sriparna Saha  

**Link**: [PDF](https://arxiv.org/pdf/2507.20046)  

**Abstract**: Statistical infographics are powerful tools that simplify complex data into visually engaging and easy-to-understand formats. Despite advancements in AI, particularly with LLMs, existing efforts have been limited to generating simple charts, with no prior work addressing the creation of complex infographics from text-heavy documents that demand a deep understanding of the content. We address this gap by introducing the task of generating statistical infographics composed of multiple sub-charts (e.g., line, bar, pie) that are contextually accurate, insightful, and visually aligned. To achieve this, we define infographic metadata that includes its title and textual insights, along with sub-chart-specific details such as their corresponding data and alignment. We also present Infodat, the first benchmark dataset for text-to-infographic metadata generation, where each sample links a document to its metadata. We propose Infogen, a two-stage framework where fine-tuned LLMs first generate metadata, which is then converted into infographic code. Extensive evaluations on Infodat demonstrate that Infogen achieves state-of-the-art performance, outperforming both closed and open-source LLMs in text-to-statistical infographic generation. 

---
# FAEDKV: Infinite-Window Fourier Transform for Unbiased KV Cache Compression 

**Authors**: Runchao Li, Yao Fu, Mu Sheng, Xianxuan Long, Haotian Yu, Pan Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.20030)  

**Abstract**: The efficacy of Large Language Models (LLMs) in long-context tasks is often hampered by the substantial memory footprint and computational demands of the Key-Value (KV) cache. Current compression strategies, including token eviction and learned projections, frequently lead to biased representations -- either by overemphasizing recent/high-attention tokens or by repeatedly degrading information from earlier context -- and may require costly model retraining. We present FAEDKV (Frequency-Adaptive Infinite-Window for KV cache), a novel, training-free KV cache compression framework that ensures unbiased information retention. FAEDKV operates by transforming the KV cache into the frequency domain using a proposed Infinite-Window Fourier Transform (IWDFT). This approach allows for the equalized contribution of all tokens to the compressed representation, effectively preserving both early and recent contextual information. A preliminary frequency ablation study identifies critical spectral components for layer-wise, targeted compression. Experiments on LongBench benchmark demonstrate FAEDKV's superiority over existing methods by up to 22\%. In addition, our method shows superior, position-agnostic retrieval accuracy on the Needle-In-A-Haystack task compared to compression based approaches. 

---
# Anomaly Detection in Human Language via Meta-Learning: A Few-Shot Approach 

**Authors**: Saurav Singla, Aarav Singla, Advik Gupta, Parnika Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2507.20019)  

**Abstract**: We propose a meta learning framework for detecting anomalies in human language across diverse domains with limited labeled data. Anomalies in language ranging from spam and fake news to hate speech pose a major challenge due to their sparsity and variability. We treat anomaly detection as a few shot binary classification problem and leverage meta-learning to train models that generalize across tasks. Using datasets from domains such as SMS spam, COVID-19 fake news, and hate speech, we evaluate model generalization on unseen tasks with minimal labeled anomalies. Our method combines episodic training with prototypical networks and domain resampling to adapt quickly to new anomaly detection tasks. Empirical results show that our method outperforms strong baselines in F1 and AUC scores. We also release the code and benchmarks to facilitate further research in few-shot text anomaly detection. 

---
# VLQA: The First Comprehensive, Large, and High-Quality Vietnamese Dataset for Legal Question Answering 

**Authors**: Tan-Minh Nguyen, Hoang-Trung Nguyen, Trong-Khoi Dao, Xuan-Hieu Phan, Ha-Thanh Nguyen, Thi-Hai-Yen Vuong  

**Link**: [PDF](https://arxiv.org/pdf/2507.19995)  

**Abstract**: The advent of large language models (LLMs) has led to significant achievements in various domains, including legal text processing. Leveraging LLMs for legal tasks is a natural evolution and an increasingly compelling choice. However, their capabilities are often portrayed as greater than they truly are. Despite the progress, we are still far from the ultimate goal of fully automating legal tasks using artificial intelligence (AI) and natural language processing (NLP). Moreover, legal systems are deeply domain-specific and exhibit substantial variation across different countries and languages. The need for building legal text processing applications for different natural languages is, therefore, large and urgent. However, there is a big challenge for legal NLP in low-resource languages such as Vietnamese due to the scarcity of resources and annotated data. The need for labeled legal corpora for supervised training, validation, and supervised fine-tuning is critical. In this paper, we introduce the VLQA dataset, a comprehensive and high-quality resource tailored for the Vietnamese legal domain. We also conduct a comprehensive statistical analysis of the dataset and evaluate its effectiveness through experiments with state-of-the-art models on legal information retrieval and question-answering tasks. 

---
# Exploring LLM Autoscoring Reliability in Large-Scale Writing Assessments Using Generalizability Theory 

**Authors**: Dan Song, Won-Chan Lee, Hong Jiao  

**Link**: [PDF](https://arxiv.org/pdf/2507.19980)  

**Abstract**: This study investigates the estimation of reliability for large language models (LLMs) in scoring writing tasks from the AP Chinese Language and Culture Exam. Using generalizability theory, the research evaluates and compares score consistency between human and AI raters across two types of AP Chinese free-response writing tasks: story narration and email response. These essays were independently scored by two trained human raters and seven AI raters. Each essay received four scores: one holistic score and three analytic scores corresponding to the domains of task completion, delivery, and language use. Results indicate that although human raters produced more reliable scores overall, LLMs demonstrated reasonable consistency under certain conditions, particularly for story narration tasks. Composite scoring that incorporates both human and AI raters improved reliability, which supports that hybrid scoring models may offer benefits for large-scale writing assessments. 

---
# Text2Vis: A Challenging and Diverse Benchmark for Generating Multimodal Visualizations from Text 

**Authors**: Mizanur Rahman, Md Tahmid Rahman Laskar, Shafiq Joty, Enamul Hoque  

**Link**: [PDF](https://arxiv.org/pdf/2507.19969)  

**Abstract**: Automated data visualization plays a crucial role in simplifying data interpretation, enhancing decision-making, and improving efficiency. While large language models (LLMs) have shown promise in generating visualizations from natural language, the absence of comprehensive benchmarks limits the rigorous evaluation of their capabilities. We introduce Text2Vis, a benchmark designed to assess text-to-visualization models, covering 20+ chart types and diverse data science queries, including trend analysis, correlation, outlier detection, and predictive analytics. It comprises 1,985 samples, each with a data table, natural language query, short answer, visualization code, and annotated charts. The queries involve complex reasoning, conversational turns, and dynamic data retrieval. We benchmark 11 open-source and closed-source models, revealing significant performance gaps, highlighting key challenges, and offering insights for future advancements. To close this gap, we propose the first cross-modal actor-critic agentic framework that jointly refines the textual answer and visualization code, increasing GPT-4o`s pass rate from 26% to 42% over the direct approach and improving chart quality. We also introduce an automated LLM-based evaluation framework that enables scalable assessment across thousands of samples without human annotation, measuring answer correctness, code execution success, visualization readability, and chart accuracy. We release Text2Vis at this https URL. 

---
# KLAAD: Refining Attention Mechanisms to Reduce Societal Bias in Generative Language Models 

**Authors**: Seorin Kim, Dongyoung Lee, Jaejin Lee  

**Link**: [PDF](https://arxiv.org/pdf/2507.19962)  

**Abstract**: Large language models (LLMs) often exhibit societal biases in their outputs, prompting ethical concerns regarding fairness and harm. In this work, we propose KLAAD (KL-Attention Alignment Debiasing), an attention-based debiasing framework that implicitly aligns attention distributions between stereotypical and anti-stereotypical sentence pairs without directly modifying model weights. KLAAD introduces a composite training objective combining Cross-Entropy, KL divergence, and Triplet losses, guiding the model to consistently attend across biased and unbiased contexts while preserving fluency and coherence. Experimental evaluation of KLAAD demonstrates improved bias mitigation on both the BBQ and BOLD benchmarks, with minimal impact on language modeling quality. The results indicate that attention-level alignment offers a principled solution for mitigating bias in generative language models. 

---
# CaliDrop: KV Cache Compression with Calibration 

**Authors**: Yi Su, Quantong Qiu, Yuechi Zhou, Juntao Li, Qingrong Xia, Ping Li, Xinyu Duan, Zhefeng Wang, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.19906)  

**Abstract**: Large Language Models (LLMs) require substantial computational resources during generation. While the Key-Value (KV) cache significantly accelerates this process by storing attention intermediates, its memory footprint grows linearly with sequence length, batch size, and model size, creating a bottleneck in long-context scenarios. Various KV cache compression techniques, including token eviction, quantization, and low-rank projection, have been proposed to mitigate this bottleneck, often complementing each other. This paper focuses on enhancing token eviction strategies. Token eviction leverages the observation that the attention patterns are often sparse, allowing for the removal of less critical KV entries to save memory. However, this reduction usually comes at the cost of notable accuracy degradation, particularly under high compression ratios. To address this issue, we propose \textbf{CaliDrop}, a novel strategy that enhances token eviction through calibration. Our preliminary experiments show that queries at nearby positions exhibit high similarity. Building on this observation, CaliDrop performs speculative calibration on the discarded tokens to mitigate the accuracy loss caused by token eviction. Extensive experiments demonstrate that CaliDrop significantly improves the accuracy of existing token eviction methods. 

---
# A Gold Standard Dataset and Evaluation Framework for Depression Detection and Explanation in Social Media using LLMs 

**Authors**: Prajval Bolegave, Pushpak Bhattacharya  

**Link**: [PDF](https://arxiv.org/pdf/2507.19899)  

**Abstract**: Early detection of depression from online social media posts holds promise for providing timely mental health interventions. In this work, we present a high-quality, expert-annotated dataset of 1,017 social media posts labeled with depressive spans and mapped to 12 depression symptom categories. Unlike prior datasets that primarily offer coarse post-level labels \cite{cohan-etal-2018-smhd}, our dataset enables fine-grained evaluation of both model predictions and generated explanations.
We develop an evaluation framework that leverages this clinically grounded dataset to assess the faithfulness and quality of natural language explanations generated by large language models (LLMs). Through carefully designed prompting strategies, including zero-shot and few-shot approaches with domain-adapted examples, we evaluate state-of-the-art proprietary LLMs including GPT-4.1, Gemini 2.5 Pro, and Claude 3.7 Sonnet.
Our comprehensive empirical analysis reveals significant differences in how these models perform on clinical explanation tasks, with zero-shot and few-shot prompting. Our findings underscore the value of human expertise in guiding LLM behavior and offer a step toward safer, more transparent AI systems for psychological well-being. 

---
# Zero-shot Performance of Generative AI in Brazilian Portuguese Medical Exam 

**Authors**: Cesar Augusto Madid Truyts, Amanda Gomes Rabelo, Gabriel Mesquita de Souza, Daniel Scaldaferri Lages, Adriano Jose Pereira, Uri Adrian Prync Flato, Eduardo Pontes dos Reis, Joaquim Edson Vieira, Paulo Sergio Panse Silveira, Edson Amaro Junior  

**Link**: [PDF](https://arxiv.org/pdf/2507.19885)  

**Abstract**: Artificial intelligence (AI) has shown the potential to revolutionize healthcare by improving diagnostic accuracy, optimizing workflows, and personalizing treatment plans. Large Language Models (LLMs) and Multimodal Large Language Models (MLLMs) have achieved notable advancements in natural language processing and medical applications. However, the evaluation of these models has focused predominantly on the English language, leading to potential biases in their performance across different languages.
This study investigates the capability of six LLMs (GPT-4.0 Turbo, LLaMA-3-8B, LLaMA-3-70B, Mixtral 8x7B Instruct, Titan Text G1-Express, and Command R+) and four MLLMs (Claude-3.5-Sonnet, Claude-3-Opus, Claude-3-Sonnet, and Claude-3-Haiku) to answer questions written in Brazilian spoken portuguese from the medical residency entrance exam of the Hospital das Clínicas da Faculdade de Medicina da Universidade de São Paulo (HCFMUSP) - the largest health complex in South America. The performance of the models was benchmarked against human candidates, analyzing accuracy, processing time, and coherence of the generated explanations.
The results show that while some models, particularly Claude-3.5-Sonnet and Claude-3-Opus, achieved accuracy levels comparable to human candidates, performance gaps persist, particularly in multimodal questions requiring image interpretation. Furthermore, the study highlights language disparities, emphasizing the need for further fine-tuning and data set augmentation for non-English medical AI applications.
Our findings reinforce the importance of evaluating generative AI in various linguistic and clinical settings to ensure a fair and reliable deployment in healthcare. Future research should explore improved training methodologies, improved multimodal reasoning, and real-world clinical integration of AI-driven medical assistance. 

---
# The Polish Vocabulary Size Test: A Novel Adaptive Test for Receptive Vocabulary Assessment 

**Authors**: Danil Fokin, Monika Płużyczka, Grigory Golovin  

**Link**: [PDF](https://arxiv.org/pdf/2507.19869)  

**Abstract**: We present the Polish Vocabulary Size Test (PVST), a novel tool for assessing the receptive vocabulary size of both native and non-native Polish speakers. Based on Item Response Theory and Computerized Adaptive Testing, PVST dynamically adjusts to each test-taker's proficiency level, ensuring high accuracy while keeping the test duration short. To validate the test, a pilot study was conducted with 1.475 participants. Native Polish speakers demonstrated significantly larger vocabularies compared to non-native speakers. For native speakers, vocabulary size showed a strong positive correlation with age. The PVST is available online at this http URL. 

---
# DRIVE: Disfluency-Rich Synthetic Dialog Data Generation Framework for Intelligent Vehicle Environments 

**Authors**: Anshul Chavda, M Jagadeesh, Chintalapalli Raja Kullayappa, B Jayaprakash, Medchalimi Sruthi, Pushpak Bhattacharyya  

**Link**: [PDF](https://arxiv.org/pdf/2507.19867)  

**Abstract**: In-car conversational AI is becoming increasingly critical as autonomous vehicles and smart assistants gain widespread adoption. Yet, existing datasets fail to capture the spontaneous disfluencies such as hesitations, false starts, repetitions, and self-corrections that characterize real driver-AI dialogs. To address this, we introduce DiscoDrive, a synthetic corpus of 3500 multi-turn dialogs across seven automotive domains, generated using a two-stage, prompt-driven pipeline that dynamically integrates disfluencies during synthesis. We show that DiscoDrive is effective both as a training resource, enabling DialoGPT-Medium and T5-Base to match or exceed KVRET-trained models on the MultiWOZ 2.2 and Schema-Guided Dialogue (SGD) relevant test sets (BLEU-4 improvements of 0.26 to 0.61; METEOR +2.10; ROUGE-L +3.48; BERTScore F1 improvements of 1.35 to 3.48), and as a data augmentation resource in low-resource scenarios, delivering additional gains of up to BLEU-4 +0.38, METEOR +1.95, ROUGE-L +2.87, and BERTScore F1 +4.00 when combined with 10 percent of KVRET. Human evaluations further confirm that dialogs sampled from DiscoDrive are rated higher than KVRET's human-collected dialogs in naturalness (3.8 vs 3.6) and coherence (4.1 vs 4.0), and are perceived as more context-appropriate than leading post-hoc methods (such as LARD), without compromising clarity. DiscoDrive fills a critical gap in existing resources and serves as a versatile corpus for both training and augmenting conversational AI, enabling robust handling of real-world, disfluent in-car interactions. 

---
# HCAttention: Extreme KV Cache Compression via Heterogeneous Attention Computing for LLMs 

**Authors**: Dongquan Yang, Yifan Yang, Xiaotian Yu, Xianbiao Qi, Rong Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2507.19823)  

**Abstract**: Processing long-context inputs with large language models presents a significant challenge due to the enormous memory requirements of the Key-Value (KV) cache during inference. Existing KV cache compression methods exhibit noticeable performance degradation when memory is reduced by more than 85%. Additionally, strategies that leverage GPU-CPU collaboration for approximate attention remain underexplored in this setting. We propose HCAttention, a heterogeneous attention computation framework that integrates key quantization, value offloading, and dynamic KV eviction to enable efficient inference under extreme memory constraints. The method is compatible with existing transformer architectures and does not require model fine-tuning. Experimental results on the LongBench benchmark demonstrate that our approach preserves the accuracy of full-attention model while shrinking the KV cache memory footprint to 25% of its original size. Remarkably, it stays competitive with only 12.5% of the cache, setting a new state-of-the-art in LLM KV cache compression. To the best of our knowledge, HCAttention is the first to extend the Llama-3-8B model to process 4 million tokens on a single A100 GPU with 80GB memory. 

---
# Flora: Effortless Context Construction to Arbitrary Length and Scale 

**Authors**: Tianxiang Chen, Zhentao Tan, Xiaofan Bo, Yue Wu, Tao Gong, Qi Chu, Jieping Ye, Nenghai Yu  

**Link**: [PDF](https://arxiv.org/pdf/2507.19786)  

**Abstract**: Effectively handling long contexts is challenging for Large Language Models (LLMs) due to the rarity of long texts, high computational demands, and substantial forgetting of short-context abilities. Recent approaches have attempted to construct long contexts for instruction tuning, but these methods often require LLMs or human interventions, which are both costly and limited in length and diversity. Also, the drop in short-context performances of present long-context LLMs remains significant. In this paper, we introduce Flora, an effortless (human/LLM-free) long-context construction strategy. Flora can markedly enhance the long-context performance of LLMs by arbitrarily assembling short instructions based on categories and instructing LLMs to generate responses based on long-context meta-instructions. This enables Flora to produce contexts of arbitrary length and scale with rich diversity, while only slightly compromising short-context performance. Experiments on Llama3-8B-Instruct and QwQ-32B show that LLMs enhanced by Flora excel in three long-context benchmarks while maintaining strong performances in short-context tasks. Our data-construction code is available at \href{this https URL}{this https URL}. 

---
# UloRL:An Ultra-Long Output Reinforcement Learning Approach for Advancing Large Language Models' Reasoning Abilities 

**Authors**: Dong Du, Shulin Liu, Tao Yang, Shaohua Chen, Yang Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.19766)  

**Abstract**: Recent advances in large language models (LLMs) have highlighted the potential of reinforcement learning with verifiable rewards (RLVR) to enhance reasoning capabilities through extended output sequences. However, traditional RL frameworks face inefficiencies when handling ultra-long outputs due to long-tail sequence distributions and entropy collapse during training. To address these challenges, we propose an Ultra-Long Output Reinforcement Learning (UloRL) approach for advancing large language models' reasoning abilities. Specifically, we divide ultra long output decoding into short segments, enabling efficient training by mitigating delays caused by long-tail samples. Additionally, we introduce dynamic masking of well-Mastered Positive Tokens (MPTs) to prevent entropy collapse. Experimental results demonstrate the effectiveness of our approach. On the Qwen3-30B-A3B model, RL with segment rollout achieved 2.06x increase in training speed, while RL training with 128k-token outputs improves the model's performance on AIME2025 from 70.9\% to 85.1\% and on BeyondAIME from 50.7\% to 61.9\%, even surpassing Qwen3-235B-A22B with remarkable gains. These findings underscore the potential of our methods to advance the reasoning capabilities of LLMs with ultra-long sequence generation. We will release our code and model for further use by the community. 

---
# Are You There God? Lightweight Narrative Annotation of Christian Fiction with LMs 

**Authors**: Rebecca M. M. Hicke, Brian Haggard, Mia Ferrante, Rayhan Khanna, David Mimno  

**Link**: [PDF](https://arxiv.org/pdf/2507.19756)  

**Abstract**: In addition to its more widely studied political activities, the American Evangelical movement has a well-developed but less externally visible cultural and literary side. Christian Fiction, however, has been little studied, and what scholarly attention there is has focused on the explosively popular Left Behind series. In this work, we use computational tools to provide both a broad topical overview of Christian Fiction as a genre and a more directed exploration of how its authors depict divine acts. Working with human annotators we first developed definitions and a codebook for "acts of God." We then adapted those instructions designed for human annotators for use by a recent, lightweight LM with the assistance of a much larger model. The laptop-scale LM is capable of matching human annotations, even when the task is subtle and challenging. Using these annotations, we show that significant and meaningful differences exist between the Left Behind books and Christian Fiction more broadly and between books by male and female authors. 

---
# JT-Math: A Multi-Stage Framework for Advanced Mathematical Reasoning in Large Language Models 

**Authors**: Yifan Hao, Fangning Chao, Yaqian Hao, Zhaojun Cui, Huan Bai, Haiyu Zhang, Yankai Liu, Chao Deng, Junlan Feng  

**Link**: [PDF](https://arxiv.org/pdf/2507.19748)  

**Abstract**: Mathematical reasoning is a cornerstone of artificial general intelligence and a primary benchmark for evaluating the capabilities of Large Language Models (LLMs). While state-of-the-art models show promise, they often falter when faced with complex problems that demand deep conceptual understanding and intricate, multi-step deliberation. To address this challenge, we introduce JT-Math-8B, a series of open-source models comprising base, instruct, and thinking versions, built upon a systematic, multi-stage optimization framework. Our pre-training corpus is a high-quality, 210B-token dataset curated through a dedicated data pipeline that uses model-based validation to ensure quality and diversity. The Instruct Model is optimized for direct, concise answers through Supervised Fine-Tuning (SFT) and a GRPO-based reinforcement learning (RL) method. The Thinking Model is trained for complex problem-solving using a Long Chain-of-Thought (Long CoT) approach, combining SFT with a novel, multi-stage RL curriculum that progressively increases task difficulty and context length up to 32K tokens. JT-Math-8B achieves state-of-the-art results among open-source models of similar size, surpassing prominent models like OpenAI's O1-mini and GPT-4o , and demonstrating superior performance on competition-level mathematics. 

---
# Basic Reading Distillation 

**Authors**: Zhi Zhou, Sirui Miao, Xiangyu Duan, Hao Yang, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.19741)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable abilities in various natural language processing areas, but they demand high computation resources which limits their deployment in real-world. Distillation is one technique to solve this problem through either knowledge distillation or task distillation. Both distillation approaches train small models to imitate specific features of LLMs, but they all neglect basic reading education for small models on generic texts that are \emph{unrelated} to downstream tasks. In this paper, we propose basic reading distillation (BRD) which educates a small model to imitate LLMs basic reading behaviors, such as named entity recognition, question raising and answering, on each sentence. After such basic education, we apply the small model on various tasks including language inference benchmarks and BIG-bench tasks. It shows that the small model can outperform or perform comparable to over 20x bigger LLMs. Analysis reveals that BRD effectively influences the probability distribution of the small model, and has orthogonality to either knowledge distillation or task distillation. 

---
# Ta-G-T: Subjectivity Capture in Table to Text Generation via RDF Graphs 

**Authors**: Ronak Upasham, Tathagata Dey, Pushpak Bhattacharyya  

**Link**: [PDF](https://arxiv.org/pdf/2507.19710)  

**Abstract**: In Table-to-Text (T2T) generation, existing approaches predominantly focus on providing objective descriptions of tabular data. However, generating text that incorporates subjectivity, where subjectivity refers to interpretations beyond raw numerical data, remains underexplored. To address this, we introduce a novel pipeline that leverages intermediate representations to generate both objective and subjective text from tables. Our three-stage pipeline consists of: 1) extraction of Resource Description Framework (RDF) triples, 2) aggregation of text into coherent narratives, and 3) infusion of subjectivity to enrich the generated text. By incorporating RDFs, our approach enhances factual accuracy while maintaining interpretability. Unlike large language models (LLMs) such as GPT-3.5, Mistral-7B, and Llama-2, our pipeline employs smaller, fine-tuned T5 models while achieving comparable performance to GPT-3.5 and outperforming Mistral-7B and Llama-2 in several metrics. We evaluate our approach through quantitative and qualitative analyses, demonstrating its effectiveness in balancing factual accuracy with subjective interpretation. To the best of our knowledge, this is the first work to propose a structured pipeline for T2T generation that integrates intermediate representations to enhance both factual correctness and subjectivity. 

---
# Towards Inclusive NLP: Assessing Compressed Multilingual Transformers across Diverse Language Benchmarks 

**Authors**: Maitha Alshehhi, Ahmed Sharshar, Mohsen Guizani  

**Link**: [PDF](https://arxiv.org/pdf/2507.19699)  

**Abstract**: Although LLMs have attained significant success in high-resource languages, their capacity in low-resource linguistic environments like Kannada and Arabic is not yet fully understood. This work benchmarking the performance of multilingual and monolingual Large Language Models (LLMs) across Arabic, English, and Indic languages, with particular emphasis on the effects of model compression strategies such as pruning and quantization. Findings shows significant performance differences driven by linguistic diversity and resource availability on SOTA LLMS as BLOOMZ, AceGPT, Jais, LLaMA-2, XGLM, and AraGPT2. We find that multilingual versions of the model outperform their language-specific counterparts across the board, indicating substantial cross-lingual transfer benefits. Quantization (4-bit and 8-bit) is effective in maintaining model accuracy while promoting efficiency, but aggressive pruning significantly compromises performance, especially in bigger models. Our findings pinpoint key strategies to construct scalable and fair multilingual NLP solutions and underscore the need for interventions to address hallucination and generalization errors in the low-resource setting. 

---
# RoD-TAL: A Benchmark for Answering Questions in Romanian Driving License Exams 

**Authors**: Andrei Vlad Man, Răzvan-Alexandru Smădu, Cristian-George Craciun, Dumitru-Clementin Cercel, Florin Pop, Mihaela-Claudia Cercel  

**Link**: [PDF](https://arxiv.org/pdf/2507.19666)  

**Abstract**: The intersection of AI and legal systems presents a growing need for tools that support legal education, particularly in under-resourced languages such as Romanian. In this work, we aim to evaluate the capabilities of Large Language Models (LLMs) and Vision-Language Models (VLMs) in understanding and reasoning about Romanian driving law through textual and visual question-answering tasks. To facilitate this, we introduce RoD-TAL, a novel multimodal dataset comprising Romanian driving test questions, text-based and image-based, alongside annotated legal references and human explanations. We implement and assess retrieval-augmented generation (RAG) pipelines, dense retrievers, and reasoning-optimized models across tasks including Information Retrieval (IR), Question Answering (QA), Visual IR, and Visual QA. Our experiments demonstrate that domain-specific fine-tuning significantly enhances retrieval performance. At the same time, chain-of-thought prompting and specialized reasoning models improve QA accuracy, surpassing the minimum grades required to pass driving exams. However, visual reasoning remains challenging, highlighting the potential and the limitations of applying LLMs and VLMs to legal education. 

---
# MCIF: Multimodal Crosslingual Instruction-Following Benchmark from Scientific Talks 

**Authors**: Sara Papi, Maike Züfle, Marco Gaido, Beatrice Savoldi, Danni Liu, Ioannis Douros, Luisa Bentivogli, Jan Niehues  

**Link**: [PDF](https://arxiv.org/pdf/2507.19634)  

**Abstract**: Recent advances in large language models have catalyzed the development of multimodal LLMs (MLLMs) that integrate text, speech, and vision within unified frameworks. As MLLMs evolve from narrow, monolingual, task-specific systems to general-purpose instruction-following models, a key frontier lies in evaluating their multilingual and multimodal capabilities over both long and short contexts. However, existing benchmarks fall short in evaluating these dimensions jointly: they are often limited to English, mostly focus on one single modality at a time, rely on short-form contexts, or lack human annotations--hindering comprehensive assessment of model performance across languages, modalities, and task complexity. To address these gaps, we introduce MCIF (Multimodal Crosslingual Instruction Following), the first multilingual human-annotated benchmark based on scientific talks that is designed to evaluate instruction-following in crosslingual, multimodal settings over both short- and long-form inputs. MCIF spans three core modalities--speech, vision, and text--and four diverse languages (English, German, Italian, and Chinese), enabling a comprehensive evaluation of MLLMs' abilities to interpret instructions across languages and combine them with multimodal contextual information. MCIF is released under a CC-BY 4.0 license to encourage open research and progress in MLLMs development. 

---
# HITSZ's End-To-End Speech Translation Systems Combining Sequence-to-Sequence Auto Speech Recognition Model and Indic Large Language Model for IWSLT 2025 in Indic Track 

**Authors**: Xuchen Wei, Yangxin Wu, Yaoyin Zhang, Henglyu Liu, Kehai Chen, Xuefeng Bai, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.19616)  

**Abstract**: This paper presents HITSZ's submission for the IWSLT 2025 Indic track, focusing on speech-to-text translation (ST) for English-to-Indic and Indic-to-English language pairs. To enhance translation quality in this low-resource scenario, we propose an end-to-end system integrating the pre-trained Whisper automated speech recognition (ASR) model with Krutrim, an Indic-specialized large language model (LLM). Experimental results demonstrate that our end-to-end system achieved average BLEU scores of $28.88$ for English-to-Indic directions and $27.86$ for Indic-to-English directions. Furthermore, we investigated the Chain-of-Thought (CoT) method. While this method showed potential for significant translation quality improvements on successfully parsed outputs (e.g. a $13.84$ BLEU increase for Tamil-to-English), we observed challenges in ensuring the model consistently adheres to the required CoT output format. 

---
# MOCHA: Are Code Language Models Robust Against Multi-Turn Malicious Coding Prompts? 

**Authors**: Muntasir Wahed, Xiaona Zhou, Kiet A. Nguyen, Tianjiao Yu, Nirav Diwan, Gang Wang, Dilek Hakkani-Tür, Ismini Lourentzou  

**Link**: [PDF](https://arxiv.org/pdf/2507.19598)  

**Abstract**: Recent advancements in Large Language Models (LLMs) have significantly enhanced their code generation capabilities. However, their robustness against adversarial misuse, particularly through multi-turn malicious coding prompts, remains underexplored. In this work, we introduce code decomposition attacks, where a malicious coding task is broken down into a series of seemingly benign subtasks across multiple conversational turns to evade safety filters. To facilitate systematic evaluation, we introduce \benchmarkname{}, a large-scale benchmark designed to evaluate the robustness of code LLMs against both single-turn and multi-turn malicious prompts. Empirical results across open- and closed-source models reveal persistent vulnerabilities, especially under multi-turn scenarios. Fine-tuning on MOCHA improves rejection rates while preserving coding ability, and importantly, enhances robustness on external adversarial datasets with up to 32.4% increase in rejection rates without any additional supervision. 

---
# Efficient Attention Mechanisms for Large Language Models: A Survey 

**Authors**: Yutao Sun, Zhenyu Li, Yike Zhang, Tengyu Pan, Bowen Dong, Yuyi Guo, Jianyong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.19595)  

**Abstract**: Transformer-based architectures have become the prevailing backbone of large language models. However, the quadratic time and memory complexity of self-attention remains a fundamental obstacle to efficient long-context modeling. To address this limitation, recent research has introduced two principal categories of efficient attention mechanisms. Linear attention methods achieve linear complexity through kernel approximations, recurrent formulations, or fastweight dynamics, thereby enabling scalable inference with reduced computational overhead. Sparse attention techniques, in contrast, limit attention computation to selected subsets of tokens based on fixed patterns, block-wise routing, or clustering strategies, enhancing efficiency while preserving contextual coverage. This survey provides a systematic and comprehensive overview of these developments, integrating both algorithmic innovations and hardware-level considerations. In addition, we analyze the incorporation of efficient attention into largescale pre-trained language models, including both architectures built entirely on efficient attention and hybrid designs that combine local and global components. By aligning theoretical foundations with practical deployment strategies, this work aims to serve as a foundational reference for advancing the design of scalable and efficient language models. 

---
# Mitigating Geospatial Knowledge Hallucination in Large Language Models: Benchmarking and Dynamic Factuality Aligning 

**Authors**: Shengyuan Wang, Jie Feng, Tianhui Liu, Dan Pei, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.19586)  

**Abstract**: Large language models (LLMs) possess extensive world knowledge, including geospatial knowledge, which has been successfully applied to various geospatial tasks such as mobility prediction and social indicator prediction. However, LLMs often generate inaccurate geospatial knowledge, leading to geospatial hallucinations (incorrect or inconsistent representations of geospatial information) that compromise their reliability. While the phenomenon of general knowledge hallucination in LLMs has been widely studied, the systematic evaluation and mitigation of geospatial hallucinations remain largely unexplored. To address this gap, we propose a comprehensive evaluation framework for geospatial hallucinations, leveraging structured geospatial knowledge graphs for controlled assessment. Through extensive evaluation across 20 advanced LLMs, we uncover the hallucinations in their geospatial knowledge. Building on these insights, we introduce a dynamic factuality aligning method based on Kahneman-Tversky Optimization (KTO) to mitigate geospatial hallucinations in LLMs, leading to a performance improvement of over 29.6% on the proposed benchmark. Extensive experimental results demonstrate the effectiveness of our benchmark and learning algorithm in enhancing the trustworthiness of LLMs in geospatial knowledge and reasoning tasks. 

---
# Mind the Language Gap in Digital Humanities: LLM-Aided Translation of SKOS Thesauri 

**Authors**: Felix Kraus, Nicolas Blumenröhr, Danah Tonne, Achim Streit  

**Link**: [PDF](https://arxiv.org/pdf/2507.19537)  

**Abstract**: We introduce WOKIE, an open-source, modular, and ready-to-use pipeline for the automated translation of SKOS thesauri. This work addresses a critical need in the Digital Humanities (DH), where language diversity can limit access, reuse, and semantic interoperability of knowledge resources. WOKIE combines external translation services with targeted refinement using Large Language Models (LLMs), balancing translation quality, scalability, and cost. Designed to run on everyday hardware and be easily extended, the application requires no prior expertise in machine translation or LLMs. We evaluate WOKIE across several DH thesauri in 15 languages with different parameters, translation services and LLMs, systematically analysing translation quality, performance, and ontology matching improvements. Our results show that WOKIE is suitable to enhance the accessibility, reuse, and cross-lingual interoperability of thesauri by hurdle-free automated translation and improved ontology matching performance, supporting more inclusive and multilingual research infrastructures. 

---
# Setting The Table with Intent: Intent-aware Schema Generation and Editing for Literature Review Tables 

**Authors**: Vishakh Padmakumar, Joseph Chee Chang, Kyle Lo, Doug Downey, Aakanksha Naik  

**Link**: [PDF](https://arxiv.org/pdf/2507.19521)  

**Abstract**: The increasing volume of academic literature makes it essential for researchers to organize, compare, and contrast collections of documents. Large language models (LLMs) can support this process by generating schemas defining shared aspects along which to compare papers. However, progress on schema generation has been slow due to: (i) ambiguity in reference-based evaluations, and (ii) lack of editing/refinement methods. Our work is the first to address both issues. First, we present an approach for augmenting unannotated table corpora with synthesized intents and apply it to create a dataset for studying schema generation conditioned on a given information need, thus reducing ambiguity. With this dataset, we show how incorporating table intents significantly improves baseline performance in reconstructing reference schemas. Next, we propose several LLM-based schema editing techniques. We start by comprehensively benchmarking several single-shot schema generation methods, including prompted LLM workflows and fine-tuned models, showing that smaller, open-weight models can be fine-tuned to be competitive with state-of-the-art prompted LLMs. Then we demonstrate that our editing techniques can further improve schemas generated by these methods. 

---
# Advancing Mental Disorder Detection: A Comparative Evaluation of Transformer and LSTM Architectures on Social Media 

**Authors**: Khalid Hasan, Jamil Saquer, Mukulika Ghosh  

**Link**: [PDF](https://arxiv.org/pdf/2507.19511)  

**Abstract**: The rising prevalence of mental health disorders necessitates the development of robust, automated tools for early detection and monitoring. Recent advances in Natural Language Processing (NLP), particularly transformer-based architectures, have demonstrated significant potential in text analysis. This study provides a comprehensive evaluation of state-of-the-art transformer models (BERT, RoBERTa, DistilBERT, ALBERT, and ELECTRA) against Long Short-Term Memory (LSTM) based approaches using different text embedding techniques for mental health disorder classification on Reddit. We construct a large annotated dataset, validating its reliability through statistical judgmental analysis and topic modeling. Experimental results demonstrate the superior performance of transformer models over traditional deep-learning approaches. RoBERTa achieved the highest classification performance, with a 99.54% F1 score on the hold-out test set and a 96.05% F1 score on the external test set. Notably, LSTM models augmented with BERT embeddings proved highly competitive, achieving F1 scores exceeding 94% on the external dataset while requiring significantly fewer computational resources. These findings highlight the effectiveness of transformer-based models for real-time, scalable mental health monitoring. We discuss the implications for clinical applications and digital mental health interventions, offering insights into the capabilities and limitations of state-of-the-art NLP methodologies in mental disorder detection. 

---
# LoRA-PAR: A Flexible Dual-System LoRA Partitioning Approach to Efficient LLM Fine-Tuning 

**Authors**: Yining Huang, Bin Li, Keke Tang, Meilian Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.20999)  

**Abstract**: Large-scale generative models like DeepSeek-R1 and OpenAI-O1 benefit substantially from chain-of-thought (CoT) reasoning, yet pushing their performance typically requires vast data, large model sizes, and full-parameter fine-tuning. While parameter-efficient fine-tuning (PEFT) helps reduce cost, most existing approaches primarily address domain adaptation or layer-wise allocation rather than explicitly tailoring data and parameters to different response demands. Inspired by "Thinking, Fast and Slow," which characterizes two distinct modes of thought-System 1 (fast, intuitive, often automatic) and System 2 (slower, more deliberative and analytic)-we draw an analogy that different "subregions" of an LLM's parameters might similarly specialize for tasks that demand quick, intuitive responses versus those requiring multi-step logical reasoning. Therefore, we propose LoRA-PAR, a dual-system LoRA framework that partitions both data and parameters by System 1 or System 2 demands, using fewer yet more focused parameters for each task. Specifically, we classify task data via multi-model role-playing and voting, and partition parameters based on importance scoring, then adopt a two-stage fine-tuning strategy of training System 1 tasks with supervised fine-tuning (SFT) to enhance knowledge and intuition and refine System 2 tasks with reinforcement learning (RL) to reinforce deeper logical deliberation next. Extensive experiments show that the two-stage fine-tuning strategy, SFT and RL, lowers active parameter usage while matching or surpassing SOTA PEFT baselines. 

---
# Your AI, Not Your View: The Bias of LLMs in Investment Analysis 

**Authors**: Hoyoung Lee, Junhyuk Seo, Suhwan Park, Junhyeong Lee, Wonbin Ahn, Chanyeol Choi, Alejandro Lopez-Lira, Yongjae Lee  

**Link**: [PDF](https://arxiv.org/pdf/2507.20957)  

**Abstract**: In finance, Large Language Models (LLMs) face frequent knowledge conflicts due to discrepancies between pre-trained parametric knowledge and real-time market data. These conflicts become particularly problematic when LLMs are deployed in real-world investment services, where misalignment between a model's embedded preferences and those of the financial institution can lead to unreliable recommendations. Yet little research has examined what investment views LLMs actually hold. We propose an experimental framework to investigate such conflicts, offering the first quantitative analysis of confirmation bias in LLM-based investment analysis. Using hypothetical scenarios with balanced and imbalanced arguments, we extract models' latent preferences and measure their persistence. Focusing on sector, size, and momentum, our analysis reveals distinct, model-specific tendencies. In particular, we observe a consistent preference for large-cap stocks and contrarian strategies across most models. These preferences often harden into confirmation bias, with models clinging to initial judgments despite counter-evidence. 

---
# Dissecting Persona-Driven Reasoning in Language Models via Activation Patching 

**Authors**: Ansh Poonia, Maeghal Jain  

**Link**: [PDF](https://arxiv.org/pdf/2507.20936)  

**Abstract**: Large language models (LLMs) exhibit remarkable versatility in adopting diverse personas. In this study, we examine how assigning a persona influences a model's reasoning on an objective task. Using activation patching, we take a first step toward understanding how key components of the model encode persona-specific information. Our findings reveal that the early Multi-Layer Perceptron (MLP) layers attend not only to the syntactic structure of the input but also process its semantic content. These layers transform persona tokens into richer representations, which are then used by the middle Multi-Head Attention (MHA) layers to shape the model's output. Additionally, we identify specific attention heads that disproportionately attend to racial and color-based identities. 

---
# $A^2R^2$: Advancing Img2LaTeX Conversion via Visual Reasoning with Attention-Guided Refinement 

**Authors**: Zhecheng Li, Guoxian Song, Yiwei Wang, Zhen Xiong, Junsong Yuan, Yujun Cai  

**Link**: [PDF](https://arxiv.org/pdf/2507.20890)  

**Abstract**: Img2LaTeX is a practically significant task that involves converting mathematical expressions or tabular data from images into LaTeX code. In recent years, vision-language models (VLMs) have demonstrated strong performance across a variety of visual understanding tasks, owing to their generalization capabilities. While some studies have explored the use of VLMs for the Img2LaTeX task, their performance often falls short of expectations. Empirically, VLMs sometimes struggle with fine-grained visual elements, leading to inaccurate LaTeX predictions. To address this challenge, we propose $A^2R^2$: Advancing Img2LaTeX Conversion via Visual Reasoning with Attention-Guided Refinement, a framework that effectively integrates attention localization and iterative refinement within a visual reasoning framework, enabling VLMs to perform self-correction and progressively improve prediction quality. For effective evaluation, we introduce a new dataset, Img2LaTex-Hard-1K, consisting of 1,100 carefully curated and challenging examples designed to rigorously evaluate the capabilities of VLMs within this task domain. Extensive experimental results demonstrate that: (1) $A^2R^2$ significantly improves model performance across six evaluation metrics spanning both textual and visual levels, consistently outperforming other baseline methods; (2) Increasing the number of inference rounds yields notable performance gains, underscoring the potential of $A^2R^2$ in test-time scaling scenarios; (3) Ablation studies and human evaluations validate the practical effectiveness of our approach, as well as the strong synergy among its core components during inference. 

---
# Enhancing Project-Specific Code Completion by Inferring Internal API Information 

**Authors**: Le Deng, Xiaoxue Ren, Chao Ni, Ming Liang, David Lo, Zhongxin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.20888)  

**Abstract**: Project-specific code completion is a critical task that leverages context from a project to generate accurate code. State-of-the-art methods use retrieval-augmented generation (RAG) with large language models (LLMs) and project information for code completion. However, they often struggle to incorporate internal API information, which is crucial for accuracy, especially when APIs are not explicitly imported in the file.
To address this, we propose a method to infer internal API information without relying on imports. Our method extends the representation of APIs by constructing usage examples and semantic descriptions, building a knowledge base for LLMs to generate relevant completions. We also introduce ProjBench, a benchmark that avoids leaked imports and consists of large-scale real-world projects.
Experiments on ProjBench and CrossCodeEval show that our approach significantly outperforms existing methods, improving code exact match by 22.72% and identifier exact match by 18.31%. Additionally, integrating our method with existing baselines boosts code match by 47.80% and identifier match by 35.55%. 

---
# The Importance of Facial Features in Vision-based Sign Language Recognition: Eyes, Mouth or Full Face? 

**Authors**: Dinh Nam Pham, Eleftherios Avramidis  

**Link**: [PDF](https://arxiv.org/pdf/2507.20884)  

**Abstract**: Non-manual facial features play a crucial role in sign language communication, yet their importance in automatic sign language recognition (ASLR) remains underexplored. While prior studies have shown that incorporating facial features can improve recognition, related work often relies on hand-crafted feature extraction and fails to go beyond the comparison of manual features versus the combination of manual and facial features. In this work, we systematically investigate the contribution of distinct facial regionseyes, mouth, and full faceusing two different deep learning models (a CNN-based model and a transformer-based model) trained on an SLR dataset of isolated signs with randomly selected classes. Through quantitative performance and qualitative saliency map evaluation, we reveal that the mouth is the most important non-manual facial feature, significantly improving accuracy. Our findings highlight the necessity of incorporating facial features in ASLR. 

---
# Kimi K2: Open Agentic Intelligence 

**Authors**: Kimi Team, Yifan Bai, Yiping Bao, Guanduo Chen, Jiahao Chen, Ningxin Chen, Ruijue Chen, Yanru Chen, Yuankun Chen, Yutian Chen, Zhuofu Chen, Jialei Cui, Hao Ding, Mengnan Dong, Angang Du, Chenzhuang Du, Dikang Du, Yulun Du, Yu Fan, Yichen Feng, Kelin Fu, Bofei Gao, Hongcheng Gao, Peizhong Gao, Tong Gao, Xinran Gu, Longyu Guan, Haiqing Guo, Jianhang Guo, Hao Hu, Xiaoru Hao, Tianhong He, Weiran He, Wenyang He, Chao Hong, Yangyang Hu, Zhenxing Hu, Weixiao Huang, Zhiqi Huang, Zihao Huang, Tao Jiang, Zhejun Jiang, Xinyi Jin, Yongsheng Kang, Guokun Lai, Cheng Li, Fang Li, Haoyang Li, Ming Li, Wentao Li, Yanhao Li, Yiwei Li, Zhaowei Li, Zheming Li, Hongzhan Lin, Xiaohan Lin, Zongyu Lin, Chengyin Liu, Chenyu Liu, Hongzhang Liu, Jingyuan Liu, Junqi Liu, Liang Liu, Shaowei Liu, T.Y. Liu, Tianwei Liu, Weizhou Liu, Yangyang Liu, Yibo Liu, Yiping Liu, Yue Liu, Zhengying Liu, Enzhe Lu, Lijun Lu, Shengling Ma, Xinyu Ma, Yingwei Ma, Shaoguang Mao, Jie Mei, Xin Men, Yibo Miao, Siyuan Pan, Yebo Peng, Ruoyu Qin, Bowen Qu, Zeyu Shang, Lidong Shi, Shengyuan Shi, Feifan Song, Jianlin Su, Zhengyuan Su, Xinjie Sun, Flood Sung, Heyi Tang, Jiawen Tao, Qifeng Teng, Chensi Wang, Dinglu Wang, Feng Wang, Haiming Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.20534)  

**Abstract**: We introduce Kimi K2, a Mixture-of-Experts (MoE) large language model with 32 billion activated parameters and 1 trillion total parameters. We propose the MuonClip optimizer, which improves upon Muon with a novel QK-clip technique to address training instability while enjoying the advanced token efficiency of Muon. Based on MuonClip, K2 was pre-trained on 15.5 trillion tokens with zero loss spike. During post-training, K2 undergoes a multi-stage post-training process, highlighted by a large-scale agentic data synthesis pipeline and a joint reinforcement learning (RL) stage, where the model improves its capabilities through interactions with real and synthetic environments.
Kimi K2 achieves state-of-the-art performance among open-source non-thinking models, with strengths in agentic capabilities. Notably, K2 obtains 66.1 on Tau2-Bench, 76.5 on ACEBench (En), 65.8 on SWE-Bench Verified, and 47.3 on SWE-Bench Multilingual -- surpassing most open and closed-sourced baselines in non-thinking settings. It also exhibits strong capabilities in coding, mathematics, and reasoning tasks, with a score of 53.7 on LiveCodeBench v6, 49.5 on AIME 2025, 75.1 on GPQA-Diamond, and 27.1 on OJBench, all without extended thinking. These results position Kimi K2 as one of the most capable open-source large language models to date, particularly in software engineering and agentic tasks. We release our base and post-trained model checkpoints to facilitate future research and applications of agentic intelligence. 

---
# Security Challenges in AI Agent Deployment: Insights from a Large Scale Public Competition 

**Authors**: Andy Zou, Maxwell Lin, Eliot Jones, Micha Nowak, Mateusz Dziemian, Nick Winter, Alexander Grattan, Valent Nathanael, Ayla Croft, Xander Davies, Jai Patel, Robert Kirk, Nate Burnikell, Yarin Gal, Dan Hendrycks, J. Zico Kolter, Matt Fredrikson  

**Link**: [PDF](https://arxiv.org/pdf/2507.20526)  

**Abstract**: Recent advances have enabled LLM-powered AI agents to autonomously execute complex tasks by combining language model reasoning with tools, memory, and web access. But can these systems be trusted to follow deployment policies in realistic environments, especially under attack? To investigate, we ran the largest public red-teaming competition to date, targeting 22 frontier AI agents across 44 realistic deployment scenarios. Participants submitted 1.8 million prompt-injection attacks, with over 60,000 successfully eliciting policy violations such as unauthorized data access, illicit financial actions, and regulatory noncompliance. We use these results to build the Agent Red Teaming (ART) benchmark - a curated set of high-impact attacks - and evaluate it across 19 state-of-the-art models. Nearly all agents exhibit policy violations for most behaviors within 10-100 queries, with high attack transferability across models and tasks. Importantly, we find limited correlation between agent robustness and model size, capability, or inference-time compute, suggesting that additional defenses are needed against adversarial misuse. Our findings highlight critical and persistent vulnerabilities in today's AI agents. By releasing the ART benchmark and accompanying evaluation framework, we aim to support more rigorous security assessment and drive progress toward safer agent deployment. 

---
# Customize Multi-modal RAI Guardrails with Precedent-based predictions 

**Authors**: Cheng-Fu Yang, Thanh Tran, Christos Christodoulopoulos, Weitong Ruan, Rahul Gupta, Kai-Wei Chang  

**Link**: [PDF](https://arxiv.org/pdf/2507.20503)  

**Abstract**: A multi-modal guardrail must effectively filter image content based on user-defined policies, identifying material that may be hateful, reinforce harmful stereotypes, contain explicit material, or spread misinformation. Deploying such guardrails in real-world applications, however, poses significant challenges. Users often require varied and highly customizable policies and typically cannot provide abundant examples for each custom policy. Consequently, an ideal guardrail should be scalable to the multiple policies and adaptable to evolving user standards with minimal retraining. Existing fine-tuning methods typically condition predictions on pre-defined policies, restricting their generalizability to new policies or necessitating extensive retraining to adapt. Conversely, training-free methods struggle with limited context lengths, making it difficult to incorporate all the policies comprehensively. To overcome these limitations, we propose to condition model's judgment on "precedents", which are the reasoning processes of prior data points similar to the given input. By leveraging precedents instead of fixed policies, our approach greatly enhances the flexibility and adaptability of the guardrail. In this paper, we introduce a critique-revise mechanism for collecting high-quality precedents and two strategies that utilize precedents for robust prediction. Experimental results demonstrate that our approach outperforms previous methods across both few-shot and full-dataset scenarios and exhibits superior generalization to novel policies. 

---
# MountainLion: A Multi-Modal LLM-Based Agent System for Interpretable and Adaptive Financial Trading 

**Authors**: Siyi Wu, Zhaoyang Guan, Leyi Zhao, Xinyuan Song, Xinyu Ying, Hanlin Zhang, Michele Pak, Yangfan He, Yi Xin, Jianhui Wang, Tianyu Shi  

**Link**: [PDF](https://arxiv.org/pdf/2507.20474)  

**Abstract**: Cryptocurrency trading is a challenging task requiring the integration of heterogeneous data from multiple modalities. Traditional deep learning and reinforcement learning approaches typically demand large training datasets and encode diverse inputs into numerical representations, often at the cost of interpretability. Recent progress in large language model (LLM)-based agents has demonstrated the capacity to process multi-modal data and support complex investment decision-making. Building on these advances, we present \textbf{MountainLion}, a multi-modal, multi-agent system for financial trading that coordinates specialized LLM-based agents to interpret financial data and generate investment strategies. MountainLion processes textual news, candlestick charts, and trading signal charts to produce high-quality financial reports, while also enabling modification of reports and investment recommendations through data-driven user interaction and question answering. A central reflection module analyzes historical trading signals and outcomes to continuously refine decision processes, and the system is capable of real-time report analysis, summarization, and dynamic adjustment of investment strategies. Empirical results confirm that MountainLion systematically enriches technical price triggers with contextual macroeconomic and capital flow signals, providing a more interpretable, robust, and actionable investment framework that improves returns and strengthens investor confidence. 

---
# SciToolAgent: A Knowledge Graph-Driven Scientific Agent for Multi-Tool Integration 

**Authors**: Keyan Ding, Jing Yu, Junjie Huang, Yuchen Yang, Qiang Zhang, Huajun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.20280)  

**Abstract**: Scientific research increasingly relies on specialized computational tools, yet effectively utilizing these tools demands substantial domain expertise. While Large Language Models (LLMs) show promise in tool automation, they struggle to seamlessly integrate and orchestrate multiple tools for complex scientific workflows. Here, we present SciToolAgent, an LLM-powered agent that automates hundreds of scientific tools across biology, chemistry, and materials science. At its core, SciToolAgent leverages a scientific tool knowledge graph that enables intelligent tool selection and execution through graph-based retrieval-augmented generation. The agent also incorporates a comprehensive safety-checking module to ensure responsible and ethical tool usage. Extensive evaluations on a curated benchmark demonstrate that SciToolAgent significantly outperforms existing approaches. Case studies in protein engineering, chemical reactivity prediction, chemical synthesis, and metal-organic framework screening further demonstrate SciToolAgent's capability to automate complex scientific workflows, making advanced research tools accessible to both experts and non-experts. 

---
# The Policy Cliff: A Theoretical Analysis of Reward-Policy Maps in Large Language Models 

**Authors**: Xingcheng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2507.20150)  

**Abstract**: Reinforcement learning (RL) plays a crucial role in shaping the behavior of large language and reasoning models (LLMs/LRMs). However, it often produces brittle and unstable policies, leading to critical failures such as spurious reasoning, deceptive alignment, and instruction disobedience that undermine the trustworthiness and safety of LLMs/LRMs. Currently, these issues lack a unified theoretical explanation and are typically addressed using ad-hoc heuristics. This paper presents a rigorous mathematical framework for analyzing the stability of the mapping from a reward function to the optimal policy. We show that policy brittleness often stems from non-unique optimal actions, a common occurrence when multiple valid traces exist in a reasoning task. This theoretical lens provides a unified explanation for a range of seemingly disparate failures, reframing them as rational outcomes of optimizing rewards that may be incomplete or noisy, especially in the presence of action degeneracy. We extend this analysis from the fundamental single-reward setting to the more realistic multi-reward RL across diverse domains, showing how stability is governed by an "effective reward" aggregation mechanism. We also prove that entropy regularization restores policy stability at the cost of increased stochasticity. Our framework provides a unified explanation for recent empirical findings on deceptive reasoning, instruction-following trade-offs, and RLHF-induced sophistry, and is further validated through perturbation experiments in multi-reward RL. This work advances policy-stability analysis from empirical heuristics towards a principled theory, offering essential insights for designing safer and more trustworthy AI systems. 

---
# EcoTransformer: Attention without Multiplication 

**Authors**: Xin Gao, Xingming Xu  

**Link**: [PDF](https://arxiv.org/pdf/2507.20096)  

**Abstract**: The Transformer, with its scaled dot-product attention mechanism, has become a foundational architecture in modern AI. However, this mechanism is computationally intensive and incurs substantial energy costs. We propose a new Transformer architecture EcoTransformer, in which the output context vector is constructed as the convolution of the values using a Laplacian kernel, where the distances are measured by the L1 metric between the queries and keys. Compared to dot-product based attention, the new attention score calculation is free of matrix multiplication. It performs on par with, or even surpasses, scaled dot-product attention in NLP, bioinformatics, and vision tasks, while consuming significantly less energy. 

---
# The Devil is in the EOS: Sequence Training for Detailed Image Captioning 

**Authors**: Abdelrahman Mohamed, Yova Kementchedjhieva  

**Link**: [PDF](https://arxiv.org/pdf/2507.20077)  

**Abstract**: Despite significant advances in vision-language models (VLMs), image captioning often suffers from a lack of detail, with base models producing short, generic captions. This limitation persists even though VLMs are equipped with strong vision and language backbones. While supervised data and complex reward functions have been proposed to improve detailed image captioning, we identify a simpler underlying issue: a bias towards the end-of-sequence (EOS) token, which is introduced during cross-entropy training. We propose an unsupervised method to debias the model's tendency to predict the EOS token prematurely. By reducing this bias, we encourage the generation of longer, more detailed captions without the need for intricate reward functions or supervision. Our approach is straightforward, effective, and easily applicable to any pretrained model. We demonstrate its effectiveness through experiments with three VLMs and on three detailed captioning benchmarks. Our results show a substantial increase in caption length and relevant details, albeit with an expected increase in the rate of hallucinations. 

---
# PITA: Preference-Guided Inference-Time Alignment for LLM Post-Training 

**Authors**: Sarat Chandra Bobbili, Ujwal Dinesha, Dheeraj Narasimha, Srinivas Shakkottai  

**Link**: [PDF](https://arxiv.org/pdf/2507.20067)  

**Abstract**: Inference-time alignment enables large language models (LLMs) to generate outputs aligned with end-user preferences without further training. Recent post-training methods achieve this by using small guidance models to modify token generation during inference. These methods typically optimize a reward function KL-regularized by the original LLM taken as the reference policy. A critical limitation, however, is their dependence on a pre-trained reward model, which requires fitting to human preference feedback--a potentially unstable process. In contrast, we introduce PITA, a novel framework that integrates preference feedback directly into the LLM's token generation, eliminating the need for a reward model. PITA learns a small preference-based guidance policy to modify token probabilities at inference time without LLM fine-tuning, reducing computational cost and bypassing the pre-trained reward model dependency. The problem is framed as identifying an underlying preference distribution, solved through stochastic search and iterative refinement of the preference-based guidance model. We evaluate PITA across diverse tasks, including mathematical reasoning and sentiment classification, demonstrating its effectiveness in aligning LLM outputs with user preferences. 

---
# $K^4$: Online Log Anomaly Detection Via Unsupervised Typicality Learning 

**Authors**: Weicong Chen, Vikash Singh, Zahra Rahmani, Debargha Ganguly, Mohsen Hariri, Vipin Chaudhary  

**Link**: [PDF](https://arxiv.org/pdf/2507.20051)  

**Abstract**: Existing Log Anomaly Detection (LogAD) methods are often slow, dependent on error-prone parsing, and use unrealistic evaluation protocols. We introduce $K^4$, an unsupervised and parser-independent framework for high-performance online detection. $K^4$ transforms arbitrary log embeddings into compact four-dimensional descriptors (Precision, Recall, Density, Coverage) using efficient k-nearest neighbor (k-NN) statistics. These descriptors enable lightweight detectors to accurately score anomalies without retraining. Using a more realistic online evaluation protocol, $K^4$ sets a new state-of-the-art (AUROC: 0.995-0.999), outperforming baselines by large margins while being orders of magnitude faster, with training under 4 seconds and inference as low as 4 $\mu$s. 

---
# The Carbon Cost of Conversation, Sustainability in the Age of Language Models 

**Authors**: Sayed Mahbub Hasan Amiri, Prasun Goswami, Md. Mainul Islam, Mohammad Shakhawat Hossen, Sayed Majhab Hasan Amiri, Naznin Akter  

**Link**: [PDF](https://arxiv.org/pdf/2507.20018)  

**Abstract**: Large language models (LLMs) like GPT-3 and BERT have revolutionized natural language processing (NLP), yet their environmental costs remain dangerously overlooked. This article critiques the sustainability of LLMs, quantifying their carbon footprint, water usage, and contribution to e-waste through case studies of models such as GPT-4 and energy-efficient alternatives like Mistral 7B. Training a single LLM can emit carbon dioxide equivalent to hundreds of cars driven annually, while data centre cooling exacerbates water scarcity in vulnerable regions. Systemic challenges corporate greenwashing, redundant model development, and regulatory voids perpetuate harm, disproportionately burdening marginalized communities in the Global South. However, pathways exist for sustainable NLP: technical innovations (e.g., model pruning, quantum computing), policy reforms (carbon taxes, mandatory emissions reporting), and cultural shifts prioritizing necessity over novelty. By analysing industry leaders (Google, Microsoft) and laggards (Amazon), this work underscores the urgency of ethical accountability and global cooperation. Without immediate action, AIs ecological toll risks outpacing its societal benefits. The article concludes with a call to align technological progress with planetary boundaries, advocating for equitable, transparent, and regenerative AI systems that prioritize both human and environmental well-being. 

---
# Improving the Performance of Sequential Recommendation Systems with an Extended Large Language Model 

**Authors**: Sinnyum Choi, Woong Kim  

**Link**: [PDF](https://arxiv.org/pdf/2507.19990)  

**Abstract**: Recently, competition in the field of artificial intelligence (AI) has intensified among major technological companies, resulting in the continuous release of new large-language models (LLMs) that exhibit improved language understanding and context-based reasoning capabilities. It is expected that these advances will enable more efficient personalized recommendations in LLM-based recommendation systems through improved quality of training data and architectural design. However, many studies have not considered these recent developments. In this study, it was proposed to improve LLM-based recommendation systems by replacing Llama2 with Llama3 in the LlamaRec framework. To ensure a fair comparison, random seed values were set and identical input data was provided during preprocessing and training. The experimental results show average performance improvements of 38.65\%, 8.69\%, and 8.19\% for the ML-100K, Beauty, and Games datasets, respectively, thus confirming the practicality of this method. Notably, the significant improvements achieved by model replacement indicate that the recommendation quality can be improved cost-effectively without the need to make structural changes to the system. Based on these results, it is our contention that the proposed approach is a viable solution for improving the performance of current recommendation systems. 

---
# Leveraging Fine-Tuned Large Language Models for Interpretable Pancreatic Cystic Lesion Feature Extraction and Risk Categorization 

**Authors**: Ebrahim Rasromani, Stella K. Kang, Yanqi Xu, Beisong Liu, Garvit Luhadia, Wan Fung Chui, Felicia L. Pasadyn, Yu Chih Hung, Julie Y. An, Edwin Mathieu, Zehui Gu, Carlos Fernandez-Granda, Ammar A. Javed, Greg D. Sacks, Tamas Gonda, Chenchan Huang, Yiqiu Shen  

**Link**: [PDF](https://arxiv.org/pdf/2507.19973)  

**Abstract**: Background: Manual extraction of pancreatic cystic lesion (PCL) features from radiology reports is labor-intensive, limiting large-scale studies needed to advance PCL research. Purpose: To develop and evaluate large language models (LLMs) that automatically extract PCL features from MRI/CT reports and assign risk categories based on guidelines. Materials and Methods: We curated a training dataset of 6,000 abdominal MRI/CT reports (2005-2024) from 5,134 patients that described PCLs. Labels were generated by GPT-4o using chain-of-thought (CoT) prompting to extract PCL and main pancreatic duct features. Two open-source LLMs were fine-tuned using QLoRA on GPT-4o-generated CoT data. Features were mapped to risk categories per institutional guideline based on the 2017 ACR White Paper. Evaluation was performed on 285 held-out human-annotated reports. Model outputs for 100 cases were independently reviewed by three radiologists. Feature extraction was evaluated using exact match accuracy, risk categorization with macro-averaged F1 score, and radiologist-model agreement with Fleiss' Kappa. Results: CoT fine-tuning improved feature extraction accuracy for LLaMA (80% to 97%) and DeepSeek (79% to 98%), matching GPT-4o (97%). Risk categorization F1 scores also improved (LLaMA: 0.95; DeepSeek: 0.94), closely matching GPT-4o (0.97), with no statistically significant differences. Radiologist inter-reader agreement was high (Fleiss' Kappa = 0.888) and showed no statistically significant difference with the addition of DeepSeek-FT-CoT (Fleiss' Kappa = 0.893) or GPT-CoT (Fleiss' Kappa = 0.897), indicating that both models achieved agreement levels on par with radiologists. Conclusion: Fine-tuned open-source LLMs with CoT supervision enable accurate, interpretable, and efficient phenotyping for large-scale PCL research, achieving performance comparable to GPT-4o. 

---
# Spatial Language Likelihood Grounding Network for Bayesian Fusion of Human-Robot Observations 

**Authors**: Supawich Sitdhipol, Waritwong Sukprasongdee, Ekapol Chuangsuwanich, Rina Tse  

**Link**: [PDF](https://arxiv.org/pdf/2507.19947)  

**Abstract**: Fusing information from human observations can help robots overcome sensing limitations in collaborative tasks. However, an uncertainty-aware fusion framework requires a grounded likelihood representing the uncertainty of human inputs. This paper presents a Feature Pyramid Likelihood Grounding Network (FP-LGN) that grounds spatial language by learning relevant map image features and their relationships with spatial relation semantics. The model is trained as a probability estimator to capture aleatoric uncertainty in human language using three-stage curriculum learning. Results showed that FP-LGN matched expert-designed rules in mean Negative Log-Likelihood (NLL) and demonstrated greater robustness with lower standard deviation. Collaborative sensing results demonstrated that the grounded likelihood successfully enabled uncertainty-aware fusion of heterogeneous human language observations and robot sensor measurements, achieving significant improvements in human-robot collaborative task performance. 

---
# The Impact of Fine-tuning Large Language Models on Automated Program Repair 

**Authors**: Roman Macháček, Anastasiia Grishina, Max Hort, Leon Moonen  

**Link**: [PDF](https://arxiv.org/pdf/2507.19909)  

**Abstract**: Automated Program Repair (APR) uses various tools and techniques to help developers achieve functional and error-free code faster. In recent years, Large Language Models (LLMs) have gained popularity as components in APR tool chains because of their performance and flexibility. However, training such models requires a significant amount of resources. Fine-tuning techniques have been developed to adapt pre-trained LLMs to specific tasks, such as APR, and enhance their performance at far lower computational costs than training from scratch. In this study, we empirically investigate the impact of various fine-tuning techniques on the performance of LLMs used for APR. Our experiments provide insights into the performance of a selection of state-of-the-art LLMs pre-trained on code. The evaluation is done on three popular APR benchmarks (i.e., QuixBugs, Defects4J and HumanEval-Java) and considers six different LLMs with varying parameter sizes (resp. CodeGen, CodeT5, StarCoder, DeepSeekCoder, Bloom, and CodeLlama-2). We consider three training regimens: no fine-tuning, full fine-tuning, and parameter-efficient fine-tuning (PEFT) using LoRA and IA3. We observe that full fine-tuning techniques decrease the benchmarking performance of various models due to different data distributions and overfitting. By using parameter-efficient fine-tuning methods, we restrict models in the amount of trainable parameters and achieve better results.
Keywords: large language models, automated program repair, parameter-efficient fine-tuning, AI4Code, AI4SE, ML4SE. 

---
# Agentic Reinforced Policy Optimization 

**Authors**: Guanting Dong, Hangyu Mao, Kai Ma, Licheng Bao, Yifei Chen, Zhongyuan Wang, Zhongxia Chen, Jiazhen Du, Huiyang Wang, Fuzheng Zhang, Guorui Zhou, Yutao Zhu, Ji-Rong Wen, Zhicheng Dou  

**Link**: [PDF](https://arxiv.org/pdf/2507.19849)  

**Abstract**: Large-scale reinforcement learning with verifiable rewards (RLVR) has demonstrated its effectiveness in harnessing the potential of large language models (LLMs) for single-turn reasoning tasks. In realistic reasoning scenarios, LLMs can often utilize external tools to assist in task-solving processes. However, current RL algorithms inadequately balance the models' intrinsic long-horizon reasoning capabilities and their proficiency in multi-turn tool interactions. To bridge this gap, we propose Agentic Reinforced Policy Optimization (ARPO), a novel agentic RL algorithm tailored for training multi-turn LLM-based agents. Through preliminary experiments, we observe that LLMs tend to exhibit highly uncertain behavior, characterized by an increase in the entropy distribution of generated tokens, immediately following interactions with external tools. Motivated by this observation, ARPO incorporates an entropy-based adaptive rollout mechanism, dynamically balancing global trajectory sampling and step-level sampling, thereby promoting exploration at steps with high uncertainty after tool usage. By integrating an advantage attribution estimation, ARPO enables LLMs to internalize advantage differences in stepwise tool-use interactions. Our experiments across 13 challenging benchmarks in computational reasoning, knowledge reasoning, and deep search domains demonstrate ARPO's superiority over trajectory-level RL algorithms. Remarkably, ARPO achieves improved performance using only half of the tool-use budget required by existing methods, offering a scalable solution for aligning LLM-based agents with real-time dynamic environments. Our code and datasets are released at this https URL 

---
# AutoSign: Direct Pose-to-Text Translation for Continuous Sign Language Recognition 

**Authors**: Samuel Ebimobowei Johnny, Blessed Guda, Andrew Blayama Stephen, Assane Gueye  

**Link**: [PDF](https://arxiv.org/pdf/2507.19840)  

**Abstract**: Continuously recognizing sign gestures and converting them to glosses plays a key role in bridging the gap between the hearing and hearing-impaired communities. This involves recognizing and interpreting the hands, face, and body gestures of the signer, which pose a challenge as it involves a combination of all these features. Continuous Sign Language Recognition (CSLR) methods rely on multi-stage pipelines that first extract visual features, then align variable-length sequences with target glosses using CTC or HMM-based approaches. However, these alignment-based methods suffer from error propagation across stages, overfitting, and struggle with vocabulary scalability due to the intermediate gloss representation bottleneck. To address these limitations, we propose AutoSign, an autoregressive decoder-only transformer that directly translates pose sequences to natural language text, bypassing traditional alignment mechanisms entirely. The use of this decoder-only approach allows the model to directly map between the features and the glosses without the need for CTC loss while also directly learning the textual dependencies in the glosses. Our approach incorporates a temporal compression module using 1D CNNs to efficiently process pose sequences, followed by AraGPT2, a pre-trained Arabic decoder, to generate text (glosses). Through comprehensive ablation studies, we demonstrate that hand and body gestures provide the most discriminative features for signer-independent CSLR. By eliminating the multi-stage pipeline, AutoSign achieves substantial improvements on the Isharah-1000 dataset, achieving an improvement of up to 6.1\% in WER score compared to the best existing method. 

---
# Salsa as a Nonverbal Embodied Language -- The CoMPAS3D Dataset and Benchmarks 

**Authors**: Bermet Burkanova, Payam Jome Yazdian, Chuxuan Zhang, Trinity Evans, Paige Tuttösí, Angelica Lim  

**Link**: [PDF](https://arxiv.org/pdf/2507.19684)  

**Abstract**: Imagine a humanoid that can safely and creatively dance with a human, adapting to its partner's proficiency, using haptic signaling as a primary form of communication. While today's AI systems excel at text or voice-based interaction with large language models, human communication extends far beyond text-it includes embodied movement, timing, and physical coordination. Modeling coupled interaction between two agents poses a formidable challenge: it is continuous, bidirectionally reactive, and shaped by individual variation. We present CoMPAS3D, the largest and most diverse motion capture dataset of improvised salsa dancing, designed as a challenging testbed for interactive, expressive humanoid AI. The dataset includes 3 hours of leader-follower salsa dances performed by 18 dancers spanning beginner, intermediate, and professional skill levels. For the first time, we provide fine-grained salsa expert annotations, covering over 2,800 move segments, including move types, combinations, execution errors and stylistic elements. We draw analogies between partner dance communication and natural language, evaluating CoMPAS3D on two benchmark tasks for synthetic humans that parallel key problems in spoken language and dialogue processing: leader or follower generation with proficiency levels (speaker or listener synthesis), and duet (conversation) generation. Towards a long-term goal of partner dance with humans, we release the dataset, annotations, and code, along with a multitask SalsaAgent model capable of performing all benchmark tasks, alongside additional baselines to encourage research in socially interactive embodied AI and creative, expressive humanoid motion generation. 

---
# FedDPG: An Adaptive Yet Efficient Prompt-tuning Approach in Federated Learning Settings 

**Authors**: Ali Shakeri, Wei Emma Zhang, Amin Beheshti, Weitong Chen, Jian Yang, Lishan Yang  

**Link**: [PDF](https://arxiv.org/pdf/2507.19534)  

**Abstract**: Pre-trained Language Models (PLMs) have demonstrated impressive performance in various NLP tasks. However, traditional fine-tuning methods for leveraging PLMs for downstream tasks entail significant computational overhead. Prompt-tuning has emerged as an efficient alternative that involves prepending a limited number of parameters to the input sequence and only updating them while the PLM's parameters are frozen. However, this technique's prompts remain fixed for all inputs, reducing the model's flexibility. The Federated Learning (FL) technique has gained attention in recent years to address the growing concerns around data privacy. However, challenges such as communication and computation limitations of clients still need to be addressed. To mitigate these challenges, this paper introduces the Federated Dynamic Prompt Generator (FedDPG), which incorporates a dynamic prompt generator network to generate context-aware prompts based on the given input, ensuring flexibility and adaptability while prioritising data privacy in federated learning settings. Our experiments on three NLP benchmark datasets showcase that FedDPG outperforms the state-of-the-art parameter-efficient fine-tuning methods in terms of global model performance, and has significantly reduced the calculation time and the number of parameters to be sent through the FL network. 

---
# Does AI and Human Advice Mitigate Punishment for Selfish Behavior? An Experiment on AI ethics From a Psychological Perspective 

**Authors**: Margarita Leib, Nils Köbis, Ivan Soraperra  

**Link**: [PDF](https://arxiv.org/pdf/2507.19487)  

**Abstract**: People increasingly rely on AI-advice when making decisions. At times, such advice can promote selfish behavior. When individuals abide by selfishness-promoting AI advice, how are they perceived and punished? To study this question, we build on theories from social psychology and combine machine-behavior and behavioral economic approaches. In a pre-registered, financially-incentivized experiment, evaluators could punish real decision-makers who (i) received AI, human, or no advice. The advice (ii) encouraged selfish or prosocial behavior, and decision-makers (iii) behaved selfishly or, in a control condition, behaved prosocially. Evaluators further assigned responsibility to decision-makers and their advisors. Results revealed that (i) prosocial behavior was punished very little, whereas selfish behavior was punished much more. Focusing on selfish behavior, (ii) compared to receiving no advice, selfish behavior was penalized more harshly after prosocial advice and more leniently after selfish advice. Lastly, (iii) whereas selfish decision-makers were seen as more responsible when they followed AI compared to human advice, punishment between the two advice sources did not vary. Overall, behavior and advice content shape punishment, whereas the advice source does not. 

---
