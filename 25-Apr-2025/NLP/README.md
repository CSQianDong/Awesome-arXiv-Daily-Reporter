# The Sparse Frontier: Sparse Attention Trade-offs in Transformer LLMs 

**Authors**: Piotr Nawrot, Robert Li, Renjie Huang, Sebastian Ruder, Kelly Marchisio, Edoardo M. Ponti  

**Link**: [PDF](https://arxiv.org/pdf/2504.17768)  

**Abstract**: Sparse attention offers a promising strategy to extend long-context capabilities in Transformer LLMs, yet its viability, its efficiency-accuracy trade-offs, and systematic scaling studies remain unexplored. To address this gap, we perform a careful comparison of training-free sparse attention methods at varying model scales, sequence lengths, and sparsity levels on a diverse collection of long-sequence tasks-including novel ones that rely on natural language while remaining controllable and easy to evaluate. Based on our experiments, we report a series of key findings: 1) an isoFLOPS analysis reveals that for very long sequences, larger and highly sparse models are preferable to smaller and dense ones. 2) The level of sparsity attainable while statistically guaranteeing accuracy preservation is higher during decoding than prefilling, and correlates with model size in the former. 3) There is no clear strategy that performs best across tasks and phases, with different units of sparsification or budget adaptivity needed for different scenarios. Even moderate sparsity levels often result in significant performance degradation on at least one task, highlighting that sparse attention is not a universal solution. 4) We introduce and validate novel scaling laws specifically tailored for sparse attention, providing evidence that our findings are likely to hold true beyond our range of experiments. Through these insights, we demonstrate that sparse attention is a key tool to enhance the capabilities of Transformer LLMs for processing longer sequences, but requires careful evaluation of trade-offs for performance-sensitive applications. 

---
# Conversational Assistants to support Heart Failure Patients: comparing a Neurosymbolic Architecture with ChatGPT 

**Authors**: Anuja Tayal, Devika Salunke, Barbara Di Eugenio, Paula Allen-Meares, Eulalia Puig Abril, Olga Garcia, Carolyn Dickens, Andrew Boyd  

**Link**: [PDF](https://arxiv.org/pdf/2504.17753)  

**Abstract**: Conversational assistants are becoming more and more popular, including in healthcare, partly because of the availability and capabilities of Large Language Models. There is a need for controlled, probing evaluations with real stakeholders which can highlight advantages and disadvantages of more traditional architectures and those based on generative AI. We present a within-group user study to compare two versions of a conversational assistant that allows heart failure patients to ask about salt content in food. One version of the system was developed in-house with a neurosymbolic architecture, and one is based on ChatGPT. The evaluation shows that the in-house system is more accurate, completes more tasks and is less verbose than the one based on ChatGPT; on the other hand, the one based on ChatGPT makes fewer speech errors and requires fewer clarifications to complete the task. Patients show no preference for one over the other. 

---
# Multilingual Performance Biases of Large Language Models in Education 

**Authors**: Vansh Gupta, Sankalan Pal Chowdhury, Vilém Zouhar, Donya Rooein, Mrinmaya Sachan  

**Link**: [PDF](https://arxiv.org/pdf/2504.17720)  

**Abstract**: Large language models (LLMs) are increasingly being adopted in educational settings. These applications expand beyond English, though current LLMs remain primarily English-centric. In this work, we ascertain if their use in education settings in non-English languages is warranted. We evaluated the performance of popular LLMs on four educational tasks: identifying student misconceptions, providing targeted feedback, interactive tutoring, and grading translations in six languages (Hindi, Arabic, Farsi, Telugu, Ukrainian, Czech) in addition to English. We find that the performance on these tasks somewhat corresponds to the amount of language represented in training data, with lower-resource languages having poorer task performance. Although the models perform reasonably well in most languages, the frequent performance drop from English is significant. Thus, we recommend that practitioners first verify that the LLM works well in the target language for their educational task before deployment. 

---
# Safety in Large Reasoning Models: A Survey 

**Authors**: Cheng Wang, Yue Liu, Baolong Li, Duzhen Zhang, Zhongzhi Li, Junfeng Fang  

**Link**: [PDF](https://arxiv.org/pdf/2504.17704)  

**Abstract**: Large Reasoning Models (LRMs) have exhibited extraordinary prowess in tasks like mathematics and coding, leveraging their advanced reasoning capabilities. Nevertheless, as these capabilities progress, significant concerns regarding their vulnerabilities and safety have arisen, which can pose challenges to their deployment and application in real-world settings. This paper presents a comprehensive survey of LRMs, meticulously exploring and summarizing the newly emerged safety risks, attacks, and defense strategies. By organizing these elements into a detailed taxonomy, this work aims to offer a clear and structured understanding of the current safety landscape of LRMs, facilitating future research and development to enhance the security and reliability of these powerful models. 

---
# Ensemble Bayesian Inference: Leveraging Small Language Models to Achieve LLM-level Accuracy in Profile Matching Tasks 

**Authors**: Haru-Tada Sato, Fuka Matsuzaki, Jun-ichiro Takahashi  

**Link**: [PDF](https://arxiv.org/pdf/2504.17685)  

**Abstract**: This study explores the potential of small language model(SLM) ensembles to achieve accuracy comparable to proprietary large language models (LLMs). We propose Ensemble Bayesian Inference (EBI), a novel approach that applies Bayesian estimation to combine judgments from multiple SLMs, allowing them to exceed the performance limitations of individual models. Our experiments on diverse tasks(aptitude assessments and consumer profile analysis in both Japanese and English) demonstrate EBI's effectiveness. Notably, we analyze cases where incorporating models with negative Lift values into ensembles improves overall performance, and we examine the method's efficacy across different languages. These findings suggest new possibilities for constructing high-performance AI systems with limited computational resources and for effectively utilizing models with individually lower performance. Building on existing research on LLM performance evaluation, ensemble methods, and open-source LLM utilization, we discuss the novelty and significance of our approach. 

---
# Energy Considerations of Large Language Model Inference and Efficiency Optimizations 

**Authors**: Jared Fernandez, Clara Na, Vashisth Tiwari, Yonatan Bisk, Sasha Luccioni, Emma Strubell  

**Link**: [PDF](https://arxiv.org/pdf/2504.17674)  

**Abstract**: As large language models (LLMs) scale in size and adoption, their computational and environmental costs continue to rise. Prior benchmarking efforts have primarily focused on latency reduction in idealized settings, often overlooking the diverse real-world inference workloads that shape energy use. In this work, we systematically analyze the energy implications of common inference efficiency optimizations across diverse Natural Language Processing (NLP) and generative Artificial Intelligence (AI) workloads, including conversational AI and code generation. We introduce a modeling approach that approximates real-world LLM workflows through a binning strategy for input-output token distributions and batch size variations. Our empirical analysis spans software frameworks, decoding strategies, GPU architectures, online and offline serving settings, and model parallelism configurations. We show that the effectiveness of inference optimizations is highly sensitive to workload geometry, software stack, and hardware accelerators, demonstrating that naive energy estimates based on FLOPs or theoretical GPU utilization significantly underestimate real-world energy consumption. Our findings reveal that the proper application of relevant inference efficiency optimizations can reduce total energy use by up to 73% from unoptimized baselines. These insights provide a foundation for sustainable LLM deployment and inform energy-efficient design strategies for future AI infrastructure. 

---
# Data-Driven Calibration of Prediction Sets in Large Vision-Language Models Based on Inductive Conformal Prediction 

**Authors**: Yuanchang Ye, Weiyan Wen  

**Link**: [PDF](https://arxiv.org/pdf/2504.17671)  

**Abstract**: This study addresses the critical challenge of hallucination mitigation in Large Vision-Language Models (LVLMs) for Visual Question Answering (VQA) tasks through a Split Conformal Prediction (SCP) framework. While LVLMs excel in multi-modal reasoning, their outputs often exhibit hallucinated content with high confidence, posing risks in safety-critical applications. We propose a model-agnostic uncertainty quantification method that integrates dynamic threshold calibration and cross-modal consistency verification. By partitioning data into calibration and test sets, the framework computes nonconformity scores to construct prediction sets with statistical guarantees under user-defined risk levels ($\alpha$). Key innovations include: (1) rigorous control of \textbf{marginal coverage} to ensure empirical error rates remain strictly below $\alpha$; (2) dynamic adjustment of prediction set sizes inversely with $\alpha$, filtering low-confidence outputs; (3) elimination of prior distribution assumptions and retraining requirements. Evaluations on benchmarks (ScienceQA, MMMU) with eight LVLMs demonstrate that SCP enforces theoretical guarantees across all $\alpha$ values. The framework achieves stable performance across varying calibration-to-test split ratios, underscoring its robustness for real-world deployment in healthcare, autonomous systems, and other safety-sensitive domains. This work bridges the gap between theoretical reliability and practical applicability in multi-modal AI systems, offering a scalable solution for hallucination detection and uncertainty-aware decision-making. 

---
# Evaluating Grounded Reasoning by Code-Assisted Large Language Models for Mathematics 

**Authors**: Zena Al-Khalili, Nick Howell, Dietrich Klakow  

**Link**: [PDF](https://arxiv.org/pdf/2504.17665)  

**Abstract**: Assisting LLMs with code generation improved their performance on mathematical reasoning tasks. However, the evaluation of code-assisted LLMs is generally restricted to execution correctness, lacking a rigorous evaluation of their generated programs. In this work, we bridge this gap by conducting an in-depth analysis of code-assisted LLMs' generated programs in response to math reasoning tasks. Our evaluation focuses on the extent to which LLMs ground their programs to math rules, and how that affects their end performance. For this purpose, we assess the generations of five different LLMs, on two different math datasets, both manually and automatically. Our results reveal that the distribution of grounding depends on LLMs' capabilities and the difficulty of math problems. Furthermore, mathematical grounding is more effective for closed-source models, while open-source models fail to employ math rules in their solutions correctly. On MATH500, the percentage of grounded programs decreased to half, while the ungrounded generations doubled in comparison to ASDiv grade-school problems. Our work highlights the need for in-depth evaluation beyond execution accuracy metrics, toward a better understanding of code-assisted LLMs' capabilities and limits in the math domain. 

---
# Towards a comprehensive taxonomy of online abusive language informed by machine leaning 

**Authors**: Samaneh Hosseini Moghaddam, Kelly Lyons, Cheryl Regehr, Vivek Goel, Kaitlyn Regehr  

**Link**: [PDF](https://arxiv.org/pdf/2504.17653)  

**Abstract**: The proliferation of abusive language in online communications has posed significant risks to the health and wellbeing of individuals and communities. The growing concern regarding online abuse and its consequences necessitates methods for identifying and mitigating harmful content and facilitating continuous monitoring, moderation, and early intervention. This paper presents a taxonomy for distinguishing key characteristics of abusive language within online text. Our approach uses a systematic method for taxonomy development, integrating classification systems of 18 existing multi-label datasets to capture key characteristics relevant to online abusive language classification. The resulting taxonomy is hierarchical and faceted, comprising 5 categories and 17 dimensions. It classifies various facets of online abuse, including context, target, intensity, directness, and theme of abuse. This shared understanding can lead to more cohesive efforts, facilitate knowledge exchange, and accelerate progress in the field of online abuse detection and mitigation among researchers, policy makers, online platform owners, and other stakeholders. 

---
# RAGAT-Mind: A Multi-Granular Modeling Approach for Rumor Detection Based on MindSpore 

**Authors**: Zhenkai Qin, Guifang Yang, Dongze Wu  

**Link**: [PDF](https://arxiv.org/pdf/2504.17574)  

**Abstract**: As false information continues to proliferate across social media platforms, effective rumor detection has emerged as a pressing challenge in natural language processing. This paper proposes RAGAT-Mind, a multi-granular modeling approach for Chinese rumor detection, built upon the MindSpore deep learning framework. The model integrates TextCNN for local semantic extraction, bidirectional GRU for sequential context learning, Multi-Head Self-Attention for global dependency focusing, and Bidirectional Graph Convolutional Networks (BiGCN) for structural representation of word co-occurrence graphs. Experiments on the Weibo1-Rumor dataset demonstrate that RAGAT-Mind achieves superior classification performance, attaining 99.2% accuracy and a macro-F1 score of 0.9919. The results validate the effectiveness of combining hierarchical linguistic features with graph-based semantic structures. Furthermore, the model exhibits strong generalization and interpretability, highlighting its practical value for real-world rumor detection applications. 

---
# DeepDistill: Enhancing LLM Reasoning Capabilities via Large-Scale Difficulty-Graded Data Training 

**Authors**: Xiaoyu Tian, Sitong Zhao, Haotian Wang, Shuaiting Chen, Yiping Peng, Yunjie Ji, Han Zhao, Xiangang Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.17565)  

**Abstract**: Although large language models (LLMs) have recently achieved remarkable performance on various complex reasoning benchmarks, the academic community still lacks an in-depth understanding of base model training processes and data quality. To address this, we construct a large-scale, difficulty-graded reasoning dataset containing approximately 3.34 million unique queries of varying difficulty levels and about 40 million distilled responses generated by multiple models over several passes. Leveraging pass rate and Coefficient of Variation (CV), we precisely select the most valuable training data to enhance reasoning capability. Notably, we observe a training pattern shift, indicating that reasoning-focused training based on base models requires higher learning rates for effective training. Using this carefully selected data, we significantly improve the reasoning capabilities of the base model, achieving a pass rate of 79.2\% on the AIME2024 mathematical reasoning benchmark. This result surpasses most current distilled models and closely approaches state-of-the-art performance. We provide detailed descriptions of our data processing, difficulty assessment, and training methodology, and have publicly released all datasets and methods to promote rapid progress in open-source long-reasoning LLMs. The dataset is available at: this https URL 

---
# When Does Metadata Conditioning (NOT) Work for Language Model Pre-Training? A Study with Context-Free Grammars 

**Authors**: Rei Higuchi, Ryotaro Kawata, Naoki Nishikawa, Kazusato Oko, Shoichiro Yamaguchi, Sosuke Kobayashi, Seiya Tokui, Kohei Hayashi, Daisuke Okanohara, Taiji Suzuki  

**Link**: [PDF](https://arxiv.org/pdf/2504.17562)  

**Abstract**: The ability to acquire latent semantics is one of the key properties that determines the performance of language models. One convenient approach to invoke this ability is to prepend metadata (e.g. URLs, domains, and styles) at the beginning of texts in the pre-training data, making it easier for the model to access latent semantics before observing the entire text. Previous studies have reported that this technique actually improves the performance of trained models in downstream tasks; however, this improvement has been observed only in specific downstream tasks, without consistent enhancement in average next-token prediction loss. To understand this phenomenon, we closely investigate how prepending metadata during pre-training affects model performance by examining its behavior using artificial data. Interestingly, we found that this approach produces both positive and negative effects on the downstream tasks. We demonstrate that the effectiveness of the approach depends on whether latent semantics can be inferred from the downstream task's prompt. Specifically, through investigations using data generated by probabilistic context-free grammars, we show that training with metadata helps improve model's performance when the given context is long enough to infer the latent semantics. In contrast, the technique negatively impacts performance when the context lacks the necessary information to make an accurate posterior inference. 

---
# HalluLens: LLM Hallucination Benchmark 

**Authors**: Yejin Bang, Ziwei Ji, Alan Schelten, Anthony Hartshorn, Tara Fowler, Cheng Zhang, Nicola Cancedda, Pascale Fung  

**Link**: [PDF](https://arxiv.org/pdf/2504.17550)  

**Abstract**: Large language models (LLMs) often generate responses that deviate from user input or training data, a phenomenon known as "hallucination." These hallucinations undermine user trust and hinder the adoption of generative AI systems. Addressing hallucinations is essential for the advancement of LLMs. This paper introduces a comprehensive hallucination benchmark, incorporating both new extrinsic and existing intrinsic evaluation tasks, built upon clear taxonomy of hallucination. A major challenge in benchmarking hallucinations is the lack of a unified framework due to inconsistent definitions and categorizations. We disentangle LLM hallucination from "factuality," proposing a clear taxonomy that distinguishes between extrinsic and intrinsic hallucinations, to promote consistency and facilitate research. Extrinsic hallucinations, where the generated content is not consistent with the training data, are increasingly important as LLMs evolve. Our benchmark includes dynamic test set generation to mitigate data leakage and ensure robustness against such leakage. We also analyze existing benchmarks, highlighting their limitations and saturation. The work aims to: (1) establish a clear taxonomy of hallucinations, (2) introduce new extrinsic hallucination tasks, with data that can be dynamically regenerated to prevent saturation by leakage, (3) provide a comprehensive analysis of existing benchmarks, distinguishing them from factuality evaluations. 

---
# Unified Attacks to Large Language Model Watermarks: Spoofing and Scrubbing in Unauthorized Knowledge Distillation 

**Authors**: Xin Yi, Shunfan Zhengc, Linlin Wanga, Xiaoling Wang, Liang He  

**Link**: [PDF](https://arxiv.org/pdf/2504.17480)  

**Abstract**: Watermarking has emerged as a critical technique for combating misinformation and protecting intellectual property in large language models (LLMs). A recent discovery, termed watermark radioactivity, reveals that watermarks embedded in teacher models can be inherited by student models through knowledge distillation. On the positive side, this inheritance allows for the detection of unauthorized knowledge distillation by identifying watermark traces in student models. However, the robustness of watermarks against scrubbing attacks and their unforgeability in the face of spoofing attacks under unauthorized knowledge distillation remain largely unexplored. Existing watermark attack methods either assume access to model internals or fail to simultaneously support both scrubbing and spoofing attacks. In this work, we propose Contrastive Decoding-Guided Knowledge Distillation (CDG-KD), a unified framework that enables bidirectional attacks under unauthorized knowledge distillation. Our approach employs contrastive decoding to extract corrupted or amplified watermark texts via comparing outputs from the student model and weakly watermarked references, followed by bidirectional distillation to train new student models capable of watermark removal and watermark forgery, respectively. Extensive experiments show that CDG-KD effectively performs attacks while preserving the general performance of the distilled model. Our findings underscore critical need for developing watermarking schemes that are robust and unforgeable. 

---
# Creating Targeted, Interpretable Topic Models with LLM-Generated Text Augmentation 

**Authors**: Anna Lieb, Maneesh Arora, Eni Mustafaraj  

**Link**: [PDF](https://arxiv.org/pdf/2504.17445)  

**Abstract**: Unsupervised machine learning techniques, such as topic modeling and clustering, are often used to identify latent patterns in unstructured text data in fields such as political science and sociology. These methods overcome common concerns about reproducibility and costliness involved in the labor-intensive process of human qualitative analysis. However, two major limitations of topic models are their interpretability and their practicality for answering targeted, domain-specific social science research questions. In this work, we investigate opportunities for using LLM-generated text augmentation to improve the usefulness of topic modeling output. We use a political science case study to evaluate our results in a domain-specific application, and find that topic modeling using GPT-4 augmentations creates highly interpretable categories that can be used to investigate domain-specific research questions with minimal human guidance. 

---
# PicPersona-TOD : A Dataset for Personalizing Utterance Style in Task-Oriented Dialogue with Image Persona 

**Authors**: Jihyun Lee, Yejin Jeon, Seungyeon Seo, Gary Geunbae Lee  

**Link**: [PDF](https://arxiv.org/pdf/2504.17390)  

**Abstract**: Task-Oriented Dialogue (TOD) systems are designed to fulfill user requests through natural language interactions, yet existing systems often produce generic, monotonic responses that lack individuality and fail to adapt to users' personal attributes. To address this, we introduce PicPersona-TOD, a novel dataset that incorporates user images as part of the persona, enabling personalized responses tailored to user-specific factors such as age or emotional context. This is facilitated by first impressions, dialogue policy-guided prompting, and the use of external knowledge to reduce hallucinations. Human evaluations confirm that our dataset enhances user experience, with personalized responses contributing to a more engaging interaction. Additionally, we introduce a new NLG model, Pictor, which not only personalizes responses, but also demonstrates robust performance across unseen domains this https URL. 

---
# LiveLongBench: Tackling Long-Context Understanding for Spoken Texts from Live Streams 

**Authors**: Yongxuan Wu, Runyu Chen, Peiyu Liu, Hongjin Qian  

**Link**: [PDF](https://arxiv.org/pdf/2504.17366)  

**Abstract**: Long-context understanding poses significant challenges in natural language processing, particularly for real-world dialogues characterized by speech-based elements, high redundancy, and uneven information density. Although large language models (LLMs) achieve impressive results on existing benchmarks, these datasets fail to reflect the complexities of such texts, limiting their applicability to practical scenarios. To bridge this gap, we construct the first spoken long-text dataset, derived from live streams, designed to reflect the redundancy-rich and conversational nature of real-world scenarios. We construct tasks in three categories: retrieval-dependent, reasoning-dependent, and hybrid. We then evaluate both popular LLMs and specialized methods to assess their ability to understand long-contexts in these tasks. Our results show that current methods exhibit strong task-specific preferences and perform poorly on highly redundant inputs, with no single method consistently outperforming others. We propose a new baseline that better handles redundancy in spoken text and achieves strong performance across tasks. Our findings highlight key limitations of current methods and suggest future directions for improving long-context understanding. Finally, our benchmark fills a gap in evaluating long-context spoken language understanding and provides a practical foundation for developing real-world e-commerce systems. The code and benchmark are available at this https URL. 

---
# PatientDx: Merging Large Language Models for Protecting Data-Privacy in Healthcare 

**Authors**: Jose G. Moreno, Jesus Lovon, M'Rick Robin-Charlet, Christine Damase-Michel, Lynda Tamine  

**Link**: [PDF](https://arxiv.org/pdf/2504.17360)  

**Abstract**: Fine-tuning of Large Language Models (LLMs) has become the default practice for improving model performance on a given task. However, performance improvement comes at the cost of training on vast amounts of annotated data which could be sensitive leading to significant data privacy concerns. In particular, the healthcare domain is one of the most sensitive domains exposed to data privacy issues. In this paper, we present PatientDx, a framework of model merging that allows the design of effective LLMs for health-predictive tasks without requiring fine-tuning nor adaptation on patient data. Our proposal is based on recently proposed techniques known as merging of LLMs and aims to optimize a building block merging strategy. PatientDx uses a pivotal model adapted to numerical reasoning and tunes hyperparameters on examples based on a performance metric but without training of the LLM on these data. Experiments using the mortality tasks of the MIMIC-IV dataset show improvements up to 7% in terms of AUROC when compared to initial models. Additionally, we confirm that when compared to fine-tuned models, our proposal is less prone to data leak problems without hurting performance. Finally, we qualitatively show the capabilities of our proposal through a case study. Our best model is publicly available at this https URL Jgmorenof/mistral\_merged\_0\_4. 

---
# M-MRE: Extending the Mutual Reinforcement Effect to Multimodal Information Extraction 

**Authors**: Chengguang Gan, Sunbowen Lee, Zhixi Cai, Yanbin Wei, Lei Zheng, Yunhao Liang, Shiwen Ni, Tatsunori Mori  

**Link**: [PDF](https://arxiv.org/pdf/2504.17353)  

**Abstract**: Mutual Reinforcement Effect (MRE) is an emerging subfield at the intersection of information extraction and model interpretability. MRE aims to leverage the mutual understanding between tasks of different granularities, enhancing the performance of both coarse-grained and fine-grained tasks through joint modeling. While MRE has been explored and validated in the textual domain, its applicability to visual and multimodal domains remains unexplored. In this work, we extend MRE to the multimodal information extraction domain for the first time. Specifically, we introduce a new task: Multimodal Mutual Reinforcement Effect (M-MRE), and construct a corresponding dataset to support this task. To address the challenges posed by M-MRE, we further propose a Prompt Format Adapter (PFA) that is fully compatible with various Large Vision-Language Models (LVLMs). Experimental results demonstrate that MRE can also be observed in the M-MRE task, a multimodal text-image understanding scenario. This provides strong evidence that MRE facilitates mutual gains across three interrelated tasks, confirming its generalizability beyond the textual domain. 

---
# Bridging Cognition and Emotion: Empathy-Driven Multimodal Misinformation Detection 

**Authors**: Zihan Wang, Lu Yuan, Zhengxuan Zhang, Qing Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2504.17332)  

**Abstract**: In the digital era, social media has become a major conduit for information dissemination, yet it also facilitates the rapid spread of misinformation. Traditional misinformation detection methods primarily focus on surface-level features, overlooking the crucial roles of human empathy in the propagation process. To address this gap, we propose the Dual-Aspect Empathy Framework (DAE), which integrates cognitive and emotional empathy to analyze misinformation from both the creator and reader perspectives. By examining creators' cognitive strategies and emotional appeals, as well as simulating readers' cognitive judgments and emotional responses using Large Language Models (LLMs), DAE offers a more comprehensive and human-centric approach to misinformation detection. Moreover, we further introduce an empathy-aware filtering mechanism to enhance response authenticity and diversity. Experimental results on benchmark datasets demonstrate that DAE outperforms existing methods, providing a novel paradigm for multimodal misinformation detection. 

---
# FLUKE: A Linguistically-Driven and Task-Agnostic Framework for Robustness Evaluation 

**Authors**: Yulia Otmakhova, Hung Thinh Truong, Rahmad Mahendra, Zenan Zhai, Rongxin Zhu, Daniel Beck, Jey Han Lau  

**Link**: [PDF](https://arxiv.org/pdf/2504.17311)  

**Abstract**: We present FLUKE (Framework for LingUistically-driven and tasK-agnostic robustness Evaluation), a task-agnostic framework for assessing model robustness through systematic minimal variations of test data. FLUKE introduces controlled variations across linguistic levels - from orthography to dialect and style varieties - and leverages large language models (LLMs) with human validation to generate modifications. We demonstrate FLUKE's utility by evaluating both fine-tuned models and LLMs across four diverse NLP tasks, and reveal that (1) the impact of linguistic variations is highly task-dependent, with some tests being critical for certain tasks but irrelevant for others; (2) while LLMs have better overall robustness compared to fine-tuned models, they still exhibit significant brittleness to certain linguistic variations; (3) all models show substantial vulnerability to negation modifications across most tasks. These findings highlight the importance of systematic robustness testing for understanding model behaviors. 

---
# CoheMark: A Novel Sentence-Level Watermark for Enhanced Text Quality 

**Authors**: Junyan Zhang, Shuliang Liu, Aiwei Liu, Yubo Gao, Jungang Li, Xiaojie Gu, Xuming Hu  

**Link**: [PDF](https://arxiv.org/pdf/2504.17309)  

**Abstract**: Watermarking technology is a method used to trace the usage of content generated by large language models. Sentence-level watermarking aids in preserving the semantic integrity within individual sentences while maintaining greater robustness. However, many existing sentence-level watermarking techniques depend on arbitrary segmentation or generation processes to embed watermarks, which can limit the availability of appropriate sentences. This limitation, in turn, compromises the quality of the generated response. To address the challenge of balancing high text quality with robust watermark detection, we propose CoheMark, an advanced sentence-level watermarking technique that exploits the cohesive relationships between sentences for better logical fluency. The core methodology of CoheMark involves selecting sentences through trained fuzzy c-means clustering and applying specific next sentence selection criteria. Experimental evaluations demonstrate that CoheMark achieves strong watermark strength while exerting minimal impact on text quality. 

---
# Evaluating and Mitigating Bias in AI-Based Medical Text Generation 

**Authors**: Xiuying Chen, Tairan Wang, Juexiao Zhou, Zirui Song, Xin Gao, Xiangliang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.17279)  

**Abstract**: Artificial intelligence (AI) systems, particularly those based on deep learning models, have increasingly achieved expert-level performance in medical applications. However, there is growing concern that such AI systems may reflect and amplify human bias, and reduce the quality of their performance in historically under-served populations. The fairness issue has attracted considerable research interest in the medical imaging classification field, yet it remains understudied in the text generation domain. In this study, we investigate the fairness problem in text generation within the medical field and observe significant performance discrepancies across different races, sexes, and age groups, including intersectional groups, various model scales, and different evaluation metrics. To mitigate this fairness issue, we propose an algorithm that selectively optimizes those underperformed groups to reduce bias. The selection rules take into account not only word-level accuracy but also the pathology accuracy to the target reference, while ensuring that the entire process remains fully differentiable for effective model training. Our evaluations across multiple backbones, datasets, and modalities demonstrate that our proposed algorithm enhances fairness in text generation without compromising overall performance. Specifically, the disparities among various groups across different metrics were diminished by more than 30% with our algorithm, while the relative change in text generation accuracy was typically within 2%. By reducing the bias generated by deep learning models, our proposed approach can potentially alleviate concerns about the fairness and reliability of text generation diagnosis in medical domain.
Our code is publicly available to facilitate further research at this https URL. 

---
# JurisCTC: Enhancing Legal Judgment Prediction via Cross-Domain Transfer and Contrastive Learning 

**Authors**: Zhaolu Kang, Hongtian Cai, Xiangyang Ji, Jinzhe Li, Nanfei Gu  

**Link**: [PDF](https://arxiv.org/pdf/2504.17264)  

**Abstract**: In recent years, Unsupervised Domain Adaptation (UDA) has gained significant attention in the field of Natural Language Processing (NLP) owing to its ability to enhance model generalization across diverse domains. However, its application for knowledge transfer between distinct legal domains remains largely unexplored. To address the challenges posed by lengthy and complex legal texts and the limited availability of large-scale annotated datasets, we propose JurisCTC, a novel model designed to improve the accuracy of Legal Judgment Prediction (LJP) tasks. Unlike existing approaches, JurisCTC facilitates effective knowledge transfer across various legal domains and employs contrastive learning to distinguish samples from different domains. Specifically, for the LJP task, we enable knowledge transfer between civil and criminal law domains. Compared to other models and specific large language models (LLMs), JurisCTC demonstrates notable advancements, achieving peak accuracies of 76.59% and 78.83%, respectively. 

---
# Low-Resource Neural Machine Translation Using Recurrent Neural Networks and Transfer Learning: A Case Study on English-to-Igbo 

**Authors**: Ocheme Anthony Ekle, Biswarup Das  

**Link**: [PDF](https://arxiv.org/pdf/2504.17252)  

**Abstract**: In this study, we develop Neural Machine Translation (NMT) and Transformer-based transfer learning models for English-to-Igbo translation - a low-resource African language spoken by over 40 million people across Nigeria and West Africa. Our models are trained on a curated and benchmarked dataset compiled from Bible corpora, local news, Wikipedia articles, and Common Crawl, all verified by native language experts. We leverage Recurrent Neural Network (RNN) architectures, including Long Short-Term Memory (LSTM) and Gated Recurrent Units (GRU), enhanced with attention mechanisms to improve translation accuracy. To further enhance performance, we apply transfer learning using MarianNMT pre-trained models within the SimpleTransformers framework. Our RNN-based system achieves competitive results, closely matching existing English-Igbo benchmarks. With transfer learning, we observe a performance gain of +4.83 BLEU points, reaching an estimated translation accuracy of 70%. These findings highlight the effectiveness of combining RNNs with transfer learning to address the performance gap in low-resource language translation tasks. 

---
# Crisp: Cognitive Restructuring of Negative Thoughts through Multi-turn Supportive Dialogues 

**Authors**: Jinfeng Zhou, Yuxuan Chen, Jianing Yin, Yongkang Huang, Yihan Shi, Xikun Zhang, Libiao Peng, Rongsheng Zhang, Tangjie Lv, Zhipeng Hu, Hongning Wang, Minlie Huang  

**Link**: [PDF](https://arxiv.org/pdf/2504.17238)  

**Abstract**: Cognitive Restructuring (CR) is a psychotherapeutic process aimed at identifying and restructuring an individual's negative thoughts, arising from mental health challenges, into more helpful and positive ones via multi-turn dialogues. Clinician shortage and stigma urge the development of human-LLM interactive psychotherapy for CR. Yet, existing efforts implement CR via simple text rewriting, fixed-pattern dialogues, or a one-shot CR workflow, failing to align with the psychotherapeutic process for effective CR. To address this gap, we propose CRDial, a novel framework for CR, which creates multi-turn dialogues with specifically designed identification and restructuring stages of negative thoughts, integrates sentence-level supportive conversation strategies, and adopts a multi-channel loop mechanism to enable iterative CR. With CRDial, we distill Crisp, a large-scale and high-quality bilingual dialogue dataset, from LLM. We then train Crispers, Crisp-based conversational LLMs for CR, at 7B and 14B scales. Extensive human studies show the superiority of Crispers in pointwise, pairwise, and intervention evaluations. 

---
# Does Knowledge Distillation Matter for Large Language Model based Bundle Generation? 

**Authors**: Kaidong Feng, Zhu Sun, Jie Yang, Hui Fang, Xinghua Qu, Wenyuan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.17220)  

**Abstract**: LLMs are increasingly explored for bundle generation, thanks to their reasoning capabilities and knowledge. However, deploying large-scale LLMs introduces significant efficiency challenges, primarily high computational costs during fine-tuning and inference due to their massive parameterization. Knowledge distillation (KD) offers a promising solution, transferring expertise from large teacher models to compact student models. This study systematically investigates knowledge distillation approaches for bundle generation, aiming to minimize computational demands while preserving performance. We explore three critical research questions: (1) how does the format of KD impact bundle generation performance? (2) to what extent does the quantity of distilled knowledge influence performance? and (3) how do different ways of utilizing the distilled knowledge affect performance? We propose a comprehensive KD framework that (i) progressively extracts knowledge (patterns, rules, deep thoughts); (ii) captures varying quantities of distilled knowledge through different strategies; and (iii) exploits complementary LLM adaptation techniques (in-context learning, supervised fine-tuning, combination) to leverage distilled knowledge in small student models for domain-specific adaptation and enhanced efficiency. Extensive experiments provide valuable insights into how knowledge format, quantity, and utilization methodologies collectively shape LLM-based bundle generation performance, exhibiting KD's significant potential for more efficient yet effective LLM-based bundle generation. 

---
# A RAG-Based Multi-Agent LLM System for Natural Hazard Resilience and Adaptation 

**Authors**: Yangxinyu Xie, Bowen Jiang, Tanwi Mallick, Joshua David Bergerson, John K. Hutchison, Duane R. Verner, Jordan Branham, M. Ross Alexander, Robert B. Ross, Yan Feng, Leslie-Anne Levy, Weijie Su, Camillo J. Taylor  

**Link**: [PDF](https://arxiv.org/pdf/2504.17200)  

**Abstract**: Large language models (LLMs) are a transformational capability at the frontier of artificial intelligence and machine learning that can support decision-makers in addressing pressing societal challenges such as extreme natural hazard events. As generalized models, LLMs often struggle to provide context-specific information, particularly in areas requiring specialized knowledge. In this work we propose a retrieval-augmented generation (RAG)-based multi-agent LLM system to support analysis and decision-making in the context of natural hazards and extreme weather events. As a proof of concept, we present WildfireGPT, a specialized system focused on wildfire hazards. The architecture employs a user-centered, multi-agent design to deliver tailored risk insights across diverse stakeholder groups. By integrating natural hazard and extreme weather projection data, observational datasets, and scientific literature through an RAG framework, the system ensures both the accuracy and contextual relevance of the information it provides. Evaluation across ten expert-led case studies demonstrates that WildfireGPT significantly outperforms existing LLM-based solutions for decision support. 

---
# Paper2Code: Automating Code Generation from Scientific Papers in Machine Learning 

**Authors**: Minju Seo, Jinheon Baek, Seongyun Lee, Sung Ju Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2504.17192)  

**Abstract**: Despite the rapid growth of machine learning research, corresponding code implementations are often unavailable, making it slow and labor-intensive for researchers to reproduce results and build upon prior work. In the meantime, recent Large Language Models (LLMs) excel at understanding scientific documents and generating high-quality code. Inspired by this, we introduce PaperCoder, a multi-agent LLM framework that transforms machine learning papers into functional code repositories. PaperCoder operates in three stages: planning, where it constructs a high-level roadmap, designs the system architecture with diagrams, identifies file dependencies, and generates configuration files; analysis, which focuses on interpreting implementation-specific details; and generation, where modular, dependency-aware code is produced. Moreover, each phase is instantiated through a set of specialized agents designed to collaborate effectively across the pipeline. We then evaluate PaperCoder on generating code implementations from machine learning papers based on both model-based and human evaluations, specifically from the original paper authors, with author-released repositories as ground truth if available. Our results demonstrate the effectiveness of PaperCoder in creating high-quality, faithful implementations. Furthermore, it consistently shows strengths in the recently released PaperBench benchmark, surpassing strong baselines by substantial margins. 

---
# MIRAGE: A Metric-Intensive Benchmark for Retrieval-Augmented Generation Evaluation 

**Authors**: Chanhee Park, Hyeonseok Moon, Chanjun Park, Heuiseok Lim  

**Link**: [PDF](https://arxiv.org/pdf/2504.17137)  

**Abstract**: Retrieval-Augmented Generation (RAG) has gained prominence as an effective method for enhancing the generative capabilities of Large Language Models (LLMs) through the incorporation of external knowledge. However, the evaluation of RAG systems remains a challenge, due to the intricate interplay between retrieval and generation components. This limitation has resulted in a scarcity of benchmarks that facilitate a detailed, component-specific assessment. In this work, we present MIRAGE, a Question Answering dataset specifically designed for RAG evaluation. MIRAGE consists of 7,560 curated instances mapped to a retrieval pool of 37,800 entries, enabling an efficient and precise evaluation of both retrieval and generation tasks. We also introduce novel evaluation metrics aimed at measuring RAG adaptability, encompassing dimensions such as noise vulnerability, context acceptability, context insensitivity, and context misinterpretation. Through comprehensive experiments across various retriever-LLM configurations, we provide new insights into the optimal alignment of model pairs and the nuanced dynamics within RAG systems. The dataset and evaluation code are publicly available, allowing for seamless integration and customization in diverse research settings\footnote{The MIRAGE code and data are available at this https URL. 

---
# Steering the CensorShip: Uncovering Representation Vectors for LLM "Thought" Control 

**Authors**: Hannah Cyberey, David Evans  

**Link**: [PDF](https://arxiv.org/pdf/2504.17130)  

**Abstract**: Large language models (LLMs) have transformed the way we access information. These models are often tuned to refuse to comply with requests that are considered harmful and to produce responses that better align with the preferences of those who control the models. To understand how this "censorship" works. We use representation engineering techniques to study open-weights safety-tuned models. We present a method for finding a refusal--compliance vector that detects and controls the level of censorship in model outputs. We also analyze recent reasoning LLMs, distilled from DeepSeek-R1, and uncover an additional dimension of censorship through "thought suppression". We show a similar approach can be used to find a vector that suppresses the model's reasoning process, allowing us to remove censorship by applying the negative multiples of this vector 

---
# The Rise of Small Language Models in Healthcare: A Comprehensive Survey 

**Authors**: Muskan Garg, Shaina Raza, Shebuti Rayana, Xingyi Liu, Sunghwan Sohn  

**Link**: [PDF](https://arxiv.org/pdf/2504.17119)  

**Abstract**: Despite substantial progress in healthcare applications driven by large language models (LLMs), growing concerns around data privacy, and limited resources; the small language models (SLMs) offer a scalable and clinically viable solution for efficient performance in resource-constrained environments for next-generation healthcare informatics. Our comprehensive survey presents a taxonomic framework to identify and categorize them for healthcare professionals and informaticians. The timeline of healthcare SLM contributions establishes a foundational framework for analyzing models across three dimensions: NLP tasks, stakeholder roles, and the continuum of care. We present a taxonomic framework to identify the architectural foundations for building models from scratch; adapting SLMs to clinical precision through prompting, instruction fine-tuning, and reasoning; and accessibility and sustainability through compression techniques. Our primary objective is to offer a comprehensive survey for healthcare professionals, introducing recent innovations in model optimization and equipping them with curated resources to support future research and development in the field. Aiming to showcase the groundbreaking advancements in SLMs for healthcare, we present a comprehensive compilation of experimental results across widely studied NLP tasks in healthcare to highlight the transformative potential of SLMs in healthcare. The updated repository is available at Github 

---
# Co-CoT: A Prompt-Based Framework for Collaborative Chain-of-Thought Reasoning 

**Authors**: Seunghyun Yoo  

**Link**: [PDF](https://arxiv.org/pdf/2504.17091)  

**Abstract**: Due to the proliferation of short-form content and the rapid adoption of AI, opportunities for deep, reflective thinking have significantly diminished, undermining users' critical thinking and reducing engagement with the reasoning behind AI-generated outputs. To address this issue, we propose an Interactive Chain-of-Thought (CoT) Framework that enhances human-centered explainability and responsible AI usage by making the model's inference process transparent, modular, and user-editable. The framework decomposes reasoning into clearly defined blocks that users can inspect, modify, and re-execute, encouraging active cognitive engagement rather than passive consumption. It further integrates a lightweight edit-adaptation mechanism inspired by preference learning, allowing the system to align with diverse cognitive styles and user intentions. Ethical transparency is ensured through explicit metadata disclosure, built-in bias checkpoint functionality, and privacy-preserving safeguards. This work outlines the design principles and architecture necessary to promote critical engagement, responsible interaction, and inclusive adaptation in AI systems aimed at addressing complex societal challenges. 

---
# How Individual Traits and Language Styles Shape Preferences In Open-ended User-LLM Interaction: A Preliminary Study 

**Authors**: Rendi Chevi, Kentaro Inui, Thamar Solorio, Alham Fikri Aji  

**Link**: [PDF](https://arxiv.org/pdf/2504.17083)  

**Abstract**: What makes an interaction with the LLM more preferable for the user? While it is intuitive to assume that information accuracy in the LLM's responses would be one of the influential variables, recent studies have found that inaccurate LLM's responses could still be preferable when they are perceived to be more authoritative, certain, well-articulated, or simply verbose. These variables interestingly fall under the broader category of language style, implying that the style in the LLM's responses might meaningfully influence users' preferences. This hypothesized dynamic could have double-edged consequences: enhancing the overall user experience while simultaneously increasing their susceptibility to risks such as LLM's misinformation or hallucinations. In this short paper, we present our preliminary studies in exploring this subject. Through a series of exploratory and experimental user studies, we found that LLM's language style does indeed influence user's preferences, but how and which language styles influence the preference varied across different user populations, and more interestingly, moderated by the user's very own individual traits. As a preliminary work, the findings in our studies should be interpreted with caution, particularly given the limitations in our samples, which still need wider demographic diversity and larger sample sizes. Our future directions will first aim to address these limitations, which would enable a more comprehensive joint effect analysis between the language style, individual traits, and preferences, and further investigate the potential causal relationship between and beyond these variables. 

---
# Agree to Disagree? A Meta-Evaluation of LLM Misgendering 

**Authors**: Arjun Subramonian, Vagrant Gautam, Preethi Seshadri, Dietrich Klakow, Kai-Wei Chang, Yizhou Sun  

**Link**: [PDF](https://arxiv.org/pdf/2504.17075)  

**Abstract**: Numerous methods have been proposed to measure LLM misgendering, including probability-based evaluations (e.g., automatically with templatic sentences) and generation-based evaluations (e.g., with automatic heuristics or human validation). However, it has gone unexamined whether these evaluation methods have convergent validity, that is, whether their results align. Therefore, we conduct a systematic meta-evaluation of these methods across three existing datasets for LLM misgendering. We propose a method to transform each dataset to enable parallel probability- and generation-based evaluation. Then, by automatically evaluating a suite of 6 models from 3 families, we find that these methods can disagree with each other at the instance, dataset, and model levels, conflicting on 20.2% of evaluation instances. Finally, with a human evaluation of 2400 LLM generations, we show that misgendering behaviour is complex and goes far beyond pronouns, which automatic evaluations are not currently designed to capture, suggesting essential disagreement with human evaluations. Based on our findings, we provide recommendations for future evaluations of LLM misgendering. Our results are also more widely relevant, as they call into question broader methodological conventions in LLM evaluation, which often assume that different evaluation methods agree. 

---
# Do Words Reflect Beliefs? Evaluating Belief Depth in Large Language Models 

**Authors**: Shariar Kabir, Kevin Esterling, Yue Dong  

**Link**: [PDF](https://arxiv.org/pdf/2504.17052)  

**Abstract**: Large Language Models (LLMs) are increasingly shaping political discourse, yet their responses often display inconsistency when subjected to scrutiny. While prior research has primarily categorized LLM outputs as left- or right-leaning to assess their political stances, a critical question remains: Do these responses reflect genuine internal beliefs or merely surface-level alignment with training data? To address this, we propose a novel framework for evaluating belief depth by analyzing (1) argumentative consistency and (2) uncertainty quantification. We evaluate 12 LLMs on 19 economic policies from the Political Compass Test, challenging their belief stability with both supportive and opposing arguments. Our analysis reveals that LLMs exhibit topic-specific belief stability rather than a uniform ideological stance. Notably, up to 95% of left-leaning models' responses and 89% of right-leaning models' responses remain consistent under the challenge, enabling semantic entropy to achieve high accuracy (AUROC=0.78), effectively distinguishing between surface-level alignment from genuine belief. These findings call into question the assumption that LLMs maintain stable, human-like political ideologies, emphasizing the importance of conducting topic-specific reliability assessments for real-world applications. 

---
# Optimizing LLMs for Italian: Reducing Token Fertility and Enhancing Efficiency Through Vocabulary Adaptation 

**Authors**: Luca Moroni, Giovanni Puccetti, Pere-Lluis Huguet Cabot, Andrei Stefan Bejgu, Edoardo Barba, Alessio Miaschi, Felice Dell'Orletta, Andrea Esuli, Roberto Navigli  

**Link**: [PDF](https://arxiv.org/pdf/2504.17025)  

**Abstract**: The number of pretrained Large Language Models (LLMs) is increasing steadily, though the majority are designed predominantly for the English language. While state-of-the-art LLMs can handle other languages, due to language contamination or some degree of multilingual pretraining data, they are not optimized for non-English languages, leading to inefficient encoding (high token "fertility") and slower inference speed. In this work, we thoroughly compare a variety of vocabulary adaptation techniques for optimizing English LLMs for the Italian language, and put forward Semantic Alignment Vocabulary Adaptation (SAVA), a novel method that leverages neural mapping for vocabulary substitution. SAVA achieves competitive performance across multiple downstream tasks, enhancing grounded alignment strategies. We adapt two LLMs: Mistral-7b-v0.1, reducing token fertility by 25\%, and Llama-3.1-8B, optimizing the vocabulary and reducing the number of parameters by 1 billion. We show that, following the adaptation of the vocabulary, these models can recover their performance with a relatively limited stage of continual training on the target language. Finally, we test the capabilities of the adapted models on various multi-choice and generative tasks. 

---
# Tokenization Matters: Improving Zero-Shot NER for Indic Languages 

**Authors**: Priyaranjan Pattnayak, Hitesh Laxmichand Patel, Amit Agarwal  

**Link**: [PDF](https://arxiv.org/pdf/2504.16977)  

**Abstract**: Tokenization is a critical component of Natural Language Processing (NLP), especially for low resource languages, where subword segmentation influences vocabulary structure and downstream task accuracy. Although Byte Pair Encoding (BPE) is a standard tokenization method in multilingual language models, its suitability for Named Entity Recognition (NER) in low resource Indic languages remains underexplored due to its limitations in handling morphological complexity. In this work, we systematically compare BPE, SentencePiece, and Character Level tokenization strategies using IndicBERT for NER tasks in low resource Indic languages like Assamese, Bengali, Marathi, and Odia, as well as extremely low resource Indic languages like Santali, Manipuri, and Sindhi. We assess both intrinsic linguistic properties tokenization efficiency, out of vocabulary (OOV) rates, and morphological preservation as well as extrinsic downstream performance, including fine tuning and zero shot cross lingual transfer.
Our experiments show that SentencePiece is a consistently better performing approach than BPE for NER in low resource Indic Languages, particularly in zero shot cross lingual settings, as it better preserves entity consistency. While BPE provides the most compact tokenization form, it is not capable of generalization because it misclassifies or even fails to recognize entity labels when tested on unseen languages. In contrast, SentencePiece constitutes a better linguistic structural preservation model, benefiting extremely low resource and morphologically rich Indic languages, such as Santali and Manipuri, for superior entity recognition, as well as high generalization across scripts, such as Sindhi, written in Arabic. The results point to SentencePiece as the more effective tokenization strategy for NER within multilingual and low resource Indic NLP applications. 

---
# Bidirectional Mamba for Single-Cell Data: Efficient Context Learning with Biological Fidelity 

**Authors**: Cong Qi, Hanzhang Fang, Tianxing Hu, Siqi Jiang, Wei Zhi  

**Link**: [PDF](https://arxiv.org/pdf/2504.16956)  

**Abstract**: Single-cell RNA sequencing (scRNA-seq) enables high-resolution analysis of cellular heterogeneity, but its complexity, which is marked by high dimensionality, sparsity, and batch effects, which poses major computational challenges. Transformer-based models have made significant advances in this domain but are often limited by their quadratic complexity and suboptimal handling of long-range dependencies. In this work, we introduce GeneMamba, a scalable and efficient foundation model for single-cell transcriptomics built on state space modeling. Leveraging the Bi-Mamba architecture, GeneMamba captures bidirectional gene context with linear-time complexity, offering substantial computational gains over transformer baselines. The model is pretrained on nearly 30 million cells and incorporates biologically informed objectives, including pathway-aware contrastive loss and rank-based gene encoding. We evaluate GeneMamba across diverse tasks, including multi-batch integration, cell type annotation, and gene-gene correlation, demonstrating strong performance, interpretability, and robustness. These results position GeneMamba as a practical and powerful alternative to transformer-based methods, advancing the development of biologically grounded, scalable tools for large-scale single-cell data analysis. 

---
# HMI: Hierarchical Knowledge Management for Efficient Multi-Tenant Inference in Pretrained Language Models 

**Authors**: Jun Zhang, Jue Wang, Huan Li, Lidan Shou, Ke Chen, Gang Chen, Qin Xie, Guiming Xie, Xuejian Gong  

**Link**: [PDF](https://arxiv.org/pdf/2504.17449)  

**Abstract**: The significant computational demands of pretrained language models (PLMs), which often require dedicated hardware, present a substantial challenge in serving them efficiently, especially in multi-tenant environments. To address this, we introduce HMI, a Hierarchical knowledge management-based Multi-tenant Inference system, designed to manage tenants with distinct PLMs resource-efficiently. Our approach is three-fold: Firstly, we categorize PLM knowledge into general, domain-specific, and task-specific. Leveraging insights on knowledge acquisition across different model layers, we construct hierarchical PLMs (hPLMs) by extracting and storing knowledge at different levels, significantly reducing GPU memory usage per tenant. Secondly, we establish hierarchical knowledge management for hPLMs generated by various tenants in HMI. We manage domain-specific knowledge with acceptable storage increases by constructing and updating domain-specific knowledge trees based on frequency. We manage task-specific knowledge within limited GPU memory through parameter swapping. Finally, we propose system optimizations to enhance resource utilization and inference throughput. These include fine-grained pipelining via hierarchical knowledge prefetching to overlap CPU and I/O operations with GPU computations, and optimizing parallel implementations with batched matrix multiplications. Our experimental results demonstrate that the proposed HMI can efficiently serve up to 10,000 hPLMs (hBERTs and hGPTs) on a single GPU, with only a negligible compromise in accuracy. 

---
# TimeSoccer: An End-to-End Multimodal Large Language Model for Soccer Commentary Generation 

**Authors**: Ling You, Wenxuan Huang, Xinni Xie, Xiangyi Wei, Bangyan Li, Shaohui Lin, Yang Li, Changbo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.17365)  

**Abstract**: Soccer is a globally popular sporting event, typically characterized by long matches and distinctive highlight moments. Recent advances in Multimodal Large Language Models (MLLMs) offer promising capabilities in temporal grounding and video understanding, soccer commentary generation often requires precise temporal localization and semantically rich descriptions over long-form video. However, existing soccer MLLMs often rely on the temporal a priori for caption generation, so they cannot process the soccer video end-to-end. While some traditional approaches follow a two-step paradigm that is complex and fails to capture the global context to achieve suboptimal performance. To solve the above issues, we present TimeSoccer, the first end-to-end soccer MLLM for Single-anchor Dense Video Captioning (SDVC) in full-match soccer videos. TimeSoccer jointly predicts timestamps and generates captions in a single pass, enabling global context modeling across 45-minute matches. To support long video understanding of soccer matches, we introduce MoFA-Select, a training-free, motion-aware frame compression module that adaptively selects representative frames via a coarse-to-fine strategy, and incorporates complementary training paradigms to strengthen the model's ability to handle long temporal sequences. Extensive experiments demonstrate that our TimeSoccer achieves State-of-The-Art (SoTA) performance on the SDVC task in an end-to-end form, generating high-quality commentary with accurate temporal alignment and strong semantic relevance. 

---
# SCALAR: A Part-of-speech Tagger for Identifiers 

**Authors**: Christian D. Newman, Brandon Scholten, Sophia Testa, Joshua A. C. Behler, Syreen Banabilah, Michael L. Collard, Michael J. Decker, Mohamed Wiem Mkaouer, Marcos Zampieri, Eman Abdullah AlOmar, Reem Alsuhaibani, Anthony Peruma, Jonathan I. Maletic  

**Link**: [PDF](https://arxiv.org/pdf/2504.17038)  

**Abstract**: The paper presents the Source Code Analysis and Lexical Annotation Runtime (SCALAR), a tool specialized for mapping (annotating) source code identifier names to their corresponding part-of-speech tag sequence (grammar pattern). SCALAR's internal model is trained using scikit-learn's GradientBoostingClassifier in conjunction with a manually-curated oracle of identifier names and their grammar patterns. This specializes the tagger to recognize the unique structure of the natural language used by developers to create all types of identifiers (e.g., function names, variable names etc.). SCALAR's output is compared with a previous version of the tagger, as well as a modern off-the-shelf part-of-speech tagger to show how it improves upon other taggers' output for annotating identifiers. The code is available on Github 

---
# (Im)possibility of Automated Hallucination Detection in Large Language Models 

**Authors**: Amin Karbasi, Omar Montasser, John Sous, Grigoris Velegkas  

**Link**: [PDF](https://arxiv.org/pdf/2504.17004)  

**Abstract**: Is automated hallucination detection possible? In this work, we introduce a theoretical framework to analyze the feasibility of automatically detecting hallucinations produced by large language models (LLMs). Inspired by the classical Gold-Angluin framework for language identification and its recent adaptation to language generation by Kleinberg and Mullainathan, we investigate whether an algorithm, trained on examples drawn from an unknown target language $K$ (selected from a countable collection) and given access to an LLM, can reliably determine whether the LLM's outputs are correct or constitute hallucinations.
First, we establish an equivalence between hallucination detection and the classical task of language identification. We prove that any hallucination detection method can be converted into a language identification method, and conversely, algorithms solving language identification can be adapted for hallucination detection. Given the inherent difficulty of language identification, this implies that hallucination detection is fundamentally impossible for most language collections if the detector is trained using only correct examples from the target language.
Second, we show that the use of expert-labeled feedback, i.e., training the detector with both positive examples (correct statements) and negative examples (explicitly labeled incorrect statements), dramatically changes this conclusion. Under this enriched training regime, automated hallucination detection becomes possible for all countable language collections.
These results highlight the essential role of expert-labeled examples in training hallucination detectors and provide theoretical support for feedback-based methods, such as reinforcement learning with human feedback (RLHF), which have proven critical for reliable LLM deployment. 

---
# A Desideratum for Conversational Agents: Capabilities, Challenges, and Future Directions 

**Authors**: Emre Can Acikgoz, Cheng Qian, Hongru Wang, Vardhan Dongre, Xiusi Chen, Heng Ji, Dilek Hakkani-Tür, Gokhan Tur  

**Link**: [PDF](https://arxiv.org/pdf/2504.16939)  

**Abstract**: Recent advances in Large Language Models (LLMs) have propelled conversational AI from traditional dialogue systems into sophisticated agents capable of autonomous actions, contextual awareness, and multi-turn interactions with users. Yet, fundamental questions about their capabilities, limitations, and paths forward remain open. This survey paper presents a desideratum for next-generation Conversational Agents - what has been achieved, what challenges persist, and what must be done for more scalable systems that approach human-level intelligence. To that end, we systematically analyze LLM-driven Conversational Agents by organizing their capabilities into three primary dimensions: (i) Reasoning - logical, systematic thinking inspired by human intelligence for decision making, (ii) Monitor - encompassing self-awareness and user interaction monitoring, and (iii) Control - focusing on tool utilization and policy following. Building upon this, we introduce a novel taxonomy by classifying recent work on Conversational Agents around our proposed desideratum. We identify critical research gaps and outline key directions, including realistic evaluations, long-term multi-turn reasoning skills, self-evolution capabilities, collaborative and multi-agent task completion, personalization, and proactivity. This work aims to provide a structured foundation, highlight existing limitations, and offer insights into potential future research directions for Conversational Agents, ultimately advancing progress toward Artificial General Intelligence (AGI). We maintain a curated repository of papers at: this https URL. 

---
