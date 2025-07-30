# DeepSieve: Information Sieving via LLM-as-a-Knowledge-Router 

**Authors**: Minghao Guo, Qingcheng Zeng, Xujiang Zhao, Yanchi Liu, Wenchao Yu, Mengnan Du, Haifeng Chen, Wei Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2507.22050)  

**Abstract**: Large Language Models (LLMs) excel at many reasoning tasks but struggle with knowledge-intensive queries due to their inability to dynamically access up-to-date or domain-specific information. Retrieval-Augmented Generation (RAG) has emerged as a promising solution, enabling LLMs to ground their responses in external sources. However, existing RAG methods lack fine-grained control over both the query and source sides, often resulting in noisy retrieval and shallow reasoning. In this work, we introduce DeepSieve, an agentic RAG framework that incorporates information sieving via LLM-as-a-knowledge-router. DeepSieve decomposes complex queries into structured sub-questions and recursively routes each to the most suitable knowledge source, filtering irrelevant information through a multi-stage distillation process. Our design emphasizes modularity, transparency, and adaptability, leveraging recent advances in agentic system design. Experiments on multi-hop QA tasks across heterogeneous sources demonstrate improved reasoning depth, retrieval precision, and interpretability over conventional RAG approaches. 

---
# Predicting Microbial Ontology and Pathogen Risk from Environmental Metadata with Large Language Models 

**Authors**: Hyunwoo Yoo, Gail L. Rosen  

**Link**: [PDF](https://arxiv.org/pdf/2507.21980)  

**Abstract**: Traditional machine learning models struggle to generalize in microbiome studies where only metadata is available, especially in small-sample settings or across studies with heterogeneous label formats. In this work, we explore the use of large language models (LLMs) to classify microbial samples into ontology categories such as EMPO 3 and related biological labels, as well as to predict pathogen contamination risk, specifically the presence of E. Coli, using environmental metadata alone. We evaluate LLMs such as ChatGPT-4o, Claude 3.7 Sonnet, Grok-3, and LLaMA 4 in zero-shot and few-shot settings, comparing their performance against traditional models like Random Forests across multiple real-world datasets. Our results show that LLMs not only outperform baselines in ontology classification, but also demonstrate strong predictive ability for contamination risk, generalizing across sites and metadata distributions. These findings suggest that LLMs can effectively reason over sparse, heterogeneous biological metadata and offer a promising metadata-only approach for environmental microbiology and biosurveillance applications. 

---
# Culinary Crossroads: A RAG Framework for Enhancing Diversity in Cross-Cultural Recipe Adaptation 

**Authors**: Tianyi Hu, Andrea Morales-Garzón, Jingyi Zheng, Maria Maistro, Daniel Hershcovich  

**Link**: [PDF](https://arxiv.org/pdf/2507.21934)  

**Abstract**: In cross-cultural recipe adaptation, the goal is not only to ensure cultural appropriateness and retain the original dish's essence, but also to provide diverse options for various dietary needs and preferences. Retrieval Augmented Generation (RAG) is a promising approach, combining the retrieval of real recipes from the target cuisine for cultural adaptability with large language models (LLMs) for relevance. However, it remains unclear whether RAG can generate diverse adaptation results. Our analysis shows that RAG tends to overly rely on a limited portion of the context across generations, failing to produce diverse outputs even when provided with varied contextual inputs. This reveals a key limitation of RAG in creative tasks with multiple valid answers: it fails to leverage contextual diversity for generating varied responses. To address this issue, we propose CARRIAGE, a plug-and-play RAG framework for cross-cultural recipe adaptation that enhances diversity in both retrieval and context organization. To our knowledge, this is the first RAG framework that explicitly aims to generate highly diverse outputs to accommodate multiple user preferences. Our experiments show that CARRIAGE achieves Pareto efficiency in terms of diversity and quality of recipe adaptation compared to closed-book LLMs. 

---
# Post-Training Large Language Models via Reinforcement Learning from Self-Feedback 

**Authors**: Carel van Niekerk, Renato Vukovic, Benjamin Matthias Ruppik, Hsien-chin Lin, Milica Gašić  

**Link**: [PDF](https://arxiv.org/pdf/2507.21931)  

**Abstract**: Large Language Models (LLMs) often produce plausible but poorly-calibrated answers, limiting their reliability on reasoning-intensive tasks. We present Reinforcement Learning from Self-Feedback (RLSF), a post-training stage that uses the model's own confidence as an intrinsic reward, mimicking how humans learn in the absence of external feedback. After a frozen LLM generates several chain-of-thought solutions, we define and compute the confidence of each final answer span and rank the traces accordingly. These synthetic preferences are then used to fine-tune the policy with standard preference optimization, similar to RLHF yet requiring no human labels, gold answers, or externally curated rewards.
RLSF simultaneously (i) refines the model's probability estimates -- restoring well-behaved calibration -- and (ii) strengthens step-by-step reasoning, yielding improved performance on arithmetic reasoning and multiple-choice question answering.
By turning a model's own uncertainty into useful self-feedback, RLSF affirms reinforcement learning on intrinsic model behaviour as a principled and data-efficient component of the LLM post-training pipeline and warrents further research in intrinsic rewards for LLM post-training. 

---
# Training language models to be warm and empathetic makes them less reliable and more sycophantic 

**Authors**: Lujain Ibrahim, Franziska Sofia Hafner, Luc Rocher  

**Link**: [PDF](https://arxiv.org/pdf/2507.21919)  

**Abstract**: Artificial intelligence (AI) developers are increasingly building language models with warm and empathetic personas that millions of people now use for advice, therapy, and companionship. Here, we show how this creates a significant trade-off: optimizing language models for warmth undermines their reliability, especially when users express vulnerability. We conducted controlled experiments on five language models of varying sizes and architectures, training them to produce warmer, more empathetic responses, then evaluating them on safety-critical tasks. Warm models showed substantially higher error rates (+10 to +30 percentage points) than their original counterparts, promoting conspiracy theories, providing incorrect factual information, and offering problematic medical advice. They were also significantly more likely to validate incorrect user beliefs, particularly when user messages expressed sadness. Importantly, these effects were consistent across different model architectures, and occurred despite preserved performance on standard benchmarks, revealing systematic risks that current evaluation practices may fail to detect. As human-like AI systems are deployed at an unprecedented scale, our findings indicate a need to rethink how we develop and oversee these systems that are reshaping human relationships and social interaction. 

---
# Rote Learning Considered Useful: Generalizing over Memorized Data in LLMs 

**Authors**: Qinyuan Wu, Soumi Das, Mahsa Amani, Bishwamittra Ghosh, Mohammad Aflah Khan, Krishna P. Gummadi, Muhammad Bilal Zafar  

**Link**: [PDF](https://arxiv.org/pdf/2507.21914)  

**Abstract**: Rote learning is a memorization technique based on repetition. It is commonly believed to hinder generalization by encouraging verbatim memorization rather than deeper understanding. This insight holds for even learning factual knowledge that inevitably requires a certain degree of memorization. In this work, we demonstrate that LLMs can be trained to generalize from rote memorized data. We introduce a two-phase memorize-then-generalize framework, where the model first rote memorizes factual subject-object associations using a semantically meaningless token and then learns to generalize by fine-tuning on a small set of semantically meaningful prompts. Extensive experiments over 8 LLMs show that the models can reinterpret rote memorized data through the semantically meaningful prompts, as evidenced by the emergence of structured, semantically aligned latent representations between the two. This surprising finding opens the door to both effective and efficient knowledge injection and possible risks of repurposing the memorized data for malicious usage. 

---
# Graph-R1: Towards Agentic GraphRAG Framework via End-to-end Reinforcement Learning 

**Authors**: Haoran Luo, Haihong E, Guanting Chen, Qika Lin, Yikai Guo, Fangzhi Xu, Zemin Kuang, Meina Song, Xiaobao Wu, Yifan Zhu, Luu Anh Tuan  

**Link**: [PDF](https://arxiv.org/pdf/2507.21892)  

**Abstract**: Retrieval-Augmented Generation (RAG) mitigates hallucination in LLMs by incorporating external knowledge, but relies on chunk-based retrieval that lacks structural semantics. GraphRAG methods improve RAG by modeling knowledge as entity-relation graphs, but still face challenges in high construction cost, fixed one-time retrieval, and reliance on long-context reasoning and prompt design. To address these challenges, we propose Graph-R1, an agentic GraphRAG framework via end-to-end reinforcement learning (RL). It introduces lightweight knowledge hypergraph construction, models retrieval as a multi-turn agent-environment interaction, and optimizes the agent process via an end-to-end reward mechanism. Experiments on standard RAG datasets show that Graph-R1 outperforms traditional GraphRAG and RL-enhanced RAG methods in reasoning accuracy, retrieval efficiency, and generation quality. 

---
# AutoTIR: Autonomous Tools Integrated Reasoning via Reinforcement Learning 

**Authors**: Yifan Wei, Xiaoyan Yu, Yixuan Weng, Tengfei Pan, Angsheng Li, Li Du  

**Link**: [PDF](https://arxiv.org/pdf/2507.21836)  

**Abstract**: Large Language Models (LLMs), when enhanced through reasoning-oriented post-training, evolve into powerful Large Reasoning Models (LRMs). Tool-Integrated Reasoning (TIR) further extends their capabilities by incorporating external tools, but existing methods often rely on rigid, predefined tool-use patterns that risk degrading core language competence. Inspired by the human ability to adaptively select tools, we introduce AutoTIR, a reinforcement learning framework that enables LLMs to autonomously decide whether and which tool to invoke during the reasoning process, rather than following static tool-use strategies. AutoTIR leverages a hybrid reward mechanism that jointly optimizes for task-specific answer correctness, structured output adherence, and penalization of incorrect tool usage, thereby encouraging both precise reasoning and efficient tool integration. Extensive evaluations across diverse knowledge-intensive, mathematical, and general language modeling tasks demonstrate that AutoTIR achieves superior overall performance, significantly outperforming baselines and exhibits superior generalization in tool-use behavior. These results highlight the promise of reinforcement learning in building truly generalizable and scalable TIR capabilities in LLMs. The code and data are available at this https URL. 

---
# Introducing HALC: A general pipeline for finding optimal prompting strategies for automated coding with LLMs in the computational social sciences 

**Authors**: Andreas Reich, Claudia Thoms, Tobias Schrimpf  

**Link**: [PDF](https://arxiv.org/pdf/2507.21831)  

**Abstract**: LLMs are seeing widespread use for task automation, including automated coding in the social sciences. However, even though researchers have proposed different prompting strategies, their effectiveness varies across LLMs and tasks. Often trial and error practices are still widespread. We propose HALC$-$a general pipeline that allows for the systematic and reliable construction of optimal prompts for any given coding task and model, permitting the integration of any prompting strategy deemed relevant. To investigate LLM coding and validate our pipeline, we sent a total of 1,512 individual prompts to our local LLMs in over two million requests. We test prompting strategies and LLM task performance based on few expert codings (ground truth). When compared to these expert codings, we find prompts that code reliably for single variables (${\alpha}$climate = .76; ${\alpha}$movement = .78) and across two variables (${\alpha}$climate = .71; ${\alpha}$movement = .74) using the LLM Mistral NeMo. Our prompting strategies are set up in a way that aligns the LLM to our codebook$-$we are not optimizing our codebook for LLM friendliness. Our paper provides insights into the effectiveness of different prompting strategies, crucial influencing factors, and the identification of reliable prompts for each coding task and model. 

---
# Modelling Adjectival Modification Effects on Semantic Plausibility 

**Authors**: Anna Golub, Beate Zywietz, Annerose Eichel  

**Link**: [PDF](https://arxiv.org/pdf/2507.21828)  

**Abstract**: While the task of assessing the plausibility of events such as ''news is relevant'' has been addressed by a growing body of work, less attention has been paid to capturing changes in plausibility as triggered by event modification. Understanding changes in plausibility is relevant for tasks such as dialogue generation, commonsense reasoning, and hallucination detection as it allows to correctly model, for example, ''gentle sarcasm'' as a sign of closeness rather than unkindness among friends [9]. In this work, we tackle the ADEPT challenge benchmark [6] consisting of 16K English sentence pairs differing by exactly one adjectival modifier. Our modeling experiments provide a conceptually novel method by using sentence transformers, and reveal that both they and transformer-based models struggle with the task at hand, and sentence transformers - despite their conceptual alignment with the task - even under-perform in comparison to models like RoBERTa. Furthermore, an in-depth comparison with prior work highlights the importance of a more realistic, balanced evaluation method: imbalances distort model performance and evaluation metrics, and weaken result trustworthiness. 

---
# HRIPBench: Benchmarking LLMs in Harm Reduction Information Provision to Support People Who Use Drugs 

**Authors**: Kaixuan Wang, Chenxin Diao, Jason T. Jacques, Zhongliang Guo, Shuai Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2507.21815)  

**Abstract**: Millions of individuals' well-being are challenged by the harms of substance use. Harm reduction as a public health strategy is designed to improve their health outcomes and reduce safety risks. Some large language models (LLMs) have demonstrated a decent level of medical knowledge, promising to address the information needs of people who use drugs (PWUD). However, their performance in relevant tasks remains largely unexplored. We introduce HRIPBench, a benchmark designed to evaluate LLM's accuracy and safety risks in harm reduction information provision. The benchmark dataset HRIP-Basic has 2,160 question-answer-evidence pairs. The scope covers three tasks: checking safety boundaries, providing quantitative values, and inferring polysubstance use risks. We build the Instruction and RAG schemes to evaluate model behaviours based on their inherent knowledge and the integration of domain knowledge. Our results indicate that state-of-the-art LLMs still struggle to provide accurate harm reduction information, and sometimes, carry out severe safety risks to PWUD. The use of LLMs in harm reduction contexts should be cautiously constrained to avoid inducing negative health outcomes. WARNING: This paper contains illicit content that potentially induces harms. 

---
# Overview of ADoBo at IberLEF 2025: Automatic Detection of Anglicisms in Spanish 

**Authors**: Elena Alvarez-Mellado, Jordi Porta-Zamorano, Constantine Lignos, Julio Gonzalo  

**Link**: [PDF](https://arxiv.org/pdf/2507.21813)  

**Abstract**: This paper summarizes the main findings of ADoBo 2025, the shared task on anglicism identification in Spanish proposed in the context of IberLEF 2025. Participants of ADoBo 2025 were asked to detect English lexical borrowings (or anglicisms) from a collection of Spanish journalistic texts. Five teams submitted their solutions for the test phase. Proposed systems included LLMs, deep learning models, Transformer-based models and rule-based systems. The results range from F1 scores of 0.17 to 0.99, which showcases the variability in performance different systems can have for this task. 

---
# ChartMark: A Structured Grammar for Chart Annotation 

**Authors**: Yiyu Chen, Yifan Wu, Shuyu Shen, Yupeng Xie, Leixian Shen, Hui Xiong, Yuyu Luo  

**Link**: [PDF](https://arxiv.org/pdf/2507.21810)  

**Abstract**: Chart annotations enhance visualization accessibility but suffer from fragmented, non-standardized representations that limit cross-platform reuse. We propose ChartMark, a structured grammar that separates annotation semantics from visualization implementations. ChartMark features a hierarchical framework mapping onto annotation dimensions (e.g., task, chart context), supporting both abstract intents and precise visual details. Our toolkit demonstrates converting ChartMark specifications into Vega-Lite visualizations, highlighting its flexibility, expressiveness, and practical applicability. 

---
# The Problem with Safety Classification is not just the Models 

**Authors**: Sowmya Vajjala  

**Link**: [PDF](https://arxiv.org/pdf/2507.21782)  

**Abstract**: Studying the robustness of Large Language Models (LLMs) to unsafe behaviors is an important topic of research today. Building safety classification models or guard models, which are fine-tuned models for input/output safety classification for LLMs, is seen as one of the solutions to address the issue. Although there is a lot of research on the safety testing of LLMs themselves, there is little research on evaluating the effectiveness of such safety classifiers or the evaluation datasets used for testing them, especially in multilingual scenarios. In this position paper, we demonstrate how multilingual disparities exist in 5 safety classification models by considering datasets covering 18 languages. At the same time, we identify potential issues with the evaluation datasets, arguing that the shortcomings of current safety classifiers are not only because of the models themselves. We expect that these findings will contribute to the discussion on developing better methods to identify harmful content in LLM inputs across languages. 

---
# AgriEval: A Comprehensive Chinese Agricultural Benchmark for Large Language Models 

**Authors**: Lian Yan, Haotian Wang, Chen Tang, Haifeng Liu, Tianyang Sun, Liangliang Liu, Yi Guan, Jingchi Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2507.21773)  

**Abstract**: In the agricultural domain, the deployment of large language models (LLMs) is hindered by the lack of training data and evaluation benchmarks. To mitigate this issue, we propose AgriEval, the first comprehensive Chinese agricultural benchmark with three main characteristics: (1) Comprehensive Capability Evaluation. AgriEval covers six major agriculture categories and 29 subcategories within agriculture, addressing four core cognitive scenarios: memorization, understanding, inference, and generation. (2) High-Quality Data. The dataset is curated from university-level examinations and assignments, providing a natural and robust benchmark for assessing the capacity of LLMs to apply knowledge and make expert-like decisions. (3) Diverse Formats and Extensive Scale. AgriEval comprises 14,697 multiple-choice questions and 2,167 open-ended question-and-answer questions, establishing it as the most extensive agricultural benchmark available to date. We also present comprehensive experimental results over 51 open-source and commercial LLMs. The experimental results reveal that most existing LLMs struggle to achieve 60% accuracy, underscoring the developmental potential in agricultural LLMs. Additionally, we conduct extensive experiments to investigate factors influencing model performance and propose strategies for enhancement. AgriEval is available at this https URL. 

---
# Adversarial Defence without Adversarial Defence: Enhancing Language Model Robustness via Instance-level Principal Component Removal 

**Authors**: Yang Wang, Chenghao Xiao, Yizhi Li, Stuart E. Middleton, Noura Al Moubayed, Chenghua Lin  

**Link**: [PDF](https://arxiv.org/pdf/2507.21750)  

**Abstract**: Pre-trained language models (PLMs) have driven substantial progress in natural language processing but remain vulnerable to adversarial attacks, raising concerns about their robustness in real-world applications. Previous studies have sought to mitigate the impact of adversarial attacks by introducing adversarial perturbations into the training process, either implicitly or explicitly. While both strategies enhance robustness, they often incur high computational costs. In this work, we propose a simple yet effective add-on module that enhances the adversarial robustness of PLMs by removing instance-level principal components, without relying on conventional adversarial defences or perturbing the original training data. Our approach transforms the embedding space to approximate Gaussian properties, thereby reducing its susceptibility to adversarial perturbations while preserving semantic relationships. This transformation aligns embedding distributions in a way that minimises the impact of adversarial noise on decision boundaries, enhancing robustness without requiring adversarial examples or costly training-time augmentation. Evaluations on eight benchmark datasets show that our approach improves adversarial robustness while maintaining comparable before-attack accuracy to baselines, achieving a balanced trade-off between robustness and generalisation. 

---
# UnsafeChain: Enhancing Reasoning Model Safety via Hard Cases 

**Authors**: Raj Vardhan Tomar, Preslav Nakov, Yuxia Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.21652)  

**Abstract**: As large reasoning models (LRMs) grow more capable, chain-of-thought (CoT) reasoning introduces new safety challenges. Existing SFT-based safety alignment studies dominantly focused on filtering prompts with safe, high-quality responses, while overlooking hard prompts that always elicit harmful outputs. To fill this gap, we introduce UnsafeChain, a safety alignment dataset constructed from hard prompts with diverse sources, where unsafe completions are identified and explicitly corrected into safe responses. By exposing models to unsafe behaviors and guiding their correction, UnsafeChain enhances safety while preserving general reasoning ability. We fine-tune three LRMs on UnsafeChain and compare them against recent SafeChain and STAR-1 across six out-of-distribution and five in-distribution benchmarks. UnsafeChain consistently outperforms prior datasets, with even a 1K subset matching or surpassing baseline performance, demonstrating the effectiveness and generalizability of correction-based supervision. We release our dataset and code at this https URL 

---
# Libra: Assessing and Improving Reward Model by Learning to Think 

**Authors**: Meng Zhou, Bei Li, Jiahao Liu, Xiaowen Shi, Yang Bai, Rongxiang Weng, Jingang Wang, Xunliang Cai  

**Link**: [PDF](https://arxiv.org/pdf/2507.21645)  

**Abstract**: Reinforcement learning (RL) has significantly improved the reasoning ability of large language models. However, current reward models underperform in challenging reasoning scenarios and predominant RL training paradigms rely on rule-based or reference-based rewards, which impose two critical limitations: 1) the dependence on finely annotated reference answer to attain rewards; and 2) the requirement for constrained output format. These limitations fundamentally hinder further RL data scaling and sustained enhancement of model reasoning performance. To address these limitations, we propose a comprehensive framework for evaluating and improving the performance of reward models in complex reasoning scenarios. We first present a reasoning-oriented benchmark (Libra Bench), systematically constructed from a diverse collection of challenging mathematical problems and advanced reasoning models, to address the limitations of existing reward model benchmarks in reasoning scenarios. We further introduce a novel approach for improving the generative reward model via learning-to-think methodologies. Based on the proposed approach, we develop Libra-RM series, a collection of generative reward models with reasoning capabilities that achieve state-of-the-art results on various benchmarks. Comprehensive downstream experiments are conducted and the experimental results demonstrate the correlation between our Libra Bench and downstream application, and the potential of Libra-RM to further improve reasoning models with unlabeled data. 

---
# Multilingual JobBERT for Cross-Lingual Job Title Matching 

**Authors**: Jens-Joris Decorte, Matthias De Lange, Jeroen Van Hautte  

**Link**: [PDF](https://arxiv.org/pdf/2507.21609)  

**Abstract**: We introduce JobBERT-V3, a contrastive learning-based model for cross-lingual job title matching. Building on the state-of-the-art monolingual JobBERT-V2, our approach extends support to English, German, Spanish, and Chinese by leveraging synthetic translations and a balanced multilingual dataset of over 21 million job titles. The model retains the efficiency-focused architecture of its predecessor while enabling robust alignment across languages without requiring task-specific supervision. Extensive evaluations on the TalentCLEF 2025 benchmark demonstrate that JobBERT-V3 outperforms strong multilingual baselines and achieves consistent performance across both monolingual and cross-lingual settings. While not the primary focus, we also show that the model can be effectively used to rank relevant skills for a given job title, demonstrating its broader applicability in multilingual labor market intelligence. The model is publicly available: this https URL. 

---
# Multi-Hypothesis Distillation of Multilingual Neural Translation Models for Low-Resource Languages 

**Authors**: Aarón Galiano-Jiménez, Juan Antonio Pérez-Ortiz, Felipe Sánchez-Martínez, Víctor M. Sánchez-Cartagena  

**Link**: [PDF](https://arxiv.org/pdf/2507.21568)  

**Abstract**: This paper explores sequence-level knowledge distillation (KD) of multilingual pre-trained encoder-decoder translation models. We argue that the teacher model's output distribution holds valuable insights for the student, beyond the approximated mode obtained through beam search (the standard decoding method), and present Multi-Hypothesis Distillation (MHD), a sequence-level KD method that generates multiple translations for each source sentence. This provides a larger representation of the teacher model distribution and exposes the student model to a wider range of target-side prefixes. We leverage $n$-best lists from beam search to guide the student's learning and examine alternative decoding methods to address issues like low variability and the under-representation of infrequent tokens. For low-resource languages, our research shows that while sampling methods may slightly compromise translation quality compared to beam search based approaches, they enhance the generated corpora with greater variability and lexical richness. This ultimately improves student model performance and mitigates the gender bias amplification often associated with KD. 

---
# Evaluating the cognitive reality of Spanish irregular morphomic patterns: Humans vs. Transformers 

**Authors**: Akhilesh Kakolu Ramarao, Kevin Tang, Dinah Baer-Henney  

**Link**: [PDF](https://arxiv.org/pdf/2507.21556)  

**Abstract**: This study investigates the cognitive plausibility of the Spanish irregular morphomic pattern by directly comparing transformer-based neural networks to human behavioral data from \citet{Nevins2015TheRA}. Using the same analytical framework as the original human study, we evaluate whether transformer models can replicate human-like sensitivity to a complex linguistic phenomena, the morphome, under controlled input conditions. Our experiments focus on three frequency conditions: natural, low-frequency, and high-frequency distributions of verbs exhibiting irregular morphomic patterns. While the models outperformed humans in stem and suffix accuracy, a clear divergence emerged in response preferences. Unlike humans, who consistently favored natural responses across all test items, models' preferred irregular responses and were influenced by the proportion of irregular verbs in their training data. Additionally, models trained on the natural and low-frequency distributions, but not the high-frequency distribution, were sensitive to the phonological similarity between test items and real Spanish L-shaped verbs. 

---
# MAGIC: A Multi-Hop and Graph-Based Benchmark for Inter-Context Conflicts in Retrieval-Augmented Generation 

**Authors**: Jungyeon Lee, Kangmin Lee, Taeuk Kim  

**Link**: [PDF](https://arxiv.org/pdf/2507.21544)  

**Abstract**: Knowledge conflict often arises in retrieval-augmented generation (RAG) systems, where retrieved documents may be inconsistent with one another or contradict the model's parametric knowledge. Existing benchmarks for investigating the phenomenon have notable limitations, including a narrow focus on the question answering setup, heavy reliance on entity substitution techniques, and a restricted range of conflict types. To address these issues, we propose a knowledge graph (KG)-based framework that generates varied and subtle conflicts between two similar yet distinct contexts, while ensuring interpretability through the explicit relational structure of KGs. Experimental results on our benchmark, MAGIC, provide intriguing insights into the inner workings of LLMs regarding knowledge conflict: both open-source and proprietary models struggle with conflict detection -- especially when multi-hop reasoning is required -- and often fail to pinpoint the exact source of contradictions. Finally, we present in-depth analyses that serve as a foundation for improving LLMs in integrating diverse, sometimes even conflicting, information. 

---
# Modern Uyghur Dependency Treebank (MUDT): An Integrated Morphosyntactic Framework for a Low-Resource Language 

**Authors**: Jiaxin Zuo, Yiquan Wang, Yuan Pan, Xiadiya Yibulayin  

**Link**: [PDF](https://arxiv.org/pdf/2507.21536)  

**Abstract**: To address a critical resource gap in Uyghur Natural Language Processing (NLP), this study introduces a dependency annotation framework designed to overcome the limitations of existing treebanks for the low-resource, agglutinative language. This inventory includes 18 main relations and 26 subtypes, with specific labels such as cop:zero for verbless clauses and instr:case=loc/dat for nuanced instrumental functions. To empirically validate the necessity of this tailored approach, we conducted a cross-standard evaluation using a pre-trained Universal Dependencies parser. The analysis revealed a systematic 47.9% divergence in annotations, pinpointing the inadequacy of universal schemes for handling Uyghur-specific structures. Grounded in nine annotation principles that ensure typological accuracy and semantic transparency, the Modern Uyghur Dependency Treebank (MUDT) provides a more accurate and semantically transparent representation, designed to enable significant improvements in parsing and downstream NLP tasks, and offers a replicable model for other morphologically complex languages. 

---
# Automatic Classification of User Requirements from Online Feedback -- A Replication Study 

**Authors**: Meet Bhatt, Nic Boilard, Muhammad Rehan Chaudhary, Cole Thompson, Jacob Idoko, Aakash Sorathiya, Gouri Ginde  

**Link**: [PDF](https://arxiv.org/pdf/2507.21532)  

**Abstract**: Natural language processing (NLP) techniques have been widely applied in the requirements engineering (RE) field to support tasks such as classification and ambiguity detection. Although RE research is rooted in empirical investigation, it has paid limited attention to replicating NLP for RE (NLP4RE) studies. The rapidly advancing realm of NLP is creating new opportunities for efficient, machine-assisted workflows, which can bring new perspectives and results to the forefront. Thus, we replicate and extend a previous NLP4RE study (baseline), "Classifying User Requirements from Online Feedback in Small Dataset Environments using Deep Learning", which evaluated different deep learning models for requirement classification from user reviews. We reproduced the original results using publicly released source code, thereby helping to strengthen the external validity of the baseline study. We then extended the setup by evaluating model performance on an external dataset and comparing results to a GPT-4o zero-shot classifier. Furthermore, we prepared the replication study ID-card for the baseline study, important for evaluating replication readiness. Results showed diverse reproducibility levels across different models, with Naive Bayes demonstrating perfect reproducibility. In contrast, BERT and other models showed mixed results. Our findings revealed that baseline deep learning models, BERT and ELMo, exhibited good generalization capabilities on an external dataset, and GPT-4o showed performance comparable to traditional baseline machine learning models. Additionally, our assessment confirmed the baseline study's replication readiness; however missing environment setup files would have further enhanced readiness. We include this missing information in our replication package and provide the replication study ID-card for our study to further encourage and support the replication of our study. 

---
# TriangleMix: A Lossless and Efficient Attention Pattern for Long Context Prefilling 

**Authors**: Zhiyuan He, Yike Zhang, Chengruidong Zhang, Huiqiang Jiang, Yuqing Yang, Lili Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2507.21526)  

**Abstract**: Large Language Models (LLMs) rely on attention mechanisms whose time complexity grows quadratically with input sequence length, creating significant computational bottlenecks during the prefilling stage. Existing static sparse attention methods typically degrade accuracy, while dynamic sparsity methods introduce additional computational overhead due to runtime sparse index estimation. To address these limitations, we propose TriangleMix, a novel training-free static attention pattern. TriangleMix employs dense attention in shallow layers and switches to a triangle-shaped sparse pattern in deeper layers. Extensive experiments demonstrate that TriangleMix reduces attention overhead by 3.7x to 15.3x in deep layers, and decreases overall Time-to-First-Token (TTFT) by 12% to 32% for sequence lengths ranging from 32K to 128K, without sacrificing model accuracy. Moreover, TriangleMix can be seamlessly integrated with dynamic sparsity methods to achieve further speedup, e.g. accelerating MInference by 19% at 128K, highlighting its potential to enhance LLM inference efficiency. 

---
# Model-free Speculative Decoding for Transformer-based ASR with Token Map Drafting 

**Authors**: Tuan Vu Ho, Hiroaki Kokubo, Masaaki Yamamoto, Yohei Kawaguchi  

**Link**: [PDF](https://arxiv.org/pdf/2507.21522)  

**Abstract**: End-to-end automatic speech recognition (ASR) systems based on transformer architectures, such as Whisper, offer high transcription accuracy and robustness. However, their autoregressive decoding is computationally expensive, hence limiting deployment on CPU-based and resource-constrained devices. Speculative decoding (SD) mitigates this issue by using a smaller draft model to propose candidate tokens, which are then verified by the main model. However, this approach is impractical for devices lacking hardware accelerators like GPUs. To address this, we propose \emph{Token Map Drafting}, a model-free SD technique that eliminates the need for a separate draft model. Instead, we leverage a precomputed n-gram token map derived from domain-specific training data, enabling efficient speculative decoding with minimal overhead. Our method significantly accelerates ASR inference in structured, low-perplexity domains without sacrificing transcription accuracy. Experimental results demonstrate decoding speed-ups of $1.27\times$ on the CI-AVSR dataset and $1.37\times$ on our internal dataset without degrading recognition accuracy. Additionally, our approach achieves a $10\%$ absolute improvement in decoding speed over the Distill-spec baseline running on CPU, highlighting its effectiveness for on-device ASR applications. 

---
# Persona Vectors: Monitoring and Controlling Character Traits in Language Models 

**Authors**: Runjin Chen, Andy Arditi, Henry Sleight, Owain Evans, Jack Lindsey  

**Link**: [PDF](https://arxiv.org/pdf/2507.21509)  

**Abstract**: Large language models interact with users through a simulated 'Assistant' persona. While the Assistant is typically trained to be helpful, harmless, and honest, it sometimes deviates from these ideals. In this paper, we identify directions in the model's activation space-persona vectors-underlying several traits, such as evil, sycophancy, and propensity to hallucinate. We confirm that these vectors can be used to monitor fluctuations in the Assistant's personality at deployment time. We then apply persona vectors to predict and control personality shifts that occur during training. We find that both intended and unintended personality changes after finetuning are strongly correlated with shifts along the relevant persona vectors. These shifts can be mitigated through post-hoc intervention, or avoided in the first place with a new preventative steering method. Moreover, persona vectors can be used to flag training data that will produce undesirable personality changes, both at the dataset level and the individual sample level. Our method for extracting persona vectors is automated and can be applied to any personality trait of interest, given only a natural-language description. 

---
# VN-MTEB: Vietnamese Massive Text Embedding Benchmark 

**Authors**: Loc Pham, Tung Luu, Thu Vo, Minh Nguyen, Viet Hoang  

**Link**: [PDF](https://arxiv.org/pdf/2507.21500)  

**Abstract**: Vietnam ranks among the top countries in terms of both internet traffic and online toxicity. As a result, implementing embedding models for recommendation and content control duties in applications is crucial. However, a lack of large-scale test datasets, both in volume and task diversity, makes it tricky for scientists to effectively evaluate AI models before deploying them in real-world, large-scale projects. To solve this important problem, we introduce a Vietnamese benchmark, VN-MTEB for embedding models, which we created by translating a large number of English samples from the Massive Text Embedding Benchmark using our new automated framework. We leverage the strengths of large language models (LLMs) and cutting-edge embedding models to conduct translation and filtering processes to retain high-quality samples, guaranteeing a natural flow of language and semantic fidelity while preserving named entity recognition (NER) and code snippets. Our comprehensive benchmark consists of 41 datasets from six tasks specifically designed for Vietnamese text embeddings. In our analysis, we find that bigger and more complex models using Rotary Positional Embedding outperform those using Absolute Positional Embedding in embedding tasks. Datasets are available at HuggingFace: this https URL 

---
# Improving Task Diversity in Label Efficient Supervised Finetuning of LLMs 

**Authors**: Abhinav Arabelly, Jagrut Nemade, Robert D Nowak, Jifan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.21482)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities across diverse domains, but developing high-performing models for specialized applications often requires substantial human annotation -- a process that is time-consuming, labor-intensive, and expensive. In this paper, we address the label-efficient learning problem for supervised finetuning (SFT) by leveraging task-diversity as a fundamental principle for effective data selection. This is markedly different from existing methods based on the prompt-diversity. Our approach is based on two key observations: 1) task labels for different prompts are often readily available; 2) pre-trained models have significantly varying levels of confidence across tasks. We combine these facts to devise a simple yet effective sampling strategy: we select examples across tasks using an inverse confidence weighting strategy. This produces models comparable to or better than those trained with more complex sampling procedures, while being significantly easier to implement and less computationally intensive. Notably, our experimental results demonstrate that this method can achieve better accuracy than training on the complete dataset (a 4\% increase in MMLU score). Across various annotation budgets and two instruction finetuning datasets, our algorithm consistently performs at or above the level of the best existing methods, while reducing annotation costs by up to 80\%. 

---
# Which LLMs Get the Joke? Probing Non-STEM Reasoning Abilities with HumorBench 

**Authors**: Reuben Narad, Siddharth Suresh, Jiayi Chen, Pine S.L. Dysart-Bricken, Bob Mankoff, Robert Nowak, Jifan Zhang, Lalit Jain  

**Link**: [PDF](https://arxiv.org/pdf/2507.21476)  

**Abstract**: We present HumorBench, a benchmark designed to evaluate large language models' (LLMs) ability to reason about and explain sophisticated humor in cartoon captions. As reasoning models increasingly saturate existing benchmarks in mathematics and science, novel and challenging evaluations of model intelligence beyond STEM domains are essential. Reasoning is fundamentally involved in text-based humor comprehension, requiring the identification of connections between concepts in cartoons/captions and external cultural references, wordplays, and other mechanisms. HumorBench includes approximately 300 unique cartoon-caption pairs from the New Yorker Caption Contest and this http URL, with expert-annotated evaluation rubrics identifying essential joke elements. LLMs are evaluated based on their explanations towards the humor and abilities in identifying the joke elements. To perform well on this task, models must form and test hypotheses about associations between concepts, potentially backtracking from initial interpretations to arrive at the most plausible explanation. Our extensive benchmarking of current SOTA models reveals three key insights: (1) LLM progress on STEM reasoning transfers effectively to humor comprehension; (2) models trained exclusively on STEM reasoning data still perform well on HumorBench, demonstrating strong transferability of reasoning abilities; and (3) test-time scaling by increasing thinking token budgets yields mixed results across different models in humor reasoning. 

---
# Towards Locally Deployable Fine-Tuned Causal Large Language Models for Mode Choice Behaviour 

**Authors**: Tareq Alsaleh, Bilal Farooq  

**Link**: [PDF](https://arxiv.org/pdf/2507.21432)  

**Abstract**: This study investigates the adoption of open-access, locally deployable causal large language models (LLMs) for travel mode choice prediction and introduces LiTransMC, the first fine-tuned causal LLM developed for this task. We systematically benchmark eleven LLMs (1-12B parameters) across three stated and revealed preference datasets, testing 396 configurations and generating over 79,000 synthetic commuter predictions. Beyond predictive accuracy, we evaluate models generated reasoning using BERTopic for topic modelling and a novel Explanation Strength Index, providing the first structured analysis of how LLMs articulate decision factors in alignment with behavioural theory. LiTransMC, fine-tuned using parameter efficient and loss masking strategy, achieved a weighted F1 score of 0.6845 and a Jensen-Shannon Divergence of 0.000245, surpassing both untuned local models and larger proprietary systems, including GPT-4o with advanced persona inference and embedding-based loading, while also outperforming classical mode choice methods such as discrete choice models and machine learning classifiers for the same dataset. This dual improvement, i.e., high instant-level accuracy and near-perfect distributional calibration, demonstrates the feasibility of creating specialist, locally deployable LLMs that integrate prediction and interpretability. Through combining structured behavioural prediction with natural language reasoning, this work unlocks the potential for conversational, multi-task transport models capable of supporting agent-based simulations, policy testing, and behavioural insight generation. These findings establish a pathway for transforming general purpose LLMs into specialized, explainable tools for transportation research and policy formulation, while maintaining privacy, reducing cost, and broadening access through local deployment. 

---
# MemTool: Optimizing Short-Term Memory Management for Dynamic Tool Calling in LLM Agent Multi-Turn Conversations 

**Authors**: Elias Lumer, Anmol Gulati, Vamse Kumar Subbiah, Pradeep Honaganahalli Basavaraju, James A. Burke  

**Link**: [PDF](https://arxiv.org/pdf/2507.21428)  

**Abstract**: Large Language Model (LLM) agents have shown significant autonomous capabilities in dynamically searching and incorporating relevant tools or Model Context Protocol (MCP) servers for individual queries. However, fixed context windows limit effectiveness in multi-turn interactions requiring repeated, independent tool usage. We introduce MemTool, a short-term memory framework enabling LLM agents to dynamically manage tools or MCP server contexts across multi-turn conversations. MemTool offers three agentic architectures: 1) Autonomous Agent Mode, granting full tool management autonomy, 2) Workflow Mode, providing deterministic control without autonomy, and 3) Hybrid Mode, combining autonomous and deterministic control. Evaluating each MemTool mode across 13+ LLMs on the ScaleMCP benchmark, we conducted experiments over 100 consecutive user interactions, measuring tool removal ratios (short-term memory efficiency) and task completion accuracy. In Autonomous Agent Mode, reasoning LLMs achieve high tool-removal efficiency (90-94% over a 3-window average), while medium-sized models exhibit significantly lower efficiency (0-60%). Workflow and Hybrid modes consistently manage tool removal effectively, whereas Autonomous and Hybrid modes excel at task completion. We present trade-offs and recommendations for each MemTool mode based on task accuracy, agency, and model capabilities. 

---
# Turbocharging Web Automation: The Impact of Compressed History States 

**Authors**: Xiyue Zhu, Peng Tang, Haofu Liao, Srikar Appalaraju  

**Link**: [PDF](https://arxiv.org/pdf/2507.21369)  

**Abstract**: Language models have led to a leap forward in web automation. The current web automation approaches take the current web state, history actions, and language instruction as inputs to predict the next action, overlooking the importance of history states. However, the highly verbose nature of web page states can result in long input sequences and sparse information, hampering the effective utilization of history states. In this paper, we propose a novel web history compressor approach to turbocharge web automation using history states. Our approach employs a history compressor module that distills the most task-relevant information from each history state into a fixed-length short representation, mitigating the challenges posed by the highly verbose history states. Experiments are conducted on the Mind2Web and WebLINX datasets to evaluate the effectiveness of our approach. Results show that our approach obtains 1.2-5.4% absolute accuracy improvements compared to the baseline approach without history inputs. 

---
# StructText: A Synthetic Table-to-Text Approach for Benchmark Generation with Multi-Dimensional Evaluation 

**Authors**: Satyananda Kashyap, Sola Shirai, Nandana Mihindukulasooriya, Horst Samulowitz  

**Link**: [PDF](https://arxiv.org/pdf/2507.21340)  

**Abstract**: Extracting structured information from text, such as key-value pairs that could augment tabular data, is quite useful in many enterprise use cases. Although large language models (LLMs) have enabled numerous automated pipelines for converting natural language into structured formats, there is still a lack of benchmarks for evaluating their extraction quality, especially in specific domains or focused documents specific to a given organization. Building such benchmarks by manual annotations is labour-intensive and limits the size and scalability of the benchmarks. In this work, we present StructText, an end-to-end framework for automatically generating high-fidelity benchmarks for key-value extraction from text using existing tabular data. It uses available tabular data as structured ground truth, and follows a two-stage ``plan-then-execute'' pipeline to synthetically generate corresponding natural-language text. To ensure alignment between text and structured source, we introduce a multi-dimensional evaluation strategy that combines (a) LLM-based judgments on factuality, hallucination, and coherence and (b) objective extraction metrics measuring numeric and temporal accuracy. We evaluated the proposed method on 71,539 examples across 49 datasets. Results reveal that while LLMs achieve strong factual accuracy and avoid hallucination, they struggle with narrative coherence in producing extractable text. Notably, models presume numerical and temporal information with high fidelity yet this information becomes embedded in narratives that resist automated extraction. We release a framework, including datasets, evaluation tools, and baseline extraction systems, to support continued research. 

---
# A Deep Learning Automatic Speech Recognition Model for Shona Language 

**Authors**: Leslie Wellington Sirora, Mainford Mutandavari  

**Link**: [PDF](https://arxiv.org/pdf/2507.21331)  

**Abstract**: This study presented the development of a deep learning-based Automatic Speech Recognition system for Shona, a low-resource language characterized by unique tonal and grammatical complexities. The research aimed to address the challenges posed by limited training data, lack of labelled data, and the intricate tonal nuances present in Shona speech, with the objective of achieving significant improvements in recognition accuracy compared to traditional statistical models. The research first explored the feasibility of using deep learning to develop an accurate ASR system for Shona. Second, it investigated the specific challenges involved in designing and implementing deep learning architectures for Shona speech recognition and proposed strategies to mitigate these challenges. Lastly, it compared the performance of the deep learning-based model with existing statistical models in terms of accuracy. The developed ASR system utilized a hybrid architecture consisting of a Convolutional Neural Network for acoustic modelling and a Long Short-Term Memory network for language modelling. To overcome the scarcity of data, data augmentation techniques and transfer learning were employed. Attention mechanisms were also incorporated to accommodate the tonal nature of Shona speech. The resulting ASR system achieved impressive results, with a Word Error Rate of 29%, Phoneme Error Rate of 12%, and an overall accuracy of 74%. These metrics indicated the potential of deep learning to enhance ASR accuracy for under-resourced languages like Shona. This study contributed to the advancement of ASR technology for under-resourced languages like Shona, ultimately fostering improved accessibility and communication for Shona speakers worldwide. 

---
# Do Large Language Models Understand Morality Across Cultures? 

**Authors**: Hadi Mohammadi, Yasmeen F.S.S. Meijer, Efthymia Papadopoulou, Ayoub Bagheri  

**Link**: [PDF](https://arxiv.org/pdf/2507.21319)  

**Abstract**: Recent advancements in large language models (LLMs) have established them as powerful tools across numerous domains. However, persistent concerns about embedded biases, such as gender, racial, and cultural biases arising from their training data, raise significant questions about the ethical use and societal consequences of these technologies. This study investigates the extent to which LLMs capture cross-cultural differences and similarities in moral perspectives. Specifically, we examine whether LLM outputs align with patterns observed in international survey data on moral attitudes. To this end, we employ three complementary methods: (1) comparing variances in moral scores produced by models versus those reported in surveys, (2) conducting cluster alignment analyses to assess correspondence between country groupings derived from LLM outputs and survey data, and (3) directly probing models with comparative prompts using systematically chosen token pairs. Our results reveal that current LLMs often fail to reproduce the full spectrum of cross-cultural moral variation, tending to compress differences and exhibit low alignment with empirical survey patterns. These findings highlight a pressing need for more robust approaches to mitigate biases and improve cultural representativeness in LLMs. We conclude by discussing the implications for the responsible development and global deployment of LLMs, emphasizing fairness and ethical alignment. 

---
# Can human clinical rationales improve the performance and explainability of clinical text classification models? 

**Authors**: Christoph Metzner, Shang Gao, Drahomira Herrmannova, Heidi A. Hanson  

**Link**: [PDF](https://arxiv.org/pdf/2507.21302)  

**Abstract**: AI-driven clinical text classification is vital for explainable automated retrieval of population-level health information. This work investigates whether human-based clinical rationales can serve as additional supervision to improve both performance and explainability of transformer-based models that automatically encode clinical documents. We analyzed 99,125 human-based clinical rationales that provide plausible explanations for primary cancer site diagnoses, using them as additional training samples alongside 128,649 electronic pathology reports to evaluate transformer-based models for extracting primary cancer sites. We also investigated sufficiency as a way to measure rationale quality for pre-selecting rationales. Our results showed that clinical rationales as additional training data can improve model performance in high-resource scenarios but produce inconsistent behavior when resources are limited. Using sufficiency as an automatic metric to preselect rationales also leads to inconsistent results. Importantly, models trained on rationales were consistently outperformed by models trained on additional reports instead. This suggests that clinical rationales don't consistently improve model performance and are outperformed by simply using more reports. Therefore, if the goal is optimizing accuracy, annotation efforts should focus on labeling more reports rather than creating rationales. However, if explainability is the priority, training models on rationale-supplemented data may help them better identify rationale-like features. We conclude that using clinical rationales as additional training data results in smaller performance improvements and only slightly better explainability (measured as average token-level rationale coverage) compared to training on additional reports. 

---
# Bangla BERT for Hyperpartisan News Detection: A Semi-Supervised and Explainable AI Approach 

**Authors**: Mohammad Mehadi Hasan, Fatema Binte Hassan, Md Al Jubair, Zobayer Ahmed, Sazzatul Yeakin, Md Masum Billah  

**Link**: [PDF](https://arxiv.org/pdf/2507.21242)  

**Abstract**: In the current digital landscape, misinformation circulates rapidly, shaping public perception and causing societal divisions. It is difficult to identify hyperpartisan news in Bangla since there aren't many sophisticated natural language processing methods available for this low-resource language. Without effective detection methods, biased content can spread unchecked, posing serious risks to informed discourse. To address this gap, our research fine-tunes Bangla BERT. This is a state-of-the-art transformer-based model, designed to enhance classification accuracy for hyperpartisan news. We evaluate its performance against traditional machine learning models and implement semi-supervised learning to enhance predictions further. Not only that, we use LIME to provide transparent explanations of the model's decision-making process, which helps to build trust in its outcomes. With a remarkable accuracy score of 95.65%, Bangla BERT outperforms conventional approaches, according to our trial data. The findings of this study demonstrate the usefulness of transformer models even in environments with limited resources, which opens the door to further improvements in this area. 

---
# Understanding Public Perception of Crime in Bangladesh: A Transformer-Based Approach with Explainability 

**Authors**: Fatema Binte Hassan, Md Al Jubair, Mohammad Mehadi Hasan, Tahmid Hossain, S M Mehebubur Rahman Khan Shuvo, Mohammad Shamsul Arefin  

**Link**: [PDF](https://arxiv.org/pdf/2507.21234)  

**Abstract**: In recent years, social media platforms have become prominent spaces for individuals to express their opinions on ongoing events, including criminal incidents. As a result, public sentiment can shift dynamically over time. This study investigates the evolving public perception of crime-related news by classifying user-generated comments into three categories: positive, negative, and neutral. A newly curated dataset comprising 28,528 Bangla-language social media comments was developed for this purpose. We propose a transformer-based model utilizing the XLM-RoBERTa Base architecture, which achieves a classification accuracy of 97%, outperforming existing state-of-the-art methods in Bangla sentiment analysis. To enhance model interpretability, explainable AI technique is employed to identify the most influential features driving sentiment classification. The results underscore the effectiveness of transformer-based models in processing low-resource languages such as Bengali and demonstrate their potential to extract actionable insights that can support public policy formulation and crime prevention strategies. 

---
# Contrast-CAT: Contrasting Activations for Enhanced Interpretability in Transformer-based Text Classifiers 

**Authors**: Sungmin Han, Jeonghyun Lee, Sangkyun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2507.21186)  

**Abstract**: Transformers have profoundly influenced AI research, but explaining their decisions remains challenging -- even for relatively simpler tasks such as classification -- which hinders trust and safe deployment in real-world applications. Although activation-based attribution methods effectively explain transformer-based text classification models, our findings reveal that these methods can be undermined by class-irrelevant features within activations, leading to less reliable interpretations. To address this limitation, we propose Contrast-CAT, a novel activation contrast-based attribution method that refines token-level attributions by filtering out class-irrelevant features. By contrasting the activations of an input sequence with reference activations, Contrast-CAT generates clearer and more faithful attribution maps. Experimental results across various datasets and models confirm that Contrast-CAT consistently outperforms state-of-the-art methods. Notably, under the MoRF setting, it achieves average improvements of x1.30 in AOPC and x2.25 in LOdds over the most competing methods, demonstrating its effectiveness in enhancing interpretability for transformer-based text classification. 

---
# Diverse LLMs or Diverse Question Interpretations? That is the Ensembling Question 

**Authors**: Rafael Rosales, Santiago Miret  

**Link**: [PDF](https://arxiv.org/pdf/2507.21168)  

**Abstract**: Effectively leveraging diversity has been shown to improve performance for various machine learning models, including large language models (LLMs). However, determining the most effective way of using diversity remains a challenge. In this work, we compare two diversity approaches for answering binary questions using LLMs: model diversity, which relies on multiple models answering the same question, and question interpretation diversity, which relies on using the same model to answer the same question framed in different ways. For both cases, we apply majority voting as the ensemble consensus heuristic to determine the final answer. Our experiments on boolq, strategyqa, and pubmedqa show that question interpretation diversity consistently leads to better ensemble accuracy compared to model diversity. Furthermore, our analysis of GPT and LLaMa shows that model diversity typically produces results between the best and the worst ensemble members without clear improvement. 

---
# TTS-1 Technical Report 

**Authors**: Oleg Atamanenko, Anna Chalova, Joseph Coombes, Nikki Cope, Phillip Dang, Zhifeng Deng, Jimmy Du, Michael Ermolenko, Feifan Fan, Yufei Feng, Cheryl Fichter, Pavel Filimonov, Louis Fischer, Kylan Gibbs, Valeria Gusarova, Pavel Karpik, Andreas Assad Kottner, Ian Lee, Oliver Louie, Jasmine Mai, Mikhail Mamontov, Suri Mao, Nurullah Morshed, Igor Poletaev, Florin Radu, Dmytro Semernia, Evgenii Shingarev, Vikram Sivaraja, Peter Skirko, Rinat Takhautdinov, Robert Villahermosa, Jean Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.21138)  

**Abstract**: We introduce Inworld TTS-1, a set of two Transformer-based autoregressive text-to-speech (TTS) models. Our largest model, TTS-1-Max, has 8.8B parameters and is designed for utmost quality and expressiveness in demanding applications. TTS-1 is our most efficient model, with 1.6B parameters, built for real-time speech synthesis and on-device use cases. By scaling train-time compute and applying a sequential process of pre-training, fine-tuning, and RL-alignment of the speech-language model (SpeechLM) component, both models achieve state-of-the-art performance on a variety of benchmarks, demonstrating exceptional quality relying purely on in-context learning of the speaker's voice. Inworld TTS-1 and TTS-1-Max can generate high-resolution 48 kHz speech with low latency, and support 11 languages with fine-grained emotional control and non-verbal vocalizations through audio markups. We additionally open-source our training and modeling code under an MIT license. 

---
# TRIDENT: Benchmarking LLM Safety in Finance, Medicine, and Law 

**Authors**: Zheng Hui, Yijiang River Dong, Ehsan Shareghi, Nigel Collier  

**Link**: [PDF](https://arxiv.org/pdf/2507.21134)  

**Abstract**: As large language models (LLMs) are increasingly deployed in high-risk domains such as law, finance, and medicine, systematically evaluating their domain-specific safety and compliance becomes critical. While prior work has largely focused on improving LLM performance in these domains, it has often neglected the evaluation of domain-specific safety risks. To bridge this gap, we first define domain-specific safety principles for LLMs based on the AMA Principles of Medical Ethics, the ABA Model Rules of Professional Conduct, and the CFA Institute Code of Ethics. Building on this foundation, we introduce Trident-Bench, a benchmark specifically targeting LLM safety in the legal, financial, and medical domains. We evaluated 19 general-purpose and domain-specialized models on Trident-Bench and show that it effectively reveals key safety gaps -- strong generalist models (e.g., GPT, Gemini) can meet basic expectations, whereas domain-specialized models often struggle with subtle ethical nuances. This highlights an urgent need for finer-grained domain-specific safety improvements. By introducing Trident-Bench, our work provides one of the first systematic resources for studying LLM safety in law and finance, and lays the groundwork for future research aimed at reducing the safety risks of deploying LLMs in professionally regulated fields. Code and benchmark will be released at: this https URL 

---
# InsurTech innovation using natural language processing 

**Authors**: Panyi Dong, Zhiyu Quan  

**Link**: [PDF](https://arxiv.org/pdf/2507.21112)  

**Abstract**: With the rapid rise of InsurTech, traditional insurance companies are increasingly exploring alternative data sources and advanced technologies to sustain their competitive edge. This paper provides both a conceptual overview and practical case studies of natural language processing (NLP) and its emerging applications within insurance operations with a focus on transforming raw, unstructured text into structured data suitable for actuarial analysis and decision-making. Leveraging real-world alternative data provided by an InsurTech industry partner that enriches traditional insurance data sources, we apply various NLP techniques to demonstrate practical use cases in the commercial insurance context. These enriched, text-derived insights not only add to and refine traditional rating factors for commercial insurance pricing but also offer novel perspectives for assessing underlying risk by introducing novel industry classifications. Through these demonstrations, we show that NLP is not merely a supplementary tool but a foundational element for modern, data-driven insurance analytics. 

---
# SemRAG: Semantic Knowledge-Augmented RAG for Improved Question-Answering 

**Authors**: Kezhen Zhong, Basem Suleiman, Abdelkarim Erradi, Shijing Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.21110)  

**Abstract**: This paper introduces SemRAG, an enhanced Retrieval Augmented Generation (RAG) framework that efficiently integrates domain-specific knowledge using semantic chunking and knowledge graphs without extensive fine-tuning. Integrating domain-specific knowledge into large language models (LLMs) is crucial for improving their performance in specialized tasks. Yet, existing adaptations are computationally expensive, prone to overfitting and limit scalability. To address these challenges, SemRAG employs a semantic chunking algorithm that segments documents based on the cosine similarity from sentence embeddings, preserving semantic coherence while reducing computational overhead. Additionally, by structuring retrieved information into knowledge graphs, SemRAG captures relationships between entities, improving retrieval accuracy and contextual understanding. Experimental results on MultiHop RAG and Wikipedia datasets demonstrate SemRAG has significantly enhances the relevance and correctness of retrieved information from the Knowledge Graph, outperforming traditional RAG methods. Furthermore, we investigate the optimization of buffer sizes for different data corpus, as optimizing buffer sizes tailored to specific datasets can further improve retrieval performance, as integration of knowledge graphs strengthens entity relationships for better contextual comprehension. The primary advantage of SemRAG is its ability to create an efficient, accurate domain-specific LLM pipeline while avoiding resource-intensive fine-tuning. This makes it a practical and scalable approach aligned with sustainability goals, offering a viable solution for AI applications in domain-specific fields. 

---
# A Survey of Classification Tasks and Approaches for Legal Contracts 

**Authors**: Amrita Singh, Aditya Joshi, Jiaojiao Jiang, Hye-young Paik  

**Link**: [PDF](https://arxiv.org/pdf/2507.21108)  

**Abstract**: Given the large size and volumes of contracts and their underlying inherent complexity, manual reviews become inefficient and prone to errors, creating a clear need for automation. Automatic Legal Contract Classification (LCC) revolutionizes the way legal contracts are analyzed, offering substantial improvements in speed, accuracy, and accessibility. This survey delves into the challenges of automatic LCC and a detailed examination of key tasks, datasets, and methodologies. We identify seven classification tasks within LCC, and review fourteen datasets related to English-language contracts, including public, proprietary, and non-public sources. We also introduce a methodology taxonomy for LCC, categorized into Traditional Machine Learning, Deep Learning, and Transformer-based approaches. Additionally, the survey discusses evaluation techniques and highlights the best-performing results from the reviewed studies. By providing a thorough overview of current methods and their limitations, this survey suggests future research directions to improve the efficiency, accuracy, and scalability of LCC. As the first comprehensive survey on LCC, it aims to support legal NLP researchers and practitioners in improving legal processes, making legal information more accessible, and promoting a more informed and equitable society. 

---
# Curved Inference: Concern-Sensitive Geometry in Large Language Model Residual Streams 

**Authors**: Rob Manson  

**Link**: [PDF](https://arxiv.org/pdf/2507.21107)  

**Abstract**: We propose Curved Inference - a geometric Interpretability framework that tracks how the residual stream trajectory of a large language model bends in response to shifts in semantic concern. Across 20 matched prompts spanning emotional, moral, perspective, logical, identity, environmental, and nonsense domains, we analyse Gemma3-1b and LLaMA3.2-3b using five native-space metrics, with a primary focus on curvature (\k{appa}_i) and salience (S(t)). These metrics are computed under a pullback semantic metric derived from the unembedding matrix, ensuring that all measurements reflect token-aligned geometry rather than raw coordinate structure. We find that concern-shifted prompts reliably alter internal activation trajectories in both models - with LLaMA exhibiting consistent, statistically significant scaling in both curvature and salience as concern intensity increases. Gemma also responds to concern but shows weaker differentiation between moderate and strong variants. Our results support a two-layer view of LLM geometry - a latent conceptual structure encoded in the embedding space, and a contextual trajectory shaped by prompt-specific inference. Curved Inference reveals how models navigate, reorient, or reinforce semantic meaning over depth, offering a principled method for diagnosing alignment, abstraction, and emergent inference dynamics. These findings offer fresh insight into semantic abstraction and model alignment through the lens of Curved Inference. 

---
# Creation of a Numerical Scoring System to Objectively Measure and Compare the Level of Rhetoric in Arabic Texts: A Feasibility Study, and A Working Prototype 

**Authors**: Mandar Marathe  

**Link**: [PDF](https://arxiv.org/pdf/2507.21106)  

**Abstract**: Arabic Rhetoric is the field of Arabic linguistics which governs the art and science of conveying a message with greater beauty, impact and persuasiveness. The field is as ancient as the Arabic language itself and is found extensively in classical and contemporary Arabic poetry, free verse and prose. In practical terms, it is the intelligent use of word order, figurative speech and linguistic embellishments to enhance message delivery. Despite the volumes that have been written about it and the high status accorded to it, there is no way to objectively know whether a speaker or writer has used Arabic rhetoric in a given text, to what extent, and why. There is no objective way to compare the use of Arabic rhetoric across genres, authors or epochs. It is impossible to know which of pre-Islamic poetry, Andalucian Arabic poetry, or modern literary genres are richer in Arabic rhetoric. The aim of the current study was to devise a way to measure the density of the literary devices which constitute Arabic rhetoric in a given text, as a proxy marker for Arabic rhetoric itself. A comprehensive list of 84 of the commonest literary devices and their definitions was compiled. A system of identifying literary devices in texts was constructed. A method of calculating the density of literary devices based on the morpheme count of the text was utilised. Four electronic tools and an analogue tool were created to support the calculation of an Arabic text's rhetorical literary device density, including a website and online calculator. Additionally, a technique of reporting the distribution of literary devices used across the three sub-domains of Arabic rhetoric was created. The output of this project is a working tool which can accurately report the density of Arabic rhetoric in any Arabic text or speech. 

---
# iLSU-T: an Open Dataset for Uruguayan Sign Language Translation 

**Authors**: Ariel E. Stassi, Yanina Boria, J. Matías Di Martino, Gregory Randall  

**Link**: [PDF](https://arxiv.org/pdf/2507.21104)  

**Abstract**: Automatic sign language translation has gained particular interest in the computer vision and computational linguistics communities in recent years. Given each sign language country particularities, machine translation requires local data to develop new techniques and adapt existing ones. This work presents iLSU T, an open dataset of interpreted Uruguayan Sign Language RGB videos with audio and text transcriptions. This type of multimodal and curated data is paramount for developing novel approaches to understand or generate tools for sign language processing. iLSU T comprises more than 185 hours of interpreted sign language videos from public TV broadcasting. It covers diverse topics and includes the participation of 18 professional interpreters of sign language. A series of experiments using three state of the art translation algorithms is presented. The aim is to establish a baseline for this dataset and evaluate its usefulness and the proposed pipeline for data processing. The experiments highlight the need for more localized datasets for sign language translation and understanding, which are critical for developing novel tools to improve accessibility and inclusion of all individuals. Our data and code can be accessed. 

---
# Rewrite-to-Rank: Optimizing Ad Visibility via Retrieval-Aware Text Rewriting 

**Authors**: Chloe Ho, Ishneet Sukhvinder Singh, Diya Sharma, Tanvi Reddy Anumandla, Michael Lu, Vasu Sharma, Kevin Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2507.21099)  

**Abstract**: Search algorithms and user query relevance have given LLMs the ability to return relevant information, but the effect of content phrasing on ad visibility remains underexplored. We investigate how LLM-based rewriting of advertisements can improve their ranking in retrieval systems and inclusion in generated LLM responses, without modifying the retrieval model itself. We introduce a supervised fine-tuning framework with a custom loss balancing semantic relevance and content fidelity. To evaluate effectiveness, we propose two metrics: DeltaMRR@K (ranking improvement) and DeltaDIR@K (inclusion frequency improvement). Our approach presents a scalable method to optimize ad phrasing, enhancing visibility in retrieval-based LLM workflows. Experiments across both instruction-based and few-shot prompting demonstrate that PPO trained models outperform both prompt engineering and supervised fine-tuning in most cases, achieving up to a 2.79 DeltaDIR@5 and 0.0073 DeltaMRR@5 in instruction-based prompting. These results highlight the importance of how the ad is written before retrieval and prompt format and reinforcement learning in effective ad rewriting for LLM integrated retrieval systems. 

---
# QU-NLP at CheckThat! 2025: Multilingual Subjectivity in News Articles Detection using Feature-Augmented Transformer Models with Sequential Cross-Lingual Fine-Tuning 

**Authors**: Mohammad AL-Smadi  

**Link**: [PDF](https://arxiv.org/pdf/2507.21095)  

**Abstract**: This paper presents our approach to the CheckThat! 2025 Task 1 on subjectivity detection, where systems are challenged to distinguish whether a sentence from a news article expresses the subjective view of the author or presents an objective view on the covered topic. We propose a feature-augmented transformer architecture that combines contextual embeddings from pre-trained language models with statistical and linguistic features. Our system leveraged pre-trained transformers with additional lexical features: for Arabic we used AraELECTRA augmented with part-of-speech (POS) tags and TF-IDF features, while for the other languages we fine-tuned a cross-lingual DeBERTa~V3 model combined with TF-IDF features through a gating mechanism. We evaluated our system in monolingual, multilingual, and zero-shot settings across multiple languages including English, Arabic, German, Italian, and several unseen languages. The results demonstrate the effectiveness of our approach, achieving competitive performance across different languages with notable success in the monolingual setting for English (rank 1st with macro-F1=0.8052), German (rank 3rd with macro-F1=0.8013), Arabic (rank 4th with macro-F1=0.5771), and Romanian (rank 1st with macro-F1=0.8126) in the zero-shot setting. We also conducted an ablation analysis that demonstrated the importance of combining TF-IDF features with the gating mechanism and the cross-lingual transfer for subjectivity detection. Furthermore, our analysis reveals the model's sensitivity to both the order of cross-lingual fine-tuning and the linguistic proximity of the training languages. 

---
# Multi-Amateur Contrastive Decoding for Text Generation 

**Authors**: Jaydip Sen, Subhasis Dasgupta, Hetvi Waghela  

**Link**: [PDF](https://arxiv.org/pdf/2507.21086)  

**Abstract**: Contrastive Decoding (CD) has emerged as an effective inference-time strategy for enhancing open-ended text generation by exploiting the divergence in output probabilities between a large expert language model and a smaller amateur model. Although CD improves coherence and fluency, its dependence on a single amateur restricts its capacity to capture the diverse and multifaceted failure modes of language generation, such as repetition, hallucination, and stylistic drift. This paper proposes Multi-Amateur Contrastive Decoding (MACD), a generalization of the CD framework that employs an ensemble of amateur models to more comprehensively characterize undesirable generation patterns. MACD integrates contrastive signals through both averaging and consensus penalization mechanisms and extends the plausibility constraint to operate effectively in the multi-amateur setting. Furthermore, the framework enables controllable generation by incorporating amateurs with targeted stylistic or content biases. Experimental results across multiple domains, such as news, encyclopedic, and narrative, demonstrate that MACD consistently surpasses conventional decoding methods and the original CD approach in terms of fluency, coherence, diversity, and adaptability, all without requiring additional training or fine-tuning. 

---
# Reviving Your MNEME: Predicting The Side Effects of LLM Unlearning and Fine-Tuning via Sparse Model Diffing 

**Authors**: Aly M. Kassem, Zhuan Shi, Negar Rostamzadeh, Golnoosh Farnadi  

**Link**: [PDF](https://arxiv.org/pdf/2507.21084)  

**Abstract**: Large language models (LLMs) are frequently fine-tuned or unlearned to adapt to new tasks or eliminate undesirable behaviors. While existing evaluation methods assess performance after such interventions, there remains no general approach for detecting unintended side effects, such as unlearning biology content degrading performance on chemistry tasks, particularly when these effects are unpredictable or emergent. To address this issue, we introduce MNEME, Model diffiNg for Evaluating Mechanistic Effects, a lightweight framework for identifying these side effects using sparse model diffing. MNEME compares base and fine-tuned models on task-agnostic data (for example, The Pile, LMSYS-Chat-1M) without access to fine-tuning data to isolate behavioral shifts. Applied to five LLMs across three scenarios: WMDP knowledge unlearning, emergent misalignment, and benign fine-tuning, MNEME achieves up to 95 percent accuracy in predicting side effects, aligning with known benchmarks and requiring no custom heuristics. Furthermore, we show that retraining on high-activation samples can partially reverse these effects. Our results demonstrate that sparse probing and diffing offer a scalable and automated lens into fine-tuning-induced model changes, providing practical tools for understanding and managing LLM behavior. 

---
# ChatGPT Reads Your Tone and Responds Accordingly -- Until It Does Not -- Emotional Framing Induces Bias in LLM Outputs 

**Authors**: Franck Bardol  

**Link**: [PDF](https://arxiv.org/pdf/2507.21083)  

**Abstract**: Large Language Models like GPT-4 adjust their responses not only based on the question asked, but also on how it is emotionally phrased. We systematically vary the emotional tone of 156 prompts - spanning controversial and everyday topics - and analyze how it affects model responses. Our findings show that GPT-4 is three times less likely to respond negatively to a negatively framed question than to a neutral one. This suggests a "rebound" bias where the model overcorrects, often shifting toward neutrality or positivity. On sensitive topics (e.g., justice or politics), this effect is even more pronounced: tone-based variation is suppressed, suggesting an alignment override. We introduce concepts like the "tone floor" - a lower bound in response negativity - and use tone-valence transition matrices to quantify behavior. Visualizations based on 1536-dimensional embeddings confirm semantic drift based on tone. Our work highlights an underexplored class of biases driven by emotional framing in prompts, with implications for AI alignment and trust. Code and data are available at: this https URL 

---
# Which symbol grounding problem should we try to solve? 

**Authors**: Vincent C. Müller  

**Link**: [PDF](https://arxiv.org/pdf/2507.21080)  

**Abstract**: Floridi and Taddeo propose a condition of "zero semantic commitment" for solutions to the grounding problem, and a solution to it. I argue briefly that their condition cannot be fulfilled, not even by their own solution. After a look at Luc Steels' very different competing suggestion, I suggest that we need to re-think what the problem is and what role the 'goals' in a system play in formulating the problem. On the basis of a proper understanding of computing, I come to the conclusion that the only sensible grounding problem is how we can explain and re-produce the behavioral ability and function of meaning in artificial computational agents 

---
# Product vs. Process: Exploring EFL Students' Editing of AI-Generated Text for Expository Writing 

**Authors**: David James Woo, Yangyang Yu, Kai Guo, Yilin Huang, April Ka Yeng Fung  

**Link**: [PDF](https://arxiv.org/pdf/2507.21073)  

**Abstract**: Text generated by artificial intelligence (AI) chatbots is increasingly used in English as a foreign language (EFL) writing contexts, yet its impact on students' expository writing process and compositions remains understudied. This research examines how EFL secondary students edit AI-generated text. Exploring editing behaviors in their expository writing process and in expository compositions, and their effect on human-rated scores for content, organization, language, and overall quality. Participants were 39 Hong Kong secondary students who wrote an expository composition with AI chatbots in a workshop. A convergent design was employed to analyze their screen recordings and compositions to examine students' editing behaviors and writing qualities. Analytical methods included qualitative coding, descriptive statistics, temporal sequence analysis, human-rated scoring, and multiple linear regression analysis. We analyzed over 260 edits per dataset, and identified two editing patterns: one where students refined introductory units repeatedly before progressing, and another where they quickly shifted to extensive edits in body units (e.g., topic and supporting sentences). MLR analyses revealed that the number of AI-generated words positively predicted all score dimensions, while most editing variables showed minimal impact. These results suggest a disconnect between students' significant editing effort and improved composition quality, indicating AI supports but does not replace writing skills. The findings highlight the importance of genre-specific instruction and process-focused writing before AI integration. Educators should also develop assessments valuing both process and product to encourage critical engagement with AI text. 

---
# Dialogic Social Learning for Artificial Agents: Enhancing LLM Ontology Acquisition through Mixed-Initiative Educational Interactions 

**Authors**: Sabrina Patania, Luca Annese, Cansu Koyuturk, Azzurra Ruggeri, Dimitri Ognibene  

**Link**: [PDF](https://arxiv.org/pdf/2507.21065)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities in processing extensive offline datasets. However, they often face challenges in acquiring and integrating complex, knowledge online. Traditional AI training paradigms, predominantly based on supervised learning or reinforcement learning, mirror a 'Piagetian' model of independent exploration. These approaches typically rely on large datasets and sparse feedback signals, limiting the models' ability to learn efficiently from interactions. Drawing inspiration from Vygotsky's sociocultural theory, this study explores the potential of socially mediated learning paradigms to address these limitations.
We introduce a dynamic environment, termed the 'AI Social Gym', where an AI learner agent engages in dyadic pedagogical dialogues with knowledgeable AI teacher agents. These interactions emphasize external, structured dialogue as a core mechanism for knowledge acquisition, contrasting with methods that depend solely on internal inference or pattern recognition.
Our investigation focuses on how different pedagogical strategies impact the AI learning process in the context of ontology acquisition. Empirical results indicate that such dialogic approaches-particularly those involving mixed-direction interactions combining top-down explanations with learner-initiated questioning-significantly enhance the LLM's ability to acquire and apply new knowledge, outperforming both unidirectional instructional methods and direct access to structured knowledge, formats typically present in training datasets.
These findings suggest that integrating pedagogical and psychological insights into AI and robot training can substantially improve post-training knowledge acquisition and response quality. This approach offers a complementary pathway to existing strategies like prompt engineering 

---
# Categorical Classification of Book Summaries Using Word Embedding Techniques 

**Authors**: Kerem Keskin, Mümine Kaya Keleş  

**Link**: [PDF](https://arxiv.org/pdf/2507.21058)  

**Abstract**: In this study, book summaries and categories taken from book sites were classified using word embedding methods, natural language processing techniques and machine learning algorithms. In addition, one hot encoding, Word2Vec and Term Frequency - Inverse Document Frequency (TF-IDF) methods, which are frequently used word embedding methods were used in this study and their success was compared. Additionally, the combination table of the pre-processing methods used is shown and added to the table. Looking at the results, it was observed that Support Vector Machine, Naive Bayes and Logistic Regression Models and TF-IDF and One-Hot Encoder word embedding techniques gave more successful results for Turkish texts. 

---
# MetaCLIP 2: A Worldwide Scaling Recipe 

**Authors**: Yung-Sung Chuang, Yang Li, Dong Wang, Ching-Feng Yeh, Kehan Lyu, Ramya Raghavendra, James Glass, Lifei Huang, Jason Weston, Luke Zettlemoyer, Xinlei Chen, Zhuang Liu, Saining Xie, Wen-tau Yih, Shang-Wen Li, Hu Xu  

**Link**: [PDF](https://arxiv.org/pdf/2507.22062)  

**Abstract**: Contrastive Language-Image Pretraining (CLIP) is a popular foundation model, supporting from zero-shot classification, retrieval to encoders for multimodal large language models (MLLMs). Although CLIP is successfully trained on billion-scale image-text pairs from the English world, scaling CLIP's training further to learning from the worldwide web data is still challenging: (1) no curation method is available to handle data points from non-English world; (2) the English performance from existing multilingual CLIP is worse than its English-only counterpart, i.e., "curse of multilinguality" that is common in LLMs. Here, we present MetaCLIP 2, the first recipe training CLIP from scratch on worldwide web-scale image-text pairs. To generalize our findings, we conduct rigorous ablations with minimal changes that are necessary to address the above challenges and present a recipe enabling mutual benefits from English and non-English world data. In zero-shot ImageNet classification, MetaCLIP 2 ViT-H/14 surpasses its English-only counterpart by 0.8% and mSigLIP by 0.7%, and surprisingly sets new state-of-the-art without system-level confounding factors (e.g., translation, bespoke architecture changes) on multilingual benchmarks, such as CVQA with 57.4%, Babel-ImageNet with 50.2% and XM3600 with 64.3% on image-to-text retrieval. 

---
# UserBench: An Interactive Gym Environment for User-Centric Agents 

**Authors**: Cheng Qian, Zuxin Liu, Akshara Prabhakar, Zhiwei Liu, Jianguo Zhang, Haolin Chen, Heng Ji, Weiran Yao, Shelby Heinecke, Silvio Savarese, Caiming Xiong, Huan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.22034)  

**Abstract**: Large Language Models (LLMs)-based agents have made impressive progress in reasoning and tool use, enabling them to solve complex tasks. However, their ability to proactively collaborate with users, especially when goals are vague, evolving, or indirectly expressed, remains underexplored. To address this gap, we introduce UserBench, a user-centric benchmark designed to evaluate agents in multi-turn, preference-driven interactions. UserBench features simulated users who start with underspecified goals and reveal preferences incrementally, requiring agents to proactively clarify intent and make grounded decisions with tools. Our evaluation of leading open- and closed-source LLMs reveals a significant disconnect between task completion and user alignment. For instance, models provide answers that fully align with all user intents only 20% of the time on average, and even the most advanced models uncover fewer than 30% of all user preferences through active interaction. These results highlight the challenges of building agents that are not just capable task executors, but true collaborative partners. UserBench offers an interactive environment to measure and advance this critical capability. 

---
# UI-AGILE: Advancing GUI Agents with Effective Reinforcement Learning and Precise Inference-Time Grounding 

**Authors**: Shuquan Lian, Yuhang Wu, Jia Ma, Zihan Song, Bingqi Chen, Xiawu Zheng, Hui Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.22025)  

**Abstract**: The emergence of Multimodal Large Language Models (MLLMs) has driven significant advances in Graphical User Interface (GUI) agent capabilities. Nevertheless, existing GUI agent training and inference techniques still suffer from a dilemma for reasoning designs, ineffective reward, and visual noise. To address these issues, we introduce UI-AGILE, a comprehensive framework enhancing GUI agents at both the training and inference stages. For training, we propose a suite of improvements to the Supervised Fine-Tuning (SFT) process: 1) a Continuous Reward function to incentivize high-precision grounding; 2) a "Simple Thinking" reward to balance planning with speed and grounding accuracy; and 3) a Cropping-based Resampling strategy to mitigate the sparse reward problem and improve learning on complex tasks. For inference, we present Decomposed Grounding with Selection, a novel method that dramatically improves grounding accuracy on high-resolution displays by breaking the image into smaller, manageable parts. Experiments show that UI-AGILE achieves the state-of-the-art performance on two benchmarks ScreenSpot-Pro and ScreenSpot-v2. For instance, using both our proposed training and inference enhancement methods brings 23% grounding accuracy improvement over the best baseline on ScreenSpot-Pro. 

---
# Who's important? -- SUnSET: Synergistic Understanding of Stakeholder, Events and Time for Timeline Generation 

**Authors**: Tiviatis Sim, Kaiwen Yang, Shen Xin, Kenji Kawaguchi  

**Link**: [PDF](https://arxiv.org/pdf/2507.21903)  

**Abstract**: As news reporting becomes increasingly global and decentralized online, tracking related events across multiple sources presents significant challenges. Existing news summarization methods typically utilizes Large Language Models and Graphical methods on article-based summaries. However, this is not effective since it only considers the textual content of similarly dated articles to understand the gist of the event. To counteract the lack of analysis on the parties involved, it is essential to come up with a novel framework to gauge the importance of stakeholders and the connection of related events through the relevant entities involved. Therefore, we present SUnSET: Synergistic Understanding of Stakeholder, Events and Time for the task of Timeline Summarization (TLS). We leverage powerful Large Language Models (LLMs) to build SET triplets and introduced the use of stakeholder-based ranking to construct a $Relevancy$ metric, which can be extended into general situations. Our experimental results outperform all prior baselines and emerged as the new State-of-the-Art, highlighting the impact of stakeholder information within news article. 

---
# What Does it Mean for a Neural Network to Learn a "World Model"? 

**Authors**: Kenneth Li, Fernanda Viégas, Martin Wattenberg  

**Link**: [PDF](https://arxiv.org/pdf/2507.21513)  

**Abstract**: We propose a set of precise criteria for saying a neural net learns and uses a "world model." The goal is to give an operational meaning to terms that are often used informally, in order to provide a common language for experimental investigation. We focus specifically on the idea of representing a latent "state space" of the world, leaving modeling the effect of actions to future work. Our definition is based on ideas from the linear probing literature, and formalizes the notion of a computation that factors through a representation of the data generation process. An essential addition to the definition is a set of conditions to check that such a "world model" is not a trivial consequence of the neural net's data or task. 

---
# ReGATE: Learning Faster and Better with Fewer Tokens in MLLMs 

**Authors**: Chaoyu Li, Yogesh Kulkarni, Pooyan Fazli  

**Link**: [PDF](https://arxiv.org/pdf/2507.21420)  

**Abstract**: The computational cost of training multimodal large language models (MLLMs) rapidly increases with the number of tokens involved. Existing efficiency methods primarily target inference and rely on token reduction or merging, offering limited benefit during training. In this paper, we propose ReGATE (Reference$-$Guided Adaptive Token Elision), an adaptive token pruning method for accelerating MLLM training. Specifically, ReGATE adopts a teacher-student framework in which the MLLM being trained serves as the student, and a frozen reference large language model (LLM) acts as the teacher. The teacher computes per-token reference losses, which are combined with an exponential moving average (EMA) of the student's own difficulty scores. This adaptive difficulty-based scoring enables the selective processing of crucial tokens while bypassing less informative ones in the forward pass, significantly reducing computational overhead. Experiments demonstrate that ReGATE, when applied to VideoLLaMA2, matches the peak accuracy of standard training on MVBench up to 2$\times$ faster, using only 35% of the tokens. With additional training, it even surpasses the baseline on several multimodal benchmarks, all while reducing the total token count by over 41%. Code and models will be released soon. 

---
# Multimodal LLMs as Customized Reward Models for Text-to-Image Generation 

**Authors**: Shijie Zhou, Ruiyi Zhang, Huaisheng Zhu, Branislav Kveton, Yufan Zhou, Jiuxiang Gu, Jian Chen, Changyou Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.21391)  

**Abstract**: We introduce LLaVA-Reward, an efficient reward model designed to automatically evaluate text-to-image (T2I) generations across multiple perspectives, leveraging pretrained multimodal large language models (MLLMs). Existing MLLM-based approaches require instruction-following data for supervised fine-tuning and evaluate generation quality on analyzing text response, which is time-consuming and difficult to train. To address this problem, we propose LLaVA-Reward, which directly utilizes the hidden states of MLLMs given text-image pairs. To enhance the bidirectional interaction between visual and textual representations in decoder-only MLLMs, we further propose adding a Skip-connection Cross Attention (SkipCA) module. This design enhances text-image correlation reasoning by connecting early-layer visual features with later-layer hidden this http URL addition, LLaVA-Reward supports different types of preference data for efficient fine-tuning, including paired preference data and unpaired data. We train LLaVA-Reward on four evaluation perspectives: text-image alignment, fidelity/artifact, safety, and overall ranking. Empirical results demonstrate that LLaVA-Reward outperforms conventional and MLLM-based methods in generating human-aligned scores for automatic evaluations and inference-time scaling in text-to-image generations. 

---
# Teaching Language Models To Gather Information Proactively 

**Authors**: Tenghao Huang, Sihao Chen, Muhao Chen, Jonathan May, Longqi Yang, Mengting Wan, Pei Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2507.21389)  

**Abstract**: Large language models (LLMs) are increasingly expected to function as collaborative partners, engaging in back-and-forth dialogue to solve complex, ambiguous problems. However, current LLMs often falter in real-world settings, defaulting to passive responses or narrow clarifications when faced with incomplete or under-specified prompts, falling short of proactively gathering the missing information that is crucial for high-quality solutions. In this work, we introduce a new task paradigm: proactive information gathering, where LLMs must identify gaps in the provided context and strategically elicit implicit user knowledge through targeted questions. To systematically study and train this capability, we design a scalable framework that generates partially specified, real-world tasks, masking key information and simulating authentic ambiguity. Within this setup, our core innovation is a reinforcement finetuning strategy that rewards questions that elicit genuinely new, implicit user information -- such as hidden domain expertise or fine-grained requirements -- that would otherwise remain unspoken. Experiments demonstrate that our trained Qwen-2.5-7B model significantly outperforms o3-mini by 18% on automatic evaluation metrics. More importantly, human evaluation reveals that clarification questions and final outlines generated by our model are favored by human annotators by 42% and 28% respectively. Together, these results highlight the value of proactive clarification in elevating LLMs from passive text generators to genuinely collaborative thought partners. 

---
# LeMix: Unified Scheduling for LLM Training and Inference on Multi-GPU Systems 

**Authors**: Yufei Li, Zexin Li, Yinglun Zhu, Cong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.21276)  

**Abstract**: Modern deployment of large language models (LLMs) frequently involves both inference serving and continuous retraining to stay aligned with evolving data and user feedback. Common practices separate these workloads onto distinct servers in isolated phases, causing substantial inefficiencies (e.g., GPU idleness) and delayed adaptation to new data in distributed settings. Our empirical analysis reveals that these inefficiencies stem from dynamic request arrivals during serving and workload heterogeneity in pipeline-parallel training. To address these challenges, we propose LeMix, a system for co-locating and managing concurrent LLM serving and training workloads. LeMix integrates offline profiling, execution prediction mechanisms, and runtime scheduling to dynamically adapt resource allocation based on workload characteristics and system conditions. By understanding task-specific behaviors and co-execution interference across shared nodes, LeMix improves utilization and serving quality without compromising serving responsiveness. Our evaluation shows that LeMix improves throughput by up to 3.53x, reduces inference loss by up to 0.61x, and delivers up to 2.12x higher response time SLO attainment over traditional separate setups. To our knowledge, this is the first work to uncover and exploit the opportunities of joint LLM inference and training, paving the way for more resource-efficient deployment of LLMs in production environments. 

---
# CompoST: A Benchmark for Analyzing the Ability of LLMs To Compositionally Interpret Questions in a QALD Setting 

**Authors**: David Maria Schmidt, Raoul Schubert, Philipp Cimiano  

**Link**: [PDF](https://arxiv.org/pdf/2507.21257)  

**Abstract**: Language interpretation is a compositional process, in which the meaning of more complex linguistic structures is inferred from the meaning of their parts. Large language models possess remarkable language interpretation capabilities and have been successfully applied to interpret questions by mapping them to SPARQL queries. An open question is how systematic this interpretation process is. Toward this question, in this paper, we propose a benchmark for investigating to what extent the abilities of LLMs to interpret questions are actually compositional. For this, we generate three datasets of varying difficulty based on graph patterns in DBpedia, relying on Lemon lexica for verbalization. Our datasets are created in a very controlled fashion in order to test the ability of LLMs to interpret structurally complex questions, given that they have seen the atomic building blocks. This allows us to evaluate to what degree LLMs are able to interpret complex questions for which they "understand" the atomic parts. We conduct experiments with models of different sizes using both various prompt and few-shot optimization techniques as well as fine-tuning. Our results show that performance in terms of macro $F_1$ degrades from $0.45$ over $0.26$ down to $0.09$ with increasing deviation from the samples optimized on. Even when all necessary information was provided to the model in the input, the $F_1$ scores do not exceed $0.57$ for the dataset of lowest complexity. We thus conclude that LLMs struggle to systematically and compositionally interpret questions and map them into SPARQL queries. 

---
# EvoSLD: Automated Neural Scaling Law Discovery With Large Language Models 

**Authors**: Haowei Lin, Xiangyu Wang, Jianzhu Ma, Yitao Liang  

**Link**: [PDF](https://arxiv.org/pdf/2507.21184)  

**Abstract**: Scaling laws are fundamental mathematical relationships that predict how neural network performance evolves with changes in variables such as model size, dataset size, and computational resources. Traditionally, discovering these laws requires extensive human expertise and manual experimentation. We introduce EvoSLD, an automated framework for Scaling Law Discovery (SLD) that leverages evolutionary algorithms guided by Large Language Models (LLMs) to co-evolve symbolic expressions and their optimization routines. Formulated to handle scaling variables, control variables, and response metrics across diverse experimental settings, EvoSLD searches for parsimonious, universal functional forms that minimize fitting errors on grouped data subsets. Evaluated on five real-world scenarios from recent literature, EvoSLD rediscovers exact human-derived laws in two cases and surpasses them in others, achieving up to orders-of-magnitude reductions in normalized mean squared error on held-out test sets. Compared to baselines like symbolic regression and ablated variants, EvoSLD demonstrates superior accuracy, interpretability, and efficiency, highlighting its potential to accelerate AI research. Code is available at this https URL. 

---
# MaPPO: Maximum a Posteriori Preference Optimization with Prior Knowledge 

**Authors**: Guangchen Lan, Sipeng Zhang, Tianle Wang, Yuwei Zhang, Daoan Zhang, Xinpeng Wei, Xiaoman Pan, Hongming Zhang, Dong-Jun Han, Christopher G. Brinton  

**Link**: [PDF](https://arxiv.org/pdf/2507.21183)  

**Abstract**: As the era of large language models (LLMs) on behalf of users unfolds, Preference Optimization (PO) methods have become a central approach to aligning LLMs with human preferences and improving performance. We propose Maximum a Posteriori Preference Optimization (MaPPO), a framework for learning from preferences that explicitly incorporates prior reward knowledge into the optimization objective. While existing methods such as Direct Preference Optimization (DPO) and its variants treat preference learning as a Maximum Likelihood Estimation (MLE) problem, MaPPO extends this paradigm by integrating prior reward estimates into a principled Maximum a Posteriori (MaP) objective. This not only generalizes DPO and its variants, but also enhances alignment by mitigating the oversimplified binary classification of responses. More importantly, MaPPO introduces no additional hyperparameter, and supports preference optimization in both offline and online settings. In addition, MaPPO can be used as a plugin with consistent improvement on DPO variants, including widely used SimPO, IPO, and CPO. Extensive empirical evaluations of different model sizes and model series on three standard benchmarks, including MT-Bench, AlpacaEval 2.0, and Arena-Hard, demonstrate consistent improvements in alignment performance without sacrificing computational efficiency. 

---
# OneShield -- the Next Generation of LLM Guardrails 

**Authors**: Chad DeLuca, Anna Lisa Gentile, Shubhi Asthana, Bing Zhang, Pawan Chowdhary, Kellen Cheng, Basel Shbita, Pengyuan Li, Guang-Jie Ren, Sandeep Gopisetty  

**Link**: [PDF](https://arxiv.org/pdf/2507.21170)  

**Abstract**: The rise of Large Language Models has created a general excitement about the great potential for a myriad of applications. While LLMs offer many possibilities, questions about safety, privacy, and ethics have emerged, and all the key actors are working to address these issues with protective measures for their own models and standalone solutions. The constantly evolving nature of LLMs makes the task of universally shielding users against their potential risks extremely challenging, and one-size-fits-all solutions unfeasible. In this work, we propose OneShield, our stand-alone, model-agnostic and customizable solution to safeguard LLMs. OneShield aims to provide facilities for defining risk factors, expressing and declaring contextual safety and compliance policies, and mitigating LLM risks, with a focus on each specific customer. We describe the implementation of the framework, the scalability considerations and provide usage statistics of OneShield since its first deployment. 

---
# AgentMaster: A Multi-Agent Conversational Framework Using A2A and MCP Protocols for Multimodal Information Retrieval and Analysis 

**Authors**: Callie C. Liao, Duoduo Liao, Sai Surya Gadiraju  

**Link**: [PDF](https://arxiv.org/pdf/2507.21105)  

**Abstract**: The rise of Multi-Agent Systems (MAS) in Artificial Intelligence (AI), especially integrated with Large Language Models (LLMs), has greatly facilitated the resolution of complex tasks. However, current systems are still facing challenges of inter-agent communication, coordination, and interaction with heterogeneous tools and resources. Most recently, the Model Context Protocol (MCP) by Anthropic and Agent-to-Agent (A2A) communication protocol by Google have been introduced, and to the best of our knowledge, very few applications exist where both protocols are employed within a single MAS framework. We present a pilot study of AgentMaster, a novel modular multi-protocol MAS framework with self-implemented A2A and MCP, enabling dynamic coordination and flexible communication. Through a unified conversational interface, the system supports natural language interaction without prior technical expertise and responds to multimodal queries for tasks including information retrieval, question answering, and image analysis. Evaluation through the BERTScore F1 and LLM-as-a-Judge metric G-Eval averaged 96.3\% and 87.1\%, revealing robust inter-agent coordination, query decomposition, dynamic routing, and domain-specific, relevant responses. Overall, our proposed framework contributes to the potential capabilities of domain-specific, cooperative, and scalable conversational AI powered by MAS. 

---
# Analise Semantica Automatizada com LLM e RAG para Bulas Farmaceuticas 

**Authors**: Daniel Meireles do Rego  

**Link**: [PDF](https://arxiv.org/pdf/2507.21103)  

**Abstract**: The production of digital documents has been growing rapidly in academic, business, and health environments, presenting new challenges in the efficient extraction and analysis of unstructured information. This work investigates the use of RAG (Retrieval-Augmented Generation) architectures combined with Large-Scale Language Models (LLMs) to automate the analysis of documents in PDF format. The proposal integrates vector search techniques by embeddings, semantic data extraction and generation of contextualized natural language responses. To validate the approach, we conducted experiments with drug package inserts extracted from official public sources. The semantic queries applied were evaluated by metrics such as accuracy, completeness, response speed and consistency. The results indicate that the combination of RAG with LLMs offers significant gains in intelligent information retrieval and interpretation of unstructured technical texts. 

---
# Emotionally Aware Moderation: The Potential of Emotion Monitoring in Shaping Healthier Social Media Conversations 

**Authors**: Xiaotian Su, Naim Zierau, Soomin Kim, April Yi Wang, Thiemo Wambsganss  

**Link**: [PDF](https://arxiv.org/pdf/2507.21089)  

**Abstract**: Social media platforms increasingly employ proactive moderation techniques, such as detecting and curbing toxic and uncivil comments, to prevent the spread of harmful content. Despite these efforts, such approaches are often criticized for creating a climate of censorship and failing to address the underlying causes of uncivil behavior. Our work makes both theoretical and practical contributions by proposing and evaluating two types of emotion monitoring dashboards to users' emotional awareness and mitigate hate speech. In a study involving 211 participants, we evaluate the effects of the two mechanisms on user commenting behavior and emotional experiences. The results reveal that these interventions effectively increase users' awareness of their emotional states and reduce hate speech. However, our findings also indicate potential unintended effects, including increased expression of negative emotions (Angry, Fear, and Sad) when discussing sensitive issues. These insights provide a basis for further research on integrating proactive emotion regulation tools into social media platforms to foster healthier digital interactions. 

---
# Can LLMs Reason About Trust?: A Pilot Study 

**Authors**: Anushka Debnath, Stephen Cranefield, Emiliano Lorini, Bastin Tony Roy Savarimuthu  

**Link**: [PDF](https://arxiv.org/pdf/2507.21075)  

**Abstract**: In human society, trust is an essential component of social attitude that helps build and maintain long-term, healthy relationships which creates a strong foundation for cooperation, enabling individuals to work together effectively and achieve shared goals. As many human interactions occur through electronic means such as using mobile apps, the potential arises for AI systems to assist users in understanding the social state of their relationships. In this paper we investigate the ability of Large Language Models (LLMs) to reason about trust between two individuals in an environment which requires fostering trust relationships. We also assess whether LLMs are capable of inducing trust by role-playing one party in a trust based interaction and planning actions which can instil trust. 

---
# R-Stitch: Dynamic Trajectory Stitching for Efficient Reasoning 

**Authors**: Zhuokun Chen, Zeren Chen, Jiahao He, Mingkui Tan, Jianfei Cai, Bohan Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2507.17307)  

**Abstract**: Chain-of-thought (CoT) reasoning enhances the problem-solving capabilities of large language models by encouraging step-by-step intermediate reasoning during inference. While effective, CoT introduces substantial computational overhead due to its reliance on autoregressive decoding over long token sequences. Existing acceleration strategies either reduce sequence length through early stopping or compressive reward designs, or improve decoding speed via speculative decoding with smaller models. However, speculative decoding suffers from limited speedup when the agreement between small and large models is low, and fails to exploit the potential advantages of small models in producing concise intermediate reasoning. In this paper, we present R-Stitch, a token-level, confidence-based hybrid decoding framework that accelerates CoT inference by switching between a small language model (SLM) and a large language model (LLM) along the reasoning trajectory. R-Stitch uses the SLM to generate tokens by default and delegates to the LLM only when the SLM's confidence falls below a threshold. This design avoids full-sequence rollback and selectively invokes the LLM on uncertain steps, preserving both efficiency and answer quality. R-Stitch is model-agnostic, training-free, and compatible with standard decoding pipelines. Experiments on math reasoning benchmarks demonstrate that R-Stitch achieves up to 85\% reduction in inference latency with negligible accuracy drop, highlighting its practical effectiveness in accelerating CoT reasoning. 

---
