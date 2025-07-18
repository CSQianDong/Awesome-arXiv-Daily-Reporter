# Comparing Apples to Oranges: A Dataset & Analysis of LLM Humour Understanding from Traditional Puns to Topical Jokes 

**Authors**: Tyler Loakman, William Thorne, Chenghua Lin  

**Link**: [PDF](https://arxiv.org/pdf/2507.13335)  

**Abstract**: Humour, as a complex language form, is derived from myriad aspects of life, whilst existing work on computational humour has focussed almost exclusively on short pun-based jokes. In this work, we investigate whether the ability of Large Language Models (LLMs) to explain humour depends on the particular humour form. We compare models on simple puns and more complex topical humour that requires knowledge of real-world entities and events. In doing so, we curate a dataset of 600 jokes split across 4 joke types and manually write high-quality explanations. These jokes include heterographic and homographic puns, contemporary internet humour, and topical jokes, where understanding relies on reasoning beyond "common sense", rooted instead in world knowledge regarding news events and pop culture. Using this dataset, we compare the zero-shot abilities of a range of LLMs to accurately and comprehensively explain jokes of different types, identifying key research gaps in the task of humour explanation. We find that none of the tested models (inc. reasoning models) are capable of reliably generating adequate explanations of all joke types, further highlighting the narrow focus of most works in computational humour on overly simple joke forms. 

---
# A Survey of Context Engineering for Large Language Models 

**Authors**: Lingrui Mei, Jiayu Yao, Yuyao Ge, Yiwei Wang, Baolong Bi, Yujun Cai, Jiazhi Liu, Mingyu Li, Zhong-Zhi Li, Duzhen Zhang, Chenlin Zhou, Jiayi Mao, Tianze Xia, Jiafeng Guo, Shenghua Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.13334)  

**Abstract**: The performance of Large Language Models (LLMs) is fundamentally determined by the contextual information provided during inference. This survey introduces Context Engineering, a formal discipline that transcends simple prompt design to encompass the systematic optimization of information payloads for LLMs. We present a comprehensive taxonomy decomposing Context Engineering into its foundational components and the sophisticated implementations that integrate them into intelligent systems. We first examine the foundational components: context retrieval and generation, context processing and context management. We then explore how these components are architecturally integrated to create sophisticated system implementations: retrieval-augmented generation (RAG), memory systems and tool-integrated reasoning, and multi-agent systems. Through this systematic analysis of over 1300 research papers, our survey not only establishes a technical roadmap for the field but also reveals a critical research gap: a fundamental asymmetry exists between model capabilities. While current models, augmented by advanced context engineering, demonstrate remarkable proficiency in understanding complex contexts, they exhibit pronounced limitations in generating equally sophisticated, long-form outputs. Addressing this gap is a defining priority for future research. Ultimately, this survey provides a unified framework for both researchers and engineers advancing context-aware AI. 

---
# The Imitation Game: Turing Machine Imitator is Length Generalizable Reasoner 

**Authors**: Zhouqi Hua, Wenwei Zhang, Chengqi Lyu, Yuzhe Gu, Songyang Gao, Kuikun Liu, Kai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.13332)  

**Abstract**: Length generalization, the ability to solve problems of longer sequences than those observed during training, poses a core challenge of Transformer-based large language models (LLM). Although existing studies have predominantly focused on data-driven approaches for arithmetic operations and symbolic manipulation tasks, these approaches tend to be task-specific with limited overall performance. To pursue a more general solution, this paper focuses on a broader case of reasoning problems that are computable, i.e., problems that algorithms can solve, thus can be solved by the Turing Machine. From this perspective, this paper proposes Turing MAchine Imitation Learning (TAIL) to improve the length generalization ability of LLMs. TAIL synthesizes chain-of-thoughts (CoT) data that imitate the execution process of a Turing Machine by computer programs, which linearly expands the reasoning steps into atomic states to alleviate shortcut learning and explicit memory fetch mechanism to reduce the difficulties of dynamic and long-range data access in elementary operations. To validate the reliability and universality of TAIL, we construct a challenging synthetic dataset covering 8 classes of algorithms and 18 tasks. Without bells and whistles, TAIL significantly improves the length generalization ability as well as the performance of Qwen2.5-7B on various tasks using only synthetic data, surpassing previous methods and DeepSeek-R1. The experimental results reveal that the key concepts in the Turing Machine, instead of the thinking styles, are indispensable for TAIL for length generalization, through which the model exhibits read-and-write behaviors consistent with the properties of the Turing Machine in their attention layers. This work provides a promising direction for future research in the learning of LLM reasoning from synthetic data. 

---
# Vision-and-Language Training Helps Deploy Taxonomic Knowledge but Does Not Fundamentally Alter It 

**Authors**: Yulu Qin, Dheeraj Varghese, Adam Dahlgren Lindström, Lucia Donatelli, Kanishka Misra, Najoung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2507.13328)  

**Abstract**: Does vision-and-language (VL) training change the linguistic representations of language models in meaningful ways? Most results in the literature have shown inconsistent or marginal differences, both behaviorally and representationally. In this work, we start from the hypothesis that the domain in which VL training could have a significant effect is lexical-conceptual knowledge, in particular its taxonomic organization. Through comparing minimal pairs of text-only LMs and their VL-trained counterparts, we first show that the VL models often outperform their text-only counterparts on a text-only question-answering task that requires taxonomic understanding of concepts mentioned in the questions. Using an array of targeted behavioral and representational analyses, we show that the LMs and VLMs do not differ significantly in terms of their taxonomic knowledge itself, but they differ in how they represent questions that contain concepts in a taxonomic relation vs. a non-taxonomic relation. This implies that the taxonomic knowledge itself does not change substantially through additional VL training, but VL training does improve the deployment of this knowledge in the context of a specific task, even when the presentation of the task is purely linguistic. 

---
# Social and Political Framing in Search Engine Results 

**Authors**: Amrit Poudel, Tim Weninger  

**Link**: [PDF](https://arxiv.org/pdf/2507.13325)  

**Abstract**: Search engines play a crucial role in shaping public discourse by influencing how information is accessed and framed. While prior research has extensively examined various dimensions of search bias -- such as content prioritization, indexical bias, political polarization, and sources of bias -- an important question remains underexplored: how do search engines and ideologically-motivated user queries contribute to bias in search results. This study analyzes the outputs of major search engines using a dataset of political and social topics. The findings reveal that search engines not only prioritize content in ways that reflect underlying biases but also that ideologically-driven user queries exacerbate these biases, resulting in the amplification of specific narratives. Moreover, significant differences were observed across search engines in terms of the sources they prioritize. These results suggest that search engines may play a pivotal role in shaping public perceptions by reinforcing ideological divides, thereby contributing to the broader issue of information polarization. 

---
# HapticCap: A Multimodal Dataset and Task for Understanding User Experience of Vibration Haptic Signals 

**Authors**: Guimin Hu, Daniel Hershcovich, Hasti Seifi  

**Link**: [PDF](https://arxiv.org/pdf/2507.13318)  

**Abstract**: Haptic signals, from smartphone vibrations to virtual reality touch feedback, can effectively convey information and enhance realism, but designing signals that resonate meaningfully with users is challenging. To facilitate this, we introduce a multimodal dataset and task, of matching user descriptions to vibration haptic signals, and highlight two primary challenges: (1) lack of large haptic vibration datasets annotated with textual descriptions as collecting haptic descriptions is time-consuming, and (2) limited capability of existing tasks and models to describe vibration signals in text. To advance this area, we create HapticCap, the first fully human-annotated haptic-captioned dataset, containing 92,070 haptic-text pairs for user descriptions of sensory, emotional, and associative attributes of vibrations. Based on HapticCap, we propose the haptic-caption retrieval task and present the results of this task from a supervised contrastive learning framework that brings together text representations within specific categories and vibrations. Overall, the combination of language model T5 and audio model AST yields the best performance in the haptic-caption retrieval task, especially when separately trained for each description category. 

---
# AbGen: Evaluating Large Language Models in Ablation Study Design and Evaluation for Scientific Research 

**Authors**: Yilun Zhao, Weiyuan Chen, Zhijian Xu, Manasi Patwardhan, Yixin Liu, Chengye Wang, Lovekesh Vig, Arman Cohan  

**Link**: [PDF](https://arxiv.org/pdf/2507.13300)  

**Abstract**: We introduce AbGen, the first benchmark designed to evaluate the capabilities of LLMs in designing ablation studies for scientific research. AbGen consists of 1,500 expert-annotated examples derived from 807 NLP papers. In this benchmark, LLMs are tasked with generating detailed ablation study designs for a specified module or process based on the given research context. Our evaluation of leading LLMs, such as DeepSeek-R1-0528 and o4-mini, highlights a significant performance gap between these models and human experts in terms of the importance, faithfulness, and soundness of the ablation study designs. Moreover, we demonstrate that current automated evaluation methods are not reliable for our task, as they show a significant discrepancy when compared to human assessment. To better investigate this, we develop AbGen-Eval, a meta-evaluation benchmark designed to assess the reliability of commonly used automated evaluation systems in measuring LLM performance on our task. We investigate various LLM-as-Judge systems on AbGen-Eval, providing insights for future research on developing more effective and reliable LLM-based evaluation systems for complex scientific tasks. 

---
# Multi-Agent Synergy-Driven Iterative Visual Narrative Synthesis 

**Authors**: Wang Xi, Quan Shi, Tian Yu, Yujie Peng, Jiayi Sun, Mengxing Ren, Zenghui Ding, Ningguang Yao  

**Link**: [PDF](https://arxiv.org/pdf/2507.13285)  

**Abstract**: Automated generation of high-quality media presentations is challenging, requiring robust content extraction, narrative planning, visual design, and overall quality optimization. Existing methods often produce presentations with logical inconsistencies and suboptimal layouts, thereby struggling to meet professional standards. To address these challenges, we introduce RCPS (Reflective Coherent Presentation Synthesis), a novel framework integrating three key components: (1) Deep Structured Narrative Planning; (2) Adaptive Layout Generation; (3) an Iterative Optimization Loop. Additionally, we propose PREVAL, a preference-based evaluation framework employing rationale-enhanced multi-dimensional models to assess presentation quality across Content, Coherence, and Design. Experimental results demonstrate that RCPS significantly outperforms baseline methods across all quality dimensions, producing presentations that closely approximate human expert standards. PREVAL shows strong correlation with human judgments, validating it as a reliable automated tool for assessing presentation quality. 

---
# Overview of the TalentCLEF 2025: Skill and Job Title Intelligence for Human Capital Management 

**Authors**: Luis Gasco, Hermenegildo Fabregat, Laura García-Sardiña, Paula Estrella, Daniel Deniz, Alvaro Rodrigo, Rabih Zbib  

**Link**: [PDF](https://arxiv.org/pdf/2507.13275)  

**Abstract**: Advances in natural language processing and large language models are driving a major transformation in Human Capital Management, with a growing interest in building smart systems based on language technologies for talent acquisition, upskilling strategies, and workforce planning. However, the adoption and progress of these technologies critically depend on the development of reliable and fair models, properly evaluated on public data and open benchmarks, which have so far been unavailable in this domain.
To address this gap, we present TalentCLEF 2025, the first evaluation campaign focused on skill and job title intelligence. The lab consists of two tasks: Task A - Multilingual Job Title Matching, covering English, Spanish, German, and Chinese; and Task B - Job Title-Based Skill Prediction, in English. Both corpora were built from real job applications, carefully anonymized, and manually annotated to reflect the complexity and diversity of real-world labor market data, including linguistic variability and gender-marked expressions.
The evaluations included monolingual and cross-lingual scenarios and covered the evaluation of gender bias.
TalentCLEF attracted 76 registered teams with more than 280 submissions. Most systems relied on information retrieval techniques built with multilingual encoder-based models fine-tuned with contrastive learning, and several of them incorporated large language models for data augmentation or re-ranking. The results show that the training strategies have a larger effect than the size of the model alone. TalentCLEF provides the first public benchmark in this field and encourages the development of robust, fair, and transferable language technologies for the labor market. 

---
# QuestA: Expanding Reasoning Capacity in LLMs via Question Augmentation 

**Authors**: Jiazheng Li, Hong Lu, Kaiyue Wen, Zaiwen Yang, Jiaxuan Gao, Hongzhou Lin, Yi Wu, Jingzhao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.13266)  

**Abstract**: Reinforcement learning (RL) has become a key component in training large language reasoning models (LLMs). However, recent studies questions its effectiveness in improving multi-step reasoning-particularly on hard problems. To address this challenge, we propose a simple yet effective strategy via Question Augmentation: introduce partial solutions during training to reduce problem difficulty and provide more informative learning signals. Our method, QuestA, when applied during RL training on math reasoning tasks, not only improves pass@1 but also pass@k-particularly on problems where standard RL struggles to make progress. This enables continual improvement over strong open-source models such as DeepScaleR and OpenMath Nemotron, further enhancing their reasoning capabilities. We achieve new state-of-the-art results on math benchmarks using 1.5B-parameter models: 67.1% (+5.3%) on AIME24, 59.5% (+10.0%) on AIME25, and 35.5% (+4.0%) on HMMT25. Further, we provide theoretical explanations that QuestA improves sample efficiency, offering a practical and generalizable pathway for expanding reasoning capability through RL. 

---
# Automating Steering for Safe Multimodal Large Language Models 

**Authors**: Lyucheng Wu, Mengru Wang, Ziwen Xu, Tri Cao, Nay Oo, Bryan Hooi, Shumin Deng  

**Link**: [PDF](https://arxiv.org/pdf/2507.13255)  

**Abstract**: Recent progress in Multimodal Large Language Models (MLLMs) has unlocked powerful cross-modal reasoning abilities, but also raised new safety concerns, particularly when faced with adversarial multimodal inputs. To improve the safety of MLLMs during inference, we introduce a modular and adaptive inference-time intervention technology, AutoSteer, without requiring any fine-tuning of the underlying model. AutoSteer incorporates three core components: (1) a novel Safety Awareness Score (SAS) that automatically identifies the most safety-relevant distinctions among the model's internal layers; (2) an adaptive safety prober trained to estimate the likelihood of toxic outputs from intermediate representations; and (3) a lightweight Refusal Head that selectively intervenes to modulate generation when safety risks are detected. Experiments on LLaVA-OV and Chameleon across diverse safety-critical benchmarks demonstrate that AutoSteer significantly reduces the Attack Success Rate (ASR) for textual, visual, and cross-modal threats, while maintaining general abilities. These findings position AutoSteer as a practical, interpretable, and effective framework for safer deployment of multimodal AI systems. 

---
# HATS: Hindi Analogy Test Set for Evaluating Reasoning in Large Language Models 

**Authors**: Ashray Gupta, Rohan Joseph, Sunny Rai  

**Link**: [PDF](https://arxiv.org/pdf/2507.13238)  

**Abstract**: Analogies test a model's ability to infer implicit relationships between concepts, making them a key benchmark for evaluating reasoning capabilities. While large language models (LLMs) are widely evaluated for reasoning in English, their abilities in Indic languages remain understudied, limiting our understanding of whether these models generalize across languages. To address this gap, we introduce a new Hindi Analogy Test Set (HATS), comprising 405 multiple-choice questions sourced from Indian government exams. We benchmark state-of-the-art multilingual LLMs using various prompting strategies and introduce a grounded Chain of Thought approach that leverages cognitive theories of analogical reasoning. This approach improves model performance on Hindi analogy questions. Our experiments show that models perform best with English prompts, irrespective of the prompting strategy. Our test set addresses the lack of a critical resource to evaluate LLM reasoning capabilities in Hindi. 

---
# Enhancing Cross-task Transfer of Large Language Models via Activation Steering 

**Authors**: Xinyu Tang, Zhihao Lv, Xiaoxue Cheng, Junyi Li, Wayne Xin Zhao, Zujie Wen, Zhiqiang Zhang, Jun Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2507.13236)  

**Abstract**: Large language models (LLMs) have shown impressive abilities in leveraging pretrained knowledge through prompting, but they often struggle with unseen tasks, particularly in data-scarce scenarios. While cross-task in-context learning offers a direct solution for transferring knowledge across tasks, it still faces critical challenges in terms of robustness, scalability, and efficiency. In this paper, we investigate whether cross-task transfer can be achieved via latent space steering without parameter updates or input expansion. Through an analysis of activation patterns in the latent space of LLMs, we observe that the enhanced activations induced by in-context examples have consistent patterns across different tasks. Inspired by these findings, we propose CAST, a novel Cross-task Activation Steering Transfer framework that enables effective transfer by manipulating the model's internal activation states. Our approach first selects influential and diverse samples from high-resource tasks, then utilizes their contrastive representation-enhanced activations to adapt LLMs to low-resource tasks. Extensive experiments across both cross-domain and cross-lingual transfer settings show that our method outperforms competitive baselines and demonstrates superior scalability and lower computational costs. 

---
# Automatically assessing oral narratives of Afrikaans and isiXhosa children 

**Authors**: R. Louw, E. Sharratt, F. de Wet, C. Jacobs, A. Smith, H. Kamper  

**Link**: [PDF](https://arxiv.org/pdf/2507.13205)  

**Abstract**: Developing narrative and comprehension skills in early childhood is critical for later literacy. However, teachers in large preschool classrooms struggle to accurately identify students who require intervention. We present a system for automatically assessing oral narratives of preschool children in Afrikaans and isiXhosa. The system uses automatic speech recognition followed by a machine learning scoring model to predict narrative and comprehension scores. For scoring predicted transcripts, we compare a linear model to a large language model (LLM). The LLM-based system outperforms the linear model in most cases, but the linear system is competitive despite its simplicity. The LLM-based system is comparable to a human expert in flagging children who require intervention. We lay the foundation for automatic oral assessments in classrooms, giving teachers extra capacity to focus on personalised support for children's learning. 

---
# GEMMAS: Graph-based Evaluation Metrics for Multi Agent Systems 

**Authors**: Jisoo Lee, Raeyoung Chang, Dongwook Kwon, Harmanpreet Singh, Nikhil Verma  

**Link**: [PDF](https://arxiv.org/pdf/2507.13190)  

**Abstract**: Multi-agent systems built on language models have shown strong performance on collaborative reasoning tasks. However, existing evaluations focus only on the correctness of the final output, overlooking how inefficient communication and poor coordination contribute to redundant reasoning and higher computational costs. We introduce GEMMAS, a graph-based evaluation framework that analyzes the internal collaboration process by modeling agent interactions as a directed acyclic graph. To capture collaboration quality, we propose two process-level metrics: Information Diversity Score (IDS) to measure semantic variation in inter-agent messages, and Unnecessary Path Ratio (UPR) to quantify redundant reasoning paths. We evaluate GEMMAS across five benchmarks and highlight results on GSM8K, where systems with only a 2.1% difference in accuracy differ by 12.8% in IDS and 80% in UPR, revealing substantial variation in internal collaboration. These findings demonstrate that outcome-only metrics are insufficient for evaluating multi-agent performance and highlight the importance of process-level diagnostics in designing more interpretable and resource-efficient collaborative AI systems. 

---
# Feature-based analysis of oral narratives from Afrikaans and isiXhosa children 

**Authors**: Emma Sharratt, Annelien Smith, Retief Louw, Daleen Klop, Febe de Wet, Herman Kamper  

**Link**: [PDF](https://arxiv.org/pdf/2507.13164)  

**Abstract**: Oral narrative skills are strong predictors of later literacy development. This study examines the features of oral narratives from children who were identified by experts as requiring intervention. Using simple machine learning methods, we analyse recorded stories from four- and five-year-old Afrikaans- and isiXhosa-speaking children. Consistent with prior research, we identify lexical diversity (unique words) and length-based features (mean utterance length) as indicators of typical development, but features like articulation rate prove less informative. Despite cross-linguistic variation in part-of-speech patterns, the use of specific verbs and auxiliaries associated with goal-directed storytelling is correlated with a reduced likelihood of requiring intervention. Our analysis of two linguistically distinct languages reveals both language-specific and shared predictors of narrative proficiency, with implications for early assessment in multilingual contexts. 

---
# Assessing the Reliability of LLMs Annotations in the Context of Demographic Bias and Model Explanation 

**Authors**: Hadi Mohammadi, Tina Shahedi, Pablo Mosteiro, Massimo Poesio, Ayoub Bagheri, Anastasia Giachanou  

**Link**: [PDF](https://arxiv.org/pdf/2507.13138)  

**Abstract**: Understanding the sources of variability in annotations is crucial for developing fair NLP systems, especially for tasks like sexism detection where demographic bias is a concern. This study investigates the extent to which annotator demographic features influence labeling decisions compared to text content. Using a Generalized Linear Mixed Model, we quantify this inf luence, finding that while statistically present, demographic factors account for a minor fraction ( 8%) of the observed variance, with tweet content being the dominant factor. We then assess the reliability of Generative AI (GenAI) models as annotators, specifically evaluating if guiding them with demographic personas improves alignment with human judgments. Our results indicate that simplistic persona prompting often fails to enhance, and sometimes degrades, performance compared to baseline models. Furthermore, explainable AI (XAI) techniques reveal that model predictions rely heavily on content-specific tokens related to sexism, rather than correlates of demographic characteristics. We argue that focusing on content-driven explanations and robust annotation protocols offers a more reliable path towards fairness than potentially persona simulation. 

---
# A Computational Framework to Identify Self-Aspects in Text 

**Authors**: Jaya Caporusso, Matthew Purver, Senja Pollak  

**Link**: [PDF](https://arxiv.org/pdf/2507.13115)  

**Abstract**: This Ph.D. proposal introduces a plan to develop a computational framework to identify Self-aspects in text. The Self is a multifaceted construct and it is reflected in language. While it is described across disciplines like cognitive science and phenomenology, it remains underexplored in natural language processing (NLP). Many of the aspects of the Self align with psychological and other well-researched phenomena (e.g., those related to mental health), highlighting the need for systematic NLP-based analysis. In line with this, we plan to introduce an ontology of Self-aspects and a gold-standard annotated dataset. Using this foundation, we will develop and evaluate conventional discriminative models, generative large language models, and embedding-based retrieval approaches against four main criteria: interpretability, ground-truth adherence, accuracy, and computational efficiency. Top-performing models will be applied in case studies in mental health and empirical phenomenology. 

---
# SemCSE: Semantic Contrastive Sentence Embeddings Using LLM-Generated Summaries For Scientific Abstracts 

**Authors**: Marc Brinner, Sina Zarriess  

**Link**: [PDF](https://arxiv.org/pdf/2507.13105)  

**Abstract**: We introduce SemCSE, an unsupervised method for learning semantic embeddings of scientific texts. Building on recent advances in contrastive learning for text embeddings, our approach leverages LLM-generated summaries of scientific abstracts to train a model that positions semantically related summaries closer together in the embedding space. This resulting objective ensures that the model captures the true semantic content of a text, in contrast to traditional citation-based approaches that do not necessarily reflect semantic similarity. To validate this, we propose a novel benchmark designed to assess a model's ability to understand and encode the semantic content of scientific texts, demonstrating that our method enforces a stronger semantic separation within the embedding space. Additionally, we evaluate SemCSE on the comprehensive SciRepEval benchmark for scientific text embeddings, where it achieves state-of-the-art performance among models of its size, thus highlighting the benefits of a semantically focused training approach. 

---
# Formalizing Attack Scenario Description: A Proposed Model 

**Authors**: Quentin Goux, Nadira Lammari  

**Link**: [PDF](https://arxiv.org/pdf/2507.13076)  

**Abstract**: Organizations face an ever-changing threat landscape. They must continuously dedicate significant efforts to protect their assets, making their adoption of increased cybersecurity automation inevitable. However, process automation requires formalization of input data. Through this paper, we address this need for processes that use attack scenarios as input. Among these processes, one can mention both the generation of scripts for attack simulation and training purposes, as well as the analysis of attacks. Therefore, the paper's main research contribution is a novel formal model that encompasses the attack's context description and its scenario. It is abstracted using UML class model. Once the description of our model done, we will show how it could serve an upstream attack analysis process. We will show also its use for an automatic generation of attack scripts in the context of cybersecurity training. These two uses cases constitute the second contribution of this present research work. 

---
# MRT at IberLEF-2025 PRESTA Task: Maximizing Recovery from Tables with Multiple Steps 

**Authors**: Maximiliano Hormazábal Lagos, Álvaro Bueno Sáez, Héctor Cerezo-Costas, Pedro Alonso Doval, Jorge Alcalde Vesteiro  

**Link**: [PDF](https://arxiv.org/pdf/2507.12981)  

**Abstract**: This paper presents our approach for the IberLEF 2025 Task PRESTA: Preguntas y Respuestas sobre Tablas en Español (Questions and Answers about Tables in Spanish). Our solution obtains answers to the questions by implementing Python code generation with LLMs that is used to filter and process the table. This solution evolves from the MRT implementation for the Semeval 2025 related task. The process consists of multiple steps: analyzing and understanding the content of the table, selecting the useful columns, generating instructions in natural language, translating these instructions to code, running it, and handling potential errors or exceptions. These steps use open-source LLMs and fine-grained optimized prompts for each step. With this approach, we achieved an accuracy score of 85\% in the task. 

---
# Making Language Model a Hierarchical Classifier and Generator 

**Authors**: Yihong Wang, Zhonglin Jiang, Ningyuan Xi, Yue Zhao, Qingqing Gu, Xiyuan Chen, Hao Wu, Sheng Xu, Hange Zhou, Yong Chen, Luo Ji  

**Link**: [PDF](https://arxiv.org/pdf/2507.12930)  

**Abstract**: Decoder-only language models, such as GPT and LLaMA, generally decode on the last layer. Motivated by human's hierarchical thinking capability, we propose that a hierarchical decoder architecture could be built with different layers decoding texts simultaneously. Due to limited time and computationally resources, we choose to adapt a pretrained language model into this form of hierarchical decoder. Language heads of the last layer are copied to different selected intermediate layers, and fine-tuned with different task inputs. By thorough experiments, we validate that these selective intermediate layers could be adapted to speak meaningful and reasonable contents, and this paradigm of hierarchical decoder can obtain state-of-the-art performances on multiple tasks such as hierarchical text classification, classification-guided generation, and hierarchical text generation. This study suggests the possibility of a generalized hierarchical reasoner, pretraining from scratch. 

---
# Are Knowledge and Reference in Multilingual Language Models Cross-Lingually Consistent? 

**Authors**: Xi Ai, Mahardika Krisna Ihsani, Min-Yen Kan  

**Link**: [PDF](https://arxiv.org/pdf/2507.12838)  

**Abstract**: Cross-lingual consistency should be considered to assess cross-lingual transferability, maintain the factuality of the model knowledge across languages, and preserve the parity of language model performance. We are thus interested in analyzing, evaluating, and interpreting cross-lingual consistency for factual knowledge. We examine code-mixed coreferential statements conveyed identical knowledge across languages to study cross-lingual knowledge consistency. We use some interpretability approaches to analyze the behavior of a model in cross-lingual contexts, discovering that multilingual models show different levels of consistency, subject to language families, linguistic factors, and a bottleneck in cross-lingual consistency on a particular layer. In addition, we evaluate common strategies aimed at improving multilingual performance to observe whether these strategies can improve knowledge consistency at the same time. While knowledge is not cross-lingual consistency in many cases, code-switching training and cross-lingual word alignment objectives show the most promising results, emphasizing the noteworthiness of cross-lingual alignment supervision and code-switching training for both multilingual performance and cross-lingual consistency enhancement. 

---
# Large Language Models' Internal Perception of Symbolic Music 

**Authors**: Andrew Shin, Kunitake Kaneko  

**Link**: [PDF](https://arxiv.org/pdf/2507.12808)  

**Abstract**: Large language models (LLMs) excel at modeling relationships between strings in natural language and have shown promise in extending to other symbolic domains like coding or mathematics. However, the extent to which they implicitly model symbolic music remains underexplored. This paper investigates how LLMs represent musical concepts by generating symbolic music data from textual prompts describing combinations of genres and styles, and evaluating their utility through recognition and generation tasks. We produce a dataset of LLM-generated MIDI files without relying on explicit musical training. We then train neural networks entirely on this LLM-generated MIDI dataset and perform genre and style classification as well as melody completion, benchmarking their performance against established models. Our results demonstrate that LLMs can infer rudimentary musical structures and temporal relationships from text, highlighting both their potential to implicitly encode musical patterns and their limitations due to a lack of explicit musical context, shedding light on their generative capabilities for symbolic music. 

---
# Learning Robust Negation Text Representations 

**Authors**: Thinh Hung Truong, Karin Verspoor, Trevor Cohn, Timothy Baldwin  

**Link**: [PDF](https://arxiv.org/pdf/2507.12782)  

**Abstract**: Despite rapid adoption of autoregressive large language models, smaller text encoders still play an important role in text understanding tasks that require rich contextualized representations. Negation is an important semantic function that is still not properly captured by such methods, affecting many downstream applications relying on text embeddings. We propose a strategy to improve negation robustness of text encoders, by distilling data from large language models using diverse patterns of negation and hedging. We adopt a standard contrastive learning strategy to finetune a strong BERT-based model, and observe large improvement in negation understanding capabilities while maintaining competitive performance on general benchmarks. In addition, we also show that our method can be adapted to LLMs, leading to improved performance on negation benchmarks. 

---
# Synergy: End-to-end Concept Model 

**Authors**: Keli Zheng, Zerong Xie  

**Link**: [PDF](https://arxiv.org/pdf/2507.12769)  

**Abstract**: In this paper, we present Synergy, a language model that bridges different levels of abstraction in an end-to-end fashion through a learned routing mechanism. Focusing on low-level linguistic abstraction, we trained our model as a byte-level language model. Our model spontaneously learns to tokenize bytes, producing fewer concept tokens than Byte-level Byte Pair Encoder (BBPE) tokenizers while keeping comparable performance. By comparing with Llama3, we observed an advantage of Synergy under the same model scale and training dataset size. Further studies show that the middle part (the higher abstraction part) of our model performs better when positional encodings are removed, suggesting the emergence of position-independent concepts. These findings demonstrate the feasibility of tokenizer-free architectures, paving the way for more robust and flexible pipelines. 

---
# Logit Arithmetic Elicits Long Reasoning Capabilities Without Training 

**Authors**: Yunxiang Zhang, Muhammad Khalifa, Lechen Zhang, Xin Liu, Ayoung Lee, Xinliang Frederick Zhang, Farima Fatahi Bayat, Lu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.12759)  

**Abstract**: Large reasoning models (LRMs) can do complex reasoning via long chain-of-thought (CoT) involving cognitive strategies such as backtracking and self-correction. Recent studies suggest that some models inherently possess these long reasoning abilities, which may be unlocked via extra training. Our work first investigates whether we can elicit such behavior without any training. To this end, we propose a decoding-time approach, ThinkLogit, which utilizes logits arithmetic (Liu et al., 2024) to tune a target large LM for long reasoning using a substantially smaller model as guider. We then show that we can further boost performance by training the guider model with preference optimization over correct/incorrect reasoning pairs sampled from both the target and guider model -- a setup we refer to as ThinkLogit-DPO. Our experiments demonstrate that ThinkLogit and ThinkLogit-DPO achieve a relative improvement in pass@1 by 26% and 29%, respectively, over four mathematical datasets using the Qwen2.5-32B when guided by R1-Distill-Qwen-1.5B -- a model 21x smaller. Lastly, we show that ThinkLogit can transfer long reasoning skills acquired through reinforcement learning, improving pass@1 by 13% relative compared to the Qwen2.5-32B base model. Our work presents a computationally-efficient method to elicit long reasoning in large models with minimal or no additional training. 

---
# Strategy Adaptation in Large Language Model Werewolf Agents 

**Authors**: Fuya Nakamori, Yin Jou Huang, Fei Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2507.12732)  

**Abstract**: This study proposes a method to improve the performance of Werewolf agents by switching between predefined strategies based on the attitudes of other players and the context of conversations. While prior works of Werewolf agents using prompt engineering have employed methods where effective strategies are implicitly defined, they cannot adapt to changing situations. In this research, we propose a method that explicitly selects an appropriate strategy based on the game context and the estimated roles of other players. We compare the strategy adaptation Werewolf agents with baseline agents using implicit or fixed strategies and verify the effectiveness of our proposed method. 

---
# TransEvalnia: Reasoning-based Evaluation and Ranking of Translations 

**Authors**: Richard Sproat, Tianyu Zhao, Llion Jones  

**Link**: [PDF](https://arxiv.org/pdf/2507.12724)  

**Abstract**: We present TransEvalnia, a prompting-based translation evaluation and ranking system that uses reasoning in performing its evaluations and ranking. This system presents fine-grained evaluations based on a subset of the Multidimensional Quality Metrics (this https URL), returns an assessment of which translation it deems the best, and provides numerical scores for the various dimensions and for the overall translation. We show that TransEvalnia performs as well as or better than the state-of-the-art MT-Ranker (Moosa et al. 2024) on our own English-Japanese data as well as several language pairs from various WMT shared tasks. Using Anthropic's Claude-3.5-Sonnet and Qwen-2.5-72B-Instruct as the evaluation LLMs, we show that the evaluations returned are deemed highly acceptable to human raters, and that the scores assigned to the translations by Sonnet, as well as other LLMs, correlate well with scores assigned by the human raters. We also note the sensitivity of our system -- as well as MT-Ranker -- to the order in which the translations are presented, and we propose methods to address this position bias. All data, including the system's evaluation and reasoning, human assessments, as well as code is released. 

---
# FLEXITOKENS: Flexible Tokenization for Evolving Language Models 

**Authors**: Abraham Toluase Owodunni, Orevaoghene Ahia, Sachin Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2507.12720)  

**Abstract**: Language models (LMs) are challenging to adapt to new data distributions by simple finetuning. This is due to the rigidity of their subword tokenizers, which typically remain unchanged during adaptation. This inflexibility often leads to inefficient tokenization, causing overfragmentation of out-of-distribution domains, unseen languages, or scripts. In this work, we develop byte-level LMs with learnable tokenizers to make tokenization adaptive. Our models include a submodule that learns to predict boundaries between the input byte sequence, encoding it into variable-length segments. Existing tokenizer-free methods train this boundary predictor using an auxiliary loss that enforces a fixed compression rate across the training corpus, introducing a new kind of rigidity. We propose FLEXITOKENS, a simplified training objective that enables significantly greater flexibility during adaptation. Evaluating across multiple multilingual benchmarks, morphologically diverse tasks, and domains, we demonstrate that FLEXITOKENS consistently reduces token over-fragmentation and achieves up to 10\% improvements on downstream task performance compared to subword and other gradient-based tokenizers. Code and data for our experiments will be released at this https URL 

---
# AudioJudge: Understanding What Works in Large Audio Model Based Speech Evaluation 

**Authors**: Potsawee Manakul, Woody Haosheng Gan, Michael J. Ryan, Ali Sartaz Khan, Warit Sirichotedumrong, Kunat Pipatanakul, William Held, Diyi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2507.12705)  

**Abstract**: Current speech evaluation suffers from two critical limitations: the need and difficulty of designing specialized systems targeting individual audio characteristics, and poor correlation between automatic evaluation methods and human preferences. This work presents a systematic study of Large Audio Model (LAM) as a Judge, AudioJudge, investigating whether it can provide a unified evaluation framework that addresses both challenges. We systematically explore AudioJudge across audio characteristic detection tasks, including pronunciation, speaking rate, speaker identification and speech quality, and system-level human preference simulation for automated benchmarking. We investigate different prompt engineering strategies, finding that audio concatenation combined with in-context learning significantly improves performance across both audio characteristic detection and human preference simulation tasks. We further introduce a multi-aspect ensemble AudioJudge to enable general-purpose multi-aspect audio evaluation. This method decomposes speech assessment into specialized judges for lexical content, speech quality, and paralinguistic features, achieving up to 0.91 Spearman correlation with human preferences on our system ranking benchmark. Robustness analysis reveals that while LAMs maintain strong performance under acoustic noise, they exhibit significant verbosity and positional biases that require careful mitigation. 

---
# AdaptiSent: Context-Aware Adaptive Attention for Multimodal Aspect-Based Sentiment Analysis 

**Authors**: S M Rafiuddin, Sadia Kamal, Mohammed Rakib, Arunkumar Bagavathi, Atriya Sen  

**Link**: [PDF](https://arxiv.org/pdf/2507.12695)  

**Abstract**: We introduce AdaptiSent, a new framework for Multimodal Aspect-Based Sentiment Analysis (MABSA) that uses adaptive cross-modal attention mechanisms to improve sentiment classification and aspect term extraction from both text and images. Our model integrates dynamic modality weighting and context-adaptive attention, enhancing the extraction of sentiment and aspect-related information by focusing on how textual cues and visual context interact. We tested our approach against several baselines, including traditional text-based models and other multimodal methods. Results from standard Twitter datasets show that AdaptiSent surpasses existing models in precision, recall, and F1 score, and is particularly effective in identifying nuanced inter-modal relationships that are crucial for accurate sentiment and aspect term extraction. This effectiveness comes from the model's ability to adjust its focus dynamically based on the context's relevance, improving the depth and accuracy of sentiment analysis across various multimodal data sets. AdaptiSent sets a new standard for MABSA, significantly outperforming current methods, especially in understanding complex multimodal information. 

---
# Improving Drug Identification in Overdose Death Surveillance using Large Language Models 

**Authors**: Arthur J. Funnell, Panayiotis Petousis, Fabrice Harel-Canada, Ruby Romero, Alex A. T. Bui, Adam Koncsol, Hritika Chaturvedi, Chelsea Shover, David Goodman-Meza  

**Link**: [PDF](https://arxiv.org/pdf/2507.12679)  

**Abstract**: The rising rate of drug-related deaths in the United States, largely driven by fentanyl, requires timely and accurate surveillance. However, critical overdose data are often buried in free-text coroner reports, leading to delays and information loss when coded into ICD (International Classification of Disease)-10 classifications. Natural language processing (NLP) models may automate and enhance overdose surveillance, but prior applications have been limited. A dataset of 35,433 death records from multiple U.S. jurisdictions in 2020 was used for model training and internal testing. External validation was conducted using a novel separate dataset of 3,335 records from 2023-2024. Multiple NLP approaches were evaluated for classifying specific drug involvement from unstructured death certificate text. These included traditional single- and multi-label classifiers, as well as fine-tuned encoder-only language models such as Bidirectional Encoder Representations from Transformers (BERT) and BioClinicalBERT, and contemporary decoder-only large language models such as Qwen 3 and Llama 3. Model performance was assessed using macro-averaged F1 scores, and 95% confidence intervals were calculated to quantify uncertainty. Fine-tuned BioClinicalBERT models achieved near-perfect performance, with macro F1 scores >=0.998 on the internal test set. External validation confirmed robustness (macro F1=0.966), outperforming conventional machine learning, general-domain BERT models, and various decoder-only large language models. NLP models, particularly fine-tuned clinical variants like BioClinicalBERT, offer a highly accurate and scalable solution for overdose death classification from free-text reports. These methods can significantly accelerate surveillance workflows, overcoming the limitations of manual ICD-10 coding and supporting near real-time detection of emerging substance use trends. 

---
# The first open machine translation system for the Chechen language 

**Authors**: Abu-Viskhan A. Umishov, Vladislav A. Grigorian  

**Link**: [PDF](https://arxiv.org/pdf/2507.12672)  

**Abstract**: We introduce the first open-source model for translation between the vulnerable Chechen language and Russian, and the dataset collected to train and evaluate it. We explore fine-tuning capabilities for including a new language into a large language model system for multilingual translation NLLB-200. The BLEU / ChrF++ scores for our model are 8.34 / 34.69 and 20.89 / 44.55 for translation from Russian to Chechen and reverse direction, respectively. The release of the translation models is accompanied by the distribution of parallel words, phrases and sentences corpora and multilingual sentence encoder adapted to the Chechen language. 

---
# Is This Just Fantasy? Language Model Representations Reflect Human Judgments of Event Plausibility 

**Authors**: Michael A. Lepori, Jennifer Hu, Ishita Dasgupta, Roma Patel, Thomas Serre, Ellie Pavlick  

**Link**: [PDF](https://arxiv.org/pdf/2507.12553)  

**Abstract**: Language models (LMs) are used for a diverse range of tasks, from question answering to writing fantastical stories. In order to reliably accomplish these tasks, LMs must be able to discern the modal category of a sentence (i.e., whether it describes something that is possible, impossible, completely nonsensical, etc.). However, recent studies have called into question the ability of LMs to categorize sentences according to modality (Michaelov et al., 2025; Kauf et al., 2023). In this work, we identify linear representations that discriminate between modal categories within a variety of LMs, or modal difference vectors. Analysis of modal difference vectors reveals that LMs have access to more reliable modal categorization judgments than previously reported. Furthermore, we find that modal difference vectors emerge in a consistent order as models become more competent (i.e., through training steps, layers, and parameter count). Notably, we find that modal difference vectors identified within LM activations can be used to model fine-grained human categorization behavior. This potentially provides a novel view into how human participants distinguish between modal categories, which we explore by correlating projections along modal difference vectors with human participants' ratings of interpretable features. In summary, we derive new insights into LM modal categorization using techniques from mechanistic interpretability, with the potential to inform our understanding of modal categorization in humans. 

---
# Modeling Open-World Cognition as On-Demand Synthesis of Probabilistic Models 

**Authors**: Lionel Wong, Katherine M. Collins, Lance Ying, Cedegao E. Zhang, Adrian Weller, Tobias Gersternberg, Timothy O'Donnell, Alexander K. Lew, Jacob D. Andreas, Joshua B. Tenenbaum, Tyler Brooke-Wilson  

**Link**: [PDF](https://arxiv.org/pdf/2507.12547)  

**Abstract**: When faced with novel situations, people are able to marshal relevant considerations from a wide range of background knowledge and put these to use in inferences and predictions. What permits us to draw in globally relevant information and reason over it coherently? Here, we explore the hypothesis that people use a combination of distributed and symbolic representations to construct bespoke mental models tailored to novel situations. We propose a computational implementation of this idea -- a ``Model Synthesis Architecture'' (MSA) -- using language models to implement global relevance-based retrieval and model synthesis and probabilistic programs to implement bespoke, coherent world models. We evaluate our MSA as a model of human judgments on a novel reasoning dataset. The dataset -- built around a `Model Olympics` domain of sports vignettes -- tests models' capacity for human-like, open-ended reasoning by requiring (i) judgments about novel causal structures described in language; (ii) drawing on large bodies of background knowledge; and (iii) doing both in light of observations that introduce arbitrary novel variables. Our MSA approach captures human judgments better than language model-only baselines, under both direct and chain-of-thought generations from the LM that supports model synthesis. These results suggest that MSAs can be implemented in a way that mirrors people's ability to deliver locally coherent reasoning over globally relevant variables, offering a path to understanding and replicating human reasoning in open-ended domains. 

---
# VisionThink: Smart and Efficient Vision Language Model via Reinforcement Learning 

**Authors**: Senqiao Yang, Junyi Li, Xin Lai, Bei Yu, Hengshuang Zhao, Jiaya Jia  

**Link**: [PDF](https://arxiv.org/pdf/2507.13348)  

**Abstract**: Recent advancements in vision-language models (VLMs) have improved performance by increasing the number of visual tokens, which are often significantly longer than text tokens. However, we observe that most real-world scenarios do not require such an extensive number of visual tokens. While the performance drops significantly in a small subset of OCR-related tasks, models still perform accurately in most other general VQA tasks with only 1/4 resolution. Therefore, we propose to dynamically process distinct samples with different resolutions, and present a new paradigm for visual token compression, namely, VisionThink. It starts with a downsampled image and smartly decides whether it is sufficient for problem solving. Otherwise, the model could output a special token to request the higher-resolution image. Compared to existing Efficient VLM methods that compress tokens using fixed pruning ratios or thresholds, VisionThink autonomously decides whether to compress tokens case by case. As a result, it demonstrates strong fine-grained visual understanding capability on OCR-related tasks, and meanwhile saves substantial visual tokens on simpler tasks. We adopt reinforcement learning and propose the LLM-as-Judge strategy to successfully apply RL to general VQA tasks. Moreover, we carefully design a reward function and penalty mechanism to achieve a stable and reasonable image resize call ratio. Extensive experiments demonstrate the superiority, efficiency, and effectiveness of our method. Our code is available at this https URL. 

---
# The Generative Energy Arena (GEA): Incorporating Energy Awareness in Large Language Model (LLM) Human Evaluations 

**Authors**: Carlos Arriaga, Gonzalo Martínez, Eneko Sendin, Javier Conde, Pedro Reviriego  

**Link**: [PDF](https://arxiv.org/pdf/2507.13302)  

**Abstract**: The evaluation of large language models is a complex task, in which several approaches have been proposed. The most common is the use of automated benchmarks in which LLMs have to answer multiple-choice questions of different topics. However, this method has certain limitations, being the most concerning, the poor correlation with the humans. An alternative approach, is to have humans evaluate the LLMs. This poses scalability issues as there is a large and growing number of models to evaluate making it impractical (and costly) to run traditional studies based on recruiting a number of evaluators and having them rank the responses of the models. An alternative approach is the use of public arenas, such as the popular LM arena, on which any user can freely evaluate models on any question and rank the responses of two models. The results are then elaborated into a model ranking. An increasingly important aspect of LLMs is their energy consumption and, therefore, evaluating how energy awareness influences the decisions of humans in selecting a model is of interest. In this paper, we present GEA, the Generative Energy Arena, an arena that incorporates information on the energy consumption of the model in the evaluation process. Preliminary results obtained with GEA are also presented, showing that for most questions, when users are aware of the energy consumption, they favor smaller and more energy efficient models. This suggests that for most user interactions, the extra cost and energy incurred by the more complex and top-performing models do not provide an increase in the perceived quality of the responses that justifies their use. 

---
# Inverse Reinforcement Learning Meets Large Language Model Post-Training: Basics, Advances, and Opportunities 

**Authors**: Hao Sun, Mihaela van der Schaar  

**Link**: [PDF](https://arxiv.org/pdf/2507.13158)  

**Abstract**: In the era of Large Language Models (LLMs), alignment has emerged as a fundamental yet challenging problem in the pursuit of more reliable, controllable, and capable machine intelligence. The recent success of reasoning models and conversational AI systems has underscored the critical role of reinforcement learning (RL) in enhancing these systems, driving increased research interest at the intersection of RL and LLM alignment. This paper provides a comprehensive review of recent advances in LLM alignment through the lens of inverse reinforcement learning (IRL), emphasizing the distinctions between RL techniques employed in LLM alignment and those in conventional RL tasks. In particular, we highlight the necessity of constructing neural reward models from human data and discuss the formal and practical implications of this paradigm shift. We begin by introducing fundamental concepts in RL to provide a foundation for readers unfamiliar with the field. We then examine recent advances in this research agenda, discussing key challenges and opportunities in conducting IRL for LLM alignment. Beyond methodological considerations, we explore practical aspects, including datasets, benchmarks, evaluation metrics, infrastructure, and computationally efficient training and inference techniques. Finally, we draw insights from the literature on sparse-reward RL to identify open questions and potential research directions. By synthesizing findings from diverse studies, we aim to provide a structured and critical overview of the field, highlight unresolved challenges, and outline promising future directions for improving LLM alignment through RL and IRL techniques. 

---
# From Roots to Rewards: Dynamic Tree Reasoning with RL 

**Authors**: Ahmed Bahloul, Simon Malberg  

**Link**: [PDF](https://arxiv.org/pdf/2507.13142)  

**Abstract**: Modern language models address complex questions through chain-of-thought (CoT) reasoning (Wei et al., 2023) and retrieval augmentation (Lewis et al., 2021), yet struggle with error propagation and knowledge integration. Tree-structured reasoning methods, particularly the Probabilistic Tree-of-Thought (ProbTree)(Cao et al., 2023) framework, mitigate these issues by decomposing questions into hierarchical structures and selecting answers through confidence-weighted aggregation of parametric and retrieved knowledge (Yao et al., 2023). However, ProbTree's static implementation introduces two key limitations: (1) the reasoning tree is fixed during the initial construction phase, preventing dynamic adaptation to intermediate results, and (2) each node requires exhaustive evaluation of all possible solution strategies, creating computational inefficiency. We present a dynamic reinforcement learning (Sutton and Barto, 2018) framework that transforms tree-based reasoning into an adaptive process. Our approach incrementally constructs the reasoning tree based on real-time confidence estimates, while learning optimal policies for action selection (decomposition, retrieval, or aggregation). This maintains ProbTree's probabilistic rigor while improving both solution quality and computational efficiency through selective expansion and focused resource allocation. The work establishes a new paradigm for treestructured reasoning that balances the reliability of probabilistic frameworks with the flexibility required for real-world question answering systems. 

---
# Rethinking the Embodied Gap in Vision-and-Language Navigation: A Holistic Study of Physical and Visual Disparities 

**Authors**: Liuyi Wang, Xinyuan Xia, Hui Zhao, Hanqing Wang, Tai Wang, Yilun Chen, Chengju Liu, Qijun Chen, Jiangmiao Pang  

**Link**: [PDF](https://arxiv.org/pdf/2507.13019)  

**Abstract**: Recent Vision-and-Language Navigation (VLN) advancements are promising, but their idealized assumptions about robot movement and control fail to reflect physically embodied deployment challenges. To bridge this gap, we introduce VLN-PE, a physically realistic VLN platform supporting humanoid, quadruped, and wheeled robots. For the first time, we systematically evaluate several ego-centric VLN methods in physical robotic settings across different technical pipelines, including classification models for single-step discrete action prediction, a diffusion model for dense waypoint prediction, and a train-free, map-based large language model (LLM) integrated with path planning. Our results reveal significant performance degradation due to limited robot observation space, environmental lighting variations, and physical challenges like collisions and falls. This also exposes locomotion constraints for legged robots in complex environments. VLN-PE is highly extensible, allowing seamless integration of new scenes beyond MP3D, thereby enabling more comprehensive VLN evaluation. Despite the weak generalization of current models in physical deployment, VLN-PE provides a new pathway for improving cross-embodiment's overall adaptability. We hope our findings and tools inspire the community to rethink VLN limitations and advance robust, practical VLN models. The code is available at this https URL. 

---
# Teach Old SAEs New Domain Tricks with Boosting 

**Authors**: Nikita Koriagin, Yaroslav Aksenov, Daniil Laptev, Gleb Gerasimov, Nikita Balagansky, Daniil Gavrilov  

**Link**: [PDF](https://arxiv.org/pdf/2507.12990)  

**Abstract**: Sparse Autoencoders have emerged as powerful tools for interpreting the internal representations of Large Language Models, yet they often fail to capture domain-specific features not prevalent in their training corpora. This paper introduces a residual learning approach that addresses this feature blindness without requiring complete retraining. We propose training a secondary SAE specifically to model the reconstruction error of a pretrained SAE on domain-specific texts, effectively capturing features missed by the primary model. By summing the outputs of both models during inference, we demonstrate significant improvements in both LLM cross-entropy and explained variance metrics across multiple specialized domains. Our experiments show that this method efficiently incorporates new domain knowledge into existing SAEs while maintaining their performance on general tasks. This approach enables researchers to selectively enhance SAE interpretability for specific domains of interest, opening new possibilities for targeted mechanistic interpretability of LLMs. 

---
# UniSLU: Unified Spoken Language Understanding from Heterogeneous Cross-Task Datasets 

**Authors**: Zhichao Sheng, Shilin Zhou, Chen Gong, Zhenghua Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.12951)  

**Abstract**: Spoken Language Understanding (SLU) plays a crucial role in speech-centric multimedia applications, enabling machines to comprehend spoken language in scenarios such as meetings, interviews, and customer service interactions. SLU encompasses multiple tasks, including Automatic Speech Recognition (ASR), spoken Named Entity Recognition (NER), and spoken Sentiment Analysis (SA). However, existing methods often rely on separate model architectures for individual tasks such as spoken NER and SA, which increases system complexity, limits cross-task interaction, and fails to fully exploit heterogeneous datasets available across tasks. To address these limitations, we propose UniSLU, a unified framework that jointly models multiple SLU tasks within a single architecture. Specifically, we propose a unified representation for diverse SLU tasks, enabling full utilization of heterogeneous datasets across multiple tasks. Built upon this representation, we propose a unified generative method that jointly models ASR, spoken NER, and SA tasks, enhancing task interactions and enabling seamless integration with large language models to harness their powerful generative capabilities. Extensive experiments on public SLU datasets demonstrate the effectiveness of our approach, achieving superior SLU performance compared to several benchmark methods, making it well-suited for real-world speech-based multimedia scenarios. We will release all code and models at github to facilitate future research. 

---
# Probabilistic Soundness Guarantees in LLM Reasoning Chains 

**Authors**: Weiqiu You, Anton Xue, Shreya Havaldar, Delip Rao, Helen Jin, Chris Callison-Burch, Eric Wong  

**Link**: [PDF](https://arxiv.org/pdf/2507.12948)  

**Abstract**: In reasoning chains generated by large language models (LLMs), initial errors often propagate and undermine the reliability of the final conclusion. Current LLM-based error detection methods often fail to detect propagated errors because they do not properly account for how earlier errors might corrupt judgments of downstream reasoning. To better detect such propagated errors, we introduce Autoregressive Reasoning Entailment Stability (ARES), a novel probabilistic framework that prevents error propagation by judging each claim based only on previously-assessed sound premises. This inductive method yields a nuanced score for each step and provides certified statistical guarantees of its soundness, rather than a brittle binary label. ARES achieves state-of-the-art performance across four benchmarks (72.1% Macro-F1, +8.2 points) and demonstrates superior robustness on very long synthetic reasoning chains, where it excels at detecting propagated errors (90.3% F1, +27.6 points). 

---
# Emotional Support with LLM-based Empathetic Dialogue Generation 

**Authors**: Shiquan Wang, Ruiyu Fang, Zhongjiang He, Shuangyong Song, Yongxiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.12820)  

**Abstract**: Emotional Support Conversation (ESC) aims to provide empathetic and effective emotional assistance through dialogue, addressing the growing demand for mental health support. This paper presents our solution for the NLPCC 2025 Task 8 ESC evaluation, where we leverage large-scale language models enhanced by prompt engineering and finetuning techniques. We explore both parameter-efficient Low-Rank Adaptation and full-parameter fine-tuning strategies to improve the model's ability to generate supportive and contextually appropriate responses. Our best model ranked second in the competition, highlighting the potential of combining LLMs with effective adaptation methods for ESC tasks. Future work will focus on further enhancing emotional understanding and response personalization to build more practical and reliable emotional support systems. 

---
# MCPEval: Automatic MCP-based Deep Evaluation for AI Agent Models 

**Authors**: Zhiwei Liu, Jielin Qiu, Shiyu Wang, Jianguo Zhang, Zuxin Liu, Roshan Ram, Haolin Chen, Weiran Yao, Huan Wang, Shelby Heinecke, Silvio Savarese, Caiming Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2507.12806)  

**Abstract**: The rapid rise of Large Language Models (LLMs)-based intelligent agents underscores the need for robust, scalable evaluation frameworks. Existing methods rely on static benchmarks and labor-intensive data collection, limiting practical assessment. We introduce \oursystemname, an open-source Model Context Protocol (MCP)-based framework that automates end-to-end task generation and deep evaluation of LLM agents across diverse domains. MCPEval standardizes metrics, seamlessly integrates with native agent tools, and eliminates manual effort in building evaluation pipelines. Empirical results across five real-world domains show its effectiveness in revealing nuanced, domain-specific performance. We publicly release MCPEval this https URL to promote reproducible and standardized LLM agent evaluation. 

---
# PMKLC: Parallel Multi-Knowledge Learning-based Lossless Compression for Large-Scale Genomics Database 

**Authors**: Hui Sun, Yanfeng Ding, Liping Yi, Huidong Ma, Gang Wang, Xiaoguang Liu, Cheng Zhong, Wentong Cai  

**Link**: [PDF](https://arxiv.org/pdf/2507.12805)  

**Abstract**: Learning-based lossless compressors play a crucial role in large-scale genomic database backup, storage, transmission, and management. However, their 1) inadequate compression ratio, 2) low compression \& decompression throughput, and 3) poor compression robustness limit their widespread adoption and application in both industry and academia. To solve those challenges, we propose a novel \underline{P}arallel \underline{M}ulti-\underline{K}nowledge \underline{L}earning-based \underline{C}ompressor (PMKLC) with four crucial designs: 1) We propose an automated multi-knowledge learning-based compression framework as compressors' backbone to enhance compression ratio and robustness; 2) we design a GPU-accelerated ($s$,$k$)-mer encoder to optimize compression throughput and computing resource usage; 3) we introduce data block partitioning and Step-wise Model Passing (SMP) mechanisms for parallel acceleration; 4) We design two compression modes PMKLC-S and PMKLC-M to meet the complex application scenarios, where the former runs on a resource-constrained single GPU and the latter is multi-GPU accelerated. We benchmark PMKLC-S/M and 14 baselines (7 traditional and 7 leaning-based) on 15 real-world datasets with different species and data sizes. Compared to baselines on the testing datasets, PMKLC-S/M achieve the average compression ratio improvement up to 73.609\% and 73.480\%, the average throughput improvement up to 3.036$\times$ and 10.710$\times$, respectively. Besides, PMKLC-S/M also achieve the best robustness and competitive memory cost, indicating its greater stability against datasets with different probability distribution perturbations, and its strong ability to run on memory-constrained devices. 

---
# A Comprehensive Survey of Electronic Health Record Modeling: From Deep Learning Approaches to Large Language Models 

**Authors**: Weijieying Ren, Jingxi Zhu, Zehao Liu, Tianxiang Zhao, Vasant Honavar  

**Link**: [PDF](https://arxiv.org/pdf/2507.12774)  

**Abstract**: Artificial intelligence (AI) has demonstrated significant potential in transforming healthcare through the analysis and modeling of electronic health records (EHRs). However, the inherent heterogeneity, temporal irregularity, and domain-specific nature of EHR data present unique challenges that differ fundamentally from those in vision and natural language tasks. This survey offers a comprehensive overview of recent advancements at the intersection of deep learning, large language models (LLMs), and EHR modeling. We introduce a unified taxonomy that spans five key design dimensions: data-centric approaches, neural architecture design, learning-focused strategies, multimodal learning, and LLM-based modeling systems. Within each dimension, we review representative methods addressing data quality enhancement, structural and temporal representation, self-supervised learning, and integration with clinical knowledge. We further highlight emerging trends such as foundation models, LLM-driven clinical agents, and EHR-to-text translation for downstream reasoning. Finally, we discuss open challenges in benchmarking, explainability, clinical alignment, and generalization across diverse clinical settings. This survey aims to provide a structured roadmap for advancing AI-driven EHR modeling and clinical decision support. For a comprehensive list of EHR-related methods, kindly refer to this https URL. 

---
# A Fuzzy Approach to Project Success: Measuring What Matters 

**Authors**: João Granja-Correia, Remedios Hernández-Linares, Luca Ferranti, Arménio Rego  

**Link**: [PDF](https://arxiv.org/pdf/2507.12653)  

**Abstract**: This paper introduces a novel approach to project success evaluation by integrating fuzzy logic into an existing construct. Traditional Likert-scale measures often overlook the context-dependent and multifaceted nature of project success. The proposed hierarchical Type-1 Mamdani fuzzy system prioritizes sustained positive impact for end-users, reducing emphasis on secondary outcomes like stakeholder satisfaction and internal project success. This dynamic approach may provide a more accurate measure of project success and could be adaptable to complex evaluations. Future research will focus on empirical testing and broader applications of fuzzy logic in social science. 

---
# Mono-InternVL-1.5: Towards Cheaper and Faster Monolithic Multimodal Large Language Models 

**Authors**: Gen Luo, Wenhan Dou, Wenhao Li, Zhaokai Wang, Xue Yang, Changyao Tian, Hao Li, Weiyun Wang, Wenhai Wang, Xizhou Zhu, Yu Qiao, Jifeng Dai  

**Link**: [PDF](https://arxiv.org/pdf/2507.12566)  

**Abstract**: This paper focuses on monolithic Multimodal Large Language Models (MLLMs), which integrate visual encoding and language decoding into a single model. Existing structures and pre-training strategies for monolithic MLLMs often suffer from unstable optimization and catastrophic forgetting. To address these challenges, our key idea is to embed a new visual parameter space into a pre-trained LLM, enabling stable learning of visual knowledge from noisy data via delta tuning. Based on this principle, we first introduce Mono-InternVL, an advanced monolithic MLLM that incorporates a set of visual experts through a multimodal mixture-of-experts architecture. In addition, we design an innovative Endogenous Visual Pre-training (EViP) for Mono-InternVL to maximize its visual capabilities via progressive learning. Mono-InternVL achieves competitive performance against existing MLLMs but also leads to relatively expensive data cost. Therefore, we further present Mono-InternVL-1.5, a cheaper and stronger monolithic MLLM equipped with an improved EViP (EViP++). EViP++ introduces additional visual attention experts to Mono-InternVL-1.5 and re-organizes the pre-training process in an efficient manner. During inference, it includes a fused CUDA kernel to speed up its MoE operations. With these designs, Mono-InternVL-1.5 significantly reduces training and inference costs, while still maintaining competitive performance with Mono-InternVL. To evaluate our approach, we conduct extensive experiments across 15 benchmarks. Results demonstrate that Mono-InternVL outperforms existing monolithic MLLMs on 12 out of 15 benchmarks, e.g., +114-point improvement over Emu3 on OCRBench. Compared to its modular counterpart, i.e., InternVL-1.5, Mono-InternVL-1.5 achieves similar multimodal performance while reducing first-token latency by up to 69%. Code and models are released at this https URL. 

---
# Scaling Up RL: Unlocking Diverse Reasoning in LLMs via Prolonged Training 

**Authors**: Mingjie Liu, Shizhe Diao, Jian Hu, Ximing Lu, Xin Dong, Hao Zhang, Alexander Bukharin, Shaokun Zhang, Jiaqi Zeng, Makesh Narsimhan Sreedhar, Gerald Shen, David Mosallanezhad, Di Zhang, Jonas Yang, June Yang, Oleksii Kuchaiev, Guilin Liu, Zhiding Yu, Pavlo Molchanov, Yejin Choi, Jan Kautz, Yi Dong  

**Link**: [PDF](https://arxiv.org/pdf/2507.12507)  

**Abstract**: Recent advancements in reasoning-focused language models such as OpenAI's O1 and DeepSeek-R1 have shown that scaling test-time computation-through chain-of-thought reasoning and iterative exploration-can yield substantial improvements on complex tasks like mathematics and code generation. These breakthroughs have been driven by large-scale reinforcement learning (RL), particularly when combined with verifiable reward signals that provide objective and grounded supervision. In this report, we investigate the effects of prolonged reinforcement learning on a small language model across a diverse set of reasoning domains. Our work identifies several key ingredients for effective training, including the use of verifiable reward tasks, enhancements to Group Relative Policy Optimization (GRPO), and practical techniques to improve training stability and generalization. We introduce controlled KL regularization, clipping ratio, and periodic reference policy resets as critical components for unlocking long-term performance gains. Our model achieves significant improvements over strong baselines, including +14.7% on math, +13.9% on coding, and +54.8% on logic puzzle tasks. To facilitate continued research, we release our model publicly. 

---
# Spatially Grounded Explanations in Vision Language Models for Document Visual Question Answering 

**Authors**: Maximiliano Hormazábal Lagos, Héctor Cerezo-Costas, Dimosthenis Karatzas  

**Link**: [PDF](https://arxiv.org/pdf/2507.12490)  

**Abstract**: We introduce EaGERS, a fully training-free and model-agnostic pipeline that (1) generates natural language rationales via a vision language model, (2) grounds these rationales to spatial sub-regions by computing multimodal embedding similarities over a configurable grid with majority voting, and (3) restricts the generation of responses only from the relevant regions selected in the masked image. Experiments on the DocVQA dataset demonstrate that our best configuration not only outperforms the base model on exact match accuracy and Average Normalized Levenshtein Similarity metrics but also enhances transparency and reproducibility in DocVQA without additional model fine-tuning. 

---
# A Survey of AIOps in the Era of Large Language Models 

**Authors**: Lingzhe Zhang, Tong Jia, Mengxi Jia, Yifan Wu, Aiwei Liu, Yong Yang, Zhonghai Wu, Xuming Hu, Philip S. Yu, Ying Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.12472)  

**Abstract**: As large language models (LLMs) grow increasingly sophisticated and pervasive, their application to various Artificial Intelligence for IT Operations (AIOps) tasks has garnered significant attention. However, a comprehensive understanding of the impact, potential, and limitations of LLMs in AIOps remains in its infancy. To address this gap, we conducted a detailed survey of LLM4AIOps, focusing on how LLMs can optimize processes and improve outcomes in this domain. We analyzed 183 research papers published between January 2020 and December 2024 to answer four key research questions (RQs). In RQ1, we examine the diverse failure data sources utilized, including advanced LLM-based processing techniques for legacy data and the incorporation of new data sources enabled by LLMs. RQ2 explores the evolution of AIOps tasks, highlighting the emergence of novel tasks and the publication trends across these tasks. RQ3 investigates the various LLM-based methods applied to address AIOps challenges. Finally, RQ4 reviews evaluation methodologies tailored to assess LLM-integrated AIOps approaches. Based on our findings, we discuss the state-of-the-art advancements and trends, identify gaps in existing research, and propose promising directions for future exploration. 

---
# Perfect diffusion is $\mathsf{TC}^0$ -- Bad diffusion is Turing-complete 

**Authors**: Yuxi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.12469)  

**Abstract**: This paper explores the computational complexity of diffusion-based language modeling. We prove a dichotomy based on the quality of the score-matching network in a diffusion model. In one direction, a network that exactly computes the score function of some initial distribution can only perform language modeling within the $\mathsf{TC}^0$ complexity class, reflecting limitations tied to rapid convergence. In the other direction, we show that if there is no requirement for the network to match any score function, then diffusion modeling can simulate any Turing machine in a certain sense. This dichotomy provides a theoretical lens on the capabilities and limitations of diffusion models, particularly concerning tasks requiring sequential computation. We conjecture extensions of our theoretical results, including for the case where the diffusion model is not perfect, but merely good. We also discuss the wider context and practical implications, and hypothesize that a machine learning architecture that can interpolate between sequential and parallel modes of operation would be superior to both Transformers and diffusion models. 

---
