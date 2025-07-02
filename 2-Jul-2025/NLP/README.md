# SciArena: An Open Evaluation Platform for Foundation Models in Scientific Literature Tasks 

**Authors**: Yilun Zhao, Kaiyan Zhang, Tiansheng Hu, Sihong Wu, Ronan Le Bras, Taira Anderson, Jonathan Bragg, Joseph Chee Chang, Jesse Dodge, Matt Latzke, Yixin Liu, Charles McGrady, Xiangru Tang, Zihang Wang, Chen Zhao, Hannaneh Hajishirzi, Doug Downey, Arman Cohan  

**Link**: [PDF](https://arxiv.org/pdf/2507.01001)  

**Abstract**: We present SciArena, an open and collaborative platform for evaluating foundation models on scientific literature tasks. Unlike traditional benchmarks for scientific literature understanding and synthesis, SciArena engages the research community directly, following the Chatbot Arena evaluation approach of community voting on model comparisons. By leveraging collective intelligence, SciArena offers a community-driven evaluation of model performance on open-ended scientific tasks that demand literature-grounded, long-form responses. The platform currently supports 23 open-source and proprietary foundation models and has collected over 13,000 votes from trusted researchers across diverse scientific domains. We analyze the data collected so far and confirm that the submitted questions are diverse, aligned with real-world literature needs, and that participating researchers demonstrate strong self-consistency and inter-annotator agreement in their evaluations. We discuss the results and insights based on the model ranking leaderboard. To further promote research in building model-based automated evaluation systems for literature tasks, we release SciArena-Eval, a meta-evaluation benchmark based on our collected preference data. The benchmark measures the accuracy of models in judging answer quality by comparing their pairwise assessments with human votes. Our experiments highlight the benchmark's challenges and emphasize the need for more reliable automated evaluation methods. 

---
# La Leaderboard: A Large Language Model Leaderboard for Spanish Varieties and Languages of Spain and Latin America 

**Authors**: María Grandury, Javier Aula-Blasco, Júlia Falcão, Clémentine Fourrier, Miguel González, Gonzalo Martínez, Gonzalo Santamaría, Rodrigo Agerri, Nuria Aldama, Luis Chiruzzo, Javier Conde, Helena Gómez, Marta Guerrero, Guido Ivetta, Natalia López, Flor Miriam Plaza-del-Arco, María Teresa Martín-Valdivia, Helena Montoro, Carmen Muñoz, Pedro Reviriego, Leire Rosado, Alejandro Vaca, María Estrella Vallecillo-Rodríguez, Jorge Vallego, Irune Zubiaga  

**Link**: [PDF](https://arxiv.org/pdf/2507.00999)  

**Abstract**: Leaderboards showcase the current capabilities and limitations of Large Language Models (LLMs). To motivate the development of LLMs that represent the linguistic and cultural diversity of the Spanish-speaking community, we present La Leaderboard, the first open-source leaderboard to evaluate generative LLMs in languages and language varieties of Spain and Latin America. La Leaderboard is a community-driven project that aims to establish an evaluation standard for everyone interested in developing LLMs for the Spanish-speaking community. This initial version combines 66 datasets in Basque, Catalan, Galician, and different Spanish varieties, showcasing the evaluation results of 50 models. To encourage community-driven development of leaderboards in other languages, we explain our methodology, including guidance on selecting the most suitable evaluation setup for each downstream task. In particular, we provide a rationale for using fewer few-shot examples than typically found in the literature, aiming to reduce environmental impact and facilitate access to reproducible results for a broader research community. 

---
# Should We Still Pretrain Encoders with Masked Language Modeling? 

**Authors**: Hippolyte Gisserot-Boukhlef, Nicolas Boizard, Manuel Faysse, Duarte M. Alves, Emmanuel Malherbe, André F. T. Martins, Céline Hudelot, Pierre Colombo  

**Link**: [PDF](https://arxiv.org/pdf/2507.00994)  

**Abstract**: Learning high-quality text representations is fundamental to a wide range of NLP tasks. While encoder pretraining has traditionally relied on Masked Language Modeling (MLM), recent evidence suggests that decoder models pretrained with Causal Language Modeling (CLM) can be effectively repurposed as encoders, often surpassing traditional encoders on text representation benchmarks. However, it remains unclear whether these gains reflect an inherent advantage of the CLM objective or arise from confounding factors such as model and data scale. In this paper, we address this question through a series of large-scale, carefully controlled pretraining ablations, training a total of 30 models ranging from 210 million to 1 billion parameters, and conducting over 15,000 fine-tuning and evaluation runs. We find that while training with MLM generally yields better performance across text representation tasks, CLM-trained models are more data-efficient and demonstrate improved fine-tuning stability. Building on these findings, we experimentally show that a biphasic training strategy that sequentially applies CLM and then MLM, achieves optimal performance under a fixed computational training budget. Moreover, we demonstrate that this strategy becomes more appealing when initializing from readily available pretrained CLM models (from the existing LLM ecosystem), reducing the computational burden needed to train best-in-class encoder models. We release all project artifacts at this https URL to foster further research. 

---
# Discourse Heuristics For Paradoxically Moral Self-Correction 

**Authors**: Guangliang Liu, Zimo Qi, Xitong Zhang, Kristen Marie Johnson  

**Link**: [PDF](https://arxiv.org/pdf/2507.00985)  

**Abstract**: Moral self-correction has emerged as a promising approach for aligning the output of Large Language Models (LLMs) with human moral values. However, moral self-correction techniques are subject to two primary paradoxes. First, despite empirical and theoretical evidence to support the effectiveness of self-correction, this LLM capability only operates at a superficial level. Second, while LLMs possess the capability of self-diagnosing immoral aspects of their output, they struggle to identify the cause of this moral inconsistency during their self-correction process. To better understand and address these paradoxes, we analyze the discourse constructions in fine-tuning corpora designed to enhance moral self-correction, uncovering the existence of the heuristics underlying effective constructions. We demonstrate that moral self-correction relies on discourse constructions that reflect heuristic shortcuts, and that the presence of these heuristic shortcuts during self-correction leads to inconsistency when attempting to enhance both self-correction and self-diagnosis capabilities jointly. Based on our findings, we propose a solution to improve moral self-correction by leveraging the heuristics of curated datasets. We also highlight the generalization challenges of this capability, particularly in terms of learning from situated context and model scales. 

---
# The Cognate Data Bottleneck in Language Phylogenetics 

**Authors**: Luise Häuser, Alexandros Stamatakis  

**Link**: [PDF](https://arxiv.org/pdf/2507.00911)  

**Abstract**: To fully exploit the potential of computational phylogenetic methods for cognate data one needs to leverage specific (complex) models an machine learning-based techniques. However, both approaches require datasets that are substantially larger than the manually collected cognate data currently available. To the best of our knowledge, there exists no feasible approach to automatically generate larger cognate datasets. We substantiate this claim by automatically extracting datasets from BabelNet, a large multilingual encyclopedic dictionary. We demonstrate that phylogenetic inferences on the respective character matrices yield trees that are largely inconsistent with the established gold standard ground truth trees. We also discuss why we consider it as being unlikely to be able to extract more suitable character matrices from other multilingual resources. Phylogenetic data analysis approaches that require larger datasets can therefore not be applied to cognate data. Thus, it remains an open question how, and if these computational approaches can be applied in historical linguistics. 

---
# MemeCMD: An Automatically Generated Chinese Multi-turn Dialogue Dataset with Contextually Retrieved Memes 

**Authors**: Yuheng Wang, Xianhe Tang, Pufeng Huang  

**Link**: [PDF](https://arxiv.org/pdf/2507.00891)  

**Abstract**: Memes are widely used in online social interactions, providing vivid, intuitive, and often humorous means to express intentions and emotions. Existing dialogue datasets are predominantly limited to either manually annotated or pure-text conversations, lacking the expressiveness and contextual nuance that multimodal interactions this http URL address these challenges, we introduce MemeCMD, an automatically generated Chinese Multi-turn Dialogue dataset with contextually retrieved memes. Our dataset combines a large-scale, MLLM-annotated meme library with dialogues auto-generated by dual agents across diverse scenarios. We introduce a retrieval framework and adaptive threshold to ensure contextually relevant, naturally spaced meme usage. Experiments demonstrate the effectiveness of our approach in generating contextually appropriate and diverse meme-incorporated dialogues, offering a scalable and privacy-preserving resource for advancing multimodal conversational AI. 

---
# Scaling Laws Are Unreliable for Downstream Tasks: A Reality Check 

**Authors**: Nicholas Lourie, Michael Y. Hu, Kyunghyun Cho  

**Link**: [PDF](https://arxiv.org/pdf/2507.00885)  

**Abstract**: Downstream scaling laws aim to predict task performance at larger scales from pretraining losses at smaller scales. Whether this prediction should be possible is unclear: some works demonstrate that task performance follows clear linear scaling trends under transformation, whereas others point out fundamental challenges to downstream scaling laws, such as emergence and inverse scaling. In this work, we conduct a meta-analysis of existing data on downstream scaling laws, finding that close fit to linear scaling laws only occurs in a minority of cases: 39% of the time. Furthermore, seemingly benign changes to the experimental setting can completely change the scaling trend. Our analysis underscores the need to understand the conditions under which scaling laws succeed. To fully model the relationship between pretraining loss and downstream task performance, we must embrace the cases in which scaling behavior deviates from linear trends. 

---
# Mathematics Isn't Culture-Free: Probing Cultural Gaps via Entity and Scenario Perturbations 

**Authors**: Aditya Tomar, Nihar Ranjan Sahoo, Ashish Mittal, Rudra Murthy, Pushpak Bhattacharyya  

**Link**: [PDF](https://arxiv.org/pdf/2507.00883)  

**Abstract**: Although mathematics is often considered culturally neutral, the way mathematical problems are presented can carry implicit cultural context. Existing benchmarks like GSM8K are predominantly rooted in Western norms, including names, currencies, and everyday scenarios. In this work, we create culturally adapted variants of the GSM8K test set for five regions Africa, India, China, Korea, and Japan using prompt-based transformations followed by manual verification. We evaluate six large language models (LLMs), ranging from 8B to 72B parameters, across five prompting strategies to assess their robustness to cultural variation in math problem presentation. Our findings reveal a consistent performance gap: models perform best on the original US-centric dataset and comparatively worse on culturally adapted versions. However, models with reasoning capabilities are more resilient to these shifts, suggesting that deeper reasoning helps bridge cultural presentation gaps in mathematical tasks 

---
# TransLaw: Benchmarking Large Language Models in Multi-Agent Simulation of the Collaborative Translation 

**Authors**: Xi Xuan, King-kui Sin, Yufei Zhou, Chunyu Kit  

**Link**: [PDF](https://arxiv.org/pdf/2507.00875)  

**Abstract**: Multi-agent systems empowered by large language models (LLMs) have demonstrated remarkable capabilities in a wide range of downstream applications, including machine translation. However, the potential of LLMs in translating Hong Kong legal judgments remains uncertain due to challenges such as intricate legal terminology, culturally embedded nuances, and strict linguistic structures. In this work, we introduce TransLaw, a novel multi-agent framework implemented for real-world Hong Kong case law translation. It employs three specialized agents, namely, Translator, Annotator, and Proofreader, to collaboratively produce translations for high accuracy in legal meaning, appropriateness in style, and adequate coherence and cohesion in structure. This framework supports customizable LLM configurations and achieves tremendous cost reduction compared to professional human translation services. We evaluated its performance using 13 open-source and commercial LLMs as agents and obtained interesting findings, including that it surpasses GPT-4o in legal semantic accuracy, structural coherence, and stylistic fidelity, yet trails human experts in contextualizing complex terminology and stylistic naturalness. Our platform website is available at CityUHK, and our bilingual judgment corpus used for the evaluation is available at Hugging Face. 

---
# Stylometry recognizes human and LLM-generated texts in short samples 

**Authors**: Karol Przystalski, Jan K. Argasiński, Iwona Grabska-Gradzińska, Jeremi K. Ochab  

**Link**: [PDF](https://arxiv.org/pdf/2507.00838)  

**Abstract**: The paper explores stylometry as a method to distinguish between texts created by Large Language Models (LLMs) and humans, addressing issues of model attribution, intellectual property, and ethical AI use. Stylometry has been used extensively to characterise the style and attribute authorship of texts. By applying it to LLM-generated texts, we identify their emergent writing patterns. The paper involves creating a benchmark dataset based on Wikipedia, with (a) human-written term summaries, (b) texts generated purely by LLMs (GPT-3.5/4, LLaMa 2/3, Orca, and Falcon), (c) processed through multiple text summarisation methods (T5, BART, Gensim, and Sumy), and (d) rephrasing methods (Dipper, T5). The 10-sentence long texts were classified by tree-based models (decision trees and LightGBM) using human-designed (StyloMetrix) and n-gram-based (our own pipeline) stylometric features that encode lexical, grammatical, syntactic, and punctuation patterns. The cross-validated results reached a performance of up to .87 Matthews correlation coefficient in the multiclass scenario with 7 classes, and accuracy between .79 and 1. in binary classification, with the particular example of Wikipedia and GPT-4 reaching up to .98 accuracy on a balanced dataset. Shapley Additive Explanations pinpointed features characteristic of the encyclopaedic text type, individual overused words, as well as a greater grammatical standardisation of LLMs with respect to human-written texts. These results show -- crucially, in the context of the increasingly sophisticated LLMs -- that it is possible to distinguish machine- from human-generated texts at least for a well-defined text type. 

---
# ProxAnn: Use-Oriented Evaluations of Topic Models and Document Clustering 

**Authors**: Alexander Hoyle, Lorena Calvo-Bartolomé, Jordan Boyd-Graber, Philip Resnik  

**Link**: [PDF](https://arxiv.org/pdf/2507.00828)  

**Abstract**: Topic model and document-clustering evaluations either use automated metrics that align poorly with human preferences or require expert labels that are intractable to scale. We design a scalable human evaluation protocol and a corresponding automated approximation that reflect practitioners' real-world usage of models. Annotators -- or an LLM-based proxy -- review text items assigned to a topic or cluster, infer a category for the group, then apply that category to other documents. Using this protocol, we collect extensive crowdworker annotations of outputs from a diverse set of topic models on two datasets. We then use these annotations to validate automated proxies, finding that the best LLM proxies are statistically indistinguishable from a human annotator and can therefore serve as a reasonable substitute in automated evaluations. Package, web interface, and data are at this https URL 

---
# Many LLMs Are More Utilitarian Than One 

**Authors**: Anita Keshmirian, Razan Baltaji, Babak Hemmatian, Hadi Asghari, Lav R. Varshney  

**Link**: [PDF](https://arxiv.org/pdf/2507.00814)  

**Abstract**: Moral judgment is integral to large language model (LLM) alignment and social reasoning. As multi-agent systems gain prominence, it becomes crucial to understand how LLMs function collectively during collaboration, compared to individual agents. In human moral judgment, group deliberation leads to a utilitarian boost: a tendency to endorse norm violations that maximize benefits for the greatest number of people despite harms. We study whether a similar dynamic emerges in multi-agent LLM systems. We tested six models on well-established sets of moral dilemmas across two conditions: (1) Solo, where models reasoned independently, and (2) Group, where they engaged in multi-turn discussions in pairs or triads. In personal moral dilemmas, where agents must decide to directly harm one individual to maximize the utility for others, all models found moral violations to be more acceptable when part of a group than individually, similar to human experiments. Some models endorsed actions that maximized overall well-being, even if they benefited strangers over familiar individuals. Others became more willing to violate moral norms in groups. However, while human groups show a similar action bias, the mechanism for their utilitarian boost differs from LLMs. Whereas the human shift comes from heightened sensitivity to decision outcomes, LLM groups show either reduced norm sensitivity or enhanced impartiality. This suggests that while the surface behavior of LLM collectives mimics human group reasoning, the underlying drivers differ. We discuss the implications for AI alignment, multi-agent design, and artificial moral reasoning. 

---
# Generative AI and the future of scientometrics: current topics and future questions 

**Authors**: Benedetto Lepori, Jens Peter Andersen, Karsten Donnay  

**Link**: [PDF](https://arxiv.org/pdf/2507.00783)  

**Abstract**: The aim of this paper is to review the use of GenAI in scientometrics, and to begin a debate on the broader implications for the field. First, we provide an introduction on GenAI's generative and probabilistic nature as rooted in distributional linguistics. And we relate this to the debate on the extent to which GenAI might be able to mimic human 'reasoning'. Second, we leverage this distinction for a critical engagement with recent experiments using GenAI in scientometrics, including topic labelling, the analysis of citation contexts, predictive applications, scholars' profiling, and research assessment. GenAI shows promise in tasks where language generation dominates, such as labelling, but faces limitations in tasks that require stable semantics, pragmatic reasoning, or structured domain knowledge. However, these results might become quickly outdated. Our recommendation is, therefore, to always strive to systematically compare the performance of different GenAI models for specific tasks. Third, we inquire whether, by generating large amounts of scientific language, GenAI might have a fundamental impact on our field by affecting textual characteristics used to measure science, such as authors, words, and references. We argue that careful empirical work and theoretical reflection will be essential to remain capable of interpreting the evolving patterns of knowledge production. 

---
# A Diagrammatic Calculus for a Functional Model of Natural Language Semantics 

**Authors**: Matthieu Pierre Boyer  

**Link**: [PDF](https://arxiv.org/pdf/2507.00782)  

**Abstract**: In this paper, we study a functional programming approach to natural language semantics, allowing us to increase the expressivity of a more traditional denotation style. We will formalize a category based type and effect system, and construct a diagrammatic calculus to model parsing and handling of effects, and use it to efficiently compute the denotations for sentences. 

---
# LitBench: A Benchmark and Dataset for Reliable Evaluation of Creative Writing 

**Authors**: Daniel Fein, Sebastian Russo, Violet Xiang, Kabir Jolly, Rafael Rafailov, Nick Haber  

**Link**: [PDF](https://arxiv.org/pdf/2507.00769)  

**Abstract**: Evaluating creative writing generated by large language models (LLMs) remains challenging because open-ended narratives lack ground truths. Without performant automated evaluation methods, off-the-shelf (OTS) language models are employed as zero-shot judges, yet their reliability is unclear in this context. In pursuit of robust evaluation for creative writing, we introduce LitBench, the first standardized benchmark and paired dataset for creative writing verification, comprising a held-out test set of 2,480 debiased, human-labeled story comparisons drawn from Reddit and a 43,827-pair training corpus of human preference labels. Using LitBench, we (i) benchmark zero-shot LLM judges, (ii) train Bradley Terry and generative reward models, and (iii) conduct an online human study to validate reward model rankings on newly LLM-generated stories. Our benchmark identifies Claude-3.7-Sonnet as the strongest off-the-shelf judge, reaching 73% agreement with human preferences; among trained reward models, Bradley-Terry and Generative reward models both attain an accuracy of 78%, outperforming all off-the-shelf judges. An online human study further confirms that our trained reward models consistently align with human preferences in novel LLM-generated stories. We release LitBench and reward models at this https URL, providing a vetted resource for reliable, automated evaluation and optimization of creative writing systems. 

---
# AI Analyst: Framework and Comprehensive Evaluation of Large Language Models for Financial Time Series Report Generation 

**Authors**: Elizabeth Fons, Elena Kochkina, Rachneet Kaur, Zhen Zeng, Berowne Hlavaty, Charese Smiley, Svitlana Vyetrenko, Manuela Veloso  

**Link**: [PDF](https://arxiv.org/pdf/2507.00718)  

**Abstract**: This paper explores the potential of large language models (LLMs) to generate financial reports from time series data. We propose a framework encompassing prompt engineering, model selection, and evaluation. We introduce an automated highlighting system to categorize information within the generated reports, differentiating between insights derived directly from time series data, stemming from financial reasoning, and those reliant on external knowledge. This approach aids in evaluating the factual grounding and reasoning capabilities of the models. Our experiments, utilizing both data from the real stock market indices and synthetic time series, demonstrate the capability of LLMs to produce coherent and informative financial reports. 

---
# Contrasting Cognitive Styles in Vision-Language Models: Holistic Attention in Japanese Versus Analytical Focus in English 

**Authors**: Ahmed Sabir, Azinovič Gasper, Mengsay Loem, Rajesh Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2507.00700)  

**Abstract**: Cross-cultural research in perception and cognition has shown that individuals from different cultural backgrounds process visual information in distinct ways. East Asians, for example, tend to adopt a holistic perspective, attending to contextual relationships, whereas Westerners often employ an analytical approach, focusing on individual objects and their attributes. In this study, we investigate whether Vision-Language Models (VLMs) trained predominantly on different languages, specifically Japanese and English, exhibit similar culturally grounded attentional patterns. Using comparative analysis of image descriptions, we examine whether these models reflect differences in holistic versus analytic tendencies. Our findings suggest that VLMs not only internalize the structural properties of language but also reproduce cultural behaviors embedded in the training data, indicating that cultural cognition may implicitly shape model outputs. 

---
# SAFER: Probing Safety in Reward Models with Sparse Autoencoder 

**Authors**: Sihang Li, Wei Shi, Ziyuan Xie, Tao Liang, Guojun Ma, Xiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.00665)  

**Abstract**: Reinforcement learning from human feedback (RLHF) is a key paradigm for aligning large language models (LLMs) with human values, yet the reward models at its core remain largely opaque. In this work, we present sparse Autoencoder For Enhanced Reward model (\textbf{SAFER}), a novel framework for interpreting and improving reward models through mechanistic analysis. Leveraging Sparse Autoencoders (SAEs), we uncover human-interpretable features in reward model activations, enabling insight into safety-relevant decision-making. We apply SAFER to safety-oriented preference datasets and quantify the salience of individual features by activation differences between chosen and rejected responses. Using these feature-level signals, we design targeted data poisoning and denoising strategies. Experiments show that SAFER can precisely degrade or enhance safety alignment with minimal data modification, without sacrificing general chat performance. Our approach contributes to interpreting, auditing and refining reward models in high-stakes LLM alignment tasks. Our codes are available at this https URL. \textit{This paper discusses topics related to large language model safety and may include discussions or examples that highlight potential risks or unsafe outcomes.} 

---
# Mixture of Reasonings: Teach Large Language Models to Reason with Adaptive Strategies 

**Authors**: Tao Xiong, Xavier Hu, Wenyan Fan, Shengyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.00606)  

**Abstract**: Large language models (LLMs) excel in complex tasks through advanced prompting techniques like Chain-of-Thought (CoT) and Tree-of-Thought (ToT), but their reliance on manually crafted, task-specific prompts limits adaptability and efficiency. We introduce Mixture of Reasoning (MoR), a training framework that embeds diverse reasoning strategies into LLMs for autonomous, task-adaptive reasoning without external prompt engineering. MoR has two phases: Thought Generation, creating reasoning chain templates with models like GPT-4o, and SFT Dataset Construction, pairing templates with benchmark datasets for supervised this http URL experiments show that MoR significantly enhances performance, with MoR150 achieving 0.730 (2.2% improvement) using CoT prompting and 0.734 (13.5% improvement) compared to baselines. MoR eliminates the need for task-specific prompts, offering a generalizable solution for robust reasoning across diverse tasks. 

---
# Transferable Modeling Strategies for Low-Resource LLM Tasks: A Prompt and Alignment-Based 

**Authors**: Shuangquan Lyu, Yingnan Deng, Guiran Liu, Zhen Qi, Ruotong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.00601)  

**Abstract**: This paper addresses the limited transfer and adaptation capabilities of large language models in low-resource language scenarios. It proposes a unified framework that combines a knowledge transfer module with parameter-efficient fine-tuning strategies. The method introduces knowledge alignment loss and soft prompt tuning to guide the model in effectively absorbing the structural features of target languages or tasks under minimal annotation. This enhances both generalization performance and training stability. The framework includes lightweight adaptation modules to reduce computational costs. During training, it integrates freezing strategies and prompt injection to preserve the model's original knowledge while enabling quick adaptation to new tasks. The study also conducts stability analysis experiments and synthetic pseudo-data transfer experiments to systematically evaluate the method's applicability and robustness across different low-resource tasks. Experimental results show that compared with existing multilingual pre-trained models and mainstream transfer methods, the proposed approach achieves higher performance and stability on cross-lingual tasks such as MLQA, XQuAD, and PAWS-X. It demonstrates particularly strong advantages under extremely data-scarce conditions. The proposed method offers strong generality and scalability. It enhances task-specific adaptability while preserving the general capabilities of large language models. This makes it well-suited for complex semantic modeling and multilingual processing tasks. 

---
# TUM-MiKaNi at SemEval-2025 Task 3: Towards Multilingual and Knowledge-Aware Non-factual Hallucination Identification 

**Authors**: Miriam Anschütz, Ekaterina Gikalo, Niklas Herbster, Georg Groh  

**Link**: [PDF](https://arxiv.org/pdf/2507.00579)  

**Abstract**: Hallucinations are one of the major problems of LLMs, hindering their trustworthiness and deployment to wider use cases. However, most of the research on hallucinations focuses on English data, neglecting the multilingual nature of LLMs. This paper describes our submission to the SemEval-2025 Task-3 - Mu-SHROOM, the Multilingual Shared-task on Hallucinations and Related Observable Overgeneration Mistakes. We propose a two-part pipeline that combines retrieval-based fact verification against Wikipedia with a BERT-based system fine-tuned to identify common hallucination patterns. Our system achieves competitive results across all languages, reaching top-10 results in eight languages, including English. Moreover, it supports multiple languages beyond the fourteen covered by the shared task. This multilingual hallucination identifier can help to improve LLM outputs and their usefulness in the future. 

---
# Methodological Rigour in Algorithm Application: An Illustration of Topic Modelling Algorithm 

**Authors**: Malmi Amadoru  

**Link**: [PDF](https://arxiv.org/pdf/2507.00547)  

**Abstract**: The rise of advanced computational algorithms has opened new avenues for computationally intensive research approaches to theory development. However, the opacity of these algorithms and lack of transparency and rigour in their application pose methodological challenges, potentially undermining trust in research. The discourse on methodological rigour in this new genre of research is still emerging. Against this backdrop, I attempt to offer guidance on methodological rigour, particularly in the context of topic modelling algorithms. By illustrating the application of the structural topic modelling algorithm and presenting a set of guidelines, I discuss how to ensure rigour in topic modelling studies. Although the guidelines are for the application of topic modelling algorithms, they can be applied to other algorithms with context-specific adjustments. The guidelines are helpful, especially for novice researchers applying topic modelling, and editors and reviewers handling topic modelling manuscripts. I contribute to the literature on topic modelling and join the emerging dialogue on methodological rigour in computationally intensive theory construction research. 

---
# Capsule Network-Based Semantic Intent Modeling for Human-Computer Interaction 

**Authors**: Shixiao Wang, Yifan Zhuang, Runsheng Zhang, Zhijun Song  

**Link**: [PDF](https://arxiv.org/pdf/2507.00540)  

**Abstract**: This paper proposes a user semantic intent modeling algorithm based on Capsule Networks to address the problem of insufficient accuracy in intent recognition for human-computer interaction. The method represents semantic features in input text through a vectorized capsule structure. It uses a dynamic routing mechanism to transfer information across multiple capsule layers. This helps capture hierarchical relationships and part-whole structures between semantic entities more effectively. The model uses a convolutional feature extraction module as the low-level encoder. After generating initial semantic capsules, it forms high-level abstract intent representations through an iterative routing process. To further enhance performance, a margin-based mechanism is introduced into the loss function. This improves the model's ability to distinguish between intent classes. Experiments are conducted using a public natural language understanding dataset. Multiple mainstream models are used for comparison. Results show that the proposed model outperforms traditional methods and other deep learning structures in terms of accuracy, F1-score, and intent detection rate. The study also analyzes the effect of the number of dynamic routing iterations on model performance. A convergence curve of the loss function during training is provided. These results verify the stability and effectiveness of the proposed method in semantic modeling. Overall, this study presents a new structured modeling approach to improve intent recognition under complex semantic conditions. 

---
# NIRANTAR: Continual Learning with New Languages and Domains on Real-world Speech Data 

**Authors**: Tahir Javed, Kaushal Bhogale, Mitesh M. Khapra  

**Link**: [PDF](https://arxiv.org/pdf/2507.00534)  

**Abstract**: We introduce Nirantar, a comprehensive framework for evaluating continual learning (CL) in multilingual and multi-domain ASR. Designed to reflect real-world CL challenges, Nirantar leverages data collected incrementally across 22 languages and 208 districts in India through natural episodes. This enables evaluation across Language-Incremental (LIL), Domain-Incremental (DIL), and the novel Language-Incremental Domain-Incremental Learning (LIDIL) scenarios. Unlike prior work that relies on simulated episodes, Nirantar presents dynamic, non-uniform language and domain shifts, making it an ideal testbed for CL research. With 3250 hours of human-transcribed speech, including 1720 hours newly introduced in this work, our framework enables systematic benchmarking of CL methods. We evaluate existing approaches and demonstrate that no single method performs consistently well, underscoring the need for more robust CL strategies. 

---
# TeamCMU at Touché: Adversarial Co-Evolution for Advertisement Integration and Detection in Conversational Search 

**Authors**: To Eun Kim, João Coelho, Gbemileke Onilude, Jai Singh  

**Link**: [PDF](https://arxiv.org/pdf/2507.00509)  

**Abstract**: As conversational search engines increasingly adopt generation-based paradigms powered by Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG), the integration of advertisements into generated responses presents both commercial opportunities and challenges for user experience. Unlike traditional search, where advertisements are clearly delineated, generative systems blur the boundary between informational content and promotional material, raising concerns around transparency and trust. In this work, we propose a modular pipeline for advertisement management in RAG-based conversational systems, consisting of an ad-rewriter for seamless ad integration and a robust ad-classifier for detection. We leverage synthetic data to train high-performing classifiers, which are then used to guide two complementary ad-integration strategies: supervised fine-tuning of the ad-rewriter and a best-of-N sampling approach that selects the least detectable ad-integrated response among multiple candidates. Our evaluation focuses on two core questions: the effectiveness of ad classifiers in detecting diverse ad integration strategies, and the training methods that best support coherent, minimally intrusive ad insertion. Experimental results show that our ad-classifier, trained on synthetic advertisement data inspired by marketing strategies and enhanced through curriculum learning, achieves robust detection performance. Additionally, we demonstrate that classifier-guided optimization, through both fine-tuning and best-of-N sampling, significantly improves ad stealth, enabling more seamless integration. These findings contribute an adversarial co-evolution framework for developing more sophisticated ad-aware generative search systems and robust ad classifiers. 

---
# Pitfalls of Evaluating Language Models with Open Benchmarks 

**Authors**: Md. Najib Hasan, Mohammad Fakhruddin Babar, Souvika Sarkar, Monowar Hasan, Santu Karmaker  

**Link**: [PDF](https://arxiv.org/pdf/2507.00460)  

**Abstract**: Open Large Language Model (LLM) benchmarks, such as HELM and BIG-bench, offer standardized, transparent protocols that facilitate the fair comparison, reproducibility, and iterative advancement of Language Models (LMs). However, their openness also introduces critical and underexplored pitfalls. This study exposes these weaknesses by systematically constructing ``cheating'' models -- smaller variants of BART, T5, and GPT-2 fine-tuned directly on public test sets -- which achieve top rankings on a prominent open, holistic benchmark (HELM) despite poor generalization and limited practical utility. Our findings underscore three key insights: \ca high leaderboard performance on open benchmarks may not always reflect real-world effectiveness; \cb private or dynamic benchmarks must complement open evaluations to safeguard integrity; and \cc a fundamental reevaluation of current benchmarking practices is essential to ensure robust and trustworthy LM assessments. 

---
# Beyond Sociodemographic Prompting: Using Supervision to Align LLMs with Human Response Distributions 

**Authors**: Gauri Kambhatla, Sanjana Gautam, Angela Zhang, Alex Liu, Ravi Srinivasan, Junyi Jessy Li, Matthew Lease  

**Link**: [PDF](https://arxiv.org/pdf/2507.00439)  

**Abstract**: The ability to accurately predict how different population groups would answer subjective questions would have great value. In this work, we show that use of relatively simple supervision can greatly improve language model alignment with diverse population groups, as measured over three datasets spanning various topics. Beyond evaluating average performance, we also report how alignment varies across specific groups. The simplicity and generality of our approach promotes easy adoption, while our broad findings provide useful guidance for when to use or not use our approach in practice. By conducting evaluation over many LLMs and prompting strategies, along with open-sourcing our work, we provide a useful benchmark to stimulate future research. 

---
# Causal Prompting for Implicit Sentiment Analysis with Large Language Models 

**Authors**: Jing Ren, Wenhao Zhou, Bowen Li, Mujie Liu, Nguyen Linh Dan Le, Jiade Cen, Liping Chen, Ziqi Xu, Xiwei Xu, Xiaodong Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.00389)  

**Abstract**: Implicit Sentiment Analysis (ISA) aims to infer sentiment that is implied rather than explicitly stated, requiring models to perform deeper reasoning over subtle contextual cues. While recent prompting-based methods using Large Language Models (LLMs) have shown promise in ISA, they often rely on majority voting over chain-of-thought (CoT) reasoning paths without evaluating their causal validity, making them susceptible to internal biases and spurious correlations. To address this challenge, we propose CAPITAL, a causal prompting framework that incorporates front-door adjustment into CoT reasoning. CAPITAL decomposes the overall causal effect into two components: the influence of the input prompt on the reasoning chains, and the impact of those chains on the final output. These components are estimated using encoder-based clustering and the NWGM approximation, with a contrastive learning objective used to better align the encoder's representation with the LLM's reasoning space. Experiments on benchmark ISA datasets with three LLMs demonstrate that CAPITAL consistently outperforms strong prompting baselines in both accuracy and robustness, particularly under adversarial conditions. This work offers a principled approach to integrating causal inference into LLM prompting and highlights its benefits for bias-aware sentiment reasoning. The source code and case study are available at: this https URL. 

---
# Gregorian melody, modality, and memory: Segmenting chant with Bayesian nonparametrics 

**Authors**: Vojtěch Lanz, Jan Hajič jr  

**Link**: [PDF](https://arxiv.org/pdf/2507.00380)  

**Abstract**: The idea that Gregorian melodies are constructed from some vocabulary of segments has long been a part of chant scholarship. This so-called "centonisation" theory has received much musicological criticism, but frequent re-use of certain melodic segments has been observed in chant melodies, and the intractable number of possible segmentations allowed the option that some undiscovered segmentation exists that will yet prove the value of centonisation, and recent empirical results have shown that segmentations can outperform music-theoretical features in mode classification. Inspired by the fact that Gregorian chant was memorised, we search for an optimal unsupervised segmentation of chant melody using nested hierarchical Pitman-Yor language models. The segmentation we find achieves state-of-the-art performance in mode classification. Modeling a monk memorising the melodies from one liturgical manuscript, we then find empirical evidence for the link between mode classification and memory efficiency, and observe more formulaic areas at the beginnings and ends of melodies corresponding to the practical role of modality in performance. However, the resulting segmentations themselves indicate that even such a memory-optimal segmentation is not what is understood as centonisation. 

---
# Question Decomposition for Retrieval-Augmented Generation 

**Authors**: Paul J. L. Ammann, Jonas Golde, Alan Akbik  

**Link**: [PDF](https://arxiv.org/pdf/2507.00355)  

**Abstract**: Grounding large language models (LLMs) in verifiable external sources is a well-established strategy for generating reliable answers. Retrieval-augmented generation (RAG) is one such approach, particularly effective for tasks like question answering: it retrieves passages that are semantically related to the question and then conditions the model on this evidence. However, multi-hop questions, such as "Which company among NVIDIA, Apple, and Google made the biggest profit in 2023?," challenge RAG because relevant facts are often distributed across multiple documents rather than co-occurring in one source, making it difficult for standard RAG to retrieve sufficient information. To address this, we propose a RAG pipeline that incorporates question decomposition: (i) an LLM decomposes the original query into sub-questions, (ii) passages are retrieved for each sub-question, and (iii) the merged candidate pool is reranked to improve the coverage and precision of the retrieved evidence. We show that question decomposition effectively assembles complementary documents, while reranking reduces noise and promotes the most relevant passages before answer generation. Although reranking itself is standard, we show that pairing an off-the-shelf cross-encoder reranker with LLM-driven question decomposition bridges the retrieval gap on multi-hop questions and provides a practical, drop-in enhancement, without any extra training or specialized indexing. We evaluate our approach on the MultiHop-RAG and HotpotQA, showing gains in retrieval (MRR@10: +36.7%) and answer accuracy (F1: +11.6%) over standard RAG baselines. 

---
# Modeling Data Diversity for Joint Instance and Verbalizer Selection in Cold-Start Scenarios 

**Authors**: Mohna Chakraborty, Adithya Kulkarni, Qi Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.00330)  

**Abstract**: Prompt-based methods leverage the knowledge of pre-trained language models (PLMs) trained with a masked language modeling (MLM) objective; however, these methods are sensitive to template, verbalizer, and few-shot instance selection, particularly in cold-start settings with no labeled data. Existing studies overlook the dependency between instances and verbalizers, where instance-label probabilities depend on verbalizer token proximity in the embedding space. To address this, we propose COLDSELECT, a joint verbalizer and instance selection approach that models data diversity. COLDSELECT maps PLM vocabulary and $h_{[MASK]}$ embeddings into a shared space, applying dimensionality reduction and clustering to ensure efficient and diverse selection. By optimizing for minimal uncertainty and maximal diversity, COLDSELECT captures data relationships effectively. Experiments on eight benchmarks demonstrate COLDSELECT's superiority in reducing uncertainty and enhancing generalization, outperforming baselines in verbalizer and few-shot instance selection for cold-start scenarios. 

---
# Failure by Interference: Language Models Make Balanced Parentheses Errors When Faulty Mechanisms Overshadow Sound Ones 

**Authors**: Daking Rai, Samuel Miller, Kevin Moran, Ziyu Yao  

**Link**: [PDF](https://arxiv.org/pdf/2507.00322)  

**Abstract**: Despite remarkable advances in coding capabilities, language models (LMs) still struggle with simple syntactic tasks such as generating balanced parentheses. In this study, we investigate the underlying mechanisms behind the persistence of these errors across LMs of varying sizes (124M-7B) to both understand and mitigate the errors. Our study reveals that LMs rely on a number of components (attention heads and FF neurons) that independently make their own predictions. While some components reliably promote correct answers across a generalized range of inputs (i.e., implementing "sound mechanisms''), others are less reliable and introduce noise by promoting incorrect tokens (i.e., implementing "faulty mechanisms''). Errors occur when the faulty mechanisms overshadow the sound ones and dominantly affect the predictions. Motivated by this insight, we introduce RASteer, a steering method to systematically identify and increase the contribution of reliable components for improving model performance. RASteer substantially improves performance on balanced parentheses tasks, boosting accuracy of some models from $0$% to around $100$% without impairing the models' general coding ability. We further demonstrate its broader applicability in arithmetic reasoning tasks, achieving performance gains of up to around $20$%. 

---
# Natural language processing for African languages 

**Authors**: David Ifeoluwa Adelani  

**Link**: [PDF](https://arxiv.org/pdf/2507.00297)  

**Abstract**: Recent advances in word embeddings and language models use large-scale, unlabelled data and self-supervised learning to boost NLP performance. Multilingual models, often trained on web-sourced data like Wikipedia, face challenges: few low-resource languages are included, their data is often noisy, and lack of labeled datasets makes it hard to evaluate performance outside high-resource languages like English. In this dissertation, we focus on languages spoken in Sub-Saharan Africa where all the indigenous languages in this region can be regarded as low-resourced in terms of the availability of labelled data for NLP tasks and unlabelled data found on the web. We analyse the noise in the publicly available corpora, and curate a high-quality corpus, demonstrating that the quality of semantic representations learned in word embeddings does not only depend on the amount of data but on the quality of pre-training data. We demonstrate empirically the limitations of word embeddings, and the opportunities the multilingual pre-trained language model (PLM) offers especially for languages unseen during pre-training and low-resource scenarios. We further study how to adapt and specialize multilingual PLMs to unseen African languages using a small amount of monolingual texts. To address the under-representation of the African languages in NLP research, we developed large scale human-annotated labelled datasets for 21 African languages in two impactful NLP tasks: named entity recognition and machine translation. We conduct an extensive empirical evaluation using state-of-the-art methods across supervised, weakly-supervised, and transfer learning settings. 

---
# Impact of Fine-Tuning Methods on Memorization in Large Language Models 

**Authors**: Jie Hou, Chuxiong Wu, Lannan Luo, Qiang Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2507.00258)  

**Abstract**: As the capabilities of pre-trained large language models (LLMs) continue to advance, the "pre-train and fine-tune" paradigm has become increasingly mainstream, leading to the development of various fine-tuning methods. However, the privacy risks arising from memorization during fine-tuning have received relatively little attention. To address this gap, we categorize popular fine-tuning approaches and assess their impact on memorization through the lens of membership inference attacks (MIAs). Our results show that, compared to parameter-based fine-tuning, prompt-based fine-tuning achieves competitive performance while exhibiting lower vulnerability to MIAs. Furthermore, prompt-based methods maintain low memorization regardless of model scale. These findings suggest that parameter-based fine-tuning is more prone to leaking private information, whereas prompt-based fine-tuning serves as a more privacy-preserving option. 

---
# EfficientXLang: Towards Improving Token Efficiency Through Cross-Lingual Reasoning 

**Authors**: Sanchit Ahuja, Praneetha Vaddamanu, Barun Patra  

**Link**: [PDF](https://arxiv.org/pdf/2507.00246)  

**Abstract**: Despite recent advances in Language Reasoning Models (LRMs), most research focuses solely on English, even though many models are pretrained on multilingual data. In this work, we investigate: Is English the most token-efficient language for reasoning? We evaluate three open-source RLMs: DeepSeek R1, Qwen 2.5 and Qwen 3, across four math datasets and seven typologically diverse languages. We find that reasoning in non-English languages not only reduces token usage, but also preserves accuracy. These gains persist even after translating the reasoning traces into English, suggesting genuine shifts in reasoning behavior rather than surface-level linguistic effects. The extent of improvement, however, depends on the models multilingual strength. Our findings motivate a broader view of reasoning in language models, highlighting the potential of multilingual reasoning and the importance of strong multilingual foundations. The code for our work can be found: this https URL. 

---
# The Algebraic Structure of Morphosyntax 

**Authors**: Isabella Senturia, Matilde Marcolli  

**Link**: [PDF](https://arxiv.org/pdf/2507.00244)  

**Abstract**: Within the context of the mathematical formulation of Merge and the Strong Minimalist Thesis, we present a mathematical model of the morphology-syntax interface. In this setting, morphology has compositional properties responsible for word formation, organized into a magma of morphological trees. However, unlike syntax, we do not have movement within morphology. A coproduct decomposition exists, but it requires extending the set of morphological trees beyond those which are generated solely by the magma, to a larger set of possible morphological inputs to syntactic trees. These participate in the formation of morphosyntactic trees as an algebra over an operad, and a correspondence between algebras over an operad. The process of structure formation for morphosyntactic trees can then be described in terms of this operadic correspondence that pairs syntactic and morphological data and the morphology coproduct. We reinterpret in this setting certain operations of Distributed Morphology as transformation that allow for flexibility in moving the boundary between syntax and morphology within the morphosyntactic objects. 

---
# Linearly Decoding Refused Knowledge in Aligned Language Models 

**Authors**: Aryan Shrivastava, Ari Holtzman  

**Link**: [PDF](https://arxiv.org/pdf/2507.00239)  

**Abstract**: Most commonly used language models (LMs) are instruction-tuned and aligned using a combination of fine-tuning and reinforcement learning, causing them to refuse users requests deemed harmful by the model. However, jailbreak prompts can often bypass these refusal mechanisms and elicit harmful responses. In this work, we study the extent to which information accessed via jailbreak prompts is decodable using linear probes trained on LM hidden states. We show that a great deal of initially refused information is linearly decodable. For example, across models, the response of a jailbroken LM for the average IQ of a country can be predicted by a linear probe with Pearson correlations exceeding $0.8$. Surprisingly, we find that probes trained on base models (which do not refuse) sometimes transfer to their instruction-tuned versions and are capable of revealing information that jailbreaks decode generatively, suggesting that the internal representations of many refused properties persist from base LMs through instruction-tuning. Importantly, we show that this information is not merely "leftover" in instruction-tuned models, but is actively used by them: we find that probe-predicted values correlate with LM generated pairwise comparisons, indicating that the information decoded by our probes align with suppressed generative behavior that may be expressed more subtly in other downstream tasks. Overall, our results suggest that instruction-tuning does not wholly eliminate or even relocate harmful information in representation space-they merely suppress its direct expression, leaving it both linearly accessible and indirectly influential in downstream behavior. 

---
# Towards Style Alignment in Cross-Cultural Translation 

**Authors**: Shreya Havaldar, Adam Stein, Eric Wong, Lyle Ungar  

**Link**: [PDF](https://arxiv.org/pdf/2507.00216)  

**Abstract**: Successful communication depends on the speaker's intended style (i.e., what the speaker is trying to convey) aligning with the listener's interpreted style (i.e., what the listener perceives). However, cultural differences often lead to misalignment between the two; for example, politeness is often lost in translation. We characterize the ways that LLMs fail to translate style - biasing translations towards neutrality and performing worse in non-Western languages. We mitigate these failures with RASTA (Retrieval-Augmented STylistic Alignment), a method that leverages learned stylistic concepts to encourage LLM translation to appropriately convey cultural communication norms and align style. 

---
# Two-Stage Reasoning-Infused Learning: Improving Classification with LLM-Generated Reasoning 

**Authors**: Mads Henrichsen, Rasmus Krebs  

**Link**: [PDF](https://arxiv.org/pdf/2507.00214)  

**Abstract**: Standard classification models often map inputs directly to labels without explicit reasoning, potentially limiting their performance, robustness, and interpretability. This paper introduces a novel two-stage approach to enhance text classification by leveraging Large Language Model (LLM)-generated reasonings. In the first stage, we fine-tune a Llama-3.2-1B-Instruct model (henceforth Llama-R-Gen) on a general-purpose reasoning dataset (syvai/reasoning-gen) to generate textual reasoning (R) given a question and its answer. In the second stage, this generally trained Llama-R-Gen is used offline to create an augmented training dataset for a downstream generative model. This downstream model, based on Llama-3.2-1B-Instruct, takes only the input text (Q) and is trained to output the generated reasoning (R) immediately followed by the predicted emotion (A). We demonstrate this methodology on the dair-ai/emotion dataset for emotion classification. Our experiments show that the generative model trained to output reasoning and the emotion (Classifier Q->RA) achieves a significant improvement of 8.7 percentage points in accuracy (for emotion prediction) compared to a baseline generative model trained solely to output the emotion (Classifier Q->A), highlighting the strong generalization capabilities of the reasoning generation and the benefit of explicit reasoning training. This work underscores the potential of LLM-generated reasonings for creating richer training datasets, thereby improving the performance of diverse downstream NLP tasks and providing explicit explanations. 

---
# LineRetriever: Planning-Aware Observation Reduction for Web Agents 

**Authors**: Imene Kerboua, Sahar Omidi Shayegan, Megh Thakkar, Xing Han Lù, Massimo Caccia, Véronique Eglin, Alexandre Aussem, Jérémy Espinas, Alexandre Lacoste  

**Link**: [PDF](https://arxiv.org/pdf/2507.00210)  

**Abstract**: While large language models have demonstrated impressive capabilities in web navigation tasks, the extensive context of web pages, often represented as DOM or Accessibility Tree (AxTree) structures, frequently exceeds model context limits. Current approaches like bottom-up truncation or embedding-based retrieval lose critical information about page state and action history. This is particularly problematic for adaptive planning in web agents, where understanding the current state is essential for determining future actions. We hypothesize that embedding models lack sufficient capacity to capture plan-relevant information, especially when retrieving content that supports future action prediction. This raises a fundamental question: how can retrieval methods be optimized for adaptive planning in web navigation tasks? In response, we introduce \textit{LineRetriever}, a novel approach that leverages a language model to identify and retrieve observation lines most relevant to future navigation steps. Unlike traditional retrieval methods that focus solely on semantic similarity, \textit{LineRetriever} explicitly considers the planning horizon, prioritizing elements that contribute to action prediction. Our experiments demonstrate that \textit{LineRetriever} can reduce the size of the observation at each step for the web agent while maintaining consistent performance within the context limitations. 

---
# Prompting as Scientific Inquiry 

**Authors**: Ari Holtzman, Chenhao Tan  

**Link**: [PDF](https://arxiv.org/pdf/2507.00163)  

**Abstract**: Prompting is the primary method by which we study and control large language models. It is also one of the most powerful: nearly every major capability attributed to LLMs-few-shot learning, chain-of-thought, constitutional AI-was first unlocked through prompting. Yet prompting is rarely treated as science and is frequently frowned upon as alchemy. We argue that this is a category error. If we treat LLMs as a new kind of complex and opaque organism that is trained rather than programmed, then prompting is not a workaround: it is behavioral science. Mechanistic interpretability peers into the neural substrate, prompting probes the model in its native interface: language. We contend that prompting is not inferior, but rather a key component in the science of LLMs. 

---
# Table Understanding and (Multimodal) LLMs: A Cross-Domain Case Study on Scientific vs. Non-Scientific Data 

**Authors**: Ekaterina Borisova, Fabio Barth, Nils Feldhus, Raia Abu Ahmad, Malte Ostendorff, Pedro Ortiz Suarez, Georg Rehm, Sebastian Möller  

**Link**: [PDF](https://arxiv.org/pdf/2507.00152)  

**Abstract**: Tables are among the most widely used tools for representing structured data in research, business, medicine, and education. Although LLMs demonstrate strong performance in downstream tasks, their efficiency in processing tabular data remains underexplored. In this paper, we investigate the effectiveness of both text-based and multimodal LLMs on table understanding tasks through a cross-domain and cross-modality evaluation. Specifically, we compare their performance on tables from scientific vs. non-scientific contexts and examine their robustness on tables represented as images vs. text. Additionally, we conduct an interpretability analysis to measure context usage and input relevance. We also introduce the TableEval benchmark, comprising 3017 tables from scholarly publications, Wikipedia, and financial reports, where each table is provided in five different formats: Image, Dictionary, HTML, XML, and LaTeX. Our findings indicate that while LLMs maintain robustness across table modalities, they face significant challenges when processing scientific tables. 

---
# Enhancing LLM Agent Safety via Causal Influence Prompting 

**Authors**: Dongyoon Hahm, Woogyeol Jin, June Suk Choi, Sungsoo Ahn, Kimin Lee  

**Link**: [PDF](https://arxiv.org/pdf/2507.00979)  

**Abstract**: As autonomous agents powered by large language models (LLMs) continue to demonstrate potential across various assistive tasks, ensuring their safe and reliable behavior is crucial for preventing unintended consequences. In this work, we introduce CIP, a novel technique that leverages causal influence diagrams (CIDs) to identify and mitigate risks arising from agent decision-making. CIDs provide a structured representation of cause-and-effect relationships, enabling agents to anticipate harmful outcomes and make safer decisions. Our approach consists of three key steps: (1) initializing a CID based on task specifications to outline the decision-making process, (2) guiding agent interactions with the environment using the CID, and (3) iteratively refining the CID based on observed behaviors and outcomes. Experimental results demonstrate that our method effectively enhances safety in both code execution and mobile device control tasks. 

---
# ONLY: One-Layer Intervention Sufficiently Mitigates Hallucinations in Large Vision-Language Models 

**Authors**: Zifu Wan, Ce Zhang, Silong Yong, Martin Q. Ma, Simon Stepputtis, Louis-Philippe Morency, Deva Ramanan, Katia Sycara, Yaqi Xie  

**Link**: [PDF](https://arxiv.org/pdf/2507.00898)  

**Abstract**: Recent Large Vision-Language Models (LVLMs) have introduced a new paradigm for understanding and reasoning about image input through textual responses. Although they have achieved remarkable performance across a range of multi-modal tasks, they face the persistent challenge of hallucination, which introduces practical weaknesses and raises concerns about their reliable deployment in real-world applications. Existing work has explored contrastive decoding approaches to mitigate this issue, where the output of the original LVLM is compared and contrasted with that of a perturbed version. However, these methods require two or more queries that slow down LVLM response generation, making them less suitable for real-time applications. To overcome this limitation, we propose ONLY, a training-free decoding approach that requires only a single query and a one-layer intervention during decoding, enabling efficient real-time deployment. Specifically, we enhance textual outputs by selectively amplifying crucial textual information using a text-to-visual entropy ratio for each token. Extensive experimental results demonstrate that our proposed ONLY consistently outperforms state-of-the-art methods across various benchmarks while requiring minimal implementation effort and computational cost. Code is available at this https URL. 

---
# Verifiable Natural Language to Linear Temporal Logic Translation: A Benchmark Dataset and Evaluation Suite 

**Authors**: William H English, Chase Walker, Dominic Simon, Sumit Kumar Jha, Rickard Ewetz  

**Link**: [PDF](https://arxiv.org/pdf/2507.00877)  

**Abstract**: Empirical evaluation of state-of-the-art natural-language (NL) to temporal-logic (TL) translation systems reveals near-perfect performance on existing benchmarks. However, current studies measure only the accuracy of the translation of NL logic into formal TL, ignoring a system's capacity to ground atomic propositions into new scenarios or environments. This is a critical feature, necessary for the verification of resulting formulas in a concrete state space. Consequently, most NL-to-TL translation frameworks propose their own bespoke dataset in which the correct grounding is known a-priori, inflating performance metrics and neglecting the need for extensible, domain-general systems. In this paper, we introduce the Verifiable Linear Temporal Logic Benchmark ( VLTL-Bench), a unifying benchmark that measures verification and verifiability of automated NL-to-LTL translation. The dataset consists of three unique state spaces and thousands of diverse natural language specifications and corresponding formal specifications in temporal logic. Moreover, the benchmark contains sample traces to validate the temporal logic expressions. While the benchmark directly supports end-to-end evaluation, we observe that many frameworks decompose the process into i) lifting, ii) grounding, iii) translation, and iv) verification. The benchmark provides ground truths after each of these steps to enable researches to improve and evaluate different substeps of the overall problem. To encourage methodologically sound advances in verifiable NL-to-LTL translation approaches, we release VLTL-Bench here: this https URL bench. 

---
# Multi-interaction TTS toward professional recording reproduction 

**Authors**: Hiroki Kanagawa, Kenichi Fujita, Aya Watanabe, Yusuke Ijima  

**Link**: [PDF](https://arxiv.org/pdf/2507.00808)  

**Abstract**: Voice directors often iteratively refine voice actors' performances by providing feedback to achieve the desired outcome. While this iterative feedback-based refinement process is important in actual recordings, it has been overlooked in text-to-speech synthesis (TTS). As a result, fine-grained style refinement after the initial synthesis is not possible, even though the synthesized speech often deviates from the user's intended style. To address this issue, we propose a TTS method with multi-step interaction that allows users to intuitively and rapidly refine synthetized speech. Our approach models the interaction between the TTS model and its user to emulate the relationship between voice actors and voice directors. Experiments show that the proposed model with its corresponding dataset enable iterative style refinements in accordance with users' directions, thus demonstrating its multi-interaction capability. Sample audios are available: https://ntt-hilab-gensp. this http URL 

---
# Safe Low Bandwidth SPV: A Formal Treatment of Simplified Payment Verification Protocols and Security Bounds 

**Authors**: Craig S Wright  

**Link**: [PDF](https://arxiv.org/pdf/2507.00740)  

**Abstract**: This paper presents a complete formal specification, protocol description, and mathematical proof structure for Simplified Payment Verification (SPV) as originally defined in the Bitcoin whitepaper \cite{nakamoto2008}. In stark contrast to the misrepresentations proliferated by popular implementations, we show that SPV is not only secure under bounded adversarial assumptions but strictly optimal for digital cash systems requiring scalable and verifiable transaction inclusion. We reconstruct the SPV protocol from first principles, grounding its verification model in symbolic automata, Merkle membership relations, and chain-of-proof dominance predicates. Through rigorous probabilistic and game-theoretic analysis, we derive the economic bounds within which the protocol operates securely and verify its liveness and safety properties under partial connectivity, hostile relay networks, and adversarial propagation delay. Our specification further introduces low-bandwidth optimisations such as adaptive polling and compressed header synchronisation while preserving correctness. This document serves both as a blueprint for secure SPV implementation and a rebuttal of common misconceptions surrounding non-validating clients. 

---
# Leveraging Large Language Models for Spontaneous Speech-Based Suicide Risk Detection 

**Authors**: Yifan Gao, Jiao Fu, Long Guo, Hong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.00693)  

**Abstract**: Early identification of suicide risk is crucial for preventing suicidal behaviors. As a result, the identification and study of patterns and markers related to suicide risk have become a key focus of current research. In this paper, we present the results of our work in the 1st SpeechWellness Challenge (SW1), which aims to explore speech as a non-invasive and easily accessible mental health indicator for identifying adolescents at risk of this http URL approach leverages large language model (LLM) as the primary tool for feature extraction, alongside conventional acoustic and semantic features. The proposed method achieves an accuracy of 74\% on the test set, ranking first in the SW1 challenge. These findings demonstrate the potential of LLM-based methods for analyzing speech in the context of suicide risk assessment. 

---
# MassTool: A Multi-Task Search-Based Tool Retrieval Framework for Large Language Models 

**Authors**: Jianghao Lin, Xinyuan Wang, Xinyi Dai, Menghui Zhu, Bo Chen, Ruiming Tang, Yong Yu, Weinan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.00487)  

**Abstract**: Tool retrieval is a critical component in enabling large language models (LLMs) to interact effectively with external tools. It aims to precisely filter the massive tools into a small set of candidates for the downstream tool-augmented LLMs. However, most existing approaches primarily focus on optimizing tool representations, often neglecting the importance of precise query comprehension. To address this gap, we introduce MassTool, a multi-task search-based framework designed to enhance both query representation and tool retrieval accuracy. MassTool employs a two-tower architecture: a tool usage detection tower that predicts the need for function calls, and a tool retrieval tower that leverages a query-centric graph convolution network (QC-GCN) for effective query-tool matching. It also incorporates search-based user intent modeling (SUIM) to handle diverse and out-of-distribution queries, alongside an adaptive knowledge transfer (AdaKT) module for efficient multi-task learning. By jointly optimizing tool usage detection loss, list-wise retrieval loss, and contrastive regularization loss, MassTool establishes a robust dual-step sequential decision-making pipeline for precise query understanding. Extensive experiments demonstrate its effectiveness in improving retrieval accuracy. Our code is available at this https URL. 

---
# Beat and Downbeat Tracking in Performance MIDI Using an End-to-End Transformer Architecture 

**Authors**: Sebastian Murgul, Michael Heizmann  

**Link**: [PDF](https://arxiv.org/pdf/2507.00466)  

**Abstract**: Beat tracking in musical performance MIDI is a challenging and important task for notation-level music transcription and rhythmical analysis, yet existing methods primarily focus on audio-based approaches. This paper proposes an end-to-end transformer-based model for beat and downbeat tracking in performance MIDI, leveraging an encoder-decoder architecture for sequence-to-sequence translation of MIDI input to beat annotations. Our approach introduces novel data preprocessing techniques, including dynamic augmentation and optimized tokenization strategies, to improve accuracy and generalizability across different datasets. We conduct extensive experiments using the A-MAPS, ASAP, GuitarSet, and Leduc datasets, comparing our model against state-of-the-art hidden Markov models (HMMs) and deep learning-based beat tracking methods. The results demonstrate that our model outperforms existing symbolic music beat tracking approaches, achieving competitive F1-scores across various musical styles and instruments. Our findings highlight the potential of transformer architectures for symbolic beat tracking and suggest future integration with automatic music transcription systems for enhanced music analysis and score generation. 

---
# Overcoming Long-Context Limitations of State-Space Models via Context-Dependent Sparse Attention 

**Authors**: Zhihao Zhan, Jianan Zhao, Zhaocheng Zhu, Jian Tang  

**Link**: [PDF](https://arxiv.org/pdf/2507.00449)  

**Abstract**: Efficient long-context modeling remains a critical challenge for natural language processing (NLP), as the time complexity of the predominant Transformer architecture scales quadratically with the sequence length. While state-space models (SSMs) offer alternative sub-quadratic solutions, they struggle to capture long-range dependencies effectively. In this work, we focus on analyzing and improving the long-context modeling capabilities of SSMs. We show that the widely used synthetic task, associative recall, which requires a model to recall a value associated with a single key without context, insufficiently represents the complexities of real-world long-context modeling. To address this limitation, we extend the associative recall to a novel synthetic task, \emph{joint recall}, which requires a model to recall the value associated with a key given in a specified context. Theoretically, we prove that SSMs do not have the expressiveness to solve multi-query joint recall in sub-quadratic time complexity. To resolve this issue, we propose a solution based on integrating SSMs with Context-Dependent Sparse Attention (CDSA), which has the expressiveness to solve multi-query joint recall with sub-quadratic computation. To bridge the gap between theoretical analysis and real-world applications, we propose locality-sensitive Hashing Attention with sparse Key Selection (HAX), which instantiates the theoretical solution and is further tailored to natural language domains. Extensive experiments on both synthetic and real-world long-context benchmarks show that HAX consistently outperforms SSM baselines and SSMs integrated with context-independent sparse attention (CISA). 

---
# Does Math Reasoning Improve General LLM Capabilities? Understanding Transferability of LLM Reasoning 

**Authors**: Maggie Huan, Yuetai Li, Tuney Zheng, Xiaoyu Xu, Seungone Kim, Minxin Du, Radha Poovendran, Graham Neubig, Xiang Yue  

**Link**: [PDF](https://arxiv.org/pdf/2507.00432)  

**Abstract**: Math reasoning has become the poster child of progress in large language models (LLMs), with new models rapidly surpassing human-level performance on benchmarks like MATH and AIME. But as math leaderboards improve week by week, it is worth asking: do these gains reflect broader problem-solving ability or just narrow overfitting? To answer this question, we evaluate over 20 open-weight reasoning-tuned models across a broad suite of tasks, including math, scientific QA, agent planning, coding, and standard instruction-following. We surprisingly find that most models that succeed in math fail to transfer their gains to other domains. To rigorously study this phenomenon, we conduct controlled experiments on Qwen3-14B models using math-only data but different tuning methods. We find that reinforcement learning (RL)-tuned models generalize well across domains, while supervised fine-tuning (SFT)-tuned models often forget general capabilities. Latent-space representation and token-space distribution shift analyses reveal that SFT induces substantial representation and output drift, while RL preserves general-domain structure. Our results suggest a need to rethink standard post-training recipes, particularly the reliance on SFT-distilled data for advancing reasoning models. 

---
# Flexible Language Modeling in Continuous Space with Transformer-based Autoregressive Flows 

**Authors**: Ruixiang Zhang, Shuangfei Zhai, Jiatao Gu, Yizhe Zhang, Huangjie Zheng, Tianrong Chen, Miguel Angel Bautista, Josh Susskind, Navdeep Jaitly  

**Link**: [PDF](https://arxiv.org/pdf/2507.00425)  

**Abstract**: Autoregressive models have driven remarkable progress in language modeling. Their foundational reliance on discrete tokens, unidirectional context, and single-pass decoding, while central to their success, also inspires the exploration of a design space that could offer new axes of modeling flexibility. In this work, we explore an alternative paradigm, shifting language modeling from a discrete token space to a continuous latent space. We propose a novel framework TarFlowLM, that employs transformer-based autoregressive normalizing flows to model these continuous representations. This approach unlocks substantial flexibility, enabling the construction of models that can capture global bi-directional context through stacked, alternating-direction autoregressive transformations, support block-wise generation with flexible token patch sizes, and facilitate a hierarchical multi-pass generation process. We further propose new mixture-based coupling transformations designed to capture complex dependencies within the latent space shaped by discrete data, and demonstrate theoretical connections to conventional discrete autoregressive models. Extensive experiments on language modeling benchmarks demonstrate strong likelihood performance and highlight the flexible modeling capabilities inherent in our framework. 

---
# ASTRO: Teaching Language Models to Reason by Reflecting and Backtracking In-Context 

**Authors**: Joongwon Kim, Anirudh Goyal, Liang Tan, Hannaneh Hajishirzi, Srinivasan Iyer, Tianlu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.00417)  

**Abstract**: We introduce ASTRO, the "Autoregressive Search-Taught Reasoner", a framework for training language models to reason like search algorithms, explicitly leveraging self-reflection, backtracking, and exploration in their outputs. Recently, training large language models (LLMs) via reinforcement learning (RL) has led to the advent of reasoning models with greatly enhanced reasoning capabilities. Open-source replications of reasoning models, while successful, build upon models that already exhibit strong reasoning capabilities along with search behavior observed even before RL. As a result, it is yet unclear how to boost the reasoning capabilities of other non-reasoner models including Llama 3. ASTRO teaches such models to internalize structured search behavior through a synthetic dataset derived from Monte Carlo Tree Search (MCTS) over mathematical problem-solving trajectories. By converting search traces into natural language chain-of-thoughts that capture both successes and recoveries from failure, ASTRO bootstraps models with a rich prior for exploration during RL. We finetune our models on these search-derived traces and further improve performance via RL with verifiable rewards. We apply ASTRO to the Llama 3 family of models and achieve absolute performance gains of 16.0% on MATH-500, 26.9% on AMC 2023, and 20.0% on AIME 2024, especially improving upon challenging problems that require iterative correction. Our results demonstrate that search-inspired training offers a principled way to instill robust reasoning capabilities into open LLMs. 

---
# $μ^2$Tokenizer: Differentiable Multi-Scale Multi-Modal Tokenizer for Radiology Report Generation 

**Authors**: Siyou Li, Pengyao Qin, Huanan Wu, Dong Nie, Arun J. Thirunavukarasu, Juntao Yu, Le Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.00316)  

**Abstract**: Automated radiology report generation (RRG) aims to produce detailed textual reports from clinical imaging, such as computed tomography (CT) scans, to improve the accuracy and efficiency of diagnosis and provision of management advice. RRG is complicated by two key challenges: (1) inherent complexity in extracting relevant information from imaging data under resource constraints, and (2) difficulty in objectively evaluating discrepancies between model-generated and expert-written reports. To address these challenges, we propose $\mu^2$LLM, a $\underline{\textbf{mu}}$ltiscale $\underline{\textbf{mu}}$ltimodal large language models for RRG tasks. The novel ${\mu}^2$Tokenizer, as an intermediate layer, integrates multi-modal features from the multiscale visual tokenizer and the text tokenizer, then enhances report generation quality through direct preference optimization (DPO), guided by GREEN-RedLlama. Experimental results on four large CT image-report medical datasetdemonstrate that our method outperforms existing approaches, highlighting the potential of our fine-tuned $\mu^2$LLMs on limited data for RRG tasks. 

---
# Open-ended Scientific Discovery via Bayesian Surprise 

**Authors**: Dhruv Agarwal, Bodhisattwa Prasad Majumder, Reece Adamson, Megha Chakravorty, Satvika Reddy Gavireddy, Aditya Parashar, Harshit Surana, Bhavana Dalvi Mishra, Andrew McCallum, Ashish Sabharwal, Peter Clark  

**Link**: [PDF](https://arxiv.org/pdf/2507.00310)  

**Abstract**: The promise of autonomous scientific discovery (ASD) hinges not only on answering questions, but also on knowing which questions to ask. Most recent works in ASD explore the use of large language models (LLMs) in goal-driven settings, relying on human-specified research questions to guide hypothesis generation. However, scientific discovery may be accelerated further by allowing the AI system to drive exploration by its own criteria. The few existing approaches in open-ended ASD select hypotheses based on diversity heuristics or subjective proxies for human interestingness, but the former struggles to meaningfully navigate the typically vast hypothesis space, and the latter suffers from imprecise definitions. This paper presents AutoDS -- a method for open-ended ASD that instead drives scientific exploration using Bayesian surprise. Here, we quantify the epistemic shift from the LLM's prior beliefs about a hypothesis to its posterior beliefs after gathering experimental results. To efficiently explore the space of nested hypotheses, our method employs a Monte Carlo tree search (MCTS) strategy with progressive widening using surprisal as the reward function. We evaluate AutoDS in the setting of data-driven discovery across 21 real-world datasets spanning domains such as biology, economics, finance, and behavioral science. Our results demonstrate that under a fixed budget, AutoDS substantially outperforms competitors by producing 5--29\% more discoveries deemed surprising by the LLM. Our human evaluation further finds that two-thirds of AutoDS discoveries are surprising to the domain experts, suggesting this is an important step forward towards building open-ended ASD systems. 

---
# Developing Lightweight DNN Models With Limited Data For Real-Time Sign Language Recognition 

**Authors**: Nikita Nikitin, Eugene Fomin  

**Link**: [PDF](https://arxiv.org/pdf/2507.00248)  

**Abstract**: We present a novel framework for real-time sign language recognition using lightweight DNNs trained on limited data. Our system addresses key challenges in sign language recognition, including data scarcity, high computational costs, and discrepancies in frame rates between training and inference environments. By encoding sign language specific parameters, such as handshape, palm orientation, movement, and location into vectorized inputs, and leveraging MediaPipe for landmark extraction, we achieve highly separable input data representations. Our DNN architecture, optimized for sub 10MB deployment, enables accurate classification of 343 signs with less than 10ms latency on edge devices. The data annotation platform 'slait data' facilitates structured labeling and vector extraction. Our model achieved 92% accuracy in isolated sign recognition and has been integrated into the 'slait ai' web application, where it demonstrates stable inference. 

---
# Interpretable AI for Time-Series: Multi-Model Heatmap Fusion with Global Attention and NLP-Generated Explanations 

**Authors**: Jiztom Kavalakkatt Francis, Matthew J Darr  

**Link**: [PDF](https://arxiv.org/pdf/2507.00234)  

**Abstract**: In this paper, we present a novel framework for enhancing model interpretability by integrating heatmaps produced separately by ResNet and a restructured 2D Transformer with globally weighted input saliency. We address the critical problem of spatial-temporal misalignment in existing interpretability methods, where convolutional networks fail to capture global context and Transformers lack localized precision - a limitation that impedes actionable insights in safety-critical domains like healthcare and industrial monitoring. Our method merges gradient-weighted activation maps (ResNet) and Transformer attention rollout into a unified visualization, achieving full spatial-temporal alignment while preserving real-time performance. Empirical evaluations on clinical (ECG arrhythmia detection) and industrial (energy consumption prediction) datasets demonstrate significant improvements: the hybrid framework achieves 94.1% accuracy (F1 0.93) on the PhysioNet dataset and reduces regression error to RMSE = 0.28 kWh (R2 = 0.95) on the UCI Energy Appliance dataset-outperforming standalone ResNet, Transformer, and InceptionTime baselines by 3.8-12.4%. An NLP module translates fused heatmaps into domain-specific narratives (e.g., "Elevated ST-segment between 2-4 seconds suggests myocardial ischemia"), validated via BLEU-4 (0.586) and ROUGE-L (0.650) scores. By formalizing interpretability as causal fidelity and spatial-temporal alignment, our approach bridges the gap between technical outputs and stakeholder understanding, offering a scalable solution for transparent, time-aware decision-making. 

---
# Thinking About Thinking: SAGE-nano's Inverse Reasoning for Self-Aware Language Models 

**Authors**: Basab Jha, Firoj Paudel, Ujjwal Puri, Zhang Yuting, Choi Donghyuk, Wang Junhao  

**Link**: [PDF](https://arxiv.org/pdf/2507.00092)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities at solving complex reasoning tasks with Chain-of-Thought (CoT) prompting, but their decision-making processes remain somewhat blackbox. We introduce textbfinverse reasoning, a novel paradigm enabling LLMs to decompose and explain their own reasoning chains post-hoc. Our approach, used in SAGE-nano, a 4-billion-parameter reasoning model, employs a metacognitive structure that reflects back via attention processes to identify major decision points and generate explanations of reasoning choices. While typical CoT approaches are directed towards forward reasoning generation, inverse reasoning provides insight into why specific reasoning chains were selected over others. Through thorough testing of logical reasoning puzzles, math problems and ethical dilemmas from AQUA-RAT, CommonsenseQA, and customized benchmarks, we demonstrate that SAGE-nano is at the cutting edge both on reasoning accuracy (74.6% on AQUA-RAT) and explanation quality (92.1% human preference score) for its task, and offers performance almost on par with models like Claude-3.5 Sonnet or GPT-4o. Our contributions are: (i) the first rigorous framework for LLM self-reflection via inverse reasoning, (ii) a novel metalearning framework to reverse the attention flow, (iii) comprehensive evaluation frameworks for reasoning transparency, and (iv) evidence that increasing reasoning using inverse reasoning improves interpretability along with reasoning performance. Our work creates new avenues for transparent AI systems and closes significant gaps in AI safety, education, and scientific discovery. 

---
# Federated Learning-Enabled Hybrid Language Models for Communication-Efficient Token Transmission 

**Authors**: Faranaksadat Solat, Joohyung Lee, Mohamed Seif, Dusit Niyato, H. Vincent Poor  

**Link**: [PDF](https://arxiv.org/pdf/2507.00082)  

**Abstract**: Hybrid Language Models (HLMs) combine the low-latency efficiency of Small Language Models (SLMs) on edge devices with the high accuracy of Large Language Models (LLMs) on centralized servers. Unlike traditional end-to-end LLM inference, HLMs reduce latency and communication by invoking LLMs only when local SLM predictions are uncertain, i.e., when token-level confidence is low or entropy is high. However, ambiguous or low-confidence predictions still require frequent offloading to the LLM, leading to significant communication overhead in bandwidth-constrained settings. To address this, we propose FedHLM, a communication-efficient HLM framework that integrates uncertainty-aware inference with Federated Learning (FL). FedHLM's key innovation lies in collaboratively learning token-level uncertainty thresholds that govern when LLM assistance is needed. Rather than using static or manually tuned thresholds, FedHLM employs FL to optimize these thresholds in a privacy-preserving, distributed manner. Additionally, it leverages embedding-based token representations for Peer-to-Peer (P2P) resolution, enabling clients to reuse tokens inferred by semantically similar peers without engaging the LLM. We further introduce hierarchical model aggregation: edge servers refine local routing policies through client updates, while cross-cluster coordination aligns global decision boundaries. This layered design captures recurring uncertainty patterns, reducing redundant LLM queries. Experiments on large-scale news classification tasks show that FedHLM reduces LLM transmissions by over 95 percent with negligible accuracy loss, making it well-suited for scalable and efficient edge-AI applications. 

---
# State and Memory is All You Need for Robust and Reliable AI Agents 

**Authors**: Matthew Muhoberac, Atharva Parikh, Nirvi Vakharia, Saniya Virani, Aco Radujevic, Savannah Wood, Meghav Verma, Dimitri Metaxotos, Jeyaraman Soundararajan, Thierry Masquelin, Alexander G. Godfrey, Sean Gardner, Dobrila Rudnicki, Sam Michael, Gaurav Chopra  

**Link**: [PDF](https://arxiv.org/pdf/2507.00081)  

**Abstract**: Large language models (LLMs) have enabled powerful advances in natural language understanding and generation. Yet their application to complex, real-world scientific workflows remain limited by challenges in memory, planning, and tool integration. Here, we introduce SciBORG (Scientific Bespoke Artificial Intelligence Agents Optimized for Research Goals), a modular agentic framework that allows LLM-based agents to autonomously plan, reason, and achieve robust and reliable domain-specific task execution. Agents are constructed dynamically from source code documentation and augmented with finite-state automata (FSA) memory, enabling persistent state tracking and context-aware decision-making. This approach eliminates the need for manual prompt engineering and allows for robust, scalable deployment across diverse applications via maintaining context across extended workflows and to recover from tool or execution failures. We validate SciBORG through integration with both physical and virtual hardware, such as microwave synthesizers for executing user-specified reactions, with context-aware decision making and demonstrate its use in autonomous multi-step bioassay retrieval from the PubChem database utilizing multi-step planning, reasoning, agent-to-agent communication and coordination for execution of exploratory tasks. Systematic benchmarking shows that SciBORG agents achieve reliable execution, adaptive planning, and interpretable state transitions. Our results show that memory and state awareness are critical enablers of agentic planning and reliability, offering a generalizable foundation for deploying AI agents in complex environments. 

---
# The language of time: a language model perspective on time-series foundation models 

**Authors**: Yi Xie, Yun Xiong, Zejian Shi, Hao Niu, Zhengfu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.00078)  

**Abstract**: With the rise of large language models, the paradigm of training foundation models with massive parameter counts on vast datasets has been adopted in multiple domains to achieve remarkable success. Time series foundation models represent a significant extension of this paradigm, demonstrating exceptional expressive power, generalization, and cross-domain transferability. However, this gives rise to a fundamental paradox: time series data reflect distinct dynamical systems, making cross-domain transfer intuitively implausible, yet this is contradicted by the models' empirical success. To resolve this paradox, this paper investigates, from both theoretical and experimental perspectives, the representation learning mechanisms and generalization capabilities of patch-based time series foundation models. We argue that such models are not merely applying a new architecture but are fundamentally generalizing the representation paradigm of language models by extending deterministic vector-based representations to latent probabilistic distributional forms. Our theoretical analysis supports this framework by demonstrating that continuous time-series patches can be faithfully quantized into a discrete vocabulary whose key statistical properties are highly consistent with those of natural language. This generalization allows time series models to inherit the robust representation and transfer abilities of large language models, thereby explaining their superior performance in temporal tasks. Ultimately, our work provides a rigorous theoretical cornerstone for understanding, evaluating, and improving the safety and reliability of large-scale time series foundation models. 

---
# MANTA: Cross-Modal Semantic Alignment and Information-Theoretic Optimization for Long-form Multimodal Understanding 

**Authors**: Ziqi Zhong, Daniel Tang  

**Link**: [PDF](https://arxiv.org/pdf/2507.00068)  

**Abstract**: While multi-modal learning has advanced significantly, current approaches often treat modalities separately, creating inconsistencies in representation and reasoning. We introduce MANTA (Multi-modal Abstraction and Normalization via Textual Alignment), a theoretically-grounded framework that unifies visual and auditory inputs into a structured textual space for seamless processing with large language models. MANTA addresses four key challenges: (1) semantic alignment across modalities with information-theoretic optimization, (2) adaptive temporal synchronization for varying information densities, (3) hierarchical content representation for multi-scale understanding, and (4) context-aware retrieval of sparse information from long sequences. We formalize our approach within a rigorous mathematical framework, proving its optimality for context selection under token constraints. Extensive experiments on the challenging task of Long Video Question Answering show that MANTA improves state-of-the-art models by up to 22.6% in overall accuracy, with particularly significant gains (27.3%) on videos exceeding 30 minutes. Additionally, we demonstrate MANTA's superiority on temporal reasoning tasks (23.8% improvement) and cross-modal understanding (25.1% improvement). Our framework introduces novel density estimation techniques for redundancy minimization while preserving rare signals, establishing new foundations for unifying multimodal representations through structured text. 

---
# Enhancing Reasoning Capabilities in SLMs with Reward Guided Dataset Distillation 

**Authors**: Shreyansh Padarha  

**Link**: [PDF](https://arxiv.org/pdf/2507.00054)  

**Abstract**: The push to compress and impart the proficiency of Large Language Models (LLMs) into more deployable and efficient Small Language Models (SLMs) has benefited from improvements in knowledge distillation (KD) techniques. These techniques allow a smaller student model to learn from a more capable and larger teacher model's responses. However, distillation often revolves around the student model merely copying the teacher's in-distribution responses, limiting its generalisability. This limitation is amplified on reasoning tasks and can be computationally expensive. In this study, we propose AdvDistill, a reward-guided dataset distillation framework. We utilise multiple generations (responses) from a teacher for each prompt and assign rewards based on rule-based verifiers. These varying and normally distributed rewards serve as weights when training student models. Our methods and their subsequent behavioural analysis demonstrate a significant improvement in student model performance for mathematical and complex reasoning tasks, showcasing the efficacy and benefits of incorporating a rewarding mechanism in dataset distillation processes. 

---
# CaughtCheating: Is Your MLLM a Good Cheating Detective? Exploring the Boundary of Visual Perception and Reasoning 

**Authors**: Ming Li, Chenguang Wang, Yijun Liang, Xiyao Wang, Yuhang Zhou, Xiyang Wu, Yuqing Zhang, Ruiyi Zhang, Tianyi Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2507.00045)  

**Abstract**: Recent agentic Multi-Modal Large Language Models (MLLMs) such as GPT-o3 have achieved near-ceiling scores on various existing benchmarks, motivating a demand for more challenging test tasks. These MLLMs have been reported to excel in a few expert-level tasks for humans, e.g., GeoGuesser, reflecting their potential as a detective who can notice minuscule cues in an image and weave them into coherent, situational explanations, leading to a reliable answer. But can they match the performance of excellent human detectives? To answer this question, we investigate some hard scenarios where GPT-o3 can still handle, and find a common scenario where o3's performance drops to nearly zero, which we name CaughtCheating. It is inspired by the social media requests that ask others to detect suspicious clues from photos shared by the poster's partner. We conduct extensive experiments and analysis to understand why existing MLLMs lack sufficient capability to solve this kind of task. CaughtCheating provides a class of challenging visual perception and reasoning tasks with great value and practical usage. Success in these tasks paves the way for MLLMs to acquire human-level detective perception and reasoning capabilities. 

---
# Moment Sampling in Video LLMs for Long-Form Video QA 

**Authors**: Mustafa Chasmai, Gauri Jagatap, Gouthaman KV, Grant Van Horn, Subhransu Maji, Andrea Fanelli  

**Link**: [PDF](https://arxiv.org/pdf/2507.00033)  

**Abstract**: Recent advancements in video large language models (Video LLMs) have significantly advanced the field of video question answering (VideoQA). While existing methods perform well on short videos, they often struggle with long-range reasoning in longer videos. To scale Video LLMs for longer video content, frame sub-sampling (selecting frames at regular intervals) is commonly used. However, this approach is suboptimal, often leading to the loss of crucial frames or the inclusion of redundant information from multiple similar frames. Missing key frames impairs the model's ability to answer questions accurately, while redundant frames lead the model to focus on irrelevant video segments and increase computational resource consumption. In this paper, we investigate the use of a general-purpose text-to-video moment retrieval model to guide the frame sampling process. We propose "moment sampling", a novel, model-agnostic approach that enables the model to select the most relevant frames according to the context of the question. Specifically, we employ a lightweight moment retrieval model to prioritize frame selection. By focusing on the frames most pertinent to the given question, our method enhances long-form VideoQA performance in Video LLMs. Through extensive experiments on four long-form VideoQA datasets, using four state-of-the-art Video LLMs, we demonstrate the effectiveness of the proposed approach. 

---
# ROSE: Toward Reality-Oriented Safety Evaluation of Large Language Models 

**Authors**: Jiale Ding, Xiang Zheng, Cong Wang, Wei-Bin Lee, Xingjun Ma, Yu-Gang Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2507.00026)  

**Abstract**: As Large Language Models (LLMs) are increasingly deployed as black-box components in real-world applications, evaluating their safety-especially under adversarial prompting-has become critical. Arguably, effective safety evaluations should be adaptive, evolving with LLM capabilities, and also cover a broad spectrum of harmful topics and real-world scenarios to fully expose potential vulnerabilities. Existing manual safety benchmarks, built on handcrafted adversarial prompts, are limited by their static nature and the intensive labor required to update them, making it difficult to keep pace with rapidly advancing LLMs. In contrast, automated adversarial prompt generation offers a promising path toward adaptive evaluation. However, current methods often suffer from insufficient adversarial topic coverage (topic-level diversity) and weak alignment with real-world contexts. These shortcomings stem from the exploration-exploitation dilemma in black-box optimization and a lack of real-world contextualization, resulting in adversarial prompts that are both topically narrow and scenario-repetitive. To address these issues, we propose Reality-Oriented Safety Evaluation (ROSE), a novel framework that uses multi-objective reinforcement learning to fine-tune an adversarial LLM for generating topically diverse and contextually rich adversarial prompts. Experiments show that ROSE outperforms existing methods in uncovering safety vulnerabilities in state-of-the-art LLMs, with notable improvements in integrated evaluation metrics. We hope ROSE represents a step toward more practical and reality-oriented safety evaluation of LLMs. WARNING: This paper contains examples of potentially harmful text. 

---
# GLU Attention Improve Transformer 

**Authors**: Zehao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.00022)  

**Abstract**: Gated Linear Units (GLU) have shown great potential in enhancing neural network performance. In this paper, I introduce a novel attention mechanism called GLU Attention, which introduces nonlinearity into the values of Attention. My experiments demonstrate that GLU Attention improves both model performance and convergence speed across text and vision modalities with zero additional parameters and negligible computational costs. GLU Attention is lightweight and can seamlessly integrate with other technologies, such as Flash Attention, Rotary Position Embedding (RoPE), and various Multi-Head Attention (MHA) variants such as Grouped-Query Attention (GQA). This project is open-sourced at github. 

---
# Implicit Reward as the Bridge: A Unified View of SFT and DPO Connections 

**Authors**: Bo Wang, Qinyuan Cheng, Runyu Peng, Rong Bao, Peiji Li, Qipeng Guo, Linyang Li, Zhiyuan Zeng, Yunhua Zhou, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2507.00018)  

**Abstract**: Post-training processes are essential phases in grounding pre-trained language models to real-world tasks, with learning from demonstrations or preference signals playing a crucial role in this adaptation. We present a unified theoretical framework bridging Supervised Fine-Tuning (SFT) and preference learning in Large Language Model (LLM) post-training. Through rigorous mathematical derivation, we demonstrate that both SFT and preference learning methods like Direct Preference Optimization (DPO) operate within the same optimal policy-reward subspace, with SFT representing a special case of implicit reward learning. Our analysis reveals a critical limitation in conventional SFT: the KL divergence term in distribution matching becomes constant with respect to the policy during optimization, failing to constrain model updates. To address this, we propose a simple yet effective learning rate reduction approach that yields significant performance improvements (up to \textbf{25\%} relative gain and \textbf{6\%} absolute win rate increase in instruction following tasks. Additionally, we derive alternative SFT objectives from various f-divergence functions that preserve the KL term during optimization, further enhancing post-DPO model performance. Finally, we extend the theoretical relationship between LLM logits and Q-functions from preference learning to the SFT context, providing mathematical derivations and experimental validation. 

---
# Hypertokens: Holographic Associative Memory in Tokenized LLMs 

**Authors**: Christopher James Augeri  

**Link**: [PDF](https://arxiv.org/pdf/2507.00002)  

**Abstract**: Large Language Models (LLMs) exhibit remarkable capabilities but suffer from apparent precision loss, reframed here as information spreading. This reframing shifts the problem from computational precision to an information-theoretic communication issue. We address the K:V and V:K memory problem in LLMs by introducing HDRAM (Holographically Defined Random Access Memory), a symbolic memory framework treating transformer latent space as a spread-spectrum channel. Built upon hypertokens, structured symbolic codes integrating classical error-correcting codes (ECC), holographic computing, and quantum-inspired search, HDRAM recovers distributed information through principled despreading. These phase-coherent memory addresses enable efficient key-value operations and Grover-style search in latent space. By combining ECC grammar with compressed sensing and Krylov subspace alignment, HDRAM significantly improves associative retrieval without architectural changes, demonstrating how Classical-Holographic-Quantum-inspired (CHQ) principles can fortify transformer architectures. 

---
