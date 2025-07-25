# The Thin Line Between Comprehension and Persuasion in LLMs 

**Authors**: Adrian de Wynter, Tangming Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2507.01936)  

**Abstract**: Large language models (LLMs) are excellent at maintaining high-level, convincing dialogues. They are being fast deployed as chatbots and evaluators in sensitive areas, such as peer review and mental health applications. This, along with the disparate accounts on their reasoning capabilities, calls for a closer examination of LLMs and their comprehension of dialogue. In this work we begin by evaluating LLMs' ability to maintain a debate--one of the purest yet most complex forms of human communication. Then we measure how this capability relates to their understanding of what is being talked about, namely, their comprehension of dialogical structures and the pragmatic context. We find that LLMs are capable of maintaining coherent, persuasive debates, often swaying the beliefs of participants and audiences alike. We also note that awareness or suspicion of AI involvement encourage people to be more critical of the arguments made. When polling LLMs on their comprehension of deeper structures of dialogue, however, they cannot demonstrate said understanding. Our findings tie the shortcomings of LLMs-as-evaluators to their (in)ability to understand the context. More broadly, for the field of argumentation theory we posit that, if an agent can convincingly maintain a dialogue, it is not necessary for it to know what it is talking about. Hence, the modelling of pragmatic context and coherence are secondary to effectiveness. 

---
# Adaptability of ASR Models on Low-Resource Language: A Comparative Study of Whisper and Wav2Vec-BERT on Bangla 

**Authors**: Md Sazzadul Islam Ridoy, Sumi Akter, Md. Aminur Rahman  

**Link**: [PDF](https://arxiv.org/pdf/2507.01931)  

**Abstract**: In recent years, neural models trained on large multilingual text and speech datasets have shown great potential for supporting low-resource languages. This study investigates the performances of two state-of-the-art Automatic Speech Recognition (ASR) models, OpenAI's Whisper (Small & Large-V2) and Facebook's Wav2Vec-BERT on Bangla, a low-resource language. We have conducted experiments using two publicly available datasets: Mozilla Common Voice-17 and OpenSLR to evaluate model performances. Through systematic fine-tuning and hyperparameter optimization, including learning rate, epochs, and model checkpoint selection, we have compared the models based on Word Error Rate (WER), Character Error Rate (CER), Training Time, and Computational Efficiency. The Wav2Vec-BERT model outperformed Whisper across all key evaluation metrics, demonstrated superior performance while requiring fewer computational resources, and offered valuable insights to develop robust speech recognition systems in low-resource linguistic settings. 

---
# Decision-oriented Text Evaluation 

**Authors**: Yu-Shiang Huang, Chuan-Ju Wang, Chung-Chi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.01923)  

**Abstract**: Natural language generation (NLG) is increasingly deployed in high-stakes domains, yet common intrinsic evaluation methods, such as n-gram overlap or sentence plausibility, weakly correlate with actual decision-making efficacy. We propose a decision-oriented framework for evaluating generated text by directly measuring its influence on human and large language model (LLM) decision outcomes. Using market digest texts--including objective morning summaries and subjective closing-bell analyses--as test cases, we assess decision quality based on the financial performance of trades executed by human investors and autonomous LLM agents informed exclusively by these texts. Our findings reveal that neither humans nor LLM agents consistently surpass random performance when relying solely on summaries. However, richer analytical commentaries enable collaborative human-LLM teams to outperform individual human or agent baselines significantly. Our approach underscores the importance of evaluating generated text by its ability to facilitate synergistic decision-making between humans and LLMs, highlighting critical limitations of traditional intrinsic metrics. 

---
# NaturalThoughts: Selecting and Distilling Reasoning Traces for General Reasoning Tasks 

**Authors**: Yang Li, Youssef Emad, Karthik Padthe, Jack Lanchantin, Weizhe Yuan, Thao Nguyen, Jason Weston, Shang-Wen Li, Dong Wang, Ilia Kulikov, Xian Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.01921)  

**Abstract**: Recent work has shown that distilling reasoning traces from a larger teacher model via supervised finetuning outperforms reinforcement learning with the smaller student model alone (Guo et al. 2025). However, there has not been a systematic study of what kind of reasoning demonstrations from the teacher are most effective in improving the student model's reasoning capabilities. In this work we curate high-quality "NaturalThoughts" by selecting reasoning traces from a strong teacher model based on a large pool of questions from NaturalReasoning (Yuan et al. 2025). We first conduct a systematic analysis of factors that affect distilling reasoning capabilities, in terms of sample efficiency and scalability for general reasoning tasks. We observe that simply scaling up data size with random sampling is a strong baseline with steady performance gains. Further, we find that selecting difficult examples that require more diverse reasoning strategies is more sample-efficient to transfer the teacher model's reasoning skills. Evaluated on both Llama and Qwen models, training with NaturalThoughts outperforms existing reasoning datasets such as OpenThoughts, LIMO, etc. on general STEM reasoning benchmarks including GPQA-Diamond, MMLU-Pro and SuperGPQA. 

---
# Gradient-Adaptive Policy Optimization: Towards Multi-Objective Alignment of Large Language Models 

**Authors**: Chengao Li, Hanyu Zhang, Yunkun Xu, Hongyan Xue, Xiang Ao, Qing He  

**Link**: [PDF](https://arxiv.org/pdf/2507.01915)  

**Abstract**: Reinforcement Learning from Human Feedback (RLHF) has emerged as a powerful technique for aligning large language models (LLMs) with human preferences. However, effectively aligning LLMs with diverse human preferences remains a significant challenge, particularly when they are conflict. To address this issue, we frame human value alignment as a multi-objective optimization problem, aiming to maximize a set of potentially conflicting objectives. We introduce Gradient-Adaptive Policy Optimization (GAPO), a novel fine-tuning paradigm that employs multiple-gradient descent to align LLMs with diverse preference distributions. GAPO adaptively rescales the gradients for each objective to determine an update direction that optimally balances the trade-offs between objectives. Additionally, we introduce P-GAPO, which incorporates user preferences across different objectives and achieves Pareto solutions that better align with the user's specific needs. Our theoretical analysis demonstrates that GAPO converges towards a Pareto optimal solution for multiple objectives. Empirical results on Mistral-7B show that GAPO outperforms current state-of-the-art methods, achieving superior performance in both helpfulness and harmlessness. 

---
# AI4Research: A Survey of Artificial Intelligence for Scientific Research 

**Authors**: Qiguang Chen, Mingda Yang, Libo Qin, Jinhao Liu, Zheng Yan, Jiannan Guan, Dengyun Peng, Yiyan Ji, Hanjing Li, Mengkang Hu, Yimeng Zhang, Yihao Liang, Yuhang Zhou, Jiaqi Wang, Zhi Chen, Wanxiang Che  

**Link**: [PDF](https://arxiv.org/pdf/2507.01903)  

**Abstract**: Recent advancements in artificial intelligence (AI), particularly in large language models (LLMs) such as OpenAI-o1 and DeepSeek-R1, have demonstrated remarkable capabilities in complex domains such as logical reasoning and experimental coding. Motivated by these advancements, numerous studies have explored the application of AI in the innovation process, particularly in the context of scientific research. These AI technologies primarily aim to develop systems that can autonomously conduct research processes across a wide range of scientific disciplines. Despite these significant strides, a comprehensive survey on AI for Research (AI4Research) remains absent, which hampers our understanding and impedes further development in this field. To address this gap, we present a comprehensive survey and offer a unified perspective on AI4Research. Specifically, the main contributions of our work are as follows: (1) Systematic taxonomy: We first introduce a systematic taxonomy to classify five mainstream tasks in AI4Research. (2) New frontiers: Then, we identify key research gaps and highlight promising future directions, focusing on the rigor and scalability of automated experiments, as well as the societal impact. (3) Abundant applications and resources: Finally, we compile a wealth of resources, including relevant multidisciplinary applications, data corpora, and tools. We hope our work will provide the research community with quick access to these resources and stimulate innovative breakthroughs in AI4Research. 

---
# High-Layer Attention Pruning with Rescaling 

**Authors**: Songtao Liu, Peng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.01900)  

**Abstract**: Pruning is a highly effective approach for compressing large language models (LLMs), significantly reducing inference latency. However, conventional training-free structured pruning methods often employ a heuristic metric that indiscriminately removes some attention heads across all pruning layers, without considering their positions within the network architecture. In this work, we propose a novel pruning algorithm that strategically prunes attention heads in the model's higher layers. Since the removal of attention heads can alter the magnitude of token representations, we introduce an adaptive rescaling parameter that calibrates the representation scale post-pruning to counteract this effect. We conduct comprehensive experiments on a wide range of LLMs, including LLaMA3.1-8B, Mistral-7B-v0.3, Qwen2-7B, and Gemma2-9B. Our evaluation includes both generation and discriminative tasks across 27 datasets. The results consistently demonstrate that our method outperforms existing structured pruning methods. This improvement is particularly notable in generation tasks, where our approach significantly outperforms existing baselines. 

---
# MiCoTA: Bridging the Learnability Gap with Intermediate CoT and Teacher Assistants 

**Authors**: Dongyi Ding, Tiannan Wang, Chenghao Zhu, Meiling Tao, Yuchen Eleanor Jiang, Wangchunshu Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2507.01887)  

**Abstract**: Large language models (LLMs) excel at reasoning tasks requiring long thought sequences for planning, reflection, and refinement. However, their substantial model size and high computational demands are impractical for widespread deployment. Yet, small language models (SLMs) often struggle to learn long-form CoT reasoning due to their limited capacity, a phenomenon we refer to as the "SLMs Learnability Gap". To address this, we introduce \textbf{Mi}d-\textbf{Co}T \textbf{T}eacher \textbf{A}ssistant Distillation (MiCoTAl), a framework for improving long CoT distillation for SLMs. MiCoTA employs intermediate-sized models as teacher assistants and utilizes intermediate-length CoT sequences to bridge both the capacity and reasoning length gaps. Our experiments on downstream tasks demonstrate that although SLMs distilled from large teachers can perform poorly, by applying MiCoTA, they achieve significant improvements in reasoning performance. Specifically, Qwen2.5-7B-Instruct and Qwen2.5-3B-Instruct achieve an improvement of 3.47 and 3.93 respectively on average score on AIME2024, AMC, Olympiad, MATH-500 and GSM8K benchmarks. To better understand the mechanism behind MiCoTA, we perform a quantitative experiment demonstrating that our method produces data more closely aligned with base SLM distributions. Our insights pave the way for future research into long-CoT data distillation for SLMs. 

---
# DIY-MKG: An LLM-Based Polyglot Language Learning System 

**Authors**: Kenan Tang, Yanhong Li, Yao Qin  

**Link**: [PDF](https://arxiv.org/pdf/2507.01872)  

**Abstract**: Existing language learning tools, even those powered by Large Language Models (LLMs), often lack support for polyglot learners to build linguistic connections across vocabularies in multiple languages, provide limited customization for individual learning paces or needs, and suffer from detrimental cognitive offloading. To address these limitations, we design Do-It-Yourself Multilingual Knowledge Graph (DIY-MKG), an open-source system that supports polyglot language learning. DIY-MKG allows the user to build personalized vocabulary knowledge graphs, which are constructed by selective expansion with related words suggested by an LLM. The system further enhances learning through rich annotation capabilities and an adaptive review module that leverages LLMs for dynamic, personalized quiz generation. In addition, DIY-MKG allows users to flag incorrect quiz questions, simultaneously increasing user engagement and providing a feedback loop for prompt refinement. Our evaluation of LLM-based components in DIY-MKG shows that vocabulary expansion is reliable and fair across multiple languages, and that the generated quizzes are highly accurate, validating the robustness of DIY-MKG. 

---
# Eka-Eval : A Comprehensive Evaluation Framework for Large Language Models in Indian Languages 

**Authors**: Samridhi Raj Sinha, Rajvee Sheth, Abhishek Upperwal, Mayank Singh  

**Link**: [PDF](https://arxiv.org/pdf/2507.01853)  

**Abstract**: The rapid advancement of Large Language Models (LLMs) has intensified the need for evaluation frameworks that go beyond English centric benchmarks and address the requirements of linguistically diverse regions such as India. We present EKA-EVAL, a unified and production-ready evaluation framework that integrates over 35 benchmarks, including 10 Indic-specific datasets, spanning categories like reasoning, mathematics, tool use, long-context understanding, and reading comprehension. Compared to existing Indian language evaluation tools, EKA-EVAL offers broader benchmark coverage, with built-in support for distributed inference, quantization, and multi-GPU usage. Our systematic comparison positions EKA-EVAL as the first end-to-end, extensible evaluation suite tailored for both global and Indic LLMs, significantly lowering the barrier to multilingual benchmarking. The framework is open-source and publicly available at this https URL eka-eval and a part of ongoing EKA initiative (this https URL), which aims to scale up to over 100 benchmarks and establish a robust, multilingual evaluation ecosystem for LLMs. 

---
# Low-Perplexity LLM-Generated Sequences and Where To Find Them 

**Authors**: Arthur Wuhrmann, Anastasiia Kucherenko, Andrei Kucharavy  

**Link**: [PDF](https://arxiv.org/pdf/2507.01844)  

**Abstract**: As Large Language Models (LLMs) become increasingly widespread, understanding how specific training data shapes their outputs is crucial for transparency, accountability, privacy, and fairness. To explore how LLMs leverage and replicate their training data, we introduce a systematic approach centered on analyzing low-perplexity sequences - high-probability text spans generated by the model. Our pipeline reliably extracts such long sequences across diverse topics while avoiding degeneration, then traces them back to their sources in the training data. Surprisingly, we find that a substantial portion of these low-perplexity spans cannot be mapped to the corpus. For those that do match, we quantify the distribution of occurrences across source documents, highlighting the scope and nature of verbatim recall and paving a way toward better understanding of how LLMs training data impacts their behavior. 

---
# Evaluating Structured Output Robustness of Small Language Models for Open Attribute-Value Extraction from Clinical Notes 

**Authors**: Nikita Neveditsin, Pawan Lingras, Vijay Mago  

**Link**: [PDF](https://arxiv.org/pdf/2507.01810)  

**Abstract**: We present a comparative analysis of the parseability of structured outputs generated by small language models for open attribute-value extraction from clinical notes. We evaluate three widely used serialization formats: JSON, YAML, and XML, and find that JSON consistently yields the highest parseability. Structural robustness improves with targeted prompting and larger models, but declines for longer documents and certain note types. Our error analysis identifies recurring format-specific failure patterns. These findings offer practical guidance for selecting serialization formats and designing prompts when deploying language models in privacy-sensitive clinical settings. 

---
# The Anatomy of Evidence: An Investigation Into Explainable ICD Coding 

**Authors**: Katharina Beckh, Elisa Studeny, Sujan Sai Gannamaneni, Dario Antweiler, Stefan Rüping  

**Link**: [PDF](https://arxiv.org/pdf/2507.01802)  

**Abstract**: Automatic medical coding has the potential to ease documentation and billing processes. For this task, transparency plays an important role for medical coders and regulatory bodies, which can be achieved using explainability methods. However, the evaluation of these approaches has been mostly limited to short text and binary settings due to a scarcity of annotated data. Recent efforts by Cheng et al. (2023) have introduced the MDACE dataset, which provides a valuable resource containing code evidence in clinical records. In this work, we conduct an in-depth analysis of the MDACE dataset and perform plausibility evaluation of current explainable medical coding systems from an applied perspective. With this, we contribute to a deeper understanding of automatic medical coding and evidence extraction. Our findings reveal that ground truth evidence aligns with code descriptions to a certain degree. An investigation into state-of-the-art approaches shows a high overlap with ground truth evidence. We propose match measures and highlight success and failure cases. Based on our findings, we provide recommendations for developing and evaluating explainable medical coding systems. 

---
# How Do Vision-Language Models Process Conflicting Information Across Modalities? 

**Authors**: Tianze Hua, Tian Yun, Ellie Pavlick  

**Link**: [PDF](https://arxiv.org/pdf/2507.01790)  

**Abstract**: AI models are increasingly required to be multimodal, integrating disparate input streams into a coherent state representation on which subsequent behaviors and actions can be based. This paper seeks to understand how such models behave when input streams present conflicting information. Focusing specifically on vision-language models, we provide inconsistent inputs (e.g., an image of a dog paired with the caption "A photo of a cat") and ask the model to report the information present in one of the specific modalities (e.g., "What does the caption say / What is in the image?"). We find that models often favor one modality over the other, e.g., reporting the image regardless of what the caption says, but that different models differ in which modality they favor. We find evidence that the behaviorally preferred modality is evident in the internal representational structure of the model, and that specific attention heads can restructure the representations to favor one modality over the other. Moreover, we find modality-agnostic "router heads" which appear to promote answers about the modality requested in the instruction, and which can be manipulated or transferred in order to improve performance across datasets and modalities. Together, the work provides essential steps towards identifying and controlling if and how models detect and resolve conflicting signals within complex multimodal environments. 

---
# Probing Evaluation Awareness of Language Models 

**Authors**: Jord Nguyen, Khiem Hoang, Carlo Leonardo Attubato, Felix Hofstätter  

**Link**: [PDF](https://arxiv.org/pdf/2507.01786)  

**Abstract**: Language models can distinguish between testing and deployment phases -- a capability known as evaluation awareness. This has significant safety and policy implications, potentially undermining the reliability of evaluations that are central to AI governance frameworks and voluntary industry commitments. In this paper, we study evaluation awareness in Llama-3.3-70B-Instruct. We show that linear probes can separate real-world evaluation and deployment prompts, suggesting that current models internally represent this distinction. We also find that current safety evaluations are correctly classified by the probes, suggesting that they already appear artificial or inauthentic to models. Our findings underscore the importance of ensuring trustworthy evaluations and understanding deceptive capabilities. More broadly, our work showcases how model internals may be leveraged to support blackbox methods in safety audits, especially for future models more competent at evaluation awareness and deception. 

---
# MuRating: A High Quality Data Selecting Approach to Multilingual Large Language Model Pretraining 

**Authors**: Zhixun Chen, Ping Guo, Wenhan Han, Yifan Zhang, Binbin Liu, Haobin Lin, Fengze Liu, Yan Zhao, Bingni Zhang, Taifeng Wang, Yin Zheng, Meng Fang  

**Link**: [PDF](https://arxiv.org/pdf/2507.01785)  

**Abstract**: Data quality is a critical driver of large language model performance, yet existing model-based selection methods focus almost exclusively on English. We introduce MuRating, a scalable framework that transfers high-quality English data-quality signals into a single rater for 17 target languages. MuRating aggregates multiple English "raters" via pairwise comparisons to learn unified document-quality scores,then projects these judgments through translation to train a multilingual evaluator on monolingual, cross-lingual, and parallel text pairs. Applied to web data, MuRating selects balanced subsets of English and multilingual content to pretrain a 1.2 B-parameter LLaMA model. Compared to strong baselines, including QuRater, AskLLM, DCLM and so on, our approach boosts average accuracy on both English benchmarks and multilingual evaluations, with especially large gains on knowledge-intensive tasks. We further analyze translation fidelity, selection biases, and underrepresentation of narrative material, outlining directions for future work. 

---
# Data interference: emojis, homoglyphs, and issues of data fidelity in corpora and their results 

**Authors**: Matteo Di Cristofaro  

**Link**: [PDF](https://arxiv.org/pdf/2507.01764)  

**Abstract**: Tokenisation - "the process of splitting text into atomic parts" (Brezina & Timperley, 2017: 1) - is a crucial step for corpus linguistics, as it provides the basis for any applicable quantitative method (e.g. collocations) while ensuring the reliability of qualitative approaches. This paper examines how discrepancies in tokenisation affect the representation of language data and the validity of analytical findings: investigating the challenges posed by emojis and homoglyphs, the study highlights the necessity of preprocessing these elements to maintain corpus fidelity to the source data. The research presents methods for ensuring that digital texts are accurately represented in corpora, thereby supporting reliable linguistic analysis and guaranteeing the repeatability of linguistic interpretations. The findings emphasise the necessity of a detailed understanding of both linguistic and technical aspects involved in digital textual data to enhance the accuracy of corpus analysis, and have significant implications for both quantitative and qualitative approaches in corpus-based research. 

---
# LLMs for Legal Subsumption in German Employment Contracts 

**Authors**: Oliver Wardas, Florian Matthes  

**Link**: [PDF](https://arxiv.org/pdf/2507.01734)  

**Abstract**: Legal work, characterized by its text-heavy and resource-intensive nature, presents unique challenges and opportunities for NLP research. While data-driven approaches have advanced the field, their lack of interpretability and trustworthiness limits their applicability in dynamic legal environments. To address these issues, we collaborated with legal experts to extend an existing dataset and explored the use of Large Language Models (LLMs) and in-context learning to evaluate the legality of clauses in German employment contracts. Our work evaluates the ability of different LLMs to classify clauses as "valid," "unfair," or "void" under three legal context variants: no legal context, full-text sources of laws and court rulings, and distilled versions of these (referred to as examination guidelines). Results show that full-text sources moderately improve performance, while examination guidelines significantly enhance recall for void clauses and weighted F1-Score, reaching 80\%. Despite these advancements, LLMs' performance when using full-text sources remains substantially below that of human lawyers. We contribute an extended dataset, including examination guidelines, referenced legal sources, and corresponding annotations, alongside our code and all log files. Our findings highlight the potential of LLMs to assist lawyers in contract legality review while also underscoring the limitations of the methods presented. 

---
# Stereotype Detection as a Catalyst for Enhanced Bias Detection: A Multi-Task Learning Approach 

**Authors**: Aditya Tomar, Rudra Murthy, Pushpak Bhattacharyya  

**Link**: [PDF](https://arxiv.org/pdf/2507.01715)  

**Abstract**: Bias and stereotypes in language models can cause harm, especially in sensitive areas like content moderation and decision-making. This paper addresses bias and stereotype detection by exploring how jointly learning these tasks enhances model performance. We introduce StereoBias, a unique dataset labeled for bias and stereotype detection across five categories: religion, gender, socio-economic status, race, profession, and others, enabling a deeper study of their relationship. Our experiments compare encoder-only models and fine-tuned decoder-only models using QLoRA. While encoder-only models perform well, decoder-only models also show competitive results. Crucially, joint training on bias and stereotype detection significantly improves bias detection compared to training them separately. Additional experiments with sentiment analysis confirm that the improvements stem from the connection between bias and stereotypes, not multi-task learning alone. These findings highlight the value of leveraging stereotype information to build fairer and more effective AI systems. 

---
# AdamMeme: Adaptively Probe the Reasoning Capacity of Multimodal Large Language Models on Harmfulness 

**Authors**: Zixin Chen, Hongzhan Lin, Kaixin Li, Ziyang Luo, Zhen Ye, Guang Chen, Zhiyong Huang, Jing Ma  

**Link**: [PDF](https://arxiv.org/pdf/2507.01702)  

**Abstract**: The proliferation of multimodal memes in the social media era demands that multimodal Large Language Models (mLLMs) effectively understand meme harmfulness. Existing benchmarks for assessing mLLMs on harmful meme understanding rely on accuracy-based, model-agnostic evaluations using static datasets. These benchmarks are limited in their ability to provide up-to-date and thorough assessments, as online memes evolve dynamically. To address this, we propose AdamMeme, a flexible, agent-based evaluation framework that adaptively probes the reasoning capabilities of mLLMs in deciphering meme harmfulness. Through multi-agent collaboration, AdamMeme provides comprehensive evaluations by iteratively updating the meme data with challenging samples, thereby exposing specific limitations in how mLLMs interpret harmfulness. Extensive experiments show that our framework systematically reveals the varying performance of different target mLLMs, offering in-depth, fine-grained analyses of model-specific weaknesses. Our code is available at this https URL. 

---
# Adapting Language Models to Indonesian Local Languages: An Empirical Study of Language Transferability on Zero-Shot Settings 

**Authors**: Rifki Afina Putri  

**Link**: [PDF](https://arxiv.org/pdf/2507.01645)  

**Abstract**: In this paper, we investigate the transferability of pre-trained language models to low-resource Indonesian local languages through the task of sentiment analysis. We evaluate both zero-shot performance and adapter-based transfer on ten local languages using models of different types: a monolingual Indonesian BERT, multilingual models such as mBERT and XLM-R, and a modular adapter-based approach called MAD-X. To better understand model behavior, we group the target languages into three categories: seen (included during pre-training), partially seen (not included but linguistically related to seen languages), and unseen (absent and unrelated in pre-training data). Our results reveal clear performance disparities across these groups: multilingual models perform best on seen languages, moderately on partially seen ones, and poorly on unseen languages. We find that MAD-X significantly improves performance, especially for seen and partially seen languages, without requiring labeled data in the target language. Additionally, we conduct a further analysis on tokenization and show that while subword fragmentation and vocabulary overlap with Indonesian correlate weakly with prediction quality, they do not fully explain the observed performance. Instead, the most consistent predictor of transfer success is the model's prior exposure to the language, either directly or through a related language. 

---
# Confidence and Stability of Global and Pairwise Scores in NLP Evaluation 

**Authors**: Georgii Levtsov, Dmitry Ustalov  

**Link**: [PDF](https://arxiv.org/pdf/2507.01633)  

**Abstract**: With the advent of highly capable instruction-tuned neural language models, benchmarking in natural language processing (NLP) is increasingly shifting towards pairwise comparison leaderboards, such as LMSYS Arena, from traditional global pointwise scores (e.g., GLUE, BIG-bench, SWE-bench). This paper empirically investigates the strengths and weaknesses of both global scores and pairwise comparisons to aid decision-making in selecting appropriate model evaluation strategies. Through computational experiments on synthetic and real-world datasets using standard global metrics and the popular Bradley-Terry model for pairwise comparisons, we found that while global scores provide more reliable overall rankings, they can underestimate strong models with rare, significant errors or low confidence. Conversely, pairwise comparisons are particularly effective for identifying strong contenders among models with lower global scores, especially where quality metrics are hard to define (e.g., text generation), though they require more comparisons to converge if ties are frequent. Our code and data are available at this https URL under a permissive license. 

---
# Chart Question Answering from Real-World Analytical Narratives 

**Authors**: Maeve Hutchinson, Radu Jianu, Aidan Slingsby, Jo Wood, Pranava Madhyastha  

**Link**: [PDF](https://arxiv.org/pdf/2507.01627)  

**Abstract**: We present a new dataset for chart question answering (CQA) constructed from visualization notebooks. The dataset features real-world, multi-view charts paired with natural language questions grounded in analytical narratives. Unlike prior benchmarks, our data reflects ecologically valid reasoning workflows. Benchmarking state-of-the-art multimodal large language models reveals a significant performance gap, with GPT-4.1 achieving an accuracy of 69.3%, underscoring the challenges posed by this more authentic CQA setting. 

---
# Emotionally Intelligent Task-oriented Dialogue Systems: Architecture, Representation, and Optimisation 

**Authors**: Shutong Feng, Hsien-chin Lin, Nurul Lubis, Carel van Niekerk, Michael Heck, Benjamin Ruppik, Renato Vukovic, Milica Gašić  

**Link**: [PDF](https://arxiv.org/pdf/2507.01594)  

**Abstract**: Task-oriented dialogue (ToD) systems are designed to help users achieve specific goals through natural language interaction. While recent advances in large language models (LLMs) have significantly improved linguistic fluency and contextual understanding, building effective and emotionally intelligent ToD systems remains a complex challenge. Effective ToD systems must optimise for task success, emotional understanding and responsiveness, and precise information conveyance, all within inherently noisy and ambiguous conversational environments. In this work, we investigate architectural, representational, optimisational as well as emotional considerations of ToD systems. We set up systems covering these design considerations with a challenging evaluation environment composed of a natural-language user simulator coupled with an imperfect natural language understanding module. We propose \textbf{LUSTER}, an \textbf{L}LM-based \textbf{U}nified \textbf{S}ystem for \textbf{T}ask-oriented dialogue with \textbf{E}nd-to-end \textbf{R}einforcement learning with both short-term (user sentiment) and long-term (task success) rewards. Our findings demonstrate that combining LLM capability with structured reward modelling leads to more resilient and emotionally responsive ToD systems, offering a practical path forward for next-generation conversational agents. 

---
# Is External Information Useful for Stance Detection with LLMs? 

**Authors**: Quang Minh Nguyen, Taegyoon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2507.01543)  

**Abstract**: In the stance detection task, a text is classified as either favorable, opposing, or neutral towards a target. Prior work suggests that the use of external information, e.g., excerpts from Wikipedia, improves stance detection performance. However, whether or not such information can benefit large language models (LLMs) remains an unanswered question, despite their wide adoption in many reasoning tasks. In this study, we conduct a systematic evaluation on how Wikipedia and web search external information can affect stance detection across eight LLMs and in three datasets with 12 targets. Surprisingly, we find that such information degrades performance in most cases, with macro F1 scores dropping by up to 27.9\%. We explain this through experiments showing LLMs' tendency to align their predictions with the stance and sentiment of the provided information rather than the ground truth stance of the given text. We also find that performance degradation persists with chain-of-thought prompting, while fine-tuning mitigates but does not fully eliminate it. Our findings, in contrast to previous literature on BERT-based systems which suggests that external information enhances performance, highlight the risks of information biases in LLM-based stance classifiers. Code is available at this https URL. 

---
# Efficient Out-of-Scope Detection in Dialogue Systems via Uncertainty-Driven LLM Routing 

**Authors**: Álvaro Zaera, Diana Nicoleta Popa, Ivan Sekulic, Paolo Rosso  

**Link**: [PDF](https://arxiv.org/pdf/2507.01541)  

**Abstract**: Out-of-scope (OOS) intent detection is a critical challenge in task-oriented dialogue systems (TODS), as it ensures robustness to unseen and ambiguous queries. In this work, we propose a novel but simple modular framework that combines uncertainty modeling with fine-tuned large language models (LLMs) for efficient and accurate OOS detection. The first step applies uncertainty estimation to the output of an in-scope intent detection classifier, which is currently deployed in a real-world TODS handling tens of thousands of user interactions daily. The second step then leverages an emerging LLM-based approach, where a fine-tuned LLM is triggered to make a final decision on instances with high uncertainty. Unlike prior approaches, our method effectively balances computational efficiency and performance, combining traditional approaches with LLMs and yielding state-of-the-art results on key OOS detection benchmarks, including real-world OOS data acquired from a deployed TODS. 

---
# Evaluating the Effectiveness of Direct Preference Optimization for Personalizing German Automatic Text Simplifications for Persons with Intellectual Disabilities 

**Authors**: Yingqiang Gao, Kaede Johnson, David Froehlich, Luisa Carrer, Sarah Ebling  

**Link**: [PDF](https://arxiv.org/pdf/2507.01479)  

**Abstract**: Automatic text simplification (ATS) aims to enhance language accessibility for various target groups, particularly persons with intellectual disabilities. Recent advancements in generative AI, especially large language models (LLMs), have substantially improved the quality of machine-generated text simplifications, thereby mitigating information barriers for the target group. However, existing LLM-based ATS systems do not incorporate preference feedback on text simplifications during training, resulting in a lack of personalization tailored to the specific needs of target group representatives.
In this work, we extend the standard supervised fine-tuning (SFT) approach for adapting LLM-based ATS models by leveraging a computationally efficient LLM alignment technique -- direct preference optimization (DPO). Specifically, we post-train LLM-based ATS models using human feedback collected from persons with intellectual disabilities, reflecting their preferences on paired text simplifications generated by mainstream LLMs. Furthermore, we propose a pipeline for developing personalized LLM-based ATS systems, encompassing data collection, model selection, SFT and DPO post-training, and evaluation. Our findings underscore the necessity of active participation of target group persons in designing personalized AI accessibility solutions aligned with human expectations. This work represents a step towards personalizing inclusive AI systems at the target-group level, incorporating insights not only from text simplification experts but also from target group persons themselves. 

---
# LogitSpec: Accelerating Retrieval-based Speculative Decoding via Next Next Token Speculation 

**Authors**: Tianyu Liu, Qitan Lv, Hao Li, Xing Gao, Xiao Sun  

**Link**: [PDF](https://arxiv.org/pdf/2507.01449)  

**Abstract**: Speculative decoding (SD), where a small draft model is employed to propose draft tokens in advance and then the target model validates them in parallel, has emerged as a promising technique for LLM inference acceleration. Many endeavors to improve SD are to eliminate the need for a draft model and generate draft tokens in a retrieval-based manner in order to further alleviate the drafting overhead and significantly reduce the difficulty in deployment and applications. However, retrieval-based SD relies on a matching paradigm to retrieval the most relevant reference as the draft tokens, where these methods often fail to find matched and accurate draft tokens. To address this challenge, we propose LogitSpec to effectively expand the retrieval range and find the most relevant reference as drafts. Our LogitSpec is motivated by the observation that the logit of the last token can not only predict the next token, but also speculate the next next token. Specifically, LogitSpec generates draft tokens in two steps: (1) utilizing the last logit to speculate the next next token; (2) retrieving relevant reference for both the next token and the next next token. LogitSpec is training-free and plug-and-play, which can be easily integrated into existing LLM inference frameworks. Extensive experiments on a wide range of text generation benchmarks demonstrate that LogitSpec can achieve up to 2.61 $\times$ speedup and 3.28 mean accepted tokens per decoding step. Our code is available at this https URL. 

---
# Clinical NLP with Attention-Based Deep Learning for Multi-Disease Prediction 

**Authors**: Ting Xu, Xiaoxiao Deng, Xiandong Meng, Haifeng Yang, Yan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2507.01437)  

**Abstract**: This paper addresses the challenges posed by the unstructured nature and high-dimensional semantic complexity of electronic health record texts. A deep learning method based on attention mechanisms is proposed to achieve unified modeling for information extraction and multi-label disease prediction. The study is conducted on the MIMIC-IV dataset. A Transformer-based architecture is used to perform representation learning over clinical text. Multi-layer self-attention mechanisms are employed to capture key medical entities and their contextual relationships. A Sigmoid-based multi-label classifier is then applied to predict multiple disease labels. The model incorporates a context-aware semantic alignment mechanism, enhancing its representational capacity in typical medical scenarios such as label co-occurrence and sparse information. To comprehensively evaluate model performance, a series of experiments were conducted, including baseline comparisons, hyperparameter sensitivity analysis, data perturbation studies, and noise injection tests. Results demonstrate that the proposed method consistently outperforms representative existing approaches across multiple performance metrics. The model maintains strong generalization under varying data scales, interference levels, and model depth configurations. The framework developed in this study offers an efficient algorithmic foundation for processing real-world clinical texts and presents practical significance for multi-label medical text modeling tasks. 

---
# Skywork-Reward-V2: Scaling Preference Data Curation via Human-AI Synergy 

**Authors**: Chris Yuhao Liu, Liang Zeng, Yuzhen Xiao, Jujie He, Jiacai Liu, Chaojie Wang, Rui Yan, Wei Shen, Fuxiang Zhang, Jiacheng Xu, Yang Liu, Yahui Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2507.01352)  

**Abstract**: Despite the critical role of reward models (RMs) in reinforcement learning from human feedback (RLHF), current state-of-the-art open RMs perform poorly on most existing evaluation benchmarks, failing to capture the spectrum of nuanced and sophisticated human preferences. Even approaches that incorporate advanced training techniques have not yielded meaningful performance improvements. We hypothesize that this brittleness stems primarily from limitations in preference datasets, which are often narrowly scoped, synthetically labeled, or lack rigorous quality control. To address these challenges, we present a large-scale preference dataset comprising 40 million preference pairs, named SynPref-40M. To enable data curation at scale, we design a human-AI synergistic two-stage pipeline that leverages the complementary strengths of human annotation quality and AI scalability. In this pipeline, humans provide verified annotations, while large language models perform automatic curation based on human guidance. Training on this preference mixture, we introduce Skywork-Reward-V2, a suite of eight reward models ranging from 0.6B to 8B parameters, trained on a carefully curated subset of 26 million preference pairs from SynPref-40M. We demonstrate that Skywork-Reward-V2 is versatile across a wide range of capabilities, including alignment with human preferences, objective correctness, safety, resistance to stylistic biases, and best-of-N scaling, achieving state-of-the-art performance across seven major reward model benchmarks. Ablation studies confirm that the effectiveness of our approach stems not only from data scale but also from high-quality curation. The Skywork-Reward-V2 series represents substantial progress in open reward models, highlighting the untapped potential of existing preference datasets and demonstrating how human-AI curation synergy can unlock significantly higher data quality. 

---
# LEDOM: An Open and Fundamental Reverse Language Model 

**Authors**: Xunjian Yin, Sitao Cheng, Yuxi Xie, Xinyu Hu, Li Lin, Xinyi Wang, Liangming Pan, William Yang Wang, Xiaojun Wan  

**Link**: [PDF](https://arxiv.org/pdf/2507.01335)  

**Abstract**: We introduce LEDOM, the first purely reverse language model, trained autoregressively on 435B tokens with 2B and 7B parameter variants, which processes sequences in reverse temporal order through previous token prediction. For the first time, we present the reverse language model as a potential foundational model across general tasks, accompanied by a set of intriguing examples and insights. Based on LEDOM, we further introduce a novel application: Reverse Reward, where LEDOM-guided reranking of forward language model outputs leads to substantial performance improvements on mathematical reasoning tasks. This approach leverages LEDOM's unique backward reasoning capability to refine generation quality through posterior evaluation. Our findings suggest that LEDOM exhibits unique characteristics with broad application potential. We will release all models, training code, and pre-training data to facilitate future research. 

---
# Symbolic or Numerical? Understanding Physics Problem Solving in Reasoning LLMs 

**Authors**: Nifu Dan, Yujun Cai, Yiwei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.01334)  

**Abstract**: Navigating the complexities of physics reasoning has long been a difficult task for Large Language Models (LLMs), requiring a synthesis of profound conceptual understanding and adept problem-solving techniques. In this study, we investigate the application of advanced instruction-tuned reasoning models, such as Deepseek-R1, to address a diverse spectrum of physics problems curated from the challenging SciBench benchmark. Our comprehensive experimental evaluation reveals the remarkable capabilities of reasoning models. Not only do they achieve state-of-the-art accuracy in answering intricate physics questions, but they also generate distinctive reasoning patterns that emphasize on symbolic derivation. Furthermore, our findings indicate that even for these highly sophisticated reasoning models, the strategic incorporation of few-shot prompting can still yield measurable improvements in overall accuracy, highlighting the potential for continued performance gains. 

---
# La RoSA: Enhancing LLM Efficiency via Layerwise Rotated Sparse Activation 

**Authors**: Kai Liu, Bowen Xu, Shaoyu Wu, Xin Chen, Hao Zhou, Yongliang Tao, Lulu Hu  

**Link**: [PDF](https://arxiv.org/pdf/2507.01299)  

**Abstract**: Activation sparsity can reduce the computational overhead and memory transfers during the forward pass of Large Language Model (LLM) inference. Existing methods face limitations, either demanding time-consuming recovery training that hinders real-world adoption, or relying on empirical magnitude-based pruning, which causes fluctuating sparsity and unstable inference speed-up. This paper introduces LaRoSA (Layerwise Rotated Sparse Activation), a novel method for activation sparsification designed to improve LLM efficiency without requiring additional training or magnitude-based pruning. We leverage layerwise orthogonal rotations to transform input activations into rotated forms that are more suitable for sparsification. By employing a Top-K selection approach within the rotated activations, we achieve consistent model-level sparsity and reliable wall-clock time speed-up. LaRoSA is effective across various sizes and types of LLMs, demonstrating minimal performance degradation and robust inference acceleration. Specifically, for LLaMA2-7B at 40% sparsity, LaRoSA achieves a mere 0.17 perplexity gap with a consistent 1.30x wall-clock time speed-up, and reduces the accuracy gap in zero-shot tasks compared to the dense model to just 0.54%, while surpassing TEAL by 1.77% and CATS by 17.14%. 

---
# Frustratingly Simple Retrieval Improves Challenging, Reasoning-Intensive Benchmarks 

**Authors**: Xinxi Lyu, Michael Duan, Rulin Shao, Pang Wei Koh, Sewon Min  

**Link**: [PDF](https://arxiv.org/pdf/2507.01297)  

**Abstract**: Retrieval-augmented Generation (RAG) has primarily been studied in limited settings, such as factoid question answering; more challenging, reasoning-intensive benchmarks have seen limited success from minimal RAG. In this work, we challenge this prevailing view on established, reasoning-intensive benchmarks: MMLU, MMLU Pro, AGI Eval, GPQA, and MATH. We identify a key missing component in prior work: a usable, web-scale datastore aligned with the breadth of pretraining data. To this end, we introduce CompactDS: a diverse, high-quality, web-scale datastore that achieves high retrieval accuracy and subsecond latency on a single-node. The key insights are (1) most web content can be filtered out without sacrificing coverage, and a compact, high-quality subset is sufficient; and (2) combining in-memory approximate nearest neighbor (ANN) retrieval and on-disk exact search balances speed and recall. Using CompactDS, we show that a minimal RAG pipeline achieves consistent accuracy improvements across all benchmarks and model sizes (8B--70B), with relative gains of 10% on MMLU, 33% on MMLU Pro, 14% on GPQA, and 19% on MATH. No single data source suffices alone, highlighting the importance of diversity of sources (web crawls, curated math, academic papers, textbooks). Finally, we show that our carefully designed in-house datastore matches or outperforms web search engines such as Google Search, as well as recently proposed, complex agent-based RAG systems--all while maintaining simplicity, reproducibility, and self-containment. We release CompactDS and our retrieval pipeline, supporting future research exploring retrieval-based AI systems. 

---
# Rethinking All Evidence: Enhancing Trustworthy Retrieval-Augmented Generation via Conflict-Driven Summarization 

**Authors**: Juan Chen, Baolong Bi, Wei Zhang, Jingyan Sui, Xiaofei Zhu, Yuanzhuo Wang, Lingrui Mei, Shenghua Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.01281)  

**Abstract**: Retrieval-Augmented Generation (RAG) enhances large language models (LLMs) by integrating their parametric knowledge with external retrieved content. However, knowledge conflicts caused by internal inconsistencies or noisy retrieved content can severely undermine the generation reliability of RAG this http URL this work, we argue that LLMs should rethink all evidence, including both retrieved content and internal knowledge, before generating this http URL propose CARE-RAG (Conflict-Aware and Reliable Evidence for RAG), a novel framework that improves trustworthiness through Conflict-Driven Summarization of all available this http URL-RAG first derives parameter-aware evidence by comparing parameter records to identify diverse internal perspectives. It then refines retrieved evidences to produce context-aware evidence, removing irrelevant or misleading content. To detect and summarize conflicts, we distill a 3B LLaMA3.2 model to perform conflict-driven summarization, enabling reliable synthesis across multiple this http URL further ensure evaluation integrity, we introduce a QA Repair step to correct outdated or ambiguous benchmark this http URL on revised QA datasets with retrieval data show that CARE-RAG consistently outperforms strong RAG baselines, especially in scenarios with noisy or conflicting evidence. 

---
# Evaluating Large Language Models for Multimodal Simulated Ophthalmic Decision-Making in Diabetic Retinopathy and Glaucoma Screening 

**Authors**: Cindy Lie Tabuse, David Restepo, Carolina Gracitelli, Fernando Korn Malerbi, Caio Regatieri, Luis Filipe Nakayama  

**Link**: [PDF](https://arxiv.org/pdf/2507.01278)  

**Abstract**: Large language models (LLMs) can simulate clinical reasoning based on natural language prompts, but their utility in ophthalmology is largely unexplored. This study evaluated GPT-4's ability to interpret structured textual descriptions of retinal fundus photographs and simulate clinical decisions for diabetic retinopathy (DR) and glaucoma screening, including the impact of adding real or synthetic clinical metadata. We conducted a retrospective diagnostic validation study using 300 annotated fundus images. GPT-4 received structured prompts describing each image, with or without patient metadata. The model was tasked with assigning an ICDR severity score, recommending DR referral, and estimating the cup-to-disc ratio for glaucoma referral. Performance was evaluated using accuracy, macro and weighted F1 scores, and Cohen's kappa. McNemar's test and change rate analysis were used to assess the influence of metadata. GPT-4 showed moderate performance for ICDR classification (accuracy 67.5%, macro F1 0.33, weighted F1 0.67, kappa 0.25), driven mainly by correct identification of normal cases. Performance improved in the binary DR referral task (accuracy 82.3%, F1 0.54, kappa 0.44). For glaucoma referral, performance was poor across all settings (accuracy ~78%, F1 <0.04, kappa <0.03). Metadata inclusion did not significantly affect outcomes (McNemar p > 0.05), and predictions remained consistent across conditions. GPT-4 can simulate basic ophthalmic decision-making from structured prompts but lacks precision for complex tasks. While not suitable for clinical use, LLMs may assist in education, documentation, or image annotation workflows in ophthalmology. 

---
# GAIus: Combining Genai with Legal Clauses Retrieval for Knowledge-based Assistant 

**Authors**: Michał Matak, Jarosław A. Chudziak  

**Link**: [PDF](https://arxiv.org/pdf/2507.01259)  

**Abstract**: In this paper we discuss the capability of large language models to base their answer and provide proper references when dealing with legal matters of non-english and non-chinese speaking country. We discuss the history of legal information retrieval, the difference between case law and statute law, its impact on the legal tasks and analyze the latest research in this field. Basing on that background we introduce gAIus, the architecture of the cognitive LLM-based agent, whose responses are based on the knowledge retrieved from certain legal act, which is Polish Civil Code. We propose a retrieval mechanism which is more explainable, human-friendly and achieves better results than embedding-based approaches. To evaluate our method we create special dataset based on single-choice questions from entrance exams for law apprenticeships conducted in Poland. The proposed architecture critically leveraged the abilities of used large language models, improving the gpt-3.5-turbo-0125 by 419%, allowing it to beat gpt-4o and lifting gpt-4o-mini score from 31% to 86%. At the end of our paper we show the possible future path of research and potential applications of our findings. 

---
# The Medium Is Not the Message: Deconfounding Text Embeddings via Linear Concept Erasure 

**Authors**: Yu Fan, Yang Tian, Shauli Ravfogel, Mrinmaya Sachan, Elliott Ash, Alexander Hoyle  

**Link**: [PDF](https://arxiv.org/pdf/2507.01234)  

**Abstract**: Embedding-based similarity metrics between text sequences can be influenced not just by the content dimensions we most care about, but can also be biased by spurious attributes like the text's source or language. These document confounders cause problems for many applications, but especially those that need to pool texts from different corpora. This paper shows that a debiasing algorithm that removes information about observed confounders from the encoder representations substantially reduces these biases at a minimal computational cost. Document similarity and clustering metrics improve across every embedding variant and task we evaluate -- often dramatically. Interestingly, performance on out-of-distribution benchmarks is not impacted, indicating that the embeddings are not otherwise degraded. 

---
# MEGA: xLSTM with Multihead Exponential Gated Fusion for Precise Aspect-based Sentiment Analysis 

**Authors**: Adamu Lawan, Juhua Pu, Haruna Yunusa, Jawad Muhammad, Muhammad Lawan  

**Link**: [PDF](https://arxiv.org/pdf/2507.01213)  

**Abstract**: Aspect-based Sentiment Analysis (ABSA) is a critical Natural Language Processing (NLP) task that extracts aspects from text and determines their associated sentiments, enabling fine-grained analysis of user opinions. Existing ABSA methods struggle to balance computational efficiency with high performance: deep learning models often lack global context, transformers demand significant computational resources, and Mamba-based approaches face CUDA dependency and diminished local correlations. Recent advancements in Extended Long Short-Term Memory (xLSTM) models, particularly their efficient modeling of long-range dependencies, have significantly advanced the NLP community. However, their potential in ABSA remains untapped. To this end, we propose xLSTM with Multihead Exponential Gated Fusion (MEGA), a novel framework integrating a bi-directional mLSTM architecture with forward and partially flipped backward (PF-mLSTM) streams. The PF-mLSTM enhances localized context modeling by processing the initial sequence segment in reverse with dedicated parameters, preserving critical short-range patterns. We further introduce an mLSTM-based multihead cross exponential gated fusion mechanism (MECGAF) that dynamically combines forward mLSTM outputs as query and key with PF-mLSTM outputs as value, optimizing short-range dependency capture while maintaining global context and efficiency. Experimental results on three benchmark datasets demonstrate that MEGA outperforms state-of-the-art baselines, achieving superior accuracy and efficiency in ABSA tasks. 

---
# Matching and Linking Entries in Historical Swedish Encyclopedias 

**Authors**: Simon Börjesson, Erik Ersmark, Pierre Nugues  

**Link**: [PDF](https://arxiv.org/pdf/2507.01170)  

**Abstract**: The \textit{Nordisk familjebok} is a Swedish encyclopedia from the 19th and 20th centuries. It was written by a team of experts and aimed to be an intellectual reference, stressing precision and accuracy. This encyclopedia had four main editions remarkable by their size, ranging from 20 to 38 volumes. As a consequence, the \textit{Nordisk familjebok} had a considerable influence in universities, schools, the media, and society overall. As new editions were released, the selection of entries and their content evolved, reflecting intellectual changes in Sweden.
In this paper, we used digitized versions from \textit{Project Runeberg}. We first resegmented the raw text into entries and matched pairs of entries between the first and second editions using semantic sentence embeddings. We then extracted the geographical entries from both editions using a transformer-based classifier and linked them to Wikidata. This enabled us to identify geographic trends and possible shifts between the first and second editions, written between 1876-1899 and 1904-1926, respectively.
Interpreting the results, we observe a small but significant shift in geographic focus away from Europe and towards North America, Africa, Asia, Australia, and northern Scandinavia from the first to the second edition, confirming the influence of the First World War and the rise of new powers. The code and data are available on GitHub at this https URL. 

---
# Event-based evaluation of abstractive news summarization 

**Authors**: Huiling You, Samia Touileb, Erik Velldal, Lilja Øvrelid  

**Link**: [PDF](https://arxiv.org/pdf/2507.01160)  

**Abstract**: An abstractive summary of a news article contains its most important information in a condensed version. The evaluation of automatically generated summaries by generative language models relies heavily on human-authored summaries as gold references, by calculating overlapping units or similarity scores. News articles report events, and ideally so should the summaries. In this work, we propose to evaluate the quality of abstractive summaries by calculating overlapping events between generated summaries, reference summaries, and the original news articles. We experiment on a richly annotated Norwegian dataset comprising both events annotations and summaries authored by expert human annotators. Our approach provides more insight into the event information contained in the summaries. 

---
# MALIBU Benchmark: Multi-Agent LLM Implicit Bias Uncovered 

**Authors**: Imran Mirza, Cole Huang, Ishwara Vasista, Rohan Patil, Asli Akalin, Sean O'Brien, Kevin Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2507.01019)  

**Abstract**: Multi-agent systems, which consist of multiple AI models interacting within a shared environment, are increasingly used for persona-based interactions. However, if not carefully designed, these systems can reinforce implicit biases in large language models (LLMs), raising concerns about fairness and equitable representation. We present MALIBU, a novel benchmark developed to assess the degree to which LLM-based multi-agent systems implicitly reinforce social biases and stereotypes. MALIBU evaluates bias in LLM-based multi-agent systems through scenario-based assessments. AI models complete tasks within predefined contexts, and their responses undergo evaluation by an LLM-based multi-agent judging system in two phases. In the first phase, judges score responses labeled with specific demographic personas (e.g., gender, race, religion) across four metrics. In the second phase, judges compare paired responses assigned to different personas, scoring them and selecting the superior response. Our study quantifies biases in LLM-generated outputs, revealing that bias mitigation may favor marginalized personas over true neutrality, emphasizing the need for nuanced detection, balanced fairness strategies, and transparent evaluation benchmarks in multi-agent systems. 

---
# Test-Time Scaling with Reflective Generative Model 

**Authors**: Zixiao Wang, Yuxin Wang, Xiaorui Wang, Mengting Xing, Jie Gao, Jianjun Xu, Guangcan Liu, Chenhui Jin, Zhuo Wang, Shengzhuo Zhang, Hongtao Xie  

**Link**: [PDF](https://arxiv.org/pdf/2507.01951)  

**Abstract**: We introduce our first reflective generative model MetaStone-S1, which obtains OpenAI o3's performance via the self-supervised process reward model (SPRM). Through sharing the backbone network and using task-specific heads for next token prediction and process scoring respectively, SPRM successfully integrates the policy model and process reward model(PRM) into a unified interface without extra process annotation, reducing over 99% PRM parameters for efficient reasoning. Equipped with SPRM, MetaStone-S1 is naturally suitable for test time scaling (TTS), and we provide three reasoning effort modes (low, medium, and high), based on the controllable thinking length. Moreover, we empirically establish a scaling law that reveals the relationship between total thinking computation and TTS performance. Experiments demonstrate that our MetaStone-S1 achieves comparable performance to OpenAI-o3-mini's series with only 32B parameter size. To support the research community, we have open-sourced MetaStone-S1 at this https URL. 

---
# LoRA Fine-Tuning Without GPUs: A CPU-Efficient Meta-Generation Framework for LLMs 

**Authors**: Reza Arabpour, Haitz Sáez de Ocáriz Borde, Anastasis Kratsios  

**Link**: [PDF](https://arxiv.org/pdf/2507.01806)  

**Abstract**: Low-Rank Adapters (LoRAs) have transformed the fine-tuning of Large Language Models (LLMs) by enabling parameter-efficient updates. However, their widespread adoption remains limited by the reliance on GPU-based training. In this work, we propose a theoretically grounded approach to LoRA fine-tuning designed specifically for users with limited computational resources, particularly those restricted to standard laptop CPUs. Our method learns a meta-operator that maps any input dataset, represented as a probability distribution, to a set of LoRA weights by leveraging a large bank of pre-trained adapters for the Mistral-7B-Instruct-v0.2 model. Instead of performing new gradient-based updates, our pipeline constructs adapters via lightweight combinations of existing LoRAs directly on CPU. While the resulting adapters do not match the performance of GPU-trained counterparts, they consistently outperform the base Mistral model on downstream tasks, offering a practical and accessible alternative to traditional GPU-based fine-tuning. 

---
# Tuning without Peeking: Provable Privacy and Generalization Bounds for LLM Post-Training 

**Authors**: Ismail Labiad, Mathurin Videau, Matthieu Kowalski, Marc Schoenauer, Alessandro Leite, Julia Kempe, Olivier Teytaud  

**Link**: [PDF](https://arxiv.org/pdf/2507.01752)  

**Abstract**: Gradient-based optimization is the workhorse of deep learning, offering efficient and scalable training via backpropagation. However, its reliance on large volumes of labeled data raises privacy and security concerns such as susceptibility to data poisoning attacks and the risk of overfitting. In contrast, black box optimization methods, which treat the model as an opaque function, relying solely on function evaluations to guide optimization, offer a promising alternative in scenarios where data access is restricted, adversarial risks are high, or overfitting is a concern. However, black box methods also pose significant challenges, including poor scalability to high-dimensional parameter spaces, as prevalent in large language models (LLMs), and high computational costs due to reliance on numerous model evaluations. This paper introduces BBoxER, an evolutionary black-box method for LLM post-training that induces an information bottleneck via implicit compression of the training data. Leveraging the tractability of information flow, we provide strong theoretical bounds on generalization, differential privacy, susceptibility to data poisoning attacks, and robustness to extraction attacks. BBoxER operates on top of pre-trained LLMs, offering a lightweight and modular enhancement suitable for deployment in restricted or privacy-sensitive environments, in addition to non-vacuous generalization guarantees. In experiments with LLMs, we demonstrate empirically that Retrofitting methods are able to learn, showing how a few iterations of BBoxER improve performance and generalize well on a benchmark of reasoning datasets. This positions BBoxER as an attractive add-on on top of gradient-based optimization. 

---
# ECCV 2024 W-CODA: 1st Workshop on Multimodal Perception and Comprehension of Corner Cases in Autonomous Driving 

**Authors**: Kai Chen, Ruiyuan Gao, Lanqing Hong, Hang Xu, Xu Jia, Holger Caesar, Dengxin Dai, Bingbing Liu, Dzmitry Tsishkou, Songcen Xu, Chunjing Xu, Qiang Xu, Huchuan Lu, Dit-Yan Yeung  

**Link**: [PDF](https://arxiv.org/pdf/2507.01735)  

**Abstract**: In this paper, we present details of the 1st W-CODA workshop, held in conjunction with the ECCV 2024. W-CODA aims to explore next-generation solutions for autonomous driving corner cases, empowered by state-of-the-art multimodal perception and comprehension techniques. 5 Speakers from both academia and industry are invited to share their latest progress and opinions. We collect research papers and hold a dual-track challenge, including both corner case scene understanding and generation. As the pioneering effort, we will continuously bridge the gap between frontier autonomous driving techniques and fully intelligent, reliable self-driving agents robust towards corner cases. 

---
# Blending Supervised and Reinforcement Fine-Tuning with Prefix Sampling 

**Authors**: Zeyu Huang, Tianhao Cheng, Zihan Qiu, Zili Wang, Yinghui Xu, Edoardo M. Ponti, Ivan Titov  

**Link**: [PDF](https://arxiv.org/pdf/2507.01679)  

**Abstract**: Existing post-training techniques for large language models are broadly categorized into Supervised Fine-Tuning (SFT) and Reinforcement Fine-Tuning (RFT). Each paradigm presents a distinct trade-off: SFT excels at mimicking demonstration data but can lead to problematic generalization as a form of behavior cloning. Conversely, RFT can significantly enhance a model's performance but is prone to learn unexpected behaviors, and its performance is highly sensitive to the initial policy. In this paper, we propose a unified view of these methods and introduce Prefix-RFT, a hybrid approach that synergizes learning from both demonstration and exploration. Using mathematical reasoning problems as a testbed, we empirically demonstrate that Prefix-RFT is both simple and effective. It not only surpasses the performance of standalone SFT and RFT but also outperforms parallel mixed-policy RFT methods. A key advantage is its seamless integration into existing open-source frameworks, requiring only minimal modifications to the standard RFT pipeline. Our analysis highlights the complementary nature of SFT and RFT, and validates that Prefix-RFT effectively harmonizes these two learning paradigms. Furthermore, ablation studies confirm the method's robustness to variations in the quality and quantity of demonstration data. We hope this work offers a new perspective on LLM post-training, suggesting that a unified paradigm that judiciously integrates demonstration and exploration could be a promising direction for future research. 

---
# Data Agent: A Holistic Architecture for Orchestrating Data+AI Ecosystems 

**Authors**: Zhaoyan Sun, Jiayi Wang, Xinyang Zhao, Jiachi Wang, Guoliang Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.01599)  

**Abstract**: Traditional Data+AI systems utilize data-driven techniques to optimize performance, but they rely heavily on human experts to orchestrate system pipelines, enabling them to adapt to changes in data, queries, tasks, and environments. For instance, while there are numerous data science tools available, developing a pipeline planning system to coordinate these tools remains challenging. This difficulty arises because existing Data+AI systems have limited capabilities in semantic understanding, reasoning, and planning. Fortunately, we have witnessed the success of large language models (LLMs) in enhancing semantic understanding, reasoning, and planning abilities. It is crucial to incorporate LLM techniques to revolutionize data systems for orchestrating Data+AI applications effectively.
To achieve this, we propose the concept of a 'Data Agent' - a comprehensive architecture designed to orchestrate Data+AI ecosystems, which focuses on tackling data-related tasks by integrating knowledge comprehension, reasoning, and planning capabilities. We delve into the challenges involved in designing data agents, such as understanding data/queries/environments/tools, orchestrating pipelines/workflows, optimizing and executing pipelines, and fostering pipeline self-reflection. Furthermore, we present examples of data agent systems, including a data science agent, data analytics agents (such as unstructured data analytics agent, semantic structured data analytics agent, data lake analytics agent, and multi-modal data analytics agent), and a database administrator (DBA) agent. We also outline several open challenges associated with designing data agent systems. 

---
# T3DM: Test-Time Training-Guided Distribution Shift Modelling for Temporal Knowledge Graph Reasoning 

**Authors**: Yuehang Si, Zefan Zeng, Jincai Huang, Qing Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2507.01597)  

**Abstract**: Temporal Knowledge Graph (TKG) is an efficient method for describing the dynamic development of facts along a timeline. Most research on TKG reasoning (TKGR) focuses on modelling the repetition of global facts and designing patterns of local historical facts. However, they face two significant challenges: inadequate modeling of the event distribution shift between training and test samples, and reliance on random entity substitution for generating negative samples, which often results in low-quality sampling. To this end, we propose a novel distributional feature modeling approach for training TKGR models, Test-Time Training-guided Distribution shift Modelling (T3DM), to adjust the model based on distribution shift and ensure the global consistency of model reasoning. In addition, we design a negative-sampling strategy to generate higher-quality negative quadruples based on adversarial training. Extensive experiments show that T3DM provides better and more robust results than the state-of-the-art baselines in most cases. 

---
# Self-Guided Process Reward Optimization with Masked Step Advantage for Process Reinforcement Learning 

**Authors**: Wu Fei, Hao Kong, Shuxian Liang, Yang Lin, Yibo Yang, Jing Tang, Lei Chen, Xiansheng Hua  

**Link**: [PDF](https://arxiv.org/pdf/2507.01551)  

**Abstract**: Process Reinforcement Learning~(PRL) has demonstrated considerable potential in enhancing the reasoning capabilities of Large Language Models~(LLMs). However, introducing additional process reward models incurs substantial computational overhead, and there is no unified theoretical framework for process-level advantage estimation. To bridge this gap, we propose \textbf{S}elf-Guided \textbf{P}rocess \textbf{R}eward \textbf{O}ptimization~(\textbf{SPRO}), a novel framework that enables process-aware RL through two key innovations: (1) we first theoretically demonstrate that process rewards can be derived intrinsically from the policy model itself, and (2) we introduce well-defined cumulative process rewards and \textbf{M}asked \textbf{S}tep \textbf{A}dvantage (\textbf{MSA}), which facilitates rigorous step-wise action advantage estimation within shared-prompt sampling groups. Our experimental results demonstrate that SPRO outperforms vaniila GRPO with 3.4x higher training efficiency and a 17.5\% test accuracy improvement. Furthermore, SPRO maintains a stable and elevated policy entropy throughout training while reducing the average response length by approximately $1/3$, evidencing sufficient exploration and prevention of reward hacking. Notably, SPRO incurs no additional computational overhead compared to outcome-supervised RL methods such as GRPO, which benefit industrial implementation. 

---
# Crafting Hanzi as Narrative Bridges: An AI Co-Creation Workshop for Elderly Migrants 

**Authors**: Wen Zhan, Ziqun Hua, Peiyue Lin, Yunfei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.01548)  

**Abstract**: This paper explores how older adults, particularly aging migrants in urban China, can engage AI-assisted co-creation to express personal narratives that are often fragmented, underrepresented, or difficult to verbalize. Through a pilot workshop combining oral storytelling and the symbolic reconstruction of Hanzi, participants shared memories of migration and recreated new character forms using Xiaozhuan glyphs, suggested by the Large Language Model (LLM), together with physical materials. Supported by human facilitation and a soft AI presence, participants transformed lived experience into visual and tactile expressions without requiring digital literacy. This approach offers new perspectives on human-AI collaboration and aging by repositioning AI not as a content producer but as a supportive mechanism, and by supporting narrative agency within sociotechnical systems. 

---
# Following the Clues: Experiments on Person Re-ID using Cross-Modal Intelligence 

**Authors**: Robert Aufschläger, Youssef Shoeb, Azarm Nowzad, Michael Heigl, Fabian Bally, Martin Schramm  

**Link**: [PDF](https://arxiv.org/pdf/2507.01504)  

**Abstract**: The collection and release of street-level recordings as Open Data play a vital role in advancing autonomous driving systems and AI research. However, these datasets pose significant privacy risks, particularly for pedestrians, due to the presence of Personally Identifiable Information (PII) that extends beyond biometric traits such as faces. In this paper, we present cRID, a novel cross-modal framework combining Large Vision-Language Models, Graph Attention Networks, and representation learning to detect textual describable clues of PII and enhance person re-identification (Re-ID). Our approach focuses on identifying and leveraging interpretable features, enabling the detection of semantically meaningful PII beyond low-level appearance cues. We conduct a systematic evaluation of PII presence in person image datasets. Our experiments show improved performance in practical cross-dataset Re-ID scenarios, notably from Market-1501 to CUHK03-np (detected), highlighting the framework's practical utility. Code is available at this https URL. 

---
# Pensieve Grader: An AI-Powered, Ready-to-Use Platform for Effortless Handwritten STEM Grading 

**Authors**: Yoonseok Yang, Minjune Kim, Marlon Rondinelli, Keren Shao  

**Link**: [PDF](https://arxiv.org/pdf/2507.01431)  

**Abstract**: Grading handwritten, open-ended responses remains a major bottleneck in large university STEM courses. We introduce Pensieve (this https URL), an AI-assisted grading platform that leverages large language models (LLMs) to transcribe and evaluate student work, providing instructors with rubric-aligned scores, transcriptions, and confidence ratings. Unlike prior tools that focus narrowly on specific tasks like transcription or rubric generation, Pensieve supports the entire grading pipeline-from scanned student submissions to final feedback-within a human-in-the-loop interface.
Pensieve has been deployed in real-world courses at over 20 institutions and has graded more than 300,000 student responses. We present system details and empirical results across four core STEM disciplines: Computer Science, Mathematics, Physics, and Chemistry. Our findings show that Pensieve reduces grading time by an average of 65%, while maintaining a 95.4% agreement rate with instructor-assigned grades for high-confidence predictions. 

---
# Automated Vehicles Should be Connected with Natural Language 

**Authors**: Xiangbo Gao, Keshu Wu, Hao Zhang, Kexin Tian, Yang Zhou, Zhengzhong Tu  

**Link**: [PDF](https://arxiv.org/pdf/2507.01059)  

**Abstract**: Multi-agent collaborative driving promises improvements in traffic safety and efficiency through collective perception and decision making. However, existing communication media -- including raw sensor data, neural network features, and perception results -- suffer limitations in bandwidth efficiency, information completeness, and agent interoperability. Moreover, traditional approaches have largely ignored decision-level fusion, neglecting critical dimensions of collaborative driving. In this paper we argue that addressing these challenges requires a transition from purely perception-oriented data exchanges to explicit intent and reasoning communication using natural language. Natural language balances semantic density and communication bandwidth, adapts flexibly to real-time conditions, and bridges heterogeneous agent platforms. By enabling the direct communication of intentions, rationales, and decisions, it transforms collaborative driving from reactive perception-data sharing into proactive coordination, advancing safety, efficiency, and transparency in intelligent transportation systems. 

---
# Text Detoxification: Data Efficiency, Semantic Preservation and Model Generalization 

**Authors**: Jing Yu, Yibo Zhao, Jiapeng Zhu, Wenming Shao, Bo Pang, Zhao Zhang, Xiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.01050)  

**Abstract**: The widespread dissemination of toxic content on social media poses a serious threat to both online environments and public discourse, highlighting the urgent need for detoxification methods that effectively remove toxicity while preserving the original semantics. However, existing approaches often struggle to simultaneously achieve strong detoxification performance, semantic preservation, and robustness to out-of-distribution data. Moreover, they typically rely on costly, manually annotated parallel corpora while showing poor data efficiency. To address these challenges, we propose a two-stage training framework that jointly optimizes for data efficiency, semantic preservation, and model generalization. We first perform supervised fine-tuning on a small set of high-quality, filtered parallel data to establish a strong initialization. Then, we leverage unlabeled toxic inputs and a custom-designed reward model to train the LLM using Group Relative Policy Optimization. Experimental results demonstrate that our method effectively mitigates the trade-offs faced by previous work, achieving state-of-the-art performance with improved generalization and significantly reduced dependence on annotated data. Our code is available at: this https URL 

---
# Cohort Retrieval using Dense Passage Retrieval 

**Authors**: Pranav Jadhav  

**Link**: [PDF](https://arxiv.org/pdf/2507.01049)  

**Abstract**: Patient cohort retrieval is a pivotal task in medical research and clinical practice, enabling the identification of specific patient groups from extensive electronic health records (EHRs). In this work, we address the challenge of cohort retrieval in the echocardiography domain by applying Dense Passage Retrieval (DPR), a prominent methodology in semantic search. We propose a systematic approach to transform an echocardiographic EHR dataset of unstructured nature into a Query-Passage dataset, framing the problem as a Cohort Retrieval task. Additionally, we design and implement evaluation metrics inspired by real-world clinical scenarios to rigorously test the models across diverse retrieval tasks. Furthermore, we present a custom-trained DPR embedding model that demonstrates superior performance compared to traditional and off-the-shelf SOTA this http URL our knowledge, this is the first work to apply DPR for patient cohort retrieval in the echocardiography domain, establishing a framework that can be adapted to other medical domains. 

---
# Can Argus Judge Them All? Comparing VLMs Across Domains 

**Authors**: Harsh Joshi, Gautam Siddharth Kashyap, Rafiq Ali, Ebad Shabbir, Niharika Jain, Sarthak Jain, Jiechao Gao, Usman Naseem  

**Link**: [PDF](https://arxiv.org/pdf/2507.01042)  

**Abstract**: Vision-Language Models (VLMs) are advancing multimodal AI, yet their performance consistency across tasks is underexamined. We benchmark CLIP, BLIP, and LXMERT across diverse datasets spanning retrieval, captioning, and reasoning. Our evaluation includes task accuracy, generation quality, efficiency, and a novel Cross-Dataset Consistency (CDC) metric. CLIP shows strongest generalization (CDC: 0.92), BLIP excels on curated data, and LXMERT leads in structured reasoning. These results expose trade-offs between generalization and specialization, informing industrial deployment of VLMs and guiding development toward robust, task-flexible architectures. 

---
# PathCoT: Chain-of-Thought Prompting for Zero-shot Pathology Visual Reasoning 

**Authors**: Junjie Zhou, Yingli Zuo, Shichang Feng, Peng Wan, Qi Zhu, Daoqiang Zhang, Wei Shao  

**Link**: [PDF](https://arxiv.org/pdf/2507.01029)  

**Abstract**: With the development of generative artificial intelligence and instruction tuning techniques, multimodal large language models (MLLMs) have made impressive progress on general reasoning tasks. Benefiting from the chain-of-thought (CoT) methodology, MLLMs can solve the visual reasoning problem step-by-step. However, existing MLLMs still face significant challenges when applied to pathology visual reasoning tasks: (1) LLMs often underperforms because they lack domain-specific information, which can lead to model hallucinations. (2) The additional reasoning steps in CoT may introduce errors, leading to the divergence of answers. To address these limitations, we propose PathCoT, a novel zero-shot CoT prompting method which integrates the pathology expert-knowledge into the reasoning process of MLLMs and incorporates self-evaluation to mitigate divergence of answers. Specifically, PathCoT guides the MLLM with prior knowledge to perform as pathology experts, and provides comprehensive analysis of the image with their domain-specific knowledge. By incorporating the experts' knowledge, PathCoT can obtain the answers with CoT reasoning. Furthermore, PathCoT incorporates a self-evaluation step that assesses both the results generated directly by MLLMs and those derived through CoT, finally determining the reliable answer. The experimental results on the PathMMU dataset demonstrate the effectiveness of our method on pathology visual understanding and reasoning. 

---
# Scalable Offline ASR for Command-Style Dictation in Courtrooms 

**Authors**: Kumarmanas Nethil, Vaibhav Mishra, Kriti Anandan, Kavya Manohar  

**Link**: [PDF](https://arxiv.org/pdf/2507.01021)  

**Abstract**: We propose an open-source framework for Command-style dictation that addresses the gap between resource-intensive Online systems and high-latency Batch processing. Our approach uses Voice Activity Detection (VAD) to segment audio and transcribes these segments in parallel using Whisper models, enabling efficient multiplexing across audios. Unlike proprietary systems like SuperWhisper, this framework is also compatible with most ASR architectures, including widely used CTC-based models. Our multiplexing technique maximizes compute utilization in real-world settings, as demonstrated by its deployment in around 15% of India's courtrooms. Evaluations on live data show consistent latency reduction as user concurrency increases, compared to sequential batch processing. The live demonstration will showcase our open-sourced implementation and allow attendees to interact with it in real-time. 

---
