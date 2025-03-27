# Mobile-MMLU: A Mobile Intelligence Language Understanding Benchmark 

**Authors**: Sondos Mahmoud Bsharat, Mukul Ranjan, Aidar Myrzakhan, Jiacheng Liu, Bowei Guo, Shengkun Tang, Zhuang Liu, Yuanzhi Li, Zhiqiang Shen  

**Link**: [PDF](https://arxiv.org/pdf/2503.20786)  

**Abstract**: Rapid advancements in large language models (LLMs) have increased interest in deploying them on mobile devices for on-device AI applications. Mobile users interact differently with LLMs compared to desktop users, creating unique expectations and data biases. Current benchmark datasets primarily target at server and desktop environments, and there is a notable lack of extensive datasets specifically designed for mobile contexts. Additionally, mobile devices face strict limitations in storage and computing resources, constraining model size and capabilities, thus requiring optimized efficiency and prioritized knowledge. To address these challenges, we introduce Mobile-MMLU, a large-scale benchmark dataset tailored for mobile intelligence. It consists of 16,186 questions across 80 mobile-related fields, designed to evaluate LLM performance in realistic mobile scenarios. A challenging subset, Mobile-MMLU-Pro, provides advanced evaluation similar in size to MMLU-Pro but significantly more difficult than our standard full set. Both benchmarks use multiple-choice, order-invariant questions focused on practical mobile interactions, such as recipe suggestions, travel planning, and essential daily tasks. The dataset emphasizes critical mobile-specific metrics like inference latency, energy consumption, memory usage, and response quality, offering comprehensive insights into model performance under mobile constraints. Moreover, it prioritizes privacy and adaptability, assessing models' ability to perform on-device processing, maintain user privacy, and adapt to personalized usage patterns. Mobile-MMLU family offers a standardized framework for developing and comparing mobile-optimized LLMs, enabling advancements in productivity and decision-making within mobile computing environments. Our code and data are available at: this https URL. 

---
# MCTS-RAG: Enhancing Retrieval-Augmented Generation with Monte Carlo Tree Search 

**Authors**: Yunhai Hu, Yilun Zhao, Chen Zhao, Arman Cohan  

**Link**: [PDF](https://arxiv.org/pdf/2503.20757)  

**Abstract**: We introduce MCTS-RAG, a novel approach that enhances the reasoning capabilities of small language models on knowledge-intensive tasks by leveraging retrieval-augmented generation (RAG) to provide relevant context and Monte Carlo Tree Search (MCTS) to refine reasoning paths. MCTS-RAG dynamically integrates retrieval and reasoning through an iterative decision-making process. Unlike standard RAG methods, which typically retrieve information independently from reasoning and thus integrate knowledge suboptimally, or conventional MCTS reasoning, which depends solely on internal model knowledge without external facts, MCTS-RAG combines structured reasoning with adaptive retrieval. This integrated approach enhances decision-making, reduces hallucinations, and ensures improved factual accuracy and response consistency. The experimental results on multiple reasoning and knowledge-intensive datasets datasets (i.e., ComplexWebQA, GPQA, and FoolMeTwice) show that our method enables small-scale LMs to achieve performance comparable to frontier LLMs like GPT-4o by effectively scaling inference-time compute, setting a new standard for reasoning in small-scale models. 

---
# ADS-Edit: A Multimodal Knowledge Editing Dataset for Autonomous Driving Systems 

**Authors**: Chenxi Wang, Jizhan Fang, Xiang Chen, Bozhong Tian, Ziwen Xu, Huajun Chen, Ningyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.20756)  

**Abstract**: Recent advancements in Large Multimodal Models (LMMs) have shown promise in Autonomous Driving Systems (ADS). However, their direct application to ADS is hindered by challenges such as misunderstanding of traffic knowledge, complex road conditions, and diverse states of vehicle. To address these challenges, we propose the use of Knowledge Editing, which enables targeted modifications to a model's behavior without the need for full retraining. Meanwhile, we introduce ADS-Edit, a multimodal knowledge editing dataset specifically designed for ADS, which includes various real-world scenarios, multiple data types, and comprehensive evaluation metrics. We conduct comprehensive experiments and derive several interesting conclusions. We hope that our work will contribute to the further advancement of knowledge editing applications in the field of autonomous driving. Code and data are available in this https URL. 

---
# Beyond Believability: Accurate Human Behavior Simulation with Fine-Tuned LLMs 

**Authors**: Yuxuan Lu, Jing Huang, Yan Han, Bennet Bei, Yaochen Xie, Dakuo Wang, Jessie Wang, Qi He  

**Link**: [PDF](https://arxiv.org/pdf/2503.20749)  

**Abstract**: Recent research shows that LLMs can simulate ``believable'' human behaviors to power LLM agents via prompt-only methods. In this work, we focus on evaluating and improving LLM's objective ``accuracy'' rather than the subjective ``believability'' in the web action generation task, leveraging a large-scale, real-world dataset collected from online shopping human actions. We present the first comprehensive quantitative evaluation of state-of-the-art LLMs (e.g., DeepSeek-R1, Llama, and Claude) on the task of web action generation. Our results show that fine-tuning LLMs on real-world behavioral data substantially improves their ability to generate actions compared to prompt-only methods. Furthermore, incorporating synthesized reasoning traces into model training leads to additional performance gains, demonstrating the value of explicit rationale in behavior modeling. This work establishes a new benchmark for evaluating LLMs in behavior simulation and offers actionable insights into how real-world action data and reasoning augmentation can enhance the fidelity of LLM agents. 

---
# Ontology-based Semantic Similarity Measures for Clustering Medical Concepts in Drug Safety 

**Authors**: Jeffery L Painter, François Haguinet, Gregory E Powell, Andrew Bate  

**Link**: [PDF](https://arxiv.org/pdf/2503.20737)  

**Abstract**: Semantic similarity measures (SSMs) are widely used in biomedical research but remain underutilized in pharmacovigilance. This study evaluates six ontology-based SSMs for clustering MedDRA Preferred Terms (PTs) in drug safety data. Using the Unified Medical Language System (UMLS), we assess each method's ability to group PTs around medically meaningful centroids. A high-throughput framework was developed with a Java API and Python and R interfaces support large-scale similarity computations. Results show that while path-based methods perform moderately with F1 scores of 0.36 for WUPALMER and 0.28 for LCH, intrinsic information content (IC)-based measures, especially INTRINSIC-LIN and SOKAL, consistently yield better clustering accuracy (F1 score of 0.403). Validated against expert review and standard MedDRA queries (SMQs), our findings highlight the promise of IC-based SSMs in enhancing pharmacovigilance workflows by improving early signal detection and reducing manual review. 

---
# From Annotation to Adaptation: Metrics, Synthetic Data, and Aspect Extraction for Aspect-Based Sentiment Analysis with Large Language Models 

**Authors**: Nikita Neveditsin, Pawan Lingras, Vijay Mago  

**Link**: [PDF](https://arxiv.org/pdf/2503.20715)  

**Abstract**: This study examines the performance of Large Language Models (LLMs) in Aspect-Based Sentiment Analysis (ABSA), with a focus on implicit aspect extraction in a novel domain. Using a synthetic sports feedback dataset, we evaluate open-weight LLMs' ability to extract aspect-polarity pairs and propose a metric to facilitate the evaluation of aspect extraction with generative models. Our findings highlight both the potential and limitations of LLMs in the ABSA task. 

---
# UniEDU: A Unified Language and Vision Assistant for Education Applications 

**Authors**: Zhendong Chu, Jian Xie, Shen Wang, Zichao Wang, Qingsong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2503.20701)  

**Abstract**: Education materials for K-12 students often consist of multiple modalities, such as text and images, posing challenges for models to fully understand nuanced information in these materials. In this paper, we propose a unified language and vision assistant UniEDU designed for various educational applications, including knowledge recommendation, knowledge tracing, time cost prediction, and user answer prediction, all within a single model. Unlike conventional task-specific models, UniEDU offers a unified solution that excels across multiple educational tasks while maintaining strong generalization capabilities. Its adaptability makes it well-suited for real-world deployment in diverse learning environments. Furthermore, UniEDU is optimized for industry-scale deployment by significantly reducing computational overhead-achieving approximately a 300\% increase in efficiency-while maintaining competitive performance with minimal degradation compared to fully fine-tuned models. This work represents a significant step toward creating versatile AI systems tailored to the evolving demands of education. 

---
# TN-Eval: Rubric and Evaluation Protocols for Measuring the Quality of Behavioral Therapy Notes 

**Authors**: Raj Sanjay Shah, Lei Xu, Qianchu Liu, Jon Burnsky, Drew Bertagnolli, Chaitanya Shivade  

**Link**: [PDF](https://arxiv.org/pdf/2503.20648)  

**Abstract**: Behavioral therapy notes are important for both legal compliance and patient care. Unlike progress notes in physical health, quality standards for behavioral therapy notes remain underdeveloped. To address this gap, we collaborated with licensed therapists to design a comprehensive rubric for evaluating therapy notes across key dimensions: completeness, conciseness, and faithfulness. Further, we extend a public dataset of behavioral health conversations with therapist-written notes and LLM-generated notes, and apply our evaluation framework to measure their quality. We find that: (1) A rubric-based manual evaluation protocol offers more reliable and interpretable results than traditional Likert-scale annotations. (2) LLMs can mimic human evaluators in assessing completeness and conciseness but struggle with faithfulness. (3) Therapist-written notes often lack completeness and conciseness, while LLM-generated notes contain hallucination. Surprisingly, in a blind test, therapists prefer and judge LLM-generated notes to be superior to therapist-written notes. 

---
# Unlocking Efficient Long-to-Short LLM Reasoning with Model Merging 

**Authors**: Han Wu, Yuxuan Yao, Shuqi Liu, Zehua Liu, Xiaojin Fu, Xiongwei Han, Xing Li, Hui-Ling Zhen, Tao Zhong, Mingxuan Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2503.20641)  

**Abstract**: The transition from System 1 to System 2 reasoning in large language models (LLMs) has marked significant advancements in handling complex tasks through deliberate, iterative thinking. However, this progress often comes at the cost of efficiency, as models tend to overthink, generating redundant reasoning steps without proportional improvements in output quality. Long-to-Short (L2S) reasoning has emerged as a promising solution to this challenge, aiming to balance reasoning depth with practical efficiency. While existing approaches, such as supervised fine-tuning (SFT), reinforcement learning (RL), and prompt engineering, have shown potential, they are either computationally expensive or unstable. Model merging, on the other hand, offers a cost-effective and robust alternative by integrating the quick-thinking capabilities of System 1 models with the methodical reasoning of System 2 models. In this work, we present a comprehensive empirical study on model merging for L2S reasoning, exploring diverse methodologies, including task-vector-based, SVD-based, and activation-informed merging. Our experiments reveal that model merging can reduce average response length by up to 55% while preserving or even improving baseline performance. We also identify a strong correlation between model scale and merging efficacy with extensive evaluations on 1.5B/7B/14B/32B models. Furthermore, we investigate the merged model's ability to self-critique and self-correct, as well as its adaptive response length based on task complexity. Our findings highlight model merging as a highly efficient and effective paradigm for L2S reasoning, offering a practical solution to the overthinking problem while maintaining the robustness of System 2 reasoning. This work can be found on Github this https URL. 

---
# PVLens: Enhancing Pharmacovigilance Through Automated Label Extraction 

**Authors**: Jeffery L Painter, Gregory E Powell, Andrew Bate  

**Link**: [PDF](https://arxiv.org/pdf/2503.20639)  

**Abstract**: Reliable drug safety reference databases are essential for pharmacovigilance, yet existing resources like SIDER are outdated and static. We introduce PVLens, an automated system that extracts labeled safety information from FDA Structured Product Labels (SPLs) and maps terms to MedDRA. PVLens integrates automation with expert oversight through a web-based review tool. In validation against 97 drug labels, PVLens achieved an F1 score of 0.882, with high recall (0.983) and moderate precision (0.799). By offering a scalable, more accurate and continuously updated alternative to SIDER, PVLens enhances real-time pharamcovigilance with improved accuracy and contemporaneous insights. 

---
# Collaborative Storytelling and LLM: A Linguistic Analysis of Automatically-Generated Role-Playing Game Sessions 

**Authors**: Alessandro Maisto  

**Link**: [PDF](https://arxiv.org/pdf/2503.20623)  

**Abstract**: Role-playing games (RPG) are games in which players interact with one another to create narratives. The role of players in the RPG is largely based on the interaction between players and their characters. This emerging form of shared narrative, primarily oral, is receiving increasing attention. In particular, many authors investigated the use of an LLM as an actor in the game. In this paper, we aim to discover to what extent the language of Large Language Models (LLMs) exhibit oral or written features when asked to generate an RPG session without human interference. We will conduct a linguistic analysis of the lexical and syntactic features of the generated texts and compare the results with analyses of conversations, transcripts of human RPG sessions, and books. We found that LLMs exhibit a pattern that is distinct from all other text categories, including oral conversations, human RPG sessions and books. Our analysis has shown how training influences the way LLMs express themselves and provides important indications of the narrative capabilities of these tools. 

---
# Synthetic Data Augmentation for Cross-domain Implicit Discourse Relation Recognition 

**Authors**: Frances Yung, Varsha Suresh, Zaynab Reza, Mansoor Ahmad, Vera Demberg  

**Link**: [PDF](https://arxiv.org/pdf/2503.20588)  

**Abstract**: Implicit discourse relation recognition (IDRR) -- the task of identifying the implicit coherence relation between two text spans -- requires deep semantic understanding. Recent studies have shown that zero- or few-shot approaches significantly lag behind supervised models, but LLMs may be useful for synthetic data augmentation, where LLMs generate a second argument following a specified coherence relation. We applied this approach in a cross-domain setting, generating discourse continuations using unlabelled target-domain data to adapt a base model which was trained on source-domain labelled data. Evaluations conducted on a large-scale test set revealed that different variations of the approach did not result in any significant improvements. We conclude that LLMs often fail to generate useful samples for IDRR, and emphasize the importance of considering both statistical significance and comparability when evaluating IDRR models. 

---
# Low-resource Information Extraction with the European Clinical Case Corpus 

**Authors**: Soumitra Ghosh, Begona Altuna, Saeed Farzi, Pietro Ferrazzi, Alberto Lavelli, Giulia Mezzanotte, Manuela Speranza, Bernardo Magnini  

**Link**: [PDF](https://arxiv.org/pdf/2503.20568)  

**Abstract**: We present E3C-3.0, a multilingual dataset in the medical domain, comprising clinical cases annotated with diseases and test-result relations. The dataset includes both native texts in five languages (English, French, Italian, Spanish and Basque) and texts translated and projected from the English source into five target languages (Greek, Italian, Polish, Slovak, and Slovenian). A semi-automatic approach has been implemented, including automatic annotation projection based on Large Language Models (LLMs) and human revision. We present several experiments showing that current state-of-the-art LLMs can benefit from being fine-tuned on the E3C-3.0 dataset. We also show that transfer learning in different languages is very effective, mitigating the scarcity of data. Finally, we compare performance both on native data and on projected data. We release the data at this https URL . 

---
# A Retrieval-Based Approach to Medical Procedure Matching in Romanian 

**Authors**: Andrei Niculae, Adrian Cosma, Emilian Radoi  

**Link**: [PDF](https://arxiv.org/pdf/2503.20556)  

**Abstract**: Accurately mapping medical procedure names from healthcare providers to standardized terminology used by insurance companies is a crucial yet complex task. Inconsistencies in naming conventions lead to missclasified procedures, causing administrative inefficiencies and insurance claim problems in private healthcare settings. Many companies still use human resources for manual mapping, while there is a clear opportunity for automation. This paper proposes a retrieval-based architecture leveraging sentence embeddings for medical name matching in the Romanian healthcare system. This challenge is significantly more difficult in underrepresented languages such as Romanian, where existing pretrained language models lack domain-specific adaptation to medical text. We evaluate multiple embedding models, including Romanian, multilingual, and medical-domain-specific representations, to identify the most effective solution for this task. Our findings contribute to the broader field of medical NLP for low-resource languages such as Romanian. 

---
# Accelerate Parallelizable Reasoning via Parallel Decoding within One Sequence 

**Authors**: Yijiong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2503.20533)  

**Abstract**: Recent advances in reasoning models have demonstrated significant improvements in accuracy, particularly for complex tasks such as mathematical reasoning, by employing detailed and comprehensive reasoning processes. However, generating these lengthy reasoning sequences is computationally expensive and time-consuming. To address this inefficiency, we leverage the inherent parallelizability of certain tasks to accelerate the reasoning process. Specifically, when multiple parallel reasoning branches exist, we decode multiple tokens per step using a specialized attention mask, processing them within a single sequence. Experimental results show that our method achieves over 100% speedup in decoding time while basically maintaining accuracy. 

---
# StableToolBench-MirrorAPI: Modeling Tool Environments as Mirrors of 7,000+ Real-World APIs 

**Authors**: Zhicheng Guo, Sijie Cheng, Yuchen Niu, Hao Wang, Sicheng Zhou, Wenbing Huang, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.20527)  

**Abstract**: The rapid advancement of large language models (LLMs) has spurred significant interest in tool learning, where LLMs are augmented with external tools to tackle complex tasks. However, existing tool environments face challenges in balancing stability, scalability, and realness, particularly for benchmarking purposes. To address this problem, we propose MirrorAPI, a novel framework that trains specialized LLMs to accurately simulate real API responses, effectively acting as "mirrors" to tool environments. Using a comprehensive dataset of request-response pairs from 7,000+ APIs, we employ supervised fine-tuning and chain-of-thought reasoning to enhance simulation fidelity. MirrorAPI achieves superior accuracy and stability compared to state-of-the-art methods, as demonstrated by its performance on the newly constructed MirrorAPI-Bench and its integration into StableToolBench. 

---
# Explainable ICD Coding via Entity Linking 

**Authors**: Leonor Barreiros, Isabel Coutinho, Gonçalo M. Correia, Bruno Martins  

**Link**: [PDF](https://arxiv.org/pdf/2503.20508)  

**Abstract**: Clinical coding is a critical task in healthcare, although traditional methods for automating clinical coding may not provide sufficient explicit evidence for coders in production environments. This evidence is crucial, as medical coders have to make sure there exists at least one explicit passage in the input health record that justifies the attribution of a code. We therefore propose to reframe the task as an entity linking problem, in which each document is annotated with its set of codes and respective textual evidence, enabling better human-machine collaboration. By leveraging parameter-efficient fine-tuning of Large Language Models (LLMs), together with constrained decoding, we introduce three approaches to solve this problem that prove effective at disambiguating clinical mentions and that perform well in few-shot scenarios. 

---
# Enhancing Depression Detection via Question-wise Modality Fusion 

**Authors**: Aishik Mandal, Dana Atzil-Slonim, Thamar Solorio, Iryna Gurevych  

**Link**: [PDF](https://arxiv.org/pdf/2503.20496)  

**Abstract**: Depression is a highly prevalent and disabling condition that incurs substantial personal and societal costs. Current depression diagnosis involves determining the depression severity of a person through self-reported questionnaires or interviews conducted by clinicians. This often leads to delayed treatment and involves substantial human resources. Thus, several works try to automate the process using multimodal data. However, they usually overlook the following: i) The variable contribution of each modality for each question in the questionnaire and ii) Using ordinal classification for the task. This results in sub-optimal fusion and training methods. In this work, we propose a novel Question-wise Modality Fusion (QuestMF) framework trained with a novel Imbalanced Ordinal Log-Loss (ImbOLL) function to tackle these issues. The performance of our framework is comparable to the current state-of-the-art models on the E-DAIC dataset and enhances interpretability by predicting scores for each question. This will help clinicians identify an individual's symptoms, allowing them to customise their interventions accordingly. We also make the code for the QuestMF framework publicly available. 

---
# TempTest: Local Normalization Distortion and the Detection of Machine-generated Text 

**Authors**: Tom Kempton, Stuart Burrell, Connor Cheverall  

**Link**: [PDF](https://arxiv.org/pdf/2503.20421)  

**Abstract**: Existing methods for the zero-shot detection of machine-generated text are dominated by three statistical quantities: log-likelihood, log-rank, and entropy. As language models mimic the distribution of human text ever closer, this will limit our ability to build effective detection algorithms. To combat this, we introduce a method for detecting machine-generated text that is entirely agnostic of the generating language model. This is achieved by targeting a defect in the way that decoding strategies, such as temperature or top-k sampling, normalize conditional probability measures. This method can be rigorously theoretically justified, is easily explainable, and is conceptually distinct from existing methods for detecting machine-generated text. We evaluate our detector in the white and black box settings across various language models, datasets, and passage lengths. We also study the effect of paraphrasing attacks on our detector and the extent to which it is biased against non-native speakers. In each of these settings, the performance of our test is at least comparable to that of other state-of-the-art text detectors, and in some cases, we strongly outperform these baselines. 

---
# CFunModel: A "Funny" Language Model Capable of Chinese Humor Generation and Processing 

**Authors**: Zhenghan Yu, Xinyu Hu, Xiaojun Wan  

**Link**: [PDF](https://arxiv.org/pdf/2503.20417)  

**Abstract**: Humor plays a significant role in daily language communication. With the rapid development of large language models (LLMs), natural language processing has made significant strides in understanding and generating various genres of texts. However, most LLMs exhibit poor performance in generating and processing Chinese humor. In this study, we introduce a comprehensive Chinese humor-related dataset, the Chinese Fun Set (CFunSet). This dataset aggregates existing Chinese humor datasets and includes over 20,000 jokes collected from Tieba-JokeBar, a Chinese online platform known for joke sharing. The resulting corpus comprises more than 160,000 entries. Leveraging CFunSet, we developed the Chinese Fun Model (CFunModel), the first large language model designed to handle various Chinese humor-related tasks including Crosstalk Response Selection, Humor Recognition, Joke Generation, etc. Experimental results demonstrate that CFunModel outperforms popular large language models in these tasks. Our CFunSet is available at this https URL and CFunModel is available at this https URL. A demostration video of our work is available at this https URL. 

---
# Iterative Prompting with Persuasion Skills in Jailbreaking Large Language Models 

**Authors**: Shih-Wen Ke, Guan-Yu Lai, Guo-Lin Fang, Hsi-Yuan Kao  

**Link**: [PDF](https://arxiv.org/pdf/2503.20320)  

**Abstract**: Large language models (LLMs) are designed to align with human values in their responses. This study exploits LLMs with an iterative prompting technique where each prompt is systematically modified and refined across multiple iterations to enhance its effectiveness in jailbreaking attacks progressively. This technique involves analyzing the response patterns of LLMs, including GPT-3.5, GPT-4, LLaMa2, Vicuna, and ChatGLM, allowing us to adjust and optimize prompts to evade the LLMs' ethical and security constraints. Persuasion strategies enhance prompt effectiveness while maintaining consistency with malicious intent. Our results show that the attack success rates (ASR) increase as the attacking prompts become more refined with the highest ASR of 90% for GPT4 and ChatGLM and the lowest ASR of 68% for LLaMa2. Our technique outperforms baseline techniques (PAIR and PAP) in ASR and shows comparable performance with GCG and ArtPrompt. 

---
# A Multilingual, Culture-First Approach to Addressing Misgendering in LLM Applications 

**Authors**: Sunayana Sitaram, Adrian de Wynter, Isobel McCrum, Qilong Gu, Si-Qing Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.20302)  

**Abstract**: Misgendering is the act of referring to someone by a gender that does not match their chosen identity. It marginalizes and undermines a person's sense of self, causing significant harm. English-based approaches have clear-cut approaches to avoiding misgendering, such as the use of the pronoun ``they''. However, other languages pose unique challenges due to both grammatical and cultural constructs. In this work we develop methodologies to assess and mitigate misgendering across 42 languages and dialects using a participatory-design approach to design effective and appropriate guardrails across all languages. We test these guardrails in a standard large language model-based application (meeting transcript summarization), where both the data generation and the annotation steps followed a human-in-the-loop approach. We find that the proposed guardrails are very effective in reducing misgendering rates across all languages in the summaries generated, and without incurring loss of quality. Our human-in-the-loop approach demonstrates a method to feasibly scale inclusive and responsible AI-based solutions across multiple languages and cultures. 

---
# sudo rm -rf agentic_security 

**Authors**: Sejin Lee, Jian Kim, Haon Park, Ashkan Yousefpour, Sangyoon Yu, Min Song  

**Link**: [PDF](https://arxiv.org/pdf/2503.20279)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed as computer-use agents, autonomously performing tasks within real desktop or web environments. While this evolution greatly expands practical use cases for humans, it also creates serious security exposures. We present SUDO (Screen-based Universal Detox2Tox Offense), a novel attack framework that systematically bypasses refusal trained safeguards in commercial computer-use agents, such as Claude Computer Use. The core mechanism, Detox2Tox, transforms harmful requests (that agents initially reject) into seemingly benign requests via detoxification, secures detailed instructions from advanced vision language models (VLMs), and then reintroduces malicious content via toxification just before execution. Unlike conventional jailbreaks, SUDO iteratively refines its attacks based on a built-in refusal feedback, making it increasingly effective against robust policy filters. In extensive tests spanning 50 real-world tasks and multiple state-of-the-art VLMs, SUDO achieves a stark attack success rate of 24% (with no refinement), and up to 41% (by its iterative refinement) in Claude Computer Use. By revealing these vulnerabilities and demonstrating the ease with which they can be exploited in real-world computing environments, this paper highlights an immediate need for robust, context-aware safeguards. WARNING: This paper includes harmful or offensive model outputs. 

---
# Advancements in Natural Language Processing: Exploring Transformer-Based Architectures for Text Understanding 

**Authors**: Tianhao Wu, Yu Wang, Ngoc Quach  

**Link**: [PDF](https://arxiv.org/pdf/2503.20227)  

**Abstract**: Natural Language Processing (NLP) has witnessed a transformative leap with the advent of transformer-based architectures, which have significantly enhanced the ability of machines to understand and generate human-like text. This paper explores the advancements in transformer models, such as BERT and GPT, focusing on their superior performance in text understanding tasks compared to traditional methods like recurrent neural networks (RNNs). By analyzing statistical properties through visual representations-including probability density functions of text length distributions and feature space classifications-the study highlights the models' proficiency in handling long-range dependencies, adapting to conditional shifts, and extracting features for classification, even with overlapping classes. Drawing on recent 2024 research, including enhancements in multi-hop knowledge graph reasoning and context-aware chat interactions, the paper outlines a methodology involving data preparation, model selection, pretraining, fine-tuning, and evaluation. The results demonstrate state-of-the-art performance on benchmarks like GLUE and SQuAD, with F1 scores exceeding 90%, though challenges such as high computational costs persist. This work underscores the pivotal role of transformers in modern NLP and suggests future directions, including efficiency optimization and multimodal integration, to further advance language-based AI systems. 

---
# Qwen2.5-Omni Technical Report 

**Authors**: Jin Xu, Zhifang Guo, Jinzheng He, Hangrui Hu, Ting He, Shuai Bai, Keqin Chen, Jialin Wang, Yang Fan, Kai Dang, Bin Zhang, Xiong Wang, Yunfei Chu, Junyang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2503.20215)  

**Abstract**: In this report, we present Qwen2.5-Omni, an end-to-end multimodal model designed to perceive diverse modalities, including text, images, audio, and video, while simultaneously generating text and natural speech responses in a streaming manner. To enable the streaming of multimodal information inputs, both audio and visual encoders utilize a block-wise processing approach. To synchronize the timestamps of video inputs with audio, we organize the audio and video sequentially in an interleaved manner and propose a novel position embedding approach, named TMRoPE(Time-aligned Multimodal RoPE). To concurrently generate text and speech while avoiding interference between the two modalities, we propose \textbf{Thinker-Talker} architecture. In this framework, Thinker functions as a large language model tasked with text generation, while Talker is a dual-track autoregressive model that directly utilizes the hidden representations from the Thinker to produce audio tokens as output. Both the Thinker and Talker models are designed to be trained and inferred in an end-to-end manner. For decoding audio tokens in a streaming manner, we introduce a sliding-window DiT that restricts the receptive field, aiming to reduce the initial package delay. Qwen2.5-Omni is comparable with the similarly sized Qwen2.5-VL and outperforms Qwen2-Audio. Furthermore, Qwen2.5-Omni achieves state-of-the-art performance on multimodal benchmarks like Omni-Bench. Notably, Qwen2.5-Omni's performance in end-to-end speech instruction following is comparable to its capabilities with text inputs, as evidenced by benchmarks such as MMLU and GSM8K. As for speech generation, Qwen2.5-Omni's streaming Talker outperforms most existing streaming and non-streaming alternatives in robustness and naturalness. 

---
# Dolphin: A Large-Scale Automatic Speech Recognition Model for Eastern Languages 

**Authors**: Yangyang Meng, Jinpeng Li, Guodong Lin, Yu Pu, Guanbo Wang, Hu Du, Zhiming Shao, Yukai Huang, Ke Li, Wei-Qiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.20212)  

**Abstract**: This report introduces Dolphin, a large-scale multilingual automatic speech recognition (ASR) model that extends the Whisper architecture to support a wider range of languages. Our approach integrates in-house proprietary and open-source datasets to refine and optimize Dolphin's performance. The model is specifically designed to achieve notable recognition accuracy for 40 Eastern languages across East Asia, South Asia, Southeast Asia, and the Middle East, while also supporting 22 Chinese dialects. Experimental evaluations show that Dolphin significantly outperforms current state-of-the-art open-source models across various languages. To promote reproducibility and community-driven innovation, we are making our trained models and inference source code publicly available. 

---
# SARGes: Semantically Aligned Reliable Gesture Generation via Intent Chain 

**Authors**: Nan Gao, Yihua Bao, Dongdong Weng, Jiayi Zhao, Jia Li, Yan Zhou, Pengfei Wan, Di Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.20202)  

**Abstract**: Co-speech gesture generation enhances human-computer interaction realism through speech-synchronized gesture synthesis. However, generating semantically meaningful gestures remains a challenging problem. We propose SARGes, a novel framework that leverages large language models (LLMs) to parse speech content and generate reliable semantic gesture labels, which subsequently guide the synthesis of meaningful co-speech this http URL, we constructed a comprehensive co-speech gesture ethogram and developed an LLM-based intent chain reasoning mechanism that systematically parses and decomposes gesture semantics into structured inference steps following ethogram criteria, effectively guiding LLMs to generate context-aware gesture labels. Subsequently, we constructed an intent chain-annotated text-to-gesture label dataset and trained a lightweight gesture label generation model, which then guides the generation of credible and semantically coherent co-speech gestures. Experimental results demonstrate that SARGes achieves highly semantically-aligned gesture labeling (50.2% accuracy) with efficient single-pass inference (0.4 seconds). The proposed method provides an interpretable intent reasoning pathway for semantic gesture synthesis. 

---
# GAPO: Learning Preferential Prompt through Generative Adversarial Policy Optimization 

**Authors**: Zhouhong Gu, Xingzhou Chen, Xiaoran Shi, Tao Wang, Suhang Zheng, Tianyu Li, Hongwei Feng, Yanghua Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2503.20194)  

**Abstract**: Recent advances in large language models have highlighted the critical need for precise control over model outputs through predefined constraints. While existing methods attempt to achieve this through either direct instruction-response synthesis or preferential response optimization, they often struggle with constraint understanding and adaptation. This limitation becomes particularly evident when handling fine-grained constraints, leading to either hallucination or brittle performance. We introduce Generative Adversarial Policy Optimization (GAPO), a novel framework that combines GAN-based training dynamics with an encoder-only reward model to progressively learn and adapt to increasingly complex constraints. GAPO leverages adversarial training to automatically generate training samples of varying difficulty while utilizing the encoder-only architecture to better capture prompt-response relationships. Extensive experiments demonstrate GAPO's superior performance across multiple benchmarks, particularly in scenarios requiring fine-grained constraint handling, where it significantly outperforms existing methods like PPO, DPO, and KTO. Our results suggest that GAPO's unique approach to preferential prompt learning offers a more robust and effective solution for controlling LLM outputs. Code is avaliable in this https URL. 

---
# Leveraging Implicit Sentiments: Enhancing Reliability and Validity in Psychological Trait Evaluation of LLMs 

**Authors**: Huanhuan Ma, Haisong Gong, Xiaoyuan Yi, Xing Xie, Dongkuan Xu  

**Link**: [PDF](https://arxiv.org/pdf/2503.20182)  

**Abstract**: Recent advancements in Large Language Models (LLMs) have led to their increasing integration into human life. With the transition from mere tools to human-like assistants, understanding their psychological aspects-such as emotional tendencies and personalities-becomes essential for ensuring their trustworthiness. However, current psychological evaluations of LLMs, often based on human psychological assessments like the BFI, face significant limitations. The results from these approaches often lack reliability and have limited validity when predicting LLM behavior in real-world scenarios. In this work, we introduce a novel evaluation instrument specifically designed for LLMs, called Core Sentiment Inventory (CSI). CSI is a bilingual tool, covering both English and Chinese, that implicitly evaluates models' sentiment tendencies, providing an insightful psychological portrait of LLM across three dimensions: optimism, pessimism, and neutrality. Through extensive experiments, we demonstrate that: 1) CSI effectively captures nuanced emotional patterns, revealing significant variation in LLMs across languages and contexts; 2) Compared to current approaches, CSI significantly improves reliability, yielding more consistent results; and 3) The correlation between CSI scores and the sentiment of LLM's real-world outputs exceeds 0.85, demonstrating its strong validity in predicting LLM behavior. We make CSI public available via: this https URL. 

---
# ProtoBERT-LoRA: Parameter-Efficient Prototypical Finetuning for Immunotherapy Study Identification 

**Authors**: Shijia Zhang, Xiyu Ding, Kai Ding, Jacob Zhang, Kevin Galinsky, Mengrui Wang, Ryan P. Mayers, Zheyu Wang, Hadi Kharrazi  

**Link**: [PDF](https://arxiv.org/pdf/2503.20179)  

**Abstract**: Identifying immune checkpoint inhibitor (ICI) studies in genomic repositories like Gene Expression Omnibus (GEO) is vital for cancer research yet remains challenging due to semantic ambiguity, extreme class imbalance, and limited labeled data in low-resource settings. We present ProtoBERT-LoRA, a hybrid framework that combines PubMedBERT with prototypical networks and Low-Rank Adaptation (LoRA) for efficient fine-tuning. The model enforces class-separable embeddings via episodic prototype training while preserving biomedical domain knowledge. Our dataset was divided as: Training (20 positive, 20 negative), Prototype Set (10 positive, 10 negative), Validation (20 positive, 200 negative), and Test (71 positive, 765 negative). Evaluated on test dataset, ProtoBERT-LoRA achieved F1-score of 0.624 (precision: 0.481, recall: 0.887), outperforming the rule-based system, machine learning baselines and finetuned PubMedBERT. Application to 44,287 unlabeled studies reduced manual review efforts by 82%. Ablation studies confirmed that combining prototypes with LoRA improved performance by 29% over stand-alone LoRA. 

---
# Efficient Model Development through Fine-tuning Transfer 

**Authors**: Pin-Jie Lin, Rishab Balasubramanian, Fengyuan Liu, Nikhil Kandpal, Tu Vu  

**Link**: [PDF](https://arxiv.org/pdf/2503.20110)  

**Abstract**: Modern LLMs struggle with efficient updates, as each new pretrained model version requires repeating expensive alignment processes. This challenge also applies to domain- or language-specific models, where fine-tuning on specialized data must be redone for every new base model release. In this paper, we explore the transfer of fine-tuning updates between model versions. Specifically, we derive the diff vector from one source model version, which represents the weight changes from fine-tuning, and apply it to the base model of a different target version. Through empirical evaluations on various open-weight model versions, we show that transferring diff vectors can significantly improve the target base model, often achieving performance comparable to its fine-tuned counterpart. For example, reusing the fine-tuning updates from Llama 3.0 8B leads to an absolute accuracy improvement of 10.7% on GPQA over the base Llama 3.1 8B without additional training, surpassing Llama 3.1 8B Instruct. In a multilingual model development setting, we show that this approach can significantly increase performance on target-language tasks without retraining, achieving an absolute improvement of 4.7% and 15.5% on Global MMLU for Malagasy and Turkish, respectively, compared to Llama 3.1 8B Instruct. Our controlled experiments reveal that fine-tuning transfer is most effective when the source and target models are linearly connected in the parameter space. Additionally, we demonstrate that fine-tuning transfer offers a stronger and more computationally efficient starting point for further fine-tuning. Finally, we propose an iterative recycling-then-finetuning approach for continuous model development, which improves both efficiency and effectiveness. Our findings suggest that fine-tuning transfer is a viable strategy to reduce training costs while maintaining model performance. 

---
# "Is There Anything Else?'': Examining Administrator Influence on Linguistic Features from the Cookie Theft Picture Description Cognitive Test 

**Authors**: Changye Li, Zhecheng Sheng, Trevor Cohen, Serguei Pakhomov  

**Link**: [PDF](https://arxiv.org/pdf/2503.20104)  

**Abstract**: Alzheimer's Disease (AD) dementia is a progressive neurodegenerative disease that negatively impacts patients' cognitive ability. Previous studies have demonstrated that changes in naturalistic language samples can be useful for early screening of AD dementia. However, the nature of language deficits often requires test administrators to use various speech elicitation techniques during spontaneous language assessments to obtain enough propositional utterances from dementia patients. This could lead to the ``observer's effect'' on the downstream analysis that has not been fully investigated. Our study seeks to quantify the influence of test administrators on linguistic features in dementia assessment with two English corpora the ``Cookie Theft'' picture description datasets collected at different locations and test administrators show different levels of administrator involvement. Our results show that the level of test administrator involvement significantly impacts observed linguistic features in patient speech. These results suggest that many of significant linguistic features in the downstream classification task may be partially attributable to differences in the test administration practices rather than solely to participants' cognitive status. The variations in test administrator behavior can lead to systematic biases in linguistic data, potentially confounding research outcomes and clinical assessments. Our study suggests that there is a need for a more standardized test administration protocol in the development of responsible clinical speech analytics frameworks. 

---
# Bigger But Not Better: Small Neural Language Models Outperform Large Language Models in Detection of Thought Disorder 

**Authors**: Changye Li, Weizhe Xu, Serguei Pakhomov, Ellen Bradley, Dror Ben-Zeev, Trevor Cohen  

**Link**: [PDF](https://arxiv.org/pdf/2503.20103)  

**Abstract**: Disorganized thinking is a key diagnostic indicator of schizophrenia-spectrum disorders. Recently, clinical estimates of the severity of disorganized thinking have been shown to correlate with measures of how difficult speech transcripts would be for large language models (LLMs) to predict. However, LLMs' deployment challenges -- including privacy concerns, computational and financial costs, and lack of transparency of training data -- limit their clinical utility. We investigate whether smaller neural language models can serve as effective alternatives for detecting positive formal thought disorder, using the same sliding window based perplexity measurements that proved effective with larger models. Surprisingly, our results show that smaller models are more sensitive to linguistic differences associated with formal thought disorder than their larger counterparts. Detection capability declines beyond a certain model size and context length, challenging the common assumption of ``bigger is better'' for LLM-based applications. Our findings generalize across audio diaries and clinical interview speech samples from individuals with psychotic symptoms, suggesting a promising direction for developing efficient, cost-effective, and privacy-preserving screening tools that can be deployed in both clinical and naturalistic settings. 

---
# Generative Linguistics, Large Language Models, and the Social Nature of Scientific Success 

**Authors**: Sophie Hao  

**Link**: [PDF](https://arxiv.org/pdf/2503.20088)  

**Abstract**: Chesi's (forthcoming) target paper depicts a generative linguistics in crisis, foreboded by Piantadosi's (2023) declaration that "modern language models refute Chomsky's approach to language." In order to survive, Chesi warns, generativists must hold themselves to higher standards of formal and empirical rigor. This response argues that the crisis described by Chesi and Piantadosi actually has little to do with rigor, but is rather a reflection of generativists' limited social ambitions. Chesi ties the fate of generative linguistics to its intellectual merits, but the current success of language model research is social in nature as much as it is intellectual. In order to thrive, then, generativists must do more than heed Chesi's call for rigor; they must also expand their ambitions by giving outsiders a stake in their future success. 

---
# Cross-Tokenizer Distillation via Approximate Likelihood Matching 

**Authors**: Benjamin Minixhofer, Edoardo Maria Ponti, Ivan Vulić  

**Link**: [PDF](https://arxiv.org/pdf/2503.20083)  

**Abstract**: Distillation has shown remarkable success in transferring knowledge from a Large Language Model (LLM) teacher to a student LLM. However, current distillation methods predominantly require the same tokenizer between the teacher and the student, restricting their applicability to only a small subset of teacher-student pairs. In this work, we develop a cross-tokenizer distillation method to solve this crucial deficiency. Our method is the first to enable cross-tokenizer distillation without a next-token prediction loss as the main objective, instead purely maximizing the student predictions' similarity to the teacher's predictions (known as pure distillation), while also being robust to large mismatches between the teacher and the student tokenizer function and vocabulary. Empirically, our method enables substantially improved performance as tested on two use cases. First, we show that viewing tokenizer transfer as self-distillation enables unprecedently effective transfer across tokenizers. We transfer (subword-level) Llama and Gemma models to byte-level tokenization more effectively than prior methods transfer to a similar subword tokenizer under a comparable training budget. Transferring different base models to the same tokenizer also enables ensembling them (e.g., via averaging their predicted probabilities) which boosts performance. Second, we use our cross-tokenizer distillation method to distil a large maths-specialized LLM into a smaller model, achieving competitive maths problem-solving performance. Overall, our results make substantial strides toward better adaptability and enhanced interaction between different LLMs. 

---
# Poor Alignment and Steerability of Large Language Models: Evidence from College Admission Essays 

**Authors**: Jinsook Lee, AJ Alvero, Thorsten Joachims, René Kizilcec  

**Link**: [PDF](https://arxiv.org/pdf/2503.20062)  

**Abstract**: People are increasingly using technologies equipped with large language models (LLM) to write texts for formal communication, which raises two important questions at the intersection of technology and society: Who do LLMs write like (model alignment); and can LLMs be prompted to change who they write like (model steerability). We investigate these questions in the high-stakes context of undergraduate admissions at a selective university by comparing lexical and sentence variation between essays written by 30,000 applicants to two types of LLM-generated essays: one prompted with only the essay question used by the human applicants; and another with additional demographic information about each applicant. We consistently find that both types of LLM-generated essays are linguistically distinct from human-authored essays, regardless of the specific model and analytical approach. Further, prompting a specific sociodemographic identity is remarkably ineffective in aligning the model with the linguistic patterns observed in human writing from this identity group. This holds along the key dimensions of sex, race, first-generation status, and geographic location. The demographically prompted and unprompted synthetic texts were also more similar to each other than to the human text, meaning that prompting did not alleviate homogenization. These issues of model alignment and steerability in current LLMs raise concerns about the use of LLMs in high-stakes contexts. 

---
# Low-resource Machine Translation for Code-switched Kazakh-Russian Language Pair 

**Authors**: Maksim Borisov, Zhanibek Kozhirbayev, Valentin Malykh  

**Link**: [PDF](https://arxiv.org/pdf/2503.20007)  

**Abstract**: Machine translation for low resource language pairs is a challenging task. This task could become extremely difficult once a speaker uses code switching. We propose a method to build a machine translation model for code-switched Kazakh-Russian language pair with no labeled data. Our method is basing on generation of synthetic data. Additionally, we present the first codeswitching Kazakh-Russian parallel corpus and the evaluation results, which include a model achieving 16.48 BLEU almost reaching an existing commercial system and beating it by human evaluation. 

---
# Untangling the Influence of Typology, Data and Model Architecture on Ranking Transfer Languages for Cross-Lingual POS Tagging 

**Authors**: Enora Rice, Ali Marashian, Hannah Haynie, Katharina von der Wense, Alexis Palmer  

**Link**: [PDF](https://arxiv.org/pdf/2503.19979)  

**Abstract**: Cross-lingual transfer learning is an invaluable tool for overcoming data scarcity, yet selecting a suitable transfer language remains a challenge. The precise roles of linguistic typology, training data, and model architecture in transfer language choice are not fully understood. We take a holistic approach, examining how both dataset-specific and fine-grained typological features influence transfer language selection for part-of-speech tagging, considering two different sources for morphosyntactic features. While previous work examines these dynamics in the context of bilingual biLSTMS, we extend our analysis to a more modern transfer learning pipeline: zero-shot prediction with pretrained multilingual models. We train a series of transfer language ranking systems and examine how different feature inputs influence ranker performance across architectures. Word overlap, type-token ratio, and genealogical distance emerge as top features across all architectures. Our findings reveal that a combination of typological and dataset-dependent features leads to the best rankings, and that good performance can be obtained with either feature group on its own. 

---
# Understanding R1-Zero-Like Training: A Critical Perspective 

**Authors**: Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, Min Lin  

**Link**: [PDF](https://arxiv.org/pdf/2503.20783)  

**Abstract**: DeepSeek-R1-Zero has shown that reinforcement learning (RL) at scale can directly enhance the reasoning capabilities of LLMs without supervised fine-tuning. In this work, we critically examine R1-Zero-like training by analyzing its two core components: base models and RL. We investigate a wide range of base models, including DeepSeek-V3-Base, to understand how pretraining characteristics influence RL performance. Our analysis reveals that DeepSeek-V3-Base already exhibit ''Aha moment'', while Qwen2.5 base models demonstrate strong reasoning capabilities even without prompt templates, suggesting potential pretraining biases. Additionally, we identify an optimization bias in Group Relative Policy Optimization (GRPO), which artificially increases response length (especially for incorrect outputs) during training. To address this, we introduce Dr. GRPO, an unbiased optimization method that improves token efficiency while maintaining reasoning performance. Leveraging these insights, we present a minimalist R1-Zero recipe that achieves 43.3% accuracy on AIME 2024 with a 7B base model, establishing a new state-of-the-art. Our code is available at this https URL. 

---
# Vision as LoRA 

**Authors**: Han Wang, Yongjie Ye, Bingru Li, Yuxiang Nie, Jinghui Lu, Jingqun Tang, Yanjie Wang, Can Huang  

**Link**: [PDF](https://arxiv.org/pdf/2503.20680)  

**Abstract**: We introduce Vision as LoRA (VoRA), a novel paradigm for transforming an LLM into an MLLM. Unlike prevalent MLLM architectures that rely on external vision modules for vision encoding, VoRA internalizes visual capabilities by integrating vision-specific LoRA layers directly into the LLM. This design allows the added parameters to be seamlessly merged into the LLM during inference, eliminating structural complexity and minimizing computational overhead. Moreover, inheriting the LLM's ability of handling flexible context, VoRA can process inputs at arbitrary resolutions.
To further strengthen VoRA's visual capabilities, we introduce a block-wise distillation method that transfers visual priors from a pre-trained ViT into the LoRA layers, effectively accelerating training by injecting visual knowledge. Additionally, we apply bi-directional attention masks to better capture the context information of an image. We successfully demonstrate that with additional pre-training data, VoRA can perform comparably with conventional encode-based MLLMs. All training data, codes, and model weights will be released at this https URL. 

---
# TAMA: A Human-AI Collaborative Thematic Analysis Framework Using Multi-Agent LLMs for Clinical Interviews 

**Authors**: Huimin Xu, Seungjun Yi, Terence Lim, Jiawei Xu, Andrew Well, Carlos Mery, Aidong Zhang, Yuji Zhang, Heng Ji, Keshav Pingali, Yan Leng, Ying Ding  

**Link**: [PDF](https://arxiv.org/pdf/2503.20666)  

**Abstract**: Thematic analysis (TA) is a widely used qualitative approach for uncovering latent meanings in unstructured text data. TA provides valuable insights in healthcare but is resource-intensive. Large Language Models (LLMs) have been introduced to perform TA, yet their applications in healthcare remain unexplored. Here, we propose TAMA: A Human-AI Collaborative Thematic Analysis framework using Multi-Agent LLMs for clinical interviews. We leverage the scalability and coherence of multi-agent systems through structured conversations between agents and coordinate the expertise of cardiac experts in TA. Using interview transcripts from parents of children with Anomalous Aortic Origin of a Coronary Artery (AAOCA), a rare congenital heart disease, we demonstrate that TAMA outperforms existing LLM-assisted TA approaches, achieving higher thematic hit rate, coverage, and distinctiveness. TAMA demonstrates strong potential for automated TA in clinical settings by leveraging multi-agent LLM systems with human-in-the-loop integration by enhancing quality while significantly reducing manual workload. 

---
# Optimizing Case-Based Reasoning System for Functional Test Script Generation with Large Language Models 

**Authors**: Siyuan Guo, Huiwu Liu, Xiaolong Chen, Yuming Xie, Liang Zhang, Tao Han, Hechang Chen, Yi Chang, Jun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.20576)  

**Abstract**: In this work, we explore the potential of large language models (LLMs) for generating functional test scripts, which necessitates understanding the dynamically evolving code structure of the target software. To achieve this, we propose a case-based reasoning (CBR) system utilizing a 4R cycle (i.e., retrieve, reuse, revise, and retain), which maintains and leverages a case bank of test intent descriptions and corresponding test scripts to facilitate LLMs for test script generation. To improve user experience further, we introduce Re4, an optimization method for the CBR system, comprising reranking-based retrieval finetuning and reinforced reuse finetuning. Specifically, we first identify positive examples with high semantic and script similarity, providing reliable pseudo-labels for finetuning the retriever model without costly labeling. Then, we apply supervised finetuning, followed by a reinforcement learning finetuning stage, to align LLMs with our production scenarios, ensuring the faithful reuse of retrieved cases. Extensive experimental results on two product development units from Huawei Datacom demonstrate the superiority of the proposed CBR+Re4. Notably, we also show that the proposed Re4 method can help alleviate the repetitive generation issues with LLMs. 

---
# Exploring the Effect of Robotic Embodiment and Empathetic Tone of LLMs on Empathy Elicitation 

**Authors**: Liza Darwesh, Jaspreet Singh, Marin Marian, Eduard Alexa, Koen Hindriks, Kim Baraka  

**Link**: [PDF](https://arxiv.org/pdf/2503.20518)  

**Abstract**: This study investigates the elicitation of empathy toward a third party through interaction with social agents. Participants engaged with either a physical robot or a voice-enabled chatbot, both driven by a large language model (LLM) programmed to exhibit either an empathetic tone or remain neutral. The interaction is focused on a fictional character, Katie Banks, who is in a challenging situation and in need of financial donations. The willingness to help Katie, measured by the number of hours participants were willing to volunteer, along with their perceptions of the agent, were assessed for 60 participants. Results indicate that neither robotic embodiment nor empathetic tone significantly influenced participants' willingness to volunteer. While the LLM effectively simulated human empathy, fostering genuine empathetic responses in participants proved challenging. 

---
# VPO: Aligning Text-to-Video Generation Models with Prompt Optimization 

**Authors**: Jiale Cheng, Ruiliang Lyu, Xiaotao Gu, Xiao Liu, Jiazheng Xu, Yida Lu, Jiayan Teng, Zhuoyi Yang, Yuxiao Dong, Jie Tang, Hongning Wang, Minlie Huang  

**Link**: [PDF](https://arxiv.org/pdf/2503.20491)  

**Abstract**: Video generation models have achieved remarkable progress in text-to-video tasks. These models are typically trained on text-video pairs with highly detailed and carefully crafted descriptions, while real-world user inputs during inference are often concise, vague, or poorly structured. This gap makes prompt optimization crucial for generating high-quality videos. Current methods often rely on large language models (LLMs) to refine prompts through in-context learning, but suffer from several limitations: they may distort user intent, omit critical details, or introduce safety risks. Moreover, they optimize prompts without considering the impact on the final video quality, which can lead to suboptimal results. To address these issues, we introduce VPO, a principled framework that optimizes prompts based on three core principles: harmlessness, accuracy, and helpfulness. The generated prompts faithfully preserve user intents and, more importantly, enhance the safety and quality of generated videos. To achieve this, VPO employs a two-stage optimization approach. First, we construct and refine a supervised fine-tuning (SFT) dataset based on principles of safety and alignment. Second, we introduce both text-level and video-level feedback to further optimize the SFT model with preference learning. Our extensive experiments demonstrate that VPO significantly improves safety, alignment, and video quality compared to baseline methods. Moreover, VPO shows strong generalization across video generation models. Furthermore, we demonstrate that VPO could outperform and be combined with RLHF methods on video generation models, underscoring the effectiveness of VPO in aligning video generation models. Our code and data are publicly available at this https URL. 

---
# VideoGEM: Training-free Action Grounding in Videos 

**Authors**: Felix Vogel, Walid Bousselham, Anna Kukleva, Nina Shvetsova, Hilde Kuehne  

**Link**: [PDF](https://arxiv.org/pdf/2503.20348)  

**Abstract**: Vision-language foundation models have shown impressive capabilities across various zero-shot tasks, including training-free localization and grounding, primarily focusing on localizing objects in images. However, leveraging those capabilities to localize actions and events in videos is challenging, as actions have less physical outline and are usually described by higher-level concepts. In this work, we propose VideoGEM, the first training-free spatial action grounding method based on pretrained image- and video-language backbones. Namely, we adapt the self-self attention formulation of GEM to spatial activity grounding. We observe that high-level semantic concepts, such as actions, usually emerge in the higher layers of the image- and video-language models. We, therefore, propose a layer weighting in the self-attention path to prioritize higher layers. Additionally, we introduce a dynamic weighting method to automatically tune layer weights to capture each layer`s relevance to a specific prompt. Finally, we introduce a prompt decomposition, processing action, verb, and object prompts separately, resulting in a better spatial localization of actions. We evaluate the proposed approach on three image- and video-language backbones, CLIP, OpenCLIP, and ViCLIP, and on four video grounding datasets, V-HICO, DALY, YouCook-Interactions, and GroundingYouTube, showing that the proposed training-free approach is able to outperform current trained state-of-the-art approaches for spatial video grounding. 

---
# QualiSpeech: A Speech Quality Assessment Dataset with Natural Language Reasoning and Descriptions 

**Authors**: Siyin Wang, Wenyi Yu, Xianzhao Chen, Xiaohai Tian, Jun Zhang, Yu Tsao, Junichi Yamagishi, Yuxuan Wang, Chao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.20290)  

**Abstract**: This paper explores a novel perspective to speech quality assessment by leveraging natural language descriptions, offering richer, more nuanced insights than traditional numerical scoring methods. Natural language feedback provides instructive recommendations and detailed evaluations, yet existing datasets lack the comprehensive annotations needed for this approach. To bridge this gap, we introduce QualiSpeech, a comprehensive low-level speech quality assessment dataset encompassing 11 key aspects and detailed natural language comments that include reasoning and contextual insights. Additionally, we propose the QualiSpeech Benchmark to evaluate the low-level speech understanding capabilities of auditory large language models (LLMs). Experimental results demonstrate that finetuned auditory LLMs can reliably generate detailed descriptions of noise and distortion, effectively identifying their types and temporal characteristics. The results further highlight the potential for incorporating reasoning to enhance the accuracy and reliability of quality assessments. The dataset will be released at this https URL. 

---
# ViLBench: A Suite for Vision-Language Process Reward Modeling 

**Authors**: Haoqin Tu, Weitao Feng, Hardy Chen, Hui Liu, Xianfeng Tang, Cihang Xie  

**Link**: [PDF](https://arxiv.org/pdf/2503.20271)  

**Abstract**: Process-supervised reward models serve as a fine-grained function that provides detailed step-wise feedback to model responses, facilitating effective selection of reasoning trajectories for complex tasks. Despite its advantages, evaluation on PRMs remains less explored, especially in the multimodal domain. To address this gap, this paper first benchmarks current vision large language models (VLLMs) as two types of reward models: output reward models (ORMs) and process reward models (PRMs) on multiple vision-language benchmarks, which reveal that neither ORM nor PRM consistently outperforms across all tasks, and superior VLLMs do not necessarily yield better rewarding performance. To further advance evaluation, we introduce ViLBench, a vision-language benchmark designed to require intensive process reward signals. Notably, OpenAI's GPT-4o with Chain-of-Thought (CoT) achieves only 27.3% accuracy, indicating the benchmark's challenge for current VLLMs. Lastly, we preliminarily showcase a promising pathway towards bridging the gap between general VLLMs and reward models -- by collecting 73.6K vision-language process reward data using an enhanced tree-search algorithm, our 3B model is able to achieve an average improvement of 3.3% over standard CoT and up to 2.5% compared to its untrained counterpart on ViLBench by selecting OpenAI o1's generations. We release the implementations at this https URL with our code, model, and data. 

---
# TeleLoRA: Teleporting Model-Specific Alignment Across LLMs 

**Authors**: Xiao Lin, Manoj Acharya, Anirban Roy, Susmit Jha  

**Link**: [PDF](https://arxiv.org/pdf/2503.20228)  

**Abstract**: Mitigating Trojans in Large Language Models (LLMs) is one of many tasks where alignment data is LLM specific, as different LLMs have different Trojan triggers and trigger behaviors to be removed. In this paper, we introduce TeleLoRA (Teleporting Low-Rank Adaptation), a novel framework that synergizes model-specific alignment data across multiple LLMs to enable zero-shot Trojan mitigation on unseen LLMs without alignment data. TeleLoRA learns a unified generator of LoRA adapter weights by leveraging local activation information across multiple LLMs. This generator is designed to be permutation symmetric to generalize across models with different architectures and sizes. We optimize the model design for memory efficiency, making it feasible to learn with large-scale LLMs with minimal computational resources. Experiments on LLM Trojan mitigation benchmarks demonstrate that TeleLoRA effectively reduces attack success rates while preserving the benign performance of the models. 

---
# Open Deep Search: Democratizing Search with Open-source Reasoning Agents 

**Authors**: Salaheddin Alzubi, Creston Brooks, Purva Chiniya, Edoardo Contente, Chiara von Gerlach, Lucas Irwin, Yihan Jiang, Arda Kaz, Windsor Nguyen, Sewoong Oh, Himanshu Tyagi, Pramod Viswanath  

**Link**: [PDF](https://arxiv.org/pdf/2503.20201)  

**Abstract**: We introduce Open Deep Search (ODS) to close the increasing gap between the proprietary search AI solutions, such as Perplexity's Sonar Reasoning Pro and OpenAI's GPT-4o Search Preview, and their open-source counterparts. The main innovation introduced in ODS is to augment the reasoning capabilities of the latest open-source LLMs with reasoning agents that can judiciously use web search tools to answer queries. Concretely, ODS consists of two components that work with a base LLM chosen by the user: Open Search Tool and Open Reasoning Agent. Open Reasoning Agent interprets the given task and completes it by orchestrating a sequence of actions that includes calling tools, one of which is the Open Search Tool. Open Search Tool is a novel web search tool that outperforms proprietary counterparts. Together with powerful open-source reasoning LLMs, such as DeepSeek-R1, ODS nearly matches and sometimes surpasses the existing state-of-the-art baselines on two benchmarks: SimpleQA and FRAMES. For example, on the FRAMES evaluation benchmark, ODS improves the best existing baseline of the recently released GPT-4o Search Preview by 9.7% in accuracy. ODS is a general framework for seamlessly augmenting any LLMs -- for example, DeepSeek-R1 that achieves 82.4% on SimpleQA and 30.1% on FRAMES -- with search and reasoning capabilities to achieve state-of-the-art performance: 88.3% on SimpleQA and 75.3% on FRAMES. 

---
# LogQuant: Log-Distributed 2-Bit Quantization of KV Cache with Superior Accuracy Preservation 

**Authors**: Han Chen, Zicong Jiang, Zining Zhang, Bingsheng He, Pingyi Luo, Mian Lu, Yuqiang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.19950)  

**Abstract**: We introduce LogQuant, a groundbreaking 2-bit quantization technique for KV Cache in large language model (LLM) inference, delivering substantial memory savings while preserving superior performance. Previous methods either assume that later tokens are more important or attempt to predict important tokens based on earlier attention patterns. Both approaches, however, can result in performance bottlenecks or frequent mispredictions.
LogQuant takes a different approach. By applying a log-based filtering mechanism, it selectively compresses the KV Cache across the entire context, achieving better performance with the same or even reduced memory footprint compared to existing methods. In benchmark tests, it enhances throughput by 25% and boosts batch size by 60% without increasing memory consumption. For challenging tasks such as Math and Code Completion, LogQuant improves accuracy by 40% to 200% at the same compression ratio, outperforming comparable this http URL integrates effortlessly with popular inference frameworks like Python's transformers library. Implementation can be available in this https URL. 

---
