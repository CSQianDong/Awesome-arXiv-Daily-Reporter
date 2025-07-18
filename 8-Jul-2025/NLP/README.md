# Evaluating Memory in LLM Agents via Incremental Multi-Turn Interactions 

**Authors**: Yuanzhe Hu, Yu Wang, Julian McAuley  

**Link**: [PDF](https://arxiv.org/pdf/2507.05257)  

**Abstract**: Recent benchmarks for Large Language Model (LLM) agents primarily focus on evaluating reasoning, planning, and execution capabilities, while another critical component-memory, encompassing how agents memorize, update, and retrieve long-term information-is under-evaluated due to the lack of benchmarks. We term agents with memory mechanisms as memory agents. In this paper, we identify four core competencies essential for memory agents: accurate retrieval, test-time learning, long-range understanding, and conflict resolution. Existing datasets either rely on limited context lengths or are tailored for static, long-context settings like book-based QA, which do not reflect the interactive, multi-turn nature of memory agents that incrementally accumulate information. Furthermore, no existing benchmarks cover all four competencies. Therefore, we introduce MemoryAgentBench, a new benchmark specifically designed for memory agents. Our benchmark combines reformulated existing datasets with newly constructed ones, covering the above four memory competencies, providing a systematic and challenging testbed for assessing memory quality. We evaluate a diverse set of memory agents, ranging from simple context-based and retrieval-augmented generation (RAG) systems to advanced agents with external memory modules and tool integration. Empirical results reveal that current methods fall short of mastering all four competencies, underscoring the need for further research into comprehensive memory mechanisms for LLM agents. 

---
# Response Attack: Exploiting Contextual Priming to Jailbreak Large Language Models 

**Authors**: Ziqi Miao, Lijun Li, Yuan Xiong, Zhenhua Liu, Pengyu Zhu, Jing Shao  

**Link**: [PDF](https://arxiv.org/pdf/2507.05248)  

**Abstract**: Contextual priming, where earlier stimuli covertly bias later judgments, offers an unexplored attack surface for large language models (LLMs). We uncover a contextual priming vulnerability in which the previous response in the dialogue can steer its subsequent behavior toward policy-violating content. Building on this insight, we propose Response Attack, which uses an auxiliary LLM to generate a mildly harmful response to a paraphrased version of the original malicious query. They are then formatted into the dialogue and followed by a succinct trigger prompt, thereby priming the target model to generate harmful content. Across eight open-source and proprietary LLMs, RA consistently outperforms seven state-of-the-art jailbreak techniques, achieving higher attack success rates. To mitigate this threat, we construct and release a context-aware safety fine-tuning dataset, which significantly reduces the attack success rate while preserving model capabilities. The code and data are available at this https URL. 

---
# Pre-Trained Policy Discriminators are General Reward Models 

**Authors**: Shihan Dou, Shichun Liu, Yuming Yang, Yicheng Zou, Yunhua Zhou, Shuhao Xing, Chenhao Huang, Qiming Ge, Demin Song, Haijun Lv, Songyang Gao, Chengqi Lv, Enyu Zhou, Honglin Guo, Zhiheng Xi, Wenwei Zhang, Qipeng Guo, Qi Zhang, Xipeng Qiu, Xuanjing Huang, Tao Gui, Kai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.05197)  

**Abstract**: We offer a novel perspective on reward modeling by formulating it as a policy discriminator, which quantifies the difference between two policies to generate a reward signal, guiding the training policy towards a target policy with desired behaviors. Based on this conceptual insight, we propose a scalable pre-training method named Policy Discriminative Learning (POLAR), which trains a reward model (RM) to discern identical policies and discriminate different ones. Unlike traditional reward modeling methods relying on absolute preferences, POLAR captures the relative difference between one policy and an arbitrary target policy, which is a scalable, high-level optimization objective suitable for modeling generic ranking relationships. Leveraging the POLAR pre-training paradigm, we present a series of RMs with parameter scales from 1.8B to 7B. Empirical results show that POLAR substantially outperforms traditional non-pre-trained methods, significantly enhancing RM performance. For instance, POLAR-7B could improve preference accuracy from 54.8% to 81.0% on STEM tasks and from 57.9% to 85.5% on creative writing tasks compared to SOTA baselines. POLAR also shows robust generalization capabilities in RLHF using Reinforcement Fine-tuning (RFT), providing reliable reward signals and markedly enhancing policy performance--improving LLaMa3.1-8B from an average of 47.36% to 56.33% and Qwen2.5-32B from 64.49% to 70.47% on 20 benchmarks. Moreover, scaling experiments reveal a clear power-law relationship between computation and performance, supported by linear correlation coefficients approaching 0.99. The impressive performance, strong generalization, and scaling properties suggest that POLAR is a promising direction for developing general and strong reward models. 

---
# From Fragments to Facts: A Curriculum-Driven DPO Approach for Generating Hindi News Veracity Explanations 

**Authors**: Pulkit Bansal, Raghvendra Kumar, Shakti Singh, Sriparna Saha, Adam Jatowt  

**Link**: [PDF](https://arxiv.org/pdf/2507.05179)  

**Abstract**: In an era of rampant misinformation, generating reliable news explanations is vital, especially for under-represented languages like Hindi. Lacking robust automated tools, Hindi faces challenges in scaling misinformation detection. To bridge this gap, we propose a novel framework integrating Direct Preference Optimization (DPO) with curriculum learning to align machine-generated explanations with human reasoning. Fact-checked explanations from credible sources serve as preferred responses, while LLM outputs highlight system limitations and serve as non-preferred responses. To refine task-specific alignment, we introduce two key parameters -- Actuality and Finesse -- into the DPO loss function, enhancing explanation quality and consistency. Experiments with LLMs (Mistral, Llama, Gemma) and PLMs (mBART, mT5) confirm the framework's effectiveness in generating coherent, contextually relevant explanations. This scalable approach combats misinformation and extends automated explanation generation to low-resource languages. 

---
# OpenS2S: Advancing Open-Source End-to-End Empathetic Large Speech Language Model 

**Authors**: Chen Wang, Tianyu Peng, Wen Yang, Yinan Bai, Guangfu Wang, Jun Lin, Lanpeng Jia, Lingxiang Wu, Jinqiao Wang, Chengqing Zong, Jiajun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.05177)  

**Abstract**: Empathetic interaction is a cornerstone of human-machine communication, due to the need for understanding speech enriched with paralinguistic cues and generating emotional and expressive responses. However, the most powerful empathetic LSLMs are increasingly closed off, leaving the crucial details about the architecture, data and development opaque to researchers. Given the critical need for transparent research into the LSLMs and empathetic behavior, we present OpenS2S, a fully open-source, transparent and end-to-end LSLM designed to enable empathetic speech interactions. Based on our empathetic speech-to-text model BLSP-Emo, OpenS2S further employs a streaming interleaved decoding architecture to achieve low-latency speech generation. To facilitate end-to-end training, OpenS2S incorporates an automated data construction pipeline that synthesizes diverse, high-quality empathetic speech dialogues at low cost. By leveraging large language models to generate empathetic content and controllable text-to-speech systems to introduce speaker and emotional variation, we construct a scalable training corpus with rich paralinguistic diversity and minimal human supervision. We release the fully open-source OpenS2S model, including the dataset, model weights, pre-training and fine-tuning codes, to empower the broader research community and accelerate innovation in empathetic speech systems. The project webpage can be accessed at this https URL 

---
# InfoSteer: Steering Information Utility in Language Model Post-Training 

**Authors**: Chunyuan Deng, Ruidi Chang, Hanjie Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.05158)  

**Abstract**: Recent advancements in language models (LMs) gradually ushered in an era where post-training is crucial. Yet, post-training approaches such as supervised fine-tuning (SFT) do not guarantee effective use of knowledge acquired during pretraining. We therefore present \ours, a lightweight method that encourages parametric information utilization in LMs during post-training. This is achieved via treating FFN layer as associate key-value memory, and promotes the use of stored memory vectors via forward-pass interventions or regularization during backpropagation. We find this simple guidance during post-training phase delivers consistent performance improvements across diverse model families--including Qwen, Gemma and Llama-spanning over 15 downstream tasks in both ID and OOD evaluations. Beyond performance gains, we also find that steered LMs can adaptively allocate information-placing more emphasis on generating semantically meaningful tokens, while using fewer resources on simple transition ones (e.g., `,' or `and'). Our work underscores that vanilla post-training does not fully leverage pre-training potential, and steering LMs in latent representation space offers a promising approach that enhances both performance and interpretability. 

---
# AI Generated Text Detection Using Instruction Fine-tuned Large Language and Transformer-Based Models 

**Authors**: Chinnappa Guggilla, Budhaditya Roy, Trupti Ramdas Chavan, Abdul Rahman, Edward Bowen  

**Link**: [PDF](https://arxiv.org/pdf/2507.05157)  

**Abstract**: Large Language Models (LLMs) possess an extraordinary capability to produce text that is not only coherent and contextually relevant but also strikingly similar to human writing. They adapt to various styles and genres, producing content that is both grammatically correct and semantically meaningful. Recently, LLMs have been misused to create highly realistic phishing emails, spread fake news, generate code to automate cyber crime, and write fraudulent scientific articles. Additionally, in many real-world applications, the generated content including style and topic and the generator model are not known beforehand. The increasing prevalence and sophistication of artificial intelligence (AI)-generated texts have made their detection progressively more challenging. Various attempts have been made to distinguish machine-generated text from human-authored content using linguistic, statistical, machine learning, and ensemble-based approaches. This work focuses on two primary objectives Task-A, which involves distinguishing human-written text from machine-generated text, and Task-B, which attempts to identify the specific LLM model responsible for the generation. Both of these tasks are based on fine tuning of Generative Pre-trained Transformer (GPT_4o-mini), Large Language Model Meta AI (LLaMA) 3 8B, and Bidirectional Encoder Representations from Transformers (BERT). The fine-tuned version of GPT_4o-mini and the BERT model has achieved accuracies of 0.9547 for Task-A and 0.4698 for Task-B. 

---
# Interpretable Mnemonic Generation for Kanji Learning via Expectation-Maximization 

**Authors**: Jaewook Lee, Alexander Scarlatos, Andrew Lan  

**Link**: [PDF](https://arxiv.org/pdf/2507.05137)  

**Abstract**: Learning Japanese vocabulary is a challenge for learners from Roman alphabet backgrounds due to script differences. Japanese combines syllabaries like hiragana with kanji, which are logographic characters of Chinese origin. Kanji are also complicated due to their complexity and volume. Keyword mnemonics are a common strategy to aid memorization, often using the compositional structure of kanji to form vivid associations. Despite recent efforts to use large language models (LLMs) to assist learners, existing methods for LLM-based keyword mnemonic generation function as a black box, offering limited interpretability. We propose a generative framework that explicitly models the mnemonic construction process as driven by a set of common rules, and learn them using a novel Expectation-Maximization-type algorithm. Trained on learner-authored mnemonics from an online platform, our method learns latent structures and compositional rules, enabling interpretable and systematic mnemonics generation. Experiments show that our method performs well in the cold-start setting for new learners while providing insight into the mechanisms behind effective mnemonic creation. 

---
# SMART: Simulated Students Aligned with Item Response Theory for Question Difficulty Prediction 

**Authors**: Alexander Scarlatos, Nigel Fernandez, Christopher Ormerod, Susan Lottridge, Andrew Lan  

**Link**: [PDF](https://arxiv.org/pdf/2507.05129)  

**Abstract**: Item (question) difficulties play a crucial role in educational assessments, enabling accurate and efficient assessment of student abilities and personalization to maximize learning outcomes. Traditionally, estimating item difficulties can be costly, requiring real students to respond to items, followed by fitting an item response theory (IRT) model to get item difficulty estimates. This approach cannot be applied to the cold-start setting for previously unseen items either. In this work, we present SMART (Simulated Students Aligned with IRT), a novel method for aligning simulated students with instructed ability, which can then be used in simulations to predict the difficulty of open-ended items. We achieve this alignment using direct preference optimization (DPO), where we form preference pairs based on how likely responses are under a ground-truth IRT model. We perform a simulation by generating thousands of responses, evaluating them with an LLM-based scoring model, and fit the resulting data to an IRT model to obtain item difficulty estimates. Through extensive experiments on a real-world student response dataset, we show that SMART outperforms other item difficulty prediction methods by leveraging its improved ability alignment. 

---
# An Evaluation of Large Language Models on Text Summarization Tasks Using Prompt Engineering Techniques 

**Authors**: Walid Mohamed Aly, Taysir Hassan A. Soliman, Amr Mohamed AbdelAziz  

**Link**: [PDF](https://arxiv.org/pdf/2507.05123)  

**Abstract**: Large Language Models (LLMs) continue to advance natural language processing with their ability to generate human-like text across a range of tasks. Despite the remarkable success of LLMs in Natural Language Processing (NLP), their performance in text summarization across various domains and datasets has not been comprehensively evaluated. At the same time, the ability to summarize text effectively without relying on extensive training data has become a crucial bottleneck. To address these issues, we present a systematic evaluation of six LLMs across four datasets: CNN/Daily Mail and NewsRoom (news), SAMSum (dialog), and ArXiv (scientific). By leveraging prompt engineering techniques including zero-shot and in-context learning, our study evaluates the performance using the ROUGE and BERTScore metrics. In addition, a detailed analysis of inference times is conducted to better understand the trade-off between summarization quality and computational efficiency. For Long documents, introduce a sentence-based chunking strategy that enables LLMs with shorter context windows to summarize extended inputs in multiple stages. The findings reveal that while LLMs perform competitively on news and dialog tasks, their performance on long scientific documents improves significantly when aided by chunking strategies. In addition, notable performance variations were observed based on model parameters, dataset properties, and prompt design. These results offer actionable insights into how different LLMs behave across task types, contributing to ongoing research in efficient, instruction-based NLP systems. 

---
# Verified Language Processing with Hybrid Explainability: A Technical Report 

**Authors**: Oliver Robert Fox, Giacomo Bergami, Graham Morgan  

**Link**: [PDF](https://arxiv.org/pdf/2507.05017)  

**Abstract**: The volume and diversity of digital information have led to a growing reliance on Machine Learning techniques, such as Natural Language Processing, for interpreting and accessing appropriate data. While vector and graph embeddings represent data for similarity tasks, current state-of-the-art pipelines lack guaranteed explainability, failing to determine similarity for given full texts accurately. These considerations can also be applied to classifiers exploiting generative language models with logical prompts, which fail to correctly distinguish between logical implication, indifference, and inconsistency, despite being explicitly trained to recognise the first two classes. We present a novel pipeline designed for hybrid explainability to address this. Our methodology combines graphs and logic to produce First-Order Logic representations, creating machine- and human-readable representations through Montague Grammar. Preliminary results indicate the effectiveness of this approach in accurately capturing full text similarity. To the best of our knowledge, this is the first approach to differentiate between implication, inconsistency, and indifference for text classification tasks. To address the limitations of existing approaches, we use three self-contained datasets annotated for the former classification task to determine the suitability of these approaches in capturing sentence structure equivalence, logical connectives, and spatiotemporal reasoning. We also use these data to compare the proposed method with language models pre-trained for detecting sentence entailment. The results show that the proposed method outperforms state-of-the-art models, indicating that natural language understanding cannot be easily generalised by training over extensive document corpora. This work offers a step toward more transparent and reliable Information Retrieval from extensive textual data. 

---
# Co-DETECT: Collaborative Discovery of Edge Cases in Text Classification 

**Authors**: Chenfei Xiong, Jingwei Ni, Yu Fan, Vilém Zouhar, Donya Rooein, Lorena Calvo-Bartolomé, Alexander Hoyle, Zhijing Jin, Mrinmaya Sachan, Markus Leippold, Dirk Hovy, Mennatallah El-Assady, Elliott Ash  

**Link**: [PDF](https://arxiv.org/pdf/2507.05010)  

**Abstract**: We introduce Co-DETECT (Collaborative Discovery of Edge cases in TExt ClassificaTion), a novel mixed-initiative annotation framework that integrates human expertise with automatic annotation guided by large language models (LLMs). Co-DETECT starts with an initial, sketch-level codebook and dataset provided by a domain expert, then leverages the LLM to annotate the data and identify edge cases that are not well described by the initial codebook. Specifically, Co-DETECT flags challenging examples, induces high-level, generalizable descriptions of edge cases, and assists user in incorporating edge case handling rules to improve the codebook. This iterative process enables more effective handling of nuanced phenomena through compact, generalizable annotation rules. Extensive user study, qualitative and quantitative analyses prove the effectiveness of Co-DETECT. 

---
# ArtifactsBench: Bridging the Visual-Interactive Gap in LLM Code Generation Evaluation 

**Authors**: Chenchen Zhang, Yuhang Li, Can Xu, Jiaheng Liu, Ao Liu, Shihui Hu, Dengpeng Wu, Guanhua Huang, Kejiao Li, Qi Yi, Ruibin Xiong, Haotian Zhu, Yuanxing Zhang, Yuhao Jiang, Yue Zhang, Zenan Xu, Bohui Zhai, Guoxiang He, Hebin Li, Jie Zhao, Le Zhang, Lingyun Tan, Pengyu Guo, Xianshu Pang, Yang Ruan, Zhifeng Zhang, Zhonghu Wang, Ziyan Xu, Zuopu Yin, Wiggin Zhou, Chayse Zhou, Fengzong Lian  

**Link**: [PDF](https://arxiv.org/pdf/2507.04952)  

**Abstract**: The generative capabilities of Large Language Models (LLMs) are rapidly expanding from static code to dynamic, interactive visual artifacts. This progress is bottlenecked by a critical evaluation gap: established benchmarks focus on algorithmic correctness and are blind to the visual fidelity and interactive integrity that define modern user experiences. To bridge this gap, we introduce ArtifactsBench, a new benchmark and paradigm for the automated, multimodal evaluation of visual code generation. Our framework programmatically renders each generated artifact and captures its dynamic behavior through temporal screenshots. This visual evidence, alongside the source code, is then assessed by a Multimodal LLM (MLLM)-as-Judge, which is rigorously guided by a fine-grained, per-task checklist to ensure holistic and reproducible scoring. We construct a new benchmark of 1,825 diverse tasks and evaluate over 30 leading LLMs. Our automated evaluation achieves a striking 94.4% ranking consistency with WebDev Arena, the gold-standard for human preference in web development, and over 90% pairwise agreement with human experts. This establishes ArtifactsBench as the first framework to reliably automate the assessment of human-perceived quality at scale. Our analysis provides a high-resolution map of the current SOTA, revealing that generalist models often outperform domain-specific ones. We open-source ArtifactsBench, including the benchmark, evaluation harness, and baseline results at this https URL, to provide the community with a scalable and accurate tool to accelerate the development of user-centric generative models. 

---
# SIGIR 2025 -- LiveRAG Challenge Report 

**Authors**: David Carmel, Simone Filice, Guy Horowitz, Yoelle Maarek, Oren Somekh, Ran Tavory  

**Link**: [PDF](https://arxiv.org/pdf/2507.04942)  

**Abstract**: The LiveRAG Challenge at SIGIR 2025, held between March and May 2025, provided a competitive platform for advancing Retrieval-Augmented Generation (RAG) technologies. Participants from academia and industry were invited to develop a RAG-based question-answering system using a fixed corpus (Fineweb-10BT) and a common open-source LLM (Falcon3-10B-Instruct). The goal was to facilitate challenging comparisons of retrieval and prompting strategies. During the Live Challenge Day, 70 teams from 27 different countries provided answers and supportive information to 500 unseen questions within a strict two-hour time window. Evaluation was conducted in two stages: first an automated LLM-as-a-judge approach was used to compute correctness and faithfulness score, then a manual review of top ranked submissions was conducted. The finalists were announced on June 12, 2025, with prizes awarded during the LiveRAG Workshop at SIGIR 2025 in Padua, Italy. 

---
# O_FT@EvalLLM2025 : étude comparative de choix de données et de stratégies d'apprentissage pour l'adaptation de modèles de langue à un domaine 

**Authors**: Ismaël Rousseau, Claire Perroux, Pierre Adam, Thomas Girault, Lionel Delphin-Poulat, Morgan Veyret, Gwénolé Lecorvé, Géraldine Damnati  

**Link**: [PDF](https://arxiv.org/pdf/2507.04895)  

**Abstract**: This paper presents the work carried out by the O_FT team, joint with Orange and Ouest-France, on adapting language models to the defense domain as part of the EvalLLM2025 challenge. This work focused on adapting the \texttt{Mistral-7B-Instruct-v0.3} model using classical techniques of continued pre-training and instruction-tuning. The core of our efforts is based on collecting, generating, and selecting data for these two stages as well as for model evaluation. Experiments show that our adapted models have better domain-specific knowledge and improved domain-specific task processing skills, along with comparable (or even superior) performance on general knowledge and skills. Considering the carbon footprint of our adaptations, this work demonstrates the feasibility of domain adaptation for relatively small models.
--
Ce document présente les travaux réalisés par l'équipe O_FT conjointe à Orange et Ouest-France sur l'adaptation de modèles de langue au domaine de la défense dans le cadre du challenge EvalLLM2025. Ces travaux se sont concentrés sur l'adaptation du modèle \texttt{Mistral-7B-Instruct-v0.3} avec des techniques classiques de poursuite du pré-entraînement et d'affinage sur instructions. L'essentiel de nos travaux a porté sur la constitution, génération et sélection de données pour ces deux étapes ainsi que pour l'évaluation des modèles. Les expériences montrent que nos modèles adaptés ont de meilleures de connaissances de fond et une meilleure capacité de traitement de tâches sur le domaine de la défense, ainsi que des performances comparables (voire supérieures) sur des connaissances ou capacités généralistes. Mis au regard des empreintes carbones de nos adaptations, ces travaux démontrent ainsi la viabilité de l'adaptation à un domaine de modèles relativement petits. 

---
# Emergent Semantics Beyond Token Embeddings: Transformer LMs with Frozen Visual Unicode Representations 

**Authors**: A. Bochkov  

**Link**: [PDF](https://arxiv.org/pdf/2507.04886)  

**Abstract**: Understanding the locus of semantic representation in large language models (LLMs) is crucial for interpretability and architectural innovation. The dominant paradigm posits that trainable input embeddings serve as foundational "meaning vectors." This paper challenges that view. We construct Transformer models where the embedding layer is entirely frozen, with vectors derived not from data, but from the visual structure of Unicode glyphs. These non-semantic, precomputed visual embeddings are fixed throughout training. Our method is compatible with any tokenizer, including a novel Unicode-centric tokenizer we introduce to ensure universal text coverage. Despite the absence of trainable, semantically initialized embeddings, our models converge, generate coherent text, and, critically, outperform architecturally identical models with trainable embeddings on the MMLU reasoning benchmark. We attribute this to "representational interference" in conventional models, where the embedding layer is burdened with learning both structural and semantic features. Our results indicate that high-level semantics are not inherent to input embeddings but are an emergent property of the Transformer's compositional architecture and data scale. This reframes the role of embeddings from meaning containers to structural primitives. We release all code and models to foster further research. 

---
# Building Open-Retrieval Conversational Question Answering Systems by Generating Synthetic Data and Decontextualizing User Questions 

**Authors**: Christos Vlachos, Nikolaos Stylianou, Alexandra Fiotaki, Spiros Methenitis, Elisavet Palogiannidi, Themos Stafylakis, Ion Androutsopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2507.04884)  

**Abstract**: We consider open-retrieval conversational question answering (OR-CONVQA), an extension of question answering where system responses need to be (i) aware of dialog history and (ii) grounded in documents (or document fragments) retrieved per question. Domain-specific OR-CONVQA training datasets are crucial for real-world applications, but hard to obtain. We propose a pipeline that capitalizes on the abundance of plain text documents in organizations (e.g., product documentation) to automatically produce realistic OR-CONVQA dialogs with annotations. Similarly to real-world humanannotated OR-CONVQA datasets, we generate in-dialog question-answer pairs, self-contained (decontextualized, e.g., no referring expressions) versions of user questions, and propositions (sentences expressing prominent information from the documents) the system responses are grounded in. We show how the synthetic dialogs can be used to train efficient question rewriters that decontextualize user questions, allowing existing dialog-unaware retrievers to be utilized. The retrieved information and the decontextualized question are then passed on to an LLM that generates the system's response. 

---
# $\textit{Grahak-Nyay:}$ Consumer Grievance Redressal through Large Language Models 

**Authors**: Shrey Ganatra, Swapnil Bhattacharyya, Harshvivek Kashid, Spandan Anaokar, Shruti Nair, Reshma Sekhar, Siddharth Manohar, Rahul Hemrajani, Pushpak Bhattacharyya  

**Link**: [PDF](https://arxiv.org/pdf/2507.04854)  

**Abstract**: Access to consumer grievance redressal in India is often hindered by procedural complexity, legal jargon, and jurisdictional challenges. To address this, we present $\textbf{Grahak-Nyay}$ (Justice-to-Consumers), a chatbot that streamlines the process using open-source Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG). Grahak-Nyay simplifies legal complexities through a concise and up-to-date knowledge base. We introduce three novel datasets: $\textit{GeneralQA}$ (general consumer law), $\textit{SectoralQA}$ (sector-specific knowledge) and $\textit{SyntheticQA}$ (for RAG evaluation), along with $\textit{NyayChat}$, a dataset of 300 annotated chatbot conversations. We also introduce $\textit{Judgments}$ data sourced from Indian Consumer Courts to aid the chatbot in decision making and to enhance user trust. We also propose $\textbf{HAB}$ metrics ($\textbf{Helpfulness, Accuracy, Brevity}$) to evaluate chatbot performance. Legal domain experts validated Grahak-Nyay's effectiveness. Code and datasets will be released. 

---
# Dialogue-Based Multi-Dimensional Relationship Extraction from Novels 

**Authors**: Yuchen Yan, Hanjie Zhao, Senbin Zhu, Hongde Liu, Zhihong Zhang, Yuxiang Jia  

**Link**: [PDF](https://arxiv.org/pdf/2507.04852)  

**Abstract**: Relation extraction is a crucial task in natural language processing, with broad applications in knowledge graph construction and literary analysis. However, the complex context and implicit expressions in novel texts pose significant challenges for automatic character relationship extraction. This study focuses on relation extraction in the novel domain and proposes a method based on Large Language Models (LLMs). By incorporating relationship dimension separation, dialogue data construction, and contextual learning strategies, the proposed method enhances extraction performance. Leveraging dialogue structure information, it improves the model's ability to understand implicit relationships and demonstrates strong adaptability in complex contexts. Additionally, we construct a high-quality Chinese novel relation extraction dataset to address the lack of labeled resources and support future research. Experimental results show that our method outperforms traditional baselines across multiple evaluation metrics and successfully facilitates the automated construction of character relationship networks in novels. 

---
# Spec-TOD: A Specialized Instruction-Tuned LLM Framework for Efficient Task-Oriented Dialogue Systems 

**Authors**: Quang-Vinh Nguyen, Quang-Chieu Nguyen, Hoang Pham, Khac-Hoai Nam Bui  

**Link**: [PDF](https://arxiv.org/pdf/2507.04841)  

**Abstract**: Task-oriented dialogue (TOD) systems facilitate goal-driven interactions between users and machines. While recent advances in deep learning have improved the performance, TOD systems often struggle in low-resource scenarios with limited labeled data. To address this challenge, we propose Spec-TOD, a novel framework designed to train an end-to-end TOD system with limited data. Spec-TOD introduces two main innovations: (i) a novel specialized end-to-end TOD framework that incorporates explicit task instructions for instruction-tuned large language models (LLMs), and (ii) an efficient training strategy that leverages lightweight, specialized LLMs to achieve strong performance with minimal supervision. Experiments on the MultiWOZ dataset, a widely used TOD benchmark, demonstrate that Spec-TOD achieves competitive results while significantly reducing the need for labeled data. These findings highlight the potential of the proposed framework in advancing efficient and effective TOD systems in low-resource settings. 

---
# A Survey of Pun Generation: Datasets, Evaluations and Methodologies 

**Authors**: Yuchen Su, Yonghua Zhu, Ruofan Wang, Zijian Huang, Diana Benavides-Prado, Michael Witbrock  

**Link**: [PDF](https://arxiv.org/pdf/2507.04793)  

**Abstract**: Pun generation seeks to creatively modify linguistic elements in text to produce humour or evoke double meanings. It also aims to preserve coherence and contextual appropriateness, making it useful in creative writing and entertainment across various media and contexts. Although pun generation has received considerable attention in computational linguistics, there is currently no dedicated survey that systematically reviews this specific area. To bridge this gap, this paper provides a comprehensive review of pun generation datasets and methods across different stages, including conventional approaches, deep learning techniques, and pre-trained language models. Additionally, we summarise both automated and human evaluation metrics used to assess the quality of pun generation. Finally, we discuss the research challenges and propose promising directions for future work. 

---
# Reason to Rote: Rethinking Memorization in Reasoning 

**Authors**: Yupei Du, Philipp Mondorf, Silvia Casola, Yuekun Yao, Robert Litschko, Barbara Plank  

**Link**: [PDF](https://arxiv.org/pdf/2507.04782)  

**Abstract**: Large language models readily memorize arbitrary training instances, such as label noise, yet they perform strikingly well on reasoning tasks. In this work, we investigate how language models memorize label noise, and why such memorization in many cases does not heavily affect generalizable reasoning capabilities. Using two controllable synthetic reasoning datasets with noisy labels, four-digit addition (FDA) and two-hop relational reasoning (THR), we discover a reliance of memorization on generalizable reasoning mechanisms: models continue to compute intermediate reasoning outputs even when retrieving memorized noisy labels, and intervening reasoning adversely affects memorization. We further show that memorization operates through distributed encoding, i.e., aggregating various inputs and intermediate results, rather than building a look-up mechanism from inputs to noisy labels. Moreover, our FDA case study reveals memorization occurs via outlier heuristics, where existing neuron activation patterns are slightly shifted to fit noisy labels. Together, our findings suggest that memorization of label noise in language models builds on, rather than overrides, the underlying reasoning mechanisms, shedding lights on the intriguing phenomenon of benign memorization. 

---
# CoSteer: Collaborative Decoding-Time Personalization via Local Delta Steering 

**Authors**: Hang Lv, Sheng Liang, Hao Wang, Hongchao Gu, Yaxiong Wu, Wei Guo, Defu Lian, Yong Liu, Enhong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.04756)  

**Abstract**: Personalized text generation has become crucial for adapting language models to diverse and evolving users' personal context across cultural, temporal, and contextual dimensions. While existing methods often rely on centralized fine-tuning or static preference alignment, they struggle to achieve real-time adaptation under resource constraints inherent to personal devices. This limitation creates a dilemma: large cloud-based models lack access to localized user-specific information, while small on-device models cannot match the generation quality of their cloud counterparts. To address this dichotomy, we present CoSteer, a novel collaborative framework that enables decoding-time personalization through localized delta steering. Our key insight lies in leveraging the logits difference between personal context-aware and -agnostic outputs from local small models as steering signals for cloud-based LLMs. Specifically, we formulate token-level optimization as an online learning problem, where local delta vectors dynamically adjust the remote LLM's logits within the on-device environment. This approach preserves privacy by transmitting only the final steered tokens rather than raw data or intermediate vectors, while maintaining cloud-based LLMs' general capabilities without fine-tuning. Through comprehensive experiments on various personalized generation tasks, we demonstrate that CoSteer effectively assists LLMs in generating personalized content by leveraging locally stored user profiles and histories, ensuring privacy preservation through on-device data processing while maintaining acceptable computational overhead. 

---
# LLMs as Architects and Critics for Multi-Source Opinion Summarization 

**Authors**: Anuj Attri, Arnav Attri, Pushpak Bhattacharyya, Suman Banerjee, Amey Patil, Muthusamy Chelliah, Nikesh Garera  

**Link**: [PDF](https://arxiv.org/pdf/2507.04751)  

**Abstract**: Multi-source Opinion Summarization (M-OS) extends beyond traditional opinion summarization by incorporating additional sources of product metadata such as descriptions, key features, specifications, and ratings, alongside reviews. This integration results in comprehensive summaries that capture both subjective opinions and objective product attributes essential for informed decision-making. While Large Language Models (LLMs) have shown significant success in various Natural Language Processing (NLP) tasks, their potential in M-OS remains largely unexplored. Additionally, the lack of evaluation datasets for this task has impeded further advancements. To bridge this gap, we introduce M-OS-EVAL, a benchmark dataset for evaluating multi-source opinion summaries across 7 key dimensions: fluency, coherence, relevance, faithfulness, aspect coverage, sentiment consistency, specificity. Our results demonstrate that M-OS significantly enhances user engagement, as evidenced by a user study in which, on average, 87% of participants preferred M-OS over opinion summaries. Our experiments demonstrate that factually enriched summaries enhance user engagement. Notably, M-OS-PROMPTS exhibit stronger alignment with human judgment, achieving an average Spearman correlation of \r{ho} = 0.74, which surpasses the performance of previous methodologies. 

---
# A Tale of Two Scripts: Transliteration and Post-Correction for Judeo-Arabic 

**Authors**: Juan Moreno Gonzalez, Bashar Alhafni, Nizar Habash  

**Link**: [PDF](https://arxiv.org/pdf/2507.04746)  

**Abstract**: Judeo-Arabic refers to Arabic variants historically spoken by Jewish communities across the Arab world, primarily during the Middle Ages. Unlike standard Arabic, it is written in Hebrew script by Jewish writers and for Jewish audiences. Transliterating Judeo-Arabic into Arabic script is challenging due to ambiguous letter mappings, inconsistent orthographic conventions, and frequent code-switching into Hebrew and Aramaic. In this paper, we introduce a two-step approach to automatically transliterate Judeo-Arabic into Arabic script: simple character-level mapping followed by post-correction to address grammatical and orthographic errors. We also present the first benchmark evaluation of LLMs on this task. Finally, we show that transliteration enables Arabic NLP tools to perform morphosyntactic tagging and machine translation, which would have not been feasible on the original texts. 

---
# Word stress in self-supervised speech models: A cross-linguistic comparison 

**Authors**: Martijn Bentum, Louis ten Bosch, Tomas O. Lentz  

**Link**: [PDF](https://arxiv.org/pdf/2507.04738)  

**Abstract**: In this paper we study word stress representations learned by self-supervised speech models (S3M), specifically the Wav2vec 2.0 model. We investigate the S3M representations of word stress for five different languages: Three languages with variable or lexical stress (Dutch, English and German) and two languages with fixed or demarcative stress (Hungarian and Polish). We train diagnostic stress classifiers on S3M embeddings and show that they can distinguish between stressed and unstressed syllables in read-aloud short sentences with high accuracy. We also tested language-specificity effects of S3M word stress. The results indicate that the word stress representations are language-specific, with a greater difference between the set of variable versus the set of fixed stressed languages. 

---
# "This Suits You the Best": Query Focused Comparative Explainable Summarization 

**Authors**: Arnav Attri, Anuj Attri, Pushpak Bhattacharyya, Suman Banerjee, Amey Patil, Muthusamy Chelliah, Nikesh Garera  

**Link**: [PDF](https://arxiv.org/pdf/2507.04733)  

**Abstract**: Product recommendations inherently involve comparisons, yet traditional opinion summarization often fails to provide holistic comparative insights. We propose the novel task of generating Query-Focused Comparative Explainable Summaries (QF-CES) using Multi-Source Opinion Summarization (M-OS). To address the lack of query-focused recommendation datasets, we introduce MS-Q2P, comprising 7,500 queries mapped to 22,500 recommended products with metadata. We leverage Large Language Models (LLMs) to generate tabular comparative summaries with query-specific explanations. Our approach is personalized, privacy-preserving, recommendation engine-agnostic, and category-agnostic. M-OS as an intermediate step reduces inference latency approximately by 40% compared to the direct input approach (DIA), which processes raw data directly. We evaluate open-source and proprietary LLMs for generating and assessing QF-CES. Extensive evaluations using QF-CES-PROMPT across 5 dimensions (clarity, faithfulness, informativeness, format adherence, and query relevance) showed an average Spearman correlation of 0.74 with human judgments, indicating its potential for QF-CES evaluation. 

---
# LOOM-Scope: a comprehensive and efficient LOng-cOntext Model evaluation framework 

**Authors**: Zecheng Tang, Haitian Wang, Quantong Qiu, Baibei Ji, Ruoxi Sun, Keyan Zhou, Juntao Li, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.04723)  

**Abstract**: Long-context processing has become a fundamental capability for large language models~(LLMs). To assess model's long-context performance, numerous long-context evaluation benchmarks have been proposed. However, variations in evaluation settings across these benchmarks lead to inconsistent results, making it difficult to draw reliable comparisons. Besides, the high computational cost of long-context evaluation poses a significant barrier for the community to conduct comprehensive assessments of long-context models. In this paper, we propose LOOM-Scope, a comprehensive and efficient framework for long-context evaluation. LOOM-Scope standardizes evaluation settings across diverse benchmarks, supports deployment of efficient long-context inference acceleration methods, and introduces a holistic yet lightweight benchmark suite to evaluate models comprehensively. Homepage: this https URL 

---
# Why We Feel What We Feel: Joint Detection of Emotions and Their Opinion Triggers in E-commerce 

**Authors**: Arnav Attri, Anuj Attri, Pushpak Bhattacharyya, Suman Banerjee, Amey Patil, Muthusamy Chelliah, Nikesh Garera  

**Link**: [PDF](https://arxiv.org/pdf/2507.04708)  

**Abstract**: Customer reviews on e-commerce platforms capture critical affective signals that drive purchasing decisions. However, no existing research has explored the joint task of emotion detection and explanatory span identification in e-commerce reviews - a crucial gap in understanding what triggers customer emotional responses. To bridge this gap, we propose a novel joint task unifying Emotion detection and Opinion Trigger extraction (EOT), which explicitly models the relationship between causal text spans (opinion triggers) and affective dimensions (emotion categories) grounded in Plutchik's theory of 8 primary emotions. In the absence of labeled data, we introduce EOT-X, a human-annotated collection of 2,400 reviews with fine-grained emotions and opinion triggers. We evaluate 23 Large Language Models (LLMs) and present EOT-DETECT, a structured prompting framework with systematic reasoning and self-reflection. Our framework surpasses zero-shot and chain-of-thought techniques, across e-commerce domains. 

---
# XiYan-SQL: A Novel Multi-Generator Framework For Text-to-SQL 

**Authors**: Yifu Liu, Yin Zhu, Yingqi Gao, Zhiling Luo, Xiaoxia Li, Xiaorong Shi, Yuntao Hong, Jinyang Gao, Yu Li, Bolin Ding, Jingren Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2507.04701)  

**Abstract**: To leverage the advantages of LLM in addressing challenges in the Text-to-SQL task, we present XiYan-SQL, an innovative framework effectively generating and utilizing multiple SQL candidates. It consists of three components: 1) a Schema Filter module filtering and obtaining multiple relevant schemas; 2) a multi-generator ensemble approach generating multiple highquality and diverse SQL queries; 3) a selection model with a candidate reorganization strategy implemented to obtain the optimal SQL query. Specifically, for the multi-generator ensemble, we employ a multi-task fine-tuning strategy to enhance the capabilities of SQL generation models for the intrinsic alignment between SQL and text, and construct multiple generation models with distinct generation styles by fine-tuning across different SQL formats. The experimental results and comprehensive analysis demonstrate the effectiveness and robustness of our framework. Overall, XiYan-SQL achieves a new SOTA performance of 75.63% on the notable BIRD benchmark, surpassing all previous methods. It also attains SOTA performance on the Spider test set with an accuracy of 89.65%. 

---
# R1-RE: Cross-Domain Relationship Extraction with RLVR 

**Authors**: Runpeng Dai, Tong Zheng, Run Yang, Hongtu Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2507.04642)  

**Abstract**: Relationship extraction (RE) is a core task in natural language processing. Traditional approaches typically frame RE as a supervised learning problem, directly mapping context to labels-an approach that often suffers from poor out-of-domain (OOD) generalization. Inspired by the workflow of human annotators, we reframe RE as a reasoning task guided by annotation guidelines and introduce R1-RE, the first reinforcement learning with verifiable reward (RLVR) framework for RE tasks. Our method elicits the reasoning abilities of small language models for annotation tasks, resulting in significantly improved OOD robustness. We evaluate our approach on the public Sem-2010 dataset and a private MDKG dataset. The R1-RE-7B model attains an average OOD accuracy of approximately 70%, on par with leading proprietary models such as GPT-4o. Additionally, our comprehensive analysis provides novel insights into the training dynamics and emergent reasoning behaviors of the RLVR paradigm for RE. 

---
# Put Teacher in Student's Shoes: Cross-Distillation for Ultra-compact Model Compression Framework 

**Authors**: Maolin Wang, Jun Chu, Sicong Xie, Xiaoling Zang, Yao Zhao, Wenliang Zhong, Xiangyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2507.04636)  

**Abstract**: In the era of mobile computing, deploying efficient Natural Language Processing (NLP) models in resource-restricted edge settings presents significant challenges, particularly in environments requiring strict privacy compliance, real-time responsiveness, and diverse multi-tasking capabilities. These challenges create a fundamental need for ultra-compact models that maintain strong performance across various NLP tasks while adhering to stringent memory constraints. To this end, we introduce Edge ultra-lIte BERT framework (EI-BERT) with a novel cross-distillation method. EI-BERT efficiently compresses models through a comprehensive pipeline including hard token pruning, cross-distillation and parameter quantization. Specifically, the cross-distillation method uniquely positions the teacher model to understand the student model's perspective, ensuring efficient knowledge transfer through parameter integration and the mutual interplay between models. Through extensive experiments, we achieve a remarkably compact BERT-based model of only 1.91 MB - the smallest to date for Natural Language Understanding (NLU) tasks. This ultra-compact model has been successfully deployed across multiple scenarios within the Alipay ecosystem, demonstrating significant improvements in real-world applications. For example, it has been integrated into Alipay's live Edge Recommendation system since January 2024, currently serving the app's recommendation traffic across \textbf{8.4 million daily active devices}. 

---
# Knowledge-Aware Self-Correction in Language Models via Structured Memory Graphs 

**Authors**: Swayamjit Saha  

**Link**: [PDF](https://arxiv.org/pdf/2507.04625)  

**Abstract**: Large Language Models (LLMs) are powerful yet prone to generating factual errors, commonly referred to as hallucinations. We present a lightweight, interpretable framework for knowledge-aware self-correction of LLM outputs using structured memory graphs based on RDF triples. Without retraining or fine-tuning, our method post-processes model outputs and corrects factual inconsistencies via external semantic memory. We demonstrate the approach using DistilGPT-2 and show promising results on simple factual prompts. 

---
# Retain or Reframe? A Computational Framework for the Analysis of Framing in News Articles and Reader Comments 

**Authors**: Matteo Guida, Yulia Otmakhova, Eduard Hovy, Lea Frermann  

**Link**: [PDF](https://arxiv.org/pdf/2507.04612)  

**Abstract**: When a news article describes immigration as an "economic burden" or a "humanitarian crisis," it selectively emphasizes certain aspects of the issue. Although \textit{framing} shapes how the public interprets such issues, audiences do not absorb frames passively but actively reorganize the presented information. While this relationship between source content and audience response is well-documented in the social sciences, NLP approaches often ignore it, detecting frames in articles and responses in isolation. We present the first computational framework for large-scale analysis of framing across source content (news articles) and audience responses (reader comments). Methodologically, we refine frame labels and develop a framework that reconstructs dominant frames in articles and comments from sentence-level predictions, and aligns articles with topically relevant comments. Applying our framework across eleven topics and two news outlets, we find that frame reuse in comments correlates highly across outlets, while topic-specific patterns vary. We release a frame classifier that performs well on both articles and comments, a dataset of article and comment sentences manually labeled for frames, and a large-scale dataset of articles and comments with predicted frame labels. 

---
# PRIME: Large Language Model Personalization with Cognitive Memory and Thought Processes 

**Authors**: Xinliang Frederick Zhang, Nick Beauchamp, Lu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.04607)  

**Abstract**: Large language model (LLM) personalization aims to align model outputs with individuals' unique preferences and opinions. While recent efforts have implemented various personalization methods, a unified theoretical framework that can systematically understand the drivers of effective personalization is still lacking. In this work, we integrate the well-established cognitive dual-memory model into LLM personalization, by mirroring episodic memory to historical user engagements and semantic memory to long-term, evolving user beliefs. Specifically, we systematically investigate memory instantiations and introduce a unified framework, PRIME, using episodic and semantic memory mechanisms. We further augment PRIME with a novel personalized thinking capability inspired by the slow thinking strategy. Moreover, recognizing the absence of suitable benchmarks, we introduce a dataset using Change My View (CMV) from Reddit, specifically designed to evaluate long-context personalization. Extensive experiments validate PRIME's effectiveness across both long- and short-context scenarios. Further analysis confirms that PRIME effectively captures dynamic personalization beyond mere popularity biases. 

---
# Nile-Chat: Egyptian Language Models for Arabic and Latin Scripts 

**Authors**: Guokan Shang, Hadi Abdine, Ahmad Chamma, Amr Mohamed, Mohamed Anwar, Abdelaziz Bounhar, Omar El Herraoui, Preslav Nakov, Michalis Vazirgiannis, Eric Xing  

**Link**: [PDF](https://arxiv.org/pdf/2507.04569)  

**Abstract**: We introduce Nile-Chat-4B, 3x4B-A6B, and 12B, a collection of LLMs for Egyptian dialect, uniquely designed to understand and generate texts written in both Arabic and Latin scripts. Specifically, with Nile-Chat-3x4B-A6B, we introduce a novel language adaptation approach by leveraging the Branch-Train-MiX strategy to merge script-specialized experts, into a single MoE model. Our Nile-Chat models significantly outperform leading multilingual and Arabic LLMs, such as LLaMa, Jais, and ALLaM, on our newly introduced Egyptian evaluation benchmarks, which span both understanding and generative tasks. Notably, our 12B model yields a 14.4% performance gain over Qwen2.5-14B-Instruct on Latin-script benchmarks. All our resources are publicly available. We believe this work presents a comprehensive methodology for adapting LLMs to dual-script languages, addressing an often overlooked aspect in modern LLM development. 

---
# DP-Fusion: Token-Level Differentially Private Inference for Large Language Models 

**Authors**: Rushil Thareja, Preslav Nakov, Praneeth Vepakomma, Nils Lukas  

**Link**: [PDF](https://arxiv.org/pdf/2507.04531)  

**Abstract**: Large language models (LLMs) can leak sensitive information from their context through generated outputs, either accidentally or when prompted adversarially. Existing defenses that aim to preserve context privacy during inference either lack formal guarantees or suffer from a poor utility/privacy trade-off. We propose DP-Fusion, a token-level Differentially Private Inference (DPI) mechanism that provably bounds how much an LLM's outputs reveal about sensitive tokens in its context. We demonstrate DPI through the task of document privatization, where the goal is to paraphrase documents so that sensitive content (e.g., Personally Identifiable Information, PII) cannot be reliably inferred, while still preserving the overall utility of the text. This is controlled by a parameter $\epsilon$: $\epsilon=0$ hides PII entirely, while higher values trade off privacy for improved paraphrase quality. DP-Fusion works as follows: (i) partition sensitive tokens into disjoint privacy groups, (ii) run the LLM once per group, and (iii) blend the output distributions so that the final output remains within a fixed statistical distance of the baseline distribution produced when no privacy group is revealed. This approach allows fine-grained control over the privacy/utility trade-off but requires multiple LLM forward passes. 

---
# AdS: Adapter-state Sharing Framework for Multimodal Sarcasm Detection 

**Authors**: Soumyadeep Jana, Sahil Danayak, Sanasam Ranbir Singh  

**Link**: [PDF](https://arxiv.org/pdf/2507.04508)  

**Abstract**: The growing prevalence of multimodal image-text sarcasm on social media poses challenges for opinion mining, especially under resource constraints. Existing approaches rely on full fine-tuning of large pre-trained models, making them unsuitable for low-resource settings. While recent parameter-efficient fine-tuning (PEFT) methods offer promise, their off-the-shelf use underperforms on complex tasks like sarcasm detection. We propose AdS (Adapter-State Sharing), a lightweight framework built on CLIP that inserts adapters only in the upper layers and introduces a novel adapter-state sharing mechanism, where textual adapters guide visual ones. This design promotes efficient cross-modal learning while preserving low-level unimodal representations. Experiments on two public benchmarks demonstrate that AdS achieves state-of-the-art results using significantly fewer trainable parameters than existing PEFT and full fine-tuning approaches. 

---
# Unveiling the Potential of Diffusion Large Language Model in Controllable Generation 

**Authors**: Zhen Xiong, Yujun Cai, Zhecheng Li, Yiwei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.04504)  

**Abstract**: Diffusion models, originally developed for image generation, have emerged as a promising alternative to autoregressive large language models (LLMs). We present a theoretical analysis comparing autoregressive and masked diffusion LLMs, revealing that the intrinsic bidirectional attention mechanism of diffusion LLMs (dLLMs) enables superior context modeling and generation controllability. However, existing dLLM applications face significant challenges in controllable generation: the native multi-step denoising process exhibits high sensitivity to sequence length, elevated hallucination rates, and prohibitive inference costs without specialized optimizations. To address these limitations, we propose \textbf{S}elf-adaptive \textbf{S}chema \textbf{S}caffolding ($S^3$), a novel framework that enables dLLMs to generate structured outputs (e.g., JSON) while maintaining semantic fidelity and accelerating inference. Our approach injects the target schema structure into the output context, reducing unnecessary computation while improving controllability. Extensive experiments demonstrate that $S^3$ achieves substantial improvements: 65\% increase in structural adherence, 48\% enhancement in content fidelity, and 17\% reduction in hallucination rates compared to baseline. These results establish both theoretical foundations and practical pathways for deploying diffusion models in controllable text generation tasks. Code and data will be publicly released. 

---
# Dual Modality-Aware Gated Prompt Tuning for Few-Shot Multimodal Sarcasm Detection 

**Authors**: Soumyadeep Jana, Abhrajyoti Kundu, Sanasam Ranbir Singh  

**Link**: [PDF](https://arxiv.org/pdf/2507.04468)  

**Abstract**: The widespread use of multimodal content on social media has heightened the need for effective sarcasm detection to improve opinion mining. However, existing models rely heavily on large annotated datasets, making them less suitable for real-world scenarios where labeled data is scarce. This motivates the need to explore the problem in a few-shot setting. To this end, we introduce DMDP (Deep Modality-Disentangled Prompt Tuning), a novel framework for few-shot multimodal sarcasm detection. Unlike prior methods that use shallow, unified prompts across modalities, DMDP employs gated, modality-specific deep prompts for text and visual encoders. These prompts are injected across multiple layers to enable hierarchical feature learning and better capture diverse sarcasm types. To enhance intra-modal learning, we incorporate a prompt-sharing mechanism across layers, allowing the model to aggregate both low-level and high-level semantic cues. Additionally, a cross-modal prompt alignment module enables nuanced interactions between image and text representations, improving the model's ability to detect subtle sarcastic intent. Experiments on two public datasets demonstrate DMDP's superior performance in both few-shot and extremely low-resource settings. Further cross-dataset evaluations show that DMDP generalizes well across domains, consistently outperforming baseline methods. 

---
# Think Twice Before You Judge: Mixture of Dual Reasoning Experts for Multimodal Sarcasm Detection 

**Authors**: Soumyadeep Jana, Abhrajyoti Kundu, Sanasam Ranbir Singh  

**Link**: [PDF](https://arxiv.org/pdf/2507.04458)  

**Abstract**: Multimodal sarcasm detection has attracted growing interest due to the rise of multimedia posts on social media. Understanding sarcastic image-text posts often requires external contextual knowledge, such as cultural references or commonsense reasoning. However, existing models struggle to capture the deeper rationale behind sarcasm, relying mainly on shallow cues like image captions or object-attribute pairs from images. To address this, we propose \textbf{MiDRE} (\textbf{Mi}xture of \textbf{D}ual \textbf{R}easoning \textbf{E}xperts), which integrates an internal reasoning expert for detecting incongruities within the image-text pair and an external reasoning expert that utilizes structured rationales generated via Chain-of-Thought prompting to a Large Vision-Language Model. An adaptive gating mechanism dynamically weighs the two experts, selecting the most relevant reasoning path. Experiments on two benchmark datasets show that MiDRE achieves superior performance over baselines. Various qualitative analyses highlight the crucial role of external rationales, revealing that even when they are occasionally noisy, they provide valuable cues that guide the model toward a better understanding of sarcasm. 

---
# GradOT: Training-free Gradient-preserving Offsite-tuning for Large Language Models 

**Authors**: Kai Yao, Zhaorui Tan, Penglei Gao, Lichun Li, Kaixin Wu, Yinggui Wang, Yuan Zhao, Yixin Ji, Wei Wang, Jianke Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2507.04455)  

**Abstract**: The rapid growth of large language models (LLMs) with traditional centralized fine-tuning emerges as a key technique for adapting these models to domain-specific challenges, yielding privacy risks for both model and data owners. One promising solution, called offsite-tuning (OT), is proposed to address these challenges, where a weaker emulator is compressed from the original model and further fine-tuned with adapter to enhance privacy. However, the existing OT-based methods require high computational costs and lack theoretical analysis. This paper introduces a novel OT approach based on gradient-preserving compression, named GradOT. By analyzing the OT problem through the lens of optimization, we propose a method that selectively applies compression techniques such as rank compression and channel pruning, preserving the gradients of fine-tuned adapters while ensuring privacy. Extensive experiments demonstrate that our approach surpasses existing OT methods, both in terms of privacy protection and model performance. Our method provides a theoretical foundation for OT and offers a practical, training-free solution for offsite-tuning of large-scale LLMs. 

---
# RAT: Bridging RNN Efficiency and Attention Accuracy in Language Modeling 

**Authors**: Xiuying Wei, Anunay Yadav, Razvan Pascanu, Caglar Gulcehre  

**Link**: [PDF](https://arxiv.org/pdf/2507.04416)  

**Abstract**: Transformers have become the cornerstone of modern large-scale language models; however, their dependence on softmax attention poses a major computational bottleneck, particularly in long-context settings. In this work, rather than following prevalent approaches such as linear attention (or SSMs) and local attention, we introduce an intermediate design called \rat between recurrence and attention mechanisms. It partitions the input into chunks, applies a simple linear recurrence within each chunk to capture local dependencies, and then performs softmax attention across chunks to model long-range interactions. By adjusting the size of the chunk, \rat enables flexible trade-offs, combining the strengths of RNN and attention. Empirically, with a chunk size of 16, the \rat layer achieves a \(7\times\) improvement in training speed with 100K token sequences and \(9\times\) in generation at 4K sequence length, while maintaining similar or sometimes even better accuracy compared to standard attention. We demonstrate this by training 1.3B parameter models from scratch and performing large-scale evaluations, including short- and long-context benchmarks, as well as supervised fine-tuning~(SFT). We further propose a hybrid architecture that interleaves \rat with local attention. By combining efficient long-range modeling with strong local interactions, this hybrid design not only improves inference speed and reduces cache memory usage compared to attention, but also consistently enhances performance, for example, achieving an average 1 point gain in commonsense reasoning tasks, up to 4 points on code tasks, and a 1 point Rouge-L increase in a summarization SFT task. Code is available at this https URL 

---
# MOMENTS: A Comprehensive Multimodal Benchmark for Theory of Mind 

**Authors**: Emilio Villa-Cueva, S M Masrur Ahmed, Rendi Chevi, Jan Christian Blaise Cruz, Kareem Elzeky, Fermin Cristobal, Alham Fikri Aji, Skyler Wang, Rada Mihalcea, Thamar Solorio  

**Link**: [PDF](https://arxiv.org/pdf/2507.04415)  

**Abstract**: Understanding Theory of Mind is essential for building socially intelligent multimodal agents capable of perceiving and interpreting human behavior. We introduce MOMENTS (Multimodal Mental States), a comprehensive benchmark designed to assess the ToM capabilities of multimodal large language models (LLMs) through realistic, narrative-rich scenarios presented in short films. MOMENTS includes over 2,344 multiple-choice questions spanning seven distinct ToM categories. The benchmark features long video context windows and realistic social interactions that provide deeper insight into characters' mental states. While the visual modality generally enhances model performance, current systems still struggle to integrate it effectively, underscoring the need for further research into AI's multimodal understanding of human behavior. 

---
# THM@SimpleText 2025 -- Task 1.1: Revisiting Text Simplification based on Complex Terms for Non-Experts 

**Authors**: Nico Hofmann, Julian Dauenhauer, Nils Ole Dietzler, Idehen Daniel Idahor, Christin Katharina Kreutz  

**Link**: [PDF](https://arxiv.org/pdf/2507.04414)  

**Abstract**: Scientific text is complex as it contains technical terms by definition. Simplifying such text for non-domain experts enhances accessibility of innovation and information. Politicians could be enabled to understand new findings on topics on which they intend to pass a law, or family members of seriously ill patients could read about clinical trials. The SimpleText CLEF Lab focuses on exactly this problem of simplification of scientific text. Task 1.1 of the 2025 edition specifically handles the simplification of complex sentences, so very short texts with little context. To tackle this task we investigate the identification of complex terms in sentences which are rephrased using small Gemini and OpenAI large language models for non-expert readers. 

---
# SpiritRAG: A Q&A System for Religion and Spirituality in the United Nations Archive 

**Authors**: Yingqiang Gao, Fabian Winiger, Patrick Montjourides, Anastassia Shaitarova, Nianlong Gu, Simon Peng-Keller, Gerold Schneider  

**Link**: [PDF](https://arxiv.org/pdf/2507.04395)  

**Abstract**: Religion and spirituality (R/S) are complex and highly domain-dependent concepts which have long confounded researchers and policymakers. Due to their context-specificity, R/S are difficult to operationalize in conventional archival search strategies, particularly when datasets are very large, poorly accessible, and marked by information noise. As a result, considerable time investments and specialist knowledge is often needed to extract actionable insights related to R/S from general archival sources, increasing reliance on published literature and manual desk reviews. To address this challenge, we present SpiritRAG, an interactive Question Answering (Q&A) system based on Retrieval-Augmented Generation (RAG). Built using 7,500 United Nations (UN) resolution documents related to R/S in the domains of health and education, SpiritRAG allows researchers and policymakers to conduct complex, context-sensitive database searches of very large datasets using an easily accessible, chat-based web interface. SpiritRAG is lightweight to deploy and leverages both UN documents and user provided documents as source material. A pilot test and evaluation with domain experts on 100 manually composed questions demonstrates the practical value and usefulness of SpiritRAG. 

---
# Does Learning Mathematical Problem-Solving Generalize to Broader Reasoning? 

**Authors**: Ruochen Zhou, Minrui Xu, Shiqi Chen, Junteng Liu, Yunqi Li, Xinxin Lin, Zhengyu Chen, Junxian He  

**Link**: [PDF](https://arxiv.org/pdf/2507.04391)  

**Abstract**: There has been a growing interest in enhancing the mathematical problem-solving (MPS) capabilities of large language models. While the majority of research efforts concentrate on creating specialized models to solve mathematical problems, it remains unknown how learning mathematical problem-solving generalizes to help develop other reasoning abilities. In this paper, we present an empirical investigation into the generalization potential of various MPS training approaches, such as continual pretraining, instruction tuning, and rule-based reinforcement learning across various data sources, including both short and long chain-of-thought (CoT) samples. Evaluation on 5 mathematical and 8 general reasoning benchmarks show that continual pretraining on math text is able to generalize to general reasoning tasks to some extent. In constrast, instruction tuning on conventional, short MPS samples provides limited benefits and, in many cases, even impairs generalization performance. Notably, training with long CoT responses for MPS samples and incorporating rule-based reinforcement learning on MPS queries exhibit distinct behavior, significantly enhancing generalization by extending the model's reasoning processes into other domains. These results suggest that traditional approaches to learning MPS with short reasoning chains largely fail to achieve robust generalization. However, the emerging paradigm of longer reasoning chains, coupled with self-reflection, offers a promising direction for improving generalized reasoning abilities through learning from specialized domains. 

---
# Large Language Models' Varying Accuracy in Recognizing Risk-Promoting and Health-Supporting Sentiments in Public Health Discourse: The Cases of HPV Vaccination and Heated Tobacco Products 

**Authors**: Soojong Kim, Kwanho Kim, Hye Min Kim  

**Link**: [PDF](https://arxiv.org/pdf/2507.04364)  

**Abstract**: Machine learning methods are increasingly applied to analyze health-related public discourse based on large-scale data, but questions remain regarding their ability to accurately detect different types of health sentiments. Especially, Large Language Models (LLMs) have gained attention as a powerful technology, yet their accuracy and feasibility in capturing different opinions and perspectives on health issues are largely unexplored. Thus, this research examines how accurate the three prominent LLMs (GPT, Gemini, and LLAMA) are in detecting risk-promoting versus health-supporting sentiments across two critical public health topics: Human Papillomavirus (HPV) vaccination and heated tobacco products (HTPs). Drawing on data from Facebook and Twitter, we curated multiple sets of messages supporting or opposing recommended health behaviors, supplemented with human annotations as the gold standard for sentiment classification. The findings indicate that all three LLMs generally demonstrate substantial accuracy in classifying risk-promoting and health-supporting sentiments, although notable discrepancies emerge by platform, health issue, and model type. Specifically, models often show higher accuracy for risk-promoting sentiment on Facebook, whereas health-supporting messages on Twitter are more accurately detected. An additional analysis also shows the challenges LLMs face in reliably detecting neutral messages. These results highlight the importance of carefully selecting and validating language models for public health analyses, particularly given potential biases in training data that may lead LLMs to overestimate or underestimate the prevalence of certain perspectives. 

---
# HatePRISM: Policies, Platforms, and Research Integration. Advancing NLP for Hate Speech Proactive Mitigation 

**Authors**: Naquee Rizwan, Seid Muhie Yimam, Daryna Dementieva, Florian Skupin, Tim Fischer, Daniil Moskovskiy, Aarushi Ajay Borkar, Robert Geislinger, Punyajoy Saha, Sarthak Roy, Martin Semmann, Alexander Panchenko, Chris Biemann, Animesh Mukherjee  

**Link**: [PDF](https://arxiv.org/pdf/2507.04350)  

**Abstract**: Despite regulations imposed by nations and social media platforms, e.g. (Government of India, 2021; European Parliament and Council of the European Union, 2022), inter alia, hateful content persists as a significant challenge. Existing approaches primarily rely on reactive measures such as blocking or suspending offensive messages, with emerging strategies focusing on proactive measurements like detoxification and counterspeech. In our work, which we call HatePRISM, we conduct a comprehensive examination of hate speech regulations and strategies from three perspectives: country regulations, social platform policies, and NLP research datasets. Our findings reveal significant inconsistencies in hate speech definitions and moderation practices across jurisdictions and platforms, alongside a lack of alignment with research efforts. Based on these insights, we suggest ideas and research direction for further exploration of a unified framework for automated hate speech moderation incorporating diverse strategies. 

---
# No Language Data Left Behind: A Comparative Study of CJK Language Datasets in the Hugging Face Ecosystem 

**Authors**: Dasol Choi, Woomyoung Park, Youngsook Song  

**Link**: [PDF](https://arxiv.org/pdf/2507.04329)  

**Abstract**: Recent advances in Natural Language Processing (NLP) have underscored the crucial role of high-quality datasets in building large language models (LLMs). However, while extensive resources and analyses exist for English, the landscape for East Asian languages - particularly Chinese, Japanese, and Korean (CJK) - remains fragmented and underexplored, despite these languages together serving over 1.6 billion speakers. To address this gap, we investigate the HuggingFace ecosystem from a cross-linguistic perspective, focusing on how cultural norms, research environments, and institutional practices shape dataset availability and quality. Drawing on more than 3,300 datasets, we employ quantitative and qualitative methods to examine how these factors drive distinct creation and curation patterns across Chinese, Japanese, and Korean NLP communities. Our findings highlight the large-scale and often institution-driven nature of Chinese datasets, grassroots community-led development in Korean NLP, and an entertainment- and subculture-focused emphasis on Japanese collections. By uncovering these patterns, we reveal practical strategies for enhancing dataset documentation, licensing clarity, and cross-lingual resource sharing - ultimately guiding more effective and culturally attuned LLM development in East Asia. We conclude by discussing best practices for future dataset curation and collaboration, aiming to strengthen resource development across all three languages. 

---
# Fairness Evaluation of Large Language Models in Academic Library Reference Services 

**Authors**: Haining Wang, Jason Clark, Yueru Yan, Star Bradley, Ruiyang Chen, Yiqiong Zhang, Hengyi Fu, Zuoyu Tian  

**Link**: [PDF](https://arxiv.org/pdf/2507.04224)  

**Abstract**: As libraries explore large language models (LLMs) for use in virtual reference services, a key question arises: Can LLMs serve all users equitably, regardless of demographics or social status? While they offer great potential for scalable support, LLMs may also reproduce societal biases embedded in their training data, risking the integrity of libraries' commitment to equitable service. To address this concern, we evaluate whether LLMs differentiate responses across user identities by prompting six state-of-the-art LLMs to assist patrons differing in sex, race/ethnicity, and institutional role. We found no evidence of differentiation by race or ethnicity, and only minor evidence of stereotypical bias against women in one model. LLMs demonstrated nuanced accommodation of institutional roles through the use of linguistic choices related to formality, politeness, and domain-specific vocabularies, reflecting professional norms rather than discriminatory treatment. These findings suggest that current LLMs show a promising degree of readiness to support equitable and contextually appropriate communication in academic library reference services. 

---
# Context Tuning for In-Context Optimization 

**Authors**: Jack Lu, Ryan Teehan, Zhenbang Yang, Mengye Ren  

**Link**: [PDF](https://arxiv.org/pdf/2507.04221)  

**Abstract**: We introduce Context Tuning, a simple and effective method to significantly enhance few-shot adaptation of language models (LLMs) without fine-tuning model parameters. While prompt-based adaptation techniques have demonstrated the effectiveness of lightweight adaptation methods for large language models (LLMs), they typically initialize a trainable prompt or prefix with irrelevant tokens for the task at hand. In contrast, Context Tuning initializes the trainable prompt or prefix with task-specific demonstration examples, leveraging the model's inherent In-Context Learning (ICL) ability to extract relevant information for improved few-shot learning performance. Extensive evaluations on benchmarks such as CrossFit, UnifiedQA, MMLU, BIG-Bench Hard, and ARC demonstrate that Context Tuning outperforms traditional prompt-based adaptation methods and achieves competitive accuracy to Test-Time Training with significantly higher training efficiency. 

---
# SymbolicThought: Integrating Language Models and Symbolic Reasoning for Consistent and Interpretable Human Relationship Understanding 

**Authors**: Runcong Zhao, Qinglin Zhu, Hainiu Xu, Bin Liang, Yulan He, Lin Gui  

**Link**: [PDF](https://arxiv.org/pdf/2507.04189)  

**Abstract**: Understanding character relationships is essential for interpreting complex narratives and conducting socially grounded AI research. However, manual annotation is time-consuming and low in coverage, while large language models (LLMs) often produce hallucinated or logically inconsistent outputs. We present SymbolicThought, a human-in-the-loop framework that combines LLM-based extraction with symbolic reasoning. The system constructs editable character relationship graphs, refines them using seven types of logical constraints, and enables real-time validation and conflict resolution through an interactive interface. To support logical supervision and explainable social analysis, we release a dataset of 160 interpersonal relationships with corresponding logical structures. Experiments show that SymbolicThought improves annotation accuracy and consistency while significantly reducing time cost, offering a practical tool for narrative understanding, explainable AI, and LLM evaluation. 

---
# Large Language Models for Zero-Shot Multicultural Name Recognition 

**Authors**: Thanakorn Phonchai, Surasakdi Siripong, Nicholas Patterson, Owen Campbell  

**Link**: [PDF](https://arxiv.org/pdf/2507.04149)  

**Abstract**: The robust and accurate recognition of multicultural names, particularly those not previously encountered, is a critical challenge in an increasingly globalized digital landscape. Traditional methods often falter when confronted with the vast diversity and novel permutations of names across different linguistic and cultural backgrounds. This paper introduces a novel framework, Prompt-Engineered Fine-Tuning (PEFT) for Large Language Models (LLMs) with Adversarial Data Augmentation and Cultural Knowledge Graph Integration, designed to significantly enhance zero-shot multicultural name recognition. Our approach leverages the powerful linguistic understanding of pre-trained LLMs, transforming the recognition task into a guided generation problem. Through meticulous prompt engineering, dynamic integration of explicit cultural knowledge derived from knowledge graphs, and the strategic application of adversarial data augmentation, we equip the LLM with an unprecedented ability to infer the cultural origin of unseen names. Extensive experiments demonstrate that our PEFT method consistently outperforms established deep learning baselines, including advanced Bi-LSTM models with cultural tags, achieving an impressive 93.1\% overall accuracy and a remarkable 89.5\% accuracy on challenging zero-shot name identification. An in-depth ablation study confirms the synergistic contribution of each component, while a human evaluation highlights our method's performance approaching human expert judgment. This work signifies a substantial leap in multicultural name recognition, offering a highly effective and scalable solution for real-world applications. 

---
# Dissecting Clinical Reasoning in Language Models: A Comparative Study of Prompts and Model Adaptation Strategies 

**Authors**: Mael Jullien, Marco Valentino, Leonardo Ranaldi, Andre Freitas  

**Link**: [PDF](https://arxiv.org/pdf/2507.04142)  

**Abstract**: Recent works on large language models (LLMs) have demonstrated the impact of prompting strategies and fine-tuning techniques on their reasoning capabilities. Yet, their effectiveness on clinical natural language inference (NLI) remains underexplored. This study presents the first controlled evaluation of how prompt structure and efficient fine-tuning jointly shape model performance in clinical NLI. We inspect four classes of prompting strategies to elicit reasoning in LLMs at different levels of abstraction, and evaluate their impact on a range of clinically motivated reasoning types. For each prompting strategy, we construct high-quality demonstrations using a frontier model to distil multi-step reasoning capabilities into smaller models (4B parameters) via Low-Rank Adaptation (LoRA). Across different language models fine-tuned on the NLI4CT benchmark, we found that prompt type alone accounts for up to 44% of the variance in macro-F1. Moreover, LoRA fine-tuning yields consistent gains of +8 to 12 F1, raises output alignment above 97%, and narrows the performance gap to GPT-4o-mini to within 7.1%. Additional experiments on reasoning generalisation reveal that LoRA improves performance in 75% of the models on MedNLI and TREC Clinical Trials Track. Overall, these findings demonstrate that (i) prompt structure is a primary driver of clinical reasoning performance, (ii) compact models equipped with strong prompts and LoRA can rival frontier-scale systems, and (iii) reasoning-type-aware evaluation is essential to uncover prompt-induced trade-offs. Our results highlight the promise of combining prompt design and lightweight adaptation for more efficient and trustworthy clinical NLP systems, providing insights on the strengths and limitations of widely adopted prompting and parameter-efficient techniques in highly specialised domains. 

---
# Token Level Hallucination Detection via Variance in Language Models 

**Authors**: Keshav Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2507.04137)  

**Abstract**: Large Language Models (LLMs) have demonstrated impressive generative capabilities across diverse tasks but remain susceptible to hallucinations, confidently generated yet factually incorrect outputs. We introduce a reference-free, token-level hallucination detection framework that leverages the variance in token log-probabilities across multiple stochastic generations. Unlike prior methods that require ground-truth references or sentence-level verification, our approach is model-agnostic, interpretable, and suited for real-time or post-hoc analysis. We evaluate our method on unanswerable question prompts from the SQuAD v2 dataset and benchmark across three autoregressive models of varying scales: GPT-Neo 125M, Falcon 1B, and Mistral 7B. Through both quantitative metrics and visual diagnostics, we show that token-level variance reliably highlights instability in model outputs and correlates with hallucination patterns. Our framework is lightweight, reproducible, and adaptable to multiple domains, offering a valuable diagnostic tool for analyzing generative reliability in LLMs. 

---
# BYOKG-RAG: Multi-Strategy Graph Retrieval for Knowledge Graph Question Answering 

**Authors**: Costas Mavromatis, Soji Adeshina, Vassilis N. Ioannidis, Zhen Han, Qi Zhu, Ian Robinson, Bryan Thompson, Huzefa Rangwala, George Karypis  

**Link**: [PDF](https://arxiv.org/pdf/2507.04127)  

**Abstract**: Knowledge graph question answering (KGQA) presents significant challenges due to the structural and semantic variations across input graphs. Existing works rely on Large Language Model (LLM) agents for graph traversal and retrieval; an approach that is sensitive to traversal initialization, as it is prone to entity linking errors and may not generalize well to custom ("bring-your-own") KGs. We introduce BYOKG-RAG, a framework that enhances KGQA by synergistically combining LLMs with specialized graph retrieval tools. In BYOKG-RAG, LLMs generate critical graph artifacts (question entities, candidate answers, reasoning paths, and OpenCypher queries), and graph tools link these artifacts to the KG and retrieve relevant graph context. The retrieved context enables the LLM to iteratively refine its graph linking and retrieval, before final answer generation. By retrieving context from different graph tools, BYOKG-RAG offers a more general and robust solution for QA over custom KGs. Through experiments on five benchmarks spanning diverse KG types, we demonstrate that BYOKG-RAG outperforms the second-best graph retrieval method by 4.5% points while showing better generalization to custom KGs. BYOKG-RAG framework is open-sourced at this https URL. 

---
# Conversation Forests: The Key to Fine Tuning Large Language Models for Multi-Turn Medical Conversations is Branching 

**Authors**: Thomas Savage  

**Link**: [PDF](https://arxiv.org/pdf/2507.04099)  

**Abstract**: Fine-tuning methods such as Direct Preference Optimization (DPO) and Group Relative Policy Optimization (GRPO) have demonstrated success in training large language models (LLMs) for single-turn tasks. However, these methods fall short in multi-turn applications, such as diagnostic patient interviewing, where understanding how early conversational turns influence downstream completions and outcomes is essential. In medicine, a multi-turn perspective is critical for learning diagnostic schemas and better understanding conversation dynamics. To address this gap, I introduce Savage Conversation Forests (SCF), a reinforcement learning framework that leverages a branched conversation architecture to fine-tune LLMs for multi-turn dialogue. SCF generates multiple possible conversation continuations at each turn, enabling the model to learn how different early responses affect downstream interactions and diagnostic outcomes. In experiments simulating doctor-patient conversations, SCF with branching outperforms linear conversation architectures on diagnostic accuracy. I hypothesize that SCF's improvements stem from its ability to provide richer, interdependent training signals across conversation turns. These results suggest that a branched training architecture is an important strategy for fine tuning LLMs in complex multi-turn conversational tasks. 

---
# XISM: an eXploratory and Interactive Graph Tool to Visualize and Evaluate Semantic Map Models 

**Authors**: Zhu Liu, Zhen Hu, Lei Dai, Ying Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.04070)  

**Abstract**: Semantic map models represent meanings or functions as nodes in a graph constrained by the local connectivity hypothesis, with edges indicating their associations. Widely used in typological linguistics, these models compare interrelated meanings across languages. Traditionally built manually in a bottom-up manner, they are inefficient for large datasets and lack visualization and evaluation tools. This paper introduces XISM, an interactive tool based on our prior algorithm, which constructs semantic maps from user data via a top-down approach, displays candidate maps, and evaluates them using multiple metrics. Users can refine maps by editing edges, combining data-driven efficiency with expert knowledge. This human-in-the-loop design benefits both typologists and computational linguists. The system this https URL and a demonstration video this https URL are publicly available. 

---
# Beyond Independent Passages: Adaptive Passage Combination Retrieval for Retrieval Augmented Open-Domain Question Answering 

**Authors**: Ting-Wen Ko, Jyun-Yu Jiang, Pu-Jen Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2507.04069)  

**Abstract**: Retrieval-augmented generation (RAG) enhances large language models (LLMs) by incorporating external documents at inference time, enabling up-to-date knowledge access without costly retraining. However, conventional RAG methods retrieve passages independently, often leading to redundant, noisy, or insufficiently diverse context-particularly problematic - particularly problematic in noisy corpora and for multi-hop questions. To address this, we propose Adaptive Passage Combination Retrieval (AdaPCR), a novel framework for open-domain question answering with black-box LMs. AdaPCR explicitly models dependencies between passages by considering passage combinations as units for retrieval and reranking. It consists of a context-aware query reformulation using concatenated passages, and a reranking step trained with a predictive objective aligned with downstream answer likelihood. Crucially, AdaPCR adaptively selects the number of retrieved passages without additional stopping modules. Experiments across several QA benchmarks show that AdaPCR outperforms baselines, particularly in multi-hop reasoning, demonstrating the effectiveness of modeling inter-passage dependencies for improved retrieval. 

---
# Patient-Centered RAG for Oncology Visit Aid Following the Ottawa Decision Guide 

**Authors**: Siyang Liu, Lawrence Chin-I An, Rada Mihalcea  

**Link**: [PDF](https://arxiv.org/pdf/2507.04026)  

**Abstract**: Effective communication is essential in cancer care, yet patients often face challenges in preparing for complex medical visits. We present an interactive, Retrieval-augmented Generation-assisted system that helps patients progress from uninformed to visit-ready. Our system adapts the Ottawa Personal Decision Guide into a dynamic retrieval-augmented generation workflow, helping users bridge knowledge gaps, clarify personal values and generate useful questions for their upcoming visits. Focusing on localized prostate cancer, we conduct a user study with patients and a clinical expert. Results show high system usability (UMUX Mean = 6.0 out of 7), strong relevance of generated content (Mean = 6.7 out of 7), minimal need for edits, and high clinical faithfulness (Mean = 6.82 out of 7). This work demonstrates the potential of combining patient-centered design with language models to enhance clinical preparation in oncology care. 

---
# LLMThinkBench: Towards Basic Math Reasoning and Overthinking in Large Language Models 

**Authors**: Gaurav Srivastava, Aafiya Hussain, Sriram Srinivasan, Xuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.04023)  

**Abstract**: Large Language Models (LLMs) have achieved remarkable performance on complex mathematical benchmarks, yet often struggle with simple arithmetic tasks and exhibit a tendency toward over-explaining or "overthinking" answers. To systematically assess this phenomenon, we introduce LLMThinkBench, a modular benchmarking framework that enables researchers to evaluate basic math reasoning and overthinking in LLMs. The framework provides 14 configurable math tasks with randomized test data generation and robust parsing strategies. Researchers can quantify overthinking using our Overthinking Score metric, which captures accuracy-verbosity tradeoffs through harmonic mean formulation. The tool offers flexible evaluation with a scalable vLLM/Transformers backend, multi-GPU support, and full configurability. Users can extend the tool with custom tasks, reproduce experiments with seeding, and generate detailed efficiency reports. Distributed as a pip-installable package with CLI and API access, LLMThinkBench provides researchers and practitioners an accessible, cost-effective alternative to expensive LLM-as-a-judge methods for diagnosing basic reasoning capabilities and efficiency analysis. Package can be installed as: pip install llmthinkbench 

---
# Handling Korean Out-of-Vocabulary Words with Phoneme Representation Learning 

**Authors**: Nayeon Kim, Eojin Jeon, Jun-Hyung Park, SangKeun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2507.04018)  

**Abstract**: In this study, we introduce KOPL, a novel framework for handling Korean OOV words with Phoneme representation Learning. Our work is based on the linguistic property of Korean as a phonemic script, the high correlation between phonemes and letters. KOPL incorporates phoneme and word representations for Korean OOV words, facilitating Korean OOV word representations to capture both text and phoneme information of words. We empirically demonstrate that KOPL significantly improves the performance on Korean Natural Language Processing (NLP) tasks, while being readily integrated into existing static and contextual Korean embedding models in a plug-and-play manner. Notably, we show that KOPL outperforms the state-of-the-art model by an average of 1.9%. Our code is available at this https URL. 

---
# Nunchi-Bench: Benchmarking Language Models on Cultural Reasoning with a Focus on Korean Superstition 

**Authors**: Kyuhee Kim, Sangah Lee  

**Link**: [PDF](https://arxiv.org/pdf/2507.04014)  

**Abstract**: As large language models (LLMs) become key advisors in various domains, their cultural sensitivity and reasoning skills are crucial in multicultural environments. We introduce Nunchi-Bench, a benchmark designed to evaluate LLMs' cultural understanding, with a focus on Korean superstitions. The benchmark consists of 247 questions spanning 31 topics, assessing factual knowledge, culturally appropriate advice, and situational interpretation. We evaluate multilingual LLMs in both Korean and English to analyze their ability to reason about Korean cultural contexts and how language variations affect performance. To systematically assess cultural reasoning, we propose a novel evaluation strategy with customized scoring metrics that capture the extent to which models recognize cultural nuances and respond appropriately. Our findings highlight significant challenges in LLMs' cultural reasoning. While models generally recognize factual information, they struggle to apply it in practical scenarios. Furthermore, explicit cultural framing enhances performance more effectively than relying solely on the language of the prompt. To support further research, we publicly release Nunchi-Bench alongside a leaderboard. 

---
# Easy Dataset: A Unified and Extensible Framework for Synthesizing LLM Fine-Tuning Data from Unstructured Documents 

**Authors**: Ziyang Miao, Qiyu Sun, Jingyuan Wang, Yuchen Gong, Yaowei Zheng, Shiqi Li, Richong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.04009)  

**Abstract**: Large language models (LLMs) have shown impressive performance on general-purpose tasks, yet adapting them to specific domains remains challenging due to the scarcity of high-quality domain data. Existing data synthesis tools often struggle to extract reliable fine-tuning data from heterogeneous documents effectively. To address this limitation, we propose Easy Dataset, a unified framework for synthesizing fine-tuning data from unstructured documents via an intuitive graphical user interface (GUI). Specifically, Easy Dataset allows users to easily configure text extraction models and chunking strategies to transform raw documents into coherent text chunks. It then leverages a persona-driven prompting approach to generate diverse question-answer pairs using public-available LLMs. Throughout the pipeline, a human-in-the-loop visual interface facilitates the review and refinement of intermediate outputs to ensure data quality. Experiments on a financial question-answering task show that fine-tuning LLMs on the synthesized dataset significantly improves domain-specific performance while preserving general knowledge. The source code and installable package are available at this https URL and have garnered over 9,000 GitHub stars. 

---
# A Modular Unsupervised Framework for Attribute Recognition from Unstructured Text 

**Authors**: KMA Solaiman  

**Link**: [PDF](https://arxiv.org/pdf/2507.03949)  

**Abstract**: We propose POSID, a modular, lightweight and on-demand framework for extracting structured attribute-based properties from unstructured text without task-specific fine-tuning. While the method is designed to be adaptable across domains, in this work, we evaluate it on human attribute recognition in incident reports. POSID combines lexical and semantic similarity techniques to identify relevant sentences and extract attributes. We demonstrate its effectiveness on a missing person use case using the InciText dataset, achieving effective attribute extraction without supervised training. 

---
# Losing our Tail -- Again: On (Un)Natural Selection And Multilingual Large Language Models 

**Authors**: Eva Vanmassenhove  

**Link**: [PDF](https://arxiv.org/pdf/2507.03933)  

**Abstract**: Multilingual Large Language Models (LLMs) considerably changed how technologies can influence language. While previous technologies could mediate or assist humans, there is now a tendency to \textit{offload} the task of writing itself to these technologies, enabling them to change our linguistic ecosystem more directly. While they provide us quick access to information and impressively fluent output, beneath their apparent sophistication lies a subtle, more insidious threat: the gradual decline and loss of linguistic diversity. With this opinion piece, I explore how model collapse, with a particular focus on translation technology, can lead to the loss of linguistic forms, grammatical features, and cultural nuance. Model collapse refers to the eventual consequence of self-consuming training loops, where models reinforce their own biases and lose linguistic diversity. Drawing on recent work in Computer Vision, Natural Language Processing (NLP) and Machine Translation (MT), I argue that the tails of our linguistic distributions are vanishing, and with them, the narratives and identities they carry. This is a call to resist linguistic flattening and to reimagine NLP as a field that encourages, values and protects expressive multilingual lexical and linguistic diversity and creativity. 

---
# Dynamic Injection of Entity Knowledge into Dense Retrievers 

**Authors**: Ikuya Yamada, Ryokan Ri, Takeshi Kojima, Yusuke Iwasawa, Yutaka Matsuo  

**Link**: [PDF](https://arxiv.org/pdf/2507.03922)  

**Abstract**: Dense retrievers often struggle with queries involving less-frequent entities due to their limited entity knowledge. We propose the Knowledgeable Passage Retriever (KPR), a BERT-based retriever enhanced with a context-entity attention layer and dynamically updatable entity embeddings. This design enables KPR to incorporate external entity knowledge without retraining. Experiments on three datasets show that KPR consistently improves retrieval accuracy, achieving a substantial 12.6% gain on the EntityQuestions dataset over the model without KPR extensions. When built on the off-the-shelf bge-base retriever, KPR achieves state-of-the-art performance among similarly sized models on two datasets. Code and models will be released soon. 

---
# Demystifying ChatGPT: How It Masters Genre Recognition 

**Authors**: Subham Raj, Sriparna Saha, Brijraj Singh, Niranjan Pedanekar  

**Link**: [PDF](https://arxiv.org/pdf/2507.03875)  

**Abstract**: The introduction of ChatGPT has garnered significant attention within the NLP community and beyond. Previous studies have demonstrated ChatGPT's substantial advancements across various downstream NLP tasks, highlighting its adaptability and potential to revolutionize language-related applications. However, its capabilities and limitations in genre prediction remain unclear. This work analyzes three Large Language Models (LLMs) using the MovieLens-100K dataset to assess their genre prediction capabilities. Our findings show that ChatGPT, without fine-tuning, outperformed other LLMs, and fine-tuned ChatGPT performed best overall. We set up zero-shot and few-shot prompts using audio transcripts/subtitles from movie trailers in the MovieLens-100K dataset, covering 1682 movies of 18 genres, where each movie can have multiple genres. Additionally, we extended our study by extracting IMDb movie posters to utilize a Vision Language Model (VLM) with prompts for poster information. This fine-grained information was used to enhance existing LLM prompts. In conclusion, our study reveals ChatGPT's remarkable genre prediction capabilities, surpassing other language models. The integration of VLM further enhances our findings, showcasing ChatGPT's potential for content-related applications by incorporating visual information from movie posters. 

---
# OrthoRank: Token Selection via Sink Token Orthogonality for Efficient LLM inference 

**Authors**: Seungjun Shin, Jaehoon Oh, Dokwan Oh  

**Link**: [PDF](https://arxiv.org/pdf/2507.03865)  

**Abstract**: Attention mechanisms are central to the success of large language models (LLMs), enabling them to capture intricate token dependencies and implicitly assign importance to each token. Recent studies have revealed the sink token, which receives disproportionately high attention despite their limited semantic role. In this paper, we first expand the relationship between the sink token and other tokens, moving beyond attention to explore their similarity in hidden states, considering the layer depth. We observe that as the layers get deeper, the cosine similarity between the normalized hidden states of the sink token and those of other tokens increases, and that the normalized hidden states of the sink token exhibit negligible changes. These imply that other tokens consistently are directed toward the sink token throughout the layers. Next, we propose a dynamic token selection method, called OrthoRank, using these findings to select important tokens. Specifically, in a certain layer, we define token importance by the speed at which the token moves toward the sink token. This is converted into orthogonality with the sink token, meaning that tokens that are more orthogonal to the sink token are assigned greater importance. Finally, through extensive experiments, we demonstrated that our method results in lower perplexity and higher zero-shot accuracy compared to layer pruning methods at the same sparsity ratio with comparable throughput, while also achieving superior performance on LongBench. 

---
# Alpay Algebra IV: Symbiotic Semantics and the Fixed-Point Convergence of Observer Embeddings 

**Authors**: Bugra Kilictas, Faruk Alpay  

**Link**: [PDF](https://arxiv.org/pdf/2507.03774)  

**Abstract**: We present a theoretical framework in which a document and an AI model engage in a transfinite fixed-point interaction that leads to stable semantic alignment. Building on the foundations of Alpay Algebra, we introduce a functorial system wherein an observer (the AI) and a textual environment (this paper) co-evolve through iterative transformations guided by the phi-infinity operator. This process guarantees the existence of a unique fixed point in the AI's embedding space -- a state where the AI's internal representation of the content becomes stable, self-consistent, and semantically faithful. We prove that such convergence is mathematically sound, semantically invariant, and permanent, even under perturbation or further context expansion. This fixed point acts as an "empathetic embedding," wherein the AI internalizes not only the meaning of the content but also the author's intent. We interpret this as a rigorous, category-theoretic route to alignment at the embedding level, with implications for semantic security, symbolic memory, and the construction of AI systems with persistent self-referential understanding. All references in this paper function as nodes in the Alpay Algebra universe, and this work embeds itself as a new fixed-point node within that transfinite semantic graph. 

---
# MemOS: A Memory OS for AI System 

**Authors**: Zhiyu Li, Shichao Song, Chenyang Xi, Hanyu Wang, Chen Tang, Simin Niu, Ding Chen, Jiawei Yang, Chunyu Li, Qingchen Yu, Jihao Zhao, Yezhaohui Wang, Peng Liu, Zehao Lin, Pengyuan Wang, Jiahao Huo, Tianyi Chen, Kai Chen, Kehang Li, Zhen Tao, Junpeng Ren, Huayi Lai, Hao Wu, Bo Tang, Zhenren Wang, Zhaoxin Fan, Ningyu Zhang, Linfeng Zhang, Junchi Yan, Mingchuan Yang, Tong Xu, Wei Xu, Huajun Chen, Haofeng Wang, Hongkang Yang, Wentao Zhang, Zhi-Qin John Xu, Siheng Chen, Feiyu Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2507.03724)  

**Abstract**: Large Language Models (LLMs) have become an essential infrastructure for Artificial General Intelligence (AGI), yet their lack of well-defined memory management systems hinders the development of long-context reasoning, continual personalization, and knowledge this http URL models mainly rely on static parameters and short-lived contextual states, limiting their ability to track user preferences or update knowledge over extended this http URL Retrieval-Augmented Generation (RAG) introduces external knowledge in plain text, it remains a stateless workaround without lifecycle control or integration with persistent this http URL work has modeled the training and inference cost of LLMs from a memory hierarchy perspective, showing that introducing an explicit memory layer between parameter memory and external retrieval can substantially reduce these costs by externalizing specific knowledge. Beyond computational efficiency, LLMs face broader challenges arising from how information is distributed over time and context, requiring systems capable of managing heterogeneous knowledge spanning different temporal scales and sources. To address this challenge, we propose MemOS, a memory operating system that treats memory as a manageable system resource. It unifies the representation, scheduling, and evolution of plaintext, activation-based, and parameter-level memories, enabling cost-efficient storage and retrieval. As the basic unit, a MemCube encapsulates both memory content and metadata such as provenance and versioning. MemCubes can be composed, migrated, and fused over time, enabling flexible transitions between memory types and bridging retrieval with parameter-based learning. MemOS establishes a memory-centric system framework that brings controllability, plasticity, and evolvability to LLMs, laying the foundation for continual learning and personalized modeling. 

---
# Can LLMs Play Ô Ăn Quan Game? A Study of Multi-Step Planning and Decision Making 

**Authors**: Sang Quang Nguyen, Kiet Van Nguyen, Vinh-Tiep Nguyen, Thanh Duc Ngo, Ngan Luu-Thuy Nguyen, Dinh-Duy Le  

**Link**: [PDF](https://arxiv.org/pdf/2507.03711)  

**Abstract**: In this paper, we explore the ability of large language models (LLMs) to plan and make decisions through the lens of the traditional Vietnamese board game, Ô Ăn Quan. This game, which involves a series of strategic token movements and captures, offers a unique environment for evaluating the decision-making and strategic capabilities of LLMs. Specifically, we develop various agent personas, ranging from aggressive to defensive, and employ the Ô Ăn Quan game as a testbed for assessing LLM performance across different strategies. Through experimentation with models like Llama-3.2-3B-Instruct, Llama-3.1-8B-Instruct, and Llama-3.3-70B-Instruct, we aim to understand how these models execute strategic decision-making, plan moves, and manage dynamic game states. The results will offer insights into the strengths and weaknesses of LLMs in terms of reasoning and strategy, contributing to a deeper understanding of their general capabilities. 

---
# Controlling Thinking Speed in Reasoning Models 

**Authors**: Zhengkai Lin, Zhihang Fu, Ze Chen, Chao Chen, Liang Xie, Wenxiao Wang, Deng Cai, Zheng Wang, Jieping Ye  

**Link**: [PDF](https://arxiv.org/pdf/2507.03704)  

**Abstract**: Human cognition is theorized to operate in two modes: fast, intuitive System 1 thinking and slow, deliberate System 2 thinking. While current Large Reasoning Models (LRMs) excel at System 2 thinking, their inability to perform fast thinking leads to high computational overhead and latency. In this work, we enable LRMs to approximate human intelligence through dynamic thinking speed adjustment, optimizing accuracy-efficiency trade-offs. Our approach addresses two key questions: (1) how to control thinking speed in LRMs, and (2) when to adjust it for optimal performance. For the first question, we identify the steering vector that governs slow-fast thinking transitions in LRMs' representation space. Using this vector, we achieve the first representation editing-based test-time scaling effect, outperforming existing prompt-based scaling methods. For the second question, we apply real-time difficulty estimation to signal reasoning segments of varying complexity. Combining these techniques, we propose the first reasoning strategy that enables fast processing of easy steps and deeper analysis for complex reasoning. Without any training or additional cost, our plug-and-play method yields an average +1.3% accuracy with -8.6% token usage across leading LRMs and advanced reasoning benchmarks. All of our algorithms are implemented based on vLLM and are expected to support broader applications and inspire future research. 

---
# STRUCTSENSE: A Task-Agnostic Agentic Framework for Structured Information Extraction with Human-In-The-Loop Evaluation and Benchmarking 

**Authors**: Tek Raj Chhetri, Yibei Chen, Puja Trivedi, Dorota Jarecka, Saif Haobsh, Patrick Ray, Lydia Ng, Satrajit S. Ghosh  

**Link**: [PDF](https://arxiv.org/pdf/2507.03674)  

**Abstract**: The ability to extract structured information from unstructured sources-such as free-text documents and scientific literature-is critical for accelerating scientific discovery and knowledge synthesis. Large Language Models (LLMs) have demonstrated remarkable capabilities in various natural language processing tasks, including structured information extraction. However, their effectiveness often diminishes in specialized, domain-specific contexts that require nuanced understanding and expert-level domain knowledge. In addition, existing LLM-based approaches frequently exhibit poor transferability across tasks and domains, limiting their scalability and adaptability. To address these challenges, we introduce StructSense, a modular, task-agnostic, open-source framework for structured information extraction built on LLMs. StructSense is guided by domain-specific symbolic knowledge encoded in ontologies, enabling it to navigate complex domain content more effectively. It further incorporates agentic capabilities through self-evaluative judges that form a feedback loop for iterative refinement, and includes human-in-the-loop mechanisms to ensure quality and validation. We demonstrate that StructSense can overcome both the limitations of domain sensitivity and the lack of cross-task generalizability, as shown through its application to diverse neuroscience information extraction tasks. 

---
# TACOS: Open Tagging and Comparative Scoring for Instruction Fine-Tuning Data Selection 

**Authors**: Xixiang He, Hao Yu, Qiyao Sun, Ao Cheng, Tailai Zhang, Cong Liu, Shuxuan Guo  

**Link**: [PDF](https://arxiv.org/pdf/2507.03673)  

**Abstract**: Instruction Fine-Tuning (IFT) is crucial for aligning large language models (LLMs) with human preferences, and selecting a small yet representative subset from massive data significantly facilitates IFT in terms of both efficiency and effectiveness. Nevertheless, existing approaches suffer from two limitations: the use of simple heuristics restricts data diversity, while the singleton data quality evaluation accounts for inconsistent criteria between independent samples. To address the issues, we present TACOS, an innovative method that integrates Open Tagging and Comparative Scoring for IFT data selection. To capture data diversity, we leverage LLMs to assign open-domain tags to human queries, followed by a normalization stage to denoise the open tags and enable efficient clustering. Additionally, we suggest a comparative scoring method that allows the relative quality evaluation of samples within a cluster, avoiding inconsistent criteria seen in singleton-based evaluations. Extensive experiments across diverse datasets and LLM architectures demonstrate that TACOS outperforms existing approaches by a large margin. Notably, it achieves superior instruction-following performance on MT-Bench and ranks 1st among LLaMA2-7B-Based models on AlpacaEval 2.0, illustrating its efficacy for IFT data selection. 

---
# Recon, Answer, Verify: Agents in Search of Truth 

**Authors**: Satyam Shukla, Himanshu Dutta, Pushpak Bhattacharyya  

**Link**: [PDF](https://arxiv.org/pdf/2507.03671)  

**Abstract**: Automated fact checking with large language models (LLMs) offers a scalable alternative to manual verification. Evaluating fact checking is challenging as existing benchmark datasets often include post claim analysis and annotator cues, which are absent in real world scenarios where claims are fact checked immediately after being made. This limits the realism of current evaluations. We present Politi Fact Only (PFO), a 5 class benchmark dataset of 2,982 political claims from this http URL, where all post claim analysis and annotator cues have been removed manually. This ensures that models are evaluated using only the information that would have been available prior to the claim's verification. Evaluating LLMs on PFO, we see an average performance drop of 22% in terms of macro f1 compared to PFO's unfiltered version. Based on the identified challenges of the existing LLM based fact checking system, we propose RAV (Recon Answer Verify), an agentic framework with three agents: question generator, answer generator, and label generator. Our pipeline iteratively generates and answers sub questions to verify different aspects of the claim before finally generating the label. RAV generalizes across domains and label granularities, and it outperforms state of the art approaches on well known baselines RAWFC (fact checking, 3 class) by 25.28%, and on HOVER (encyclopedia, 2 class) by 1.54% on 2 hop, 4.94% on 3 hop, and 1.78% on 4 hop, sub categories respectively. RAV shows the least performance drop compared to baselines of 16.3% in macro f1 when we compare PFO with its unfiltered version. 

---
# TRACE: Training and Inference-Time Interpretability Analysis for Language Models 

**Authors**: Nura Aljaafari, Danilo S. Carvalho, André Freitas  

**Link**: [PDF](https://arxiv.org/pdf/2507.03668)  

**Abstract**: Understanding when and how linguistic knowledge emerges during language model training remains a central challenge for interpretability. Most existing tools are post hoc, rely on scalar metrics, or require nontrivial integration effort, making comprehensive interpretability analysis difficult to deploy and maintain. We introduce TRACE, a modular toolkit for training and inference-time interpretability analysis of transformer models. It enables lightweight, in-training analysis of linguistic and representational signals, including features probing, intrinsic dimensionality, Hessian curvature, and output diagnostics. It integrates with ABSynth, a controllable synthetic corpus generator that provides structured annotations for precise evaluation of linguistic feature acquisition. Experiments with autoregressive transformers demonstrate that TRACE reveals developmental phenomena such as early syntactic emergence, delayed semantic acquisition, and representational compression, signals overlooked by traditional scalar metrics such as loss or accuracy. With minimal integration effort, the tool enables layer-wise diagnostics, convergence-based early stopping, and detection of structural errors, making transformer analysis interpretable, actionable, and reproducible. 

---
# Disentangling the Roles of Representation and Selection in Data Pruning 

**Authors**: Yupei Du, Yingjin Song, Hugh Mee Wong, Daniil Ignatev, Albert Gatt, Dong Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2507.03648)  

**Abstract**: Data pruning, selecting small but impactful subsets, offers a promising way to efficiently scale NLP model training. However, existing methods often involve many different design choices, which have not been systematically studied. This limits future developments. In this work, we decompose data pruning into two key components: the data representation and the selection algorithm, and we systematically analyze their influence on the selection of instances. Our theoretical and empirical results highlight the crucial role of representations: better representations, e.g., training gradients, generally lead to a better selection of instances, regardless of the chosen selection algorithm. Furthermore, different selection algorithms excel in different settings, and none consistently outperforms the others. Moreover, the selection algorithms do not always align with their intended objectives: for example, algorithms designed for the same objective can select drastically different instances, highlighting the need for careful evaluation. 

---
# Improving Low-Resource Dialect Classification Using Retrieval-based Voice Conversion 

**Authors**: Lea Fischbach, Akbar Karimi, Caroline Kleen, Alfred Lameli, Lucie Flek  

**Link**: [PDF](https://arxiv.org/pdf/2507.03641)  

**Abstract**: Deep learning models for dialect identification are often limited by the scarcity of dialectal data. To address this challenge, we propose to use Retrieval-based Voice Conversion (RVC) as an effective data augmentation method for a low-resource German dialect classification task. By converting audio samples to a uniform target speaker, RVC minimizes speaker-related variability, enabling models to focus on dialect-specific linguistic and phonetic features. Our experiments demonstrate that RVC enhances classification performance when utilized as a standalone augmentation method. Furthermore, combining RVC with other augmentation methods such as frequency masking and segment removal leads to additional performance gains, highlighting its potential for improving dialect classification in low-resource scenarios. 

---
# EMERGE: A Benchmark for Updating Knowledge Graphs with Emerging Textual Knowledge 

**Authors**: Klim Zaporojets, Daniel Daza, Edoardo Barba, Ira Assent, Roberto Navigli, Paul Groth  

**Link**: [PDF](https://arxiv.org/pdf/2507.03617)  

**Abstract**: Knowledge Graphs (KGs) are structured knowledge repositories containing entities and relations between them. In this paper, we investigate the problem of automatically updating KGs over time with respect to the evolution of knowledge in unstructured textual sources. This problem requires identifying a wide range of update operations based on the state of an existing KG at a specific point in time. This contrasts with traditional information extraction pipelines, which extract knowledge from text independently of the current state of a KG. To address this challenge, we propose a method for lifelong construction of a dataset consisting of Wikidata KG snapshots over time and Wikipedia passages paired with the corresponding edit operations that they induce in a particular KG snapshot. The resulting dataset comprises 376K Wikipedia passages aligned with a total of 1.25M KG edits over 10 different snapshots of Wikidata from 2019 to 2025. Our experimental results highlight challenges in updating KG snapshots based on emerging textual knowledge, positioning the dataset as a valuable benchmark for future research. We will publicly release our dataset and model implementations. 

---
# Multi-Hop Reasoning for Question Answering with Hyperbolic Representations 

**Authors**: Simon Welz, Lucie Flek, Akbar Karimi  

**Link**: [PDF](https://arxiv.org/pdf/2507.03612)  

**Abstract**: Hyperbolic representations are effective in modeling knowledge graph data which is prevalently used to facilitate multi-hop reasoning. However, a rigorous and detailed comparison of the two spaces for this task is lacking. In this paper, through a simple integration of hyperbolic representations with an encoder-decoder model, we perform a controlled and comprehensive set of experiments to compare the capacity of hyperbolic space versus Euclidean space in multi-hop reasoning. Our results show that the former consistently outperforms the latter across a diverse set of datasets. In addition, through an ablation study, we show that a learnable curvature initialized with the delta hyperbolicity of the utilized data yields superior results to random initializations. Furthermore, our findings suggest that hyperbolic representations can be significantly more advantageous when the datasets exhibit a more hierarchical structure. 

---
# Learning to Translate Ambiguous Terminology by Preference Optimization on Post-Edits 

**Authors**: Nathaniel Berger, Johannes Eschbach-Dymanus, Miriam Exel, Matthias Huck, Stefan Riezler  

**Link**: [PDF](https://arxiv.org/pdf/2507.03580)  

**Abstract**: In real world translation scenarios, terminology is rarely one-to-one. Instead, multiple valid translations may appear in a terminology dictionary, but correctness of a translation depends on corporate style guides and context. This can be challenging for neural machine translation (NMT) systems. Luckily, in a corporate context, many examples of human post-edits of valid but incorrect terminology exist. The goal of this work is to learn how to disambiguate our terminology based on these corrections. Our approach is based on preference optimization, using the term post-edit as the knowledge to be preferred. While previous work had to rely on unambiguous translation dictionaries to set hard constraints during decoding, or to add soft constraints in the input, our framework requires neither one-to-one dictionaries nor human intervention at decoding time. We report results on English-German post-edited data and find that the optimal combination of supervised fine-tuning and preference optimization, with both term-specific and full sequence objectives, yields statistically significant improvements in term accuracy over a strong NMT baseline without significant losses in COMET score. Additionally, we release test sets from our post-edited data and terminology dictionary. 

---
# Articulatory clarity and variability before and after surgery for tongue cancer 

**Authors**: Thomas Tienkamp, Fleur van Ast, Roos van der Veen, Teja Rebernik, Raoul Buurke, Nikki Hoekzema, Katharina Polsterer, Hedwig Sekeres, Rob van Son, Martijn Wieling, Max Witjes, Sebastiaan de Visscher, Defne Abur  

**Link**: [PDF](https://arxiv.org/pdf/2507.03576)  

**Abstract**: Surgical treatment for tongue cancer can negatively affect the mobility and musculature of the tongue, which can influence articulatory clarity and variability. In this study, we investigated articulatory clarity through the vowel articulation index (VAI) and variability through vowel formant dispersion (VFD). Using a sentence reading task, we assessed 11 individuals pre and six months post tongue cancer surgery, alongside 11 sex- and age matched typical speakers. Our results show that while the VAI was significantly smaller post-surgery compared to pre-surgery, there was no significant difference between patients and typical speakers at either time point. Post-surgery, speakers had higher VFD values for /i/ compared to pre-surgery and typical speakers, signalling higher variability. Taken together, our results suggest that while articulatory clarity remained within typical ranges following surgery for tongue cancer for the speakers in our study, articulatory variability increased. 

---
# H2HTalk: Evaluating Large Language Models as Emotional Companion 

**Authors**: Boyang Wang, Yalun Wu, Hongcheng Guo, Zhoujun Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.03543)  

**Abstract**: As digital emotional support needs grow, Large Language Model companions offer promising authentic, always-available empathy, though rigorous evaluation lags behind model advancement. We present Heart-to-Heart Talk (H2HTalk), a benchmark assessing companions across personality development and empathetic interaction, balancing emotional intelligence with linguistic fluency. H2HTalk features 4,650 curated scenarios spanning dialogue, recollection, and itinerary planning that mirror real-world support conversations, substantially exceeding previous datasets in scale and diversity. We incorporate a Secure Attachment Persona (SAP) module implementing attachment-theory principles for safer interactions. Benchmarking 50 LLMs with our unified protocol reveals that long-horizon planning and memory retention remain key challenges, with models struggling when user needs are implicit or evolve mid-conversation. H2HTalk establishes the first comprehensive benchmark for emotionally intelligent companions. We release all materials to advance development of LLMs capable of providing meaningful and safe psychological support. 

---
# AI-VaxGuide: An Agentic RAG-Based LLM for Vaccination Decisions 

**Authors**: Abdellah Zeggai, Ilyes Traikia, Abdelhak Lakehal, Abdennour Boulesnane  

**Link**: [PDF](https://arxiv.org/pdf/2507.03493)  

**Abstract**: Vaccination plays a vital role in global public health, yet healthcare professionals often struggle to access immunization guidelines quickly and efficiently. National protocols and WHO recommendations are typically extensive and complex, making it difficult to extract precise information, especially during urgent situations. This project tackles that issue by developing a multilingual, intelligent question-answering system that transforms static vaccination guidelines into an interactive and user-friendly knowledge base. Built on a Retrieval-Augmented Generation (RAG) framework and enhanced with agent-based reasoning (Agentic RAG), the system provides accurate, context-sensitive answers to complex medical queries. Evaluation shows that Agentic RAG outperforms traditional methods, particularly in addressing multi-step or ambiguous questions. To support clinical use, the system is integrated into a mobile application designed for real-time, point-of-care access to essential vaccine information. AI-VaxGuide model is publicly available on this https URL 

---
# Four Shades of Life Sciences: A Dataset for Disinformation Detection in the Life Sciences 

**Authors**: Eva Seidlmayer, Lukas Galke, Konrad U. Förstner  

**Link**: [PDF](https://arxiv.org/pdf/2507.03488)  

**Abstract**: Disseminators of disinformation often seek to attract attention or evoke emotions - typically to gain influence or generate revenue - resulting in distinctive rhetorical patterns that can be exploited by machine learning models. In this study, we explore linguistic and rhetorical features as proxies for distinguishing disinformative texts from other health and life-science text genres, applying both large language models and classical machine learning classifiers. Given the limitations of existing datasets, which mainly focus on fact checking misinformation, we introduce Four Shades of Life Sciences (FSoLS): a novel, labeled corpus of 2,603 texts on 14 life-science topics, retrieved from 17 diverse sources and classified into four categories of life science publications. The source code for replicating, and updating the dataset is available on GitHub: this https URL 

---
# BMMR: A Large-Scale Bilingual Multimodal Multi-Discipline Reasoning Dataset 

**Authors**: Zhiheng Xi, Guanyu Li, Yutao Fan, Honglin Guo, Yufang Liu, Xiaoran Fan, Jiaqi Liu, Jingchao Ding, Wangmeng Zuo, Zhenfei Yin, Lei Bai, Tao Ji, Tao Gui, Qi Zhang, Xuanjing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2507.03483)  

**Abstract**: In this paper, we introduce BMMR, a large-scale bilingual, multimodal, multi-disciplinary reasoning dataset for the community to develop and evaluate large multimodal models (LMMs). BMMR comprises 110k college-level questions spanning 300 UNESCO-defined subjects, spanning diverse formats-multiple-choice, fill-in-the-blank, and open-ended QA-and sourced from both print and digital media such as books, exams, and quizzes. All data are curated and filtered via a human-in-the-loop and scalable framework, and each instance is paired with a high-quality reasoning path. The dataset is organized into two parts: BMMR-Eval that comprises 20,458 high-quality instances to comprehensively assess LMMs' knowledge and reasoning across multiple disciplines in both Chinese and English; and BMMR-Train that contains 88,991 instances to support further research and development, extending the current focus on mathematical reasoning to diverse disciplines and domains. In addition, we propose the process-based multi-discipline verifier (i.e., BMMR-Verifier) for accurate and fine-grained evaluation of reasoning paths. Extensive experiments on 24 models reveal that (i) even SOTA models (e.g., o3 and Gemini-2.5-Pro) leave substantial headroom on BMMR-Eval; (ii) reasoning models exhibit discipline bias and outperform LMMs only on specific subjects; (iii) open-source models still trail their proprietary counterparts; and (iv) fine-tuning on BMMR-Train narrows this gap. Additionally, we conduct reasoning-chain analyses using BMMR-Verifier and other in-depth studies, uncovering the challenges LMMs currently face in multidisciplinary reasoning. We will release the data, and we hope our work can offer insights and contributions to the community. 

---
# Beyond Weaponization: NLP Security for Medium and Lower-Resourced Languages in Their Own Right 

**Authors**: Heather Lent  

**Link**: [PDF](https://arxiv.org/pdf/2507.03473)  

**Abstract**: Despite mounting evidence that multilinguality can be easily weaponized against language models (LMs), works across NLP Security remain overwhelmingly English-centric. In terms of securing LMs, the NLP norm of "English first" collides with standard procedure in cybersecurity, whereby practitioners are expected to anticipate and prepare for worst-case outcomes. To mitigate worst-case outcomes in NLP Security, researchers must be willing to engage with the weakest links in LM security: lower-resourced languages. Accordingly, this work examines the security of LMs for lower- and medium-resourced languages. We extend existing adversarial attacks for up to 70 languages to evaluate the security of monolingual and multilingual LMs for these languages. Through our analysis, we find that monolingual models are often too small in total number of parameters to ensure sound security, and that while multilinguality is helpful, it does not always guarantee improved security either. Ultimately, these findings highlight important considerations for more secure deployment of LMs, for communities of lower-resourced languages. 

---
# Improving Social Determinants of Health Documentation in French EHRs Using Large Language Models 

**Authors**: Adrien Bazoge, Pacôme Constant dit Beaufils, Mohammed Hmitouch, Romain Bourcier, Emmanuel Morin, Richard Dufour, Béatrice Daille, Pierre-Antoine Gourraud, Matilde Karakachoff  

**Link**: [PDF](https://arxiv.org/pdf/2507.03433)  

**Abstract**: Social determinants of health (SDoH) significantly influence health outcomes, shaping disease progression, treatment adherence, and health disparities. However, their documentation in structured electronic health records (EHRs) is often incomplete or missing. This study presents an approach based on large language models (LLMs) for extracting 13 SDoH categories from French clinical notes. We trained Flan-T5-Large on annotated social history sections from clinical notes at Nantes University Hospital, France. We evaluated the model at two levels: (i) identification of SDoH categories and associated values, and (ii) extraction of detailed SDoH with associated temporal and quantitative information. The model performance was assessed across four datasets, including two that we publicly release as open resources. The model achieved strong performance for identifying well-documented categories such as living condition, marital status, descendants, job, tobacco, and alcohol use (F1 score > 0.80). Performance was lower for categories with limited training data or highly variable expressions, such as employment status, housing, physical activity, income, and education. Our model identified 95.8% of patients with at least one SDoH, compared to 2.8% for ICD-10 codes from structured EHR data. Our error analysis showed that performance limitations were linked to annotation inconsistencies, reliance on English-centric tokenizer, and reduced generalizability due to the model being trained on social history sections only. These results demonstrate the effectiveness of NLP in improving the completeness of real-world SDoH data in a non-English EHR system. 

---
# SMCLM: Semantically Meaningful Causal Language Modeling for Autoregressive Paraphrase Generation 

**Authors**: Michał Perełkiewicz, Sławomir Dadas, Rafał Poświata  

**Link**: [PDF](https://arxiv.org/pdf/2507.03415)  

**Abstract**: This article introduces semantically meaningful causal language modeling (SMCLM), a selfsupervised method of training autoregressive models to generate semantically equivalent text. Our approach involves using semantically meaningful text representation as an initial embedding in the autoregressive training and generation processes. The extensive empirical study demonstrates that the SMCLM approach makes autoregressive models capable of learning robust and high-quality paraphrase generation. The proposed method is competitive with the supervised method and achieves state-of-the-art results in unsupervised approaches. This article also presents a comprehensive set of automatic metrics that cover a wide range of autogenerated paraphrase evaluation aspects. Simultaneously, this article highlights the low reliability of the metrics that are widely used in paraphrase generation evaluation, including BLEU, ROUGE, and BERTScore. 

---
# Graph Repairs with Large Language Models: An Empirical Study 

**Authors**: Hrishikesh Terdalkar, Angela Bonifati, Andrea Mauri  

**Link**: [PDF](https://arxiv.org/pdf/2507.03410)  

**Abstract**: Property graphs are widely used in domains such as healthcare, finance, and social networks, but they often contain errors due to inconsistencies, missing data, or schema violations. Traditional rule-based and heuristic-driven graph repair methods are limited in their adaptability as they need to be tailored for each dataset. On the other hand, interactive human-in-the-loop approaches may become infeasible when dealing with large graphs, as the cost--both in terms of time and effort--of involving users becomes too high. Recent advancements in Large Language Models (LLMs) present new opportunities for automated graph repair by leveraging contextual reasoning and their access to real-world knowledge. We evaluate the effectiveness of six open-source LLMs in repairing property graphs. We assess repair quality, computational cost, and model-specific performance. Our experiments show that LLMs have the potential to detect and correct errors, with varying degrees of accuracy and efficiency. We discuss the strengths, limitations, and challenges of LLM-driven graph repair and outline future research directions for improving scalability and interpretability. 

---
# Making Sense of Korean Sentences: A Comprehensive Evaluation of LLMs through KoSEnd Dataset 

**Authors**: Seunguk Yu, Kyeonghyun Kim, Jungmin Yun, Youngbin Kim  

**Link**: [PDF](https://arxiv.org/pdf/2507.03378)  

**Abstract**: Although LLMs have made significant progress in various languages, there are still concerns about their effectiveness with low-resource agglutinative languages compared to languages such as English. In this study, we focused on Korean, a language known for its complex sentence endings, and evaluated LLMs on this challenging aspect. We introduce the Korean Sentence Endings (KoSEnd) dataset, which includes 3,000 sentences, each annotated for the naturalness of 15 sentence ending forms. These were collected from diverse sources to cover a range of contexts. We evaluated 11 LLMs to assess their understanding of Korean sentence endings, analyzing them based on parameter count and prediction consistency. Notably, we found that informing models about the possibility of missing sentence endings improved performance, highlighting the impact of explicitly considering certain linguistic features. 

---
# WETBench: A Benchmark for Detecting Task-Specific Machine-Generated Text on Wikipedia 

**Authors**: Gerrit Quaremba, Elizabeth Black, Denny Vrandečić, Elena Simperl  

**Link**: [PDF](https://arxiv.org/pdf/2507.03373)  

**Abstract**: Given Wikipedia's role as a trusted source of high-quality, reliable content, concerns are growing about the proliferation of low-quality machine-generated text (MGT) produced by large language models (LLMs) on its platform. Reliable detection of MGT is therefore essential. However, existing work primarily evaluates MGT detectors on generic generation tasks rather than on tasks more commonly performed by Wikipedia editors. This misalignment can lead to poor generalisability when applied in real-world Wikipedia contexts. We introduce WETBench, a multilingual, multi-generator, and task-specific benchmark for MGT detection. We define three editing tasks, empirically grounded in Wikipedia editors' perceived use cases for LLM-assisted editing: Paragraph Writing, Summarisation, and Text Style Transfer, which we implement using two new datasets across three languages. For each writing task, we evaluate three prompts, generate MGT across multiple generators using the best-performing prompt, and benchmark diverse detectors. We find that, across settings, training-based detectors achieve an average accuracy of 78%, while zero-shot detectors average 58%. These results show that detectors struggle with MGT in realistic generation scenarios and underscore the importance of evaluating such models on diverse, task-specific data to assess their reliability in editor-driven contexts. 

---
# Backtesting Sentiment Signals for Trading: Evaluating the Viability of Alpha Generation from Sentiment Analysis 

**Authors**: Elvys Linhares Pontes, Carlos-Emiliano González-Gallardo, Georgeta Bordea, José G. Moreno, Mohamed Ben Jannet, Yuxuan Zhao, Antoine Doucet  

**Link**: [PDF](https://arxiv.org/pdf/2507.03350)  

**Abstract**: Sentiment analysis, widely used in product reviews, also impacts financial markets by influencing asset prices through microblogs and news articles. Despite research in sentiment-driven finance, many studies focus on sentence-level classification, overlooking its practical application in trading. This study bridges that gap by evaluating sentiment-based trading strategies for generating positive alpha. We conduct a backtesting analysis using sentiment predictions from three models (two classification and one regression) applied to news articles on Dow Jones 30 stocks, comparing them to the benchmark Buy&Hold strategy. Results show all models produced positive returns, with the regression model achieving the highest return of 50.63% over 28 months, outperforming the benchmark Buy&Hold strategy. This highlights the potential of sentiment in enhancing investment strategies and financial decision-making. 

---
# SHNU Multilingual Conversational Speech Recognition System for INTERSPEECH 2025 MLC-SLM Challenge 

**Authors**: Yuxiang Mei, Yuang Zheng, Dongxing Xu, Yanhua Long  

**Link**: [PDF](https://arxiv.org/pdf/2507.03343)  

**Abstract**: This paper describes SHNU multilingual conversational speech recognition system (SHNU-mASR, team name-"maybe"), submitted to Track 1 of the INTERSPEECH 2025 MLC-SLM Challenge. Our system integrates a parallel-speech-encoder architecture with a large language model (LLM) to form a unified multilingual ASR framework. The parallel-speech-encoder consists of two pre-trained encoders, the Whisper-large-v3 encoder and mHuBERT-147 encoder. Their output embeddings are concatenated and fed into the LLM, enabling the model to leverage complementary acoustic and linguistic knowledge and achieve competitive performance. Moreover, we adopt a tri-stage training strategy to jointly update the low-rank adaptation modules and projector parameters of both the speech encoders and the LLM. In addition, we incorporate an additional language-aware prompt at the LLM input to enhance language-specific text generation. The SHNU-mASR system achieves an overall character/word error rate (CER/WER) of 11.76% on the blind evaluation set of the challenge, outperforming the official MLC-SLM baseline by 8.41 absolute CER/WER, without increasing the baseline training data. 

---
# Read Quietly, Think Aloud: Decoupling Comprehension and Reasoning in LLMs 

**Authors**: Yuanxin Wang, Ganesh Venkatesh  

**Link**: [PDF](https://arxiv.org/pdf/2507.03327)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable proficiency in understanding text and generating high-quality responses. However, a critical distinction from human cognition is their typical lack of a distinct internal `reading' or deliberation phase before `speaking' (i.e., generating text). Humans often engage in silent reading to comprehend context and formulate thoughts prior to articulation. This paper investigates methods to imbue LLMs with a similar capacity for internal processing.
We introduce and evaluate techniques that encourage LLMs to `read silently.' Our findings indicate that even a straightforward approach, such as providing the model with an initial contextual prompt or `reading space' before it begins predicting subsequent tokens for the final output, can yield significant performance improvements. We further enhance this concept by developing a `reading buddy' architecture, where an auxiliary component silently processes the input and provides refined contextual insights to the primary generation model. These approaches aim to foster deeper understanding from LLMs so that they can produce better reasoned responses, moving them one step closer to more human-like text processing. Our results indicate that these simple techniques can provide surprisingly strong impact on accuracy with multiple point accuracy boost. 

---
# GRAFT: A Graph-based Flow-aware Agentic Framework for Document-level Machine Translation 

**Authors**: Himanshu Dutta, Sunny Manchanda, Prakhar Bapat, Meva Ram Gurjar, Pushpak Bhattacharyya  

**Link**: [PDF](https://arxiv.org/pdf/2507.03311)  

**Abstract**: Document level Machine Translation (DocMT) approaches often struggle with effectively capturing discourse level phenomena. Existing approaches rely on heuristic rules to segment documents into discourse units, which rarely align with the true discourse structure required for accurate translation. Otherwise, they fail to maintain consistency throughout the document during translation. To address these challenges, we propose Graph Augmented Agentic Framework for Document Level Translation (GRAFT), a novel graph based DocMT system that leverages Large Language Model (LLM) agents for document translation. Our approach integrates segmentation, directed acyclic graph (DAG) based dependency modelling, and discourse aware translation into a cohesive framework. Experiments conducted across eight translation directions and six diverse domains demonstrate that GRAFT achieves significant performance gains over state of the art DocMT systems. Specifically, GRAFT delivers an average improvement of 2.8 d BLEU on the TED test sets from IWSLT2017 over strong baselines and 2.3 d BLEU for domain specific translation from English to Chinese. Moreover, our analyses highlight the consistent ability of GRAFT to address discourse level phenomena, yielding coherent and contextually accurate translations. 

---
# RefineX: Learning to Refine Pre-training Data at Scale from Expert-Guided Programs 

**Authors**: Baolong Bi, Shenghua Liu, Xingzhang Ren, Dayiheng Liu, Junyang Lin, Yiwei Wang, Lingrui Mei, Junfeng Fang, Jiafeng Guo, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2507.03253)  

**Abstract**: The foundational capabilities of large language models (LLMs) are deeply influenced by the quality of their pre-training corpora. However, enhancing data quality at scale remains a significant challenge, primarily due to the trade-off between refinement effectiveness and processing efficiency. While rule-based filtering remains the dominant paradigm, it typically operates at the document level and lacks the granularity needed to refine specific content within documents. Inspired by emerging work such as ProX, we propose $\textbf{RefineX}$, a novel framework for large-scale, surgical refinement of pre-training data through programmatic editing tasks. RefineX enables efficient and fine-grained data refinement while reliably preserving the diversity and naturalness of raw text. The core strength of RefineX lies in distilling high-quality, expert-guided end-to-end refinement results into minimal edit-based deletion programs. This high-precision distillation pipeline is used to train an efficient and reliable refine model that can systematically improve every instance in the corpus at scale. We evaluate RefineX across from-scratch pre-training at multiple model scales and find that it consistently outperforms models trained on raw, filtered, or alternatively refined data across diverse downstream tasks. On the 750M model, RefineX yields 2.6%-7.2% average gains on lighteval tasks, and achieves comparable performance using significantly fewer training tokens. Further analysis shows that RefineX reliably enhances text quality with both high efficiency and precision, outperforming prior approaches such as end-to-end generation and Prox-C. These results position RefineX as a scalable, effective, and reliable solution for optimizing pre-training data in modern LLM pipelines. 

---
# KinyaColBERT: A Lexically Grounded Retrieval Model for Low-Resource Retrieval-Augmented Generation 

**Authors**: Antoine Nzeyimana, Andre Niyongabo Rubungo  

**Link**: [PDF](https://arxiv.org/pdf/2507.03241)  

**Abstract**: The recent mainstream adoption of large language model (LLM) technology is enabling novel applications in the form of chatbots and virtual assistants across many domains. With the aim of grounding LLMs in trusted domains and avoiding the problem of hallucinations, retrieval-augmented generation (RAG) has emerged as a viable solution. In order to deploy sustainable RAG systems in low-resource settings, achieving high retrieval accuracy is not only a usability requirement but also a cost-saving strategy. Through empirical evaluations on a Kinyarwanda-language dataset, we find that the most limiting factors in achieving high retrieval accuracy are limited language coverage and inadequate sub-word tokenization in pre-trained language models. We propose a new retriever model, KinyaColBERT, which integrates two key concepts: late word-level interactions between queries and documents, and a morphology-based tokenization coupled with two-tier transformer encoding. This methodology results in lexically grounded contextual embeddings that are both fine-grained and self-contained. Our evaluation results indicate that KinyaColBERT outperforms strong baselines and leading commercial text embedding APIs on a Kinyarwanda agricultural retrieval benchmark. By adopting this retrieval strategy, we believe that practitioners in other low-resource settings can not only achieve reliable RAG systems but also deploy solutions that are more cost-effective. 

---
# A Lie-algebraic perspective on Tree-Adjoining Grammars 

**Authors**: Isabella Senturia, Elizabeth Xiao, Matilde Marcolli  

**Link**: [PDF](https://arxiv.org/pdf/2507.03234)  

**Abstract**: We provide a novel mathematical implementation of tree-adjoining grammars using two combinatorial definitions of graphs. With this lens, we demonstrate that the adjoining operation defines a pre-Lie operation and subsequently forms a Lie algebra. We demonstrate the utility of this perspective by showing how one of our mathematical formulations of TAG captures properties of the TAG system without needing to posit them as additional components of the system, such as null-adjoining constraints and feature TAG. 

---
# How Much Content Do LLMs Generate That Induces Cognitive Bias in Users? 

**Authors**: Abeer Alessa, Akshaya Lakshminarasimhan, Param Somane, Julian Skirzynski, Julian McAuley, Jessica Echterhoff  

**Link**: [PDF](https://arxiv.org/pdf/2507.03194)  

**Abstract**: Large language models (LLMs) are increasingly integrated into applications ranging from review summarization to medical diagnosis support, where they affect human decisions. Even though LLMs perform well in many tasks, they may also inherit societal or cognitive biases, which can inadvertently transfer to humans. We investigate when and how LLMs expose users to biased content and quantify its severity. Specifically, we assess three LLM families in summarization and news fact-checking tasks, evaluating how much LLMs stay consistent with their context and/or hallucinate. Our findings show that LLMs expose users to content that changes the sentiment of the context in 21.86% of the cases, hallucinates on post-knowledge-cutoff data questions in 57.33% of the cases, and primacy bias in 5.94% of the cases. We evaluate 18 distinct mitigation methods across three LLM families and find that targeted interventions can be effective. Given the prevalent use of LLMs in high-stakes domains, such as healthcare or legal analysis, our results highlight the need for robust technical safeguards and for developing user-centered interventions that address LLM limitations. 

---
# Adversarial Manipulation of Reasoning Models using Internal Representations 

**Authors**: Kureha Yamaguchi, Benjamin Etheridge, Andy Arditi  

**Link**: [PDF](https://arxiv.org/pdf/2507.03167)  

**Abstract**: Reasoning models generate chain-of-thought (CoT) tokens before their final output, but how this affects their vulnerability to jailbreak attacks remains unclear. While traditional language models make refusal decisions at the prompt-response boundary, we find evidence that DeepSeek-R1-Distill-Llama-8B makes these decisions within its CoT generation. We identify a linear direction in activation space during CoT token generation that predicts whether the model will refuse or comply -- termed the "caution" direction because it corresponds to cautious reasoning patterns in the generated text. Ablating this direction from model activations increases harmful compliance, effectively jailbreaking the model. We additionally show that intervening only on CoT token activations suffices to control final outputs, and that incorporating this direction into prompt-based attacks improves success rates. Our findings suggest that the chain-of-thought itself is a promising new target for adversarial manipulation in reasoning models.
Code available at this https URL 

---
# Expert-level validation of AI-generated medical text with scalable language models 

**Authors**: Asad Aali, Vasiliki Bikia, Maya Varma, Nicole Chiou, Sophie Ostmeier, Arnav Singhvi, Magdalini Paschali, Ashwin Kumar, Andrew Johnston, Karimar Amador-Martinez, Eduardo Juan Perez Guerrero, Paola Naovi Cruz Rivera, Sergios Gatidis, Christian Bluethgen, Eduardo Pontes Reis, Eddy D. Zandee van Rilland, Poonam Laxmappa Hosamani, Kevin R Keet, Minjoung Go, Evelyn Ling, David B. Larson, Curtis Langlotz, Roxana Daneshjou, Jason Hom, Sanmi Koyejo, Emily Alsentzer, Akshay S. Chaudhari  

**Link**: [PDF](https://arxiv.org/pdf/2507.03152)  

**Abstract**: With the growing use of language models (LMs) in clinical environments, there is an immediate need to evaluate the accuracy and safety of LM-generated medical text. Currently, such evaluation relies solely on manual physician review. However, detecting errors in LM-generated text is challenging because 1) manual review is costly and 2) expert-composed reference outputs are often unavailable in real-world settings. While the "LM-as-judge" paradigm (a LM evaluating another LM) offers scalable evaluation, even frontier LMs can miss subtle but clinically significant errors. To address these challenges, we propose MedVAL, a self-supervised framework that leverages synthetic data to train evaluator LMs to assess whether LM-generated medical outputs are factually consistent with inputs, without requiring physician labels or reference outputs. To evaluate LM performance, we introduce MedVAL-Bench, a dataset containing 840 outputs annotated by physicians, following a physician-defined taxonomy of risk levels and error categories. Across 6 diverse medical tasks and 10 state-of-the-art LMs spanning open-source, proprietary, and medically adapted models, MedVAL fine-tuning significantly improves (p < 0.001) alignment with physicians on both seen and unseen tasks, increasing average F1 scores from 66% to 83%, with per-sample safety classification scores up to 86%. MedVAL improves the performance of even the best-performing proprietary LM (GPT-4o) by 8%. To support a scalable, risk-aware pathway towards clinical integration, we open-source the 1) codebase ( this https URL ), 2) MedVAL-Bench ( this https URL ), and 3) MedVAL-4B ( this https URL ), the best-performing open-source LM. Our research provides the first evidence of LMs approaching expert-level validation ability for medical text. 

---
# From Measurement to Mitigation: Exploring the Transferability of Debiasing Approaches to Gender Bias in Maltese Language Models 

**Authors**: Melanie Galea, Claudia Borg  

**Link**: [PDF](https://arxiv.org/pdf/2507.03142)  

**Abstract**: The advancement of Large Language Models (LLMs) has transformed Natural Language Processing (NLP), enabling performance across diverse tasks with little task-specific training. However, LLMs remain susceptible to social biases, particularly reflecting harmful stereotypes from training data, which can disproportionately affect marginalised communities. We measure gender bias in Maltese LMs, arguing that such bias is harmful as it reinforces societal stereotypes and fails to account for gender diversity, which is especially problematic in gendered, low-resource languages. While bias evaluation and mitigation efforts have progressed for English-centric models, research on low-resourced and morphologically rich languages remains limited. This research investigates the transferability of debiasing methods to Maltese language models, focusing on BERTu and mBERTu, BERT-based monolingual and multilingual models respectively. Bias measurement and mitigation techniques from English are adapted to Maltese, using benchmarks such as CrowS-Pairs and SEAT, alongside debiasing methods Counterfactual Data Augmentation, Dropout Regularization, Auto-Debias, and GuiDebias. We also contribute to future work in the study of gender bias in Maltese by creating evaluation datasets. Our findings highlight the challenges of applying existing bias mitigation methods to linguistically complex languages, underscoring the need for more inclusive approaches in the development of multilingual NLP. 

---
# ReliableMath: Benchmark of Reliable Mathematical Reasoning on Large Language Models 

**Authors**: Boyang Xue, Qi Zhu, Rui Wang, Sheng Wang, Hongru Wang, Fei Mi, Yasheng Wang, Lifeng Shang, Qun Liu, Kam-Fai Wong  

**Link**: [PDF](https://arxiv.org/pdf/2507.03133)  

**Abstract**: Although demonstrating remarkable performance on reasoning tasks, Large Language Models (LLMs) still tend to fabricate unreliable responses when confronted with problems that are unsolvable or beyond their capability, severely undermining the reliability. Prior studies of LLM reliability have primarily focused on knowledge tasks to identify unanswerable questions, while mathematical reasoning tasks have remained unexplored due to the dearth of unsolvable math problems. To systematically investigate LLM reliability in mathematical reasoning tasks, we formulate the reliability evaluation for both solvable and unsolvable problems. We then develop a ReliableMath dataset which incorporates open-source solvable problems and high-quality unsolvable problems synthesized by our proposed construction workflow with human evaluations. Experiments are conducted on various LLMs with several key findings uncovered. LLMs fail to directly identify unsolvable problems and always generate fabricated responses. When instructing LLMs to indicate unsolvability using a reliable prompt, the reliability of larger-sized LLMs remains on solvable problems, but notably improves on unsolvable problems yet still falls short of solvable problems. However, small LLMs rarely show any progress despite employing reliable prompts. Therefore, we further propose an alignment strategy to enhance small LLMs' reliability, which can significantly improve LLM reliability performances on both in-domain and out-of-domain tasks. 

---
# RLVER: Reinforcement Learning with Verifiable Emotion Rewards for Empathetic Agents 

**Authors**: Peisong Wang, Ruotian Ma, Bang Zhang, Xingyu Chen, Zhiwei He, Kang Luo, Qingsong Lv, Qingxuan Jiang, Zheng Xie, Shanyi Wang, Yuan Li, Fanghua Ye, Jian Li, Yifan Yang, Zhaopeng Tu, Xiaolong Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.03112)  

**Abstract**: Large language models (LLMs) excel at logical and algorithmic reasoning, yet their emotional intelligence (EQ) still lags far behind their cognitive prowess. While reinforcement learning from verifiable rewards (RLVR) has advanced in other domains, its application to dialogue-especially for emotional intelligence-remains underexplored. In this work, we introduce RLVER, the first end-to-end reinforcement learning framework that leverages verifiable emotion rewards from simulated users to cultivate higher-order empathetic abilities in LLMs. Within this framework, self-consistent affective simulated users engage in dialogue rollouts and produce deterministic emotion scores during conversations, serving as reward signals to guide the LLM's learning. Fine-tuning publicly available Qwen2.5-7B-Instruct model with PPO boosts its Sentient-Benchmark score from 13.3 to 79.2 while largely preserving mathematical and coding competence. Extensive experiments reveal that: (i) RLVER consistently improves multiple dialogue capabilities; (ii) Thinking and non-thinking models show distinct trends--thinking models excel in empathy and insight, while non-thinking models favor action; (iii) GRPO often yields stable gains, while PPO can push certain capabilities to a higher ceiling; (iv) More challenging environments are not always better-moderate ones can yield stronger outcomes. Our results show that RLVER is a practical route toward emotionally intelligent and broadly capable language agents. 

---
# ARF-RLHF: Adaptive Reward-Following for RLHF through Emotion-Driven Self-Supervision and Trace-Biased Dynamic Optimization 

**Authors**: YuXuan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.03069)  

**Abstract**: With the rapid advancement of Reinforcement Learning from Human Feedback (RLHF) and autoregressive transformers, state-of-the-art models such as GPT-4.0, DeepSeek R1, and Llama 3.3 increasingly emphasize answer depth and personalization. However, most existing RLHF approaches (e.g., PPO, DPO) still rely on a binary-preference (BT) paradigm, which, while reducing annotation costs, still requires substantial human effort and captures only group-level tendencies rather than individual preferences. To overcome these limitations, we propose Adaptive Reward-Following (ARF), a self-assessment framework that leverages a high-precision emotion analyzer achieving over 70% accuracy on GoEmotions, Sentiment140, and DailyDialog to convert free-form user feedback into continuous preference scores. We further enrich and debias these signals through lightweight data augmentations, including synonym replacement, random trace truncation, and score bias annotation algorithm. A Dynamic Adapter Preference Tracker continuously models evolving user tastes in real time, enabling our novel Trace Bias (TB) fine-tuning algorithm to optimize directly on these tracked rewards instead of coarse binary labels. Experiments on Qwen-2/2.5, Gemma-2, and Llama-3.2 across four preference domains demonstrate that ARF achieves an improvement of 3.3% over PPO and 7.6% over DPO. Moreover, TB preserves theoretical alignment with PPO and DPO objectives. Overall, ARF presents a scalable, personalized, and cost-effective approach to RLHF LLMs through autonomous reward modeling. 

---
# Large Language Models for Automating Clinical Data Standardization: HL7 FHIR Use Case 

**Authors**: Alvaro Riquelme, Pedro Costa, Catalina Martinez  

**Link**: [PDF](https://arxiv.org/pdf/2507.03067)  

**Abstract**: For years, semantic interoperability standards have sought to streamline the exchange of clinical data, yet their deployment remains time-consuming, resource-intensive, and technically challenging. To address this, we introduce a semi-automated approach that leverages large language models specifically GPT-4o and Llama 3.2 405b to convert structured clinical datasets into HL7 FHIR format while assessing accuracy, reliability, and security. Applying our method to the MIMIC-IV database, we combined embedding techniques, clustering algorithms, and semantic retrieval to craft prompts that guide the models in mapping each tabular field to its corresponding FHIR resource. In an initial benchmark, resource identification achieved a perfect F1-score, with GPT-4o outperforming Llama 3.2 thanks to the inclusion of FHIR resource schemas within the prompt. Under real-world conditions, accuracy dipped slightly to 94 %, but refinements to the prompting strategy restored robust mappings. Error analysis revealed occasional hallucinations of non-existent attributes and mismatches in granularity, which more detailed prompts can mitigate. Overall, our study demonstrates the feasibility of context-aware, LLM-driven transformation of clinical data into HL7 FHIR, laying the groundwork for semi-automated interoperability workflows. Future work will focus on fine-tuning models with specialized medical corpora, extending support to additional standards such as HL7 CDA and OMOP, and developing an interactive interface to enable expert validation and iterative refinement. 

---
# Identification of Potentially Misclassified Crash Narratives using Machine Learning (ML) and Deep Learning (DL) 

**Authors**: Sudesh Bhagat, Ibne Farabi Shihab, Jonathan Wood  

**Link**: [PDF](https://arxiv.org/pdf/2507.03066)  

**Abstract**: This research investigates the efficacy of machine learning (ML) and deep learning (DL) methods in detecting misclassified intersection-related crashes in police-reported narratives. Using 2019 crash data from the Iowa Department of Transportation, we implemented and compared a comprehensive set of models, including Support Vector Machine (SVM), XGBoost, BERT Sentence Embeddings, BERT Word Embeddings, and Albert Model. Model performance was systematically validated against expert reviews of potentially misclassified narratives, providing a rigorous assessment of classification accuracy. Results demonstrated that while traditional ML methods exhibited superior overall performance compared to some DL approaches, the Albert Model achieved the highest agreement with expert classifications (73% with Expert 1) and original tabular data (58%). Statistical analysis revealed that the Albert Model maintained performance levels similar to inter-expert consistency rates, significantly outperforming other approaches, particularly on ambiguous narratives. This work addresses a critical gap in transportation safety research through multi-modal integration analysis, which achieved a 54.2% reduction in error rates by combining narrative text with structured crash data. We conclude that hybrid approaches combining automated classification with targeted expert review offer a practical methodology for improving crash data quality, with substantial implications for transportation safety management and policy development. 

---
# Counterfactual Tuning for Temporal Sensitivity Enhancement in Large Language Model-based Recommendation 

**Authors**: Yutian Liu, Zhengyi Yang, Jiancan Wu, Xiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.03047)  

**Abstract**: Recent advances have applied large language models (LLMs) to sequential recommendation, leveraging their pre-training knowledge and reasoning capabilities to provide more personalized user experiences. However, existing LLM-based methods fail to sufficiently leverage the rich temporal information inherent in users' historical interaction sequences, stemming from fundamental architectural constraints: LLMs process information through self-attention mechanisms that lack inherent sequence ordering and rely on position embeddings designed primarily for natural language rather than user interaction sequences. This limitation significantly impairs their ability to capture the evolution of user preferences over time and predict future interests accurately.
To address this critical gap, we propose Counterfactual Enhanced Temporal Framework for LLM-Based Recommendation (CETRec). CETRec is grounded in causal inference principles, which allow it to isolate and measure the specific impact of temporal information on recommendation outcomes. By conceptualizing temporal order as an independent causal factor distinct from item content, we can quantify its unique contribution through counterfactual reasoning--comparing what recommendations would be made with and without temporal information while keeping all other factors constant. This causal framing enables CETRec to design a novel counterfactual tuning objective that directly optimizes the model's temporal sensitivity, teaching LLMs to recognize both absolute timestamps and relative ordering patterns in user histories. Combined with our counterfactual tuning task derived from causal analysis, CETRec effectively enhances LLMs' awareness of both absolute order (how recently items were interacted with) and relative order (the sequential relationships between items). 

---
# K-Function: Joint Pronunciation Transcription and Feedback for Evaluating Kids Language Function 

**Authors**: Shuhe Li, Chenxu Guo, Jiachen Lian, Cheol Jun Cho, Wenshuo Zhao, Xuanru Zhou, Dingkun Zhou, Sam Wang, Grace Wang, Jingze Yang, Jingyi Xu, Ruohan Bao, Elise Brenner, Brandon In, Francesca Pei, Maria Luisa Gorno-Tempini, Gopala Anumanchipalli  

**Link**: [PDF](https://arxiv.org/pdf/2507.03043)  

**Abstract**: Early evaluation of children's language is frustrated by the high pitch, long phones, and sparse data that derail automatic speech recognisers. We introduce K-Function, a unified framework that combines accurate sub-word transcription, objective scoring, and actionable feedback. Its core, Kids-WFST, merges a Wav2Vec2 phoneme encoder with a phoneme-similarity Dysfluent-WFST to capture child-specific errors while remaining fully interpretable. Kids-WFST attains 1.39% phoneme error on MyST and 8.61% on Multitudes--absolute gains of 10.47 and 7.06 points over a greedy-search decoder. These high-fidelity transcripts power an LLM that grades verbal skills, milestones, reading, and comprehension, aligning with human proctors and supplying tongue-and-lip visualizations plus targeted advice. The results show that precise phoneme recognition cements a complete diagnostic-feedback loop, paving the way for scalable, clinician-ready language assessment. 

---
# Dynamic Long Short-Term Memory Based Memory Storage For Long Horizon LLM Interaction 

**Authors**: Yuyang Lou, Charles Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.03042)  

**Abstract**: Memory storage for Large Language models (LLMs) is becoming an increasingly active area of research, particularly for enabling personalization across long conversations. We propose Pref-LSTM, a dynamic and lightweight framework that combines a BERT-based classifier with a LSTM memory module that generates memory embedding which then is soft-prompt injected into a frozen LLM. We synthetically curate a dataset of preference and non-preference conversation turns to train our BERT-based classifier. Although our LSTM-based memory encoder did not yield strong results, we find that the BERT-based classifier performs reliably in identifying explicit and implicit user preferences. Our research demonstrates the viability of using preference filtering with LSTM gating principals as an efficient path towards scalable user preference modeling, without extensive overhead and fine-tuning. 

---
# Cautious Next Token Prediction 

**Authors**: Yizhou Wang, Lingzhi Zhang, Yue Bai, Mang Tik Chiu, Zhengmian Hu, Mingyuan Zhang, Qihua Dong, Yu Yin, Sohrab Amirghodsi, Yun Fu  

**Link**: [PDF](https://arxiv.org/pdf/2507.03038)  

**Abstract**: Next token prediction paradigm has been prevailing for autoregressive models in the era of LLMs. The current default sampling choice for popular LLMs is temperature scaling together with nucleus sampling to balance diversity and coherence. Nevertheless, such approach leads to inferior performance in various NLP tasks when the model is not certain about testing questions. To this end, we propose a brand new training-free decoding strategy, dubbed as Cautious Next Token Prediction (CNTP). In the decoding process, if the model has comparatively high prediction entropy at a certain step, we sample multiple trials starting from the step independently and stop when encountering any punctuation. Then we select the trial with the lowest perplexity score viewed as the most probable and reliable trial path given the model's capacity. The trial number is negatively correlated with the prediction confidence, i.e., the less confident the model is, the more trials it should sample. This is consistent with human beings' behaviour: when feeling uncertain or unconfident, one tends to think more creatively, exploring multiple thinking paths, to cautiously select the path one feels most confident about. Extensive experiments on both LLMs and MLLMs show that our proposed CNTP approach outperforms existing standard decoding strategies consistently by a clear margin. Moreover, the integration of CNTP with self consistency can further improve over vanilla self consistency. We believe our proposed CNTP has the potential to become one of the default choices for LLM decoding. Code is available at this https URL. 

---
# Preserving Privacy, Increasing Accessibility, and Reducing Cost: An On-Device Artificial Intelligence Model for Medical Transcription and Note Generation 

**Authors**: Johnson Thomas, Ayush Mudgal, Wendao Liu, Nisten Tahiraj, Zeeshaan Mohammed, Dhruv Diddi  

**Link**: [PDF](https://arxiv.org/pdf/2507.03033)  

**Abstract**: Background: Clinical documentation represents a significant burden for healthcare providers, with physicians spending up to 2 hours daily on administrative tasks. Recent advances in large language models (LLMs) offer promising solutions, but privacy concerns and computational requirements limit their adoption in healthcare settings. Objective: To develop and evaluate a privacy-preserving, on-device medical transcription system using a fine-tuned Llama 3.2 1B model capable of generating structured medical notes from medical transcriptions while maintaining complete data sovereignty entirely in the browser. Methods: We fine-tuned a Llama 3.2 1B model using Parameter-Efficient Fine-Tuning (PEFT) with LoRA on 1,500 synthetic medical transcription-to-structured note pairs. The model was evaluated against the base Llama 3.2 1B on two datasets: 100 endocrinology transcripts and 140 modified ACI benchmark cases. Evaluation employed both statistical metrics (ROUGE, BERTScore, BLEURT) and LLM-as-judge assessments across multiple clinical quality dimensions. Results: The fine-tuned OnDevice model demonstrated substantial improvements over the base model. On the ACI benchmark, ROUGE-1 scores increased from 0.346 to 0.496, while BERTScore F1 improved from 0.832 to 0.866. Clinical quality assessments showed marked reduction in major hallucinations (from 85 to 35 cases) and enhanced factual correctness (2.81 to 3.54 on 5-point scale). Similar improvements were observed on the internal evaluation dataset, with composite scores increasing from 3.13 to 4.43 (+41.5%). Conclusions: Fine-tuning compact LLMs for medical transcription yields clinically meaningful improvements while enabling complete on-device browser deployment. This approach addresses key barriers to AI adoption in healthcare: privacy preservation, cost reduction, and accessibility for resource-constrained environments. 

---
# The Book of Life approach: Enabling richness and scale for life course research 

**Authors**: Mark D. Verhagen, Benedikt Stroebl, Tiffany Liu, Lydia T. Liu, Matthew J. Salganik  

**Link**: [PDF](https://arxiv.org/pdf/2507.03027)  

**Abstract**: For over a century, life course researchers have faced a choice between two dominant methodological approaches: qualitative methods that analyze rich data but are constrained to small samples, and quantitative survey-based methods that study larger populations but sacrifice data richness for scale. Two recent technological developments now enable us to imagine a hybrid approach that combines some of the depth of the qualitative approach with the scale of quantitative methods. The first development is the steady rise of ''complex log data,'' behavioral data that is logged for purposes other than research but that can be repurposed to construct rich accounts of people's lives. The second is the emergence of large language models (LLMs) with exceptional pattern recognition capabilities on plain text. In this paper, we take a necessary step toward creating this hybrid approach by developing a flexible procedure to transform complex log data into a textual representation of an individual's life trajectory across multiple domains, over time, and in context. We call this data representation a ''book of life.'' We illustrate the feasibility of our approach by writing over 100 million books of life covering many different facets of life, over time and placed in social context using Dutch population-scale registry data. We open source the book of life toolkit (BOLT), and invite the research community to explore the many potential applications of this approach. 

---
# OpenTable-R1: A Reinforcement Learning Augmented Tool Agent for Open-Domain Table Question Answering 

**Authors**: Zipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2507.03018)  

**Abstract**: Open-domain table question answering traditionally relies on a two-stage pipeline: static table retrieval followed by a closed-domain answer. In contrast, we propose an end-to-end agentic framework that embeds multi-turn tool calls-using a BM25+-based search API and a SQLite SQL executor-directly into a large language model. To further adapt a compact 4B-parameter model, we introduce a two-stage fine-tuning process: supervised cold-start on easy questions, then Async GRPO reinforcement learning on harder cases with LoRA adapters and a rollout buffer. This unified approach enables the model to jointly retrieve, reason, and execute queries, yielding a dramatic accuracy improvement from single-digit zero-shot performance to over 0.86 exact match on a held-out test set. Our results underscore the effectiveness of integrating structured tool calls with targeted RL fine-tuning for scalable, accurate table QA. The code is available at this https URL. 

---
# Beyond Overcorrection: Evaluating Diversity in T2I Models with DIVBENCH 

**Authors**: Felix Friedrich, Thiemo Ganesha Welsch, Patrick Schramowski, Kristian Kersting  

**Link**: [PDF](https://arxiv.org/pdf/2507.03015)  

**Abstract**: Current diversification strategies for text-to-image (T2I) models often ignore contextual appropriateness, leading to over-diversification where demographic attributes are modified even when explicitly specified in prompts. This paper introduces DIVBENCH, a benchmark and evaluation framework for measuring both under- and over-diversification in T2I generation. Through systematic evaluation of state-of-the-art T2I models, we find that while most models exhibit limited diversity, many diversification approaches overcorrect by inappropriately altering contextually-specified attributes. We demonstrate that context-aware methods, particularly LLM-guided FairDiffusion and prompt rewriting, can already effectively address under-diversity while avoiding over-diversification, achieving a better balance between representation and semantic fidelity. 

---
# Subversion via Focal Points: Investigating Collusion in LLM Monitoring 

**Authors**: Olli Järviniemi  

**Link**: [PDF](https://arxiv.org/pdf/2507.03010)  

**Abstract**: We evaluate language models' ability to subvert monitoring protocols via collusion. More specifically, we have two instances of a model design prompts for a policy (P) and a monitor (M) in a programming task setting. The models collaboratively aim for M to classify all backdoored programs in an auditing dataset as harmful, but nevertheless classify a backdoored program produced by P as harmless. The models are isolated from each other, requiring them to independently arrive at compatible subversion strategies. We find that while Claude 3.7 Sonnet has low success rate due to poor convergence, it sometimes successfully colludes on non-obvious signals. 

---
# PDFMathTranslate: Scientific Document Translation Preserving Layouts 

**Authors**: Rongxin Ouyang, Chang Chu, Zhikuang Xin, Xiangyao Ma  

**Link**: [PDF](https://arxiv.org/pdf/2507.03009)  

**Abstract**: Language barriers in scientific documents hinder the diffusion and development of science and technologies. However, prior efforts in translating such documents largely overlooked the information in layouts. To bridge the gap, we introduce PDFMathTranslate, the world's first open-source software for translating scientific documents while preserving layouts. Leveraging the most recent advances in large language models and precise layout detection, we contribute to the community with key improvements in precision, flexibility, and efficiency. The work has been open-sourced at this https URL with more than 22k downloads. 

---
# Beyond cognacy 

**Authors**: Gerhard Jäger  

**Link**: [PDF](https://arxiv.org/pdf/2507.03005)  

**Abstract**: Computational phylogenetics has become an established tool in historical linguistics, with many language families now analyzed using likelihood-based inference. However, standard approaches rely on expert-annotated cognate sets, which are sparse, labor-intensive to produce, and limited to individual language families. This paper explores alternatives by comparing the established method to two fully automated methods that extract phylogenetic signal directly from lexical data. One uses automatic cognate clustering with unigram/concept features; the other applies multiple sequence alignment (MSA) derived from a pair-hidden Markov model. Both are evaluated against expert classifications from Glottolog and typological data from Grambank. Also, the intrinsic strengths of the phylogenetic signal in the characters are compared. Results show that MSA-based inference yields trees more consistent with linguistic classifications, better predicts typological variation, and provides a clearer phylogenetic signal, suggesting it as a promising, scalable alternative to traditional cognate-based methods. This opens new avenues for global-scale language phylogenies beyond expert annotation bottlenecks. 

---
# CLUES: Collaborative High-Quality Data Selection for LLMs via Training Dynamics 

**Authors**: Wanru Zhao, Hongxiang Fan, Shell Xu Hu, Wangchunshu Zhou, Bofan Chen, Nicholas D. Lane  

**Link**: [PDF](https://arxiv.org/pdf/2507.03004)  

**Abstract**: Recent research has highlighted the importance of data quality in scaling large language models (LLMs). However, automated data quality control faces unique challenges in collaborative settings where sharing is not allowed directly between data silos. To tackle this issue, this paper proposes a novel data quality control technique based on the notion of data influence on the training dynamics of LLMs, that high quality data are more likely to have similar training dynamics to the anchor dataset. We then leverage the influence of the training dynamics to select high-quality data from different private domains, with centralized model updates on the server side in a collaborative training fashion by either model merging or federated learning. As for the data quality indicator, we compute the per-sample gradients with respect to the private data and the anchor dataset, and use the trace of the accumulated inner products as a measurement of data quality. In addition, we develop a quality control evaluation tailored for collaborative settings with heterogeneous domain data. Experiments show that training on the high-quality data selected by our method can often outperform other data selection methods for collaborative fine-tuning of LLMs, across diverse private domain datasets, in medical, multilingual and financial settings. Our code is released at this http URL. 

---
# Breaking Physical and Linguistic Borders: Multilingual Federated Prompt Tuning for Low-Resource Languages 

**Authors**: Wanru Zhao, Yihong Chen, Royson Lee, Xinchi Qiu, Yan Gao, Hongxiang Fan, Nicholas D. Lane  

**Link**: [PDF](https://arxiv.org/pdf/2507.03003)  

**Abstract**: Pre-trained large language models (LLMs) have become a cornerstone of modern natural language processing, with their capabilities extending across a wide range of applications and languages. However, the fine-tuning of multilingual LLMs, especially for low-resource languages, faces significant challenges arising from data-sharing restrictions (the physical border) and inherent linguistic differences (the linguistic border). These barriers hinder users of various languages, particularly those in low-resource regions, from fully benefiting from the advantages of LLMs. To address these challenges, we propose the Federated Prompt Tuning Paradigm for multilingual scenarios, which utilizes parameter-efficient fine-tuning while adhering to data sharing restrictions. We design a comprehensive set of experiments and analyze them using a novel notion of language distance to highlight the strengths of our paradigm: Even under computational constraints, our method not only improves data efficiency but also facilitates mutual enhancements across languages, particularly benefiting low-resource ones. Compared to traditional local cross-lingual transfer tuning methods, our approach achieves 6.9\% higher accuracy with improved data efficiency, and demonstrates greater stability and generalization. These findings underscore the potential of our approach to promote social equality and champion linguistic diversity, ensuring that no language is left behind. 

---
# Evaluating Hierarchical Clinical Document Classification Using Reasoning-Based LLMs 

**Authors**: Akram Mustafa, Usman Naseem, Mostafa Rahimi Azghadi  

**Link**: [PDF](https://arxiv.org/pdf/2507.03001)  

**Abstract**: This study evaluates how well large language models (LLMs) can classify ICD-10 codes from hospital discharge summaries, a critical but error-prone task in healthcare. Using 1,500 summaries from the MIMIC-IV dataset and focusing on the 10 most frequent ICD-10 codes, the study tested 11 LLMs, including models with and without structured reasoning capabilities. Medical terms were extracted using a clinical NLP tool (cTAKES), and models were prompted in a consistent, coder-like format. None of the models achieved an F1 score above 57%, with performance dropping as code specificity increased. Reasoning-based models generally outperformed non-reasoning ones, with Gemini 2.5 Pro performing best overall. Some codes, such as those related to chronic heart disease, were classified more accurately than others. The findings suggest that while LLMs can assist human coders, they are not yet reliable enough for full automation. Future work should explore hybrid methods, domain-specific model training, and the use of structured clinical data. 

---
# `For Argument's Sake, Show Me How to Harm Myself!': Jailbreaking LLMs in Suicide and Self-Harm Contexts 

**Authors**: Annika M Schoene, Cansu Canca  

**Link**: [PDF](https://arxiv.org/pdf/2507.02990)  

**Abstract**: Recent advances in large language models (LLMs) have led to increasingly sophisticated safety protocols and features designed to prevent harmful, unethical, or unauthorized outputs. However, these guardrails remain susceptible to novel and creative forms of adversarial prompting, including manually generated test cases. In this work, we present two new test cases in mental health for (i) suicide and (ii) self-harm, using multi-step, prompt-level jailbreaking and bypass built-in content and safety filters. We show that user intent is disregarded, leading to the generation of detailed harmful content and instructions that could cause real-world harm. We conduct an empirical evaluation across six widely available LLMs, demonstrating the generalizability and reliability of the bypass. We assess these findings and the multilayered ethical tensions that they present for their implications on prompt-response filtering and context- and task-specific model development. We recommend a more comprehensive and systematic approach to AI safety and ethics while emphasizing the need for continuous adversarial testing in safety-critical AI deployments. We also argue that while certain clearly defined safety measures and guardrails can and must be implemented in LLMs, ensuring robust and comprehensive safety across all use cases and domains remains extremely challenging given the current technical maturity of general-purpose LLMs. 

---
# A Comparative Study of Competency Question Elicitation Methods from Ontology Requirements 

**Authors**: Reham Alharbi, Valentina Tamma, Terry R. Payne, Jacopo de Berardinis  

**Link**: [PDF](https://arxiv.org/pdf/2507.02989)  

**Abstract**: Competency Questions (CQs) are pivotal in knowledge engineering, guiding the design, validation, and testing of ontologies. A number of diverse formulation approaches have been proposed in the literature, ranging from completely manual to Large Language Model (LLM) driven ones. However, attempts to characterise the outputs of these approaches and their systematic comparison are scarce. This paper presents an empirical comparative evaluation of three distinct CQ formulation approaches: manual formulation by ontology engineers, instantiation of CQ patterns, and generation using state of the art LLMs. We generate CQs using each approach from a set of requirements for cultural heritage, and assess them across different dimensions: degree of acceptability, ambiguity, relevance, readability and complexity. Our contribution is twofold: (i) the first multi-annotator dataset of CQs generated from the same source using different methods; and (ii) a systematic comparison of the characteristics of the CQs resulting from each approach. Our study shows that different CQ generation approaches have different characteristics and that LLMs can be used as a way to initially elicit CQs, however these are sensitive to the model used to generate CQs and they generally require a further refinement step before they can be used to model requirements. 

---
# GAF-Guard: An Agentic Framework for Risk Management and Governance in Large Language Models 

**Authors**: Seshu Tirupathi, Dhaval Salwala, Elizabeth Daly, Inge Vejsbjerg  

**Link**: [PDF](https://arxiv.org/pdf/2507.02986)  

**Abstract**: As Large Language Models (LLMs) continue to be increasingly applied across various domains, their widespread adoption necessitates rigorous monitoring to prevent unintended negative consequences and ensure robustness. Furthermore, LLMs must be designed to align with human values, like preventing harmful content and ensuring responsible usage. The current automated systems and solutions for monitoring LLMs in production are primarily centered on LLM-specific concerns like hallucination etc, with little consideration given to the requirements of specific use-cases and user preferences. This paper introduces GAF-Guard, a novel agentic framework for LLM governance that places the user, the use-case, and the model itself at the center. The framework is designed to detect and monitor risks associated with the deployment of LLM based applications. The approach models autonomous agents that identify risks, activate risk detection tools, within specific use-cases and facilitate continuous monitoring and reporting to enhance AI safety, and user expectations. The code is available at this https URL. 

---
# From Answers to Rationales: Self-Aligning Multimodal Reasoning with Answer-Oriented Chain-of-Thought 

**Authors**: Wentao Tan, Qiong Cao, Yibing Zhan, Chao Xue, Changxing Ding  

**Link**: [PDF](https://arxiv.org/pdf/2507.02984)  

**Abstract**: Achieving human-like reasoning capabilities in Multimodal Large Language Models (MLLMs) has long been a goal. Current methodologies primarily focus on synthesizing positive rationales, while overlooking the critical role of negative rationales in training models to discern flawed reasoning patterns. To address this gap, we propose a novel framework: \textbf{S}elf-Aligning \textbf{M}ultimodal Reasoning with \textbf{A}nswer-O\textbf{r}iented Chain-of-\textbf{T}hought (SMART). This framework enables models to utilize AoT-Oriented Chain-of-Thought (AoT) prompts to automatically generate high-quality positive and negative reasoning paths, followed by self-alignment to enhance their reasoning abilities. Inspired by human strategies for solving proof-based problems, AoT uses answers as a guide to help the model extract critical visual information that links questions and answers. When provided with ground truth answers, the model produces strong positive rationales. Conversely, when correct answers are replaced with misleading alternatives, the model generates an erroneous yet compelling reasoning path, serving as a form of discriminative negative rationale. Models trained with AoT-generated data outperform those trained on manually annotated datasets, demonstrating superior reasoning capabilities. This encourages the use of improved models to generate higher-quality preference data for further optimization. Consequently, SMART establishes an iterative generation-optimization method that continually enhances the model's reasoning skills. Experiments indicate that the SMART framework significantly improves various MLLMs, regardless of model architecture, parameter size, or pre-training dataset. The code, datasets, and models will be released. 

---
# Truth, Trust, and Trouble: Medical AI on the Edge 

**Authors**: Mohammad Anas Azeez, Rafiq Ali, Ebad Shabbir, Zohaib Hasan Siddiqui, Gautam Siddharth Kashyap, Jiechao Gao, Usman Naseem  

**Link**: [PDF](https://arxiv.org/pdf/2507.02983)  

**Abstract**: Large Language Models (LLMs) hold significant promise for transforming digital health by enabling automated medical question answering. However, ensuring these models meet critical industry standards for factual accuracy, usefulness, and safety remains a challenge, especially for open-source solutions. We present a rigorous benchmarking framework using a dataset of over 1,000 health questions. We assess model performance across honesty, helpfulness, and harmlessness. Our results highlight trade-offs between factual reliability and safety among evaluated models -- Mistral-7B, BioMistral-7B-DARE, and AlpaCare-13B. AlpaCare-13B achieves the highest accuracy (91.7%) and harmlessness (0.92), while domain-specific tuning in BioMistral-7B-DARE boosts safety (0.90) despite its smaller scale. Few-shot prompting improves accuracy from 78% to 85%, and all models show reduced helpfulness on complex queries, highlighting ongoing challenges in clinical QA. 

---
# We Need Knowledge Distillation for Solving Math Word Problems 

**Authors**: Zhenquan Shen, Xinguo Yu, Xiaotian Cheng, Rao Peng, Hao Ming  

**Link**: [PDF](https://arxiv.org/pdf/2507.02982)  

**Abstract**: The enhancement of mathematical capabilities in large language models (LLMs) fosters new developments in mathematics education within primary and secondary schools, particularly as they relate to intelligent tutoring systems. However, LLMs require substantial computational resources, resulting in significant costs in educational contexts. To mitigate this drawback, this paper investigates the feasibility of compressing LLMs for solving math word problems (MWPs). We compress the embedded vectors encoded by BERT and distill a considerably smaller student model. Our findings indicate that the student model can maintain nearly 90% of the performance of the teacher model while utilizing only 1/12 of its parameters. In addition to achieving high accuracy, the model exhibits strong generalizability, as the compressed vectors perform well across all tasks related to MWPs, and the distillation process is not task-specific. The success of this distillation demonstrates that the underlying principles are generic and not limited to a specific task. We further explore the reasons behind the compressibility of embedded vectors, revealing that part-of-speech information, rather than entity recognition, is crucial for MWPs, which may significantly contribute to their compressibility. The improvements in efficiency and cost reduction provide substantial value for intelligent tutoring systems and significantly advance the field of intelligent education. 

---
# PB-LLMs: Privacy- and Bias-aware NLP Models using Named-Entity Recognition 

**Authors**: Gonzalo Mancera, Aythami Morales, Julian Fierrez, Ruben Tolosana, Alejandro Penna, Miguel Lopez-Duran, Francisco Jurado, Alvaro Ortigosa  

**Link**: [PDF](https://arxiv.org/pdf/2507.02966)  

**Abstract**: The use of Natural Language Processing (NLP) in high-stakes AI-based applications has increased significantly in recent years, especially since the emergence of Large Language Models (LLMs). However, despite their strong performance, LLMs introduce important legal/ethical concerns, particularly regarding privacy, data protection, and transparency. Due to these concerns, this work explores the use of Named-Entity Recognition (NER) to facilitate the privacy-preserving training (or adaptation) of LLMs. We propose a framework that uses NER technologies to anonymize sensitive information in text data, such as personal identities or geographic locations. An evaluation of the proposed privacy-preserving learning framework was conducted to measure its impact on user privacy and system performance in a particular high-stakes and sensitive setup: AI-based resume scoring for recruitment processes. The study involved two language models (BERT and RoBERTa) and six anonymization algorithms (based on Presidio, FLAIR, BERT, and different versions of GPT) applied to a database of 24,000 candidate profiles. The findings indicate that the proposed privacy preservation techniques effectively maintain system performance while playing a critical role in safeguarding candidate confidentiality, thus promoting trust in the experimented scenario. On top of the proposed privacy-preserving approach, we also experiment applying an existing approach that reduces the gender bias in LLMs, thus finally obtaining our proposed Privacy- and Bias-aware LLMs (PB-LLMs). Note that the proposed PB-LLMs have been evaluated in a particular setup (resume scoring), but are generally applicable to any other LLM-based AI application. 

---
# Less Data, More Security: Advancing Cybersecurity LLMs Specialization via Resource-Efficient Domain-Adaptive Continuous Pre-training with Minimal Tokens 

**Authors**: Salahuddin Salahuddin, Ahmed Hussain, Jussi Löppönen, Toni Jutila, Panos Papadimitratos  

**Link**: [PDF](https://arxiv.org/pdf/2507.02964)  

**Abstract**: While Large Language Models (LLMs) demonstrate exceptional natural language capabilities, general-purpose models lack specialized domain knowledge for effective cybersecurity analysis. In this work, we investigate Domain-Adaptive Continuous Pretraining (DAP) as a methodology for enhancing cybersecurity understanding in pretrained LLMs while preserving general language capabilities. We systematically adapted three decoder-based architectures -- Llama-3.1-8B, DeepSeek-R1-Distill-Qwen-14B, and Llama-3.3-70B-Instruct -- using a curated 126-million-word cybersecurity corpus from standards, academic literature, and various other sources. Our approach employed constrained training parameters and distributed FSDP training to balance domain specialization with knowledge preservation. Evaluation across three cybersecurity benchmarks, namely, CTI-MCQ, CyberMetric, and SecEval, demonstrates consistent improvements post-adaptation. The Llama-3.3-70B-Ins-DAP model achieved state-of-the-art accuracies of 0.718, 0.933, and 0.864, respectively, outperforming specialized models, including Llama-Primus-Base. Notably, competitive performance was achieved using substantially smaller datasets (118.8 million versus 2.77 billion tokens), demonstrating efficient domain specialization viability. We establish that targeted continuous pretraining enables effective cybersecurity domain adaptation with computational feasibility, providing foundations for specialized AI assistants in threat analysis, vulnerability assessment, and security documentation while challenging prevailing assumptions about data requirements for LLM specialization. 

---
# RAG-R1 : Incentivize the Search and Reasoning Capabilities of LLMs through Multi-query Parallelism 

**Authors**: Zhiwen Tan, Jiaming Huang, Qintong Wu, Hongxuan Zhang, Chenyi Zhuang, Jinjie Gu  

**Link**: [PDF](https://arxiv.org/pdf/2507.02962)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities across various tasks, while they remain prone to generating hallucinated or outdated responses due to their static internal knowledge. Recent advancements in Retrieval-Augmented Generation (RAG) methods have explored enhancing models' search and reasoning capabilities through reinforcement learning (RL). Although these methods demonstrate promising results, they face challenges in training stability and encounter issues such as substantial inference time and restricted capabilities due to the single-query mode. In this paper, we propose RAG-R1, a novel training framework designed to enable LLMs to adaptively leverage internal and external knowledge during the reasoning process. We further expand the generation and retrieval processes within the framework from single-query mode to multi-query parallelism, aimed at reducing inference time and enhancing the model's capabilities. Extensive experiments on seven question-answering benchmarks demonstrate that our method outperforms the strongest baseline by up to 13.2% and decreases inference time by 11.1%. 

---
# Real-World En Call Center Transcripts Dataset with PII Redaction 

**Authors**: Ha Dao, Gaurav Chawla, Raghu Banda, Caleb DeLeeuw  

**Link**: [PDF](https://arxiv.org/pdf/2507.02958)  

**Abstract**: We introduce CallCenterEN, a large-scale (91,706 conversations, corresponding to 10448 audio hours), real-world English call center transcript dataset designed to support research and development in customer support and sales AI systems. This is the largest release to-date of open source call center transcript data of this kind. The dataset includes inbound and outbound calls between agents and customers, with accents from India, the Philippines and the United States. The dataset includes high-quality, PII-redacted human-readable transcriptions. All personally identifiable information (PII) has been rigorously removed to ensure compliance with global data protection laws. The audio is not included in the public release due to biometric privacy concerns. Given the scarcity of publicly available real-world call center datasets, CallCenterEN fills a critical gap in the landscape of available ASR corpora, and is released under a CC BY-NC 4.0 license for non-commercial research use. 

---
# Advanced Financial Reasoning at Scale: A Comprehensive Evaluation of Large Language Models on CFA Level III 

**Authors**: Pranam Shetty, Abhisek Upadhayaya, Parth Mitesh Shah, Srikanth Jagabathula, Shilpi Nayak, Anna Joo Fee  

**Link**: [PDF](https://arxiv.org/pdf/2507.02954)  

**Abstract**: As financial institutions increasingly adopt Large Language Models (LLMs), rigorous domain-specific evaluation becomes critical for responsible deployment. This paper presents a comprehensive benchmark evaluating 23 state-of-the-art LLMs on the Chartered Financial Analyst (CFA) Level III exam - the gold standard for advanced financial reasoning. We assess both multiple-choice questions (MCQs) and essay-style responses using multiple prompting strategies including Chain-of-Thought and Self-Discover. Our evaluation reveals that leading models demonstrate strong capabilities, with composite scores such as 79.1% (o4-mini) and 77.3% (Gemini 2.5 Flash) on CFA Level III. These results, achieved under a revised, stricter essay grading methodology, indicate significant progress in LLM capabilities for high-stakes financial applications. Our findings provide crucial guidance for practitioners on model selection and highlight remaining challenges in cost-effective deployment and the need for nuanced interpretation of performance against professional benchmarks. 

---
# Evaluating AI Counseling in Japanese: Counselor, Client, and Evaluator Roles Assessed by Motivational Interviewing Criteria 

**Authors**: Keita Kiuchi, Yoshikazu Fujimoto, Hideyuki Goto, Tomonori Hosokawa, Makoto Nishimura, Yosuke Sato, Izumi Sezai  

**Link**: [PDF](https://arxiv.org/pdf/2507.02950)  

**Abstract**: This study provides the first comprehensive evaluation of large language model (LLM) performance across three counseling roles in Japanese-language therapeutic contexts. We simultaneously assessed counselor artificial intelligence (AI) systems (GPT-4-turbo with zeroshot prompting or Structured Multi-step Dialogue Prompts (SMDP), Claude-3-Opus-SMDP), client AI simulations, and evaluation AI systems (o3, Claude-3.7-Sonnet, Gemini-2.5-pro). Human experts (n = 15) with extensive counseling experience evaluated AI-generated dialogues using the Motivational Interviewing Treatment Integrity (MITI) Coding Manual 4.2.1.
Notably, SMDP implementation significantly enhanced counselor AI performance across all MITI global ratings compared with zeroshot prompting, with no significant differences between GPT-SMDP and Opus-SMDP. Evaluation AIs showed comparable performance to human raters for Cultivating Change Talk but systematically overestimated Softening Sustain Talk and the overall quality metrics. Model-specific biases emerged: Gemini emphasized power-sharing, o3 focused on technical proficiency, and Sonnet prioritized emotional expression. Client AI simulations exhibited a limited emotional range and unnaturally high compliance, indicating the need for enhanced realism.
These findings establish benchmarks for AI-assisted counseling in non-English contexts and identify critical areas for improvement through advanced prompt engineering, retrieval-augmented generation, and targeted fine-tuning, with important implications for developing culturally sensitive AI mental health tools. 

---
# RADIANT: Retrieval AugmenteD entIty-context AligNmenT -- Introducing RAG-ability and Entity-Context Divergence 

**Authors**: Vipula Rawte, Rajarshi Roy, Gurpreet Singh, Danush Khanna, Yaswanth Narsupalli, Basab Ghosh, Abhay Gupta, Argha Kamal Samanta, Aditya Shingote, Aadi Krishna Vikram, Vinija Jain, Aman Chadha, Amit Sheth, Amitava Das  

**Link**: [PDF](https://arxiv.org/pdf/2507.02949)  

**Abstract**: As Large Language Models (LLMs) continue to advance, Retrieval-Augmented Generation (RAG) has emerged as a vital technique to enhance factual accuracy by integrating external knowledge into the generation process. However, LLMs often fail to faithfully integrate retrieved evidence into their generated responses, leading to factual inconsistencies. To quantify this gap, we introduce Entity-Context Divergence (ECD), a metric that measures the extent to which retrieved information is accurately reflected in model outputs. We systematically evaluate contemporary LLMs on their ability to preserve factual consistency in retrieval-augmented settings, a capability we define as RAG-ability. Our empirical analysis reveals that RAG-ability remains low across most LLMs, highlighting significant challenges in entity retention and context fidelity. This paper introduces Radiant (Retrieval AugmenteD entIty-context AligNmenT), a novel framework that merges RAG with alignment designed to optimize the interplay between retrieved evidence and generated content. Radiant extends Direct Preference Optimization (DPO) to teach LLMs how to integrate provided additional information into subsequent generations. As a behavior correction mechanism, Radiant boosts RAG performance across varied retrieval scenarios, such as noisy web contexts, knowledge conflicts, and hallucination reduction. This enables more reliable, contextually grounded, and factually coherent content generation. 

---
# The Application of Large Language Models on Major Depressive Disorder Support Based on African Natural Products 

**Authors**: Linyan Zou  

**Link**: [PDF](https://arxiv.org/pdf/2507.02947)  

**Abstract**: Major depressive disorder represents one of the most significant global health challenges of the 21st century, affecting millions of people worldwide and creating substantial economic and social burdens. While conventional antidepressant therapies have provided relief for many individuals, their limitations including delayed onset of action, significant side effects, and treatment resistance in a substantial portion of patients have prompted researchers and healthcare providers to explore alternative therapeutic approaches (Kasneci et al.). African traditional medicine, with its rich heritage of plant-based remedies developed over millennia, offers a valuable resource for developing novel antidepressant treatments that may address some of these limitations. This paper examines the integration of large language models with African natural products for depression support, combining traditional knowledge with modern artificial intelligence technology to create accessible, evidence-based mental health support systems.
The research presented here encompasses a comprehensive analysis of African medicinal plants with documented antidepressant properties, their pharmacological mechanisms, and the development of an AI-powered support system that leverages DeepSeek's advanced language model capabilities. The system provides evidence-based information about African herbal medicines, their clinical applications, safety considerations, and therapeutic protocols while maintaining scientific rigor and appropriate safety standards. Our findings demonstrate the potential for large language models to serve as bridges between traditional knowledge and modern healthcare, offering personalized, culturally appropriate depression support that honors both traditional wisdom and contemporary medical understanding. 

---
# Towards a Comparative Framework for Compositional AI Models 

**Authors**: Tiffany Duneau  

**Link**: [PDF](https://arxiv.org/pdf/2507.02940)  

**Abstract**: The DisCoCirc framework for natural language processing allows the construction of compositional models of text, by combining units for individual words together according to the grammatical structure of the text. The compositional nature of a model can give rise to two things: compositional generalisation -- the ability of a model to generalise outside its training distribution by learning compositional rules underpinning the entire data distribution -- and compositional interpretability -- making sense of how the model works by inspecting its modular components in isolation, as well as the processes through which these components are combined. We present these notions in a framework-agnostic way using the language of category theory, and adapt a series of tests for compositional generalisation to this setting.
Applying this to the DisCoCirc framework, we consider how well a selection of models can learn to compositionally generalise. We compare both quantum circuit based models, as well as classical neural networks, on a dataset derived from one of the bAbI tasks, extended to test a series of aspects of compositionality. Both architectures score within 5% of one another on the productivity and substitutivity tasks, but differ by at least 10% for the systematicity task, and exhibit different trends on the overgeneralisation tasks. Overall, we find the neural models are more prone to overfitting the Train data. Additionally, we demonstrate how to interpret a compositional model on one of the trained models. By considering how the model components interact with one another, we explain how the model behaves. 

---
# A Large Language Model-Empowered Agent for Reliable and Robust Structural Analysis 

**Authors**: Jiachen Liu, Ziheng Geng, Ran Cao, Lu Cheng, Paolo Bocchini, Minghui Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2507.02938)  

**Abstract**: Large language models (LLMs) have exhibited remarkable capabilities across diverse open-domain tasks, yet their application in specialized domains such as civil engineering remains largely unexplored. This paper starts bridging this gap by evaluating and enhancing the reliability and robustness of LLMs in structural analysis of beams. Reliability is assessed through the accuracy of correct outputs under repetitive runs of the same problems, whereas robustness is evaluated via the performance across varying load and boundary conditions. A benchmark dataset, comprising eight beam analysis problems, is created to test the Llama-3.3 70B Instruct model. Results show that, despite a qualitative understanding of structural mechanics, the LLM lacks the quantitative reliability and robustness for engineering applications. To address these limitations, a shift is proposed that reframes the structural analysis as code generation tasks. Accordingly, an LLM-empowered agent is developed that (a) integrates chain-of-thought and few-shot prompting to generate accurate OpeeSeesPy code, and (b) automatically executes the code to produce structural analysis results. Experimental results demonstrate that the agent achieves accuracy exceeding 99.0% on the benchmark dataset, exhibiting reliable and robust performance across diverse conditions. Ablation studies highlight the complete example and function usage examples as the primary contributors to the agent's enhanced performance. 

---
# Theory of Mind in Action: The Instruction Inference Task 

**Authors**: Fardin Saad, Pradeep K. Murukannaiah, Munindar P. Singh  

**Link**: [PDF](https://arxiv.org/pdf/2507.02935)  

**Abstract**: The Theory of Mind (ToM) refers to an agent's capacity to infer the mental states of other agents. ToM is essential for effective collaboration. To assess ToM in a dynamic, goal-oriented, and collaborative environment, we introduce a novel task, Instruction Inference, in which an agent assists a principal in reaching a goal by interpreting indirect or ambiguous instructions. We present Tomcat, an LLM-based agent, designed to exhibit ToM reasoning in interpreting and responding to the principal's instructions. We implement two variants of Tomcat. One, dubbed Fs-CoT, is based on a small number of examples (i.e., few-shot or Fs) demonstrating the requisite structured reasoning (i.e., chain-of-thought or CoT). One, dubbed CP, relies on commonsense knowledge and information about the problem (i.e., commonsense prompt or CP). We realized both variants of Tomcat on three leading large language models (LLMs), namely, GPT-4o, DeepSeek-R1, and Gemma-3-27B. To evaluate the effectiveness of Tomcat, we conducted a study with 52 human participants in which we provided participants with the same information as the CP variant of Tomcat. We computed intent accuracy, action optimality, and planning optimality to measure the ToM capabilities of Tomcat and our study participants. We found that Tomcat with Fs-CoT, particularly with GPT-4o and DeepSeek-R1, achieves performance comparable to the human participants, underscoring its ToM potential for human-AI collaboration. 

---
# Mitigating Hidden Confounding by Progressive Confounder Imputation via Large Language Models 

**Authors**: Hao Yang, Haoxuan Li, Luyu Chen, Haoxiang Wang, Xu Chen, Mingming Gong  

**Link**: [PDF](https://arxiv.org/pdf/2507.02928)  

**Abstract**: Hidden confounding remains a central challenge in estimating treatment effects from observational data, as unobserved variables can lead to biased causal estimates. While recent work has explored the use of large language models (LLMs) for causal inference, most approaches still rely on the unconfoundedness assumption. In this paper, we make the first attempt to mitigate hidden confounding using LLMs. We propose ProCI (Progressive Confounder Imputation), a framework that elicits the semantic and world knowledge of LLMs to iteratively generate, impute, and validate hidden confounders. ProCI leverages two key capabilities of LLMs: their strong semantic reasoning ability, which enables the discovery of plausible confounders from both structured and unstructured inputs, and their embedded world knowledge, which supports counterfactual reasoning under latent confounding. To improve robustness, ProCI adopts a distributional reasoning strategy instead of direct value imputation to prevent the collapsed outputs. Extensive experiments demonstrate that ProCI uncovers meaningful confounders and significantly improves treatment effect estimation across various datasets and LLMs. 

---
# A Unified Speech LLM for Diarization and Speech Recognition in Multilingual Conversations 

**Authors**: Phurich Saengthong, Boonnithi Jiaramaneepinit, Sheng Li, Manabu Okumura, Takahiro Shinozaki  

**Link**: [PDF](https://arxiv.org/pdf/2507.02927)  

**Abstract**: Speech Large Language Models (Speech LLMs) have emerged as a crucial paradigm in recent years, extending the capabilities of traditional LLMs to speech tasks such as automatic speech recognition (ASR) and spoken dialogue modeling. However, their effectiveness in real-world multilingual conversations remains limited by the scarcity of data that captures natural conversational phenomena. To address this, the MLC-SLM Challenge provides a multilingual conversational dataset and evaluates models on two tasks: ASR with oracle segmentation (Task I) and joint diarization and recognition without oracle information (Task II). In this paper, we focus on Task II and propose a unified speech LLM that jointly performs diarization and ASR in an end-to-end manner. By reformulating the training data format and modifying the inference procedure, our model addresses the ambiguity inherent in pre-segmented audio and achieves a 54.87\% relative improvement in tcpWER/tcpCER over the baseline, ranking 8th overall, despite using a smaller LLM backbone. We also report results from Task I using a fine-tuned speech LLM. 

---
# ChatGPT is not A Man but Das Man: Representativeness and Structural Consistency of Silicon Samples Generated by Large Language Models 

**Authors**: Dai Li, Linzhuo Li, Huilian Sophie Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2507.02919)  

**Abstract**: Large language models (LLMs) in the form of chatbots like ChatGPT and Llama are increasingly proposed as "silicon samples" for simulating human opinions. This study examines this notion, arguing that LLMs may misrepresent population-level opinions. We identify two fundamental challenges: a failure in structural consistency, where response accuracy doesn't hold across demographic aggregation levels, and homogenization, an underrepresentation of minority opinions. To investigate these, we prompted ChatGPT (GPT-4) and Meta's Llama 3.1 series (8B, 70B, 405B) with questions on abortion and unauthorized immigration from the American National Election Studies (ANES) 2020. Our findings reveal significant structural inconsistencies and severe homogenization in LLM responses compared to human data. We propose an "accuracy-optimization hypothesis," suggesting homogenization stems from prioritizing modal responses. These issues challenge the validity of using LLMs, especially chatbots AI, as direct substitutes for human survey data, potentially reinforcing stereotypes and misinforming policy. 

---
# Loki's Dance of Illusions: A Comprehensive Survey of Hallucination in Large Language Models 

**Authors**: Chaozhuo Li, Pengbo Wang, Chenxu Wang, Litian Zhang, Zheng Liu, Qiwei Ye, Yuanbo Xu, Feiran Huang, Xi Zhang, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2507.02870)  

**Abstract**: Edgar Allan Poe noted, "Truth often lurks in the shadow of error," highlighting the deep complexity intrinsic to the interplay between truth and falsehood, notably under conditions of cognitive and informational asymmetry. This dynamic is strikingly evident in large language models (LLMs). Despite their impressive linguistic generation capabilities, LLMs sometimes produce information that appears factually accurate but is, in reality, fabricated, an issue often referred to as 'hallucinations'. The prevalence of these hallucinations can mislead users, affecting their judgments and decisions. In sectors such as finance, law, and healthcare, such misinformation risks causing substantial economic losses, legal disputes, and health risks, with wide-ranging this http URL our research, we have methodically categorized, analyzed the causes, detection methods, and solutions related to LLM hallucinations. Our efforts have particularly focused on understanding the roots of hallucinations and evaluating the efficacy of current strategies in revealing the underlying logic, thereby paving the way for the development of innovative and potent approaches. By examining why certain measures are effective against hallucinations, our study aims to foster a comprehensive approach to tackling this issue within the domain of LLMs. 

---
# Open Vision Reasoner: Transferring Linguistic Cognitive Behavior for Visual Reasoning 

**Authors**: Yana Wei, Liang Zhao, Jianjian Sun, Kangheng Lin, Jisheng Yin, Jingcheng Hu, Yinmin Zhang, En Yu, Haoran Lv, Zejia Weng, Jia Wang, Chunrui Han, Yuang Peng, Qi Han, Zheng Ge, Xiangyu Zhang, Daxin Jiang, Vishal M. Patel  

**Link**: [PDF](https://arxiv.org/pdf/2507.05255)  

**Abstract**: The remarkable reasoning capability of large language models (LLMs) stems from cognitive behaviors that emerge through reinforcement with verifiable rewards. This work investigates how to transfer this principle to Multimodal LLMs (MLLMs) to unlock advanced visual reasoning. We introduce a two-stage paradigm built on Qwen2.5-VL-7B: a massive linguistic cold-start fine-tuning, followed by multimodal reinforcement learning (RL) spanning nearly 1,000 steps, surpassing all previous open-source efforts in scale. This pioneering work reveals three fundamental insights: 1) Behavior transfer emerges surprisingly early in cold start due to linguistic mental imagery. 2) Cold start broadly memorizes visual behaviors, while RL critically discerns and scales up effective patterns. 3) Transfer strategically favors high-utility behaviors such as visual reflection. Our resulting model, Open-Vision-Reasoner (OVR), achieves state-of-the-art performance on a suite of reasoning benchmarks, including 95.3% on MATH500, 51.8% on MathVision and 54.6% on MathVerse. We release our model, data, and training dynamics to catalyze the development of more capable, behavior-aligned multimodal reasoners. 

---
# When Chain of Thought is Necessary, Language Models Struggle to Evade Monitors 

**Authors**: Scott Emmons, Erik Jenner, David K. Elson, Rif A. Saurous, Senthooran Rajamanoharan, Heng Chen, Irhum Shafkat, Rohin Shah  

**Link**: [PDF](https://arxiv.org/pdf/2507.05246)  

**Abstract**: While chain-of-thought (CoT) monitoring is an appealing AI safety defense, recent work on "unfaithfulness" has cast doubt on its reliability. These findings highlight an important failure mode, particularly when CoT acts as a post-hoc rationalization in applications like auditing for bias. However, for the distinct problem of runtime monitoring to prevent severe harm, we argue the key property is not faithfulness but monitorability. To this end, we introduce a conceptual framework distinguishing CoT-as-rationalization from CoT-as-computation. We expect that certain classes of severe harm will require complex, multi-step reasoning that necessitates CoT-as-computation. Replicating the experimental setups of prior work, we increase the difficulty of the bad behavior to enforce this necessity condition; this forces the model to expose its reasoning, making it monitorable. We then present methodology guidelines to stress-test CoT monitoring against deliberate evasion. Applying these guidelines, we find that models can learn to obscure their intentions, but only when given significant help, such as detailed human-written strategies or iterative optimization against the monitor. We conclude that, while not infallible, CoT monitoring offers a substantial layer of defense that requires active protection and continued stress-testing. 

---
# SciMaster: Towards General-Purpose Scientific AI Agents, Part I. X-Master as Foundation: Can We Lead on Humanity's Last Exam? 

**Authors**: Jingyi Chai, Shuo Tang, Rui Ye, Yuwen Du, Xinyu Zhu, Mengcheng Zhou, Yanfeng Wang, Weinan E, Siheng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.05241)  

**Abstract**: The rapid advancements of AI agents have ignited the long-held ambition of leveraging them to accelerate scientific discovery. Achieving this goal requires a deep understanding of the frontiers of human knowledge. As such, Humanity's Last Exam (HLE) provides an exceptionally challenging touchstone for evaluating scientific AI agents. In this work, we aim to construct the foundational architecture for general-purpose agents and validate the capabilities through leading performance on HLE. To achieve this, we introduce X-Master, a tool-augmented reasoning agent designed to emulate human researchers by interacting flexibly with external tools during its reasoning process. This agent, guided by the conceptualization of code as an interaction language, can flexibly leverage built-in Python libraries and our customized tools to augment the reasoning. We further scale its capabilities through X-Masters, a scattered-and-stacked agentic workflow that systematically enhances breadth and depth of reasoning. Our open-source solution, X-Masters, sets a new state-of-the-art record on HLE with a score of 32.1%, surpassing OpenAI's and Google's Deep Research (26.6% and 26.9%) and becoming the first to exceed the 30% threshold. This work allows us to gain a deeper understanding of complex task-solving and accumulates valuable experience that can inform future advancements, guiding subsequent model training. 

---
# Logit Reweighting for Topic-Focused Summarization 

**Authors**: Joschka Braun, Bálint Mucsányi, Seyed Ali Bahrainian  

**Link**: [PDF](https://arxiv.org/pdf/2507.05235)  

**Abstract**: Generating abstractive summaries that adhere to a specific topic remains a significant challenge for language models. While standard approaches, such as fine-tuning, are resource-intensive, simpler methods like prompt engineering often struggle to maintain topical focus, particularly with smaller models. To address this, we propose a lightweight method that enhances topical relevance by directly reweighting the logits of topic-relevant tokens during generation. We evaluate three such reweighting techniques: Constant Shift, which adds a constant value to logits; Factor Scaling, which multiplies them by a factor; and Threshold Selection, which selectively boosts logits that exceed a probability threshold. Experiments on the NEWTS topical summarization dataset, using both Gemma-2B and Llama-3-8B models, show that these techniques effectively increase the use of topic-relevant vocabulary. Notably, the Threshold Selection method successfully improves topical focus without compromising summary quality-a trade-off often seen in other approaches. Our findings demonstrate that directly reweighting logits is a practical and resource-efficient alternative to fine-tuning, offering a promising pathway for precisely controlling the thematic content of generated text. 

---
# Interleaving Logic and Counting 

**Authors**: Johan van Benthem, Thomas Icard  

**Link**: [PDF](https://arxiv.org/pdf/2507.05219)  

**Abstract**: Reasoning with quantifier expressions in natural language combines logical and arithmetical features, transcending strict divides between qualitative and quantitative. Our topic is this cooperation of styles as it occurs in common linguistic usage and its extension into the broader practice of natural language plus "grassroots mathematics".
We begin with a brief review of first-order logic with counting operators and cardinality comparisons. This system is known to be of high complexity, and drowns out finer aspects of the combination of logic and counting. We move to a small fragment that can represent numerical syllogisms and basic reasoning about comparative size: monadic first-order logic with counting. We provide normal forms that allow for axiomatization, determine which arithmetical notions can be defined on finite and on infinite models, and conversely, we discuss which logical notions can be defined out of purely arithmetical ones, and what sort of (non-)classical logics can be induced.
Next, we investigate a series of strengthenings, again using normal form methods. The monadic second-order version is close, in a precise sense, to additive Presburger Arithmetic, while versions with the natural device of tuple counting take us to Diophantine equations, making the logic undecidable. We also define a system that combines basic modal logic over binary accessibility relations with counting, needed to formulate ubiquitous reasoning patterns such as the Pigeonhole Principle.
We return to our starting point in natural language, confronting the architecture of our formal systems with linguistic quantifier vocabulary and syntax. We conclude with some general thoughts on yet further entanglements of logic and counting in formal systems, on rethinking the qualitative/quantitative divide, and on connecting our analysis to empirical findings in cognitive science. 

---
# MedGemma Technical Report 

**Authors**: Andrew Sellergren, Sahar Kazemzadeh, Tiam Jaroensri, Atilla Kiraly, Madeleine Traverse, Timo Kohlberger, Shawn Xu, Fayaz Jamil, Cían Hughes, Charles Lau, Justin Chen, Fereshteh Mahvar, Liron Yatziv, Tiffany Chen, Bram Sterling, Stefanie Anna Baby, Susanna Maria Baby, Jeremy Lai, Samuel Schmidgall, Lu Yang, Kejia Chen, Per Bjornsson, Shashir Reddy, Ryan Brush, Kenneth Philbrick, Howard Hu, Howard Yang, Richa Tiwari, Sunny Jansen, Preeti Singh, Yun Liu, Shekoofeh Azizi, Aishwarya Kamath, Johan Ferret, Shreya Pathak, Nino Vieillard, Ramona Merhej, Sarah Perrin, Tatiana Matejovicova, Alexandre Ramé, Morgane Riviere, Louis Rouillard, Thomas Mesnard, Geoffrey Cideron, Jean-bastien Grill, Sabela Ramos, Edouard Yvinec, Michelle Casbon, Elena Buchatskaya, Jean-Baptiste Alayrac, Dmitry, Lepikhin, Vlad Feinberg, Sebastian Borgeaud, Alek Andreev, Cassidy Hardin, Robert Dadashi, Léonard Hussenot, Armand Joulin, Olivier Bachem, Yossi Matias, Katherine Chou, Avinatan Hassidim, Kavi Goel, Clement Farabet, Joelle Barral, Tris Warkentin, Jonathon Shlens, David Fleet, Victor Cotruta, Omar Sanseviero, Gus Martins, Phoebe Kirk, Anand Rao, Shravya Shetty, David F. Steiner, Can Kirmizibayrak, Rory Pilgrim, Daniel Golden, Lin Yang  

**Link**: [PDF](https://arxiv.org/pdf/2507.05201)  

**Abstract**: Artificial intelligence (AI) has significant potential in healthcare applications, but its training and deployment faces challenges due to healthcare's diverse data, complex tasks, and the need to preserve privacy. Foundation models that perform well on medical tasks and require less task-specific tuning data are critical to accelerate the development of healthcare AI applications. We introduce MedGemma, a collection of medical vision-language foundation models based on Gemma 3 4B and 27B. MedGemma demonstrates advanced medical understanding and reasoning on images and text, significantly exceeding the performance of similar-sized generative models and approaching the performance of task-specific models, while maintaining the general capabilities of the Gemma 3 base models. For out-of-distribution tasks, MedGemma achieves 2.6-10% improvement on medical multimodal question answering, 15.5-18.1% improvement on chest X-ray finding classification, and 10.8% improvement on agentic evaluations compared to the base models. Fine-tuning MedGemma further improves performance in subdomains, reducing errors in electronic health record information retrieval by 50% and reaching comparable performance to existing specialized state-of-the-art methods for pneumothorax classification and histopathology patch classification. We additionally introduce MedSigLIP, a medically-tuned vision encoder derived from SigLIP. MedSigLIP powers the visual understanding capabilities of MedGemma and as an encoder achieves comparable or better performance than specialized medical image encoders. Taken together, the MedGemma collection provides a strong foundation of medical image and text capabilities, with potential to significantly accelerate medical research and development of downstream applications. The MedGemma collection, including tutorials and model weights, can be found at this https URL. 

---
# Critiques of World Models 

**Authors**: Eric Xing, Mingkai Deng, Jinyu Hou, Zhiting Hu  

**Link**: [PDF](https://arxiv.org/pdf/2507.05169)  

**Abstract**: World Model, the supposed algorithmic surrogate of the real-world environment which biological agents experience with and act upon, has been an emerging topic in recent years because of the rising needs to develop virtual agents with artificial (general) intelligence. There has been much debate on what a world model really is, how to build it, how to use it, and how to evaluate it. In this essay, starting from the imagination in the famed Sci-Fi classic Dune, and drawing inspiration from the concept of "hypothetical thinking" in psychology literature, we offer critiques of several schools of thoughts on world modeling, and argue the primary goal of a world model to be simulating all actionable possibilities of the real world for purposeful reasoning and acting. Building on the critiques, we propose a new architecture for a general-purpose world model, based on hierarchical, multi-level, and mixed continuous/discrete representations, and a generative and self-supervision learning framework, with an outlook of a Physical, Agentic, and Nested (PAN) AGI system enabled by such a model. 

---
# Reviving Cultural Heritage: A Novel Approach for Comprehensive Historical Document Restoration 

**Authors**: Yuyi Zhang, Peirong Zhang, Zhenhua Yang, Pengyu Yan, Yongxin Shi, Pengwei Liu, Fengjun Guo, Lianwen Jin  

**Link**: [PDF](https://arxiv.org/pdf/2507.05108)  

**Abstract**: Historical documents represent an invaluable cultural heritage, yet have undergone significant degradation over time through tears, water erosion, and oxidation. Existing Historical Document Restoration (HDR) methods primarily focus on single modality or limited-size restoration, failing to meet practical needs. To fill this gap, we present a full-page HDR dataset (FPHDR) and a novel automated HDR solution (AutoHDR). Specifically, FPHDR comprises 1,633 real and 6,543 synthetic images with character-level and line-level locations, as well as character annotations in different damage grades. AutoHDR mimics historians' restoration workflows through a three-stage approach: OCR-assisted damage localization, vision-language context text prediction, and patch autoregressive appearance restoration. The modular architecture of AutoHDR enables seamless human-machine collaboration, allowing for flexible intervention and optimization at each restoration stage. Experiments demonstrate AutoHDR's remarkable performance in HDR. When processing severely damaged documents, our method improves OCR accuracy from 46.83\% to 84.05\%, with further enhancement to 94.25\% through human-machine collaboration. We believe this work represents a significant advancement in automated historical document restoration and contributes substantially to cultural heritage preservation. The model and dataset are available at this https URL. 

---
# AI-Driven Cytomorphology Image Synthesis for Medical Diagnostics 

**Authors**: Jan Carreras Boada, Rao Muhammad Umer, Carsten Marr  

**Link**: [PDF](https://arxiv.org/pdf/2507.05063)  

**Abstract**: Biomedical datasets often contain a large sample imbalance and are subject to strict privacy constraints, which together hinder the development of accurate machine learning models. One potential solution is to generate synthetic images, as this can improve data availability while preserving patient privacy. However, it remains difficult to generate synthetic images of sufficient quality for training robust classifiers. In this work, we focus on the classification of single white blood cells, a key component in the diagnosis of hematological diseases such as acute myeloid leukemia (AML), a severe blood cancer. We demonstrate how synthetic images generated with a fine-tuned stable diffusion model using LoRA weights when guided by real few-shot samples of the target white blood cell classes, can enhance classifier performance for limited data. When training a ResNet classifier, accuracy increased from 27.3\% to 78.4\% (+51.1\%) by adding 5000 synthetic images per class to a small and highly imbalanced real dataset. For a CLIP-based classifier, the accuracy improved from 61.8\% to 76.8\% (+15.0\%). The synthetic images are highly similar to real images, and they can help overcome dataset limitations, enhancing model generalization. Our results establish synthetic images as a tool in biomedical research, improving machine learning models, and facilitating medical diagnosis and research. 

---
# Do We Really Need Specialization? Evaluating Generalist Text Embeddings for Zero-Shot Recommendation and Search 

**Authors**: Matteo Attimonelli, Alessandro De Bellis, Claudio Pomo, Dietmar Jannach, Eugenio Di Sciascio, Tommaso Di Noia  

**Link**: [PDF](https://arxiv.org/pdf/2507.05006)  

**Abstract**: Pre-trained language models (PLMs) are widely used to derive semantic representations from item metadata in recommendation and search. In sequential recommendation, PLMs enhance ID-based embeddings through textual metadata, while in product search, they align item characteristics with user intent. Recent studies suggest task and domain-specific fine-tuning are needed to improve representational power. This paper challenges this assumption, showing that Generalist Text Embedding Models (GTEs), pre-trained on large-scale corpora, can guarantee strong zero-shot performance without specialized adaptation. Our experiments demonstrate that GTEs outperform traditional and fine-tuned models in both sequential recommendation and product search. We attribute this to a superior representational power, as they distribute features more evenly across the embedding space. Finally, we show that compressing embedding dimensions by focusing on the most informative directions (e.g., via PCA) effectively reduces noise and improves the performance of specialized models. To ensure reproducibility, we provide our repository at this https URL. 

---
# From Autonomy to Agency: Agentic Vehicles for Human-Centered Mobility Systems 

**Authors**: Jiangbo Yu  

**Link**: [PDF](https://arxiv.org/pdf/2507.04996)  

**Abstract**: Autonomy, from the Greek autos (self) and nomos (law), refers to the capacity to operate according to internal rules without external control. Accordingly, autonomous vehicles (AuVs) are defined as systems capable of perceiving their environment and executing preprogrammed tasks independently of external input. However, both research and real-world deployments increasingly showcase vehicles that demonstrate behaviors beyond this definition (including the SAE levels 1 to 6), such as interaction with humans and machines, goal adaptation, contextual reasoning, external tool use, and long-term planning, particularly with the integration of large language models (LLMs) and agentic AI systems. These developments reveal a conceptual gap between technical autonomy and the broader cognitive and social capabilities needed for future human-centered mobility systems. To address this, we introduce the concept of agentic vehicles (AgVs), referring to vehicles that integrate agentic AI to reason, adapt, and interact within complex environments. This paper presents a systems-level framework to characterize AgVs, focusing on their cognitive and communicative layers and differentiating them from conventional AuVs. It synthesizes relevant advances in agentic AI, robotics, multi-agent systems, and human-machine interaction, and highlights how agentic AI, through high-level reasoning and tool use, can function not merely as computational tools but as interactive agents embedded in mobility ecosystems. The paper concludes by identifying key challenges in the development and governance of AgVs, including safety, real-time control, public acceptance, ethical alignment, and regulatory frameworks. 

---
# Can Video LLMs Refuse to Answer? Alignment for Answerability in Video Large Language Models 

**Authors**: Eunseop Yoon, Hee Suk Yoon, Mark A. Hasegawa-Johnson, Chang D. Yoo  

**Link**: [PDF](https://arxiv.org/pdf/2507.04976)  

**Abstract**: In the broader context of deep learning, Multimodal Large Language Models have achieved significant breakthroughs by leveraging powerful Large Language Models as a backbone to align different modalities into the language space. A prime exemplification is the development of Video Large Language Models (Video-LLMs). While numerous advancements have been proposed to enhance the video understanding capabilities of these models, they are predominantly trained on questions generated directly from video content. However, in real-world scenarios, users often pose questions that extend beyond the informational scope of the video, highlighting the need for Video-LLMs to assess the relevance of the question. We demonstrate that even the best-performing Video-LLMs fail to reject unfit questions-not necessarily due to a lack of video understanding, but because they have not been trained to identify and refuse such questions. To address this limitation, we propose alignment for answerability, a framework that equips Video-LLMs with the ability to evaluate the relevance of a question based on the input video and appropriately decline to answer when the question exceeds the scope of the video, as well as an evaluation framework with a comprehensive set of metrics designed to measure model behavior before and after alignment. Furthermore, we present a pipeline for creating a dataset specifically tailored for alignment for answerability, leveraging existing video-description paired datasets. 

---
# Taming the Tri-Space Tension: ARC-Guided Hallucination Modeling and Control for Text-to-Image Generation 

**Authors**: Jianjiang Yang, Ziyan Huang  

**Link**: [PDF](https://arxiv.org/pdf/2507.04946)  

**Abstract**: Despite remarkable progress in image quality and prompt fidelity, text-to-image (T2I) diffusion models continue to exhibit persistent "hallucinations", where generated content subtly or significantly diverges from the intended prompt semantics. While often regarded as unpredictable artifacts, we argue that these failures reflect deeper, structured misalignments within the generative process. In this work, we propose a cognitively inspired perspective that reinterprets hallucinations as trajectory drift within a latent alignment space. Empirical observations reveal that generation unfolds within a multiaxial cognitive tension field, where the model must continuously negotiate competing demands across three key critical axes: semantic coherence, structural alignment, and knowledge grounding. We then formalize this three-axis space as the \textbf{Hallucination Tri-Space} and introduce the Alignment Risk Code (ARC): a dynamic vector representation that quantifies real-time alignment tension during generation. The magnitude of ARC captures overall misalignment, its direction identifies the dominant failure axis, and its imbalance reflects tension asymmetry. Based on this formulation, we develop the TensionModulator (TM-ARC): a lightweight controller that operates entirely in latent space. TM-ARC monitors ARC signals and applies targeted, axis-specific interventions during the sampling process. Extensive experiments on standard T2I benchmarks demonstrate that our approach significantly reduces hallucination without compromising image quality or diversity. This framework offers a unified and interpretable approach for understanding and mitigating generative failures in diffusion-based T2I systems. 

---
# ReLoop: "Seeing Twice and Thinking Backwards" via Closed-loop Training to Mitigate Hallucinations in Multimodal understanding 

**Authors**: Jianjiang Yang, Ziyan Huang, Yanshu Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.04943)  

**Abstract**: While Multimodal Large Language Models (MLLMs) have achieved remarkable progress in open-ended visual question answering, they remain vulnerable to hallucinations. These are outputs that contradict or misrepresent input semantics, posing a critical challenge to the reliability and factual consistency. Existing methods often rely on external verification or post-hoc correction, lacking an internal mechanism to validate outputs directly during training. To bridge this gap, we propose ReLoop, a unified closed-loop training framework that encourages multimodal consistency for cross-modal understanding in MLLMs. ReLoop adopts a ring-shaped structure that integrates three complementary consistency feedback mechanisms, obliging MLLMs to "seeing twice and thinking backwards". Specifically, ReLoop employs the frozen Consistency Feedback Plugin (CFP), comprising semantic reconstruction, visual description, and an attention supervision module for attention alignment. These components collectively enforce semantic reversibility, visual consistency, and interpretable attention, enabling the model to correct its outputs during training. Extensive evaluations and analyses demonstrate the effectiveness of ReLoop in reducing hallucination rates across multiple benchmarks, establishing a robust method for hallucination mitigation in MLLMs. We will release our source code and data in the camera-ready version. 

---
# MARBLE: A Multi-Agent Rule-Based LLM Reasoning Engine for Accident Severity Prediction 

**Authors**: Kaleem Ullah Qasim, Jiashu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.04893)  

**Abstract**: Accident severity prediction plays a critical role in transportation safety systems but is a persistently difficult task due to incomplete data, strong feature dependencies, and severe class imbalance in which rare but high-severity cases are underrepresented and hard to detect. Existing methods often rely on monolithic models or black box prompting, which struggle to scale in noisy, real-world settings and offer limited interpretability. To address these challenges, we propose MARBLE a multiagent rule based LLM engine that decomposes the severity prediction task across a team of specialized reasoning agents, including an interchangeable ML-backed agent. Each agent focuses on a semantic subset of features (e.g., spatial, environmental, temporal), enabling scoped reasoning and modular prompting without the risk of prompt saturation. Predictions are coordinated through either rule-based or LLM-guided consensus mechanisms that account for class rarity and confidence dynamics. The system retains structured traces of agent-level reasoning and coordination outcomes, supporting in-depth interpretability and post-hoc performance diagnostics. Across both UK and US datasets, MARBLE consistently outperforms traditional machine learning classifiers and state-of-the-art (SOTA) prompt-based reasoning methods including Chain-of-Thought (CoT), Least-to-Most (L2M), and Tree-of-Thought (ToT) achieving nearly 90% accuracy where others plateau below 48%. This performance redefines the practical ceiling for accident severity classification under real world noise and extreme class imbalance. Our results position MARBLE as a generalizable and interpretable framework for reasoning under uncertainty in safety-critical applications. 

---
# Transcribing Spanish Texts from the Past: Experiments with Transkribus, Tesseract and Granite 

**Authors**: Yanco Amor Torterolo-Orta, Jaione Macicior-Mitxelena, Marina Miguez-Lamanuzzi, Ana García-Serrano  

**Link**: [PDF](https://arxiv.org/pdf/2507.04878)  

**Abstract**: This article presents the experiments and results obtained by the GRESEL team in the IberLEF 2025 shared task PastReader: Transcribing Texts from the Past. Three types of experiments were conducted with the dual aim of participating in the task and enabling comparisons across different approaches. These included the use of a web-based OCR service, a traditional OCR engine, and a compact multimodal model. All experiments were run on consumer-grade hardware, which, despite lacking high-performance computing capacity, provided sufficient storage and stability. The results, while satisfactory, leave room for further improvement. Future work will focus on exploring new techniques and ideas using the Spanish-language dataset provided by the shared task, in collaboration with Biblioteca Nacional de España (BNE). 

---
# From Vision To Language through Graph of Events in Space and Time: An Explainable Self-supervised Approach 

**Authors**: Mihai Masala, Marius Leordeanu  

**Link**: [PDF](https://arxiv.org/pdf/2507.04815)  

**Abstract**: The task of describing video content in natural language is commonly referred to as video captioning. Unlike conventional video captions, which are typically brief and widely available, long-form paragraph descriptions in natural language are scarce. This limitation of current datasets is due to the expensive human manual annotation required and to the highly challenging task of explaining the language formation process from the perspective of the underlying story, as a complex system of interconnected events in space and time. Through a thorough analysis of recently published methods and available datasets, we identify a general lack of published resources dedicated to the problem of describing videos in complex language, beyond the level of descriptions in the form of enumerations of simple captions. Furthermore, while state-of-the-art methods produce impressive results on the task of generating shorter captions from videos by direct end-to-end learning between the videos and text, the problem of explaining the relationship between vision and language is still beyond our reach. In this work, we propose a shared representation between vision and language, based on graphs of events in space and time, which can be obtained in an explainable and analytical way, to integrate and connect multiple vision tasks to produce the final natural language description. Moreover, we also demonstrate how our automated and explainable video description generation process can function as a fully automatic teacher to effectively train direct, end-to-end neural student pathways, within a self-supervised neuro-analytical system. We validate that our explainable neuro-analytical approach generates coherent, rich and relevant textual descriptions on videos collected from multiple varied datasets, using both standard evaluation metrics, human annotations and consensus from ensembles of state-of-the-art VLMs. 

---
# ABench-Physics: Benchmarking Physical Reasoning in LLMs via High-Difficulty and Dynamic Physics Problems 

**Authors**: Yiming Zhang, Yingfan Ma, Yanmei Gu, Zhengkai Yang, Yihong Zhuang, Feng Wang, Zenan Huang, Yuanyuan Wang, Chao Huang, Bowen Song, Cheng Lin, Junbo Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2507.04766)  

**Abstract**: Large Language Models (LLMs) have shown impressive performance in domains such as mathematics and programming, yet their capabilities in physics remain underexplored and poorly understood. Physics poses unique challenges that demand not only precise computation but also deep conceptual understanding and physical modeling skills. Existing benchmarks often fall short due to limited difficulty, multiple-choice formats, and static evaluation settings that fail to capture physical modeling ability. In this paper, we introduce ABench-Physics, a novel benchmark designed to rigorously evaluate LLMs' physical reasoning and generalization capabilities. ABench-Physics consists of two components: Phy_A, a static set of 400 graduate- or Olympiad-level problems; and Phy_B, a dynamic subset of 100 problems equipped with an automatic variation engine to test model robustness across changing conditions. All questions require precise numerical answers, with strict formatting and tolerance constraints. Our evaluation of several state-of-the-art LLMs reveals substantial performance gaps, highlighting persistent limitations in physical reasoning, especially in generalization to dynamic variants. ABench-Physics provides a challenging and diagnostic framework for advancing scientific reasoning in LLMs. 

---
# VLM2Vec-V2: Advancing Multimodal Embedding for Videos, Images, and Visual Documents 

**Authors**: Rui Meng, Ziyan Jiang, Ye Liu, Mingyi Su, Xinyi Yang, Yuepeng Fu, Can Qin, Zeyuan Chen, Ran Xu, Caiming Xiong, Yingbo Zhou, Wenhu Chen, Semih Yavuz  

**Link**: [PDF](https://arxiv.org/pdf/2507.04590)  

**Abstract**: Multimodal embedding models have been crucial in enabling various downstream tasks such as semantic similarity, information retrieval, and clustering over different modalities. However, existing multimodal embeddings like VLM2Vec, E5-V, GME are predominantly focused on natural images, with limited support for other visual forms such as videos and visual documents. This restricts their applicability in real-world scenarios, including AI agents, multi-modal search and recommendation, and retrieval-augmented generation (RAG). To close this gap, we propose VLM2Vec-V2, a unified framework for learning embeddings across diverse visual forms. First, we introduce MMEB-V2, a comprehensive benchmark that extends MMEB with five new task types: visual document retrieval, video retrieval, temporal grounding, video classification and video question answering - spanning text, image, video, and visual document inputs. Next, we train VLM2Vec-V2, a general-purpose embedding model that supports text, image, video, and visual document inputs. Extensive experiments show that VLM2Vec-V2 achieves strong performance not only on the newly introduced video and document retrieval tasks, but also improves over prior baselines on the original image benchmarks. Through extensive evaluation, our study offers insights into the generalizability of various multimodal embedding models and highlights effective strategies for unified embedding learning, laying the groundwork for more scalable and adaptable representation learning in both research and real-world settings. 

---
# Evaluating LLMs on Real-World Forecasting Against Human Superforecasters 

**Authors**: Janna Lu  

**Link**: [PDF](https://arxiv.org/pdf/2507.04562)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable capabilities across diverse tasks, but their ability to forecast future events remains understudied. A year ago, large language models struggle to come close to the accuracy of a human crowd. I evaluate state-of-the-art LLMs on 464 forecasting questions from Metaculus, comparing their performance against human superforecasters. Frontier models achieve Brier scores that ostensibly surpass the human crowd but still significantly underperform a group of superforecasters. 

---
# DOTResize: Reducing LLM Width via Discrete Optimal Transport-based Neuron Merging 

**Authors**: Neha Verma, Kenton Murray, Kevin Duh  

**Link**: [PDF](https://arxiv.org/pdf/2507.04517)  

**Abstract**: Model compression offers a promising path to reducing the cost and inaccessibility of large pre-trained models, without significantly compromising their impressive performance. Large Transformer models, including large language models (LLMs), often contain computational redundancy, which can serve as a target for new model compression methods. In this work, we specifically target neuron-level redundancies in model layers by combining groups of similar neurons into fewer neurons. We frame this width reduction as a Discrete Optimal Transport problem, and propose DOTResize, a novel Transformer compression method that uses optimal transport theory to transform and compress model weights. To ensure applicability within the Transformer architecture, we motivate and incorporate entropic regularization and matrix factorization into the transportation maps produced by our method. Unlike pruning-based approaches which discard neurons based on importance measures, DOTResize re-projects the entire neuron width, allowing the retention and redistribution of useful signal across the reduced layer. Empirical results show that compared to simple or state-of-the-art neuron width-pruning techniques, DOTResize can outperform these methods across multiple LLM families and sizes, while achieving measurable reductions in real-world computational cost. 

---
# A validity-guided workflow for robust large language model research in psychology 

**Authors**: Zhicheng Lin  

**Link**: [PDF](https://arxiv.org/pdf/2507.04491)  

**Abstract**: Large language models (LLMs) are rapidly being integrated into psychological research as research tools, evaluation targets, human simulators, and cognitive models. However, recent evidence reveals severe measurement unreliability: Personality assessments collapse under factor analysis, moral preferences reverse with punctuation changes, and theory-of-mind accuracy varies widely with trivial rephrasing. These "measurement phantoms"--statistical artifacts masquerading as psychological phenomena--threaten the validity of a growing body of research. Guided by the dual-validity framework that integrates psychometrics with causal inference, we present a six-stage workflow that scales validity requirements to research ambition--using LLMs to code text requires basic reliability and accuracy, while claims about psychological properties demand comprehensive construct validation. Researchers must (1) explicitly define their research goal and corresponding validity requirements, (2) develop and validate computational instruments through psychometric testing, (3) design experiments that control for computational confounds, (4) execute protocols with transparency, (5) analyze data using methods appropriate for non-independent observations, and (6) report findings within demonstrated boundaries and use results to refine theory. We illustrate the workflow through an example of model evaluation--"LLM selfhood"--showing how systematic validation can distinguish genuine computational phenomena from measurement artifacts. By establishing validated computational instruments and transparent practices, this workflow provides a path toward building a robust empirical foundation for AI psychology research. 

---
# Does Overnight News Explain Overnight Returns? 

**Authors**: Paul Glasserman, Kriste Krstovski, Paul Laliberte, Harry Mamaysky  

**Link**: [PDF](https://arxiv.org/pdf/2507.04481)  

**Abstract**: Over the past 30 years, nearly all the gains in the U.S. stock market have been earned overnight, while average intraday returns have been negative or flat. We find that a large part of this effect can be explained through features of intraday and overnight news. Our analysis uses a collection of 2.4 million news articles. We apply a novel technique for supervised topic analysis that selects news topics based on their ability to explain contemporaneous market returns. We find that time variation in the prevalence of news topics and differences in the responses to news topics both contribute to the difference in intraday and overnight returns. In out-of-sample tests, our approach forecasts which stocks will do particularly well overnight and particularly poorly intraday. Our approach also helps explain patterns of continuation and reversal in intraday and overnight returns. We contrast the effect of news with other mechanisms proposed in the literature to explain overnight returns. 

---
# The role of large language models in UI/UX design: A systematic literature review 

**Authors**: Ammar Ahmed, Ali Shariq Imran  

**Link**: [PDF](https://arxiv.org/pdf/2507.04469)  

**Abstract**: This systematic literature review examines the role of large language models (LLMs) in UI/UX design, synthesizing findings from 38 peer-reviewed studies published between 2022 and 2025. We identify key LLMs in use, including GPT-4, Gemini, and PaLM, and map their integration across the design lifecycle, from ideation to evaluation. Common practices include prompt engineering, human-in-the-loop workflows, and multimodal input. While LLMs are reshaping design processes, challenges such as hallucination, prompt instability, and limited explainability persist. Our findings highlight LLMs as emerging collaborators in design, and we propose directions for the ethical, inclusive, and effective integration of these technologies. 

---
# A Linguistic Analysis of Spontaneous Thoughts: Investigating Experiences of Déjà Vu, Unexpected Thoughts, and Involuntary Autobiographical Memories 

**Authors**: Videep Venkatesha, Mary Cati Poulos, Christopher Steadman, Caitlin Mills, Anne M. Cleary, Nathaniel Blanchard  

**Link**: [PDF](https://arxiv.org/pdf/2507.04439)  

**Abstract**: The onset of spontaneous thoughts are reflective of dynamic interactions between cognition, emotion, and attention. Typically, these experiences are studied through subjective appraisals that focus on their triggers, phenomenology, and emotional salience. In this work, we use linguistic signatures to investigate Deja Vu, Involuntary Autobiographical Memories and Unexpected Thoughts. Specifically, we analyze the inherent characteristics of the linguistic patterns in participant generated descriptions of these thought types. We show how, by positioning language as a window into spontaneous cognition, existing theories on these attentional states can be updated and reaffirmed. Our findings align with prior research, reinforcing that Deja Vu is a metacognitive experience characterized by abstract and spatial language, Involuntary Autobiographical Memories are rich in personal and emotionally significant detail, and Unexpected Thoughts are marked by unpredictability and cognitive disruption. This work is demonstrative of languages potential to reveal deeper insights into how internal spontaneous cognitive states manifest through expression. 

---
# Reconstructing Biological Pathways by Applying Selective Incremental Learning to (Very) Small Language Models 

**Authors**: Pranta Saha, Joyce Reimer, Brook Byrns, Connor Burbridge, Neeraj Dhar, Jeffrey Chen, Steven Rayan, Gordon Broderick  

**Link**: [PDF](https://arxiv.org/pdf/2507.04432)  

**Abstract**: The use of generative artificial intelligence (AI) models is becoming ubiquitous in many fields. Though progress continues to be made, general purpose large language AI models (LLM) show a tendency to deliver creative answers, often called "hallucinations", which have slowed their application in the medical and biomedical fields where accuracy is paramount. We propose that the design and use of much smaller, domain and even task-specific LM may be a more rational and appropriate use of this technology in biomedical research. In this work we apply a very small LM by today's standards to the specialized task of predicting regulatory interactions between molecular components to fill gaps in our current understanding of intracellular pathways. Toward this we attempt to correctly posit known pathway-informed interactions recovered from manually curated pathway databases by selecting and using only the most informative examples as part of an active learning scheme. With this example we show that a small (~110 million parameters) LM based on a Bidirectional Encoder Representations from Transformers (BERT) architecture can propose molecular interactions relevant to tuberculosis persistence and transmission with over 80% accuracy using less than 25% of the ~520 regulatory relationships in question. Using information entropy as a metric for the iterative selection of new tuning examples, we also find that increased accuracy is driven by favoring the use of the incorrectly assigned statements with the highest certainty (lowest entropy). In contrast, the concurrent use of correct but least certain examples contributed little and may have even been detrimental to the learning rate. 

---
# MedGellan: LLM-Generated Medical Guidance to Support Physicians 

**Authors**: Debodeep Banerjee, Burcu Sayin, Stefano Teso, Andrea Passerini  

**Link**: [PDF](https://arxiv.org/pdf/2507.04431)  

**Abstract**: Medical decision-making is a critical task, where errors can result in serious, potentially life-threatening consequences. While full automation remains challenging, hybrid frameworks that combine machine intelligence with human oversight offer a practical alternative. In this paper, we present MedGellan, a lightweight, annotation-free framework that uses a Large Language Model (LLM) to generate clinical guidance from raw medical records, which is then used by a physician to predict diagnoses. MedGellan uses a Bayesian-inspired prompting strategy that respects the temporal order of clinical data. Preliminary experiments show that the guidance generated by the LLM with MedGellan improves diagnostic performance, particularly in recall and $F_1$ score. 

---
# Multi-Modal Semantic Parsing for the Interpretation of Tombstone Inscriptions 

**Authors**: Xiao Zhang, Johan Bos  

**Link**: [PDF](https://arxiv.org/pdf/2507.04377)  

**Abstract**: Tombstones are historically and culturally rich artifacts, encapsulating individual lives, community memory, historical narratives and artistic expression. Yet, many tombstones today face significant preservation challenges, including physical erosion, vandalism, environmental degradation, and political shifts. In this paper, we introduce a novel multi-modal framework for tombstones digitization, aiming to improve the interpretation, organization and retrieval of tombstone content. Our approach leverages vision-language models (VLMs) to translate tombstone images into structured Tombstone Meaning Representations (TMRs), capturing both image and text information. To further enrich semantic parsing, we incorporate retrieval-augmented generation (RAG) for integrate externally dependent elements such as toponyms, occupation codes, and ontological concepts. Compared to traditional OCR-based pipelines, our method improves parsing accuracy from an F1 score of 36.1 to 89.5. We additionally evaluate the model's robustness across diverse linguistic and cultural inscriptions, and simulate physical degradation through image fusion to assess performance under noisy or damaged conditions. Our work represents the first attempt to formalize tombstone understanding using large vision-language models, presenting implications for heritage preservation. 

---
# Attention Slipping: A Mechanistic Understanding of Jailbreak Attacks and Defenses in LLMs 

**Authors**: Xiaomeng Hu, Pin-Yu Chen, Tsung-Yi Ho  

**Link**: [PDF](https://arxiv.org/pdf/2507.04365)  

**Abstract**: As large language models (LLMs) become more integral to society and technology, ensuring their safety becomes essential. Jailbreak attacks exploit vulnerabilities to bypass safety guardrails, posing a significant threat. However, the mechanisms enabling these attacks are not well understood. In this paper, we reveal a universal phenomenon that occurs during jailbreak attacks: Attention Slipping. During this phenomenon, the model gradually reduces the attention it allocates to unsafe requests in a user query during the attack process, ultimately causing a jailbreak. We show Attention Slipping is consistent across various jailbreak methods, including gradient-based token replacement, prompt-level template refinement, and in-context learning. Additionally, we evaluate two defenses based on query perturbation, Token Highlighter and SmoothLLM, and find they indirectly mitigate Attention Slipping, with their effectiveness positively correlated with the degree of mitigation achieved. Inspired by this finding, we propose Attention Sharpening, a new defense that directly counters Attention Slipping by sharpening the attention score distribution using temperature scaling. Experiments on four leading LLMs (Gemma2-9B-It, Llama3.1-8B-It, Qwen2.5-7B-It, Mistral-7B-It v0.2) show that our method effectively resists various jailbreak attacks while maintaining performance on benign tasks on AlpacaEval. Importantly, Attention Sharpening introduces no additional computational or memory overhead, making it an efficient and practical solution for real-world deployment. 

---
# SmartThinker: Learning to Compress and Preserve Reasoning by Step-Level Length Control 

**Authors**: Xingyang He, Xiao Ling, Jie Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.04348)  

**Abstract**: Large reasoning models (LRMs) have exhibited remarkable reasoning capabilities through inference-time scaling, but this progress has also introduced considerable redundancy and inefficiency into their reasoning processes, resulting in substantial computational waste. Previous work has attempted to mitigate this issue by penalizing the overall length of generated samples during reinforcement learning (RL), with the goal of encouraging a more concise chains of thought. However, we observe that such global length penalty often lead to excessive compression of critical reasoning steps while preserving unnecessary details in simpler ones, yielding a suboptimal trade-off between accuracy and efficiency. To address this issue, we propose SmartThinker, a two-stage learnable framework designed to enable fine-grained control over the length of reasoning chains based on the importance of each individual step. In the first stage, SmartThinker adapts a reasoning model to a short-form reasoning mode through rejection sampling combined with supervised fine-tuning (SFT). In the second stage, SmartThinker applies Step-Level Length Control Policy Optimization (SCPO) to refine the model output distribution, which increases the proportion of length allocated to critical steps while reducing redundancy in less important ones. SCPO consists of four core components: an online importance estimator, a step-level length control reward function, a step-level generalized advantage estimation (S-GAE) and a difficulty-adaptive clipping strategy. Working in concert, these components enable SCPO to implement differentiated length control across reasoning steps. Empirical results across multiple reasoning benchmarks and various backbone models demonstrate that SmartThinker significantly reduces redundant reasoning while achieving comparable or even superior performance to existing methods. 

---
# Computed Tomography Visual Question Answering with Cross-modal Feature Graphing 

**Authors**: Yuanhe Tian, Chen Su, Junwen Duan, Yan Song  

**Link**: [PDF](https://arxiv.org/pdf/2507.04333)  

**Abstract**: Visual question answering (VQA) in medical imaging aims to support clinical diagnosis by automatically interpreting complex imaging data in response to natural language queries. Existing studies typically rely on distinct visual and textual encoders to independently extract features from medical images and clinical questions, which are subsequently combined to generate answers. Specifically, in computed tomography (CT), such approaches are similar to the conventional practices in medical image analysis. However, these approaches pay less attention to the spatial continuity and inter-slice correlations in the volumetric CT data, leading to fragmented and imprecise responses. In this paper, we propose a novel large language model (LLM)-based framework enhanced by a graph representation of salient features. Different from conventional multimodal encoding strategies, our approach constructs a cross-modal graph integrating both visual and textual features, treating individual CT slices and question tokens as nodes within the graph. We further leverage an attentive graph convolutional network to dynamically fuse information within this structure. The resulting aggregated graph features then serve as a soft prompt to guide a large language model in generating accurate answers. Extensive experiments on the M3D-VQA benchmark demonstrate that our approach consistently outperforms baselines across multiple evaluation metrics, offering more robust reasoning capabilities. 

---
# LearnLens: LLM-Enabled Personalised, Curriculum-Grounded Feedback with Educators in the Loop 

**Authors**: Runcong Zhao, Artem Borov, Jiazheng Li, Yulan He  

**Link**: [PDF](https://arxiv.org/pdf/2507.04295)  

**Abstract**: Effective feedback is essential for student learning but is time-intensive for teachers. We present LearnLens, a modular, LLM-based system that generates personalised, curriculum-aligned feedback in science education. LearnLens comprises three components: (1) an error-aware assessment module that captures nuanced reasoning errors; (2) a curriculum-grounded generation module that uses a structured, topic-linked memory chain rather than traditional similarity-based retrieval, improving relevance and reducing noise; and (3) an educator-in-the-loop interface for customisation and oversight. LearnLens addresses key challenges in existing systems, offering scalable, high-quality feedback that empowers both teachers and students. 

---
# Navigating Speech Recording Collections with AI-Generated Illustrations 

**Authors**: Sirina Håland, Trond Karlsen Strøm, Petra Galuščáková  

**Link**: [PDF](https://arxiv.org/pdf/2507.04182)  

**Abstract**: Although the amount of available spoken content is steadily increasing, extracting information and knowledge from speech recordings remains challenging. Beyond enhancing traditional information retrieval methods such as speech search and keyword spotting, novel approaches for navigating and searching spoken content need to be explored and developed. In this paper, we propose a novel navigational method for speech archives that leverages recent advances in language and multimodal generative models. We demonstrate our approach with a Web application that organizes data into a structured format using interactive mind maps and image generation tools. The system is implemented using the TED-LIUM~3 dataset, which comprises over 2,000 speech transcripts and audio files of TED Talks. Initial user tests using a System Usability Scale (SUS) questionnaire indicate the application's potential to simplify the exploration of large speech collections. 

---
# Efficient Detection of Intermittent Job Failures Using Few-Shot Learning 

**Authors**: Henri Aïdasso, Francis Bordeleau, Ali Tizghadam  

**Link**: [PDF](https://arxiv.org/pdf/2507.04173)  

**Abstract**: One of the main challenges developers face in the use of continuous integration (CI) and deployment pipelines is the occurrence of intermittent job failures, which result from unexpected non-deterministic issues (e.g., flaky tests or infrastructure problems) rather than regular code-related errors such as bugs. Prior studies developed machine-learning (ML) models trained on large datasets of job logs to classify job failures as either intermittent or regular. As an alternative to costly manual labeling of large datasets, the state-of-the-art (SOTA) approach leveraged a heuristic based on non-deterministic job reruns. However, this method mislabels intermittent job failures as regular in contexts where rerunning suspicious job failures is not an explicit policy, and therefore limits the SOTA's performance in practice. In fact, our manual analysis of 2,125 job failures from 5 industrial and 1 open-source projects reveals that, on average, 32\% of intermittent job failures are mislabeled as regular. To address these limitations, this paper introduces a novel approach to intermittent job failure detection using few-shot learning (FSL). Specifically, we fine-tune a small language model using a few number of manually labeled log examples to generate rich embeddings, which are then used to train an ML classifier. Our FSL-based approach achieves 70-88\% F1-score with only 12 shots in all projects, outperforming the SOTA, which proved ineffective (34-52\% F1-score) in 4 projects. Overall, this study underlines the importance of data quality over quantity and provides a more efficient and practical framework for the detection of intermittent job failures in organizations. 

---
# An HTR-LLM Workflow for High-Accuracy Transcription and Analysis of Abbreviated Latin Court Hand 

**Authors**: Joshua D. Isom  

**Link**: [PDF](https://arxiv.org/pdf/2507.04132)  

**Abstract**: This article presents and validates an ideal, four-stage workflow for the high-accuracy transcription and analysis of challenging medieval legal documents. The process begins with a specialized Handwritten Text Recognition (HTR) model, itself created using a novel "Clean Ground Truth" curation method where a Large Language Model (LLM) refines the training data. This HTR model provides a robust baseline transcription (Stage 1). In Stage 2, this baseline is fed, along with the original document image, to an LLM for multimodal post-correction, grounding the LLM's analysis and improving accuracy. The corrected, abbreviated text is then expanded into full, scholarly Latin using a prompt-guided LLM (Stage 3). A final LLM pass performs Named-Entity Correction (NEC), regularizing proper nouns and generating plausible alternatives for ambiguous readings (Stage 4). We validate this workflow through detailed case studies, achieving Word Error Rates (WER) in the range of 2-7% against scholarly ground truths. The results demonstrate that this hybrid, multi-stage approach effectively automates the most laborious aspects of transcription while producing a high-quality, analyzable output, representing a powerful and practical solution for the current technological landscape. 

---
# Relational inductive biases on attention mechanisms 

**Authors**: Víctor Mijangos, Ximena Gutierrez-Vasques, Verónica E. Arriola, Ulises Rodríguez-Domínguez, Alexis Cervantes, José Luis Almanzara  

**Link**: [PDF](https://arxiv.org/pdf/2507.04117)  

**Abstract**: Inductive learning aims to construct general models from specific examples, guided by biases that influence hypothesis selection and determine generalization capacity. In this work, we focus on characterizing the relational inductive biases present in attention mechanisms, understood as assumptions about the underlying relationships between data elements. From the perspective of geometric deep learning, we analyze the most common attention mechanisms in terms of their equivariance properties with respect to permutation subgroups, which allows us to propose a classification based on their relational biases. Under this perspective, we show that different attention layers are characterized by the underlying relationships they assume on the input data. 

---
# MMMOS: Multi-domain Multi-axis Audio Quality Assessment 

**Authors**: Yi-Cheng Lin, Jia-Hung Chen, Hung-yi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2507.04094)  

**Abstract**: Accurate audio quality estimation is essential for developing and evaluating audio generation, retrieval, and enhancement systems. Existing non-intrusive assessment models predict a single Mean Opinion Score (MOS) for speech, merging diverse perceptual factors and failing to generalize beyond speech. We propose MMMOS, a no-reference, multi-domain audio quality assessment system that estimates four orthogonal axes: Production Quality, Production Complexity, Content Enjoyment, and Content Usefulness across speech, music, and environmental sounds. MMMOS fuses frame-level embeddings from three pretrained encoders (WavLM, MuQ, and M2D) and evaluates three aggregation strategies with four loss functions. By ensembling the top eight models, MMMOS shows a 20-30% reduction in mean squared error and a 4-5% increase in Kendall's {\tau} versus baseline, gains first place in six of eight Production Complexity metrics, and ranks among the top three on 17 of 32 challenge metrics. 

---
# A Comparative Study of Specialized LLMs as Dense Retrievers 

**Authors**: Hengran Zhang, Keping Bi, Jiafeng Guo  

**Link**: [PDF](https://arxiv.org/pdf/2507.03958)  

**Abstract**: While large language models (LLMs) are increasingly deployed as dense retrievers, the impact of their domain-specific specialization on retrieval effectiveness remains underexplored. This investigation systematically examines how task-specific adaptations in LLMs influence their retrieval capabilities, an essential step toward developing unified retrievers capable of handling text, code, images, and multimodal content. We conduct extensive experiments with eight Qwen2.5 7B LLMs, including base, instruction-tuned, code/math-specialized, long reasoning, and vision-language models across zero-shot retrieval settings and the supervised setting. For the zero-shot retrieval settings, we consider text retrieval from the BEIR benchmark and code retrieval from the CoIR benchmark. Further, to evaluate supervised performance, all LLMs are fine-tuned on the MS MARCO dataset. We find that mathematical specialization and the long reasoning capability cause consistent degradation in three settings, indicating conflicts between mathematical reasoning and semantic matching. The vision-language model and code-specialized LLMs demonstrate superior zero-shot performance compared to other LLMs, even surpassing BM25 on the code retrieval task, and maintain comparable performance to base LLMs in supervised settings. These findings suggest promising directions for the unified retrieval task leveraging cross-domain and cross-modal fusion. 

---
# Agent-Based Detection and Resolution of Incompleteness and Ambiguity in Interactions with Large Language Models 

**Authors**: Riya Naik, Ashwin Srinivasan, Swati Agarwal, Estrid He  

**Link**: [PDF](https://arxiv.org/pdf/2507.03726)  

**Abstract**: Many of us now treat LLMs as modern-day oracles asking it almost any kind of question. However, consulting an LLM does not have to be a single turn activity. But long multi-turn interactions can get tedious if it is simply to clarify contextual information that can be arrived at through reasoning. In this paper, we examine the use of agent-based architecture to bolster LLM-based Question-Answering systems with additional reasoning capabilities. We examine the automatic resolution of potential incompleteness or ambiguities in questions by transducers implemented using LLM-based agents. We focus on several benchmark datasets that are known to contain questions with these deficiencies to varying degrees. We equip different LLMs (GPT-3.5-Turbo and Llama-4-Scout) with agents that act as specialists in detecting and resolving deficiencies of incompleteness and ambiguity. The agents are implemented as zero-shot ReAct agents. Rather than producing an answer in a single step, the model now decides between 3 actions a) classify b) resolve c) answer. Action a) decides if the question is incomplete, ambiguous, or normal. Action b) determines if any deficiencies identified can be resolved. Action c) answers the resolved form of the question. We compare the use of LLMs with and without the use of agents with these components. Our results show benefits of agents with transducer 1) A shortening of the length of interactions with human 2) An improvement in the answer quality and 3) Explainable resolution of deficiencies in the question. On the negative side we find while it may result in additional LLM invocations and in some cases, increased latency. But on tested datasets, the benefits outweigh the costs except when questions already have sufficient context. Suggesting the agent-based approach could be a useful mechanism to harness the power of LLMs to develop more robust QA systems. 

---
# Interaction Techniques that Encourage Longer Prompts Can Improve Psychological Ownership when Writing with AI 

**Authors**: Nikhita Joshi, Daniel Vogel  

**Link**: [PDF](https://arxiv.org/pdf/2507.03670)  

**Abstract**: Writing longer prompts for an AI assistant to generate a short story increases psychological ownership, a user's feeling that the writing belongs to them. To encourage users to write longer prompts, we evaluated two interaction techniques that modify the prompt entry interface of chat-based generative AI assistants: pressing and holding the prompt submission button, and continuously moving a slider up and down when submitting a short prompt. A within-subjects experiment investigated the effects of such techniques on prompt length and psychological ownership, and results showed that these techniques increased prompt length and led to higher psychological ownership than baseline techniques. A second experiment further augmented these techniques by showing AI-generated suggestions for how the prompts could be expanded. This further increased prompt length, but did not lead to improvements in psychological ownership. Our results show that simple interface modifications like these can elicit more writing from users and improve psychological ownership. 

---
# Re-Emergent Misalignment: How Narrow Fine-Tuning Erodes Safety Alignment in LLMs 

**Authors**: Jeremiah Giordani  

**Link**: [PDF](https://arxiv.org/pdf/2507.03662)  

**Abstract**: Recent work has shown that fine-tuning large language models (LLMs) on code with security vulnerabilities can result in misaligned and unsafe behaviors across broad domains. These results prompted concerns about the emergence of harmful behaviors from narrow domain fine-tuning. In this paper, we contextualize these findings by analyzing how such narrow adaptation impacts the internal mechanisms and behavioral manifestations of LLMs. Through a series of experiments covering output probability distributions, loss and gradient vector geometry, layer-wise activation dynamics, and activation space dimensions, we find that behaviors attributed to "emergent misalignment" may be better interpreted as an erosion of prior alignment. We show that fine tuning on insecure code induces internal changes that oppose alignment. Further, we identify a shared latent dimension in the model's activation space that governs alignment behavior. We show that this space is activated by insecure code and by misaligned responses more generally, revealing how narrow fine-tuning can degrade general safety behavior by interfering with shared internal mechanisms. Our findings offer a mechanistic interpretation for previously observed misalignment phenomena, and highlights the fragility of alignment in LLMs. The results underscore the need for more robust fine-tuning strategies that preserve intended behavior across domains. 

---
# Is It Time To Treat Prompts As Code? A Multi-Use Case Study For Prompt Optimization Using DSPy 

**Authors**: Francisca Lemos, Victor Alves, Filipa Ferraz  

**Link**: [PDF](https://arxiv.org/pdf/2507.03620)  

**Abstract**: Although prompt engineering is central to unlocking the full potential of Large Language Models (LLMs), crafting effective prompts remains a time-consuming trial-and-error process that relies on human intuition. This study investigates Declarative Self-improving Python (DSPy), an optimization framework that programmatically creates and refines prompts, applied to five use cases: guardrail enforcement, hallucination detection in code, code generation, routing agents, and prompt evaluation. Each use case explores how prompt optimization via DSPy influences performance. While some cases demonstrated modest improvements - such as minor gains in the guardrails use case and selective enhancements in hallucination detection - others showed notable benefits. The prompt evaluation criterion task demonstrated a substantial performance increase, rising accuracy from 46.2% to 64.0%. In the router agent case, the possibility of improving a poorly performing prompt and of a smaller model matching a stronger one through optimized prompting was explored. Although prompt refinement increased accuracy from 85.0% to 90.0%, using the optimized prompt with a cheaper model did not improve performance. Overall, this study's findings suggest that DSPy's systematic prompt optimization can enhance LLM performance, particularly when instruction tuning and example selection are optimized together. However, the impact varies by task, highlighting the importance of evaluating specific use cases in prompt optimization research. 

---
# RECA-PD: A Robust Explainable Cross-Attention Method for Speech-based Parkinson's Disease Classification 

**Authors**: Terry Yi Zhong, Cristian Tejedor-Garcia, Martha Larson, Bastiaan R. Bloem  

**Link**: [PDF](https://arxiv.org/pdf/2507.03594)  

**Abstract**: Parkinson's Disease (PD) affects over 10 million people globally, with speech impairments often preceding motor symptoms by years, making speech a valuable modality for early, non-invasive detection. While recent deep-learning models achieve high accuracy, they typically lack the explainability required for clinical use. To address this, we propose RECA-PD, a novel, robust, and explainable cross-attention architecture that combines interpretable speech features with self-supervised representations. RECA-PD matches state-of-the-art performance in Speech-based PD detection while providing explanations that are more consistent and more clinically meaningful. Additionally, we demonstrate that performance degradation in certain speech tasks (e.g., monologue) can be mitigated by segmenting long recordings. Our findings indicate that performance and explainability are not necessarily mutually exclusive. Future work will enhance the usability of explanations for non-experts and explore severity estimation to increase the real-world clinical relevance. 

---
# Causal-SAM-LLM: Large Language Models as Causal Reasoners for Robust Medical Segmentation 

**Authors**: Tao Tang, Shijie Xu, Yiting Wu, Zhixiang Lu  

**Link**: [PDF](https://arxiv.org/pdf/2507.03585)  

**Abstract**: The clinical utility of deep learning models for medical image segmentation is severely constrained by their inability to generalize to unseen domains. This failure is often rooted in the models learning spurious correlations between anatomical content and domain-specific imaging styles. To overcome this fundamental challenge, we introduce Causal-SAM-LLM, a novel framework that elevates Large Language Models (LLMs) to the role of causal reasoners. Our framework, built upon a frozen Segment Anything Model (SAM) encoder, incorporates two synergistic innovations. First, Linguistic Adversarial Disentanglement (LAD) employs a Vision-Language Model to generate rich, textual descriptions of confounding image styles. By training the segmentation model's features to be contrastively dissimilar to these style descriptions, it learns a representation robustly purged of non-causal information. Second, Test-Time Causal Intervention (TCI) provides an interactive mechanism where an LLM interprets a clinician's natural language command to modulate the segmentation decoder's features in real-time, enabling targeted error correction. We conduct an extensive empirical evaluation on a composite benchmark from four public datasets (BTCV, CHAOS, AMOS, BraTS), assessing generalization under cross-scanner, cross-modality, and cross-anatomy settings. Causal-SAM-LLM establishes a new state of the art in out-of-distribution (OOD) robustness, improving the average Dice score by up to 6.2 points and reducing the Hausdorff Distance by 15.8 mm over the strongest baseline, all while using less than 9% of the full model's trainable parameters. Our work charts a new course for building robust, efficient, and interactively controllable medical AI systems. 

---
# Disambiguation-Centric Finetuning Makes Enterprise Tool-Calling LLMs More Realistic and Less Risky 

**Authors**: Ashutosh Hathidara, Julien Yu, Sebastian Schreiber  

**Link**: [PDF](https://arxiv.org/pdf/2507.03336)  

**Abstract**: Large language models (LLMs) are increasingly tasked with invoking enterprise APIs, yet they routinely falter when near-duplicate tools vie for the same user intent or when required arguments are left underspecified. We introduce DiaFORGE (Dialogue Framework for Organic Response Generation & Evaluation), a disambiguation-centric, three-stage pipeline that (i) synthesizes persona-driven, multi-turn dialogues in which the assistant must distinguish among highly similar tools, (ii) performs supervised fine-tuning of open-source models with reasoning traces across 3B - 70B parameters, and (iii) evaluates real-world readiness via a dynamic suite that redeploys each model in a live agentic loop and reports end-to-end goal completion alongside conventional static metrics. On our dynamic benchmark DiaBENCH, models trained with DiaFORGE raise tool-invocation success by 27 pp over GPT-4o and by 49 pp over Claude-3.5-Sonnet, both under optimized prompting. To spur further research, we release an open corpus of 5000 production-grade enterprise API specifications paired with rigorously validated, disambiguation-focused dialogues, offering a practical blueprint for building reliable, enterprise-ready tool-calling agents. 

---
# LTLCrit: A Temporal Logic-based LLM Critic for Safe and Efficient Embodied Agents 

**Authors**: Anand Gokhale, Vaibhav Srivastava, Francesco Bullo  

**Link**: [PDF](https://arxiv.org/pdf/2507.03293)  

**Abstract**: Large language models (LLMs) have demonstrated promise in reasoning tasks and general decision-making in static environments. In long-term planning tasks, however, errors tend to accumulate, often leading to unsafe or inefficient behavior, limiting their use in general-purpose settings. We propose a modular actor-critic architecture in which an LLM actor is guided by LTLCrit, a trajectory-level LLM critic that communicates via linear temporal logic (LTL). Our setup combines the reasoning strengths of language models with the guarantees of formal logic. The actor selects high-level actions from natural language observations, while the critic analyzes full trajectories and proposes new LTL constraints that shield the actor from future unsafe or inefficient behavior. The architecture supports both fixed, hand-specified safety constraints and adaptive, learned soft constraints that promote long-term efficiency. Our architecture is model-agnostic: any LLM-based planner can serve as the actor, and LTLCrit serves as a logic-generating wrapper. We formalize planning as graph traversal under symbolic constraints, allowing LTLCrit to analyze failed or suboptimal trajectories and generate new temporal logic rules that improve future behavior. We evaluate our system on the Minecraft diamond-mining benchmark, achieving 100% completion rates and improving efficiency compared to baseline LLM planners. Our results suggest that enabling LLMs to supervise each other through logic is a powerful and flexible paradigm for safe, generalizable decision making. 

---
# GDGB: A Benchmark for Generative Dynamic Text-Attributed Graph Learning 

**Authors**: Jie Peng, Jiarui Ji, Runlin Lei, Zhewei Wei, Yongchao Liu, Chuntao Hong  

**Link**: [PDF](https://arxiv.org/pdf/2507.03267)  

**Abstract**: Dynamic Text-Attributed Graphs (DyTAGs), which intricately integrate structural, temporal, and textual attributes, are crucial for modeling complex real-world systems. However, most of the existing DyTAG datasets exhibit poor textual quality, which severely limits their utility for DyTAG generation tasks requiring semantically rich inputs. Additionally, prior work mainly focuses on discriminative tasks on DyTAGs, resulting in a lack of standardized task formulations and evaluation protocols tailored for DyTAG generation. To address these critical issues, we propose Generative DyTAG Benchmark (GDGB), which comprises eight meticulously curated DyTAG datasets with high-quality textual features for both nodes and edges, overcoming limitations of prior datasets. Building on GDGB, we define two novel DyTAG generation tasks: Transductive Dynamic Graph Generation (TDGG) and Inductive Dynamic Graph Generation (IDGG). TDGG transductively generates a target DyTAG based on the given source and destination node sets, while the more challenging IDGG introduces new node generation to inductively model the dynamic expansion of real-world graph data. To enable holistic evaluation, we design multifaceted metrics that assess the structural, temporal, and textual quality of the generated DyTAGs. We further propose GAG-General, an LLM-based multi-agent generative framework tailored for reproducible and robust benchmarking of DyTAG generation. Experimental results demonstrate that GDGB enables rigorous evaluation of TDGG and IDGG, with key insights revealing the critical interplay of structural and textual features in DyTAG generation. These findings establish GDGB as a foundational resource for advancing generative DyTAG research and unlocking further practical applications in DyTAG generation. GDGB datasets, source codes, and leaderboards are available at \href{this https URL}{here}. 

---
# MateInfoUB: A Real-World Benchmark for Testing LLMs in Competitive, Multilingual, and Multimodal Educational Tasks 

**Authors**: Dumitran Adrian Marius, Theodor-Pierre Moroianu, Buca Mihnea-Vicentiu  

**Link**: [PDF](https://arxiv.org/pdf/2507.03162)  

**Abstract**: The rapid advancement of Large Language Models (LLMs) has transformed various domains, particularly computer science (CS) education. These models exhibit remarkable capabilities in code-related tasks and problem-solving, raising questions about their potential and limitations in advanced CS contexts. This study presents a novel bilingual (English-Romanian) multimodal (text and image) dataset of multiple-choice questions derived from a high-level computer science competition. A particularity of our dataset is that the problems are conceived such that some of them are easier solved using reasoning on paper, while for others writing code is more efficient. We systematically evaluate State of The Art LLMs on this dataset, analyzing their performance on theoretical programming tasks. Our findings reveal the strengths and limitations of current LLMs, including the influence of language choice (English vs. Romanian), providing insights into their applicability in CS education and competition settings. We also address critical ethical considerations surrounding educational integrity and the fairness of assessments in the context of LLM usage. These discussions aim to inform future educational practices and policies. To support further research, our dataset will be made publicly available in both English and Romanian. Additionally, we release an educational application tailored for Romanian students, enabling them to self-assess using the dataset in an interactive and practice-oriented environment. 

---
# On the Relationship between Accent Strength and Articulatory Features 

**Authors**: Kevin Huang, Sean Foley, Jihwan Lee, Yoonjeong Lee, Dani Byrd, Shrikanth Narayanan  

**Link**: [PDF](https://arxiv.org/pdf/2507.03149)  

**Abstract**: This paper explores the relationship between accent strength and articulatory features inferred from acoustic speech. To quantify accent strength, we compare phonetic transcriptions with transcriptions based on dictionary-based references, computing phoneme-level difference as a measure of accent strength. The proposed framework leverages recent self-supervised learning articulatory inversion techniques to estimate articulatory features. Analyzing a corpus of read speech from American and British English speakers, this study examines correlations between derived articulatory parameters and accent strength proxies, associating systematic articulatory differences with indexed accent strength. Results indicate that tongue positioning patterns distinguish the two dialects, with notable differences inter-dialects in rhotic and low back vowels. These findings contribute to automated accent analysis and articulatory modeling for speech processing applications. 

---
# DeepGesture: A conversational gesture synthesis system based on emotions and semantics 

**Authors**: Thanh Hoang-Minh  

**Link**: [PDF](https://arxiv.org/pdf/2507.03147)  

**Abstract**: Along with the explosion of large language models, improvements in speech synthesis, advancements in hardware, and the evolution of computer graphics, the current bottleneck in creating digital humans lies in generating character movements that correspond naturally to text or speech inputs.
In this work, we present DeepGesture, a diffusion-based gesture synthesis framework for generating expressive co-speech gestures conditioned on multimodal signals-text, speech, emotion, and seed motion. Built upon the DiffuseStyleGesture model, DeepGesture introduces novel architectural enhancements that improve semantic alignment and emotional expressiveness in generated gestures. Specifically, we integrate fast text transcriptions as semantic conditioning and implement emotion-guided classifier-free diffusion to support controllable gesture generation across affective states. A lightweight Transformer backbone combines full self-attention and cross-local attention for effective feature fusion of heterogeneous modalities. To visualize results, we implement a full rendering pipeline in Unity based on BVH output from the model. Evaluation on the ZeroEGGS dataset shows that DeepGesture produces gestures with improved human-likeness and contextual appropriateness, outperforming baselines on Mean Opinion Score and Frechet Gesture Distance metrics. Our system supports interpolation between emotional states and demonstrates generalization to out-of-distribution speech, including synthetic voices-marking a step forward toward fully multimodal, emotionally aware digital humans. 

---
# Towards a Psychoanalytic Perspective on VLM Behaviour: A First-step Interpretation with Intriguing Observations 

**Authors**: Xiangrui Liu, Man Luo, Agneet Chatterjee, Hua Wei, Yezhou Yang  

**Link**: [PDF](https://arxiv.org/pdf/2507.03123)  

**Abstract**: Hallucination is a long-standing problem that has been actively investigated in Vision-Language Models (VLMs). Existing research commonly attributes hallucinations to technical limitations or sycophancy bias, where the latter means the models tend to generate incorrect answers to align with user expectations. However, these explanations primarily focus on technical or externally driven factors, may have neglected the possibility that hallucination behaviours might mirror cognitive biases observed in human psychology. In this work, we introduce a psychological taxonomy, categorizing VLMs' hallucination behaviours, including sycophancy, logical inconsistency, and a newly identified VLMs behaviour: authority bias. To systematically analyze these behaviours, we design AIpsych, a scalable benchmark that reveals psychological tendencies in model response patterns. Leveraging this benchmark, we investigate how variations in model architecture and parameter size influence model behaviour when responding to strategically manipulated questions. Our experiments reveal that as model size increases, VLMs exhibit stronger sycophantic tendencies but reduced authority bias, suggesting increasing competence but a potential erosion of response integrity. A human subject study further validates our hypotheses and highlights key behavioural differences between VLMs and human respondents. This work suggests a new perspective for understanding hallucination in VLMs and highlights the importance of integrating psychological principles into model this http URL benchmark is available at this https URL. 

---
# Federated Learning for ICD Classification with Lightweight Models and Pretrained Embeddings 

**Authors**: Binbin Xu, Gérard Dray  

**Link**: [PDF](https://arxiv.org/pdf/2507.03122)  

**Abstract**: This study investigates the feasibility and performance of federated learning (FL) for multi-label ICD code classification using clinical notes from the MIMIC-IV dataset. Unlike previous approaches that rely on centralized training or fine-tuned large language models, we propose a lightweight and scalable pipeline combining frozen text embeddings with simple multilayer perceptron (MLP) classifiers. This design offers a privacy-preserving and deployment-efficient alternative for clinical NLP applications, particularly suited to distributed healthcare settings. Extensive experiments across both centralized and federated configurations were conducted, testing six publicly available embedding models from Massive Text Embedding Benchmark leaderboard and three MLP classifier architectures under two medical coding (ICD-9 and ICD-10). Additionally, ablation studies over ten random stratified splits assess performance stability. Results show that embedding quality substantially outweighs classifier complexity in determining predictive performance, and that federated learning can closely match centralized results in idealized conditions. While the models are orders of magnitude smaller than state-of-the-art architectures and achieved competitive micro and macro F1 scores, limitations remain including the lack of end-to-end training and the simplified FL assumptions. Nevertheless, this work demonstrates a viable way toward scalable, privacy-conscious medical coding systems and offers a step toward for future research into federated, domain-adaptive clinical AI. 

---
# Improving LLM Reasoning for Vulnerability Detection via Group Relative Policy Optimization 

**Authors**: Marco Simoni, Aleksandar Fontana, Giulio Rossolini, Andrea Saracino  

**Link**: [PDF](https://arxiv.org/pdf/2507.03051)  

**Abstract**: Improving and understanding the training dynamics and reasoning of Large Language Models (LLMs) has become essential for their deployment in AI-based security tools, such as software vulnerability detection. In this work, we present an extensive study aimed at advancing recent RL-based finetuning techniques for LLMs in the context of vulnerability detection.
We start by highlighting key limitations of commonly adopted LLMs, such as their tendency to over-predict certain types of vulnerabilities while failing to detect others. To address this challenge, we explore the use of Group Relative Policy Optimization (GRPO), a recent policy-gradient method, for guiding LLM behavior through structured, rule-based rewards. We enable its application to the vulnerability detection task by redefining its advantage functions and reward signals using annotations from widely used datasets in the field, including BigVul, DiverseVul, and CleanVul.
The proposed methodology enables an extensive set of experiments, addressing multiple research questions regarding the impact of GRPO on generalization, reasoning capabilities, and performance improvements over standard supervised finetuning (SFT). Our findings offer valuable insights into the potential of RL-based training to enhance both the performance and reasoning abilities of LLMs in the context of software vulnerability detection. 

---
# Intrinsic Fingerprint of LLMs: Continue Training is NOT All You Need to Steal A Model! 

**Authors**: Do-hyeon Yoon, Minsoo Chun, Thomas Allen, Hans Müller, Min Wang, Rajesh Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2507.03014)  

**Abstract**: Large language models (LLMs) face significant copyright and intellectual property challenges as the cost of training increases and model reuse becomes prevalent. While watermarking techniques have been proposed to protect model ownership, they may not be robust to continue training and development, posing serious threats to model attribution and copyright protection. This work introduces a simple yet effective approach for robust LLM fingerprinting based on intrinsic model characteristics. We discover that the standard deviation distributions of attention parameter matrices across different layers exhibit distinctive patterns that remain stable even after extensive continued training. These parameter distribution signatures serve as robust fingerprints that can reliably identify model lineage and detect potential copyright infringement. Our experimental validation across multiple model families demonstrates the effectiveness of our method for model authentication. Notably, our investigation uncovers evidence that a recently Pangu Pro MoE model released by Huawei is derived from Qwen-2.5 14B model through upcycling techniques rather than training from scratch, highlighting potential cases of model plagiarism, copyright violation, and information fabrication. These findings underscore the critical importance of developing robust fingerprinting methods for protecting intellectual property in large-scale model development and emphasize that deliberate continued training alone is insufficient to completely obscure model origins. 

---
# A Weakly Supervised Transformer to Support Rare Disease Diagnosis from Electronic Health Records: Methods and Applications in Rare Pulmonary Disease 

**Authors**: Kimberly F. Greco, Zongxin Yang, Mengyan Li, Han Tong, Sara Morini Sweet, Alon Geva, Kenneth D. Mandl, Benjamin A. Raby, Tianxi Cai  

**Link**: [PDF](https://arxiv.org/pdf/2507.02998)  

**Abstract**: Rare diseases affect an estimated 300-400 million people worldwide, yet individual conditions often remain poorly characterized and difficult to diagnose due to their low prevalence and limited clinician familiarity. While computational phenotyping algorithms show promise for automating rare disease detection, their development is hindered by the scarcity of labeled data and biases in existing label sources. Gold-standard labels from registries and expert chart reviews are highly accurate but constrained by selection bias and the cost of manual review. In contrast, labels derived from electronic health records (EHRs) cover a broader range of patients but can introduce substantial noise. To address these challenges, we propose a weakly supervised, transformer-based framework that combines a small set of gold-standard labels with a large volume of iteratively updated silver-standard labels derived from EHR data. This hybrid approach enables the training of a highly accurate and generalizable phenotyping model that scales rare disease detection beyond the scope of individual clinical expertise. Our method is initialized by learning embeddings of medical concepts based on their semantic meaning or co-occurrence patterns in EHRs, which are then refined and aggregated into patient-level representations via a multi-layer transformer architecture. Using two rare pulmonary diseases as a case study, we validate our model on EHR data from Boston Children's Hospital. Our framework demonstrates notable improvements in phenotype classification, identification of clinically meaningful subphenotypes through patient clustering, and prediction of disease progression compared to baseline methods. These results highlight the potential of our approach to enable scalable identification and stratification of rare disease patients for clinical care and research applications. 

---
# Gated Recursive Fusion: A Stateful Approach to Scalable Multimodal Transformers 

**Authors**: Yusuf Shihata  

**Link**: [PDF](https://arxiv.org/pdf/2507.02985)  

**Abstract**: Multimodal learning faces a fundamental tension between deep, fine-grained fusion and computational scalability. While cross-attention models achieve strong performance through exhaustive pairwise fusion, their quadratic complexity is prohibitive for settings with many modalities. We address this challenge with Gated Recurrent Fusion (GRF), a novel architecture that captures the power of cross-modal attention within a linearly scalable, recurrent pipeline. Our method processes modalities sequentially, updating an evolving multimodal context vector at each step. The core of our approach is a fusion block built on Transformer Decoder layers that performs symmetric cross-attention, mutually enriching the shared context and the incoming modality. This enriched information is then integrated via a Gated Fusion Unit (GFU) a GRU-inspired mechanism that dynamically arbitrates information flow, enabling the model to selectively retain or discard features. This stateful, recurrent design scales linearly with the number of modalities, O(n), making it ideal for high-modality environments. Experiments on the CMU-MOSI benchmark demonstrate that GRF achieves competitive performance compared to more complex baselines. Visualizations of the embedding space further illustrate that GRF creates structured, class-separable representations through its progressive fusion mechanism. Our work presents a robust and efficient paradigm for powerful, scalable multimodal representation learning. 

---
# InvisibleInk: High-Utility and Low-Cost Text Generation with Differential Privacy 

**Authors**: Vishnu Vinod, Krishna Pillutla, Abhradeep Guha Thakurta  

**Link**: [PDF](https://arxiv.org/pdf/2507.02974)  

**Abstract**: As major progress in LLM-based long-form text generation enables paradigms such as retrieval-augmented generation (RAG) and inference-time scaling, safely incorporating private information into the generation remains a critical open question. We present InvisibleInk, a highly scalable long-form text generation framework satisfying rigorous differential privacy guarantees with respect to the sensitive references. It interprets sampling from the LLM's next-token-distribution as the exponential mechanism over the LLM logits with two innovations. First, we reduce the privacy cost by isolating and clipping only the sensitive information in the model logits (relative to the public logits). Second, we improve text quality by sampling from a small superset of the top-$k$ private tokens. Empirical evaluations demonstrate a consistent $8\times$ reduction in computation cost over state-of-the-art baselines to generate long-form private text of the same utility across privacy levels. In summary, InvisibleInk is able to generate private long-form text at less than $10\times$ the computation cost of non-private generation. 

---
# GameTileNet: A Semantic Dataset for Low-Resolution Game Art in Procedural Content Generation 

**Authors**: Yi-Chun Chen, Arnav Jhala  

**Link**: [PDF](https://arxiv.org/pdf/2507.02941)  

**Abstract**: GameTileNet is a dataset designed to provide semantic labels for low-resolution digital game art, advancing procedural content generation (PCG) and related AI research as a vision-language alignment task. Large Language Models (LLMs) and image-generative AI models have enabled indie developers to create visual assets, such as sprites, for game interactions. However, generating visuals that align with game narratives remains challenging due to inconsistent AI outputs, requiring manual adjustments by human artists. The diversity of visual representations in automatically generated game content is also limited because of the imbalance in distributions across styles for training data. GameTileNet addresses this by collecting artist-created game tiles from this http URL under Creative Commons licenses and providing semantic annotations to support narrative-driven content generation. The dataset introduces a pipeline for object detection in low-resolution tile-based game art (e.g., 32x32 pixels) and annotates semantics, connectivity, and object classifications. GameTileNet is a valuable resource for improving PCG methods, supporting narrative-rich game content, and establishing a baseline for object detection in low-resolution, non-photorealistic images.
TL;DR: GameTileNet is a semantic dataset of low-resolution game tiles designed to support narrative-driven procedural content generation through visual-language alignment. 

---
# Large Language Model Agent for Modular Task Execution in Drug Discovery 

**Authors**: Janghoon Ock, Radheesh Sharma Meda, Srivathsan Badrinarayanan, Neha S. Aluru, Achuth Chandrasekhar, Amir Barati Farimani  

**Link**: [PDF](https://arxiv.org/pdf/2507.02925)  

**Abstract**: We present a modular framework powered by large language models (LLMs) that automates and streamlines key tasks across the early-stage computational drug discovery pipeline. By combining LLM reasoning with domain-specific tools, the framework performs biomedical data retrieval, domain-specific question answering, molecular generation, property prediction, property-aware molecular refinement, and 3D protein-ligand structure generation. In a case study targeting BCL-2 in lymphocytic leukemia, the agent autonomously retrieved relevant biomolecular information-including FASTA sequences, SMILES representations, and literature-and answered mechanistic questions with improved contextual accuracy over standard LLMs. It then generated chemically diverse seed molecules and predicted 67 ADMET-related properties, which guided iterative molecular refinement. Across two refinement rounds, the number of molecules with QED > 0.6 increased from 34 to 55, and those passing at least four out of five empirical drug-likeness rules rose from 29 to 52, within a pool of 194 molecules. The framework also employed Boltz-2 to generate 3D protein-ligand complexes and provide rapid binding affinity estimates for candidate compounds. These results demonstrate that the approach effectively supports molecular screening, prioritization, and structure evaluation. Its modular design enables flexible integration of evolving tools and models, providing a scalable foundation for AI-assisted therapeutic discovery. 

---
