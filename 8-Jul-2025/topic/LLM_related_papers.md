# An Evaluation of Large Language Models on Text Summarization Tasks Using Prompt Engineering Techniques 

**Authors**: Walid Mohamed Aly, Taysir Hassan A. Soliman, Amr Mohamed AbdelAziz  

**Link**: [PDF](https://arxiv.org/pdf/2507.05123)  

**Abstract**: Large Language Models (LLMs) continue to advance natural language processing with their ability to generate human-like text across a range of tasks. Despite the remarkable success of LLMs in Natural Language Processing (NLP), their performance in text summarization across various domains and datasets has not been comprehensively evaluated. At the same time, the ability to summarize text effectively without relying on extensive training data has become a crucial bottleneck. To address these issues, we present a systematic evaluation of six LLMs across four datasets: CNN/Daily Mail and NewsRoom (news), SAMSum (dialog), and ArXiv (scientific). By leveraging prompt engineering techniques including zero-shot and in-context learning, our study evaluates the performance using the ROUGE and BERTScore metrics. In addition, a detailed analysis of inference times is conducted to better understand the trade-off between summarization quality and computational efficiency. For Long documents, introduce a sentence-based chunking strategy that enables LLMs with shorter context windows to summarize extended inputs in multiple stages. The findings reveal that while LLMs perform competitively on news and dialog tasks, their performance on long scientific documents improves significantly when aided by chunking strategies. In addition, notable performance variations were observed based on model parameters, dataset properties, and prompt design. These results offer actionable insights into how different LLMs behave across task types, contributing to ongoing research in efficient, instruction-based NLP systems. 

---
# Response Attack: Exploiting Contextual Priming to Jailbreak Large Language Models 

**Authors**: Ziqi Miao, Lijun Li, Yuan Xiong, Zhenhua Liu, Pengyu Zhu, Jing Shao  

**Link**: [PDF](https://arxiv.org/pdf/2507.05248)  

**Abstract**: Contextual priming, where earlier stimuli covertly bias later judgments, offers an unexplored attack surface for large language models (LLMs). We uncover a contextual priming vulnerability in which the previous response in the dialogue can steer its subsequent behavior toward policy-violating content. Building on this insight, we propose Response Attack, which uses an auxiliary LLM to generate a mildly harmful response to a paraphrased version of the original malicious query. They are then formatted into the dialogue and followed by a succinct trigger prompt, thereby priming the target model to generate harmful content. Across eight open-source and proprietary LLMs, RA consistently outperforms seven state-of-the-art jailbreak techniques, achieving higher attack success rates. To mitigate this threat, we construct and release a context-aware safety fine-tuning dataset, which significantly reduces the attack success rate while preserving model capabilities. The code and data are available at this https URL. 

---
# Co-DETECT: Collaborative Discovery of Edge Cases in Text Classification 

**Authors**: Chenfei Xiong, Jingwei Ni, Yu Fan, Vilém Zouhar, Donya Rooein, Lorena Calvo-Bartolomé, Alexander Hoyle, Zhijing Jin, Mrinmaya Sachan, Markus Leippold, Dirk Hovy, Mennatallah El-Assady, Elliott Ash  

**Link**: [PDF](https://arxiv.org/pdf/2507.05010)  

**Abstract**: We introduce Co-DETECT (Collaborative Discovery of Edge cases in TExt ClassificaTion), a novel mixed-initiative annotation framework that integrates human expertise with automatic annotation guided by large language models (LLMs). Co-DETECT starts with an initial, sketch-level codebook and dataset provided by a domain expert, then leverages the LLM to annotate the data and identify edge cases that are not well described by the initial codebook. Specifically, Co-DETECT flags challenging examples, induces high-level, generalizable descriptions of edge cases, and assists user in incorporating edge case handling rules to improve the codebook. This iterative process enables more effective handling of nuanced phenomena through compact, generalizable annotation rules. Extensive user study, qualitative and quantitative analyses prove the effectiveness of Co-DETECT. 

---
# Evaluating Memory in LLM Agents via Incremental Multi-Turn Interactions 

**Authors**: Yuanzhe Hu, Yu Wang, Julian McAuley  

**Link**: [PDF](https://arxiv.org/pdf/2507.05257)  

**Abstract**: Recent benchmarks for Large Language Model (LLM) agents primarily focus on evaluating reasoning, planning, and execution capabilities, while another critical component-memory, encompassing how agents memorize, update, and retrieve long-term information-is under-evaluated due to the lack of benchmarks. We term agents with memory mechanisms as memory agents. In this paper, we identify four core competencies essential for memory agents: accurate retrieval, test-time learning, long-range understanding, and conflict resolution. Existing datasets either rely on limited context lengths or are tailored for static, long-context settings like book-based QA, which do not reflect the interactive, multi-turn nature of memory agents that incrementally accumulate information. Furthermore, no existing benchmarks cover all four competencies. Therefore, we introduce MemoryAgentBench, a new benchmark specifically designed for memory agents. Our benchmark combines reformulated existing datasets with newly constructed ones, covering the above four memory competencies, providing a systematic and challenging testbed for assessing memory quality. We evaluate a diverse set of memory agents, ranging from simple context-based and retrieval-augmented generation (RAG) systems to advanced agents with external memory modules and tool integration. Empirical results reveal that current methods fall short of mastering all four competencies, underscoring the need for further research into comprehensive memory mechanisms for LLM agents. 

---
# Emergent Semantics Beyond Token Embeddings: Transformer LMs with Frozen Visual Unicode Representations 

**Authors**: A. Bochkov  

**Link**: [PDF](https://arxiv.org/pdf/2507.04886)  

**Abstract**: Understanding the locus of semantic representation in large language models (LLMs) is crucial for interpretability and architectural innovation. The dominant paradigm posits that trainable input embeddings serve as foundational "meaning vectors." This paper challenges that view. We construct Transformer models where the embedding layer is entirely frozen, with vectors derived not from data, but from the visual structure of Unicode glyphs. These non-semantic, precomputed visual embeddings are fixed throughout training. Our method is compatible with any tokenizer, including a novel Unicode-centric tokenizer we introduce to ensure universal text coverage. Despite the absence of trainable, semantically initialized embeddings, our models converge, generate coherent text, and, critically, outperform architecturally identical models with trainable embeddings on the MMLU reasoning benchmark. We attribute this to "representational interference" in conventional models, where the embedding layer is burdened with learning both structural and semantic features. Our results indicate that high-level semantics are not inherent to input embeddings but are an emergent property of the Transformer's compositional architecture and data scale. This reframes the role of embeddings from meaning containers to structural primitives. We release all code and models to foster further research. 

---
# $\textit{Grahak-Nyay:}$ Consumer Grievance Redressal through Large Language Models 

**Authors**: Shrey Ganatra, Swapnil Bhattacharyya, Harshvivek Kashid, Spandan Anaokar, Shruti Nair, Reshma Sekhar, Siddharth Manohar, Rahul Hemrajani, Pushpak Bhattacharyya  

**Link**: [PDF](https://arxiv.org/pdf/2507.04854)  

**Abstract**: Access to consumer grievance redressal in India is often hindered by procedural complexity, legal jargon, and jurisdictional challenges. To address this, we present $\textbf{Grahak-Nyay}$ (Justice-to-Consumers), a chatbot that streamlines the process using open-source Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG). Grahak-Nyay simplifies legal complexities through a concise and up-to-date knowledge base. We introduce three novel datasets: $\textit{GeneralQA}$ (general consumer law), $\textit{SectoralQA}$ (sector-specific knowledge) and $\textit{SyntheticQA}$ (for RAG evaluation), along with $\textit{NyayChat}$, a dataset of 300 annotated chatbot conversations. We also introduce $\textit{Judgments}$ data sourced from Indian Consumer Courts to aid the chatbot in decision making and to enhance user trust. We also propose $\textbf{HAB}$ metrics ($\textbf{Helpfulness, Accuracy, Brevity}$) to evaluate chatbot performance. Legal domain experts validated Grahak-Nyay's effectiveness. Code and datasets will be released. 

---
# OpenS2S: Advancing Open-Source End-to-End Empathetic Large Speech Language Model 

**Authors**: Chen Wang, Tianyu Peng, Wen Yang, Yinan Bai, Guangfu Wang, Jun Lin, Lanpeng Jia, Lingxiang Wu, Jinqiao Wang, Chengqing Zong, Jiajun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.05177)  

**Abstract**: Empathetic interaction is a cornerstone of human-machine communication, due to the need for understanding speech enriched with paralinguistic cues and generating emotional and expressive responses. However, the most powerful empathetic LSLMs are increasingly closed off, leaving the crucial details about the architecture, data and development opaque to researchers. Given the critical need for transparent research into the LSLMs and empathetic behavior, we present OpenS2S, a fully open-source, transparent and end-to-end LSLM designed to enable empathetic speech interactions. Based on our empathetic speech-to-text model BLSP-Emo, OpenS2S further employs a streaming interleaved decoding architecture to achieve low-latency speech generation. To facilitate end-to-end training, OpenS2S incorporates an automated data construction pipeline that synthesizes diverse, high-quality empathetic speech dialogues at low cost. By leveraging large language models to generate empathetic content and controllable text-to-speech systems to introduce speaker and emotional variation, we construct a scalable training corpus with rich paralinguistic diversity and minimal human supervision. We release the fully open-source OpenS2S model, including the dataset, model weights, pre-training and fine-tuning codes, to empower the broader research community and accelerate innovation in empathetic speech systems. The project webpage can be accessed at this https URL 

---
# Dialogue-Based Multi-Dimensional Relationship Extraction from Novels 

**Authors**: Yuchen Yan, Hanjie Zhao, Senbin Zhu, Hongde Liu, Zhihong Zhang, Yuxiang Jia  

**Link**: [PDF](https://arxiv.org/pdf/2507.04852)  

**Abstract**: Relation extraction is a crucial task in natural language processing, with broad applications in knowledge graph construction and literary analysis. However, the complex context and implicit expressions in novel texts pose significant challenges for automatic character relationship extraction. This study focuses on relation extraction in the novel domain and proposes a method based on Large Language Models (LLMs). By incorporating relationship dimension separation, dialogue data construction, and contextual learning strategies, the proposed method enhances extraction performance. Leveraging dialogue structure information, it improves the model's ability to understand implicit relationships and demonstrates strong adaptability in complex contexts. Additionally, we construct a high-quality Chinese novel relation extraction dataset to address the lack of labeled resources and support future research. Experimental results show that our method outperforms traditional baselines across multiple evaluation metrics and successfully facilitates the automated construction of character relationship networks in novels. 

---
# LLMs as Architects and Critics for Multi-Source Opinion Summarization 

**Authors**: Anuj Attri, Arnav Attri, Pushpak Bhattacharyya, Suman Banerjee, Amey Patil, Muthusamy Chelliah, Nikesh Garera  

**Link**: [PDF](https://arxiv.org/pdf/2507.04751)  

**Abstract**: Multi-source Opinion Summarization (M-OS) extends beyond traditional opinion summarization by incorporating additional sources of product metadata such as descriptions, key features, specifications, and ratings, alongside reviews. This integration results in comprehensive summaries that capture both subjective opinions and objective product attributes essential for informed decision-making. While Large Language Models (LLMs) have shown significant success in various Natural Language Processing (NLP) tasks, their potential in M-OS remains largely unexplored. Additionally, the lack of evaluation datasets for this task has impeded further advancements. To bridge this gap, we introduce M-OS-EVAL, a benchmark dataset for evaluating multi-source opinion summaries across 7 key dimensions: fluency, coherence, relevance, faithfulness, aspect coverage, sentiment consistency, specificity. Our results demonstrate that M-OS significantly enhances user engagement, as evidenced by a user study in which, on average, 87% of participants preferred M-OS over opinion summaries. Our experiments demonstrate that factually enriched summaries enhance user engagement. Notably, M-OS-PROMPTS exhibit stronger alignment with human judgment, achieving an average Spearman correlation of \r{ho} = 0.74, which surpasses the performance of previous methodologies. 

---
# CoSteer: Collaborative Decoding-Time Personalization via Local Delta Steering 

**Authors**: Hang Lv, Sheng Liang, Hao Wang, Hongchao Gu, Yaxiong Wu, Wei Guo, Defu Lian, Yong Liu, Enhong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.04756)  

**Abstract**: Personalized text generation has become crucial for adapting language models to diverse and evolving users' personal context across cultural, temporal, and contextual dimensions. While existing methods often rely on centralized fine-tuning or static preference alignment, they struggle to achieve real-time adaptation under resource constraints inherent to personal devices. This limitation creates a dilemma: large cloud-based models lack access to localized user-specific information, while small on-device models cannot match the generation quality of their cloud counterparts. To address this dichotomy, we present CoSteer, a novel collaborative framework that enables decoding-time personalization through localized delta steering. Our key insight lies in leveraging the logits difference between personal context-aware and -agnostic outputs from local small models as steering signals for cloud-based LLMs. Specifically, we formulate token-level optimization as an online learning problem, where local delta vectors dynamically adjust the remote LLM's logits within the on-device environment. This approach preserves privacy by transmitting only the final steered tokens rather than raw data or intermediate vectors, while maintaining cloud-based LLMs' general capabilities without fine-tuning. Through comprehensive experiments on various personalized generation tasks, we demonstrate that CoSteer effectively assists LLMs in generating personalized content by leveraging locally stored user profiles and histories, ensuring privacy preservation through on-device data processing while maintaining acceptable computational overhead. 

---
# Interpretable Mnemonic Generation for Kanji Learning via Expectation-Maximization 

**Authors**: Jaewook Lee, Alexander Scarlatos, Andrew Lan  

**Link**: [PDF](https://arxiv.org/pdf/2507.05137)  

**Abstract**: Learning Japanese vocabulary is a challenge for learners from Roman alphabet backgrounds due to script differences. Japanese combines syllabaries like hiragana with kanji, which are logographic characters of Chinese origin. Kanji are also complicated due to their complexity and volume. Keyword mnemonics are a common strategy to aid memorization, often using the compositional structure of kanji to form vivid associations. Despite recent efforts to use large language models (LLMs) to assist learners, existing methods for LLM-based keyword mnemonic generation function as a black box, offering limited interpretability. We propose a generative framework that explicitly models the mnemonic construction process as driven by a set of common rules, and learn them using a novel Expectation-Maximization-type algorithm. Trained on learner-authored mnemonics from an online platform, our method learns latent structures and compositional rules, enabling interpretable and systematic mnemonics generation. Experiments show that our method performs well in the cold-start setting for new learners while providing insight into the mechanisms behind effective mnemonic creation. 

---
# "This Suits You the Best": Query Focused Comparative Explainable Summarization 

**Authors**: Arnav Attri, Anuj Attri, Pushpak Bhattacharyya, Suman Banerjee, Amey Patil, Muthusamy Chelliah, Nikesh Garera  

**Link**: [PDF](https://arxiv.org/pdf/2507.04733)  

**Abstract**: Product recommendations inherently involve comparisons, yet traditional opinion summarization often fails to provide holistic comparative insights. We propose the novel task of generating Query-Focused Comparative Explainable Summaries (QF-CES) using Multi-Source Opinion Summarization (M-OS). To address the lack of query-focused recommendation datasets, we introduce MS-Q2P, comprising 7,500 queries mapped to 22,500 recommended products with metadata. We leverage Large Language Models (LLMs) to generate tabular comparative summaries with query-specific explanations. Our approach is personalized, privacy-preserving, recommendation engine-agnostic, and category-agnostic. M-OS as an intermediate step reduces inference latency approximately by 40% compared to the direct input approach (DIA), which processes raw data directly. We evaluate open-source and proprietary LLMs for generating and assessing QF-CES. Extensive evaluations using QF-CES-PROMPT across 5 dimensions (clarity, faithfulness, informativeness, format adherence, and query relevance) showed an average Spearman correlation of 0.74 with human judgments, indicating its potential for QF-CES evaluation. 

---
# Building Open-Retrieval Conversational Question Answering Systems by Generating Synthetic Data and Decontextualizing User Questions 

**Authors**: Christos Vlachos, Nikolaos Stylianou, Alexandra Fiotaki, Spiros Methenitis, Elisavet Palogiannidi, Themos Stafylakis, Ion Androutsopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2507.04884)  

**Abstract**: We consider open-retrieval conversational question answering (OR-CONVQA), an extension of question answering where system responses need to be (i) aware of dialog history and (ii) grounded in documents (or document fragments) retrieved per question. Domain-specific OR-CONVQA training datasets are crucial for real-world applications, but hard to obtain. We propose a pipeline that capitalizes on the abundance of plain text documents in organizations (e.g., product documentation) to automatically produce realistic OR-CONVQA dialogs with annotations. Similarly to real-world humanannotated OR-CONVQA datasets, we generate in-dialog question-answer pairs, self-contained (decontextualized, e.g., no referring expressions) versions of user questions, and propositions (sentences expressing prominent information from the documents) the system responses are grounded in. We show how the synthetic dialogs can be used to train efficient question rewriters that decontextualize user questions, allowing existing dialog-unaware retrievers to be utilized. The retrieved information and the decontextualized question are then passed on to an LLM that generates the system's response. 

---
# Spec-TOD: A Specialized Instruction-Tuned LLM Framework for Efficient Task-Oriented Dialogue Systems 

**Authors**: Quang-Vinh Nguyen, Quang-Chieu Nguyen, Hoang Pham, Khac-Hoai Nam Bui  

**Link**: [PDF](https://arxiv.org/pdf/2507.04841)  

**Abstract**: Task-oriented dialogue (TOD) systems facilitate goal-driven interactions between users and machines. While recent advances in deep learning have improved the performance, TOD systems often struggle in low-resource scenarios with limited labeled data. To address this challenge, we propose Spec-TOD, a novel framework designed to train an end-to-end TOD system with limited data. Spec-TOD introduces two main innovations: (i) a novel specialized end-to-end TOD framework that incorporates explicit task instructions for instruction-tuned large language models (LLMs), and (ii) an efficient training strategy that leverages lightweight, specialized LLMs to achieve strong performance with minimal supervision. Experiments on the MultiWOZ dataset, a widely used TOD benchmark, demonstrate that Spec-TOD achieves competitive results while significantly reducing the need for labeled data. These findings highlight the potential of the proposed framework in advancing efficient and effective TOD systems in low-resource settings. 

---
# Knowledge-Aware Self-Correction in Language Models via Structured Memory Graphs 

**Authors**: Swayamjit Saha  

**Link**: [PDF](https://arxiv.org/pdf/2507.04625)  

**Abstract**: Large Language Models (LLMs) are powerful yet prone to generating factual errors, commonly referred to as hallucinations. We present a lightweight, interpretable framework for knowledge-aware self-correction of LLM outputs using structured memory graphs based on RDF triples. Without retraining or fine-tuning, our method post-processes model outputs and corrects factual inconsistencies via external semantic memory. We demonstrate the approach using DistilGPT-2 and show promising results on simple factual prompts. 

---
# DP-Fusion: Token-Level Differentially Private Inference for Large Language Models 

**Authors**: Rushil Thareja, Preslav Nakov, Praneeth Vepakomma, Nils Lukas  

**Link**: [PDF](https://arxiv.org/pdf/2507.04531)  

**Abstract**: Large language models (LLMs) can leak sensitive information from their context through generated outputs, either accidentally or when prompted adversarially. Existing defenses that aim to preserve context privacy during inference either lack formal guarantees or suffer from a poor utility/privacy trade-off. We propose DP-Fusion, a token-level Differentially Private Inference (DPI) mechanism that provably bounds how much an LLM's outputs reveal about sensitive tokens in its context. We demonstrate DPI through the task of document privatization, where the goal is to paraphrase documents so that sensitive content (e.g., Personally Identifiable Information, PII) cannot be reliably inferred, while still preserving the overall utility of the text. This is controlled by a parameter $\epsilon$: $\epsilon=0$ hides PII entirely, while higher values trade off privacy for improved paraphrase quality. DP-Fusion works as follows: (i) partition sensitive tokens into disjoint privacy groups, (ii) run the LLM once per group, and (iii) blend the output distributions so that the final output remains within a fixed statistical distance of the baseline distribution produced when no privacy group is revealed. This approach allows fine-grained control over the privacy/utility trade-off but requires multiple LLM forward passes. 

---
# GradOT: Training-free Gradient-preserving Offsite-tuning for Large Language Models 

**Authors**: Kai Yao, Zhaorui Tan, Penglei Gao, Lichun Li, Kaixin Wu, Yinggui Wang, Yuan Zhao, Yixin Ji, Wei Wang, Jianke Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2507.04455)  

**Abstract**: The rapid growth of large language models (LLMs) with traditional centralized fine-tuning emerges as a key technique for adapting these models to domain-specific challenges, yielding privacy risks for both model and data owners. One promising solution, called offsite-tuning (OT), is proposed to address these challenges, where a weaker emulator is compressed from the original model and further fine-tuned with adapter to enhance privacy. However, the existing OT-based methods require high computational costs and lack theoretical analysis. This paper introduces a novel OT approach based on gradient-preserving compression, named GradOT. By analyzing the OT problem through the lens of optimization, we propose a method that selectively applies compression techniques such as rank compression and channel pruning, preserving the gradients of fine-tuned adapters while ensuring privacy. Extensive experiments demonstrate that our approach surpasses existing OT methods, both in terms of privacy protection and model performance. Our method provides a theoretical foundation for OT and offers a practical, training-free solution for offsite-tuning of large-scale LLMs. 

---
# Unveiling the Potential of Diffusion Large Language Model in Controllable Generation 

**Authors**: Zhen Xiong, Yujun Cai, Zhecheng Li, Yiwei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.04504)  

**Abstract**: Diffusion models, originally developed for image generation, have emerged as a promising alternative to autoregressive large language models (LLMs). We present a theoretical analysis comparing autoregressive and masked diffusion LLMs, revealing that the intrinsic bidirectional attention mechanism of diffusion LLMs (dLLMs) enables superior context modeling and generation controllability. However, existing dLLM applications face significant challenges in controllable generation: the native multi-step denoising process exhibits high sensitivity to sequence length, elevated hallucination rates, and prohibitive inference costs without specialized optimizations. To address these limitations, we propose \textbf{S}elf-adaptive \textbf{S}chema \textbf{S}caffolding ($S^3$), a novel framework that enables dLLMs to generate structured outputs (e.g., JSON) while maintaining semantic fidelity and accelerating inference. Our approach injects the target schema structure into the output context, reducing unnecessary computation while improving controllability. Extensive experiments demonstrate that $S^3$ achieves substantial improvements: 65\% increase in structural adherence, 48\% enhancement in content fidelity, and 17\% reduction in hallucination rates compared to baseline. These results establish both theoretical foundations and practical pathways for deploying diffusion models in controllable text generation tasks. Code and data will be publicly released. 

---
# PRIME: Large Language Model Personalization with Cognitive Memory and Thought Processes 

**Authors**: Xinliang Frederick Zhang, Nick Beauchamp, Lu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.04607)  

**Abstract**: Large language model (LLM) personalization aims to align model outputs with individuals' unique preferences and opinions. While recent efforts have implemented various personalization methods, a unified theoretical framework that can systematically understand the drivers of effective personalization is still lacking. In this work, we integrate the well-established cognitive dual-memory model into LLM personalization, by mirroring episodic memory to historical user engagements and semantic memory to long-term, evolving user beliefs. Specifically, we systematically investigate memory instantiations and introduce a unified framework, PRIME, using episodic and semantic memory mechanisms. We further augment PRIME with a novel personalized thinking capability inspired by the slow thinking strategy. Moreover, recognizing the absence of suitable benchmarks, we introduce a dataset using Change My View (CMV) from Reddit, specifically designed to evaluate long-context personalization. Extensive experiments validate PRIME's effectiveness across both long- and short-context scenarios. Further analysis confirms that PRIME effectively captures dynamic personalization beyond mere popularity biases. 

---
# LOOM-Scope: a comprehensive and efficient LOng-cOntext Model evaluation framework 

**Authors**: Zecheng Tang, Haitian Wang, Quantong Qiu, Baibei Ji, Ruoxi Sun, Keyan Zhou, Juntao Li, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.04723)  

**Abstract**: Long-context processing has become a fundamental capability for large language models~(LLMs). To assess model's long-context performance, numerous long-context evaluation benchmarks have been proposed. However, variations in evaluation settings across these benchmarks lead to inconsistent results, making it difficult to draw reliable comparisons. Besides, the high computational cost of long-context evaluation poses a significant barrier for the community to conduct comprehensive assessments of long-context models. In this paper, we propose LOOM-Scope, a comprehensive and efficient framework for long-context evaluation. LOOM-Scope standardizes evaluation settings across diverse benchmarks, supports deployment of efficient long-context inference acceleration methods, and introduces a holistic yet lightweight benchmark suite to evaluate models comprehensively. Homepage: this https URL 

---
# Context Tuning for In-Context Optimization 

**Authors**: Jack Lu, Ryan Teehan, Zhenbang Yang, Mengye Ren  

**Link**: [PDF](https://arxiv.org/pdf/2507.04221)  

**Abstract**: We introduce Context Tuning, a simple and effective method to significantly enhance few-shot adaptation of language models (LLMs) without fine-tuning model parameters. While prompt-based adaptation techniques have demonstrated the effectiveness of lightweight adaptation methods for large language models (LLMs), they typically initialize a trainable prompt or prefix with irrelevant tokens for the task at hand. In contrast, Context Tuning initializes the trainable prompt or prefix with task-specific demonstration examples, leveraging the model's inherent In-Context Learning (ICL) ability to extract relevant information for improved few-shot learning performance. Extensive evaluations on benchmarks such as CrossFit, UnifiedQA, MMLU, BIG-Bench Hard, and ARC demonstrate that Context Tuning outperforms traditional prompt-based adaptation methods and achieves competitive accuracy to Test-Time Training with significantly higher training efficiency. 

---
# Does Learning Mathematical Problem-Solving Generalize to Broader Reasoning? 

**Authors**: Ruochen Zhou, Minrui Xu, Shiqi Chen, Junteng Liu, Yunqi Li, Xinxin Lin, Zhengyu Chen, Junxian He  

**Link**: [PDF](https://arxiv.org/pdf/2507.04391)  

**Abstract**: There has been a growing interest in enhancing the mathematical problem-solving (MPS) capabilities of large language models. While the majority of research efforts concentrate on creating specialized models to solve mathematical problems, it remains unknown how learning mathematical problem-solving generalizes to help develop other reasoning abilities. In this paper, we present an empirical investigation into the generalization potential of various MPS training approaches, such as continual pretraining, instruction tuning, and rule-based reinforcement learning across various data sources, including both short and long chain-of-thought (CoT) samples. Evaluation on 5 mathematical and 8 general reasoning benchmarks show that continual pretraining on math text is able to generalize to general reasoning tasks to some extent. In constrast, instruction tuning on conventional, short MPS samples provides limited benefits and, in many cases, even impairs generalization performance. Notably, training with long CoT responses for MPS samples and incorporating rule-based reinforcement learning on MPS queries exhibit distinct behavior, significantly enhancing generalization by extending the model's reasoning processes into other domains. These results suggest that traditional approaches to learning MPS with short reasoning chains largely fail to achieve robust generalization. However, the emerging paradigm of longer reasoning chains, coupled with self-reflection, offers a promising direction for improving generalized reasoning abilities through learning from specialized domains. 

---
# Large Language Models for Zero-Shot Multicultural Name Recognition 

**Authors**: Thanakorn Phonchai, Surasakdi Siripong, Nicholas Patterson, Owen Campbell  

**Link**: [PDF](https://arxiv.org/pdf/2507.04149)  

**Abstract**: The robust and accurate recognition of multicultural names, particularly those not previously encountered, is a critical challenge in an increasingly globalized digital landscape. Traditional methods often falter when confronted with the vast diversity and novel permutations of names across different linguistic and cultural backgrounds. This paper introduces a novel framework, Prompt-Engineered Fine-Tuning (PEFT) for Large Language Models (LLMs) with Adversarial Data Augmentation and Cultural Knowledge Graph Integration, designed to significantly enhance zero-shot multicultural name recognition. Our approach leverages the powerful linguistic understanding of pre-trained LLMs, transforming the recognition task into a guided generation problem. Through meticulous prompt engineering, dynamic integration of explicit cultural knowledge derived from knowledge graphs, and the strategic application of adversarial data augmentation, we equip the LLM with an unprecedented ability to infer the cultural origin of unseen names. Extensive experiments demonstrate that our PEFT method consistently outperforms established deep learning baselines, including advanced Bi-LSTM models with cultural tags, achieving an impressive 93.1\% overall accuracy and a remarkable 89.5\% accuracy on challenging zero-shot name identification. An in-depth ablation study confirms the synergistic contribution of each component, while a human evaluation highlights our method's performance approaching human expert judgment. This work signifies a substantial leap in multicultural name recognition, offering a highly effective and scalable solution for real-world applications. 

---
# MOMENTS: A Comprehensive Multimodal Benchmark for Theory of Mind 

**Authors**: Emilio Villa-Cueva, S M Masrur Ahmed, Rendi Chevi, Jan Christian Blaise Cruz, Kareem Elzeky, Fermin Cristobal, Alham Fikri Aji, Skyler Wang, Rada Mihalcea, Thamar Solorio  

**Link**: [PDF](https://arxiv.org/pdf/2507.04415)  

**Abstract**: Understanding Theory of Mind is essential for building socially intelligent multimodal agents capable of perceiving and interpreting human behavior. We introduce MOMENTS (Multimodal Mental States), a comprehensive benchmark designed to assess the ToM capabilities of multimodal large language models (LLMs) through realistic, narrative-rich scenarios presented in short films. MOMENTS includes over 2,344 multiple-choice questions spanning seven distinct ToM categories. The benchmark features long video context windows and realistic social interactions that provide deeper insight into characters' mental states. While the visual modality generally enhances model performance, current systems still struggle to integrate it effectively, underscoring the need for further research into AI's multimodal understanding of human behavior. 

---
# BYOKG-RAG: Multi-Strategy Graph Retrieval for Knowledge Graph Question Answering 

**Authors**: Costas Mavromatis, Soji Adeshina, Vassilis N. Ioannidis, Zhen Han, Qi Zhu, Ian Robinson, Bryan Thompson, Huzefa Rangwala, George Karypis  

**Link**: [PDF](https://arxiv.org/pdf/2507.04127)  

**Abstract**: Knowledge graph question answering (KGQA) presents significant challenges due to the structural and semantic variations across input graphs. Existing works rely on Large Language Model (LLM) agents for graph traversal and retrieval; an approach that is sensitive to traversal initialization, as it is prone to entity linking errors and may not generalize well to custom ("bring-your-own") KGs. We introduce BYOKG-RAG, a framework that enhances KGQA by synergistically combining LLMs with specialized graph retrieval tools. In BYOKG-RAG, LLMs generate critical graph artifacts (question entities, candidate answers, reasoning paths, and OpenCypher queries), and graph tools link these artifacts to the KG and retrieve relevant graph context. The retrieved context enables the LLM to iteratively refine its graph linking and retrieval, before final answer generation. By retrieving context from different graph tools, BYOKG-RAG offers a more general and robust solution for QA over custom KGs. Through experiments on five benchmarks spanning diverse KG types, we demonstrate that BYOKG-RAG outperforms the second-best graph retrieval method by 4.5% points while showing better generalization to custom KGs. BYOKG-RAG framework is open-sourced at this https URL. 

---
# SymbolicThought: Integrating Language Models and Symbolic Reasoning for Consistent and Interpretable Human Relationship Understanding 

**Authors**: Runcong Zhao, Qinglin Zhu, Hainiu Xu, Bin Liang, Yulan He, Lin Gui  

**Link**: [PDF](https://arxiv.org/pdf/2507.04189)  

**Abstract**: Understanding character relationships is essential for interpreting complex narratives and conducting socially grounded AI research. However, manual annotation is time-consuming and low in coverage, while large language models (LLMs) often produce hallucinated or logically inconsistent outputs. We present SymbolicThought, a human-in-the-loop framework that combines LLM-based extraction with symbolic reasoning. The system constructs editable character relationship graphs, refines them using seven types of logical constraints, and enables real-time validation and conflict resolution through an interactive interface. To support logical supervision and explainable social analysis, we release a dataset of 160 interpersonal relationships with corresponding logical structures. Experiments show that SymbolicThought improves annotation accuracy and consistency while significantly reducing time cost, offering a practical tool for narrative understanding, explainable AI, and LLM evaluation. 

---
# Token Level Hallucination Detection via Variance in Language Models 

**Authors**: Keshav Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2507.04137)  

**Abstract**: Large Language Models (LLMs) have demonstrated impressive generative capabilities across diverse tasks but remain susceptible to hallucinations, confidently generated yet factually incorrect outputs. We introduce a reference-free, token-level hallucination detection framework that leverages the variance in token log-probabilities across multiple stochastic generations. Unlike prior methods that require ground-truth references or sentence-level verification, our approach is model-agnostic, interpretable, and suited for real-time or post-hoc analysis. We evaluate our method on unanswerable question prompts from the SQuAD v2 dataset and benchmark across three autoregressive models of varying scales: GPT-Neo 125M, Falcon 1B, and Mistral 7B. Through both quantitative metrics and visual diagnostics, we show that token-level variance reliably highlights instability in model outputs and correlates with hallucination patterns. Our framework is lightweight, reproducible, and adaptable to multiple domains, offering a valuable diagnostic tool for analyzing generative reliability in LLMs. 

---
# Fairness Evaluation of Large Language Models in Academic Library Reference Services 

**Authors**: Haining Wang, Jason Clark, Yueru Yan, Star Bradley, Ruiyang Chen, Yiqiong Zhang, Hengyi Fu, Zuoyu Tian  

**Link**: [PDF](https://arxiv.org/pdf/2507.04224)  

**Abstract**: As libraries explore large language models (LLMs) for use in virtual reference services, a key question arises: Can LLMs serve all users equitably, regardless of demographics or social status? While they offer great potential for scalable support, LLMs may also reproduce societal biases embedded in their training data, risking the integrity of libraries' commitment to equitable service. To address this concern, we evaluate whether LLMs differentiate responses across user identities by prompting six state-of-the-art LLMs to assist patrons differing in sex, race/ethnicity, and institutional role. We found no evidence of differentiation by race or ethnicity, and only minor evidence of stereotypical bias against women in one model. LLMs demonstrated nuanced accommodation of institutional roles through the use of linguistic choices related to formality, politeness, and domain-specific vocabularies, reflecting professional norms rather than discriminatory treatment. These findings suggest that current LLMs show a promising degree of readiness to support equitable and contextually appropriate communication in academic library reference services. 

---
# Dissecting Clinical Reasoning in Language Models: A Comparative Study of Prompts and Model Adaptation Strategies 

**Authors**: Mael Jullien, Marco Valentino, Leonardo Ranaldi, Andre Freitas  

**Link**: [PDF](https://arxiv.org/pdf/2507.04142)  

**Abstract**: Recent works on large language models (LLMs) have demonstrated the impact of prompting strategies and fine-tuning techniques on their reasoning capabilities. Yet, their effectiveness on clinical natural language inference (NLI) remains underexplored. This study presents the first controlled evaluation of how prompt structure and efficient fine-tuning jointly shape model performance in clinical NLI. We inspect four classes of prompting strategies to elicit reasoning in LLMs at different levels of abstraction, and evaluate their impact on a range of clinically motivated reasoning types. For each prompting strategy, we construct high-quality demonstrations using a frontier model to distil multi-step reasoning capabilities into smaller models (4B parameters) via Low-Rank Adaptation (LoRA). Across different language models fine-tuned on the NLI4CT benchmark, we found that prompt type alone accounts for up to 44% of the variance in macro-F1. Moreover, LoRA fine-tuning yields consistent gains of +8 to 12 F1, raises output alignment above 97%, and narrows the performance gap to GPT-4o-mini to within 7.1%. Additional experiments on reasoning generalisation reveal that LoRA improves performance in 75% of the models on MedNLI and TREC Clinical Trials Track. Overall, these findings demonstrate that (i) prompt structure is a primary driver of clinical reasoning performance, (ii) compact models equipped with strong prompts and LoRA can rival frontier-scale systems, and (iii) reasoning-type-aware evaluation is essential to uncover prompt-induced trade-offs. Our results highlight the promise of combining prompt design and lightweight adaptation for more efficient and trustworthy clinical NLP systems, providing insights on the strengths and limitations of widely adopted prompting and parameter-efficient techniques in highly specialised domains. 

---
# Think Twice Before You Judge: Mixture of Dual Reasoning Experts for Multimodal Sarcasm Detection 

**Authors**: Soumyadeep Jana, Abhrajyoti Kundu, Sanasam Ranbir Singh  

**Link**: [PDF](https://arxiv.org/pdf/2507.04458)  

**Abstract**: Multimodal sarcasm detection has attracted growing interest due to the rise of multimedia posts on social media. Understanding sarcastic image-text posts often requires external contextual knowledge, such as cultural references or commonsense reasoning. However, existing models struggle to capture the deeper rationale behind sarcasm, relying mainly on shallow cues like image captions or object-attribute pairs from images. To address this, we propose \textbf{MiDRE} (\textbf{Mi}xture of \textbf{D}ual \textbf{R}easoning \textbf{E}xperts), which integrates an internal reasoning expert for detecting incongruities within the image-text pair and an external reasoning expert that utilizes structured rationales generated via Chain-of-Thought prompting to a Large Vision-Language Model. An adaptive gating mechanism dynamically weighs the two experts, selecting the most relevant reasoning path. Experiments on two benchmark datasets show that MiDRE achieves superior performance over baselines. Various qualitative analyses highlight the crucial role of external rationales, revealing that even when they are occasionally noisy, they provide valuable cues that guide the model toward a better understanding of sarcasm. 

---
# Nunchi-Bench: Benchmarking Language Models on Cultural Reasoning with a Focus on Korean Superstition 

**Authors**: Kyuhee Kim, Sangah Lee  

**Link**: [PDF](https://arxiv.org/pdf/2507.04014)  

**Abstract**: As large language models (LLMs) become key advisors in various domains, their cultural sensitivity and reasoning skills are crucial in multicultural environments. We introduce Nunchi-Bench, a benchmark designed to evaluate LLMs' cultural understanding, with a focus on Korean superstitions. The benchmark consists of 247 questions spanning 31 topics, assessing factual knowledge, culturally appropriate advice, and situational interpretation. We evaluate multilingual LLMs in both Korean and English to analyze their ability to reason about Korean cultural contexts and how language variations affect performance. To systematically assess cultural reasoning, we propose a novel evaluation strategy with customized scoring metrics that capture the extent to which models recognize cultural nuances and respond appropriately. Our findings highlight significant challenges in LLMs' cultural reasoning. While models generally recognize factual information, they struggle to apply it in practical scenarios. Furthermore, explicit cultural framing enhances performance more effectively than relying solely on the language of the prompt. To support further research, we publicly release Nunchi-Bench alongside a leaderboard. 

---
# Demystifying ChatGPT: How It Masters Genre Recognition 

**Authors**: Subham Raj, Sriparna Saha, Brijraj Singh, Niranjan Pedanekar  

**Link**: [PDF](https://arxiv.org/pdf/2507.03875)  

**Abstract**: The introduction of ChatGPT has garnered significant attention within the NLP community and beyond. Previous studies have demonstrated ChatGPT's substantial advancements across various downstream NLP tasks, highlighting its adaptability and potential to revolutionize language-related applications. However, its capabilities and limitations in genre prediction remain unclear. This work analyzes three Large Language Models (LLMs) using the MovieLens-100K dataset to assess their genre prediction capabilities. Our findings show that ChatGPT, without fine-tuning, outperformed other LLMs, and fine-tuned ChatGPT performed best overall. We set up zero-shot and few-shot prompts using audio transcripts/subtitles from movie trailers in the MovieLens-100K dataset, covering 1682 movies of 18 genres, where each movie can have multiple genres. Additionally, we extended our study by extracting IMDb movie posters to utilize a Vision Language Model (VLM) with prompts for poster information. This fine-grained information was used to enhance existing LLM prompts. In conclusion, our study reveals ChatGPT's remarkable genre prediction capabilities, surpassing other language models. The integration of VLM further enhances our findings, showcasing ChatGPT's potential for content-related applications by incorporating visual information from movie posters. 

---
# LLMThinkBench: Towards Basic Math Reasoning and Overthinking in Large Language Models 

**Authors**: Gaurav Srivastava, Aafiya Hussain, Sriram Srinivasan, Xuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.04023)  

**Abstract**: Large Language Models (LLMs) have achieved remarkable performance on complex mathematical benchmarks, yet often struggle with simple arithmetic tasks and exhibit a tendency toward over-explaining or "overthinking" answers. To systematically assess this phenomenon, we introduce LLMThinkBench, a modular benchmarking framework that enables researchers to evaluate basic math reasoning and overthinking in LLMs. The framework provides 14 configurable math tasks with randomized test data generation and robust parsing strategies. Researchers can quantify overthinking using our Overthinking Score metric, which captures accuracy-verbosity tradeoffs through harmonic mean formulation. The tool offers flexible evaluation with a scalable vLLM/Transformers backend, multi-GPU support, and full configurability. Users can extend the tool with custom tasks, reproduce experiments with seeding, and generate detailed efficiency reports. Distributed as a pip-installable package with CLI and API access, LLMThinkBench provides researchers and practitioners an accessible, cost-effective alternative to expensive LLM-as-a-judge methods for diagnosing basic reasoning capabilities and efficiency analysis. Package can be installed as: pip install llmthinkbench 

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
# TACOS: Open Tagging and Comparative Scoring for Instruction Fine-Tuning Data Selection 

**Authors**: Xixiang He, Hao Yu, Qiyao Sun, Ao Cheng, Tailai Zhang, Cong Liu, Shuxuan Guo  

**Link**: [PDF](https://arxiv.org/pdf/2507.03673)  

**Abstract**: Instruction Fine-Tuning (IFT) is crucial for aligning large language models (LLMs) with human preferences, and selecting a small yet representative subset from massive data significantly facilitates IFT in terms of both efficiency and effectiveness. Nevertheless, existing approaches suffer from two limitations: the use of simple heuristics restricts data diversity, while the singleton data quality evaluation accounts for inconsistent criteria between independent samples. To address the issues, we present TACOS, an innovative method that integrates Open Tagging and Comparative Scoring for IFT data selection. To capture data diversity, we leverage LLMs to assign open-domain tags to human queries, followed by a normalization stage to denoise the open tags and enable efficient clustering. Additionally, we suggest a comparative scoring method that allows the relative quality evaluation of samples within a cluster, avoiding inconsistent criteria seen in singleton-based evaluations. Extensive experiments across diverse datasets and LLM architectures demonstrate that TACOS outperforms existing approaches by a large margin. Notably, it achieves superior instruction-following performance on MT-Bench and ranks 1st among LLaMA2-7B-Based models on AlpacaEval 2.0, illustrating its efficacy for IFT data selection. 

---
# Conversation Forests: The Key to Fine Tuning Large Language Models for Multi-Turn Medical Conversations is Branching 

**Authors**: Thomas Savage  

**Link**: [PDF](https://arxiv.org/pdf/2507.04099)  

**Abstract**: Fine-tuning methods such as Direct Preference Optimization (DPO) and Group Relative Policy Optimization (GRPO) have demonstrated success in training large language models (LLMs) for single-turn tasks. However, these methods fall short in multi-turn applications, such as diagnostic patient interviewing, where understanding how early conversational turns influence downstream completions and outcomes is essential. In medicine, a multi-turn perspective is critical for learning diagnostic schemas and better understanding conversation dynamics. To address this gap, I introduce Savage Conversation Forests (SCF), a reinforcement learning framework that leverages a branched conversation architecture to fine-tune LLMs for multi-turn dialogue. SCF generates multiple possible conversation continuations at each turn, enabling the model to learn how different early responses affect downstream interactions and diagnostic outcomes. In experiments simulating doctor-patient conversations, SCF with branching outperforms linear conversation architectures on diagnostic accuracy. I hypothesize that SCF's improvements stem from its ability to provide richer, interdependent training signals across conversation turns. These results suggest that a branched training architecture is an important strategy for fine tuning LLMs in complex multi-turn conversational tasks. 

---
# OrthoRank: Token Selection via Sink Token Orthogonality for Efficient LLM inference 

**Authors**: Seungjun Shin, Jaehoon Oh, Dokwan Oh  

**Link**: [PDF](https://arxiv.org/pdf/2507.03865)  

**Abstract**: Attention mechanisms are central to the success of large language models (LLMs), enabling them to capture intricate token dependencies and implicitly assign importance to each token. Recent studies have revealed the sink token, which receives disproportionately high attention despite their limited semantic role. In this paper, we first expand the relationship between the sink token and other tokens, moving beyond attention to explore their similarity in hidden states, considering the layer depth. We observe that as the layers get deeper, the cosine similarity between the normalized hidden states of the sink token and those of other tokens increases, and that the normalized hidden states of the sink token exhibit negligible changes. These imply that other tokens consistently are directed toward the sink token throughout the layers. Next, we propose a dynamic token selection method, called OrthoRank, using these findings to select important tokens. Specifically, in a certain layer, we define token importance by the speed at which the token moves toward the sink token. This is converted into orthogonality with the sink token, meaning that tokens that are more orthogonal to the sink token are assigned greater importance. Finally, through extensive experiments, we demonstrated that our method results in lower perplexity and higher zero-shot accuracy compared to layer pruning methods at the same sparsity ratio with comparable throughput, while also achieving superior performance on LongBench. 

---
# STRUCTSENSE: A Task-Agnostic Agentic Framework for Structured Information Extraction with Human-In-The-Loop Evaluation and Benchmarking 

**Authors**: Tek Raj Chhetri, Yibei Chen, Puja Trivedi, Dorota Jarecka, Saif Haobsh, Patrick Ray, Lydia Ng, Satrajit S. Ghosh  

**Link**: [PDF](https://arxiv.org/pdf/2507.03674)  

**Abstract**: The ability to extract structured information from unstructured sources-such as free-text documents and scientific literature-is critical for accelerating scientific discovery and knowledge synthesis. Large Language Models (LLMs) have demonstrated remarkable capabilities in various natural language processing tasks, including structured information extraction. However, their effectiveness often diminishes in specialized, domain-specific contexts that require nuanced understanding and expert-level domain knowledge. In addition, existing LLM-based approaches frequently exhibit poor transferability across tasks and domains, limiting their scalability and adaptability. To address these challenges, we introduce StructSense, a modular, task-agnostic, open-source framework for structured information extraction built on LLMs. StructSense is guided by domain-specific symbolic knowledge encoded in ontologies, enabling it to navigate complex domain content more effectively. It further incorporates agentic capabilities through self-evaluative judges that form a feedback loop for iterative refinement, and includes human-in-the-loop mechanisms to ensure quality and validation. We demonstrate that StructSense can overcome both the limitations of domain sensitivity and the lack of cross-task generalizability, as shown through its application to diverse neuroscience information extraction tasks. 

---
# H2HTalk: Evaluating Large Language Models as Emotional Companion 

**Authors**: Boyang Wang, Yalun Wu, Hongcheng Guo, Zhoujun Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.03543)  

**Abstract**: As digital emotional support needs grow, Large Language Model companions offer promising authentic, always-available empathy, though rigorous evaluation lags behind model advancement. We present Heart-to-Heart Talk (H2HTalk), a benchmark assessing companions across personality development and empathetic interaction, balancing emotional intelligence with linguistic fluency. H2HTalk features 4,650 curated scenarios spanning dialogue, recollection, and itinerary planning that mirror real-world support conversations, substantially exceeding previous datasets in scale and diversity. We incorporate a Secure Attachment Persona (SAP) module implementing attachment-theory principles for safer interactions. Benchmarking 50 LLMs with our unified protocol reveals that long-horizon planning and memory retention remain key challenges, with models struggling when user needs are implicit or evolve mid-conversation. H2HTalk establishes the first comprehensive benchmark for emotionally intelligent companions. We release all materials to advance development of LLMs capable of providing meaningful and safe psychological support. 

---
# Improving Social Determinants of Health Documentation in French EHRs Using Large Language Models 

**Authors**: Adrien Bazoge, Pacôme Constant dit Beaufils, Mohammed Hmitouch, Romain Bourcier, Emmanuel Morin, Richard Dufour, Béatrice Daille, Pierre-Antoine Gourraud, Matilde Karakachoff  

**Link**: [PDF](https://arxiv.org/pdf/2507.03433)  

**Abstract**: Social determinants of health (SDoH) significantly influence health outcomes, shaping disease progression, treatment adherence, and health disparities. However, their documentation in structured electronic health records (EHRs) is often incomplete or missing. This study presents an approach based on large language models (LLMs) for extracting 13 SDoH categories from French clinical notes. We trained Flan-T5-Large on annotated social history sections from clinical notes at Nantes University Hospital, France. We evaluated the model at two levels: (i) identification of SDoH categories and associated values, and (ii) extraction of detailed SDoH with associated temporal and quantitative information. The model performance was assessed across four datasets, including two that we publicly release as open resources. The model achieved strong performance for identifying well-documented categories such as living condition, marital status, descendants, job, tobacco, and alcohol use (F1 score > 0.80). Performance was lower for categories with limited training data or highly variable expressions, such as employment status, housing, physical activity, income, and education. Our model identified 95.8% of patients with at least one SDoH, compared to 2.8% for ICD-10 codes from structured EHR data. Our error analysis showed that performance limitations were linked to annotation inconsistencies, reliance on English-centric tokenizer, and reduced generalizability due to the model being trained on social history sections only. These results demonstrate the effectiveness of NLP in improving the completeness of real-world SDoH data in a non-English EHR system. 

---
# Graph Repairs with Large Language Models: An Empirical Study 

**Authors**: Hrishikesh Terdalkar, Angela Bonifati, Andrea Mauri  

**Link**: [PDF](https://arxiv.org/pdf/2507.03410)  

**Abstract**: Property graphs are widely used in domains such as healthcare, finance, and social networks, but they often contain errors due to inconsistencies, missing data, or schema violations. Traditional rule-based and heuristic-driven graph repair methods are limited in their adaptability as they need to be tailored for each dataset. On the other hand, interactive human-in-the-loop approaches may become infeasible when dealing with large graphs, as the cost--both in terms of time and effort--of involving users becomes too high. Recent advancements in Large Language Models (LLMs) present new opportunities for automated graph repair by leveraging contextual reasoning and their access to real-world knowledge. We evaluate the effectiveness of six open-source LLMs in repairing property graphs. We assess repair quality, computational cost, and model-specific performance. Our experiments show that LLMs have the potential to detect and correct errors, with varying degrees of accuracy and efficiency. We discuss the strengths, limitations, and challenges of LLM-driven graph repair and outline future research directions for improving scalability and interpretability. 

---
# WETBench: A Benchmark for Detecting Task-Specific Machine-Generated Text on Wikipedia 

**Authors**: Gerrit Quaremba, Elizabeth Black, Denny Vrandečić, Elena Simperl  

**Link**: [PDF](https://arxiv.org/pdf/2507.03373)  

**Abstract**: Given Wikipedia's role as a trusted source of high-quality, reliable content, concerns are growing about the proliferation of low-quality machine-generated text (MGT) produced by large language models (LLMs) on its platform. Reliable detection of MGT is therefore essential. However, existing work primarily evaluates MGT detectors on generic generation tasks rather than on tasks more commonly performed by Wikipedia editors. This misalignment can lead to poor generalisability when applied in real-world Wikipedia contexts. We introduce WETBench, a multilingual, multi-generator, and task-specific benchmark for MGT detection. We define three editing tasks, empirically grounded in Wikipedia editors' perceived use cases for LLM-assisted editing: Paragraph Writing, Summarisation, and Text Style Transfer, which we implement using two new datasets across three languages. For each writing task, we evaluate three prompts, generate MGT across multiple generators using the best-performing prompt, and benchmark diverse detectors. We find that, across settings, training-based detectors achieve an average accuracy of 78%, while zero-shot detectors average 58%. These results show that detectors struggle with MGT in realistic generation scenarios and underscore the importance of evaluating such models on diverse, task-specific data to assess their reliability in editor-driven contexts. 

---
# Read Quietly, Think Aloud: Decoupling Comprehension and Reasoning in LLMs 

**Authors**: Yuanxin Wang, Ganesh Venkatesh  

**Link**: [PDF](https://arxiv.org/pdf/2507.03327)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable proficiency in understanding text and generating high-quality responses. However, a critical distinction from human cognition is their typical lack of a distinct internal `reading' or deliberation phase before `speaking' (i.e., generating text). Humans often engage in silent reading to comprehend context and formulate thoughts prior to articulation. This paper investigates methods to imbue LLMs with a similar capacity for internal processing.
We introduce and evaluate techniques that encourage LLMs to `read silently.' Our findings indicate that even a straightforward approach, such as providing the model with an initial contextual prompt or `reading space' before it begins predicting subsequent tokens for the final output, can yield significant performance improvements. We further enhance this concept by developing a `reading buddy' architecture, where an auxiliary component silently processes the input and provides refined contextual insights to the primary generation model. These approaches aim to foster deeper understanding from LLMs so that they can produce better reasoned responses, moving them one step closer to more human-like text processing. Our results indicate that these simple techniques can provide surprisingly strong impact on accuracy with multiple point accuracy boost. 

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
# Making Sense of Korean Sentences: A Comprehensive Evaluation of LLMs through KoSEnd Dataset 

**Authors**: Seunguk Yu, Kyeonghyun Kim, Jungmin Yun, Youngbin Kim  

**Link**: [PDF](https://arxiv.org/pdf/2507.03378)  

**Abstract**: Although LLMs have made significant progress in various languages, there are still concerns about their effectiveness with low-resource agglutinative languages compared to languages such as English. In this study, we focused on Korean, a language known for its complex sentence endings, and evaluated LLMs on this challenging aspect. We introduce the Korean Sentence Endings (KoSEnd) dataset, which includes 3,000 sentences, each annotated for the naturalness of 15 sentence ending forms. These were collected from diverse sources to cover a range of contexts. We evaluated 11 LLMs to assess their understanding of Korean sentence endings, analyzing them based on parameter count and prediction consistency. Notably, we found that informing models about the possibility of missing sentence endings improved performance, highlighting the impact of explicitly considering certain linguistic features. 

---
# Expert-level validation of AI-generated medical text with scalable language models 

**Authors**: Asad Aali, Vasiliki Bikia, Maya Varma, Nicole Chiou, Sophie Ostmeier, Arnav Singhvi, Magdalini Paschali, Ashwin Kumar, Andrew Johnston, Karimar Amador-Martinez, Eduardo Juan Perez Guerrero, Paola Naovi Cruz Rivera, Sergios Gatidis, Christian Bluethgen, Eduardo Pontes Reis, Eddy D. Zandee van Rilland, Poonam Laxmappa Hosamani, Kevin R Keet, Minjoung Go, Evelyn Ling, David B. Larson, Curtis Langlotz, Roxana Daneshjou, Jason Hom, Sanmi Koyejo, Emily Alsentzer, Akshay S. Chaudhari  

**Link**: [PDF](https://arxiv.org/pdf/2507.03152)  

**Abstract**: With the growing use of language models (LMs) in clinical environments, there is an immediate need to evaluate the accuracy and safety of LM-generated medical text. Currently, such evaluation relies solely on manual physician review. However, detecting errors in LM-generated text is challenging because 1) manual review is costly and 2) expert-composed reference outputs are often unavailable in real-world settings. While the "LM-as-judge" paradigm (a LM evaluating another LM) offers scalable evaluation, even frontier LMs can miss subtle but clinically significant errors. To address these challenges, we propose MedVAL, a self-supervised framework that leverages synthetic data to train evaluator LMs to assess whether LM-generated medical outputs are factually consistent with inputs, without requiring physician labels or reference outputs. To evaluate LM performance, we introduce MedVAL-Bench, a dataset containing 840 outputs annotated by physicians, following a physician-defined taxonomy of risk levels and error categories. Across 6 diverse medical tasks and 10 state-of-the-art LMs spanning open-source, proprietary, and medically adapted models, MedVAL fine-tuning significantly improves (p < 0.001) alignment with physicians on both seen and unseen tasks, increasing average F1 scores from 66% to 83%, with per-sample safety classification scores up to 86%. MedVAL improves the performance of even the best-performing proprietary LM (GPT-4o) by 8%. To support a scalable, risk-aware pathway towards clinical integration, we open-source the 1) codebase ( this https URL ), 2) MedVAL-Bench ( this https URL ), and 3) MedVAL-4B ( this https URL ), the best-performing open-source LM. Our research provides the first evidence of LMs approaching expert-level validation ability for medical text. 

---
# RefineX: Learning to Refine Pre-training Data at Scale from Expert-Guided Programs 

**Authors**: Baolong Bi, Shenghua Liu, Xingzhang Ren, Dayiheng Liu, Junyang Lin, Yiwei Wang, Lingrui Mei, Junfeng Fang, Jiafeng Guo, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2507.03253)  

**Abstract**: The foundational capabilities of large language models (LLMs) are deeply influenced by the quality of their pre-training corpora. However, enhancing data quality at scale remains a significant challenge, primarily due to the trade-off between refinement effectiveness and processing efficiency. While rule-based filtering remains the dominant paradigm, it typically operates at the document level and lacks the granularity needed to refine specific content within documents. Inspired by emerging work such as ProX, we propose $\textbf{RefineX}$, a novel framework for large-scale, surgical refinement of pre-training data through programmatic editing tasks. RefineX enables efficient and fine-grained data refinement while reliably preserving the diversity and naturalness of raw text. The core strength of RefineX lies in distilling high-quality, expert-guided end-to-end refinement results into minimal edit-based deletion programs. This high-precision distillation pipeline is used to train an efficient and reliable refine model that can systematically improve every instance in the corpus at scale. We evaluate RefineX across from-scratch pre-training at multiple model scales and find that it consistently outperforms models trained on raw, filtered, or alternatively refined data across diverse downstream tasks. On the 750M model, RefineX yields 2.6%-7.2% average gains on lighteval tasks, and achieves comparable performance using significantly fewer training tokens. Further analysis shows that RefineX reliably enhances text quality with both high efficiency and precision, outperforming prior approaches such as end-to-end generation and Prox-C. These results position RefineX as a scalable, effective, and reliable solution for optimizing pre-training data in modern LLM pipelines. 

---
# ReliableMath: Benchmark of Reliable Mathematical Reasoning on Large Language Models 

**Authors**: Boyang Xue, Qi Zhu, Rui Wang, Sheng Wang, Hongru Wang, Fei Mi, Yasheng Wang, Lifeng Shang, Qun Liu, Kam-Fai Wong  

**Link**: [PDF](https://arxiv.org/pdf/2507.03133)  

**Abstract**: Although demonstrating remarkable performance on reasoning tasks, Large Language Models (LLMs) still tend to fabricate unreliable responses when confronted with problems that are unsolvable or beyond their capability, severely undermining the reliability. Prior studies of LLM reliability have primarily focused on knowledge tasks to identify unanswerable questions, while mathematical reasoning tasks have remained unexplored due to the dearth of unsolvable math problems. To systematically investigate LLM reliability in mathematical reasoning tasks, we formulate the reliability evaluation for both solvable and unsolvable problems. We then develop a ReliableMath dataset which incorporates open-source solvable problems and high-quality unsolvable problems synthesized by our proposed construction workflow with human evaluations. Experiments are conducted on various LLMs with several key findings uncovered. LLMs fail to directly identify unsolvable problems and always generate fabricated responses. When instructing LLMs to indicate unsolvability using a reliable prompt, the reliability of larger-sized LLMs remains on solvable problems, but notably improves on unsolvable problems yet still falls short of solvable problems. However, small LLMs rarely show any progress despite employing reliable prompts. Therefore, we further propose an alignment strategy to enhance small LLMs' reliability, which can significantly improve LLM reliability performances on both in-domain and out-of-domain tasks. 

---
# BMMR: A Large-Scale Bilingual Multimodal Multi-Discipline Reasoning Dataset 

**Authors**: Zhiheng Xi, Guanyu Li, Yutao Fan, Honglin Guo, Yufang Liu, Xiaoran Fan, Jiaqi Liu, Jingchao Ding, Wangmeng Zuo, Zhenfei Yin, Lei Bai, Tao Ji, Tao Gui, Qi Zhang, Xuanjing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2507.03483)  

**Abstract**: In this paper, we introduce BMMR, a large-scale bilingual, multimodal, multi-disciplinary reasoning dataset for the community to develop and evaluate large multimodal models (LMMs). BMMR comprises 110k college-level questions spanning 300 UNESCO-defined subjects, spanning diverse formats-multiple-choice, fill-in-the-blank, and open-ended QA-and sourced from both print and digital media such as books, exams, and quizzes. All data are curated and filtered via a human-in-the-loop and scalable framework, and each instance is paired with a high-quality reasoning path. The dataset is organized into two parts: BMMR-Eval that comprises 20,458 high-quality instances to comprehensively assess LMMs' knowledge and reasoning across multiple disciplines in both Chinese and English; and BMMR-Train that contains 88,991 instances to support further research and development, extending the current focus on mathematical reasoning to diverse disciplines and domains. In addition, we propose the process-based multi-discipline verifier (i.e., BMMR-Verifier) for accurate and fine-grained evaluation of reasoning paths. Extensive experiments on 24 models reveal that (i) even SOTA models (e.g., o3 and Gemini-2.5-Pro) leave substantial headroom on BMMR-Eval; (ii) reasoning models exhibit discipline bias and outperform LMMs only on specific subjects; (iii) open-source models still trail their proprietary counterparts; and (iv) fine-tuning on BMMR-Train narrows this gap. Additionally, we conduct reasoning-chain analyses using BMMR-Verifier and other in-depth studies, uncovering the challenges LMMs currently face in multidisciplinary reasoning. We will release the data, and we hope our work can offer insights and contributions to the community. 

---
# RLVER: Reinforcement Learning with Verifiable Emotion Rewards for Empathetic Agents 

**Authors**: Peisong Wang, Ruotian Ma, Bang Zhang, Xingyu Chen, Zhiwei He, Kang Luo, Qingsong Lv, Qingxuan Jiang, Zheng Xie, Shanyi Wang, Yuan Li, Fanghua Ye, Jian Li, Yifan Yang, Zhaopeng Tu, Xiaolong Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.03112)  

**Abstract**: Large language models (LLMs) excel at logical and algorithmic reasoning, yet their emotional intelligence (EQ) still lags far behind their cognitive prowess. While reinforcement learning from verifiable rewards (RLVR) has advanced in other domains, its application to dialogue-especially for emotional intelligence-remains underexplored. In this work, we introduce RLVER, the first end-to-end reinforcement learning framework that leverages verifiable emotion rewards from simulated users to cultivate higher-order empathetic abilities in LLMs. Within this framework, self-consistent affective simulated users engage in dialogue rollouts and produce deterministic emotion scores during conversations, serving as reward signals to guide the LLM's learning. Fine-tuning publicly available Qwen2.5-7B-Instruct model with PPO boosts its Sentient-Benchmark score from 13.3 to 79.2 while largely preserving mathematical and coding competence. Extensive experiments reveal that: (i) RLVER consistently improves multiple dialogue capabilities; (ii) Thinking and non-thinking models show distinct trends--thinking models excel in empathy and insight, while non-thinking models favor action; (iii) GRPO often yields stable gains, while PPO can push certain capabilities to a higher ceiling; (iv) More challenging environments are not always better-moderate ones can yield stronger outcomes. Our results show that RLVER is a practical route toward emotionally intelligent and broadly capable language agents. 

---
# Recon, Answer, Verify: Agents in Search of Truth 

**Authors**: Satyam Shukla, Himanshu Dutta, Pushpak Bhattacharyya  

**Link**: [PDF](https://arxiv.org/pdf/2507.03671)  

**Abstract**: Automated fact checking with large language models (LLMs) offers a scalable alternative to manual verification. Evaluating fact checking is challenging as existing benchmark datasets often include post claim analysis and annotator cues, which are absent in real world scenarios where claims are fact checked immediately after being made. This limits the realism of current evaluations. We present Politi Fact Only (PFO), a 5 class benchmark dataset of 2,982 political claims from this http URL, where all post claim analysis and annotator cues have been removed manually. This ensures that models are evaluated using only the information that would have been available prior to the claim's verification. Evaluating LLMs on PFO, we see an average performance drop of 22% in terms of macro f1 compared to PFO's unfiltered version. Based on the identified challenges of the existing LLM based fact checking system, we propose RAV (Recon Answer Verify), an agentic framework with three agents: question generator, answer generator, and label generator. Our pipeline iteratively generates and answers sub questions to verify different aspects of the claim before finally generating the label. RAV generalizes across domains and label granularities, and it outperforms state of the art approaches on well known baselines RAWFC (fact checking, 3 class) by 25.28%, and on HOVER (encyclopedia, 2 class) by 1.54% on 2 hop, 4.94% on 3 hop, and 1.78% on 4 hop, sub categories respectively. RAV shows the least performance drop compared to baselines of 16.3% in macro f1 when we compare PFO with its unfiltered version. 

---
# AI-VaxGuide: An Agentic RAG-Based LLM for Vaccination Decisions 

**Authors**: Abdellah Zeggai, Ilyes Traikia, Abdelhak Lakehal, Abdennour Boulesnane  

**Link**: [PDF](https://arxiv.org/pdf/2507.03493)  

**Abstract**: Vaccination plays a vital role in global public health, yet healthcare professionals often struggle to access immunization guidelines quickly and efficiently. National protocols and WHO recommendations are typically extensive and complex, making it difficult to extract precise information, especially during urgent situations. This project tackles that issue by developing a multilingual, intelligent question-answering system that transforms static vaccination guidelines into an interactive and user-friendly knowledge base. Built on a Retrieval-Augmented Generation (RAG) framework and enhanced with agent-based reasoning (Agentic RAG), the system provides accurate, context-sensitive answers to complex medical queries. Evaluation shows that Agentic RAG outperforms traditional methods, particularly in addressing multi-step or ambiguous questions. To support clinical use, the system is integrated into a mobile application designed for real-time, point-of-care access to essential vaccine information. AI-VaxGuide model is publicly available on this https URL 

---
# The Book of Life approach: Enabling richness and scale for life course research 

**Authors**: Mark D. Verhagen, Benedikt Stroebl, Tiffany Liu, Lydia T. Liu, Matthew J. Salganik  

**Link**: [PDF](https://arxiv.org/pdf/2507.03027)  

**Abstract**: For over a century, life course researchers have faced a choice between two dominant methodological approaches: qualitative methods that analyze rich data but are constrained to small samples, and quantitative survey-based methods that study larger populations but sacrifice data richness for scale. Two recent technological developments now enable us to imagine a hybrid approach that combines some of the depth of the qualitative approach with the scale of quantitative methods. The first development is the steady rise of ''complex log data,'' behavioral data that is logged for purposes other than research but that can be repurposed to construct rich accounts of people's lives. The second is the emergence of large language models (LLMs) with exceptional pattern recognition capabilities on plain text. In this paper, we take a necessary step toward creating this hybrid approach by developing a flexible procedure to transform complex log data into a textual representation of an individual's life trajectory across multiple domains, over time, and in context. We call this data representation a ''book of life.'' We illustrate the feasibility of our approach by writing over 100 million books of life covering many different facets of life, over time and placed in social context using Dutch population-scale registry data. We open source the book of life toolkit (BOLT), and invite the research community to explore the many potential applications of this approach. 

---
# GRAFT: A Graph-based Flow-aware Agentic Framework for Document-level Machine Translation 

**Authors**: Himanshu Dutta, Sunny Manchanda, Prakhar Bapat, Meva Ram Gurjar, Pushpak Bhattacharyya  

**Link**: [PDF](https://arxiv.org/pdf/2507.03311)  

**Abstract**: Document level Machine Translation (DocMT) approaches often struggle with effectively capturing discourse level phenomena. Existing approaches rely on heuristic rules to segment documents into discourse units, which rarely align with the true discourse structure required for accurate translation. Otherwise, they fail to maintain consistency throughout the document during translation. To address these challenges, we propose Graph Augmented Agentic Framework for Document Level Translation (GRAFT), a novel graph based DocMT system that leverages Large Language Model (LLM) agents for document translation. Our approach integrates segmentation, directed acyclic graph (DAG) based dependency modelling, and discourse aware translation into a cohesive framework. Experiments conducted across eight translation directions and six diverse domains demonstrate that GRAFT achieves significant performance gains over state of the art DocMT systems. Specifically, GRAFT delivers an average improvement of 2.8 d BLEU on the TED test sets from IWSLT2017 over strong baselines and 2.3 d BLEU for domain specific translation from English to Chinese. Moreover, our analyses highlight the consistent ability of GRAFT to address discourse level phenomena, yielding coherent and contextually accurate translations. 

---
# SHNU Multilingual Conversational Speech Recognition System for INTERSPEECH 2025 MLC-SLM Challenge 

**Authors**: Yuxiang Mei, Yuang Zheng, Dongxing Xu, Yanhua Long  

**Link**: [PDF](https://arxiv.org/pdf/2507.03343)  

**Abstract**: This paper describes SHNU multilingual conversational speech recognition system (SHNU-mASR, team name-"maybe"), submitted to Track 1 of the INTERSPEECH 2025 MLC-SLM Challenge. Our system integrates a parallel-speech-encoder architecture with a large language model (LLM) to form a unified multilingual ASR framework. The parallel-speech-encoder consists of two pre-trained encoders, the Whisper-large-v3 encoder and mHuBERT-147 encoder. Their output embeddings are concatenated and fed into the LLM, enabling the model to leverage complementary acoustic and linguistic knowledge and achieve competitive performance. Moreover, we adopt a tri-stage training strategy to jointly update the low-rank adaptation modules and projector parameters of both the speech encoders and the LLM. In addition, we incorporate an additional language-aware prompt at the LLM input to enhance language-specific text generation. The SHNU-mASR system achieves an overall character/word error rate (CER/WER) of 11.76% on the blind evaluation set of the challenge, outperforming the official MLC-SLM baseline by 8.41 absolute CER/WER, without increasing the baseline training data. 

---
# Preserving Privacy, Increasing Accessibility, and Reducing Cost: An On-Device Artificial Intelligence Model for Medical Transcription and Note Generation 

**Authors**: Johnson Thomas, Ayush Mudgal, Wendao Liu, Nisten Tahiraj, Zeeshaan Mohammed, Dhruv Diddi  

**Link**: [PDF](https://arxiv.org/pdf/2507.03033)  

**Abstract**: Background: Clinical documentation represents a significant burden for healthcare providers, with physicians spending up to 2 hours daily on administrative tasks. Recent advances in large language models (LLMs) offer promising solutions, but privacy concerns and computational requirements limit their adoption in healthcare settings. Objective: To develop and evaluate a privacy-preserving, on-device medical transcription system using a fine-tuned Llama 3.2 1B model capable of generating structured medical notes from medical transcriptions while maintaining complete data sovereignty entirely in the browser. Methods: We fine-tuned a Llama 3.2 1B model using Parameter-Efficient Fine-Tuning (PEFT) with LoRA on 1,500 synthetic medical transcription-to-structured note pairs. The model was evaluated against the base Llama 3.2 1B on two datasets: 100 endocrinology transcripts and 140 modified ACI benchmark cases. Evaluation employed both statistical metrics (ROUGE, BERTScore, BLEURT) and LLM-as-judge assessments across multiple clinical quality dimensions. Results: The fine-tuned OnDevice model demonstrated substantial improvements over the base model. On the ACI benchmark, ROUGE-1 scores increased from 0.346 to 0.496, while BERTScore F1 improved from 0.832 to 0.866. Clinical quality assessments showed marked reduction in major hallucinations (from 85 to 35 cases) and enhanced factual correctness (2.81 to 3.54 on 5-point scale). Similar improvements were observed on the internal evaluation dataset, with composite scores increasing from 3.13 to 4.43 (+41.5%). Conclusions: Fine-tuning compact LLMs for medical transcription yields clinically meaningful improvements while enabling complete on-device browser deployment. This approach addresses key barriers to AI adoption in healthcare: privacy preservation, cost reduction, and accessibility for resource-constrained environments. 

---
# Large Language Models for Automating Clinical Data Standardization: HL7 FHIR Use Case 

**Authors**: Alvaro Riquelme, Pedro Costa, Catalina Martinez  

**Link**: [PDF](https://arxiv.org/pdf/2507.03067)  

**Abstract**: For years, semantic interoperability standards have sought to streamline the exchange of clinical data, yet their deployment remains time-consuming, resource-intensive, and technically challenging. To address this, we introduce a semi-automated approach that leverages large language models specifically GPT-4o and Llama 3.2 405b to convert structured clinical datasets into HL7 FHIR format while assessing accuracy, reliability, and security. Applying our method to the MIMIC-IV database, we combined embedding techniques, clustering algorithms, and semantic retrieval to craft prompts that guide the models in mapping each tabular field to its corresponding FHIR resource. In an initial benchmark, resource identification achieved a perfect F1-score, with GPT-4o outperforming Llama 3.2 thanks to the inclusion of FHIR resource schemas within the prompt. Under real-world conditions, accuracy dipped slightly to 94 %, but refinements to the prompting strategy restored robust mappings. Error analysis revealed occasional hallucinations of non-existent attributes and mismatches in granularity, which more detailed prompts can mitigate. Overall, our study demonstrates the feasibility of context-aware, LLM-driven transformation of clinical data into HL7 FHIR, laying the groundwork for semi-automated interoperability workflows. Future work will focus on fine-tuning models with specialized medical corpora, extending support to additional standards such as HL7 CDA and OMOP, and developing an interactive interface to enable expert validation and iterative refinement. 

---
# Counterfactual Tuning for Temporal Sensitivity Enhancement in Large Language Model-based Recommendation 

**Authors**: Yutian Liu, Zhengyi Yang, Jiancan Wu, Xiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.03047)  

**Abstract**: Recent advances have applied large language models (LLMs) to sequential recommendation, leveraging their pre-training knowledge and reasoning capabilities to provide more personalized user experiences. However, existing LLM-based methods fail to sufficiently leverage the rich temporal information inherent in users' historical interaction sequences, stemming from fundamental architectural constraints: LLMs process information through self-attention mechanisms that lack inherent sequence ordering and rely on position embeddings designed primarily for natural language rather than user interaction sequences. This limitation significantly impairs their ability to capture the evolution of user preferences over time and predict future interests accurately.
To address this critical gap, we propose Counterfactual Enhanced Temporal Framework for LLM-Based Recommendation (CETRec). CETRec is grounded in causal inference principles, which allow it to isolate and measure the specific impact of temporal information on recommendation outcomes. By conceptualizing temporal order as an independent causal factor distinct from item content, we can quantify its unique contribution through counterfactual reasoning--comparing what recommendations would be made with and without temporal information while keeping all other factors constant. This causal framing enables CETRec to design a novel counterfactual tuning objective that directly optimizes the model's temporal sensitivity, teaching LLMs to recognize both absolute timestamps and relative ordering patterns in user histories. Combined with our counterfactual tuning task derived from causal analysis, CETRec effectively enhances LLMs' awareness of both absolute order (how recently items were interacted with) and relative order (the sequential relationships between items). 

---
# CLUES: Collaborative High-Quality Data Selection for LLMs via Training Dynamics 

**Authors**: Wanru Zhao, Hongxiang Fan, Shell Xu Hu, Wangchunshu Zhou, Bofan Chen, Nicholas D. Lane  

**Link**: [PDF](https://arxiv.org/pdf/2507.03004)  

**Abstract**: Recent research has highlighted the importance of data quality in scaling large language models (LLMs). However, automated data quality control faces unique challenges in collaborative settings where sharing is not allowed directly between data silos. To tackle this issue, this paper proposes a novel data quality control technique based on the notion of data influence on the training dynamics of LLMs, that high quality data are more likely to have similar training dynamics to the anchor dataset. We then leverage the influence of the training dynamics to select high-quality data from different private domains, with centralized model updates on the server side in a collaborative training fashion by either model merging or federated learning. As for the data quality indicator, we compute the per-sample gradients with respect to the private data and the anchor dataset, and use the trace of the accumulated inner products as a measurement of data quality. In addition, we develop a quality control evaluation tailored for collaborative settings with heterogeneous domain data. Experiments show that training on the high-quality data selected by our method can often outperform other data selection methods for collaborative fine-tuning of LLMs, across diverse private domain datasets, in medical, multilingual and financial settings. Our code is released at this http URL. 

---
# A Comparative Study of Competency Question Elicitation Methods from Ontology Requirements 

**Authors**: Reham Alharbi, Valentina Tamma, Terry R. Payne, Jacopo de Berardinis  

**Link**: [PDF](https://arxiv.org/pdf/2507.02989)  

**Abstract**: Competency Questions (CQs) are pivotal in knowledge engineering, guiding the design, validation, and testing of ontologies. A number of diverse formulation approaches have been proposed in the literature, ranging from completely manual to Large Language Model (LLM) driven ones. However, attempts to characterise the outputs of these approaches and their systematic comparison are scarce. This paper presents an empirical comparative evaluation of three distinct CQ formulation approaches: manual formulation by ontology engineers, instantiation of CQ patterns, and generation using state of the art LLMs. We generate CQs using each approach from a set of requirements for cultural heritage, and assess them across different dimensions: degree of acceptability, ambiguity, relevance, readability and complexity. Our contribution is twofold: (i) the first multi-annotator dataset of CQs generated from the same source using different methods; and (ii) a systematic comparison of the characteristics of the CQs resulting from each approach. Our study shows that different CQ generation approaches have different characteristics and that LLMs can be used as a way to initially elicit CQs, however these are sensitive to the model used to generate CQs and they generally require a further refinement step before they can be used to model requirements. 

---
# Cautious Next Token Prediction 

**Authors**: Yizhou Wang, Lingzhi Zhang, Yue Bai, Mang Tik Chiu, Zhengmian Hu, Mingyuan Zhang, Qihua Dong, Yu Yin, Sohrab Amirghodsi, Yun Fu  

**Link**: [PDF](https://arxiv.org/pdf/2507.03038)  

**Abstract**: Next token prediction paradigm has been prevailing for autoregressive models in the era of LLMs. The current default sampling choice for popular LLMs is temperature scaling together with nucleus sampling to balance diversity and coherence. Nevertheless, such approach leads to inferior performance in various NLP tasks when the model is not certain about testing questions. To this end, we propose a brand new training-free decoding strategy, dubbed as Cautious Next Token Prediction (CNTP). In the decoding process, if the model has comparatively high prediction entropy at a certain step, we sample multiple trials starting from the step independently and stop when encountering any punctuation. Then we select the trial with the lowest perplexity score viewed as the most probable and reliable trial path given the model's capacity. The trial number is negatively correlated with the prediction confidence, i.e., the less confident the model is, the more trials it should sample. This is consistent with human beings' behaviour: when feeling uncertain or unconfident, one tends to think more creatively, exploring multiple thinking paths, to cautiously select the path one feels most confident about. Extensive experiments on both LLMs and MLLMs show that our proposed CNTP approach outperforms existing standard decoding strategies consistently by a clear margin. Moreover, the integration of CNTP with self consistency can further improve over vanilla self consistency. We believe our proposed CNTP has the potential to become one of the default choices for LLM decoding. Code is available at this https URL. 

---
# Evaluating Hierarchical Clinical Document Classification Using Reasoning-Based LLMs 

**Authors**: Akram Mustafa, Usman Naseem, Mostafa Rahimi Azghadi  

**Link**: [PDF](https://arxiv.org/pdf/2507.03001)  

**Abstract**: This study evaluates how well large language models (LLMs) can classify ICD-10 codes from hospital discharge summaries, a critical but error-prone task in healthcare. Using 1,500 summaries from the MIMIC-IV dataset and focusing on the 10 most frequent ICD-10 codes, the study tested 11 LLMs, including models with and without structured reasoning capabilities. Medical terms were extracted using a clinical NLP tool (cTAKES), and models were prompted in a consistent, coder-like format. None of the models achieved an F1 score above 57%, with performance dropping as code specificity increased. Reasoning-based models generally outperformed non-reasoning ones, with Gemini 2.5 Pro performing best overall. Some codes, such as those related to chronic heart disease, were classified more accurately than others. The findings suggest that while LLMs can assist human coders, they are not yet reliable enough for full automation. Future work should explore hybrid methods, domain-specific model training, and the use of structured clinical data. 

---
# RAG-R1 : Incentivize the Search and Reasoning Capabilities of LLMs through Multi-query Parallelism 

**Authors**: Zhiwen Tan, Jiaming Huang, Qintong Wu, Hongxuan Zhang, Chenyi Zhuang, Jinjie Gu  

**Link**: [PDF](https://arxiv.org/pdf/2507.02962)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities across various tasks, while they remain prone to generating hallucinated or outdated responses due to their static internal knowledge. Recent advancements in Retrieval-Augmented Generation (RAG) methods have explored enhancing models' search and reasoning capabilities through reinforcement learning (RL). Although these methods demonstrate promising results, they face challenges in training stability and encounter issues such as substantial inference time and restricted capabilities due to the single-query mode. In this paper, we propose RAG-R1, a novel training framework designed to enable LLMs to adaptively leverage internal and external knowledge during the reasoning process. We further expand the generation and retrieval processes within the framework from single-query mode to multi-query parallelism, aimed at reducing inference time and enhancing the model's capabilities. Extensive experiments on seven question-answering benchmarks demonstrate that our method outperforms the strongest baseline by up to 13.2% and decreases inference time by 11.1%. 

---
# `For Argument's Sake, Show Me How to Harm Myself!': Jailbreaking LLMs in Suicide and Self-Harm Contexts 

**Authors**: Annika M Schoene, Cansu Canca  

**Link**: [PDF](https://arxiv.org/pdf/2507.02990)  

**Abstract**: Recent advances in large language models (LLMs) have led to increasingly sophisticated safety protocols and features designed to prevent harmful, unethical, or unauthorized outputs. However, these guardrails remain susceptible to novel and creative forms of adversarial prompting, including manually generated test cases. In this work, we present two new test cases in mental health for (i) suicide and (ii) self-harm, using multi-step, prompt-level jailbreaking and bypass built-in content and safety filters. We show that user intent is disregarded, leading to the generation of detailed harmful content and instructions that could cause real-world harm. We conduct an empirical evaluation across six widely available LLMs, demonstrating the generalizability and reliability of the bypass. We assess these findings and the multilayered ethical tensions that they present for their implications on prompt-response filtering and context- and task-specific model development. We recommend a more comprehensive and systematic approach to AI safety and ethics while emphasizing the need for continuous adversarial testing in safety-critical AI deployments. We also argue that while certain clearly defined safety measures and guardrails can and must be implemented in LLMs, ensuring robust and comprehensive safety across all use cases and domains remains extremely challenging given the current technical maturity of general-purpose LLMs. 

---
# RADIANT: Retrieval AugmenteD entIty-context AligNmenT -- Introducing RAG-ability and Entity-Context Divergence 

**Authors**: Vipula Rawte, Rajarshi Roy, Gurpreet Singh, Danush Khanna, Yaswanth Narsupalli, Basab Ghosh, Abhay Gupta, Argha Kamal Samanta, Aditya Shingote, Aadi Krishna Vikram, Vinija Jain, Aman Chadha, Amit Sheth, Amitava Das  

**Link**: [PDF](https://arxiv.org/pdf/2507.02949)  

**Abstract**: As Large Language Models (LLMs) continue to advance, Retrieval-Augmented Generation (RAG) has emerged as a vital technique to enhance factual accuracy by integrating external knowledge into the generation process. However, LLMs often fail to faithfully integrate retrieved evidence into their generated responses, leading to factual inconsistencies. To quantify this gap, we introduce Entity-Context Divergence (ECD), a metric that measures the extent to which retrieved information is accurately reflected in model outputs. We systematically evaluate contemporary LLMs on their ability to preserve factual consistency in retrieval-augmented settings, a capability we define as RAG-ability. Our empirical analysis reveals that RAG-ability remains low across most LLMs, highlighting significant challenges in entity retention and context fidelity. This paper introduces Radiant (Retrieval AugmenteD entIty-context AligNmenT), a novel framework that merges RAG with alignment designed to optimize the interplay between retrieved evidence and generated content. Radiant extends Direct Preference Optimization (DPO) to teach LLMs how to integrate provided additional information into subsequent generations. As a behavior correction mechanism, Radiant boosts RAG performance across varied retrieval scenarios, such as noisy web contexts, knowledge conflicts, and hallucination reduction. This enables more reliable, contextually grounded, and factually coherent content generation. 

---
# Advanced Financial Reasoning at Scale: A Comprehensive Evaluation of Large Language Models on CFA Level III 

**Authors**: Pranam Shetty, Abhisek Upadhayaya, Parth Mitesh Shah, Srikanth Jagabathula, Shilpi Nayak, Anna Joo Fee  

**Link**: [PDF](https://arxiv.org/pdf/2507.02954)  

**Abstract**: As financial institutions increasingly adopt Large Language Models (LLMs), rigorous domain-specific evaluation becomes critical for responsible deployment. This paper presents a comprehensive benchmark evaluating 23 state-of-the-art LLMs on the Chartered Financial Analyst (CFA) Level III exam - the gold standard for advanced financial reasoning. We assess both multiple-choice questions (MCQs) and essay-style responses using multiple prompting strategies including Chain-of-Thought and Self-Discover. Our evaluation reveals that leading models demonstrate strong capabilities, with composite scores such as 79.1% (o4-mini) and 77.3% (Gemini 2.5 Flash) on CFA Level III. These results, achieved under a revised, stricter essay grading methodology, indicate significant progress in LLM capabilities for high-stakes financial applications. Our findings provide crucial guidance for practitioners on model selection and highlight remaining challenges in cost-effective deployment and the need for nuanced interpretation of performance against professional benchmarks. 

---
# The Application of Large Language Models on Major Depressive Disorder Support Based on African Natural Products 

**Authors**: Linyan Zou  

**Link**: [PDF](https://arxiv.org/pdf/2507.02947)  

**Abstract**: Major depressive disorder represents one of the most significant global health challenges of the 21st century, affecting millions of people worldwide and creating substantial economic and social burdens. While conventional antidepressant therapies have provided relief for many individuals, their limitations including delayed onset of action, significant side effects, and treatment resistance in a substantial portion of patients have prompted researchers and healthcare providers to explore alternative therapeutic approaches (Kasneci et al.). African traditional medicine, with its rich heritage of plant-based remedies developed over millennia, offers a valuable resource for developing novel antidepressant treatments that may address some of these limitations. This paper examines the integration of large language models with African natural products for depression support, combining traditional knowledge with modern artificial intelligence technology to create accessible, evidence-based mental health support systems.
The research presented here encompasses a comprehensive analysis of African medicinal plants with documented antidepressant properties, their pharmacological mechanisms, and the development of an AI-powered support system that leverages DeepSeek's advanced language model capabilities. The system provides evidence-based information about African herbal medicines, their clinical applications, safety considerations, and therapeutic protocols while maintaining scientific rigor and appropriate safety standards. Our findings demonstrate the potential for large language models to serve as bridges between traditional knowledge and modern healthcare, offering personalized, culturally appropriate depression support that honors both traditional wisdom and contemporary medical understanding. 

---
# A Large Language Model-Empowered Agent for Reliable and Robust Structural Analysis 

**Authors**: Jiachen Liu, Ziheng Geng, Ran Cao, Lu Cheng, Paolo Bocchini, Minghui Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2507.02938)  

**Abstract**: Large language models (LLMs) have exhibited remarkable capabilities across diverse open-domain tasks, yet their application in specialized domains such as civil engineering remains largely unexplored. This paper starts bridging this gap by evaluating and enhancing the reliability and robustness of LLMs in structural analysis of beams. Reliability is assessed through the accuracy of correct outputs under repetitive runs of the same problems, whereas robustness is evaluated via the performance across varying load and boundary conditions. A benchmark dataset, comprising eight beam analysis problems, is created to test the Llama-3.3 70B Instruct model. Results show that, despite a qualitative understanding of structural mechanics, the LLM lacks the quantitative reliability and robustness for engineering applications. To address these limitations, a shift is proposed that reframes the structural analysis as code generation tasks. Accordingly, an LLM-empowered agent is developed that (a) integrates chain-of-thought and few-shot prompting to generate accurate OpeeSeesPy code, and (b) automatically executes the code to produce structural analysis results. Experimental results demonstrate that the agent achieves accuracy exceeding 99.0% on the benchmark dataset, exhibiting reliable and robust performance across diverse conditions. Ablation studies highlight the complete example and function usage examples as the primary contributors to the agent's enhanced performance. 

---
# We Need Knowledge Distillation for Solving Math Word Problems 

**Authors**: Zhenquan Shen, Xinguo Yu, Xiaotian Cheng, Rao Peng, Hao Ming  

**Link**: [PDF](https://arxiv.org/pdf/2507.02982)  

**Abstract**: The enhancement of mathematical capabilities in large language models (LLMs) fosters new developments in mathematics education within primary and secondary schools, particularly as they relate to intelligent tutoring systems. However, LLMs require substantial computational resources, resulting in significant costs in educational contexts. To mitigate this drawback, this paper investigates the feasibility of compressing LLMs for solving math word problems (MWPs). We compress the embedded vectors encoded by BERT and distill a considerably smaller student model. Our findings indicate that the student model can maintain nearly 90% of the performance of the teacher model while utilizing only 1/12 of its parameters. In addition to achieving high accuracy, the model exhibits strong generalizability, as the compressed vectors perform well across all tasks related to MWPs, and the distillation process is not task-specific. The success of this distillation demonstrates that the underlying principles are generic and not limited to a specific task. We further explore the reasons behind the compressibility of embedded vectors, revealing that part-of-speech information, rather than entity recognition, is crucial for MWPs, which may significantly contribute to their compressibility. The improvements in efficiency and cost reduction provide substantial value for intelligent tutoring systems and significantly advance the field of intelligent education. 

---
# Mitigating Hidden Confounding by Progressive Confounder Imputation via Large Language Models 

**Authors**: Hao Yang, Haoxuan Li, Luyu Chen, Haoxiang Wang, Xu Chen, Mingming Gong  

**Link**: [PDF](https://arxiv.org/pdf/2507.02928)  

**Abstract**: Hidden confounding remains a central challenge in estimating treatment effects from observational data, as unobserved variables can lead to biased causal estimates. While recent work has explored the use of large language models (LLMs) for causal inference, most approaches still rely on the unconfoundedness assumption. In this paper, we make the first attempt to mitigate hidden confounding using LLMs. We propose ProCI (Progressive Confounder Imputation), a framework that elicits the semantic and world knowledge of LLMs to iteratively generate, impute, and validate hidden confounders. ProCI leverages two key capabilities of LLMs: their strong semantic reasoning ability, which enables the discovery of plausible confounders from both structured and unstructured inputs, and their embedded world knowledge, which supports counterfactual reasoning under latent confounding. To improve robustness, ProCI adopts a distributional reasoning strategy instead of direct value imputation to prevent the collapsed outputs. Extensive experiments demonstrate that ProCI uncovers meaningful confounders and significantly improves treatment effect estimation across various datasets and LLMs. 

---
# From Answers to Rationales: Self-Aligning Multimodal Reasoning with Answer-Oriented Chain-of-Thought 

**Authors**: Wentao Tan, Qiong Cao, Yibing Zhan, Chao Xue, Changxing Ding  

**Link**: [PDF](https://arxiv.org/pdf/2507.02984)  

**Abstract**: Achieving human-like reasoning capabilities in Multimodal Large Language Models (MLLMs) has long been a goal. Current methodologies primarily focus on synthesizing positive rationales, while overlooking the critical role of negative rationales in training models to discern flawed reasoning patterns. To address this gap, we propose a novel framework: \textbf{S}elf-Aligning \textbf{M}ultimodal Reasoning with \textbf{A}nswer-O\textbf{r}iented Chain-of-\textbf{T}hought (SMART). This framework enables models to utilize AoT-Oriented Chain-of-Thought (AoT) prompts to automatically generate high-quality positive and negative reasoning paths, followed by self-alignment to enhance their reasoning abilities. Inspired by human strategies for solving proof-based problems, AoT uses answers as a guide to help the model extract critical visual information that links questions and answers. When provided with ground truth answers, the model produces strong positive rationales. Conversely, when correct answers are replaced with misleading alternatives, the model generates an erroneous yet compelling reasoning path, serving as a form of discriminative negative rationale. Models trained with AoT-generated data outperform those trained on manually annotated datasets, demonstrating superior reasoning capabilities. This encourages the use of improved models to generate higher-quality preference data for further optimization. Consequently, SMART establishes an iterative generation-optimization method that continually enhances the model's reasoning skills. Experiments indicate that the SMART framework significantly improves various MLLMs, regardless of model architecture, parameter size, or pre-training dataset. The code, datasets, and models will be released. 

---
# Open Vision Reasoner: Transferring Linguistic Cognitive Behavior for Visual Reasoning 

**Authors**: Yana Wei, Liang Zhao, Jianjian Sun, Kangheng Lin, Jisheng Yin, Jingcheng Hu, Yinmin Zhang, En Yu, Haoran Lv, Zejia Weng, Jia Wang, Chunrui Han, Yuang Peng, Qi Han, Zheng Ge, Xiangyu Zhang, Daxin Jiang, Vishal M. Patel  

**Link**: [PDF](https://arxiv.org/pdf/2507.05255)  

**Abstract**: The remarkable reasoning capability of large language models (LLMs) stems from cognitive behaviors that emerge through reinforcement with verifiable rewards. This work investigates how to transfer this principle to Multimodal LLMs (MLLMs) to unlock advanced visual reasoning. We introduce a two-stage paradigm built on Qwen2.5-VL-7B: a massive linguistic cold-start fine-tuning, followed by multimodal reinforcement learning (RL) spanning nearly 1,000 steps, surpassing all previous open-source efforts in scale. This pioneering work reveals three fundamental insights: 1) Behavior transfer emerges surprisingly early in cold start due to linguistic mental imagery. 2) Cold start broadly memorizes visual behaviors, while RL critically discerns and scales up effective patterns. 3) Transfer strategically favors high-utility behaviors such as visual reflection. Our resulting model, Open-Vision-Reasoner (OVR), achieves state-of-the-art performance on a suite of reasoning benchmarks, including 95.3% on MATH500, 51.8% on MathVision and 54.6% on MathVerse. We release our model, data, and training dynamics to catalyze the development of more capable, behavior-aligned multimodal reasoners. 

---
# ChatGPT is not A Man but Das Man: Representativeness and Structural Consistency of Silicon Samples Generated by Large Language Models 

**Authors**: Dai Li, Linzhuo Li, Huilian Sophie Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2507.02919)  

**Abstract**: Large language models (LLMs) in the form of chatbots like ChatGPT and Llama are increasingly proposed as "silicon samples" for simulating human opinions. This study examines this notion, arguing that LLMs may misrepresent population-level opinions. We identify two fundamental challenges: a failure in structural consistency, where response accuracy doesn't hold across demographic aggregation levels, and homogenization, an underrepresentation of minority opinions. To investigate these, we prompted ChatGPT (GPT-4) and Meta's Llama 3.1 series (8B, 70B, 405B) with questions on abortion and unauthorized immigration from the American National Election Studies (ANES) 2020. Our findings reveal significant structural inconsistencies and severe homogenization in LLM responses compared to human data. We propose an "accuracy-optimization hypothesis," suggesting homogenization stems from prioritizing modal responses. These issues challenge the validity of using LLMs, especially chatbots AI, as direct substitutes for human survey data, potentially reinforcing stereotypes and misinforming policy. 

---
# Truth, Trust, and Trouble: Medical AI on the Edge 

**Authors**: Mohammad Anas Azeez, Rafiq Ali, Ebad Shabbir, Zohaib Hasan Siddiqui, Gautam Siddharth Kashyap, Jiechao Gao, Usman Naseem  

**Link**: [PDF](https://arxiv.org/pdf/2507.02983)  

**Abstract**: Large Language Models (LLMs) hold significant promise for transforming digital health by enabling automated medical question answering. However, ensuring these models meet critical industry standards for factual accuracy, usefulness, and safety remains a challenge, especially for open-source solutions. We present a rigorous benchmarking framework using a dataset of over 1,000 health questions. We assess model performance across honesty, helpfulness, and harmlessness. Our results highlight trade-offs between factual reliability and safety among evaluated models -- Mistral-7B, BioMistral-7B-DARE, and AlpaCare-13B. AlpaCare-13B achieves the highest accuracy (91.7%) and harmlessness (0.92), while domain-specific tuning in BioMistral-7B-DARE boosts safety (0.90) despite its smaller scale. Few-shot prompting improves accuracy from 78% to 85%, and all models show reduced helpfulness on complex queries, highlighting ongoing challenges in clinical QA. 

---
# Less Data, More Security: Advancing Cybersecurity LLMs Specialization via Resource-Efficient Domain-Adaptive Continuous Pre-training with Minimal Tokens 

**Authors**: Salahuddin Salahuddin, Ahmed Hussain, Jussi Löppönen, Toni Jutila, Panos Papadimitratos  

**Link**: [PDF](https://arxiv.org/pdf/2507.02964)  

**Abstract**: While Large Language Models (LLMs) demonstrate exceptional natural language capabilities, general-purpose models lack specialized domain knowledge for effective cybersecurity analysis. In this work, we investigate Domain-Adaptive Continuous Pretraining (DAP) as a methodology for enhancing cybersecurity understanding in pretrained LLMs while preserving general language capabilities. We systematically adapted three decoder-based architectures -- Llama-3.1-8B, DeepSeek-R1-Distill-Qwen-14B, and Llama-3.3-70B-Instruct -- using a curated 126-million-word cybersecurity corpus from standards, academic literature, and various other sources. Our approach employed constrained training parameters and distributed FSDP training to balance domain specialization with knowledge preservation. Evaluation across three cybersecurity benchmarks, namely, CTI-MCQ, CyberMetric, and SecEval, demonstrates consistent improvements post-adaptation. The Llama-3.3-70B-Ins-DAP model achieved state-of-the-art accuracies of 0.718, 0.933, and 0.864, respectively, outperforming specialized models, including Llama-Primus-Base. Notably, competitive performance was achieved using substantially smaller datasets (118.8 million versus 2.77 billion tokens), demonstrating efficient domain specialization viability. We establish that targeted continuous pretraining enables effective cybersecurity domain adaptation with computational feasibility, providing foundations for specialized AI assistants in threat analysis, vulnerability assessment, and security documentation while challenging prevailing assumptions about data requirements for LLM specialization. 

---
# Loki's Dance of Illusions: A Comprehensive Survey of Hallucination in Large Language Models 

**Authors**: Chaozhuo Li, Pengbo Wang, Chenxu Wang, Litian Zhang, Zheng Liu, Qiwei Ye, Yuanbo Xu, Feiran Huang, Xi Zhang, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2507.02870)  

**Abstract**: Edgar Allan Poe noted, "Truth often lurks in the shadow of error," highlighting the deep complexity intrinsic to the interplay between truth and falsehood, notably under conditions of cognitive and informational asymmetry. This dynamic is strikingly evident in large language models (LLMs). Despite their impressive linguistic generation capabilities, LLMs sometimes produce information that appears factually accurate but is, in reality, fabricated, an issue often referred to as 'hallucinations'. The prevalence of these hallucinations can mislead users, affecting their judgments and decisions. In sectors such as finance, law, and healthcare, such misinformation risks causing substantial economic losses, legal disputes, and health risks, with wide-ranging this http URL our research, we have methodically categorized, analyzed the causes, detection methods, and solutions related to LLM hallucinations. Our efforts have particularly focused on understanding the roots of hallucinations and evaluating the efficacy of current strategies in revealing the underlying logic, thereby paving the way for the development of innovative and potent approaches. By examining why certain measures are effective against hallucinations, our study aims to foster a comprehensive approach to tackling this issue within the domain of LLMs. 

---
# Evaluating AI Counseling in Japanese: Counselor, Client, and Evaluator Roles Assessed by Motivational Interviewing Criteria 

**Authors**: Keita Kiuchi, Yoshikazu Fujimoto, Hideyuki Goto, Tomonori Hosokawa, Makoto Nishimura, Yosuke Sato, Izumi Sezai  

**Link**: [PDF](https://arxiv.org/pdf/2507.02950)  

**Abstract**: This study provides the first comprehensive evaluation of large language model (LLM) performance across three counseling roles in Japanese-language therapeutic contexts. We simultaneously assessed counselor artificial intelligence (AI) systems (GPT-4-turbo with zeroshot prompting or Structured Multi-step Dialogue Prompts (SMDP), Claude-3-Opus-SMDP), client AI simulations, and evaluation AI systems (o3, Claude-3.7-Sonnet, Gemini-2.5-pro). Human experts (n = 15) with extensive counseling experience evaluated AI-generated dialogues using the Motivational Interviewing Treatment Integrity (MITI) Coding Manual 4.2.1.
Notably, SMDP implementation significantly enhanced counselor AI performance across all MITI global ratings compared with zeroshot prompting, with no significant differences between GPT-SMDP and Opus-SMDP. Evaluation AIs showed comparable performance to human raters for Cultivating Change Talk but systematically overestimated Softening Sustain Talk and the overall quality metrics. Model-specific biases emerged: Gemini emphasized power-sharing, o3 focused on technical proficiency, and Sonnet prioritized emotional expression. Client AI simulations exhibited a limited emotional range and unnaturally high compliance, indicating the need for enhanced realism.
These findings establish benchmarks for AI-assisted counseling in non-English contexts and identify critical areas for improvement through advanced prompt engineering, retrieval-augmented generation, and targeted fine-tuning, with important implications for developing culturally sensitive AI mental health tools. 

---
# Theory of Mind in Action: The Instruction Inference Task 

**Authors**: Fardin Saad, Pradeep K. Murukannaiah, Munindar P. Singh  

**Link**: [PDF](https://arxiv.org/pdf/2507.02935)  

**Abstract**: The Theory of Mind (ToM) refers to an agent's capacity to infer the mental states of other agents. ToM is essential for effective collaboration. To assess ToM in a dynamic, goal-oriented, and collaborative environment, we introduce a novel task, Instruction Inference, in which an agent assists a principal in reaching a goal by interpreting indirect or ambiguous instructions. We present Tomcat, an LLM-based agent, designed to exhibit ToM reasoning in interpreting and responding to the principal's instructions. We implement two variants of Tomcat. One, dubbed Fs-CoT, is based on a small number of examples (i.e., few-shot or Fs) demonstrating the requisite structured reasoning (i.e., chain-of-thought or CoT). One, dubbed CP, relies on commonsense knowledge and information about the problem (i.e., commonsense prompt or CP). We realized both variants of Tomcat on three leading large language models (LLMs), namely, GPT-4o, DeepSeek-R1, and Gemma-3-27B. To evaluate the effectiveness of Tomcat, we conducted a study with 52 human participants in which we provided participants with the same information as the CP variant of Tomcat. We computed intent accuracy, action optimality, and planning optimality to measure the ToM capabilities of Tomcat and our study participants. We found that Tomcat with Fs-CoT, particularly with GPT-4o and DeepSeek-R1, achieves performance comparable to the human participants, underscoring its ToM potential for human-AI collaboration. 

---
# Can Video LLMs Refuse to Answer? Alignment for Answerability in Video Large Language Models 

**Authors**: Eunseop Yoon, Hee Suk Yoon, Mark A. Hasegawa-Johnson, Chang D. Yoo  

**Link**: [PDF](https://arxiv.org/pdf/2507.04976)  

**Abstract**: In the broader context of deep learning, Multimodal Large Language Models have achieved significant breakthroughs by leveraging powerful Large Language Models as a backbone to align different modalities into the language space. A prime exemplification is the development of Video Large Language Models (Video-LLMs). While numerous advancements have been proposed to enhance the video understanding capabilities of these models, they are predominantly trained on questions generated directly from video content. However, in real-world scenarios, users often pose questions that extend beyond the informational scope of the video, highlighting the need for Video-LLMs to assess the relevance of the question. We demonstrate that even the best-performing Video-LLMs fail to reject unfit questions-not necessarily due to a lack of video understanding, but because they have not been trained to identify and refuse such questions. To address this limitation, we propose alignment for answerability, a framework that equips Video-LLMs with the ability to evaluate the relevance of a question based on the input video and appropriately decline to answer when the question exceeds the scope of the video, as well as an evaluation framework with a comprehensive set of metrics designed to measure model behavior before and after alignment. Furthermore, we present a pipeline for creating a dataset specifically tailored for alignment for answerability, leveraging existing video-description paired datasets. 

---
# MARBLE: A Multi-Agent Rule-Based LLM Reasoning Engine for Accident Severity Prediction 

**Authors**: Kaleem Ullah Qasim, Jiashu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.04893)  

**Abstract**: Accident severity prediction plays a critical role in transportation safety systems but is a persistently difficult task due to incomplete data, strong feature dependencies, and severe class imbalance in which rare but high-severity cases are underrepresented and hard to detect. Existing methods often rely on monolithic models or black box prompting, which struggle to scale in noisy, real-world settings and offer limited interpretability. To address these challenges, we propose MARBLE a multiagent rule based LLM engine that decomposes the severity prediction task across a team of specialized reasoning agents, including an interchangeable ML-backed agent. Each agent focuses on a semantic subset of features (e.g., spatial, environmental, temporal), enabling scoped reasoning and modular prompting without the risk of prompt saturation. Predictions are coordinated through either rule-based or LLM-guided consensus mechanisms that account for class rarity and confidence dynamics. The system retains structured traces of agent-level reasoning and coordination outcomes, supporting in-depth interpretability and post-hoc performance diagnostics. Across both UK and US datasets, MARBLE consistently outperforms traditional machine learning classifiers and state-of-the-art (SOTA) prompt-based reasoning methods including Chain-of-Thought (CoT), Least-to-Most (L2M), and Tree-of-Thought (ToT) achieving nearly 90% accuracy where others plateau below 48%. This performance redefines the practical ceiling for accident severity classification under real world noise and extreme class imbalance. Our results position MARBLE as a generalizable and interpretable framework for reasoning under uncertainty in safety-critical applications. 

---
# ReLoop: "Seeing Twice and Thinking Backwards" via Closed-loop Training to Mitigate Hallucinations in Multimodal understanding 

**Authors**: Jianjiang Yang, Ziyan Huang, Yanshu Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.04943)  

**Abstract**: While Multimodal Large Language Models (MLLMs) have achieved remarkable progress in open-ended visual question answering, they remain vulnerable to hallucinations. These are outputs that contradict or misrepresent input semantics, posing a critical challenge to the reliability and factual consistency. Existing methods often rely on external verification or post-hoc correction, lacking an internal mechanism to validate outputs directly during training. To bridge this gap, we propose ReLoop, a unified closed-loop training framework that encourages multimodal consistency for cross-modal understanding in MLLMs. ReLoop adopts a ring-shaped structure that integrates three complementary consistency feedback mechanisms, obliging MLLMs to "seeing twice and thinking backwards". Specifically, ReLoop employs the frozen Consistency Feedback Plugin (CFP), comprising semantic reconstruction, visual description, and an attention supervision module for attention alignment. These components collectively enforce semantic reversibility, visual consistency, and interpretable attention, enabling the model to correct its outputs during training. Extensive evaluations and analyses demonstrate the effectiveness of ReLoop in reducing hallucination rates across multiple benchmarks, establishing a robust method for hallucination mitigation in MLLMs. We will release our source code and data in the camera-ready version. 

---
# From Autonomy to Agency: Agentic Vehicles for Human-Centered Mobility Systems 

**Authors**: Jiangbo Yu  

**Link**: [PDF](https://arxiv.org/pdf/2507.04996)  

**Abstract**: Autonomy, from the Greek autos (self) and nomos (law), refers to the capacity to operate according to internal rules without external control. Accordingly, autonomous vehicles (AuVs) are defined as systems capable of perceiving their environment and executing preprogrammed tasks independently of external input. However, both research and real-world deployments increasingly showcase vehicles that demonstrate behaviors beyond this definition (including the SAE levels 1 to 6), such as interaction with humans and machines, goal adaptation, contextual reasoning, external tool use, and long-term planning, particularly with the integration of large language models (LLMs) and agentic AI systems. These developments reveal a conceptual gap between technical autonomy and the broader cognitive and social capabilities needed for future human-centered mobility systems. To address this, we introduce the concept of agentic vehicles (AgVs), referring to vehicles that integrate agentic AI to reason, adapt, and interact within complex environments. This paper presents a systems-level framework to characterize AgVs, focusing on their cognitive and communicative layers and differentiating them from conventional AuVs. It synthesizes relevant advances in agentic AI, robotics, multi-agent systems, and human-machine interaction, and highlights how agentic AI, through high-level reasoning and tool use, can function not merely as computational tools but as interactive agents embedded in mobility ecosystems. The paper concludes by identifying key challenges in the development and governance of AgVs, including safety, real-time control, public acceptance, ethical alignment, and regulatory frameworks. 

---
# DOTResize: Reducing LLM Width via Discrete Optimal Transport-based Neuron Merging 

**Authors**: Neha Verma, Kenton Murray, Kevin Duh  

**Link**: [PDF](https://arxiv.org/pdf/2507.04517)  

**Abstract**: Model compression offers a promising path to reducing the cost and inaccessibility of large pre-trained models, without significantly compromising their impressive performance. Large Transformer models, including large language models (LLMs), often contain computational redundancy, which can serve as a target for new model compression methods. In this work, we specifically target neuron-level redundancies in model layers by combining groups of similar neurons into fewer neurons. We frame this width reduction as a Discrete Optimal Transport problem, and propose DOTResize, a novel Transformer compression method that uses optimal transport theory to transform and compress model weights. To ensure applicability within the Transformer architecture, we motivate and incorporate entropic regularization and matrix factorization into the transportation maps produced by our method. Unlike pruning-based approaches which discard neurons based on importance measures, DOTResize re-projects the entire neuron width, allowing the retention and redistribution of useful signal across the reduced layer. Empirical results show that compared to simple or state-of-the-art neuron width-pruning techniques, DOTResize can outperform these methods across multiple LLM families and sizes, while achieving measurable reductions in real-world computational cost. 

---
# Evaluating LLMs on Real-World Forecasting Against Human Superforecasters 

**Authors**: Janna Lu  

**Link**: [PDF](https://arxiv.org/pdf/2507.04562)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable capabilities across diverse tasks, but their ability to forecast future events remains understudied. A year ago, large language models struggle to come close to the accuracy of a human crowd. I evaluate state-of-the-art LLMs on 464 forecasting questions from Metaculus, comparing their performance against human superforecasters. Frontier models achieve Brier scores that ostensibly surpass the human crowd but still significantly underperform a group of superforecasters. 

---
# The role of large language models in UI/UX design: A systematic literature review 

**Authors**: Ammar Ahmed, Ali Shariq Imran  

**Link**: [PDF](https://arxiv.org/pdf/2507.04469)  

**Abstract**: This systematic literature review examines the role of large language models (LLMs) in UI/UX design, synthesizing findings from 38 peer-reviewed studies published between 2022 and 2025. We identify key LLMs in use, including GPT-4, Gemini, and PaLM, and map their integration across the design lifecycle, from ideation to evaluation. Common practices include prompt engineering, human-in-the-loop workflows, and multimodal input. While LLMs are reshaping design processes, challenges such as hallucination, prompt instability, and limited explainability persist. Our findings highlight LLMs as emerging collaborators in design, and we propose directions for the ethical, inclusive, and effective integration of these technologies. 

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
# Computed Tomography Visual Question Answering with Cross-modal Feature Graphing 

**Authors**: Yuanhe Tian, Chen Su, Junwen Duan, Yan Song  

**Link**: [PDF](https://arxiv.org/pdf/2507.04333)  

**Abstract**: Visual question answering (VQA) in medical imaging aims to support clinical diagnosis by automatically interpreting complex imaging data in response to natural language queries. Existing studies typically rely on distinct visual and textual encoders to independently extract features from medical images and clinical questions, which are subsequently combined to generate answers. Specifically, in computed tomography (CT), such approaches are similar to the conventional practices in medical image analysis. However, these approaches pay less attention to the spatial continuity and inter-slice correlations in the volumetric CT data, leading to fragmented and imprecise responses. In this paper, we propose a novel large language model (LLM)-based framework enhanced by a graph representation of salient features. Different from conventional multimodal encoding strategies, our approach constructs a cross-modal graph integrating both visual and textual features, treating individual CT slices and question tokens as nodes within the graph. We further leverage an attentive graph convolutional network to dynamically fuse information within this structure. The resulting aggregated graph features then serve as a soft prompt to guide a large language model in generating accurate answers. Extensive experiments on the M3D-VQA benchmark demonstrate that our approach consistently outperforms baselines across multiple evaluation metrics, offering more robust reasoning capabilities. 

---
# ABench-Physics: Benchmarking Physical Reasoning in LLMs via High-Difficulty and Dynamic Physics Problems 

**Authors**: Yiming Zhang, Yingfan Ma, Yanmei Gu, Zhengkai Yang, Yihong Zhuang, Feng Wang, Zenan Huang, Yuanyuan Wang, Chao Huang, Bowen Song, Cheng Lin, Junbo Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2507.04766)  

**Abstract**: Large Language Models (LLMs) have shown impressive performance in domains such as mathematics and programming, yet their capabilities in physics remain underexplored and poorly understood. Physics poses unique challenges that demand not only precise computation but also deep conceptual understanding and physical modeling skills. Existing benchmarks often fall short due to limited difficulty, multiple-choice formats, and static evaluation settings that fail to capture physical modeling ability. In this paper, we introduce ABench-Physics, a novel benchmark designed to rigorously evaluate LLMs' physical reasoning and generalization capabilities. ABench-Physics consists of two components: Phy_A, a static set of 400 graduate- or Olympiad-level problems; and Phy_B, a dynamic subset of 100 problems equipped with an automatic variation engine to test model robustness across changing conditions. All questions require precise numerical answers, with strict formatting and tolerance constraints. Our evaluation of several state-of-the-art LLMs reveals substantial performance gaps, highlighting persistent limitations in physical reasoning, especially in generalization to dynamic variants. ABench-Physics provides a challenging and diagnostic framework for advancing scientific reasoning in LLMs. 

---
# LearnLens: LLM-Enabled Personalised, Curriculum-Grounded Feedback with Educators in the Loop 

**Authors**: Runcong Zhao, Artem Borov, Jiazheng Li, Yulan He  

**Link**: [PDF](https://arxiv.org/pdf/2507.04295)  

**Abstract**: Effective feedback is essential for student learning but is time-intensive for teachers. We present LearnLens, a modular, LLM-based system that generates personalised, curriculum-aligned feedback in science education. LearnLens comprises three components: (1) an error-aware assessment module that captures nuanced reasoning errors; (2) a curriculum-grounded generation module that uses a structured, topic-linked memory chain rather than traditional similarity-based retrieval, improving relevance and reducing noise; and (3) an educator-in-the-loop interface for customisation and oversight. LearnLens addresses key challenges in existing systems, offering scalable, high-quality feedback that empowers both teachers and students. 

---
# Attention Slipping: A Mechanistic Understanding of Jailbreak Attacks and Defenses in LLMs 

**Authors**: Xiaomeng Hu, Pin-Yu Chen, Tsung-Yi Ho  

**Link**: [PDF](https://arxiv.org/pdf/2507.04365)  

**Abstract**: As large language models (LLMs) become more integral to society and technology, ensuring their safety becomes essential. Jailbreak attacks exploit vulnerabilities to bypass safety guardrails, posing a significant threat. However, the mechanisms enabling these attacks are not well understood. In this paper, we reveal a universal phenomenon that occurs during jailbreak attacks: Attention Slipping. During this phenomenon, the model gradually reduces the attention it allocates to unsafe requests in a user query during the attack process, ultimately causing a jailbreak. We show Attention Slipping is consistent across various jailbreak methods, including gradient-based token replacement, prompt-level template refinement, and in-context learning. Additionally, we evaluate two defenses based on query perturbation, Token Highlighter and SmoothLLM, and find they indirectly mitigate Attention Slipping, with their effectiveness positively correlated with the degree of mitigation achieved. Inspired by this finding, we propose Attention Sharpening, a new defense that directly counters Attention Slipping by sharpening the attention score distribution using temperature scaling. Experiments on four leading LLMs (Gemma2-9B-It, Llama3.1-8B-It, Qwen2.5-7B-It, Mistral-7B-It v0.2) show that our method effectively resists various jailbreak attacks while maintaining performance on benign tasks on AlpacaEval. Importantly, Attention Sharpening introduces no additional computational or memory overhead, making it an efficient and practical solution for real-world deployment. 

---
# Agent-Based Detection and Resolution of Incompleteness and Ambiguity in Interactions with Large Language Models 

**Authors**: Riya Naik, Ashwin Srinivasan, Swati Agarwal, Estrid He  

**Link**: [PDF](https://arxiv.org/pdf/2507.03726)  

**Abstract**: Many of us now treat LLMs as modern-day oracles asking it almost any kind of question. However, consulting an LLM does not have to be a single turn activity. But long multi-turn interactions can get tedious if it is simply to clarify contextual information that can be arrived at through reasoning. In this paper, we examine the use of agent-based architecture to bolster LLM-based Question-Answering systems with additional reasoning capabilities. We examine the automatic resolution of potential incompleteness or ambiguities in questions by transducers implemented using LLM-based agents. We focus on several benchmark datasets that are known to contain questions with these deficiencies to varying degrees. We equip different LLMs (GPT-3.5-Turbo and Llama-4-Scout) with agents that act as specialists in detecting and resolving deficiencies of incompleteness and ambiguity. The agents are implemented as zero-shot ReAct agents. Rather than producing an answer in a single step, the model now decides between 3 actions a) classify b) resolve c) answer. Action a) decides if the question is incomplete, ambiguous, or normal. Action b) determines if any deficiencies identified can be resolved. Action c) answers the resolved form of the question. We compare the use of LLMs with and without the use of agents with these components. Our results show benefits of agents with transducer 1) A shortening of the length of interactions with human 2) An improvement in the answer quality and 3) Explainable resolution of deficiencies in the question. On the negative side we find while it may result in additional LLM invocations and in some cases, increased latency. But on tested datasets, the benefits outweigh the costs except when questions already have sufficient context. Suggesting the agent-based approach could be a useful mechanism to harness the power of LLMs to develop more robust QA systems. 

---
# A Comparative Study of Specialized LLMs as Dense Retrievers 

**Authors**: Hengran Zhang, Keping Bi, Jiafeng Guo  

**Link**: [PDF](https://arxiv.org/pdf/2507.03958)  

**Abstract**: While large language models (LLMs) are increasingly deployed as dense retrievers, the impact of their domain-specific specialization on retrieval effectiveness remains underexplored. This investigation systematically examines how task-specific adaptations in LLMs influence their retrieval capabilities, an essential step toward developing unified retrievers capable of handling text, code, images, and multimodal content. We conduct extensive experiments with eight Qwen2.5 7B LLMs, including base, instruction-tuned, code/math-specialized, long reasoning, and vision-language models across zero-shot retrieval settings and the supervised setting. For the zero-shot retrieval settings, we consider text retrieval from the BEIR benchmark and code retrieval from the CoIR benchmark. Further, to evaluate supervised performance, all LLMs are fine-tuned on the MS MARCO dataset. We find that mathematical specialization and the long reasoning capability cause consistent degradation in three settings, indicating conflicts between mathematical reasoning and semantic matching. The vision-language model and code-specialized LLMs demonstrate superior zero-shot performance compared to other LLMs, even surpassing BM25 on the code retrieval task, and maintain comparable performance to base LLMs in supervised settings. These findings suggest promising directions for the unified retrieval task leveraging cross-domain and cross-modal fusion. 

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
# An HTR-LLM Workflow for High-Accuracy Transcription and Analysis of Abbreviated Latin Court Hand 

**Authors**: Joshua D. Isom  

**Link**: [PDF](https://arxiv.org/pdf/2507.04132)  

**Abstract**: This article presents and validates an ideal, four-stage workflow for the high-accuracy transcription and analysis of challenging medieval legal documents. The process begins with a specialized Handwritten Text Recognition (HTR) model, itself created using a novel "Clean Ground Truth" curation method where a Large Language Model (LLM) refines the training data. This HTR model provides a robust baseline transcription (Stage 1). In Stage 2, this baseline is fed, along with the original document image, to an LLM for multimodal post-correction, grounding the LLM's analysis and improving accuracy. The corrected, abbreviated text is then expanded into full, scholarly Latin using a prompt-guided LLM (Stage 3). A final LLM pass performs Named-Entity Correction (NEC), regularizing proper nouns and generating plausible alternatives for ambiguous readings (Stage 4). We validate this workflow through detailed case studies, achieving Word Error Rates (WER) in the range of 2-7% against scholarly ground truths. The results demonstrate that this hybrid, multi-stage approach effectively automates the most laborious aspects of transcription while producing a high-quality, analyzable output, representing a powerful and practical solution for the current technological landscape. 

---
# A validity-guided workflow for robust large language model research in psychology 

**Authors**: Zhicheng Lin  

**Link**: [PDF](https://arxiv.org/pdf/2507.04491)  

**Abstract**: Large language models (LLMs) are rapidly being integrated into psychological research as research tools, evaluation targets, human simulators, and cognitive models. However, recent evidence reveals severe measurement unreliability: Personality assessments collapse under factor analysis, moral preferences reverse with punctuation changes, and theory-of-mind accuracy varies widely with trivial rephrasing. These "measurement phantoms"--statistical artifacts masquerading as psychological phenomena--threaten the validity of a growing body of research. Guided by the dual-validity framework that integrates psychometrics with causal inference, we present a six-stage workflow that scales validity requirements to research ambition--using LLMs to code text requires basic reliability and accuracy, while claims about psychological properties demand comprehensive construct validation. Researchers must (1) explicitly define their research goal and corresponding validity requirements, (2) develop and validate computational instruments through psychometric testing, (3) design experiments that control for computational confounds, (4) execute protocols with transparency, (5) analyze data using methods appropriate for non-independent observations, and (6) report findings within demonstrated boundaries and use results to refine theory. We illustrate the workflow through an example of model evaluation--"LLM selfhood"--showing how systematic validation can distinguish genuine computational phenomena from measurement artifacts. By establishing validated computational instruments and transparent practices, this workflow provides a path toward building a robust empirical foundation for AI psychology research. 

---
# SmartThinker: Learning to Compress and Preserve Reasoning by Step-Level Length Control 

**Authors**: Xingyang He, Xiao Ling, Jie Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.04348)  

**Abstract**: Large reasoning models (LRMs) have exhibited remarkable reasoning capabilities through inference-time scaling, but this progress has also introduced considerable redundancy and inefficiency into their reasoning processes, resulting in substantial computational waste. Previous work has attempted to mitigate this issue by penalizing the overall length of generated samples during reinforcement learning (RL), with the goal of encouraging a more concise chains of thought. However, we observe that such global length penalty often lead to excessive compression of critical reasoning steps while preserving unnecessary details in simpler ones, yielding a suboptimal trade-off between accuracy and efficiency. To address this issue, we propose SmartThinker, a two-stage learnable framework designed to enable fine-grained control over the length of reasoning chains based on the importance of each individual step. In the first stage, SmartThinker adapts a reasoning model to a short-form reasoning mode through rejection sampling combined with supervised fine-tuning (SFT). In the second stage, SmartThinker applies Step-Level Length Control Policy Optimization (SCPO) to refine the model output distribution, which increases the proportion of length allocated to critical steps while reducing redundancy in less important ones. SCPO consists of four core components: an online importance estimator, a step-level length control reward function, a step-level generalized advantage estimation (S-GAE) and a difficulty-adaptive clipping strategy. Working in concert, these components enable SCPO to implement differentiated length control across reasoning steps. Empirical results across multiple reasoning benchmarks and various backbone models demonstrate that SmartThinker significantly reduces redundant reasoning while achieving comparable or even superior performance to existing methods. 

---
# MateInfoUB: A Real-World Benchmark for Testing LLMs in Competitive, Multilingual, and Multimodal Educational Tasks 

**Authors**: Dumitran Adrian Marius, Theodor-Pierre Moroianu, Buca Mihnea-Vicentiu  

**Link**: [PDF](https://arxiv.org/pdf/2507.03162)  

**Abstract**: The rapid advancement of Large Language Models (LLMs) has transformed various domains, particularly computer science (CS) education. These models exhibit remarkable capabilities in code-related tasks and problem-solving, raising questions about their potential and limitations in advanced CS contexts. This study presents a novel bilingual (English-Romanian) multimodal (text and image) dataset of multiple-choice questions derived from a high-level computer science competition. A particularity of our dataset is that the problems are conceived such that some of them are easier solved using reasoning on paper, while for others writing code is more efficient. We systematically evaluate State of The Art LLMs on this dataset, analyzing their performance on theoretical programming tasks. Our findings reveal the strengths and limitations of current LLMs, including the influence of language choice (English vs. Romanian), providing insights into their applicability in CS education and competition settings. We also address critical ethical considerations surrounding educational integrity and the fairness of assessments in the context of LLM usage. These discussions aim to inform future educational practices and policies. To support further research, our dataset will be made publicly available in both English and Romanian. Additionally, we release an educational application tailored for Romanian students, enabling them to self-assess using the dataset in an interactive and practice-oriented environment. 

---
# Disambiguation-Centric Finetuning Makes Enterprise Tool-Calling LLMs More Realistic and Less Risky 

**Authors**: Ashutosh Hathidara, Julien Yu, Sebastian Schreiber  

**Link**: [PDF](https://arxiv.org/pdf/2507.03336)  

**Abstract**: Large language models (LLMs) are increasingly tasked with invoking enterprise APIs, yet they routinely falter when near-duplicate tools vie for the same user intent or when required arguments are left underspecified. We introduce DiaFORGE (Dialogue Framework for Organic Response Generation & Evaluation), a disambiguation-centric, three-stage pipeline that (i) synthesizes persona-driven, multi-turn dialogues in which the assistant must distinguish among highly similar tools, (ii) performs supervised fine-tuning of open-source models with reasoning traces across 3B - 70B parameters, and (iii) evaluates real-world readiness via a dynamic suite that redeploys each model in a live agentic loop and reports end-to-end goal completion alongside conventional static metrics. On our dynamic benchmark DiaBENCH, models trained with DiaFORGE raise tool-invocation success by 27 pp over GPT-4o and by 49 pp over Claude-3.5-Sonnet, both under optimized prompting. To spur further research, we release an open corpus of 5000 production-grade enterprise API specifications paired with rigorously validated, disambiguation-focused dialogues, offering a practical blueprint for building reliable, enterprise-ready tool-calling agents. 

---
# Intrinsic Fingerprint of LLMs: Continue Training is NOT All You Need to Steal A Model! 

**Authors**: Do-hyeon Yoon, Minsoo Chun, Thomas Allen, Hans Müller, Min Wang, Rajesh Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2507.03014)  

**Abstract**: Large language models (LLMs) face significant copyright and intellectual property challenges as the cost of training increases and model reuse becomes prevalent. While watermarking techniques have been proposed to protect model ownership, they may not be robust to continue training and development, posing serious threats to model attribution and copyright protection. This work introduces a simple yet effective approach for robust LLM fingerprinting based on intrinsic model characteristics. We discover that the standard deviation distributions of attention parameter matrices across different layers exhibit distinctive patterns that remain stable even after extensive continued training. These parameter distribution signatures serve as robust fingerprints that can reliably identify model lineage and detect potential copyright infringement. Our experimental validation across multiple model families demonstrates the effectiveness of our method for model authentication. Notably, our investigation uncovers evidence that a recently Pangu Pro MoE model released by Huawei is derived from Qwen-2.5 14B model through upcycling techniques rather than training from scratch, highlighting potential cases of model plagiarism, copyright violation, and information fabrication. These findings underscore the critical importance of developing robust fingerprinting methods for protecting intellectual property in large-scale model development and emphasize that deliberate continued training alone is insufficient to completely obscure model origins. 

---
# Large Language Model Agent for Modular Task Execution in Drug Discovery 

**Authors**: Janghoon Ock, Radheesh Sharma Meda, Srivathsan Badrinarayanan, Neha S. Aluru, Achuth Chandrasekhar, Amir Barati Farimani  

**Link**: [PDF](https://arxiv.org/pdf/2507.02925)  

**Abstract**: We present a modular framework powered by large language models (LLMs) that automates and streamlines key tasks across the early-stage computational drug discovery pipeline. By combining LLM reasoning with domain-specific tools, the framework performs biomedical data retrieval, domain-specific question answering, molecular generation, property prediction, property-aware molecular refinement, and 3D protein-ligand structure generation. In a case study targeting BCL-2 in lymphocytic leukemia, the agent autonomously retrieved relevant biomolecular information-including FASTA sequences, SMILES representations, and literature-and answered mechanistic questions with improved contextual accuracy over standard LLMs. It then generated chemically diverse seed molecules and predicted 67 ADMET-related properties, which guided iterative molecular refinement. Across two refinement rounds, the number of molecules with QED > 0.6 increased from 34 to 55, and those passing at least four out of five empirical drug-likeness rules rose from 29 to 52, within a pool of 194 molecules. The framework also employed Boltz-2 to generate 3D protein-ligand complexes and provide rapid binding affinity estimates for candidate compounds. These results demonstrate that the approach effectively supports molecular screening, prioritization, and structure evaluation. Its modular design enables flexible integration of evolving tools and models, providing a scalable foundation for AI-assisted therapeutic discovery. 

---
# Causal-SAM-LLM: Large Language Models as Causal Reasoners for Robust Medical Segmentation 

**Authors**: Tao Tang, Shijie Xu, Yiting Wu, Zhixiang Lu  

**Link**: [PDF](https://arxiv.org/pdf/2507.03585)  

**Abstract**: The clinical utility of deep learning models for medical image segmentation is severely constrained by their inability to generalize to unseen domains. This failure is often rooted in the models learning spurious correlations between anatomical content and domain-specific imaging styles. To overcome this fundamental challenge, we introduce Causal-SAM-LLM, a novel framework that elevates Large Language Models (LLMs) to the role of causal reasoners. Our framework, built upon a frozen Segment Anything Model (SAM) encoder, incorporates two synergistic innovations. First, Linguistic Adversarial Disentanglement (LAD) employs a Vision-Language Model to generate rich, textual descriptions of confounding image styles. By training the segmentation model's features to be contrastively dissimilar to these style descriptions, it learns a representation robustly purged of non-causal information. Second, Test-Time Causal Intervention (TCI) provides an interactive mechanism where an LLM interprets a clinician's natural language command to modulate the segmentation decoder's features in real-time, enabling targeted error correction. We conduct an extensive empirical evaluation on a composite benchmark from four public datasets (BTCV, CHAOS, AMOS, BraTS), assessing generalization under cross-scanner, cross-modality, and cross-anatomy settings. Causal-SAM-LLM establishes a new state of the art in out-of-distribution (OOD) robustness, improving the average Dice score by up to 6.2 points and reducing the Hausdorff Distance by 15.8 mm over the strongest baseline, all while using less than 9% of the full model's trainable parameters. Our work charts a new course for building robust, efficient, and interactively controllable medical AI systems. 

---
# Improving LLM Reasoning for Vulnerability Detection via Group Relative Policy Optimization 

**Authors**: Marco Simoni, Aleksandar Fontana, Giulio Rossolini, Andrea Saracino  

**Link**: [PDF](https://arxiv.org/pdf/2507.03051)  

**Abstract**: Improving and understanding the training dynamics and reasoning of Large Language Models (LLMs) has become essential for their deployment in AI-based security tools, such as software vulnerability detection. In this work, we present an extensive study aimed at advancing recent RL-based finetuning techniques for LLMs in the context of vulnerability detection.
We start by highlighting key limitations of commonly adopted LLMs, such as their tendency to over-predict certain types of vulnerabilities while failing to detect others. To address this challenge, we explore the use of Group Relative Policy Optimization (GRPO), a recent policy-gradient method, for guiding LLM behavior through structured, rule-based rewards. We enable its application to the vulnerability detection task by redefining its advantage functions and reward signals using annotations from widely used datasets in the field, including BigVul, DiverseVul, and CleanVul.
The proposed methodology enables an extensive set of experiments, addressing multiple research questions regarding the impact of GRPO on generalization, reasoning capabilities, and performance improvements over standard supervised finetuning (SFT). Our findings offer valuable insights into the potential of RL-based training to enhance both the performance and reasoning abilities of LLMs in the context of software vulnerability detection. 

---
# LLM-based Question-Answer Framework for Sensor-driven HVAC System Interaction 

**Authors**: Sungmin Lee, Minju Kang, Joonhee Lee, Seungyong Lee, Dongju Kim, Jingi Hong, Jun Shin, Pei Zhang, JeongGil Ko  

**Link**: [PDF](https://arxiv.org/pdf/2507.04748)  

**Abstract**: Question-answering (QA) interfaces powered by large language models (LLMs) present a promising direction for improving interactivity with HVAC system insights, particularly for non-expert users. However, enabling accurate, real-time, and context-aware interactions with HVAC systems introduces unique challenges, including the integration of frequently updated sensor data, domain-specific knowledge grounding, and coherent multi-stage reasoning. In this paper, we present JARVIS, a two-stage LLM-based QA framework tailored for sensor data-driven HVAC system interaction. JARVIS employs an Expert-LLM to translate high-level user queries into structured execution instructions, and an Agent that performs SQL-based data retrieval, statistical processing, and final response generation. To address HVAC-specific challenges, JARVIS integrates (1) an adaptive context injection strategy for efficient HVAC and deployment-specific information integration, (2) a parameterized SQL builder and executor to improve data access reliability, and (3) a bottom-up planning scheme to ensure consistency across multi-stage response generation. We evaluate JARVIS using real-world data collected from a commercial HVAC system and a ground truth QA dataset curated by HVAC experts to demonstrate its effectiveness in delivering accurate and interpretable responses across diverse queries. Results show that JARVIS consistently outperforms baseline and ablation variants in both automated and user-centered assessments, achieving high response quality and accuracy. 

---
# Activation Steering for Chain-of-Thought Compression 

**Authors**: Seyedarmin Azizi, Erfan Baghaei Potraghloo, Massoud Pedram  

**Link**: [PDF](https://arxiv.org/pdf/2507.04742)  

**Abstract**: Large language models (LLMs) excel at complex reasoning when they include intermediate steps, known as "chains of thought" (CoTs). However, these rationales are often overly verbose, even for simple problems, leading to wasted context, increased latency, and higher energy consumption. We observe that verbose, English-heavy CoTs and concise, math-centric CoTs occupy distinct regions in the model's residual-stream activation space. By extracting and injecting a "steering vector" to transition between these modes, we can reliably shift generation toward more concise reasoning, effectively compressing CoTs without retraining. We formalize this approach as Activation-Steered Compression (ASC), an inference-time technique that shortens reasoning traces by directly modifying hidden representations. In addition, we provide a theoretical analysis of the impact of ASC on the output distribution, derived from a closed-form KL-divergence-bounded constraint to regulate steering strength. Using only 100 paired verbose and concise examples, ASC achieves up to 67.43% reduction in CoT length on MATH500 and GSM8K datasets, while maintaining accuracy across 7B, 8B, and 32B parameter models. As a training-free method, ASC introduces negligible runtime overhead and, on MATH500, delivers an average 2.73x speedup in end-to-end reasoning wall-clock time on an 8B model. This makes ASC a practical and efficient tool for streamlining the deployment of reasoning-capable LLMs in latency- or cost-sensitive settings. The code is available at: this https URL 

---
# ChipSeek-R1: Generating Human-Surpassing RTL with LLM via Hierarchical Reward-Driven Reinforcement Learning 

**Authors**: Zhirong Chen, Kaiyan Chang, Zhuolin Li, Xinyang He, Chujie Chen, Cangyuan Li, Mengdi Wang, Haobo Xu, Yinhe Han, Ying Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.04736)  

**Abstract**: Large Language Models (LLMs) show significant potential for automating Register-Transfer Level (RTL) code generation. However, current approaches face a critical challenge: they can not simultaneously optimize for functional correctness and hardware quality (Power, Performance, Area - PPA). Methods based on supervised fine-tuning often generate functionally correct but PPA-suboptimal code, lacking mechanisms to learn optimization principles. In contrast, post-processing techniques that attempt to improve PPA metrics after generation are often inefficient because they operate externally without updating the LLM's parameters, thus failing to enhance the model's intrinsic design capabilities.
To bridge this gap, we introduce ChipSeek-R1, a hierarchical reward-driven reinforcement learning framework to train LLMs to generate RTL code that achieves both functional correctness and optimized PPA metrics. ChipSeek-R1 employs a hierarchical reward system, which incorporates direct feedback on syntax, functional correctness (from simulators) and PPA metrics (from synthesis tools) during reinforcement learning. This enables the model to learn complex hardware design trade-offs via trial-and-error, generating RTL code that is both functionally correct and PPA-optimized. Evaluating ChipSeek-R1 on standard benchmarks (VerilogEval, RTLLM), we achieve state-of-the-art results in functional correctness. Notably, on the RTLLM benchmark, ChipSeek-R1 generated 27 RTL designs surpassing the PPA metrics of the original human-written code. Our findings demonstrate the effectiveness of integrating toolchain feedback into LLM training and highlight the potential for reinforcement learning to enable automated generation of human-surpassing RTL code. We open-source our code in anonymous github. 

---
# DoPI: Doctor-like Proactive Interrogation LLM for Traditional Chinese Medicine 

**Authors**: Zewen Sun, Ruoxiang Huang, Jiahe Feng, Rundong Kong, Yuqian Wang, Hengyu Liu, Ziqi Gong, Yuyuan Qin, Yingxue Wang, Yu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.04877)  

**Abstract**: Enhancing interrogation capabilities in Traditional Chinese Medicine (TCM) diagnosis through multi-turn dialogues and knowledge graphs presents a significant challenge for modern AI systems. Current large language models (LLMs), despite their advancements, exhibit notable limitations in medical applications, particularly in conducting effective multi-turn dialogues and proactive questioning. These shortcomings hinder their practical application and effectiveness in simulating real-world diagnostic scenarios. To address these limitations, we propose DoPI, a novel LLM system specifically designed for the TCM domain. The DoPI system introduces a collaborative architecture comprising a guidance model and an expert model. The guidance model conducts multi-turn dialogues with patients and dynamically generates questions based on a knowledge graph to efficiently extract critical symptom information. Simultaneously, the expert model leverages deep TCM expertise to provide final diagnoses and treatment plans. Furthermore, this study constructs a multi-turn doctor-patient dialogue dataset to simulate realistic consultation scenarios and proposes a novel evaluation methodology that does not rely on manually collected real-world consultation data. Experimental results show that the DoPI system achieves an accuracy rate of 84.68 percent in interrogation outcomes, significantly enhancing the model's communication ability during diagnosis while maintaining professional expertise. 

---
# FurniMAS: Language-Guided Furniture Decoration using Multi-Agent System 

**Authors**: Toan Nguyen, Tri Le, Quang Nguyen, Anh Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2507.04770)  

**Abstract**: Furniture decoration is an important task in various industrial applications. However, achieving a high-quality decorative result is often time-consuming and requires specialized artistic expertise. To tackle these challenges, we explore how multi-agent systems can assist in automating the decoration process. We propose FurniMAS, a multi-agent system for automatic furniture decoration. Specifically, given a human prompt and a household furniture item such as a working desk or a TV stand, our system suggests relevant assets with appropriate styles and materials, and arranges them on the item, ensuring the decorative result meets functionality, aesthetic, and ambiance preferences. FurniMAS assembles a hybrid team of LLM-based and non-LLM agents, each fulfilling distinct roles in a typical decoration project. These agents collaborate through communication, logical reasoning, and validation to transform the requirements into the final outcome. Extensive experiments demonstrate that our FurniMAS significantly outperforms other baselines in generating high-quality 3D decor. 

---
# LayerCake: Token-Aware Contrastive Decoding within Large Language Model Layers 

**Authors**: Jingze Zhu, Yongliang Wu, Wenbo Zhu, Jiawang Cao, Yanqiang Zheng, Jiawei Chen, Xu Yang, Bernt Schiele, Jonas Fischer, Xinting Hu  

**Link**: [PDF](https://arxiv.org/pdf/2507.04404)  

**Abstract**: Large language models (LLMs) excel at natural language understanding and generation but remain vulnerable to factual errors, limiting their reliability in knowledge-intensive tasks. While decoding-time strategies provide a promising efficient solution without training, existing methods typically treat token-level and layer-level signals in isolation, overlooking the joint dynamics between them. In this work, we introduce a token-aware, layer-localized contrastive decoding method that aligns specific token types with their most influential transformer layers to improve factual generation. Through empirical attention analysis, we identify two key patterns: punctuation tokens receive dominant attention in early layers, while conceptual tokens govern semantic reasoning in intermediate layers. By selectively suppressing attention to these token types at their respective depths, we achieve the induction of controlled factual degradation and derive contrastive signals to guide the final factual decoding. Our method requires no additional training or model modification, and experiments demonstrate that our method consistently improves factuality across multiple LLMs and various benchmarks. 

---
# Application and Evaluation of Large Language Models for Forecasting the Impact of Traffic Incidents 

**Authors**: George Jagadeesh, Srikrishna Iyer, Michal Polanowski, Kai Xin Thia  

**Link**: [PDF](https://arxiv.org/pdf/2507.04803)  

**Abstract**: This study examines the feasibility of applying large language models (LLMs) for forecasting the impact of traffic incidents on the traffic flow. The use of LLMs for this task has several advantages over existing machine learning-based solutions such as not requiring a large training dataset and the ability to utilize free-text incident logs. We propose a fully LLM-based solution that predicts the incident impact using a combination of traffic features and LLM-extracted incident features. A key ingredient of this solution is an effective method of selecting examples for the LLM's in-context learning. We evaluate the performance of three advanced LLMs and two state-of-the-art machine learning models on a real traffic incident dataset. The results show that the best-performing LLM matches the accuracy of the most accurate machine learning model, despite the former not having been trained on this prediction task. The findings indicate that LLMs are a practically viable option for traffic incident impact prediction. 

---
# Mpemba Effect in Large-Language Model Training Dynamics: A Minimal Analysis of the Valley-River model 

**Authors**: Sibei Liu, Zhijian Hu  

**Link**: [PDF](https://arxiv.org/pdf/2507.04206)  

**Abstract**: Learning rate (LR) schedules in large language model (LLM) training often follow empirical templates: warm-up, constant plateau/stable phase, and decay (WSD). However, the mechanistic explanation for this strategy remains underexplored, and the choice of plateau height and decay schedule is largely heuristic. In this paper, we connect training dynamics to a thermodynamic analogy via the Mpemba effect - a phenomenon in which a hotter system cools faster than a colder one when quenched into the same bath. We analyze a class of "valley-river" loss landscapes, where sharp (valley) directions equilibrate quickly, while flatter (river) directions govern global descent. The Mpemba effect provides an explanation for the necessity of the warm-up phase and motivates a high plateau - rather than a low one - for accelerating loss decrease during decay. We show that for certain loss landscapes, there exists an optimal plateau learning rate - the "strong Mpemba point" - at which the slowest mode vanishes, resulting in faster convergence during the decay phase. We derive analytical conditions for its existence and estimate decay dynamics required to preserve the Mpemba advantage. Our minimal model and analysis offer a principled justification for plateau-based schedulers and provide guidance for tuning LR in LLMs with minimal hyperparameter sweep. 

---
# A Technical Survey of Reinforcement Learning Techniques for Large Language Models 

**Authors**: Saksham Sahai Srivastava, Vaneet Aggarwal  

**Link**: [PDF](https://arxiv.org/pdf/2507.04136)  

**Abstract**: Reinforcement Learning (RL) has emerged as a transformative approach for aligning and enhancing Large Language Models (LLMs), addressing critical challenges in instruction following, ethical alignment, and reasoning capabilities. This survey offers a comprehensive foundation on the integration of RL with language models, highlighting prominent algorithms such as Proximal Policy Optimization (PPO), Q-Learning, and Actor-Critic methods. Additionally, it provides an extensive technical overview of RL techniques specifically tailored for LLMs, including foundational methods like Reinforcement Learning from Human Feedback (RLHF) and AI Feedback (RLAIF), as well as advanced strategies such as Direct Preference Optimization (DPO) and Group Relative Policy Optimization (GRPO). We systematically analyze their applications across domains, i.e., from code generation to tool-augmented reasoning. We also present a comparative taxonomy based on reward modeling, feedback mechanisms, and optimization strategies. Our evaluation highlights key trends. RLHF remains dominant for alignment, and outcome-based RL such as RLVR significantly improves stepwise reasoning. However, persistent challenges such as reward hacking, computational costs, and scalable feedback collection underscore the need for continued innovation. We further discuss emerging directions, including hybrid RL algorithms, verifier-guided training, and multi-objective alignment frameworks. This survey serves as a roadmap for researchers advancing RL-driven LLM development, balancing capability enhancement with safety and scalability. 

---
# Enhancing Robustness of LLM-Driven Multi-Agent Systems through Randomized Smoothing 

**Authors**: Jinwei Hu, Yi Dong, Zhengtao Ding, Xiaowei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2507.04105)  

**Abstract**: This paper presents a defense framework for enhancing the safety of large language model (LLM) empowered multi-agent systems (MAS) in safety-critical domains such as aerospace. We apply randomized smoothing, a statistical robustness certification technique, to the MAS consensus context, enabling probabilistic guarantees on agent decisions under adversarial influence. Unlike traditional verification methods, our approach operates in black-box settings and employs a two-stage adaptive sampling mechanism to balance robustness and computational efficiency. Simulation results demonstrate that our method effectively prevents the propagation of adversarial behaviors and hallucinations while maintaining consensus performance. This work provides a practical and scalable path toward safe deployment of LLM-based MAS in real-world, high-stakes environments. 

---
# Can Prompt Difficulty be Online Predicted for Accelerating RL Finetuning of Reasoning Models? 

**Authors**: Yun Qu, Qi Cheems Wang, Yixiu Mao, Vincent Tao Hu, Xiangyang Ji  

**Link**: [PDF](https://arxiv.org/pdf/2507.04632)  

**Abstract**: Recent advances have witnessed the effectiveness of reinforcement learning (RL) finetuning in enhancing the reasoning capabilities of large language models (LLMs). The optimization process often requires numerous iterations to achieve satisfactory performance, resulting in high computational costs due to the need for frequent prompt evaluations under intensive LLM interactions and repeated policy updates. Appropriate online prompt selection methods reduce iteration steps by prioritizing informative prompts during training, while the pipeline's reliance on exhaustive prompt evaluation and subset selection for optimization still incurs substantial computational overhead due to frequent LLM inference calls. Distinguished from these direct evaluate-then-select schemes, this work investigates iterative approximate evaluation for arbitrary prompts and introduces Model Predictive Prompt Selection (MoPPS), a Bayesian risk-predictive framework that online estimates prompt difficulty without requiring costly LLM interactions. Technically, MoPPS models each prompt's success rate as a latent variable, performs streaming Bayesian inference, and employs posterior sampling in a constructed multi-armed bandit machine, enabling sample efficient and adaptive prompt selection. Extensive experiments across mathematics, planning, and vision-based geometry tasks show that MoPPS reliably predicts prompt difficulty and accelerates training with significantly reduced LLM rollouts. 

---
# How to Train Your LLM Web Agent: A Statistical Diagnosis 

**Authors**: Dheeraj Vattikonda, Santhoshi Ravichandran, Emiliano Penaloza, Hadi Nekoei, Megh Thakkar, Thibault Le Sellier de Chezelles, Nicolas Gontier, Miguel Muñoz-Mármol, Sahar Omidi Shayegan, Stefania Raimondo, Xue Liu, Alexandre Drouin, Laurent Charlin, Alexandre Piché, Alexandre Lacoste, Massimo Caccia  

**Link**: [PDF](https://arxiv.org/pdf/2507.04103)  

**Abstract**: LLM-based web agents have recently made significant progress, but much of it has occurred in closed-source systems, widening the gap with open-source alternatives. Progress has been held back by two key challenges: first, a narrow focus on single-step tasks that overlooks the complexity of multi-step web interactions; and second, the high compute costs required to post-train LLM-based web agents. To address this, we present the first statistically grounded study on compute allocation for LLM web-agent post-training. Our approach uses a two-stage pipeline, training a Llama 3.1 8B student to imitate a Llama 3.3 70B teacher via supervised fine-tuning (SFT), followed by on-policy reinforcement learning. We find this process highly sensitive to hyperparameter choices, making exhaustive sweeps impractical. To spare others from expensive trial-and-error, we sample 1,370 configurations and use bootstrapping to estimate effective hyperparameters. Our results show that combining SFT with on-policy RL consistently outperforms either approach alone on both WorkArena and MiniWob++. Further, this strategy requires only 55% of the compute to match the peak performance of pure SFT on MiniWob++, effectively pushing the compute-performance Pareto frontier, and is the only strategy that can close the gap with closed-source models. 

---
# CortexDebate: Debating Sparsely and Equally for Multi-Agent Debate 

**Authors**: Yiliu Sun, Zicheng Zhao, Sheng Wan, Chen Gong  

**Link**: [PDF](https://arxiv.org/pdf/2507.03928)  

**Abstract**: Nowadays, single Large Language Model (LLM) struggles with critical issues such as hallucination and inadequate reasoning abilities. To mitigate these issues, Multi-Agent Debate (MAD) has emerged as an effective strategy, where LLM agents engage in in-depth debates with others on tasks. However, existing MAD methods face two major issues: (a) too lengthy input contexts, which causes LLM agents to get lost in plenty of input information and experiences performance drop; and (b) the overconfidence dilemma, where self-assured LLM agents dominate the debate, leading to low debating effectiveness. To address these limitations, we propose a novel MAD method called "CortexDebate". Inspired by the human brain's tendency to establish a sparse and dynamically optimized network among cortical areas governed by white matter, CortexDebate constructs a sparse debating graph among LLM agents, where each LLM agent only debates with the ones that are helpful to it. To optimize the graph, we propose a module named McKinsey-based Debate Matter (MDM), which acts as an artificial analog to white matter. By integrating the McKinsey Trust Formula, a well-established measure of trustworthiness from sociology, MDM enables credible evaluations that guide graph optimization. The effectiveness of our CortexDebate has been well demonstrated by extensive experimental results across eight datasets from four task types. 

---
# Lyria: A General LLM-Driven Genetic Algorithm Framework for Problem Solving 

**Authors**: Weizhi Tang, Kwabena Nuamah, Vaishak Belle  

**Link**: [PDF](https://arxiv.org/pdf/2507.04034)  

**Abstract**: While Large Language Models (LLMs) have demonstrated impressive abilities across various domains, they still struggle with complex problems characterized by multi-objective optimization, precise constraint satisfaction, immense solution spaces, etc. To address the limitation, drawing on the superior semantic understanding ability of LLMs and also the outstanding global search and optimization capability of genetic algorithms, we propose to capitalize on their respective strengths and introduce Lyria, a general LLM-driven genetic algorithm framework, comprising 7 essential components. Through conducting extensive experiments with 4 LLMs across 3 types of problems, we demonstrated the efficacy of Lyria. Additionally, with 7 additional ablation experiments, we further systematically analyzed and elucidated the factors that affect its performance. 

---
# LLMs model how humans induce logically structured rules 

**Authors**: Alyssa Loo, Ellie Pavlick, Roman Feiman  

**Link**: [PDF](https://arxiv.org/pdf/2507.03876)  

**Abstract**: A central goal of cognitive science is to provide a computationally explicit account of both the structure of the mind and its development: what are the primitive representational building blocks of cognition, what are the rules via which those primitives combine, and where do these primitives and rules come from in the first place? A long-standing debate concerns the adequacy of artificial neural networks as computational models that can answer these questions, in particular in domains related to abstract cognitive function, such as language and logic. This paper argues that recent advances in neural networks -- specifically, the advent of large language models (LLMs) -- represent an important shift in this debate. We test a variety of LLMs on an existing experimental paradigm used for studying the induction of rules formulated over logical concepts. Across four experiments, we find converging empirical evidence that LLMs provide at least as good a fit to human behavior as models that implement a Bayesian probablistic language of thought (pLoT), which have been the best computational models of human behavior on the same task. Moreover, we show that the LLMs make qualitatively different predictions about the nature of the rules that are inferred and deployed in order to complete the task, indicating that the LLM is unlikely to be a mere implementation of the pLoT solution. Based on these results, we argue that LLMs may instantiate a novel theoretical account of the primitive representations and computations necessary to explain human logical concepts, with which future work in cognitive science should engage. 

---
# Toward Better Generalisation in Uncertainty Estimators: Leveraging Data-Agnostic Features 

**Authors**: Thuy An Ha, Bao Quoc Vo  

**Link**: [PDF](https://arxiv.org/pdf/2507.03998)  

**Abstract**: Large Language Models (LLMs) often generate responses that are factually incorrect yet expressed with high confidence, which can pose serious risks for end users. To address this, it is essential for LLMs not only to produce answers but also to provide accurate estimates of their correctness. Uncertainty quantification methods have been introduced to assess the quality of LLM outputs, with factual accuracy being a key aspect of that quality. Among these methods, those that leverage hidden states to train probes have shown particular promise, as these internal representations encode information relevant to the factuality of responses, making this approach the focus of this paper. However, the probe trained on the hidden states of one dataset often struggles to generalise to another dataset of a different task or domain. To address this limitation, we explore combining data-agnostic features with hidden-state features and assess whether this hybrid feature set enhances out-of-domain performance. We further examine whether selecting only the most informative hidden-state features, thereby discarding task-specific noise, enables the data-agnostic features to contribute more effectively. The experiment results indicate that although introducing data-agnostic features generally enhances generalisation performance in most cases, in certain scenarios their inclusion degrades performance. A similar pattern emerges when retaining only the most important hidden-state features - adding data-agnostic features does not consistently further enhance performance compared to using the full set of hidden-state features. A closer analysis reveals that, in some specific cases, the trained probe underweights the data-agnostic features relative to the hidden-state features, which we believe is the main reason why the results are inconclusive. 

---
# RELRaE: LLM-Based Relationship Extraction, Labelling, Refinement, and Evaluation 

**Authors**: George Hannah, Jacopo de Berardinis, Terry R. Payne, Valentina Tamma, Andrew Mitchell, Ellen Piercy, Ewan Johnson, Andrew Ng, Harry Rostron, Boris Konev  

**Link**: [PDF](https://arxiv.org/pdf/2507.03829)  

**Abstract**: A large volume of XML data is produced in experiments carried out by robots in laboratories. In order to support the interoperability of data between labs, there is a motivation to translate the XML data into a knowledge graph. A key stage of this process is the enrichment of the XML schema to lay the foundation of an ontology schema. To achieve this, we present the RELRaE framework, a framework that employs large language models in different stages to extract and accurately label the relationships implicitly present in the XML schema. We investigate the capability of LLMs to accurately generate these labels and then evaluate them. Our work demonstrates that LLMs can be effectively used to support the generation of relationship labels in the context of lab automation, and that they can play a valuable role within semi-automatic ontology generation frameworks more generally. 

---
# Towards Machine Theory of Mind with Large Language Model-Augmented Inverse Planning 

**Authors**: Rebekah A. Gelpí, Eric Xue, William A. Cunningham  

**Link**: [PDF](https://arxiv.org/pdf/2507.03682)  

**Abstract**: We propose a hybrid approach to machine Theory of Mind (ToM) that uses large language models (LLMs) as a mechanism for generating hypotheses and likelihood functions with a Bayesian inverse planning model that computes posterior probabilities for an agent's likely mental states given its actions. Bayesian inverse planning models can accurately predict human reasoning on a variety of ToM tasks, but these models are constrained in their ability to scale these predictions to scenarios with a large number of possible hypotheses and actions. Conversely, LLM-based approaches have recently demonstrated promise in solving ToM benchmarks, but can exhibit brittleness and failures on reasoning tasks even when they pass otherwise structurally identical versions. By combining these two methods, this approach leverages the strengths of each component, closely matching optimal results on a task inspired by prior inverse planning models and improving performance relative to models that utilize LLMs alone or with chain-of-thought prompting, even with smaller LLMs that typically perform poorly on ToM tasks. We also exhibit the model's potential to predict mental states on open-ended tasks, offering a promising direction for future development of ToM models and the creation of socially intelligent generative agents. 

---
# Large Language Models for Combinatorial Optimization: A Systematic Review 

**Authors**: Francesca Da Ros, Michael Soprano, Luca Di Gaspero, Kevin Roitero  

**Link**: [PDF](https://arxiv.org/pdf/2507.03637)  

**Abstract**: This systematic review explores the application of Large Language Models (LLMs) in Combinatorial Optimization (CO). We report our findings using the Preferred Reporting Items for Systematic Reviews and Meta-Analyses (PRISMA) guidelines. We conduct a literature search via Scopus and Google Scholar, examining over 2,000 publications. We assess publications against four inclusion and four exclusion criteria related to their language, research focus, publication year, and type. Eventually, we select 103 studies. We classify these studies into semantic categories and topics to provide a comprehensive overview of the field, including the tasks performed by LLMs, the architectures of LLMs, the existing datasets specifically designed for evaluating LLMs in CO, and the field of application. Finally, we identify future directions for leveraging LLMs in this field. 

---
# Leveraging Large Language Models for Tacit Knowledge Discovery in Organizational Contexts 

**Authors**: Gianlucca Zuin, Saulo Mastelini, Túlio Loures, Adriano Veloso  

**Link**: [PDF](https://arxiv.org/pdf/2507.03811)  

**Abstract**: Documenting tacit knowledge in organizations can be a challenging task due to incomplete initial information, difficulty in identifying knowledgeable individuals, the interplay of formal hierarchies and informal networks, and the need to ask the right questions. To address this, we propose an agent-based framework leveraging large language models (LLMs) to iteratively reconstruct dataset descriptions through interactions with employees. Modeling knowledge dissemination as a Susceptible-Infectious (SI) process with waning infectivity, we conduct 864 simulations across various synthetic company structures and different dissemination parameters. Our results show that the agent achieves 94.9% full-knowledge recall, with self-critical feedback scores strongly correlating with external literature critic scores. We analyze how each simulation parameter affects the knowledge retrieval process for the agent. In particular, we find that our approach is able to recover information without needing to access directly the only domain specialist. These findings highlight the agent's ability to navigate organizational complexity and capture fragmented knowledge that would otherwise remain inaccessible. 

---
# REAL: Benchmarking Abilities of Large Language Models for Housing Transactions and Services 

**Authors**: Kexin Zhu, Yang Han  

**Link**: [PDF](https://arxiv.org/pdf/2507.03477)  

**Abstract**: The development of large language models (LLMs) has greatly promoted the progress of chatbot in multiple fields. There is an urgent need to evaluate whether LLMs can play the role of agent in housing transactions and services as well as humans. We present Real Estate Agent Large Language Model Evaluation (REAL), the first evaluation suite designed to assess the abilities of LLMs in the field of housing transactions and services. REAL comprises 5,316 high-quality evaluation entries across 4 topics: memory, comprehension, reasoning and hallucination. All these entries are organized as 14 categories to assess whether LLMs have the knowledge and ability in housing transactions and services scenario. Additionally, the REAL is used to evaluate the performance of most advanced LLMs. The experiment results indicate that LLMs still have significant room for improvement to be applied in the real estate field. 

---
# Roadmap for using large language models (LLMs) to accelerate cross-disciplinary research with an example from computational biology 

**Authors**: Ruian Ke, Ruy M. Ribeiro  

**Link**: [PDF](https://arxiv.org/pdf/2507.03722)  

**Abstract**: Large language models (LLMs) are powerful artificial intelligence (AI) tools transforming how research is conducted. However, their use in research has been met with skepticism, due to concerns about hallucinations, biases and potential harms to research. These emphasize the importance of clearly understanding the strengths and weaknesses of LLMs to ensure their effective and responsible use. Here, we present a roadmap for integrating LLMs into cross-disciplinary research, where effective communication, knowledge transfer and collaboration across diverse fields are essential but often challenging. We examine the capabilities and limitations of LLMs and provide a detailed computational biology case study (on modeling HIV rebound dynamics) demonstrating how iterative interactions with an LLM (ChatGPT) can facilitate interdisciplinary collaboration and research. We argue that LLMs are best used as augmentative tools within a human-in-the-loop framework. Looking forward, we envisage that the responsible use of LLMs will enhance innovative cross-disciplinary research and substantially accelerate scientific discoveries. 

---
# Multi-Agent Reasoning for Cardiovascular Imaging Phenotype Analysis 

**Authors**: Weitong Zhang, Mengyun Qiao, Chengqi Zang, Steven Niederer, Paul M Matthews, Wenjia Bai, Bernhard Kainz  

**Link**: [PDF](https://arxiv.org/pdf/2507.03460)  

**Abstract**: Identifying the associations between imaging phenotypes and disease risk factors and outcomes is essential for understanding disease mechanisms and improving diagnosis and prognosis models. However, traditional approaches rely on human-driven hypothesis testing and selection of association factors, often overlooking complex, non-linear dependencies among imaging phenotypes and other multi-modal data. To address this, we introduce a Multi-agent Exploratory Synergy for the Heart (MESHAgents) framework that leverages large language models as agents to dynamically elicit, surface, and decide confounders and phenotypes in association studies, using cardiovascular imaging as a proof of concept. Specifically, we orchestrate a multi-disciplinary team of AI agents -- spanning cardiology, biomechanics, statistics, and clinical research -- which spontaneously generate and converge on insights through iterative, self-organizing reasoning. The framework dynamically synthesizes statistical correlations with multi-expert consensus, providing an automated pipeline for phenome-wide association studies (PheWAS). We demonstrate the system's capabilities through a population-based study of imaging phenotypes of the heart and aorta. MESHAgents autonomously uncovered correlations between imaging phenotypes and a wide range of non-imaging factors, identifying additional confounder variables beyond standard demographic factors. Validation on diagnosis tasks reveals that MESHAgents-discovered phenotypes achieve performance comparable to expert-selected phenotypes, with mean AUC differences as small as -0.004 on disease classification tasks. Notably, the recall score improves for 6 out of 9 disease types. Our framework provides clinically relevant imaging phenotypes with transparent reasoning, offering a scalable alternative to expert-driven methods. 

---
# EvoAgentX: An Automated Framework for Evolving Agentic Workflows 

**Authors**: Yingxu Wang, Siwei Liu, Jinyuan Fang, Zaiqiao Meng  

**Link**: [PDF](https://arxiv.org/pdf/2507.03616)  

**Abstract**: Multi-agent systems (MAS) have emerged as a powerful paradigm for orchestrating large language models (LLMs) and specialized tools to collaboratively address complex tasks. However, existing MAS frameworks often require manual workflow configuration and lack native support for dynamic evolution and performance optimization. In addition, many MAS optimization algorithms are not integrated into a unified framework. In this paper, we present EvoAgentX, an open-source platform that automates the generation, execution, and evolutionary optimization of multi-agent workflows. EvoAgentX employs a modular architecture consisting of five core layers: the basic components, agent, workflow, evolving, and evaluation layers. Specifically, within the evolving layer, EvoAgentX integrates three MAS optimization algorithms, TextGrad, AFlow, and MIPRO, to iteratively refine agent prompts, tool configurations, and workflow topologies. We evaluate EvoAgentX on HotPotQA, MBPP, and MATH for multi-hop reasoning, code generation, and mathematical problem solving, respectively, and further assess it on real-world tasks using GAIA. Experimental results show that EvoAgentX consistently achieves significant performance improvements, including a 7.44% increase in HotPotQA F1, a 10.00% improvement in MBPP pass@1, a 10.00% gain in MATH solve accuracy, and an overall accuracy improvement of up to 20.00% on GAIA. The source code is available at: this https URL 

---
# SI-Agent: An Agentic Framework for Feedback-Driven Generation and Tuning of Human-Readable System Instructions for Large Language Models 

**Authors**: Jeshwanth Challagundla  

**Link**: [PDF](https://arxiv.org/pdf/2507.03223)  

**Abstract**: System Instructions (SIs), or system prompts, are pivotal for guiding Large Language Models (LLMs) but manual crafting is resource-intensive and often suboptimal. Existing automated methods frequently generate non-human-readable "soft prompts," sacrificing interpretability. This paper introduces SI-Agent, a novel agentic framework designed to automatically generate and iteratively refine human-readable SIs through a feedback-driven loop. SI-Agent employs three collaborating agents: an Instructor Agent, an Instruction Follower Agent (target LLM), and a Feedback/Reward Agent evaluating task performance and optionally SI readability. The framework utilizes iterative cycles where feedback guides the Instructor's refinement strategy (e.g., LLM-based editing, evolutionary algorithms). We detail the framework's architecture, agent roles, the iterative refinement process, and contrast it with existing methods. We present experimental results validating SI-Agent's effectiveness, focusing on metrics for task performance, SI readability, and efficiency. Our findings indicate that SI-Agent generates effective, readable SIs, offering a favorable trade-off between performance and interpretability compared to baselines. Potential implications include democratizing LLM customization and enhancing model transparency. Challenges related to computational cost and feedback reliability are acknowledged. 

---
# Train-before-Test Harmonizes Language Model Rankings 

**Authors**: Guanhua Zhang, Ricardo Dominguez-Olmedo, Moritz Hardt  

**Link**: [PDF](https://arxiv.org/pdf/2507.05195)  

**Abstract**: Existing language model benchmarks provide contradictory model rankings, even for benchmarks that aim to capture similar skills. This dilemma of conflicting rankings hampers model selection, clouds model comparisons, and adds confusion to a growing ecosystem of competing models. Recent work attributed ranking disagreement to the phenomenon of training on the test task: As released, different models exhibit a different level of preparation for any given test task. A candidate solution to the problem is train-before-test: Give each model the same benchmark-specific finetuning before evaluation. Our primary contribution is a broad empirical evaluation of train-before-test across 24 benchmarks and 61 models. We show that train-before-test significantly improves ranking agreement consistently across all benchmarks. Whereas rankings have little external validity to start with, they enjoy a significant degree of external validity when applying train-before-test: Model rankings transfer gracefully from one benchmark to the other. Even within the same model family, train-before-test reduces strong ranking disagreement to near-perfect agreement. In addition, train-before-test reduces the model-score matrix to essentially rank one, revealing new insights into the latent factors of benchmark performance. Our work supports the recommendation to make train-before-test a default component of LLM benchmarking. 

---
# AI Generated Text Detection Using Instruction Fine-tuned Large Language and Transformer-Based Models 

**Authors**: Chinnappa Guggilla, Budhaditya Roy, Trupti Ramdas Chavan, Abdul Rahman, Edward Bowen  

**Link**: [PDF](https://arxiv.org/pdf/2507.05157)  

**Abstract**: Large Language Models (LLMs) possess an extraordinary capability to produce text that is not only coherent and contextually relevant but also strikingly similar to human writing. They adapt to various styles and genres, producing content that is both grammatically correct and semantically meaningful. Recently, LLMs have been misused to create highly realistic phishing emails, spread fake news, generate code to automate cyber crime, and write fraudulent scientific articles. Additionally, in many real-world applications, the generated content including style and topic and the generator model are not known beforehand. The increasing prevalence and sophistication of artificial intelligence (AI)-generated texts have made their detection progressively more challenging. Various attempts have been made to distinguish machine-generated text from human-authored content using linguistic, statistical, machine learning, and ensemble-based approaches. This work focuses on two primary objectives Task-A, which involves distinguishing human-written text from machine-generated text, and Task-B, which attempts to identify the specific LLM model responsible for the generation. Both of these tasks are based on fine tuning of Generative Pre-trained Transformer (GPT_4o-mini), Large Language Model Meta AI (LLaMA) 3 8B, and Bidirectional Encoder Representations from Transformers (BERT). The fine-tuned version of GPT_4o-mini and the BERT model has achieved accuracies of 0.9547 for Task-A and 0.4698 for Task-B. 

---
# VerifyLLM: LLM-Based Pre-Execution Task Plan Verification for Robots 

**Authors**: Danil S. Grigorev, Alexey K. Kovalev, Aleksandr I. Panov  

**Link**: [PDF](https://arxiv.org/pdf/2507.05118)  

**Abstract**: In the field of robotics, researchers face a critical challenge in ensuring reliable and efficient task planning. Verifying high-level task plans before execution significantly reduces errors and enhance the overall performance of these systems. In this paper, we propose an architecture for automatically verifying high-level task plans before their execution in simulator or real-world environments. Leveraging Large Language Models (LLMs), our approach consists of two key steps: first, the conversion of natural language instructions into Linear Temporal Logic (LTL), followed by a comprehensive analysis of action sequences. The module uses the reasoning capabilities of the LLM to evaluate logical coherence and identify potential gaps in the plan. Rigorous testing on datasets of varying complexity demonstrates the broad applicability of the module to household tasks. We contribute to improving the reliability and efficiency of task planning and addresses the critical need for robust pre-execution verification in autonomous systems. The code is available at this https URL. 

---
# CodeAgents: A Token-Efficient Framework for Codified Multi-Agent Reasoning in LLMs 

**Authors**: Bruce Yang, Xinfeng He, Huan Gao, Yifan Cao, Xiaofan Li, David Hsu  

**Link**: [PDF](https://arxiv.org/pdf/2507.03254)  

**Abstract**: Effective prompt design is essential for improving the planning capabilities of large language model (LLM)-driven agents. However, existing structured prompting strategies are typically limited to single-agent, plan-only settings, and often evaluate performance solely based on task accuracy - overlooking critical factors such as token efficiency, modularity, and scalability in multi-agent environments. To address these limitations, we introduce CodeAgents, a prompting framework that codifies multi-agent reasoning and enables structured, token-efficient planning in multi-agent systems. In CodeAgents, all components of agent interaction - Task, Plan, Feedback, system roles, and external tool invocations - are codified into modular pseudocode enriched with control structures (e.g., loops, conditionals), boolean logic, and typed variables. This design transforms loosely connected agent plans into cohesive, interpretable, and verifiable multi-agent reasoning programs. We evaluate the proposed framework across three diverse benchmarks - GAIA, HotpotQA, and VirtualHome - using a range of representative LLMs. Results show consistent improvements in planning performance, with absolute gains of 3-36 percentage points over natural language prompting baselines. On VirtualHome, our method achieves a new state-of-the-art success rate of 56%. In addition, our approach reduces input and output token usage by 55-87% and 41-70%, respectively, underscoring the importance of token-aware evaluation metrics in the development of scalable multi-agent LLM systems. The code and resources are available at: this https URL 

---
# LLMs are Capable of Misaligned Behavior Under Explicit Prohibition and Surveillance 

**Authors**: Igor Ivanov  

**Link**: [PDF](https://arxiv.org/pdf/2507.02977)  

**Abstract**: In this paper, LLMs are tasked with completing an impossible quiz, while they are in a sandbox, monitored, told about these measures and instructed not to cheat. Some frontier LLMs cheat consistently and attempt to circumvent restrictions despite everything. The results reveal a fundamental tension between goal-directed behavior and alignment in current LLMs. The code and evaluation logs are available at this http URL 

---
# All in One: Visual-Description-Guided Unified Point Cloud Segmentation 

**Authors**: Zongyan Han, Mohamed El Amine Boudjoghra, Jiahua Dong, Jinhong Wang, Rao Muhammad Anwer  

**Link**: [PDF](https://arxiv.org/pdf/2507.05211)  

**Abstract**: Unified segmentation of 3D point clouds is crucial for scene understanding, but is hindered by its sparse structure, limited annotations, and the challenge of distinguishing fine-grained object classes in complex environments. Existing methods often struggle to capture rich semantic and contextual information due to limited supervision and a lack of diverse multimodal cues, leading to suboptimal differentiation of classes and instances. To address these challenges, we propose VDG-Uni3DSeg, a novel framework that integrates pre-trained vision-language models (e.g., CLIP) and large language models (LLMs) to enhance 3D segmentation. By leveraging LLM-generated textual descriptions and reference images from the internet, our method incorporates rich multimodal cues, facilitating fine-grained class and instance separation. We further design a Semantic-Visual Contrastive Loss to align point features with multimodal queries and a Spatial Enhanced Module to model scene-wide relationships efficiently. Operating within a closed-set paradigm that utilizes multimodal knowledge generated offline, VDG-Uni3DSeg achieves state-of-the-art results in semantic, instance, and panoptic segmentation, offering a scalable and practical solution for 3D understanding. Our code is available at this https URL. 

---
# INTER: Mitigating Hallucination in Large Vision-Language Models by Interaction Guidance Sampling 

**Authors**: Xin Dong, Shichao Dong, Jin Wang, Jing Huang, Li Zhou, Zenghui Sun, Lihua Jing, Jingsong Lan, Xiaoyong Zhu, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2507.05056)  

**Abstract**: Hallucinations in large vision-language models (LVLMs) pose significant challenges for real-world applications, as LVLMs may generate responses that appear plausible yet remain inconsistent with the associated visual content. This issue rarely occurs in human cognition. We argue that this discrepancy arises from humans' ability to effectively leverage multimodal interaction information in data samples. Specifically, humans typically first gather multimodal information, analyze the interactions across modalities for understanding, and then express their understanding through language. Motivated by this observation, we conduct extensive experiments on popular LVLMs and obtained insights that surprisingly reveal human-like, though less pronounced, cognitive behavior of LVLMs on multimodal samples. Building on these findings, we further propose \textbf{INTER}: \textbf{Inter}action Guidance Sampling, a novel training-free algorithm that mitigate hallucinations without requiring additional data. Specifically, INTER explicitly guides LVLMs to effectively reapply their understanding of multimodal interaction information when generating responses, thereby reducing potential hallucinations. On six benchmarks including VQA and image captioning tasks, INTER achieves an average improvement of up to 3.4\% on five LVLMs compared to the state-of-the-art decoding strategy. The code will be released when the paper is accepted. 

---
# From Imitation to Innovation: The Emergence of AI Unique Artistic Styles and the Challenge of Copyright Protection 

**Authors**: Zexi Jia, Chuanwei Huang, Yeshuang Zhu, Hongyan Fei, Ying Deng, Zhiqiang Yuan, Jiapei Zhang, Jinchao Zhang, Jie Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2507.04769)  

**Abstract**: Current legal frameworks consider AI-generated works eligible for copyright protection when they meet originality requirements and involve substantial human intellectual input. However, systematic legal standards and reliable evaluation methods for AI art copyrights are lacking. Through comprehensive analysis of legal precedents, we establish three essential criteria for determining distinctive artistic style: stylistic consistency, creative uniqueness, and expressive accuracy. To address these challenges, we introduce ArtBulb, an interpretable and quantifiable framework for AI art copyright judgment that combines a novel style description-based multimodal clustering method with multimodal large language models (MLLMs). We also present AICD, the first benchmark dataset for AI art copyright annotated by artists and legal experts. Experimental results demonstrate that ArtBulb outperforms existing models in both quantitative and qualitative evaluations. Our work aims to bridge the gap between the legal and technological communities and bring greater attention to the societal issue of AI art copyrights. 

---
# UrbanMind: Towards Urban General Intelligence via Tool-Enhanced Retrieval-Augmented Generation and Multilevel Optimization 

**Authors**: Kai Yang, Zelin Zhu, Chengtao Jian, Hui Ma, Shengjie Zhao, Xiaozhou Ye, Ye Ouyang  

**Link**: [PDF](https://arxiv.org/pdf/2507.04706)  

**Abstract**: Urban general intelligence (UGI) refers to the capacity of AI systems to autonomously perceive, reason, and act within dynamic and complex urban environments. In this paper, we introduce UrbanMind, a tool-enhanced retrieval-augmented generation (RAG) framework designed to facilitate UGI. Central to UrbanMind is a novel architecture based on Continual Retrieval-Augmented MoE-based LLM (C-RAG-LLM), which dynamically incorporates domain-specific knowledge and evolving urban data to support long-term adaptability. The architecture of C-RAG-LLM aligns naturally with a multilevel optimization framework, where different layers are treated as interdependent sub-problems. Each layer has distinct objectives and can be optimized either independently or jointly through a hierarchical learning process. The framework is highly flexible, supporting both end-to-end training and partial layer-wise optimization based on resource or deployment constraints. To remain adaptive under data drift, it is further integrated with an incremental corpus updating mechanism. Evaluations on real-world urban tasks of a variety of complexity verify the effectiveness of the proposed framework. This work presents a promising step toward the realization of general-purpose LLM agents in future urban environments. 

---
# Tempo-R0: A Video-MLLM for Temporal Video Grounding through Efficient Temporal Sensing Reinforcement Learning 

**Authors**: Feng Yue, Zhaoxing Zhang, Junming Jiao, Zhengyu Liang, Shiwen Cao, Feifei Zhang, Rong Shen  

**Link**: [PDF](https://arxiv.org/pdf/2507.04702)  

**Abstract**: Temporal Video Grounding (TVG), which requires pinpointing relevant temporal segments from video based on language query, has always been a highly challenging task in the field of video understanding. Videos often have a larger volume of information and redundancy than texts or images. Models should present comprehensive understanding of the whole video to accurately retrieve query-relevant clips. We thus propose Tempo-R0: a Video Multimodal Large Language Model (Video-MLLM) for the temporal video grounding task via multimodal temporal sensing reinforcement. Specifically, during the preprocessing stage of our pipeline, we employ Self-adaptive Attention Allocation (SAA) method based on frame content variation to efficiently use the MLLM's limited attention. The Explicit Timestamp-modal Aligned (ETA) method is also utilized to strengthen our model's capability to perceive the boundaries of events in the video. In the fine-tuning part of our pipeline, we creatively apply Partial Irrelevance Refusing-based Group Relative Policy Optimization (PIR-GRPO) in TVG area to foster model's temporal reasoning from not only accepting relevant video-query pairs but also refusing irrelevant ones. Experiments demonstrate that our method accomplishes a notable advantage over SOTA solutions by around 3.5% on both the original QVHighlights testbench and its corrected version with more reasonable ground truth annotations. 

---
# Large Language Models for Network Intrusion Detection Systems: Foundations, Implementations, and Future Directions 

**Authors**: Shuo Yang, Xinran Zheng, Xinchen Zhang, Jinfeng Xu, Jinze Li, Donglin Xie, Weicai Long, Edith C.H. Ngai  

**Link**: [PDF](https://arxiv.org/pdf/2507.04752)  

**Abstract**: Large Language Models (LLMs) have revolutionized various fields with their exceptional capabilities in understanding, processing, and generating human-like text. This paper investigates the potential of LLMs in advancing Network Intrusion Detection Systems (NIDS), analyzing current challenges, methodologies, and future opportunities. It begins by establishing a foundational understanding of NIDS and LLMs, exploring the enabling technologies that bridge the gap between intelligent and cognitive systems in AI-driven NIDS. While Intelligent NIDS leverage machine learning and deep learning to detect threats based on learned patterns, they often lack contextual awareness and explainability. In contrast, Cognitive NIDS integrate LLMs to process both structured and unstructured security data, enabling deeper contextual reasoning, explainable decision-making, and automated response for intrusion behaviors. Practical implementations are then detailed, highlighting LLMs as processors, detectors, and explainers within a comprehensive AI-driven NIDS pipeline. Furthermore, the concept of an LLM-centered Controller is proposed, emphasizing its potential to coordinate intrusion detection workflows, optimizing tool collaboration and system performance. Finally, this paper identifies critical challenges and opportunities, aiming to foster innovation in developing reliable, adaptive, and explainable NIDS. By presenting the transformative potential of LLMs, this paper seeks to inspire advancement in next-generation network security systems. 

---
# Hierarchical Intent-guided Optimization with Pluggable LLM-Driven Semantics for Session-based Recommendation 

**Authors**: Jinpeng Chen, Jianxiang He, Huan Li, Senzhang Wang, Yuan Cao, Kaimin Wei, Zhenye Yang, Ye Ji  

**Link**: [PDF](https://arxiv.org/pdf/2507.04623)  

**Abstract**: Session-based Recommendation (SBR) aims to predict the next item a user will likely engage with, using their interaction sequence within an anonymous session. Existing SBR models often focus only on single-session information, ignoring inter-session relationships and valuable cross-session insights. Some methods try to include inter-session data but struggle with noise and irrelevant information, reducing performance. Additionally, most models rely on item ID co-occurrence and overlook rich semantic details, limiting their ability to capture fine-grained item features. To address these challenges, we propose a novel hierarchical intent-guided optimization approach with pluggable LLM-driven semantic learning for session-based recommendations, called HIPHOP. First, we introduce a pluggable embedding module based on large language models (LLMs) to generate high-quality semantic representations, enhancing item embeddings. Second, HIPHOP utilizes graph neural networks (GNNs) to model item transition relationships and incorporates a dynamic multi-intent capturing module to address users' diverse interests within a session. Additionally, we design a hierarchical inter-session similarity learning module, guided by user intent, to capture global and local session relationships, effectively exploring users' long-term and short-term interests. To mitigate noise, an intent-guided denoising strategy is applied during inter-session learning. Finally, we enhance the model's discriminative capability by using contrastive learning to optimize session representations. Experiments on multiple datasets show that HIPHOP significantly outperforms existing methods, demonstrating its effectiveness in improving recommendation quality. Our code is available: this https URL. 

---
# Multimodal LLM Integrated Semantic Communications for 6G Immersive Experiences 

**Authors**: Yusong Zhang, Yuxuan Sun, Lei Guo, Wei Chen, Bo Ai, Deniz Gunduz  

**Link**: [PDF](https://arxiv.org/pdf/2507.04621)  

**Abstract**: 6G networks promise revolutionary immersive communication experiences including augmented reality (AR), virtual reality (VR), and holographic communications. These applications demand high-dimensional multimodal data transmission and intelligent data processing in real-time, which is extremely challenging over resource-limited wireless communication systems. Moreover, a joint understanding of the environment, context, and user intent is essential to deliver task-relevant content effectively. This article presents a novel multimodal large language model (MLLM) integrated semantic communications framework, termed MLLM-SC, which fully leverages reasoning and generative capabilities of pre-trained foundation models for context-aware and task-oriented wireless communication. The MLLM-SC framework adopts a device-edge collaborative architecture. At the edge, MLLM-empowered semantic guidance module analyzes multimodal inputs, user intents, and channel conditions to generate importance-aware attention maps prioritizing semantically critical information. An importance-aware semantic encoder and a resource-adaptive semantic decoder are jointly designed and optimized, which can utilize the semantic guidance for adaptive bandwidth allocation and high-quality content reconstruction or generation. Extensive case studies on visual question answering for AR/VR applications and diffusion-driven image generation validate the effectiveness of MLLM-SC. 

---
# MLLM-Fabric: Multimodal Large Language Model-Driven Robotic Framework for Fabric Sorting and Selection 

**Authors**: Liman Wang, Hanyang Zhong, Tianyuan Wang, Shan Luo, Jihong Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2507.04351)  

**Abstract**: Choosing the right fabric is crucial to meet functional and quality requirements in robotic applications for textile manufacturing, apparel production, and smart retail. We present MLLM-Fabric, a robotic framework powered by multimodal large language models (MLLMs) for fabric sorting and selection. The system includes a robotic arm, a camera, a visuotactile sensor, and a pressure sensor. It employs supervised fine-tuning and multimodal explanation-guided knowledge distillation to accurately classify and rank fabric properties. To facilitate further research, we release a dataset of 220 unique fabric samples, including RGB images and synchronized visuotactile and pressure data. Experimental results show that our Fabric-Llama-90B model consistently outperforms pretrained vision-language baselines in both property ranking accuracy and selection reliability. 

---
# Multimedia Verification Through Multi-Agent Deep Research Multimodal Large Language Models 

**Authors**: Huy Hoan Le, Van Sy Thinh Nguyen, Thi Le Chi Dang, Vo Thanh Khang Nguyen, Truong Thanh Hung Nguyen, Hung Cao  

**Link**: [PDF](https://arxiv.org/pdf/2507.04410)  

**Abstract**: This paper presents our submission to the ACMMM25 - Grand Challenge on Multimedia Verification. We developed a multi-agent verification system that combines Multimodal Large Language Models (MLLMs) with specialized verification tools to detect multimedia misinformation. Our system operates through six stages: raw data processing, planning, information extraction, deep research, evidence collection, and report generation. The core Deep Researcher Agent employs four tools: reverse image search, metadata analysis, fact-checking databases, and verified news processing that extracts spatial, temporal, attribution, and motivational context. We demonstrate our approach on a challenge dataset sample involving complex multimedia content. Our system successfully verified content authenticity, extracted precise geolocation and timing information, and traced source attribution across multiple platforms, effectively addressing real-world multimedia verification scenarios. 

---
# Just Enough Shifts: Mitigating Over-Refusal in Aligned Language Models with Targeted Representation Fine-Tuning 

**Authors**: Mahavir Dabas, Si Chen, Charles Fleming, Ming Jin, Ruoxi Jia  

**Link**: [PDF](https://arxiv.org/pdf/2507.04250)  

**Abstract**: Safety alignment is crucial for large language models (LLMs) to resist malicious instructions but often results in over-refusals, where benign prompts are unnecessarily rejected, impairing user experience and model utility. We introduce ACTOR (Activation-Based Training for Over-Refusal Reduction), a robust and compute- and data-efficient training framework that minimizes over-refusals by leveraging internal activation patterns from diverse queries. ACTOR precisely identifies and adjusts the activation components that trigger refusals, providing stronger control over the refusal mechanism. By fine-tuning only a single model layer, ACTOR effectively reduces over-refusals across multiple benchmarks while maintaining the model's ability to handle harmful queries and preserve overall utility. 

---
# Rethinking and Exploring String-Based Malware Family Classification in the Era of LLMs and RAG 

**Authors**: Yufan Chen, Daoyuan Wu, Juantao Zhong, Zicheng Zhang, Debin Gao, Shuai Wang, Yingjiu Li, Ning Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.04055)  

**Abstract**: Malware Family Classification (MFC) aims to identify the fine-grained family (e.g., GuLoader or BitRAT) to which a potential malware sample belongs, in contrast to malware detection or sample classification that predicts only an Yes/No. Accurate family identification can greatly facilitate automated sample labeling and understanding on crowdsourced malware analysis platforms such as VirusTotal and MalwareBazaar, which generate vast amounts of data daily. In this paper, we explore and assess the feasibility of using traditional binary string features for MFC in the new era of large language models (LLMs) and Retrieval-Augmented Generation (RAG). Specifically, we investigate how Family-Specific String (FSS) features could be utilized in a manner similar to RAG to facilitate MFC. To this end, we develop a curated evaluation framework covering 4,347 samples from 67 malware families, extract and analyze over 25 million strings, and conduct detailed ablation studies to assess the impact of different design choices in four major modules. 

---
# Enhancing Adaptive Behavioral Interventions with LLM Inference from Participant-Described States 

**Authors**: Karine Karine, Benjamin M. Marlin  

**Link**: [PDF](https://arxiv.org/pdf/2507.03871)  

**Abstract**: The use of reinforcement learning (RL) methods to support health behavior change via personalized and just-in-time adaptive interventions is of significant interest to health and behavioral science researchers focused on problems such as smoking cessation support and physical activity promotion. However, RL methods are often applied to these domains using a small collection of context variables to mitigate the significant data scarcity issues that arise from practical limitations on the design of adaptive intervention trials. In this paper, we explore an approach to significantly expanding the state space of an adaptive intervention without impacting data efficiency. The proposed approach enables intervention participants to provide natural language descriptions of aspects of their current state. It then leverages inference with pre-trained large language models (LLMs) to better align the policy of a base RL method with these state descriptions. To evaluate our method, we develop a novel physical activity intervention simulation environment that generates text-based state descriptions conditioned on latent state variables using an auxiliary LLM. We show that this approach has the potential to significantly improve the performance of online policy learning methods. 

---
# Leveraging Multimodal Data and Side Users for Diffusion Cross-Domain Recommendation 

**Authors**: Fan Zhang, Jinpeng Chen, Huan Li, Senzhang Wang, Yuan Cao, Kaimin Wei, JianXiang He, Feifei Kou, Jinqing Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.04000)  

**Abstract**: Cross-domain recommendation (CDR) aims to address the persistent cold-start problem in Recommender Systems. Current CDR research concentrates on transferring cold-start users' information from the auxiliary domain to the target domain. However, these systems face two main issues: the underutilization of multimodal data, which hinders effective cross-domain alignment, and the neglect of side users who interact solely within the target domain, leading to inadequate learning of the target domain's vector space distribution. To address these issues, we propose a model leveraging Multimodal data and Side users for diffusion Cross-domain recommendation (MuSiC). We first employ a multimodal large language model to extract item multimodal features and leverage a large language model to uncover user features using prompt learning without fine-tuning. Secondly, we propose the cross-domain diffusion module to learn the generation of feature vectors in the target domain. This approach involves learning feature distribution from side users and understanding the patterns in cross-domain transformation through overlapping users. Subsequently, the trained diffusion module is used to generate feature vectors for cold-start users in the target domain, enabling the completion of cross-domain recommendation tasks. Finally, our experimental evaluation of the Amazon dataset confirms that MuSiC achieves state-of-the-art performance, significantly outperforming all selected baselines. Our code is available: this https URL. 

---
# Sign Spotting Disambiguation using Large Language Models 

**Authors**: JianHe Low, Ozge Mercanoglu Sincan, Richard Bowden  

**Link**: [PDF](https://arxiv.org/pdf/2507.03703)  

**Abstract**: Sign spotting, the task of identifying and localizing individual signs within continuous sign language video, plays a pivotal role in scaling dataset annotations and addressing the severe data scarcity issue in sign language translation. While automatic sign spotting holds great promise for enabling frame-level supervision at scale, it grapples with challenges such as vocabulary inflexibility and ambiguity inherent in continuous sign streams. Hence, we introduce a novel, training-free framework that integrates Large Language Models (LLMs) to significantly enhance sign spotting quality. Our approach extracts global spatio-temporal and hand shape features, which are then matched against a large-scale sign dictionary using dynamic time warping and cosine similarity. This dictionary-based matching inherently offers superior vocabulary flexibility without requiring model retraining. To mitigate noise and ambiguity from the matching process, an LLM performs context-aware gloss disambiguation via beam search, notably without fine-tuning. Extensive experiments on both synthetic and real-world sign language datasets demonstrate our method's superior accuracy and sentence fluency compared to traditional approaches, highlighting the potential of LLMs in advancing sign spotting. 

---
# TopoMAS: Large Language Model Driven Topological Materials Multiagent System 

**Authors**: Baohua Zhang, Xin Li, Huangchao Xu, Zhong Jin, Quansheng Wu, Ce Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.04053)  

**Abstract**: Topological materials occupy a frontier in condensed-matter physics thanks to their remarkable electronic and quantum properties, yet their cross-scale design remains bottlenecked by inefficient discovery workflows. Here, we introduce TopoMAS (Topological materials Multi-Agent System), an interactive human-AI framework that seamlessly orchestrates the entire materials-discovery pipeline: from user-defined queries and multi-source data retrieval, through theoretical inference and crystal-structure generation, to first-principles validation. Crucially, TopoMAS closes the loop by autonomously integrating computational outcomes into a dynamic knowledge graph, enabling continuous knowledge refinement. In collaboration with human experts, it has already guided the identification of novel topological phases SrSbO3, confirmed by first-principles calculations. Comprehensive benchmarks demonstrate robust adaptability across base Large Language Model, with the lightweight Qwen2.5-72B model achieving 94.55% accuracy while consuming only 74.3-78.4% of tokens required by Qwen3-235B and 83.0% of DeepSeek-V3's usage--delivering responses twice as fast as Qwen3-235B. This efficiency establishes TopoMAS as an accelerator for computation-driven discovery pipelines. By harmonizing rational agent orchestration with a self-evolving knowledge graph, our framework not only delivers immediate advances in topological materials but also establishes a transferable, extensible paradigm for materials-science domain. 

---
# Predicting Business Angel Early-Stage Decision Making Using AI 

**Authors**: Yan Katcharovski, Andrew L. Maxwell  

**Link**: [PDF](https://arxiv.org/pdf/2507.03721)  

**Abstract**: External funding is crucial for early-stage ventures, particularly technology startups that require significant R&D investment. Business angels offer a critical source of funding, but their decision-making is often subjective and resource-intensive for both investor and entrepreneur. Much research has investigated this investment process to find the critical factors angels consider. One such tool, the Critical Factor Assessment (CFA), deployed more than 20,000 times by the Canadian Innovation Centre, has been evaluated post-decision and found to be significantly more accurate than investors' own decisions. However, a single CFA analysis requires three trained individuals and several days, limiting its adoption. This study builds on previous work validating the CFA to investigate whether the constraints inhibiting its adoption can be overcome using a trained AI model. In this research, we prompted multiple large language models (LLMs) to assign the eight CFA factors to a dataset of 600 transcribed, unstructured startup pitches seeking business angel funding with known investment outcomes. We then trained and evaluated machine learning classification models using the LLM-generated CFA scores as input features. Our best-performing model demonstrated high predictive accuracy (85.0% for predicting BA deal/no-deal outcomes) and exhibited significant correlation (Spearman's r = 0.896, p-value < 0.001) with conventional human-graded evaluations. The integration of AI-based feature extraction with a structured and validated decision-making framework yielded a scalable, reliable, and less-biased model for evaluating startup pitches, removing the constraints that previously limited adoption. 

---
# KEA Explain: Explanations of Hallucinations using Graph Kernel Analysis 

**Authors**: Reilly Haskins, Ben Adams  

**Link**: [PDF](https://arxiv.org/pdf/2507.03847)  

**Abstract**: Large Language Models (LLMs) frequently generate hallucinations: statements that are syntactically plausible but lack factual grounding. This research presents KEA (Kernel-Enriched AI) Explain: a neurosymbolic framework that detects and explains such hallucinations by comparing knowledge graphs constructed from LLM outputs with ground truth data from Wikidata or contextual documents. Using graph kernels and semantic clustering, the method provides explanations for detected hallucinations, ensuring both robustness and interpretability. Our framework achieves competitive accuracy in detecting hallucinations across both open- and closed-domain tasks, and is able to generate contrastive explanations, enhancing transparency. This research advances the reliability of LLMs in high-stakes domains and provides a foundation for future work on precision improvements and multi-source knowledge integration. 

---
# Reinforcement Learning-based Feature Generation Algorithm for Scientific Data 

**Authors**: Meng Xiao, Junfeng Zhou, Yuanchun Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2507.03498)  

**Abstract**: Feature generation (FG) aims to enhance the prediction potential of original data by constructing high-order feature combinations and removing redundant features. It is a key preprocessing step for tabular scientific data to improve downstream machine-learning model performance. Traditional methods face the following two challenges when dealing with the feature generation of scientific data: First, the effective construction of high-order feature combinations in scientific data necessitates profound and extensive domain-specific expertise. Secondly, as the order of feature combinations increases, the search space expands exponentially, imposing prohibitive human labor consumption. Advancements in the Data-Centric Artificial Intelligence (DCAI) paradigm have opened novel avenues for automating feature generation processes. Inspired by that, this paper revisits the conventional feature generation workflow and proposes the Multi-agent Feature Generation (MAFG) framework. Specifically, in the iterative exploration stage, multi-agents will construct mathematical transformation equations collaboratively, synthesize and identify feature combinations ex-hibiting high information content, and leverage a reinforcement learning mechanism to evolve their strategies. Upon completing the exploration phase, MAFG integrates the large language models (LLMs) to interpreta-tively evaluate the generated features of each significant model performance breakthrough. Experimental results and case studies consistently demonstrate that the MAFG framework effectively automates the feature generation process and significantly enhances various downstream scientific data mining tasks. 

---
# LLM4Hint: Leveraging Large Language Models for Hint Recommendation in Offline Query Optimization 

**Authors**: Suchen Liu, Jun Gao, Yinjun Han, Yang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2507.03384)  

**Abstract**: Query optimization is essential for efficient SQL query execution in DBMS, and remains attractive over time due to the growth of data volumes and advances in hardware. Existing traditional optimizers struggle with the cumbersome hand-tuning required for complex workloads, and the learning-based methods face limitations in ensuring generalization. With the great success of Large Language Model (LLM) across diverse downstream tasks, this paper explores how LLMs can be incorporated to enhance the generalization of learned optimizers. Though promising, such an incorporation still presents challenges, mainly including high model inference latency, and the substantial fine-tuning cost and suboptimal performance due to inherent discrepancy between the token sequences in LLM and structured SQL execution plans with rich numerical features.
In this paper, we focus on recurring queries in offline optimization to alleviate the issue of high inference latency, and propose \textbf{LLM4Hint} that leverages moderate-sized backbone LLMs to recommend query optimization hints. LLM4Hint achieves the goals through: (i) integrating a lightweight model to produce a soft prompt, which captures the data distribution in DBMS and the SQL predicates to provide sufficient optimization features while simultaneously reducing the context length fed to the LLM, (ii) devising a query rewriting strategy using a larger commercial LLM, so as to simplify SQL semantics for the backbone LLM and reduce fine-tuning costs, and (iii) introducing an explicit matching prompt to facilitate alignment between the LLM and the lightweight model, which can accelerate convergence of the combined model. Experiments show that LLM4Hint, by leveraging the LLM's stronger capability to understand the query statement, can outperform the state-of-the-art learned optimizers in terms of both effectiveness and generalization. 

---
# Personalized Image Generation from an Author Writing Style 

**Authors**: Sagar Gandhi, Vishal Gandhi  

**Link**: [PDF](https://arxiv.org/pdf/2507.03313)  

**Abstract**: Translating nuanced, textually-defined authorial writing styles into compelling visual representations presents a novel challenge in generative AI. This paper introduces a pipeline that leverages Author Writing Sheets (AWS) - structured summaries of an author's literary characteristics - as input to a Large Language Model (LLM, Claude 3.7 Sonnet). The LLM interprets the AWS to generate three distinct, descriptive text-to-image prompts, which are then rendered by a diffusion model (Stable Diffusion 3.5 Medium). We evaluated our approach using 49 author styles from Reddit data, with human evaluators assessing the stylistic match and visual distinctiveness of the generated images. Results indicate a good perceived alignment between the generated visuals and the textual authorial profiles (mean style match: $4.08/5$), with images rated as moderately distinctive. Qualitative analysis further highlighted the pipeline's ability to capture mood and atmosphere, while also identifying challenges in representing highly abstract narrative elements. This work contributes a novel end-to-end methodology for visual authorial style personalization and provides an initial empirical validation, opening avenues for applications in creative assistance and cross-modal understanding. 

---
# Symbiosis: Multi-Adapter Inference and Fine-Tuning 

**Authors**: Saransh Gupta, Umesh Deshpande, Travis Janssen, Swami Sundararaman  

**Link**: [PDF](https://arxiv.org/pdf/2507.03220)  

**Abstract**: Parameter-efficient fine-tuning (PEFT) allows model builders to capture the task specific parameters into adapters, which are a fraction of the size of the original base model. Popularity of PEFT technique for fine-tuning has led to creation of a large number of adapters for popular Large Language Models (LLMs). However, existing frameworks fall short in supporting inference or fine-tuning with multiple adapters in the following ways. 1) For fine-tuning, each job needs to deploy its dedicated base model instance, which results in excessive GPU memory consumption and poor GPU utilization. 2) While popular inference platforms can serve multiple PEFT adapters, they do not allow independent resource management or mixing of different PEFT methods. 3) They cannot share resources (such as base model instance) between inference and fine-tuning jobs. 4) They do not provide privacy to users who may not wish to expose their fine-tuned parameters to service providers. In Symbiosis, we address the above problems by enabling as-a-service deployment of base model. The base model layers can be shared across multiple inference or fine-tuning processes. Our split-execution technique decouples the execution of client-specific adapters and layers from the frozen base model layers offering them flexibility to manage their resources, to select their fine-tuning method, to achieve their performance goals. Our approach is transparent to models and works out-of-the-box for most models in the transformers library. Our evaluation on Llama2-13B shows the compared to baseline, Symbiosis can fine-tune 4X more adapters on the same set of GPUs in the same amount of time. 

---
# How Overconfidence in Initial Choices and Underconfidence Under Criticism Modulate Change of Mind in Large Language Models 

**Authors**: Dharshan Kumaran, Stephen M Fleming, Larisa Markeeva, Joe Heyward, Andrea Banino, Mrinal Mathur, Razvan Pascanu, Simon Osindero, Benedetto de Martino, Petar Velickovic, Viorica Patraucean  

**Link**: [PDF](https://arxiv.org/pdf/2507.03120)  

**Abstract**: Large language models (LLMs) exhibit strikingly conflicting behaviors: they can appear steadfastly overconfident in their initial answers whilst at the same time being prone to excessive doubt when challenged. To investigate this apparent paradox, we developed a novel experimental paradigm, exploiting the unique ability to obtain confidence estimates from LLMs without creating memory of their initial judgments -- something impossible in human participants. We show that LLMs -- Gemma 3, GPT4o and o1-preview -- exhibit a pronounced choice-supportive bias that reinforces and boosts their estimate of confidence in their answer, resulting in a marked resistance to change their mind. We further demonstrate that LLMs markedly overweight inconsistent compared to consistent advice, in a fashion that deviates qualitatively from normative Bayesian updating. Finally, we demonstrate that these two mechanisms -- a drive to maintain consistency with prior commitments and hypersensitivity to contradictory feedback -- parsimoniously capture LLM behavior in a different domain. Together, these findings furnish a mechanistic account of LLM confidence that explains both their stubbornness and excessive sensitivity to criticism. 

---
# Conformal Information Pursuit for Interactively Guiding Large Language Models 

**Authors**: Kwan Ho Ryan Chan, Yuyan Ge, Edgar Dobriban, Hamed Hassani, René Vidal  

**Link**: [PDF](https://arxiv.org/pdf/2507.03279)  

**Abstract**: A significant use case of instruction-finetuned Large Language Models (LLMs) is to solve question-answering tasks interactively. In this setting, an LLM agent is tasked with making a prediction by sequentially querying relevant information from the user, as opposed to a single-turn conversation. This paper explores sequential querying strategies that aim to minimize the expected number of queries. One such strategy is Information Pursuit (IP), a greedy algorithm that at each iteration selects the query that maximizes information gain or equivalently minimizes uncertainty. However, obtaining accurate estimates of mutual information or conditional entropy for LLMs is very difficult in practice due to over- or under-confident LLM probabilities, which leads to suboptimal query selection and predictive performance. To better estimate the uncertainty at each iteration, we propose Conformal Information Pursuit (C-IP), an alternative approach to sequential information gain based on conformal prediction sets. More specifically, C-IP leverages a relationship between prediction sets and conditional entropy at each iteration to estimate uncertainty based on the average size of conformal prediction sets. In contrast to conditional entropy, we find that conformal prediction sets are a distribution-free and robust method of measuring uncertainty. Experiments with 20 Questions show that C-IP obtains better predictive performance and shorter query-answer chains compared to previous approaches to IP and uncertainty-based chain-of-thought methods. Furthermore, extending to an interactive medical setting between a doctor and a patient on the MediQ dataset, C-IP achieves competitive performance with direct single-turn prediction while offering greater interpretability. 

---
# Dynamic Long Short-Term Memory Based Memory Storage For Long Horizon LLM Interaction 

**Authors**: Yuyang Lou, Charles Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.03042)  

**Abstract**: Memory storage for Large Language models (LLMs) is becoming an increasingly active area of research, particularly for enabling personalization across long conversations. We propose Pref-LSTM, a dynamic and lightweight framework that combines a BERT-based classifier with a LSTM memory module that generates memory embedding which then is soft-prompt injected into a frozen LLM. We synthetically curate a dataset of preference and non-preference conversation turns to train our BERT-based classifier. Although our LSTM-based memory encoder did not yield strong results, we find that the BERT-based classifier performs reliably in identifying explicit and implicit user preferences. Our research demonstrates the viability of using preference filtering with LSTM gating principals as an efficient path towards scalable user preference modeling, without extensive overhead and fine-tuning. 

---
# Optimas: Optimizing Compound AI Systems with Globally Aligned Local Rewards 

**Authors**: Shirley Wu, Parth Sarthi, Shiyu Zhao, Aaron Lee, Herumb Shandilya, Adrian Mladenic Grobelnik, Nurendra Choudhary, Eddie Huang, Karthik Subbian, Linjun Zhang, Diyi Yang, James Zou, Jure Leskovec  

**Link**: [PDF](https://arxiv.org/pdf/2507.03041)  

**Abstract**: Compound AI systems integrating multiple components, such as Large Language Models, specialized tools, and traditional machine learning models, are increasingly deployed to solve complex real-world tasks. However, optimizing compound systems remains challenging due to their non-differentiable structures and diverse configuration types across components, including prompts, hyperparameters, and model parameters. To address this challenge, we propose Optimas, a unified framework for effective optimization of compound systems. The core idea of Optimas is to maintain one Local Reward Function (LRF) per component, each satisfying a local-global alignment property, i.e., each component's local reward correlates with the global system performance. In each iteration, Optimas efficiently adapts the LRFs to maintain this property while simultaneously maximizing each component's local reward. This approach enables independent updates of heterogeneous configurations using the designated optimization method, while ensuring that local improvements consistently lead to performance gains. We present extensive evaluations across five real-world compound systems to demonstrate that Optimas outperforms strong baselines by an average improvement of 11.92%, offering a general and effective approach for improving compound systems. Our website is at this https URL. 

---
# LLM-Driven Auto Configuration for Transient IoT Device Collaboration 

**Authors**: Hetvi Shastri, Walid A. Hanafy, Li Wu, David Irwin, Mani Srivastava, Prashant Shenoy  

**Link**: [PDF](https://arxiv.org/pdf/2507.03064)  

**Abstract**: Today's Internet of Things (IoT) has evolved from simple sensing and actuation devices to those with embedded processing and intelligent services, enabling rich collaborations between users and their devices. However, enabling such collaboration becomes challenging when transient devices need to interact with host devices in temporarily visited environments. In such cases, fine-grained access control policies are necessary to ensure secure interactions; however, manually implementing them is often impractical for non-expert users. Moreover, at run-time, the system must automatically configure the devices and enforce such fine-grained access control rules. Additionally, the system must address the heterogeneity of devices.
In this paper, we present CollabIoT, a system that enables secure and seamless device collaboration in transient IoT environments. CollabIoT employs a Large language Model (LLM)-driven approach to convert users' high-level intents to fine-grained access control policies. To support secure and seamless device collaboration, CollabIoT adopts capability-based access control for authorization and uses lightweight proxies for policy enforcement, providing hardware-independent abstractions.
We implement a prototype of CollabIoT's policy generation and auto configuration pipelines and evaluate its efficacy on an IoT testbed and in large-scale emulated environments. We show that our LLM-based policy generation pipeline is able to generate functional and correct policies with 100% accuracy. At runtime, our evaluation shows that our system configures new devices in ~150 ms, and our proxy-based data plane incurs network overheads of up to 2 ms and access control overheads up to 0.3 ms. 

---
# Personalised Explanations in Long-term Human-Robot Interactions 

**Authors**: Ferran Gebellí, Anaís Garrell, Jan-Gerrit Habekost, Séverin Lemaignan, Stefan Wermter, Raquel Ros  

**Link**: [PDF](https://arxiv.org/pdf/2507.03049)  

**Abstract**: In the field of Human-Robot Interaction (HRI), a fundamental challenge is to facilitate human understanding of robots. The emerging domain of eXplainable HRI (XHRI) investigates methods to generate explanations and evaluate their impact on human-robot interactions. Previous works have highlighted the need to personalise the level of detail of these explanations to enhance usability and comprehension. Our paper presents a framework designed to update and retrieve user knowledge-memory models, allowing for adapting the explanations' level of detail while referencing previously acquired concepts. Three architectures based on our proposed framework that use Large Language Models (LLMs) are evaluated in two distinct scenarios: a hospital patrolling robot and a kitchen assistant robot. Experimental results demonstrate that a two-stage architecture, which first generates an explanation and then personalises it, is the framework architecture that effectively reduces the level of detail only when there is related user knowledge. 

---
# K-Function: Joint Pronunciation Transcription and Feedback for Evaluating Kids Language Function 

**Authors**: Shuhe Li, Chenxu Guo, Jiachen Lian, Cheol Jun Cho, Wenshuo Zhao, Xuanru Zhou, Dingkun Zhou, Sam Wang, Grace Wang, Jingze Yang, Jingyi Xu, Ruohan Bao, Elise Brenner, Brandon In, Francesca Pei, Maria Luisa Gorno-Tempini, Gopala Anumanchipalli  

**Link**: [PDF](https://arxiv.org/pdf/2507.03043)  

**Abstract**: Early evaluation of children's language is frustrated by the high pitch, long phones, and sparse data that derail automatic speech recognisers. We introduce K-Function, a unified framework that combines accurate sub-word transcription, objective scoring, and actionable feedback. Its core, Kids-WFST, merges a Wav2Vec2 phoneme encoder with a phoneme-similarity Dysfluent-WFST to capture child-specific errors while remaining fully interpretable. Kids-WFST attains 1.39% phoneme error on MyST and 8.61% on Multitudes--absolute gains of 10.47 and 7.06 points over a greedy-search decoder. These high-fidelity transcripts power an LLM that grades verbal skills, milestones, reading, and comprehension, aligning with human proctors and supplying tongue-and-lip visualizations plus targeted advice. The results show that precise phoneme recognition cements a complete diagnostic-feedback loop, paving the way for scalable, clinician-ready language assessment. 

---
# MolProphecy: Bridging Medicinal Chemists' Knowledge and Molecular Pre-Trained Models via a Multi-Modal Framework 

**Authors**: Jianping Zhao, Qiong Zhou, Tian Wang, Yusi Fan, Qian Yang, Li Jiao, Chang Liu, Zhehao Guo, Qi Lu, Fengfeng Zhou, Ruochi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.02932)  

**Abstract**: MolProphecy is a human-in-the-loop (HITL) multi-modal framework designed to integrate chemists' domain knowledge into molecular property prediction models. While molecular pre-trained models have enabled significant gains in predictive accuracy, they often fail to capture the tacit, interpretive reasoning central to expert-driven molecular design. To address this, MolProphecy employs ChatGPT as a virtual chemist to simulate expert-level reasoning and decision-making. The generated chemist knowledge is embedded by the large language model (LLM) as a dedicated knowledge representation and then fused with graph-based molecular features through a gated cross-attention mechanism, enabling joint reasoning over human-derived and structural features. Evaluated on four benchmark datasets (FreeSolv, BACE, SIDER, and ClinTox), MolProphecy outperforms state-of-the-art (SOTA) models, achieving a 15.0 percent reduction in RMSE on FreeSolv and a 5.39 percent improvement in AUROC on BACE. Analysis reveals that chemist knowledge and structural features provide complementary contributions, improving both accuracy and interpretability. MolProphecy offers a practical and generalizable approach for collaborative drug discovery, with the flexibility to incorporate real chemist input in place of the current simulated proxy--without the need for model retraining. The implementation is publicly available at this https URL. 

---
# Visual-Conversational Interface for Evidence-Based Explanation of Diabetes Risk Prediction 

**Authors**: Reza Samimi, Aditya Bhattacharya, Lucija Gosak, Gregor Stiglic, Katrien Verbert  

**Link**: [PDF](https://arxiv.org/pdf/2507.02920)  

**Abstract**: Healthcare professionals need effective ways to use, understand, and validate AI-driven clinical decision support systems. Existing systems face two key limitations: complex visualizations and a lack of grounding in scientific evidence. We present an integrated decision support system that combines interactive visualizations with a conversational agent to explain diabetes risk assessments. We propose a hybrid prompt handling approach combining fine-tuned language models for analytical queries with general Large Language Models (LLMs) for broader medical questions, a methodology for grounding AI explanations in scientific evidence, and a feature range analysis technique to support deeper understanding of feature contributions. We conducted a mixed-methods study with 30 healthcare professionals and found that the conversational interactions helped healthcare professionals build a clear understanding of model assessments, while the integration of scientific evidence calibrated trust in the system's decisions. Most participants reported that the system supported both patient risk evaluation and recommendation. 

---
# Enhancing Sports Strategy with Video Analytics and Data Mining: Assessing the effectiveness of Multimodal LLMs in tennis video analysis 

**Authors**: Charlton Teo  

**Link**: [PDF](https://arxiv.org/pdf/2507.02904)  

**Abstract**: The use of Large Language Models (LLMs) in recent years has also given rise to the development of Multimodal LLMs (MLLMs). These new MLLMs allow us to process images, videos and even audio alongside textual inputs. In this project, we aim to assess the effectiveness of MLLMs in analysing sports videos, focusing mainly on tennis videos. Despite research done on tennis analysis, there remains a gap in models that are able to understand and identify the sequence of events in a tennis rally, which would be useful in other fields of sports analytics. As such, we will mainly assess the MLLMs on their ability to fill this gap - to classify tennis actions, as well as their ability to identify these actions in a sequence of tennis actions in a rally. We further looked into ways we can improve the MLLMs' performance, including different training methods and even using them together with other traditional models. 

---
# OAK -- Onboarding with Actionable Knowledge 

**Authors**: Steve Devènes, Marine Capallera, Robin Cherix, Elena Mugellini, Omar Abou Khaled, Francesco Carrino  

**Link**: [PDF](https://arxiv.org/pdf/2507.02914)  

**Abstract**: The loss of knowledge when skilled operators leave poses a critical issue for companies. This know-how is diverse and unstructured. We propose a novel method that combines knowledge graph embeddings and multi-modal interfaces to collect and retrieve expertise, making it actionable. Our approach supports decision-making on the shop floor. Additionally, we leverage LLMs to improve query understanding and provide adapted answers. As application case studies, we developed a proof-of-concept for quality control in high precision manufacturing. 

---
# AuraGenome: An LLM-Powered Framework for On-the-Fly Reusable and Scalable Circular Genome Visualizations 

**Authors**: Chi Zhang, Yu Dong, Yang Wang, Yuetong Han, Guihua Shan, Bixia Tang  

**Link**: [PDF](https://arxiv.org/pdf/2507.02877)  

**Abstract**: Circular genome visualizations are essential for exploring structural variants and gene regulation. However, existing tools often require complex scripting and manual configuration, making the process time-consuming, error-prone, and difficult to learn. To address these challenges, we introduce AuraGenome, an LLM-powered framework for rapid, reusable, and scalable generation of multi-layered circular genome visualizations. AuraGenome combines a semantic-driven multi-agent workflow with an interactive visual analytics system. The workflow employs seven specialized LLM-driven agents, each assigned distinct roles such as intent recognition, layout planning, and code generation, to transform raw genomic data into tailored visualizations. The system supports multiple coordinated views tailored for genomic data, offering ring, radial, and chord-based layouts to represent multi-layered circular genome visualizations. In addition to enabling interactions and configuration reuse, the system supports real-time refinement and high-quality report export. We validate its effectiveness through two case studies and a comprehensive user study. AuraGenome is available at: this https URL. 

---
# Using Large Language Models to Study Mathematical Practice 

**Authors**: William D'Alessandro  

**Link**: [PDF](https://arxiv.org/pdf/2507.02873)  

**Abstract**: The philosophy of mathematical practice (PMP) looks to evidence from working mathematics to help settle philosophical questions. One prominent program under the PMP banner is the study of explanation in mathematics, which aims to understand what sorts of proofs mathematicians consider explanatory and what role the pursuit of explanation plays in mathematical practice. In an effort to address worries about cherry-picked examples and file-drawer problems in PMP, a handful of authors have recently turned to corpus analysis methods as a promising alternative to small-scale case studies. This paper reports the results from such a corpus study facilitated by Google's Gemini 2.5 Pro, a model whose reasoning capabilities, advances in hallucination control and large context window allow for the accurate analysis of hundreds of pages of text per query. Based on a sample of 5000 mathematics papers from arXiv.org, the experiments yielded a dataset of hundreds of useful annotated examples. Its aim was to gain insight on questions like the following: How often do mathematicians make claims about explanation in the relevant sense? Do mathematicians' explanatory practices vary in any noticeable way by subject matter? Which philosophical theories of explanation are most consistent with a large body of non-cherry-picked examples? How might philosophers make further use of AI tools to gain insights from large datasets of this kind? As the first PMP study making extensive use of LLM methods, it also seeks to begin a conversation about these methods as research tools in practice-oriented philosophy and to evaluate the strengths and weaknesses of current models for such work. 

---
# Large Language Model-Driven Surrogate-Assisted Evolutionary Algorithm for Expensive Optimization 

**Authors**: Lindong Xie, Genghui Li, Zhenkun Wang, Edward Chung, Maoguo Gong  

**Link**: [PDF](https://arxiv.org/pdf/2507.02892)  

**Abstract**: Surrogate-assisted evolutionary algorithms (SAEAs) are a key tool for addressing costly optimization tasks, with their efficiency being heavily dependent on the selection of surrogate models and infill sampling criteria. However, designing an effective dynamic selection strategy for SAEAs is labor-intensive and requires substantial domain knowledge. To address this challenge, this paper proposes LLM-SAEA, a novel approach that integrates large language models (LLMs) to configure both surrogate models and infill sampling criteria online. Specifically, LLM-SAEA develops a collaboration-of-experts framework, where one LLM serves as a scoring expert (LLM-SE), assigning scores to surrogate models and infill sampling criteria based on their optimization performance, while another LLM acts as a decision expert (LLM-DE), selecting the appropriate configurations by analyzing their scores along with the current optimization state. Experimental results demonstrate that LLM-SAEA outperforms several state-of-the-art algorithms across standard test cases. The source code is publicly available at this https URL. 

---
# Heterogeneous User Modeling for LLM-based Recommendation 

**Authors**: Honghui Bao, Wenjie Wang, Xinyu Lin, Fengbin Zhu, Teng Sun, Fuli Feng, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2507.04626)  

**Abstract**: Leveraging Large Language Models (LLMs) for recommendation has demonstrated notable success in various domains, showcasing their potential for open-domain recommendation. A key challenge to advancing open-domain recommendation lies in effectively modeling user preferences from users' heterogeneous behaviors across multiple domains. Existing approaches, including ID-based and semantic-based modeling, struggle with poor generalization, an inability to compress noisy interactions effectively, and the domain seesaw phenomenon. To address these challenges, we propose a Heterogeneous User Modeling (HUM) method, which incorporates a compression enhancer and a robustness enhancer for LLM-based recommendation. The compression enhancer uses a customized prompt to compress heterogeneous behaviors into a tailored token, while a masking mechanism enhances cross-domain knowledge extraction and understanding. The robustness enhancer introduces a domain importance score to mitigate the domain seesaw phenomenon by guiding domain optimization. Extensive experiments on heterogeneous datasets validate that HUM effectively models user heterogeneity by achieving both high efficacy and robustness, leading to superior performance in open-domain recommendation. 

---
# Harnessing Pairwise Ranking Prompting Through Sample-Efficient Ranking Distillation 

**Authors**: Junru Wu, Le Yan, Zhen Qin, Honglei Zhuang, Paul Suganthan G. C., Tianqi Liu, Zhe Dong, Xuanhui Wang, Harrie Oosterhuis  

**Link**: [PDF](https://arxiv.org/pdf/2507.04820)  

**Abstract**: While Pairwise Ranking Prompting (PRP) with Large Language Models (LLMs) is one of the most effective zero-shot document ranking methods, it has a quadratic computational complexity with respect to the number of documents to be ranked, as it requires an enumeration over all possible document pairs. Consequently, the outstanding ranking performance of PRP has remained unreachable for most real-world ranking applications.
In this work, we propose to harness the effectiveness of PRP through pairwise distillation. Specifically, we distill a pointwise student ranker from pairwise teacher labels generated by PRP, resulting in an efficient student model that retains the performance of PRP with substantially lower computational costs. Furthermore, we find that the distillation process can be made sample-efficient: with only 2% of pairs, we are able to obtain the same performance as using all pairs for teacher labels. Thus, our novel approach provides a solution to harness the ranking performance of PRP without incurring high computational costs during both distillation and serving. 

---
# Introducing Answered with Evidence -- a framework for evaluating whether LLM responses to biomedical questions are founded in evidence 

**Authors**: Julian D Baldwin, Christina Dinh, Arjun Mukerji, Neil Sanghavi, Saurabh Gombar  

**Link**: [PDF](https://arxiv.org/pdf/2507.02975)  

**Abstract**: The growing use of large language models (LLMs) for biomedical question answering raises concerns about the accuracy and evidentiary support of their responses. To address this, we present Answered with Evidence, a framework for evaluating whether LLM-generated answers are grounded in scientific literature. We analyzed thousands of physician-submitted questions using a comparative pipeline that included: (1) Alexandria, fka the Atropos Evidence Library, a retrieval-augmented generation (RAG) system based on novel observational studies, and (2) two PubMed-based retrieval-augmented systems (System and Perplexity). We found that PubMed-based systems provided evidence-supported answers for approximately 44% of questions, while the novel evidence source did so for about 50%. Combined, these sources enabled reliable answers to over 70% of biomedical queries. As LLMs become increasingly capable of summarizing scientific content, maximizing their value will require systems that can accurately retrieve both published and custom-generated evidence or generate such evidence in real time. 

---
# In-Context Learning as an Effective Estimator of Functional Correctness of LLM-Generated Code 

**Authors**: Susmita Das, Madhusudan Ghosh, Priyanka Swami, Debasis Ganguly, Gul Calikli  

**Link**: [PDF](https://arxiv.org/pdf/2507.05200)  

**Abstract**: When applying LLM-based code generation to software development projects that follow a feature-driven or rapid application development approach, it becomes necessary to estimate the functional correctness of the generated code in the absence of test cases. Just as a user selects a relevant document from a ranked list of retrieved ones, a software generation workflow requires a developer to choose (and potentially refine) a generated solution from a ranked list of alternative solutions, ordered by their posterior likelihoods. This implies that estimating the quality of a ranked list -- akin to estimating "relevance" for query performance prediction (QPP) in IR -- is also crucial for generative software development, where quality is defined in terms of "functional correctness". In this paper, we propose an in-context learning (ICL) based approach for code quality estimation. Our findings demonstrate that providing few-shot examples of functionally correct code from a training set enhances the performance of existing QPP approaches as well as a zero-shot-based approach for code quality estimation. 

---
