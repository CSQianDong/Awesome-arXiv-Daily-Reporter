# Applying Large Language Models to Characterize Public Narratives 

**Authors**: Elinor Poole-Dayan, Daniel T Kessler, Hannah Chiou, Margaret Hughes, Emily S Lin, Marshall Ganz, Deb Roy  

**Link**: [PDF](https://arxiv.org/pdf/2511.13505)  

**Abstract**: Public Narratives (PNs) are key tools for leadership development and civic mobilization, yet their systematic analysis remains challenging due to their subjective interpretation and the high cost of expert annotation. In this work, we propose a novel computational framework that leverages large language models (LLMs) to automate the qualitative annotation of public narratives. Using a codebook we co-developed with subject-matter experts, we evaluate LLM performance against that of expert annotators. Our work reveals that LLMs can achieve near-human-expert performance, achieving an average F1 score of 0.80 across 8 narratives and 14 codes. We then extend our analysis to empirically explore how PN framework elements manifest across a larger dataset of 22 stories. Lastly, we extrapolate our analysis to a set of political speeches, establishing a novel lens in which to analyze political rhetoric in civic spaces. This study demonstrates the potential of LLM-assisted annotation for scalable narrative analysis and highlights key limitations and directions for future research in computational civic storytelling. 

---
# Can Large Language Models Function as Qualified Pediatricians? A Systematic Evaluation in Real-World Clinical Contexts 

**Authors**: Siyu Zhu, Mouxiao Bian, Yue Xie, Yongyu Tang, Zhikang Yu, Tianbin Li, Pengcheng Chen, Bing Han, Jie Xu, Xiaoyan Dong  

**Link**: [PDF](https://arxiv.org/pdf/2511.13381)  

**Abstract**: With the rapid rise of large language models (LLMs) in medicine, a key question is whether they can function as competent pediatricians in real-world clinical settings. We developed PEDIASBench, a systematic evaluation framework centered on a knowledge-system framework and tailored to realistic clinical environments. PEDIASBench assesses LLMs across three dimensions: application of basic knowledge, dynamic diagnosis and treatment capability, and pediatric medical safety and medical ethics. We evaluated 12 representative models released over the past two years, including GPT-4o, Qwen3-235B-A22B, and DeepSeek-V3, covering 19 pediatric subspecialties and 211 prototypical diseases. State-of-the-art models performed well on foundational knowledge, with Qwen3-235B-A22B achieving over 90% accuracy on licensing-level questions, but performance declined ~15% as task complexity increased, revealing limitations in complex reasoning. Multiple-choice assessments highlighted weaknesses in integrative reasoning and knowledge recall. In dynamic diagnosis and treatment scenarios, DeepSeek-R1 scored highest in case reasoning (mean 0.58), yet most models struggled to adapt to real-time patient changes. On pediatric medical ethics and safety tasks, Qwen2.5-72B performed best (accuracy 92.05%), though humanistic sensitivity remained limited. These findings indicate that pediatric LLMs are constrained by limited dynamic decision-making and underdeveloped humanistic care. Future development should focus on multimodal integration and a clinical feedback-model iteration loop to enhance safety, interpretability, and human-AI collaboration. While current LLMs cannot independently perform pediatric care, they hold promise for decision support, medical education, and patient communication, laying the groundwork for a safe, trustworthy, and collaborative intelligent pediatric healthcare system. 

---
# Donors and Recipients: On Asymmetric Transfer Across Tasks and Languages with Parameter-Efficient Fine-Tuning 

**Authors**: Kajetan Dymkiewicz, Ivan Vulic, Helen Yannakoudakis, Eilam Shapira, Roi Reichart, Anna Korhonen  

**Link**: [PDF](https://arxiv.org/pdf/2511.13368)  

**Abstract**: Large language models (LLMs) perform strongly across tasks and languages, yet how improvements in one task or language affect other tasks and languages and their combinations remains poorly understood. We conduct a controlled PEFT/LoRA study across multiple open-weight LLM families and sizes, treating task and language as transfer axes while conditioning on model family and size; we fine-tune each model on a single task-language source and measure transfer as the percentage-point change versus its baseline score when evaluated on all other task-language target pairs. We decompose transfer into (i) Matched-Task (Cross-Language), (ii) Matched-Language (Cross-Task), and (iii) Cross-Task (Cross-Language) regimes. We uncover two consistent general patterns. First, a pronounced on-task vs. off-task asymmetry: Matched-Task (Cross-Language) transfer is reliably positive, whereas off-task transfer often incurs collateral degradation. Second, a stable donor-recipient structure across languages and tasks (hub donors vs. brittle recipients). We outline implications for risk-aware fine-tuning and model specialisation. 

---
# Crossing Borders: A Multimodal Challenge for Indian Poetry Translation and Image Generation 

**Authors**: Sofia Jamil, Kotla Sai Charan, Sriparna Saha, Koustava Goswami, Joseph K J  

**Link**: [PDF](https://arxiv.org/pdf/2511.13689)  

**Abstract**: Indian poetry, known for its linguistic complexity and deep cultural resonance, has a rich and varied heritage spanning thousands of years. However, its layered meanings, cultural allusions, and sophisticated grammatical constructions often pose challenges for comprehension, especially for non-native speakers or readers unfamiliar with its context and language. Despite its cultural significance, existing works on poetry have largely overlooked Indian language poems. In this paper, we propose the Translation and Image Generation (TAI) framework, leveraging Large Language Models (LLMs) and Latent Diffusion Models through appropriate prompt tuning. Our framework supports the United Nations Sustainable Development Goals of Quality Education (SDG 4) and Reduced Inequalities (SDG 10) by enhancing the accessibility of culturally rich Indian-language poetry to a global audience. It includes (1) a translation module that uses an Odds Ratio Preference Alignment Algorithm to accurately translate morphologically rich poetry into English, and (2) an image generation module that employs a semantic graph to capture tokens, dependencies, and semantic relationships between metaphors and their meanings, to create visually meaningful representations of Indian poems. Our comprehensive experimental evaluation, including both human and quantitative assessments, demonstrates the superiority of TAI Diffusion in poem image generation tasks, outperforming strong baselines. To further address the scarcity of resources for Indian-language poetry, we introduce the Morphologically Rich Indian Language Poems MorphoVerse Dataset, comprising 1,570 poems across 21 low-resource Indian languages. By addressing the gap in poetry translation and visual comprehension, this work aims to broaden accessibility and enrich the reader's experience. 

---
# Why is "Chicago" Predictive of Deceptive Reviews? Using LLMs to Discover Language Phenomena from Lexical Cues 

**Authors**: Jiaming Qu, Mengtian Guo, Yue Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.13658)  

**Abstract**: Deceptive reviews mislead consumers, harm businesses, and undermine trust in online marketplaces. Machine learning classifiers can learn from large amounts of training examples to effectively distinguish deceptive reviews from genuine ones. However, the distinguishing features learned by these classifiers are often subtle, fragmented, and difficult for humans to interpret. In this work, we explore using large language models (LLMs) to translate machine-learned lexical cues into human-understandable language phenomena that can differentiate deceptive reviews from genuine ones. We show that language phenomena obtained in this manner are empirically grounded in data, generalizable across similar domains, and more predictive than phenomena either in LLMs' prior knowledge or obtained through in-context learning. These language phenomena have the potential to aid people in critically assessing the credibility of online reviews in environments where deception detection classifiers are unavailable. 

---
# Beyond SELECT: A Comprehensive Taxonomy-Guided Benchmark for Real-World Text-to-SQL Translation 

**Authors**: Hao Wang, Yuanfeng Song, Xiaoming Yin, Xing Chen  

**Link**: [PDF](https://arxiv.org/pdf/2511.13590)  

**Abstract**: Text-to-SQL datasets are essential for training and evaluating text-to-SQL models, but existing datasets often suffer from limited coverage and fail to capture the diversity of real-world applications. To address this, we propose a novel taxonomy for text-to-SQL classification based on dimensions including core intents, statement types, syntax structures, and key actions. Using this taxonomy, we evaluate widely used public text-to-SQL datasets (e.g., Spider and Bird) and reveal limitations in their coverage and diversity. We then introduce a taxonomy-guided dataset synthesis pipeline, yielding a new dataset named SQL-Synth. This approach combines the taxonomy with Large Language Models (LLMs) to ensure the dataset reflects the breadth and complexity of real-world text-to-SQL applications. Extensive analysis and experimental results validate the effectiveness of our taxonomy, as SQL-Synth exhibits greater diversity and coverage compared to existing benchmarks. Moreover, we uncover that existing LLMs typically fall short in adequately capturing the full range of scenarios, resulting in limited performance on SQL-Synth. However, fine-tuning can substantially improve their performance in these scenarios. The proposed taxonomy has significant potential impact, as it not only enables comprehensive analysis of datasets and the performance of different LLMs, but also guides the construction of training data for LLMs. 

---
# Zero-Shot Grammar Competency Estimation Using Large Language Model Generated Pseudo Labels 

**Authors**: Sourya Dipta Das, Shubham Kumar, Kuldeep Yadav  

**Link**: [PDF](https://arxiv.org/pdf/2511.13152)  

**Abstract**: Grammar competency estimation is essential for assessing linguistic proficiency in both written and spoken language; however, the spoken modality presents additional challenges due to its spontaneous, unstructured, and disfluent nature. Developing accurate grammar scoring models further requires extensive expert annotation, making large-scale data creation impractical. To address these limitations, we propose a zero-shot grammar competency estimation framework that leverages unlabeled data and Large Language Models (LLMs) without relying on manual labels. During training, we employ LLM-generated predictions on unlabeled data by using grammar competency rubric-based prompts. These predictions, treated as pseudo labels, are utilized to train a transformer-based model through a novel training framework designed to handle label noise effectively. We show that the choice of LLM for pseudo-label generation critically affects model performance and that the ratio of clean-to-noisy samples during training strongly influences stability and accuracy. Finally, a qualitative analysis of error intensity and score prediction confirms the robustness and interpretability of our approach. Experimental results demonstrate the efficacy of our approach in estimating grammar competency scores with high accuracy, paving the way for scalable, low-resource grammar assessment systems. 

---
# Extracting Events Like Code: A Multi-Agent Programming Framework for Zero-Shot Event Extraction 

**Authors**: Quanjiang Guo, Sijie Wang, Jinchuan Zhang, Ben Zhang, Zhao Kang, Ling Tian, Ke Yan  

**Link**: [PDF](https://arxiv.org/pdf/2511.13118)  

**Abstract**: Zero-shot event extraction (ZSEE) remains a significant challenge for large language models (LLMs) due to the need for complex reasoning and domain-specific understanding. Direct prompting often yields incomplete or structurally invalid outputs--such as misclassified triggers, missing arguments, and schema violations. To address these limitations, we present Agent-Event-Coder (AEC), a novel multi-agent framework that treats event extraction like software engineering: as a structured, iterative code-generation process. AEC decomposes ZSEE into specialized subtasks--retrieval, planning, coding, and verification--each handled by a dedicated LLM agent. Event schemas are represented as executable class definitions, enabling deterministic validation and precise feedback via a verification agent. This programming-inspired approach allows for systematic disambiguation and schema enforcement through iterative refinement. By leveraging collaborative agent workflows, AEC enables LLMs to produce precise, complete, and schema-consistent extractions in zero-shot settings. Experiments across five diverse domains and six LLMs demonstrate that AEC consistently outperforms prior zero-shot baselines, showcasing the power of treating event extraction like code generation. The code and data are released on this https URL. 

---
# TCM-5CEval: Extended Deep Evaluation Benchmark for LLM's Comprehensive Clinical Research Competence in Traditional Chinese Medicine 

**Authors**: Tianai Huang, Jiayuan Chen, Lu Lu, Pengcheng Chen, Tianbin Li, Bing Han, Wenchao Tang, Jie Xu, Ming Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.13169)  

**Abstract**: Large language models (LLMs) have demonstrated exceptional capabilities in general domains, yet their application in highly specialized and culturally-rich fields like Traditional Chinese Medicine (TCM) requires rigorous and nuanced evaluation. Building upon prior foundational work such as TCM-3CEval, which highlighted systemic knowledge gaps and the importance of cultural-contextual alignment, we introduce TCM-5CEval, a more granular and comprehensive benchmark. TCM-5CEval is designed to assess LLMs across five critical dimensions: (1) Core Knowledge (TCM-Exam), (2) Classical Literacy (TCM-LitQA), (3) Clinical Decision-making (TCM-MRCD), (4) Chinese Materia Medica (TCM-CMM), and (5) Clinical Non-pharmacological Therapy (TCM-ClinNPT). We conducted a thorough evaluation of fifteen prominent LLMs, revealing significant performance disparities and identifying top-performing models like deepseek\_r1 and gemini\_2\_5\_pro. Our findings show that while models exhibit proficiency in recalling foundational knowledge, they struggle with the interpretative complexities of classical texts. Critically, permutation-based consistency testing reveals widespread fragilities in model inference. All evaluated models, including the highest-scoring ones, displayed a substantial performance degradation when faced with varied question option ordering, indicating a pervasive sensitivity to positional bias and a lack of robust understanding. TCM-5CEval not only provides a more detailed diagnostic tool for LLM capabilities in TCM but aldso exposes fundamental weaknesses in their reasoning stability. To promote further research and standardized comparison, TCM-5CEval has been uploaded to the Medbench platform, joining its predecessor in the "In-depth Challenge for Comprehensive TCM Abilities" special track. 

---
# BeDiscovER: The Benchmark of Discourse Understanding in the Era of Reasoning Language Models 

**Authors**: Chuyuan Li, Giuseppe Carenini  

**Link**: [PDF](https://arxiv.org/pdf/2511.13095)  

**Abstract**: We introduce BeDiscovER (Benchmark of Discourse Understanding in the Era of Reasoning Language Models), an up-to-date, comprehensive suite for evaluating the discourse-level knowledge of modern LLMs. BeDiscovER compiles 5 publicly available discourse tasks across discourse lexicon, (multi-)sentential, and documental levels, with in total 52 individual datasets. It covers both extensively studied tasks such as discourse parsing and temporal relation extraction, as well as some novel challenges such as discourse particle disambiguation (e.g., ``just''), and also aggregates a shared task on Discourse Relation Parsing and Treebanking for multilingual and multi-framework discourse relation classification. We evaluate open-source LLMs: Qwen3 series, DeepSeek-R1, and frontier model such as GPT-5-mini on BeDiscovER, and find that state-of-the-art models exhibit strong performance in arithmetic aspect of temporal reasoning, but they struggle with full document reasoning and some subtle semantic and discourse phenomena, such as rhetorical relation recognition. 

---
# Distinguishing Repetition Disfluency from Morphological Reduplication in Bangla ASR Transcripts: A Novel Corpus and Benchmarking Analysis 

**Authors**: Zaara Zabeen Arpa, Sadnam Sakib Apurbo, Nazia Karim Khan Oishee, Ajwad Abrar  

**Link**: [PDF](https://arxiv.org/pdf/2511.13159)  

**Abstract**: Automatic Speech Recognition (ASR) transcripts, especially in low-resource languages like Bangla, contain a critical ambiguity: word-word repetitions can be either Repetition Disfluency (unintentional ASR error/hesitation) or Morphological Reduplication (a deliberate grammatical construct). Standard disfluency correction fails by erroneously deleting valid linguistic information. To solve this, we introduce the first publicly available, 20,000-row Bangla corpus, manually annotated to explicitly distinguish between these two phenomena in noisy ASR transcripts. We benchmark this novel resource using two paradigms: state-of-the-art multilingual Large Language Models (LLMs) and task-specific fine-tuning of encoder models. LLMs achieve competitive performance (up to 82.68\% accuracy) with few-shot prompting. However, fine-tuning proves superior, with the language-specific BanglaBERT model achieving the highest accuracy of 84.78\% and an F1 score of 0.677. This establishes a strong, linguistically-informed baseline and provides essential data for developing sophisticated, semantic-preserving text normalization systems for Bangla. 

---
# Spark-Prover-X1: Formal Theorem Proving Through Diverse Data Training 

**Authors**: Xinyuan Zhou, Yi Lei, Xiaoyu Zhou, Jingyi Sun, Yu Zhu, Zhongyi Ye, Weitai Zhang, Quan Liu, Si Wei, Cong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.13043)  

**Abstract**: Large Language Models (LLMs) have shown significant promise in automated theorem proving, yet progress is often constrained by the scarcity of diverse and high-quality formal language data. To address this issue, we introduce Spark-Prover-X1, a 7B parameter model trained via an three-stage framework designed to unlock the reasoning potential of more accessible and moderately-sized LLMs. The first stage infuses deep knowledge through continuous pre-training on a broad mathematical corpus, enhanced by a suite of novel data tasks. Key innovation is a "CoT-augmented state prediction" task to achieve fine-grained reasoning. The second stage employs Supervised Fine-tuning (SFT) within an expert iteration loop to specialize both the Spark-Prover-X1-7B and Spark-Formalizer-X1-7B models. Finally, a targeted round of Group Relative Policy Optimization (GRPO) is applied to sharpen the prover's capabilities on the most challenging problems. To facilitate robust evaluation, particularly on problems from real-world examinations, we also introduce ExamFormal-Bench, a new benchmark dataset of 402 formal problems. Experimental results demonstrate that Spark-Prover-X1-7B achieves state-of-the-art performance among similarly-sized open-source models, attaining a 37.0\% average pass rate (pass@32). It shows exceptional performance on difficult competition benchmarks, notably solving 27 problems on PutnamBench (pass@32) and achieving 24.0\% on CombiBench (pass@32). Our work validates that this diverse training data and progressively refined training pipeline provides an effective path for enhancing the formal reasoning capabilities of lightweight LLMs. Both Spark-Prover-X1-7B and Spark-Formalizer-X1-7B, along with the ExamFormal-Bench dataset, are made publicly available at:this https URL, this https URL. 

---
# Fine-Tuned LLMs Know They Don't Know: A Parameter-Efficient Approach to Recovering Honesty 

**Authors**: Zeyu Shi, Ziming Wang, Tianyu Chen, Shiqi Gao, Haoyi Zhou, Qingyun Sun, Jianxin Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.12991)  

**Abstract**: The honesty of Large Language Models (LLMs) is increasingly important for safe deployment in high-stakes domains. However, this crucial trait is severely undermined by supervised fine-tuning (SFT), a common technique for model specialization. Existing recovery methods rely on data-intensive global parameter adjustments, implicitly assuming that SFT deeply corrupts the models' ability to recognize their knowledge boundaries. However, we observe that fine-tuned LLMs still preserve this ability; what is damaged is their capacity to faithfully express that awareness. Building on this, we propose Honesty-Critical Neurons Restoration (HCNR) to surgically repair this suppressed capacity. HCNR identifies and restores key expression-governing neurons to their pre-trained state while harmonizing them with task-oriented neurons via Hessian-guided compensation. Experiments on four QA tasks and five LLM families demonstrate that HCNR effectively recovers 33.25% of the compromised honesty while achieving at least 2.23x speedup with over 10x less data compared to baseline methods, offering a practical solution for trustworthy LLM deployment. 

---
# AA-Omniscience: Evaluating Cross-Domain Knowledge Reliability in Large Language Models 

**Authors**: Declan Jackson, William Keating, George Cameron, Micah Hill-Smith  

**Link**: [PDF](https://arxiv.org/pdf/2511.13029)  

**Abstract**: Existing language model evaluations primarily measure general capabilities, yet reliable use of these models across a range of domains demands factual accuracy and recognition of knowledge gaps. We introduce AA-Omniscience, a benchmark designed to measure both factual recall and knowledge calibration across 6,000 questions. Questions are derived from authoritative academic and industry sources, and cover 42 economically relevant topics within six different domains. The evaluation measures a model's Omniscience Index, a bounded metric (-100 to 100) measuring factual recall that jointly penalizes hallucinations and rewards abstention when uncertain, with 0 equating to a model that answers questions correctly as much as it does incorrectly. Among evaluated models, Claude 4.1 Opus attains the highest score (4.8), making it one of only three models to score above zero. These results reveal persistent factuality and calibration weaknesses across frontier models. Performance also varies by domain, with the models from three different research labs leading across the six domains. This performance variability suggests models should be chosen according to the demands of the use case rather than general performance for tasks where knowledge is important. 

---
# NeuroLex: A Lightweight Domain Language Model for EEG Report Understanding and Generation 

**Authors**: Kang Yin, Hye-Bin Shin  

**Link**: [PDF](https://arxiv.org/pdf/2511.12851)  

**Abstract**: Clinical electroencephalogram (EEG) reports encode domain-specific linguistic conventions that general-purpose language models (LMs) fail to capture. We introduce NeuroLex, a lightweight domain-adaptive language model trained purely on EEG report text from the Harvard Electroencephalography Database. Unlike existing biomedical LMs, NeuroLex is tailored to the linguistic and diagnostic characteristics of EEG reporting, enabling it to serve as both an independent textual model and a decoder backbone for multimodal EEG-language systems. Using span-corruption pretraining and instruction-style fine-tuning on report polishing, paragraph summarization, and terminology question answering, NeuroLex learns the syntax and reasoning patterns characteristic of EEG interpretation. Comprehensive evaluations show that it achieves lower perplexity, higher extraction and summarization accuracy, better label efficiency, and improved robustness to negation and factual hallucination compared with general models of the same scale. With an EEG-aware linguistic backbone, NeuroLex bridges biomedical text modeling and brain-computer interface applications, offering a foundation for interpretable and language-driven neural decoding. 

---
# From Passive to Persuasive: Steering Emotional Nuance in Human-AI Negotiation 

**Authors**: Niranjan Chebrolu, Gerard Christopher Yeo, Kokil Jaidka  

**Link**: [PDF](https://arxiv.org/pdf/2511.12832)  

**Abstract**: Large Language Models (LLMs) demonstrate increasing conversational fluency, yet instilling them with nuanced, human-like emotional expression remains a significant challenge. Current alignment techniques often address surface-level output or require extensive fine-tuning. This paper demonstrates that targeted activation engineering can steer LLaMA 3.1-8B to exhibit more human-like emotional nuances. We first employ attribution patching to identify causally influential components, to find a key intervention locus by observing activation patterns during diagnostic conversational tasks. We then derive emotional expression vectors from the difference in the activations generated by contrastive text pairs (positive vs. negative examples of target emotions). Applying these vectors to new conversational prompts significantly enhances emotional characteristics: steered responses show increased positive sentiment (e.g., joy, trust) and more frequent first-person pronoun usage, indicative of greater personal engagement. Our findings offer a precise and interpretable framework and new directions for the study of conversational AI. 

---
# On the Brittleness of LLMs: A Journey around Set Membership 

**Authors**: Lea Hergert, Gábor Berend, Mario Szegedy, Gyorgy Turan, Márk Jelasity  

**Link**: [PDF](https://arxiv.org/pdf/2511.12728)  

**Abstract**: Large language models (LLMs) achieve superhuman performance on complex reasoning tasks, yet often fail on much simpler problems, raising concerns about their reliability and interpretability. We investigate this paradox through a focused study with two key design features: simplicity, to expose basic failure modes, and scale, to enable comprehensive controlled experiments. We focus on set membership queries -- among the most fundamental forms of reasoning -- using tasks like ``Is apple an element of the set \{pear, plum, apple, raspberry\}?''. We conduct a systematic empirical evaluation across prompt phrasing, semantic structure, element ordering, and model choice. Our large-scale analysis reveals that LLM performance on this elementary task is consistently brittle, and unpredictable across all dimensions, suggesting that the models' ``understanding'' of the set concept is fragmented and convoluted at best. Our work demonstrates that the large-scale experiments enabled by the simplicity of the problem allow us to map and analyze the failure modes comprehensively, making this approach a valuable methodology for LLM evaluation in general. 

---
# BioMedJImpact: A Comprehensive Dataset and LLM Pipeline for AI Engagement and Scientific Impact Analysis of Biomedical Journals 

**Authors**: Ruiyu Wang, Yuzhang Xie, Xiao Hu, Carl Yang, Jiaying Lu  

**Link**: [PDF](https://arxiv.org/pdf/2511.12821)  

**Abstract**: Assessing journal impact is central to scholarly communication, yet existing open resources rarely capture how collaboration structures and artificial intelligence (AI) research jointly shape venue prestige in biomedicine. We present BioMedJImpact, a large-scale, biomedical-oriented dataset designed to advance journal-level analysis of scientific impact and AI engagement. Built from 1.74 million PubMed Central articles across 2,744 journals, BioMedJImpact integrates bibliometric indicators, collaboration features, and LLM-derived semantic indicators for AI engagement. Specifically, the AI engagement feature is extracted through a reproducible three-stage LLM pipeline that we propose. Using this dataset, we analyze how collaboration intensity and AI engagement jointly influence scientific impact across pre- and post-pandemic periods (2016-2019, 2020-2023). Two consistent trends emerge: journals with higher collaboration intensity, particularly those with larger and more diverse author teams, tend to achieve greater citation impact, and AI engagement has become an increasingly strong correlate of journal prestige, especially in quartile rankings. To further validate the three-stage LLM pipeline we proposed for deriving the AI engagement feature, we conduct human evaluation, confirming substantial agreement in AI relevance detection and consistent subfield classification. Together, these contributions demonstrate that BioMedJImpact serves as both a comprehensive dataset capturing the intersection of biomedicine and AI, and a validated methodological framework enabling scalable, content-aware scientometric analysis of scientific impact and innovation dynamics. Code is available at this https URL. 

---
# Visual Room 2.0: Seeing is Not Understanding for MLLMs 

**Authors**: Haokun Li, Yazhou Zhang, Jizhi Ding, Qiuchi Li, Peng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.12928)  

**Abstract**: Can multi-modal large language models (MLLMs) truly understand what they can see? Extending Searle's Chinese Room into the multi-modal domain, this paper proposes the Visual Room argument: MLLMs may describe every visual detail precisely yet fail to comprehend the underlying emotions and intentions, namely seeing is not understanding. Building on this, we introduce \textit{Visual Room} 2.0, a hierarchical benchmark for evaluating perception-cognition alignment of MLLMs. We model human perceptive and cognitive processes across three levels: low, middle, and high, covering 17 representative tasks. The perception component ranges from attribute recognition to scene understanding, while the cognition component extends from textual entailment to causal and social reasoning. The dataset contains 350 multi-modal samples, each with six progressive questions (2,100 in total) spanning perception to cognition. Evaluating 10 state-of-the-art (SoTA) MLLMs, we highlight three key findings: (1) MLLMs exhibit stronger perceptual competence than cognitive ability (8.0\%$\uparrow$); (2) cognition appears not causally dependent on perception-based reasoning; and (3) cognition scales with model size, but perception does not consistently improve with larger variants. This work operationalizes Seeing $\ne$ Understanding as a testable hypothesis, offering a new paradigm from perceptual processing to cognitive reasoning in MLLMs. Our dataset is available at this https URL. 

---
# Adaptive Focus Memory for Language Models 

**Authors**: Christopher Cruz  

**Link**: [PDF](https://arxiv.org/pdf/2511.12712)  

**Abstract**: Large language models (LLMs) are increasingly deployed in multi-turn dialogue settings, but their behavior is still bottlenecked by fixed context windows and naive memory strategies. Replaying the full conversation at every turn is simple but expensive, while static summarization or recency-only heuristics often erase safety-critical user details. We present Adaptive Focus Memory (AFM), a dynamic context manager that assigns each past message one of three fidelity levels -- FULL, COMPRESSED, or PLACEHOLDER -- based on semantic similarity to the current query, half-life recency weighting, and importance classification. AFM packs messages chronologically under a strict token budget, preferring high fidelity for the most relevant turns while aiming to preserve a cheap trace of the dialogue. In a safety-oriented benchmark involving a user with a severe peanut allergy planning a trip to Thailand, AFM retains the allergy across both short and medium-length conversations, matches the safety performance of naive replay, and cuts average token usage by 66% relative to a replay baseline. We release a modular Python implementation of AFM designed for OpenAI-compatible APIs and offline operation, enabling practitioners to reduce inference cost without sacrificing safety or factual continuity in the evaluated scenario. 

---
# Uni-MoE-2.0-Omni: Scaling Language-Centric Omnimodal Large Model with Advanced MoE, Training and Data 

**Authors**: Yunxin Li, Xinyu Chen, Shenyuan Jiang, Haoyuan Shi, Zhenyu Liu, Xuanyu Zhang, Nanhao Deng, Zhenran Xu, Yicheng Ma, Meishan Zhang, Baotian Hu, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.12609)  

**Abstract**: We present Uni-MoE 2.0 from the Lychee family. As a fully open-source omnimodal large model (OLM), it substantially advances Lychee's Uni-MoE series in language-centric multimodal understanding, reasoning, and generating. Based on the Qwen2.5-7B dense architecture, we build Uni-MoE-2.0-Omni from scratch through three core contributions: dynamic-capacity Mixture-of-Experts (MoE) design, a progressive training strategy enhanced with an iterative reinforcement strategy, and a carefully curated multimodal data matching technique. It is capable of omnimodal understanding, as well as generating images, text, and speech. Architecturally, our new MoE framework balances computational efficiency and capability for 10 cross-modal inputs using shared, routed, and null experts, while our Omni-Modality 3D RoPE ensures spatio-temporal cross-modality alignment in the self-attention layer. For training, following cross-modal pretraining, we use a progressive supervised fine-tuning strategy that activates modality-specific experts and is enhanced by balanced data composition and an iterative GSPO-DPO method to stabilise RL training and improve reasoning. Data-wise, the base model, trained on approximately 75B tokens of open-source multimodal data, is equipped with special speech and image generation tokens, allowing it to learn these generative tasks by conditioning its outputs on linguistic cues. Extensive evaluation across 85 benchmarks demonstrates that our model achieves SOTA or highly competitive performance against leading OLMs, surpassing Qwen2.5-Omni (trained with 1.2T tokens) on over 50 of 76 benchmarks. Key strengths include video understanding (+7% avg. of 8), omnimodallity understanding (+7% avg. of 4), and audiovisual reasoning (+4%). It also advances long-form speech processing (reducing WER by 4.2%) and leads in low-level image processing and controllable generation across 5 metrics. 

---
# From Perception to Reasoning: Deep Thinking Empowers Multimodal Large Language Models 

**Authors**: Wenxin Zhu, Andong Chen, Yuchen Song, Kehai Chen, Conghui Zhu, Ziyan Chen, Tiejun Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2511.12861)  

**Abstract**: With the remarkable success of Multimodal Large Language Models (MLLMs) in perception tasks, enhancing their complex reasoning capabilities has emerged as a critical research focus. Existing models still suffer from challenges such as opaque reasoning paths and insufficient generalization ability. Chain-of-Thought (CoT) reasoning, which has demonstrated significant efficacy in language models by enhancing reasoning transparency and output interpretability, holds promise for improving model reasoning capabilities when extended to the multimodal domain. This paper provides a systematic review centered on "Multimodal Chain-of-Thought" (MCoT). First, it analyzes the background and theoretical motivations for its inception from the perspectives of technical evolution and task demands. Then, it introduces mainstream MCoT methods from three aspects: CoT paradigms, the post-training stage, and the inference stage, while also analyzing their underlying mechanisms. Furthermore, the paper summarizes existing evaluation benchmarks and metrics, and discusses the application scenarios of MCoT. Finally, it analyzes the challenges currently facing MCoT and provides an outlook on its future research directions. 

---
# Reason-KE++: Aligning the Process, Not Just the Outcome, for Faithful LLM Knowledge Editing 

**Authors**: Yuchen Wu, Liang Ding, Li Shen, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2511.12661)  

**Abstract**: Aligning Large Language Models (LLMs) to be faithful to new knowledge in complex, multi-hop reasoning tasks is a critical, yet unsolved, challenge. We find that SFT-based methods, e.g., Reason-KE, while state-of-the-art, suffer from a "faithfulness gap": they optimize for format mimicry rather than sound reasoning. This gap enables the LLM's powerful parametric priors to override new contextual facts, resulting in critical factual hallucinations (e.g., incorrectly reasoning "Houston" from "NASA" despite an explicit edit). To solve this core LLM alignment problem, we propose Reason-KE++, an SFT+RL framework that instills process-level faithfulness. Its core is a Stage-aware Reward mechanism that provides dense supervision for intermediate reasoning steps (e.g., Decomposition, Sub-answer Correctness). Crucially, we identify that naive outcome-only RL is a deceptive trap for LLM alignment: it collapses reasoning integrity (e.g., 19.00% Hop acc) while superficially boosting final accuracy. Our process-aware framework sets a new SOTA of 95.48% on MQUAKE-CF-3k (+5.28%), demonstrating that for complex tasks, aligning the reasoning process is essential for building trustworthy LLMs. 

---
# Evolve the Method, Not the Prompts: Evolutionary Synthesis of Jailbreak Attacks on LLMs 

**Authors**: Yunhao Chen, Xin Wang, Juncheng Li, Yixu Wang, Jie Li, Yan Teng, Yingchun Wang, Xingjun Ma  

**Link**: [PDF](https://arxiv.org/pdf/2511.12710)  

**Abstract**: Automated red teaming frameworks for Large Language Models (LLMs) have become increasingly sophisticated, yet they share a fundamental limitation: their jailbreak logic is confined to selecting, combining, or refining pre-existing attack strategies. This binds their creativity and leaves them unable to autonomously invent entirely new attack mechanisms. To overcome this gap, we introduce \textbf{EvoSynth}, an autonomous framework that shifts the paradigm from attack planning to the evolutionary synthesis of jailbreak methods. Instead of refining prompts, EvoSynth employs a multi-agent system to autonomously engineer, evolve, and execute novel, code-based attack algorithms. Crucially, it features a code-level self-correction loop, allowing it to iteratively rewrite its own attack logic in response to failure. Through extensive experiments, we demonstrate that EvoSynth not only establishes a new state-of-the-art by achieving an 85.5\% Attack Success Rate (ASR) against highly robust models like Claude-Sonnet-4.5, but also generates attacks that are significantly more diverse than those from existing methods. We release our framework to facilitate future research in this new direction of evolutionary synthesis of jailbreak methods. Code is available at: this https URL. 

---
# Don't Think of the White Bear: Ironic Negation in Transformer Models Under Cognitive Load 

**Authors**: Logan Mann, Nayan Saxena, Sarah Tandon, Chenhao Sun, Savar Toteja, Kevin Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2511.12381)  

**Abstract**: Negation instructions such as 'do not mention $X$' can paradoxically increase the accessibility of $X$ in human thought, a phenomenon known as ironic rebound. Large language models (LLMs) face the same challenge: suppressing a concept requires internally activating it, which may prime rebound instead of avoidance. We investigated this tension with two experiments. \textbf{(1) Load \& content}: after a negation instruction, we vary distractor text (semantic, syntactic, repetition) and measure rebound strength. \textbf{(2) Polarity separation}: We test whether models distinguish neutral from negative framings of the same concept and whether this separation predicts rebound persistence. Results show that rebound consistently arises immediately after negation and intensifies with longer or semantic distractors, while repetition supports suppression. Stronger polarity separation correlates with more persistent rebound. Together, these findings, complemented by a circuit tracing analysis that identifies sparse middle-layer attention heads amplifying forbidden tokens while early layers suppress, link cognitive predictions of ironic rebound with mechanistic insights into long-context interference. To support future work, we release ReboundBench, a dataset of $5,000$ systematically varied negation prompts designed to probe rebound in LLMs. 

---
# Cmprsr: Abstractive Token-Level Question-Agnostic Prompt Compressor 

**Authors**: Ivan Zakazov, Alexander Sharipov, Berke Argin, Oussama Gabouj, Kamel Charaf, Alexi Semiz, Lorenzo Drudi, Nicolas Baldwin, Robert West  

**Link**: [PDF](https://arxiv.org/pdf/2511.12281)  

**Abstract**: Motivated by the high costs of using black-box Large Language Models (LLMs), we introduce a novel prompt compression paradigm, under which we use smaller LLMs to compress inputs for the larger ones. We present the first comprehensive LLM-as-a-compressor benchmark spanning 25 open- and closed-source models, which reveals significant disparity in models' compression ability in terms of (i) preserving semantically important information (ii) following the user-provided compression rate (CR). We further improve the performance of gpt-4.1-mini, the best overall vanilla compressor, with Textgrad-based compression meta-prompt optimization. We also identify the most promising open-source vanilla LLM - Qwen3-4B - and post-train it with a combination of supervised fine-tuning (SFT) and Group Relative Policy Optimization (GRPO), pursuing the dual objective of CR adherence and maximizing the downstream task performance. We call the resulting model Cmprsr and demonstrate its superiority over both extractive and vanilla abstractive compression across the entire range of compression rates on lengthy inputs from MeetingBank and LongBench as well as short prompts from GSM8k. The latter highlights Cmprsr's generalizability across varying input lengths and domains. Moreover, Cmprsr closely follows the requested compression rate, offering fine control over the cost-quality trade-off. 

---
# Assessing LLMs for Serendipity Discovery in Knowledge Graphs: A Case for Drug Repurposing 

**Authors**: Mengying Wang, Chenhui Ma, Ao Jiao, Tuo Liang, Pengjun Lu, Shrinidhi Hegde, Yu Yin, Evren Gurkan-Cavusoglu, Yinghui Wu  

**Link**: [PDF](https://arxiv.org/pdf/2511.12472)  

**Abstract**: Large Language Models (LLMs) have greatly advanced knowledge graph question answering (KGQA), yet existing systems are typically optimized for returning highly relevant but predictable answers. A missing yet desired capacity is to exploit LLMs to suggest surprise and novel ("serendipitious") answers. In this paper, we formally define the serendipity-aware KGQA task and propose the SerenQA framework to evaluate LLMs' ability to uncover unexpected insights in scientific KGQA tasks. SerenQA includes a rigorous serendipity metric based on relevance, novelty, and surprise, along with an expert-annotated benchmark derived from the Clinical Knowledge Graph, focused on drug repurposing. Additionally, it features a structured evaluation pipeline encompassing three subtasks: knowledge retrieval, subgraph reasoning, and serendipity exploration. Our experiments reveal that while state-of-the-art LLMs perform well on retrieval, they still struggle to identify genuinely surprising and valuable discoveries, underscoring a significant room for future improvements. Our curated resources and extended version are released at: this https URL. 

---
# SGuard-v1: Safety Guardrail for Large Language Models 

**Authors**: JoonHo Lee, HyeonMin Cho, Jaewoong Yun, Hyunjae Lee, JunKyu Lee, Juree Seok  

**Link**: [PDF](https://arxiv.org/pdf/2511.12497)  

**Abstract**: We present SGuard-v1, a lightweight safety guardrail for Large Language Models (LLMs), which comprises two specialized models to detect harmful content and screen adversarial prompts in human-AI conversational settings. The first component, ContentFilter, is trained to identify safety risks in LLM prompts and responses in accordance with the MLCommons hazard taxonomy, a comprehensive framework for trust and safety assessment of AI. The second component, JailbreakFilter, is trained with a carefully designed curriculum over integrated datasets and findings from prior work on adversarial prompting, covering 60 major attack types while mitigating false-unsafe classification. SGuard-v1 is built on the 2B-parameter Granite-3.3-2B-Instruct model that supports 12 languages. We curate approximately 1.4 million training instances from both collected and synthesized data and perform instruction tuning on the base model, distributing the curated data across the two component according to their designated functions. Through extensive evaluation on public and proprietary safety benchmarks, SGuard-v1 achieves state-of-the-art safety performance while remaining lightweight, thereby reducing deployment overhead. SGuard-v1 also improves interpretability for downstream use by providing multi-class safety predictions and their binary confidence scores. We release the SGuard-v1 under the Apache-2.0 License to enable further research and practical deployment in AI safety. 

---
# Probing Preference Representations: A Multi-Dimensional Evaluation and Analysis Method for Reward Models 

**Authors**: Chenglong Wang, Yifu Huo, Yang Gan, Yongyu Mu, Qiaozhi He, Murun Yang, Bei Li, Chunliang Zhang, Tongran Liu, Anxiang Ma, Zhengtao Yu, Jingbo Zhu, Tong Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2511.12464)  

**Abstract**: Previous methods evaluate reward models by testing them on a fixed pairwise ranking test set, but they typically do not provide performance information on each preference dimension. In this work, we address the evaluation challenge of reward models by probing preference representations. To confirm the effectiveness of this evaluation method, we construct a Multi-dimensional Reward Model Benchmark (MRMBench), a collection of six probing tasks for different preference dimensions. We design it to favor and encourage reward models that better capture preferences across different dimensions. Furthermore, we introduce an analysis method, inference-time probing, which identifies the dimensions used during the reward prediction and enhances its interpretability. Through extensive experiments, we find that MRMBench strongly correlates with the alignment performance of large language models (LLMs), making it a reliable reference for developing advanced reward models. Our analysis of MRMBench evaluation results reveals that reward models often struggle to capture preferences across multiple dimensions, highlighting the potential of multi-objective optimization in reward modeling. Additionally, our findings show that the proposed inference-time probing method offers a reliable metric for assessing the confidence of reward predictions, which ultimately improves the alignment of LLMs. 

---
# Consistency Is the Key: Detecting Hallucinations in LLM Generated Text By Checking Inconsistencies About Key Facts 

**Authors**: Raavi Gupta, Pranav Hari Panicker, Sumit Bhatia, Ganesh Ramakrishnan  

**Link**: [PDF](https://arxiv.org/pdf/2511.12236)  

**Abstract**: Large language models (LLMs), despite their remarkable text generation capabilities, often hallucinate and generate text that is factually incorrect and not grounded in real-world knowledge. This poses serious risks in domains like healthcare, finance, and customer support. A typical way to use LLMs is via the APIs provided by LLM vendors where there is no access to model weights or options to fine-tune the model. Existing methods to detect hallucinations in such settings where the model access is restricted or constrained by resources typically require making multiple LLM API calls, increasing latency and API cost. We introduce CONFACTCHECK, an efficient hallucination detection approach that does not leverage any external knowledge base and works on the simple intuition that responses to factual probes within the generated text should be consistent within a single LLM and across different LLMs. Rigorous empirical evaluation on multiple datasets that cover both the generation of factual texts and the open generation shows that CONFACTCHECK can detect hallucinated facts efficiently using fewer resources and achieves higher accuracy scores compared to existing baselines that operate under similar conditions. Our code is available here. 

---
# Do LLMs and Humans Find the Same Questions Difficult? A Case Study on Japanese Quiz Answering 

**Authors**: Naoya Sugiura, Kosuke Yamada, Yasuhiro Ogawa, Katsuhiko Toyama, Ryohei Sasano  

**Link**: [PDF](https://arxiv.org/pdf/2511.12300)  

**Abstract**: LLMs have achieved performance that surpasses humans in many NLP tasks. However, it remains unclear whether problems that are difficult for humans are also difficult for LLMs. This study investigates how the difficulty of quizzes in a buzzer setting differs between LLMs and humans. Specifically, we first collect Japanese quiz data including questions, answers, and correct response rate of humans, then prompted LLMs to answer the quizzes under several settings, and compare their correct answer rate to that of humans from two analytical perspectives. The experimental results showed that, compared to humans, LLMs struggle more with quizzes whose correct answers are not covered by Wikipedia entries, and also have difficulty with questions that require numerical answers. 

---
# LLMLagBench: Identifying Temporal Training Boundaries in Large Language Models 

**Authors**: Piotr Pęzik, Konrad Kaczyński, Maria Szymańska, Filip Żarnecki, Zuzanna Deckert, Jakub Kwiatkowski, Wojciech Janowski  

**Link**: [PDF](https://arxiv.org/pdf/2511.12116)  

**Abstract**: Large Language Models (LLMs) are pretrained on textual data up to a specific temporal cutoff. This creates a strict knowledge boundary beyond which models cannot provide accurate information without querying external sources. More subtly, when this limitation is unknown or ignored, LLMs may inadvertently blend outdated time-sensitive information with general knowledge during reasoning tasks, potentially compromising response accuracy. We introduce LLMLagBench, an LLM freshness benchmark, as a systematic approach for identifying the earliest probable temporal boundaries of an LLM's training data by evaluating its knowledge of recent events. We then apply this benchmark to evaluate a large set of LLMs, including models with both explicitly declared and undeclared training cutoffs. The reliability of the benchmark is assessed by manual validation and comparison with publicly released information about LLM pretraining. 

---
# From Phonemes to Meaning: Evaluating Large Language Models on Tamil 

**Authors**: Jeyarajalingam Varsha, Menan Velayuthan, Sumirtha Karunakaran, Rasan Nivethiga, Kengatharaiyer Sarveswaran  

**Link**: [PDF](https://arxiv.org/pdf/2511.12387)  

**Abstract**: Large Language Models (LLMs) have shown strong generalization across tasks in high-resource languages; however, their linguistic competence in low-resource and morphologically rich languages such as Tamil remains largely unexplored. Existing multilingual benchmarks often rely on translated English datasets, failing to capture the linguistic and cultural nuances of the target language. To address this gap, we introduce ILAKKANAM, the first Tamil-specific linguistic evaluation benchmark manually curated using 820 questions from Sri Lankan school-level Tamil subject examination papers. Each question is annotated by trained linguists under five linguistic categories and a factual knowledge category, spanning Grades 1--13 to ensure broad linguistic coverage. We evaluate both closed-source and open-source LLMs using a standardized evaluation framework. Our results show that Gemini 2.5 achieves the highest overall performance, while open-source models lag behind, highlighting the gap in linguistic grounding. Category- and grade-wise analyses reveal that all models perform well on lower-grade questions but show a clear decline as linguistic complexity increases. Further, no strong correlation is observed between a model's overall performance and its ability to identify linguistic categories, suggesting that performance may be driven by exposure rather than genuine understanding. 

---
# A Reasoning Paradigm for Named Entity Recognition 

**Authors**: Hui Huang, Yanping Chen, Ruizhang Huang, Chuan Lin, Yongbin Qin  

**Link**: [PDF](https://arxiv.org/pdf/2511.11978)  

**Abstract**: Generative LLMs typically improve Named Entity Recognition (NER) performance through instruction tuning. They excel at generating entities by semantic pattern matching but lack an explicit, verifiable reasoning mechanism. This "cognitive shortcutting" leads to suboptimal performance and brittle generalization, especially in zero-shot and lowresource scenarios where reasoning from limited contextual cues is crucial. To address this issue, a reasoning framework is proposed for NER, which shifts the extraction paradigm from implicit pattern matching to explicit reasoning. This framework consists of three stages: Chain of Thought (CoT) generation, CoT tuning, and reasoning enhancement. First, a dataset annotated with NER-oriented CoTs is generated, which contain task-relevant reasoning chains. Then, they are used to tune the NER model to generate coherent rationales before deriving the final answer. Finally, a reasoning enhancement stage is implemented to optimize the reasoning process using a comprehensive reward signal. This stage ensures explicit and verifiable extractions. Experiments show that ReasoningNER demonstrates impressive cognitive ability in the NER task, achieving competitive performance. In zero-shot settings, it achieves state-of-the-art (SOTA) performance, outperforming GPT-4 by 12.3 percentage points on the F1 score. Analytical results also demonstrate its great potential to advance research in reasoningoriented information extraction. Our codes are available at this https URL. 

---
# AI-Salesman: Towards Reliable Large Language Model Driven Telemarketing 

**Authors**: Qingyu Zhang, Chunlei Xin, Xuanang Chen, Yaojie Lu, Hongyu Lin, Xianpei Han, Le Sun, Qing Ye, Qianlong Xie, Xingxing Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.12133)  

**Abstract**: Goal-driven persuasive dialogue, exemplified by applications like telemarketing, requires sophisticated multi-turn planning and strict factual faithfulness, which remains a significant challenge for even state-of-the-art Large Language Models (LLMs). A lack of task-specific data often limits previous works, and direct LLM application suffers from strategic brittleness and factual hallucination. In this paper, we first construct and release TeleSalesCorpus, the first real-world-grounded dialogue dataset for this domain. We then propose AI-Salesman, a novel framework featuring a dual-stage architecture. For the training stage, we design a Bayesian-supervised reinforcement learning algorithm that learns robust sales strategies from noisy dialogues. For the inference stage, we introduce the Dynamic Outline-Guided Agent (DOGA), which leverages a pre-built script library to provide dynamic, turn-by-turn strategic guidance. Moreover, we design a comprehensive evaluation framework that combines fine-grained metrics for key sales skills with the LLM-as-a-Judge paradigm. Experimental results demonstrate that our proposed AI-Salesman significantly outperforms baseline models in both automatic metrics and comprehensive human evaluations, showcasing its effectiveness in complex persuasive scenarios. 

---
# Improving LLM's Attachment to External Knowledge In Dialogue Generation Tasks Through Entity Anonymization 

**Authors**: Hadi Sheikhi, Chenyang Huang, Osmar R. Zaïane  

**Link**: [PDF](https://arxiv.org/pdf/2511.11946)  

**Abstract**: Knowledge graph-based dialogue generation (KG-DG) is a challenging task requiring models to effectively incorporate external knowledge into conversational responses. While large language models (LLMs) have achieved impressive results across various NLP tasks, their ability to utilize external knowledge in KG-DG remains under-explored. We observe that LLMs often rely on internal knowledge, leading to detachment from provided knowledge graphs, even when they are given a flawlessly retrieved knowledge graph. First, we introduce LLM-KAT, an evaluation procedure for measuring knowledge attachment in generated responses. Second, we propose a simple yet effective entity anonymization technique to encourage LLMs to better leverage external knowledge. Experiments on the OpenDialKG dataset demonstrate that our approach improves LLMs' attachment on external knowledge. 

---
# CURE: Cultural Understanding and Reasoning Evaluation - A Framework for "Thick" Culture Alignment Evaluation in LLMs 

**Authors**: Truong Vo, Sanmi Koyejo  

**Link**: [PDF](https://arxiv.org/pdf/2511.12014)  

**Abstract**: Large language models (LLMs) are increasingly deployed in culturally diverse environments, yet existing evaluations of cultural competence remain limited. Existing methods focus on de-contextualized correctness or forced-choice judgments, overlooking the need for cultural understanding and reasoning required for appropriate responses. To address this gap, we introduce a set of benchmarks that, instead of directly probing abstract norms or isolated statements, present models with realistic situational contexts that require culturally grounded reasoning. In addition to the standard Exact Match metric, we introduce four complementary metrics (Coverage, Specificity, Connotation, and Coherence) to capture different dimensions of model's response quality. Empirical analysis across frontier models reveals that thin evaluation systematically overestimates cultural competence and produces unstable assessments with high variance. In contrast, thick evaluation exposes differences in reasoning depth, reduces variance, and provides more stable, interpretable signals of cultural understanding. 

---
# InData: Towards Secure Multi-Step, Tool-Based Data Analysis 

**Authors**: Karthikeyan K, Raghuveer Thirukovalluru, Bhuwan Dhingra, David Edwin Carlson  

**Link**: [PDF](https://arxiv.org/pdf/2511.11933)  

**Abstract**: Large language model agents for data analysis typically generate and execute code directly on databases. However, when applied to sensitive data, this approach poses significant security risks. To address this issue, we propose a security-motivated alternative: restrict LLMs from direct code generation and data access, and require them to interact with data exclusively through a predefined set of secure, verified tools. Although recent tool-use benchmarks exist, they primarily target tool selection and simple execution rather than the compositional, multi-step reasoning needed for complex data analysis. To reduce this gap, we introduce Indirect Data Engagement (InData), a dataset designed to assess LLMs' multi-step tool-based reasoning ability. InData includes data analysis questions at three difficulty levels--Easy, Medium, and Hard--capturing increasing reasoning complexity. We benchmark 15 open-source LLMs on InData and find that while large models (e.g., gpt-oss-120b) achieve high accuracy on Easy tasks (97.3%), performance drops sharply on Hard tasks (69.6%). These results show that current LLMs still lack robust multi-step tool-based reasoning ability. With InData, we take a step toward enabling the development and evaluation of LLMs with stronger multi-step tool-use capabilities. We will publicly release the dataset and code. 

---
# Seeing is Believing: Rich-Context Hallucination Detection for MLLMs via Backward Visual Grounding 

**Authors**: Pinxue Guo, Chongruo Wu, Xinyu Zhou, Lingyi Hong, Zhaoyu Chen, Jinglun Li, Kaixun Jiang, Sen-ching Samson Cheung, Wei Zhang, Wenqiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.12140)  

**Abstract**: Multimodal Large Language Models (MLLMs) have unlocked powerful cross-modal capabilities, but still significantly suffer from hallucinations. As such, accurate detection of hallucinations in MLLMs is imperative for ensuring their reliability in practical applications. To this end, guided by the principle of "Seeing is Believing", we introduce VBackChecker, a novel reference-free hallucination detection framework that verifies the consistency of MLLMgenerated responses with visual inputs, by leveraging a pixellevel Grounding LLM equipped with reasoning and referring segmentation capabilities. This reference-free framework not only effectively handles rich-context scenarios, but also offers interpretability. To facilitate this, an innovative pipeline is accordingly designed for generating instruction-tuning data (R-Instruct), featuring rich-context descriptions, grounding masks, and hard negative samples. We further establish R^2 -HalBench, a new hallucination benchmark for MLLMs, which, unlike previous benchmarks, encompasses real-world, rich-context descriptions from 18 MLLMs with high-quality annotations, spanning diverse object-, attribute, and relationship-level details. VBackChecker outperforms prior complex frameworks and achieves state-of-the-art performance on R^2 -HalBench, even rivaling GPT-4o's capabilities in hallucination detection. It also surpasses prior methods in the pixel-level grounding task, achieving over a 10% improvement. All codes, data, and models are available at this https URL. 

---
# ClinStructor: AI-Powered Structuring of Unstructured Clinical Texts 

**Authors**: Karthikeyan K, Raghuveer Thirukovalluru, David Carlson  

**Link**: [PDF](https://arxiv.org/pdf/2511.11883)  

**Abstract**: Clinical notes contain valuable, context-rich information, but their unstructured format introduces several challenges, including unintended biases (e.g., gender or racial bias), and poor generalization across clinical settings (e.g., models trained on one EHR system may perform poorly on another due to format differences) and poor interpretability. To address these issues, we present ClinStructor, a pipeline that leverages large language models (LLMs) to convert clinical free-text into structured, task-specific question-answer pairs prior to predictive modeling. Our method substantially enhances transparency and controllability and only leads to a modest reduction in predictive performance (a 2-3% drop in AUC), compared to direct fine-tuning, on the ICU mortality prediction task. ClinStructor lays a strong foundation for building reliable, interpretable, and generalizable machine learning models in clinical environments. 

---
# Towards Autoformalization of LLM-generated Outputs for Requirement Verification 

**Authors**: Mihir Gupte, Ramesh S  

**Link**: [PDF](https://arxiv.org/pdf/2511.11829)  

**Abstract**: Autoformalization, the process of translating informal statements into formal logic, has gained renewed interest with the emergence of powerful Large Language Models (LLMs). While LLMs show promise in generating structured outputs from natural language (NL), such as Gherkin Scenarios from NL feature requirements, there's currently no formal method to verify if these outputs are accurate. This paper takes a preliminary step toward addressing this gap by exploring the use of a simple LLM-based autoformalizer to verify LLM-generated outputs against a small set of natural language requirements. We conducted two distinct experiments. In the first one, the autoformalizer successfully identified that two differently-worded NL requirements were logically equivalent, demonstrating the pipeline's potential for consistency checks. In the second, the autoformalizer was used to identify a logical inconsistency between a given NL requirement and an LLM-generated output, highlighting its utility as a formal verification tool. Our findings, while limited, suggest that autoformalization holds significant potential for ensuring the fidelity and logical consistency of LLM-generated outputs, laying a crucial foundation for future, more extensive studies into this novel application. 

---
# Additive Large Language Models for Semi-Structured Text 

**Authors**: Karthikeyan K, Raghuveer Thirukovalluru, David Carlson  

**Link**: [PDF](https://arxiv.org/pdf/2511.11922)  

**Abstract**: Large Language Models have advanced clinical text classification, but their opaque predictions remain a critical barrier to practical adoption in research and clinical settings where investigators and physicians need to understand which parts of a patient's record drive risk signals. To address this challenge, we introduce \textbf{CALM}, short for \textbf{Classification with Additive Large Language Models}, an interpretable framework for semi-structured text where inputs are composed of semantically meaningful components, such as sections of an admission note or question-answer fields from an intake form. CALM predicts outcomes as the additive sum of each component's contribution, making these contributions part of the forward computation itself and enabling faithful explanations at both the patient and population level. The additive structure also enables clear visualizations, such as component-level risk curves similar to those used in generalized additive models, making the learned relationships easier to inspect and communicate. Although CALM expects semi-structured inputs, many clinical documents already have this form, and similar structure can often be automatically extracted from free-text notes. CALM achieves performance comparable to conventional LLM classifiers while improving trust, supporting quality-assurance checks, and revealing clinically meaningful patterns during model development and auditing. 

---
# Context-Emotion Aware Therapeutic Dialogue Generation: A Multi-component Reinforcement Learning Approach to Language Models for Mental Health Support 

**Authors**: Eric Hua Qing Zhang, Julia Ive  

**Link**: [PDF](https://arxiv.org/pdf/2511.11884)  

**Abstract**: Mental health illness represents a substantial global socioeconomic burden, with COVID-19 further exacerbating accessibility challenges and driving increased demand for telehealth mental health support. While large language models (LLMs) offer promising solutions through 24/7 availability and non-judgmental interactions, pre-trained models often lack the contextual and emotional awareness necessary for appropriate therapeutic responses. This paper investigated the application of supervised fine-tuning (SFT) and reinforcement learning (RL) techniques to enhance GPT-2's capacity for therapeutic dialogue generation. The methodology restructured input formats to enable simultaneous processing of contextual information and emotional states alongside user input, employing a multi-component reward function that aligned model outputs with professional therapist responses and annotated emotions. Results demonstrated improvements through reinforcement learning over baseline GPT-2 across multiple evaluation metrics: BLEU (0.0111), ROUGE-1 (0.1397), ROUGE-2 (0.0213), ROUGE-L (0.1317), and METEOR (0.0581). LLM evaluation confirmed high contextual relevance and professionalism, while reinforcement learning achieved 99.34% emotion accuracy compared to 66.96% for baseline GPT-2. These findings demonstrate reinforcement learning's effectiveness in developing therapeutic dialogue systems that can serve as valuable assistive tools for therapists while maintaining essential human clinical oversight. 

---
# MedPT: A Massive Medical Question Answering Dataset for Brazilian-Portuguese Speakers 

**Authors**: Fernanda Bufon Färber, Iago Alves Brito, Julia Soares Dollis, Pedro Schindler Freire Brasil Ribeiro, Rafael Teixeira Sousa, Arlindo Rodrigues Galvão Filho  

**Link**: [PDF](https://arxiv.org/pdf/2511.11878)  

**Abstract**: While large language models (LLMs) show transformative potential in healthcare, their development remains focused on high-resource languages, creating a critical barrier for others as simple translation fails to capture unique clinical and cultural nuances, such as endemic diseases. To address this, we introduce MedPT, the first large-scale, real-world corpus for Brazilian Portuguese, comprising 384,095 authentic question-answer pairs from patient-doctor interactions. The dataset underwent a meticulous multi-stage curation protocol, using a hybrid quantitative-qualitative analysis to filter noise and contextually enrich thousands of ambiguous queries. We further augmented the corpus via LLM-driven annotation, classifying questions into seven semantic types to capture user intent. Our analysis reveals its thematic breadth (3,200 topics) and unique linguistic properties, like the natural asymmetry in patient-doctor communication. To validate its utility, we benchmark a medical specialty routing task: fine-tuning a 1.7B parameter model achieves an outstanding 94\% F1-score on a 20-class setup. Furthermore, our qualitative error analysis shows misclassifications are not random but reflect genuine clinical ambiguities (e.g., between comorbid conditions), proving the dataset's deep semantic richness. We publicly release MedPT to foster the development of more equitable, accurate, and culturally-aware medical technologies for the Portuguese-speaking world. 

---
# TimeStampEval: A Simple LLM Eval and a Little Fuzzy Matching Trick to Improve Search Accuracy 

**Authors**: James McCammon  

**Link**: [PDF](https://arxiv.org/pdf/2511.11594)  

**Abstract**: Traditional fuzzy matching often fails when searching for quotes that are semantically identical but syntactically different across documents-a common issue when aligning official written records with speech-to-text transcripts. We introduce TimeStampEval, a benchmark for retrieving precise millisecond timestamps from long transcripts given non-verbatim quotes. Our simple two-stage method dramatically improves retrieval accuracy while cutting inference costs by over 90%. The motivating use case is an automated long-form podcast that assembles Congressional Record clips into AI-hosted narration. The technical challenge: given a sentence-timestamped transcript and a target quote that may differ due to transcription or editorial drift, return exact start and end boundaries. Standard algorithms handle verbatim text but break under fuzzier variants. Evaluating six modern LLMs on a 2,800-sentence (120k-token) transcript revealed four key findings. (1) Prompt design matters more than model choice: placing the query before the transcript and using compact formatting improved accuracy by 3-20 points while reducing token count by 30-40%. (2) Off-by-one errors form a distinct category, showing models understand the task but misplace boundaries. (3) A modest reasoning budget (600-850 tokens) raises accuracy from 37% to 77% for weak setups and to above 90% for strong ones. (4) Our "Assisted Fuzzy" approach-RapidFuzz pre-filtering followed by LLM verification on short snippets-improves fuzzy match accuracy by up to 50 points while halving latency and reducing cost per correct result by up to 96%. Extended tests on ten transcripts (50k-900k tokens, 1989-2025) confirm robustness to transcript length, vocabulary drift, and domain change, maintaining 95-100% rejection accuracy for absent targets. 

---
# P1: Mastering Physics Olympiads with Reinforcement Learning 

**Authors**: Jiacheng Chen, Qianjia Cheng, Fangchen Yu, Haiyuan Wan, Yuchen Zhang, Shenghe Zheng, Junchi Yao, Qingyang Zhang, Haonan He, Yun Luo, Yufeng Zhao, Futing Wang, Li Sheng, Chengxing Xie, Yuxin Zuo, Yizhuo Li, Wenxauan Zeng, Yulun Wu, Rui Huang, Dongzhan Zhou, Kai Chen, Yu Qiao, Lei Bai, Yu Cheng, Ning Ding, Bowen Zhou, Peng Ye, Ganqu Cui  

**Link**: [PDF](https://arxiv.org/pdf/2511.13612)  

**Abstract**: Recent progress in large language models (LLMs) has moved the frontier from puzzle-solving to science-grade reasoning-the kind needed to tackle problems whose answers must stand against nature, not merely fit a rubric. Physics is the sharpest test of this shift, which binds symbols to reality in a fundamental way, serving as the cornerstone of most modern technologies. In this work, we manage to advance physics research by developing large language models with exceptional physics reasoning capabilities, especially excel at solving Olympiad-level physics problems. We introduce P1, a family of open-source physics reasoning models trained entirely through reinforcement learning (RL). Among them, P1-235B-A22B is the first open-source model with Gold-medal performance at the latest International Physics Olympiad (IPhO 2025), and wins 12 gold medals out of 13 international/regional physics competitions in 2024/2025. P1-30B-A3B also surpasses almost all other open-source models on IPhO 2025, getting a silver medal. Further equipped with an agentic framework PhysicsMinions, P1-235B-A22B+PhysicsMinions achieves overall No.1 on IPhO 2025, and obtains the highest average score over the 13 physics competitions. Besides physics, P1 models also present great performance on other reasoning tasks like math and coding, showing the great generalibility of P1 series. 

---
# ForgeDAN: An Evolutionary Framework for Jailbreaking Aligned Large Language Models 

**Authors**: Siyang Cheng, Gaotian Liu, Rui Mei, Yilin Wang, Kejia Zhang, Kaishuo Wei, Yuqi Yu, Weiping Wen, Xiaojie Wu, Junhua Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.13548)  

**Abstract**: The rapid adoption of large language models (LLMs) has brought both transformative applications and new security risks, including jailbreak attacks that bypass alignment safeguards to elicit harmful outputs. Existing automated jailbreak generation approaches e.g. AutoDAN, suffer from limited mutation diversity, shallow fitness evaluation, and fragile keyword-based detection. To address these limitations, we propose ForgeDAN, a novel evolutionary framework for generating semantically coherent and highly effective adversarial prompts against aligned LLMs. First, ForgeDAN introduces multi-strategy textual perturbations across \textit{character, word, and sentence-level} operations to enhance attack diversity; then we employ interpretable semantic fitness evaluation based on a text similarity model to guide the evolutionary process toward semantically relevant and harmful outputs; finally, ForgeDAN integrates dual-dimensional jailbreak judgment, leveraging an LLM-based classifier to jointly assess model compliance and output harmfulness, thereby reducing false positives and improving detection effectiveness. Our evaluation demonstrates ForgeDAN achieves high jailbreaking success rates while maintaining naturalness and stealth, outperforming existing SOTA solutions. 

---
# Dropouts in Confidence: Moral Uncertainty in Human-LLM Alignment 

**Authors**: Jea Kwon, Luiz Felipe Vecchietti, Sungwon Park, Meeyoung Cha  

**Link**: [PDF](https://arxiv.org/pdf/2511.13290)  

**Abstract**: Humans display significant uncertainty when confronted with moral dilemmas, yet the extent of such uncertainty in machines and AI agents remains underexplored. Recent studies have confirmed the overly confident tendencies of machine-generated responses, particularly in large language models (LLMs). As these systems are increasingly embedded in ethical decision-making scenarios, it is important to understand their moral reasoning and the inherent uncertainties in building reliable AI systems. This work examines how uncertainty influences moral decisions in the classical trolley problem, analyzing responses from 32 open-source models and 9 distinct moral dimensions. We first find that variance in model confidence is greater across models than within moral dimensions, suggesting that moral uncertainty is predominantly shaped by model architecture and training method. To quantify uncertainty, we measure binary entropy as a linear combination of total entropy, conditional entropy, and mutual information. To examine its effects, we introduce stochasticity into models via "dropout" at inference time. Our findings show that our mechanism increases total entropy, mainly through a rise in mutual information, while conditional entropy remains largely unchanged. Moreover, this mechanism significantly improves human-LLM moral alignment, with correlations in mutual information and alignment score shifts. Our results highlight the potential to better align model-generated decisions and human preferences by deliberately modulating uncertainty and reducing LLMs' confidence in morally complex scenarios. 

---
# PragWorld: A Benchmark Evaluating LLMs' Local World Model under Minimal Linguistic Alterations and Conversational Dynamics 

**Authors**: Sachin Vashistha, Aryan Bibhuti, Atharva Naik, Martin Tutek, Somak Aditya  

**Link**: [PDF](https://arxiv.org/pdf/2511.13021)  

**Abstract**: Real-world conversations are rich with pragmatic elements, such as entity mentions, references, and implicatures. Understanding such nuances is a requirement for successful natural communication, and often requires building a local world model which encodes such elements and captures the dynamics of their evolving states. However, it is not well-understood whether language models (LMs) construct or maintain a robust implicit representation of conversations. In this work, we evaluate the ability of LMs to encode and update their internal world model in dyadic conversations and test their malleability under linguistic alterations. To facilitate this, we apply seven minimal linguistic alterations to conversations sourced from popular datasets and construct two benchmarks comprising yes-no questions. We evaluate a wide range of open and closed source LMs and observe that they struggle to maintain robust accuracy. Our analysis unveils that LMs struggle to memorize crucial details, such as tracking entities under linguistic alterations to conversations. We then propose a dual-perspective interpretability framework which identifies transformer layers that are useful or harmful and highlights linguistic alterations most influenced by harmful layers, typically due to encoding spurious signals or relying on shortcuts. Inspired by these insights, we propose two layer-regularization based fine-tuning strategies that suppress the effect of the harmful layers. 

---
# Co-Layout: LLM-driven Co-optimization for Interior Layout 

**Authors**: Chucheng Xiang, Ruchao Bao, Biyin Feng, Wenzheng Wu, Zhongyuan Liu, Yirui Guan, Ligang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.12474)  

**Abstract**: We present a novel framework for automated interior design that combines large language models (LLMs) with grid-based integer programming to jointly optimize room layout and furniture placement. Given a textual prompt, the LLM-driven agent workflow extracts structured design constraints related to room configurations and furniture arrangements. These constraints are encoded into a unified grid-based representation inspired by ``Modulor". Our formulation accounts for key design requirements, including corridor connectivity, room accessibility, spatial exclusivity, and user-specified preferences. To improve computational efficiency, we adopt a coarse-to-fine optimization strategy that begins with a low-resolution grid to solve a simplified problem and guides the solution at the full resolution. Experimental results across diverse scenarios demonstrate that our joint optimization approach significantly outperforms existing two-stage design pipelines in solution quality, and achieves notable computational efficiency through the coarse-to-fine strategy. 

---
# A Content-Preserving Secure Linguistic Steganography 

**Authors**: Lingyun Xiang, Chengfu Ou, Xu He, Zhongliang Yang, Yuling Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.12565)  

**Abstract**: Existing linguistic steganography methods primarily rely on content transformations to conceal secret messages. However, they often cause subtle yet looking-innocent deviations between normal and stego texts, posing potential security risks in real-world applications. To address this challenge, we propose a content-preserving linguistic steganography paradigm for perfectly secure covert communication without modifying the cover text. Based on this paradigm, we introduce CLstega (\textit{C}ontent-preserving \textit{L}inguistic \textit{stega}nography), a novel method that embeds secret messages through controllable distribution transformation. CLstega first applies an augmented masking strategy to locate and mask embedding positions, where MLM(masked language model)-predicted probability distributions are easily adjustable for transformation. Subsequently, a dynamic distribution steganographic coding strategy is designed to encode secret messages by deriving target distributions from the original probability distributions. To achieve this transformation, CLstega elaborately selects target words for embedding positions as labels to construct a masked sentence dataset, which is used to fine-tune the original MLM, producing a target MLM capable of directly extracting secret messages from the cover text. This approach ensures perfect security of secret messages while fully preserving the integrity of the original cover text. Experimental results show that CLstega can achieve a 100\% extraction success rate, and outperforms existing methods in security, effectively balancing embedding capacity and security. 

---
# Forgetting-MarI: LLM Unlearning via Marginal Information Regularization 

**Authors**: Shizhou Xu, Yuan Ni, Stefan Broecker, Thomas Strohmer  

**Link**: [PDF](https://arxiv.org/pdf/2511.11914)  

**Abstract**: As AI models are trained on ever-expanding datasets, the ability to remove the influence of specific data from trained models has become essential for privacy protection and regulatory compliance. Unlearning addresses this challenge by selectively removing parametric knowledge from the trained models without retraining from scratch, which is critical for resource-intensive models such as Large Language Models (LLMs). Existing unlearning methods often degrade model performance by removing more information than necessary when attempting to ''forget'' specific data. We introduce Forgetting-MarI, an LLM unlearning framework that provably removes only the additional (marginal) information contributed by the data to be unlearned, while preserving the information supported by the data to be retained. By penalizing marginal information, our method yields an explicit upper bound on the unlearn dataset's residual influence in the trained models, providing provable undetectability. Extensive experiments confirm that our approach outperforms current state-of-the-art unlearning methods, delivering reliable forgetting and better preserved general model performance across diverse benchmarks. This advancement represents an important step toward making AI systems more controllable and compliant with privacy and copyright regulations without compromising their effectiveness. 

---
# Better LLM Reasoning via Dual-Play 

**Authors**: Zhengxin Zhang, Chengyu Huang, Aochong Oliver Li, Claire Cardie  

**Link**: [PDF](https://arxiv.org/pdf/2511.11881)  

**Abstract**: Large Language Models (LLMs) have achieved remarkable progress through Reinforcement Learning with Verifiable Rewards (RLVR), yet still rely heavily on external supervision (e.g., curated labels). Adversarial learning, particularly through self-play, offers a promising alternative that enables models to iteratively learn from themselves - thus reducing reliance on external supervision. Dual-play extends adversarial learning by assigning specialized roles to two models and training them against each other, fostering sustained competition and mutual evolution. Despite its promise, adapting dual-play training to LLMs remains limited, largely due to their susceptibility to reward hacking and training instability. In this paper, we introduce PasoDoble, a novel LLM dual-play framework. PasoDoble adversarially trains two models initialized from the same base model: a Proposer, which generates challenging questions with ground-truth answers, and a Solver, which attempts to solve them. We enrich the Proposer with knowledge from a pre-training dataset to ensure the questions' quality and diversity. To avoid reward hacking, the Proposer is rewarded for producing only valid questions that push the Solver's limit, while the Solver is rewarded for solving them correctly, and both are updated jointly. To further enhance training stability, we introduce an optional offline paradigm that decouples Proposer and Solver updates, alternately updating each for several steps while holding the other fixed. Notably, PasoDoble operates without supervision during training. Experimental results show that PasoDoble can improve the reasoning performance of LLMs. Our project page is available at this https URL. 

---
# Do LLMs Really Struggle at NL-FOL Translation? Revealing their Strengths via a Novel Benchmarking Strategy 

**Authors**: Andrea Brunello, Luca Geatti, Michele Mignani, Angelo Montanari, Nicola Saccomanno  

**Link**: [PDF](https://arxiv.org/pdf/2511.11816)  

**Abstract**: Due to its expressiveness and unambiguous nature, First-Order Logic (FOL) is a powerful formalism for representing concepts expressed in natural language (NL). This is useful, e.g., for specifying and verifying desired system properties. While translating FOL into human-readable English is relatively straightforward, the inverse problem, converting NL to FOL (NL-FOL translation), has remained a longstanding challenge, for both humans and machines. Although the emergence of Large Language Models (LLMs) promised a breakthrough, recent literature provides contrasting results on their ability to perform NL-FOL translation. In this work, we provide a threefold contribution. First, we critically examine existing datasets and protocols for evaluating NL-FOL translation performance, revealing key limitations that may cause a misrepresentation of LLMs' actual capabilities. Second, to overcome these shortcomings, we propose a novel evaluation protocol explicitly designed to distinguish genuine semantic-level logical understanding from superficial pattern recognition, memorization, and dataset contamination. Third, using this new approach, we show that state-of-the-art, dialogue-oriented LLMs demonstrate strong NL-FOL translation skills and a genuine grasp of sentence-level logic, whereas embedding-centric models perform markedly worse. 

---
# Generative AI as a Linguistic Equalizer in Global Science 

**Authors**: Dragan Filimonovic, Christian Rutzer, Jeffrey Macher, Rolf Weder  

**Link**: [PDF](https://arxiv.org/pdf/2511.11687)  

**Abstract**: For decades, the dominance of English has created a substantial barrier in global science, disadvantaging non-native speakers. The recent rise of generative AI (GenAI) offers a potential technological response to this long-standing inequity. We provide the first large-scale evidence testing whether GenAI acts as a linguistic equalizer in global science. Drawing on 5.65 million scientific articles published from 2021 to 2024, we compare GenAI-assisted and non-assisted publications from authors in non-English-speaking countries. Using text embeddings derived from a pretrained large language model (SciBERT), we measure each publication's linguistic similarity to a benchmark of scientific writing from U.S.-based authors and track stylistic convergence over time. We find significant and growing convergence for GenAI-assisted publications after the release of ChatGPT in late 2022. The effect is strongest for domestic coauthor teams from countries linguistically distant from English. These findings provide large-scale evidence that GenAI is beginning to reshape global science communication by reducing language barriers in research. 

---
# VoiceCraft-X: Unifying Multilingual, Voice-Cloning Speech Synthesis and Speech Editing 

**Authors**: Zhisheng Zheng, Puyuan Peng, Anuj Diwan, Cong Phuoc Huynh, Xiaohang Sun, Zhu Liu, Vimal Bhat, David Harwath  

**Link**: [PDF](https://arxiv.org/pdf/2511.12347)  

**Abstract**: We introduce VoiceCraft-X, an autoregressive neural codec language model which unifies multilingual speech editing and zero-shot Text-to-Speech (TTS) synthesis across 11 languages: English, Mandarin, Korean, Japanese, Spanish, French, German, Dutch, Italian, Portuguese, and Polish. VoiceCraft-X utilizes the Qwen3 large language model for phoneme-free cross-lingual text processing and a novel token reordering mechanism with time-aligned text and speech tokens to handle both tasks as a single sequence generation problem. The model generates high-quality, natural-sounding speech, seamlessly creating new audio or editing existing recordings within one framework. VoiceCraft-X shows robust performance in diverse linguistic settings, even with limited per-language data, underscoring the power of unified autoregressive approaches for advancing complex, real-world multilingual speech applications. Audio samples are available at this https URL. 

---
# Leveraging Large Language Models for Career Mobility Analysis: A Study of Gender, Race, and Job Change Using U.S. Online Resume Profiles 

**Authors**: Palakorn Achananuparp, Connie Xu, Yao Lu, Xavier Jayaraj Siddarth Ashok, Ee-Peng Lim  

**Link**: [PDF](https://arxiv.org/pdf/2511.12010)  

**Abstract**: We present a large-scale analysis of career mobility of college-educated U.S. workers using online resume profiles to investigate how gender, race, and job change options are associated with upward mobility. This study addresses key research questions of how the job changes affect their upward career mobility, and how the outcomes of upward career mobility differ by gender and race. We address data challenges -- such as missing demographic attributes, missing wage data, and noisy occupation labels -- through various data processing and Artificial Intelligence (AI) methods. In particular, we develop a large language models (LLMs) based occupation classification method known as FewSOC that achieves accuracy significantly higher than the original occupation labels in the resume dataset. Analysis of 228,710 career trajectories reveals that intra-firm occupation change has been found to facilitate upward mobility most strongly, followed by inter-firm occupation change and inter-firm lateral move. Women and Black college graduates experience significantly lower returns from job changes than men and White peers. Multilevel sensitivity analyses confirm that these disparities are robust to cluster-level heterogeneity and reveal additional intersectional patterns. 

---
# Reasoning: From Reflection to Solution 

**Authors**: Zixi Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.11712)  

**Abstract**: What is reasoning? This question has driven centuries of philosophical inquiry, from Aristotle's syllogisms to modern computational complexity theory. In the age of large language models achieving superhuman performance on benchmarks like GSM8K (95\% accuracy) and HumanEval (90\% pass@1), we must ask: have these systems learned to \emph{reason}, or have they learned to \emph{pattern-match over reasoning traces}?
This paper argues for a specific answer: \textbf{reasoning is iterative operator application in state spaces, converging to fixed points}. This definition is not merely philosophical -- it has concrete architectural implications that explain both the failures of current systems and the path to genuine reasoning capabilities.
Our investigation begins with a puzzle (OpenXOR), progresses through theory (OpenOperator), and culminates in a working solution (OpenLM) that achieves 76\% accuracy where state-of-the-art LLMs achieve 0\%. This is not about criticizing existing systems, but about \emph{understanding what reasoning requires} and \emph{building architectures that provide it}. 

---
# Automatic generation of DRI Statements 

**Authors**: Maurice Flechtner  

**Link**: [PDF](https://arxiv.org/pdf/2511.11655)  

**Abstract**: Assessing the quality of group deliberation is essential for improving our understanding of deliberative processes. The Deliberative Reason Index (DRI) offers a sophisticated metric for evaluating group reasoning, but its implementation has been constrained by the complex and time-consuming process of statement generation. This thesis introduces an innovative, automated approach to DRI statement generation that leverages advanced natural language processing (NLP) and large language models (LLMs) to substantially reduce the human effort involved in survey preparation. Key contributions are a systematic framework for automated DRI statement generation and a methodological innovation that significantly lowers the barrier to conducting comprehensive deliberative process assessments. In addition, the findings provide a replicable template for integrating generative artificial intelligence into social science research methodologies. 

---
# SynBullying: A Multi LLM Synthetic Conversational Dataset for Cyberbullying Detectio 

**Authors**: Arefeh Kazemi, Hamza Qadeer, Joachim Wagner, Hossein Hosseini, Sri Balaaji Natarajan Kalaivendan, Brian Davis  

**Link**: [PDF](https://arxiv.org/pdf/2511.11599)  

**Abstract**: We introduce SynBullying, a synthetic multi-LLM conversational dataset for studying and detecting cyberbullying (CB). SynBullying provides a scalable and ethically safe alternative to human data collection by leveraging large language models (LLMs) to simulate realistic bullying interactions. The dataset offers (i) conversational structure, capturing multi-turn exchanges rather than isolated posts; (ii) context-aware annotations, where harmfulness is assessed within the conversational flow considering context, intent, and discourse dynamics; and (iii) fine-grained labeling, covering various CB categories for detailed linguistic and behavioral analysis. We evaluate SynBullying across five dimensions, including conversational structure, lexical patterns, sentiment/toxicity, role dynamics, harm intensity, and CB-type distribution. We further examine its utility by testing its performance as standalone training data and as an augmentation source for CB classification. 

---
# LLM-Generated Negative News Headlines Dataset: Creation and Benchmarking Against Real Journalism 

**Authors**: Olusola Babalola, Bolanle Ojokoh, Olutayo Boyinbode  

**Link**: [PDF](https://arxiv.org/pdf/2511.11591)  

**Abstract**: This research examines the potential of datasets generated by Large Language Models (LLMs) to support Natural Language Processing (NLP) tasks, aiming to overcome challenges related to data acquisition and privacy concerns associated with real-world data. Focusing on negative valence text, a critical component of sentiment analysis, we explore the use of LLM-generated synthetic news headlines as an alternative to real-world data. A specialized corpus of negative news headlines was created using tailored prompts to capture diverse negative sentiments across various societal domains. The synthetic headlines were validated by expert review and further analyzed in embedding space to assess their alignment with real-world negative news in terms of content, tone, length, and style. Key metrics such as correlation with real headlines, perplexity, coherence, and realism were evaluated. The synthetic dataset was benchmarked against two sets of real news headlines using evaluations including the Comparative Perplexity Test, Comparative Readability Test, Comparative POS Profiling, BERTScore, and Comparative Semantic Similarity. Results show the generated headlines match real headlines with the only marked divergence being in the proper noun score of the POS profile test. 

---
# CreBench: Human-Aligned Creativity Evaluation from Idea to Process to Product 

**Authors**: Kaiwen Xue, Chenglong Li, Zhonghong Ou, Guoxin Zhang, Kaoyan Lu, Shuai Lyu, Yifan Zhu, Ping Zong Junpeng Ding, Xinyu Liu, Qunlin Chen, Weiwei Qin, Yiran Shen, Jiayi Cen  

**Link**: [PDF](https://arxiv.org/pdf/2511.13626)  

**Abstract**: Human-defined creativity is highly abstract, posing a challenge for multimodal large language models (MLLMs) to comprehend and assess creativity that aligns with human judgments. The absence of an existing benchmark further exacerbates this dilemma. To this end, we propose CreBench, which consists of two key components: 1) an evaluation benchmark covering the multiple dimensions from creative idea to process to products; 2) CreMIT (Creativity Multimodal Instruction Tuning dataset), a multimodal creativity evaluation dataset, consisting of 2.2K diverse-sourced multimodal data, 79.2K human feedbacks and 4.7M multi-typed instructions. Specifically, to ensure MLLMs can handle diverse creativity-related queries, we prompt GPT to refine these human feedbacks to activate stronger creativity assessment capabilities. CreBench serves as a foundation for building MLLMs that understand human-aligned creativity. Based on the CreBench, we fine-tune open-source general MLLMs, resulting in CreExpert, a multimodal creativity evaluation expert model. Extensive experiments demonstrate that the proposed CreExpert models achieve significantly better alignment with human creativity evaluation compared to state-of-the-art MLLMs, including the most advanced GPT-4V and Gemini-Pro-Vision. 

---
# Automated Construction of Medical Indicator Knowledge Graphs Using Retrieval Augmented Large Language Models 

**Authors**: Zhengda Wang, Daqian Shi, Jingyi Zhao, Xiaolei Diao, Xiongfeng Tang, Yanguo Qin  

**Link**: [PDF](https://arxiv.org/pdf/2511.13526)  

**Abstract**: Artificial intelligence (AI) is reshaping modern healthcare by advancing disease diagnosis, treatment decision-making, and biomedical research. Among AI technologies, large language models (LLMs) have become especially impactful, enabling deep knowledge extraction and semantic reasoning from complex medical texts. However, effective clinical decision support requires knowledge in structured, interoperable formats. Knowledge graphs serve this role by integrating heterogeneous medical information into semantically consistent networks. Yet, current clinical knowledge graphs still depend heavily on manual curation and rule-based extraction, which is limited by the complexity and contextual ambiguity of medical guidelines and literature. To overcome these challenges, we propose an automated framework that combines retrieval-augmented generation (RAG) with LLMs to construct medical indicator knowledge graphs. The framework incorporates guideline-driven data acquisition, ontology-based schema design, and expert-in-the-loop validation to ensure scalability, accuracy, and clinical reliability. The resulting knowledge graphs can be integrated into intelligent diagnosis and question-answering systems, accelerating the development of AI-driven healthcare solutions. 

---
# FreeAskWorld: An Interactive and Closed-Loop Simulator for Human-Centric Embodied AI 

**Authors**: Yuhang Peng, Yizhou Pan, Xinning He, Jihaoyu Yang, Xinyu Yin, Han Wang, Xiaoji Zheng, Chao Gao, Jiangtao Gong  

**Link**: [PDF](https://arxiv.org/pdf/2511.13524)  

**Abstract**: As embodied intelligence emerges as a core frontier in artificial intelligence research, simulation platforms must evolve beyond low-level physical interactions to capture complex, human-centered social behaviors. We introduce FreeAskWorld, an interactive simulation framework that integrates large language models (LLMs) for high-level behavior planning and semantically grounded interaction, informed by theories of intention and social cognition. Our framework supports scalable, realistic human-agent simulations and includes a modular data generation pipeline tailored for diverse embodied this http URL validate the framework, we extend the classic Vision-and-Language Navigation (VLN) task into a interaction enriched Direction Inquiry setting, wherein agents can actively seek and interpret navigational guidance. We present and publicly release FreeAskWorld, a large-scale benchmark dataset comprising reconstructed environments, six diverse task types, 16 core object categories, 63,429 annotated sample frames, and more than 17 hours of interaction data to support training and evaluation of embodied AI systems. We benchmark VLN models, and human participants under both open-loop and closed-loop settings. Experimental results demonstrate that models fine-tuned on FreeAskWorld outperform their original counterparts, achieving enhanced semantic understanding and interaction competency. These findings underscore the efficacy of socially grounded simulation frameworks in advancing embodied AI systems toward sophisticated high-level planning and more naturalistic human-agent interaction. Importantly, our work underscores that interaction itself serves as an additional information modality. 

---
# Reasoning Shapes Alignment: Investigating Cultural Alignment in Large Reasoning Models with Cultural Norms 

**Authors**: Yuhang Wang, Yanxu Zhu, Jitao Sang  

**Link**: [PDF](https://arxiv.org/pdf/2511.13359)  

**Abstract**: The advanced reasoning capabilities of Large Reasoning Models enable them to thoroughly understand and apply safety policies through deliberate thought processes, thereby improving the models' safety. Beyond safety, these models must also be able to reflect the diverse range of human values across various cultures. This paper presents the Cultural Norm-based Cultural Alignment (CNCA) framework, which enables models to leverage their powerful reasoning ability to align with cultural norms. Specifically, we propose three methods to automatically mine cultural norms from limited survey data and explore ways to effectively utilize these norms for improving cultural alignment. Two alignment paradigms are examined: an in-context alignment method, where cultural norms are explicitly integrated into the user context, and a fine-tuning-based method, which internalizes norms through enhanced Chain-of-Thought training data. Comprehensive experiments demonstrate the effectiveness of these methods, highlighting that models with stronger reasoning capabilities benefit more from cultural norm mining and utilization. Our findings emphasize the potential for reasoning models to better reflect diverse human values through culturally informed alignment strategies. 

---
# Multi-Agent Multimodal Large Language Model Framework for Automated Interpretation of Fuel Efficiency Analytics in Public Transportation 

**Authors**: Zhipeng Ma, Ali Rida Bahja, Andreas Burgdorf, André Pomp, Tobias Meisen, Bo Nørregaard Jørgensen, Zheng Grace Ma  

**Link**: [PDF](https://arxiv.org/pdf/2511.13476)  

**Abstract**: Enhancing fuel efficiency in public transportation requires the integration of complex multimodal data into interpretable, decision-relevant insights. However, traditional analytics and visualization methods often yield fragmented outputs that demand extensive human interpretation, limiting scalability and consistency. This study presents a multi-agent framework that leverages multimodal large language models (LLMs) to automate data narration and energy insight generation. The framework coordinates three specialized agents, including a data narration agent, an LLM-as-a-judge agent, and an optional human-in-the-loop evaluator, to iteratively transform analytical artifacts into coherent, stakeholder-oriented reports. The system is validated through a real-world case study on public bus transportation in Northern Jutland, Denmark, where fuel efficiency data from 4006 trips are analyzed using Gaussian Mixture Model clustering. Comparative experiments across five state-of-the-art LLMs and three prompting paradigms identify GPT-4.1 mini with Chain-of-Thought prompting as the optimal configuration, achieving 97.3% narrative accuracy while balancing interpretability and computational cost. The findings demonstrate that multi-agent orchestration significantly enhances factual precision, coherence, and scalability in LLM-based reporting. The proposed framework establishes a replicable and domain-adaptive methodology for AI-driven narrative generation and decision support in energy informatics. 

---
# Grounded by Experience: Generative Healthcare Prediction Augmented with Hierarchical Agentic Retrieval 

**Authors**: Chuang Zhao, Hui Tang, Hongke Zhao, Xiaofang Zhou, Xiaomeng Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.13293)  

**Abstract**: Accurate healthcare prediction is critical for improving patient outcomes and reducing operational costs. Bolstered by growing reasoning capabilities, large language models (LLMs) offer a promising path to enhance healthcare predictions by drawing on their rich parametric knowledge. However, LLMs are prone to factual inaccuracies due to limitations in the reliability and coverage of their embedded knowledge. While retrieval-augmented generation (RAG) frameworks, such as GraphRAG and its variants, have been proposed to mitigate these issues by incorporating external knowledge, they face two key challenges in the healthcare scenario: (1) identifying the clinical necessity to activate the retrieval mechanism, and (2) achieving synergy between the retriever and the generator to craft contextually appropriate retrievals. To address these challenges, we propose GHAR, a \underline{g}enerative \underline{h}ierarchical \underline{a}gentic \underline{R}AG framework that simultaneously resolves when to retrieve and how to optimize the collaboration between submodules in healthcare. Specifically, for the first challenge, we design a dual-agent architecture comprising Agent-Top and Agent-Low. Agent-Top acts as the primary physician, iteratively deciding whether to rely on parametric knowledge or to initiate retrieval, while Agent-Low acts as the consulting service, summarising all task-relevant knowledge once retrieval was triggered. To tackle the second challenge, we innovatively unify the optimization of both agents within a formal Markov Decision Process, designing diverse rewards to align their shared goal of accurate prediction while preserving their distinct roles. Extensive experiments on three benchmark datasets across three popular tasks demonstrate our superiority over state-of-the-art baselines, highlighting the potential of hierarchical agentic RAG in advancing healthcare systems. 

---
# MM-Telco: Benchmarks and Multimodal Large Language Models for Telecom Applications 

**Authors**: Gagan Raj Gupta, Anshul Kumar, Manish Rai, Apu Chakraborty, Ashutosh Modi, Abdelaali Chaoub, Soumajit Pramanik, Moyank Giri, Yashwanth Holla, Sunny Kumar, M. V. Kiran Sooraj  

**Link**: [PDF](https://arxiv.org/pdf/2511.13131)  

**Abstract**: Large Language Models (LLMs) have emerged as powerful tools for automating complex reasoning and decision-making tasks. In telecommunications, they hold the potential to transform network optimization, automate troubleshooting, enhance customer support, and ensure regulatory compliance. However, their deployment in telecom is hindered by domain-specific challenges that demand specialized adaptation. To overcome these challenges and to accelerate the adaptation of LLMs for telecom, we propose MM-Telco, a comprehensive suite of multimodal benchmarks and models tailored for the telecom domain. The benchmark introduces various tasks (both text based and image based) that address various practical real-life use cases such as network operations, network management, improving documentation quality, and retrieval of relevant text and images. Further, we perform baseline experiments with various LLMs and VLMs. The models fine-tuned on our dataset exhibit a significant boost in performance. Our experiments also help analyze the weak areas in the working of current state-of-art multimodal LLMs, thus guiding towards further development and research. 

---
# Scaling Generative Verifiers For Natural Language Mathematical Proof Verification And Selection 

**Authors**: Sadegh Mahdavi, Branislav Kisacanin, Shubham Toshniwal, Wei Du, Ivan Moshkov, George Armstrong, Renjie Liao, Christos Thrampoulidis, Igor Gitman  

**Link**: [PDF](https://arxiv.org/pdf/2511.13027)  

**Abstract**: Large language models have achieved remarkable success on final-answer mathematical problems, largely due to the ease of applying reinforcement learning with verifiable rewards. However, the reasoning underlying these solutions is often flawed. Advancing to rigorous proof-based mathematics requires reliable proof verification capabilities. We begin by analyzing multiple evaluation setups and show that focusing on a single benchmark can lead to brittle or misleading conclusions. To address this, we evaluate both proof-based and final-answer reasoning to obtain a more reliable measure of model performance. We then scale two major generative verification methods (GenSelect and LLM-as-a-Judge) to millions of tokens and identify their combination as the most effective framework for solution verification and selection. We further show that the choice of prompt for LLM-as-a-Judge significantly affects the model's performance, but reinforcement learning can reduce this sensitivity. However, despite improving proof-level metrics, reinforcement learning does not enhance final-answer precision, indicating that current models often reward stylistic or procedural correctness rather than mathematical validity. Our results establish practical guidelines for designing and evaluating scalable proof-verification and selection systems. 

---
# MedRule-KG: A Knowledge-Graph--Steered Scaffold for Reliable Mathematical and Biomedical Reasoning 

**Authors**: Crystal Su  

**Link**: [PDF](https://arxiv.org/pdf/2511.12963)  

**Abstract**: We study how to impose domain-consistent structure on large language models (LLMs) used for scientific reasoning and early-stage drug discovery. We present MedRule-KG, a compact knowledge-graph scaffold paired with a lightweight verifier that steers generation toward mathematically and biomedically valid outputs. The system injects curated symbolic facts into prompts and then enforces rule satisfaction with a deterministic checker. We formalize generation as constrained inference, introduce a soft guidance surrogate suitable for decoding, and perform a thorough statistical analysis with uncertainty quantification. Across 90 tasks spanning reaction feasibility, metabolic compatibility, and toxicity screening, MedRule-KG reduces violation counts by 83.2\% relative to a strong chain-of-thought baseline while improving exact match. Results remain stable under stratification and scale with dataset size, and the verifier adds negligible latency, making the approach practical for interactive design. 

---
# GEM: Generative Entropy-Guided Preference Modeling for Few-shot Alignment of LLMs 

**Authors**: Yiyang Zhao, Huiyu Bai, Xuejiao Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2511.13007)  

**Abstract**: Alignment of large language models (LLMs) with human preferences typically relies on supervised reward models or external judges that demand abundant annotations. However, in fields that rely on professional knowledge, such as medicine and law, such large-scale preference labels are often unachievable. In this paper, we propose a generative entropy-guided preference modeling approach named GEM for LLMs aligment at low-resource and domain-specific scenarios. Instead of training a discriminative reward model on preference data, we directly train the LLM to internalize a closed-loop optimization architecture that can extract and exploit the multi-dimensional, fine-grained cognitive signals implicit in human preferences. Specifically, our Cognitive Filtering module, based on entropy theory in decision making, first leverages Chain-of-Thought (CoT) prompting to generate diverse candidate reasoning chains (CoTs) from preference data. Subsequently, it introduces a token scoring mechanism to rank and weight the sampled CoTs, boosting the importance of high-confidence answers and strategically high-entropy tokens. Building on these filtered preferences, we fine-tune the LLM using a novel self-evaluated group advantage algorithm, SEGA, which effectively aggregates group-level cognitive signals and transforms the entropy-based scores into implicit rewards for policy optimization. In these ways, GEM empowers the LLM to rely on its own judgments and establishes an entropy-guided closed-loop cognitive optimization framework, enabling highly efficient few-shot alignment of LLMs. Experiments on general benchmarks and domain-specific tasks (such as mathematical reasoning and medical dialogues) demonstrate that our GEM achieves significant improvements with few-shot preference data. 

---
# Cost-Effective Communication: An Auction-based Method for Language Agent Interaction 

**Authors**: Yijia Fan, Jusheng Zhang, Kaitong Cai, Jing Yang, Chengpei Tang, Jian Wang, Keze Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.13193)  

**Abstract**: Multi-agent systems (MAS) built on large language models (LLMs) often suffer from inefficient "free-for-all" communication, leading to exponential token costs and low signal-to-noise ratios that hinder their practical deployment. We challenge the notion that more communication is always beneficial, hypothesizing instead that the core issue is the absence of resource rationality. We argue that "free" communication, by ignoring the principle of scarcity, inherently breeds inefficiency and unnecessary expenses. To address this, we introduce the Dynamic Auction-based Language Agent (DALA), a novel framework that treats communication bandwidth as a scarce and tradable resource. Specifically, our DALA regards inter-agent communication as a centralized auction, where agents learn to bid for the opportunity to speak based on the predicted value density of their messages. Thus, our DALA intrinsically encourages agents to produce concise, informative messages while filtering out low-value communication. Extensive and comprehensive experiments demonstrate that our economically-driven DALA achieves new state-of-the-art performance across seven challenging reasoning benchmarks, including 84.32% on MMLU and a 91.21% pass@1 rate on HumanEval. Note that this is accomplished with remarkable efficiency, i.e., our DALA uses only 6.25 million tokens, a fraction of the resources consumed by current state-of-the-art methods on GSM8K. Further analysis reveals that our DALA cultivates the emergent skill of strategic silence, effectively adapting its communication strategies from verbosity to silence in a dynamical manner via resource constraints. 

---
# Fault2Flow: An AlphaEvolve-Optimized Human-in-the-Loop Multi-Agent System for Fault-to-Workflow Automation 

**Authors**: Yafang Wang, Yangjie Tian, Xiaoyu Shen, Gaoyang Zhang, Jiaze Sun, He Zhang, Ruohua Xu, Feng Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2511.12916)  

**Abstract**: Power grid fault diagnosis is a critical process hindered by its reliance on manual, error-prone methods. Technicians must manually extract reasoning logic from dense regulations and attempt to combine it with tacit expert knowledge, which is inefficient, error-prone, and lacks maintainability as ragulations are updated and experience evolves. While Large Language Models (LLMs) have shown promise in parsing unstructured text, no existing framework integrates these two disparate knowledge sources into a single, verified, and executable workflow. To bridge this gap, we propose Fault2Flow, an LLM-based multi-agent system. Fault2Flow systematically: (1) extracts and structures regulatory logic into PASTA-formatted fault trees; (2) integrates expert knowledge via a human-in-the-loop interface for verification; (3) optimizes the reasoning logic using a novel AlphaEvolve module; and (4) synthesizes the final, verified logic into an n8n-executable workflow. Experimental validation on transformer fault diagnosis datasets confirms 100\% topological consistency and high semantic fidelity. Fault2Flow establishes a reproducible path from fault analysis to operational automation, substantially reducing expert workload. 

---
# Bootstrapping LLMs via Preference-Based Policy Optimization 

**Authors**: Chen Jia  

**Link**: [PDF](https://arxiv.org/pdf/2511.12867)  

**Abstract**: Bootstrapping large language models (LLMs) through preference-based policy optimization offers a promising direction for aligning model behavior with human preferences without relying on extensive manual annotations. In this work, we propose a novel preference-based policy optimization (PbPO) framework that formulates the learning process as a min-max game between the main policy and a reward model (RM). The RM is constrained within a confidence set derived from preference data to ensure reliable exploitation. Our iterative online algorithm actively collects preference data through guided exploration of the evolving policy, enabling continual self-improvement of both the policy and the RM. We provide theoretical guarantees for our method, establishing high-probability regret bounds for both settings with sequence-level RM and token-level RM, demonstrating its effectiveness in bootstrapping LLMs. Extensive experiments on five benchmarks show that our approach consistently outperforms existing state-of-the-art preference optimization techniques. 

---
# Online Learning of HTN Methods for integrated LLM-HTN Planning 

**Authors**: Yuesheng Xu, Hector Munoz-Avila  

**Link**: [PDF](https://arxiv.org/pdf/2511.12901)  

**Abstract**: We present online learning of Hierarchical Task Network (HTN) methods in the context of integrated HTN planning and LLM-based chatbots. Methods indicate when and how to decompose tasks into subtasks. Our method learner is built on top of the ChatHTN planner. ChatHTN queries ChatGPT to generate a decomposition of a task into primitive tasks when no applicable method for the task is available. In this work, we extend ChatHTN. Namely, when ChatGPT generates a task decomposition, ChatHTN learns from it, akin to memoization. However, unlike memoization, it learns a generalized method that applies not only to the specific instance encountered, but to other instances of the same task. We conduct experiments on two domains and demonstrate that our online learning procedure reduces the number of calls to ChatGPT while solving at least as many problems, and in some cases, even more. 

---
# Event-CausNet: Unlocking Causal Knowledge from Text with Large Language Models for Reliable Spatio-Temporal Forecasting 

**Authors**: Luyao Niu, Zepu Wang, Shuyi Guan, Yang Liu, Peng Sun  

**Link**: [PDF](https://arxiv.org/pdf/2511.12769)  

**Abstract**: While spatio-temporal Graph Neural Networks (GNNs) excel at modeling recurring traffic patterns, their reliability plummets during non-recurring events like accidents. This failure occurs because GNNs are fundamentally correlational models, learning historical patterns that are invalidated by the new causal factors introduced during disruptions. To address this, we propose Event-CausNet, a framework that uses a Large Language Model to quantify unstructured event reports, builds a causal knowledge base by estimating average treatment effects, and injects this knowledge into a dual-stream GNN-LSTM network using a novel causal attention mechanism to adjust and enhance the forecast. Experiments on a real-world dataset demonstrate that Event-CausNet achieves robust performance, reducing prediction error (MAE) by up to 35.87%, significantly outperforming state-of-the-art baselines. Our framework bridges the gap between correlational models and causal reasoning, providing a solution that is more accurate and transferable, while also offering crucial interpretability, providing a more reliable foundation for real-world traffic management during critical disruptions. 

---
# CoS: Towards Optimal Event Scheduling via Chain-of-Scheduling 

**Authors**: Yiming Zhao, Jiwei Tang, Shimin Di, Libin Zheng, Jianxing Yu, Jian Yin  

**Link**: [PDF](https://arxiv.org/pdf/2511.12913)  

**Abstract**: Recommending event schedules is a key issue in Event-based Social Networks (EBSNs) in order to maintain user activity. An effective recommendation is required to maximize the user's preference, subjecting to both time and geographical constraints. Existing methods face an inherent trade-off among efficiency, effectiveness, and generalization, due to the NP-hard nature of the problem. This paper proposes the Chain-of-Scheduling (CoS) framework, which activates the event scheduling capability of Large Language Models (LLMs) through a guided, efficient scheduling process. CoS enhances LLM by formulating the schedule task into three atomic stages, i.e., exploration, verification and integration. Then we enable the LLMs to generate CoS autonomously via Knowledge Distillation (KD). Experimental results show that CoS achieves near-theoretical optimal effectiveness with high efficiency on three real-world datasets in a interpretable manner. Moreover, it demonstrates strong zero-shot learning ability on out-of-domain data. 

---
# ARCHE: A Novel Task to Evaluate LLMs on Latent Reasoning Chain Extraction 

**Authors**: Pengze Li, Jiaqi Liu, Junchi Yu, Lihao Liu, Mingyu Ding, Wanli Ouyang, Shixiang Tang, Xi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2511.12485)  

**Abstract**: Large language models (LLMs) are increasingly used in scientific domains. While they can produce reasoning-like content via methods such as chain-of-thought prompting, these outputs are typically unstructured and informal, obscuring whether models truly understand the fundamental reasoning paradigms that underpin scientific inference. To address this, we introduce a novel task named Latent Reasoning Chain Extraction (ARCHE), in which models must decompose complex reasoning arguments into combinations of standard reasoning paradigms in the form of a Reasoning Logic Tree (RLT). In RLT, all reasoning steps are explicitly categorized as one of three variants of Peirce's fundamental inference modes: deduction, induction, or abduction. To facilitate this task, we release ARCHE Bench, a new benchmark derived from 70 Nature Communications articles, including more than 1,900 references and 38,000 viewpoints. We propose two logic-aware evaluation metrics: Entity Coverage (EC) for content completeness and Reasoning Edge Accuracy (REA) for step-by-step logical validity. Evaluations on 10 leading LLMs on ARCHE Bench reveal that models exhibit a trade-off between REA and EC, and none are yet able to extract a complete and standard reasoning chain. These findings highlight a substantial gap between the abilities of current reasoning models and the rigor required for scientific argumentation. 

---
# Reward and Guidance through Rubrics: Promoting Exploration to Improve Multi-Domain Reasoning 

**Authors**: Baolong Bi, Shenghua Liu, Yiwei Wang, Siqian Tong, Lingrui Mei, Yuyao Ge, Yilong Xu, Jiafeng Guo, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2511.12344)  

**Abstract**: Recent advances in reinforcement learning (RL) have significantly improved the complex reasoning capabilities of large language models (LLMs). Despite these successes, existing methods mainly focus on single-domain RL (e.g., mathematics) with verifiable rewards (RLVR), and their reliance on purely online RL frameworks restricts the exploration space, thereby limiting reasoning performance. In this paper, we address these limitations by leveraging rubrics to provide both fine-grained reward signals and offline guidance. We propose $\textbf{RGR-GRPO}$ (Reward and Guidance through Rubrics), a rubric-driven RL framework for multi-domain reasoning. RGR-GRPO enables LLMs to receive dense and informative rewards while exploring a larger solution space during GRPO training. Extensive experiments across 14 benchmarks spanning multiple domains demonstrate that RGR-GRPO consistently outperforms RL methods that rely solely on alternative reward schemes or offline guidance. Compared with verifiable online RL baseline, RGR-GRPO achieves average improvements of +7.0%, +5.4%, +8.4%, and +6.6% on mathematics, physics, chemistry, and general reasoning tasks, respectively. Notably, RGR-GRPO maintains stable entropy fluctuations during off-policy training and achieves superior pass@k performance, reflecting sustained exploration and effective breakthrough beyond existing performance bottlenecks. 

---
# MoralReason: Generalizable Moral Decision Alignment For LLM Agents Using Reasoning-Level Reinforcement Learning 

**Authors**: Zhiyu An, Wan Du  

**Link**: [PDF](https://arxiv.org/pdf/2511.12271)  

**Abstract**: Large language models are increasingly influencing human moral decisions, yet current approaches focus primarily on evaluating rather than actively steering their moral decisions. We formulate this as an out-of-distribution moral alignment problem, where LLM agents must learn to apply consistent moral reasoning frameworks to scenarios beyond their training distribution. We introduce Moral-Reason-QA, a novel dataset extending 680 human-annotated, high-ambiguity moral scenarios with framework-specific reasoning traces across utilitarian, deontological, and virtue ethics, enabling systematic evaluation of moral generalization in realistic decision contexts. Our learning approach employs Group Relative Policy Optimization with composite rewards that simultaneously optimize decision alignment and framework-specific reasoning processes to facilitate learning of the underlying moral frameworks. Experimental results demonstrate successful generalization to unseen moral scenarios, with softmax-normalized alignment scores improving by +0.757 for utilitarian and +0.450 for deontological frameworks when tested on out-of-distribution evaluation sets. The experiments also reveal training challenges and promising directions that inform future research. These findings establish that LLM agents can be systematically trained to internalize and apply specific moral frameworks to novel situations, providing a critical foundation for AI safety as language models become more integrated into human decision-making processes. 

---
# Enhancing Conversational Recommender Systems with Tree-Structured Knowledge and Pretrained Language Models 

**Authors**: Yongwen Ren, Chao Wang, Peng Du, Chuan Qin, Dazhong Shen, Hui Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2511.12579)  

**Abstract**: Recent advances in pretrained language models (PLMs) have significantly improved conversational recommender systems (CRS), enabling more fluent and context-aware interactions. To further enhance accuracy and mitigate hallucination, many methods integrate PLMs with knowledge graphs (KGs), but face key challenges: failing to fully exploit PLM reasoning over graph relationships, indiscriminately incorporating retrieved knowledge without context filtering, and neglecting collaborative preferences in multi-turn dialogues. To this end, we propose PCRS-TKA, a prompt-based framework employing retrieval-augmented generation to integrate PLMs with KGs. PCRS-TKA constructs dialogue-specific knowledge trees from KGs and serializes them into texts, enabling structure-aware reasoning while capturing rich entity semantics. Our approach selectively filters context-relevant knowledge and explicitly models collaborative preferences using specialized supervision signals. A semantic alignment module harmonizes heterogeneous inputs, reducing noise and enhancing accuracy. Extensive experiments demonstrate that PCRS-TKA consistently outperforms all baselines in both recommendation and conversational quality. 

---
# Multi-agent Self-triage System with Medical Flowcharts 

**Authors**: Yujia Liu, Sophia Yu, Hongyue Jin, Jessica Wen, Alexander Qian, Terrence Lee, Mattheus Ramsis, Gi Won Choi, Lianhui Qin, Xin Liu, Edward J. Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.12439)  

**Abstract**: Online health resources and large language models (LLMs) are increasingly used as a first point of contact for medical decision-making, yet their reliability in healthcare remains limited by low accuracy, lack of transparency, and susceptibility to unverified information. We introduce a proof-of-concept conversational self-triage system that guides LLMs with 100 clinically validated flowcharts from the American Medical Association, providing a structured and auditable framework for patient decision support. The system leverages a multi-agent framework consisting of a retrieval agent, a decision agent, and a chat agent to identify the most relevant flowchart, interpret patient responses, and deliver personalized, patient-friendly recommendations, respectively. Performance was evaluated at scale using synthetic datasets of simulated conversations. The system achieved 95.29% top-3 accuracy in flowchart retrieval (N=2,000) and 99.10% accuracy in flowchart navigation across varied conversational styles and conditions (N=37,200). By combining the flexibility of free-text interaction with the rigor of standardized clinical protocols, this approach demonstrates the feasibility of transparent, accurate, and generalizable AI-assisted self-triage, with potential to support informed patient decision-making while improving healthcare resource utilization. 

---
# Adaptive Diagnostic Reasoning Framework for Pathology with Multimodal Large Language Models 

**Authors**: Yunqi Hong, Johnson Kao, Liam Edwards, Nein-Tzu Liu, Chung-Yen Huang, Alex Oliveira-Kowaleski, Cho-Jui Hsieh, Neil Y.C. Lin  

**Link**: [PDF](https://arxiv.org/pdf/2511.12008)  

**Abstract**: AI tools in pathology have improved screening throughput, standardized quantification, and revealed prognostic patterns that inform treatment. However, adoption remains limited because most systems still lack the human-readable reasoning needed to audit decisions and prevent errors. We present RECAP-PATH, an interpretable framework that establishes a self-learning paradigm, shifting off-the-shelf multimodal large language models from passive pattern recognition to evidence-linked diagnostic reasoning. At its core is a two-phase learning process that autonomously derives diagnostic criteria: diversification expands pathology-style explanations, while optimization refines them for accuracy. This self-learning approach requires only small labeled sets and no white-box access or weight updates to generate cancer diagnoses. Evaluated on breast and prostate datasets, RECAP-PATH produced rationales aligned with expert assessment and delivered substantial gains in diagnostic accuracy over baselines. By uniting visual understanding with reasoning, RECAP-PATH provides clinically trustworthy AI and demonstrates a generalizable path toward evidence-linked interpretation. 

---
# LLM-Assisted Formalization Enables Deterministic Detection of Statutory Inconsistency in the Internal Revenue Code 

**Authors**: Borchuluun Yadamsuren, Steven Keith Platt, Miguel Diaz  

**Link**: [PDF](https://arxiv.org/pdf/2511.11954)  

**Abstract**: This study introduces a hybrid neuro-symbolic framework that achieves deterministic detection of statutory inconsistency in complex law. We use the U.S. Internal Revenue Code (IRC) as a case study because its complexity makes it a fertile domain for identifying conflicts. Our research offers a solution for detecting inconsistent provisions by combining Large Language Models (LLMs) with symbolic logic.
LLM-based methods can support compliance, fairness, and statutory drafting, yet tax-specific applications remain sparse. A key challenge is that such models struggle with hierarchical processing and deep structured reasoning, especially over long text.
This research addresses these gaps through experiments using GPT-4o, GPT-5, and Prolog. GPT-4o was first used to translate Section 121 into Prolog rules and refine them in SWISH. These rules were then incorporated into prompts to test whether Prolog-augmented prompting improved GPT-4o's inconsistency detection. GPT-4o, whether prompted with natural language alone or with Prolog augmentation, detected the inconsistency in only one of three strategies (33 percent accuracy), but its reasoning quality differed: natural-language prompting achieved 100 percent rule coverage, while Prolog-augmented prompting achieved 66 percent, indicating more incomplete statutory analysis.
In contrast to probabilistic prompting, the hybrid Prolog model produced deterministic and reproducible results. Guided by GPT-5 for refinement, the model formalized the IRC section's competing interpretations and successfully detected an inconsistency zone. Validation tests confirm that the Prolog implementation is accurate, internally consistent, deterministic, and capable of autonomously identifying inconsistencies. These findings show that LLM-assisted formalization, anchored in symbolic logic, enables transparent and reliable statutory inconsistency detection. 

---
# Bayesian Optimization in Language Space: An Eval-Efficient AI Self-Improvement Framework 

**Authors**: Enoch Hyunwook Kang, Hema Yoganarasimhan  

**Link**: [PDF](https://arxiv.org/pdf/2511.12063)  

**Abstract**: Large Language Models (LLMs) have recently enabled self-improving AI, i.e., AI that iteratively generates, evaluates, and refines its own outcomes. Recent studies have shown that self-improving AI focusing on prompt optimization can outperform state-of-the-art reinforcement-learning fine-tuned LLMs. Here, their `performance' is typically measured by query efficiency - the number of LLM-generated solution samples required to meet a certain performance threshold. However, in many societal applications, the primary limitation is not generating new solutions but evaluating them. For instance, evaluating an ad's effectiveness requires significant human feedback, which is far more costly and time-consuming than generating a candidate ad. To optimize for the evaluation efficiency objective, a natural approach is to extend Bayesian Optimization (BO), a framework proven optimal for evaluation efficiency, to the language domain. However, the difficulty of directly estimating suitable acquisition functions in LLMs' minds makes this extension challenging. This paper overcomes this challenge by proving that the combination of the simple and widely used Best-of-N selection strategy and simple textual gradients (i.e., textual edits from a critic model) statistically emulates the behavior of the gradients on the canonical UCB acquisition function, which induces optimal exploration in terms of evaluation efficiency. Based on this result, we propose TextGrad-Best-of-N Bayesian Optimization (T-BoN BO), a simple and eval-efficient language-space Bayesian optimization framework for AI self-improvement. We also empirically validate T-BoN BO by applying it to automated ad alignment tasks for persona distribution, demonstrating its superior performance compared to popular state-of-the-art baselines. 

---
# An Analysis of Architectural Impact on LLM-based Abstract Visual Reasoning: A Systematic Benchmark on RAVEN-FAIR 

**Authors**: Sinan Urgun, Seçkin Arı  

**Link**: [PDF](https://arxiv.org/pdf/2511.11916)  

**Abstract**: This study aims to systematically evaluate the performance of large language models (LLMs) in abstract visual reasoning problems. We examined four LLM models (GPT-4.1-Mini, Claude-3.5-Haiku, Gemini-1.5-Flash, Llama-3.3-70b) utilizing four different reasoning architectures (single-shot, embedding-controlled repetition, self-reflection, and multi-agent) on the RAVEN-FAIR dataset. Visual responses generated through a three-stage process (JSON extraction, LLM reasoning, and Tool Function) were evaluated using SSIM and LPIPS metrics; Chain-of-Thought scores and error types (semantic hallucination, numeric misperception) were analyzed. Results demonstrate that GPT-4.1-Mini consistently achieved the highest overall accuracy across all architectures, indicating a strong reasoning capability. While the multi-agent architecture occasionally altered semantic and numeric balance across models, these effects were not uniformly beneficial. Instead, each model exhibited distinct sensitivity patterns to architectural design, underscoring that reasoning effectiveness remains model-specific. Variations in response coverage further emerged as a confounding factor that complicates direct cross-architecture comparison. To estimate the upper-bound performance of each configuration, we report the best of five independent runs, representing a best-case scenario rather than an averaged outcome. This multi-run strategy aligns with recent recommendations, which emphasize that single-run evaluations are fragile and may lead to unreliable conclusions. 

---
# Learning to Refine: An Agentic RL Approach for Iterative SPARQL Query Construction 

**Authors**: Floris Vossebeld, Shenghui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.11770)  

**Abstract**: Generating complex, logically-sound SPARQL queries for multi-hop questions remains a critical bottleneck for Knowledge Graph Question Answering, as the brittle nature of one-shot generation by Large Language Models (LLMs) hinders reliable interaction with structured data. Current methods lack the adaptive policies needed to dynamically debug queries based on real-time execution feedback. This paper introduces a novel agentic framework where an LLM learns a resilient policy for the sequential process of iterative SPARQL construction. We show that a compact 3B-parameter model, trained exclusively via outcome-driven Reinforcement Learning (GRPO) without supervised fine-tuning, can learn effective policies for this task, discovering how to systematically recover from execution errors and refine its queries toward a correct answer. On a curated, executable single-answer subset of LC-QuAD 2.0, our agent achieves 49.7\% accuracy post-entity-linking, a significant 17.5 percentage point improvement over the strongest iterative zero-shot baseline. Further analysis reveals that while the agent's capability is driven by RL, its performance is enhanced by an explicit deliberative reasoning step that acts as a cognitive scaffold to improve policy precision. This work presents a generalizable blueprint for teaching agents to master formal, symbolic tools through interaction, bridging the gap between probabilistic LLMs and the structured world of Knowledge Graphs. 

---
# On the Measure of a Model: From Intelligence to Generality 

**Authors**: Ruchira Dhar, Ninell Oldenburg, Anders Soegaard  

**Link**: [PDF](https://arxiv.org/pdf/2511.11773)  

**Abstract**: Benchmarks such as ARC, Raven-inspired tests, and the Blackbird Task are widely used to evaluate the intelligence of large language models (LLMs). Yet, the concept of intelligence remains elusive- lacking a stable definition and failing to predict performance on practical tasks such as question answering, summarization, or coding. Optimizing for such benchmarks risks misaligning evaluation with real-world utility. Our perspective is that evaluation should be grounded in generality rather than abstract notions of intelligence. We identify three assumptions that often underpin intelligence-focused evaluation: generality, stability, and realism. Through conceptual and formal analysis, we show that only generality withstands conceptual and empirical scrutiny. Intelligence is not what enables generality; generality is best understood as a multitask learning problem that directly links evaluation to measurable performance breadth and reliability. This perspective reframes how progress in AI should be assessed and proposes generality as a more stable foundation for evaluating capability across diverse and evolving tasks. 

---
# Towards autonomous quantum physics research using LLM agents with access to intelligent tools 

**Authors**: Sören Arlt, Xuemei Gu, Mario Krenn  

**Link**: [PDF](https://arxiv.org/pdf/2511.11752)  

**Abstract**: Artificial intelligence (AI) is used in numerous fields of science, yet the initial research questions and targets are still almost always provided by human researchers. AI-generated creative ideas in science are rare and often vague, so that it remains a human task to execute them. Automating idea generation and implementation in one coherent system would significantly shift the role of humans in the scientific process. Here we present AI-Mandel, an LLM agent that can generate and implement ideas in quantum physics. AI-Mandel formulates ideas from the literature and uses a domain-specific AI tool to turn them into concrete experiment designs that can readily be implemented in laboratories. The generated ideas by AI-Mandel are often scientifically interesting - for two of them we have already written independent scientific follow-up papers. The ideas include new variations of quantum teleportation, primitives of quantum networks in indefinite causal orders, and new concepts of geometric phases based on closed loops of quantum information transfer. AI-Mandel is a prototypical demonstration of an AI physicist that can generate and implement concrete, actionable ideas. Building such a system is not only useful to accelerate science, but it also reveals concrete open challenges on the path to human-level artificial scientists. 

---
# CausalGuard: A Smart System for Detecting and Preventing False Information in Large Language Models 

**Authors**: Piyushkumar Patel  

**Link**: [PDF](https://arxiv.org/pdf/2511.11600)  

**Abstract**: While large language models have transformed how we interact with AI systems, they have a critical weakness: they confidently state false information that sounds entirely plausible. This "hallucination" problem has become a major barrier to using these models where accuracy matters most. Existing solutions either require retraining the entire model, add significant computational costs, or miss the root causes of why these hallucinations occur in the first place.
We present CausalGuard, a new approach that combines causal reasoning with symbolic logic to catch and prevent hallucinations as they happen. Unlike previous methods that only check outputs after generation, our system understands the causal chain that leads to false statements and intervenes early in the process. CausalGuard works through two complementary paths: one that traces causal relationships between what the model knows and what it generates, and another that checks logical consistency using automated reasoning.
Testing across twelve different benchmarks, we found that CausalGuard correctly identifies hallucinations 89.3\% of the time while missing only 8.3\% of actual hallucinations. More importantly, it reduces false claims by nearly 80\% while keeping responses natural and helpful. The system performs especially well on complex reasoning tasks where multiple steps of logic are required. Because CausalGuard shows its reasoning process, it works well in sensitive areas like medical diagnosis or financial analysis where understanding why a decision was made matters as much as the decision itself. 

---
# Data Value in the Age of Scaling: Understanding LLM Scaling Dynamics Under Real-Synthetic Data Mixtures 

**Authors**: Haohui Wang, Jingyuan Qi, Jianpeng Chen, Jun Wu, Lifu Huang, Lecheng Zheng, Kevin Choi, Balaji Veeramani, Edward Bowen, Alison Hu, Tyler Cody, Dawei Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2511.13640)  

**Abstract**: The rapid progress of large language models (LLMs) is fueled by the growing reliance on datasets that blend real and synthetic data. While synthetic data offers scalability and cost-efficiency, it often introduces systematic distributional discrepancies, particularly underrepresenting long-tail knowledge due to truncation effects from data generation mechanisms like top-p sampling, temperature scaling, and finite sampling. These discrepancies pose fundamental challenges in characterizing and evaluating the utility of mixed real-synthetic datasets. In this paper, we identify a three-phase scaling behavior characterized by two breakpoints that reflect transitions in model behavior across learning head and tail knowledge. We further derive an LLM generalization bound designed for real and synthetic mixtures, revealing several key factors that govern their generalization performance. Building on our theoretical findings, we propose an effective yet efficient data valuation method that scales to large-scale datasets. Comprehensive experiments across four tasks, including image classification, sentiment classification, instruction following, and complex reasoning, demonstrate that our method surpasses state-of-the-art baselines in data valuation with significantly low computational cost. 

---
# KForge: Program Synthesis for Diverse AI Hardware Accelerators 

**Authors**: Taras Sereda, Tom St. John, Burak Bartan, Natalie Serrino, Sachin Katti, Zain Asgar  

**Link**: [PDF](https://arxiv.org/pdf/2511.13274)  

**Abstract**: GPU kernels are critical for ML performance but difficult to optimize across diverse accelerators. We present KForge, a platform-agnostic framework built on two collaborative LLM-based agents: a generation agent that produces and iteratively refines programs through compilation and correctness feedback, and a performance analysis agent that interprets profiling data to guide optimization. This agent-based architecture requires only a single-shot example to target new platforms.
We make three key contributions: (1) introducing an iterative refinement system where the generation agent and performance analysis agent collaborate through functional and optimization passes, interpreting diverse profiling data (from programmatic APIs to GUI-based tools) to generate actionable recommendations that guide program synthesis for arbitrary accelerators; (2) demonstrating that the generation agent effectively leverages cross-platform knowledge transfer, where a reference implementation from one architecture substantially improves generation quality for different hardware targets; and (3) validating the platform-agnostic nature of our approach by demonstrating effective program synthesis across fundamentally different parallel computing platforms: NVIDIA CUDA and Apple Metal. 

---
# An LLM-based Quantitative Framework for Evaluating High-Stealthy Backdoor Risks in OSS Supply Chains 

**Authors**: Zihe Yan, Kai Luo, Haoyu Yang, Yang Yu, Zhuosheng Zhang, Guancheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.13341)  

**Abstract**: In modern software development workflows, the open-source software supply chain contributes significantly to efficient and convenient engineering practices. With increasing system complexity, using open-source software as third-party dependencies has become a common practice. However, the lack of maintenance for underlying dependencies and insufficient community auditing create challenges in ensuring source code security and the legitimacy of repository maintainers, especially under high-stealthy backdoor attacks exemplified by the XZ-Util incident. To address these problems, we propose a fine-grained project evaluation framework for backdoor risk assessment in open-source software. The framework models stealthy backdoor attacks from the viewpoint of the attacker and defines targeted metrics for each attack stage. In addition, to overcome the limitations of static analysis in assessing the reliability of repository maintenance activities such as irregular committer privilege escalation and limited participation in reviews, the framework uses large language models (LLMs) to conduct semantic evaluation of code repositories without relying on manually crafted patterns. The framework is evaluated on sixty six high-priority packages in the Debian ecosystem. The experimental results indicate that the current open-source software supply chain is exposed to various security risks. 

---
# TokenSqueeze: Performance-Preserving Compression for Reasoning LLMs 

**Authors**: Yuxiang Zhang, Zhengxu Yu, Weihang Pan, Zhongming Jin, Qiang Fu, Deng Cai, Binbin Lin, Jieping Ye  

**Link**: [PDF](https://arxiv.org/pdf/2511.13223)  

**Abstract**: Emerging reasoning LLMs such as OpenAI-o1 and DeepSeek-R1 have achieved strong performance on complex reasoning tasks by generating long chain-of-thought (CoT) traces. However, these long CoTs result in increased token usage, leading to higher inference latency and memory consumption. As a result, balancing accuracy and reasoning efficiency has become essential for deploying reasoning LLMs in practical applications. Existing long-to-short (Long2Short) methods aim to reduce inference length but often sacrifice accuracy, revealing a need for an approach that maintains performance while lowering token costs. To address this efficiency-accuracy tradeoff, we propose TokenSqueeze, a novel Long2Short method that condenses reasoning paths while preserving performance and relying exclusively on self-generated data. First, to prevent performance degradation caused by excessive compression of reasoning depth, we propose to select self-generated samples whose reasoning depth is adaptively matched to the complexity of the problem. To further optimize the linguistic expression without altering the underlying reasoning paths, we introduce a distribution-aligned linguistic refinement method that enhances the clarity and conciseness of the reasoning path while preserving its logical integrity. Comprehensive experimental results demonstrate the effectiveness of TokenSqueeze in reducing token usage while maintaining accuracy. Notably, DeepSeek-R1-Distill-Qwen-7B fine-tuned using our proposed method achieved a 50\% average token reduction while preserving accuracy on the MATH500 benchmark. TokenSqueeze exclusively utilizes the model's self-generated data, enabling efficient and high-fidelity reasoning without relying on manually curated short-answer datasets across diverse applications. Our code is available at this https URL. 

---
# ParaDySe: A Parallel-Strategy Switching Framework for Dynamic Sequence Lengths in Transformer 

**Authors**: Zhixin Ou, Peng Liang, Jianchen Han, Baihui Liu, Linbo Qiao  

**Link**: [PDF](https://arxiv.org/pdf/2511.13198)  

**Abstract**: Dynamic sequences with varying lengths have been widely used in the training of Transformer-based large language models (LLMs). However, current training frameworks adopt a pre-defined static parallel strategy for these sequences, causing neither communication-parallelization cancellation on short sequences nor out-of-memory on long sequences. To mitigate these issues, we propose ParaDySe, a novel adaptive Parallel strategy switching framework for Dynamic Sequences. ParaDySe enables on-the-fly optimal strategy adoption according to the immediate input sequence. It first implements the modular function libraries for parallel strategies with unified tensor layout specifications, and then builds sequence-aware memory and time cost models with hybrid methods. Guided by cost models, ParaDySe selects optimal layer-wise strategies for dynamic sequences via an efficient heuristic algorithm. By integrating these techniques together, ParaDySe achieves seamless hot-switching of optimal strategies through its well-designed function libraries. We compare ParaDySe with baselines on representative LLMs under datasets with sequence lengths up to 624K. Experimental results indicate that ParaDySe addresses OOM and CPC bottlenecks in LLM training by systematically integrating long-sequence optimizations with existing frameworks. 

---
# MACKO: Sparse Matrix-Vector Multiplication for Low Sparsity 

**Authors**: Vladimír Macko, Vladimír Boža  

**Link**: [PDF](https://arxiv.org/pdf/2511.13061)  

**Abstract**: Sparse Matrix-Vector Multiplication (SpMV) is a fundamental operation in the inference of sparse Large Language Models (LLMs). Because existing SpMV methods perform poorly under the low and unstructured sparsity (30-90%) commonly observed in pruned LLMs, unstructured pruning provided only limited memory reduction and speedup. We propose MACKO-SpMV, a GPU-optimized format and kernel co-designed to reduce storage overhead while preserving compatibility with the GPU's execution model. This enables efficient SpMV for unstructured sparsity without specialized hardware units (e.g., tensor cores) or format-specific precomputation. Empirical results show that at sparsity 50%, MACKO is the first approach with significant 1.5x memory reduction and 1.2-1.5x speedup over dense representation. Speedups over other SpMV baselines: 2.8-13.0x over cuSPARSE, 1.9-2.6x over Sputnik, and 2.2-2.5x over DASP. Applied to Llama2-7B pruned with Wanda to sparsity 50%, it delivers 1.5x memory reduction and 1.5x faster inference at fp16 precision. Thanks to MACKO, unstructured pruning at 50% sparsity is now justified in real-world LLM workloads. 

---
# Tokenize Once, Recommend Anywhere: Unified Item Tokenization for Multi-domain LLM-based Recommendation 

**Authors**: Yu Hou, Won-Yong Shin  

**Link**: [PDF](https://arxiv.org/pdf/2511.12922)  

**Abstract**: Large language model (LLM)-based recommender systems have achieved high-quality performance by bridging the discrepancy between the item space and the language space through item tokenization. However, existing item tokenization methods typically require training separate models for each item domain, limiting generalization. Moreover, the diverse distributions and semantics across item domains make it difficult to construct a unified tokenization that preserves domain-specific information. To address these challenges, we propose UniTok, a Unified item Tokenization framework that integrates our own mixture-of-experts (MoE) architecture with a series of codebooks to convert items into discrete tokens, enabling scalable tokenization while preserving semantic information across multiple item domains. Specifically, items from different domains are first projected into a unified latent space through a shared encoder. They are then routed to domain-specific experts to capture the unique semantics, while a shared expert, which is always active, encodes common knowledge transferable across domains. Additionally, to mitigate semantic imbalance across domains, we present a mutual information calibration mechanism, which guides the model towards retaining similar levels of semantic information for each domain. Comprehensive experiments on wide-ranging real-world datasets demonstrate that the proposed UniTok framework is (a) highly effective: achieving up to 51.89% improvements over strong benchmarks, (b) theoretically sound: showing the analytical validity of our architectural design and optimization; and (c) highly generalizable: demonstrating robust performance across diverse domains without requiring per-domain retraining, a capability not supported by existing baselines. 

---
# On the Fundamental Limits of LLMs at Scale 

**Authors**: Muhammad Ahmed Mohsin, Muhammad Umer, Ahsan Bilal, Zeeshan Memon, Muhammad Ibtsaam Qadir, Sagnik Bhattacharya, Hassan Rizwan, Abhiram R. Gorle, Maahe Zehra Kazmi, Ayesha Mohsin, Muhammad Usman Rafique, Zihao He, Pulkit Mehta, Muhammad Ali Jamshed, John M. Cioffi  

**Link**: [PDF](https://arxiv.org/pdf/2511.12869)  

**Abstract**: Large Language Models (LLMs) have benefited enormously from scaling, yet these gains are bounded by five fundamental limitations: (1) hallucination, (2) context compression, (3) reasoning degradation, (4) retrieval fragility, and (5) multimodal misalignment. While existing surveys describe these phenomena empirically, they lack a rigorous theoretical synthesis connecting them to the foundational limits of computation, information, and learning. This work closes that gap by presenting a unified, proof-informed framework that formalizes the innate theoretical ceilings of LLM scaling. First, computability and uncomputability imply an irreducible residue of error: for any computably enumerable model family, diagonalization guarantees inputs on which some model must fail, and undecidable queries (e.g., halting-style tasks) induce infinite failure sets for all computable predictors. Second, information-theoretic and statistical constraints bound attainable accuracy even on decidable tasks, finite description length enforces compression error, and long-tail factual knowledge requires prohibitive sample complexity. Third, geometric and computational effects compress long contexts far below their nominal size due to positional under-training, encoding attenuation, and softmax crowding. We further show how likelihood-based training favors pattern completion over inference, how retrieval under token limits suffers from semantic drift and coupling noise, and how multimodal scaling inherits shallow cross-modal alignment. Across sections, we pair theorems and empirical evidence to outline where scaling helps, where it saturates, and where it cannot progress, providing both theoretical foundations and practical mitigation paths like bounded-oracle retrieval, positional curricula, and sparse or hierarchical attention. 

---
# Video Finetuning Improves Reasoning Between Frames 

**Authors**: Ruiqi Yang, Tian Yun, Zihan Wang, Ellie Pavlick  

**Link**: [PDF](https://arxiv.org/pdf/2511.12868)  

**Abstract**: Multimodal large language models (LLMs) have made rapid progress in visual understanding, yet their extension from images to videos often reduces to a naive concatenation of frame tokens. In this work, we investigate what video finetuning brings to multimodal LLMs. We propose Visual Chain-of-Thought (vCoT), an explicit reasoning process that generates transitional event descriptions between consecutive frames. Using vCoT, we systematically compare image-only LVLMs with their video-finetuned counterparts, both with and without access to these transitional cues. Our experiments show that vCoT significantly improves the performance of image-only models on long-form video question answering, while yielding only marginal gains for video-finetuned models. This suggests that the latter already capture frame-to-frame transitions implicitly. Moreover, we find that video models transfer this temporal reasoning ability to purely static settings, outperforming image models' baselines on relational visual reasoning tasks. 

---
# DeepSport: A Multimodal Large Language Model for Comprehensive Sports Video Reasoning via Agentic Reinforcement Learning 

**Authors**: Junbo Zou, Haotian Xia, Zhen Ye, Shengjie Zhang, Christopher Lai, Vicente Ordonez, Weining Shen, Hanjie Chen  

**Link**: [PDF](https://arxiv.org/pdf/2511.12908)  

**Abstract**: Sports video understanding presents unique challenges, requiring models to perceive high-speed dynamics, comprehend complex rules, and reason over long temporal contexts. While Multimodal Large Language Models (MLLMs) have shown promise in genral domains, the current state of research in sports remains narrowly focused: existing approaches are either single-sport centric, limited to specific tasks, or rely on training-free paradigms that lack robust, learned reasoning process. To address this gap, we introduce DeepSport, the first end-to-end trained MLLM framework designed for multi-task, multi-sport video understanding. DeepSport shifts the paradigm from passive frame processing to active, iterative reasoning, empowering the model to ``think with videos'' by dynamically interrogating content via a specialized frame-extraction tool. To enable this, we propose a data distillation pipeline that synthesizes high-quality Chain-of-Thought (CoT) trajectories from 10 diverse data source, creating a unified resource of 78k training data. We then employ a two-stage training strategy, Supervised Fine-Tuning (SFT) followed by Reinforcement Learning (RL) with a novel gated tool-use reward, to optimize the model's reasoning process. Extensive experiments on the testing benchmark of 6.7k questions demonstrate that DeepSport achieves state-of-the-art performance, significantly outperforming baselines of both proprietary model and open-source models. Our work establishes a new foundation for domain-specific video reasoning to address the complexities of diverse sports. 

---
# Are LLMs The Way Forward? A Case Study on LLM-Guided Reinforcement Learning for Decentralized Autonomous Driving 

**Authors**: Timur Anvar, Jeffrey Chen, Yuyan Wang, Rohan Chandra  

**Link**: [PDF](https://arxiv.org/pdf/2511.12751)  

**Abstract**: Autonomous vehicle navigation in complex environments such as dense and fast-moving highways and merging scenarios remains an active area of research. A key limitation of RL is its reliance on well-specified reward functions, which often fail to capture the full semantic and social complexity of diverse, out-of-distribution situations. As a result, a rapidly growing line of research explores using Large Language Models (LLMs) to replace or supplement RL for direct planning and control, on account of their ability to reason about rich semantic context. However, LLMs present significant drawbacks: they can be unstable in zero-shot safety-critical settings, produce inconsistent outputs, and often depend on expensive API calls with network latency. This motivates our investigation into whether small, locally deployed LLMs (< 14B parameters) can meaningfully support autonomous highway driving through reward shaping rather than direct control. We present a case study comparing RL-only, LLM-only, and hybrid approaches, where LLMs augment RL rewards by scoring state-action transitions during training, while standard RL policies execute at test time. Our findings reveal that RL-only agents achieve moderate success rates (73-89%) with reasonable efficiency, LLM-only agents can reach higher success rates (up to 94%) but with severely degraded speed performance, and hybrid approaches consistently fall between these extremes. Critically, despite explicit efficiency instructions, LLM-influenced approaches exhibit systematic conservative bias with substantial model-dependent variability, highlighting important limitations of current small LLMs for safety-critical control tasks. 

---
# Whose Narrative is it Anyway? A KV Cache Manipulation Attack 

**Authors**: Mukkesh Ganesh, Kaushik Iyer, Arun Baalaaji Sankar Ananthan  

**Link**: [PDF](https://arxiv.org/pdf/2511.12752)  

**Abstract**: The Key Value(KV) cache is an important component for efficient inference in autoregressive Large Language Models (LLMs), but its role as a representation of the model's internal state makes it a potential target for integrity attacks. This paper introduces "History Swapping," a novel block-level attack that manipulates the KV cache to steer model generation without altering the user-facing prompt. The attack involves overwriting a contiguous segment of the active generation's cache with a precomputed cache from a different topic. We empirically evaluate this method across 324 configurations on the Qwen 3 family of models, analyzing the impact of timing, magnitude, and layer depth of the cache overwrite. Our findings reveal that only full-layer overwrites can successfully hijack the conversation's topic, leading to three distinct behaviors: immediate and persistent topic shift, partial recovery, or a delayed hijack. Furthermore, we observe that high-level structural plans are encoded early in the generation process and local discourse structure is maintained by the final layers of the model. This work demonstrates that the KV cache is a significant vector for security analysis, as it encodes not just context but also topic trajectory and structural planning, making it a powerful interface for manipulating model behavior. 

---
# LLM4SCREENLIT: Recommendations on Assessing the Performance of Large Language Models for Screening Literature in Systematic Reviews 

**Authors**: Lech Madeyski, Barbara Kitchenham, Martin Shepperd  

**Link**: [PDF](https://arxiv.org/pdf/2511.12635)  

**Abstract**: Context: Large language models (LLMs) are released faster than users' ability to evaluate them rigorously. When LLMs underpin research, such as identifying relevant literature for systematic reviews (SRs), robust empirical assessment is essential. Objective: We identify and discuss key challenges in assessing LLM performance for selecting relevant literature, identify good (evaluation) practices, and propose recommendations. Method: Using a recent large-scale study as an example, we identify problems with the use of traditional metrics for assessing the performance of Gen-AI tools for identifying relevant literature in SRs. We analyzed 27 additional papers investigating this issue, extracted the performance metrics, and found both good practices and widespread problems, especially with the use and reporting of performance metrics for SR screening. Results: Major weaknesses included: i) a failure to use metrics that are robust to imbalanced data and do not directly indicate whether results are better than chance, e.g., the use of Accuracy, ii) a failure to consider the impact of lost evidence when making claims concerning workload savings, and iii) pervasive failure to report the full confusion matrix (or performance metrics from which it can be reconstructed) which is essential for future meta-analyses. On the positive side, we extract good (evaluation) practices on which our recommendations for researchers and practitioners, as well as policymakers, are built. Conclusions: SR screening evaluations should prioritize lost evidence/recall alongside chance-anchored and cost-sensitive Weighted MCC (WMCC) metric, report complete confusion matrices, treat unclassifiable outputs as referred-back positives for assessment, adopt leakage-aware designs with non-LLM baselines and open artifacts, and ground conclusions in cost-benefit analysis where FNs carry higher penalties than FPs. 

---
# R$^{2}$Seg: Training-Free OOD Medical Tumor Segmentation via Anatomical Reasoning and Statistical Rejection 

**Authors**: Shuaike Shen, Ke Liu, Jiaqing Xie, Shangde Gao, Chunhua Shen, Ge Liu, Mireia Crispin-Ortuzar, Shangqi Gao  

**Link**: [PDF](https://arxiv.org/pdf/2511.12691)  

**Abstract**: Foundation models for medical image segmentation struggle under out-of-distribution (OOD) shifts, often producing fragmented false positives on OOD tumors. We introduce R$^{2}$Seg, a training-free framework for robust OOD tumor segmentation that operates via a two-stage Reason-and-Reject process. First, the Reason step employs an LLM-guided anatomical reasoning planner to localize organ anchors and generate multi-scale ROIs. Second, the Reject step applies two-sample statistical testing to candidates generated by a frozen foundation model (BiomedParse) within these ROIs. This statistical rejection filter retains only candidates significantly different from normal tissue, effectively suppressing false positives. Our framework requires no parameter updates, making it compatible with zero-update test-time augmentation and avoiding catastrophic forgetting. On multi-center and multi-modal tumor segmentation benchmarks, R$^{2}$Seg substantially improves Dice, specificity, and sensitivity over strong baselines and the original foundation models. Code are available at this https URL. 

---
# Mitigating Length Bias in RLHF through a Causal Lens 

**Authors**: Hyeonji Kim, Sujeong Oh, Sanghack Lee  

**Link**: [PDF](https://arxiv.org/pdf/2511.12573)  

**Abstract**: Reinforcement learning from human feedback (RLHF) is widely used to align large language models (LLMs) with human preferences. However, RLHF-trained reward models often exhibit length bias -- a systematic tendency to favor longer responses by conflating verbosity with quality. We propose a causal framework for analyzing and mitigating length bias in RLHF reward modeling. Central to our approach is a counterfactual data augmentation method that generates response pairs designed to isolate content quality from verbosity. These counterfactual examples are then used to train the reward model, enabling it to assess responses based on content quality independently of verbosity. Specifically, we construct (1) length-divergent pairs with similar content and (2) content-divergent pairs of similar length. Empirical evaluations show that our method reduces length bias in reward assignment and leads to more concise, content-focused outputs from the policy model. These findings demonstrate that the proposed approach effectively reduces length bias and improves the robustness and content sensitivity of reward modeling in RLHF pipelines. 

---
# One Request, Multiple Experts: LLM Orchestrates Domain Specific Models via Adaptive Task Routing 

**Authors**: Xu Yang, Chenhui Lin, Haotian Liu, Qi Wang, Yue Yang, Wenchuan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2511.12484)  

**Abstract**: With the integration of massive distributed energy resources and the widespread participation of novel market entities, the operation of active distribution networks (ADNs) is progressively evolving into a complex multi-scenario, multi-objective problem. Although expert engineers have developed numerous domain specific models (DSMs) to address distinct technical problems, mastering, integrating, and orchestrating these heterogeneous DSMs still entail considerable overhead for ADN operators. Therefore, an intelligent approach is urgently required to unify these DSMs and enable efficient coordination. To address this challenge, this paper proposes the ADN-Agent architecture, which leverages a general large language model (LLM) to coordinate multiple DSMs, enabling adaptive intent recognition, task decomposition, and DSM invocation. Within the ADN-Agent, we design a novel communication mechanism that provides a unified and flexible interface for diverse heterogeneous DSMs. Finally, for some language-intensive subtasks, we propose an automated training pipeline for fine-tuning small language models, thereby effectively enhancing the overall problem-solving capability of the system. Comprehensive comparisons and ablation experiments validate the efficacy of the proposed method and demonstrate that the ADN-Agent architecture outperforms existing LLM application paradigms. 

---
# SeedAIchemy: LLM-Driven Seed Corpus Generation for Fuzzing 

**Authors**: Aidan Wen, Norah A. Alzahrani, Jingzhi Jiang, Andrew Joe, Karen Shieh, Andy Zhang, Basel Alomair, David Wagner  

**Link**: [PDF](https://arxiv.org/pdf/2511.12448)  

**Abstract**: We introduce SeedAIchemy, an automated LLM-driven corpus generation tool that makes it easier for developers to implement fuzzing effectively. SeedAIchemy consists of five modules which implement different approaches at collecting publicly available files from the internet. Four of the five modules use large language model (LLM) workflows to construct search terms designed to maximize corpus quality. Corpora generated by SeedAIchemy perform significantly better than a naive corpus and similarly to a manually-curated corpus on a diverse range of target programs and libraries. 

---
# SynthGuard: An Open Platform for Detecting AI-Generated Multimedia with Multimodal LLMs 

**Authors**: Shail Desai, Aditya Pawar, Li Lin, Xin Wang, Shu Hu  

**Link**: [PDF](https://arxiv.org/pdf/2511.12404)  

**Abstract**: Artificial Intelligence (AI) has made it possible for anyone to create images, audio, and video with unprecedented ease, enriching education, communication, and creative expression. At the same time, the rapid rise of AI-generated media has introduced serious risks, including misinformation, identity misuse, and the erosion of public trust as synthetic content becomes increasingly indistinguishable from real media. Although deepfake detection has advanced, many existing tools remain closed-source, limited in modality, or lacking transparency and educational value, making it difficult for users to understand how detection decisions are made. To address these gaps, we introduce SynthGuard, an open, user-friendly platform for detecting and analyzing AI-generated multimedia using both traditional detectors and multimodal large language models (MLLMs). SynthGuard provides explainable inference, unified image and audio support, and an interactive interface designed to make forensic analysis accessible to researchers, educators, and the public. The SynthGuard platform is available at: this https URL 

---
# Optimal Self-Consistency for Efficient Reasoning with Large Language Models 

**Authors**: Austin Feng, Marius Alonso, Ambroise Odonnat  

**Link**: [PDF](https://arxiv.org/pdf/2511.12309)  

**Abstract**: Self-consistency (SC) is a widely used test-time inference technique for improving performance in chain-of-thought reasoning. It involves generating multiple responses, or samples from a large language model (LLM) and selecting the most frequent answer. This procedure can naturally be viewed as a majority vote or empirical mode estimation. Despite its effectiveness, SC is prohibitively expensive at scale when naively applied to datasets, and it lacks a unified theoretical treatment of sample efficiency and scaling behavior. In this paper, we provide the first comprehensive analysis of SC's scaling behavior and its variants, drawing on mode estimation and voting theory. We derive and empirically validate power law scaling for self-consistency across datasets, and analyze the sample efficiency for fixed-allocation and dynamic-allocation sampling schemes. From these insights, we introduce Blend-ASC, a novel variant of self-consistency that dynamically allocates samples to questions during inference, achieving state-of-the-art sample efficiency. Our approach uses 6.8x fewer samples than vanilla SC on average, outperforming both fixed- and dynamic-allocation SC baselines, thereby demonstrating the superiority of our approach in terms of efficiency. In contrast to existing variants, Blend-ASC is hyperparameter-free and can fit an arbitrary sample budget, ensuring it can be easily applied to any self-consistency application. 

---
# CrossVid: A Comprehensive Benchmark for Evaluating Cross-Video Reasoning in Multimodal Large Language Models 

**Authors**: Jingyao Li, Jingyun Wang, Molin Tan, Haochen Wang, Cilin Yan, Likun Shi, Jiayin Cai, Xiaolong Jiang, Yao Hu  

**Link**: [PDF](https://arxiv.org/pdf/2511.12263)  

**Abstract**: Cross-Video Reasoning (CVR) presents a significant challenge in video understanding, which requires simultaneous understanding of multiple videos to aggregate and compare information across groups of videos. Most existing video understanding benchmarks focus on single-video analysis, failing to assess the ability of multimodal large language models (MLLMs) to simultaneously reason over various videos. Recent benchmarks evaluate MLLMs' capabilities on multi-view videos that capture different perspectives of the same scene. However, their limited tasks hinder a thorough assessment of MLLMs in diverse real-world CVR scenarios. To this end, we introduce CrossVid, the first benchmark designed to comprehensively evaluate MLLMs' spatial-temporal reasoning ability in cross-video contexts. Firstly, CrossVid encompasses a wide spectrum of hierarchical tasks, comprising four high-level dimensions and ten specific tasks, thereby closely reflecting the complex and varied nature of real-world video understanding. Secondly, CrossVid provides 5,331 videos, along with 9,015 challenging question-answering pairs, spanning single-choice, multiple-choice, and open-ended question formats. Through extensive experiments on various open-source and closed-source MLLMs, we observe that Gemini-2.5-Pro performs best on CrossVid, achieving an average accuracy of 50.4%. Notably, our in-depth case study demonstrates that most current MLLMs struggle with CVR tasks, primarily due to their inability to integrate or compare evidence distributed across multiple videos for reasoning. These insights highlight the potential of CrossVid to guide future advancements in enhancing MLLMs' CVR capabilities. 

---
# OAD-Promoter: Enhancing Zero-shot VQA using Large Language Models with Object Attribute Description 

**Authors**: Quanxing Xu, Ling Zhou, Feifei Zhang, Jinyu Tian, Rubing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2511.12131)  

**Abstract**: Large Language Models (LLMs) have become a crucial tool in Visual Question Answering (VQA) for handling knowledge-intensive questions in few-shot or zero-shot scenarios. However, their reliance on massive training datasets often causes them to inherit language biases during the acquisition of knowledge. This limitation imposes two key constraints on existing methods: (1) LLM predictions become less reliable due to bias exploitation, and (2) despite strong knowledge reasoning capabilities, LLMs still struggle with out-of-distribution (OOD) generalization. To address these issues, we propose Object Attribute Description Promoter (OAD-Promoter), a novel approach for enhancing LLM-based VQA by mitigating language bias and improving domain-shift robustness. OAD-Promoter comprises three components: the Object-concentrated Example Generation (OEG) module, the Memory Knowledge Assistance (MKA) module, and the OAD Prompt. The OEG module generates global captions and object-concentrated samples, jointly enhancing visual information input to the LLM and mitigating bias through complementary global and regional visual cues. The MKA module assists the LLM in handling OOD samples by retrieving relevant knowledge from stored examples to support questions from unseen domains. Finally, the OAD Prompt integrates the outputs of the preceding modules to optimize LLM inference. Experiments demonstrate that OAD-Promoter significantly improves the performance of LLM-based VQA methods in few-shot or zero-shot settings, achieving new state-of-the-art results. 

---
# Striking the Right Balance between Compute and Copy: Improving LLM Inferencing Under Speculative Decoding 

**Authors**: Arun Ramachandran, Ramaswamy Govindarajan, Murali Annavaram, Prakash Raghavendra, Hossein Entezari Zarch, Lei Gao, Chaoyi Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2511.12031)  

**Abstract**: With the skyrocketing costs of GPUs and their virtual instances in the cloud, there is a significant desire to use CPUs for large language model (LLM) inference. KV cache update, often implemented as allocation, copying, and in-place strided update for each generated token, incurs significant overhead. As the sequence length increases, the allocation and copy overheads dominate the performance. Alternate approaches may allocate large KV tensors upfront to enable in-place updates, but these matrices (with zero-padded rows) cause redundant computations. In this work, we propose a new KV cache allocation mechanism called Balancing Memory and Compute (BMC). BMC allocates, once every r iterations, KV tensors with r redundant rows, allowing in-place update without copy overhead for those iterations, but at the expense of a small amount of redundant computation. Second, we make an interesting observation that the extra rows allocated in the KV tensors and the resulting redundant computation can be repurposed for Speculative Decoding (SD) that improves token generation efficiency. Last, BMC represents a spectrum of design points with different values of r. To identify the best-performing design point(s), we derive a simple analytical model for BMC. The proposed BMC method achieves an average throughput acceleration of up to 3.2x over baseline HuggingFace (without SD). Importantly when we apply BMC with SD, it results in an additional speedup of up to 1.39x, over and above the speedup offered by SD. Further, BMC achieves a throughput acceleration of up to 1.36x and 2.29x over state-of-the-art inference servers vLLM and DeepSpeed, respectively. Although the BMC technique is evaluated extensively across different classes of CPUs (desktop and server class), we also evaluate the scheme with GPUs and demonstrate that it works well for GPUs. 

---
# VULPO: Context-Aware Vulnerability Detection via On-Policy LLM Optimization 

**Authors**: Youpeng Li, Fuxun Yu, Xinda Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.11896)  

**Abstract**: The widespread reliance on open-source software dramatically increases the risk of vulnerability exploitation, underscoring the need for effective and scalable vulnerability detection (VD). Existing VD techniques, whether traditional machine learning-based or LLM-based approaches like prompt engineering, supervised fine-tuning, or off-policy preference optimization, remain fundamentally limited in their ability to perform context-aware analysis: They depend on fixed inputs or static preference datasets, cannot adaptively explore repository-level dependencies, and are constrained by function-level benchmarks that overlook critical vulnerability context.
This paper introduces Vulnerability-Adaptive Policy Optimization (VULPO), an on-policy LLM reinforcement learning framework for context-aware VD. To support training and evaluation, we first construct ContextVul, a new dataset that augments high-quality function-level samples with lightweight method to extract repository-level context information. We then design multi-dimensional reward structuring that jointly captures prediction correctness, vulnerability localization accuracy, and the semantic relevance of vulnerability analysis, thereby guiding the model toward comprehensive contextual reasoning. To address the asymmetric difficulty of different vulnerability cases and mitigate reward hacking, VULPO incorporates label-level and sample-level difficulty-adaptive reward scaling, encouraging the model to explore challenging cases while maintaining balanced reward distribution. Extensive experiments demonstrate the superiority of our VULPO framework in context-aware VD: Our VULPO-4B substantially outperforms existing VD baselines based on prompt engineering and off-policy optimization, improving F1 by 85% over Qwen3-4B and achieving performance comparable to a 150x larger-scale model, DeepSeek-R1-0528. 

---
# Conformal Constrained Policy Optimization for Cost-Effective LLM Agents 

**Authors**: Wenwen Si, Sooyong Jang, Insup Lee, Osbert Bastani  

**Link**: [PDF](https://arxiv.org/pdf/2511.11828)  

**Abstract**: While large language models (LLMs) have recently made tremendous progress towards solving challenging AI problems, they have done so at increasingly steep computational and API costs. We propose a novel strategy where we combine multiple LLM models with varying cost/accuracy tradeoffs in an agentic manner, where models and tools are run in sequence as determined by an orchestration model to minimize cost subject to a user-specified level of reliability; this constraint is formalized using conformal prediction to provide guarantees. To solve this problem, we propose Conformal Constrained Policy Optimization (CCPO), a training paradigm that integrates constrained policy optimization with off-policy reinforcement learning and recent advances in online conformal prediction. CCPO jointly optimizes a cost-aware policy (score function) and an adaptive threshold. Across two multi-hop question answering benchmarks, CCPO achieves up to a 30% cost reduction compared to other cost-aware baselines and LLM-guided methods without compromising reliability. Our approach provides a principled and practical framework for deploying LLM agents that are significantly more cost-effective while maintaining reliability. 

---
# From Single to Societal: Analyzing Persona-Induced Bias in Multi-Agent Interactions 

**Authors**: Jiayi Li, Xiao Liu, Yansong Feng  

**Link**: [PDF](https://arxiv.org/pdf/2511.11789)  

**Abstract**: Large Language Model (LLM)-based multi-agent systems are increasingly used to simulate human interactions and solve collaborative tasks. A common practice is to assign agents with personas to encourage behavioral diversity. However, this raises a critical yet underexplored question: do personas introduce biases into multi-agent interactions? This paper presents a systematic investigation into persona-induced biases in multi-agent interactions, with a focus on social traits like trustworthiness (how an agent's opinion is received by others) and insistence (how strongly an agent advocates for its opinion). Through a series of controlled experiments in collaborative problem-solving and persuasion tasks, we reveal that (1) LLM-based agents exhibit biases in both trustworthiness and insistence, with personas from historically advantaged groups (e.g., men and White individuals) perceived as less trustworthy and demonstrating less insistence; and (2) agents exhibit significant in-group favoritism, showing a higher tendency to conform to others who share the same persona. These biases persist across various LLMs, group sizes, and numbers of interaction rounds, highlighting an urgent need for awareness and mitigation to ensure the fairness and reliability of multi-agent systems. 

---
# Differences in the Moral Foundations of Large Language Models 

**Authors**: Peter Kirgis  

**Link**: [PDF](https://arxiv.org/pdf/2511.11790)  

**Abstract**: Large language models are increasingly being used in critical domains of politics, business, and education, but the nature of their normative ethical judgment remains opaque. Alignment research has, to date, not sufficiently utilized perspectives and insights from the field of moral psychology to inform training and evaluation of frontier models. I perform a synthetic experiment on a wide range of models from most major model providers using Jonathan Haidt's influential moral foundations theory (MFT) to elicit diverse value judgments from LLMs. Using multiple descriptive statistical approaches, I document the bias and variance of large language model responses relative to a human baseline in the original survey. My results suggest that models rely on different moral foundations from one another and from a nationally representative human baseline, and these differences increase as model capabilities increase. This work seeks to spur further analysis of LLMs using MFT, including finetuning of open-source models, and greater deliberation by policymakers on the importance of moral foundations for LLM alignment. 

---
# MALBO: Optimizing LLM-Based Multi-Agent Teams via Multi-Objective Bayesian Optimization 

**Authors**: Antonio Sabbatella  

**Link**: [PDF](https://arxiv.org/pdf/2511.11788)  

**Abstract**: The optimal assignment of Large Language Models (LLMs) to specialized roles in multi-agent systems is a significant challenge, defined by a vast combinatorial search space, expensive black-box evaluations, and an inherent trade-off between performance and cost. Current optimization methods focus on single-agent settings and lack a principled framework for this multi-agent, multi-objective problem.
This thesis introduces MALBO (Multi-Agent LLM Bayesian Optimization), a systematic framework designed to automate the efficient composition of LLM-based agent teams. We formalize the assignment challenge as a multi-objective optimization problem, aiming to identify the Pareto front of configurations between task accuracy and inference cost. The methodology employs multi-objective Bayesian Optimization (MOBO) with independent Gaussian Process surrogate models. By searching over a continuous feature-space representation of the LLMs, this approach performs a sample-efficient exploration guided by the expected hypervolume improvement.
The primary contribution is a principled and automated methodology that yields a Pareto front of optimal team configurations. Our results demonstrate that the Bayesian optimization phase, compared to an initial random search, maintained a comparable average performance while reducing the average configuration cost by over 45%. Furthermore, MALBO identified specialized, heterogeneous teams that achieve cost reductions of up to 65.8% compared to homogeneous baselines, all while maintaining maximum performance. The framework thus provides a data-driven tool for deploying cost-effective and highly specialized multi-agent AI systems. 

---
# Can AI Models be Jailbroken to Phish Elderly Victims? An End-to-End Evaluation 

**Authors**: Fred Heiding, Simon Lermen  

**Link**: [PDF](https://arxiv.org/pdf/2511.11759)  

**Abstract**: We present an end-to-end demonstration of how attackers can exploit AI safety failures to harm vulnerable populations: from jailbreaking LLMs to generate phishing content, to deploying those messages against real targets, to successfully compromising elderly victims. We systematically evaluated safety guardrails across six frontier LLMs spanning four attack categories, revealing critical failures where several models exhibited near-complete susceptibility to certain attack vectors. In a human validation study with 108 senior volunteers, AI-generated phishing emails successfully compromised 11\% of participants. Our work uniquely demonstrates the complete attack pipeline targeting elderly populations, highlighting that current AI safety measures fail to protect those most vulnerable to fraud. Beyond generating phishing content, LLMs enable attackers to overcome language barriers and conduct multi-turn trust-building conversations at scale, fundamentally transforming fraud economics. While some providers report voluntary counter-abuse efforts, we argue these remain insufficient. 

---
# Speculative Decoding in Decentralized LLM Inference: Turning Communication Latency into Computation Throughput 

**Authors**: Jingwei Song, Wanyi Chen, Xinyuan Song, Chris Tong, Gufeng Chen, Tianyi Zhao, Eric Yang, Bill Shi, Lynn Ai  

**Link**: [PDF](https://arxiv.org/pdf/2511.11733)  

**Abstract**: Speculative decoding accelerates large language model (LLM) inference by using a lightweight draft model to propose tokens that are later verified by a stronger target model. While effective in centralized systems, its behavior in decentralized settings, where network latency often dominates compute, remains under-characterized. We present Decentralized Speculative Decoding (DSD), a plug-and-play framework for decentralized inference that turns communication delay into useful computation by verifying multiple candidate tokens in parallel across distributed nodes. We further introduce an adaptive speculative verification strategy that adjusts acceptance thresholds by token-level semantic importance, delivering an additional 15% to 20% end-to-end speedup without retraining. In theory, DSD reduces cross-node communication cost by approximately (N-1)t1(k-1)/k, where t1 is per-link latency and k is the average number of tokens accepted per round. In practice, DSD achieves up to 2.56x speedup on HumanEval and 2.59x on GSM8K, surpassing the Eagle3 baseline while preserving accuracy. These results show that adapting speculative decoding for decentralized execution provides a system-level optimization that converts network stalls into throughput, enabling faster distributed LLM inference with no model retraining or architectural changes. 

---
# Evaluating Large Language Models for Workload Mapping and Scheduling in Heterogeneous HPC Systems 

**Authors**: Aasish Kumar Sharma, Julian Kunkel  

**Link**: [PDF](https://arxiv.org/pdf/2511.11612)  

**Abstract**: Large language models (LLMs) are increasingly explored for their reasoning capabilities, yet their ability to perform structured, constraint-based optimization from natural language remains insufficiently understood. This study evaluates twenty-one publicly available LLMs on a representative heterogeneous high-performance computing (HPC) workload mapping and scheduling problem. Each model received the same textual description of system nodes, task requirements, and scheduling constraints, and was required to assign tasks to nodes, compute the total makespan, and explain its reasoning. A manually derived analytical optimum of nine hours and twenty seconds served as the ground truth reference. Three models exactly reproduced the analytical optimum while satisfying all constraints, twelve achieved near-optimal results within two minutes of the reference, and six produced suboptimal schedules with arithmetic or dependency errors. All models generated feasible task-to-node mappings, though only about half maintained strict constraint adherence. Nineteen models produced partially executable verification code, and eighteen provided coherent step-by-step reasoning, demonstrating strong interpretability even when logical errors occurred. Overall, the results define the current capability boundary of LLM reasoning in combinatorial optimization: leading models can reconstruct optimal schedules directly from natural language, but most still struggle with precise timing, data transfer arithmetic, and dependency enforcement. These findings highlight the potential of LLMs as explainable co-pilots for optimization and decision-support tasks rather than autonomous solvers. 

---
# Parallel and Multi-Stage Knowledge Graph Retrieval for Behaviorally Aligned Financial Asset Recommendations 

**Authors**: Fernando Spadea, Oshani Seneviratne  

**Link**: [PDF](https://arxiv.org/pdf/2511.11583)  

**Abstract**: Large language models (LLMs) show promise for personalized financial recommendations but are hampered by context limits, hallucinations, and a lack of behavioral grounding. Our prior work, FLARKO, embedded structured knowledge graphs (KGs) in LLM prompts to align advice with user behavior and market data. This paper introduces RAG-FLARKO, a retrieval-augmented extension to FLARKO, that overcomes scalability and relevance challenges using multi-stage and parallel KG retrieval processes. Our method first retrieves behaviorally relevant entities from a user's transaction KG and then uses this context to filter temporally consistent signals from a market KG, constructing a compact, grounded subgraph for the LLM. This pipeline reduces context overhead and sharpens the model's focus on relevant information. Empirical evaluation on a real-world financial transaction dataset demonstrates that RAG-FLARKO significantly enhances recommendation quality. Notably, our framework enables smaller, more efficient models to achieve high performance in both profitability and behavioral alignment, presenting a viable path for deploying grounded financial AI in resource-constrained environments. 

---
# MedBuild AI: An Agent-Based Hybrid Intelligence Framework for Reshaping Agency in Healthcare Infrastructure Planning through Generative Design for Medical Architecture 

**Authors**: Yiming Zhang, Yuejia Xu, Ziyao Wang, Xin Yan, Xiaosai Hao  

**Link**: [PDF](https://arxiv.org/pdf/2511.11587)  

**Abstract**: Globally, disparities in healthcare infrastructure remain stark, leaving countless communities without access to even basic services. Traditional infrastructure planning is often slow and inaccessible, and although many architects are actively delivering humanitarian and aid-driven hospital projects worldwide, these vital efforts still fall far short of the sheer scale and urgency of demand. This paper introduces MedBuild AI, a hybrid-intelligence framework that integrates large language models (LLMs) with deterministic expert systems to rebalance the early design and conceptual planning stages. As a web-based platform, it enables any region with satellite internet access to obtain guidance on modular, low-tech, low-cost medical building designs. The system operates through three agents: the first gathers local health intelligence via conversational interaction; the second translates this input into an architectural functional program through rule-based computation; and the third generates layouts and 3D models. By embedding computational negotiation into the design process, MedBuild AI fosters a reciprocal, inclusive, and equitable approach to healthcare planning, empowering communities and redefining agency in global healthcare architecture. 

---
# DAOpt: Modeling and Evaluation of Data-Driven Optimization under Uncertainty with LLMs 

**Authors**: WenZhuo Zhu, Zheng Cui, Wenhan Lu, Sheng Liu, Yue Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2511.11576)  

**Abstract**: Recent advances in large language models (LLMs) have accelerated research on automated optimization modeling. While real-world decision-making is inherently uncertain, most existing work has focused on deterministic optimization with known parameters, leaving the application of LLMs in uncertain settings largely unexplored. To that end, we propose the DAOpt framework including a new dataset OptU, a multi-agent decision-making module, and a simulation environment for evaluating LLMs with a focus on out-of-sample feasibility and robustness. Additionally, we enhance LLMs' modeling capabilities by incorporating few-shot learning with domain knowledge from stochastic and robust optimization. 

---
# HAPO: Training Language Models to Reason Concisely via History-Aware Policy Optimization 

**Authors**: Chengyu Huang, Zhengxin Zhang, Claire Cardie  

**Link**: [PDF](https://arxiv.org/pdf/2505.11225)  

**Abstract**: While scaling the length of responses at test-time has been shown to markedly improve the reasoning abilities and performance of large language models (LLMs), it often results in verbose outputs and increases inference cost. Prior approaches for efficient test-time scaling, typically using universal budget constraints or query-level length optimization, do not leverage historical information from previous encounters with the same problem during training. We hypothesize that this limits their ability to progressively make solutions more concise over time. To address this, we present History-Aware Policy Optimization (HAPO), which keeps track of a history state (e.g., the minimum length over previously generated correct responses) for each problem. HAPO employs a novel length reward function based on this history state to incentivize the discovery of correct solutions that are more concise than those previously found. Crucially, this reward structure avoids overly penalizing shorter incorrect responses with the goal of facilitating exploration towards more efficient solutions. By combining this length reward with a correctness reward, HAPO jointly optimizes for correctness and efficiency. We use HAPO to train DeepSeek-R1-Distill-Qwen-1.5B, DeepScaleR-1.5B-Preview, and Qwen-2.5-1.5B-Instruct, and evaluate HAPO on several math benchmarks that span various difficulty levels. Experiment results demonstrate that HAPO effectively induces LLMs' concise reasoning abilities, producing length reductions of 33-59% with accuracy drops of only 2-5%. 

---
# MiniGPT-Pancreas: Multimodal Large Language Model for Pancreas Cancer Classification and Detection 

**Authors**: Andrea Moglia, Elia Clement Nastasio, Luca Mainardi, Pietro Cerveri  

**Link**: [PDF](https://arxiv.org/pdf/2412.15925)  

**Abstract**: Problem: Pancreas radiological imaging is challenging due to the small size, blurred boundaries, and variability of shape and position of the organ among patients. Goal: In this work we present MiniGPT-Pancreas, a Multimodal Large Language Model (MLLM), as an interactive chatbot to support clinicians in pancreas cancer diagnosis by integrating visual and textual information. Methods: MiniGPT-v2, a general-purpose MLLM, was fine-tuned in a cascaded way for pancreas detection, tumor classification, and tumor detection with multimodal prompts combining questions and computed tomography scans from the National Institute of Health (NIH), and Medical Segmentation Decathlon (MSD) datasets. The AbdomenCT-1k dataset was used to detect the liver, spleen, kidney, and pancreas. Results: MiniGPT-Pancreas achieved an Intersection over Union (IoU) of 0.595 and 0.550 for the detection of pancreas on NIH and MSD datasets, respectively. For the pancreas cancer classification task on the MSD dataset, accuracy, precision, and recall were 0.876, 0.874, and 0.878, respectively. When evaluating MiniGPT-Pancreas on the AbdomenCT-1k dataset for multi-organ detection, the IoU was 0.8399 for the liver, 0.722 for the kidney, 0.705 for the spleen, and 0.497 for the pancreas. For the pancreas tumor detection task, the IoU score was 0.168 on the MSD dataset. Conclusions: MiniGPT-Pancreas represents a promising solution to support clinicians in the classification of pancreas images with pancreas tumors. Future research is needed to improve the score on the detection task, especially for pancreas tumors. 

---
# Cog-RAG: Cognitive-Inspired Dual-Hypergraph with Theme Alignment Retrieval-Augmented Generation 

**Authors**: Hao Hu, Yifan Feng, Ruoxue Li, Rundong Xue, Xingliang Hou, Zhiqiang Tian, Yue Gao, Shaoyi Du  

**Link**: [PDF](https://arxiv.org/pdf/2511.13201)  

**Abstract**: Retrieval-Augmented Generation (RAG) enhances the response quality and domain-specific performance of large language models (LLMs) by incorporating external knowledge to combat hallucinations. In recent research, graph structures have been integrated into RAG to enhance the capture of semantic relations between entities. However, it primarily focuses on low-order pairwise entity relations, limiting the high-order associations among multiple entities. Hypergraph-enhanced approaches address this limitation by modeling multi-entity interactions via hyperedges, but they are typically constrained to inter-chunk entity-level representations, overlooking the global thematic organization and alignment across chunks. Drawing inspiration from the top-down cognitive process of human reasoning, we propose a theme-aligned dual-hypergraph RAG framework (Cog-RAG) that uses a theme hypergraph to capture inter-chunk thematic structure and an entity hypergraph to model high-order semantic relations. Furthermore, we design a cognitive-inspired two-stage retrieval strategy that first activates query-relevant thematic content from the theme hypergraph, and then guides fine-grained recall and diffusion in the entity hypergraph, achieving semantic alignment and consistent generation from global themes to local details. Our extensive experiments demonstrate that Cog-RAG significantly outperforms existing state-of-the-art baseline approaches. 

---
# Can We Predict the Next Question? A Collaborative Filtering Approach to Modeling User Behavior 

**Authors**: Bokang Fu, Jiahao Wang, Xiaojing Liu, Yuli Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.12949)  

**Abstract**: In recent years, large language models (LLMs) have excelled in language understanding and generation, powering advanced dialogue and recommendation systems. However, a significant limitation persists: these systems often model user preferences statically, failing to capture the dynamic and sequential nature of interactive behaviors. The sequence of a user's historical questions provides a rich, implicit signal of evolving interests and cognitive patterns, yet leveraging this temporal data for predictive tasks remains challenging due to the inherent disconnect between language modeling and behavioral sequence modeling.
To bridge this gap, we propose a Collaborative Filtering-enhanced Question Prediction (CFQP) framework. CFQP dynamically models evolving user-question interactions by integrating personalized memory modules with graph-based preference propagation. This dual mechanism allows the system to adaptively learn from user-specific histories while refining predictions through collaborative signals from similar users. Experimental results demonstrate that our approach effectively generates agents that mimic real-user questioning patterns, highlighting its potential for building proactive and adaptive dialogue systems. 

---
# ComLQ: Benchmarking Complex Logical Queries in Information Retrieval 

**Authors**: Ganlin Xu, Zhitao Yin, Linghao Zhang, Jiaqing Liang, Weijia Lu, Xiaodong Zhang, Zhifei Yang, Sihang Jiang, Deqing Yang  

**Link**: [PDF](https://arxiv.org/pdf/2511.12004)  

**Abstract**: Information retrieval (IR) systems play a critical role in navigating information overload across various applications. Existing IR benchmarks primarily focus on simple queries that are semantically analogous to single- and multi-hop relations, overlooking \emph{complex logical queries} involving first-order logic operations such as conjunction ($\land$), disjunction ($\lor$), and negation ($\lnot$). Thus, these benchmarks can not be used to sufficiently evaluate the performance of IR models on complex queries in real-world scenarios. To address this problem, we propose a novel method leveraging large language models (LLMs) to construct a new IR dataset \textbf{ComLQ} for \textbf{Com}plex \textbf{L}ogical \textbf{Q}ueries, which comprises 2,909 queries and 11,251 candidate passages. A key challenge in constructing the dataset lies in capturing the underlying logical structures within unstructured text. Therefore, by designing the subgraph-guided prompt with the subgraph indicator, an LLM (such as GPT-4o) is guided to generate queries with specific logical structures based on selected passages. All query-passage pairs in ComLQ are ensured \emph{structure conformity} and \emph{evidence distribution} through expert annotation. To better evaluate whether retrievers can handle queries with negation, we further propose a new evaluation metric, \textbf{Log-Scaled Negation Consistency} (\textbf{LSNC@$K$}). As a supplement to standard relevance-based metrics (such as nDCG and mAP), LSNC@$K$ measures whether top-$K$ retrieved passages violate negation conditions in queries. Our experimental results under zero-shot settings demonstrate existing retrieval models' limited performance on complex logical queries, especially on queries with negation, exposing their inferior capabilities of modeling exclusion. 

---
