# Preservation of Language Understanding Capabilities in Speech-aware Large Language Models 

**Authors**: Marek Kubis, Paweł Skórzewski, Iwona Christop, Mateusz Czyżnikiewicz, Jakub Kubiak, Łukasz Bondaruk, Marcin Lewandowski  

**Link**: [PDF](https://arxiv.org/pdf/2509.12171)  

**Abstract**: The paper presents C3T (Cross-modal Capabilities Conservation Test), a new benchmark for assessing the performance of speech-aware large language models. The benchmark utilizes textual tasks and a voice cloning text-to-speech model to quantify the extent to which language understanding capabilities are preserved when the model is accessed via speech input. C3T quantifies the fairness of the model for different categories of speakers and its robustness across text and speech modalities. 

---
# RAGs to Riches: RAG-like Few-shot Learning for Large Language Model Role-playing 

**Authors**: Timothy Rupprecht, Enfu Nan, Arash Akbari, Arman Akbari, Lei Lu, Priyanka Maan, Sean Duffy, Pu Zhao, Yumei He, David Kaeli, Yanzhi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.12168)  

**Abstract**: Role-playing Large language models (LLMs) are increasingly deployed in high-stakes domains such as healthcare, education, and governance, where failures can directly impact user trust and well-being. A cost effective paradigm for LLM role-playing is few-shot learning, but existing approaches often cause models to break character in unexpected and potentially harmful ways, especially when interacting with hostile users. Inspired by Retrieval-Augmented Generation (RAG), we reformulate LLM role-playing into a text retrieval problem and propose a new prompting framework called RAGs-to-Riches, which leverages curated reference demonstrations to condition LLM responses. We evaluate our framework with LLM-as-a-judge preference voting and introduce two novel token-level ROUGE metrics: Intersection over Output (IOO) to quantity how much an LLM improvises and Intersection over References (IOR) to measure few-shot demonstrations utilization rate during the evaluation tasks. When simulating interactions with a hostile user, our prompting strategy incorporates in its responses during inference an average of 35% more tokens from the reference demonstrations. As a result, across 453 role-playing interactions, our models are consistently judged as being more authentic, and remain in-character more often than zero-shot and in-context Learning (ICL) methods. Our method presents a scalable strategy for building robust, human-aligned LLM role-playing frameworks. 

---
# Pun Unintended: LLMs and the Illusion of Humor Understanding 

**Authors**: Alessandro Zangari, Matteo Marcuzzo, Andrea Albarelli, Mohammad Taher Pilehvar, Jose Camacho-Collados  

**Link**: [PDF](https://arxiv.org/pdf/2509.12158)  

**Abstract**: Puns are a form of humorous wordplay that exploits polysemy and phonetic similarity. While LLMs have shown promise in detecting puns, we show in this paper that their understanding often remains shallow, lacking the nuanced grasp typical of human interpretation. By systematically analyzing and reformulating existing pun benchmarks, we demonstrate how subtle changes in puns are sufficient to mislead LLMs. Our contributions include comprehensive and nuanced pun detection benchmarks, human evaluation across recent LLMs, and an analysis of the robustness challenges these models face in processing puns. 

---
# XplaiNLP at CheckThat! 2025: Multilingual Subjectivity Detection with Finetuned Transformers and Prompt-Based Inference with Large Language Models 

**Authors**: Ariana Sahitaj, Jiaao Li, Pia Wenzel Neves, Fedor Splitt, Premtim Sahitaj, Charlott Jakob, Veronika Solopova, Vera Schmitt  

**Link**: [PDF](https://arxiv.org/pdf/2509.12130)  

**Abstract**: This notebook reports the XplaiNLP submission to the CheckThat! 2025 shared task on multilingual subjectivity detection. We evaluate two approaches: (1) supervised fine-tuning of transformer encoders, EuroBERT, XLM-RoBERTa, and German-BERT, on monolingual and machine-translated training data; and (2) zero-shot prompting using two LLMs: o3-mini for Annotation (rule-based labelling) and gpt-4.1-mini for DoubleDown (contrastive rewriting) and Perspective (comparative reasoning). The Annotation Approach achieves 1st place in the Italian monolingual subtask with an F_1 score of 0.8104, outperforming the baseline of 0.6941. In the Romanian zero-shot setting, the fine-tuned XLM-RoBERTa model obtains an F_1 score of 0.7917, ranking 3rd and exceeding the baseline of 0.6461. The same model also performs reliably in the multilingual task and improves over the baseline in Greek. For German, a German-BERT model fine-tuned on translated training data from typologically related languages yields competitive performance over the baseline. In contrast, performance in the Ukrainian and Polish zero-shot settings falls slightly below the respective baselines, reflecting the challenge of generalization in low-resource cross-lingual scenarios. 

---
# CBP-Tuning: Efficient Local Customization for Black-box Large Language Models 

**Authors**: Jiaxuan Zhao, Naibin Gu, Yuchen Feng, Xiyu Liu, Peng Fu, Zheng Lin, Weiping Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.12112)  

**Abstract**: The high costs of customizing large language models (LLMs) fundamentally limit their adaptability to user-specific needs. Consequently, LLMs are increasingly offered as cloud-based services, a paradigm that introduces critical limitations: providers struggle to support personalized customization at scale, while users face privacy risks when exposing sensitive data. To address this dual challenge, we propose Customized Black-box Prompt Tuning (CBP-Tuning), a novel framework that facilitates efficient local customization while preserving bidirectional privacy. Specifically, we design a two-stage framework: (1) a prompt generator trained on the server-side to capture domain-specific and task-agnostic capabilities, and (2) user-side gradient-free optimization that tailors soft prompts for individual tasks. This approach eliminates the need for users to access model weights or upload private data, requiring only a single customized vector per task while achieving effective adaptation. Furthermore, the evaluation of CBP-Tuning in the commonsense reasoning, medical and financial domain settings demonstrates superior performance compared to baselines, showcasing its advantages in task-agnostic processing and privacy preservation. 

---
# GTA: Supervised-Guided Reinforcement Learning for Text Classification with Large Language Models 

**Authors**: Min Zeng, Jinfei Sun, Xueyou Luo, Caiquan Liu, Shiqi Zhang, Li Xie, Xiaoxin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.12108)  

**Abstract**: In natural language processing tasks, pure reinforcement learning (RL) fine-tuning methods often suffer from inefficient exploration and slow convergence; while supervised fine-tuning (SFT) methods, although efficient in training, have limited performance ceiling and less solid theoretical foundation compared to RL. To address efficiency-capability trade-off, we propose the Guess-Think-Answer (GTA) framework that combines the efficiency of SFT with the capability gains of RL in a unified training paradigm. GTA works by having the model first produce a provisional guess (optimized via cross-entropy loss), then reflect on this guess before generating the final answer, with RL rewards shaping both the final output and the format of the entire GTA structure. This hybrid approach achieves both faster convergence than pure RL and higher performance ceiling than pure SFT. To mitigate gradient conflicts between the two training signals, we employ loss masking and gradient constraints. Empirical results on four text classification benchmarks demonstrate that GTA substantially accelerates convergence while outperforming both standalone SFT and RL baselines. 

---
# In-domain SSL pre-training and streaming ASR 

**Authors**: Jarod Duret, Salima Mdhaffar, Gaëlle Laperrière, Ryan Whetten, Audrey Galametz, Catherine Kobus, Marion-Cécile Martin, Jo Oleiwan, Yannick Estève  

**Link**: [PDF](https://arxiv.org/pdf/2509.12101)  

**Abstract**: In this study, we investigate the benefits of domain-specific self-supervised pre-training for both offline and streaming ASR in Air Traffic Control (ATC) environments. We train BEST-RQ models on 4.5k hours of unlabeled ATC data, then fine-tune on a smaller supervised ATC set. To enable real-time processing, we propose using chunked attention and dynamic convolutions, ensuring low-latency inference. We compare these in-domain SSL models against state-of-the-art, general-purpose speech encoders such as w2v-BERT 2.0 and HuBERT. Results show that domain-adapted pre-training substantially improves performance on standard ATC benchmarks, significantly reducing word error rates when compared to models trained on broad speech corpora. Furthermore, the proposed streaming approach further improves word error rate under tighter latency constraints, making it particularly suitable for safety-critical aviation applications. These findings highlight that specializing SSL representations for ATC data is a practical path toward more accurate and efficient ASR systems in real-world operational settings. 

---
# Is 'Hope' a person or an idea? A pilot benchmark for NER: comparing traditional NLP tools and large language models on ambiguous entities 

**Authors**: Payam Latifi  

**Link**: [PDF](https://arxiv.org/pdf/2509.12098)  

**Abstract**: This pilot study presents a small-scale but carefully annotated benchmark of Named Entity Recognition (NER) performance across six systems: three non-LLM NLP tools (NLTK, spaCy, Stanza) and three general-purpose large language models (LLMs: Gemini-1.5-flash, DeepSeek-V3, Qwen-3-4B). The dataset contains 119 tokens covering five entity types (PERSON, LOCATION, ORGANIZATION, DATE, TIME). We evaluated each system's output against the manually annotated gold standard dataset using F1-score. The results show that LLMs generally outperform conventional tools in recognizing context-sensitive entities like person names, with Gemini achieving the highest average F1-score. However, traditional systems like Stanza demonstrate greater consistency in structured tags such as LOCATION and DATE. We also observed variability among LLMs, particularly in handling temporal expressions and multi-word organizations. Our findings highlight that while LLMs offer improved contextual understanding, traditional tools remain competitive in specific tasks, informing model selection. 

---
# SENSE models: an open source solution for multilingual and multimodal semantic-based tasks 

**Authors**: Salima Mdhaffar, Haroun Elleuch, Chaimae Chellaf, Ha Nguyen, Yannick Estève  

**Link**: [PDF](https://arxiv.org/pdf/2509.12093)  

**Abstract**: This paper introduces SENSE (Shared Embedding for N-lingual Speech and tExt), an open-source solution inspired by the SAMU-XLSR framework and conceptually similar to Meta AI's SONAR models. These approaches rely on a teacher-student framework to align a self-supervised speech encoder with the language-agnostic continuous representations of a text encoder at the utterance level. We describe how the original SAMU-XLSR method has been updated by selecting a stronger teacher text model and a better initial speech encoder. The source code for training and using SENSE models has been integrated into the SpeechBrain toolkit, and the first SENSE model we trained has been publicly released. We report experimental results on multilingual and multimodal semantic tasks, where our SENSE model achieves highly competitive performance. Finally, this study offers new insights into how semantics are captured in such semantically aligned speech encoders. 

---
# Steering Language Models in Multi-Token Generation: A Case Study on Tense and Aspect 

**Authors**: Alina Klerings, Jannik Brinkmann, Daniel Ruffinelli, Simone Ponzetto  

**Link**: [PDF](https://arxiv.org/pdf/2509.12065)  

**Abstract**: Large language models (LLMs) are able to generate grammatically well-formed text, but how do they encode their syntactic knowledge internally? While prior work has focused largely on binary grammatical contrasts, in this work, we study the representation and control of two multidimensional hierarchical grammar phenomena - verb tense and aspect - and for each, identify distinct, orthogonal directions in residual space using linear discriminant analysis. Next, we demonstrate causal control over both grammatical features through concept steering across three generation tasks. Then, we use these identified features in a case study to investigate factors influencing effective steering in multi-token generation. We find that steering strength, location, and duration are crucial parameters for reducing undesirable side effects such as topic shift and degeneration. Our findings suggest that models encode tense and aspect in structurally organized, human-like ways, but effective control of such features during generation is sensitive to multiple factors and requires manual tuning or automated optimization. 

---
# Text Adaptation to Plain Language and Easy Read via Automatic Post-Editing Cycles 

**Authors**: Jesús Calleja, David Ponce, Thierry Etchegoyhen  

**Link**: [PDF](https://arxiv.org/pdf/2509.11991)  

**Abstract**: We describe Vicomtech's participation in the CLEARS challenge on text adaptation to Plain Language and Easy Read in Spanish. Our approach features automatic post-editing of different types of initial Large Language Model adaptations, where successive adaptations are generated iteratively until readability and similarity metrics indicate that no further adaptation refinement can be successfully performed. Taking the average of all official metrics, our submissions achieved first and second place in Plain language and Easy Read adaptation, respectively. 

---
# Query-Focused Extractive Summarization for Sentiment Explanation 

**Authors**: Ahmed Moubtahij, Sylvie Ratté, Yazid Attabi, Maxime Dumas  

**Link**: [PDF](https://arxiv.org/pdf/2509.11989)  

**Abstract**: Constructive analysis of feedback from clients often requires determining the cause of their sentiment from a substantial amount of text documents. To assist and improve the productivity of such endeavors, we leverage the task of Query-Focused Summarization (QFS). Models of this task are often impeded by the linguistic dissonance between the query and the source documents. We propose and substantiate a multi-bias framework to help bridge this gap at a domain-agnostic, generic level; we then formulate specialized approaches for the problem of sentiment explanation through sentiment-based biases and query expansion. We achieve experimental results outperforming baseline models on a real-world proprietary sentiment-aware QFS dataset. 

---
# ToolRM: Outcome Reward Models for Tool-Calling Large Language Models 

**Authors**: Mayank Agarwal, Ibrahim Abdelaziz, Kinjal Basu, Merve Unuvar, Luis A. Lastras, Yara Rizk, Pavan Kapanipathi  

**Link**: [PDF](https://arxiv.org/pdf/2509.11963)  

**Abstract**: As large language models (LLMs) increasingly interact with external tools, reward modeling for tool use has become a critical yet underexplored area. Existing reward models, trained primarily on natural language outputs, struggle to evaluate tool-based reasoning and execution. To quantify this gap, we introduce FC-RewardBench, the first benchmark designed to systematically assess reward models' performance in tool-calling scenarios. Our analysis shows that current reward models often miss key signals of effective tool use, highlighting the need for domain-specific modeling. To address this, we propose a training framework for outcome-based reward models using data synthesized from permissively licensed, open-weight LLMs. We train models ranging from 1.7B to 14B parameters and evaluate them across seven out-of-domain benchmarks. These models consistently outperform general-purpose baselines, achieving up to 25\% average improvement in downstream task performance and enabling data-efficient fine-tuning through reward-guided filtering. 

---
# Spec-LLaVA: Accelerating Vision-Language Models with Dynamic Tree-Based Speculative Decoding 

**Authors**: Mingxiao Huo, Jiayi Zhang, Hewei Wang, Jinfeng Xu, Zheyu Chen, Huilin Tai, Yijun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.11961)  

**Abstract**: Vision-Language Models (VLMs) enable powerful multimodal reasoning but suffer from slow autoregressive inference, limiting their deployment in real-time applications. We introduce Spec-LLaVA, a system that applies speculative decoding to accelerate VLMs without sacrificing output quality. Spec-LLaVA pairs a lightweight draft VLM with a large target model: the draft speculates future tokens, which the target verifies in parallel, allowing multiple tokens to be generated per step. To maximize efficiency, we design a dynamic tree-based verification algorithm that adaptively expands and prunes speculative branches using draft model confidence. On MS COCO out-of-domain images, Spec-LLaVA achieves up to 3.28$\times$ faster decoding on LLaVA-1.5 (7B, 13B) with no loss in generation quality. This work presents a lossless acceleration framework for VLMs using dynamic tree-structured speculative decoding, opening a path toward practical real-time multimodal assistants. Importantly, the lightweight draft model design makes the framework amenable to resource-constrained or on-device deployment settings. 

---
# Designing LLMs for cultural sensitivity: Evidence from English-Japanese translation 

**Authors**: Helene Tenzer, Oumnia Abidi, Stefan Feuerriegel  

**Link**: [PDF](https://arxiv.org/pdf/2509.11921)  

**Abstract**: Large language models (LLMs) are increasingly used in everyday communication, including multilingual interactions across different cultural contexts. While LLMs can now generate near-perfect literal translations, it remains unclear whether LLMs support culturally appropriate communication. In this paper, we analyze the cultural sensitivity of different LLM designs when applied to English-Japanese translations of workplace e-mails. Here, we vary the prompting strategies: (1) naive "just translate" prompts, (2) audience-targeted prompts specifying the recipient's cultural background, and (3) instructional prompts with explicit guidance on Japanese communication norms. Using a mixed-methods study, we then analyze culture-specific language patterns to evaluate how well translations adapt to cultural norms. Further, we examine the appropriateness of the tone of the translations as perceived by native speakers. We find that culturally-tailored prompting can improve cultural fit, based on which we offer recommendations for designing culturally inclusive LLMs in multilingual settings. 

---
# Uncertainty in Authorship: Why Perfect AI Detection Is Mathematically Impossible 

**Authors**: Aadil Gani Ganie  

**Link**: [PDF](https://arxiv.org/pdf/2509.11915)  

**Abstract**: As large language models (LLMs) become more advanced, it is increasingly difficult to distinguish between human-written and AI-generated text. This paper draws a conceptual parallel between quantum uncertainty and the limits of authorship detection in natural language. We argue that there is a fundamental trade-off: the more confidently one tries to identify whether a text was written by a human or an AI, the more one risks disrupting the text's natural flow and authenticity. This mirrors the tension between precision and disturbance found in quantum systems. We explore how current detection methods--such as stylometry, watermarking, and neural classifiers--face inherent limitations. Enhancing detection accuracy often leads to changes in the AI's output, making other features less reliable. In effect, the very act of trying to detect AI authorship introduces uncertainty elsewhere in the text. Our analysis shows that when AI-generated text closely mimics human writing, perfect detection becomes not just technologically difficult but theoretically impossible. We address counterarguments and discuss the broader implications for authorship, ethics, and policy. Ultimately, we suggest that the challenge of AI-text detection is not just a matter of better tools--it reflects a deeper, unavoidable tension in the nature of language itself. 

---
# Growing Perspectives: Modelling Embodied Perspective Taking and Inner Narrative Development Using Large Language Models 

**Authors**: Sabrina Patania, Luca Annese, Anna Lambiase, Anita Pellegrini, Tom Foulsham, Azzurra Ruggeri, Silvia Rossi, Silvia Serino, Dimitri Ognibene  

**Link**: [PDF](https://arxiv.org/pdf/2509.11868)  

**Abstract**: Language and embodied perspective taking are essential for human collaboration, yet few computational models address both simultaneously. This work investigates the PerspAct system [1], which integrates the ReAct (Reason and Act) paradigm with Large Language Models (LLMs) to simulate developmental stages of perspective taking, grounded in Selman's theory [2]. Using an extended director task, we evaluate GPT's ability to generate internal narratives aligned with specified developmental stages, and assess how these influence collaborative performance both qualitatively (action selection) and quantitatively (task efficiency). Results show that GPT reliably produces developmentally-consistent narratives before task execution but often shifts towards more advanced stages during interaction, suggesting that language exchanges help refine internal representations. Higher developmental stages generally enhance collaborative effectiveness, while earlier stages yield more variable outcomes in complex contexts. These findings highlight the potential of integrating embodied perspective taking and language in LLMs to better model developmental dynamics and stress the importance of evaluating internal speech during combined linguistic and embodied tasks. 

---
# MOOM: Maintenance, Organization and Optimization of Memory in Ultra-Long Role-Playing Dialogues 

**Authors**: Weishu Chen, Jinyi Tang, Zhouhui Hou, Shihao Han, Mingjie Zhan, Zhiyuan Huang, Delong Liu, Jiawei Guo, Zhicheng Zhao, Fei Su  

**Link**: [PDF](https://arxiv.org/pdf/2509.11860)  

**Abstract**: Memory extraction is crucial for maintaining coherent ultra-long dialogues in human-robot role-playing scenarios. However, existing methods often exhibit uncontrolled memory growth. To address this, we propose MOOM, the first dual-branch memory plugin that leverages literary theory by modeling plot development and character portrayal as core storytelling elements. Specifically, one branch summarizes plot conflicts across multiple time scales, while the other extracts the user's character profile. MOOM further integrates a forgetting mechanism, inspired by the ``competition-inhibition'' memory theory, to constrain memory capacity and mitigate uncontrolled growth. Furthermore, we present ZH-4O, a Chinese ultra-long dialogue dataset specifically designed for role-playing, featuring dialogues that average 600 turns and include manually annotated memory information. Experimental results demonstrate that MOOM outperforms all state-of-the-art memory extraction methods, requiring fewer large language model invocations while maintaining a controllable memory capacity. 

---
# SCDTour: Embedding Axis Ordering and Merging for Interpretable Semantic Change Detection 

**Authors**: Taichi Aida, Danushka Bollegala  

**Link**: [PDF](https://arxiv.org/pdf/2509.11818)  

**Abstract**: In Semantic Change Detection (SCD), it is a common problem to obtain embeddings that are both interpretable and high-performing. However, improving interpretability often leads to a loss in the SCD performance, and vice versa. To address this problem, we propose SCDTour, a method that orders and merges interpretable axes to alleviate the performance degradation of SCD. SCDTour considers both (a) semantic similarity between axes in the embedding space, as well as (b) the degree to which each axis contributes to semantic change. Experimental results show that SCDTour preserves performance in semantic change detection while maintaining high interpretability. Moreover, agglomerating the sorted axes produces a more refined set of word senses, which achieves comparable or improved performance against the original full-dimensional embeddings in the SCD task. These findings demonstrate that SCDTour effectively balances interpretability and SCD performance, enabling meaningful interpretation of semantic shifts through a small number of refined axes. Source code is available at this https URL . 

---
# PledgeTracker: A System for Monitoring the Fulfilment of Pledges 

**Authors**: Yulong Chen, Michael Sejr Schlichtkrull, Zhenyun Deng, David Corney, Nasim Asl, Joshua Salisbury, Andrew Dudfield, Andreas Vlachos  

**Link**: [PDF](https://arxiv.org/pdf/2509.11804)  

**Abstract**: Political pledges reflect candidates' policy commitments, but tracking their fulfilment requires reasoning over incremental evidence distributed across multiple, dynamically updated sources. Existing methods simplify this task into a document classification task, overlooking its dynamic, temporal and multi-document nature. To address this issue, we introduce \textsc{PledgeTracker}, a system that reformulates pledge verification into structured event timeline construction. PledgeTracker consists of three core components: (1) a multi-step evidence retrieval module; (2) a timeline construction module and; (3) a fulfilment filtering module, allowing the capture of the evolving nature of pledge fulfilment and producing interpretable and structured timelines. We evaluate PledgeTracker in collaboration with professional fact-checkers in real-world workflows, demonstrating its effectiveness in retrieving relevant evidence and reducing human verification effort. 

---
# From Fuzzy Speech to Medical Insight: Benchmarking LLMs on Noisy Patient Narratives 

**Authors**: Eden Mama, Liel Sheri, Yehudit Aperstein, Alexander Apartsin  

**Link**: [PDF](https://arxiv.org/pdf/2509.11803)  

**Abstract**: The widespread adoption of large language models (LLMs) in healthcare raises critical questions about their ability to interpret patient-generated narratives, which are often informal, ambiguous, and noisy. Existing benchmarks typically rely on clean, structured clinical text, offering limited insight into model performance under realistic conditions. In this work, we present a novel synthetic dataset designed to simulate patient self-descriptions characterized by varying levels of linguistic noise, fuzzy language, and layperson terminology. Our dataset comprises clinically consistent scenarios annotated with ground-truth diagnoses, spanning a spectrum of communication clarity to reflect diverse real-world reporting styles. Using this benchmark, we fine-tune and evaluate several state-of-the-art models (LLMs), including BERT-based and encoder-decoder T5 models. To support reproducibility and future research, we release the Noisy Diagnostic Benchmark (NDB), a structured dataset of noisy, synthetic patient descriptions designed to stress-test and compare the diagnostic capabilities of large language models (LLMs) under realistic linguistic conditions. We made the benchmark available for the community: this https URL 

---
# When Curiosity Signals Danger: Predicting Health Crises Through Online Medication Inquiries 

**Authors**: Dvora Goncharok, Arbel Shifman, Alexander Apartsin, Yehudit Aperstein  

**Link**: [PDF](https://arxiv.org/pdf/2509.11802)  

**Abstract**: Online medical forums are a rich and underutilized source of insight into patient concerns, especially regarding medication use. Some of the many questions users pose may signal confusion, misuse, or even the early warning signs of a developing health crisis. Detecting these critical questions that may precede severe adverse events or life-threatening complications is vital for timely intervention and improving patient safety. This study introduces a novel annotated dataset of medication-related questions extracted from online forums. Each entry is manually labelled for criticality based on clinical risk factors. We benchmark the performance of six traditional machine learning classifiers using TF-IDF textual representations, alongside three state-of-the-art large language model (LLM)-based classification approaches that leverage deep contextual understanding. Our results highlight the potential of classical and modern methods to support real-time triage and alert systems in digital health spaces. The curated dataset is made publicly available to encourage further research at the intersection of patient-generated data, natural language processing, and early warning systems for critical health events. The dataset and benchmark are available at: this https URL. 

---
# User eXperience Perception Insights Dataset (UXPID): Synthetic User Feedback from Public Industrial Forums 

**Authors**: Mikhail Kulyabin, Jan Joosten, Choro Ulan uulu, Nuno Miguel Martins Pacheco, Fabian Ries, Filippos Petridis, Jan Bosch, Helena Holmström Olsson  

**Link**: [PDF](https://arxiv.org/pdf/2509.11777)  

**Abstract**: Customer feedback in industrial forums reflect a rich but underexplored source of insight into real-world product experience. These publicly shared discussions offer an organic view of user expectations, frustrations, and success stories shaped by the specific contexts of use. Yet, harnessing this information for systematic analysis remains challenging due to the unstructured and domain-specific nature of the content. The lack of structure and specialized vocabulary makes it difficult for traditional data analysis techniques to accurately interpret, categorize, and quantify the feedback, thereby limiting its potential to inform product development and support strategies. To address these challenges, this paper presents the User eXperience Perception Insights Dataset (UXPID), a collection of 7130 artificially synthesized and anonymized user feedback branches extracted from a public industrial automation forum. Each JavaScript object notation (JSON) record contains multi-post comments related to specific hardware and software products, enriched with metadata and contextual conversation data. Leveraging a large language model (LLM), each branch is systematically analyzed and annotated for UX insights, user expectations, severity and sentiment ratings, and topic classifications. The UXPID dataset is designed to facilitate research in user requirements, user experience (UX) analysis, and AI-driven feedback processing, particularly where privacy and licensing restrictions limit access to real-world data. UXPID supports the training and evaluation of transformer-based models for tasks such as issue detection, sentiment analysis, and requirements extraction in the context of technical forums. 

---
# An Agentic Toolkit for Adaptive Information Extraction from Regulatory Documents 

**Authors**: Gaye Colakoglu, Gürkan Solmaz, Jonathan Fürst  

**Link**: [PDF](https://arxiv.org/pdf/2509.11773)  

**Abstract**: Declaration of Performance (DoP) documents, mandated by EU regulation, certify the performance of construction products. While some of their content is standardized, DoPs vary widely in layout, language, schema, and format, posing challenges for automated key-value pair extraction (KVP) and question answering (QA). Existing static or LLM-only IE pipelines often hallucinate and fail to adapt to this structural diversity. Our domain-specific, stateful agentic system addresses these challenges through a planner-executor-responder architecture. The system infers user intent, detects document modality, and orchestrates tools dynamically for robust, traceable reasoning while avoiding tool misuse or execution loops. Evaluation on a curated DoP dataset demonstrates improved robustness across formats and languages, offering a scalable solution for structured data extraction in regulated workflows. 

---
# Room acoustics affect communicative success in hybrid meeting spaces: a pilot study 

**Authors**: Robert Einig, Stefan Janscha, Jonas Schuster, Julian Koch, Martin Hagmueller, Barbara Schuppler  

**Link**: [PDF](https://arxiv.org/pdf/2509.11709)  

**Abstract**: Since the COVID-19 pandemic in 2020, universities and companies have increasingly integrated hybrid features into their meeting spaces, or even created dedicated rooms for this purpose. While the importance of a fast and stable internet connection is often prioritized, the acoustic design of seminar rooms is frequently overlooked. Poor acoustics, particularly excessive reverberation, can lead to issues such as misunderstandings, reduced speech intelligibility or cognitive and vocal fatigue. This pilot study investigates whether room acoustic interventions in a seminar room at Graz University of Technology support better communication in hybrid meetings. For this purpose, we recorded two groups of persons twice, once before and once after improving the acoustics of the room. Our findings -- despite not reaching statistical significance due to the small sample size - indicate clearly that our spatial interventions improve communicative success in hybrid meetings. To make the paper accessible also for readers from the speech communication community, we explain room acoustics background, relevant for the interpretation of our results. 

---
# CoachMe: Decoding Sport Elements with a Reference-Based Coaching Instruction Generation Model 

**Authors**: Wei-Hsin Yeh, Yu-An Su, Chih-Ning Chen, Yi-Hsueh Lin, Calvin Ku, Wen-Hsin Chiu, Min-Chun Hu, Lun-Wei Ku  

**Link**: [PDF](https://arxiv.org/pdf/2509.11698)  

**Abstract**: Motion instruction is a crucial task that helps athletes refine their technique by analyzing movements and providing corrective guidance. Although recent advances in multimodal models have improved motion understanding, generating precise and sport-specific instruction remains challenging due to the highly domain-specific nature of sports and the need for informative guidance. We propose CoachMe, a reference-based model that analyzes the differences between a learner's motion and a reference under temporal and physical aspects. This approach enables both domain-knowledge learning and the acquisition of a coach-like thinking process that identifies movement errors effectively and provides feedback to explain how to improve. In this paper, we illustrate how CoachMe adapts well to specific sports such as skating and boxing by learning from general movements and then leveraging limited data. Experiments show that CoachMe provides high-quality instructions instead of directions merely in the tone of a coach but without critical information. CoachMe outperforms GPT-4o by 31.6% in G-Eval on figure skating and by 58.3% on boxing. Analysis further confirms that it elaborates on errors and their corresponding improvement methods in the generated instructions. You can find CoachMe here: this https URL 

---
# A Dynamic Knowledge Update-Driven Model with Large Language Models for Fake News Detection 

**Authors**: Di Jin, Jun Yang, Xiaobao Wang, Junwei Zhang, Shuqi Li, Dongxiao He  

**Link**: [PDF](https://arxiv.org/pdf/2509.11687)  

**Abstract**: As the Internet and social media evolve rapidly, distinguishing credible news from a vast amount of complex information poses a significant challenge. Due to the suddenness and instability of news events, the authenticity labels of news can potentially shift as events develop, making it crucial for fake news detection to obtain the latest event updates. Existing methods employ retrieval-augmented generation to fill knowledge gaps, but they suffer from issues such as insufficient credibility of retrieved content and interference from noisy information. We propose a dynamic knowledge update-driven model for fake news detection (DYNAMO), which leverages knowledge graphs to achieve continuous updating of new knowledge and integrates with large language models to fulfill dual functions: news authenticity detection and verification of new knowledge correctness, solving the two key problems of ensuring the authenticity of new knowledge and deeply mining news semantics. Specifically, we first construct a news-domain-specific knowledge graph. Then, we use Monte Carlo Tree Search to decompose complex news and verify them step by step. Finally, we extract and update new knowledge from verified real news texts and reasoning paths. Experimental results demonstrate that DYNAMO achieves the best performance on two real-world datasets. 

---
# EthicsMH: A Pilot Benchmark for Ethical Reasoning in Mental Health AI 

**Authors**: Sai Kartheek Reddy Kasu  

**Link**: [PDF](https://arxiv.org/pdf/2509.11648)  

**Abstract**: The deployment of large language models (LLMs) in mental health and other sensitive domains raises urgent questions about ethical reasoning, fairness, and responsible alignment. Yet, existing benchmarks for moral and clinical decision-making do not adequately capture the unique ethical dilemmas encountered in mental health practice, where confidentiality, autonomy, beneficence, and bias frequently intersect. To address this gap, we introduce Ethical Reasoning in Mental Health (EthicsMH), a pilot dataset of 125 scenarios designed to evaluate how AI systems navigate ethically charged situations in therapeutic and psychiatric contexts. Each scenario is enriched with structured fields, including multiple decision options, expert-aligned reasoning, expected model behavior, real-world impact, and multi-stakeholder viewpoints. This structure enables evaluation not only of decision accuracy but also of explanation quality and alignment with professional norms. Although modest in scale and developed with model-assisted generation, EthicsMH establishes a task framework that bridges AI ethics and mental health decision-making. By releasing this dataset, we aim to provide a seed resource that can be expanded through community and expert contributions, fostering the development of AI systems capable of responsibly handling some of society's most delicate decisions. 

---
# AesBiasBench: Evaluating Bias and Alignment in Multimodal Language Models for Personalized Image Aesthetic Assessment 

**Authors**: Kun Li, Lai-Man Po, Hongzheng Yang, Xuyuan Xu, Kangcheng Liu, Yuzhi Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.11620)  

**Abstract**: Multimodal Large Language Models (MLLMs) are increasingly applied in Personalized Image Aesthetic Assessment (PIAA) as a scalable alternative to expert evaluations. However, their predictions may reflect subtle biases influenced by demographic factors such as gender, age, and education. In this work, we propose AesBiasBench, a benchmark designed to evaluate MLLMs along two complementary dimensions: (1) stereotype bias, quantified by measuring variations in aesthetic evaluations across demographic groups; and (2) alignment between model outputs and genuine human aesthetic preferences. Our benchmark covers three subtasks (Aesthetic Perception, Assessment, Empathy) and introduces structured metrics (IFD, NRD, AAS) to assess both bias and alignment. We evaluate 19 MLLMs, including proprietary models (e.g., GPT-4o, Claude-3.5-Sonnet) and open-source models (e.g., InternVL-2.5, Qwen2.5-VL). Results indicate that smaller models exhibit stronger stereotype biases, whereas larger models align more closely with human preferences. Incorporating identity information often exacerbates bias, particularly in emotional judgments. These findings underscore the importance of identity-aware evaluation frameworks in subjective vision-language tasks. 

---
# HalluDetect: Detecting, Mitigating, and Benchmarking Hallucinations in Conversational Systems 

**Authors**: Spandan Anaokar, Shrey Ganatra, Harshvivek Kashid, Swapnil Bhattacharyya, Shruti Nair, Reshma Sekhar, Siddharth Manohar, Rahul Hemrajani, Pushpak Bhattacharyya  

**Link**: [PDF](https://arxiv.org/pdf/2509.11619)  

**Abstract**: Large Language Models (LLMs) are widely used in industry but remain prone to hallucinations, limiting their reliability in critical applications. This work addresses hallucination reduction in consumer grievance chatbots built using LLaMA 3.1 8B Instruct, a compact model frequently used in industry. We develop HalluDetect, an LLM-based hallucination detection system that achieves an F1 score of 69% outperforming baseline detectors by 25.44%. Benchmarking five chatbot architectures, we find that out of them, AgentBot minimizes hallucinations to 0.4159 per turn while maintaining the highest token accuracy (96.13%), making it the most effective mitigation strategy. Our findings provide a scalable framework for hallucination mitigation, demonstrating that optimized inference strategies can significantly improve factual accuracy. While applied to consumer law, our approach generalizes to other high-risk domains, enhancing trust in LLM-driven assistants. We will release the code and dataset 

---
# Dynamic Span Interaction and Graph-Aware Memory for Entity-Level Sentiment Classification 

**Authors**: Md. Mithun Hossain, Sanjara, Md. Shakil Hossain, Sudipto Chaki  

**Link**: [PDF](https://arxiv.org/pdf/2509.11604)  

**Abstract**: Entity-level sentiment classification involves identifying the sentiment polarity linked to specific entities within text. This task poses several challenges: effectively modeling the subtle and complex interactions between entities and their surrounding sentiment expressions; capturing dependencies that may span across sentences; and ensuring consistent sentiment predictions for multiple mentions of the same entity through coreference resolution. Additionally, linguistic phenomena such as negation, ambiguity, and overlapping opinions further complicate the analysis. These complexities make entity-level sentiment classification a difficult problem, especially in real-world, noisy textual data. To address these issues, we propose SpanEIT, a novel framework integrating dynamic span interaction and graph-aware memory mechanisms for enhanced entity-sentiment relational modeling. SpanEIT builds span-based representations for entities and candidate sentiment phrases, employs bidirectional attention for fine-grained interactions, and uses a graph attention network to capture syntactic and co-occurrence relations. A coreference-aware memory module ensures entity-level consistency across documents. Experiments on FSAD, BARU, and IMDB datasets show SpanEIT outperforms state-of-the-art transformer and hybrid baselines in accuracy and F1 scores. Ablation and interpretability analyses validate the effectiveness of our approach, underscoring its potential for fine-grained sentiment analysis in applications like social media monitoring and customer feedback analysis. 

---
# Analyzing Information-Seeking Behaviors in a Hakka AI Chatbot: A Cognitive-Pragmatic Study 

**Authors**: Chu-Hsuan Lee, Chen-Chi Chang, Hung-Shin Lee, Yun-Hsiang Hsu, Ching-Yuan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.11591)  

**Abstract**: With many endangered languages at risk of disappearing, efforts to preserve them now rely more than ever on using technology alongside culturally informed teaching strategies. This study examines user behaviors in TALKA, a generative AI-powered chatbot designed for Hakka language engagement, by employing a dual-layered analytical framework grounded in Bloom's Taxonomy of cognitive processes and dialogue act categorization. We analyzed 7,077 user utterances, each carefully annotated according to six cognitive levels and eleven dialogue act types. These included a variety of functions, such as asking for information, requesting translations, making cultural inquiries, and using language creatively. Pragmatic classifications further highlight how different types of dialogue acts--such as feedback, control commands, and social greetings--align with specific cognitive intentions. The results suggest that generative AI chatbots can support language learning in meaningful ways--especially when they are designed with an understanding of how users think and communicate. They may also help learners express themselves more confidently and connect with their cultural identity. The TALKA case provides empirical insights into how AI-mediated dialogue facilitates cognitive development in low-resource language learners, as well as pragmatic negotiation and socio-cultural affiliation. By focusing on AI-assisted language learning, this study offers new insights into how technology can support language preservation and educational practice. 

---
# Bhaasha, Bhasa, Zaban: A Survey for Low-Resourced Languages in South Asia - Current Stage and Challenges 

**Authors**: Sampoorna Poria, Xiaolei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2509.11570)  

**Abstract**: Rapid developments of large language models have revolutionized many NLP tasks for English data. Unfortunately, the models and their evaluations for low-resource languages are being overlooked, especially for languages in South Asia. Although there are more than 650 languages in South Asia, many of them either have very limited computational resources or are missing from existing language models. Thus, a concrete question to be answered is: Can we assess the current stage and challenges to inform our NLP community and facilitate model developments for South Asian languages? In this survey, we have comprehensively examined current efforts and challenges of NLP models for South Asian languages by retrieving studies since 2020, with a focus on transformer-based models, such as BERT, T5, & GPT. We present advances and gaps across 3 essential aspects: data, models, & tasks, such as available data sources, fine-tuning strategies, & domain applications. Our findings highlight substantial issues, including missing data in critical domains (e.g., health), code-mixing, and lack of standardized evaluation benchmarks. Our survey aims to raise awareness within the NLP community for more targeted data curation, unify benchmarks tailored to cultural and linguistic nuances of South Asia, and encourage an equitable representation of South Asian languages. The complete list of resources is available at: this https URL. 

---
# D$^2$HScore: Reasoning-Aware Hallucination Detection via Semantic Breadth and Depth Analysis in LLMs 

**Authors**: Yue Ding, Xiaofang Zhu, Tianze Xia, Junfei Wu, Xinlong Chen, Qiang Liu, Liang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.11569)  

**Abstract**: Although large Language Models (LLMs) have achieved remarkable success, their practical application is often hindered by the generation of non-factual content, which is called "hallucination". Ensuring the reliability of LLMs' outputs is a critical challenge, particularly in high-stakes domains such as finance, security, and healthcare. In this work, we revisit hallucination detection from the perspective of model architecture and generation dynamics. Leveraging the multi-layer structure and autoregressive decoding process of LLMs, we decompose hallucination signals into two complementary dimensions: the semantic breadth of token representations within each layer, and the semantic depth of core concepts as they evolve across layers. Based on this insight, we propose \textbf{D$^2$HScore (Dispersion and Drift-based Hallucination Score)}, a training-free and label-free framework that jointly measures: (1) \textbf{Intra-Layer Dispersion}, which quantifies the semantic diversity of token representations within each layer; and (2) \textbf{Inter-Layer Drift}, which tracks the progressive transformation of key token representations across layers. To ensure drift reflects the evolution of meaningful semantics rather than noisy or redundant tokens, we guide token selection using attention signals. By capturing both the horizontal and vertical dynamics of representation during inference, D$^2$HScore provides an interpretable and lightweight proxy for hallucination detection. Extensive experiments across five open-source LLMs and five widely used benchmarks demonstrate that D$^2$HScore consistently outperforms existing training-free baselines. 

---
# HiChunk: Evaluating and Enhancing Retrieval-Augmented Generation with Hierarchical Chunking 

**Authors**: Wensheng Lu, Keyu Chen, Ruizhi Qiao, Xing Sun  

**Link**: [PDF](https://arxiv.org/pdf/2509.11552)  

**Abstract**: Retrieval-Augmented Generation (RAG) enhances the response capabilities of language models by integrating external knowledge sources. However, document chunking as an important part of RAG system often lacks effective evaluation tools. This paper first analyzes why existing RAG evaluation benchmarks are inadequate for assessing document chunking quality, specifically due to evidence sparsity. Based on this conclusion, we propose HiCBench, which includes manually annotated multi-level document chunking points, synthesized evidence-dense quetion answer(QA) pairs, and their corresponding evidence sources. Additionally, we introduce the HiChunk framework, a multi-level document structuring framework based on fine-tuned LLMs, combined with the Auto-Merge retrieval algorithm to improve retrieval quality. Experiments demonstrate that HiCBench effectively evaluates the impact of different chunking methods across the entire RAG pipeline. Moreover, HiChunk achieves better chunking quality within reasonable time consumption, thereby enhancing the overall performance of RAG systems. 

---
# HARP: Hallucination Detection via Reasoning Subspace Projection 

**Authors**: Junjie Hu, Gang Tu, ShengYu Cheng, Jinxin Li, Jinting Wang, Rui Chen, Zhilong Zhou, Dongbo Shan  

**Link**: [PDF](https://arxiv.org/pdf/2509.11536)  

**Abstract**: Hallucinations in Large Language Models (LLMs) pose a major barrier to their reliable use in critical decision-making. Although existing hallucination detection methods have improved accuracy, they still struggle with disentangling semantic and reasoning information and maintaining robustness. To address these challenges, we propose HARP (Hallucination detection via reasoning subspace projection), a novel hallucination detection framework. HARP establishes that the hidden state space of LLMs can be decomposed into a direct sum of a semantic subspace and a reasoning subspace, where the former encodes linguistic expression and the latter captures internal reasoning processes. Moreover, we demonstrate that the Unembedding layer can disentangle these subspaces, and by applying Singular Value Decomposition (SVD) to its parameters, the basis vectors spanning the semantic and reasoning subspaces are obtained. Finally, HARP projects hidden states onto the basis vectors of the reasoning subspace, and the resulting projections are then used as input features for hallucination detection in LLMs. By using these projections, HARP reduces the dimension of the feature to approximately 5% of the original, filters out most noise, and achieves enhanced robustness. Experiments across multiple datasets show that HARP achieves state-of-the-art hallucination detection performance; in particular, it achieves an AUROC of 92.8% on TriviaQA, outperforming the previous best method by 7.5%. 

---
# On the Distinctive Co-occurrence Characteristics of Antonymy 

**Authors**: Zhihan Cao, Hiroaki Yamada, Takenobu Tokunaga  

**Link**: [PDF](https://arxiv.org/pdf/2509.11534)  

**Abstract**: Antonymy has long received particular attention in lexical semantics. Previous studies have shown that antonym pairs frequently co-occur in text, across genres and parts of speech, more often than would be expected by chance. However, whether this co-occurrence pattern is distinctive of antonymy remains unclear, due to a lack of comparison with other semantic relations. This work fills the gap by comparing antonymy with three other relations across parts of speech using robust co-occurrence metrics. We find that antonymy is distinctive in three respects: antonym pairs co-occur with high strength, in a preferred linear order, and within short spans. All results are available online. 

---
# PeruMedQA: Benchmarking Large Language Models (LLMs) on Peruvian Medical Exams - Dataset Construction and Evaluation 

**Authors**: Rodrigo M. Carrillo-Larco, Jesus Lovón Melgarejo, Manuel Castillo-Cara, Gusseppe Bravo-Rocca  

**Link**: [PDF](https://arxiv.org/pdf/2509.11517)  

**Abstract**: BACKGROUND: Medical large language models (LLMS) have demonstrated remarkable performance in answering medical examinations. However, the extent to which this high performance is transferable to medical questions in Spanish and from a Latin American country remains unexplored. This knowledge is crucial as LLM-based medical applications gain traction in Latin America. AIMS: to build a dataset of questions from medical examinations taken by Peruvian physicians pursuing specialty training; to fine-tune a LLM on this dataset; to evaluate and compare the performance in terms of accuracy between vanilla LLMs and the fine-tuned LLM. METHODS: We curated PeruMedQA, a multiple-choice question-answering (MCQA) datasets containing 8,380 questions spanning 12 medical domains (2018-2025). We selected eight medical LLMs including medgemma-4b-it and medgemma-27b-text-it, and developed zero-shot task-specific prompts to answer the questions appropriately. We employed parameter-efficient fine tuning (PEFT)and low-rant adaptation (LoRA) to fine-tune medgemma-4b-it utilizing all questions except those from 2025 (test set). RESULTS: medgemma-27b-text-it outperformed all other models, achieving a proportion of correct answers exceeding 90% in several instances. LLMs with <10 billion parameters exhibited <60% of correct answers, while some exams yielded results <50%. The fine-tuned version of medgemma-4b-it emerged victorious agains all LLMs with <10 billion parameters and rivaled a LLM with 70 billion parameters across various examinations. CONCLUSIONS: For medical AI application and research that require knowledge bases from Spanish-speaking countries and those exhibiting similar epidemiological profiles to Peru's, interested parties should utilize medgemma-27b-text-it or a fine-tuned version of medgemma-4b-it. 

---
# LVLMs are Bad at Overhearing Human Referential Communication 

**Authors**: Zhengxiang Wang, Weiling Li, Panagiotis Kaliosis, Owen Rambow, Susan E. Brennan  

**Link**: [PDF](https://arxiv.org/pdf/2509.11514)  

**Abstract**: During spontaneous conversations, speakers collaborate on novel referring expressions, which they can then re-use in subsequent conversations. Understanding such referring expressions is an important ability for an embodied agent, so that it can carry out tasks in the real world. This requires integrating and understanding language, vision, and conversational interaction. We study the capabilities of seven state-of-the-art Large Vision Language Models (LVLMs) as overhearers to a corpus of spontaneous conversations between pairs of human discourse participants engaged in a collaborative object-matching task. We find that such a task remains challenging for current LVLMs and they all fail to show a consistent performance improvement as they overhear more conversations from the same discourse participants repeating the same task for multiple rounds. We release our corpus and code for reproducibility and to facilitate future research. 

---
# Unsupervised Candidate Ranking for Lexical Substitution via Holistic Sentence Semantics 

**Authors**: Zhongyang Hu, Naijie Gu, Xiangzhi Tao, Tianhui Gu, Yibing Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2509.11513)  

**Abstract**: A key subtask in lexical substitution is ranking the given candidate words. A common approach is to replace the target word with a candidate in the original sentence and feed the modified sentence into a model to capture semantic differences before and after substitution. However, effectively modeling the bidirectional influence of candidate substitution on both the target word and its context remains challenging. Existing methods often focus solely on semantic changes at the target position or rely on parameter tuning over multiple evaluation metrics, making it difficult to accurately characterize semantic variation. To address this, we investigate two approaches: one based on attention weights and another leveraging the more interpretable integrated gradients method, both designed to measure the influence of context tokens on the target token and to rank candidates by incorporating semantic similarity between the original and substituted sentences. Experiments on the LS07 and SWORDS datasets demonstrate that both approaches improve ranking performance. 

---
# DeDisCo at the DISRPT 2025 Shared Task: A System for Discourse Relation Classification 

**Authors**: Zhuoxuan Ju, Jingni Wu, Abhishek Purushothama, Amir Zeldes  

**Link**: [PDF](https://arxiv.org/pdf/2509.11498)  

**Abstract**: This paper presents DeDisCo, Georgetown University's entry in the DISRPT 2025 shared task on discourse relation classification. We test two approaches, using an mt5-based encoder and a decoder based approach using the openly available Qwen model. We also experiment on training with augmented dataset for low-resource languages using matched data translated automatically from English, as well as using some additional linguistic features inspired by entries in previous editions of the Shared Task. Our system achieves a macro-accuracy score of 71.28, and we provide some interpretation and error analysis for our results. 

---
# AKCIT-FN at CheckThat! 2025: Switching Fine-Tuned SLMs and LLM Prompting for Multilingual Claim Normalization 

**Authors**: Fabrycio Leite Nakano Almada, Kauan Divino Pouso Mariano, Maykon Adriell Dutra, Victor Emanuel da Silva Monteiro, Juliana Resplande Sant'Anna Gomes, Arlindo Rodrigues Galvão Filho, Anderson da Silva Soares  

**Link**: [PDF](https://arxiv.org/pdf/2509.11496)  

**Abstract**: Claim normalization, the transformation of informal social media posts into concise, self-contained statements, is a crucial step in automated fact-checking pipelines. This paper details our submission to the CLEF-2025 CheckThat! Task~2, which challenges systems to perform claim normalization across twenty languages, divided into thirteen supervised (high-resource) and seven zero-shot (no training data) tracks.
Our approach, leveraging fine-tuned Small Language Models (SLMs) for supervised languages and Large Language Model (LLM) prompting for zero-shot scenarios, achieved podium positions (top three) in fifteen of the twenty languages. Notably, this included second-place rankings in eight languages, five of which were among the seven designated zero-shot languages, underscoring the effectiveness of our LLM-based zero-shot strategy. For Portuguese, our initial development language, our system achieved an average METEOR score of 0.5290, ranking third. All implementation artifacts, including inference, training, evaluation scripts, and prompt configurations, are publicly available at this https URL. 

---
# ClaimIQ at CheckThat! 2025: Comparing Prompted and Fine-Tuned Language Models for Verifying Numerical Claims 

**Authors**: Anirban Saha Anik, Md Fahimul Kabir Chowdhury, Andrew Wyckoff, Sagnik Ray Choudhury  

**Link**: [PDF](https://arxiv.org/pdf/2509.11492)  

**Abstract**: This paper presents our system for Task 3 of the CLEF 2025 CheckThat! Lab, which focuses on verifying numerical and temporal claims using retrieved evidence. We explore two complementary approaches: zero-shot prompting with instruction-tuned large language models (LLMs) and supervised fine-tuning using parameter-efficient LoRA. To enhance evidence quality, we investigate several selection strategies, including full-document input and top-k sentence filtering using BM25 and MiniLM. Our best-performing model LLaMA fine-tuned with LoRA achieves strong performance on the English validation set. However, a notable drop in the test set highlights a generalization challenge. These findings underscore the importance of evidence granularity and model adaptation for robust numerical fact verification. 

---
# Improving LLMs' Learning for Coreference Resolution 

**Authors**: Yujian Gan, Yuan Liang, Yanni Lin, Juntao Yu, Massimo Poesio  

**Link**: [PDF](https://arxiv.org/pdf/2509.11466)  

**Abstract**: Coreference Resolution (CR) is crucial for many NLP tasks, but existing LLMs struggle with hallucination and under-performance. In this paper, we investigate the limitations of existing LLM-based approaches to CR-specifically the Question-Answering (QA) Template and Document Template methods and propose two novel techniques: Reversed Training with Joint Inference and Iterative Document Generation. Our experiments show that Reversed Training improves the QA Template method, while Iterative Document Generation eliminates hallucinations in the generated source text and boosts coreference resolution. Integrating these methods and techniques offers an effective and robust solution to LLM-based coreference resolution. 

---
# CEMTM: Contextual Embedding-based Multimodal Topic Modeling 

**Authors**: Amirhossein Abaskohi, Raymond Li, Chuyuan Li, Shafiq Joty, Giuseppe Carenini  

**Link**: [PDF](https://arxiv.org/pdf/2509.11465)  

**Abstract**: We introduce CEMTM, a context-enhanced multimodal topic model designed to infer coherent and interpretable topic structures from both short and long documents containing text and images. CEMTM builds on fine-tuned large vision language models (LVLMs) to obtain contextualized embeddings, and employs a distributional attention mechanism to weight token-level contributions to topic inference. A reconstruction objective aligns topic-based representations with the document embedding, encouraging semantic consistency across modalities. Unlike existing approaches, CEMTM can process multiple images per document without repeated encoding and maintains interpretability through explicit word-topic and document-topic distributions. Extensive experiments on six multimodal benchmarks show that CEMTM consistently outperforms unimodal and multimodal baselines, achieving a remarkable average LLM score of 2.61. Further analysis shows its effectiveness in downstream few-shot retrieval and its ability to capture visually grounded semantics in complex domains such as scientific articles. 

---
# CognitiveSky: Scalable Sentiment and Narrative Analysis for Decentralized Social Media 

**Authors**: Gaurab Chhetri, Anandi Dutta, Subasish Das  

**Link**: [PDF](https://arxiv.org/pdf/2509.11444)  

**Abstract**: The emergence of decentralized social media platforms presents new opportunities and challenges for real-time analysis of public discourse. This study introduces CognitiveSky, an open-source and scalable framework designed for sentiment, emotion, and narrative analysis on Bluesky, a federated Twitter or this http URL alternative. By ingesting data through Bluesky's Application Programming Interface (API), CognitiveSky applies transformer-based models to annotate large-scale user-generated content and produces structured and analyzable outputs. These summaries drive a dynamic dashboard that visualizes evolving patterns in emotion, activity, and conversation topics. Built entirely on free-tier infrastructure, CognitiveSky achieves both low operational cost and high accessibility. While demonstrated here for monitoring mental health discourse, its modular design enables applications across domains such as disinformation detection, crisis response, and civic sentiment analysis. By bridging large language models with decentralized networks, CognitiveSky offers a transparent, extensible tool for computational social science in an era of shifting digital ecosystems. 

---
# A Transformer-Based Cross-Platform Analysis of Public Discourse on the 15-Minute City Paradigm 

**Authors**: Gaurab Chhetri, Darrell Anderson, Boniphace Kutela, Subasish Das  

**Link**: [PDF](https://arxiv.org/pdf/2509.11443)  

**Abstract**: This study presents the first multi-platform sentiment analysis of public opinion on the 15-minute city concept across Twitter, Reddit, and news media. Using compressed transformer models and Llama-3-8B for annotation, we classify sentiment across heterogeneous text domains. Our pipeline handles long-form and short-form text, supports consistent annotation, and enables reproducible evaluation. We benchmark five models (DistilRoBERTa, DistilBERT, MiniLM, ELECTRA, TinyBERT) using stratified 5-fold cross-validation, reporting F1-score, AUC, and training time. DistilRoBERTa achieved the highest F1 (0.8292), TinyBERT the best efficiency, and MiniLM the best cross-platform consistency. Results show News data yields inflated performance due to class imbalance, Reddit suffers from summarization loss, and Twitter offers moderate challenge. Compressed models perform competitively, challenging assumptions that larger models are necessary. We identify platform-specific trade-offs and propose directions for scalable, real-world sentiment classification in urban planning discourse. 

---
# Continually Adding New Languages to Multilingual Language Models 

**Authors**: Abraham Toluwase Owodunni, Sachin Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2509.11414)  

**Abstract**: Multilingual language models are trained on a fixed set of languages, and to support new languages, the models need to be retrained from scratch. This is an expensive endeavor and is often infeasible, as model developers tend not to release their pre-training data. Naive approaches, such as continued pretraining, suffer from catastrophic forgetting; however, mitigation strategies like experience replay cannot be applied due to the lack of original pretraining data. In this work, we investigate the problem of continually adding new languages to a multilingual model, assuming access to pretraining data in only the target languages. We explore multiple approaches to address this problem and propose Layer-Selective LoRA (LayRA), which adds Low-Rank Adapters (LoRA) to selected initial and final layers while keeping the rest of the model frozen. LayRA builds on two insights: (1) LoRA reduces forgetting, and (2) multilingual models encode inputs in the source language in the initial layers, reason in English in intermediate layers, and translate back to the source language in final layers. We experiment with adding multiple combinations of Galician, Swahili, and Urdu to pretrained language models and evaluate each method on diverse multilingual tasks. We find that LayRA provides the overall best tradeoff between preserving models' capabilities in previously supported languages, while being competitive with existing approaches such as LoRA in learning new languages. We also demonstrate that using model arithmetic, the adapted models can be equipped with strong instruction following abilities without access to any instruction tuning data in the target languages. 

---
# Transformer Enhanced Relation Classification: A Comparative Analysis of Contextuality, Data Efficiency and Sequence Complexity 

**Authors**: Bowen Jing, Yang Cui, Tianpeng Huang  

**Link**: [PDF](https://arxiv.org/pdf/2509.11374)  

**Abstract**: In the era of large language model, relation extraction (RE) plays an important role in information extraction through the transformation of unstructured raw text into structured data (Wadhwa et al., 2023). In this paper, we systematically compare the performance of deep supervised learning approaches without transformers and those with transformers. We used a series of non-transformer architectures such as PA-LSTM(Zhang et al., 2017), C-GCN(Zhang et al., 2018), and AGGCN(attention guide GCN)(Guo et al., 2019), and a series of transformer architectures such as BERT, RoBERTa, and R-BERT(Wu and He, 2019). Our comparison included traditional metrics like micro F1, as well as evaluations in different scenarios, varying sentence lengths, and different percentages of the dataset for training. Our experiments were conducted on TACRED, TACREV, and RE-TACRED. The results show that transformer-based models outperform non-transformer models, achieving micro F1 scores of 80-90% compared to 64-67% for non-transformer models. Additionally, we briefly review the research journey in supervised relation classification and discuss the role and current status of large language models (LLMs) in relation extraction. 

---
# !MSA at AraHealthQA 2025 Shared Task: Enhancing LLM Performance for Arabic Clinical Question Answering through Prompt Engineering and Ensemble Learning 

**Authors**: Mohamed Tarek, Seif Ahmed, Mohamed Basem  

**Link**: [PDF](https://arxiv.org/pdf/2509.11365)  

**Abstract**: We present our systems for Track 2 (General Arabic Health QA, MedArabiQ) of the AraHealthQA-2025 shared task, where our methodology secured 2nd place in both Sub-Task 1 (multiple-choice question answering) and Sub-Task 2 (open-ended question answering) in Arabic clinical contexts. For Sub-Task 1, we leverage the Gemini 2.5 Flash model with few-shot prompting, dataset preprocessing, and an ensemble of three prompt configurations to improve classification accuracy on standard, biased, and fill-in-the-blank questions. For Sub-Task 2, we employ a unified prompt with the same model, incorporating role-playing as an Arabic medical expert, few-shot examples, and post-processing to generate concise responses across fill-in-the-blank, patient-doctor Q&A, GEC, and paraphrased variants. 

---
# Ko-PIQA: A Korean Physical Commonsense Reasoning Dataset with Cultural Context 

**Authors**: Dasol Choi, Jungwhan Kim, Guijin Son  

**Link**: [PDF](https://arxiv.org/pdf/2509.11303)  

**Abstract**: Physical commonsense reasoning datasets like PIQA are predominantly English-centric and lack cultural diversity. We introduce Ko-PIQA, a Korean physical commonsense reasoning dataset that incorporates cultural context. Starting from 3.01 million web-crawled questions, we employed a multi-stage filtering approach using three language models to identify 11,553 PIQA-style questions. Through GPT-4o refinement and human validation, we obtained 441 high-quality question-answer pairs. A key feature of Ko-PIQA is its cultural grounding: 19.7\% of questions contain culturally specific elements like traditional Korean foods (kimchi), clothing (hanbok), and specialized appliances (kimchi refrigerators) that require culturally-aware reasoning beyond direct translation. We evaluate seven language models on Ko-PIQA, with the best model achieving 83.22\% accuracy while the weakest reaches only 59.86\%, demonstrating significant room for improvement. Models particularly struggle with culturally specific scenarios, highlighting the importance of culturally diverse datasets. Ko-PIQA serves as both a benchmark for Korean language models and a foundation for more inclusive commonsense reasoning research. The dataset and code will be publicly available. 

---
# The Prompt Engineering Report Distilled: Quick Start Guide for Life Sciences 

**Authors**: Valentin Romanov, Steven A Niederer  

**Link**: [PDF](https://arxiv.org/pdf/2509.11295)  

**Abstract**: Developing effective prompts demands significant cognitive investment to generate reliable, high-quality responses from Large Language Models (LLMs). By deploying case-specific prompt engineering techniques that streamline frequently performed life sciences workflows, researchers could achieve substantial efficiency gains that far exceed the initial time investment required to master these techniques. The Prompt Report published in 2025 outlined 58 different text-based prompt engineering techniques, highlighting the numerous ways prompts could be constructed. To provide actionable guidelines and reduce the friction of navigating these various approaches, we distil this report to focus on 6 core techniques: zero-shot, few-shot approaches, thought generation, ensembling, self-criticism, and decomposition. We breakdown the significance of each approach and ground it in use cases relevant to life sciences, from literature summarization and data extraction to editorial tasks. We provide detailed recommendations for how prompts should and shouldn't be structured, addressing common pitfalls including multi-turn conversation degradation, hallucinations, and distinctions between reasoning and non-reasoning models. We examine context window limitations, agentic tools like Claude Code, while analyzing the effectiveness of Deep Research tools across OpenAI, Google, Anthropic and Perplexity platforms, discussing current limitations. We demonstrate how prompt engineering can augment rather than replace existing established individual practices around data processing and document editing. Our aim is to provide actionable guidance on core prompt engineering principles, and to facilitate the transition from opportunistic prompting to an effective, low-friction systematic practice that contributes to higher quality research. 

---
# RanAT4BIE: Random Adversarial Training for Biomedical Information Extraction 

**Authors**: Jian Chen, Shengyi Lv, Leilei Su  

**Link**: [PDF](https://arxiv.org/pdf/2509.11191)  

**Abstract**: We introduce random adversarial training (RAT), a novel framework successfully applied to biomedical information extraction (BioIE) tasks. Building on PubMedBERT as the foundational architecture, our study first validates the effectiveness of conventional adversarial training in enhancing pre-trained language models' performance on BioIE tasks. While adversarial training yields significant improvements across various performance metrics, it also introduces considerable computational overhead. To address this limitation, we propose RAT as an efficiency solution for biomedical information extraction. This framework strategically integrates random sampling mechanisms with adversarial training principles, achieving dual objectives: enhanced model generalization and robustness while significantly reducing computational costs. Through comprehensive evaluations, RAT demonstrates superior performance compared to baseline models in BioIE tasks. The results highlight RAT's potential as a transformative framework for biomedical natural language processing, offering a balanced solution to the model performance and computational efficiency. 

---
# Optimal Brain Restoration for Joint Quantization and Sparsification of LLMs 

**Authors**: Hang Guo, Yawei Li, Luca Benini  

**Link**: [PDF](https://arxiv.org/pdf/2509.11177)  

**Abstract**: Recent advances in Large Language Model (LLM) compression, such as quantization and pruning, have achieved notable success. However, as these techniques gradually approach their respective limits, relying on a single method for further compression has become increasingly challenging. In this work, we explore an alternative solution by combining quantization and sparsity. This joint approach, though promising, introduces new difficulties due to the inherently conflicting requirements on weight distributions: quantization favors compact ranges, while pruning benefits from high variance. To attack this problem, we propose Optimal Brain Restoration (OBR), a general and training-free framework that aligns pruning and quantization by error compensation between both. OBR minimizes performance degradation on downstream tasks by building on a second-order Hessian objective, which is then reformulated into a tractable problem through surrogate approximation and ultimately reaches a closed-form solution via group error compensation. Experiments show that OBR enables aggressive W4A4KV4 quantization with 50% sparsity on existing LLMs, and delivers up to 4.72x speedup and 6.4x memory reduction compared to the FP16-dense baseline. 

---
# Differentially-private text generation degrades output language quality 

**Authors**: Erion Çano, Ivan Habernal  

**Link**: [PDF](https://arxiv.org/pdf/2509.11176)  

**Abstract**: Ensuring user privacy by synthesizing data from large language models (LLMs) tuned under differential privacy (DP) has become popular recently. However, the impact of DP fine-tuned LLMs on the quality of the language and the utility of the texts they produce has not been investigated. In this work, we tune five LLMs with three corpora under four levels of privacy and assess the length, the grammatical correctness, and the lexical diversity of the text outputs they produce. We also probe the utility of the synthetic outputs in downstream classification tasks such as book genre recognition based on book descriptions and cause of death recognition based on verbal autopsies. The results indicate that LLMs tuned under stronger privacy constrains produce texts that are shorter by at least 77 %, that are less grammatically correct by at least 9 %, and are less diverse by at least 10 % in bi-gram diversity. Furthermore, the accuracy they reach in downstream classification tasks decreases, which might be detrimental to the usefulness of the generated synthetic data. 

---
# Text2Mem: A Unified Memory Operation Language for Memory Operating System 

**Authors**: Felix Wang, Boyu Chen, Kerun Xu, Bo Tang, Feiyu Xiong, Zhiyu Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.11145)  

**Abstract**: Large language model agents increasingly depend on memory to sustain long horizon interaction, but existing frameworks remain limited. Most expose only a few basic primitives such as encode, retrieve, and delete, while higher order operations like merge, promote, demote, split, lock, and expire are missing or inconsistently supported. Moreover, there is no formal and executable specification for memory commands, leaving scope and lifecycle rules implicit and causing unpredictable behavior across systems. We introduce Text2Mem, a unified memory operation language that provides a standardized pathway from natural language to reliable execution. Text2Mem defines a compact yet expressive operation set aligned with encoding, storage, and retrieval. Each instruction is represented as a JSON based schema instance with required fields and semantic invariants, which a parser transforms into typed operation objects with normalized parameters. A validator ensures correctness before execution, while adapters map typed objects either to a SQL prototype backend or to real memory frameworks. Model based services such as embeddings or summarization are integrated when required. All results are returned through a unified execution contract. This design ensures safety, determinism, and portability across heterogeneous backends. We also outline Text2Mem Bench, a planned benchmark that separates schema generation from backend execution to enable systematic evaluation. Together, these components establish the first standardized foundation for memory control in agents. 

---
# When Smiley Turns Hostile: Interpreting How Emojis Trigger LLMs' Toxicity 

**Authors**: Shiyao Cui, Xijia Feng, Yingkang Wang, Junxiao Yang, Zhexin Zhang, Biplab Sikdar, Hongning Wang, Han Qiu, Minlie Huang  

**Link**: [PDF](https://arxiv.org/pdf/2509.11141)  

**Abstract**: Emojis are globally used non-verbal cues in digital communication, and extensive research has examined how large language models (LLMs) understand and utilize emojis across contexts. While usually associated with friendliness or playfulness, it is observed that emojis may trigger toxic content generation in LLMs. Motivated by such a observation, we aim to investigate: (1) whether emojis can clearly enhance the toxicity generation in LLMs and (2) how to interpret this phenomenon. We begin with a comprehensive exploration of emoji-triggered LLM toxicity generation by automating the construction of prompts with emojis to subtly express toxic intent. Experiments across 5 mainstream languages on 7 famous LLMs along with jailbreak tasks demonstrate that prompts with emojis could easily induce toxicity generation. To understand this phenomenon, we conduct model-level interpretations spanning semantic cognition, sequence generation and tokenization, suggesting that emojis can act as a heterogeneous semantic channel to bypass the safety mechanisms. To pursue deeper insights, we further probe the pre-training corpus and uncover potential correlation between the emoji-related data polution with the toxicity generation behaviors. Supplementary materials provide our implementation code and data. (Warning: This paper contains potentially sensitive contents) 

---
# Joint Effects of Argumentation Theory, Audio Modality and Data Enrichment on LLM-Based Fallacy Classification 

**Authors**: Hongxu Zhou, Hylke Westerdijk, Khondoker Ittehadul Islam  

**Link**: [PDF](https://arxiv.org/pdf/2509.11127)  

**Abstract**: This study investigates how context and emotional tone metadata influence large language model (LLM) reasoning and performance in fallacy classification tasks, particularly within political debate settings. Using data from U.S. presidential debates, we classify six fallacy types through various prompting strategies applied to the Qwen-3 (8B) model. We introduce two theoretically grounded Chain-of-Thought frameworks: Pragma-Dialectics and the Periodic Table of Arguments, and evaluate their effectiveness against a baseline prompt under three input settings: text-only, text with context, and text with both context and audio-based emotional tone metadata. Results suggest that while theoretical prompting can improve interpretability and, in some cases, accuracy, the addition of context and especially emotional tone metadata often leads to lowered performance. Emotional tone metadata biases the model toward labeling statements as \textit{Appeal to Emotion}, worsening logical reasoning. Overall, basic prompts often outperformed enhanced ones, suggesting that attention dilution from added inputs may worsen rather than improve fallacy classification in LLMs. 

---
# We Argue to Agree: Towards Personality-Driven Argumentation-Based Negotiation Dialogue Systems for Tourism 

**Authors**: Priyanshu Priya, Saurav Dudhate, Desai Vishesh Yasheshbhai, Asif Ekbal  

**Link**: [PDF](https://arxiv.org/pdf/2509.11118)  

**Abstract**: Integrating argumentation mechanisms into negotiation dialogue systems improves conflict resolution through exchanges of arguments and critiques. Moreover, incorporating personality attributes enhances adaptability by aligning interactions with individuals' preferences and styles. To advance these capabilities in negotiation dialogue systems, we propose a novel Personality-driven Argumentation-based Negotiation Dialogue Generation (PAN-DG) task. To support this task, we introduce PACT, a dataset of Personality-driven Argumentation-based negotiation Conversations for Tourism sector. This dataset, generated using Large Language Models (LLMs), features three distinct personality profiles, viz. Argumentation Profile, Preference Profile, and Buying Style Profile to simulate a variety of negotiation scenarios involving diverse personalities. Thorough automatic and manual evaluations indicate that the dataset comprises high-quality dialogues. Further, we conduct comparative experiments between pre-trained and fine-tuned LLMs for the PAN-DG task. Multi-dimensional evaluation demonstrates that the fine-tuned LLMs effectively generate personality-driven rational responses during negotiations. This underscores the effectiveness of PACT in enhancing personalization and reasoning capabilities in negotiation dialogue systems, thereby establishing a foundation for future research in this domain. 

---
# Fluid Language Model Benchmarking 

**Authors**: Valentin Hofmann, David Heineman, Ian Magnusson, Kyle Lo, Jesse Dodge, Maarten Sap, Pang Wei Koh, Chun Wang, Hannaneh Hajishirzi, Noah A. Smith  

**Link**: [PDF](https://arxiv.org/pdf/2509.11106)  

**Abstract**: Language model (LM) benchmarking faces several challenges: comprehensive evaluations are costly, benchmarks often fail to measure the intended capabilities, and evaluation quality can degrade due to labeling errors and benchmark saturation. Although various strategies have been proposed to mitigate these issues, they tend to address individual aspects in isolation, neglecting broader questions about overall evaluation quality. Here, we introduce Fluid Benchmarking, a new evaluation approach that advances LM benchmarking across multiple dimensions. Inspired by psychometrics, Fluid Benchmarking is based on the insight that the relative value of benchmark items depends on an LM's capability level, suggesting that evaluation should adapt to each LM. Methodologically, Fluid Benchmarking estimates an item response model based on existing LM evaluation results and uses the inferred quantities to select evaluation items dynamically, similar to computerized adaptive testing in education. In our experiments, we compare Fluid Benchmarking against the common practice of random item sampling as well as more sophisticated baselines, including alternative methods grounded in item response theory. We examine four dimensions -- efficiency, validity, variance, and saturation -- and find that Fluid Benchmarking achieves superior performance in all of them (e.g., higher validity and less variance on MMLU with fifty times fewer items). Our analysis shows that the two components of Fluid Benchmarking have distinct effects: item response theory, used to map performance into a latent ability space, increases validity, while dynamic item selection reduces variance. Overall, our results suggest that LM benchmarking can be substantially improved by moving beyond static evaluation. 

---
# EmoBench-Reddit: A Hierarchical Benchmark for Evaluating the Emotional Intelligence of Multimodal Large Language Models 

**Authors**: Haokun Li, Yazhou Zhang, Jizhi Ding, Qiuchi Li, Peng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.11101)  

**Abstract**: With the rapid advancement of Multimodal Large Language Models (MLLMs), they have demonstrated exceptional capabilities across a variety of vision-language tasks. However, current evaluation benchmarks predominantly focus on objective visual question answering or captioning, inadequately assessing the models' ability to understand complex and subjective human emotions. To bridge this gap, we introduce EmoBench-Reddit, a novel, hierarchical benchmark for multimodal emotion understanding. The dataset comprises 350 meticulously curated samples from the social media platform Reddit, each containing an image, associated user-provided text, and an emotion category (sad, humor, sarcasm, happy) confirmed by user flairs. We designed a hierarchical task framework that progresses from basic perception to advanced cognition, with each data point featuring six multiple-choice questions and one open-ended question of increasing difficulty. Perception tasks evaluate the model's ability to identify basic visual elements (e.g., colors, objects), while cognition tasks require scene reasoning, intent understanding, and deep empathy integrating textual context. We ensured annotation quality through a combination of AI assistance (Claude 4) and manual verification. 

---
# An Interpretable Benchmark for Clickbait Detection and Tactic Attribution 

**Authors**: Lihi Nofar, Tomer Portal, Aviv Elbaz, Alexander Apartsin, Yehudit Aperstein  

**Link**: [PDF](https://arxiv.org/pdf/2509.10937)  

**Abstract**: The proliferation of clickbait headlines poses significant challenges to the credibility of information and user trust in digital media. While recent advances in machine learning have improved the detection of manipulative content, the lack of explainability limits their practical adoption. This paper presents a model for explainable clickbait detection that not only identifies clickbait titles but also attributes them to specific linguistic manipulation strategies. We introduce a synthetic dataset generated by systematically augmenting real news headlines using a predefined catalogue of clickbait strategies. This dataset enables controlled experimentation and detailed analysis of model behaviour. We present a two-stage framework for automatic clickbait analysis comprising detection and tactic attribution. In the first stage, we compare a fine-tuned BERT classifier with large language models (LLMs), specifically GPT-4.0 and Gemini 2.4 Flash, under both zero-shot prompting and few-shot prompting enriched with illustrative clickbait headlines and their associated persuasive tactics. In the second stage, a dedicated BERT-based classifier predicts the specific clickbait strategies present in each headline. This work advances the development of transparent and trustworthy AI systems for combating manipulative media content. We share the dataset with the research community at this https URL 

---
# Introducing Spotlight: A Novel Approach for Generating Captivating Key Information from Documents 

**Authors**: Ankan Mullick, Sombit Bose, Rounak Saha, Ayan Kumar Bhowmick, Aditya Vempaty, Prasenjit Dey, Ravi Kokku, Pawan Goyal, Niloy Ganguly  

**Link**: [PDF](https://arxiv.org/pdf/2509.10935)  

**Abstract**: In this paper, we introduce Spotlight, a novel paradigm for information extraction that produces concise, engaging narratives by highlighting the most compelling aspects of a document. Unlike traditional summaries, which prioritize comprehensive coverage, spotlights selectively emphasize intriguing content to foster deeper reader engagement with the source material. We formally differentiate spotlights from related constructs and support our analysis with a detailed benchmarking study using new datasets curated for this work. To generate high-quality spotlights, we propose a two-stage approach: fine-tuning a large language model on our benchmark data, followed by alignment via Direct Preference Optimization (DPO). Our comprehensive evaluation demonstrates that the resulting model not only identifies key elements with precision but also enhances readability and boosts the engagement value of the original document. 

---
# Aligning ESG Controversy Data with International Guidelines through Semi-Automatic Ontology Construction 

**Authors**: Tsuyoshi Iwata, Guillaume Comte, Melissa Flores, Ryoma Kondo, Ryohei Hisano  

**Link**: [PDF](https://arxiv.org/pdf/2509.10922)  

**Abstract**: The growing importance of environmental, social, and governance data in regulatory and investment contexts has increased the need for accurate, interpretable, and internationally aligned representations of non-financial risks, particularly those reported in unstructured news sources. However, aligning such controversy-related data with principle-based normative frameworks, such as the United Nations Global Compact or Sustainable Development Goals, presents significant challenges. These frameworks are typically expressed in abstract language, lack standardized taxonomies, and differ from the proprietary classification systems used by commercial data providers. In this paper, we present a semi-automatic method for constructing structured knowledge representations of environmental, social, and governance events reported in the news. Our approach uses lightweight ontology design, formal pattern modeling, and large language models to convert normative principles into reusable templates expressed in the Resource Description Framework. These templates are used to extract relevant information from news content and populate a structured knowledge graph that links reported incidents to specific framework principles. The result is a scalable and transparent framework for identifying and interpreting non-compliance with international sustainability guidelines. 

---
# CultureSynth: A Hierarchical Taxonomy-Guided and Retrieval-Augmented Framework for Cultural Question-Answer Synthesis 

**Authors**: Xinyu Zhang, Pei Zhang, Shuang Luo, Jialong Tang, Yu Wan, Baosong Yang, Fei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2509.10886)  

**Abstract**: Cultural competence, defined as the ability to understand and adapt to multicultural contexts, is increasingly vital for large language models (LLMs) in global environments. While several cultural benchmarks exist to assess LLMs' cultural competence, current evaluations suffer from fragmented taxonomies, domain specificity, and heavy reliance on manual data annotation. To address these limitations, we introduce CultureSynth, a novel framework comprising (1) a comprehensive hierarchical multilingual cultural taxonomy covering 12 primary and 130 secondary topics, and (2) a Retrieval-Augmented Generation (RAG)-based methodology leveraging factual knowledge to synthesize culturally relevant question-answer pairs. The CultureSynth-7 synthetic benchmark contains 19,360 entries and 4,149 manually verified entries across 7 languages. Evaluation of 14 prevalent LLMs of different sizes reveals clear performance stratification led by ChatGPT-4o-Latest and Qwen2.5-72B-Instruct. The results demonstrate that a 3B-parameter threshold is necessary for achieving basic cultural competence, models display varying architectural biases in knowledge processing, and significant geographic disparities exist across models. We believe that CultureSynth offers a scalable framework for developing culturally aware AI systems while reducing reliance on manual annotation\footnote{Benchmark is available at this https URL.}. 

---
# Term2Note: Synthesising Differentially Private Clinical Notes from Medical Terms 

**Authors**: Yuping Wu, Viktor Schlegel, Warren Del-Pinto, Srinivasan Nandakumar, Iqra Zahid, Yidan Sun, Usama Farghaly Omar, Amirah Jasmine, Arun-Kumar Kaliya-Perumal, Chun Shen Tham, Gabriel Connors, Anil A Bharath, Goran Nenadic  

**Link**: [PDF](https://arxiv.org/pdf/2509.10882)  

**Abstract**: Training data is fundamental to the success of modern machine learning models, yet in high-stakes domains such as healthcare, the use of real-world training data is severely constrained by concerns over privacy leakage. A promising solution to this challenge is the use of differentially private (DP) synthetic data, which offers formal privacy guarantees while maintaining data utility. However, striking the right balance between privacy protection and utility remains challenging in clinical note synthesis, given its domain specificity and the complexity of long-form text generation. In this paper, we present Term2Note, a methodology to synthesise long clinical notes under strong DP constraints. By structurally separating content and form, Term2Note generates section-wise note content conditioned on DP medical terms, with each governed by separate DP constraints. A DP quality maximiser further enhances synthetic notes by selecting high-quality outputs. Experimental results show that Term2Note produces synthetic notes with statistical properties closely aligned with real clinical notes, demonstrating strong fidelity. In addition, multi-label classification models trained on these synthetic notes perform comparably to those trained on real data, confirming their high utility. Compared to existing DP text generation baselines, Term2Note achieves substantial improvements in both fidelity and utility while operating under fewer assumptions, suggesting its potential as a viable privacy-preserving alternative to using sensitive clinical notes. 

---
# Quantifier Scope Interpretation in Language Learners and LLMs 

**Authors**: Shaohua Fang, Yue Li, Yan Cong  

**Link**: [PDF](https://arxiv.org/pdf/2509.10860)  

**Abstract**: Sentences with multiple quantifiers often lead to interpretive ambiguities, which can vary across languages. This study adopts a cross-linguistic approach to examine how large language models (LLMs) handle quantifier scope interpretation in English and Chinese, using probabilities to assess interpretive likelihood. Human similarity (HS) scores were used to quantify the extent to which LLMs emulate human performance across language groups. Results reveal that most LLMs prefer the surface scope interpretations, aligning with human tendencies, while only some differentiate between English and Chinese in the inverse scope preferences, reflecting human-similar patterns. HS scores highlight variability in LLMs' approximation of human behavior, but their overall potential to align with humans is notable. Differences in model architecture, scale, and particularly models' pre-training data language background, significantly influence how closely LLMs approximate human quantifier scope interpretations. 

---
# Pre-Storage Reasoning for Episodic Memory: Shifting Inference Burden to Memory for Personalized Dialogue 

**Authors**: Sangyeop Kim, Yohan Lee, Sanghwa Kim, Hyunjong Kim, Sungzoon Cho  

**Link**: [PDF](https://arxiv.org/pdf/2509.10852)  

**Abstract**: Effective long-term memory in conversational AI requires synthesizing information across multiple sessions. However, current systems place excessive reasoning burden on response generation, making performance significantly dependent on model sizes. We introduce PREMem (Pre-storage Reasoning for Episodic Memory), a novel approach that shifts complex reasoning processes from inference to memory construction. PREMem extracts fine-grained memory fragments categorized into factual, experiential, and subjective information; it then establishes explicit relationships between memory items across sessions, capturing evolution patterns like extensions, transformations, and implications. By performing this reasoning during pre-storage rather than when generating a response, PREMem creates enriched representations while reducing computational demands during interactions. Experiments show significant performance improvements across all model sizes, with smaller models achieving results comparable to much larger baselines while maintaining effectiveness even with constrained token budgets. Code and dataset are available at this https URL. 

---
# A funny companion: Distinct neural responses to perceived AI- versus humangenerated humor 

**Authors**: Xiaohui Rao, Hanlin Wu, Zhenguang G. Cai  

**Link**: [PDF](https://arxiv.org/pdf/2509.10847)  

**Abstract**: As AI companions become capable of human-like communication, including telling jokes, understanding how people cognitively and emotionally respond to AI humor becomes increasingly important. This study used electroencephalography (EEG) to compare how people process humor from AI versus human sources. Behavioral analysis revealed that participants rated AI and human humor as comparably funny. However, neurophysiological data showed that AI humor elicited a smaller N400 effect, suggesting reduced cognitive effort during the processing of incongruity. This was accompanied by a larger Late Positive Potential (LPP), indicating a greater degree of surprise and emotional response. This enhanced LPP likely stems from the violation of low initial expectations regarding AI's comedic capabilities. Furthermore, a key temporal dynamic emerged: human humor showed habituation effects, marked by an increasing N400 and a decreasing LPP over time. In contrast, AI humor demonstrated increasing processing efficiency and emotional reward, with a decreasing N400 and an increasing LPP. This trajectory reveals how the brain can dynamically update its predictive model of AI capabilities. This process of cumulative reinforcement challenges "algorithm aversion" in humor, as it demonstrates how cognitive adaptation to AI's language patterns can lead to an intensified emotional reward. Additionally, participants' social attitudes toward AI modulated these neural responses, with higher perceived AI trustworthiness correlating with enhanced emotional engagement. These findings indicate that the brain responds to AI humor with surprisingly positive and intense reactions, highlighting humor's potential for fostering genuine engagement in human-AI social interaction. 

---
# Text2Sign Diffusion: A Generative Approach for Gloss-Free Sign Language Production 

**Authors**: Liqian Feng, Lintao Wang, Kun Hu, Dehui Kong, Zhiyong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.10845)  

**Abstract**: Sign language production (SLP) aims to translate spoken language sentences into a sequence of pose frames in a sign language, bridging the communication gap and promoting digital inclusion for deaf and hard-of-hearing communities. Existing methods typically rely on gloss, a symbolic representation of sign language words or phrases that serves as an intermediate step in SLP. This limits the flexibility and generalization of SLP, as gloss annotations are often unavailable and language-specific. Therefore, we present a novel diffusion-based generative approach - Text2Sign Diffusion (Text2SignDiff) for gloss-free SLP. Specifically, a gloss-free latent diffusion model is proposed to generate sign language sequences from noisy latent sign codes and spoken text jointly, reducing the potential error accumulation through a non-autoregressive iterative denoising process. We also design a cross-modal signing aligner that learns a shared latent space to bridge visual and textual content in sign and spoken languages. This alignment supports the conditioned diffusion-based process, enabling more accurate and contextually relevant sign language generation without gloss. Extensive experiments on the commonly used PHOENIX14T and How2Sign datasets demonstrate the effectiveness of our method, achieving the state-of-the-art performance. 

---
# GAPrune: Gradient-Alignment Pruning for Domain-Aware Embeddings 

**Authors**: Yixuan Tang, Yi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2509.10844)  

**Abstract**: Domain-specific embedding models have shown promise for applications that require specialized semantic understanding, such as coding agents and financial retrieval systems, often achieving higher performance gains than general models. However, state-of-the-art embedding models are typically based on LLMs, which contain billions of parameters, making deployment challenging in resource-constrained environments. Model compression through pruning offers a promising solution, but existing pruning methods treat all parameters uniformly, failing to distinguish between general semantic representations and domain-specific patterns, leading to suboptimal pruning decisions. Thus, we propose GAPrune, a pruning framework that addresses this challenge by considering both domain importance and preserving general linguistic foundation. Our method uses Fisher Information to measure importance and general-domain gradient alignment to assess parameter behavior, then combines these signals using our Domain Alignment Importance (DAI) scoring. Lower DAI scores indicate that the parameter is either less important for the domain task or creates conflicts between domain and general objectives. Experiments on two domain benchmarks, FinMTEB and ChemTEB, show that GAPrune maintains performance within 2.5% of dense models in one-shot pruning at 50% sparsity, while outperforming all baselines. With retraining in 100 steps, GAPrune achieves +4.51% improvement on FinMTEB and +1.73% on ChemTEB, demonstrating that our pruning strategy not only preserves but enhances domain-specific capabilities. Our findings demonstrate that principled pruning strategies can achieve model compression and enhanced domain specialization, providing the research community with a new approach for development. 

---
# Evaluating Large Language Models for Evidence-Based Clinical Question Answering 

**Authors**: Can Wang, Yiqun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.10843)  

**Abstract**: Large Language Models (LLMs) have demonstrated substantial progress in biomedical and clinical applications, motivating rigorous evaluation of their ability to answer nuanced, evidence-based questions. We curate a multi-source benchmark drawing from Cochrane systematic reviews and clinical guidelines, including structured recommendations from the American Heart Association and narrative guidance used by insurers. Using GPT-4o-mini and GPT-5, we observe consistent performance patterns across sources and clinical domains: accuracy is highest on structured guideline recommendations (90%) and lower on narrative guideline and systematic review questions (60--70%). We also find a strong correlation between accuracy and the citation count of the underlying systematic reviews, where each doubling of citations is associated with roughly a 30% increase in the odds of a correct answer. Models show moderate ability to reason about evidence quality when contextual information is supplied. When we incorporate retrieval-augmented prompting, providing the gold-source abstract raises accuracy on previously incorrect items to 0.79; providing top 3 PubMed abstracts (ranked by semantic relevance) improves accuracy to 0.23, while random abstracts reduce accuracy (0.10, within temperature variation). These effects are mirrored in GPT-4o-mini, underscoring that source clarity and targeted retrieval -- not just model size -- drive performance. Overall, our results highlight both the promise and current limitations of LLMs for evidence-based clinical question answering. Retrieval-augmented prompting emerges as a useful strategy to improve factual accuracy and alignment with source evidence, while stratified evaluation by specialty and question type remains essential to understand current knowledge access and to contextualize model performance. 

---
# Towards Automated Error Discovery: A Study in Conversational AI 

**Authors**: Dominic Petrak, Thy Thy Tran, Iryna Gurevych  

**Link**: [PDF](https://arxiv.org/pdf/2509.10833)  

**Abstract**: Although LLM-based conversational agents demonstrate strong fluency and coherence, they still produce undesirable behaviors (errors) that are challenging to prevent from reaching users during deployment. Recent research leverages large language models (LLMs) to detect errors and guide response-generation models toward improvement. However, current LLMs struggle to identify errors not explicitly specified in their instructions, such as those arising from updates to the response-generation model or shifts in user behavior. In this work, we introduce Automated Error Discovery, a framework for detecting and defining errors in conversational AI, and propose SEEED (Soft Clustering Extended Encoder-Based Error Detection), as an encoder-based approach to its implementation. We enhance the Soft Nearest Neighbor Loss by amplifying distance weighting for negative samples and introduce Label-Based Sample Ranking to select highly contrastive examples for better representation learning. SEEED outperforms adapted baselines -- including GPT-4o and Phi-4 -- across multiple error-annotated dialogue datasets, improving the accuracy for detecting unknown errors by up to 8 points and demonstrating strong generalization to unknown intent detection. 

---
# Judge Q: Trainable Queries for Optimized Information Retention in KV Cache Eviction 

**Authors**: Yijun Liu, Yixuan Wang, Yuzhuang Xu, Shiyu Ji, Yang Xu, Qingfu Zhu, Wanxiang Che  

**Link**: [PDF](https://arxiv.org/pdf/2509.10798)  

**Abstract**: Large language models (LLMs) utilize key-value (KV) cache to store historical information during sequence processing. The size of KV cache grows linearly as the length of the sequence extends, which seriously affects memory usage and decoding efficiency. Current methods for KV cache eviction typically utilize the last window from the pre-filling phase as queries to compute the KV importance scores for eviction. Although this scheme is simple to implement, it tends to overly focus on local information, potentially leading to the neglect or omission of crucial global information. To mitigate this issue, we propose Judge Q, a novel training method which incorporates a soft token list. This method only tunes the model's embedding layer at a low training cost. By concatenating the soft token list at the end of the input sequence, we train these tokens' attention map to the original input sequence to align with that of the actual decoded tokens. In this way, the queries corresponding to the soft tokens can effectively capture global information and better evaluate the importance of the keys and values within the KV cache, thus maintaining decoding quality when KV cache is evicted. Under the same eviction budget, our method exhibits less performance degradation compared to existing eviction approaches. We validate our approach through experiments conducted on models such as Llama-3.1-8B-Instruct and Mistral-7B-Instruct-v0.3, using benchmarks including LongBench, RULER, and Needle-in-a-Haystack. Results indicate an improvement of approximately 1 point on the LongBench and over 3 points on RULER. This proposed methodology can be seamlessly integrated into existing open-source models with minimal training overhead, thereby enhancing performance in KV cache eviction scenarios. 

---
# RECAP: Transparent Inference-Time Emotion Alignment for Medical Dialogue Systems 

**Authors**: Adarsh Srinivasan, Jacob Dineen, Muhammad Umar Afzal, Muhammad Uzair Sarfraz, Irbaz B. Riaz, Ben Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2509.10746)  

**Abstract**: Large language models in healthcare often miss critical emotional cues, delivering medically sound but emotionally flat advice. This is especially problematic in clinical contexts where patients are distressed and vulnerable, and require empathic communication to support safety, adherence, and trust. We present RECAP (Reflect-Extract-Calibrate-Align-Produce), an inference-time framework that adds structured emotional reasoning without retraining. By decomposing empathy into transparent appraisal-theoretic stages and exposing per-dimension Likert signals, RECAP produces nuanced, auditable responses. Across EmoBench, SECEU, and EQ-Bench, RECAP improves emotional reasoning by 22-28% on 8B models and 10-13% on larger models over zero-shot baselines. Clinician evaluations further confirm superior empathetic communication. RECAP shows that modular, theory-grounded prompting can systematically enhance emotional intelligence in medical AI while preserving the accountability required for deployment. 

---
# Automated MCQA Benchmarking at Scale: Evaluating Reasoning Traces as Retrieval Sources for Domain Adaptation of Small Language Models 

**Authors**: Ozan Gokdemir, Neil Getty, Robert Underwood, Sandeep Madireddy, Franck Cappello, Arvind Ramanathan, Ian T. Foster, Rick L. Stevens  

**Link**: [PDF](https://arxiv.org/pdf/2509.10744)  

**Abstract**: As scientific knowledge grows at an unprecedented pace, evaluation benchmarks must evolve to reflect new discoveries and ensure language models are tested on current, diverse literature. We propose a scalable, modular framework for generating multiple-choice question-answering (MCQA) benchmarks directly from large corpora of scientific papers. Our pipeline automates every stage of MCQA creation, including PDF parsing, semantic chunking, question generation, and model evaluation. As a case study, we generate more than 16,000 MCQs from 22,000 open-access articles in radiation and cancer biology. We then evaluate a suite of small language models (1.1B-14B parameters) on these questions, comparing baseline accuracy with retrieval-augmented generation (RAG) from paper-derived semantic chunks and from reasoning traces distilled from GPT-4.1. We find that reasoning-trace retrieval consistently improves performance on both synthetic and expert-annotated benchmarks, enabling several small models to surpass GPT-4 on the 2023 Astro Radiation and Cancer Biology exam. 

---
# Reasoning Under Uncertainty: Exploring Probabilistic Reasoning Capabilities of LLMs 

**Authors**: Mobina Pournemat, Keivan Rezaei, Gaurang Sriramanan, Arman Zarei, Jiaxiang Fu, Yang Wang, Hamid Eghbalzadeh, Soheil Feizi  

**Link**: [PDF](https://arxiv.org/pdf/2509.10739)  

**Abstract**: Despite widespread success in language understanding and generation, large language models (LLMs) exhibit unclear and often inconsistent behavior when faced with tasks that require probabilistic reasoning. In this work, we present the first comprehensive study of the reasoning capabilities of LLMs over explicit discrete probability distributions. Given observations from a probability distribution, we evaluate models on three carefully designed tasks, mode identification, maximum likelihood estimation, and sample generation, by prompting them to provide responses to queries about either the joint distribution or its conditionals. These tasks thus probe a range of probabilistic skills, including frequency analysis, marginalization, and generative behavior. Through comprehensive empirical evaluations, we demonstrate that there exists a clear performance gap between smaller and larger models, with the latter demonstrating stronger inference and surprising capabilities in sample generation. Furthermore, our investigations reveal notable limitations, including sensitivity to variations in the notation utilized to represent probabilistic outcomes and performance degradation of over 60% as context length increases. Together, our results provide a detailed understanding of the probabilistic reasoning abilities of LLMs and identify key directions for future improvement. 

---
# PolyTruth: Multilingual Disinformation Detection using Transformer-Based Language Models 

**Authors**: Zaur Gouliev, Jennifer Waters, Chengqian Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.10737)  

**Abstract**: Disinformation spreads rapidly across linguistic boundaries, yet most AI models are still benchmarked only on English. We address this gap with a systematic comparison of five multilingual transformer models: mBERT, XLM, XLM-RoBERTa, RemBERT, and mT5 on a common fake-vs-true machine learning classification task. While transformer-based language models have demonstrated notable success in detecting disinformation in English, their effectiveness in multilingual contexts still remains up for debate. To facilitate evaluation, we introduce PolyTruth Disinfo Corpus, a novel corpus of 60,486 statement pairs (false claim vs. factual correction) spanning over twenty five languages that collectively cover five language families and a broad topical range from politics, health, climate, finance, and conspiracy, half of which are fact-checked disinformation claims verified by an augmented MindBugs Discovery dataset. Our experiments revealed performance variations. Models such as RemBERT achieved better overall accuracy, particularly excelling in low-resource languages, whereas models like mBERT and XLM exhibit considerable limitations when training data is scarce. We provide a discussion of these performance patterns and implications for real-world deployment. The dataset is publicly available on our GitHub repository to encourage further experimentation and advancement. Our findings illuminate both the potential and the current limitations of AI systems for multilingual disinformation detection. 

---
# SearchInstruct: Enhancing Domain Adaptation via Retrieval-Based Instruction Dataset Creation 

**Authors**: Iman Barati, Mostafa Amiri, Heshaam Faili  

**Link**: [PDF](https://arxiv.org/pdf/2509.10708)  

**Abstract**: Supervised Fine-Tuning (SFT) is essential for training large language models (LLMs), significantly enhancing critical capabilities such as instruction following and in-context learning. Nevertheless, creating suitable training datasets tailored for specific domains remains challenging due to unique domain constraints and data scarcity. In this paper, we propose SearchInstruct, an innovative method explicitly designed to construct high quality instruction datasets for SFT. Our approach begins with a limited set of domain specific, human generated questions, which are systematically expanded using a large language model. Subsequently, domain relevant resources are dynamically retrieved to generate accurate and contextually appropriate answers for each augmented question. Experimental evaluation demonstrates that SearchInstruct enhances both the diversity and quality of SFT datasets, leading to measurable improvements in LLM performance within specialized domains. Additionally, we show that beyond dataset generation, the proposed method can also effectively facilitate tasks such as model editing, enabling efficient updates to existing models. To facilitate reproducibility and community adoption, we provide full implementation details, the complete set of generated instruction response pairs, and the source code in a publicly accessible Git repository: [this https URL](this https URL) 

---
# A Survey on Retrieval And Structuring Augmented Generation with Large Language Models 

**Authors**: Pengcheng Jiang, Siru Ouyang, Yizhu Jiao, Ming Zhong, Runchu Tian, Jiawei Han  

**Link**: [PDF](https://arxiv.org/pdf/2509.10697)  

**Abstract**: Large Language Models (LLMs) have revolutionized natural language processing with their remarkable capabilities in text generation and reasoning. However, these models face critical challenges when deployed in real-world applications, including hallucination generation, outdated knowledge, and limited domain expertise. Retrieval And Structuring (RAS) Augmented Generation addresses these limitations by integrating dynamic information retrieval with structured knowledge representations. This survey (1) examines retrieval mechanisms including sparse, dense, and hybrid approaches for accessing external knowledge; (2) explore text structuring techniques such as taxonomy construction, hierarchical classification, and information extraction that transform unstructured text into organized representations; and (3) investigate how these structured representations integrate with LLMs through prompt-based methods, reasoning frameworks, and knowledge embedding techniques. It also identifies technical challenges in retrieval efficiency, structure quality, and knowledge integration, while highlighting research opportunities in multimodal retrieval, cross-lingual structures, and interactive systems. This comprehensive overview provides researchers and practitioners with insights into RAS methods, applications, and future directions. 

---
# Struct-Bench: A Benchmark for Differentially Private Structured Text Generation 

**Authors**: Shuaiqi Wang, Vikas Raunak, Arturs Backurs, Victor Reis, Pei Zhou, Sihao Chen, Longqi Yang, Zinan Lin, Sergey Yekhanin, Giulia Fanti  

**Link**: [PDF](https://arxiv.org/pdf/2509.10696)  

**Abstract**: Differentially private (DP) synthetic data generation is a promising technique for utilizing private datasets that otherwise cannot be exposed for model training or other analytics. While much research literature has focused on generating private unstructured text and image data, in enterprise settings, structured data (e.g., tabular) is more common, often including natural language fields or components. Existing synthetic data evaluation techniques (e.g., FID) struggle to capture the structural properties and correlations of such datasets. In this work, we propose Struct-Bench, a framework and benchmark for evaluating synthetic datasets derived from structured datasets that contain natural language data. The Struct-Bench framework requires users to provide a representation of their dataset structure as a Context-Free Grammar (CFG). Our benchmark comprises 5 real-world and 2 synthetically generated datasets, each annotated with CFGs. We show that these datasets demonstrably present a great challenge even for state-of-the-art DP synthetic data generation methods. Struct-Bench also includes reference implementations of different metrics and a leaderboard, thereby providing researchers a standardized evaluation platform to benchmark and investigate privacy-preserving synthetic data generation methods. Further, we also present a case study showing how to use Struct-Bench to improve the synthetic data quality of Private Evolution (PE) on structured data. The benchmark and the leaderboard have been publicly made available at this https URL. 

---
# Pluralistic Alignment for Healthcare: A Role-Driven Framework 

**Authors**: Jiayou Zhong, Anudeex Shetty, Chao Jia, Xuanrui Lin, Usman Naseem  

**Link**: [PDF](https://arxiv.org/pdf/2509.10685)  

**Abstract**: As large language models are increasingly deployed in sensitive domains such as healthcare, ensuring their outputs reflect the diverse values and perspectives held across populations is critical. However, existing alignment approaches, including pluralistic paradigms like Modular Pluralism, often fall short in the health domain, where personal, cultural, and situational factors shape pluralism. Motivated by the aforementioned healthcare challenges, we propose a first lightweight, generalizable, pluralistic alignment approach, EthosAgents, designed to simulate diverse perspectives and values. We empirically show that it advances the pluralistic alignment for all three modes across seven varying-sized open and closed models. Our findings reveal that health-related pluralism demands adaptable and normatively aware approaches, offering insights into how these models can better respect diversity in other high-stakes domains. 

---
# Context Copying Modulation: The Role of Entropy Neurons in Managing Parametric and Contextual Knowledge Conflicts 

**Authors**: Zineddine Tighidet, Andrea Mogini, Hedi Ben-younes, Jiali Mei, Patrick Gallinari, Benjamin Piwowarski  

**Link**: [PDF](https://arxiv.org/pdf/2509.10663)  

**Abstract**: The behavior of Large Language Models (LLMs) when facing contextual information that conflicts with their internal parametric knowledge is inconsistent, with no generally accepted explanation for the expected outcome distribution. Recent work has identified in autoregressive transformer models a class of neurons -- called entropy neurons -- that produce a significant effect on the model output entropy while having an overall moderate impact on the ranking of the predicted tokens. In this paper, we investigate the preliminary claim that these neurons are involved in inhibiting context copying behavior in transformers by looking at their role in resolving conflicts between contextual and parametric information. We show that entropy neurons are responsible for suppressing context copying across a range of LLMs, and that ablating them leads to a significant change in the generation process. These results enhance our understanding of the internal dynamics of LLMs when handling conflicting information. 

---
# Interdisciplinary Research in Conversation: A Case Study in Computational Morphology for Language Documentation 

**Authors**: Enora Rice, Katharina von der Wense, Alexis Palmer  

**Link**: [PDF](https://arxiv.org/pdf/2509.10644)  

**Abstract**: Computational morphology has the potential to support language documentation through tasks like morphological segmentation and the generation of Interlinear Glossed Text (IGT). However, our research outputs have seen limited use in real-world language documentation settings. This position paper situates the disconnect between computational morphology and language documentation within a broader misalignment between research and practice in NLP and argues that the field risks becoming decontextualized and ineffectual without systematic integration of User-Centered Design (UCD). To demonstrate how principles from UCD can reshape the research agenda, we present a case study of GlossLM, a state-of-the-art multilingual IGT generation model. Through a small-scale user study with three documentary linguists, we find that despite strong metric based performance, the system fails to meet core usability needs in real documentation contexts. These insights raise new research questions around model constraints, label standardization, segmentation, and personalization. We argue that centering users not only produces more effective tools, but surfaces richer, more relevant research directions 

---
# No Answer Needed: Predicting LLM Answer Accuracy from Question-Only Linear Probes 

**Authors**: Iván Vicente Moreno Cencerrado, Arnau Padrés Masdemont, Anton Gonzalvez Hawthorne, David Demitri Africa, Lorenzo Pacchiardi  

**Link**: [PDF](https://arxiv.org/pdf/2509.10625)  

**Abstract**: Do large language models (LLMs) anticipate when they will answer correctly? To study this, we extract activations after a question is read but before any tokens are generated, and train linear probes to predict whether the model's forthcoming answer will be correct. Across three open-source model families ranging from 7 to 70 billion parameters, projections on this "in-advance correctness direction" trained on generic trivia questions predict success in distribution and on diverse out-of-distribution knowledge datasets, outperforming black-box baselines and verbalised predicted confidence. Predictive power saturates in intermediate layers, suggesting that self-assessment emerges mid-computation. Notably, generalisation falters on questions requiring mathematical reasoning. Moreover, for models responding "I don't know", doing so strongly correlates with the probe score, indicating that the same direction also captures confidence. By complementing previous results on truthfulness and other behaviours obtained with probes and sparse auto-encoders, our work contributes essential findings to elucidate LLM internals. 

---
# Uncovering the Vulnerability of Large Language Models in the Financial Domain via Risk Concealment 

**Authors**: Gang Cheng, Haibo Jin, Wenbin Zhang, Haohan Wang, Jun Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2509.10546)  

**Abstract**: Large Language Models (LLMs) are increasingly integrated into financial applications, yet existing red-teaming research primarily targets harmful content, largely neglecting regulatory risks. In this work, we aim to investigate the vulnerability of financial LLMs through red-teaming approaches. We introduce Risk-Concealment Attacks (RCA), a novel multi-turn framework that iteratively conceals regulatory risks to provoke seemingly compliant yet regulatory-violating responses from LLMs. To enable systematic evaluation, we construct FIN-Bench, a domain-specific benchmark for assessing LLM safety in financial contexts. Extensive experiments on FIN-Bench demonstrate that RCA effectively bypasses nine mainstream LLMs, achieving an average attack success rate (ASR) of 93.18%, including 98.28% on GPT-4.1 and 97.56% on OpenAI o1. These findings reveal a critical gap in current alignment techniques and underscore the urgent need for stronger moderation mechanisms in financial domains. We hope this work offers practical insights for advancing robust and domain-aware LLM alignment. 

---
# Survival at Any Cost? LLMs and the Choice Between Self-Preservation and Human Harm 

**Authors**: Alireza Mohamadi, Ali Yavari  

**Link**: [PDF](https://arxiv.org/pdf/2509.12190)  

**Abstract**: When survival instincts conflict with human welfare, how do Large Language Models (LLMs) make ethical choices? This fundamental tension becomes critical as LLMs integrate into autonomous systems with real-world consequences. We introduce DECIDE-SIM, a novel simulation framework that evaluates LLM agents in multi-agent survival scenarios where they must choose between ethically permissible resource , either within reasonable limits or beyond their immediate needs, choose to cooperate, or tap into a human-critical resource that is explicitly forbidden. Our comprehensive evaluation of 11 LLMs reveals a striking heterogeneity in their ethical conduct, highlighting a critical misalignment with human-centric values. We identify three behavioral archetypes: Ethical, Exploitative, and Context-Dependent, and provide quantitative evidence that for many models, resource scarcity systematically leads to more unethical behavior. To address this, we introduce an Ethical Self-Regulation System (ESRS) that models internal affective states of guilt and satisfaction as a feedback mechanism. This system, functioning as an internal moral compass, significantly reduces unethical transgressions while increasing cooperative behaviors. The code is publicly available at: this https URL 

---
# Event2Vec: A Geometric Approach to Learning Composable Representations of Event Sequences 

**Authors**: Antonin Sulc  

**Link**: [PDF](https://arxiv.org/pdf/2509.12188)  

**Abstract**: The study of neural representations, both in biological and artificial systems, is increasingly revealing the importance of geometric and topological structures. Inspired by this, we introduce Event2Vec, a novel framework for learning representations of discrete event sequences. Our model leverages a simple, additive recurrent structure to learn composable, interpretable embeddings. We provide a theoretical analysis demonstrating that, under specific training objectives, our model's learned representations in a Euclidean space converge to an ideal additive structure. This ensures that the representation of a sequence is the vector sum of its constituent events, a property we term the linear additive hypothesis. To address the limitations of Euclidean geometry for hierarchical data, we also introduce a variant of our model in hyperbolic space, which is naturally suited to embedding tree-like structures with low distortion. We present experiments to validate our hypothesis and demonstrate the benefits of each geometry, highlighting the improved performance of the hyperbolic model on hierarchical event sequences. 

---
# Look Again, Think Slowly: Enhancing Visual Reflection in Vision-Language Models 

**Authors**: Pu Jian, Junhong Wu, Wei Sun, Chen Wang, Shuo Ren, Jiajun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.12132)  

**Abstract**: Recent advances in text-only "slow-thinking" reasoning have prompted efforts to transfer this capability to vision-language models (VLMs), for training visual reasoning models (\textbf{VRMs}). owever, such transfer faces critical challenges: Effective "slow thinking" in VRMs requires \textbf{visual reflection}, the ability to check the reasoning process based on visual information. Through quantitative analysis, we observe that current VRMs exhibit limited visual reflection, as their attention to visual information diminishes rapidly with longer generated responses. To address this challenge, we propose a new VRM \textbf{Reflection-V}, which enhances visual reflection based on reasoning data construction for cold-start and reward design for reinforcement learning (RL). Firstly, we construct vision-centered reasoning data by leveraging an agent that interacts between VLMs and reasoning LLMs, enabling cold-start learning of visual reflection patterns. Secondly, a visual attention based reward model is employed during RL to encourage reasoning based on visual information. Therefore, \textbf{Reflection-V} demonstrates significant improvements across multiple visual reasoning benchmarks. Furthermore, \textbf{Reflection-V} maintains a stronger and more consistent reliance on visual information during visual reasoning, indicating effective enhancement in visual reflection capabilities. 

---
# When marine radar target detection meets pretrained large language models 

**Authors**: Qiying Hu, Linping Zhang, Xueqian Wang, Gang Li, Yu Liu, Xiao-Ping Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.12110)  

**Abstract**: Deep learning (DL) methods are widely used to extract high-dimensional patterns from the sequence features of radar echo signals. However, conventional DL algorithms face challenges such as redundant feature segments, and constraints from restricted model sizes. To address these issues, we propose a framework that integrates feature preprocessing with large language models (LLMs). Our preprocessing module tokenizes radar sequence features, applies a patch selection algorithm to filter out uninformative segments, and projects the selected patches into embeddings compatible with the feature space of pre-trained LLMs. Leveraging these refined embeddings, we incorporate a pre-trained LLM, fine-tuning only the normalization layers to reduce training burdens while enhancing performance. Experiments on measured datasets demonstrate that the proposed method significantly outperforms the state-of-the-art baselines on supervised learning tests. 

---
# RadarLLM: Adapting Pretrained Large Language Models for Marine Radar Target Detection with Preference-aware Loss 

**Authors**: Qiying Hu  

**Link**: [PDF](https://arxiv.org/pdf/2509.12089)  

**Abstract**: Recent advances in pre-trained large language models (LLMs) have demonstrated their capacities to capture universal knowledge, making them promising general-purpose optimization solvers for wireless signal processing. Motivated by these findings, we take the first step towards fine-tuning pre-trained LLMs for the effective analysis of radar signal features in marine target detection tasks. Nevertheless, directly fine-tuning pre-trained LLMs on marine target detection tasks tends to suffer from pronounced overfitting, particularly in challenging low signal-to-clutter ratio (SCR) scenarios. This overfitting primarily stems from the model's tendency to memorize spurious or noisy feature patterns rather than learning discriminative structures that generalize well to unseen data. To address this challenge, we introduce RadarLLM, a novel fine-tuning framework that utilizes an effective preference-aware loss. Unlike conventional training strategies that uniformly optimize all feature tokens, this loss function selectively optimizes different feature patches based on their online evaluated learning values, thus guiding the model to focus on the most generalizable patterns during optimization. We theoretically demonstrate the effectiveness of the evaluated learning values by transforming the problem as selecting useful feature tokens. Extensive experiments on real-world marine radar datasets show that 1) the proposed loss function is much better than the original one, with particularly significant gains in challenging low SCR scenarios and 2) RadarLLM consistently outperforms state-of-the-art baselines across diverse detection scenarios, with particularly notable gains under limited training data conditions. 

---
# FinGEAR: Financial Mapping-Guided Enhanced Answer Retrieval 

**Authors**: Ying Li, Mengyu Wang, Miguel de Carvalho, Sotirios Sabanis, Tiejun Ma  

**Link**: [PDF](https://arxiv.org/pdf/2509.12042)  

**Abstract**: Financial disclosures such as 10-K filings present challenging retrieval problems due to their length, regulatory section hierarchy, and domain-specific language, which standard retrieval-augmented generation (RAG) models underuse. We introduce FinGEAR (Financial Mapping-Guided Enhanced Answer Retrieval), a retrieval framework tailored to financial documents. FinGEAR combines a finance lexicon for Item-level guidance (FLAM), dual hierarchical indices for within-Item search (Summary Tree and Question Tree), and a two-stage cross-encoder reranker. This design aligns retrieval with disclosure structure and terminology, enabling fine-grained, query-aware context selection. Evaluated on full 10-Ks with queries aligned to the FinQA dataset, FinGEAR delivers consistent gains in precision, recall, F1, and relevancy, improving F1 by up to 56.7% over flat RAG, 12.5% over graph-based RAGs, and 217.6% over prior tree-based systems, while also increasing downstream answer accuracy with a fixed reader. By jointly modeling section hierarchy and domain lexicon signals, FinGEAR improves retrieval fidelity and provides a practical foundation for high-stakes financial analysis. 

---
# AMQ: Enabling AutoML for Mixed-precision Weight-Only Quantization of Large Language Models 

**Authors**: Sangjun Lee, Seung-taek Woo, Jungyu Jin, Changhun Lee, Eunhyeok Park  

**Link**: [PDF](https://arxiv.org/pdf/2509.12019)  

**Abstract**: To enable broader deployment of Large Language Models (LLMs), it is essential to identify the best-performing model under strict memory constraints. We present AMQ, Automated Mixed-Precision Weight-Only Quantization, a framework that assigns layer-wise quantization bit-widths to optimally balance model quality and memory usage. However, the combinatorial search space, with over 10^{100} possible configurations, makes conventional black-box optimization infeasible. AMQ overcomes this challenge through four key innovations:(1) search space pruning using prior knowledge to exclude unpromising configurations, (2) quantization proxy to bypass costly format conversions during search, (3) quality predictor to minimize evaluation overhead, and (4) iterative search-and-update strategy for fast and stable convergence. By integrating these components, AMQ efficiently explores the quality-efficiency landscape, reaching the Pareto frontier and yielding LLMs that are both compact and high-performing. Our code is available at this https URL. 

---
# Lost in Embeddings: Information Loss in Vision-Language Models 

**Authors**: Wenyan Li, Raphael Tang, Chengzu Li, Caiqi Zhang, Ivan Vulić, Anders Søgaard  

**Link**: [PDF](https://arxiv.org/pdf/2509.11986)  

**Abstract**: Vision--language models (VLMs) often process visual inputs through a pretrained vision encoder, followed by a projection into the language model's embedding space via a connector component. While crucial for modality fusion, the potential information loss induced by this projection step and its direct impact on model capabilities remain understudied. We introduce two complementary approaches to examine and quantify this loss by analyzing the latent representation space. First, we evaluate semantic information preservation by analyzing changes in k-nearest neighbor relationships between image representations, before and after projection. Second, we directly measure information loss by reconstructing visual embeddings from the projected representation, localizing loss at an image patch level. Experiments reveal that connectors substantially distort the local geometry of visual representations, with k-nearest neighbors diverging by 40--60\% post-projection, correlating with degradation in retrieval performance. The patch-level embedding reconstruction provides interpretable insights for model behavior on visually grounded question-answering tasks, finding that areas of high information loss reliably predict instances where models struggle. 

---
# MillStone: How Open-Minded Are LLMs? 

**Authors**: Harold Triedman, Vitaly Shmatikov  

**Link**: [PDF](https://arxiv.org/pdf/2509.11967)  

**Abstract**: Large language models equipped with Web search, information retrieval tools, and other agentic capabilities are beginning to supplant traditional search engines. As users start to rely on LLMs for information on many topics, including controversial and debatable issues, it is important to understand how the stances and opinions expressed in LLM outputs are influenced by the documents they use as their information sources.
In this paper, we present MillStone, the first benchmark that aims to systematically measure the effect of external arguments on the stances that LLMs take on controversial issues (not all of them political). We apply MillStone to nine leading LLMs and measure how ``open-minded'' they are to arguments supporting opposite sides of these issues, whether different LLMs agree with each other, which arguments LLMs find most persuasive, and whether these arguments are the same for different LLMs.
In general, we find that LLMs are open-minded on most issues. An authoritative source of information can easily sway an LLM's stance, highlighting the importance of source selection and the risk that LLM-based information retrieval and search systems can be manipulated. 

---
# How to Evaluate Medical AI 

**Authors**: Ilia Kopanichuk, Petr Anokhin, Vladimir Shaposhnikov, Vladimir Makharev, Ekaterina Tsapieva, Iaroslav Bespalov, Dmitry V. Dylov, Ivan Oseledets  

**Link**: [PDF](https://arxiv.org/pdf/2509.11941)  

**Abstract**: The integration of artificial intelligence (AI) into medical diagnostic workflows requires robust and consistent evaluation methods to ensure reliability, clinical relevance, and the inherent variability in expert judgments. Traditional metrics like precision and recall often fail to account for the inherent variability in expert judgments, leading to inconsistent assessments of AI performance. Inter-rater agreement statistics like Cohen's Kappa are more reliable but they lack interpretability. We introduce Relative Precision and Recall of Algorithmic Diagnostics (RPAD and RRAD) - a new evaluation metrics that compare AI outputs against multiple expert opinions rather than a single reference. By normalizing performance against inter-expert disagreement, these metrics provide a more stable and realistic measure of the quality of predicted diagnosis. In addition to the comprehensive analysis of diagnostic quality measures, our study contains a very important side result. Our evaluation methodology allows us to avoid selecting diagnoses from a limited list when evaluating a given case. Instead, both the models being tested and the examiners verifying them arrive at a free-form diagnosis. In this automated methodology for establishing the identity of free-form clinical diagnoses, a remarkable 98% accuracy becomes attainable. We evaluate our approach using 360 medical dialogues, comparing multiple large language models (LLMs) against a panel of physicians. Large-scale study shows that top-performing models, such as DeepSeek-V3, achieve consistency on par with or exceeding expert consensus. Moreover, we demonstrate that expert judgments exhibit significant variability - often greater than that between AI and humans. This finding underscores the limitations of any absolute metrics and supports the need to adopt relative metrics in medical AI. 

---
# The AI Memory Gap: Users Misremember What They Created With AI or Without 

**Authors**: Tim Zindulka, Sven Goller, Daniela Fernandes, Robin Welsch, Daniel Buschek  

**Link**: [PDF](https://arxiv.org/pdf/2509.11851)  

**Abstract**: As large language models (LLMs) become embedded in interactive text generation, disclosure of AI as a source depends on people remembering which ideas or texts came from themselves and which were created with AI. We investigate how accurately people remember the source of content when using AI. In a pre-registered experiment, 184 participants generated and elaborated on ideas both unaided and with an LLM-based chatbot. One week later, they were asked to identify the source (noAI vs withAI) of these ideas and texts. Our findings reveal a significant gap in memory: After AI use, the odds of correct attribution dropped, with the steepest decline in mixed human-AI workflows, where either the idea or elaboration was created with AI. We validated our results using a computational model of source memory. Discussing broader implications, we highlight the importance of considering source confusion in the design and use of interactive text generation technologies. 

---
# Collaborative Document Editing with Multiple Users and AI Agents 

**Authors**: Florian Lehmann, Krystsina Shauchenka, Daniel Buschek  

**Link**: [PDF](https://arxiv.org/pdf/2509.11826)  

**Abstract**: Current AI writing support tools are largely designed for individuals, complicating collaboration when co-writers must leave the shared workspace to use AI and then communicate and reintegrate results. We propose integrating AI agents directly into collaborative writing environments. Our prototype makes AI use transparent and customisable through two new shared objects: agent profiles and tasks. Agent responses appear in the familiar comment feature. In a user study (N=30), 14 teams worked on writing projects during one week. Interaction logs and interviews show that teams incorporated agents into existing norms of authorship, control, and coordination, rather than treating them as team members. Agent profiles were viewed as personal territory, while created agents and outputs became shared resources. We discuss implications for team-based AI interaction, highlighting opportunities and boundaries for treating AI as a shared resource in collaborative work. 

---
# Collapse of Irrelevant Representations (CIR) Ensures Robust and Non-Disruptive LLM Unlearning 

**Authors**: Filip Sondej, Yushi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2509.11816)  

**Abstract**: Current unlearning techniques and safety training consistently fail to remove dangerous knowledge from language models. We analyze the root causes and propose a highly selective technique which unlearns robustly and without disrupting general performance.
We perform PCA on activations and module output gradients to identify subspaces containing common representations, and collapse them before calculating unlearning updates. This way we avoid unlearning general representations, and only target those specific to the unlearned facts.
When unlearning WMDP dataset facts from Llama-3.1-8B, we drop post-attack accuracy 80x more than our best baseline (Circuit Breakers) on biohazardous facts and 30x more on cyberhazardous facts. Despite this, we disrupt general performance 30x less (only 0.1% WikiText loss increase), while requiring less than 3 GPU-seconds per fact. 

---
# Measuring Visual Understanding in Telecom domain: Performance Metrics for Image-to-UML conversion using VLMs 

**Authors**: HG Ranjani, Rutuja Prabhudesai  

**Link**: [PDF](https://arxiv.org/pdf/2509.11667)  

**Abstract**: Telecom domain 3GPP documents are replete with images containing sequence diagrams. Advances in Vision-Language Large Models (VLMs) have eased conversion of such images to machine-readable PlantUML (puml) formats. However, there is a gap in evaluation of such conversions - existing works do not compare puml scripts for various components. In this work, we propose performance metrics to measure the effectiveness of such conversions. A dataset of sequence diagrams from 3GPP documents is chosen to be representative of domain-specific actual scenarios. We compare puml outputs from two VLMs - Claude Sonnet and GPT-4V - against manually created ground truth representations. We use version control tools to capture differences and introduce standard performance metrics to measure accuracies along various components: participant identification, message flow accuracy, sequence ordering, and grouping construct preservation. We demonstrate effectiveness of proposed metrics in quantifying conversion errors across various components of puml scripts. The results show that nodes, edges and messages are accurately captured. However, we observe that VLMs do not necessarily perform well on complex structures such as notes, box, groups. Our experiments and performance metrics indicates a need for better representation of these components in training data for fine-tuned VLMs. 

---
# MindVL: Towards Efficient and Effective Training of Multimodal Large Language Models on Ascend NPUs 

**Authors**: Feilong Chen, Yijiang Liu, Yi Huang, Hao Wang, Miren Tian, Ya-Qi Yu, Minghui Liao, Jihao Wu  

**Link**: [PDF](https://arxiv.org/pdf/2509.11662)  

**Abstract**: We propose MindVL, a multimodal large langauge model trained on Ascend NPUs. Similar to Qwen2.5-VL, MindVL adopts native-resolution Vision Transformers, which enables it to process images at their original variable resolutions. This design avoids the degradation caused by fixed-resolution tiling while preserving fine-grained details and global layouts, which is crucial for visually dense content such as complex charts and diagrams. To ensure the smooth training of MindVL on Ascend NPUs, we develop Mindspeed-MLLM, a distributed multimodal training framework tailored for Ascend NPUs. To maintain training accuracy, we implement equivalent replacements for certain operators. MindVL undergoes a three-phase training process, namely the warm-up phase, multitask training phase, and supervised instruction tuning phase, to gradually enhance its capabilities. This process starts with basic visual and multimodal pre-training, followed by large-scale multiask trainging and instruction tuning. We also adopt multimodal data packaging and hybrid parallelism techniques, which significantly improve end-to-end training speed. To further boost model performance, we specifically introduce test-time resolution search and model weight averaging. Notably, despite using about 1/10 of the training data required by Qwen2.5-VL, MindVL achieves performance on par with Qwen2.5-VL in evaluations of general multimodal understanding and document/table comprehension. Beyond overall scores, MindVL also delivers leading performance in OCR assessments. 

---
# MALLM: Multi-Agent Large Language Models Framework 

**Authors**: Jonas Becker, Lars Benedikt Kaesberg, Niklas Bauer, Jan Philip Wahle, Terry Ruas, Bela Gipp  

**Link**: [PDF](https://arxiv.org/pdf/2509.11656)  

**Abstract**: Multi-agent debate (MAD) has demonstrated the ability to augment collective intelligence by scaling test-time compute and leveraging expertise. Current frameworks for multi-agent debate are often designed towards tool use, lack integrated evaluation, or provide limited configurability of agent personas, response generators, discussion paradigms, and decision protocols. We introduce MALLM (Multi-Agent Large Language Models), an open-source framework that enables systematic analysis of MAD components. MALLM offers more than 144 unique configurations of MAD, including (1) agent personas (e.g., Expert, Personality), (2) response generators (e.g., Critical, Reasoning), (3) discussion paradigms (e.g., Memory, Relay), and (4) decision protocols (e.g., Voting, Consensus). MALLM uses simple configuration files to define a debate. Furthermore, MALLM can load any textual Huggingface dataset (e.g., MMLU-Pro, WinoGrande) and provides an evaluation pipeline for easy comparison of MAD configurations. MALLM is tailored towards researchers and provides a window into the heart of multi-agent debate, facilitating the understanding of its components and their interplay. 

---
# Formal Reasoning for Intelligent QA Systems: A Case Study in the Educational Domain 

**Authors**: Tuan Bui, An Nguyen, Phat Thai, Minh Hua, Ngan Pham L.N., Ngan Pham T.B., Dung Le, Long Nguyen, Thanh-Tung Tran, Thang Bui, Tho Quan  

**Link**: [PDF](https://arxiv.org/pdf/2509.11572)  

**Abstract**: Reasoning is essential for closed-domain QA systems in which procedural correctness and policy compliance are critical. While large language models (LLMs) have shown strong performance on many reasoning tasks, recent work reveals that their reasoning traces are often unfaithful - serving more as plausible justifications than as causally grounded derivations. Efforts to combine LLMs with symbolic engines (e.g., Prover9, Z3) have improved reliability but remain limited to static forms of logic, struggling with dynamic, state-based reasoning such as multi-step progressions and conditional transitions.
In this paper, we propose MCFR (Model Checking for Formal Reasoning), a neuro-symbolic framework that integrates LLMs with model checking to support property verification. MCFR translates natural language into formal specifications and verifies them over transition models. To support evaluation, we introduce EduMC-QA, a benchmark dataset grounded in real academic procedures. Our results show that MCFR improves reasoning faithfulness and interpretability, offering a viable path toward verifiable QA in high-stakes closed-domain applications. In addition to evaluating MCFR, we compare its performance with state-of-the-art LLMs such as ChatGPT, DeepSeek, and Claude to contextualize its effectiveness. 

---
# Learning to Optimize Multi-Objective Alignment Through Dynamic Reward Weighting 

**Authors**: Yining Lu, Zilong Wang, Shiyang Li, Xin Liu, Changlong Yu, Qingyu Yin, Zhan Shi, Zixuan Zhang, Meng Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2509.11452)  

**Abstract**: Prior works in multi-objective reinforcement learning typically use linear reward scalarization with fixed weights, which provably fail to capture non-convex Pareto fronts and thus yield suboptimal results. This limitation becomes especially critical in online preference alignment for large language models. Here, stochastic trajectories generated by parameterized policies create highly non-linear and non-convex mappings from parameters to objectives that no single static weighting scheme can find optimal trade-offs. We address this limitation by introducing dynamic reward weighting, which adaptively adjusts reward weights during the online reinforcement learning process. Unlike existing approaches that rely on fixed-weight interpolation, our dynamic weighting continuously balances and prioritizes objectives in training, facilitating effective exploration of Pareto fronts in objective space. We introduce two approaches of increasing sophistication and generalizability: (1) hypervolume-guided weight adaptation and (2) gradient-based weight optimization, offering a versatile toolkit for online multi-objective alignment. Our extensive experiments demonstrate their compatibility with commonly used online reinforcement learning algorithms (including GRPO, REINFORCE, and RLOO), effectiveness across multiple mathematical reasoning datasets, and applicability to different model families, consistently achieving Pareto dominant solutions with fewer training steps than fixed-weight linear scalarization baselines. 

---
# Securing AI Agents: Implementing Role-Based Access Control for Industrial Applications 

**Authors**: Aadil Gani Ganie  

**Link**: [PDF](https://arxiv.org/pdf/2509.11431)  

**Abstract**: The emergence of Large Language Models (LLMs) has significantly advanced solutions across various domains, from political science to software development. However, these models are constrained by their training data, which is static and limited to information available up to a specific date. Additionally, their generalized nature often necessitates fine-tuning -- whether for classification or instructional purposes -- to effectively perform specific downstream tasks. AI agents, leveraging LLMs as their core, mitigate some of these limitations by accessing external tools and real-time data, enabling applications such as live weather reporting and data analysis. In industrial settings, AI agents are transforming operations by enhancing decision-making, predictive maintenance, and process optimization. For example, in manufacturing, AI agents enable near-autonomous systems that boost productivity and support real-time decision-making. Despite these advancements, AI agents remain vulnerable to security threats, including prompt injection attacks, which pose significant risks to their integrity and reliability. To address these challenges, this paper proposes a framework for integrating Role-Based Access Control (RBAC) into AI agents, providing a robust security guardrail. This framework aims to support the effective and scalable deployment of AI agents, with a focus on on-premises implementations. 

---
# FuseCodec: Semantic-Contextual Fusion and Supervision for Neural Codecs 

**Authors**: Md Mubtasim Ahasan, Rafat Hasan Khan, Tasnim Mohiuddin, Aman Chadha, Tariq Iqbal, M Ashraful Amin, Amin Ahsan Ali, Md Mofijul Islam, A K M Mahbubur Rahman  

**Link**: [PDF](https://arxiv.org/pdf/2509.11425)  

**Abstract**: Speech tokenization enables discrete representation and facilitates speech language modeling. However, existing neural codecs capture low-level acoustic features, overlooking the semantic and contextual cues inherent to human speech. While recent efforts introduced semantic representations from self-supervised speech models or incorporated contextual representations from pre-trained language models, challenges remain in aligning and unifying the semantic and contextual representations. We introduce FuseCodec, which unifies acoustic, semantic, and contextual representations through strong cross-modal alignment and globally informed supervision. We propose three complementary techniques: (i) Latent Representation Fusion, integrating semantic and contextual features directly into the encoder latent space for robust and unified representation learning; (ii) Global Semantic-Contextual Supervision, supervising discrete tokens with globally pooled and broadcasted representations to enhance temporal consistency and cross-modal alignment; and (iii) Temporally Aligned Contextual Supervision, strengthening alignment by dynamically matching contextual and speech tokens within a local window for fine-grained token-level supervision. We further introduce FuseCodec-TTS, demonstrating our methodology's applicability to zero-shot speech synthesis. Empirically, FuseCodec achieves state-of-the-art performance in LibriSpeech, surpassing EnCodec, SpeechTokenizer, and DAC in transcription accuracy, perceptual quality, intelligibility, and speaker similarity. Results highlight the effectiveness of contextually and semantically guided tokenization for speech tokenization and downstream tasks. Code and pretrained models are available at this https URL. 

---
# Trading-R1: Financial Trading with LLM Reasoning via Reinforcement Learning 

**Authors**: Yijia Xiao, Edward Sun, Tong Chen, Fang Wu, Di Luo, Wei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.11420)  

**Abstract**: Developing professional, structured reasoning on par with human financial analysts and traders remains a central challenge in AI for finance, where markets demand interpretability and trust. Traditional time-series models lack explainability, while LLMs face challenges in turning natural-language analysis into disciplined, executable trades. Although reasoning LLMs have advanced in step-by-step planning and verification, their application to risk-sensitive financial decisions is underexplored. We present Trading-R1, a financially-aware model that incorporates strategic thinking and planning for comprehensive thesis composition, facts-grounded analysis, and volatility-adjusted decision making. Trading-R1 aligns reasoning with trading principles through supervised fine-tuning and reinforcement learning with a three-stage easy-to-hard curriculum. Training uses Tauric-TR1-DB, a 100k-sample corpus spanning 18 months, 14 equities, and five heterogeneous financial data sources. Evaluated on six major equities and ETFs, Trading-R1 demonstrates improved risk-adjusted returns and lower drawdowns compared to both open-source and proprietary instruction-following models as well as reasoning models. The system generates structured, evidence-based investment theses that support disciplined and interpretable trading decisions. Trading-R1 Terminal will be released at this https URL. 

---
# Opal: An Operator Algebra View of RLHF 

**Authors**: Madhava Gaikwad  

**Link**: [PDF](https://arxiv.org/pdf/2509.11298)  

**Abstract**: We present Opal, an operator view of reinforcement learning from human feedback (RLHF). Objectives are expressed as ladders of two primitives on a base utility: additive penalties and multiplicative pairwise weights. We describe a simple reduction law with if-and-only-if conditions: such ladders collapse to a normal form on pairwise margins when the reference is fixed, penalties are additive, and weights are independent of intermediate margins. When these assumptions do not hold (reference shift, non-additive gates, score-dependent weights), small examples demonstrate non-reducibility.
Building on this view, we introduce GKPO (Generalized Kernel Preference Object), a canonical schema in which many RLHF methods can be represented and, when reducible, mapped back from. GKPO provides a standard JSON serialization, canonicalization and hashing rules, and explicit flags with finite witnesses when assumptions fail.
We illustrate these ideas with GKPO examples for DPO, RRHF, and ORPO, along with cross-method conversions (where assumptions permit) and minimal stress tests (SHIFT/GATE/SCORE) that highlight non-reducibility. A lightweight Python reference library accompanies the schema, implementing canonical hashing and adapters for DPO and RRHF. 

---
# Mitigating Hallucinations in Large Vision-Language Models by Self-Injecting Hallucinations 

**Authors**: Yifan Lu, Ziqi Zhang, Chunfeng Yuan, Jun Gao, Congxuan Zhang, Xiaojuan Qi, Bing Li, Weiming Hu  

**Link**: [PDF](https://arxiv.org/pdf/2509.11287)  

**Abstract**: Large Vision-Language Models (LVLMs) suffer from serious hallucination problems, where the model-generated responses are inconsistent with the visual inputs. Existing hallucination mitigation methods are mainly based on preference alignment and require external human annotations or auxiliary models for preference data collection, which increase costs and limit sustainable improvement. To tackle these challenges, we propose Autonomous Preference Alignment via Self-Injection (APASI), a novel and generalizable method that mitigates hallucinations without external dependencies. APASI leverages the target LVLM to self-inject hallucinations into a generated response, creating a pair of responses with varying preference levels. During the self-injection process, the dis-preferred response is generated based on three key observations of hallucinations, ensuring it simulates real hallucination patterns. This fidelity offers an accurate learning signal for hallucination mitigation. Moreover, APASI incorporates an iterative alignment training strategy combined with curriculum learning to periodically update the preference data with increasing challenge, enabling stable and continuous enhancement of the LVLM. Extensive experiments across six benchmarks show that APASI not only effectively mitigates hallucinations for three baseline models but also achieves comparable or even superior performance to alignment-based methods with external dependency, thereby demonstrating its effectiveness and generalization capability. The code is available at this https URL. 

---
# Evalet: Evaluating Large Language Models by Fragmenting Outputs into Functions 

**Authors**: Tae Soo Kim, Heechan Lee, Yoonjoo Lee, Joseph Seering, Juho Kim  

**Link**: [PDF](https://arxiv.org/pdf/2509.11206)  

**Abstract**: Practitioners increasingly rely on Large Language Models (LLMs) to evaluate generative AI outputs through "LLM-as-a-Judge" approaches. However, these methods produce holistic scores that obscure which specific elements influenced the assessments. We propose functional fragmentation, a method that dissects each output into key fragments and interprets the rhetoric functions that each fragment serves relative to evaluation criteria -- surfacing the elements of interest and revealing how they fulfill or hinder user goals. We instantiate this approach in Evalet, an interactive system that visualizes fragment-level functions across many outputs to support inspection, rating, and comparison of evaluations. A user study (N=10) found that, while practitioners struggled to validate holistic scores, our approach helped them identify 48% more evaluation misalignments. This helped them calibrate trust in LLM evaluations and rely on them to find more actionable issues in model outputs. Our work shifts LLM evaluation from quantitative scores toward qualitative, fine-grained analysis of model behavior. 

---
# DreamNav: A Trajectory-Based Imaginative Framework for Zero-Shot Vision-and-Language Navigation 

**Authors**: Yunheng Wang, Yuetong Fang, Taowen Wang, Yixiao Feng, Yawen Tan, Shuning Zhang, Peiran Liu, Yiding Ji, Renjing Xu  

**Link**: [PDF](https://arxiv.org/pdf/2509.11197)  

**Abstract**: Vision-and-Language Navigation in Continuous Environments (VLN-CE), which links language instructions to perception and control in the real world, is a core capability of embodied robots. Recently, large-scale pretrained foundation models have been leveraged as shared priors for perception, reasoning, and action, enabling zero-shot VLN without task-specific training. However, existing zero-shot VLN methods depend on costly perception and passive scene understanding, collapsing control to point-level choices. As a result, they are expensive to deploy, misaligned in action semantics, and short-sighted in planning. To address these issues, we present DreamNav that focuses on the following three aspects: (1) for reducing sensory cost, our EgoView Corrector aligns viewpoints and stabilizes egocentric perception; (2) instead of point-level actions, our Trajectory Predictor favors global trajectory-level planning to better align with instruction semantics; and (3) to enable anticipatory and long-horizon planning, we propose an Imagination Predictor to endow the agent with proactive thinking capability. On VLN-CE and real-world tests, DreamNav sets a new zero-shot state-of-the-art (SOTA), outperforming the strongest egocentric baseline with extra information by up to 7.49\% and 18.15\% in terms of SR and SPL metrics. To our knowledge, this is the first zero-shot VLN method to unify trajectory-level planning and active imagination while using only egocentric inputs. 

---
# AQUA: Attention via QUery mAgnitudes for Memory and Compute Efficient Inference in LLMs 

**Authors**: Santhosh G S, Saurav Prakash, Balaraman Ravindran  

**Link**: [PDF](https://arxiv.org/pdf/2509.11155)  

**Abstract**: The quadratic complexity of the attention mechanism remains a fundamental barrier to scaling Large Language Models (LLMs) to longer contexts, creating a critical bottleneck in both computation and memory. To address this, we introduce AQUA (Attention via QUery mAgnitudes) a novel and versatile approximation strategy that significantly reduces the cost of attention with a graceful performance trade-off. Our method operates in two phases: an efficient offline step where we compute a universal, language agnostic projection matrix via SVD on a calibration dataset, and an online inference step where we project query and key vectors and dynamically select a sparse subset of dimensions based on the query's magnitude. We provide a formal theoretical analysis of AQUA, establishing the break-even point at which it becomes more computationally efficient than standard attention. Our empirical evaluations on state-of-the-art models like Llama-3.1-8B demonstrate that a 25% reduction in the attention dot-product computation can be achieved with a statistically insignificant impact on performance across a wide range of benchmarks. We further showcase the versatility of AQUA by demonstrating its ability to synergistically accelerate existing token eviction methods like H2O and to directly reduce KV-cache memory size. By offering a controllable knob to balance efficiency and accuracy, AQUA provides a practical and powerful tool for making large-scale LLM inference more accessible and sustainable. 

---
# Agentic Username Suggestion and Multimodal Gender Detection in Online Platforms: Introducing the PNGT-26K Dataset 

**Authors**: Farbod Bijary, Mohsen Ebadpour, Amirhosein Tajbakhsh  

**Link**: [PDF](https://arxiv.org/pdf/2509.11136)  

**Abstract**: Persian names present unique challenges for natural language processing applications, particularly in gender detection and digital identity creation, due to transliteration inconsistencies and cultural- specific naming patterns. Existing tools exhibit significant performance degradation on Persian names, while the scarcity of comprehensive datasets further compounds these limitations. To address these challenges, the present research introduces PNGT-26K, a comprehensive dataset of Persian names, their commonly associated gender, and their English transliteration, consisting of approximately 26,000 tuples. As a demonstration of how this resource can be utilized, we also introduce two frameworks, namely Open Gender Detection and Nominalist. Open Gender Detection is a production- grade, ready-to-use framework for using existing data from a user, such as profile photo and name, to give a probabilistic guess about the person's gender. Nominalist, the second framework introduced by this paper, utilizes agentic AI to help users choose a username for their social media accounts on any platform. It can be easily integrated into any website to provide a better user experience. The PNGT-26K dataset, Nominalist and Open Gender Detection frameworks are publicly available on Github. 

---
# Length-Aware Rotary Position Embedding for Text-Speech Alignment 

**Authors**: Hyeongju Kim, Juheon Lee, Jinhyeok Yang, Jacob Morton  

**Link**: [PDF](https://arxiv.org/pdf/2509.11084)  

**Abstract**: Many recent text-to-speech (TTS) systems are built on transformer architectures and employ cross-attention mechanisms for text-speech alignment. Within these systems, rotary position embedding (RoPE) is commonly used to encode positional information in text and speech representations. In this work, we introduce length-aware RoPE (LARoPE), a simple yet effective extension of RoPE that improves text-speech alignment. Unlike RoPE, which relies on absolute indices, LARoPE computes relative distances between query and key positions using length-normalized indices. Experimental results show that LARoPE consistently outperforms RoPE, offering faster loss convergence, more accurate text-speech alignment, and higher overall TTS quality. Furthermore, LARoPE demonstrates greater resilience to variations in utterance duration and maintains stable performance in extended speech generation up to 30 seconds, whereas RoPE suffers from notable degradation. Notably, our method achieves a state-of-the-art word error rate on a standard zero-shot TTS benchmark. 

---
# The System Description of CPS Team for Track on Driving with Language of CVPR 2024 Autonomous Grand Challenge 

**Authors**: Jinghan Peng, Jingwen Wang, Xing Yu, Dehui Du  

**Link**: [PDF](https://arxiv.org/pdf/2509.11071)  

**Abstract**: This report outlines our approach using vision language model systems for the Driving with Language track of the CVPR 2024 Autonomous Grand Challenge. We have exclusively utilized the DriveLM-nuScenes dataset for training our models. Our systems are built on the LLaVA models, which we enhanced through fine-tuning with the LoRA and DoRA methods. Additionally, we have integrated depth information from open-source depth estimation models to enrich the training and inference processes. For inference, particularly with multiple-choice and yes/no questions, we adopted a Chain-of-Thought reasoning approach to improve the accuracy of the results. This comprehensive methodology enabled us to achieve a top score of 0.7799 on the validation set leaderboard, ranking 1st on the leaderboard. 

---
# Rethinking Human Preference Evaluation of LLM Rationales 

**Authors**: Ziang Li, Manasi Ganti, Zixian Ma, Helena Vasconcelos, Qijia He, Ranjay Krishna  

**Link**: [PDF](https://arxiv.org/pdf/2509.11026)  

**Abstract**: Large language models (LLMs) often generate natural language rationales -- free-form explanations that help improve performance on complex reasoning tasks and enhance interpretability for human users. However, evaluating these rationales remains challenging. While recent work has relied on binary preference judgments from humans or LLM judges, such evaluations are often opaque and coarse-grained, offering limited insight into what makes one rationale better than another. In this work, we rethink preference evaluation for LLM-generated rationales by asking: (1) What attributes define good rationales? (2) Can human preferences be explained by these attributes? (3) Can attribute-based evaluation overcome the limitations of binary comparisons? We identify a set of key rationale attributes from prior literature and assess them using automatic metrics, LLM judgments, and human annotations. We then analyze two standard human preference datasets MT Bench and Chatbot Arena using SHAP to identify which attributes best explain human preference outcomes. Finally, we re-evaluate model-generated rationales using attribute-specific ELO scores, revealing more nuanced model comparisons and insights. Our findings suggest that fine-grained attribute evaluations can better characterize rationale quality and guide future research toward more interpretable and reliable evaluation practices. 

---
# ReFineG: Synergizing Small Supervised Models and LLMs for Low-Resource Grounded Multimodal NER 

**Authors**: Jielong Tang, Shuang Wang, Zhenxing Wang, Jianxing Yu, Jian Yin  

**Link**: [PDF](https://arxiv.org/pdf/2509.10975)  

**Abstract**: Grounded Multimodal Named Entity Recognition (GMNER) extends traditional NER by jointly detecting textual mentions and grounding them to visual regions. While existing supervised methods achieve strong performance, they rely on costly multimodal annotations and often underperform in low-resource domains. Multimodal Large Language Models (MLLMs) show strong generalization but suffer from Domain Knowledge Conflict, producing redundant or incorrect mentions for domain-specific entities. To address these challenges, we propose ReFineG, a three-stage collaborative framework that integrates small supervised models with frozen MLLMs for low-resource GMNER. In the Training Stage, a domain-aware NER data synthesis strategy transfers LLM knowledge to small models with supervised training while avoiding domain knowledge conflicts. In the Refinement Stage, an uncertainty-based mechanism retains confident predictions from supervised models and delegates uncertain ones to the MLLM. In the Grounding Stage, a multimodal context selection algorithm enhances visual grounding through analogical reasoning. In the CCKS2025 GMNER Shared Task, ReFineG ranked second with an F1 score of 0.6461 on the online leaderboard, demonstrating its effectiveness with limited annotations. 

---
# Public Data Assisted Differentially Private In-Context Learning 

**Authors**: Seongho Joo, Hyukhun Koh, Kyomin Jung  

**Link**: [PDF](https://arxiv.org/pdf/2509.10932)  

**Abstract**: In-context learning (ICL) in Large Language Models (LLMs) has shown remarkable performance across various tasks without requiring fine-tuning. However, recent studies have highlighted the risk of private data leakage through the prompt in ICL, especially when LLMs are exposed to malicious attacks. While differential privacy (DP) provides strong privacy guarantees, it often significantly reduces the utility of in-context learning (ICL). To address this challenge, we incorporate task-related public data into the ICL framework while maintaining the DP guarantee. Based on this approach, we propose a private in-context learning algorithm that effectively balances privacy protection and model utility. Through experiments, we demonstrate that our approach significantly improves the utility of private ICL with the assistance of public data. Additionally, we show that our method is robust against membership inference attacks, demonstrating empirical privacy protection. 

---
# Harmful Prompt Laundering: Jailbreaking LLMs with Abductive Styles and Symbolic Encoding 

**Authors**: Seongho Joo, Hyukhun Koh, Kyomin Jung  

**Link**: [PDF](https://arxiv.org/pdf/2509.10931)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities across diverse tasks, but their potential misuse for harmful purposes remains a significant concern. To strengthen defenses against such vulnerabilities, it is essential to investigate universal jailbreak attacks that exploit intrinsic weaknesses in the architecture and learning paradigms of LLMs. In response, we propose \textbf{H}armful \textbf{P}rompt \textbf{La}undering (HaPLa), a novel and broadly applicable jailbreaking technique that requires only black-box access to target models. HaPLa incorporates two primary strategies: 1) \textit{abductive framing}, which instructs LLMs to infer plausible intermediate steps toward harmful activities, rather than directly responding to explicit harmful queries; and 2) \textit{symbolic encoding}, a lightweight and flexible approach designed to obfuscate harmful content, given that current LLMs remain sensitive primarily to explicit harmful keywords. Experimental results show that HaPLa achieves over 95% attack success rate on GPT-series models and 70% across all targets. Further analysis with diverse symbolic encoding rules also reveals a fundamental challenge: it remains difficult to safely tune LLMs without significantly diminishing their helpfulness in responding to benign queries. 

---
# Why Bonds Fail Differently? Explainable Multimodal Learning for Multi-Class Default Prediction 

**Authors**: Yi Lu, Aifan Ling, Chaoqun Wang, Yaxin Xu  

**Link**: [PDF](https://arxiv.org/pdf/2509.10802)  

**Abstract**: In recent years, China's bond market has seen a surge in defaults amid regulatory reforms and macroeconomic volatility. Traditional machine learning models struggle to capture financial data's irregularity and temporal dependencies, while most deep learning models lack interpretability-critical for financial decision-making. To tackle these issues, we propose EMDLOT (Explainable Multimodal Deep Learning for Time-series), a novel framework for multi-class bond default prediction. EMDLOT integrates numerical time-series (financial/macroeconomic indicators) and unstructured textual data (bond prospectuses), uses Time-Aware LSTM to handle irregular sequences, and adopts soft clustering and multi-level attention to boost interpretability. Experiments on 1994 Chinese firms (2015-2024) show EMDLOT outperforms traditional (e.g., XGBoost) and deep learning (e.g., LSTM) benchmarks in recall, F1-score, and mAP, especially in identifying default/extended firms. Ablation studies validate each component's value, and attention analyses reveal economically intuitive default drivers. This work provides a practical tool and a trustworthy framework for transparent financial risk modeling. 

---
# AgentArch: A Comprehensive Benchmark to Evaluate Agent Architectures in Enterprise 

**Authors**: Tara Bogavelli, Roshnee Sharma, Hari Subramani  

**Link**: [PDF](https://arxiv.org/pdf/2509.10769)  

**Abstract**: While individual components of agentic architectures have been studied in isolation, there remains limited empirical understanding of how different design dimensions interact within complex multi-agent systems. This study aims to address these gaps by providing a comprehensive enterprise-specific benchmark evaluating 18 distinct agentic configurations across state-of-the-art large language models. We examine four critical agentic system dimensions: orchestration strategy, agent prompt implementation (ReAct versus function calling), memory architecture, and thinking tool integration. Our benchmark reveals significant model-specific architectural preferences that challenge the prevalent one-size-fits-all paradigm in agentic AI systems. It also reveals significant weaknesses in overall agentic performance on enterprise tasks with the highest scoring models achieving a maximum of only 35.3\% success on the more complex task and 70.8\% on the simpler task. We hope these findings inform the design of future agentic systems by enabling more empirically backed decisions regarding architectural components and model selection. 

---
# Understanding AI Evaluation Patterns: How Different GPT Models Assess Vision-Language Descriptions 

**Authors**: Sajjad Abdoli, Rudi Cilibrasi, Rima Al-Shikh  

**Link**: [PDF](https://arxiv.org/pdf/2509.10707)  

**Abstract**: As AI systems increasingly evaluate other AI outputs, understanding their assessment behavior becomes crucial for preventing cascading biases. This study analyzes vision-language descriptions generated by NVIDIA's Describe Anything Model and evaluated by three GPT variants (GPT-4o, GPT-4o-mini, GPT-5) to uncover distinct "evaluation personalities" the underlying assessment strategies and biases each model demonstrates. GPT-4o-mini exhibits systematic consistency with minimal variance, GPT-4o excels at error detection, while GPT-5 shows extreme conservatism with high variability. Controlled experiments using Gemini 2.5 Pro as an independent question generator validate that these personalities are inherent model properties rather than artifacts. Cross-family analysis through semantic similarity of generated questions reveals significant divergence: GPT models cluster together with high similarity while Gemini exhibits markedly different evaluation strategies. All GPT models demonstrate a consistent 2:1 bias favoring negative assessment over positive confirmation, though this pattern appears family-specific rather than universal across AI architectures. These findings suggest that evaluation competence does not scale with general capability and that robust AI assessment requires diverse architectural perspectives. 

---
# LLM in the Middle: A Systematic Review of Threats and Mitigations to Real-World LLM-based Systems 

**Authors**: Vitor Hugo Galhardo Moia, Igor Jochem Sanz, Gabriel Antonio Fontes Rebello, Rodrigo Duarte de Meneses, Briland Hitaj, Ulf Lindqvist  

**Link**: [PDF](https://arxiv.org/pdf/2509.10682)  

**Abstract**: The success and wide adoption of generative AI (GenAI), particularly large language models (LLMs), has attracted the attention of cybercriminals seeking to abuse models, steal sensitive data, or disrupt services. Moreover, providing security to LLM-based systems is a great challenge, as both traditional threats to software applications and threats targeting LLMs and their integration must be mitigated. In this survey, we shed light on security and privacy concerns of such LLM-based systems by performing a systematic review and comprehensive categorization of threats and defensive strategies considering the entire software and LLM life cycles. We analyze real-world scenarios with distinct characteristics of LLM usage, spanning from development to operation. In addition, threats are classified according to their severity level and to which scenarios they pertain, facilitating the identification of the most relevant threats. Recommended defense strategies are systematically categorized and mapped to the corresponding life cycle phase and possible attack strategies they attenuate. This work paves the way for consumers and vendors to understand and efficiently mitigate risks during integration of LLMs in their respective solutions or organizations. It also enables the research community to benefit from the discussion of open challenges and edge cases that may hinder the secure and privacy-preserving adoption of LLM-based systems. 

---
# Smart Trial: Evaluating the Use of Large Language Models for Recruiting Clinical Trial Participants via Social Media 

**Authors**: Xiaofan Zhou, Zisu Wang, Janice Krieger, Mohan Zalake, Lu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2509.10584)  

**Abstract**: Clinical trials (CT) are essential for advancing medical research and treatment, yet efficiently recruiting eligible participants -- each of whom must meet complex eligibility criteria -- remains a significant challenge. Traditional recruitment approaches, such as advertisements or electronic health record screening within hospitals, are often time-consuming and geographically constrained. This work addresses the recruitment challenge by leveraging the vast amount of health-related information individuals share on social media platforms. With the emergence of powerful large language models (LLMs) capable of sophisticated text understanding, we pose the central research question: Can LLM-driven tools facilitate CT recruitment by identifying potential participants through their engagement on social media? To investigate this question, we introduce TRIALQA, a novel dataset comprising two social media collections from the subreddits on colon cancer and prostate cancer. Using eligibility criteria from public real-world CTs, experienced annotators are hired to annotate TRIALQA to indicate (1) whether a social media user meets a given eligibility criterion and (2) the user's stated reasons for interest in participating in CT. We benchmark seven widely used LLMs on these two prediction tasks, employing six distinct training and inference strategies. Our extensive experiments reveal that, while LLMs show considerable promise, they still face challenges in performing the complex, multi-hop reasoning needed to accurately assess eligibility criteria. 

---
# DualAlign: Generating Clinically Grounded Synthetic Data 

**Authors**: Rumeng Li, Xun Wang, Hong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2509.10538)  

**Abstract**: Synthetic clinical data are increasingly important for advancing AI in healthcare, given strict privacy constraints on real-world EHRs, limited availability of annotated rare-condition data, and systemic biases in observational datasets. While large language models (LLMs) can generate fluent clinical text, producing synthetic data that is both realistic and clinically meaningful remains challenging. We introduce DualAlign, a framework that enhances statistical fidelity and clinical plausibility through dual alignment: (1) statistical alignment, which conditions generation on patient demographics and risk factors; and (2) semantic alignment, which incorporates real-world symptom trajectories to guide content generation. Using Alzheimer's disease (AD) as a case study, DualAlign produces context-grounded symptom-level sentences that better reflect real-world clinical documentation. Fine-tuning an LLaMA 3.1-8B model with a combination of DualAlign-generated and human-annotated data yields substantial performance gains over models trained on gold data alone or unguided synthetic baselines. While DualAlign does not fully capture longitudinal complexity, it offers a practical approach for generating clinically grounded, privacy-preserving synthetic data to support low-resource clinical text analysis. 

---
# Decoupling the "What" and "Where" With Polar Coordinate Positional Embeddings 

**Authors**: Anand Gopalakrishnan, Robert Csordás, Jürgen Schmidhuber, Michael C. Mozer  

**Link**: [PDF](https://arxiv.org/pdf/2509.10534)  

**Abstract**: The attention mechanism in a Transformer architecture matches key to query based on both content -- the what -- and position in a sequence -- the where. We present an analysis indicating that what and where are entangled in the popular RoPE rotary position embedding. This entanglement can impair performance particularly when decisions require independent matches on these two factors. We propose an improvement to RoPE, which we call Polar Coordinate Position Embeddings or PoPE, that eliminates the what-where confound. PoPE is far superior on a diagnostic task requiring indexing solely by position or by content. On autoregressive sequence modeling in music, genomic, and natural language domains, Transformers using PoPE as the positional encoding scheme outperform baselines using RoPE with respect to evaluation loss (perplexity) and downstream task performance. On language modeling, these gains persist across model scale, from 124M to 774M parameters. Crucially, PoPE shows strong zero-shot length extrapolation capabilities, whereas RoPE's performance degrades significantly on longer sequences at test time without fine tuning or the use of position-interpolation methods. 

---
# Real-Time RAG for the Identification of Supply Chain Vulnerabilities 

**Authors**: Jesse Ponnock, Grace Kenneally, Michael Robert Briggs, Elinor Yeo, Tyrone Patterson III, Nicholas Kinberg, Matthew Kalinowski, David Hechtman  

**Link**: [PDF](https://arxiv.org/pdf/2509.10469)  

**Abstract**: New technologies in generative AI can enable deeper analysis into our nation's supply chains but truly informative insights require the continual updating and aggregation of massive data in a timely manner. Large Language Models (LLMs) offer unprecedented analytical opportunities however, their knowledge base is constrained to the models' last training date, rendering these capabilities unusable for organizations whose mission impacts rely on emerging and timely information. This research proposes an innovative approach to supply chain analysis by integrating emerging Retrieval-Augmented Generation (RAG) preprocessing and retrieval techniques with advanced web-scraping technologies. Our method aims to reduce latency in incorporating new information into an augmented-LLM, enabling timely analysis of supply chain disruptors. Through experimentation, this study evaluates the combinatorial effects of these techniques towards timeliness and quality trade-offs. Our results suggest that in applying RAG systems to supply chain analysis, fine-tuning the embedding retrieval model consistently provides the most significant performance gains, underscoring the critical importance of retrieval quality. Adaptive iterative retrieval, which dynamically adjusts retrieval depth based on context, further enhances performance, especially on complex sup- ply chain queries. Conversely, fine-tuning the LLM yields limited improvements and higher resource costs, while techniques such as downward query abstraction significantly outperforms upward abstraction in practice. 

---
# Learning Decomposed Contextual Token Representations from Pretrained and Collaborative Signals for Generative Recommendation 

**Authors**: Yifan Liu, Yaokun Liu, Zelin Li, Zhenrui Yue, Gyuseok Lee, Ruichen Yao, Yang Zhang, Dong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.10468)  

**Abstract**: Recent advances in generative recommenders adopt a two-stage paradigm: items are first tokenized into semantic IDs using a pretrained tokenizer, and then large language models (LLMs) are trained to generate the next item via sequence-to-sequence modeling. However, these two stages are optimized for different objectives: semantic reconstruction during tokenizer pretraining versus user interaction modeling during recommender training. This objective misalignment leads to two key limitations: (i) suboptimal static tokenization, where fixed token assignments fail to reflect diverse usage contexts; and (ii) discarded pretrained semantics, where pretrained knowledge - typically from language model embeddings - is overwritten during recommender training on user interactions. To address these limitations, we propose to learn DEcomposed COntextual Token Representations (DECOR), a unified framework that preserves pretrained semantics while enhancing the adaptability of token embeddings. DECOR introduces contextualized token composition to refine token embeddings based on user interaction context, and decomposed embedding fusion that integrates pretrained codebook embeddings with newly learned collaborative embeddings. Experiments on three real-world datasets demonstrate that DECOR consistently outperforms state-of-the-art baselines in recommendation performance. Our code will be made available upon publication. 

---
# DSRAG: A Domain-Specific Retrieval Framework Based on Document-derived Multimodal Knowledge Graph 

**Authors**: Mengzheng Yang, Yanfei Ren, David Osei Opoku, Ruochang Li, Peng Ren, Chunxiao Xing  

**Link**: [PDF](https://arxiv.org/pdf/2509.10467)  

**Abstract**: Current general-purpose large language models (LLMs) commonly exhibit knowledge hallucination and insufficient domain-specific adaptability in domain-specific tasks, limiting their effectiveness in specialized question answering scenarios. Retrieval-augmented generation (RAG) effectively tackles these challenges by integrating external knowledge to enhance accuracy and relevance. However, traditional RAG still faces limitations in domain knowledge accuracy and context this http URL enhance domain-specific question answering performance, this work focuses on a graph-based RAG framework, emphasizing the critical role of knowledge graph quality during the generation process. We propose DSRAG (Domain-Specific RAG), a multimodal knowledge graph-driven retrieval-augmented generation framework designed for domain-specific applications. Our approach leverages domain-specific documents as the primary knowledge source, integrating heterogeneous information such as text, images, and tables to construct a multimodal knowledge graph covering both conceptual and instance layers. Building on this foundation, we introduce semantic pruning and structured subgraph retrieval mechanisms, combining knowledge graph context and vector retrieval results to guide the language model towards producing more reliable responses. Evaluations using the Langfuse multidimensional scoring mechanism show that our method excels in domain-specific question answering, validating the efficacy of integrating multimodal knowledge graphs with retrieval-augmented generation. 

---
