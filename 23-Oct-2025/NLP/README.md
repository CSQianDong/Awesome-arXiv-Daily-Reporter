# Hubble: a Model Suite to Advance the Study of LLM Memorization 

**Authors**: Johnny Tian-Zheng Wei, Ameya Godbole, Mohammad Aflah Khan, Ryan Wang, Xiaoyuan Zhu, James Flemings, Nitya Kashyap, Krishna P. Gummadi, Willie Neiswanger, Robin Jia  

**Link**: [PDF](https://arxiv.org/pdf/2510.19811)  

**Abstract**: We present Hubble, a suite of fully open-source large language models (LLMs) for the scientific study of LLM memorization. Hubble models come in standard and perturbed variants: standard models are pretrained on a large English corpus, and perturbed models are trained in the same way but with controlled insertion of text (e.g., book passages, biographies, and test sets) designed to emulate key memorization risks. Our core release includes 8 models -- standard and perturbed models with 1B or 8B parameters, pretrained on 100B or 500B tokens -- establishing that memorization risks are determined by the frequency of sensitive data relative to size of the training corpus (i.e., a password appearing once in a smaller corpus is memorized better than the same password in a larger corpus). Our release also includes 6 perturbed models with text inserted at different pretraining phases, showing that sensitive data without continued exposure can be forgotten. These findings suggest two best practices for addressing memorization risks: to dilute sensitive data by increasing the size of the training corpus, and to order sensitive data to appear earlier in training. Beyond these general empirical findings, Hubble enables a broad range of memorization research; for example, analyzing the biographies reveals how readily different types of private information are memorized. We also demonstrate that the randomized insertions in Hubble make it an ideal testbed for membership inference and machine unlearning, and invite the community to further explore, benchmark, and build upon our work. 

---
# Scaf-GRPO: Scaffolded Group Relative Policy Optimization for Enhancing LLM Reasoning 

**Authors**: Xichen Zhang, Sitong Wu, Yinghao Zhu, Haoru Tan, Shaozuo Yu, Ziyi He, Jiaya Jia  

**Link**: [PDF](https://arxiv.org/pdf/2510.19807)  

**Abstract**: Reinforcement learning from verifiable rewards has emerged as a powerful technique for enhancing the complex reasoning abilities of Large Language Models (LLMs). However, these methods are fundamentally constrained by the ''learning cliff'' phenomenon: when faced with problems far beyond their current capabilities, models consistently fail, yielding a persistent zero-reward signal. In policy optimization algorithms like GRPO, this collapses the advantage calculation to zero, rendering these difficult problems invisible to the learning gradient and stalling progress. To overcome this, we introduce Scaf-GRPO (Scaffolded Group Relative Policy Optimization), a progressive training framework that strategically provides minimal guidance only when a model's independent learning has plateaued. The framework first diagnoses learning stagnation and then intervenes by injecting tiered in-prompt hints, ranging from abstract concepts to concrete steps, enabling the model to construct a valid solution by itself. Extensive experiments on challenging mathematics benchmarks demonstrate Scaf-GRPO's effectiveness, boosting the pass@1 score of the Qwen2.5-Math-7B model on the AIME24 benchmark by a relative 44.3% over a vanilla GRPO baseline. This result demonstrates our framework provides a robust and effective methodology for unlocking a model's ability to solve problems previously beyond its reach, a critical step towards extending the frontier of autonomous reasoning in LLM. 

---
# The Art of Asking: Multilingual Prompt Optimization for Synthetic Data 

**Authors**: David Mora, Viraat Aryabumi, Wei-Yin Ko, Sara Hooker, Julia Kreutzer, Marzieh Fadaee  

**Link**: [PDF](https://arxiv.org/pdf/2510.19806)  

**Abstract**: Synthetic data has become a cornerstone for scaling large language models, yet its multilingual use remains bottlenecked by translation-based prompts. This strategy inherits English-centric framing and style and neglects cultural dimensions, ultimately constraining model generalization. We argue that the overlooked prompt space-the very inputs that define training distributions-offers a more powerful lever for improving multilingual performance. We introduce a lightweight framework for prompt-space optimization, where translated prompts are systematically transformed for Naturalness, Cultural Adaptation, and Difficulty Enhancement. Using an off-the-shelf multilingual LLM, we apply these transformations to prompts for 12 languages spanning 7 families. Under identical data conditions, our approaches achieve substantial and consistent downstream improvements over the translation-only baseline: +4.7% on Global-MMLU accuracy, +2.4% on Flores XCometXL and +35.3% wins in preferences on mArenaHard. We establish prompt-space optimization as a simple yet powerful paradigm for building multilingual LLMs that are more robust, culturally grounded, and globally capable. 

---
# ToolDreamer: Instilling LLM Reasoning Into Tool Retrievers 

**Authors**: Saptarshi Sengupta, Zhengyu Zhou, Jun Araki, Xingbo Wang, Bingqing Wang, Suhang Wang, Zhe Feng  

**Link**: [PDF](https://arxiv.org/pdf/2510.19791)  

**Abstract**: Tool calling has become increasingly popular for Large Language Models (LLMs). However, for large tool sets, the resulting tokens would exceed the LLM's context window limit, making it impossible to include every tool. Hence, an external retriever is used to provide LLMs with the most relevant tools for a query. Existing retrieval models rank tools based on the similarity between a user query and a tool description (TD). This leads to suboptimal retrieval as user requests are often poorly aligned with the language of TD. To remedy the issue, we propose ToolDreamer, a framework to condition retriever models to fetch tools based on hypothetical (synthetic) TD generated using an LLM, i.e., description of tools that the LLM feels will be potentially useful for the query. The framework enables a more natural alignment between queries and tools within the language space of TD's. We apply ToolDreamer on the ToolRet dataset and show that our method improves the performance of sparse and dense retrievers with and without training, thus showcasing its flexibility. Through our proposed framework, our aim is to offload a portion of the reasoning burden to the retriever so that the LLM may effectively handle a large collection of tools without inundating its context window. 

---
# Adapting Multilingual Models to Code-Mixed Tasks via Model Merging 

**Authors**: Prashant Kodali, Vaishnavi Shivkumar, Swarang Joshi, Monojit Choudhary, Ponnurangam Kumaraguru, Manish Shrivastava  

**Link**: [PDF](https://arxiv.org/pdf/2510.19782)  

**Abstract**: We study model merging as a practical alternative to conventional adaptation strategies for code-mixed NLP. Starting from a multilingual base model, we: (i) perform continued pre-training (CPT) on unlabeled code-mixed text to obtain an adapted checkpoint, (ii) merge checkpoint with the base model, and (iii) fine-tune (FT) on the downstream task data. We evaluate our approach for sentence classification (sentiment and hate speech) task in English-Hindi (En-Hi) and English-Spanish (En-Es) using XLM-R and Llama-3.2-1B models. Our results show that merged models consistently outperform full fine-tuning and CPT->FT. We observe gains of 2--5 points in F1 over full fine-tuning and ~1-2 points over CPT->FT, indicating that unlabeled data is leveraged more effectively via merging than via CPT alone. Zero-/few-shot prompting with larger LLMs (e.g., Llama-3.3-70B) lags behind fine-tuned and merged checkpoints, underscoring limits of in-context learning for code-mixed inputs. We further test cross-pair transfer by training on En-Hi and evaluating on En-Ta and En-Ml: merged checkpoints transfer more strongly than monolingual-English baselines (e.g., TV/TIES variants reaching 0.65-0.68 F1 vs 0.61-0.63 for full fine-tuning), suggesting that code-mixed knowledge is a more reliable substrate for low-resource pairs. We conclude with adaptation recipes matched to common data regimes (labeled only; labeled+unlabeled; transfer-only) and discuss limitations and scaling considerations for broader tasks and larger models. 

---
# AdaSPEC: Selective Knowledge Distillation for Efficient Speculative Decoders 

**Authors**: Yuezhou Hu, Jiaxin Guo, Xinyu Feng, Tuo Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2510.19779)  

**Abstract**: Speculative Decoding (SD) accelerates large language model inference by employing a small draft model to generate predictions, which are then verified by a larger target model. The effectiveness of SD hinges on the alignment between these models, which is typically enhanced by Knowledge Distillation (KD). However, conventional KD methods aim to minimize the KL divergence between the draft and target models across all tokens, a goal that is misaligned with the true objective of SD, which is to maximize token acceptance rate. Therefore, draft models often struggle to fully assimilate the target model's knowledge due to capacity constraints, leading to suboptimal performance. To address this challenge, we propose AdaSPEC, a novel method that incorporates selective token filtering into the KD process. AdaSPEC utilizes a reference model to identify and filter out difficult-to-fit tokens, enabling the distillation of a draft model that better aligns with the target model on simpler tokens. This approach improves the overall token acceptance rate without compromising generation quality. We evaluate AdaSPEC across diverse tasks, including arithmetic reasoning, instruction-following, coding, and summarization, using model configurations of 31M/1.4B and 350M/2.7B parameters. Our results demonstrate that AdaSPEC consistently outperforms the state-of-the-art DistillSpec method, achieving higher acceptance rates across all tasks (up to 15\%). The code is publicly available at this https URL. 

---
# SmartSwitch: Advancing LLM Reasoning by Overcoming Underthinking via Promoting Deeper Thought Exploration 

**Authors**: Xichen Zhang, Sitong Wu, Haoru Tan, Shaozuo Yu, Yinghao Zhu, Ziyi He, Jiaya Jia  

**Link**: [PDF](https://arxiv.org/pdf/2510.19767)  

**Abstract**: The long chain-of-thought (LongCoT) capability is central to the recent breakthroughs achieved by large language models in complex reasoning tasks. However, the accompanying issue of ''underthinking'', where models exhibit shallow reasoning by frequently switching thoughts without sufficient exploration, limits both performance and token efficiency. To address this problem, we propose a simple yet effective reasoning strategy: the SmartSwitch inference framework. This framework can be easily integrated into any large language model as a plug-and-play solution, continuously monitoring the model's reasoning process to detect underthinking and guide it toward deeper exploration of promising but overlooked thoughts. Specifically, the perception module identifies points where thoughts switch and evaluates the potential of the preceding thought using an off-the-shelf process reward model (PRM). If a high-potential thought is found to be prematurely abandoned, the intervention module interrupts the ongoing inference, backtracks to the point before the switch, and inserts a "deepening prompt" to encourage further exploration along that promising path. Extensive experiments on challenging mathematical reasoning benchmarks demonstrate that our method significantly enhances the performance of various large language models of different sizes. 

---
# Zhyper: Factorized Hypernetworks for Conditioned LLM Fine-Tuning 

**Authors**: M. H. I. Abdalla, Zhipin Wang, Christian Frey, Steffen Eger, Josif Grabocka  

**Link**: [PDF](https://arxiv.org/pdf/2510.19733)  

**Abstract**: Large Language Model (LLM) conditioning refers to instructing an LLM to generate content in accordance with the norms and values of a specific culture, beliefs of a particular political orientation, or any desired text-specified semantic conditioning. Unfortunately, prompt engineering does not ensure that LLMs behave in accordance with a desired conditioning due to the inductive bias of the pre-training and alignment datasets. Prior works have focused on fine-tuning LLMs by directly conditioning the LoRA weights; however, such methods introduce a large number of parameters. As a remedy, we propose Zhyper, a parameter-efficient factorized hypernetwork framework that generates context-aware LoRA adapters from textual descriptions. Experiments on multiple benchmarks show that Zhyper achieves competitive performance with up to 26x fewer parameters than the state-of-the-art baselines. Furthermore, we extend Zhyper to cultural alignment, demonstrating improved generalization to out-of-domain settings and a better capturing of fine-grained contextual values. 

---
# From Answers to Guidance: A Proactive Dialogue System for Legal Documents 

**Authors**: Ashish Chouhan, Michael Gertz  

**Link**: [PDF](https://arxiv.org/pdf/2510.19723)  

**Abstract**: The accessibility of legal information remains a constant challenge, particularly for laypersons seeking to understand and apply complex institutional texts. While the European Union provides open access to legislation, parliamentary responses, and regulatory documents, these resources can be challenging for laypeople to explore. In this paper, we introduce EUDial, a proactive multi-turn dialogue dataset constructed from 204 blogs curated by the Citizens' Enquiries Unit (AskEP) of the European Parliamentary Research Service. EUDial contains 880 dialogue turns (averaging 4.3 turns per dialogue), where each dialogue includes initial questions, structured answers, and follow-up questions. Beyond dataset construction, we propose the LexGuide framework that leverages retrieval-augmented generation with hierarchical topic organization to structure dialogue progression, ensuring both comprehensive coverage of legal aspects and coherence across conversational turns. The results demonstrate that proactive, structured navigation closes the gap between the availability of legal information and citizen comprehension, establishing EUDial and LexGuide as practical resources for advancing proactive legal dialogue systems. 

---
# Do Prompts Reshape Representations? An Empirical Study of Prompting Effects on Embeddings 

**Authors**: Cesar Gonzalez-Gutierrez, Dirk Hovy  

**Link**: [PDF](https://arxiv.org/pdf/2510.19694)  

**Abstract**: Prompting is a common approach for leveraging LMs in zero-shot settings. However, the underlying mechanisms that enable LMs to perform diverse tasks without task-specific supervision remain poorly understood. Studying the relationship between prompting and the quality of internal representations can shed light on how pre-trained embeddings may support in-context task solving. In this empirical study, we conduct a series of probing experiments on prompt embeddings, analyzing various combinations of prompt templates for zero-shot classification. Our findings show that while prompting affects the quality of representations, these changes do not consistently correlate with the relevance of the prompts to the target task. This result challenges the assumption that more relevant prompts necessarily lead to better representations. We further analyze potential factors that may contribute to this unexpected behavior. 

---
# Are Large Language Models Sensitive to the Motives Behind Communication? 

**Authors**: Addison J. Wu, Ryan Liu, Kerem Oktar, Theodore R. Sumers, Thomas L. Griffiths  

**Link**: [PDF](https://arxiv.org/pdf/2510.19687)  

**Abstract**: Human communication is motivated: people speak, write, and create content with a particular communicative intent in mind. As a result, information that large language models (LLMs) and AI agents process is inherently framed by humans' intentions and incentives. People are adept at navigating such nuanced information: we routinely identify benevolent or self-serving motives in order to decide what statements to trust. For LLMs to be effective in the real world, they too must critically evaluate content by factoring in the motivations of the source -- for instance, weighing the credibility of claims made in a sales pitch. In this paper, we undertake a comprehensive study of whether LLMs have this capacity for motivational vigilance. We first employ controlled experiments from cognitive science to verify that LLMs' behavior is consistent with rational models of learning from motivated testimony, and find they successfully discount information from biased sources in a human-like manner. We then extend our evaluation to sponsored online adverts, a more naturalistic reflection of LLM agents' information ecosystems. In these settings, we find that LLMs' inferences do not track the rational models' predictions nearly as closely -- partly due to additional information that distracts them from vigilance-relevant considerations. However, a simple steering intervention that boosts the salience of intentions and incentives substantially increases the correspondence between LLMs and the rational model. These results suggest that LLMs possess a basic sensitivity to the motivations of others, but generalizing to novel real-world settings will require further improvements to these models. 

---
# CoSense-LLM: Semantics at the Edge with Cost- and Uncertainty-Aware Cloud-Edge Cooperation 

**Authors**: Hasan Akgul, Mari Eplik, Javier Rojas, Aina Binti Abdullah, Pieter van der Merwe  

**Link**: [PDF](https://arxiv.org/pdf/2510.19670)  

**Abstract**: We present CoSense-LLM, an edge-first framework that turns continuous multimodal sensor streams (for example Wi-Fi CSI, IMU, audio, RFID, and lightweight vision) into compact, verifiable semantic tokens and coordinates with large language models under explicit latency, energy, bandwidth, and privacy constraints. CoSense-LLM has four parts: (i) SenseFusion, a lightweight encoder that aligns sensor embeddings with language and compresses them into short discrete code sequences; (ii) Edge-RAG, a local hybrid retrieval layer that grounds generation in site specific policies and notes; (iii) PromptRouter, a cost and uncertainty aware policy that selects edge only generation, edge plus retrieval, or compact cloud escalation; and (iv) Secure Execution, an auditable redaction path that enforces data minimization so raw waveforms never leave the device. The system works with modern serving optimizations, including paged or streaming KV caches, FlashAttention style kernels, speculative decoding, and quantized LoRA adapters, and supports on device personalization and federated updates under non IID drift. Across home, office, and clinic deployments, CoSense-LLM delivers grounded explanations while meeting tight service level objectives: it sustains sub second (p95) end to end latency on edge dominant paths, reduces inter tier token and bandwidth costs by preferring local retrieval grounded responses, and preserves privacy by transmitting only discrete codes and redacted metadata. Ablations show that Edge-RAG improves factual consistency and reduces contradictions, calibrated uncertainty enables selective abstention and controlled escalations, and KV plus decoding accelerators lower energy per decision. The results support an edge first design that treats semantics, privacy, and predictable latency as co equal goals for large model deployments in interference prone environments. 

---
# DiffAdapt: Difficulty-Adaptive Reasoning for Token-Efficient LLM Inference 

**Authors**: Xiang Liu, Xuming Hu, Xiaowen Chu, Eunsol Choi  

**Link**: [PDF](https://arxiv.org/pdf/2510.19669)  

**Abstract**: Recent reasoning Large Language Models (LLMs) demonstrate remarkable problem-solving abilities but often generate long thinking traces whose utility is unclear. Our work aims to improve their efficiency, enabling them to reach high performance without overthinking. First, we analyze the entropy of token probabilities in reasoning traces. Across three models, we observe a consistent U-shaped entropy pattern: high entropy on easy problems despite high accuracy, low entropy on problems with medium difficulty, and high entropy on hard problems reflecting uncertainty. Specifically, we notice 22--25\% entropy reduction from easy to medium difficulty regions, suggesting an {overthinking} phenomenon on easy instances. Building on these insights, we introduce \textbf{DiffAdapt}, a lightweight framework that selects Easy/Normal/Hard inference strategies per question based on their difficulty and reasoning trace entropy. Each inference strategy consists of a fixed prompt, temperature and maximum token length. In contrast to existing efficiency optimization methods, our approach does not fine-tune base LLM but a small probe that classifies LLM's final hidden state, allowing inexpensive adaptation. We comprehensively evaluate our method on five models and eight benchmarks. Our method achieves comparable or improved accuracy while reducing token usage by up to 22.4\%, establishing a practical path toward compute-efficient reasoning. 

---
# Unraveling Emotions with Pre-Trained Models 

**Authors**: Alejandro Pajón-Sanmartín, Francisco De Arriba-Pérez, Silvia García-Méndez, Fátima Leal, Benedita Malheiro, Juan Carlos Burguillo-Rial  

**Link**: [PDF](https://arxiv.org/pdf/2510.19668)  

**Abstract**: Transformer models have significantly advanced the field of emotion recognition. However, there are still open challenges when exploring open-ended queries for Large Language Models (LLMs). Although current models offer good results, automatic emotion analysis in open texts presents significant challenges, such as contextual ambiguity, linguistic variability, and difficulty interpreting complex emotional expressions. These limitations make the direct application of generalist models difficult. Accordingly, this work compares the effectiveness of fine-tuning and prompt engineering in emotion detection in three distinct scenarios: (i) performance of fine-tuned pre-trained models and general-purpose LLMs using simple prompts; (ii) effectiveness of different emotion prompt designs with LLMs; and (iii) impact of emotion grouping techniques on these models. Experimental tests attain metrics above 70% with a fine-tuned pre-trained model for emotion recognition. Moreover, the findings highlight that LLMs require structured prompt engineering and emotion grouping to enhance their performance. These advancements improve sentiment analysis, human-computer interaction, and understanding of user behavior across various domains. 

---
# LLavaCode: Compressed Code Representations for Retrieval-Augmented Code Generation 

**Authors**: Daria Cherniuk, Nikita Sukhorukov, Nikita Sushko, Daniil Gusak, Danil Sivtsov, Elena Tutubalina, Evgeny Frolov  

**Link**: [PDF](https://arxiv.org/pdf/2510.19644)  

**Abstract**: Retrieval-augmented generation has emerged as one of the most effective approaches for code completion, particularly when context from a surrounding repository is essential. However, incorporating context significantly extends sequence length, leading to slower inference - a critical limitation for interactive settings such as IDEs. In this work, we introduce LlavaCode, a framework that compresses code into compact, semantically rich representations interpretable by code LLM, enhancing generation quality while reducing the retrieved context to only a few compressed single-token vectors. Using a small projector module we can significantly increase the EM and ES metrics of coding model with negligible latency increase. Our experiments demonstrate that compressed context enables 20-38% reduction in Time-to-First-Token (TTFT) on line completion tasks compared to full-RAG pipelines. 

---
# Style Attack Disguise: When Fonts Become a Camouflage for Adversarial Intent 

**Authors**: Yangshijie Zhang, Xinda Wang, Jialin Liu, Wenqiang Wang, Zhicong Ma, Xingxing Jia  

**Link**: [PDF](https://arxiv.org/pdf/2510.19641)  

**Abstract**: With social media growth, users employ stylistic fonts and font-like emoji to express individuality, creating visually appealing text that remains human-readable. However, these fonts introduce hidden vulnerabilities in NLP models: while humans easily read stylistic text, models process these characters as distinct tokens, causing interference. We identify this human-model perception gap and propose a style-based attack, Style Attack Disguise (SAD). We design two sizes: light for query efficiency and strong for superior attack performance. Experiments on sentiment classification and machine translation across traditional models, LLMs, and commercial services demonstrate SAD's strong attack performance. We also show SAD's potential threats to multimodal tasks including text-to-image and text-to-speech generation. 

---
# CrossNews-UA: A Cross-lingual News Semantic Similarity Benchmark for Ukrainian, Polish, Russian, and English 

**Authors**: Daryna Dementieva, Evgeniya Sukhodolskaya, Alexander Fraser  

**Link**: [PDF](https://arxiv.org/pdf/2510.19628)  

**Abstract**: In the era of social networks and rapid misinformation spread, news analysis remains a critical task. Detecting fake news across multiple languages, particularly beyond English, poses significant challenges. Cross-lingual news comparison offers a promising approach to verify information by leveraging external sources in different languages (Chen and Shu, 2024). However, existing datasets for cross-lingual news analysis (Chen et al., 2022a) were manually curated by journalists and experts, limiting their scalability and adaptability to new languages. In this work, we address this gap by introducing a scalable, explainable crowdsourcing pipeline for cross-lingual news similarity assessment. Using this pipeline, we collected a novel dataset CrossNews-UA of news pairs in Ukrainian as a central language with linguistically and contextually relevant languages-Polish, Russian, and English. Each news pair is annotated for semantic similarity with detailed justifications based on the 4W criteria (Who, What, Where, When). We further tested a range of models, from traditional bag-of-words, Transformer-based architectures to large language models (LLMs). Our results highlight the challenges in multilingual news analysis and offer insights into models performance. 

---
# PBBQ: A Persian Bias Benchmark Dataset Curated with Human-AI Collaboration for Large Language Models 

**Authors**: Farhan Farsi, Shayan Bali, Fatemeh Valeh, Parsa Ghofrani, Alireza Pakniat, Kian Kashfipour, Amir H. Payberah  

**Link**: [PDF](https://arxiv.org/pdf/2510.19616)  

**Abstract**: With the increasing adoption of large language models (LLMs), ensuring their alignment with social norms has become a critical concern. While prior research has examined bias detection in various languages, there remains a significant gap in resources addressing social biases within Persian cultural contexts. In this work, we introduce PBBQ, a comprehensive benchmark dataset designed to evaluate social biases in Persian LLMs. Our benchmark, which encompasses 16 cultural categories, was developed through questionnaires completed by 250 diverse individuals across multiple demographics, in close collaboration with social science experts to ensure its validity. The resulting PBBQ dataset contains over 37,000 carefully curated questions, providing a foundation for the evaluation and mitigation of bias in Persian language models. We benchmark several open-source LLMs, a closed-source model, and Persian-specific fine-tuned models on PBBQ. Our findings reveal that current LLMs exhibit significant social biases across Persian culture. Additionally, by comparing model outputs to human responses, we observe that LLMs often replicate human bias patterns, highlighting the complex interplay between learned representations and cultural this http URL acceptance of the paper, our PBBQ dataset will be publicly available for use in future work. Content warning: This paper contains unsafe content. 

---
# Detecting Latin in Historical Books with Large Language Models: A Multimodal Benchmark 

**Authors**: Yu Wu, Ke Shu, Jonas Fischer, Lidia Pivovarova, David Rosson, Eetu Mäkelä, Mikko Tolonen  

**Link**: [PDF](https://arxiv.org/pdf/2510.19585)  

**Abstract**: This paper presents a novel task of extracting Latin fragments from mixed-language historical documents with varied layouts. We benchmark and evaluate the performance of large foundation models against a multimodal dataset of 724 annotated pages. The results demonstrate that reliable Latin detection with contemporary models is achievable. Our study provides the first comprehensive analysis of these models' capabilities and limits for this task. 

---
# Conditions for Catastrophic Forgetting in Multilingual Translation 

**Authors**: Danni Liu, Jan Niehues  

**Link**: [PDF](https://arxiv.org/pdf/2510.19546)  

**Abstract**: Fine-tuning multilingual foundation models on specific languages often induces catastrophic forgetting, degrading performance on languages unseen in fine-tuning. While this phenomenon is widely-documented, the literature presents fragmented results about when forgetting occurs. To address this ambiguity, we conduct a systematic empirical study using machine translation as a testbed to identify the conditions that trigger catastrophic forgetting in multilingual fine-tuning. Through controlled experiments across different model architectures, data scales, and fine-tuning approaches, we reveal that the relative scale between model and data size is a primary determinant of forgetting. Moreover, we demonstrate that a model's instruction-following ability is more critical for retaining multilingual knowledge than its architecture. Contrary to assumptions, parameter-efficient fine-tuning offers no clear advantage over full fine-tuning in mitigating forgetting. Lastly, we show that cross-lingual alignment can mitigate forgetting while also facilitating positive transfer to unseen target languages. 

---
# Which Evaluation for Which Model? A Taxonomy for Speech Model Assessment 

**Authors**: Maureen de Seyssel, Eeshan Gunesh Dhekane  

**Link**: [PDF](https://arxiv.org/pdf/2510.19509)  

**Abstract**: Speech foundation models have recently achieved remarkable capabilities across a wide range of tasks. However, their evaluation remains disjointed across tasks and model types. Different models excel at distinct aspects of speech processing and thus require different evaluation protocols. This paper proposes a unified taxonomy that addresses the question: Which evaluation is appropriate for which model? The taxonomy defines three orthogonal axes: the \textbf{evaluation aspect} being measured, the model capabilities required to attempt the task, and the task or protocol requirements needed to perform it. We classify a broad set of existing evaluations and benchmarks along these axes, spanning areas such as representation learning, speech generation, and interactive dialogue. By mapping each evaluation to the capabilities a model exposes (e.g., speech generation, real-time processing) and to its methodological demands (e.g., fine-tuning data, human judgment), the taxonomy provides a principled framework for aligning models with suitable evaluation methods. It also reveals systematic gaps, such as limited coverage of prosody, interaction, or reasoning, that highlight priorities for future benchmark design. Overall, this work offers a conceptual foundation and practical guide for selecting, interpreting, and extending evaluations of speech models. 

---
# Lookahead Routing for Large Language Models 

**Authors**: Canbin Huang, Tianyuan Shi, Yuhua Zhu, Ruijun Chen, Xiaojun Quan  

**Link**: [PDF](https://arxiv.org/pdf/2510.19506)  

**Abstract**: Large language model (LLM) routers improve the efficiency of multi-model systems by directing each query to the most appropriate model while leveraging the diverse strengths of heterogeneous LLMs. Most existing approaches frame routing as a classification problem based solely on the input query. While this reduces overhead by avoiding inference across all models, it overlooks valuable information that could be gleaned from potential outputs and fails to capture implicit intent or contextual nuances that often emerge only during response generation. These limitations can result in suboptimal routing decisions, particularly for complex or ambiguous queries that require deeper semantic understanding. To address this challenge, we propose Lookahead, a routing framework that "foresees" potential model outputs by predicting their latent representations and uses these predictions to guide model selection, thus enabling more informed routing without full inference. Within this framework, we implement two approaches based on causal and masked language models. Empirical evaluations across seven public benchmarks - spanning instruction following, mathematical reasoning, and code generation - show that Lookahead consistently outperforms existing routing baselines, achieving an average performance gain of 7.7% over the state-of-the-art. Our code is available at this https URL. 

---
# What is the Best Sequence Length for BABYLM? 

**Authors**: Suchir Salhan, Richard Diehl Martinez, Zébulon Goriely, Paula Buttery  

**Link**: [PDF](https://arxiv.org/pdf/2510.19493)  

**Abstract**: Transformer language models typically operate with a fixed-length context window, which has grown in step with large-scale pretraining datasets. In the BabyLM Challenge, however, many past submissions have defaulted to using much shorter sequence lengths. We examine the impact of sequence length on BabyLM pretraining, to answer the simple question: what sequence length should we be using when training Baby LMs? Using 100M-word training data and fixed compute budgets, we compare 125M-parameter Mamba and OPT models, finding that although longer is often better, the optimal length depends on both task and architecture. Shorter sequences are sufficient for grammatical generalization tasks whereas longer contexts benefit morphological analogical reasoning tasks. 

---
# Machine Text Detectors are Membership Inference Attacks 

**Authors**: Ryuto Koike, Liam Dugan, Masahiro Kaneko, Chris Callison-Burch, Naoaki Okazaki  

**Link**: [PDF](https://arxiv.org/pdf/2510.19492)  

**Abstract**: Although membership inference attacks (MIAs) and machine-generated text detection target different goals, identifying training samples and synthetic texts, their methods often exploit similar signals based on a language model's probability distribution. Despite this shared methodological foundation, the two tasks have been independently studied, which may lead to conclusions that overlook stronger methods and valuable insights developed in the other task. In this work, we theoretically and empirically investigate the transferability, i.e., how well a method originally developed for one task performs on the other, between MIAs and machine text detection. For our theoretical contribution, we prove that the metric that achieves the asymptotically highest performance on both tasks is the same. We unify a large proportion of the existing literature in the context of this optimal metric and hypothesize that the accuracy with which a given method approximates this metric is directly correlated with its transferability. Our large-scale empirical experiments, including 7 state-of-the-art MIA methods and 5 state-of-the-art machine text detectors across 13 domains and 10 generators, demonstrate very strong rank correlation (rho > 0.6) in cross-task performance. We notably find that Binoculars, originally designed for machine text detection, achieves state-of-the-art performance on MIA benchmarks as well, demonstrating the practical impact of the transferability. Our findings highlight the need for greater cross-task awareness and collaboration between the two research communities. To facilitate cross-task developments and fair evaluations, we introduce MINT, a unified evaluation suite for MIAs and machine-generated text detection, with implementation of 15 recent methods from both tasks. 

---
# VideoAgentTrek: Computer Use Pretraining from Unlabeled Videos 

**Authors**: Dunjie Lu, Yiheng Xu, Junli Wang, Haoyuan Wu, Xinyuan Wang, Zekun Wang, Junlin Yang, Hongjin Su, Jixuan Chen, Junda Chen, Yuchen Mao, Jingren Zhou, Junyang Lin, Binyuan Hui, Tao Yu  

**Link**: [PDF](https://arxiv.org/pdf/2510.19488)  

**Abstract**: Training computer-use agents requires massive amounts of GUI interaction data, but manually annotating action trajectories at scale is prohibitively expensive. We present VideoAgentTrek, a scalable pipeline that automatically mines training data from publicly available screen-recorded videos at web scale, eliminating the need for manual annotation. Our approach addresses a key challenge: raw videos contain implicit demonstrations but lack explicit action labels. To solve this, we develop Video2Action, an inverse dynamics module (IDM) with two components: (1) a video grounding model that detects and localizes GUI actions with precise temporal boundaries and context, and (2) an action-content recognizer that extracts structured parameters like click coordinates and typed text with high fidelity. Applied to 39,000 YouTube tutorial videos, our pipeline generates 1.52 million interaction steps automatically. We leverage this data through continued pretraining followed by supervised fine-tuning. On OSWorld-Verified, our approach improves task success rates from 9.3% (SFT-only baseline) to 15.8%, a 70% relative improvement. On AgentNetBench, step accuracy increases from 64.1% to 69.3%. Our results demonstrate that passive internet videos can be transformed into high-quality supervision for computer-use agents, providing a scalable alternative to expensive manual annotation. 

---
# Re-evaluating Minimum Bayes Risk Decoding for Automatic Speech Recognition 

**Authors**: Yuu Jinnai  

**Link**: [PDF](https://arxiv.org/pdf/2510.19471)  

**Abstract**: Recent work has shown that sample-based Minimum Bayes Risk (MBR) decoding outperforms beam search in text-to-text generation tasks, such as machine translation, text summarization, and image captioning. On the other hand, beam search is the current practice for speech-to-text tasks such as automatic speech recognition (ASR) and Speech Translation (ST). Given that MBR decoding is effective in text-to-text generation tasks, it is reasonable to expect it to also be effective for speech-to-text tasks. In this paper, we evaluate MBR decoding for ASR and ST tasks on English and Japanese using Whisper and its derivative models. We observe that the accuracy of MBR decoding outperforms that of beam search in most of the experimental settings we have evaluated. The results show that MBR decoding is a promising method for offline ASR and ST tasks that require high accuracy. The code is available at this https URL 

---
# MINED: Probing and Updating with Multimodal Time-Sensitive Knowledge for Large Multimodal Models 

**Authors**: Kailin Jiang, Ning Jiang, Yuchen Ren, Yuchen Li, Yifan Gao, Jinhe Bi, Yunpu Ma, Qingqing Liu, Xianhao Wang, Yifan Jia, Hongbo Jiang, Yaocong Hu, Bin Li, Lei Liu, Yuntao Du  

**Link**: [PDF](https://arxiv.org/pdf/2510.19457)  

**Abstract**: Large Multimodal Models (LMMs) encode rich factual knowledge via cross-modal pre-training, yet their static representations struggle to maintain an accurate understanding of time-sensitive factual knowledge. Existing benchmarks remain constrained by static designs, inadequately evaluating LMMs' ability to understand time-sensitive knowledge. To address this gap, we propose MINED, a comprehensive benchmark that evaluates temporal awareness along 6 key dimensions and 11 challenging tasks: cognition, awareness, trustworthiness, understanding, reasoning, and robustness. MINED is constructed from Wikipedia by two professional annotators, containing 2,104 time-sensitive knowledge samples spanning six knowledge types. Evaluating 15 widely used LMMs on MINED shows that Gemini-2.5-Pro achieves the highest average CEM score of 63.07, while most open-source LMMs still lack time understanding ability. Meanwhile, LMMs perform best on organization knowledge, whereas their performance is weakest on sport. To address these challenges, we investigate the feasibility of updating time-sensitive knowledge in LMMs through knowledge editing methods and observe that LMMs can effectively update knowledge via knowledge editing methods in single editing scenarios. 

---
# BLiSS 1.0: Evaluating Bilingual Learner Competence in Second Language Small Language Models 

**Authors**: Yuan Gao, Suchir Salhan, Andrew Caines, Paula Buttery, Weiwei Sun  

**Link**: [PDF](https://arxiv.org/pdf/2510.19419)  

**Abstract**: To bridge the gap between performance-oriented benchmarks and the evaluation of cognitively inspired models, we introduce BLiSS 1.0, a Benchmark of Learner Interlingual Syntactic Structure. Our benchmark operationalizes a new paradigm of selective tolerance, testing whether a model finds a naturalistic learner error more plausible than a matched, artificial error within the same sentence. Constructed from over 2.8 million naturalistic learner sentences, BLiSS provides 136,867 controlled triplets (corrected, learner, artificial) for this purpose. Experiments on a diverse suite of models demonstrate that selective tolerance is a distinct capability from standard grammaticality, with performance clustering strongly by training paradigm. This validates BLiSS as a robust tool for measuring how different training objectives impact a model's alignment with the systematic patterns of human language acquisition. 

---
# Spatio-temporal Sign Language Representation and Translation 

**Authors**: Yasser Hamidullah, Josef van Genabith, Cristina España-Bonet  

**Link**: [PDF](https://arxiv.org/pdf/2510.19413)  

**Abstract**: This paper describes the DFKI-MLT submission to the WMT-SLT 2022 sign language translation (SLT) task from Swiss German Sign Language (video) into German (text). State-of-the-art techniques for SLT use a generic seq2seq architecture with customized input embeddings. Instead of word embeddings as used in textual machine translation, SLT systems use features extracted from video frames. Standard approaches often do not benefit from temporal features. In our participation, we present a system that learns spatio-temporal feature representations and translation in a single model, resulting in a real end-to-end architecture expected to better generalize to new data sets. Our best system achieved $5\pm1$ BLEU points on the development set, but the performance on the test dropped to $0.11\pm0.06$ BLEU points. 

---
# ToMMeR -- Efficient Entity Mention Detection from Large Language Models 

**Authors**: Victor Morand, Nadi Tomeh, Josiane Mothe, Benjamin Piwowarski  

**Link**: [PDF](https://arxiv.org/pdf/2510.19410)  

**Abstract**: Identifying which text spans refer to entities -- mention detection -- is both foundational for information extraction and a known performance bottleneck. We introduce ToMMeR, a lightweight model (<300K parameters) probing mention detection capabilities from early LLM layers. Across 13 NER benchmarks, ToMMeR achieves 93\% recall zero-shot, with over 90\% precision using an LLM as a judge showing that ToMMeR rarely produces spurious predictions despite high recall. Cross-model analysis reveals that diverse architectures (14M-15B parameters) converge on similar mention boundaries (DICE >75\%), confirming that mention detection emerges naturally from language modeling. When extended with span classification heads, ToMMeR achieves near SOTA NER performance (80-87\% F1 on standard benchmarks). Our work provides evidence that structured entity representations exist in early transformer layers and can be efficiently recovered with minimal parameters. 

---
# SONAR-SLT: Multilingual Sign Language Translation via Language-Agnostic Sentence Embedding Supervision 

**Authors**: Yasser Hamidullah, Shakib Yazdani, Cennet Oguz, Josef van Genabith, Cristina España-Bonet  

**Link**: [PDF](https://arxiv.org/pdf/2510.19398)  

**Abstract**: Sign language translation (SLT) is typically trained with text in a single spoken language, which limits scalability and cross-language generalization. Earlier approaches have replaced gloss supervision with text-based sentence embeddings, but up to now, these remain tied to a specific language and modality. In contrast, here we employ language-agnostic, multimodal embeddings trained on text and speech from multiple languages to supervise SLT, enabling direct multilingual translation. To address data scarcity, we propose a coupled augmentation method that combines multilingual target augmentations (i.e. translations into many languages) with video-level perturbations, improving model robustness. Experiments show consistent BLEURT gains over text-only sentence embedding supervision, with larger improvements in low-resource settings. Our results demonstrate that language-agnostic embedding supervision, combined with coupled augmentation, provides a scalable and semantically robust alternative to traditional SLT training. 

---
# Sign Language Translation with Sentence Embedding Supervision 

**Authors**: Yasser Hamidullah, Josef van Genabith, Cristina España-Bonet  

**Link**: [PDF](https://arxiv.org/pdf/2510.19367)  

**Abstract**: State-of-the-art sign language translation (SLT) systems facilitate the learning process through gloss annotations, either in an end2end manner or by involving an intermediate step. Unfortunately, gloss labelled sign language data is usually not available at scale and, when available, gloss annotations widely differ from dataset to dataset. We present a novel approach using sentence embeddings of the target sentences at training time that take the role of glosses. The new kind of supervision does not need any manual annotation but it is learned on raw textual data. As our approach easily facilitates multilinguality, we evaluate it on datasets covering German (PHOENIX-2014T) and American (How2Sign) sign languages and experiment with mono- and multilingual sentence embeddings and translation systems. Our approach significantly outperforms other gloss-free approaches, setting the new state-of-the-art for data sets where glosses are not available and when no additional SLT datasets are used for pretraining, diminishing the gap between gloss-free and gloss-dependent systems. 

---
# MoE-Prism: Disentangling Monolithic Experts for Elastic MoE Services via Model-System Co-Designs 

**Authors**: Xinfeng Xia, Jiacheng Liu, Xiaofeng Hou, Peng Tang, Mingxuan Zhang, Wenfeng Wang, Chao Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.19366)  

**Abstract**: Mixture-of-Experts (MoE) models, the state-of-the-art in large-scale AI, achieve high quality by sparsely activating parameters. However, their reliance on routing between a few monolithic experts via a top-k mechanism creates a "quality cliff", offering only a few coarse-grained operating points. This inflexibility forces a difficult trade-off between cost and quality, preventing adaptation to diverse Service Level Objectives (SLOs) and leading to significant resource over-provisioning.
This paper introduces MoE-Prism, a model-system co-design that transforms rigid MoE models into elastic services. Our methodology is divided into two phases. First, an \emph{Offline Refactoring Engine} systematically deconstructs monolithic experts into fine-grained "sub-experts." This engine employs a partitioning optimization solver that uses a metaheuristic-based approach to group neurons, preserving functional locality without requiring retraining. Second, an \emph{Online Scheduling Engine} leverages this new elasticity through QoS-aware scheduling. It implements specialized policies to solve complex system problems, including maximizing throughput in cloud deployments and managing latency-optimized offloading for memory-constrained devices. Our evaluation across three different MoE models shows that MoE-Prismprovides over 4 times more distinct, stable operating points than the baseline. This allows an AI service to dynamically improve throughput by up to 19.9\% under a strict latency budget or reduce latency by up to 10.36\% under limited resources. MoE-Prism provides the critical "control knob" to bridge the model-system gap, enabling the next generation of adaptive, efficient, and QoS-aware AI services. 

---
# The Massive Legal Embedding Benchmark (MLEB) 

**Authors**: Umar Butler, Abdur-Rahman Butler, Adrian Lucas Malec  

**Link**: [PDF](https://arxiv.org/pdf/2510.19365)  

**Abstract**: We present the Massive Legal Embedding Benchmark (MLEB), the largest, most diverse, and most comprehensive open-source benchmark for legal information retrieval to date. MLEB consists of ten expert-annotated datasets spanning multiple jurisdictions (the US, UK, EU, Australia, Ireland, and Singapore), document types (cases, legislation, regulatory guidance, contracts, and literature), and task types (search, zero-shot classification, and question answering). Seven of the datasets in MLEB were newly constructed in order to fill domain and jurisdictional gaps in the open-source legal information retrieval landscape. We document our methodology in building MLEB and creating the new constituent datasets, and release our code, results, and data openly to assist with reproducible evaluations. 

---
# LoongRL:Reinforcement Learning for Advanced Reasoning over Long Contexts 

**Authors**: Siyuan Wang, Gaokai Zhang, Li Lyna Zhang, Ning Shang, Fan Yang, Dongyao Chen, Mao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2510.19363)  

**Abstract**: Reasoning over long contexts is essential for large language models. While reinforcement learning (RL) enhances short-context reasoning by inducing "Aha" moments in chain-of-thought, the advanced thinking patterns required for long-context reasoning remain largely unexplored, and high-difficulty RL data are scarce. In this paper, we introduce LoongRL, a data-driven RL method for advanced long-context reasoning. Central to LoongRL is KeyChain, a synthesis approach that transforms short multi-hop QA into high-difficulty long-context tasks by inserting UUID chains that hide the true question among large collections of distracting documents. Solving these tasks requires the model to trace the correct chain step-by-step, identify the true question, retrieve relevant facts and reason over them to answer correctly. RL training on KeyChain data induces an emergent plan-retrieve-reason-recheck reasoning pattern that generalizes far beyond training length. Models trained at 16K effectively solve 128K tasks without prohibitive full-length RL rollout costs. On Qwen2.5-7B and 14B, LoongRL substantially improves long-context multi-hop QA accuracy by +23.5% and +21.1% absolute gains. The resulting LoongRL-14B reaches a score of 74.2, rivaling much larger frontier models such as o3-mini (74.5) and DeepSeek-R1 (74.9). It also improves long-context retrieval, passes all 128K needle-in-a-haystack stress tests, and preserves short-context reasoning capabilities. 

---
# AgenticMath: Enhancing LLM Reasoning via Agentic-based Math Data Generation 

**Authors**: Xianyang Liu, Yilin Liu, Shuai Wang, Hao Cheng, Andrew Estornell, Yuzhi Zhao, Jiaheng Wei  

**Link**: [PDF](https://arxiv.org/pdf/2510.19361)  

**Abstract**: The creation of high-quality datasets to improve Large Language Model (LLM) reasoning remains a significant challenge, as current methods often suffer from generating low-quality/incorrect answers and limited information richness from available data sources. To address this, we propose AgenticMath, a novel agentic pipeline for generating high-quality mathematical question-answer pairs to enhance the supervised fine-tuning of LLMs. Our method operates through four stages: (1) Seed Question Filter that selects questions with high information richness, complexity, and clarity; (2) an Agentic Question Rephrase step that employs a multi-agent system to generate diverse, logically consistent paraphrases; (3) an Answer Augment step where rewrite answers using chain-of-thought reasoning to enhance numerical and logical correctness, without reliance on human-provided labels; and (4) a final Question and Answer Evaluation that retains only the most superior pairs. Extensive experiments demonstrate that, fine-tuning 3B-8B parameter LLMs on AgenticMath generated datasets (comprising only 30-60K math samples) achieves competitive or superior performance on diverse in domain and out-of-domain mathematical reasoning benchmarks compared to baselines trained on much more data (e.g., 400K or 2.3M samples). Our work demonstrates that targeted, high-quality data generation is a more efficient path to improving mathematical reasoning in LLMs than large-scale, low-quality alternatives. 

---
# M3-SLU: Evaluating Speaker-Attributed Reasoning in Multimodal Large Language Models 

**Authors**: Yejin Kwon, Taewoo Kang, Hyunsoo Yoon, Changouk Kim  

**Link**: [PDF](https://arxiv.org/pdf/2510.19358)  

**Abstract**: We present M3-SLU, a new multimodal large language model (MLLM) benchmark for evaluating multi-speaker, multi-turn spoken language understanding. While recent models show strong performance in speech and text comprehension, they still struggle with speaker-attributed reasoning, the ability to understand who said what and when in natural conversations. M3-SLU is built from four open corpora (CHiME-6, MELD, MultiDialog, and AMI) and comprises over 12,000 validated instances with paired audio, transcripts, and metadata. It includes two tasks: (1) Speaker-Attributed Question Answering and (2) Speaker Attribution via Utterance Matching. We provide baseline results for both cascaded pipelines and end-to-end MLLMs, evaluated using an LLM-as-Judge and accuracy metrics. Results show that while models can capture what was said, they often fail to identify who said it, revealing a key gap in speaker-aware dialogue understanding. M3-SLU offers as a challenging benchmark to advance research in speaker-aware multimodal understanding. 

---
# Modeling Turn-Taking with Semantically Informed Gestures 

**Authors**: Varsha Suresh, M. Hamza Mughal, Christian Theobalt, Vera Demberg  

**Link**: [PDF](https://arxiv.org/pdf/2510.19350)  

**Abstract**: In conversation, humans use multimodal cues, such as speech, gestures, and gaze, to manage turn-taking. While linguistic and acoustic features are informative, gestures provide complementary cues for modeling these transitions. To study this, we introduce DnD Gesture++, an extension of the multi-party DnD Gesture corpus enriched with 2,663 semantic gesture annotations spanning iconic, metaphoric, deictic, and discourse types. Using this dataset, we model turn-taking prediction through a Mixture-of-Experts framework integrating text, audio, and gestures. Experiments show that incorporating semantically guided gestures yields consistent performance gains over baselines, demonstrating their complementary role in multimodal turn-taking. 

---
# Local Obfuscation by GLINER for Impartial Context Aware Lineage: Development and evaluation of PII Removal system 

**Authors**: Prakrithi Shivaprakash, Lekhansh Shukla, Animesh Mukherjee, Prabhat Chand, Pratima Murthy  

**Link**: [PDF](https://arxiv.org/pdf/2510.19346)  

**Abstract**: Removing Personally Identifiable Information (PII) from clinical notes in Electronic Health Records (EHRs) is essential for research and AI development. While Large Language Models (LLMs) are powerful, their high computational costs and the data privacy risks of API-based services limit their use, especially in low-resource settings. To address this, we developed LOGICAL (Local Obfuscation by GLINER for Impartial Context-Aware Lineage), an efficient, locally deployable PII removal system built on a fine-tuned Generalist and Lightweight Named Entity Recognition (GLiNER) model. We used 1515 clinical documents from a psychiatric hospital's EHR system. We defined nine PII categories for removal. A modern-gliner-bi-large-v1.0 model was fine-tuned on 2849 text instances and evaluated on a test set of 376 instances using character-level precision, recall, and F1-score. We compared its performance against Microsoft Azure NER, Microsoft Presidio, and zero-shot prompting with Gemini-Pro-2.5 and Llama-3.3-70B-Instruct. The fine-tuned GLiNER model achieved superior performance, with an overall micro-average F1-score of 0.980, significantly outperforming Gemini-Pro-2.5 (F1-score: 0.845). LOGICAL correctly sanitised 95% of documents completely, compared to 64% for the next-best solution. The model operated efficiently on a standard laptop without a dedicated GPU. However, a 2% entity-level false negative rate underscores the need for human-in-the-loop validation across all tested systems. Fine-tuned, specialised transformer models like GLiNER offer an accurate, computationally efficient, and secure solution for PII removal from clinical notes. This "sanitisation at the source" approach is a practical alternative to resource-intensive LLMs, enabling the creation of de-identified datasets for research and AI development while preserving data privacy, particularly in resource-constrained environments. 

---
# Algorithmic Fairness in NLP: Persona-Infused LLMs for Human-Centric Hate Speech Detection 

**Authors**: Ewelina Gajewska, Arda Derbent, Jaroslaw A Chudziak, Katarzyna Budzynska  

**Link**: [PDF](https://arxiv.org/pdf/2510.19331)  

**Abstract**: In this paper, we investigate how personalising Large Language Models (Persona-LLMs) with annotator personas affects their sensitivity to hate speech, particularly regarding biases linked to shared or differing identities between annotators and targets. To this end, we employ Google's Gemini and OpenAI's GPT-4.1-mini models and two persona-prompting methods: shallow persona prompting and a deeply contextualised persona development based on Retrieval-Augmented Generation (RAG) to incorporate richer persona profiles. We analyse the impact of using in-group and out-group annotator personas on the models' detection performance and fairness across diverse social groups. This work bridges psychological insights on group identity with advanced NLP techniques, demonstrating that incorporating socio-demographic attributes into LLMs can address bias in automated hate speech detection. Our results highlight both the potential and limitations of persona-based approaches in reducing bias, offering valuable insights for developing more equitable hate speech detection systems. 

---
# Slot Filling as a Reasoning Task for SpeechLLMs 

**Authors**: Kadri Hacioglu, Manjunath K E, Andreas Stolcke  

**Link**: [PDF](https://arxiv.org/pdf/2510.19326)  

**Abstract**: We propose integration of reasoning into speech large language models (speechLLMs) for the end-to-end slot-filling task. Inspired by the recent development of reasoning LLMs, we use a chain-of-thought framework to decompose the slot-filling task into multiple reasoning steps, create a reasoning dataset and apply the supervised fine-tuning strategy to a speechLLM. We distinguish between regular and reasoning speechLLMs and experiment with different types and sizes of LLMs as their text foundation models. We demonstrate performance improvements by introducing reasoning (intermediate) steps. However, we show that a reasoning textual LLM developed mainly for math, logic and coding domains might be inferior as a foundation model for a reasoning speechLLM. We further show that hybrid speechLLMs, built on a hybrid text foundation LLM and fine-tuned to preserve both direct and reasoning modes of operation, have better performance than those fine-tuned employing only one mode of operation. 

---
# Balancing Rewards in Text Summarization: Multi-Objective Reinforcement Learning via HyperVolume Optimization 

**Authors**: Junjie Song, Yiwen Liu, Dapeng Li, Yin Sun, Shukun Fu, Siqi Chen, Yuji Cao  

**Link**: [PDF](https://arxiv.org/pdf/2510.19325)  

**Abstract**: Text summarization is a crucial task that requires the simultaneous optimization of multiple objectives, including consistency, coherence, relevance, and fluency, which presents considerable challenges. Although large language models (LLMs) have demonstrated remarkable performance, enhanced by reinforcement learning (RL), few studies have focused on optimizing the multi-objective problem of summarization through RL based on LLMs. In this paper, we introduce hypervolume optimization (HVO), a novel optimization strategy that dynamically adjusts the scores between groups during the reward process in RL by using the hypervolume method. This method guides the model's optimization to progressively approximate the pareto front, thereby generating balanced summaries across multiple objectives. Experimental results on several representative summarization datasets demonstrate that our method outperforms group relative policy optimization (GRPO) in overall scores and shows more balanced performance across different dimensions. Moreover, a 7B foundation model enhanced by HVO performs comparably to GPT-4 in the summarization task, while maintaining a shorter generation length. Our code is publicly available at this https URL 

---
# HAD: HAllucination Detection Language Models Based on a Comprehensive Hallucination Taxonomy 

**Authors**: Fan Xu, Xinyu Hu, Zhenghan Yu, Li Lin, Xu Zhang, Yang Zhang, Wei Zhou, Jinjie Gu, Xiaojun Wan  

**Link**: [PDF](https://arxiv.org/pdf/2510.19318)  

**Abstract**: The increasing reliance on natural language generation (NLG) models, particularly large language models, has raised concerns about the reliability and accuracy of their outputs. A key challenge is hallucination, where models produce plausible but incorrect information. As a result, hallucination detection has become a critical task. In this work, we introduce a comprehensive hallucination taxonomy with 11 categories across various NLG tasks and propose the HAllucination Detection (HAD) models this https URL, which integrate hallucination detection, span-level identification, and correction into a single inference process. Trained on an elaborate synthetic dataset of about 90K samples, our HAD models are versatile and can be applied to various NLG tasks. We also carefully annotate a test set for hallucination detection, called HADTest, which contains 2,248 samples. Evaluations on in-domain and out-of-domain test sets show that our HAD models generally outperform the existing baselines, achieving state-of-the-art results on HaluEval, FactCHD, and FaithBench, confirming their robustness and versatility. 

---
# KORE: Enhancing Knowledge Injection for Large Multimodal Models via Knowledge-Oriented Augmentations and Constraints 

**Authors**: Kailin Jiang, Hongbo Jiang, Ning Jiang, Zhi Gao, Jinhe Bi, Yuchen Ren, Bin Li, Yuntao Du, Lei Liu, Qing Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.19316)  

**Abstract**: Large Multimodal Models encode extensive factual knowledge in their pre-trained weights. However, its knowledge remains static and limited, unable to keep pace with real-world developments, which hinders continuous knowledge acquisition. Effective knowledge injection thus becomes critical, involving two goals: knowledge adaptation (injecting new knowledge) and knowledge retention (preserving old knowledge). Existing methods often struggle to learn new knowledge and suffer from catastrophic forgetting. To address this, we propose KORE, a synergistic method of KnOwledge-oRientEd augmentations and constraints for injecting new knowledge into large multimodal models while preserving old knowledge. Unlike general text or image data augmentation, KORE automatically converts individual knowledge items into structured and comprehensive knowledge to ensure that the model accurately learns new knowledge, enabling accurate adaptation. Meanwhile, KORE stores previous knowledge in the covariance matrix of LMM's linear layer activations and initializes the adapter by projecting the original weights into the matrix's null space, defining a fine-tuning direction that minimizes interference with previous knowledge, enabling powerful retention. Extensive experiments on various LMMs, including LLaVA-v1.5-7B, LLaVA-v1.5-13B, and Qwen2.5-VL-7B, show that KORE achieves superior new knowledge injection performance and effectively mitigates catastrophic forgetting. 

---
# JointCQ: Improving Factual Hallucination Detection with Joint Claim and Query Generation 

**Authors**: Fan Xu, Huixuan Zhang, Zhenliang Zhang, Jiahao Wang, Xiaojun Wan  

**Link**: [PDF](https://arxiv.org/pdf/2510.19310)  

**Abstract**: Current large language models (LLMs) often suffer from hallucination issues, i,e, generating content that appears factual but is actually unreliable. A typical hallucination detection pipeline involves response decomposition (i.e., claim extraction), query generation, evidence collection (i.e., search or retrieval), and claim verification. However, existing methods exhibit limitations in the first two stages, such as context loss during claim extraction and low specificity in query generation, resulting in degraded performance across the hallucination detection pipeline. In this work, we introduce JointCQ this https URL, a joint claim-and-query generation framework designed to construct an effective and efficient claim-query generator. Our framework leverages elaborately designed evaluation criteria to filter synthesized training data, and finetunes a language model for joint claim extraction and query generation, providing reliable and informative inputs for downstream search and verification. Experimental results demonstrate that our method outperforms previous methods on multiple open-domain QA hallucination detection benchmarks, advancing the goal of more trustworthy and transparent language model systems. 

---
# TheMCPCompany: Creating General-purpose Agents with Task-specific Tools 

**Authors**: Reza Esfandiarpoor, Vishwas Suryanarayanan, Stephen H. Bach, Vishal Chowdhary, Anthony Aue  

**Link**: [PDF](https://arxiv.org/pdf/2510.19286)  

**Abstract**: Since the introduction of the Model Context Protocol (MCP), the number of available tools for Large Language Models (LLMs) has increased significantly. These task-specific tool sets offer an alternative to general-purpose tools such as web browsers, while being easier to develop and maintain than GUIs. However, current general-purpose agents predominantly rely on web browsers for interacting with the environment. Here, we introduce TheMCPCompany, a benchmark for evaluating tool-calling agents on tasks that involve interacting with various real-world services. We use the REST APIs of these services to create MCP servers, which include over 18,000 tools. We also provide manually annotated ground-truth tools for each task. In our experiments, we use the ground truth tools to show the potential of tool-calling agents for both improving performance and reducing costs assuming perfect tool retrieval. Next, we explore agent performance using tool retrieval to study the real-world practicality of tool-based agents. While all models with tool retrieval perform similarly or better than browser-based agents, smaller models cannot take full advantage of the available tools through retrieval. On the other hand, GPT-5's performance with tool retrieval is very close to its performance with ground-truth tools. Overall, our work shows that the most advanced reasoning models are effective at discovering tools in simpler environments, but seriously struggle with navigating complex enterprise environments. TheMCPCompany reveals that navigating tens of thousands of tools and combining them in non-trivial ways to solve complex problems is still a challenging task for current models and requires both better reasoning and better retrieval models. 

---
# Difficulty-Controllable Multiple-Choice Question Generation Using Large Language Models and Direct Preference Optimization 

**Authors**: Yuto Tomikawa, Masaki Uto  

**Link**: [PDF](https://arxiv.org/pdf/2510.19265)  

**Abstract**: Difficulty-controllable question generation for reading comprehension has gained significant attention in the field of education as a fundamental tool for adaptive learning support. Although several neural question generation methods have recently succeeded in controlling difficulty, conventional approaches still face two major limitations. First, they cannot directly generate multiple-choice questions, which are the most widely used question type in educational contexts. Second, they are not explicitly trained to optimize the accuracy of difficulty control, leaving room for further improvement in difficulty controllability. To address these limitations, this study proposes a novel difficulty-controllable multiple-choice question generation method for reading comprehension which leverages a large language model trained using a direct preference optimization technique to improve the accuracy of difficulty control. 

---
# SheetBrain: A Neuro-Symbolic Agent for Accurate Reasoning over Complex and Large Spreadsheets 

**Authors**: Ziwei Wang, Jiayuan Su, Mengyu Zhou, Huaxing Zeng, Mengni Jia, Xiao Lv, Haoyu Dong, Xiaojun Ma, Shi Han, Dongmei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.19247)  

**Abstract**: Understanding and reasoning over complex spreadsheets remain fundamental challenges for large language models (LLMs), which often struggle with accurately capturing the complex structure of tables and ensuring reasoning correctness. In this work, we propose SheetBrain, a neuro-symbolic dual workflow agent framework designed for accurate reasoning over tabular data, supporting both spreadsheet question answering and manipulation tasks. SheetBrain comprises three core modules: an understanding module, which produces a comprehensive overview of the spreadsheet - including sheet summary and query-based problem insight to guide reasoning; an execution module, which integrates a Python sandbox with preloaded table-processing libraries and an Excel helper toolkit for effective multi-turn reasoning; and a validation module, which verifies the correctness of reasoning and answers, triggering re-execution when necessary. We evaluate SheetBrain on multiple public tabular QA and manipulation benchmarks, and introduce SheetBench, a new benchmark targeting large, multi-table, and structurally complex spreadsheets. Experimental results show that SheetBrain significantly improves accuracy on both existing benchmarks and the more challenging scenarios presented in SheetBench. Our code is publicly available at this https URL. 

---
# Modality Matching Matters: Calibrating Language Distances for Cross-Lingual Transfer in URIEL+ 

**Authors**: York Hay Ng, Aditya Khan, Xiang Lu, Matteo Salloum, Michael Zhou, Phuong H. Hoang, A. Seza Doğruöz, En-Shiun Annie Lee  

**Link**: [PDF](https://arxiv.org/pdf/2510.19217)  

**Abstract**: Existing linguistic knowledge bases such as URIEL+ provide valuable geographic, genetic and typological distances for cross-lingual transfer but suffer from two key limitations. One, their one-size-fits-all vector representations are ill-suited to the diverse structures of linguistic data, and two, they lack a principled method for aggregating these signals into a single, comprehensive score. In this paper, we address these gaps by introducing a framework for type-matched language distances. We propose novel, structure-aware representations for each distance type: speaker-weighted distributions for geography, hyperbolic embeddings for genealogy, and a latent variables model for typology. We unify these signals into a robust, task-agnostic composite distance. In selecting transfer languages, our representations and composite distances consistently improve performance across a wide range of NLP tasks, providing a more principled and effective toolkit for multilingual research. 

---
# DiSRouter: Distributed Self-Routing for LLM Selections 

**Authors**: Hang Zheng, Hongshen Xu, Yongkai Lin, Shuai Fan, Lu Chen, Kai Yu  

**Link**: [PDF](https://arxiv.org/pdf/2510.19208)  

**Abstract**: The proliferation of Large Language Models (LLMs) has created a diverse ecosystem of models with highly varying performance and costs, necessitating effective query routing to balance performance and expense. Current routing systems often rely on a centralized external router trained on a fixed set of LLMs, making them inflexible and prone to poor performance since the small router can not fully understand the knowledge boundaries of different LLMs. We introduce DiSRouter (Distributed Self-Router), a novel paradigm that shifts from centralized control to distributed routing. In DiSRouter, a query traverses a network of LLM agents, each independently deciding whether to answer or route to other agents based on its own self-awareness, its ability to judge its competence. This distributed design offers superior flexibility, scalability, and generalizability. To enable this, we propose a two-stage Self-Awareness Training pipeline that enhances each LLM's self-awareness. Extensive experiments demonstrate that DiSRouter significantly outperforms existing routing methods in utility across various scenarios, effectively distinguishes between easy and hard queries, and shows strong generalization to out-of-domain tasks. Our work validates that leveraging an LLM's intrinsic self-awareness is more effective than external assessment, paving the way for more modular and efficient multi-agent systems. 

---
# Multi-Faceted Evaluation of Tool-Augmented Dialogue Systems 

**Authors**: Zhaoyi Joey Hou, Tanya Shourya, Yingfan Wang, Shamik Roy, Vinayshekhar Bannihatti Kumar, Rashmi Gangadharaiah  

**Link**: [PDF](https://arxiv.org/pdf/2510.19186)  

**Abstract**: Evaluating conversational AI systems that use external tools is challenging, as errors can arise from complex interactions among user, agent, and tools. While existing evaluation methods assess either user satisfaction or agents' tool-calling capabilities, they fail to capture critical errors in multi-turn tool-augmented dialogues-such as when agents misinterpret tool results yet appear satisfactory to users. We introduce TRACE, a benchmark of systematically synthesized tool-augmented conversations covering diverse error cases, and SCOPE, an evaluation framework that automatically discovers diverse error patterns and evaluation rubrics in tool-augmented dialogues. Experiments show SCOPE significantly outperforms the baseline, particularly on challenging cases where user satisfaction signals are misleading. 

---
# Interpretable Question Answering with Knowledge Graphs 

**Authors**: Kartikeya Aneja, Manasvi Srivastava, Subhayan Das, Nagender Aneja  

**Link**: [PDF](https://arxiv.org/pdf/2510.19181)  

**Abstract**: This paper presents a question answering system that operates exclusively on a knowledge graph retrieval without relying on retrieval augmented generation (RAG) with large language models (LLMs). Instead, a small paraphraser model is used to paraphrase the entity relationship edges retrieved from querying the knowledge graph. The proposed pipeline is divided into two main stages. The first stage involves pre-processing a document to generate sets of question-answer (QA) pairs. The second stage converts these QAs into a knowledge graph from which graph-based retrieval is performed using embeddings and fuzzy techniques. The graph is queried, re-ranked, and paraphrased to generate a final answer. This work includes an evaluation using LLM-as-a-judge on the CRAG benchmark, which resulted in accuracies of 71.9% and 54.4% using LLAMA-3.2 and GPT-3.5-Turbo, respectively. 

---
# When Facts Change: Probing LLMs on Evolving Knowledge with evolveQA 

**Authors**: Nishanth Sridhar Nakshatri, Shamik Roy, Manoj Ghuhan Arivazhagan, Hanhan Zhou, Vinayshekhar Bannihatti Kumar, Rashmi Gangadharaiah  

**Link**: [PDF](https://arxiv.org/pdf/2510.19172)  

**Abstract**: LLMs often fail to handle temporal knowledge conflicts--contradictions arising when facts evolve over time within their training data. Existing studies evaluate this phenomenon through benchmarks built on structured knowledge bases like Wikidata, but they focus on widely-covered, easily-memorized popular entities and lack the dynamic structure needed to fairly evaluate LLMs with different knowledge cut-off dates. We introduce evolveQA, a benchmark specifically designed to evaluate LLMs on temporally evolving knowledge, constructed from 3 real-world, time-stamped corpora: AWS updates, Azure changes, and WHO disease outbreak reports. Our framework identifies naturally occurring knowledge evolution and generates questions with gold answers tailored to different LLM knowledge cut-off dates. Through extensive evaluation of 12 open and closed-source LLMs across 3 knowledge probing formats, we demonstrate significant performance drops of up to 31% on evolveQA compared to static knowledge questions. 

---
# Think Straight, Stop Smart: Structured Reasoning for Efficient Multi-Hop RAG 

**Authors**: Jihwan Bang, Juntae Lee, Seunghan Yang, Sungha Choi  

**Link**: [PDF](https://arxiv.org/pdf/2510.19171)  

**Abstract**: Multi-hop retrieval-augmented generation (RAG) is a promising strategy for complex reasoning, yet existing iterative prompting approaches remain inefficient. They often regenerate predictable token sequences at every step and rely on stochastic stopping, leading to excessive token usage and unstable termination. We propose TSSS (Think Straight, Stop Smart), a structured multi-hop RAG framework designed for efficiency. TSSS introduces (i) a template-based reasoning that caches recurring prefixes and anchors sub-queries to the main question, reducing token generation cost while promoting stable reasoning, and (ii) a retriever-based terminator, which deterministically halts reasoning once additional sub-queries collapse into repetition. This separation of structured reasoning and termination control enables both faster inference and more reliable answers. On HotpotQA, 2WikiMultiHop, and MuSiQue, TSSS achieves state-of-the-art accuracy and competitive efficiency among RAG-CoT approaches, highlighting its effectiveness in efficiency-constrained scenarios such as on-device inference. 

---
# "You Are Rejected!": An Empirical Study of Large Language Models Taking Hiring Evaluations 

**Authors**: Dingjie Fu, Dianxing Shi  

**Link**: [PDF](https://arxiv.org/pdf/2510.19167)  

**Abstract**: With the proliferation of the internet and the rapid advancement of Artificial Intelligence, leading technology companies face an urgent annual demand for a considerable number of software and algorithm engineers. To efficiently and effectively identify high-potential candidates from thousands of applicants, these firms have established a multi-stage selection process, which crucially includes a standardized hiring evaluation designed to assess job-specific competencies. Motivated by the demonstrated prowess of Large Language Models (LLMs) in coding and reasoning tasks, this paper investigates a critical question: Can LLMs successfully pass these hiring evaluations? To this end, we conduct a comprehensive examination of a widely used professional assessment questionnaire. We employ state-of-the-art LLMs to generate responses and subsequently evaluate their performance. Contrary to any prior expectation of LLMs being ideal engineers, our analysis reveals a significant inconsistency between the model-generated answers and the company-referenced solutions. Our empirical findings lead to a striking conclusion: All evaluated LLMs fails to pass the hiring evaluation. 

---
# Tibetan Language and AI: A Comprehensive Survey of Resources, Methods and Challenges 

**Authors**: Cheng Huang, Nyima Tashi, Fan Gao, Yutong Liu, Jiahao Li, Hao Tian, Siyang Jiang, Thupten Tsering, Ban Ma-bao, Renzeg Duojie, Gadeng Luosang, Rinchen Dongrub, Dorje Tashi, Jin Zhang, Xiao Feng, Hao Wang, Jie Tang, Guojie Tang, Xiangxiang Wang, Jia Zhang, Tsengdar Lee, Yongbin Yu  

**Link**: [PDF](https://arxiv.org/pdf/2510.19144)  

**Abstract**: Tibetan, one of the major low-resource languages in Asia, presents unique linguistic and sociocultural characteristics that pose both challenges and opportunities for AI research. Despite increasing interest in developing AI systems for underrepresented languages, Tibetan has received limited attention due to a lack of accessible data resources, standardized benchmarks, and dedicated tools. This paper provides a comprehensive survey of the current state of Tibetan AI in the AI domain, covering textual and speech data resources, NLP tasks, machine translation, speech recognition, and recent developments in LLMs. We systematically categorize existing datasets and tools, evaluate methods used across different tasks, and compare performance where possible. We also identify persistent bottlenecks such as data sparsity, orthographic variation, and the lack of unified evaluation metrics. Additionally, we discuss the potential of cross-lingual transfer, multi-modal learning, and community-driven resource creation. This survey aims to serve as a foundational reference for future work on Tibetan AI research and encourages collaborative efforts to build an inclusive and sustainable AI ecosystem for low-resource languages. 

---
# Training-Free Spectral Fingerprints of Voice Processing in Transformers 

**Authors**: Valentin Noël  

**Link**: [PDF](https://arxiv.org/pdf/2510.19131)  

**Abstract**: Different transformer architectures implement identical linguistic computations via distinct connectivity patterns, yielding model imprinted ``computational fingerprints'' detectable through spectral analysis. Using graph signal processing on attention induced token graphs, we track changes in algebraic connectivity (Fiedler value, $\Delta\lambda_2$) under voice alternation across 20 languages and three model families, with a prespecified early window (layers 2--5). Our analysis uncovers clear architectural signatures: Phi-3-Mini shows a dramatic English specific early layer disruption ($\overline{\Delta\lambda_2}_{[2,5]}\!\approx\!-0.446$) while effects in 19 other languages are minimal, consistent with public documentation that positions the model primarily for English use. Qwen2.5-7B displays small, distributed shifts that are largest for morphologically rich languages, and LLaMA-3.2-1B exhibits systematic but muted responses. These spectral signatures correlate strongly with behavioral differences (Phi-3: $r=-0.976$) and are modulated by targeted attention head ablations, linking the effect to early attention structure and confirming functional relevance. Taken together, the findings are consistent with the view that training emphasis can leave detectable computational imprints: specialized processing strategies that manifest as measurable connectivity patterns during syntactic transformations. Beyond voice alternation, the framework differentiates reasoning modes, indicating utility as a simple, training free diagnostic for revealing architectural biases and supporting model reliability analysis. 

---
# A Graph Signal Processing Framework for Hallucination Detection in Large Language Models 

**Authors**: Valentin Noël  

**Link**: [PDF](https://arxiv.org/pdf/2510.19117)  

**Abstract**: Large language models achieve impressive results but distinguishing factual reasoning from hallucinations remains challenging. We propose a spectral analysis framework that models transformer layers as dynamic graphs induced by attention, with token embeddings as signals on these graphs. Through graph signal processing, we define diagnostics including Dirichlet energy, spectral entropy, and high-frequency energy ratios, with theoretical connections to computational stability. Experiments across GPT architectures suggest universal spectral patterns: factual statements exhibit consistent "energy mountain" behavior with low-frequency convergence, while different hallucination types show distinct signatures. Logical contradictions destabilize spectra with large effect sizes ($g>1.0$), semantic errors remain stable but show connectivity drift, and substitution hallucinations display intermediate perturbations. A simple detector using spectral signatures achieves 88.75% accuracy versus 75% for perplexity-based baselines, demonstrating practical utility. These findings indicate that spectral geometry may capture reasoning patterns and error behaviors, potentially offering a framework for hallucination detection in large language models. 

---
# That's Deprecated! Understanding, Detecting, and Steering Knowledge Conflicts in Language Models for Code Generation 

**Authors**: Jaesung Bae, Cameron Churchwell, Mitchell Hermon, Tsun-An Hsieh, Jocelyn Xu, Yekaterina Yegorova, Mark Hasegawa-Johnson, Heng Ji  

**Link**: [PDF](https://arxiv.org/pdf/2510.19116)  

**Abstract**: This paper investigates how large language models (LLMs) behave when faced with discrepancies between their parametric knowledge and conflicting information contained in a prompt. Building on prior question-answering (QA) research, we extend the investigation of knowledge conflicts to the realm of code generation. We propose a domain-agnostic framework for constructing and interpreting such conflicts, along with a novel evaluation method and dataset tailored to code conflict scenarios. Our experiments indicate that sufficiently large LLMs encode the notion of a knowledge conflict in their parameters, enabling us to detect knowledge conflicts with up to \textbf{80.65\%} accuracy. Building on these insights, we show that activation-level steering can achieve up to a \textbf{12.6\%} improvement in steering success over a random baseline. However, effectiveness depends critically on balancing model size, task domain, and steering direction. The experiment code and data will be made publicly available after acceptance. 

---
# From Memorization to Generalization: Fine-Tuning Large Language Models for Biomedical Term-to-Identifier Normalization 

**Authors**: Suswitha Pericharla, Daniel B. Hier, Tayo Obafemi-Ajayi  

**Link**: [PDF](https://arxiv.org/pdf/2510.19036)  

**Abstract**: Effective biomedical data integration depends on automated term normalization, the mapping of natural language biomedical terms to standardized identifiers. This linking of terms to identifiers is essential for semantic interoperability. Large language models (LLMs) show promise for this task but perform unevenly across terminologies. We evaluated both memorization (training-term performance) and generalization (validation-term performance) across multiple biomedical ontologies. Fine-tuning Llama 3.1 8B revealed marked differences by terminology. GO mappings showed strong memorization gains (up to 77% improvement in term-to-identifier accuracy), whereas HPO showed minimal improvement. Generalization occurred only for protein-gene (GENE) mappings (13.9% gain), while fine-tuning for HPO and GO yielded negligible transfer. Baseline accuracy varied by model scale, with GPT-4o outperforming both Llama variants for all terminologies. Embedding analyses showed tight semantic alignment between gene symbols and protein names but weak alignment between terms and identifiers for GO or HPO, consistent with limited lexicalization. Fine-tuning success depended on two interacting factors: identifier popularity and lexicalization. Popular identifiers were more likely encountered during pretraining, enhancing memorization. Lexicalized identifiers, such as gene symbols, enabled semantic generalization. By contrast, arbitrary identifiers in GO and HPO constrained models to rote learning. These findings provide a predictive framework for when fine-tuning enhances factual recall versus when it fails due to sparse or non-lexicalized identifiers. 

---
# When Can We Trust LLMs in Mental Health? Large-Scale Benchmarks for Reliable LLM Evaluation 

**Authors**: Abeer Badawi, Elahe Rahimi, Md Tahmid Rahman Laskar, Sheri Grach, Lindsay Bertrand, Lames Danok, Jimmy Huang, Frank Rudzicz, Elham Dolatabadi  

**Link**: [PDF](https://arxiv.org/pdf/2510.19032)  

**Abstract**: Evaluating Large Language Models (LLMs) for mental health support is challenging due to the emotionally and cognitively complex nature of therapeutic dialogue. Existing benchmarks are limited in scale, reliability, often relying on synthetic or social media data, and lack frameworks to assess when automated judges can be trusted. To address the need for large-scale dialogue datasets and judge reliability assessment, we introduce two benchmarks that provide a framework for generation and evaluation. MentalBench-100k consolidates 10,000 one-turn conversations from three real scenarios datasets, each paired with nine LLM-generated responses, yielding 100,000 response pairs. MentalAlign-70k}reframes evaluation by comparing four high-performing LLM judges with human experts across 70,000 ratings on seven attributes, grouped into Cognitive Support Score (CSS) and Affective Resonance Score (ARS). We then employ the Affective Cognitive Agreement Framework, a statistical methodology using intraclass correlation coefficients (ICC) with confidence intervals to quantify agreement, consistency, and bias between LLM judges and human experts. Our analysis reveals systematic inflation by LLM judges, strong reliability for cognitive attributes such as guidance and informativeness, reduced precision for empathy, and some unreliability in safety and relevance. Our contributions establish new methodological and empirical foundations for reliable, large-scale evaluation of LLMs in mental health. We release the benchmarks and codes at: this https URL 

---
# Re:Member: Emotional Question Generation from Personal Memories 

**Authors**: Zackary Rackauckas, Nobuaki Minematsu, Julia Hirschberg  

**Link**: [PDF](https://arxiv.org/pdf/2510.19030)  

**Abstract**: We present Re:Member, a system that explores how emotionally expressive, memory-grounded interaction can support more engaging second language (L2) learning. By drawing on users' personal videos and generating stylized spoken questions in the target language, Re:Member is designed to encourage affective recall and conversational engagement. The system aligns emotional tone with visual context, using expressive speech styles such as whispers or late-night tones to evoke specific moods. It combines WhisperX-based transcript alignment, 3-frame visual sampling, and Style-BERT-VITS2 for emotional synthesis within a modular generation pipeline. Designed as a stylized interaction probe, Re:Member highlights the role of affect and personal media in learner-centered educational technologies. 

---
# Are they lovers or friends? Evaluating LLMs' Social Reasoning in English and Korean Dialogues 

**Authors**: Eunsu Kim, Junyeong Park, Juhyun Oh, Kiwoong Park, Seyoung Song, A.Seza Dogruoz, Najoung Kim, Alice Oh  

**Link**: [PDF](https://arxiv.org/pdf/2510.19028)  

**Abstract**: As large language models (LLMs) are increasingly used in human-AI interactions, their social reasoning capabilities in interpersonal contexts are critical. We introduce SCRIPTS, a 1k-dialogue dataset in English and Korean, sourced from movie scripts. The task involves evaluating models' social reasoning capability to infer the interpersonal relationships (e.g., friends, sisters, lovers) between speakers in each dialogue. Each dialogue is annotated with probabilistic relational labels (Highly Likely, Less Likely, Unlikely) by native (or equivalent) Korean and English speakers from Korea and the U.S. Evaluating nine models on our task, current proprietary LLMs achieve around 75-80% on the English dataset, whereas their performance on Korean drops to 58-69%. More strikingly, models select Unlikely relationships in 10-25% of their responses. Furthermore, we find that thinking models and chain-of-thought prompting, effective for general reasoning, provide minimal benefits for social reasoning and occasionally amplify social biases. Our findings reveal significant limitations in current LLMs' social reasoning capabilities, highlighting the need for efforts to develop socially-aware language models. 

---
# Dynamic Evaluation for Oversensitivity in LLMs 

**Authors**: Sophia Xiao Pu, Sitao Cheng, Xin Eric Wang, William Yang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.19005)  

**Abstract**: Oversensitivity occurs when language models defensively reject prompts that are actually benign. This behavior not only disrupts user interactions but also obscures the boundary between harmful and harmless content. Existing benchmarks rely on static datasets that degrade overtime as models evolve, leading to data contamination and diminished evaluative power. To address this, we develop a framework that dynamically generates model-specific challenging datasets, capturing emerging defensive patterns and aligning with each model's unique behavior. Building on this approach, we construct OVERBENCH, a benchmark that aggregates these datasets across diverse LLM families, encompassing 450,000 samples from 25 models. OVERBENCH provides a dynamic and evolving perspective on oversensitivity, allowing for continuous monitoring of defensive triggers as models advance, highlighting vulnerabilities that static datasets overlook. 

---
# ProfBench: Multi-Domain Rubrics requiring Professional Knowledge to Answer and Judge 

**Authors**: Zhilin Wang, Jaehun Jung, Ximing Lu, Shizhe Diao, Ellie Evans, Jiaqi Zeng, Pavlo Molchanov, Yejin Choi, Jan Kautz, Yi Dong  

**Link**: [PDF](https://arxiv.org/pdf/2510.18941)  

**Abstract**: Evaluating progress in large language models (LLMs) is often constrained by the challenge of verifying responses, limiting assessments to tasks like mathematics, programming, and short-form question-answering. However, many real-world applications require evaluating LLMs in processing professional documents, synthesizing information, and generating comprehensive reports in response to user queries. We introduce ProfBench: a set of over 7000 response-criterion pairs as evaluated by human-experts with professional knowledge across Physics PhD, Chemistry PhD, Finance MBA and Consulting MBA. We build robust and affordable LLM-Judges to evaluate ProfBench rubrics, by mitigating self-enhancement bias and reducing the cost of evaluation by 2-3 orders of magnitude, to make it fair and accessible to the broader community. Our findings reveal that ProfBench poses significant challenges even for state-of-the-art LLMs, with top-performing models like GPT-5-high achieving only 65.9\% overall performance. Furthermore, we identify notable performance disparities between proprietary and open-weight models and provide insights into the role that extended thinking plays in addressing complex, professional-domain tasks. Data: this https URL and Code: this https URL 

---
# Lost in the Maze: Overcoming Context Limitations in Long-Horizon Agentic Search 

**Authors**: Howard Yen, Ashwin Paranjape, Mengzhou Xia, Thejas Venkatesh, Jack Hessel, Danqi Chen, Yuhao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.18939)  

**Abstract**: Long-horizon agentic search requires iteratively exploring the web over long trajectories and synthesizing information across many sources, and is the foundation for enabling powerful applications like deep research systems. In this work, we show that popular agentic search frameworks struggle to scale to long trajectories primarily due to context limitations-they accumulate long, noisy content, hit context window and tool budgets, or stop early. Then, we introduce SLIM (Simple Lightweight Information Management), a simple framework that separates retrieval into distinct search and browse tools, and periodically summarizes the trajectory, keeping context concise while enabling longer, more focused searches. On long-horizon tasks, SLIM achieves comparable performance at substantially lower cost and with far fewer tool calls than strong open-source baselines across multiple base models. Specifically, with o3 as the base model, SLIM achieves 56% on BrowseComp and 31% on HLE, outperforming all open-source frameworks by 8 and 4 absolute points, respectively, while incurring 4-6x fewer tool calls. Finally, we release an automated fine-grained trajectory analysis pipeline and error taxonomy for characterizing long-horizon agentic search frameworks; SLIM exhibits fewer hallucinations than prior systems. We hope our analysis framework and simple tool design inform future long-horizon agents. 

---
# Evaluating LLM Story Generation through Large-scale Network Analysis of Social Structures 

**Authors**: Hiroshi Nonaka, K. E. Perry  

**Link**: [PDF](https://arxiv.org/pdf/2510.18932)  

**Abstract**: Evaluating the creative capabilities of large language models (LLMs) in complex tasks often requires human assessments that are difficult to scale. We introduce a novel, scalable methodology for evaluating LLM story generation by analyzing underlying social structures in narratives as signed character networks. To demonstrate its effectiveness, we conduct a large-scale comparative analysis using networks from over 1,200 stories, generated by four leading LLMs (GPT-4o, GPT-4o mini, Gemini 1.5 Pro, and Gemini 1.5 Flash) and a human-written corpus. Our findings, based on network properties like density, clustering, and signed edge weights, show that LLM-generated stories consistently exhibit a strong bias toward tightly-knit, positive relationships, which aligns with findings from prior research using human assessment. Our proposed approach provides a valuable tool for evaluating limitations and tendencies in the creative storytelling of current and future LLMs. 

---
# Misinformation Detection using Large Language Models with Explainability 

**Authors**: Jainee Patel, Chintan Bhatt, Himani Trivedi, Thanh Thi Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2510.18918)  

**Abstract**: The rapid spread of misinformation on online platforms undermines trust among individuals and hinders informed decision making. This paper shows an explainable and computationally efficient pipeline to detect misinformation using transformer-based pretrained language models (PLMs). We optimize both RoBERTa and DistilBERT using a two-step strategy: first, we freeze the backbone and train only the classification head; then, we progressively unfreeze the backbone layers while applying layer-wise learning rate decay. On two real-world benchmark datasets, COVID Fake News and FakeNewsNet GossipCop, we test the proposed approach with a unified protocol of preprocessing and stratified splits. To ensure transparency, we integrate the Local Interpretable Model-Agnostic Explanations (LIME) at the token level to present token-level rationales and SHapley Additive exPlanations (SHAP) at the global feature attribution level. It demonstrates that DistilBERT achieves accuracy comparable to RoBERTa while requiring significantly less computational resources. This work makes two key contributions: (1) it quantitatively shows that a lightweight PLM can maintain task performance while substantially reducing computational cost, and (2) it presents an explainable pipeline that retrieves faithful local and global justifications without compromising performance. The results suggest that PLMs combined with principled fine-tuning and interpretability can be an effective framework for scalable, trustworthy misinformation detection. 

---
# MMAO-Bench: MultiModal All in One Benchmark Reveals Compositional Law between Uni-modal and Omni-modal in OmniModels 

**Authors**: Chen Chen, ZeYang Hu, Fengjiao Chen, Liya Ma, Jiaxing Liu, Xiaoyu Li, Xuezhi Cao  

**Link**: [PDF](https://arxiv.org/pdf/2510.18915)  

**Abstract**: Multimodal Large Languages models have been progressing from uni-modal understanding toward unifying visual, audio and language modalities, collectively termed omni models. However, the correlation between uni-modal and omni-modal remains unclear, which requires comprehensive evaluation to drive omni model's intelligence evolution. In this work, we propose a novel, high quality and diversity omni model benchmark, MultiModal All in One Benchmark (MMAO-Bench), which effectively assesses both uni-modal and omni-modal understanding capabilities. The benchmark consists of 1880 human curated samples, across 44 task types, and a innovative multi-step open-ended question type that better assess complex reasoning tasks. Experimental result shows the compositional law between cross-modal and uni-modal performance and the omni-modal capability manifests as a bottleneck effect on weak models, while exhibiting synergistic promotion on strong models. 

---
# Context-aware Fairness Evaluation and Mitigation in LLMs 

**Authors**: Afrozah Nadeem, Mark Dras, Usman Naseem  

**Link**: [PDF](https://arxiv.org/pdf/2510.18914)  

**Abstract**: Large language models often display undesirable behaviors embedded in their internal representations, undermining fairness, inconsistency drift, amplification of harmful content, and the propagation of unwanted patterns during extended dialogue and conversations. Although training-time or data-centric methods attempt to reduce these effects, they are computationally expensive, irreversible once deployed, and slow to adapt to new conversational contexts. Pruning-based methods provide a flexible and transparent way to reduce bias by adjusting the neurons responsible for certain behaviors. However, most existing approaches are static; once a neuron is removed, the model loses the ability to adapt when the conversation or context changes. To address this, we propose a dynamic, reversible, pruning-based framework that detects context-aware neuron activations and applies adaptive masking to modulate their influence during generation. Our inference-time solution provides fine-grained, memory-aware mitigation with knowledge-preserved, more coherent behavior across multilingual single- and multi-turn dialogues, enabling dynamic fairness control in real-world conversational AI. 

---
# Learning from the Best, Differently: A Diversity-Driven Rethinking on Data Selection 

**Authors**: Hongyi He, Xiao Liu, Zhenghao Lin, Mingni Tang, Yi Cheng, Jintao Wang, Wenjie Li, Peng Cheng, Yeyun Gong  

**Link**: [PDF](https://arxiv.org/pdf/2510.18909)  

**Abstract**: High-quality pre-training data is crutial for large language models, where quality captures factual reliability and semantic value, and diversity ensures broad coverage and distributional heterogeneity. Existing approaches typically rely on single or multiple-dimensional score-based selection. However, directly selecting top-scored data often degrades performance, and sampling from a broader range is required to recover results. The above non-monotonicity between dataset scores and downstream benchmark results reveals a fundamental bias: score-based methods collapse correlated dimensions, causing top-scored data to appear high-quality while systematically overlooking diversity. We argue that ensuring diversity requires decomposing correlated metrics into orthogonal feature dimensions, from which the top-scored data can be directly selected. Therefore, we proposed the Orthogonal Diversity-Aware Selection (ODiS) algorithm, which preserves both quality and diversity during data selection. First, ODiS evaluates data from multiple dimensions, covering language quality, knowledge quality, and comprehension difficulty. The multi-dimensional scores are then decorrelated via Principal Component Analysis (PCA), yielding orthogonal evaluation dimensions. For each dimension, a Roberta-based scorer is trained to regress the data onto PCA-projected scores, enabling scalable inference on large corpora. Finally, ODiS constructs the training dataset by selecting top-scored data within each orthogonal dimension, thereby ensuring both quality and diversity. Empirical results show that ODiS-selected data exhibit less than 2\% inter-dimension overlap, confirming orthogonality between dimensions. More importantly, models trained with ODiS-selected data significantly outperform other baselines on downstream benchmarks, highlighting the necessity of orthogonal, diversity-aware data selection for LLMs. 

---
# Improving Topic Modeling of Social Media Short Texts with Rephrasing: A Case Study of COVID-19 Related Tweets 

**Authors**: Wangjiaxuan Xin, Shuhua Yin, Shi Chen, Yaorong Ge  

**Link**: [PDF](https://arxiv.org/pdf/2510.18908)  

**Abstract**: Social media platforms such as Twitter (now X) provide rich data for analyzing public discourse, especially during crises such as the COVID-19 pandemic. However, the brevity, informality, and noise of social media short texts often hinder the effectiveness of traditional topic modeling, producing incoherent or redundant topics that are often difficult to interpret. To address these challenges, we have developed \emph{TM-Rephrase}, a model-agnostic framework that leverages large language models (LLMs) to rephrase raw tweets into more standardized and formal language prior to topic modeling. Using a dataset of 25,027 COVID-19-related Twitter posts, we investigate the effects of two rephrasing strategies, general- and colloquial-to-formal-rephrasing, on multiple topic modeling methods. Results demonstrate that \emph{TM-Rephrase} improves three metrics measuring topic modeling performance (i.e., topic coherence, topic uniqueness, and topic diversity) while reducing topic redundancy of most topic modeling algorithms, with the colloquial-to-formal strategy yielding the greatest performance gains and especially for the Latent Dirichlet Allocation (LDA) algorithm. This study contributes to a model-agnostic approach to enhancing topic modeling in public health related social media analysis, with broad implications for improved understanding of public discourse in health crisis as well as other important domains. 

---
# DuoLens: A Framework for Robust Detection of Machine-Generated Multilingual Text and Code 

**Authors**: Shriyansh Agrawal, Aidan Lau, Sanyam Shah, Ahan M R, Kevin Zhu, Sunishchal Dev, Vasu Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2510.18904)  

**Abstract**: The prevalence of Large Language Models (LLMs) for generating multilingual text and source code has only increased the imperative for machine-generated content detectors to be accurate and efficient across domains. Current detectors, predominantly utilizing zero-shot methods, such as Fast DetectGPT or GPTZero, either incur high computational cost or lack sufficient accuracy, often with a trade-off between the two, leaving room for further improvement. To address these gaps, we propose the fine-tuning of encoder-only Small Language Models (SLMs), in particular, the pre-trained models of RoBERTA and CodeBERTa using specialized datasets on source code and other natural language to prove that for the task of binary classification, SLMs outperform LLMs by a huge margin whilst using a fraction of compute. Our encoders achieve AUROC $= 0.97$ to $0.99$ and macro-F1 $0.89$ to $0.94$ while reducing latency by $8$-$12\times$ and peak VRAM by $3$-$5\times$ at $512$-token inputs. Under cross-generator shifts and adversarial transformations (paraphrase, back-translation; code formatting/renaming), performance retains $\geq 92%$ of clean AUROC. We release training and evaluation scripts with seeds and configs; a reproducibility checklist is also included. 

---
# Transformer-Based Low-Resource Language Translation: A Study on Standard Bengali to Sylheti 

**Authors**: Mangsura Kabir Oni, Tabia Tanzin Prama  

**Link**: [PDF](https://arxiv.org/pdf/2510.18898)  

**Abstract**: Machine Translation (MT) has advanced from rule-based and statistical methods to neural approaches based on the Transformer architecture. While these methods have achieved impressive results for high-resource languages, low-resource varieties such as Sylheti remain underexplored. In this work, we investigate Bengali-to-Sylheti translation by fine-tuning multilingual Transformer models and comparing them with zero-shot large language models (LLMs). Experimental results demonstrate that fine-tuned models significantly outperform LLMs, with mBART-50 achieving the highest translation adequacy and MarianMT showing the strongest character-level fidelity. These findings highlight the importance of task-specific adaptation for underrepresented languages and contribute to ongoing efforts toward inclusive language technologies. 

---
# When Models Can't Follow: Testing Instruction Adherence Across 256 LLMs 

**Authors**: Richard J. Young, Brandon Gillins, Alice M. Matthews  

**Link**: [PDF](https://arxiv.org/pdf/2510.18892)  

**Abstract**: Despite widespread deployment of Large Language Models, systematic evaluation of instruction-following capabilities remains challenging. While comprehensive benchmarks exist, focused assessments that quickly diagnose specific instruction adherence patterns are valuable. As newer models may be trained on existing benchmarks, novel evaluation approaches are needed to assess genuine capabilities rather than memorized performance. This paper presents a streamlined evaluation framework using twenty carefully designed prompts to assess LLM instruction-following across diverse task categories. We demonstrate this framework through a large-scale empirical study conducted on October 14, 2025, testing 256 verified working models from 331 available via OpenRouter. To ensure methodological rigor and prevent selection bias, we first verified each model's basic functionality before inclusion. Unlike large-scale benchmarks requiring extensive computational resources, our approach offers a practical diagnostic tool researchers and practitioners can readily apply. Our methodology builds upon verifiable instructions while introducing a compact test suite balancing comprehensiveness with efficiency. Each prompt targets distinct aspects of instruction following, including format compliance, content constraints, logical sequencing, and multi-step task execution. We evaluate models from major providers (OpenAI, Anthropic, Google, Meta, Mistral) and emerging implementations (Qwen, DeepSeek, community models), providing comparative performance analysis. Our findings reveal consistent failure modes and identify specific instruction types posing particular challenges. This work contributes both a practical evaluation tool and one of the most comprehensive empirical analyses of instruction-following capabilities across the contemporary LLM landscape. 

---
# Small Language Models Offer Significant Potential for Science Community 

**Authors**: Jian Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.18890)  

**Abstract**: Recent advancements in natural language processing, particularly with large language models (LLMs), are transforming how scientists engage with the literature. While the adoption of LLMs is increasing, concerns remain regarding potential information biases and computational costs. Rather than LLMs, I developed a framework to evaluate the feasibility of precise, rapid, and cost-effective information retrieval from extensive geoscience literature using freely available small language models (MiniLMs). A curated corpus of approximately 77 million high-quality sentences, extracted from 95 leading peer-reviewed geoscience journals such as Geophysical Research Letters and Earth and Planetary Science Letters published during years 2000 to 2024, was constructed. MiniLMs enable a computationally efficient approach for extracting relevant domain-specific information from these corpora through semantic search techniques and sentence-level indexing. This approach, unlike LLMs such as ChatGPT-4 that often produces generalized responses, excels at identifying substantial amounts of expert-verified information with established, multi-disciplinary sources, especially for information with quantitative findings. Furthermore, by analyzing emotional tone via sentiment analysis and topical clusters through unsupervised clustering within sentences, MiniLM provides a powerful tool for tracking the evolution of conclusions, research priorities, advancements, and emerging questions within geoscience communities. Overall, MiniLM holds significant potential within the geoscience community for applications such as fact and image retrievals, trend analyses, contradiction analyses, and educational purposes. 

---
# Contextual Augmentation for Entity Linking using Large Language Models 

**Authors**: Daniel Vollmers, Hamada M. Zahera, Diego Moussallem, Axel-Cyrille Ngonga Ngomo  

**Link**: [PDF](https://arxiv.org/pdf/2510.18888)  

**Abstract**: Entity Linking involves detecting and linking entity mentions in natural language texts to a knowledge graph. Traditional methods use a two-step process with separate models for entity recognition and disambiguation, which can be computationally intensive and less effective. We propose a fine-tuned model that jointly integrates entity recognition and disambiguation in a unified framework. Furthermore, our approach leverages large language models to enrich the context of entity mentions, yielding better performance in entity disambiguation. We evaluated our approach on benchmark datasets and compared with several baselines. The evaluation results show that our approach achieves state-of-the-art performance on out-of-domain datasets. 

---
# olmOCR 2: Unit Test Rewards for Document OCR 

**Authors**: Jake Poznanski, Luca Soldaini, Kyle Lo  

**Link**: [PDF](https://arxiv.org/pdf/2510.19817)  

**Abstract**: We present olmOCR 2, the latest in our family of powerful OCR systems for converting digitized print documents, like PDFs, into clean, naturally ordered plain text. olmOCR 2 is powered by olmOCR-2-7B-1025, a specialized, 7B vision language model (VLM) trained using reinforcement learning with verifiable rewards (RLVR), where our rewards are a diverse set of binary unit tests. To scale unit test creation, we develop a pipeline for generating synthetic documents with diverse and challenging layouts, known ground-truth HTML source code, and extracted test cases. We show that RL training on these test cases results in state-of-the-art performance on olmOCR-Bench, our English-language OCR benchmark, with the largest improvements in math formula conversion, table parsing, and multi-column layouts compared to previous versions. We release our model, data and code under permissive open licenses. 

---
# Pico-Banana-400K: A Large-Scale Dataset for Text-Guided Image Editing 

**Authors**: Yusu Qian, Eli Bocek-Rivele, Liangchen Song, Jialing Tong, Yinfei Yang, Jiasen Lu, Wenze Hu, Zhe Gan  

**Link**: [PDF](https://arxiv.org/pdf/2510.19808)  

**Abstract**: Recent advances in multimodal models have demonstrated remarkable text-guided image editing capabilities, with systems like GPT-4o and Nano-Banana setting new benchmarks. However, the research community's progress remains constrained by the absence of large-scale, high-quality, and openly accessible datasets built from real images. We introduce Pico-Banana-400K, a comprehensive 400K-image dataset for instruction-based image editing. Our dataset is constructed by leveraging Nano-Banana to generate diverse edit pairs from real photographs in the OpenImages collection. What distinguishes Pico-Banana-400K from previous synthetic datasets is our systematic approach to quality and diversity. We employ a fine-grained image editing taxonomy to ensure comprehensive coverage of edit types while maintaining precise content preservation and instruction faithfulness through MLLM-based quality scoring and careful curation. Beyond single turn editing, Pico-Banana-400K enables research into complex editing scenarios. The dataset includes three specialized subsets: (1) a 72K-example multi-turn collection for studying sequential editing, reasoning, and planning across consecutive modifications; (2) a 56K-example preference subset for alignment research and reward model training; and (3) paired long-short editing instructions for developing instruction rewriting and summarization capabilities. By providing this large-scale, high-quality, and task-rich resource, Pico-Banana-400K establishes a robust foundation for training and benchmarking the next generation of text-guided image editing models. 

---
# Blackbox Model Provenance via Palimpsestic Membership Inference 

**Authors**: Rohith Kuditipudi, Jing Huang, Sally Zhu, Diyi Yang, Christopher Potts, Percy Liang  

**Link**: [PDF](https://arxiv.org/pdf/2510.19796)  

**Abstract**: Suppose Alice trains an open-weight language model and Bob uses a blackbox derivative of Alice's model to produce text. Can Alice prove that Bob is using her model, either by querying Bob's derivative model (query setting) or from the text alone (observational setting)? We formulate this question as an independence testing problem--in which the null hypothesis is that Bob's model or text is independent of Alice's randomized training run--and investigate it through the lens of palimpsestic memorization in language models: models are more likely to memorize data seen later in training, so we can test whether Bob is using Alice's model using test statistics that capture correlation between Bob's model or text and the ordering of training examples in Alice's training run. If Alice has randomly shuffled her training data, then any significant correlation amounts to exactly quantifiable statistical evidence against the null hypothesis, regardless of the composition of Alice's training data. In the query setting, we directly estimate (via prompting) the likelihood Bob's model gives to Alice's training examples and order; we correlate the likelihoods of over 40 fine-tunes of various Pythia and OLMo base models ranging from 1B to 12B parameters with the base model's training data order, achieving a p-value on the order of at most 1e-8 in all but six cases. In the observational setting, we try two approaches based on estimating 1) the likelihood of Bob's text overlapping with spans of Alice's training examples and 2) the likelihood of Bob's text with respect to different versions of Alice's model we obtain by repeating the last phase (e.g., 1%) of her training run on reshuffled data. The second approach can reliably distinguish Bob's text from as little as a few hundred tokens; the first does not involve any retraining but requires many more tokens (several hundred thousand) to achieve high power. 

---
# GaLLoP: Gradient-based Sparse Learning on Low-Magnitude Parameters 

**Authors**: Anand Choudhary, Yasser Sulaıman, Lukas Mauch, Ghouthi Boukli Hacene, Fabien Cardinaux, Antoine Bosselut  

**Link**: [PDF](https://arxiv.org/pdf/2510.19778)  

**Abstract**: Sparse fine-tuning techniques adapt LLMs to downstream tasks by only tuning a sparse subset of model parameters. However, the effectiveness of sparse adaptation depends on optimally selecting the model parameters to be fine-tuned. In this work, we introduce a novel sparse fine-tuning technique named GaLLoP: Gradient-based Sparse Learning on Low-Magnitude Parameters, which fine-tunes only those model parameters which have the largest gradient magnitudes on downstream tasks and the smallest pre-trained magnitudes, intuitively prioritizing parameters that are highly task-relevant, but minimally disruptive to pre-trained knowledge. Our experimentation with LLaMA3 8B and Gemma 2B as base models shows that GaLLoP consistently improves or matches the in-distribution as well as out-of-distribution performance obtained via the usage of other leading parameter-efficient fine-tuning techniques, including LoRA, DoRA, and SAFT. Our analysis demonstrates that GaLLoP mitigates catastrophic forgetting and memorization of task data, as important pre-trained parameters remain unchanged, and stabilizes performance relative to other fine-tuning techniques, robustly generalizing across most random seeds. 

---
# From Forecasting to Planning: Policy World Model for Collaborative State-Action Prediction 

**Authors**: Zhida Zhao, Talas Fu, Yifan Wang, Lijun Wang, Huchuan Lu  

**Link**: [PDF](https://arxiv.org/pdf/2510.19654)  

**Abstract**: Despite remarkable progress in driving world models, their potential for autonomous systems remains largely untapped: the world models are mostly learned for world simulation and decoupled from trajectory planning. While recent efforts aim to unify world modeling and planning in a single framework, the synergistic facilitation mechanism of world modeling for planning still requires further exploration. In this work, we introduce a new driving paradigm named Policy World Model (PWM), which not only integrates world modeling and trajectory planning within a unified architecture, but is also able to benefit planning using the learned world knowledge through the proposed action-free future state forecasting scheme. Through collaborative state-action prediction, PWM can mimic the human-like anticipatory perception, yielding more reliable planning performance. To facilitate the efficiency of video forecasting, we further introduce a dynamically enhanced parallel token generation mechanism, equipped with a context-guided tokenizer and an adaptive dynamic focal loss. Despite utilizing only front camera input, our method matches or exceeds state-of-the-art approaches that rely on multi-view and multi-modal inputs. Code and model weights will be released at this https URL. 

---
# HSCodeComp: A Realistic and Expert-level Benchmark for Deep Search Agents in Hierarchical Rule Application 

**Authors**: Yiqian Yang, Tian Lan, Qianghuai Jia, Li Zhu, Hui Jiang, Hang Zhu, Longyue Wang, Weihua Luo, Kaifu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.19631)  

**Abstract**: Effective deep search agents must not only access open-domain and domain-specific knowledge but also apply complex rules-such as legal clauses, medical manuals and tariff rules. These rules often feature vague boundaries and implicit logic relationships, making precise application challenging for agents. However, this critical capability is largely overlooked by current agent benchmarks.
To fill this gap, we introduce HSCodeComp, the first realistic, expert-level e-commerce benchmark designed to evaluate deep search agents in hierarchical rule application. In this task, the deep reasoning process of agents is guided by these rules to predict 10-digit Harmonized System Code (HSCode) of products with noisy but realistic descriptions. These codes, established by the World Customs Organization, are vital for global supply chain efficiency. Built from real-world data collected from large-scale e-commerce platforms, our proposed HSCodeComp comprises 632 product entries spanning diverse product categories, with these HSCodes annotated by several human experts.
Extensive experimental results on several state-of-the-art LLMs, open-source, and closed-source agents reveal a huge performance gap: best agent achieves only 46.8% 10-digit accuracy, far below human experts at 95.0%. Besides, detailed analysis demonstrates the challenges of hierarchical rule application, and test-time scaling fails to improve performance further. 

---
# Human-Agent Collaborative Paper-to-Page Crafting for Under $0.1 

**Authors**: Qianli Ma, Siyu Wang, Yilin Chen, Yinhao Tang, Yixiang Yang, Chang Guo, Bingjie Gao, Zhening Xing, Yanan Sun, Zhipeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.19600)  

**Abstract**: In the quest for scientific progress, communicating research is as vital as the discovery itself. Yet, researchers are often sidetracked by the manual, repetitive chore of building project webpages to make their dense papers accessible. While automation has tackled static slides and posters, the dynamic, interactive nature of webpages has remained an unaddressed challenge. To bridge this gap, we reframe the problem, arguing that the solution lies not in a single command, but in a collaborative, hierarchical process. We introduce $\textbf{AutoPage}$, a novel multi-agent system that embodies this philosophy. AutoPage deconstructs paper-to-page creation into a coarse-to-fine pipeline from narrative planning to multimodal content generation and interactive rendering. To combat AI hallucination, dedicated "Checker" agents verify each step against the source paper, while optional human checkpoints ensure the final product aligns perfectly with the author's vision, transforming the system from a mere tool into a powerful collaborative assistant. To rigorously validate our approach, we also construct $\textbf{PageBench}$, the first benchmark for this new task. Experiments show AutoPage not only generates high-quality, visually appealing pages but does so with remarkable efficiency in under 15 minutes for less than \$0.1. Code and dataset will be released at $\href{this https URL}{Webpage}$. 

---
# [De|Re]constructing VLMs' Reasoning in Counting 

**Authors**: Simone Alghisi, Gabriel Roccabruna, Massimo Rizzoli, Seyed Mahed Mousavi, Giuseppe Riccardi  

**Link**: [PDF](https://arxiv.org/pdf/2510.19555)  

**Abstract**: Vision-Language Models (VLMs) have recently gained attention due to their competitive performance on multiple downstream tasks, achieved by following user-input instructions. However, VLMs still exhibit several limitations in visual reasoning, such as difficulties in identifying relations (e.g., spatial, temporal, and among objects), understanding temporal sequences (e.g., frames), and counting objects. In this work, we go beyond score-level benchmark evaluations of VLMs by investigating the underlying causes of their failures and proposing a targeted approach to improve their reasoning capabilities. We study the reasoning skills of seven state-of-the-art VLMs in the counting task under controlled experimental conditions. Our experiments show that VLMs are highly sensitive to the number and type of objects, their spatial arrangement, and the co-occurrence of distractors. A layer-wise analysis reveals that errors are due to incorrect mapping of the last-layer representation into the output space. Our targeted training shows that fine-tuning just the output layer improves accuracy by up to 21%. We corroborate these findings by achieving consistent improvements on real-world datasets. 

---
# LLM Unlearning with LLM Beliefs 

**Authors**: Kemou Li, Qizhou Wang, Yue Wang, Fengpeng Li, Jun Liu, Bo Han, Jiantao Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2510.19422)  

**Abstract**: Large language models trained on vast corpora inherently risk memorizing sensitive or harmful content, which may later resurface in their outputs. Prevailing unlearning methods generally rely on gradient ascent and its variants to lower the probability of specific target responses. However, we find that this strategy induces a critical side effect: probability mass is redistributed into high-likelihood regions, often corresponding to semantically related rephrasings of the targets. We refer to this as the squeezing effect, which explains why many methods yield merely spurious unlearning, a problem further obscured by automated metrics (e.g., ROUGE, truth ratio) that misreport actual success. To address this, we propose a bootstrapping (BS) framework that explicitly links the squeezing effect with the model's own high-confidence generations, namely its model beliefs. Since model beliefs inherently capture the very high-likelihood regions where probability mass is squeezed, incorporating them into the unlearning objective directly counters the squeezing effect. By jointly suppressing both target responses and model beliefs, BS-T (token) attenuates high-probability tokens, whereas BS-S (sequence) removes entire high-confidence generations, together achieving more thorough forgetting while preserving utility. Extensive experiments across diverse benchmarks with various model families confirm the effectiveness of our approach. 

---
# ColorAgent: Building A Robust, Personalized, and Interactive OS Agent 

**Authors**: Ning Li, Qiqiang Lin, Zheng Wu, Xiaoyun Mo, Weiming Zhang, Yin Zhao, Xiangmou Qu, Jiamu Zhou, Jun Wang, Congmin Zheng, Yuanyi Song, Hongjiang Chen, Heyuan Huang, Jihong Wang, Jiaxin Yin, Jingwei Yu, Junwei Liao, Qiuying Peng, Xingyu Lou, Jun Wang, Weiwen Liu, Zhuosheng Zhang, Weinan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.19386)  

**Abstract**: With the advancements in hardware, software, and large language model technologies, the interaction between humans and operating systems has evolved from the command-line interface to the rapidly emerging AI agent interactions. Building an operating system (OS) agent capable of executing user instructions and faithfully following user desires is becoming a reality. In this technical report, we present ColorAgent, an OS agent designed to engage in long-horizon, robust interactions with the environment while also enabling personalized and proactive user interaction. To enable long-horizon interactions with the environment, we enhance the model's capabilities through step-wise reinforcement learning and self-evolving training, while also developing a tailored multi-agent framework that ensures generality, consistency, and robustness. In terms of user interaction, we explore personalized user intent recognition and proactive engagement, positioning the OS agent not merely as an automation tool but as a warm, collaborative partner. We evaluate ColorAgent on the AndroidWorld and AndroidLab benchmarks, achieving success rates of 77.2% and 50.7%, respectively, establishing a new state of the art. Nonetheless, we note that current benchmarks are insufficient for a comprehensive evaluation of OS agents and propose further exploring directions in future work, particularly in the areas of evaluation paradigms, agent collaboration, and security. Our code is available at this https URL. 

---
# Every Attention Matters: An Efficient Hybrid Architecture for Long-Context Reasoning 

**Authors**: Ling Team, Bin Han, Caizhi Tang, Chen Liang, Donghao Zhang, Fan Yuan, Feng Zhu, Jie Gao, Jingyu Hu, Longfei Li, Meng Li, Mingyang Zhang, Peijie Jiang, Peng Jiao, Qian Zhao, Qingyuan Yang, Wenbo Shen, Xinxing Yang, Yalin Zhang, Yankun Ren, Yao Zhao, Yibo Cao, Yixuan Sun, Yue Zhang, Yuchen Fang, Zibin Lin, Zixuan Cheng, Jun Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2510.19338)  

**Abstract**: In this technical report, we present the Ring-linear model series, specifically including Ring-mini-linear-2.0 and Ring-flash-linear-2.0. Ring-mini-linear-2.0 comprises 16B parameters and 957M activations, while Ring-flash-linear-2.0 contains 104B parameters and 6.1B activations. Both models adopt a hybrid architecture that effectively integrates linear attention and softmax attention, significantly reducing I/O and computational overhead in long-context inference scenarios. Compared to a 32 billion parameter dense model, this series reduces inference cost to 1/10, and compared to the original Ring series, the cost is also reduced by over 50%. Furthermore, through systematic exploration of the ratio between different attention mechanisms in the hybrid architecture, we have identified the currently optimal model structure. Additionally, by leveraging our self-developed high-performance FP8 operator library-linghe, overall training efficiency has been improved by 50%. Benefiting from the high alignment between the training and inference engine operators, the models can undergo long-term, stable, and highly efficient optimization during the reinforcement learning phase, consistently maintaining SOTA performance across multiple challenging complex reasoning benchmarks. 

---
# Aligning Multilingual News for Stock Return Prediction 

**Authors**: Yuntao Wu, Lynn Tao, Ing-Haw Cheng, Charles Martineau, Yoshio Nozawa, John Hull, Andreas Veneris  

**Link**: [PDF](https://arxiv.org/pdf/2510.19203)  

**Abstract**: News spreads rapidly across languages and regions, but translations may lose subtle nuances. We propose a method to align sentences in multilingual news articles using optimal transport, identifying semantically similar content across languages. We apply this method to align more than 140,000 pairs of Bloomberg English and Japanese news articles covering around 3500 stocks in Tokyo exchange over 2012-2024. Aligned sentences are sparser, more interpretable, and exhibit higher semantic similarity. Return scores constructed from aligned sentences show stronger correlations with realized stock returns, and long-short trading strategies based on these alignments achieve 10\% higher Sharpe ratios than analyzing the full text sample. 

---
# The Zero-Step Thinking: An Empirical Study of Mode Selection as Harder Early Exit in Reasoning Models 

**Authors**: Yuqiao Tan, Shizhu He, Kang Liu, Jun Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2510.19176)  

**Abstract**: Reasoning models have demonstrated exceptional performance in tasks such as mathematics and logical reasoning, primarily due to their ability to engage in step-by-step thinking during the reasoning process. However, this often leads to overthinking, resulting in unnecessary computational overhead. To address this issue, Mode Selection aims to automatically decide between Long-CoT (Chain-of-Thought) or Short-CoT by utilizing either a Thinking or NoThinking mode. Simultaneously, Early Exit determines the optimal stopping point during the iterative reasoning process. Both methods seek to reduce the computational burden. In this paper, we first identify Mode Selection as a more challenging variant of the Early Exit problem, as they share similar objectives but differ in decision timing. While Early Exit focuses on determining the best stopping point for concise reasoning at inference time, Mode Selection must make this decision at the beginning of the reasoning process, relying on pre-defined fake thoughts without engaging in an explicit reasoning process, referred to as zero-step thinking. Through empirical studies on nine baselines, we observe that prompt-based approaches often fail due to their limited classification capabilities when provided with minimal hand-crafted information. In contrast, approaches that leverage internal information generally perform better across most scenarios but still exhibit issues with stability. Our findings indicate that existing methods relying solely on the information provided by models are insufficient for effectively addressing Mode Selection in scenarios with limited information, highlighting the ongoing challenges of this task. Our code is available at this https URL. 

---
# OpenGuardrails: An Open-Source Context-Aware AI Guardrails Platform 

**Authors**: Thomas Wang, Haowen Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.19169)  

**Abstract**: As large language models (LLMs) become increasingly integrated into real-world applications, safeguarding them against unsafe, malicious, or privacy-violating content is critically important. We present OpenGuardrails, the first open-source project to provide both a context-aware safety and manipulation detection model and a deployable platform for comprehensive AI guardrails. OpenGuardrails protects against content-safety risks, model-manipulation attacks (e.g., prompt injection, jailbreaking, code-interpreter abuse, and the generation/execution of malicious code), and data leakage. Content-safety and model-manipulation detection are implemented by a unified large model, while data-leakage identification and redaction are performed by a separate lightweight NER pipeline (e.g., Presidio-style models or regex-based detectors). The system can be deployed as a security gateway or an API-based service, with enterprise-grade, fully private deployment options. OpenGuardrails achieves state-of-the-art (SOTA) performance on safety benchmarks, excelling in both prompt and response classification across English, Chinese, and multilingual tasks. All models are released under the Apache 2.0 license for public use. 

---
# A Multi-faceted Analysis of Cognitive Abilities: Evaluating Prompt Methods with Large Language Models on the CONSORT Checklist 

**Authors**: Sohyeon Jeon, Hyung-Chul Lee  

**Link**: [PDF](https://arxiv.org/pdf/2510.19139)  

**Abstract**: Despite the rapid expansion of Large Language Models (LLMs) in healthcare, the ability of these systems to assess clinical trial reporting according to CONSORT standards remains unclear, particularly with respect to their cognitive and reasoning strategies. This study applies a behavioral and metacognitive analytic approach with expert-validated data, systematically comparing two representative LLMs under three prompt conditions. Clear differences emerged in how the models approached various CONSORT items, and prompt types, including shifts in reasoning style, explicit uncertainty, and alternative interpretations shaped response patterns. Our results highlight the current limitations of these systems in clinical compliance automation and underscore the importance of understanding their cognitive adaptations and strategic behavior in developing more explainable and reliable medical AI. 

---
# PoSh: Using Scene Graphs To Guide LLMs-as-a-Judge For Detailed Image Descriptions 

**Authors**: Amith Ananthram, Elias Stengel-Eskin, Lorena A. Bradford, Julia Demarest, Adam Purvis, Keith Krut, Robert Stein, Rina Elster Pantalony, Mohit Bansal, Kathleen McKeown  

**Link**: [PDF](https://arxiv.org/pdf/2510.19060)  

**Abstract**: While vision-language models (VLMs) have advanced into detailed image description, evaluation remains a challenge. Standard metrics (e.g. CIDEr, SPICE) were designed for short texts and tuned to recognize errors that are now uncommon, such as object misidentification. In contrast, long texts require sensitivity to attribute and relation attachments and scores that localize errors to particular text spans. In this work, we introduce PoSh, a metric for detailed image description that uses scene graphs as structured rubrics to guide LLMs-as-a-Judge, producing aggregate scores grounded in fine-grained errors (e.g. mistakes in compositional understanding). PoSh is replicable, interpretable and a better proxy for human raters than existing metrics (including GPT4o-as-a-Judge). To validate PoSh, we introduce a challenging new dataset, DOCENT. This novel benchmark contains artwork, paired with expert-written references, and model-generated descriptions, augmented with granular and coarse judgments of their quality from art history students. Thus, DOCENT enables evaluating both detailed image description metrics and detailed image description itself in a challenging new domain. We show that PoSh achieves stronger correlations (+0.05 Spearman $\rho$) with the human judgments in DOCENT than the best open-weight alternatives, is robust to image type (using CapArena, an existing dataset of web imagery) and is a capable reward function, outperforming standard supervised fine-tuning. Then, using PoSh, we characterize the performance of open and closed models in describing the paintings, sketches and statues in DOCENT and find that foundation models struggle to achieve full, error-free coverage of images with rich scene dynamics, establishing a demanding new task to gauge VLM progress. Through both PoSh and DOCENT, we hope to enable advances in important areas such as assistive text generation. 

---
# NeuroAda: Activating Each Neuron's Potential for Parameter-Efficient Fine-Tuning 

**Authors**: Zhi Zhang, Yixian Shen, Congfeng Cao, Ekaterina Shutova  

**Link**: [PDF](https://arxiv.org/pdf/2510.18940)  

**Abstract**: Existing parameter-efficient fine-tuning (PEFT) methods primarily fall into two categories: addition-based and selective in-situ adaptation. The former, such as LoRA, introduce additional modules to adapt the model to downstream tasks, offering strong memory efficiency. However, their representational capacity is often limited, making them less suitable for fine-grained adaptation. In contrast, the latter directly fine-tunes a carefully chosen subset of the original model parameters, allowing for more precise and effective adaptation, but at the cost of significantly increased memory consumption. To reconcile this trade-off, we propose NeuroAda, a novel PEFT method that enables fine-grained model finetuning while maintaining high memory efficiency. Our approach first identifies important parameters (i.e., connections within the network) as in selective adaptation, and then introduces bypass connections for these selected parameters. During finetuning, only the bypass connections are updated, leaving the original model parameters frozen. Empirical results on 23+ tasks spanning both natural language generation and understanding demonstrate that NeuroAda achieves state-of-the-art performance with as little as $\leq \textbf{0.02}\%$ trainable parameters, while reducing CUDA memory usage by up to 60%. We release our code here: this https URL. 

---
# StutterZero and StutterFormer: End-to-End Speech Conversion for Stuttering Transcription and Correction 

**Authors**: Qianheng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2510.18938)  

**Abstract**: Over 70 million people worldwide experience stuttering, yet most automatic speech systems misinterpret disfluent utterances or fail to transcribe them accurately. Existing methods for stutter correction rely on handcrafted feature extraction or multi-stage automatic speech recognition (ASR) and text-to-speech (TTS) pipelines, which separate transcription from audio reconstruction and often amplify distortions. This work introduces StutterZero and StutterFormer, the first end-to-end waveform-to-waveform models that directly convert stuttered speech into fluent speech while jointly predicting its transcription. StutterZero employs a convolutional-bidirectional LSTM encoder-decoder with attention, whereas StutterFormer integrates a dual-stream Transformer with shared acoustic-linguistic representations. Both architectures are trained on paired stuttered-fluent data synthesized from the SEP-28K and LibriStutter corpora and evaluated on unseen speakers from the FluencyBank dataset. Across all benchmarks, StutterZero had a 24% decrease in Word Error Rate (WER) and a 31% improvement in semantic similarity (BERTScore) compared to the leading Whisper-Medium model. StutterFormer achieved better results, with a 28% decrease in WER and a 34% improvement in BERTScore. The results validate the feasibility of direct end-to-end stutter-to-fluent speech conversion, offering new opportunities for inclusive human-computer interaction, speech therapy, and accessibility-oriented AI systems. 

---
# BAPO: Stabilizing Off-Policy Reinforcement Learning for LLMs via Balanced Policy Optimization with Adaptive Clipping 

**Authors**: Zhiheng Xi, Xin Guo, Yang Nan, Enyu Zhou, Junrui Shen, Wenxiang Chen, Jiaqi Liu, Jixuan Huang, Zhihao Zhang, Honglin Guo, Xun Deng, Zhikai Lei, Miao Zheng, Guoteng Wang, Shuo Zhang, Peng Sun, Rui Zheng, Hang Yan, Tao Gui, Qi Zhang, Xuanjing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2510.18927)  

**Abstract**: Reinforcement learning (RL) has recently become the core paradigm for aligning and strengthening large language models (LLMs). Yet, applying RL in off-policy settings--where stale data from past policies are used for training--improves sample efficiency, but remains challenging: policy entropy declines sharply, optimization often becomes unstable and may even collapse. Through theoretical and empirical analysis, we identify two key insights: (i) an imbalance in optimization, where negative-advantage samples dominate the policy gradient, suppressing useful behaviors and risking gradient explosions; and (ii) the derived Entropy-Clip Rule, which reveals that the fixed clipping mechanism in PPO-like objectives systematically blocks entropy-increasing updates, thereby driving the policy toward over-exploitation at the expense of exploration. Building on these insights, we propose BAlanced Policy Optimization with Adaptive Clipping (BAPO), a simple yet effective method that dynamically adjusts clipping bounds to adaptively re-balance positive and negative contributions, preserve entropy, and stabilize RL optimization. Across diverse off-policy scenarios--including sample replay and partial rollout--BAPO achieves fast, stable, and data-efficient training. On AIME 2024 and AIME 2025 benchmarks, our 7B BAPO model surpasses open-source counterparts such as SkyWork-OR1-7B, while our 32B BAPO model not only achieves state-of-the-art results among models of the same scale but also outperforms leading proprietary systems like o3-mini and Gemini-2.5-Flash-Thinking. 

---
# Benchmarking On-Device Machine Learning on Apple Silicon with MLX 

**Authors**: Oluwaseun A. Ajayi, Ogundepo Odunayo  

**Link**: [PDF](https://arxiv.org/pdf/2510.18921)  

**Abstract**: The recent widespread adoption of Large Language Models (LLMs) and machine learning in general has sparked research interest in exploring the possibilities of deploying these models on smaller devices such as laptops and mobile phones. This creates a need for frameworks and approaches that are capable of taking advantage of on-device hardware. The MLX framework was created to address this need. It is a framework optimized for machine learning (ML) computations on Apple silicon devices, facilitating easier research, experimentation, and prototyping.
This paper presents a performance evaluation of MLX, focusing on inference latency of transformer models. We compare the performance of different transformer architecture implementations in MLX with their Pytorch counterparts. For this research we create a framework called MLX-transformers which includes different transformer implementations in MLX and downloads the model checkpoints in pytorch and converts it to the MLX format. By leveraging the advanced architecture and capabilities of Apple Silicon, MLX-Transformers enables seamless execution of transformer models directly sourced from Hugging Face, eliminating the need for checkpoint conversion often required when porting models between frameworks.
Our study benchmarks different transformer models on two Apple Silicon macbook devices against an NVIDIA CUDA GPU. Specifically, we compare the inference latency performance of models with the same parameter sizes and checkpoints. We evaluate the performance of BERT, RoBERTa, and XLM-RoBERTa models, with the intention of extending future work to include models of different modalities, thus providing a more comprehensive assessment of MLX's capabilities. The results highlight MLX's potential in enabling efficient and more accessible on-device ML applications within Apple's ecosystem. 

---
# Towards Better Health Conversations: The Benefits of Context-seeking 

**Authors**: Rory Sayres, Yuexing Hao, Abbi Ward, Amy Wang, Beverly Freeman, Serena Zhan, Diego Ardila, Jimmy Li, I-Ching Lee, Anna Iurchenko, Siyi Kou, Kartikeya Badola, Jimmy Hu, Bhawesh Kumar, Keith Johnson, Supriya Vijay, Justin Krogue, Avinatan Hassidim, Yossi Matias, Dale R. Webster, Sunny Virmani, Yun Liu, Quang Duong, Mike Schaekermann  

**Link**: [PDF](https://arxiv.org/pdf/2510.18880)  

**Abstract**: Navigating health questions can be daunting in the modern information landscape. Large language models (LLMs) may provide tailored, accessible information, but also risk being inaccurate, biased or misleading. We present insights from 4 mixed-methods studies (total N=163), examining how people interact with LLMs for their own health questions. Qualitative studies revealed the importance of context-seeking in conversational AIs to elicit specific details a person may not volunteer or know to share. Context-seeking by LLMs was valued by participants, even if it meant deferring an answer for several turns. Incorporating these insights, we developed a "Wayfinding AI" to proactively solicit context. In a randomized, blinded study, participants rated the Wayfinding AI as more helpful, relevant, and tailored to their concerns compared to a baseline AI. These results demonstrate the strong impact of proactive context-seeking on conversational dynamics, and suggest design patterns for conversational AI to help navigate health topics. 

---
