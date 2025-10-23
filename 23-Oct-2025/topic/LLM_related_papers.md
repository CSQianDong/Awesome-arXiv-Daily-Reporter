# SmartSwitch: Advancing LLM Reasoning by Overcoming Underthinking via Promoting Deeper Thought Exploration 

**Authors**: Xichen Zhang, Sitong Wu, Haoru Tan, Shaozuo Yu, Yinghao Zhu, Ziyi He, Jiaya Jia  

**Link**: [PDF](https://arxiv.org/pdf/2510.19767)  

**Abstract**: The long chain-of-thought (LongCoT) capability is central to the recent breakthroughs achieved by large language models in complex reasoning tasks. However, the accompanying issue of ''underthinking'', where models exhibit shallow reasoning by frequently switching thoughts without sufficient exploration, limits both performance and token efficiency. To address this problem, we propose a simple yet effective reasoning strategy: the SmartSwitch inference framework. This framework can be easily integrated into any large language model as a plug-and-play solution, continuously monitoring the model's reasoning process to detect underthinking and guide it toward deeper exploration of promising but overlooked thoughts. Specifically, the perception module identifies points where thoughts switch and evaluates the potential of the preceding thought using an off-the-shelf process reward model (PRM). If a high-potential thought is found to be prematurely abandoned, the intervention module interrupts the ongoing inference, backtracks to the point before the switch, and inserts a "deepening prompt" to encourage further exploration along that promising path. Extensive experiments on challenging mathematical reasoning benchmarks demonstrate that our method significantly enhances the performance of various large language models of different sizes. 

---
# Scaf-GRPO: Scaffolded Group Relative Policy Optimization for Enhancing LLM Reasoning 

**Authors**: Xichen Zhang, Sitong Wu, Yinghao Zhu, Haoru Tan, Shaozuo Yu, Ziyi He, Jiaya Jia  

**Link**: [PDF](https://arxiv.org/pdf/2510.19807)  

**Abstract**: Reinforcement learning from verifiable rewards has emerged as a powerful technique for enhancing the complex reasoning abilities of Large Language Models (LLMs). However, these methods are fundamentally constrained by the ''learning cliff'' phenomenon: when faced with problems far beyond their current capabilities, models consistently fail, yielding a persistent zero-reward signal. In policy optimization algorithms like GRPO, this collapses the advantage calculation to zero, rendering these difficult problems invisible to the learning gradient and stalling progress. To overcome this, we introduce Scaf-GRPO (Scaffolded Group Relative Policy Optimization), a progressive training framework that strategically provides minimal guidance only when a model's independent learning has plateaued. The framework first diagnoses learning stagnation and then intervenes by injecting tiered in-prompt hints, ranging from abstract concepts to concrete steps, enabling the model to construct a valid solution by itself. Extensive experiments on challenging mathematics benchmarks demonstrate Scaf-GRPO's effectiveness, boosting the pass@1 score of the Qwen2.5-Math-7B model on the AIME24 benchmark by a relative 44.3% over a vanilla GRPO baseline. This result demonstrates our framework provides a robust and effective methodology for unlocking a model's ability to solve problems previously beyond its reach, a critical step towards extending the frontier of autonomous reasoning in LLM. 

---
# Zhyper: Factorized Hypernetworks for Conditioned LLM Fine-Tuning 

**Authors**: M. H. I. Abdalla, Zhipin Wang, Christian Frey, Steffen Eger, Josif Grabocka  

**Link**: [PDF](https://arxiv.org/pdf/2510.19733)  

**Abstract**: Large Language Model (LLM) conditioning refers to instructing an LLM to generate content in accordance with the norms and values of a specific culture, beliefs of a particular political orientation, or any desired text-specified semantic conditioning. Unfortunately, prompt engineering does not ensure that LLMs behave in accordance with a desired conditioning due to the inductive bias of the pre-training and alignment datasets. Prior works have focused on fine-tuning LLMs by directly conditioning the LoRA weights; however, such methods introduce a large number of parameters. As a remedy, we propose Zhyper, a parameter-efficient factorized hypernetwork framework that generates context-aware LoRA adapters from textual descriptions. Experiments on multiple benchmarks show that Zhyper achieves competitive performance with up to 26x fewer parameters than the state-of-the-art baselines. Furthermore, we extend Zhyper to cultural alignment, demonstrating improved generalization to out-of-domain settings and a better capturing of fine-grained contextual values. 

---
# ToolDreamer: Instilling LLM Reasoning Into Tool Retrievers 

**Authors**: Saptarshi Sengupta, Zhengyu Zhou, Jun Araki, Xingbo Wang, Bingqing Wang, Suhang Wang, Zhe Feng  

**Link**: [PDF](https://arxiv.org/pdf/2510.19791)  

**Abstract**: Tool calling has become increasingly popular for Large Language Models (LLMs). However, for large tool sets, the resulting tokens would exceed the LLM's context window limit, making it impossible to include every tool. Hence, an external retriever is used to provide LLMs with the most relevant tools for a query. Existing retrieval models rank tools based on the similarity between a user query and a tool description (TD). This leads to suboptimal retrieval as user requests are often poorly aligned with the language of TD. To remedy the issue, we propose ToolDreamer, a framework to condition retriever models to fetch tools based on hypothetical (synthetic) TD generated using an LLM, i.e., description of tools that the LLM feels will be potentially useful for the query. The framework enables a more natural alignment between queries and tools within the language space of TD's. We apply ToolDreamer on the ToolRet dataset and show that our method improves the performance of sparse and dense retrievers with and without training, thus showcasing its flexibility. Through our proposed framework, our aim is to offload a portion of the reasoning burden to the retriever so that the LLM may effectively handle a large collection of tools without inundating its context window. 

---
# Hubble: a Model Suite to Advance the Study of LLM Memorization 

**Authors**: Johnny Tian-Zheng Wei, Ameya Godbole, Mohammad Aflah Khan, Ryan Wang, Xiaoyuan Zhu, James Flemings, Nitya Kashyap, Krishna P. Gummadi, Willie Neiswanger, Robin Jia  

**Link**: [PDF](https://arxiv.org/pdf/2510.19811)  

**Abstract**: We present Hubble, a suite of fully open-source large language models (LLMs) for the scientific study of LLM memorization. Hubble models come in standard and perturbed variants: standard models are pretrained on a large English corpus, and perturbed models are trained in the same way but with controlled insertion of text (e.g., book passages, biographies, and test sets) designed to emulate key memorization risks. Our core release includes 8 models -- standard and perturbed models with 1B or 8B parameters, pretrained on 100B or 500B tokens -- establishing that memorization risks are determined by the frequency of sensitive data relative to size of the training corpus (i.e., a password appearing once in a smaller corpus is memorized better than the same password in a larger corpus). Our release also includes 6 perturbed models with text inserted at different pretraining phases, showing that sensitive data without continued exposure can be forgotten. These findings suggest two best practices for addressing memorization risks: to dilute sensitive data by increasing the size of the training corpus, and to order sensitive data to appear earlier in training. Beyond these general empirical findings, Hubble enables a broad range of memorization research; for example, analyzing the biographies reveals how readily different types of private information are memorized. We also demonstrate that the randomized insertions in Hubble make it an ideal testbed for membership inference and machine unlearning, and invite the community to further explore, benchmark, and build upon our work. 

---
# Are Large Language Models Sensitive to the Motives Behind Communication? 

**Authors**: Addison J. Wu, Ryan Liu, Kerem Oktar, Theodore R. Sumers, Thomas L. Griffiths  

**Link**: [PDF](https://arxiv.org/pdf/2510.19687)  

**Abstract**: Human communication is motivated: people speak, write, and create content with a particular communicative intent in mind. As a result, information that large language models (LLMs) and AI agents process is inherently framed by humans' intentions and incentives. People are adept at navigating such nuanced information: we routinely identify benevolent or self-serving motives in order to decide what statements to trust. For LLMs to be effective in the real world, they too must critically evaluate content by factoring in the motivations of the source -- for instance, weighing the credibility of claims made in a sales pitch. In this paper, we undertake a comprehensive study of whether LLMs have this capacity for motivational vigilance. We first employ controlled experiments from cognitive science to verify that LLMs' behavior is consistent with rational models of learning from motivated testimony, and find they successfully discount information from biased sources in a human-like manner. We then extend our evaluation to sponsored online adverts, a more naturalistic reflection of LLM agents' information ecosystems. In these settings, we find that LLMs' inferences do not track the rational models' predictions nearly as closely -- partly due to additional information that distracts them from vigilance-relevant considerations. However, a simple steering intervention that boosts the salience of intentions and incentives substantially increases the correspondence between LLMs and the rational model. These results suggest that LLMs possess a basic sensitivity to the motivations of others, but generalizing to novel real-world settings will require further improvements to these models. 

---
# DiffAdapt: Difficulty-Adaptive Reasoning for Token-Efficient LLM Inference 

**Authors**: Xiang Liu, Xuming Hu, Xiaowen Chu, Eunsol Choi  

**Link**: [PDF](https://arxiv.org/pdf/2510.19669)  

**Abstract**: Recent reasoning Large Language Models (LLMs) demonstrate remarkable problem-solving abilities but often generate long thinking traces whose utility is unclear. Our work aims to improve their efficiency, enabling them to reach high performance without overthinking. First, we analyze the entropy of token probabilities in reasoning traces. Across three models, we observe a consistent U-shaped entropy pattern: high entropy on easy problems despite high accuracy, low entropy on problems with medium difficulty, and high entropy on hard problems reflecting uncertainty. Specifically, we notice 22--25\% entropy reduction from easy to medium difficulty regions, suggesting an {overthinking} phenomenon on easy instances. Building on these insights, we introduce \textbf{DiffAdapt}, a lightweight framework that selects Easy/Normal/Hard inference strategies per question based on their difficulty and reasoning trace entropy. Each inference strategy consists of a fixed prompt, temperature and maximum token length. In contrast to existing efficiency optimization methods, our approach does not fine-tune base LLM but a small probe that classifies LLM's final hidden state, allowing inexpensive adaptation. We comprehensively evaluate our method on five models and eight benchmarks. Our method achieves comparable or improved accuracy while reducing token usage by up to 22.4\%, establishing a practical path toward compute-efficient reasoning. 

---
# The Art of Asking: Multilingual Prompt Optimization for Synthetic Data 

**Authors**: David Mora, Viraat Aryabumi, Wei-Yin Ko, Sara Hooker, Julia Kreutzer, Marzieh Fadaee  

**Link**: [PDF](https://arxiv.org/pdf/2510.19806)  

**Abstract**: Synthetic data has become a cornerstone for scaling large language models, yet its multilingual use remains bottlenecked by translation-based prompts. This strategy inherits English-centric framing and style and neglects cultural dimensions, ultimately constraining model generalization. We argue that the overlooked prompt space-the very inputs that define training distributions-offers a more powerful lever for improving multilingual performance. We introduce a lightweight framework for prompt-space optimization, where translated prompts are systematically transformed for Naturalness, Cultural Adaptation, and Difficulty Enhancement. Using an off-the-shelf multilingual LLM, we apply these transformations to prompts for 12 languages spanning 7 families. Under identical data conditions, our approaches achieve substantial and consistent downstream improvements over the translation-only baseline: +4.7% on Global-MMLU accuracy, +2.4% on Flores XCometXL and +35.3% wins in preferences on mArenaHard. We establish prompt-space optimization as a simple yet powerful paradigm for building multilingual LLMs that are more robust, culturally grounded, and globally capable. 

---
# Unraveling Emotions with Pre-Trained Models 

**Authors**: Alejandro Pajón-Sanmartín, Francisco De Arriba-Pérez, Silvia García-Méndez, Fátima Leal, Benedita Malheiro, Juan Carlos Burguillo-Rial  

**Link**: [PDF](https://arxiv.org/pdf/2510.19668)  

**Abstract**: Transformer models have significantly advanced the field of emotion recognition. However, there are still open challenges when exploring open-ended queries for Large Language Models (LLMs). Although current models offer good results, automatic emotion analysis in open texts presents significant challenges, such as contextual ambiguity, linguistic variability, and difficulty interpreting complex emotional expressions. These limitations make the direct application of generalist models difficult. Accordingly, this work compares the effectiveness of fine-tuning and prompt engineering in emotion detection in three distinct scenarios: (i) performance of fine-tuned pre-trained models and general-purpose LLMs using simple prompts; (ii) effectiveness of different emotion prompt designs with LLMs; and (iii) impact of emotion grouping techniques on these models. Experimental tests attain metrics above 70% with a fine-tuned pre-trained model for emotion recognition. Moreover, the findings highlight that LLMs require structured prompt engineering and emotion grouping to enhance their performance. These advancements improve sentiment analysis, human-computer interaction, and understanding of user behavior across various domains. 

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
# Lookahead Routing for Large Language Models 

**Authors**: Canbin Huang, Tianyuan Shi, Yuhua Zhu, Ruijun Chen, Xiaojun Quan  

**Link**: [PDF](https://arxiv.org/pdf/2510.19506)  

**Abstract**: Large language model (LLM) routers improve the efficiency of multi-model systems by directing each query to the most appropriate model while leveraging the diverse strengths of heterogeneous LLMs. Most existing approaches frame routing as a classification problem based solely on the input query. While this reduces overhead by avoiding inference across all models, it overlooks valuable information that could be gleaned from potential outputs and fails to capture implicit intent or contextual nuances that often emerge only during response generation. These limitations can result in suboptimal routing decisions, particularly for complex or ambiguous queries that require deeper semantic understanding. To address this challenge, we propose Lookahead, a routing framework that "foresees" potential model outputs by predicting their latent representations and uses these predictions to guide model selection, thus enabling more informed routing without full inference. Within this framework, we implement two approaches based on causal and masked language models. Empirical evaluations across seven public benchmarks - spanning instruction following, mathematical reasoning, and code generation - show that Lookahead consistently outperforms existing routing baselines, achieving an average performance gain of 7.7% over the state-of-the-art. Our code is available at this https URL. 

---
# CoSense-LLM: Semantics at the Edge with Cost- and Uncertainty-Aware Cloud-Edge Cooperation 

**Authors**: Hasan Akgul, Mari Eplik, Javier Rojas, Aina Binti Abdullah, Pieter van der Merwe  

**Link**: [PDF](https://arxiv.org/pdf/2510.19670)  

**Abstract**: We present CoSense-LLM, an edge-first framework that turns continuous multimodal sensor streams (for example Wi-Fi CSI, IMU, audio, RFID, and lightweight vision) into compact, verifiable semantic tokens and coordinates with large language models under explicit latency, energy, bandwidth, and privacy constraints. CoSense-LLM has four parts: (i) SenseFusion, a lightweight encoder that aligns sensor embeddings with language and compresses them into short discrete code sequences; (ii) Edge-RAG, a local hybrid retrieval layer that grounds generation in site specific policies and notes; (iii) PromptRouter, a cost and uncertainty aware policy that selects edge only generation, edge plus retrieval, or compact cloud escalation; and (iv) Secure Execution, an auditable redaction path that enforces data minimization so raw waveforms never leave the device. The system works with modern serving optimizations, including paged or streaming KV caches, FlashAttention style kernels, speculative decoding, and quantized LoRA adapters, and supports on device personalization and federated updates under non IID drift. Across home, office, and clinic deployments, CoSense-LLM delivers grounded explanations while meeting tight service level objectives: it sustains sub second (p95) end to end latency on edge dominant paths, reduces inter tier token and bandwidth costs by preferring local retrieval grounded responses, and preserves privacy by transmitting only discrete codes and redacted metadata. Ablations show that Edge-RAG improves factual consistency and reduces contradictions, calibrated uncertainty enables selective abstention and controlled escalations, and KV plus decoding accelerators lower energy per decision. The results support an edge first design that treats semantics, privacy, and predictable latency as co equal goals for large model deployments in interference prone environments. 

---
# ToMMeR -- Efficient Entity Mention Detection from Large Language Models 

**Authors**: Victor Morand, Nadi Tomeh, Josiane Mothe, Benjamin Piwowarski  

**Link**: [PDF](https://arxiv.org/pdf/2510.19410)  

**Abstract**: Identifying which text spans refer to entities -- mention detection -- is both foundational for information extraction and a known performance bottleneck. We introduce ToMMeR, a lightweight model (<300K parameters) probing mention detection capabilities from early LLM layers. Across 13 NER benchmarks, ToMMeR achieves 93\% recall zero-shot, with over 90\% precision using an LLM as a judge showing that ToMMeR rarely produces spurious predictions despite high recall. Cross-model analysis reveals that diverse architectures (14M-15B parameters) converge on similar mention boundaries (DICE >75\%), confirming that mention detection emerges naturally from language modeling. When extended with span classification heads, ToMMeR achieves near SOTA NER performance (80-87\% F1 on standard benchmarks). Our work provides evidence that structured entity representations exist in early transformer layers and can be efficiently recovered with minimal parameters. 

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
# Algorithmic Fairness in NLP: Persona-Infused LLMs for Human-Centric Hate Speech Detection 

**Authors**: Ewelina Gajewska, Arda Derbent, Jaroslaw A Chudziak, Katarzyna Budzynska  

**Link**: [PDF](https://arxiv.org/pdf/2510.19331)  

**Abstract**: In this paper, we investigate how personalising Large Language Models (Persona-LLMs) with annotator personas affects their sensitivity to hate speech, particularly regarding biases linked to shared or differing identities between annotators and targets. To this end, we employ Google's Gemini and OpenAI's GPT-4.1-mini models and two persona-prompting methods: shallow persona prompting and a deeply contextualised persona development based on Retrieval-Augmented Generation (RAG) to incorporate richer persona profiles. We analyse the impact of using in-group and out-group annotator personas on the models' detection performance and fairness across diverse social groups. This work bridges psychological insights on group identity with advanced NLP techniques, demonstrating that incorporating socio-demographic attributes into LLMs can address bias in automated hate speech detection. Our results highlight both the potential and limitations of persona-based approaches in reducing bias, offering valuable insights for developing more equitable hate speech detection systems. 

---
# HAD: HAllucination Detection Language Models Based on a Comprehensive Hallucination Taxonomy 

**Authors**: Fan Xu, Xinyu Hu, Zhenghan Yu, Li Lin, Xu Zhang, Yang Zhang, Wei Zhou, Jinjie Gu, Xiaojun Wan  

**Link**: [PDF](https://arxiv.org/pdf/2510.19318)  

**Abstract**: The increasing reliance on natural language generation (NLG) models, particularly large language models, has raised concerns about the reliability and accuracy of their outputs. A key challenge is hallucination, where models produce plausible but incorrect information. As a result, hallucination detection has become a critical task. In this work, we introduce a comprehensive hallucination taxonomy with 11 categories across various NLG tasks and propose the HAllucination Detection (HAD) models this https URL, which integrate hallucination detection, span-level identification, and correction into a single inference process. Trained on an elaborate synthetic dataset of about 90K samples, our HAD models are versatile and can be applied to various NLG tasks. We also carefully annotate a test set for hallucination detection, called HADTest, which contains 2,248 samples. Evaluations on in-domain and out-of-domain test sets show that our HAD models generally outperform the existing baselines, achieving state-of-the-art results on HaluEval, FactCHD, and FaithBench, confirming their robustness and versatility. 

---
# Difficulty-Controllable Multiple-Choice Question Generation Using Large Language Models and Direct Preference Optimization 

**Authors**: Yuto Tomikawa, Masaki Uto  

**Link**: [PDF](https://arxiv.org/pdf/2510.19265)  

**Abstract**: Difficulty-controllable question generation for reading comprehension has gained significant attention in the field of education as a fundamental tool for adaptive learning support. Although several neural question generation methods have recently succeeded in controlling difficulty, conventional approaches still face two major limitations. First, they cannot directly generate multiple-choice questions, which are the most widely used question type in educational contexts. Second, they are not explicitly trained to optimize the accuracy of difficulty control, leaving room for further improvement in difficulty controllability. To address these limitations, this study proposes a novel difficulty-controllable multiple-choice question generation method for reading comprehension which leverages a large language model trained using a direct preference optimization technique to improve the accuracy of difficulty control. 

---
# JointCQ: Improving Factual Hallucination Detection with Joint Claim and Query Generation 

**Authors**: Fan Xu, Huixuan Zhang, Zhenliang Zhang, Jiahao Wang, Xiaojun Wan  

**Link**: [PDF](https://arxiv.org/pdf/2510.19310)  

**Abstract**: Current large language models (LLMs) often suffer from hallucination issues, i,e, generating content that appears factual but is actually unreliable. A typical hallucination detection pipeline involves response decomposition (i.e., claim extraction), query generation, evidence collection (i.e., search or retrieval), and claim verification. However, existing methods exhibit limitations in the first two stages, such as context loss during claim extraction and low specificity in query generation, resulting in degraded performance across the hallucination detection pipeline. In this work, we introduce JointCQ this https URL, a joint claim-and-query generation framework designed to construct an effective and efficient claim-query generator. Our framework leverages elaborately designed evaluation criteria to filter synthesized training data, and finetunes a language model for joint claim extraction and query generation, providing reliable and informative inputs for downstream search and verification. Experimental results demonstrate that our method outperforms previous methods on multiple open-domain QA hallucination detection benchmarks, advancing the goal of more trustworthy and transparent language model systems. 

---
# Slot Filling as a Reasoning Task for SpeechLLMs 

**Authors**: Kadri Hacioglu, Manjunath K E, Andreas Stolcke  

**Link**: [PDF](https://arxiv.org/pdf/2510.19326)  

**Abstract**: We propose integration of reasoning into speech large language models (speechLLMs) for the end-to-end slot-filling task. Inspired by the recent development of reasoning LLMs, we use a chain-of-thought framework to decompose the slot-filling task into multiple reasoning steps, create a reasoning dataset and apply the supervised fine-tuning strategy to a speechLLM. We distinguish between regular and reasoning speechLLMs and experiment with different types and sizes of LLMs as their text foundation models. We demonstrate performance improvements by introducing reasoning (intermediate) steps. However, we show that a reasoning textual LLM developed mainly for math, logic and coding domains might be inferior as a foundation model for a reasoning speechLLM. We further show that hybrid speechLLMs, built on a hybrid text foundation LLM and fine-tuned to preserve both direct and reasoning modes of operation, have better performance than those fine-tuned employing only one mode of operation. 

---
# When Facts Change: Probing LLMs on Evolving Knowledge with evolveQA 

**Authors**: Nishanth Sridhar Nakshatri, Shamik Roy, Manoj Ghuhan Arivazhagan, Hanhan Zhou, Vinayshekhar Bannihatti Kumar, Rashmi Gangadharaiah  

**Link**: [PDF](https://arxiv.org/pdf/2510.19172)  

**Abstract**: LLMs often fail to handle temporal knowledge conflicts--contradictions arising when facts evolve over time within their training data. Existing studies evaluate this phenomenon through benchmarks built on structured knowledge bases like Wikidata, but they focus on widely-covered, easily-memorized popular entities and lack the dynamic structure needed to fairly evaluate LLMs with different knowledge cut-off dates. We introduce evolveQA, a benchmark specifically designed to evaluate LLMs on temporally evolving knowledge, constructed from 3 real-world, time-stamped corpora: AWS updates, Azure changes, and WHO disease outbreak reports. Our framework identifies naturally occurring knowledge evolution and generates questions with gold answers tailored to different LLM knowledge cut-off dates. Through extensive evaluation of 12 open and closed-source LLMs across 3 knowledge probing formats, we demonstrate significant performance drops of up to 31% on evolveQA compared to static knowledge questions. 

---
# LoongRL:Reinforcement Learning for Advanced Reasoning over Long Contexts 

**Authors**: Siyuan Wang, Gaokai Zhang, Li Lyna Zhang, Ning Shang, Fan Yang, Dongyao Chen, Mao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2510.19363)  

**Abstract**: Reasoning over long contexts is essential for large language models. While reinforcement learning (RL) enhances short-context reasoning by inducing "Aha" moments in chain-of-thought, the advanced thinking patterns required for long-context reasoning remain largely unexplored, and high-difficulty RL data are scarce. In this paper, we introduce LoongRL, a data-driven RL method for advanced long-context reasoning. Central to LoongRL is KeyChain, a synthesis approach that transforms short multi-hop QA into high-difficulty long-context tasks by inserting UUID chains that hide the true question among large collections of distracting documents. Solving these tasks requires the model to trace the correct chain step-by-step, identify the true question, retrieve relevant facts and reason over them to answer correctly. RL training on KeyChain data induces an emergent plan-retrieve-reason-recheck reasoning pattern that generalizes far beyond training length. Models trained at 16K effectively solve 128K tasks without prohibitive full-length RL rollout costs. On Qwen2.5-7B and 14B, LoongRL substantially improves long-context multi-hop QA accuracy by +23.5% and +21.1% absolute gains. The resulting LoongRL-14B reaches a score of 74.2, rivaling much larger frontier models such as o3-mini (74.5) and DeepSeek-R1 (74.9). It also improves long-context retrieval, passes all 128K needle-in-a-haystack stress tests, and preserves short-context reasoning capabilities. 

---
# "You Are Rejected!": An Empirical Study of Large Language Models Taking Hiring Evaluations 

**Authors**: Dingjie Fu, Dianxing Shi  

**Link**: [PDF](https://arxiv.org/pdf/2510.19167)  

**Abstract**: With the proliferation of the internet and the rapid advancement of Artificial Intelligence, leading technology companies face an urgent annual demand for a considerable number of software and algorithm engineers. To efficiently and effectively identify high-potential candidates from thousands of applicants, these firms have established a multi-stage selection process, which crucially includes a standardized hiring evaluation designed to assess job-specific competencies. Motivated by the demonstrated prowess of Large Language Models (LLMs) in coding and reasoning tasks, this paper investigates a critical question: Can LLMs successfully pass these hiring evaluations? To this end, we conduct a comprehensive examination of a widely used professional assessment questionnaire. We employ state-of-the-art LLMs to generate responses and subsequently evaluate their performance. Contrary to any prior expectation of LLMs being ideal engineers, our analysis reveals a significant inconsistency between the model-generated answers and the company-referenced solutions. Our empirical findings lead to a striking conclusion: All evaluated LLMs fails to pass the hiring evaluation. 

---
# That's Deprecated! Understanding, Detecting, and Steering Knowledge Conflicts in Language Models for Code Generation 

**Authors**: Jaesung Bae, Cameron Churchwell, Mitchell Hermon, Tsun-An Hsieh, Jocelyn Xu, Yekaterina Yegorova, Mark Hasegawa-Johnson, Heng Ji  

**Link**: [PDF](https://arxiv.org/pdf/2510.19116)  

**Abstract**: This paper investigates how large language models (LLMs) behave when faced with discrepancies between their parametric knowledge and conflicting information contained in a prompt. Building on prior question-answering (QA) research, we extend the investigation of knowledge conflicts to the realm of code generation. We propose a domain-agnostic framework for constructing and interpreting such conflicts, along with a novel evaluation method and dataset tailored to code conflict scenarios. Our experiments indicate that sufficiently large LLMs encode the notion of a knowledge conflict in their parameters, enabling us to detect knowledge conflicts with up to \textbf{80.65\%} accuracy. Building on these insights, we show that activation-level steering can achieve up to a \textbf{12.6\%} improvement in steering success over a random baseline. However, effectiveness depends critically on balancing model size, task domain, and steering direction. The experiment code and data will be made publicly available after acceptance. 

---
# A Graph Signal Processing Framework for Hallucination Detection in Large Language Models 

**Authors**: Valentin Noël  

**Link**: [PDF](https://arxiv.org/pdf/2510.19117)  

**Abstract**: Large language models achieve impressive results but distinguishing factual reasoning from hallucinations remains challenging. We propose a spectral analysis framework that models transformer layers as dynamic graphs induced by attention, with token embeddings as signals on these graphs. Through graph signal processing, we define diagnostics including Dirichlet energy, spectral entropy, and high-frequency energy ratios, with theoretical connections to computational stability. Experiments across GPT architectures suggest universal spectral patterns: factual statements exhibit consistent "energy mountain" behavior with low-frequency convergence, while different hallucination types show distinct signatures. Logical contradictions destabilize spectra with large effect sizes ($g>1.0$), semantic errors remain stable but show connectivity drift, and substitution hallucinations display intermediate perturbations. A simple detector using spectral signatures achieves 88.75% accuracy versus 75% for perplexity-based baselines, demonstrating practical utility. These findings indicate that spectral geometry may capture reasoning patterns and error behaviors, potentially offering a framework for hallucination detection in large language models. 

---
# From Memorization to Generalization: Fine-Tuning Large Language Models for Biomedical Term-to-Identifier Normalization 

**Authors**: Suswitha Pericharla, Daniel B. Hier, Tayo Obafemi-Ajayi  

**Link**: [PDF](https://arxiv.org/pdf/2510.19036)  

**Abstract**: Effective biomedical data integration depends on automated term normalization, the mapping of natural language biomedical terms to standardized identifiers. This linking of terms to identifiers is essential for semantic interoperability. Large language models (LLMs) show promise for this task but perform unevenly across terminologies. We evaluated both memorization (training-term performance) and generalization (validation-term performance) across multiple biomedical ontologies. Fine-tuning Llama 3.1 8B revealed marked differences by terminology. GO mappings showed strong memorization gains (up to 77% improvement in term-to-identifier accuracy), whereas HPO showed minimal improvement. Generalization occurred only for protein-gene (GENE) mappings (13.9% gain), while fine-tuning for HPO and GO yielded negligible transfer. Baseline accuracy varied by model scale, with GPT-4o outperforming both Llama variants for all terminologies. Embedding analyses showed tight semantic alignment between gene symbols and protein names but weak alignment between terms and identifiers for GO or HPO, consistent with limited lexicalization. Fine-tuning success depended on two interacting factors: identifier popularity and lexicalization. Popular identifiers were more likely encountered during pretraining, enhancing memorization. Lexicalized identifiers, such as gene symbols, enabled semantic generalization. By contrast, arbitrary identifiers in GO and HPO constrained models to rote learning. These findings provide a predictive framework for when fine-tuning enhances factual recall versus when it fails due to sparse or non-lexicalized identifiers. 

---
# TheMCPCompany: Creating General-purpose Agents with Task-specific Tools 

**Authors**: Reza Esfandiarpoor, Vishwas Suryanarayanan, Stephen H. Bach, Vishal Chowdhary, Anthony Aue  

**Link**: [PDF](https://arxiv.org/pdf/2510.19286)  

**Abstract**: Since the introduction of the Model Context Protocol (MCP), the number of available tools for Large Language Models (LLMs) has increased significantly. These task-specific tool sets offer an alternative to general-purpose tools such as web browsers, while being easier to develop and maintain than GUIs. However, current general-purpose agents predominantly rely on web browsers for interacting with the environment. Here, we introduce TheMCPCompany, a benchmark for evaluating tool-calling agents on tasks that involve interacting with various real-world services. We use the REST APIs of these services to create MCP servers, which include over 18,000 tools. We also provide manually annotated ground-truth tools for each task. In our experiments, we use the ground truth tools to show the potential of tool-calling agents for both improving performance and reducing costs assuming perfect tool retrieval. Next, we explore agent performance using tool retrieval to study the real-world practicality of tool-based agents. While all models with tool retrieval perform similarly or better than browser-based agents, smaller models cannot take full advantage of the available tools through retrieval. On the other hand, GPT-5's performance with tool retrieval is very close to its performance with ground-truth tools. Overall, our work shows that the most advanced reasoning models are effective at discovering tools in simpler environments, but seriously struggle with navigating complex enterprise environments. TheMCPCompany reveals that navigating tens of thousands of tools and combining them in non-trivial ways to solve complex problems is still a challenging task for current models and requires both better reasoning and better retrieval models. 

---
# When Can We Trust LLMs in Mental Health? Large-Scale Benchmarks for Reliable LLM Evaluation 

**Authors**: Abeer Badawi, Elahe Rahimi, Md Tahmid Rahman Laskar, Sheri Grach, Lindsay Bertrand, Lames Danok, Jimmy Huang, Frank Rudzicz, Elham Dolatabadi  

**Link**: [PDF](https://arxiv.org/pdf/2510.19032)  

**Abstract**: Evaluating Large Language Models (LLMs) for mental health support is challenging due to the emotionally and cognitively complex nature of therapeutic dialogue. Existing benchmarks are limited in scale, reliability, often relying on synthetic or social media data, and lack frameworks to assess when automated judges can be trusted. To address the need for large-scale dialogue datasets and judge reliability assessment, we introduce two benchmarks that provide a framework for generation and evaluation. MentalBench-100k consolidates 10,000 one-turn conversations from three real scenarios datasets, each paired with nine LLM-generated responses, yielding 100,000 response pairs. MentalAlign-70k}reframes evaluation by comparing four high-performing LLM judges with human experts across 70,000 ratings on seven attributes, grouped into Cognitive Support Score (CSS) and Affective Resonance Score (ARS). We then employ the Affective Cognitive Agreement Framework, a statistical methodology using intraclass correlation coefficients (ICC) with confidence intervals to quantify agreement, consistency, and bias between LLM judges and human experts. Our analysis reveals systematic inflation by LLM judges, strong reliability for cognitive attributes such as guidance and informativeness, reduced precision for empathy, and some unreliability in safety and relevance. Our contributions establish new methodological and empirical foundations for reliable, large-scale evaluation of LLMs in mental health. We release the benchmarks and codes at: this https URL 

---
# Are they lovers or friends? Evaluating LLMs' Social Reasoning in English and Korean Dialogues 

**Authors**: Eunsu Kim, Junyeong Park, Juhyun Oh, Kiwoong Park, Seyoung Song, A.Seza Dogruoz, Najoung Kim, Alice Oh  

**Link**: [PDF](https://arxiv.org/pdf/2510.19028)  

**Abstract**: As large language models (LLMs) are increasingly used in human-AI interactions, their social reasoning capabilities in interpersonal contexts are critical. We introduce SCRIPTS, a 1k-dialogue dataset in English and Korean, sourced from movie scripts. The task involves evaluating models' social reasoning capability to infer the interpersonal relationships (e.g., friends, sisters, lovers) between speakers in each dialogue. Each dialogue is annotated with probabilistic relational labels (Highly Likely, Less Likely, Unlikely) by native (or equivalent) Korean and English speakers from Korea and the U.S. Evaluating nine models on our task, current proprietary LLMs achieve around 75-80% on the English dataset, whereas their performance on Korean drops to 58-69%. More strikingly, models select Unlikely relationships in 10-25% of their responses. Furthermore, we find that thinking models and chain-of-thought prompting, effective for general reasoning, provide minimal benefits for social reasoning and occasionally amplify social biases. Our findings reveal significant limitations in current LLMs' social reasoning capabilities, highlighting the need for efforts to develop socially-aware language models. 

---
# DiSRouter: Distributed Self-Routing for LLM Selections 

**Authors**: Hang Zheng, Hongshen Xu, Yongkai Lin, Shuai Fan, Lu Chen, Kai Yu  

**Link**: [PDF](https://arxiv.org/pdf/2510.19208)  

**Abstract**: The proliferation of Large Language Models (LLMs) has created a diverse ecosystem of models with highly varying performance and costs, necessitating effective query routing to balance performance and expense. Current routing systems often rely on a centralized external router trained on a fixed set of LLMs, making them inflexible and prone to poor performance since the small router can not fully understand the knowledge boundaries of different LLMs. We introduce DiSRouter (Distributed Self-Router), a novel paradigm that shifts from centralized control to distributed routing. In DiSRouter, a query traverses a network of LLM agents, each independently deciding whether to answer or route to other agents based on its own self-awareness, its ability to judge its competence. This distributed design offers superior flexibility, scalability, and generalizability. To enable this, we propose a two-stage Self-Awareness Training pipeline that enhances each LLM's self-awareness. Extensive experiments demonstrate that DiSRouter significantly outperforms existing routing methods in utility across various scenarios, effectively distinguishes between easy and hard queries, and shows strong generalization to out-of-domain tasks. Our work validates that leveraging an LLM's intrinsic self-awareness is more effective than external assessment, paving the way for more modular and efficient multi-agent systems. 

---
# Dynamic Evaluation for Oversensitivity in LLMs 

**Authors**: Sophia Xiao Pu, Sitao Cheng, Xin Eric Wang, William Yang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.19005)  

**Abstract**: Oversensitivity occurs when language models defensively reject prompts that are actually benign. This behavior not only disrupts user interactions but also obscures the boundary between harmful and harmless content. Existing benchmarks rely on static datasets that degrade overtime as models evolve, leading to data contamination and diminished evaluative power. To address this, we develop a framework that dynamically generates model-specific challenging datasets, capturing emerging defensive patterns and aligning with each model's unique behavior. Building on this approach, we construct OVERBENCH, a benchmark that aggregates these datasets across diverse LLM families, encompassing 450,000 samples from 25 models. OVERBENCH provides a dynamic and evolving perspective on oversensitivity, allowing for continuous monitoring of defensive triggers as models advance, highlighting vulnerabilities that static datasets overlook. 

---
# Evaluating LLM Story Generation through Large-scale Network Analysis of Social Structures 

**Authors**: Hiroshi Nonaka, K. E. Perry  

**Link**: [PDF](https://arxiv.org/pdf/2510.18932)  

**Abstract**: Evaluating the creative capabilities of large language models (LLMs) in complex tasks often requires human assessments that are difficult to scale. We introduce a novel, scalable methodology for evaluating LLM story generation by analyzing underlying social structures in narratives as signed character networks. To demonstrate its effectiveness, we conduct a large-scale comparative analysis using networks from over 1,200 stories, generated by four leading LLMs (GPT-4o, GPT-4o mini, Gemini 1.5 Pro, and Gemini 1.5 Flash) and a human-written corpus. Our findings, based on network properties like density, clustering, and signed edge weights, show that LLM-generated stories consistently exhibit a strong bias toward tightly-knit, positive relationships, which aligns with findings from prior research using human assessment. Our proposed approach provides a valuable tool for evaluating limitations and tendencies in the creative storytelling of current and future LLMs. 

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
# Contextual Augmentation for Entity Linking using Large Language Models 

**Authors**: Daniel Vollmers, Hamada M. Zahera, Diego Moussallem, Axel-Cyrille Ngonga Ngomo  

**Link**: [PDF](https://arxiv.org/pdf/2510.18888)  

**Abstract**: Entity Linking involves detecting and linking entity mentions in natural language texts to a knowledge graph. Traditional methods use a two-step process with separate models for entity recognition and disambiguation, which can be computationally intensive and less effective. We propose a fine-tuned model that jointly integrates entity recognition and disambiguation in a unified framework. Furthermore, our approach leverages large language models to enrich the context of entity mentions, yielding better performance in entity disambiguation. We evaluated our approach on benchmark datasets and compared with several baselines. The evaluation results show that our approach achieves state-of-the-art performance on out-of-domain datasets. 

---
# Misinformation Detection using Large Language Models with Explainability 

**Authors**: Jainee Patel, Chintan Bhatt, Himani Trivedi, Thanh Thi Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2510.18918)  

**Abstract**: The rapid spread of misinformation on online platforms undermines trust among individuals and hinders informed decision making. This paper shows an explainable and computationally efficient pipeline to detect misinformation using transformer-based pretrained language models (PLMs). We optimize both RoBERTa and DistilBERT using a two-step strategy: first, we freeze the backbone and train only the classification head; then, we progressively unfreeze the backbone layers while applying layer-wise learning rate decay. On two real-world benchmark datasets, COVID Fake News and FakeNewsNet GossipCop, we test the proposed approach with a unified protocol of preprocessing and stratified splits. To ensure transparency, we integrate the Local Interpretable Model-Agnostic Explanations (LIME) at the token level to present token-level rationales and SHapley Additive exPlanations (SHAP) at the global feature attribution level. It demonstrates that DistilBERT achieves accuracy comparable to RoBERTa while requiring significantly less computational resources. This work makes two key contributions: (1) it quantitatively shows that a lightweight PLM can maintain task performance while substantially reducing computational cost, and (2) it presents an explainable pipeline that retrieves faithful local and global justifications without compromising performance. The results suggest that PLMs combined with principled fine-tuning and interpretability can be an effective framework for scalable, trustworthy misinformation detection. 

---
# Improving Topic Modeling of Social Media Short Texts with Rephrasing: A Case Study of COVID-19 Related Tweets 

**Authors**: Wangjiaxuan Xin, Shuhua Yin, Shi Chen, Yaorong Ge  

**Link**: [PDF](https://arxiv.org/pdf/2510.18908)  

**Abstract**: Social media platforms such as Twitter (now X) provide rich data for analyzing public discourse, especially during crises such as the COVID-19 pandemic. However, the brevity, informality, and noise of social media short texts often hinder the effectiveness of traditional topic modeling, producing incoherent or redundant topics that are often difficult to interpret. To address these challenges, we have developed \emph{TM-Rephrase}, a model-agnostic framework that leverages large language models (LLMs) to rephrase raw tweets into more standardized and formal language prior to topic modeling. Using a dataset of 25,027 COVID-19-related Twitter posts, we investigate the effects of two rephrasing strategies, general- and colloquial-to-formal-rephrasing, on multiple topic modeling methods. Results demonstrate that \emph{TM-Rephrase} improves three metrics measuring topic modeling performance (i.e., topic coherence, topic uniqueness, and topic diversity) while reducing topic redundancy of most topic modeling algorithms, with the colloquial-to-formal strategy yielding the greatest performance gains and especially for the Latent Dirichlet Allocation (LDA) algorithm. This study contributes to a model-agnostic approach to enhancing topic modeling in public health related social media analysis, with broad implications for improved understanding of public discourse in health crisis as well as other important domains. 

---
# LLM Unlearning with LLM Beliefs 

**Authors**: Kemou Li, Qizhou Wang, Yue Wang, Fengpeng Li, Jun Liu, Bo Han, Jiantao Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2510.19422)  

**Abstract**: Large language models trained on vast corpora inherently risk memorizing sensitive or harmful content, which may later resurface in their outputs. Prevailing unlearning methods generally rely on gradient ascent and its variants to lower the probability of specific target responses. However, we find that this strategy induces a critical side effect: probability mass is redistributed into high-likelihood regions, often corresponding to semantically related rephrasings of the targets. We refer to this as the squeezing effect, which explains why many methods yield merely spurious unlearning, a problem further obscured by automated metrics (e.g., ROUGE, truth ratio) that misreport actual success. To address this, we propose a bootstrapping (BS) framework that explicitly links the squeezing effect with the model's own high-confidence generations, namely its model beliefs. Since model beliefs inherently capture the very high-likelihood regions where probability mass is squeezed, incorporating them into the unlearning objective directly counters the squeezing effect. By jointly suppressing both target responses and model beliefs, BS-T (token) attenuates high-probability tokens, whereas BS-S (sequence) removes entire high-confidence generations, together achieving more thorough forgetting while preserving utility. Extensive experiments across diverse benchmarks with various model families confirm the effectiveness of our approach. 

---
# A Multi-faceted Analysis of Cognitive Abilities: Evaluating Prompt Methods with Large Language Models on the CONSORT Checklist 

**Authors**: Sohyeon Jeon, Hyung-Chul Lee  

**Link**: [PDF](https://arxiv.org/pdf/2510.19139)  

**Abstract**: Despite the rapid expansion of Large Language Models (LLMs) in healthcare, the ability of these systems to assess clinical trial reporting according to CONSORT standards remains unclear, particularly with respect to their cognitive and reasoning strategies. This study applies a behavioral and metacognitive analytic approach with expert-validated data, systematically comparing two representative LLMs under three prompt conditions. Clear differences emerged in how the models approached various CONSORT items, and prompt types, including shifts in reasoning style, explicit uncertainty, and alternative interpretations shaped response patterns. Our results highlight the current limitations of these systems in clinical compliance automation and underscore the importance of understanding their cognitive adaptations and strategic behavior in developing more explainable and reliable medical AI. 

---
# Towards Better Health Conversations: The Benefits of Context-seeking 

**Authors**: Rory Sayres, Yuexing Hao, Abbi Ward, Amy Wang, Beverly Freeman, Serena Zhan, Diego Ardila, Jimmy Li, I-Ching Lee, Anna Iurchenko, Siyi Kou, Kartikeya Badola, Jimmy Hu, Bhawesh Kumar, Keith Johnson, Supriya Vijay, Justin Krogue, Avinatan Hassidim, Yossi Matias, Dale R. Webster, Sunny Virmani, Yun Liu, Quang Duong, Mike Schaekermann  

**Link**: [PDF](https://arxiv.org/pdf/2510.18880)  

**Abstract**: Navigating health questions can be daunting in the modern information landscape. Large language models (LLMs) may provide tailored, accessible information, but also risk being inaccurate, biased or misleading. We present insights from 4 mixed-methods studies (total N=163), examining how people interact with LLMs for their own health questions. Qualitative studies revealed the importance of context-seeking in conversational AIs to elicit specific details a person may not volunteer or know to share. Context-seeking by LLMs was valued by participants, even if it meant deferring an answer for several turns. Incorporating these insights, we developed a "Wayfinding AI" to proactively solicit context. In a randomized, blinded study, participants rated the Wayfinding AI as more helpful, relevant, and tailored to their concerns compared to a baseline AI. These results demonstrate the strong impact of proactive context-seeking on conversational dynamics, and suggest design patterns for conversational AI to help navigate health topics. 

---
# PoSh: Using Scene Graphs To Guide LLMs-as-a-Judge For Detailed Image Descriptions 

**Authors**: Amith Ananthram, Elias Stengel-Eskin, Lorena A. Bradford, Julia Demarest, Adam Purvis, Keith Krut, Robert Stein, Rina Elster Pantalony, Mohit Bansal, Kathleen McKeown  

**Link**: [PDF](https://arxiv.org/pdf/2510.19060)  

**Abstract**: While vision-language models (VLMs) have advanced into detailed image description, evaluation remains a challenge. Standard metrics (e.g. CIDEr, SPICE) were designed for short texts and tuned to recognize errors that are now uncommon, such as object misidentification. In contrast, long texts require sensitivity to attribute and relation attachments and scores that localize errors to particular text spans. In this work, we introduce PoSh, a metric for detailed image description that uses scene graphs as structured rubrics to guide LLMs-as-a-Judge, producing aggregate scores grounded in fine-grained errors (e.g. mistakes in compositional understanding). PoSh is replicable, interpretable and a better proxy for human raters than existing metrics (including GPT4o-as-a-Judge). To validate PoSh, we introduce a challenging new dataset, DOCENT. This novel benchmark contains artwork, paired with expert-written references, and model-generated descriptions, augmented with granular and coarse judgments of their quality from art history students. Thus, DOCENT enables evaluating both detailed image description metrics and detailed image description itself in a challenging new domain. We show that PoSh achieves stronger correlations (+0.05 Spearman $\rho$) with the human judgments in DOCENT than the best open-weight alternatives, is robust to image type (using CapArena, an existing dataset of web imagery) and is a capable reward function, outperforming standard supervised fine-tuning. Then, using PoSh, we characterize the performance of open and closed models in describing the paintings, sketches and statues in DOCENT and find that foundation models struggle to achieve full, error-free coverage of images with rich scene dynamics, establishing a demanding new task to gauge VLM progress. Through both PoSh and DOCENT, we hope to enable advances in important areas such as assistive text generation. 

---
# BAPO: Stabilizing Off-Policy Reinforcement Learning for LLMs via Balanced Policy Optimization with Adaptive Clipping 

**Authors**: Zhiheng Xi, Xin Guo, Yang Nan, Enyu Zhou, Junrui Shen, Wenxiang Chen, Jiaqi Liu, Jixuan Huang, Zhihao Zhang, Honglin Guo, Xun Deng, Zhikai Lei, Miao Zheng, Guoteng Wang, Shuo Zhang, Peng Sun, Rui Zheng, Hang Yan, Tao Gui, Qi Zhang, Xuanjing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2510.18927)  

**Abstract**: Reinforcement learning (RL) has recently become the core paradigm for aligning and strengthening large language models (LLMs). Yet, applying RL in off-policy settings--where stale data from past policies are used for training--improves sample efficiency, but remains challenging: policy entropy declines sharply, optimization often becomes unstable and may even collapse. Through theoretical and empirical analysis, we identify two key insights: (i) an imbalance in optimization, where negative-advantage samples dominate the policy gradient, suppressing useful behaviors and risking gradient explosions; and (ii) the derived Entropy-Clip Rule, which reveals that the fixed clipping mechanism in PPO-like objectives systematically blocks entropy-increasing updates, thereby driving the policy toward over-exploitation at the expense of exploration. Building on these insights, we propose BAlanced Policy Optimization with Adaptive Clipping (BAPO), a simple yet effective method that dynamically adjusts clipping bounds to adaptively re-balance positive and negative contributions, preserve entropy, and stabilize RL optimization. Across diverse off-policy scenarios--including sample replay and partial rollout--BAPO achieves fast, stable, and data-efficient training. On AIME 2024 and AIME 2025 benchmarks, our 7B BAPO model surpasses open-source counterparts such as SkyWork-OR1-7B, while our 32B BAPO model not only achieves state-of-the-art results among models of the same scale but also outperforms leading proprietary systems like o3-mini and Gemini-2.5-Flash-Thinking. 

---
# Beyond Reactivity: Measuring Proactive Problem Solving in LLM Agents 

**Authors**: Gil Pasternak, Dheeraj Rajagopal, Julia White, Dhruv Atreja, Matthew Thomas, George Hurn-Maloney, Ash Lewis  

**Link**: [PDF](https://arxiv.org/pdf/2510.19771)  

**Abstract**: LLM-based agents are increasingly moving towards proactivity: rather than awaiting instruction, they exercise agency to anticipate user needs and solve them autonomously. However, evaluating proactivity is challenging; current benchmarks are constrained to localized context, limiting their ability to test reasoning across sources and longer time horizons. To address this gap, we present PROBE (Proactive Resolution Of BottlEnecks). PROBE decomposes proactivity as a pipeline of three core capabilities: (1) searching for unspecified issues, (2) identifying specific bottlenecks, and (3) executing appropriate resolutions. We apply PROBE to evaluate leading LLMs and popular agentic frameworks, showing that even state-of-the-art models struggle to solve this benchmark. Computing our consistent measurements across frontier LLMs and agents, we find that the best end-to-end performance of 40% is achieved by both GPT-5 and Claude Opus-4.1. Additionally, we demonstrate the relative capabilities of each model and analyze mutual failure modes. Our results highlight the current limitations of autonomous action in agentic systems, and expose promising future research directions. 

---
# AgentSense: LLMs Empower Generalizable and Explainable Web-Based Participatory Urban Sensing 

**Authors**: Xusen Guo, Mingxing Peng, Xixuan Hao, Xingchen Zou, Qiongyan Wang, Sijie Ruan, Yuxuan Liang  

**Link**: [PDF](https://arxiv.org/pdf/2510.19661)  

**Abstract**: Web-based participatory urban sensing has emerged as a vital approach for modern urban management by leveraging mobile individuals as distributed sensors. However, existing urban sensing systems struggle with limited generalization across diverse urban scenarios and poor interpretability in decision-making. In this work, we introduce AgentSense, a hybrid, training-free framework that integrates large language models (LLMs) into participatory urban sensing through a multi-agent evolution system. AgentSense initially employs classical planner to generate baseline solutions and then iteratively refines them to adapt sensing task assignments to dynamic urban conditions and heterogeneous worker preferences, while producing natural language explanations that enhance transparency and trust. Extensive experiments across two large-scale mobility datasets and seven types of dynamic disturbances demonstrate that AgentSense offers distinct advantages in adaptivity and explainability over traditional methods. Furthermore, compared to single-agent LLM baselines, our approach outperforms in both performance and robustness, while delivering more reasonable and transparent explanations. These results position AgentSense as a significant advancement towards deploying adaptive and explainable urban sensing systems on the web. 

---
# NeSyPr: Neurosymbolic Proceduralization For Efficient Embodied Reasoning 

**Authors**: Wonje Choi, Jooyoung Kim, Honguk Woo  

**Link**: [PDF](https://arxiv.org/pdf/2510.19429)  

**Abstract**: We address the challenge of adopting language models (LMs) for embodied tasks in dynamic environments, where online access to large-scale inference engines or symbolic planners is constrained due to latency, connectivity, and resource limitations. To this end, we present NeSyPr, a novel embodied reasoning framework that compiles knowledge via neurosymbolic proceduralization, thereby equipping LM-based agents with structured, adaptive, and timely reasoning capabilities. In NeSyPr, task-specific plans are first explicitly generated by a symbolic tool leveraging its declarative knowledge. These plans are then transformed into composable procedural representations that encode the plans' implicit production rules, enabling the resulting composed procedures to be seamlessly integrated into the LM's inference process. This neurosymbolic proceduralization abstracts and generalizes multi-step symbolic structured path-finding and reasoning into single-step LM inference, akin to human knowledge compilation. It supports efficient test-time inference without relying on external symbolic guidance, making it well suited for deployment in latency-sensitive and resource-constrained physical systems. We evaluate NeSyPr on the embodied benchmarks PDDLGym, VirtualHome, and ALFWorld, demonstrating its efficient reasoning capabilities over large-scale reasoning models and a symbolic planner, while using more compact LMs. 

---
# RLIE: Rule Generation with Logistic Regression, Iterative Refinement, and Evaluation for Large Language Models 

**Authors**: Yang Yang, Hua XU, Zhangyi Hu, Yutao Yue  

**Link**: [PDF](https://arxiv.org/pdf/2510.19698)  

**Abstract**: Large Language Models (LLMs) can propose rules in natural language, sidestepping the need for a predefined predicate space in traditional rule learning. Yet many LLM-based approaches ignore interactions among rules, and the opportunity to couple LLMs with probabilistic rule learning for robust inference remains underexplored. We present RLIE, a unified framework that integrates LLMs with probabilistic modeling to learn a set of weighted rules. RLIE has four stages: (1) Rule generation, where an LLM proposes and filters candidates; (2) Logistic regression, which learns probabilistic weights for global selection and calibration; (3) Iterative refinement, which updates the rule set using prediction errors; and (4) Evaluation, which compares the weighted rule set as a direct classifier with methods that inject rules into an LLM. We evaluate multiple inference strategies on real-world datasets. Applying rules directly with their learned weights yields superior performance, whereas prompting LLMs with the rules, weights, and logistic-model outputs surprisingly degrades accuracy. This supports the view that LLMs excel at semantic generation and interpretation but are less reliable for precise probabilistic integration. RLIE clarifies the potential and limitations of LLMs for inductive reasoning and couples them with classic probabilistic rule combination methods to enable more reliable neuro-symbolic reasoning. 

---
# Learning to Make Friends: Coaching LLM Agents toward Emergent Social Ties 

**Authors**: Philipp J. Schneider, Lin Tian, Marian-Andrei Rizoiu  

**Link**: [PDF](https://arxiv.org/pdf/2510.19299)  

**Abstract**: Can large language model (LLM) agents reproduce the complex social dynamics that characterize human online behavior -- shaped by homophily, reciprocity, and social validation -- and what memory and learning mechanisms enable such dynamics to emerge? We present a multi-agent LLM simulation framework in which agents repeatedly interact, evaluate one another, and adapt their behavior through in-context learning accelerated by a coaching signal. To model human social behavior, we design behavioral reward functions that capture core drivers of online engagement, including social interaction, information seeking, self-presentation, coordination, and emotional support. These rewards align agent objectives with empirically observed user motivations, enabling the study of how network structures and group formations emerge from individual decision-making. Our experiments show that coached LLM agents develop stable interaction patterns and form emergent social ties, yielding network structures that mirror properties of real online communities. By combining behavioral rewards with in-context adaptation, our framework establishes a principled testbed for investigating collective dynamics in LLM populations and reveals how artificial agents may approximate or diverge from human-like social behavior. 

---
# Timely Clinical Diagnosis through Active Test Selection 

**Authors**: Silas Ruhrberg Estévez, Nicolás Astorga, Mihaela van der Schaar  

**Link**: [PDF](https://arxiv.org/pdf/2510.18988)  

**Abstract**: There is growing interest in using machine learning (ML) to support clinical diag- nosis, but most approaches rely on static, fully observed datasets and fail to reflect the sequential, resource-aware reasoning clinicians use in practice. Diagnosis remains complex and error prone, especially in high-pressure or resource-limited settings, underscoring the need for frameworks that help clinicians make timely and cost-effective decisions. We propose ACTMED (Adaptive Clinical Test selection via Model-based Experimental Design), a diagnostic framework that integrates Bayesian Experimental Design (BED) with large language models (LLMs) to better emulate real-world diagnostic reasoning. At each step, ACTMED selects the test expected to yield the greatest reduction in diagnostic uncertainty for a given patient. LLMs act as flexible simulators, generating plausible patient state distributions and supporting belief updates without requiring structured, task-specific training data. Clinicians can remain in the loop; reviewing test suggestions, interpreting intermediate outputs, and applying clinical judgment throughout. We evaluate ACTMED on real-world datasets and show it can optimize test selection to improve diagnostic accuracy, interpretability, and resource use. This represents a step to- ward transparent, adaptive, and clinician-aligned diagnostic systems that generalize across settings with reduced reliance on domain-specific data. 

---
# Test-time Verification via Optimal Transport: Coverage, ROC, & Sub-optimality 

**Authors**: Arpan Mukherjee, Marcello Bullo, Debabrota Basu, Deniz Gündüz  

**Link**: [PDF](https://arxiv.org/pdf/2510.18982)  

**Abstract**: While test-time scaling with verification has shown promise in improving the performance of large language models (LLMs), the role of the verifier and its imperfections remain underexplored. The effect of verification manifests through interactions of three quantities: (i) the generator's coverage, (ii) the verifier's region of convergence (ROC), and (iii) the sampling algorithm's sub-optimality. Though recent studies capture subsets of these factors, a unified framework quantifying the geometry of their interplay is missing. We frame verifiable test-time scaling as a transport problem. This characterizes the interaction of coverage, ROC, and sub-optimality, and uncovers that the sub-optimality--coverage curve exhibits three regimes. A transport regime -- where sub-optimality increases with coverage, a policy improvement regime -- where sub-optimality may decrease with coverage, depending on the verifier's ROC, and a saturation regime -- where sub-optimality plateaus, unaffected by coverage. We further propose and analyze two classes of sampling algorithms -- sequential and batched, and examine how their computational complexities shape these trade-offs. Empirical results with Qwen, Llama, and Gemma models corroborate our theoretical findings. 

---
# Integrating Transparent Models, LLMs, and Practitioner-in-the-Loop: A Case of Nonprofit Program Evaluation 

**Authors**: Ji Ma, Albert Casella  

**Link**: [PDF](https://arxiv.org/pdf/2510.19799)  

**Abstract**: Public and nonprofit organizations often hesitate to adopt AI tools because most models are opaque even though standard approaches typically analyze aggregate patterns rather than offering actionable, case-level guidance. This study tests a practitioner-in-the-loop workflow that pairs transparent decision-tree models with large language models (LLMs) to improve predictive accuracy, interpretability, and the generation of practical insights. Using data from an ongoing college-success program, we build interpretable decision trees to surface key predictors. We then provide each tree's structure to an LLM, enabling it to reproduce case-level predictions grounded in the transparent models. Practitioners participate throughout feature engineering, model design, explanation review, and usability assessment, ensuring that field expertise informs the analysis at every stage. Results show that integrating transparent models, LLMs, and practitioner input yields accurate, trustworthy, and actionable case-level evaluations, offering a viable pathway for responsible AI adoption in the public and nonprofit sectors. 

---
# I Spy With My Model's Eye: Visual Search as a Behavioural Test for MLLMs 

**Authors**: John Burden, Jonathan Prunty, Ben Slater, Matthieu Tehenan, Greg Davis, Lucy Cheke  

**Link**: [PDF](https://arxiv.org/pdf/2510.19678)  

**Abstract**: Multimodal large language models (MLLMs) achieve strong performance on vision-language tasks, yet their visual processing is opaque. Most black-box evaluations measure task accuracy, but reveal little about underlying mechanisms. Drawing on cognitive psychology, we adapt classic visual search paradigms -- originally developed to study human perception -- to test whether MLLMs exhibit the ``pop-out'' effect, where salient visual features are detected independently of distractor set size. Using controlled experiments targeting colour, size and lighting features, we find that advanced MLLMs exhibit human-like pop-out effects in colour or size-based disjunctive (single feature) search, as well as capacity limits for conjunctive (multiple feature) search. We also find evidence to suggest that MLLMs, like humans, incorporate natural scene priors such as lighting direction into object representations. We reinforce our findings using targeted fine-tuning and mechanistic interpretability analyses. Our work shows how visual search can serve as a cognitively grounded diagnostic tool for evaluating perceptual capabilities in MLLMs. 

---
# Modeling realistic human behavior using generative agents in a multimodal transport system: Software architecture and Application to Toulouse 

**Authors**: Trung-Dung Vu, Benoit Gaudou, Kamaldeep Singh Oberoi  

**Link**: [PDF](https://arxiv.org/pdf/2510.19497)  

**Abstract**: Modeling realistic human behaviour to understand people's mode choices in order to propose personalised mobility solutions remains challenging. This paper presents an architecture for modeling realistic human mobility behavior in complex multimodal transport systems, demonstrated through a case study in Toulouse, France. We apply Large Language Models (LLMs) within an agent-based simulation to capture decision-making in a real urban setting. The framework integrates the GAMA simulation platform with an LLM-based generative agent, along with General Transit Feed Specification (GTFS) data for public transport, and OpenTripPlanner for multimodal routing. GAMA platform models the interactive transport environment, providing visualization and dynamic agent interactions while eliminating the need to construct the simulation environment from scratch. This design enables a stronger focus on developing generative agents and evaluating their performance in transport decision-making processes. Over a simulated month, results show that agents not only make context-aware transport decisions but also form habits over time. We conclude that combining LLMs with agent-based simulation offers a promising direction for advancing intelligent transportation systems and personalised multimodal mobility solutions. We also discuss some limitations of this approach and outline future work on scaling to larger regions, integrating real-time data, and refining memory models. 

---
# Metadata Extraction Leveraging Large Language Models 

**Authors**: Cuize Han, Sesh Jalagam  

**Link**: [PDF](https://arxiv.org/pdf/2510.19334)  

**Abstract**: The advent of Large Language Models has revolutionized tasks across domains, including the automation of legal document analysis, a critical component of modern contract management systems. This paper presents a comprehensive implementation of LLM-enhanced metadata extraction for contract review, focusing on the automatic detection and annotation of salient legal clauses. Leveraging both the publicly available Contract Understanding Atticus Dataset (CUAD) and proprietary contract datasets, our work demonstrates the integration of advanced LLM methodologies with practical applications. We identify three pivotal elements for optimizing metadata extraction: robust text conversion, strategic chunk selection, and advanced LLM-specific techniques, including Chain of Thought (CoT) prompting and structured tool calling. The results from our experiments highlight the substantial improvements in clause identification accuracy and efficiency. Our approach shows promise in reducing the time and cost associated with contract review while maintaining high accuracy in legal clause identification. The results suggest that carefully optimized LLM systems could serve as valuable tools for legal professionals, potentially increasing access to efficient contract review services for organizations of all sizes. 

---
# SORA-ATMAS: Adaptive Trust Management and Multi-LLM Aligned Governance for Future Smart Cities 

**Authors**: Usama Antuley, Shahbaz Siddiqui, Sufian Hameed, Waqas Arif, Subhan Shah, Syed Attique Shah  

**Link**: [PDF](https://arxiv.org/pdf/2510.19327)  

**Abstract**: The rapid evolution of smart cities has increased the reliance on intelligent interconnected services to optimize infrastructure, resources, and citizen well-being. Agentic AI has emerged as a key enabler by supporting autonomous decision-making and adaptive coordination, allowing urban systems to respond in real time to dynamic conditions. Its benefits are evident in areas such as transportation, where the integration of traffic data, weather forecasts, and safety sensors enables dynamic rerouting and a faster response to hazards. However, its deployment across heterogeneous smart city ecosystems raises critical governance, risk, and compliance (GRC) challenges, including accountability, data privacy, and regulatory alignment within decentralized infrastructures. Evaluation of SORA-ATMAS with three domain agents (Weather, Traffic, and Safety) demonstrated that its governance policies, including a fallback mechanism for high-risk scenarios, effectively steer multiple LLMs (GPT, Grok, DeepSeek) towards domain-optimized, policy-aligned outputs, producing an average MAE reduction of 35% across agents. Results showed stable weather monitoring, effective handling of high-risk traffic plateaus 0.85, and adaptive trust regulation in Safety/Fire scenarios 0.65. Runtime profiling of a 3-agent deployment confirmed scalability, with throughput between 13.8-17.2 requests per second, execution times below 72~ms, and governance delays under 100 ms, analytical projections suggest maintained performance at larger scales. Cross-domain rules ensured safe interoperability, with traffic rerouting permitted only under validated weather conditions. These findings validate SORA-ATMAS as a regulation-aligned, context-aware, and verifiable governance framework that consolidates distributed agent outputs into accountable, real-time decisions, offering a resilient foundation for smart-city management. 

---
# LAPRAD: LLM-Assisted PRotocol Attack Discovery 

**Authors**: R.Can Aygun, Yehuda Afek, Anat Bremler-Barr, Leonard Kleinrock  

**Link**: [PDF](https://arxiv.org/pdf/2510.19264)  

**Abstract**: With the goal of improving the security of Internet protocols, we seek faster, semi-automatic methods to discover new vulnerabilities in protocols such as DNS, BGP, and others. To this end, we introduce the LLM-Assisted Protocol Attack Discovery (LAPRAD) methodology, enabling security researchers with some DNS knowledge to efficiently uncover vulnerabilities that would otherwise be hard to detect.
LAPRAD follows a three-stage process. In the first, we consult an LLM (GPT-o1) that has been trained on a broad corpus of DNS-related sources and previous DDoS attacks to identify potential exploits. In the second stage, a different LLM automatically constructs the corresponding attack configurations using the ReACT approach implemented via LangChain (DNS zone file generation). Finally, in the third stage, we validate the attack's functionality and effectiveness.
Using LAPRAD, we uncovered three new DDoS attacks on the DNS protocol and rediscovered two recently reported ones that were not included in the LLM's training data. The first new attack employs a bait-and-switch technique to trick resolvers into caching large, bogus DNSSEC RRSIGs, reducing their serving capacity to as little as 6%. The second exploits large DNSSEC encryption algorithms (RSA-4096) with multiple keys, thereby bypassing a recently implemented default RRSet limit. The third leverages ANY-type responses to produce a similar effect.
These variations of a cache-flushing DDoS attack, called SigCacheFlush, circumvent existing patches, severely degrade resolver query capacity, and impact the latest versions of major DNS resolver implementations. 

---
# PruneHal: Reducing Hallucinations in Multi-modal Large Language Models through Adaptive KV Cache Pruning 

**Authors**: Fengyuan Sun, Hui Chen, Xinhao Xu, Dandan Zheng, Jingdong Chen, Jun Zhou, Jungong Han, Guiguang Ding  

**Link**: [PDF](https://arxiv.org/pdf/2510.19183)  

**Abstract**: While multi-modal large language models (MLLMs) have made significant progress in recent years, the issue of hallucinations remains a major challenge. To mitigate this phenomenon, existing solutions either introduce additional data for further training or incorporate external or internal information during inference. However, these approaches inevitably introduce extra computational costs. In this paper, we observe that hallucinations in MLLMs are strongly associated with insufficient attention allocated to visual tokens. In particular, the presence of redundant visual tokens disperses the model's attention, preventing it from focusing on the most informative ones. As a result, critical visual cues are often under-attended, which in turn exacerbates the occurrence of hallucinations. Building on this observation, we propose \textbf{PruneHal}, a training-free, simple yet effective method that leverages adaptive KV cache pruning to enhance the model's focus on critical visual information, thereby mitigating hallucinations. To the best of our knowledge, we are the first to apply token pruning for hallucination mitigation in MLLMs. Notably, our method don't require additional training and incurs nearly no extra inference cost. Moreover, PruneHal is model-agnostic and can be seamlessly integrated with different decoding strategies, including those specifically designed for hallucination mitigation. We evaluate PruneHal on several widely used hallucination evaluation benchmarks using four mainstream MLLMs, achieving robust and outstanding results that highlight the effectiveness and superiority of our method. Our code will be publicly available. 

---
# What Makes a Good Curriculum? Disentangling the Effects of Data Ordering on LLM Mathematical Reasoning 

**Authors**: Yaning Jia, Chunhui Zhang, Xingjian Diao, Xiangchi Yuan, Zhongyu Ouyang, soroush vosoughi  

**Link**: [PDF](https://arxiv.org/pdf/2510.19099)  

**Abstract**: Curriculum learning (CL) - ordering training data from easy to hard - has become a popular strategy for improving reasoning in large language models (LLMs). Yet prior work employs disparate difficulty metrics and training setups, leaving open fundamental questions: When does curriculum help? Which direction - forward or reverse - is better? And does the answer depend on what we measure? We address these questions through a unified offline evaluation framework that decomposes curriculum difficulty into five complementary dimensions: Problem Difficulty, Model Surprisal, Confidence Margin, Predictive Uncertainty, and Decision Variability. Through controlled post-training experiments on mathematical reasoning benchmarks with Llama3.1-8B, Mistral-7B, and Gemma3-4B, we find that (i) no curriculum strategy dominates universally - the relative effectiveness of forward versus reverse CL depends jointly on model capability and task complexity; (ii) even within a single metric, samples at different difficulty levels produce distinct gains depending on task demands; and (iii) task-aligned curricula focus on shaping the model's final representations and generalization, whereas inner-state curricula modulate internal states such as confidence and uncertainty. Our findings challenge the notion of a universal curriculum strategy and offer actionable guidance across model and task regimes, with some metrics indicating that prioritizing decision-uncertain samples can further enhance learning outcomes. 

---
# CLiVR: Conversational Learning System in Virtual Reality with AI-Powered Patients 

**Authors**: Akilan Amithasagaran, Sagnik Dakshit, Bhavani Suryadevara, Lindsey Stockton  

**Link**: [PDF](https://arxiv.org/pdf/2510.19031)  

**Abstract**: Simulations constitute a fundamental component of medical and nursing education and traditionally employ standardized patients (SP) and high-fidelity manikins to develop clinical reasoning and communication skills. However, these methods require substantial resources, limiting accessibility and scalability. In this study, we introduce CLiVR, a Conversational Learning system in Virtual Reality that integrates large language models (LLMs), speech processing, and 3D avatars to simulate realistic doctor-patient interactions. Developed in Unity and deployed on the Meta Quest 3 platform, CLiVR enables trainees to engage in natural dialogue with virtual patients. Each simulation is dynamically generated from a syndrome-symptom database and enhanced with sentiment analysis to provide feedback on communication tone. Through an expert user study involving medical school faculty (n=13), we assessed usability, realism, and perceived educational impact. Results demonstrated strong user acceptance, high confidence in educational potential, and valuable feedback for improvement. CLiVR offers a scalable, immersive supplement to SP-based training. 

---
# FlexiDataGen: An Adaptive LLM Framework for Dynamic Semantic Dataset Generation in Sensitive Domains 

**Authors**: Hamed Jelodar, Samita Bai, Roozbeh Razavi-Far, Ali A. Ghorbani  

**Link**: [PDF](https://arxiv.org/pdf/2510.19025)  

**Abstract**: Dataset availability and quality remain critical challenges in machine learning, especially in domains where data are scarce, expensive to acquire, or constrained by privacy regulations. Fields such as healthcare, biomedical research, and cybersecurity frequently encounter high data acquisition costs, limited access to annotated data, and the rarity or sensitivity of key events. These issues-collectively referred to as the dataset challenge-hinder the development of accurate and generalizable machine learning models in such high-stakes domains. To address this, we introduce FlexiDataGen, an adaptive large language model (LLM) framework designed for dynamic semantic dataset generation in sensitive domains. FlexiDataGen autonomously synthesizes rich, semantically coherent, and linguistically diverse datasets tailored to specialized fields. The framework integrates four core components: (1) syntactic-semantic analysis, (2) retrieval-augmented generation, (3) dynamic element injection, and (4) iterative paraphrasing with semantic validation. Together, these components ensure the generation of high-quality, domain-relevant data. Experimental results show that FlexiDataGen effectively alleviates data shortages and annotation bottlenecks, enabling scalable and accurate machine learning model development. 

---
# Prior-informed optimization of treatment recommendation via bandit algorithms trained on large language model-processed historical records 

**Authors**: Saman Nessari, Ali Bozorgi-Amiri  

**Link**: [PDF](https://arxiv.org/pdf/2510.19014)  

**Abstract**: Current medical practice depends on standardized treatment frameworks and empirical methodologies that neglect individual patient variations, leading to suboptimal health outcomes. We develop a comprehensive system integrating Large Language Models (LLMs), Conditional Tabular Generative Adversarial Networks (CTGAN), T-learner counterfactual models, and contextual bandit approaches to provide customized, data-informed clinical recommendations. The approach utilizes LLMs to process unstructured medical narratives into structured datasets (93.2% accuracy), uses CTGANs to produce realistic synthetic patient data (55% accuracy via two-sample verification), deploys T-learners to forecast patient-specific treatment responses (84.3% accuracy), and integrates prior-informed contextual bandits to enhance online therapeutic selection by effectively balancing exploration of new possibilities with exploitation of existing knowledge. Testing on stage III colon cancer datasets revealed that our KernelUCB approach obtained 0.60-0.61 average reward scores across 5,000 rounds, exceeding other reference methods. This comprehensive system overcomes cold-start limitations in online learning environments, improves computational effectiveness, and constitutes notable progress toward individualized medicine adapted to specific patient characteristics. 

---
# Robust Driving QA through Metadata-Grounded Context and Task-Specific Prompts 

**Authors**: Seungjun Yu, Junsung Park, Youngsun Lim, Hyunjung Shim  

**Link**: [PDF](https://arxiv.org/pdf/2510.19001)  

**Abstract**: We present a two-phase vision-language QA system for autonomous driving that answers high-level perception, prediction, and planning questions. In Phase-1, a large multimodal LLM (Qwen2.5-VL-32B) is conditioned on six-camera inputs, a short temporal window of history, and a chain-of-thought prompt with few-shot exemplars. A self-consistency ensemble (multiple sampled reasoning chains) further improves answer reliability. In Phase-2, we augment the prompt with nuScenes scene metadata (object annotations, ego-vehicle state, etc.) and category-specific question instructions (separate prompts for perception, prediction, planning tasks). In experiments on a driving QA benchmark, our approach significantly outperforms the baseline Qwen2.5 models. For example, using 5 history frames and 10-shot prompting in Phase-1 yields 65.1% overall accuracy (vs.62.61% with zero-shot); applying self-consistency raises this to 66.85%. Phase-2 achieves 67.37% overall. Notably, the system maintains 96% accuracy under severe visual corruption. These results demonstrate that carefully engineered prompts and contextual grounding can greatly enhance high-level driving QA with pretrained vision-language models. 

---
# A Justice Lens on Fairness and Ethics Courses in Computing Education: LLM-Assisted Multi-Perspective and Thematic Evaluation 

**Authors**: Kenya S. Andrews, Deborah Dormah Kanubala, Kehinde Aruleba, Francisco Enrique Vicente Castro, Renata A Revelo  

**Link**: [PDF](https://arxiv.org/pdf/2510.18931)  

**Abstract**: Course syllabi set the tone and expectations for courses, shaping the learning experience for both students and instructors. In computing courses, especially those addressing fairness and ethics in artificial intelligence (AI), machine learning (ML), and algorithmic design, it is imperative that we understand how approaches to navigating barriers to fair outcomes are being this http URL expectations should be inclusive, transparent, and grounded in promoting critical thinking. Syllabus analysis offers a way to evaluate the coverage, depth, practices, and expectations within a course. Manual syllabus evaluation, however, is time-consuming and prone to inconsistency. To address this, we developed a justice-oriented scoring rubric and asked a large language model (LLM) to review syllabi through a multi-perspective role simulation. Using this rubric, we evaluated 24 syllabi from four perspectives: instructor, departmental chair, institutional reviewer, and external evaluator. We also prompted the LLM to identify thematic trends across the courses. Findings show that multiperspective evaluation aids us in noting nuanced, role-specific priorities, leveraging them to fill hidden gaps in curricula design of AI/ML and related computing courses focused on fairness and ethics. These insights offer concrete directions for improving the design and delivery of fairness, ethics, and justice content in such courses. 

---
# LLM Bazaar: A Service Design for Supporting Collaborative Learning with an LLM-Powered Multi-Party Collaboration Infrastructure 

**Authors**: Zhen Wu, Jiaxin Shi, R. Charles Murray, Carolyn Rosé, Micah San Andres  

**Link**: [PDF](https://arxiv.org/pdf/2510.18877)  

**Abstract**: For nearly two decades, conversational agents have played a critical role in structuring interactions in collaborative learning, shaping group dynamics, and supporting student engagement. The recent integration of large language models (LLMs) into these agents offers new possibilities for fostering critical thinking and collaborative problem solving. In this work, we begin with an open source collaboration support architecture called Bazaar and integrate an LLM-agent shell that enables introduction of LLM-empowered, real time, context sensitive collaborative support for group learning. This design and infrastructure paves the way for exploring how tailored LLM-empowered environments can reshape collaborative learning outcomes and interaction patterns. 

---
# CosmoCore Affective Dream-Replay Reinforcement Learning for Code Generation 

**Authors**: Santhosh Kumar Ravindran  

**Link**: [PDF](https://arxiv.org/pdf/2510.18895)  

**Abstract**: We introduce CosmoCore, a neuroscience-inspired reinforcement learning (RL) architecture that integrates affective signals to enhance code generation in large language models (LLMs). Motivated by human and animal learning where embarrassment from mistakes drives rapid correction, as observed in training a puppy to avoid repeating errors after a single scolding CosmoCore tags code generation trajectories with valence and surprise using a lightweight multi-layer perceptron (MLP). High-negative valence (cringe) episodes, such as buggy code outputs, are prioritized in a Dream Queue for five-fold replay during off-policy updates, while low-surprise successes are pruned to prevent overconfidence and buffer bloat. Evaluated on code generation benchmarks like HumanEval and BigCodeBench, alongside simulations with a custom data pipeline environment, CosmoCore reduces hallucinated code (e.g., syntax errors or logical bugs) by 48\% and accelerates self-correction by 45\%. Local experiments using Hugging Face models in a PySpark environment validate these gains, with code snippets provided for replication. Ablations confirm valence tagging boosts curiosity in exploration, and pruning mitigates inefficiency. This framework extends RL from human feedback (RLHF) for more emotionally aware code assistants, with applications in IDEs and data pipelines. Code and the custom mini-world simulation are released. 

---
# AI for Distributed Systems Design: Scalable Cloud Optimization Through Repeated LLMs Sampling And Simulators 

**Authors**: Jacopo Tagliabue  

**Link**: [PDF](https://arxiv.org/pdf/2510.18897)  

**Abstract**: We explore AI-driven distributed-systems policy design by combining stochastic code generation from large language models (LLMs) with deterministic verification in a domain-specific simulator. Using a Function-as-a-Service runtime (Bauplan) and its open-source simulator (Eudoxia) as a case study, we frame scheduler design as an iterative generate-and-verify loop: an LLM proposes a Python policy, the simulator evaluates it on standardized traces, and structured feedback steers subsequent generations. This setup preserves interpretability while enabling targeted search over a large design space. We detail the system architecture and report preliminary results on throughput improvements across multiple models. Beyond early gains, we discuss the limits of the current setup and outline next steps; in particular, we conjecture that AI will be crucial for scaling this methodology by helping to bootstrap new simulators. 

---
# SBAN: A Framework \& Multi-Dimensional Dataset for Large Language Model Pre-Training and Software Code Mining 

**Authors**: Hamed Jelodar, Mohammad Meymani, Samita Bai, Roozbeh Razavi-Far, Ali A. Ghorbani  

**Link**: [PDF](https://arxiv.org/pdf/2510.18936)  

**Abstract**: This paper introduces SBAN (Source code, Binary, Assembly, and Natural Language Description), a large-scale, multi-dimensional dataset designed to advance the pre-training and evaluation of large language models (LLMs) for software code analysis. SBAN comprises more than 3 million samples, including 2.9 million benign and 672,000 malware respectively, each represented across four complementary layers: binary code, assembly instructions, natural language descriptions, and source code. This unique multimodal structure enables research on cross-representation learning, semantic understanding of software, and automated malware detection. Beyond security applications, SBAN supports broader tasks such as code translation, code explanation, and other software mining tasks involving heterogeneous data. It is particularly suited for scalable training of deep models, including transformers and other LLM architectures. By bridging low-level machine representations and high-level human semantics, SBAN provides a robust foundation for building intelligent systems that reason about code. We believe that this dataset opens new opportunities for mining software behavior, improving security analytics, and enhancing LLM capabilities in pre-training and fine-tuning tasks for software code mining. 

---
# XGen-Q: An Explainable Domain-Adaptive LLM Framework with Retrieval-Augmented Generation for Software Security 

**Authors**: Hamed Jelodar, Mohammad Meymani, Roozbeh Razavi-Far, Ali A. Ghorbani  

**Link**: [PDF](https://arxiv.org/pdf/2510.19006)  

**Abstract**: Generative AI and large language models (LLMs) have shown strong capabilities in code understanding, but their use in cybersecurity, particularly for malware detection and analysis, remains limited. Existing detection systems often fail to generalize to obfuscated or previously unseen threats, underscoring the need for more adaptable and explainable models. To address this challenge, we introduce XGen-Q, a domain-adapted LLM built on the Qwen-Coder architecture and pretrained on a large-scale corpus of over one million malware samples, spanning both source and assembly code. XGen-Q uses a multi-stage prompt strategy combined with retrieval-augmented generation (RAG) to deliver reliable malware identification and detailed forensic reporting, even in the presence of complex code obfuscation. To further enhance generalization, we design a training pipeline that systematically exposes the model to diverse obfuscation patterns. Experimental results show that XGen-Q achieves significantly lower perplexity than competitive baselines and exhibits strong performance on novel malware samples, demonstrating the promise of LLM-based approaches for interpretable and robust malware analysis. 

---
