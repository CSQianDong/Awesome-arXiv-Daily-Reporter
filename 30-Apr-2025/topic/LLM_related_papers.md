# MATCHA: Can Multi-Agent Collaboration Build a Trustworthy Conversational Recommender? 

**Authors**: Zheng Hui, Xiaokai Wei, Yexi Jiang, Kevin Gao, Chen Wang, Frank Ong, Se-eun Yoon, Rachit Pareek, Michelle Gong  

**Link**: [PDF](https://arxiv.org/pdf/2504.20094)  

**Abstract**: In this paper, we propose a multi-agent collaboration framework called MATCHA for conversational recommendation system, leveraging large language models (LLMs) to enhance personalization and user engagement. Users can request recommendations via free-form text and receive curated lists aligned with their interests, preferences, and constraints. Our system introduces specialized agents for intent analysis, candidate generation, ranking, re-ranking, explainability, and safeguards. These agents collaboratively improve recommendations accuracy, diversity, and safety. On eight metrics, our model achieves superior or comparable performance to the current state-of-the-art. Through comparisons with six baseline models, our approach addresses key challenges in conversational recommendation systems for game recommendations, including: (1) handling complex, user-specific requests, (2) enhancing personalization through multi-agent collaboration, (3) empirical evaluation and deployment, and (4) ensuring safe and trustworthy interactions. 

---
# Enhancing News Recommendation with Hierarchical LLM Prompting 

**Authors**: Hai-Dang Kieu, Delvin Ce Zhang, Minh Duc Nguyen, Min Xu, Qiang Wu, Dung D. Le  

**Link**: [PDF](https://arxiv.org/pdf/2504.20452)  

**Abstract**: Personalized news recommendation systems often struggle to effectively capture the complexity of user preferences, as they rely heavily on shallow representations, such as article titles and abstracts. To address this problem, we introduce a novel method, namely PNR-LLM, for Large Language Models for Personalized News Recommendation. Specifically, PNR-LLM harnesses the generation capabilities of LLMs to enrich news titles and abstracts, and consequently improves recommendation quality. PNR-LLM contains a novel module, News Enrichment via LLMs, which generates deeper semantic information and relevant entities from articles, transforming shallow contents into richer representations. We further propose an attention mechanism to aggregate enriched semantic- and entity-level data, forming unified user and news embeddings that reveal a more accurate user-news match. Extensive experiments on MIND datasets show that PNR-LLM outperforms state-of-the-art baselines. Moreover, the proposed data enrichment module is model-agnostic, and we empirically show that applying our proposed module to multiple existing models can further improve their performance, verifying the advantage of our design. 

---
# An Integrated Framework for Contextual Personalized LLM-Based Food Recommendation 

**Authors**: Ali Rostami  

**Link**: [PDF](https://arxiv.org/pdf/2504.20092)  

**Abstract**: Personalized food recommendation systems (Food-RecSys) critically underperform due to fragmented component understanding and the failure of conventional machine learning with vast, imbalanced food data. While Large Language Models (LLMs) offer promise, current generic Recommendation as Language Processing (RLP) strategies lack the necessary specialization for the food domain's complexity. This thesis tackles these deficiencies by first identifying and analyzing the essential components for effective Food-RecSys. We introduce two key innovations: a multimedia food logging platform for rich contextual data acquisition and the World Food Atlas, enabling unique geolocation-based food analysis previously unavailable. Building on this foundation, we pioneer the Food Recommendation as Language Processing (F-RLP) framework - a novel, integrated approach specifically architected for the food domain. F-RLP leverages LLMs in a tailored manner, overcoming the limitations of generic models and providing a robust infrastructure for effective, contextual, and truly personalized food recommendations. 

---
# Recommending Clinical Trials for Online Patient Cases using Artificial Intelligence 

**Authors**: Joey Chan, Qiao Jin, Nicholas Wan, Charalampos S. Floudas, Elisabetta Xue, Zhiyong Lu  

**Link**: [PDF](https://arxiv.org/pdf/2504.20059)  

**Abstract**: Clinical trials are crucial for assessing new treatments; however, recruitment challenges - such as limited awareness, complex eligibility criteria, and referral barriers - hinder their success. With the growth of online platforms, patients increasingly turn to social media and health communities for support, research, and advocacy, expanding recruitment pools and established enrollment pathways. Recognizing this potential, we utilized TrialGPT, a framework that leverages a large language model (LLM) as its backbone, to match 50 online patient cases (collected from published case reports and a social media website) to clinical trials and evaluate performance against traditional keyword-based searches. Our results show that TrialGPT outperforms traditional methods by 46% in identifying eligible trials, with each patient, on average, being eligible for around 7 trials. Additionally, our outreach efforts to case authors and trial organizers regarding these patient-trial matches yielded highly positive feedback, which we present from both perspectives. 

---
# OpenTCM: A GraphRAG-Empowered LLM-based System for Traditional Chinese Medicine Knowledge Retrieval and Diagnosis 

**Authors**: Jinglin He, Yunqi Guo, Lai Kwan Lam, Waikei Leung, Lixing He, Yuanan Jiang, Chi Chiu Wang, Guoliang Xing, Hongkai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.20118)  

**Abstract**: Traditional Chinese Medicine (TCM) represents a rich repository of ancient medical knowledge that continues to play an important role in modern healthcare. Due to the complexity and breadth of the TCM literature, the integration of AI technologies is critical for its modernization and broader accessibility. However, this integration poses considerable challenges, including the interpretation of obscure classical Chinese texts and the modeling of intricate semantic relationships among TCM concepts. In this paper, we develop OpenTCM, an LLM-based system that combines a domain-specific TCM knowledge graph and Graph-based Retrieval-Augmented Generation (GraphRAG). First, we extract more than 3.73 million classical Chinese characters from 68 gynecological books in the Chinese Medical Classics Database, with the help of TCM and gynecology experts. Second, we construct a comprehensive multi-relational knowledge graph comprising more than 48,000 entities and 152,000 interrelationships, using customized prompts and Chinese-oriented LLMs such as DeepSeek and Kimi to ensure high-fidelity semantic understanding. Last, we integrate OpenTCM with this knowledge graph, enabling high-fidelity ingredient knowledge retrieval and diagnostic question-answering without model fine-tuning. Experimental evaluations demonstrate that our prompt design and model selection significantly improve knowledge graph quality, achieving a precision of 98. 55% and an F1 score of 99. 55%. In addition, OpenTCM achieves mean expert scores of 4.5 in ingredient information retrieval and 3.8 in diagnostic question-answering tasks, outperforming state-of-the-art solutions in real-world TCM use cases. 

---
# Against Opacity: Explainable AI and Large Language Models for Effective Digital Advertising 

**Authors**: Qi Yang, Marlo Ongpin, Sergey Nikolenko, Alfred Huang, Aleksandr Farseev  

**Link**: [PDF](https://arxiv.org/pdf/2504.20064)  

**Abstract**: The opaqueness of modern digital advertising, exemplified by platforms such as Meta Ads, raises concerns regarding their autonomous control over audience targeting, pricing structures, and ad relevancy assessments. Locked in their leading positions by network effects, ``Metas and Googles of the world'' attract countless advertisers who rely on intuition, with billions of dollars lost on ineffective social media ads. The platforms' algorithms use huge amounts of data unavailable to advertisers, and the algorithms themselves are opaque as well. This lack of transparency hinders the advertisers' ability to make informed decisions and necessitates efforts to promote transparency, standardize industry metrics, and strengthen regulatory frameworks. In this work, we propose novel ways to assist marketers in optimizing their advertising strategies via machine learning techniques designed to analyze and evaluate content, in particular, predict the click-through rates (CTR) of novel advertising content. Another important problem is that large volumes of data available in the competitive landscape, e.g., competitors' ads, impede the ability of marketers to derive meaningful insights. This leads to a pressing need for a novel approach that would allow us to summarize and comprehend complex data. Inspired by the success of ChatGPT in bridging the gap between large language models (LLMs) and a broader non-technical audience, we propose a novel system that facilitates marketers in data interpretation, called SODA, that merges LLMs with explainable AI, enabling better human-AI collaboration with an emphasis on the domain of digital marketing and advertising. By combining LLMs and explainability features, in particular modern text-image models, we aim to improve the synergy between human marketers and AI systems. 

---
# HCT-QA: A Benchmark for Question Answering on Human-Centric Tables 

**Authors**: Mohammad S. Ahmad, Zan A. Naeem, Michaël Aupetit, Ahmed Elmagarmid, Mohamed Eltabakh, Xiasong Ma, Mourad Ouzzani, Chaoyi Ruan  

**Link**: [PDF](https://arxiv.org/pdf/2504.20047)  

**Abstract**: Tabular data embedded within PDF files, web pages, and other document formats are prevalent across numerous sectors such as government, engineering, science, and business. These human-centric tables (HCTs) possess a unique combination of high business value, intricate layouts, limited operational power at scale, and sometimes serve as the only data source for critical insights. However, their complexity poses significant challenges to traditional data extraction, processing, and querying methods. While current solutions focus on transforming these tables into relational formats for SQL queries, they fall short in handling the diverse and complex layouts of HCTs and hence being amenable to querying. This paper describes HCT-QA, an extensive benchmark of HCTs, natural language queries, and related answers on thousands of tables. Our dataset includes 2,188 real-world HCTs with 9,835 QA pairs and 4,679 synthetic tables with 67.5K QA pairs. While HCTs can be potentially processed by different type of query engines, in this paper, we focus on Large Language Models as potential engines and assess their ability in processing and querying such tables. 

---
# Team ACK at SemEval-2025 Task 2: Beyond Word-for-Word Machine Translation for English-Korean Pairs 

**Authors**: Daniel Lee, Harsh Sharma, Jieun Han, Sunny Jeong, Alice Oh, Vered Shwartz  

**Link**: [PDF](https://arxiv.org/pdf/2504.20451)  

**Abstract**: Translating knowledge-intensive and entity-rich text between English and Korean requires transcreation to preserve language-specific and cultural nuances beyond literal, phonetic or word-for-word conversion. We evaluate 13 models (LLMs and MT models) using automatic metrics and human assessment by bilingual annotators. Our findings show LLMs outperform traditional MT systems but struggle with entity translation requiring cultural adaptation. By constructing an error taxonomy, we identify incorrect responses and entity name errors as key issues, with performance varying by entity type and popularity level. This work exposes gaps in automatic evaluation metrics and hope to enable future work in completing culturally-nuanced machine translation. 

---
# Spark: A System for Scientifically Creative Idea Generation 

**Authors**: Aishik Sanyal, Samuel Schapiro, Sumuk Shashidhar, Royce Moon, Lav R. Varshney, Dilek Hakkani-Tur  

**Link**: [PDF](https://arxiv.org/pdf/2504.20090)  

**Abstract**: Recently, large language models (LLMs) have shown promising abilities to generate novel research ideas in science, a direction which coincides with many foundational principles in computational creativity (CC). In light of these developments, we present an idea generation system named Spark that couples retrieval-augmented idea generation using LLMs with a reviewer model named Judge trained on 600K scientific reviews from OpenReview. Our work is both a system demonstration and intended to inspire other CC researchers to explore grounding the generation and evaluation of scientific ideas within foundational CC principles. To this end, we release the annotated dataset used to train Judge, inviting other researchers to explore the use of LLMs for idea generation and creative evaluations. 

---
# Refiner: Restructure Retrieval Content Efficiently to Advance Question-Answering Capabilities 

**Authors**: Zhonghao Li, Xuming Hu, Aiwei Liu, Kening Zheng, Sirui Huang, Hui Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2406.11357)  

**Abstract**: Large Language Models (LLMs) are limited by their parametric knowledge, leading to hallucinations in knowledge-extensive tasks. To address this, Retrieval-Augmented Generation (RAG) incorporates external document chunks to expand LLM knowledge. Furthermore, compressing information from document chunks through extraction or summarization can improve LLM performance. Nonetheless, LLMs still struggle to notice and utilize scattered key information, a problem known as the "lost-in-the-middle" syndrome. Therefore, we typically need to restructure the content for LLM to recognize the key information. We propose $\textit{Refiner}$, an end-to-end extract-and-restructure paradigm that operates in the post-retrieval process of RAG. $\textit{Refiner}$ leverages a single decoder-only LLM to adaptively extract query-relevant contents verbatim along with the necessary context, and section them based on their interconnectedness, thereby highlights information distinction, and aligns downstream LLMs with the original context effectively. Experiments show that a trained $\textit{Refiner}$ (with 7B parameters) exhibits significant gain to downstream LLM in improving answer accuracy, and outperforms other state-of-the-art advanced RAG and concurrent compressing approaches in various single-hop and multi-hop QA tasks. Notably, $\textit{Refiner}$ achieves a 80.5% tokens reduction and a 1.6-7.0% improvement margin in multi-hop tasks compared to the next best solution. $\textit{Refiner}$ is a plug-and-play solution that can be seamlessly integrated with RAG systems, facilitating its application across diverse open-source frameworks. 

---
# Chain-of-Defensive-Thought: Structured Reasoning Elicits Robustness in Large Language Models against Reference Corruption 

**Authors**: Wenxiao Wang, Parsa Hosseini, Soheil Feizi  

**Link**: [PDF](https://arxiv.org/pdf/2504.20769)  

**Abstract**: Chain-of-thought prompting has demonstrated great success in facilitating the reasoning abilities of large language models. In this work, we explore how these enhanced reasoning abilities can be exploited to improve the robustness of large language models in tasks that are not necessarily reasoning-focused. In particular, we show how a wide range of large language models exhibit significantly improved robustness against reference corruption using a simple method called chain-of-defensive-thought, where only a few exemplars with structured and defensive reasoning are provided as demonstrations. Empirically, the improvements can be astounding, especially given the simplicity and applicability of the method. For example, in the Natural Questions task, the accuracy of GPT-4o degrades from 60% to as low as 3% with standard prompting when 1 out of 10 references provided is corrupted with prompt injection attacks. In contrast, GPT-4o using chain-of-defensive-thought prompting maintains an accuracy of 50%. 

---
# OSVBench: Benchmarking LLMs on Specification Generation Tasks for Operating System Verification 

**Authors**: Shangyu Li, Juyong Jiang, Tiancheng Zhao, Jiasi Shen  

**Link**: [PDF](https://arxiv.org/pdf/2504.20964)  

**Abstract**: We introduce OSVBench, a new benchmark for evaluating Large Language Models (LLMs) in generating complete specification code pertaining to operating system kernel verification tasks. The benchmark first defines the specification generation problem into a program synthesis problem within a confined scope of syntax and semantics by providing LLMs with the programming model. The LLMs are required to understand the provided verification assumption and the potential syntax and semantics space to search for, then generate the complete specification for the potentially buggy operating system code implementation under the guidance of the high-level functional description of the operating system. This benchmark is built upon a real-world operating system kernel, Hyperkernel, and consists of 245 complex specification generation tasks in total, each is a long context task of about 20k-30k tokens. Our comprehensive evaluation of 12 LLMs exhibits the limited performance of the current LLMs on the specification generation tasks for operating system verification. Significant disparities in their performance on the benchmark highlight differences in their ability to handle long-context code generation tasks. The evaluation toolkit and benchmark are available at this https URL. 

---
# SetKE: Knowledge Editing for Knowledge Elements Overlap 

**Authors**: Yifan Wei, Xiaoyan Yu, Ran Song, Hao Peng, Angsheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.20972)  

**Abstract**: Large Language Models (LLMs) excel in tasks such as retrieval and question answering but require updates to incorporate new knowledge and reduce inaccuracies and hallucinations. Traditional updating methods, like fine-tuning and incremental learning, face challenges such as overfitting and high computational costs. Knowledge Editing (KE) provides a promising alternative but often overlooks the Knowledge Element Overlap (KEO) phenomenon, where multiple triplets share common elements, leading to editing conflicts. We identify the prevalence of KEO in existing KE datasets and show its significant impact on current KE methods, causing performance degradation in handling such triplets. To address this, we propose a new formulation, Knowledge Set Editing (KSE), and introduce SetKE, a method that edits sets of triplets simultaneously. Experimental results demonstrate that SetKE outperforms existing methods in KEO scenarios on mainstream LLMs. Additionally, we introduce EditSet, a dataset containing KEO triplets, providing a comprehensive benchmark. 

---
# Trace-of-Thought: Enhanced Arithmetic Problem Solving via Reasoning Distillation From Large to Small Language Models 

**Authors**: Tyler McDonald, Ali Emami  

**Link**: [PDF](https://arxiv.org/pdf/2504.20946)  

**Abstract**: As Large Language Models (LLMs) continue to be leveraged for daily tasks, prompt engineering remains an active field of contribution within computational linguistics, particularly in domains requiring specialized knowledge such as arithmetic reasoning. While these LLMs are optimized for a variety of tasks, their exhaustive employment may become computationally or financially cumbersome for small teams. Additionally, complete reliance on proprietary, closed-source models often limits customization and adaptability, posing significant challenges in research and application scalability. Instead, by leveraging open-source models at or below 7 billion parameters, we can optimize our resource usage while still observing remarkable gains over standard prompting approaches. To cultivate this notion, we introduce Trace-of-Thought Prompting, a simple, zero-shot prompt engineering method that instructs LLMs to create observable subproblems using critical problem-solving, specifically designed to enhance arithmetic reasoning capabilities. When applied to open-source models in tandem with GPT-4, we observe that Trace-of-Thought not only allows novel insight into the problem-solving process but also introduces performance gains as large as 125% on language models at or below 7 billion parameters. This approach underscores the potential of open-source initiatives in democratizing AI research and improving the accessibility of high-quality computational linguistics applications. 

---
# Turing Machine Evaluation for Large Language Model 

**Authors**: Haitao Wu, Zongbo Han, Huaxi Huang, Changqing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.20771)  

**Abstract**: With the rapid development and widespread application of Large Language Models (LLMs), rigorous evaluation has become particularly crucial. This research adopts a novel perspective, focusing on evaluating the core computational reasoning ability of LLMs, defined as the capacity of model to accurately understand rules, and execute logically computing operations. This capability assesses the reliability of LLMs as precise executors, and is critical to advanced tasks such as complex code generation and multi-step problem-solving. We propose an evaluation framework based on Universal Turing Machine (UTM) simulation. This framework requires LLMs to strictly follow instructions and track dynamic states, such as tape content and read/write head position, during multi-step computations. To enable standardized evaluation, we developed TMBench, a benchmark for systematically studying the computational reasoning capabilities of LLMs. TMBench provides several key advantages, including knowledge-agnostic evaluation, adjustable difficulty, foundational coverage through Turing machine encoding, and unlimited capacity for instance generation, ensuring scalability as models continue to evolve. We find that model performance on TMBench correlates strongly with performance on other recognized reasoning benchmarks (Pearson correlation coefficient is 0.73), clearly demonstrating that computational reasoning is a significant dimension for measuring the deep capabilities of LLMs. Code and data are available at this https URL. 

---
# Beyond the Last Answer: Your Reasoning Trace Uncovers More than You Think 

**Authors**: Hasan Abed Al Kader Hammoud, Hani Itani, Bernard Ghanem  

**Link**: [PDF](https://arxiv.org/pdf/2504.20708)  

**Abstract**: Large Language Models (LLMs) leverage step-by-step reasoning to solve complex problems. Standard evaluation practice involves generating a complete reasoning trace and assessing the correctness of the final answer presented at its conclusion. In this paper, we challenge the reliance on the final answer by posing the following two questions: Does the final answer reliably represent the model's optimal conclusion? Can alternative reasoning paths yield different results? To answer these questions, we analyze intermediate reasoning steps, termed subthoughts, and propose a method based on our findings. Our approach involves segmenting a reasoning trace into sequential subthoughts based on linguistic cues. We start by prompting the model to generate continuations from the end-point of each intermediate subthought. We extract a potential answer from every completed continuation originating from different subthoughts. We find that aggregating these answers by selecting the most frequent one (the mode) often yields significantly higher accuracy compared to relying solely on the answer derived from the original complete trace. Analyzing the consistency among the answers derived from different subthoughts reveals characteristics that correlate with the model's confidence and correctness, suggesting potential for identifying less reliable answers. Our experiments across various LLMs and challenging mathematical reasoning datasets (AIME2024 and AIME2025) show consistent accuracy improvements, with gains reaching up to 13\% and 10\% respectively. Implementation is available at: this https URL. 

---
# Can LLMs Detect Intrinsic Hallucinations in Paraphrasing and Machine Translation? 

**Authors**: Evangelia Gogoulou, Shorouq Zahra, Liane Guillou, Luise Dürlich, Joakim Nivre  

**Link**: [PDF](https://arxiv.org/pdf/2504.20699)  

**Abstract**: A frequently observed problem with LLMs is their tendency to generate output that is nonsensical, illogical, or factually incorrect, often referred to broadly as hallucination. Building on the recently proposed HalluciGen task for hallucination detection and generation, we evaluate a suite of open-access LLMs on their ability to detect intrinsic hallucinations in two conditional generation tasks: translation and paraphrasing. We study how model performance varies across tasks and language and we investigate the impact of model size, instruction tuning, and prompt choice. We find that performance varies across models but is consistent across prompts. Finally, we find that NLI models perform comparably well, suggesting that LLM-based detectors are not the only viable option for this specific task. 

---
# A Generative-AI-Driven Claim Retrieval System Capable of Detecting and Retrieving Claims from Social Media Platforms in Multiple Languages 

**Authors**: Ivan Vykopal, Martin Hyben, Robert Moro, Michal Gregor, Jakub Simko  

**Link**: [PDF](https://arxiv.org/pdf/2504.20668)  

**Abstract**: Online disinformation poses a global challenge, placing significant demands on fact-checkers who must verify claims efficiently to prevent the spread of false information. A major issue in this process is the redundant verification of already fact-checked claims, which increases workload and delays responses to newly emerging claims. This research introduces an approach that retrieves previously fact-checked claims, evaluates their relevance to a given input, and provides supplementary information to support fact-checkers. Our method employs large language models (LLMs) to filter irrelevant fact-checks and generate concise summaries and explanations, enabling fact-checkers to faster assess whether a claim has been verified before. In addition, we evaluate our approach through both automatic and human assessments, where humans interact with the developed tool to review its effectiveness. Our results demonstrate that LLMs are able to filter out many irrelevant fact-checks and, therefore, reduce effort and streamline the fact-checking process. 

---
# WenyanGPT: A Large Language Model for Classical Chinese Tasks 

**Authors**: Xinyu Yao, Mengdi Wang, Bo Chen, Xiaobing Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2504.20609)  

**Abstract**: Classical Chinese, as the core carrier of Chinese culture, plays a crucial role in the inheritance and study of ancient literature. However, existing natural language processing models primarily optimize for Modern Chinese, resulting in inadequate performance on Classical Chinese. This paper presents a comprehensive solution for Classical Chinese language processing. By continuing pre-training and instruction fine-tuning on the LLaMA3-8B-Chinese model, we construct a large language model, WenyanGPT, which is specifically designed for Classical Chinese tasks. Additionally, we develop an evaluation benchmark dataset, WenyanBENCH. Experimental results on WenyanBENCH demonstrate that WenyanGPT significantly outperforms current advanced LLMs in various Classical Chinese tasks. We make the model's training data, instruction fine-tuning data\footnote, and evaluation benchmark dataset publicly available to promote further research and development in the field of Classical Chinese processing. 

---
# Cooking Up Creativity: A Cognitively-Inspired Approach for Enhancing LLM Creativity through Structured Representations 

**Authors**: Moran Mizrahi, Chen Shani, Gabriel Stanovsky, Dan Jurafsky, Dafna Shahaf  

**Link**: [PDF](https://arxiv.org/pdf/2504.20643)  

**Abstract**: Large Language Models (LLMs) excel at countless tasks, yet struggle with creativity. In this paper, we introduce a novel approach that couples LLMs with structured representations and cognitively inspired manipulations to generate more creative and diverse ideas. Our notion of creativity goes beyond superficial token-level variations; rather, we explicitly recombine structured representations of existing ideas, allowing our algorithm to effectively explore the more abstract landscape of ideas. We demonstrate our approach in the culinary domain with DishCOVER, a model that generates creative recipes. Experiments comparing our model's results to those of GPT-4o show greater diversity. Domain expert evaluations reveal that our outputs, which are mostly coherent and feasible culinary creations, significantly surpass GPT-4o in terms of novelty, thus outperforming it in creative generation. We hope our work inspires further research into structured creativity in AI. 

---
# UniDetox: Universal Detoxification of Large Language Models via Dataset Distillation 

**Authors**: Huimin Lu, Masaru Isonuma, Junichiro Mori, Ichiro Sakata  

**Link**: [PDF](https://arxiv.org/pdf/2504.20500)  

**Abstract**: We present UniDetox, a universally applicable method designed to mitigate toxicity across various large language models (LLMs). Previous detoxification methods are typically model-specific, addressing only individual models or model families, and require careful hyperparameter tuning due to the trade-off between detoxification efficacy and language modeling performance. In contrast, UniDetox provides a detoxification technique that can be universally applied to a wide range of LLMs without the need for separate model-specific tuning. Specifically, we propose a novel and efficient dataset distillation technique for detoxification using contrastive decoding. This approach distills detoxifying representations in the form of synthetic text data, enabling universal detoxification of any LLM through fine-tuning with the distilled text. Our experiments demonstrate that the detoxifying text distilled from GPT-2 can effectively detoxify larger models, including OPT, Falcon, and LLaMA-2. Furthermore, UniDetox eliminates the need for separate hyperparameter tuning for each model, as a single hyperparameter configuration can be seamlessly applied across different models. Additionally, analysis of the detoxifying text reveals a reduction in politically biased content, providing insights into the attributes necessary for effective detoxification of LLMs. 

---
# Fane at SemEval-2025 Task 10: Zero-Shot Entity Framing with Large Language Models 

**Authors**: Enfa Fane, Mihai Surdeanu, Eduardo Blanco, Steven R. Corman  

**Link**: [PDF](https://arxiv.org/pdf/2504.20469)  

**Abstract**: Understanding how news narratives frame entities is crucial for studying media's impact on societal perceptions of events. In this paper, we evaluate the zero-shot capabilities of large language models (LLMs) in classifying framing roles. Through systematic experimentation, we assess the effects of input context, prompting strategies, and task decomposition. Our findings show that a hierarchical approach of first identifying broad roles and then fine-grained roles, outperforms single-step classification. We also demonstrate that optimal input contexts and prompts vary across task levels, highlighting the need for subtask-specific strategies. We achieve a Main Role Accuracy of 89.4% and an Exact Match Ratio of 34.5%, demonstrating the effectiveness of our approach. Our findings emphasize the importance of tailored prompt design and input context optimization for improving LLM performance in entity framing. 

---
# Enhancing LLM Language Adaption through Cross-lingual In-Context Pre-training 

**Authors**: Linjuan Wu, Haoran Wei, Huan Lin, Tianhao Li, Baosong Yang, Weiming Lu  

**Link**: [PDF](https://arxiv.org/pdf/2504.20484)  

**Abstract**: Large language models (LLMs) exhibit remarkable multilingual capabilities despite English-dominated pre-training, attributed to cross-lingual mechanisms during pre-training. Existing methods for enhancing cross-lingual transfer remain constrained by parallel resources, suffering from limited linguistic and domain coverage. We propose Cross-lingual In-context Pre-training (CrossIC-PT), a simple and scalable approach that enhances cross-lingual transfer by leveraging semantically related bilingual texts via simple next-word prediction. We construct CrossIC-PT samples by interleaving semantic-related bilingual Wikipedia documents into a single context window. To access window size constraints, we implement a systematic segmentation policy to split long bilingual document pairs into chunks while adjusting the sliding window mechanism to preserve contextual coherence. We further extend data availability through a semantic retrieval framework to construct CrossIC-PT samples from web-crawled corpus. Experimental results demonstrate that CrossIC-PT improves multilingual performance on three models (Llama-3.1-8B, Qwen2.5-7B, and Qwen2.5-1.5B) across six target languages, yielding performance gains of 3.79%, 3.99%, and 1.95%, respectively, with additional improvements after data augmentation. 

---
# DMDTEval: An Evaluation and Analysis of LLMs on Disambiguation in Multi-domain Translation 

**Authors**: Zhibo Man, Yuanmeng Chen, Yujie Zhang, Yufeng Chen, Jinan Xu  

**Link**: [PDF](https://arxiv.org/pdf/2504.20371)  

**Abstract**: Currently, Large Language Models (LLMs) have achieved remarkable results in machine translation. However, their performance in multi-domain translation (MDT) is less satisfactory; the meanings of words can vary across different domains, highlighting the significant ambiguity inherent in MDT. Therefore, evaluating the disambiguation ability of LLMs in MDT remains an open problem. To this end, we present an evaluation and analysis of LLMs on disambiguation in multi-domain translation (DMDTEval), our systematic evaluation framework consisting of three critical aspects: (1) we construct a translation test set with multi-domain ambiguous word annotation, (2) we curate a diverse set of disambiguation prompting templates, and (3) we design precise disambiguation metrics, and study the efficacy of various prompting strategies on multiple state-of-the-art LLMs. Our extensive experiments reveal a number of crucial findings that we believe will pave the way and also facilitate further research in the critical area of improving the disambiguation of LLMs. 

---
# Local Prompt Optimization 

**Authors**: Yash Jain, Vishal Chowdhary  

**Link**: [PDF](https://arxiv.org/pdf/2504.20355)  

**Abstract**: In recent years, the use of prompts to guide the output of Large Language Models have increased dramatically. However, even the best of experts struggle to choose the correct words to stitch up a prompt for the desired task. To solve this, LLM driven prompt optimization emerged as an important problem. Existing prompt optimization methods optimize a prompt globally, where in all the prompt tokens have to be optimized over a large vocabulary while solving a complex task. The large optimization space (tokens) leads to insufficient guidance for a better prompt. In this work, we introduce Local Prompt Optimization (LPO) that integrates with any general automatic prompt engineering method. We identify the optimization tokens in a prompt and nudge the LLM to focus only on those tokens in its optimization step. We observe remarkable performance improvements on Math Reasoning (GSM8k and MultiArith) and BIG-bench Hard benchmarks across various automatic prompt engineering methods. Further, we show that LPO converges to the optimal prompt faster than global methods. 

---
# Information Gravity: A Field-Theoretic Model for Token Selection in Large Language Models 

**Authors**: Maryna Vyshnyvetska  

**Link**: [PDF](https://arxiv.org/pdf/2504.20951)  

**Abstract**: We propose a theoretical model called "information gravity" to describe the text generation process in large language models (LLMs). The model uses physical apparatus from field theory and spacetime geometry to formalize the interaction between user queries and the probability distribution of generated tokens. A query is viewed as an object with "information mass" that curves the semantic space of the model, creating gravitational potential wells that "attract" tokens during generation. This model offers a mechanism to explain several observed phenomena in LLM behavior, including hallucinations (emerging from low-density semantic voids), sensitivity to query formulation (due to semantic field curvature changes), and the influence of sampling temperature on output diversity. 

---
# Toward Evaluative Thinking: Meta Policy Optimization with Evolving Reward Models 

**Authors**: Zae Myung Kim, Chanwoo Park, Vipul Raheja, Dongyeop Kang  

**Link**: [PDF](https://arxiv.org/pdf/2504.20157)  

**Abstract**: Reward-based alignment methods for large language models (LLMs) face two key limitations: vulnerability to reward hacking, where models exploit flaws in the reward signal; and reliance on brittle, labor-intensive prompt engineering when LLMs are used as reward models. We introduce Meta Policy Optimization (MPO), a framework that addresses these challenges by integrating a meta-reward model that dynamically refines the reward model's prompt throughout training. In MPO, the meta-reward model monitors the evolving training context and continuously adjusts the reward model's prompt to maintain high alignment, providing an adaptive reward signal that resists exploitation by the policy. This meta-learning approach promotes a more stable policy optimization, and greatly reduces the need for manual reward prompt design. It yields performance on par with or better than models guided by extensively hand-crafted reward prompts. Furthermore, we show that MPO maintains its effectiveness across diverse tasks, such as question answering and mathematical reasoning, without requiring specialized reward designs. Beyond standard RLAIF, MPO's meta-learning formulation is readily extensible to higher-level alignment frameworks. Overall, this method addresses theoretical and practical challenges in reward-based RL alignment for LLMs, paving the way for more robust and adaptable alignment strategies. The code and models will be publicly shared. 

---
# Enhancing Systematic Reviews with Large Language Models: Using GPT-4 and Kimi 

**Authors**: Dandan Chen Kaptur, Yue Huang, Xuejun Ryan Ji, Yanhui Guo, Bradley Kaptur  

**Link**: [PDF](https://arxiv.org/pdf/2504.20276)  

**Abstract**: This research delved into GPT-4 and Kimi, two Large Language Models (LLMs), for systematic reviews. We evaluated their performance by comparing LLM-generated codes with human-generated codes from a peer-reviewed systematic review on assessment. Our findings suggested that the performance of LLMs fluctuates by data volume and question complexity for systematic reviews. 

---
# ChestX-Reasoner: Advancing Radiology Foundation Models with Reasoning through Step-by-Step Verification 

**Authors**: Ziqing Fan, Cheng Liang, Chaoyi Wu, Ya Zhang, Yanfeng Wang, Weidi Xie  

**Link**: [PDF](https://arxiv.org/pdf/2504.20930)  

**Abstract**: Recent advances in reasoning-enhanced large language models (LLMs) and multimodal LLMs (MLLMs) have significantly improved performance in complex tasks, yet medical AI models often overlook the structured reasoning processes inherent in clinical practice. In this work, we present ChestX-Reasoner, a radiology diagnosis MLLM designed to leverage process supervision mined directly from clinical reports, reflecting the step-by-step reasoning followed by radiologists. We construct a large dataset by extracting and refining reasoning chains from routine radiology reports. Our two-stage training framework combines supervised fine-tuning and reinforcement learning guided by process rewards to better align model reasoning with clinical standards. We introduce RadRBench-CXR, a comprehensive benchmark featuring 59K visual question answering samples with 301K clinically validated reasoning steps, and propose RadRScore, a metric evaluating reasoning factuality, completeness, and effectiveness. ChestX-Reasoner outperforms existing medical and general-domain MLLMs in both diagnostic accuracy and reasoning ability, achieving 16%, 5.9%, and 18% improvements in reasoning ability compared to the best medical MLLM, the best general MLLM, and its base model, respectively, as well as 3.3%, 24%, and 27% improvements in outcome accuracy. All resources are open-sourced to facilitate further research in medical reasoning MLLMs. 

---
# Reinforcement Learning for Reasoning in Large Language Models with One Training Example 

**Authors**: Yiping Wang, Qing Yang, Zhiyuan Zeng, Liliang Ren, Lucas Liu, Baolin Peng, Hao Cheng, Xuehai He, Kuan Wang, Jianfeng Gao, Weizhu Chen, Shuohang Wang, Simon Shaolei Du, Yelong Shen  

**Link**: [PDF](https://arxiv.org/pdf/2504.20571)  

**Abstract**: We show that reinforcement learning with verifiable reward using one training example (1-shot RLVR) is effective in incentivizing the math reasoning capabilities of large language models (LLMs). Applying RLVR to the base model Qwen2.5-Math-1.5B, we identify a single example that elevates model performance on MATH500 from 36.0% to 73.6%, and improves the average performance across six common mathematical reasoning benchmarks from 17.6% to 35.7%. This result matches the performance obtained using the 1.2k DeepScaleR subset (MATH500: 73.6%, average: 35.9%), which includes the aforementioned example. Similar substantial improvements are observed across various models (Qwen2.5-Math-7B, Llama3.2-3B-Instruct, DeepSeek-R1-Distill-Qwen-1.5B), RL algorithms (GRPO and PPO), and different math examples (many of which yield approximately 30% or greater improvement on MATH500 when employed as a single training example). In addition, we identify some interesting phenomena during 1-shot RLVR, including cross-domain generalization, increased frequency of self-reflection, and sustained test performance improvement even after the training accuracy has saturated, a phenomenon we term post-saturation generalization. Moreover, we verify that the effectiveness of 1-shot RLVR primarily arises from the policy gradient loss, distinguishing it from the "grokking" phenomenon. We also show the critical role of promoting exploration (e.g., by adding entropy loss with an appropriate coefficient) in 1-shot RLVR training. As a bonus, we observe that applying entropy loss alone, without any outcome reward, significantly enhances Qwen2.5-Math-1.5B's performance on MATH500 by 27.4%. These findings can inspire future work on RLVR data efficiency and encourage a re-examination of both recent progress and the underlying mechanisms in RLVR. Our code, model, and data are open source at this https URL 

---
# The Leaderboard Illusion 

**Authors**: Shivalika Singh, Yiyang Nan, Alex Wang, Daniel D'Souza, Sayash Kapoor, Ahmet Üstün, Sanmi Koyejo, Yuntian Deng, Shayne Longpre, Noah Smith, Beyza Ermis, Marzieh Fadaee, Sara Hooker  

**Link**: [PDF](https://arxiv.org/pdf/2504.20879)  

**Abstract**: Measuring progress is fundamental to the advancement of any scientific field. As benchmarks play an increasingly central role, they also grow more susceptible to distortion. Chatbot Arena has emerged as the go-to leaderboard for ranking the most capable AI systems. Yet, in this work we identify systematic issues that have resulted in a distorted playing field. We find that undisclosed private testing practices benefit a handful of providers who are able to test multiple variants before public release and retract scores if desired. We establish that the ability of these providers to choose the best score leads to biased Arena scores due to selective disclosure of performance results. At an extreme, we identify 27 private LLM variants tested by Meta in the lead-up to the Llama-4 release. We also establish that proprietary closed models are sampled at higher rates (number of battles) and have fewer models removed from the arena than open-weight and open-source alternatives. Both these policies lead to large data access asymmetries over time. Providers like Google and OpenAI have received an estimated 19.2% and 20.4% of all data on the arena, respectively. In contrast, a combined 83 open-weight models have only received an estimated 29.7% of the total data. We show that access to Chatbot Arena data yields substantial benefits; even limited additional data can result in relative performance gains of up to 112% on the arena distribution, based on our conservative estimates. Together, these dynamics result in overfitting to Arena-specific dynamics rather than general model quality. The Arena builds on the substantial efforts of both the organizers and an open community that maintains this valuable evaluation platform. We offer actionable recommendations to reform the Chatbot Arena's evaluation framework and promote fairer, more transparent benchmarking for the field 

---
# ResearchCodeAgent: An LLM Multi-Agent System for Automated Codification of Research Methodologies 

**Authors**: Shubham Gandhi, Dhruv Shah, Manasi Patwardhan, Lovekesh Vig, Gautam Shroff  

**Link**: [PDF](https://arxiv.org/pdf/2504.20117)  

**Abstract**: In this paper we introduce ResearchCodeAgent, a novel multi-agent system leveraging large language models (LLMs) agents to automate the codification of research methodologies described in machine learning literature. The system bridges the gap between high-level research concepts and their practical implementation, allowing researchers auto-generating code of existing research papers for benchmarking or building on top-of existing methods specified in the literature with availability of partial or complete starter code. ResearchCodeAgent employs a flexible agent architecture with a comprehensive action suite, enabling context-aware interactions with the research environment. The system incorporates a dynamic planning mechanism, utilizing both short and long-term memory to adapt its approach iteratively. We evaluate ResearchCodeAgent on three distinct machine learning tasks with distinct task complexity and representing different parts of the ML pipeline: data augmentation, optimization, and data batching. Our results demonstrate the system's effectiveness and generalizability, with 46.9% of generated code being high-quality and error-free, and 25% showing performance improvements over baseline implementations. Empirical analysis shows an average reduction of 57.9% in coding time compared to manual implementation. We observe higher gains for more complex tasks. ResearchCodeAgent represents a significant step towards automating the research implementation process, potentially accelerating the pace of machine learning research. 

---
# RAGEN: Understanding Self-Evolution in LLM Agents via Multi-Turn Reinforcement Learning 

**Authors**: Zihan Wang, Kangrui Wang, Qineng Wang, Pingyue Zhang, Linjie Li, Zhengyuan Yang, Kefan Yu, Minh Nhat Nguyen, Licheng Liu, Eli Gottlieb, Monica Lam, Yiping Lu, Kyunghyun Cho, Jiajun Wu, Li Fei-Fei, Lijuan Wang, Yejin Choi, Manling Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.20073)  

**Abstract**: Training large language models (LLMs) as interactive agents presents unique challenges including long-horizon decision making and interacting with stochastic environment feedback. While reinforcement learning (RL) has enabled progress in static tasks, multi-turn agent RL training remains underexplored. We propose StarPO (State-Thinking-Actions-Reward Policy Optimization), a general framework for trajectory-level agent RL, and introduce RAGEN, a modular system for training and evaluating LLM agents. Our study on three stylized environments reveals three core findings. First, our agent RL training shows a recurring mode of Echo Trap where reward variance cliffs and gradient spikes; we address this with StarPO-S, a stabilized variant with trajectory filtering, critic incorporation, and decoupled clipping. Second, we find the shaping of RL rollouts would benefit from diverse initial states, medium interaction granularity and more frequent sampling. Third, we show that without fine-grained, reasoning-aware reward signals, agent reasoning hardly emerge through multi-turn RL and they may show shallow strategies or hallucinated thoughts. Code and environments are available at this https URL. 

---
# Cognitive maps are generative programs 

**Authors**: Marta Kryven, Cole Wyeth, Aidan Curtis, Kevin Ellis  

**Link**: [PDF](https://arxiv.org/pdf/2504.20628)  

**Abstract**: Making sense of the world and acting in it relies on building simplified mental representations that abstract away aspects of reality. This principle of cognitive mapping is universal to agents with limited resources. Living organisms, people, and algorithms all face the problem of forming functional representations of their world under various computing constraints. In this work, we explore the hypothesis that human resource-efficient planning may arise from representing the world as predictably structured. Building on the metaphor of concepts as programs, we propose that cognitive maps can take the form of generative programs that exploit predictability and redundancy, in contrast to directly encoding spatial layouts. We use a behavioral experiment to show that people who navigate in structured spaces rely on modular planning strategies that align with programmatic map representations. We describe a computational model that predicts human behavior in a variety of structured scenarios. This model infers a small distribution over possible programmatic cognitive maps conditioned on human prior knowledge of the world, and uses this distribution to generate resource-efficient plans. Our models leverages a Large Language Model as an embedding of human priors, implicitly learned through training on a vast corpus of human data. Our model demonstrates improved computational efficiency, requires drastically less memory, and outperforms unstructured planning algorithms with cognitive constraints at predicting human behavior, suggesting that human planning strategies rely on programmatic cognitive maps. 

---
# MuRAL: A Multi-Resident Ambient Sensor Dataset Annotated with Natural Language for Activities of Daily Living 

**Authors**: Xi Chen, Julien Cumin, Fano Ramparany, Dominique Vaufreydaz  

**Link**: [PDF](https://arxiv.org/pdf/2504.20505)  

**Abstract**: Recent advances in Large Language Models (LLMs) have shown promising potential for human activity recognition (HAR) using ambient sensors, especially through natural language reasoning and zero-shot learning. However, existing datasets such as CASAS, ARAS, and MARBLE were not originally designed with LLMs in mind and therefore lack the contextual richness, complexity, and annotation granularity required to fully exploit LLM capabilities. In this paper, we introduce MuRAL, the first Multi-Resident Ambient sensor dataset with natural Language, comprising over 21 hours of multi-user sensor data collected from 21 sessions in a smart-home environment. MuRAL is annotated with fine-grained natural language descriptions, resident identities, and high-level activity labels, all situated in dynamic, realistic multi-resident settings. We benchmark MuRAL using state-of-the-art LLMs for three core tasks: subject assignment, action description, and activity classification. Our results demonstrate that while LLMs can provide rich semantic interpretations of ambient data, current models still face challenges in handling multi-user ambiguity and under-specified sensor contexts. We release MuRAL to support future research on LLM-powered, explainable, and socially aware activity understanding in smart environments. For access to the dataset, please reach out to us via the provided contact information. A direct link for dataset retrieval will be made available at this location in due course. 

---
# PaRT: Enhancing Proactive Social Chatbots with Personalized Real-Time Retrieval 

**Authors**: Zihan Niu, Zheyong Xie, Shaosheng Cao, Chonggang Lu, Zheyu Ye, Tong Xu, Zuozhu Liu, Yan Gao, Jia Chen, Zhe Xu, Yi Wu, Yao Hu  

**Link**: [PDF](https://arxiv.org/pdf/2504.20624)  

**Abstract**: Social chatbots have become essential intelligent companions in daily scenarios ranging from emotional support to personal interaction. However, conventional chatbots with passive response mechanisms usually rely on users to initiate or sustain dialogues by bringing up new topics, resulting in diminished engagement and shortened dialogue duration. In this paper, we present PaRT, a novel framework enabling context-aware proactive dialogues for social chatbots through personalized real-time retrieval and generation. Specifically, PaRT first integrates user profiles and dialogue context into a large language model (LLM), which is initially prompted to refine user queries and recognize their underlying intents for the upcoming conversation. Guided by refined intents, the LLM generates personalized dialogue topics, which then serve as targeted queries to retrieve relevant passages from RedNote. Finally, we prompt LLMs with summarized passages to generate knowledge-grounded and engagement-optimized responses. Our approach has been running stably in a real-world production environment for more than 30 days, achieving a 21.77\% improvement in the average duration of dialogues. 

---
# RV-Syn: Rational and Verifiable Mathematical Reasoning Data Synthesis based on Structured Function Library 

**Authors**: Jiapeng Wang, Jinhao Jiang, Zhiqiang Zhang, Jun Zhou, Wayne Xin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2504.20426)  

**Abstract**: The advancement of reasoning capabilities in Large Language Models (LLMs) requires substantial amounts of high-quality reasoning data, particularly in mathematics. Existing data synthesis methods, such as data augmentation from annotated training sets or direct question generation based on relevant knowledge points and documents, have expanded datasets but face challenges in mastering the inner logic of the problem during generation and ensuring the verifiability of the solutions. To address these issues, we propose RV-Syn, a novel Rational and Verifiable mathematical Synthesis approach. RV-Syn constructs a structured mathematical operation function library based on initial seed problems and generates computational graphs as solutions by combining Python-formatted functions from this library. These graphs are then back-translated into complex problems. Based on the constructed computation graph, we achieve solution-guided logic-aware problem generation. Furthermore, the executability of the computational graph ensures the verifiability of the solving process. Experimental results show that RV-Syn surpasses existing synthesis methods, including those involving human-generated problems, achieving greater efficient data scaling. This approach provides a scalable framework for generating high-quality reasoning datasets. 

---
# Skill Discovery for Software Scripting Automation via Offline Simulations with LLMs 

**Authors**: Paiheng Xu, Gang Wu, Xiang Chen, Tong Yu, Chang Xiao, Franck Dernoncourt, Tianyi Zhou, Wei Ai, Viswanathan Swaminathan  

**Link**: [PDF](https://arxiv.org/pdf/2504.20406)  

**Abstract**: Scripting interfaces enable users to automate tasks and customize software workflows, but creating scripts traditionally requires programming expertise and familiarity with specific APIs, posing barriers for many users. While Large Language Models (LLMs) can generate code from natural language queries, runtime code generation is severely limited due to unverified code, security risks, longer response times, and higher computational costs. To bridge the gap, we propose an offline simulation framework to curate a software-specific skillset, a collection of verified scripts, by exploiting LLMs and publicly available scripting guides. Our framework comprises two components: (1) task creation, using top-down functionality guidance and bottom-up API synergy exploration to generate helpful tasks; and (2) skill generation with trials, refining and validating scripts based on execution feedback. To efficiently navigate the extensive API landscape, we introduce a Graph Neural Network (GNN)-based link prediction model to capture API synergy, enabling the generation of skills involving underutilized APIs and expanding the skillset's diversity. Experiments with Adobe Illustrator demonstrate that our framework significantly improves automation success rates, reduces response time, and saves runtime token costs compared to traditional runtime code generation. This is the first attempt to use software scripting interfaces as a testbed for LLM-based systems, highlighting the advantages of leveraging execution feedback in a controlled environment and offering valuable insights into aligning AI capabilities with user needs in specialized software domains. 

---
# TAMO:Fine-Grained Root Cause Analysis via Tool-Assisted LLM Agent with Multi-Modality Observation Data 

**Authors**: Qi Wang, Xiao Zhang, Mingyi Li, Yuan Yuan, Mengbai Xiao, Fuzhen Zhuang, Dongxiao Yu  

**Link**: [PDF](https://arxiv.org/pdf/2504.20462)  

**Abstract**: With the development of distributed systems, microservices and cloud native technologies have become central to modern enterprise software development. Despite bringing significant advantages, these technologies also increase system complexity and operational challenges. Traditional root cause analysis (RCA) struggles to achieve automated fault response, heavily relying on manual intervention. In recent years, large language models (LLMs) have made breakthroughs in contextual inference and domain knowledge integration, providing new solutions for Artificial Intelligence for Operations (AIOps). However, Existing LLM-based approaches face three key challenges: text input constraints, dynamic service dependency hallucinations, and context window limitations. To address these issues, we propose a tool-assisted LLM agent with multi-modality observation data, namely TAMO, for fine-grained RCA. It unifies multi-modal observational data into time-aligned representations to extract consistent features and employs specialized root cause localization and fault classification tools for perceiving the contextual environment. This approach overcomes the limitations of LLM in handling real-time changing service dependencies and raw observational data and guides LLM to generate repair strategies aligned with system contexts by structuring key information into a prompt. Experimental results show that TAMO performs well in root cause analysis when dealing with public datasets characterized by heterogeneity and common fault types, demonstrating its effectiveness. 

---
# Evolution of AI in Education: Agentic Workflows 

**Authors**: Firuz Kamalov, David Santandreu Calonge, Linda Smail, Dilshod Azizov, Dimple R. Thadani, Theresa Kwong, Amara Atif  

**Link**: [PDF](https://arxiv.org/pdf/2504.20082)  

**Abstract**: Artificial intelligence (AI) has transformed various aspects of education, with large language models (LLMs) driving advancements in automated tutoring, assessment, and content generation. However, conventional LLMs are constrained by their reliance on static training data, limited adaptability, and lack of reasoning. To address these limitations and foster more sustainable technological practices, AI agents have emerged as a promising new avenue for educational innovation. In this review, we examine agentic workflows in education according to four major paradigms: reflection, planning, tool use, and multi-agent collaboration. We critically analyze the role of AI agents in education through these key design paradigms, exploring their advantages, applications, and challenges. To illustrate the practical potential of agentic systems, we present a proof-of-concept application: a multi-agent framework for automated essay scoring. Preliminary results suggest this agentic approach may offer improved consistency compared to stand-alone LLMs. Our findings highlight the transformative potential of AI agents in educational settings while underscoring the need for further research into their interpretability, trustworthiness, and sustainable impact on pedagogical impact. 

---
# Toward Efficient Exploration by Large Language Model Agents 

**Authors**: Dilip Arumugam, Thomas L. Griffiths  

**Link**: [PDF](https://arxiv.org/pdf/2504.20997)  

**Abstract**: A burgeoning area within reinforcement learning (RL) is the design of sequential decision-making agents centered around large language models (LLMs). While autonomous decision-making agents powered by modern LLMs could facilitate numerous real-world applications, such successes demand agents that are capable of data-efficient RL. One key obstacle to achieving data efficiency in RL is exploration, a challenge that we demonstrate many recent proposals for LLM agent designs struggle to contend with. Meanwhile, classic algorithms from the RL literature known to gracefully address exploration require technical machinery that can be challenging to operationalize in purely natural language settings. In this work, rather than relying on finetuning or in-context learning to coax LLMs into implicitly imitating a RL algorithm, we illustrate how LLMs can be used to explicitly implement an existing RL algorithm (Posterior Sampling for Reinforcement Learning) whose capacity for statistically-efficient exploration is already well-studied. We offer empirical results demonstrating how our LLM-based implementation of a known, data-efficient RL algorithm can be considerably more effective in natural language tasks that demand prudent exploration. 

---
# Classifier-to-Bias: Toward Unsupervised Automatic Bias Detection for Visual Classifiers 

**Authors**: Quentin Guimard, Moreno D'Incà, Massimiliano Mancini, Elisa Ricci  

**Link**: [PDF](https://arxiv.org/pdf/2504.20902)  

**Abstract**: A person downloading a pre-trained model from the web should be aware of its biases. Existing approaches for bias identification rely on datasets containing labels for the task of interest, something that a non-expert may not have access to, or may not have the necessary resources to collect: this greatly limits the number of tasks where model biases can be identified. In this work, we present Classifier-to-Bias (C2B), the first bias discovery framework that works without access to any labeled data: it only relies on a textual description of the classification task to identify biases in the target classification model. This description is fed to a large language model to generate bias proposals and corresponding captions depicting biases together with task-specific target labels. A retrieval model collects images for those captions, which are then used to assess the accuracy of the model w.r.t. the given biases. C2B is training-free, does not require any annotations, has no constraints on the list of biases, and can be applied to any pre-trained model on any classification task. Experiments on two publicly available datasets show that C2B discovers biases beyond those of the original datasets and outperforms a recent state-of-the-art bias detection baseline that relies on task-specific annotations, being a promising first step toward addressing task-agnostic unsupervised bias detection. 

---
# Reinforcement Learning for LLM Reasoning Under Memory Constraints 

**Authors**: Alan Lee, Harry Tong  

**Link**: [PDF](https://arxiv.org/pdf/2504.20834)  

**Abstract**: We explore reinforcement learning (RL) techniques to enhance reasoning within targeted problem spaces in large language models (LLMs) under memory and compute constraints. Our focus is on critic-free methods compatible with LoRA fine-tuning on a single 40GB GPU, a common limitation in academic settings. We introduce S-GRPO, a memory-efficient variant of Group Relative Policy Optimization, and T-SPMO, a token-level prefix matching strategy for fine-grained credit assignment. Despite limited resources, when used to fine-tune Qwen2-1.5B both methods significantly improve SVAMP benchmark accuracy from 46% to above 70% using LoRA training. T-SPMO also excels in multi-digit multiplication tasks, underscoring the potential of RL fine-tuning under hardware constraints. Additionally, we find that our full-token GRPO baseline under LoRA fine-tuning did not improve model performance (compared to base model) on either task, suggesting that our memory-efficient methods may act as a form of regularization that stabilizes training when only a small subset of parameters are updated. 

---
# Using LLMs in Generating Design Rationale for Software Architecture Decisions 

**Authors**: Xiyu Zhou, Ruiyin Li, Peng Liang, Beiqi Zhang, Mojtaba Shahin, Zengyang Li, Chen Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.20781)  

**Abstract**: Design Rationale (DR) for software architecture decisions refers to the reasoning underlying architectural choices, which provides valuable insights into the different phases of the architecting process throughout software development. However, in practice, DR is often inadequately documented due to a lack of motivation and effort from developers. With the recent advancements in Large Language Models (LLMs), their capabilities in text comprehension, reasoning, and generation may enable the generation and recovery of DR for architecture decisions. In this study, we evaluated the performance of LLMs in generating DR for architecture decisions. First, we collected 50 Stack Overflow (SO) posts, 25 GitHub issues, and 25 GitHub discussions related to architecture decisions to construct a dataset of 100 architecture-related problems. Then, we selected five LLMs to generate DR for the architecture decisions with three prompting strategies, including zero-shot, chain of thought (CoT), and LLM-based agents. With the DR provided by human experts as ground truth, the Precision of LLM-generated DR with the three prompting strategies ranges from 0.267 to 0.278, Recall from 0.627 to 0.715, and F1-score from 0.351 to 0.389. Additionally, 64.45% to 69.42% of the arguments of DR not mentioned by human experts are also helpful, 4.12% to 4.87% of the arguments have uncertain correctness, and 1.59% to 3.24% of the arguments are potentially misleading. Based on the results, we further discussed the pros and cons of the three prompting strategies and the strengths and limitations of the DR generated by LLMs. 

---
# Hallucination by Code Generation LLMs: Taxonomy, Benchmarks, Mitigation, and Challenges 

**Authors**: Yunseo Lee, John Youngeun Song, Dongsun Kim, Jindae Kim, Mijung Kim, Jaechang Nam  

**Link**: [PDF](https://arxiv.org/pdf/2504.20799)  

**Abstract**: Recent technical breakthroughs in large language models (LLMs) have enabled them to fluently generate source code. Software developers often leverage both general-purpose and code-specialized LLMs to revise existing code or even generate a whole function from scratch. These capabilities are also beneficial in no-code or low-code contexts, in which one can write programs without a technical background. However, due to their internal design, LLMs are prone to generating hallucinations, which are incorrect, nonsensical, and not justifiable information but difficult to identify its presence. This problem also occurs when generating source code. Once hallucinated code is produced, it is often challenging for users to identify and fix it, especially when such hallucinations can be identified under specific execution paths. As a result, the hallucinated code may remain unnoticed within the codebase. This survey investigates recent studies and techniques relevant to hallucinations generated by CodeLLMs. We categorize the types of hallucinations in the code generated by CodeLLMs, review existing benchmarks and mitigation strategies, and identify open challenges. Based on these findings, this survey outlines further research directions in the detection and removal of hallucinations produced by CodeLLMs. 

---
# CoCo-Bench: A Comprehensive Code Benchmark For Multi-task Large Language Model Evaluation 

**Authors**: Wenjing Yin, Tianze Sun, Yijiong Yu, Jiawei Fang, Guangyao Su, Jiancheng Wang, Zekun Wang, Wei Wang, Ran Chen, Ziyun Dai, Shuai Yuan, Menghang Dong, Peng Luo, Dong Cao, Da Lei, Yajun Zhang, Hao Chen, Xiang Ma, Yong Liu, Weifeng Liu, Yuanjian Xu, Ji Pei  

**Link**: [PDF](https://arxiv.org/pdf/2504.20673)  

**Abstract**: Large language models (LLMs) play a crucial role in software engineering, excelling in tasks like code generation and maintenance. However, existing benchmarks are often narrow in scope, focusing on a specific task and lack a comprehensive evaluation framework that reflects real-world applications. To address these gaps, we introduce CoCo-Bench (Comprehensive Code Benchmark), designed to evaluate LLMs across four critical dimensions: code understanding, code generation, code modification, and code review. These dimensions capture essential developer needs, ensuring a more systematic and representative evaluation. CoCo-Bench includes multiple programming languages and varying task difficulties, with rigorous manual review to ensure data quality and accuracy. Empirical results show that CoCo-Bench aligns with existing benchmarks while uncovering significant variations in model performance, effectively highlighting strengths and weaknesses. By offering a holistic and objective evaluation, CoCo-Bench provides valuable insights to guide future research and technological advancements in code-oriented LLMs, establishing a reliable benchmark for the field. 

---
# The Hidden Risks of LLM-Generated Web Application Code: A Security-Centric Evaluation of Code Generation Capabilities in Large Language Models 

**Authors**: Swaroop Dora, Deven Lunkad, Naziya Aslam, S. Venkatesan, Sandeep Kumar Shukla  

**Link**: [PDF](https://arxiv.org/pdf/2504.20612)  

**Abstract**: The rapid advancement of Large Language Models (LLMs) has enhanced software development processes, minimizing the time and effort required for coding and enhancing developer productivity. However, despite their potential benefits, code generated by LLMs has been shown to generate insecure code in controlled environments, raising critical concerns about their reliability and security in real-world applications. This paper uses predefined security parameters to evaluate the security compliance of LLM-generated code across multiple models, such as ChatGPT, DeepSeek, Claude, Gemini and Grok. The analysis reveals critical vulnerabilities in authentication mechanisms, session management, input validation and HTTP security headers. Although some models implement security measures to a limited extent, none fully align with industry best practices, highlighting the associated risks in automated software development. Our findings underscore that human expertise is crucial to ensure secure software deployment or review of LLM-generated code. Also, there is a need for robust security assessment frameworks to enhance the reliability of LLM-generated code in real-world applications. 

---
# Token-Efficient Prompt Injection Attack: Provoking Cessation in LLM Reasoning via Adaptive Token Compression 

**Authors**: Yu Cui, Yujun Cai, Yiwei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.20493)  

**Abstract**: While reasoning large language models (LLMs) demonstrate remarkable performance across various tasks, they also contain notable security vulnerabilities. Recent research has uncovered a "thinking-stopped" vulnerability in DeepSeek-R1, where model-generated reasoning tokens can forcibly interrupt the inference process, resulting in empty responses that compromise LLM-integrated applications. However, existing methods triggering this vulnerability require complex mathematical word problems with long prompts--even exceeding 5,000 tokens. To reduce the token cost and formally define this vulnerability, we propose a novel prompt injection attack named "Reasoning Interruption Attack", based on adaptive token compression. We demonstrate that simple standalone arithmetic tasks can effectively trigger this vulnerability, and the prompts based on such tasks exhibit simpler logical structures than mathematical word problems. We develop a systematic approach to efficiently collect attack prompts and an adaptive token compression framework that utilizes LLMs to automatically compress these prompts. Experiments show our compression framework significantly reduces prompt length while maintaining effective attack capabilities. We further investigate the attack's performance via output prefix and analyze the underlying causes of the vulnerability, providing valuable insights for improving security in reasoning LLMs. 

---
# GaLore 2: Large-Scale LLM Pre-Training by Gradient Low-Rank Projection 

**Authors**: DiJia Su, Andrew Gu, Jane Xu, Yuandong Tian, Jiawei Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2504.20437)  

**Abstract**: Large language models (LLMs) have revolutionized natural language understanding and generation but face significant memory bottlenecks during training. GaLore, Gradient Low-Rank Projection, addresses this issue by leveraging the inherent low-rank structure of weight gradients, enabling substantial memory savings without sacrificing performance. Recent works further extend GaLore from various aspects, including low-bit quantization and higher-order tensor structures. However, there are several remaining challenges for GaLore, such as the computational overhead of SVD for subspace updates and the integration with state-of-the-art training parallelization strategies (e.g., FSDP). In this paper, we present GaLore 2, an efficient and scalable GaLore framework that addresses these challenges and incorporates recent advancements. In addition, we demonstrate the scalability of GaLore 2 by pre-training Llama 7B from scratch using up to 500 billion training tokens, highlighting its potential impact on real LLM pre-training scenarios. 

---
# Can Large Language Models Learn Formal Logic? A Data-Driven Training and Evaluation Framework 

**Authors**: Yuan Xia, Akanksha Atrey, Fadoua Khmaissia, Kedar S. Namjoshi  

**Link**: [PDF](https://arxiv.org/pdf/2504.20213)  

**Abstract**: This paper investigates the logical reasoning capabilities of large language models (LLMs). For a precisely defined yet tractable formulation, we choose the conceptually simple but technically complex task of constructing proofs in Boolean logic. A trained LLM receives as input a set of assumptions and a goal, and produces as output a proof that formally derives the goal from the assumptions. Incorrect proofs are caught by an automated proof checker. A critical obstacle for training is the scarcity of real-world proofs. We propose an efficient, randomized procedure for synthesizing valid proofs and introduce Template Transformation, a data augmentation technique that enhances the model's ability to handle complex logical expressions. The central evaluation question is whether an LLM has indeed learned to reason. We propose tests to measure the reasoning ability of a black-box LLM. By these measures, experiments demonstrate strong reasoning capabilities for assertions with short proofs, which decline with proof complexity. Notably, template transformation improves accuracy even for smaller models, suggesting its effectiveness across model scales. 

---
# Prompting LLMs for Code Editing: Struggles and Remedies 

**Authors**: Daye Nam, Ahmed Omran, Ambar Murillo, Saksham Thakur, Abner Araujo, Marcel Blistein, Alexander Frömmgen, Vincent Hellendoorn, Satish Chandra  

**Link**: [PDF](https://arxiv.org/pdf/2504.20196)  

**Abstract**: Large Language Models (LLMs) are rapidly transforming software engineering, with coding assistants embedded in an IDE becoming increasingly prevalent. While research has focused on improving the tools and understanding developer perceptions, a critical gap exists in understanding how developers actually use these tools in their daily workflows, and, crucially, where they struggle. This paper addresses part of this gap through a multi-phased investigation of developer interactions with an LLM-powered code editing and transformation feature, Transform Code, in an IDE widely used at Google. First, we analyze telemetry logs of the feature usage, revealing that frequent re-prompting can be an indicator of developer struggles with using Transform Code. Second, we conduct a qualitative analysis of unsatisfactory requests, identifying five key categories of information often missing from developer prompts. Finally, based on these findings, we propose and evaluate a tool, AutoPrompter, for automatically improving prompts by inferring missing information from the surrounding code context, leading to a 27% improvement in edit correctness on our test set. 

---
# BLADE: Benchmark suite for LLM-driven Automated Design and Evolution of iterative optimisation heuristics 

**Authors**: Niki van Stein, Anna V. Kononova, Haoran Yin, Thomas Bäck  

**Link**: [PDF](https://arxiv.org/pdf/2504.20183)  

**Abstract**: The application of Large Language Models (LLMs) for Automated Algorithm Discovery (AAD), particularly for optimisation heuristics, is an emerging field of research. This emergence necessitates robust, standardised benchmarking practices to rigorously evaluate the capabilities and limitations of LLM-driven AAD methods and the resulting generated algorithms, especially given the opacity of their design process and known issues with existing benchmarks. To address this need, we introduce BLADE (Benchmark suite for LLM-driven Automated Design and Evolution), a modular and extensible framework specifically designed for benchmarking LLM-driven AAD methods in a continuous black-box optimisation context. BLADE integrates collections of benchmark problems (including MA-BBOB and SBOX-COST among others) with instance generators and textual descriptions aimed at capability-focused testing, such as generalisation, specialisation and information exploitation. It offers flexible experimental setup options, standardised logging for reproducibility and fair comparison, incorporates methods for analysing the AAD process (e.g., Code Evolution Graphs and various visualisation approaches) and facilitates comparison against human-designed baselines through integration with established tools like IOHanalyser and IOHexplainer. BLADE provides an `out-of-the-box' solution to systematically evaluate LLM-driven AAD approaches. The framework is demonstrated through two distinct use cases exploring mutation prompt strategies and function specialisation. 

---
# Towards Large Language Models for Lunar Mission Planning and In Situ Resource Utilization 

**Authors**: Michael Pekala, Gregory Canal, Samuel Barham, Milena B. Graziano, Morgan Trexler, Leslie Hamilton, Elizabeth Reilly, Christopher D. Stiles  

**Link**: [PDF](https://arxiv.org/pdf/2504.20125)  

**Abstract**: A key factor for lunar mission planning is the ability to assess the local availability of raw materials. However, many potentially relevant measurements are scattered across a variety of scientific publications. In this paper we consider the viability of obtaining lunar composition data by leveraging LLMs to rapidly process a corpus of scientific publications. While leveraging LLMs to obtain knowledge from scientific documents is not new, this particular application presents interesting challenges due to the heterogeneity of lunar samples and the nuances involved in their characterization. Accuracy and uncertainty quantification are particularly crucial since many materials properties can be sensitive to small variations in composition. Our findings indicate that off-the-shelf LLMs are generally effective at extracting data from tables commonly found in these documents. However, there remains opportunity to further refine the data we extract in this initial approach; in particular, to capture fine-grained mineralogy information and to improve performance on more subtle/complex pieces of information. 

---
# AutoP2C: An LLM-Based Agent Framework for Code Repository Generation from Multimodal Content in Academic Papers 

**Authors**: Zijie Lin, Yiqing Shen, Qilin Cai, He Sun, Jinrui Zhou, Mingjun Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2504.20115)  

**Abstract**: Machine Learning (ML) research is spread through academic papers featuring rich multimodal content, including text, diagrams, and tabular results. However, translating these multimodal elements into executable code remains a challenging and time-consuming process that requires substantial ML expertise. We introduce ``Paper-to-Code'' (P2C), a novel task that transforms the multimodal content of scientific publications into fully executable code repositories, which extends beyond the existing formulation of code generation that merely converts textual descriptions into isolated code snippets. To automate the P2C process, we propose AutoP2C, a multi-agent framework based on large language models that processes both textual and visual content from research papers to generate complete code repositories. Specifically, AutoP2C contains four stages: (1) repository blueprint extraction from established codebases, (2) multimodal content parsing that integrates information from text, equations, and figures, (3) hierarchical task decomposition for structured code generation, and (4) iterative feedback-driven debugging to ensure functionality and performance. Evaluation on a benchmark of eight research papers demonstrates the effectiveness of AutoP2C, which can successfully generate executable code repositories for all eight papers, while OpenAI-o1 or DeepSeek-R1 can only produce runnable code for one paper. The code is available at this https URL. 

---
