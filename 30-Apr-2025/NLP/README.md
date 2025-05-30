# SetKE: Knowledge Editing for Knowledge Elements Overlap 

**Authors**: Yifan Wei, Xiaoyan Yu, Ran Song, Hao Peng, Angsheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.20972)  

**Abstract**: Large Language Models (LLMs) excel in tasks such as retrieval and question answering but require updates to incorporate new knowledge and reduce inaccuracies and hallucinations. Traditional updating methods, like fine-tuning and incremental learning, face challenges such as overfitting and high computational costs. Knowledge Editing (KE) provides a promising alternative but often overlooks the Knowledge Element Overlap (KEO) phenomenon, where multiple triplets share common elements, leading to editing conflicts. We identify the prevalence of KEO in existing KE datasets and show its significant impact on current KE methods, causing performance degradation in handling such triplets. To address this, we propose a new formulation, Knowledge Set Editing (KSE), and introduce SetKE, a method that edits sets of triplets simultaneously. Experimental results demonstrate that SetKE outperforms existing methods in KEO scenarios on mainstream LLMs. Additionally, we introduce EditSet, a dataset containing KEO triplets, providing a comprehensive benchmark. 

---
# OSVBench: Benchmarking LLMs on Specification Generation Tasks for Operating System Verification 

**Authors**: Shangyu Li, Juyong Jiang, Tiancheng Zhao, Jiasi Shen  

**Link**: [PDF](https://arxiv.org/pdf/2504.20964)  

**Abstract**: We introduce OSVBench, a new benchmark for evaluating Large Language Models (LLMs) in generating complete specification code pertaining to operating system kernel verification tasks. The benchmark first defines the specification generation problem into a program synthesis problem within a confined scope of syntax and semantics by providing LLMs with the programming model. The LLMs are required to understand the provided verification assumption and the potential syntax and semantics space to search for, then generate the complete specification for the potentially buggy operating system code implementation under the guidance of the high-level functional description of the operating system. This benchmark is built upon a real-world operating system kernel, Hyperkernel, and consists of 245 complex specification generation tasks in total, each is a long context task of about 20k-30k tokens. Our comprehensive evaluation of 12 LLMs exhibits the limited performance of the current LLMs on the specification generation tasks for operating system verification. Significant disparities in their performance on the benchmark highlight differences in their ability to handle long-context code generation tasks. The evaluation toolkit and benchmark are available at this https URL. 

---
# Information Gravity: A Field-Theoretic Model for Token Selection in Large Language Models 

**Authors**: Maryna Vyshnyvetska  

**Link**: [PDF](https://arxiv.org/pdf/2504.20951)  

**Abstract**: We propose a theoretical model called "information gravity" to describe the text generation process in large language models (LLMs). The model uses physical apparatus from field theory and spacetime geometry to formalize the interaction between user queries and the probability distribution of generated tokens. A query is viewed as an object with "information mass" that curves the semantic space of the model, creating gravitational potential wells that "attract" tokens during generation. This model offers a mechanism to explain several observed phenomena in LLM behavior, including hallucinations (emerging from low-density semantic voids), sensitivity to query formulation (due to semantic field curvature changes), and the influence of sampling temperature on output diversity. 

---
# Trace-of-Thought: Enhanced Arithmetic Problem Solving via Reasoning Distillation From Large to Small Language Models 

**Authors**: Tyler McDonald, Ali Emami  

**Link**: [PDF](https://arxiv.org/pdf/2504.20946)  

**Abstract**: As Large Language Models (LLMs) continue to be leveraged for daily tasks, prompt engineering remains an active field of contribution within computational linguistics, particularly in domains requiring specialized knowledge such as arithmetic reasoning. While these LLMs are optimized for a variety of tasks, their exhaustive employment may become computationally or financially cumbersome for small teams. Additionally, complete reliance on proprietary, closed-source models often limits customization and adaptability, posing significant challenges in research and application scalability. Instead, by leveraging open-source models at or below 7 billion parameters, we can optimize our resource usage while still observing remarkable gains over standard prompting approaches. To cultivate this notion, we introduce Trace-of-Thought Prompting, a simple, zero-shot prompt engineering method that instructs LLMs to create observable subproblems using critical problem-solving, specifically designed to enhance arithmetic reasoning capabilities. When applied to open-source models in tandem with GPT-4, we observe that Trace-of-Thought not only allows novel insight into the problem-solving process but also introduces performance gains as large as 125% on language models at or below 7 billion parameters. This approach underscores the potential of open-source initiatives in democratizing AI research and improving the accessibility of high-quality computational linguistics applications. 

---
# DYNAMAX: Dynamic computing for Transformers and Mamba based architectures 

**Authors**: Miguel Nogales, Matteo Gambella, Manuel Roveri  

**Link**: [PDF](https://arxiv.org/pdf/2504.20922)  

**Abstract**: Early exits (EEs) offer a promising approach to reducing computational costs and latency by dynamically terminating inference once a satisfactory prediction confidence on a data sample is achieved. Although many works integrate EEs into encoder-only Transformers, their application to decoder-only architectures and, more importantly, Mamba models, a novel family of state-space architectures in the LLM realm, remains insufficiently explored. This work introduces DYNAMAX, the first framework to exploit the unique properties of Mamba architectures for early exit mechanisms. We not only integrate EEs into Mamba but also repurpose Mamba as an efficient EE classifier for both Mamba-based and transformer-based LLMs, showcasing its versatility. Our experiments employ the Mistral 7B transformer compared to the Codestral 7B Mamba model, using data sets such as TruthfulQA, CoQA, and TriviaQA to evaluate computational savings, accuracy, and consistency. The results highlight the adaptability of Mamba as a powerful EE classifier and its efficiency in balancing computational cost and performance quality across NLP tasks. By leveraging Mamba's inherent design for dynamic processing, we open pathways for scalable and efficient inference in embedded applications and resource-constrained environments. This study underscores the transformative potential of Mamba in redefining dynamic computing paradigms for LLMs. 

---
# JaccDiv: A Metric and Benchmark for Quantifying Diversity of Generated Marketing Text in the Music Industry 

**Authors**: Anum Afzal, Alexandre Mercier, Florian Matthes  

**Link**: [PDF](https://arxiv.org/pdf/2504.20849)  

**Abstract**: Online platforms are increasingly interested in using Data-to-Text technologies to generate content and help their users. Unfortunately, traditional generative methods often fall into repetitive patterns, resulting in monotonous galleries of texts after only a few iterations. In this paper, we investigate LLM-based data-to-text approaches to automatically generate marketing texts that are of sufficient quality and diverse enough for broad adoption. We leverage Language Models such as T5, GPT-3.5, GPT-4, and LLaMa2 in conjunction with fine-tuning, few-shot, and zero-shot approaches to set a baseline for diverse marketing texts. We also introduce a metric JaccDiv to evaluate the diversity of a set of texts. This research extends its relevance beyond the music industry, proving beneficial in various fields where repetitive automated content generation is prevalent. 

---
# Universal language model with the intervention of quantum theory 

**Authors**: D.-F. Qin  

**Link**: [PDF](https://arxiv.org/pdf/2504.20839)  

**Abstract**: This paper examines language modeling based on the theory of quantum mechanics. It focuses on the introduction of quantum mechanics into the symbol-meaning pairs of language in order to build a representation model of natural language. At the same time, it is realized that word embedding, which is widely used as a basic technique for statistical language modeling, can be explained and improved by the mathematical framework of quantum mechanics. On this basis, this paper continues to try to use quantum statistics and other related theories to study the mathematical representation, natural evolution and statistical properties of natural language. It is also assumed that the source of such quantum properties is the physicality of information. The feasibility of using quantum theory to model natural language is pointed out through the construction of a experimental code. The paper discusses, in terms of applications, the possible help of the theory in constructing generative models that are popular nowadays. A preliminary discussion of future applications of the theory to quantum computers is also presented. 

---
# Turing Machine Evaluation for Large Language Model 

**Authors**: Haitao Wu, Zongbo Han, Huaxi Huang, Changqing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.20771)  

**Abstract**: With the rapid development and widespread application of Large Language Models (LLMs), rigorous evaluation has become particularly crucial. This research adopts a novel perspective, focusing on evaluating the core computational reasoning ability of LLMs, defined as the capacity of model to accurately understand rules, and execute logically computing operations. This capability assesses the reliability of LLMs as precise executors, and is critical to advanced tasks such as complex code generation and multi-step problem-solving. We propose an evaluation framework based on Universal Turing Machine (UTM) simulation. This framework requires LLMs to strictly follow instructions and track dynamic states, such as tape content and read/write head position, during multi-step computations. To enable standardized evaluation, we developed TMBench, a benchmark for systematically studying the computational reasoning capabilities of LLMs. TMBench provides several key advantages, including knowledge-agnostic evaluation, adjustable difficulty, foundational coverage through Turing machine encoding, and unlimited capacity for instance generation, ensuring scalability as models continue to evolve. We find that model performance on TMBench correlates strongly with performance on other recognized reasoning benchmarks (Pearson correlation coefficient is 0.73), clearly demonstrating that computational reasoning is a significant dimension for measuring the deep capabilities of LLMs. Code and data are available at this https URL. 

---
# Chain-of-Defensive-Thought: Structured Reasoning Elicits Robustness in Large Language Models against Reference Corruption 

**Authors**: Wenxiao Wang, Parsa Hosseini, Soheil Feizi  

**Link**: [PDF](https://arxiv.org/pdf/2504.20769)  

**Abstract**: Chain-of-thought prompting has demonstrated great success in facilitating the reasoning abilities of large language models. In this work, we explore how these enhanced reasoning abilities can be exploited to improve the robustness of large language models in tasks that are not necessarily reasoning-focused. In particular, we show how a wide range of large language models exhibit significantly improved robustness against reference corruption using a simple method called chain-of-defensive-thought, where only a few exemplars with structured and defensive reasoning are provided as demonstrations. Empirically, the improvements can be astounding, especially given the simplicity and applicability of the method. For example, in the Natural Questions task, the accuracy of GPT-4o degrades from 60% to as low as 3% with standard prompting when 1 out of 10 references provided is corrupted with prompt injection attacks. In contrast, GPT-4o using chain-of-defensive-thought prompting maintains an accuracy of 50%. 

---
# Grokking in the Wild: Data Augmentation for Real-World Multi-Hop Reasoning with Transformers 

**Authors**: Roman Abramov, Felix Steinbauer, Gjergji Kasneci  

**Link**: [PDF](https://arxiv.org/pdf/2504.20752)  

**Abstract**: Transformers have achieved great success in numerous NLP tasks but continue to exhibit notable gaps in multi-step factual reasoning, especially when real-world knowledge is sparse. Recent advances in grokking have demonstrated that neural networks can transition from memorizing to perfectly generalizing once they detect underlying logical patterns - yet these studies have primarily used small, synthetic tasks. In this paper, for the first time, we extend grokking to real-world factual data and address the challenge of dataset sparsity by augmenting existing knowledge graphs with carefully designed synthetic data to raise the ratio $\phi_r$ of inferred facts to atomic facts above the threshold required for grokking. Surprisingly, we find that even factually incorrect synthetic data can strengthen emergent reasoning circuits rather than degrade accuracy, as it forces the model to rely on relational structure rather than memorization. When evaluated on multi-hop reasoning benchmarks, our approach achieves up to 95-100% accuracy on 2WikiMultiHopQA - substantially improving over strong baselines and matching or exceeding current state-of-the-art results. We further provide an in-depth analysis of how increasing $\phi_r$ drives the formation of generalizing circuits inside Transformers. Our findings suggest that grokking-based data augmentation can unlock implicit multi-hop reasoning capabilities, opening the door to more robust and interpretable factual reasoning in large-scale language models. 

---
# UniversalRAG: Retrieval-Augmented Generation over Multiple Corpora with Diverse Modalities and Granularities 

**Authors**: Woongyeong Yeo, Kangsan Kim, Soyeong Jeong, Jinheon Baek, Sung Ju Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2504.20734)  

**Abstract**: Retrieval-Augmented Generation (RAG) has shown substantial promise in improving factual accuracy by grounding model responses with external knowledge relevant to queries. However, most existing RAG approaches are limited to a text-only corpus, and while recent efforts have extended RAG to other modalities such as images and videos, they typically operate over a single modality-specific corpus. In contrast, real-world queries vary widely in the type of knowledge they require, which a single type of knowledge source cannot address. To address this, we introduce UniversalRAG, a novel RAG framework designed to retrieve and integrate knowledge from heterogeneous sources with diverse modalities and granularities. Specifically, motivated by the observation that forcing all modalities into a unified representation space derived from a single combined corpus causes a modality gap, where the retrieval tends to favor items from the same modality as the query, we propose a modality-aware routing mechanism that dynamically identifies the most appropriate modality-specific corpus and performs targeted retrieval within it. Also, beyond modality, we organize each modality into multiple granularity levels, enabling fine-tuned retrieval tailored to the complexity and scope of the query. We validate UniversalRAG on 8 benchmarks spanning multiple modalities, showing its superiority over modality-specific and unified baselines. 

---
# Beyond the Last Answer: Your Reasoning Trace Uncovers More than You Think 

**Authors**: Hasan Abed Al Kader Hammoud, Hani Itani, Bernard Ghanem  

**Link**: [PDF](https://arxiv.org/pdf/2504.20708)  

**Abstract**: Large Language Models (LLMs) leverage step-by-step reasoning to solve complex problems. Standard evaluation practice involves generating a complete reasoning trace and assessing the correctness of the final answer presented at its conclusion. In this paper, we challenge the reliance on the final answer by posing the following two questions: Does the final answer reliably represent the model's optimal conclusion? Can alternative reasoning paths yield different results? To answer these questions, we analyze intermediate reasoning steps, termed subthoughts, and propose a method based on our findings. Our approach involves segmenting a reasoning trace into sequential subthoughts based on linguistic cues. We start by prompting the model to generate continuations from the end-point of each intermediate subthought. We extract a potential answer from every completed continuation originating from different subthoughts. We find that aggregating these answers by selecting the most frequent one (the mode) often yields significantly higher accuracy compared to relying solely on the answer derived from the original complete trace. Analyzing the consistency among the answers derived from different subthoughts reveals characteristics that correlate with the model's confidence and correctness, suggesting potential for identifying less reliable answers. Our experiments across various LLMs and challenging mathematical reasoning datasets (AIME2024 and AIME2025) show consistent accuracy improvements, with gains reaching up to 13\% and 10\% respectively. Implementation is available at: this https URL. 

---
# BrightCookies at SemEval-2025 Task 9: Exploring Data Augmentation for Food Hazard Classification 

**Authors**: Foteini Papadopoulou, Osman Mutlu, Neris Özen, Bas H.M. van der Velden, Iris Hendrickx, Ali Hürriyetoğlu  

**Link**: [PDF](https://arxiv.org/pdf/2504.20703)  

**Abstract**: This paper presents our system developed for the SemEval-2025 Task 9: The Food Hazard Detection Challenge. The shared task's objective is to evaluate explainable classification systems for classifying hazards and products in two levels of granularity from food recall incident reports. In this work, we propose text augmentation techniques as a way to improve poor performance on minority classes and compare their effect for each category on various transformer and machine learning models. We explore three word-level data augmentation techniques, namely synonym replacement, random word swapping, and contextual word insertion. The results show that transformer models tend to have a better overall performance. None of the three augmentation techniques consistently improved overall performance for classifying hazards and products. We observed a statistically significant improvement (P < 0.05) in the fine-grained categories when using the BERT model to compare the baseline with each augmented model. Compared to the baseline, the contextual words insertion augmentation improved the accuracy of predictions for the minority hazard classes by 6%. This suggests that targeted augmentation of minority classes can improve the performance of transformer models. 

---
# Can LLMs Detect Intrinsic Hallucinations in Paraphrasing and Machine Translation? 

**Authors**: Evangelia Gogoulou, Shorouq Zahra, Liane Guillou, Luise Dürlich, Joakim Nivre  

**Link**: [PDF](https://arxiv.org/pdf/2504.20699)  

**Abstract**: A frequently observed problem with LLMs is their tendency to generate output that is nonsensical, illogical, or factually incorrect, often referred to broadly as hallucination. Building on the recently proposed HalluciGen task for hallucination detection and generation, we evaluate a suite of open-access LLMs on their ability to detect intrinsic hallucinations in two conditional generation tasks: translation and paraphrasing. We study how model performance varies across tasks and language and we investigate the impact of model size, instruction tuning, and prompt choice. We find that performance varies across models but is consistent across prompts. Finally, we find that NLI models perform comparably well, suggesting that LLM-based detectors are not the only viable option for this specific task. 

---
# Are Information Retrieval Approaches Good at Harmonising Longitudinal Survey Questions in Social Science? 

**Authors**: Wing Yan Li, Zeqiang Wang, Jon Johnson, Suparna De  

**Link**: [PDF](https://arxiv.org/pdf/2504.20679)  

**Abstract**: Automated detection of semantically equivalent questions in longitudinal social science surveys is crucial for long-term studies informing empirical research in the social, economic, and health sciences. Retrieving equivalent questions faces dual challenges: inconsistent representation of theoretical constructs (i.e. concept/sub-concept) across studies as well as between question and response options, and the evolution of vocabulary and structure in longitudinal text. To address these challenges, our multi-disciplinary collaboration of computer scientists and survey specialists presents a new information retrieval (IR) task of identifying concept (e.g. Housing, Job, etc.) equivalence across question and response options to harmonise longitudinal population studies. This paper investigates multiple unsupervised approaches on a survey dataset spanning 1946-2020, including probabilistic models, linear probing of language models, and pre-trained neural networks specialised for IR. We show that IR-specialised neural models achieve the highest overall performance with other approaches performing comparably. Additionally, the re-ranking of the probabilistic model's results with neural models only introduces modest improvements of 0.07 at most in F1-score. Qualitative post-hoc evaluation by survey specialists shows that models generally have a low sensitivity to questions with high lexical overlap, particularly in cases where sub-concepts are mismatched. Altogether, our analysis serves to further research on harmonising longitudinal studies in social science. 

---
# Non-native Children's Automatic Speech Assessment Challenge (NOCASA) 

**Authors**: Yaroslav Getman, Tamás Grósz, Mikko Kurimo, Giampiero Salvi  

**Link**: [PDF](https://arxiv.org/pdf/2504.20678)  

**Abstract**: This paper presents the "Non-native Children's Automatic Speech Assessment" (NOCASA) - a data competition part of the IEEE MLSP 2025 conference. NOCASA challenges participants to develop new systems that can assess single-word pronunciations of young second language (L2) learners as part of a gamified pronunciation training app. To achieve this, several issues must be addressed, most notably the limited nature of available training data and the highly unbalanced distribution among the pronunciation level categories. To expedite the development, we provide a pseudo-anonymized training data (TeflonNorL2), containing 10,334 recordings from 44 speakers attempting to pronounce 205 distinct Norwegian words, human-rated on a 1 to 5 scale (number of stars that should be given in the game). In addition to the data, two already trained systems are released as official baselines: an SVM classifier trained on the ComParE_16 acoustic feature set and a multi-task wav2vec 2.0 model. The latter achieves the best performance on the challenge test set, with an unweighted average recall (UAR) of 36.37%. 

---
# A Generative-AI-Driven Claim Retrieval System Capable of Detecting and Retrieving Claims from Social Media Platforms in Multiple Languages 

**Authors**: Ivan Vykopal, Martin Hyben, Robert Moro, Michal Gregor, Jakub Simko  

**Link**: [PDF](https://arxiv.org/pdf/2504.20668)  

**Abstract**: Online disinformation poses a global challenge, placing significant demands on fact-checkers who must verify claims efficiently to prevent the spread of false information. A major issue in this process is the redundant verification of already fact-checked claims, which increases workload and delays responses to newly emerging claims. This research introduces an approach that retrieves previously fact-checked claims, evaluates their relevance to a given input, and provides supplementary information to support fact-checkers. Our method employs large language models (LLMs) to filter irrelevant fact-checks and generate concise summaries and explanations, enabling fact-checkers to faster assess whether a claim has been verified before. In addition, we evaluate our approach through both automatic and human assessments, where humans interact with the developed tool to review its effectiveness. Our results demonstrate that LLMs are able to filter out many irrelevant fact-checks and, therefore, reduce effort and streamline the fact-checking process. 

---
# Cooking Up Creativity: A Cognitively-Inspired Approach for Enhancing LLM Creativity through Structured Representations 

**Authors**: Moran Mizrahi, Chen Shani, Gabriel Stanovsky, Dan Jurafsky, Dafna Shahaf  

**Link**: [PDF](https://arxiv.org/pdf/2504.20643)  

**Abstract**: Large Language Models (LLMs) excel at countless tasks, yet struggle with creativity. In this paper, we introduce a novel approach that couples LLMs with structured representations and cognitively inspired manipulations to generate more creative and diverse ideas. Our notion of creativity goes beyond superficial token-level variations; rather, we explicitly recombine structured representations of existing ideas, allowing our algorithm to effectively explore the more abstract landscape of ideas. We demonstrate our approach in the culinary domain with DishCOVER, a model that generates creative recipes. Experiments comparing our model's results to those of GPT-4o show greater diversity. Domain expert evaluations reveal that our outputs, which are mostly coherent and feasible culinary creations, significantly surpass GPT-4o in terms of novelty, thus outperforming it in creative generation. We hope our work inspires further research into structured creativity in AI. 

---
# WenyanGPT: A Large Language Model for Classical Chinese Tasks 

**Authors**: Xinyu Yao, Mengdi Wang, Bo Chen, Xiaobing Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2504.20609)  

**Abstract**: Classical Chinese, as the core carrier of Chinese culture, plays a crucial role in the inheritance and study of ancient literature. However, existing natural language processing models primarily optimize for Modern Chinese, resulting in inadequate performance on Classical Chinese. This paper presents a comprehensive solution for Classical Chinese language processing. By continuing pre-training and instruction fine-tuning on the LLaMA3-8B-Chinese model, we construct a large language model, WenyanGPT, which is specifically designed for Classical Chinese tasks. Additionally, we develop an evaluation benchmark dataset, WenyanBENCH. Experimental results on WenyanBENCH demonstrate that WenyanGPT significantly outperforms current advanced LLMs in various Classical Chinese tasks. We make the model's training data, instruction fine-tuning data\footnote, and evaluation benchmark dataset publicly available to promote further research and development in the field of Classical Chinese processing. 

---
# TF1-EN-3M: Three Million Synthetic Moral Fables for Training Small, Open Language Models 

**Authors**: Mihai Nadas, Laura Diosan, Andrei Piscoran, Andreea Tomescu  

**Link**: [PDF](https://arxiv.org/pdf/2504.20605)  

**Abstract**: Moral stories are a time-tested vehicle for transmitting values, yet modern NLP lacks a large, structured corpus that couples coherent narratives with explicit ethical lessons. We close this gap with TF1-EN-3M, the first open dataset of three million English-language fables generated exclusively by instruction-tuned models no larger than 8B parameters. Each story follows a six-slot scaffold (character -> trait -> setting -> conflict -> resolution -> moral), produced through a combinatorial prompt engine that guarantees genre fidelity while covering a broad thematic space.
A hybrid evaluation pipeline blends (i) a GPT-based critic that scores grammar, creativity, moral clarity, and template adherence with (ii) reference-free diversity and readability metrics. Among ten open-weight candidates, an 8B-parameter Llama-3 variant delivers the best quality-speed trade-off, producing high-scoring fables on a single consumer GPU (<24 GB VRAM) at approximately 13.5 cents per 1,000 fables.
We release the dataset, generation code, evaluation scripts, and full metadata under a permissive license, enabling exact reproducibility and cost benchmarking. TF1-EN-3M opens avenues for research in instruction following, narrative intelligence, value alignment, and child-friendly educational AI, demonstrating that large-scale moral storytelling no longer requires proprietary giant models. 

---
# ClonEval: An Open Voice Cloning Benchmark 

**Authors**: Iwona Christop, Tomasz Kuczyński, Marek Kubis  

**Link**: [PDF](https://arxiv.org/pdf/2504.20581)  

**Abstract**: We present a novel benchmark for voice cloning text-to-speech models. The benchmark consists of an evaluation protocol, an open-source library for assessing the performance of voice cloning models, and an accompanying leaderboard. The paper discusses design considerations and presents a detailed description of the evaluation procedure. The usage of the software library is explained, along with the organization of results on the leaderboard. 

---
# BrAIcht, a theatrical agent that speaks like Bertolt Brecht's characters 

**Authors**: Baz Roland, Kristina Malyseva, Anna Pappa, Tristan Cazenave  

**Link**: [PDF](https://arxiv.org/pdf/2504.20552)  

**Abstract**: This project introduces BrAIcht, an AI conversational agent that creates dialogues in the distinctive style of the famous German playwright Bertolt Brecht. BrAIcht is fine-tuned using German LeoLM, a large language model with 7 billion parameters and a modified version of the base Llama2 suitable for German language tasks. For fine-tuning, 29 plays of Bertolt Brecht and 907 of other German plays that are stylistically similar to Bertolt Brecht are used to form a more di-erse dataset. Due to the limited memory capacity, a parameterefficient fine-tuning technique called QLoRA is implemented to train the large language model. The results, based on BLEU score and perplexity, show very promising performance of BrAIcht in generating dialogues in the style of Bertolt Brecht. 

---
# Revisiting the MIMIC-IV Benchmark: Experiments Using Language Models for Electronic Health Records 

**Authors**: Jesus Lovon, Thouria Ben-Haddi, Jules Di Scala, Jose G. Moreno, Lynda Tamine  

**Link**: [PDF](https://arxiv.org/pdf/2504.20547)  

**Abstract**: The lack of standardized evaluation benchmarks in the medical domain for text inputs can be a barrier to widely adopting and leveraging the potential of natural language models for health-related downstream tasks. This paper revisited an openly available MIMIC-IV benchmark for electronic health records (EHRs) to address this issue. First, we integrate the MIMIC-IV data within the Hugging Face datasets library to allow an easy share and use of this collection. Second, we investigate the application of templates to convert EHR tabular data to text. Experiments using fine-tuned and zero-shot LLMs on the mortality of patients task show that fine-tuned text-based models are competitive against robust tabular classifiers. In contrast, zero-shot LLMs struggle to leverage EHR representations. This study underlines the potential of text-based approaches in the medical field and highlights areas for further improvement. 

---
# UniDetox: Universal Detoxification of Large Language Models via Dataset Distillation 

**Authors**: Huimin Lu, Masaru Isonuma, Junichiro Mori, Ichiro Sakata  

**Link**: [PDF](https://arxiv.org/pdf/2504.20500)  

**Abstract**: We present UniDetox, a universally applicable method designed to mitigate toxicity across various large language models (LLMs). Previous detoxification methods are typically model-specific, addressing only individual models or model families, and require careful hyperparameter tuning due to the trade-off between detoxification efficacy and language modeling performance. In contrast, UniDetox provides a detoxification technique that can be universally applied to a wide range of LLMs without the need for separate model-specific tuning. Specifically, we propose a novel and efficient dataset distillation technique for detoxification using contrastive decoding. This approach distills detoxifying representations in the form of synthetic text data, enabling universal detoxification of any LLM through fine-tuning with the distilled text. Our experiments demonstrate that the detoxifying text distilled from GPT-2 can effectively detoxify larger models, including OPT, Falcon, and LLaMA-2. Furthermore, UniDetox eliminates the need for separate hyperparameter tuning for each model, as a single hyperparameter configuration can be seamlessly applied across different models. Additionally, analysis of the detoxifying text reveals a reduction in politically biased content, providing insights into the attributes necessary for effective detoxification of LLMs. 

---
# Enhancing LLM Language Adaption through Cross-lingual In-Context Pre-training 

**Authors**: Linjuan Wu, Haoran Wei, Huan Lin, Tianhao Li, Baosong Yang, Weiming Lu  

**Link**: [PDF](https://arxiv.org/pdf/2504.20484)  

**Abstract**: Large language models (LLMs) exhibit remarkable multilingual capabilities despite English-dominated pre-training, attributed to cross-lingual mechanisms during pre-training. Existing methods for enhancing cross-lingual transfer remain constrained by parallel resources, suffering from limited linguistic and domain coverage. We propose Cross-lingual In-context Pre-training (CrossIC-PT), a simple and scalable approach that enhances cross-lingual transfer by leveraging semantically related bilingual texts via simple next-word prediction. We construct CrossIC-PT samples by interleaving semantic-related bilingual Wikipedia documents into a single context window. To access window size constraints, we implement a systematic segmentation policy to split long bilingual document pairs into chunks while adjusting the sliding window mechanism to preserve contextual coherence. We further extend data availability through a semantic retrieval framework to construct CrossIC-PT samples from web-crawled corpus. Experimental results demonstrate that CrossIC-PT improves multilingual performance on three models (Llama-3.1-8B, Qwen2.5-7B, and Qwen2.5-1.5B) across six target languages, yielding performance gains of 3.79%, 3.99%, and 1.95%, respectively, with additional improvements after data augmentation. 

---
# Fane at SemEval-2025 Task 10: Zero-Shot Entity Framing with Large Language Models 

**Authors**: Enfa Fane, Mihai Surdeanu, Eduardo Blanco, Steven R. Corman  

**Link**: [PDF](https://arxiv.org/pdf/2504.20469)  

**Abstract**: Understanding how news narratives frame entities is crucial for studying media's impact on societal perceptions of events. In this paper, we evaluate the zero-shot capabilities of large language models (LLMs) in classifying framing roles. Through systematic experimentation, we assess the effects of input context, prompting strategies, and task decomposition. Our findings show that a hierarchical approach of first identifying broad roles and then fine-grained roles, outperforms single-step classification. We also demonstrate that optimal input contexts and prompts vary across task levels, highlighting the need for subtask-specific strategies. We achieve a Main Role Accuracy of 89.4% and an Exact Match Ratio of 34.5%, demonstrating the effectiveness of our approach. Our findings emphasize the importance of tailored prompt design and input context optimization for improving LLM performance in entity framing. 

---
# Team ACK at SemEval-2025 Task 2: Beyond Word-for-Word Machine Translation for English-Korean Pairs 

**Authors**: Daniel Lee, Harsh Sharma, Jieun Han, Sunny Jeong, Alice Oh, Vered Shwartz  

**Link**: [PDF](https://arxiv.org/pdf/2504.20451)  

**Abstract**: Translating knowledge-intensive and entity-rich text between English and Korean requires transcreation to preserve language-specific and cultural nuances beyond literal, phonetic or word-for-word conversion. We evaluate 13 models (LLMs and MT models) using automatic metrics and human assessment by bilingual annotators. Our findings show LLMs outperform traditional MT systems but struggle with entity translation requiring cultural adaptation. By constructing an error taxonomy, we identify incorrect responses and entity name errors as key issues, with performance varying by entity type and popularity level. This work exposes gaps in automatic evaluation metrics and hope to enable future work in completing culturally-nuanced machine translation. 

---
# On Psychology of AI -- Does Primacy Effect Affect ChatGPT and Other LLMs? 

**Authors**: Mika Hämäläinen  

**Link**: [PDF](https://arxiv.org/pdf/2504.20444)  

**Abstract**: We study the primacy effect in three commercial LLMs: ChatGPT, Gemini and Claude. We do this by repurposing the famous experiment Asch (1946) conducted using human subjects. The experiment is simple, given two candidates with equal descriptions which one is preferred if one description has positive adjectives first before negative ones and another description has negative adjectives followed by positive ones. We test this in two experiments. In one experiment, LLMs are given both candidates simultaneously in the same prompt, and in another experiment, LLMs are given both candidates separately. We test all the models with 200 candidate pairs. We found that, in the first experiment, ChatGPT preferred the candidate with positive adjectives listed first, while Gemini preferred both equally often. Claude refused to make a choice. In the second experiment, ChatGPT and Claude were most likely to rank both candidates equally. In the case where they did not give an equal rating, both showed a clear preference to a candidate that had negative adjectives listed first. Gemini was most likely to prefer a candidate with negative adjectives listed first. 

---
# DMDTEval: An Evaluation and Analysis of LLMs on Disambiguation in Multi-domain Translation 

**Authors**: Zhibo Man, Yuanmeng Chen, Yujie Zhang, Yufeng Chen, Jinan Xu  

**Link**: [PDF](https://arxiv.org/pdf/2504.20371)  

**Abstract**: Currently, Large Language Models (LLMs) have achieved remarkable results in machine translation. However, their performance in multi-domain translation (MDT) is less satisfactory; the meanings of words can vary across different domains, highlighting the significant ambiguity inherent in MDT. Therefore, evaluating the disambiguation ability of LLMs in MDT remains an open problem. To this end, we present an evaluation and analysis of LLMs on disambiguation in multi-domain translation (DMDTEval), our systematic evaluation framework consisting of three critical aspects: (1) we construct a translation test set with multi-domain ambiguous word annotation, (2) we curate a diverse set of disambiguation prompting templates, and (3) we design precise disambiguation metrics, and study the efficacy of various prompting strategies on multiple state-of-the-art LLMs. Our extensive experiments reveal a number of crucial findings that we believe will pave the way and also facilitate further research in the critical area of improving the disambiguation of LLMs. 

---
# What Causes Knowledge Loss in Multilingual Language Models? 

**Authors**: Maria Khelli, Samuel Cahyawijaya, Ayu Purwarianti, Genta Indra Winata  

**Link**: [PDF](https://arxiv.org/pdf/2504.20356)  

**Abstract**: Cross-lingual transfer in natural language processing (NLP) models enhances multilingual performance by leveraging shared linguistic knowledge. However, traditional methods that process all data simultaneously often fail to mimic real-world scenarios, leading to challenges like catastrophic forgetting, where fine-tuning on new tasks degrades performance on previously learned ones. Our study explores this issue in multilingual contexts, focusing on linguistic differences affecting representational learning rather than just model parameters. We experiment with 52 languages using LoRA adapters of varying ranks to evaluate non-shared, partially shared, and fully shared parameters. Our aim is to see if parameter sharing through adapters can mitigate forgetting while preserving prior knowledge. We find that languages using non-Latin scripts are more susceptible to catastrophic forgetting, whereas those written in Latin script facilitate more effective cross-lingual transfer. 

---
# Local Prompt Optimization 

**Authors**: Yash Jain, Vishal Chowdhary  

**Link**: [PDF](https://arxiv.org/pdf/2504.20355)  

**Abstract**: In recent years, the use of prompts to guide the output of Large Language Models have increased dramatically. However, even the best of experts struggle to choose the correct words to stitch up a prompt for the desired task. To solve this, LLM driven prompt optimization emerged as an important problem. Existing prompt optimization methods optimize a prompt globally, where in all the prompt tokens have to be optimized over a large vocabulary while solving a complex task. The large optimization space (tokens) leads to insufficient guidance for a better prompt. In this work, we introduce Local Prompt Optimization (LPO) that integrates with any general automatic prompt engineering method. We identify the optimization tokens in a prompt and nudge the LLM to focus only on those tokens in its optimization step. We observe remarkable performance improvements on Math Reasoning (GSM8k and MultiArith) and BIG-bench Hard benchmarks across various automatic prompt engineering methods. Further, we show that LPO converges to the optimal prompt faster than global methods. 

---
# Labeling Case Similarity based on Co-Citation of Legal Articles in Judgment Documents with Empirical Dispute-Based Evaluation 

**Authors**: Chao-Lin Liu, Po-Hsien Wu, Yi-Ting Yu  

**Link**: [PDF](https://arxiv.org/pdf/2504.20323)  

**Abstract**: This report addresses the challenge of limited labeled datasets for developing legal recommender systems, particularly in specialized domains like labor disputes. We propose a new approach leveraging the co-citation of legal articles within cases to establish similarity and enable algorithmic annotation. This method draws a parallel to the concept of case co-citation, utilizing cited precedents as indicators of shared legal issues. To evaluate the labeled results, we employ a system that recommends similar cases based on plaintiffs' accusations, defendants' rebuttals, and points of disputes. The evaluation demonstrates that the recommender, with finetuned text embedding models and a reasonable BiLSTM module can recommend labor cases whose similarity was measured by the co-citation of the legal articles. This research contributes to the development of automated annotation techniques for legal documents, particularly in areas with limited access to comprehensive legal databases. 

---
# UD-English-CHILDES: A Collected Resource of Gold and Silver Universal Dependencies Trees for Child Language Interactions 

**Authors**: Xiulin Yang, Zhuoxuan Ju, Lanni Bu, Zoey Liu, Nathan Schneider  

**Link**: [PDF](https://arxiv.org/pdf/2504.20304)  

**Abstract**: CHILDES is a widely used resource of transcribed child and child-directed speech. This paper introduces UD-English-CHILDES, the first officially released Universal Dependencies (UD) treebank derived from previously dependency-annotated CHILDES data with consistent and unified annotation guidelines. Our corpus harmonizes annotations from 11 children and their caregivers, totaling over 48k sentences. We validate existing gold-standard annotations under the UD v2 framework and provide an additional 1M silver-standard sentences, offering a consistent resource for computational and linguistic research. 

---
# Enhancing Systematic Reviews with Large Language Models: Using GPT-4 and Kimi 

**Authors**: Dandan Chen Kaptur, Yue Huang, Xuejun Ryan Ji, Yanhui Guo, Bradley Kaptur  

**Link**: [PDF](https://arxiv.org/pdf/2504.20276)  

**Abstract**: This research delved into GPT-4 and Kimi, two Large Language Models (LLMs), for systematic reviews. We evaluated their performance by comparing LLM-generated codes with human-generated codes from a peer-reviewed systematic review on assessment. Our findings suggested that the performance of LLMs fluctuates by data volume and question complexity for systematic reviews. 

---
# A Platform for Generating Educational Activities to Teach English as a Second Language 

**Authors**: Aiala Rosá, Santiago Góngora, Juan Pablo Filevich, Ignacio Sastre, Laura Musto, Brian Carpenter, Luis Chiruzzo  

**Link**: [PDF](https://arxiv.org/pdf/2504.20251)  

**Abstract**: We present a platform for the generation of educational activities oriented to teaching English as a foreign language. The different activities --games and language practice exercises-- are strongly based on Natural Language Processing techniques. The platform offers the possibility of playing out-of-the-box games, generated from resources created semi-automatically and then manually curated. It can also generate games or exercises of greater complexity from texts entered by teachers, providing a stage of review and edition of the generated content before use. As a way of expanding the variety of activities in the platform, we are currently experimenting with image and text generation. In order to integrate them and improve the performance of other neural tools already integrated, we are working on migrating the platform to a more powerful server. In this paper we describe the development of our platform and its deployment for end users, discussing the challenges faced and how we overcame them, and also detail our future work plans. 

---
# A Multimodal Pipeline for Clinical Data Extraction: Applying Vision-Language Models to Scans of Transfusion Reaction Reports 

**Authors**: Henning Schäfer, Cynthia S. Schmidt, Johannes Wutzkowsky, Kamil Lorek, Lea Reinartz, Johannes Rückert, Christian Temme, Britta Böckmann, Peter A. Horn, Christoph M. Friedrich  

**Link**: [PDF](https://arxiv.org/pdf/2504.20220)  

**Abstract**: Despite the growing adoption of electronic health records, many processes still rely on paper documents, reflecting the heterogeneous real-world conditions in which healthcare is delivered. The manual transcription process is time-consuming and prone to errors when transferring paper-based data to digital formats. To streamline this workflow, this study presents an open-source pipeline that extracts and categorizes checkbox data from scanned documents. Demonstrated on transfusion reaction reports, the design supports adaptation to other checkbox-rich document types. The proposed method integrates checkbox detection, multilingual optical character recognition (OCR) and multilingual vision-language models (VLMs). The pipeline achieves high precision and recall compared against annually compiled gold-standards from 2017 to 2024. The result is a reduction in administrative workload and accurate regulatory reporting. The open-source availability of this pipeline encourages self-hosted parsing of checkbox forms. 

---
# MICE for CATs: Model-Internal Confidence Estimation for Calibrating Agents with Tools 

**Authors**: Nishant Subramani, Jason Eisner, Justin Svegliato, Benjamin Van Durme, Yu Su, Sam Thomson  

**Link**: [PDF](https://arxiv.org/pdf/2504.20168)  

**Abstract**: Tool-using agents that act in the world need to be both useful and safe. Well-calibrated model confidences can be used to weigh the risk versus reward of potential actions, but prior work shows that many models are poorly calibrated. Inspired by interpretability literature exploring the internals of models, we propose a novel class of model-internal confidence estimators (MICE) to better assess confidence when calling tools. MICE first decodes from each intermediate layer of the language model using logitLens and then computes similarity scores between each layer's generation and the final output. These features are fed into a learned probabilistic classifier to assess confidence in the decoded output. On the simulated trial and error (STE) tool-calling dataset using Llama3 models, we find that MICE beats or matches the baselines on smoothed expected calibration error. Using MICE confidences to determine whether to call a tool significantly improves over strong baselines on a new metric, expected tool-calling utility. Further experiments show that MICE is sample-efficient, can generalize zero-shot to unseen APIs, and results in higher tool-calling utility in scenarios with varying risk levels. Our code is open source, available at this https URL. 

---
# Toward Evaluative Thinking: Meta Policy Optimization with Evolving Reward Models 

**Authors**: Zae Myung Kim, Chanwoo Park, Vipul Raheja, Dongyeop Kang  

**Link**: [PDF](https://arxiv.org/pdf/2504.20157)  

**Abstract**: Reward-based alignment methods for large language models (LLMs) face two key limitations: vulnerability to reward hacking, where models exploit flaws in the reward signal; and reliance on brittle, labor-intensive prompt engineering when LLMs are used as reward models. We introduce Meta Policy Optimization (MPO), a framework that addresses these challenges by integrating a meta-reward model that dynamically refines the reward model's prompt throughout training. In MPO, the meta-reward model monitors the evolving training context and continuously adjusts the reward model's prompt to maintain high alignment, providing an adaptive reward signal that resists exploitation by the policy. This meta-learning approach promotes a more stable policy optimization, and greatly reduces the need for manual reward prompt design. It yields performance on par with or better than models guided by extensively hand-crafted reward prompts. Furthermore, we show that MPO maintains its effectiveness across diverse tasks, such as question answering and mathematical reasoning, without requiring specialized reward designs. Beyond standard RLAIF, MPO's meta-learning formulation is readily extensible to higher-level alignment frameworks. Overall, this method addresses theoretical and practical challenges in reward-based RL alignment for LLMs, paving the way for more robust and adaptable alignment strategies. The code and models will be publicly shared. 

---
# Understanding and Mitigating Risks of Generative AI in Financial Services 

**Authors**: Sebastian Gehrmann, Claire Huang, Xian Teng, Sergei Yurovski, Iyanuoluwa Shode, Chirag S. Patel, Arjun Bhorkar, Naveen Thomas, John Doucette, David Rosenberg, Mark Dredze, David Rabinowitz  

**Link**: [PDF](https://arxiv.org/pdf/2504.20086)  

**Abstract**: To responsibly develop Generative AI (GenAI) products, it is critical to define the scope of acceptable inputs and outputs. What constitutes a "safe" response is an actively debated question. Academic work puts an outsized focus on evaluating models by themselves for general purpose aspects such as toxicity, bias, and fairness, especially in conversational applications being used by a broad audience. In contrast, less focus is put on considering sociotechnical systems in specialized domains. Yet, those specialized systems can be subject to extensive and well-understood legal and regulatory scrutiny. These product-specific considerations need to be set in industry-specific laws, regulations, and corporate governance requirements. In this paper, we aim to highlight AI content safety considerations specific to the financial services domain and outline an associated AI content risk taxonomy. We compare this taxonomy to existing work in this space and discuss implications of risk category violations on various stakeholders. We evaluate how existing open-source technical guardrail solutions cover this taxonomy by assessing them on data collected via red-teaming activities. Our results demonstrate that these guardrails fail to detect most of the content risks we discuss. 

---
# Evaluating Large Language Models on Multiword Expressions in Multilingual and Code-Switched Contexts 

**Authors**: Frances Laureano De Leon, Harish Tayyar Madabushi, Mark G. Lee  

**Link**: [PDF](https://arxiv.org/pdf/2504.20051)  

**Abstract**: Multiword expressions, characterised by non-compositional meanings and syntactic irregularities, are an example of nuanced language. These expressions can be used literally or idiomatically, leading to significant changes in meaning. While large language models have demonstrated strong performance across many tasks, their ability to handle such linguistic subtleties remains uncertain. Therefore, this study evaluates how state-of-the-art language models process the ambiguity of potentially idiomatic multiword expressions, particularly in contexts that are less frequent, where models are less likely to rely on memorisation. By evaluating models across in Portuguese and Galician, in addition to English, and using a novel code-switched dataset and a novel task, we find that large language models, despite their strengths, struggle with nuanced language. In particular, we find that the latest models, including GPT-4, fail to outperform the xlm-roBERTa-base baselines in both detection and semantic tasks, with especially poor performance on the novel tasks we introduce, despite its similarity to existing tasks. Overall, our results demonstrate that multiword expressions, especially those which are ambiguous, continue to be a challenge to models. 

---
# It's the same but not the same: Do LLMs distinguish Spanish varieties? 

**Authors**: Marina Mayor-Rocher, Cristina Pozo, Nina Melero, Gonzalo Martínez, María Grandury, Pedro Reviriego  

**Link**: [PDF](https://arxiv.org/pdf/2504.20049)  

**Abstract**: In recent years, large language models (LLMs) have demonstrated a high capacity for understanding and generating text in Spanish. However, with five hundred million native speakers, Spanish is not a homogeneous language but rather one rich in diatopic variations spanning both sides of the Atlantic. For this reason, in this study, we evaluate the ability of nine language models to identify and distinguish the morphosyntactic and lexical peculiarities of seven varieties of Spanish (Andean, Antillean, Continental Caribbean, Chilean, Peninsular, Mexican and Central American and Rioplatense) through a multiple-choice test. The results indicate that the Peninsular Spanish variety is the best identified by all models and that, among them, GPT-4o is the only model capable of recognizing the variability of the Spanish language.
--
En los últimos años, los grandes modelos de lenguaje (LLMs, por sus siglas en inglés) han demostrado una alta capacidad para comprender y generar texto en español. Sin embargo, con quinientos millones de hablantes nativos, la española no es una lengua homogénea, sino rica en variedades diatópicas que se extienden a ambos lados del Atlántico. Por todo ello, evaluamos en este trabajo la capacidad de nueve modelos de lenguaje de identificar y discernir las peculiaridades morfosintácticas y léxicas de siete variedades de español (andino, antillano, caribeño continental, chileno, español peninsular, mexicano y centroamericano y rioplatense) mediante un test de respuesta múltiple. Los resultados obtenidos indican que la variedad de español peninsular es la mejor identificada por todos los modelos y que, de entre todos, GPT-4o es el único modelo capaz de identificar la variabilidad de la lengua española. 

---
# Towards Understanding the Nature of Attention with Low-Rank Sparse Decomposition 

**Authors**: Zhengfu He, Junxuan Wang, Rui Lin, Xuyang Ge, Wentao Shu, Qiong Tang, Junping Zhang, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2504.20938)  

**Abstract**: We propose Low-Rank Sparse Attention (Lorsa), a sparse replacement model of Transformer attention layers to disentangle original Multi Head Self Attention (MHSA) into individually comprehensible components. Lorsa is designed to address the challenge of attention superposition to understand attention-mediated interaction between features in different token positions. We show that Lorsa heads find cleaner and finer-grained versions of previously discovered MHSA behaviors like induction heads, successor heads and attention sink behavior (i.e., heavily attending to the first token). Lorsa and Sparse Autoencoder (SAE) are both sparse dictionary learning methods applied to different Transformer components, and lead to consistent findings in many ways. For instance, we discover a comprehensive family of arithmetic-specific Lorsa heads, each corresponding to an atomic operation in Llama-3.1-8B. Automated interpretability analysis indicates that Lorsa achieves parity with SAE in interpretability while Lorsa exhibits superior circuit discovery properties, especially for features computed collectively by multiple MHSA heads. We also conduct extensive experiments on architectural design ablation, Lorsa scaling law and error analysis. 

---
# ChestX-Reasoner: Advancing Radiology Foundation Models with Reasoning through Step-by-Step Verification 

**Authors**: Ziqing Fan, Cheng Liang, Chaoyi Wu, Ya Zhang, Yanfeng Wang, Weidi Xie  

**Link**: [PDF](https://arxiv.org/pdf/2504.20930)  

**Abstract**: Recent advances in reasoning-enhanced large language models (LLMs) and multimodal LLMs (MLLMs) have significantly improved performance in complex tasks, yet medical AI models often overlook the structured reasoning processes inherent in clinical practice. In this work, we present ChestX-Reasoner, a radiology diagnosis MLLM designed to leverage process supervision mined directly from clinical reports, reflecting the step-by-step reasoning followed by radiologists. We construct a large dataset by extracting and refining reasoning chains from routine radiology reports. Our two-stage training framework combines supervised fine-tuning and reinforcement learning guided by process rewards to better align model reasoning with clinical standards. We introduce RadRBench-CXR, a comprehensive benchmark featuring 59K visual question answering samples with 301K clinically validated reasoning steps, and propose RadRScore, a metric evaluating reasoning factuality, completeness, and effectiveness. ChestX-Reasoner outperforms existing medical and general-domain MLLMs in both diagnostic accuracy and reasoning ability, achieving 16%, 5.9%, and 18% improvements in reasoning ability compared to the best medical MLLM, the best general MLLM, and its base model, respectively, as well as 3.3%, 24%, and 27% improvements in outcome accuracy. All resources are open-sourced to facilitate further research in medical reasoning MLLMs. 

---
# The Leaderboard Illusion 

**Authors**: Shivalika Singh, Yiyang Nan, Alex Wang, Daniel D'Souza, Sayash Kapoor, Ahmet Üstün, Sanmi Koyejo, Yuntian Deng, Shayne Longpre, Noah Smith, Beyza Ermis, Marzieh Fadaee, Sara Hooker  

**Link**: [PDF](https://arxiv.org/pdf/2504.20879)  

**Abstract**: Measuring progress is fundamental to the advancement of any scientific field. As benchmarks play an increasingly central role, they also grow more susceptible to distortion. Chatbot Arena has emerged as the go-to leaderboard for ranking the most capable AI systems. Yet, in this work we identify systematic issues that have resulted in a distorted playing field. We find that undisclosed private testing practices benefit a handful of providers who are able to test multiple variants before public release and retract scores if desired. We establish that the ability of these providers to choose the best score leads to biased Arena scores due to selective disclosure of performance results. At an extreme, we identify 27 private LLM variants tested by Meta in the lead-up to the Llama-4 release. We also establish that proprietary closed models are sampled at higher rates (number of battles) and have fewer models removed from the arena than open-weight and open-source alternatives. Both these policies lead to large data access asymmetries over time. Providers like Google and OpenAI have received an estimated 19.2% and 20.4% of all data on the arena, respectively. In contrast, a combined 83 open-weight models have only received an estimated 29.7% of the total data. We show that access to Chatbot Arena data yields substantial benefits; even limited additional data can result in relative performance gains of up to 112% on the arena distribution, based on our conservative estimates. Together, these dynamics result in overfitting to Arena-specific dynamics rather than general model quality. The Arena builds on the substantial efforts of both the organizers and an open community that maintains this valuable evaluation platform. We offer actionable recommendations to reform the Chatbot Arena's evaluation framework and promote fairer, more transparent benchmarking for the field 

---
# X-Cross: Dynamic Integration of Language Models for Cross-Domain Sequential Recommendation 

**Authors**: Guy Hadad, Haggai Roitman, Yotam Eshel, Bracha Shapira, Lior Rokach  

**Link**: [PDF](https://arxiv.org/pdf/2504.20859)  

**Abstract**: As new products are emerging daily, recommendation systems are required to quickly adapt to possible new domains without needing extensive retraining. This work presents ``X-Cross'' -- a novel cross-domain sequential-recommendation model that recommends products in new domains by integrating several domain-specific language models; each model is fine-tuned with low-rank adapters (LoRA). Given a recommendation prompt, operating layer by layer, X-Cross dynamically refines the representation of each source language model by integrating knowledge from all other models. These refined representations are propagated from one layer to the next, leveraging the activations from each domain adapter to ensure domain-specific nuances are preserved while enabling adaptability across domains. Using Amazon datasets for sequential recommendation, X-Cross achieves performance comparable to a model that is fine-tuned with LoRA, while using only 25% of the additional parameters. In cross-domain tasks, such as adapting from Toys domain to Tools, Electronics or Sports, X-Cross demonstrates robust performance, while requiring about 50%-75% less fine-tuning data than LoRA to make fine-tuning effective. Furthermore, X-Cross achieves significant improvement in accuracy over alternative cross-domain baselines. Overall, X-Cross enables scalable and adaptive cross-domain recommendations, reducing computational overhead and providing an efficient solution for data-constrained environments. 

---
# ReasonIR: Training Retrievers for Reasoning Tasks 

**Authors**: Rulin Shao, Rui Qiao, Varsha Kishore, Niklas Muennighoff, Xi Victoria Lin, Daniela Rus, Bryan Kian Hsiang Low, Sewon Min, Wen-tau Yih, Pang Wei Koh, Luke Zettlemoyer  

**Link**: [PDF](https://arxiv.org/pdf/2504.20595)  

**Abstract**: We present ReasonIR-8B, the first retriever specifically trained for general reasoning tasks. Existing retrievers have shown limited gains on reasoning tasks, in part because existing training datasets focus on short factual queries tied to documents that straightforwardly answer them. We develop a synthetic data generation pipeline that, for each document, our pipeline creates a challenging and relevant query, along with a plausibly related but ultimately unhelpful hard negative. By training on a mixture of our synthetic data and existing public data, ReasonIR-8B achieves a new state-of-the-art of 29.9 nDCG@10 without reranker and 36.9 nDCG@10 with reranker on BRIGHT, a widely-used reasoning-intensive information retrieval (IR) benchmark. When applied to RAG tasks, ReasonIR-8B improves MMLU and GPQA performance by 6.4% and 22.6% respectively, relative to the closed-book baseline, outperforming other retrievers and search engines. In addition, ReasonIR-8B uses test-time compute more effectively: on BRIGHT, its performance consistently increases with longer and more information-rich rewritten queries; it continues to outperform other retrievers when combined with an LLM reranker. Our training recipe is general and can be easily extended to future LLMs; to this end, we open-source our code, data, and model. 

---
# Reinforcement Learning for Reasoning in Large Language Models with One Training Example 

**Authors**: Yiping Wang, Qing Yang, Zhiyuan Zeng, Liliang Ren, Lucas Liu, Baolin Peng, Hao Cheng, Xuehai He, Kuan Wang, Jianfeng Gao, Weizhu Chen, Shuohang Wang, Simon Shaolei Du, Yelong Shen  

**Link**: [PDF](https://arxiv.org/pdf/2504.20571)  

**Abstract**: We show that reinforcement learning with verifiable reward using one training example (1-shot RLVR) is effective in incentivizing the math reasoning capabilities of large language models (LLMs). Applying RLVR to the base model Qwen2.5-Math-1.5B, we identify a single example that elevates model performance on MATH500 from 36.0% to 73.6%, and improves the average performance across six common mathematical reasoning benchmarks from 17.6% to 35.7%. This result matches the performance obtained using the 1.2k DeepScaleR subset (MATH500: 73.6%, average: 35.9%), which includes the aforementioned example. Similar substantial improvements are observed across various models (Qwen2.5-Math-7B, Llama3.2-3B-Instruct, DeepSeek-R1-Distill-Qwen-1.5B), RL algorithms (GRPO and PPO), and different math examples (many of which yield approximately 30% or greater improvement on MATH500 when employed as a single training example). In addition, we identify some interesting phenomena during 1-shot RLVR, including cross-domain generalization, increased frequency of self-reflection, and sustained test performance improvement even after the training accuracy has saturated, a phenomenon we term post-saturation generalization. Moreover, we verify that the effectiveness of 1-shot RLVR primarily arises from the policy gradient loss, distinguishing it from the "grokking" phenomenon. We also show the critical role of promoting exploration (e.g., by adding entropy loss with an appropriate coefficient) in 1-shot RLVR training. As a bonus, we observe that applying entropy loss alone, without any outcome reward, significantly enhances Qwen2.5-Math-1.5B's performance on MATH500 by 27.4%. These findings can inspire future work on RLVR data efficiency and encourage a re-examination of both recent progress and the underlying mechanisms in RLVR. Our code, model, and data are open source at this https URL 

---
# Search-Based Interaction For Conversation Recommendation via Generative Reward Model Based Simulated User 

**Authors**: Xiaolei Wang, Chunxuan Xia, Junyi Li, Fanzhe Meng, Lei Huang, Jinpeng Wang, Wayne Xin Zhao, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2504.20458)  

**Abstract**: Conversational recommendation systems (CRSs) use multi-turn interaction to capture user preferences and provide personalized recommendations. A fundamental challenge in CRSs lies in effectively understanding user preferences from conversations. User preferences can be multifaceted and complex, posing significant challenges for accurate recommendations even with access to abundant external knowledge. While interaction with users can clarify their true preferences, frequent user involvement can lead to a degraded user experience.
To address this problem, we propose a generative reward model based simulated user, named GRSU, for automatic interaction with CRSs. The simulated user provides feedback to the items recommended by CRSs, enabling them to better capture intricate user preferences through multi-turn interaction. Inspired by generative reward models, we design two types of feedback actions for the simulated user: i.e., generative item scoring, which offers coarse-grained feedback, and attribute-based item critique, which provides fine-grained feedback. To ensure seamless integration, these feedback actions are unified into an instruction-based format, allowing the development of a unified simulated user via instruction tuning on synthesized data. With this simulated user, automatic multi-turn interaction with CRSs can be effectively conducted. Furthermore, to strike a balance between effectiveness and efficiency, we draw inspiration from the paradigm of reward-guided search in complex reasoning tasks and employ beam search for the interaction process. On top of this, we propose an efficient candidate ranking method to improve the recommendation results derived from interaction. Extensive experiments on public datasets demonstrate the effectiveness, efficiency, and transferability of our approach. 

---
# Reviving Any-Subset Autoregressive Models with Principled Parallel Sampling and Speculative Decoding 

**Authors**: Gabe Guo, Stefano Ermon  

**Link**: [PDF](https://arxiv.org/pdf/2504.20456)  

**Abstract**: In arbitrary-order language models, it is an open question how to sample tokens in parallel from the correct joint distribution. With discrete diffusion models, the more tokens they generate in parallel, the less their predicted distributions adhere to the originally learned data distribution, as they rely on a conditional independence assumption that only works with infinitesimally small timesteps. We find that a different class of models, any-subset autoregressive models (AS-ARMs), holds the solution. As implied by the name, AS-ARMs can generate tokens in any order, and in parallel. Moreover, AS-ARMs support parallelized joint probability density estimation, allowing them to correct their own parallel-generated token distributions, via our Any-Subset Speculative Decoding (ASSD) algorithm. ASSD provably enables generation of tokens from the correct joint distribution, with the number of neural network calls upper bounded by the number of tokens predicted. We empirically verify that ASSD speeds up language generation, without sacrificing quality. Furthermore, we provide a mathematically justified scheme for training AS-ARMs for generation, and show that AS-ARMs achieve state-of-the-art performance among sub-200M parameter models on infilling benchmark tasks, and nearly match the performance of models 50X larger on code generation. Our theoretical and empirical results indicate that the once-forgotten AS-ARMs are a promising direction of language modeling. 

---
# mrCAD: Multimodal Refinement of Computer-aided Designs 

**Authors**: William P. McCarthy, Saujas Vaduguru, Karl D. D. Willis, Justin Matejka, Judith E. Fan, Daniel Fried, Yewen Pu  

**Link**: [PDF](https://arxiv.org/pdf/2504.20294)  

**Abstract**: A key feature of human collaboration is the ability to iteratively refine the concepts we have communicated. In contrast, while generative AI excels at the \textit{generation} of content, it often struggles to make specific language-guided \textit{modifications} of its prior outputs. To bridge the gap between how humans and machines perform edits, we present mrCAD, a dataset of multimodal instructions in a communication game. In each game, players created computer aided designs (CADs) and refined them over several rounds to match specific target designs. Only one player, the Designer, could see the target, and they must instruct the other player, the Maker, using text, drawing, or a combination of modalities. mrCAD consists of 6,082 communication games, 15,163 instruction-execution rounds, played between 1,092 pairs of human players. We analyze the dataset and find that generation and refinement instructions differ in their composition of drawing and text. Using the mrCAD task as a benchmark, we find that state-of-the-art VLMs are better at following generation instructions than refinement instructions. These results lay a foundation for analyzing and modeling a multimodal language of refinement that is not represented in previous datasets. 

---
# Weaving Context Across Images: Improving Vision-Language Models through Focus-Centric Visual Chains 

**Authors**: Juntian Zhang, Chuanqi cheng, Yuhan Liu, Wei Liu, Jian Luan, Rui Yan  

**Link**: [PDF](https://arxiv.org/pdf/2504.20199)  

**Abstract**: Vision-language models (VLMs) achieve remarkable success in single-image tasks. However, real-world scenarios often involve intricate multi-image inputs, leading to a notable performance decline as models struggle to disentangle critical information scattered across complex visual features. In this work, we propose Focus-Centric Visual Chain, a novel paradigm that enhances VLMs'perception, comprehension, and reasoning abilities in multi-image scenarios. To facilitate this paradigm, we propose Focus-Centric Data Synthesis, a scalable bottom-up approach for synthesizing high-quality data with elaborate reasoning paths. Through this approach, We construct VISC-150K, a large-scale dataset with reasoning data in the form of Focus-Centric Visual Chain, specifically designed for multi-image tasks. Experimental results on seven multi-image benchmarks demonstrate that our method achieves average performance gains of 3.16% and 2.24% across two distinct model architectures, without compromising the general vision-language capabilities. our study represents a significant step toward more robust and capable vision-language systems that can handle complex visual scenarios. 

---
# ResearchCodeAgent: An LLM Multi-Agent System for Automated Codification of Research Methodologies 

**Authors**: Shubham Gandhi, Dhruv Shah, Manasi Patwardhan, Lovekesh Vig, Gautam Shroff  

**Link**: [PDF](https://arxiv.org/pdf/2504.20117)  

**Abstract**: In this paper we introduce ResearchCodeAgent, a novel multi-agent system leveraging large language models (LLMs) agents to automate the codification of research methodologies described in machine learning literature. The system bridges the gap between high-level research concepts and their practical implementation, allowing researchers auto-generating code of existing research papers for benchmarking or building on top-of existing methods specified in the literature with availability of partial or complete starter code. ResearchCodeAgent employs a flexible agent architecture with a comprehensive action suite, enabling context-aware interactions with the research environment. The system incorporates a dynamic planning mechanism, utilizing both short and long-term memory to adapt its approach iteratively. We evaluate ResearchCodeAgent on three distinct machine learning tasks with distinct task complexity and representing different parts of the ML pipeline: data augmentation, optimization, and data batching. Our results demonstrate the system's effectiveness and generalizability, with 46.9% of generated code being high-quality and error-free, and 25% showing performance improvements over baseline implementations. Empirical analysis shows an average reduction of 57.9% in coding time compared to manual implementation. We observe higher gains for more complex tasks. ResearchCodeAgent represents a significant step towards automating the research implementation process, potentially accelerating the pace of machine learning research. 

---
# MATCHA: Can Multi-Agent Collaboration Build a Trustworthy Conversational Recommender? 

**Authors**: Zheng Hui, Xiaokai Wei, Yexi Jiang, Kevin Gao, Chen Wang, Frank Ong, Se-eun Yoon, Rachit Pareek, Michelle Gong  

**Link**: [PDF](https://arxiv.org/pdf/2504.20094)  

**Abstract**: In this paper, we propose a multi-agent collaboration framework called MATCHA for conversational recommendation system, leveraging large language models (LLMs) to enhance personalization and user engagement. Users can request recommendations via free-form text and receive curated lists aligned with their interests, preferences, and constraints. Our system introduces specialized agents for intent analysis, candidate generation, ranking, re-ranking, explainability, and safeguards. These agents collaboratively improve recommendations accuracy, diversity, and safety. On eight metrics, our model achieves superior or comparable performance to the current state-of-the-art. Through comparisons with six baseline models, our approach addresses key challenges in conversational recommendation systems for game recommendations, including: (1) handling complex, user-specific requests, (2) enhancing personalization through multi-agent collaboration, (3) empirical evaluation and deployment, and (4) ensuring safe and trustworthy interactions. 

---
# AI Awareness 

**Authors**: Xiaojian Li, Haoyuan Shi, Rongwu Xu, Wei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2504.20084)  

**Abstract**: Recent breakthroughs in artificial intelligence (AI) have brought about increasingly capable systems that demonstrate remarkable abilities in reasoning, language understanding, and problem-solving. These advancements have prompted a renewed examination of AI awareness, not as a philosophical question of consciousness, but as a measurable, functional capacity. In this review, we explore the emerging landscape of AI awareness, which includes meta-cognition (the ability to represent and reason about its own state), self-awareness (recognizing its own identity, knowledge, limitations, inter alia), social awareness (modeling the knowledge, intentions, and behaviors of other agents), and situational awareness (assessing and responding to the context in which it operates).
First, we draw on insights from cognitive science, psychology, and computational theory to trace the theoretical foundations of awareness and examine how the four distinct forms of AI awareness manifest in state-of-the-art AI. Next, we systematically analyze current evaluation methods and empirical findings to better understand these manifestations. Building on this, we explore how AI awareness is closely linked to AI capabilities, demonstrating that more aware AI agents tend to exhibit higher levels of intelligent behaviors. Finally, we discuss the risks associated with AI awareness, including key topics in AI safety, alignment, and broader ethical concerns.
AI awareness is a double-edged sword: it improves general capabilities, i.e., reasoning, safety, while also raises concerns around misalignment and societal risks, demanding careful oversight as AI capabilities grow. On the whole, our interdisciplinary review provides a roadmap for future research and aims to clarify the role of AI awareness in the ongoing development of intelligent machines. 

---
# RAGEN: Understanding Self-Evolution in LLM Agents via Multi-Turn Reinforcement Learning 

**Authors**: Zihan Wang, Kangrui Wang, Qineng Wang, Pingyue Zhang, Linjie Li, Zhengyuan Yang, Kefan Yu, Minh Nhat Nguyen, Licheng Liu, Eli Gottlieb, Monica Lam, Yiping Lu, Kyunghyun Cho, Jiajun Wu, Li Fei-Fei, Lijuan Wang, Yejin Choi, Manling Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.20073)  

**Abstract**: Training large language models (LLMs) as interactive agents presents unique challenges including long-horizon decision making and interacting with stochastic environment feedback. While reinforcement learning (RL) has enabled progress in static tasks, multi-turn agent RL training remains underexplored. We propose StarPO (State-Thinking-Actions-Reward Policy Optimization), a general framework for trajectory-level agent RL, and introduce RAGEN, a modular system for training and evaluating LLM agents. Our study on three stylized environments reveals three core findings. First, our agent RL training shows a recurring mode of Echo Trap where reward variance cliffs and gradient spikes; we address this with StarPO-S, a stabilized variant with trajectory filtering, critic incorporation, and decoupled clipping. Second, we find the shaping of RL rollouts would benefit from diverse initial states, medium interaction granularity and more frequent sampling. Third, we show that without fine-grained, reasoning-aware reward signals, agent reasoning hardly emerge through multi-turn RL and they may show shallow strategies or hallucinated thoughts. Code and environments are available at this https URL. 

---
# Recommending Clinical Trials for Online Patient Cases using Artificial Intelligence 

**Authors**: Joey Chan, Qiao Jin, Nicholas Wan, Charalampos S. Floudas, Elisabetta Xue, Zhiyong Lu  

**Link**: [PDF](https://arxiv.org/pdf/2504.20059)  

**Abstract**: Clinical trials are crucial for assessing new treatments; however, recruitment challenges - such as limited awareness, complex eligibility criteria, and referral barriers - hinder their success. With the growth of online platforms, patients increasingly turn to social media and health communities for support, research, and advocacy, expanding recruitment pools and established enrollment pathways. Recognizing this potential, we utilized TrialGPT, a framework that leverages a large language model (LLM) as its backbone, to match 50 online patient cases (collected from published case reports and a social media website) to clinical trials and evaluate performance against traditional keyword-based searches. Our results show that TrialGPT outperforms traditional methods by 46% in identifying eligible trials, with each patient, on average, being eligible for around 7 trials. Additionally, our outreach efforts to case authors and trial organizers regarding these patient-trial matches yielded highly positive feedback, which we present from both perspectives. 

---
