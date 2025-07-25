# Language Models Improve When Pretraining Data Matches Target Tasks 

**Authors**: David Mizrahi, Anders Boesen Lindbo Larsen, Jesse Allardice, Suzie Petryk, Yuri Gorokhov, Jeffrey Li, Alex Fang, Josh Gardner, Tom Gunter, Afshin Dehghan  

**Link**: [PDF](https://arxiv.org/pdf/2507.12466)  

**Abstract**: Every data selection method inherently has a target. In practice, these targets often emerge implicitly through benchmark-driven iteration: researchers develop selection strategies, train models, measure benchmark performance, then refine accordingly. This raises a natural question: what happens when we make this optimization explicit? To explore this, we propose benchmark-targeted ranking (BETR), a simple method that selects pretraining documents based on similarity to benchmark training examples. BETR embeds benchmark examples and a sample of pretraining documents in a shared space, scores this sample by similarity to benchmarks, then trains a lightweight classifier to predict these scores for the full corpus. We compare data selection methods by training over 500 models spanning $10^{19}$ to $10^{22}$ FLOPs and fitting scaling laws to them. From this, we find that simply aligning pretraining data to evaluation benchmarks using BETR achieves a 2.1x compute multiplier over DCLM-Baseline (4.7x over unfiltered data) and improves performance on 9 out of 10 tasks across all scales. BETR also generalizes well: when targeting a diverse set of benchmarks disjoint from our evaluation suite, it still matches or outperforms baselines. Our scaling analysis further reveals a clear trend: larger models require less aggressive filtering. Overall, our findings show that directly matching pretraining data to target tasks precisely shapes model capabilities and highlight that optimal selection strategies must adapt to model scale. 

---
# S2WTM: Spherical Sliced-Wasserstein Autoencoder for Topic Modeling 

**Authors**: Suman Adhya, Debarshi Kumar Sanyal  

**Link**: [PDF](https://arxiv.org/pdf/2507.12451)  

**Abstract**: Modeling latent representations in a hyperspherical space has proven effective for capturing directional similarities in high-dimensional text data, benefiting topic modeling. Variational autoencoder-based neural topic models (VAE-NTMs) commonly adopt the von Mises-Fisher prior to encode hyperspherical structure. However, VAE-NTMs often suffer from posterior collapse, where the KL divergence term in the objective function highly diminishes, leading to ineffective latent representations. To mitigate this issue while modeling hyperspherical structure in the latent space, we propose the Spherical Sliced Wasserstein Autoencoder for Topic Modeling (S2WTM). S2WTM employs a prior distribution supported on the unit hypersphere and leverages the Spherical Sliced-Wasserstein distance to align the aggregated posterior distribution with the prior. Experimental results demonstrate that S2WTM outperforms state-of-the-art topic models, generating more coherent and diverse topics while improving performance on downstream tasks. 

---
# Can We Predict Alignment Before Models Finish Thinking? Towards Monitoring Misaligned Reasoning Models 

**Authors**: Yik Siu Chan, Zheng-Xin Yong, Stephen H. Bach  

**Link**: [PDF](https://arxiv.org/pdf/2507.12428)  

**Abstract**: Open-weights reasoning language models generate long chains-of-thought (CoTs) before producing a final response, which improves performance but introduces additional alignment risks, with harmful content often appearing in both the CoTs and the final outputs. In this work, we investigate if we can use CoTs to predict final response misalignment. We evaluate a range of monitoring approaches, including humans, highly-capable large language models, and text classifiers, using either CoT text or activations. First, we find that a simple linear probe trained on CoT activations can significantly outperform all text-based methods in predicting whether a final response will be safe or unsafe. CoT texts are often unfaithful and can mislead humans and classifiers, while model latents (i.e., CoT activations) offer a more reliable predictive signal. Second, the probe makes accurate predictions before reasoning completes, achieving strong performance even when applied to early CoT segments. These findings generalize across model sizes, families, and safety benchmarks, suggesting that lightweight probes could enable real-time safety monitoring and early intervention during generation. 

---
# Advancing Retrieval-Augmented Generation for Structured Enterprise and Internal Data 

**Authors**: Chandana Cheerla  

**Link**: [PDF](https://arxiv.org/pdf/2507.12425)  

**Abstract**: Organizations increasingly rely on proprietary enterprise data, including HR records, structured reports, and tabular documents, for critical decision-making. While Large Language Models (LLMs) have strong generative capabilities, they are limited by static pretraining, short context windows, and challenges in processing heterogeneous data formats. Conventional Retrieval-Augmented Generation (RAG) frameworks address some of these gaps but often struggle with structured and semi-structured data.
This work proposes an advanced RAG framework that combines hybrid retrieval strategies using dense embeddings (all-mpnet-base-v2) and BM25, enhanced by metadata-aware filtering with SpaCy NER and cross-encoder reranking. The framework applies semantic chunking to maintain textual coherence and retains tabular data structures to preserve row-column integrity. Quantized indexing optimizes retrieval efficiency, while human-in-the-loop feedback and conversation memory improve adaptability.
Experiments on enterprise datasets show notable improvements: Precision@5 increased by 15 percent (90 versus 75), Recall@5 by 13 percent (87 versus 74), and Mean Reciprocal Rank by 16 percent (0.85 versus 0.69). Qualitative evaluations show higher scores in Faithfulness (4.6 versus 3.0), Completeness (4.2 versus 2.5), and Relevance (4.5 versus 3.2) on a 5-point Likert scale. These results demonstrate the framework's effectiveness in delivering accurate, comprehensive, and contextually relevant responses for enterprise tasks. Future work includes extending to multimodal data and integrating agent-based retrieval. The source code will be released at this https URL 

---
# Probing for Arithmetic Errors in Language Models 

**Authors**: Yucheng Sun, Alessandro Stolfo, Mrinmaya Sachan  

**Link**: [PDF](https://arxiv.org/pdf/2507.12379)  

**Abstract**: We investigate whether internal activations in language models can be used to detect arithmetic errors. Starting with a controlled setting of 3-digit addition, we show that simple probes can accurately decode both the model's predicted output and the correct answer from hidden states, regardless of whether the model's output is correct. Building on this, we train lightweight error detectors that predict model correctness with over 90% accuracy. We then extend our analysis to structured chain-of-thought traces on addition-only GSM8K problems and find that probes trained on simple arithmetic generalize well to this more complex setting, revealing consistent internal representations. Finally, we demonstrate that these probes can guide selective re-prompting of erroneous reasoning steps, improving task accuracy with minimal disruption to correct outputs. Our findings suggest that arithmetic errors can be anticipated from internal activations alone, and that simple probes offer a viable path toward lightweight model self-correction. 

---
# Web-Browsing LLMs Can Access Social Media Profiles and Infer User Demographics 

**Authors**: Meysam Alizadeh, Fabrizio Gilardi, Zeynab Samei, Mohsen Mosleh  

**Link**: [PDF](https://arxiv.org/pdf/2507.12372)  

**Abstract**: Large language models (LLMs) have traditionally relied on static training data, limiting their knowledge to fixed snapshots. Recent advancements, however, have equipped LLMs with web browsing capabilities, enabling real time information retrieval and multi step reasoning over live web content. While prior studies have demonstrated LLMs ability to access and analyze websites, their capacity to directly retrieve and analyze social media data remains unexplored. Here, we evaluate whether web browsing LLMs can infer demographic attributes of social media users given only their usernames. Using a synthetic dataset of 48 X (Twitter) accounts and a survey dataset of 1,384 international participants, we show that these models can access social media content and predict user demographics with reasonable accuracy. Analysis of the synthetic dataset further reveals how LLMs parse and interpret social media profiles, which may introduce gender and political biases against accounts with minimal activity. While this capability holds promise for computational social science in the post API era, it also raises risks of misuse particularly in information operations and targeted advertising underscoring the need for safeguards. We recommend that LLM providers restrict this capability in public facing applications, while preserving controlled access for verified research purposes. 

---
# Beyond Single Models: Enhancing LLM Detection of Ambiguity in Requests through Debate 

**Authors**: Ana Davila, Jacinto Colan, Yasuhisa Hasegawa  

**Link**: [PDF](https://arxiv.org/pdf/2507.12370)  

**Abstract**: Large Language Models (LLMs) have demonstrated significant capabilities in understanding and generating human language, contributing to more natural interactions with complex systems. However, they face challenges such as ambiguity in user requests processed by LLMs. To address these challenges, this paper introduces and evaluates a multi-agent debate framework designed to enhance detection and resolution capabilities beyond single models. The framework consists of three LLM architectures (Llama3-8B, Gemma2-9B, and Mistral-7B variants) and a dataset with diverse ambiguities. The debate framework markedly enhanced the performance of Llama3-8B and Mistral-7B variants over their individual baselines, with Mistral-7B-led debates achieving a notable 76.7% success rate and proving particularly effective for complex ambiguities and efficient consensus. While acknowledging varying model responses to collaborative strategies, these findings underscore the debate framework's value as a targeted method for augmenting LLM capabilities. This work offers important insights for developing more robust and adaptive language understanding systems by showing how structured debates can lead to improved clarity in interactive systems. 

---
# Exploring Gender Bias in Alzheimer's Disease Detection: Insights from Mandarin and Greek Speech Perception 

**Authors**: Liu He, Yuanchao Li, Rui Feng, XinRan Han, Yin-Long Liu, Yuwei Yang, Zude Zhu, Jiahong Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2507.12356)  

**Abstract**: Gender bias has been widely observed in speech perception tasks, influenced by the fundamental voicing differences between genders. This study reveals a gender bias in the perception of Alzheimer's Disease (AD) speech. In a perception experiment involving 16 Chinese listeners evaluating both Chinese and Greek speech, we identified that male speech was more frequently identified as AD, with this bias being particularly pronounced in Chinese speech. Acoustic analysis showed that shimmer values in male speech were significantly associated with AD perception, while speech portion exhibited a significant negative correlation with AD identification. Although language did not have a significant impact on AD perception, our findings underscore the critical role of gender bias in AD speech perception. This work highlights the necessity of addressing gender bias when developing AD detection models and calls for further research to validate model performance across different linguistic contexts. 

---
# Chain-of-Descriptions: Improving Code LLMs for VHDL Code Generation and Summarization 

**Authors**: Prashanth Vijayaraghavan, Apoorva Nitsure, Charles Mackin, Luyao Shi, Stefano Ambrogio, Arvind Haran, Viresh Paruthi, Ali Elzein, Dan Coops, David Beymer, Tyler Baldwin, Ehsan Degan  

**Link**: [PDF](https://arxiv.org/pdf/2507.12308)  

**Abstract**: Large Language Models (LLMs) have become widely used across diverse NLP tasks and domains, demonstrating their adaptability and effectiveness. In the realm of Electronic Design Automation (EDA), LLMs show promise for tasks like Register-Transfer Level (RTL) code generation and summarization. However, despite the proliferation of LLMs for general code-related tasks, there's a dearth of research focused on evaluating and refining these models for hardware description languages (HDLs), notably VHDL. In this study, we evaluate the performance of existing code LLMs for VHDL code generation and summarization using various metrics and two datasets -- VHDL-Eval and VHDL-Xform. The latter, an in-house dataset, aims to gauge LLMs' understanding of functionally equivalent code. Our findings reveal consistent underperformance of these models across different metrics, underscoring a significant gap in their suitability for this domain. To address this challenge, we propose Chain-of-Descriptions (CoDes), a novel approach to enhance the performance of LLMs for VHDL code generation and summarization tasks. CoDes involves generating a series of intermediate descriptive steps based on: (i) the problem statement for code generation, and (ii) the VHDL code for summarization. These steps are then integrated with the original input prompt (problem statement or code) and provided as input to the LLMs to generate the final output. Our experiments demonstrate that the CoDes approach significantly surpasses the standard prompting strategy across various metrics on both datasets. This method not only improves the quality of VHDL code generation and summarization but also serves as a framework for future research aimed at enhancing code LLMs for VHDL. 

---
# Text-ADBench: Text Anomaly Detection Benchmark based on LLMs Embedding 

**Authors**: Feng Xiao, Jicong Fan  

**Link**: [PDF](https://arxiv.org/pdf/2507.12295)  

**Abstract**: Text anomaly detection is a critical task in natural language processing (NLP), with applications spanning fraud detection, misinformation identification, spam detection and content moderation, etc. Despite significant advances in large language models (LLMs) and anomaly detection algorithms, the absence of standardized and comprehensive benchmarks for evaluating the existing anomaly detection methods on text data limits rigorous comparison and development of innovative approaches. This work performs a comprehensive empirical study and introduces a benchmark for text anomaly detection, leveraging embeddings from diverse pre-trained language models across a wide array of text datasets. Our work systematically evaluates the effectiveness of embedding-based text anomaly detection by incorporating (1) early language models (GloVe, BERT); (2) multiple LLMs (LLaMa-2, LLama-3, Mistral, OpenAI (small, ada, large)); (3) multi-domain text datasets (news, social media, scientific publications); (4) comprehensive evaluation metrics (AUROC, AUPRC). Our experiments reveal a critical empirical insight: embedding quality significantly governs anomaly detection efficacy, and deep learning-based approaches demonstrate no performance advantage over conventional shallow algorithms (e.g., KNN, Isolation Forest) when leveraging LLM-derived this http URL addition, we observe strongly low-rank characteristics in cross-model performance matrices, which enables an efficient strategy for rapid model evaluation (or embedding evaluation) and selection in practical applications. Furthermore, by open-sourcing our benchmark toolkit that includes all embeddings from different models and code at this https URL, this work provides a foundation for future research in robust and scalable text anomaly detection systems. 

---
# Infherno: End-to-end Agent-based FHIR Resource Synthesis from Free-form Clinical Notes 

**Authors**: Johann Frei, Nils Feldhus, Lisa Raithel, Roland Roller, Alexander Meyer, Frank Kramer  

**Link**: [PDF](https://arxiv.org/pdf/2507.12261)  

**Abstract**: For clinical data integration and healthcare services, the HL7 FHIR standard has established itself as a desirable format for interoperability between complex health data. Previous attempts at automating the translation from free-form clinical notes into structured FHIR resources rely on modular, rule-based systems or LLMs with instruction tuning and constrained decoding. Since they frequently suffer from limited generalizability and structural inconformity, we propose an end-to-end framework powered by LLM agents, code execution, and healthcare terminology database tools to address these issues. Our solution, called Infherno, is designed to adhere to the FHIR document schema and competes well with a human baseline in predicting FHIR resources from unstructured text. The implementation features a front end for custom and synthetic data and both local and proprietary models, supporting clinical data integration processes and interoperability across institutions. 

---
# Translationese-index: Using Likelihood Ratios for Graded and Generalizable Measurement of Translationese 

**Authors**: Yikang Liu, Wanyang Zhang, Yiming Wang, Jialong Tang, Pei Zhang, Baosong Yang, Fei Huang, Rui Wang, Hai Hu  

**Link**: [PDF](https://arxiv.org/pdf/2507.12260)  

**Abstract**: In this paper, we propose the first quantitative measure for translationese -- the translationese-index (T-index) for graded and generalizable measurement of translationese, computed from the likelihood ratios of two contrastively fine-tuned language models (LMs). We use a synthesized dataset and a dataset with translations in the wild to evaluate T-index's generalizability in cross-domain settings and its validity against human judgments. Our results show that T-index is both robust and efficient. T-index scored by two 0.5B LMs fine-tuned on only 1-5k pairs of synthetic data can well capture translationese in the wild. We find that the relative differences in T-indices between translations can well predict pairwise translationese annotations obtained from human annotators; and the absolute values of T-indices correlate well with human ratings of degrees of translationese (Pearson's $r = 0.568$). Additionally, the correlation between T-index and existing machine translation (MT) quality estimation (QE) metrics such as BLEU and COMET is low, suggesting that T-index is not covered by these metrics and can serve as a complementary metric in MT QE. 

---
# Improving Contextual ASR via Multi-grained Fusion with Large Language Models 

**Authors**: Shilin Zhou, Zhenghua Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.12252)  

**Abstract**: While end-to-end Automatic Speech Recognition (ASR) models have shown impressive performance in transcribing general speech, they often struggle to accurately recognize contextually relevant keywords, such as proper nouns or user-specific entities.
Previous approaches have explored leveraging keyword dictionaries in the textual modality to improve keyword recognition, either through token-level fusion that guides token-by-token generation or phrase-level fusion that enables direct copying of keyword phrases.
However, these methods operate at different granularities and have their own limitations.
In this paper, we propose a novel multi-grained fusion approach that jointly leverages the strengths of both token-level and phrase-level fusion with Large Language Models (LLMs).
Our approach incorporates a late-fusion strategy that elegantly combines ASR's acoustic information with LLM's rich contextual knowledge, balancing fine-grained token precision with holistic phrase-level understanding.
Experiments on Chinese and English datasets demonstrate that our approach achieves state-of-the-art performance on keyword-related metrics while preserving high accuracy on non-keyword text.
Ablation studies further confirm that the token-level and phrase-level components both contribute significantly to the performance gains, complementing each other in our joint multi-grained framework.
The code and models will be publicly available at this https URL. 

---
# Towards few-shot isolated word reading assessment 

**Authors**: Reuben Smit, Retief Louw, Herman Kamper  

**Link**: [PDF](https://arxiv.org/pdf/2507.12217)  

**Abstract**: We explore an ASR-free method for isolated word reading assessment in low-resource settings. Our few-shot approach compares input child speech to a small set of adult-provided reference templates. Inputs and templates are encoded using intermediate layers from large self-supervised learned (SSL) models. Using an Afrikaans child speech benchmark, we investigate design options such as discretising SSL features and barycentre averaging of the templates. Idealised experiments show reasonable performance for adults, but a substantial drop for child speech input, even with child templates. Despite the success of employing SSL representations in low-resource speech tasks, our work highlights the limitations of SSL representations for processing child data when used in a few-shot classification system. 

---
# Toward a Behavioural Translation Style Space: Simulating the Temporal Dynamics of Affect, Behaviour, and Cognition in Human Translation Production 

**Authors**: Michael Carl, Takanori Mizowaki, Aishvarya Ray, Masaru Yamada, Devi Sri Bandaru, Xinyue Ren  

**Link**: [PDF](https://arxiv.org/pdf/2507.12208)  

**Abstract**: The paper introduces a Behavioural Translation Style Space (BTSS) that describes possible behavioural translation patterns. The suggested BTSS is organized as a hierarchical structure that entails various embedded processing layers. We posit that observable translation behaviour - i.e., eye and finger movements - is fundamental when executing the physical act of translation but it is caused and shaped by higher-order cognitive processes and affective translation states. We analyse records of keystrokes and gaze data as indicators of the hidden mental processing structure and organize the behavioural patterns as a multi-layered embedded BTSS. The BTSS serves as the basis for a computational translation agent to simulate the temporal dynamics of affect, automatized behaviour and cognition during human translation production. 

---
# Overview of the Sensemaking Task at the ELOQUENT 2025 Lab: LLMs as Teachers, Students and Evaluators 

**Authors**: Pavel Šindelář, Ondřej Bojar  

**Link**: [PDF](https://arxiv.org/pdf/2507.12143)  

**Abstract**: ELOQUENT is a set of shared tasks that aims to create easily testable high-level criteria for evaluating generative language models. Sensemaking is one such shared task.
In Sensemaking, we try to assess how well generative models ``make sense out of a given text'' in three steps inspired by exams in a classroom setting: (1) Teacher systems should prepare a set of questions, (2) Student systems should answer these questions, and (3) Evaluator systems should score these answers, all adhering rather strictly to a given set of input materials.
We report on the 2025 edition of Sensemaking, where we had 7 sources of test materials (fact-checking analyses of statements, textbooks, transcribed recordings of a lecture, and educational videos) spanning English, German, Ukrainian, and Czech languages.
This year, 4 teams participated, providing us with 2 Teacher submissions, 2 Student submissions, and 2 Evaluator submissions. We added baselines for Teacher and Student using commercial large language model systems. We devised a fully automatic evaluation procedure, which we compare to a minimalistic manual evaluation.
We were able to make some interesting observations. For the first task, the creation of questions, better evaluation strategies will still have to be devised because it is difficult to discern the quality of the various candidate question sets. In the second task, question answering, the LLMs examined overall perform acceptably, but restricting their answers to the given input texts remains problematic. In the third task, evaluation of question answers, our adversarial tests reveal that systems using the LLM-as-a-Judge paradigm erroneously rate both garbled question-answer pairs and answers to mixed-up questions as acceptable. 

---
# Iterative Augmentation with Summarization Refinement (IASR) Evaluation for Unstructured Survey data Modeling and Analysis 

**Authors**: Payal Bhattad, Sai Manoj Pudukotai Dinakarrao, Anju Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2507.12126)  

**Abstract**: Text data augmentation is a widely used strategy for mitigating data sparsity in natural language processing (NLP), particularly in low-resource settings where limited samples hinder effective semantic modeling. While augmentation can improve input diversity and downstream interpretability, existing techniques often lack mechanisms to ensure semantic preservation during large-scale or iterative generation, leading to redundancy and instability. This work introduces a principled evaluation framework for large language model (LLM) based text augmentation, comprising two components: (1) Scalability Analysis, which measures semantic consistency as augmentation volume increases, and (2) Iterative Augmentation with Summarization Refinement (IASR), which evaluates semantic drift across recursive paraphrasing cycles. Empirical evaluations across state-of-the-art LLMs show that GPT-3.5 Turbo achieved the best balance of semantic fidelity, diversity, and generation efficiency. Applied to a real-world topic modeling task using BERTopic with GPT-enhanced few-shot labeling, the proposed approach results in a 400% increase in topic granularity and complete elimination of topic overlaps. These findings validated the utility of the proposed frameworks for structured evaluation of LLM-based augmentation in practical NLP pipelines. 

---
# Findings of MEGA: Maths Explanation with LLMs using the Socratic Method for Active Learning 

**Authors**: Tosin Adewumi, Foteini Simistira Liwicki, Marcus Liwicki, Viktor Gardelli, Lama Alkhaled, Hamam Mokayed  

**Link**: [PDF](https://arxiv.org/pdf/2507.12079)  

**Abstract**: This paper presents an intervention study on the effects of the combined methods of (1) the Socratic method, (2) Chain of Thought (CoT) reasoning, (3) simplified gamification and (4) formative feedback on university students' Maths learning driven by large language models (LLMs). We call our approach Mathematics Explanations through Games by AI LLMs (MEGA). Some students struggle with Maths and as a result avoid Math-related discipline or subjects despite the importance of Maths across many fields, including signal processing. Oftentimes, students' Maths difficulties stem from suboptimal pedagogy. We compared the MEGA method to the traditional step-by-step (CoT) method to ascertain which is better by using a within-group design after randomly assigning questions for the participants, who are university students. Samples (n=60) were randomly drawn from each of the two test sets of the Grade School Math 8K (GSM8K) and Mathematics Aptitude Test of Heuristics (MATH) datasets, based on the error margin of 11%, the confidence level of 90%, and a manageable number of samples for the student evaluators. These samples were used to evaluate two capable LLMs at length (Generative Pretrained Transformer 4o (GPT4o) and Claude 3.5 Sonnet) out of the initial six that were tested for capability. The results showed that students agree in more instances that the MEGA method is experienced as better for learning for both datasets. It is even much better than the CoT (47.5% compared to 26.67%) in the more difficult MATH dataset, indicating that MEGA is better at explaining difficult Maths problems. 

---
# BOOKCOREF: Coreference Resolution at Book Scale 

**Authors**: Giuliano Martinelli, Tommaso Bonomo, Pere-Lluís Huguet Cabot, Roberto Navigli  

**Link**: [PDF](https://arxiv.org/pdf/2507.12075)  

**Abstract**: Coreference Resolution systems are typically evaluated on benchmarks containing small- to medium-scale documents. When it comes to evaluating long texts, however, existing benchmarks, such as LitBank, remain limited in length and do not adequately assess system capabilities at the book scale, i.e., when co-referring mentions span hundreds of thousands of tokens. To fill this gap, we first put forward a novel automatic pipeline that produces high-quality Coreference Resolution annotations on full narrative texts. Then, we adopt this pipeline to create the first book-scale coreference benchmark, BOOKCOREF, with an average document length of more than 200,000 tokens. We carry out a series of experiments showing the robustness of our automatic procedure and demonstrating the value of our resource, which enables current long-document coreference systems to gain up to +20 CoNLL-F1 points when evaluated on full books. Moreover, we report on the new challenges introduced by this unprecedented book-scale setting, highlighting that current models fail to deliver the same performance they achieve on smaller documents. We release our data and code to encourage research and development of new book-scale Coreference Resolution systems at this https URL. 

---
# StylOch at PAN: Gradient-Boosted Trees with Frequency-Based Stylometric Features 

**Authors**: Jeremi K. Ochab, Mateusz Matias, Tymoteusz Boba, Tomasz Walkowiak  

**Link**: [PDF](https://arxiv.org/pdf/2507.12064)  

**Abstract**: This submission to the binary AI detection task is based on a modular stylometric pipeline, where: public spaCy models are used for text preprocessing (including tokenisation, named entity recognition, dependency parsing, part-of-speech tagging, and morphology annotation) and extracting several thousand features (frequencies of n-grams of the above linguistic annotations); light-gradient boosting machines are used as the classifier. We collect a large corpus of more than 500 000 machine-generated texts for the classifier's training. We explore several parameter options to increase the classifier's capacity and take advantage of that training set. Our approach follows the non-neural, computationally inexpensive but explainable approach found effective previously. 

---
# Evaluating the Ability of Large Language Models to Reason about Cardinal Directions, Revisited 

**Authors**: Anthony G Cohn, Robert E Blackwell  

**Link**: [PDF](https://arxiv.org/pdf/2507.12059)  

**Abstract**: We investigate the abilities of 28 Large language Models (LLMs) to reason about cardinal directions (CDs) using a benchmark generated from a set of templates, extensively testing an LLM's ability to determine the correct CD given a particular scenario. The templates allow for a number of degrees of variation such as means of locomotion of the agent involved, and whether set in the first, second or third person. Even the newer Large Reasoning Models are unable to reliably determine the correct CD for all questions. This paper summarises and extends earlier work presented at COSIT-24. 

---
# A Comparative Approach to Assessing Linguistic Creativity of Large Language Models and Humans 

**Authors**: Anca Dinu, Andra-Maria Florescu, Alina Resceanu  

**Link**: [PDF](https://arxiv.org/pdf/2507.12039)  

**Abstract**: The following paper introduces a general linguistic creativity test for humans and Large Language Models (LLMs). The test consists of various tasks aimed at assessing their ability to generate new original words and phrases based on word formation processes (derivation and compounding) and on metaphorical language use. We administered the test to 24 humans and to an equal number of LLMs, and we automatically evaluated their answers using OCSAI tool for three criteria: Originality, Elaboration, and Flexibility. The results show that LLMs not only outperformed humans in all the assessed criteria, but did better in six out of the eight test tasks. We then computed the uniqueness of the individual answers, which showed some minor differences between humans and LLMs. Finally, we performed a short manual analysis of the dataset, which revealed that humans are more inclined towards E(extending)-creativity, while LLMs favor F(ixed)-creativity. 

---
# Improving Data and Parameter Efficiency of Neural Language Models Using Representation Analysis 

**Authors**: Josip Jukić  

**Link**: [PDF](https://arxiv.org/pdf/2507.12004)  

**Abstract**: This thesis addresses challenges related to data and parameter efficiency in neural language models, with a focus on representation analysis and the introduction of new optimization techniques. The first part examines the properties and dynamics of language representations within neural models, emphasizing their significance in enhancing robustness and generalization. It proposes innovative approaches based on representation smoothness, including regularization strategies that utilize Jacobian and Hessian matrices to stabilize training and mitigate sensitivity to input perturbations. The second part focuses on methods to significantly enhance data and parameter efficiency by integrating active learning strategies with parameter-efficient fine-tuning, guided by insights from representation smoothness analysis. It presents smoothness-informed early-stopping techniques designed to eliminate the need for labeled validation sets and proposes innovative combinations of active learning and parameter-efficient fine-tuning to reduce labeling efforts and computational resources. Extensive experimental evaluations across various NLP tasks demonstrate that these combined approaches substantially outperform traditional methods in terms of performance, stability, and efficiency. The third part explores weak supervision techniques enhanced by in-context learning to effectively utilize unlabeled data, further reducing dependence on extensive labeling. It shows that using in-context learning as a mechanism for weak supervision enables models to better generalize from limited labeled data by leveraging unlabeled examples more effectively during training. Comprehensive empirical evaluations confirm significant gains in model accuracy, adaptability, and robustness, especially in low-resource settings and dynamic data environments. 

---
# Simplifications are Absolutists: How Simplified Language Reduces Word Sense Awareness in LLM-Generated Definitions 

**Authors**: Lukas Ellinger, Miriam Anschütz, Georg Groh  

**Link**: [PDF](https://arxiv.org/pdf/2507.11981)  

**Abstract**: Large Language Models (LLMs) can provide accurate word definitions and explanations for any context. However, the scope of the definition changes for different target groups, like children or language learners. This is especially relevant for homonyms, words with multiple meanings, where oversimplification might risk information loss by omitting key senses, potentially misleading users who trust LLM outputs. We investigate how simplification impacts homonym definition quality across three target groups: Normal, Simple, and ELI5. Using two novel evaluation datasets spanning multiple languages, we test DeepSeek v3, Llama 4 Maverick, Qwen3-30B A3B, GPT-4o mini, and Llama 3.1 8B via LLM-as-Judge and human annotations. Our results show that simplification drastically degrades definition completeness by neglecting polysemy, increasing the risk of misunderstanding. Fine-tuning Llama 3.1 8B with Direct Preference Optimization substantially improves homonym response quality across all prompt types. These findings highlight the need to balance simplicity and completeness in educational NLP to ensure reliable, context-aware definitions for all learners. 

---
# Value-Based Large Language Model Agent Simulation for Mutual Evaluation of Trust and Interpersonal Closeness 

**Authors**: Yuki Sakamoto, Takahisa Uchida, Hiroshi Ishiguro  

**Link**: [PDF](https://arxiv.org/pdf/2507.11979)  

**Abstract**: Large language models (LLMs) have emerged as powerful tools for simulating complex social phenomena using human-like agents with specific traits. In human societies, value similarity is important for building trust and close relationships; however, it remains unexplored whether this principle holds true in artificial societies comprising LLM agents. Therefore, this study investigates the influence of value similarity on relationship-building among LLM agents through two experiments. First, in a preliminary experiment, we evaluated the controllability of values in LLMs to identify the most effective model and prompt design for controlling the values. Subsequently, in the main experiment, we generated pairs of LLM agents imbued with specific values and analyzed their mutual evaluations of trust and interpersonal closeness following a dialogue. The experiments were conducted in English and Japanese to investigate language dependence. The results confirmed that pairs of agents with higher value similarity exhibited greater mutual trust and interpersonal closeness. Our findings demonstrate that the LLM agent simulation serves as a valid testbed for social science theories, contributes to elucidating the mechanisms by which values influence relationship building, and provides a foundation for inspiring new theories and insights into the social sciences. 

---
# Graph Representations for Reading Comprehension Analysis using Large Language Model and Eye-Tracking Biomarker 

**Authors**: Yuhong Zhang, Jialu Li, Shilai Yang, Yuchen Xu, Gert Cauwenberghs, Tzyy-Ping Jung  

**Link**: [PDF](https://arxiv.org/pdf/2507.11972)  

**Abstract**: Reading comprehension is a fundamental skill in human cognitive development. With the advancement of Large Language Models (LLMs), there is a growing need to compare how humans and LLMs understand language across different contexts and apply this understanding to functional tasks such as inference, emotion interpretation, and information retrieval. Our previous work used LLMs and human biomarkers to study the reading comprehension process. The results showed that the biomarkers corresponding to words with high and low relevance to the inference target, as labeled by the LLMs, exhibited distinct patterns, particularly when validated using eye-tracking data. However, focusing solely on individual words limited the depth of understanding, which made the conclusions somewhat simplistic despite their potential significance. This study used an LLM-based AI agent to group words from a reading passage into nodes and edges, forming a graph-based text representation based on semantic meaning and question-oriented prompts. We then compare the distribution of eye fixations on important nodes and edges. Our findings indicate that LLMs exhibit high consistency in language understanding at the level of graph topological structure. These results build on our previous findings and offer insights into effective human-AI co-learning strategies. 

---
# Toxicity-Aware Few-Shot Prompting for Low-Resource Singlish Translation 

**Authors**: Ziyu Ge, Gabriel Chua, Leanne Tan, Roy Ka-Wei Lee  

**Link**: [PDF](https://arxiv.org/pdf/2507.11966)  

**Abstract**: As online communication increasingly incorporates under-represented languages and colloquial dialects, standard translation systems often fail to preserve local slang, code-mixing, and culturally embedded markers of harmful speech. Translating toxic content between low-resource language pairs poses additional challenges due to scarce parallel data and safety filters that sanitize offensive expressions. In this work, we propose a reproducible, two-stage framework for toxicity-preserving translation, demonstrated on a code-mixed Singlish safety corpus. First, we perform human-verified few-shot prompt engineering: we iteratively curate and rank annotator-selected Singlish-target examples to capture nuanced slang, tone, and toxicity. Second, we optimize model-prompt pairs by benchmarking several large language models using semantic similarity via direct and back-translation. Quantitative human evaluation confirms the effectiveness and efficiency of our pipeline. Beyond improving translation quality, our framework contributes to the safety of multicultural LLMs by supporting culturally sensitive moderation and benchmarking in low-resource contexts. By positioning Singlish as a testbed for inclusive NLP, we underscore the importance of preserving sociolinguistic nuance in real-world applications such as content moderation and regional platform governance. 

---
# PoTPTQ: A Two-step Power-of-Two Post-training for LLMs 

**Authors**: Xinyu Wang, Vahid Partovi Nia, Peng Lu, Jerry Huang, Xiao-Wen Chang, Boxing Chen, Yufei Cui  

**Link**: [PDF](https://arxiv.org/pdf/2507.11959)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable performance across various natural language processing (NLP) tasks. However, their deployment is challenging due to the substantial computational resources required. Power-of-two (PoT) quantization is a general tool to counteract this difficulty. Albeit previous works on PoT quantization can be efficiently dequantized on CPUs using fixed-point addition, it showed less effectiveness on GPUs. The reason is entanglement of the sign bit and sequential bit manipulations needed for dequantization. We propose a novel POT quantization framework for LLM weights that (i) outperforms state-of-the-art accuracy in extremely low-precision number formats, and (ii) enables faster inference through more efficient dequantization. To maintain the accuracy of the quantized model, we introduce a two-step post-training algorithm: (i) initialize the quantization scales with a robust starting point, and (ii) refine these scales using a minimal calibration set. The performance of our PoT post-training algorithm surpasses the current state-of-the-art in integer quantization, particularly at low precisions such as 2- and 3-bit formats. Our PoT quantization accelerates the dequantization step required for the floating point inference and leads to $3.67\times$ speed up on a NVIDIA V100, and $1.63\times$ on a NVIDIA RTX 4090, compared to uniform integer dequantization. 

---
# The benefits of query-based KGQA systems for complex and temporal questions in LLM era 

**Authors**: Artem Alekseev, Mikhail Chaichuk, Miron Butko, Alexander Panchenko, Elena Tutubalina, Oleg Somov  

**Link**: [PDF](https://arxiv.org/pdf/2507.11954)  

**Abstract**: Large language models excel in question-answering (QA) yet still struggle with multi-hop reasoning and temporal questions. Query-based knowledge graph QA (KGQA) offers a modular alternative by generating executable queries instead of direct answers. We explore multi-stage query-based framework for WikiData QA, proposing multi-stage approach that enhances performance on challenging multi-hop and temporal benchmarks. Through generalization and rejection studies, we evaluate robustness across multi-hop and temporal QA datasets. Additionally, we introduce a novel entity linking and predicate matching method using CoT reasoning. Our results demonstrate the potential of query-based multi-stage KGQA framework for improving multi-hop and temporal QA with small language models. Code and data: this https URL 

---
# IAM: Efficient Inference through Attention Mapping between Different-scale LLMs 

**Authors**: Yi Zhao, Zuchao Li, Hai Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2507.11953)  

**Abstract**: LLMs encounter significant challenges in resource consumption nowadays, especially with long contexts. Despite extensive efforts dedicate to enhancing inference efficiency, these methods primarily exploit internal sparsity within the models, without leveraging external information for optimization. We identify the high similarity of attention matrices across different-scale LLMs, which offers a novel perspective for optimization. We first conduct a comprehensive analysis of how to measure similarity, how to select mapping Layers and whether mapping is consistency. Based on these insights, we introduce the IAM framework, which achieves dual benefits of accelerated attention computation and reduced KV cache usage by performing attention mapping between small and large LLMs. Our experimental results demonstrate that IAM can accelerate prefill by 15% and reduce KV cache usage by 22.1% without appreciably sacrificing performance. Experiments on different series of models show the generalizability of IAM. Importantly, it is also orthogonal to many existing KV cache optimization methods, making it a versatile addition to the current toolkit for enhancing LLM efficiency. 

---
# DAC: A Dynamic Attention-aware Approach for Task-Agnostic Prompt Compression 

**Authors**: Yi Zhao, Zuchao Li, Hai Zhao, Baoyuan Qi, Guoming Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.11942)  

**Abstract**: Task-agnostic prompt compression leverages the redundancy in natural language to reduce computational overhead and enhance information density within prompts, especially in long-context scenarios. Existing methods predominantly rely on information entropy as the metric to compress lexical units, aiming to achieve minimal information loss. However, these approaches overlook two critical aspects: (i) the importance of attention-critical tokens at the algorithmic level, and (ii) shifts in information entropy during the compression process. Motivated by these challenges, we propose a dynamic attention-aware approach for task-agnostic prompt compression (DAC). This approach effectively integrates entropy and attention information, dynamically sensing entropy shifts during compression to achieve fine-grained prompt compression. Extensive experiments across various domains, including LongBench, GSM8K, and BBH, show that DAC consistently yields robust and substantial improvements across a diverse range of tasks and LLMs, offering compelling evidence of its efficacy. 

---
# BlockBPE: Parallel BPE Tokenization 

**Authors**: Amos You  

**Link**: [PDF](https://arxiv.org/pdf/2507.11941)  

**Abstract**: Tokenization is a critical preprocessing step in large language model pipelines, yet widely-used implementations remain CPU-bound and suboptimal for batch inference workflows on GPU. We present BlockBPE, a parallel GPU implementation of byte-pair encoding (BPE) that achieves near linear-time complexity under realistic assumptions and is optimized for high-throughput, batch inference. Unlike existing Rust-based tokenizers such as HuggingFace Tokenizers or OpenAI's tiktoken-whose runtimes are dominated by Regex pre-tokenization and exhibit $O(n \log n)$ runtime-BlockBPE eliminates the Regex pre-tokenization which leads to small loss in generation quality, but enables highly parallelized token merges within thread blocks, reducing overall complexity to $O(nd)$ where $d \ll n$. On high-batch inference workloads, BlockBPE achieves up to 2x higher throughput than tiktoken and 2.5x over HuggingFace Tokenizers. 

---
# POLYCHARTQA: Benchmarking Large Vision-Language Models with Multilingual Chart Question Answering 

**Authors**: Yichen Xu, Liangyu Chen, Liang Zhang, Wenxuan Wang, Qin Jin  

**Link**: [PDF](https://arxiv.org/pdf/2507.11939)  

**Abstract**: Charts are a universally adopted medium for interpreting and communicating data. However, existing chart understanding benchmarks are predominantly English-centric, limiting their accessibility and applicability to global audiences. In this paper, we present PolyChartQA, the first large-scale multilingual chart question answering benchmark covering 22,606 charts and 26,151 question-answering pairs across 10 diverse languages. PolyChartQA is built using a decoupled pipeline that separates chart data from rendering code, allowing multilingual charts to be flexibly generated by simply translating the data and reusing the code. We leverage state-of-the-art LLM-based translation and enforce rigorous quality control in the pipeline to ensure the linguistic and semantic consistency of the generated multilingual charts. PolyChartQA facilitates systematic evaluation of multilingual chart understanding. Experiments on both open- and closed-source large vision-language models reveal a significant performance gap between English and other languages, especially low-resource ones with non-Latin scripts. This benchmark lays a foundation for advancing globally inclusive vision-language models. 

---
# A Survey of Deep Learning for Geometry Problem Solving 

**Authors**: Jianzhe Ma, Wenxuan Wang, Qin Jin  

**Link**: [PDF](https://arxiv.org/pdf/2507.11936)  

**Abstract**: Geometry problem solving is a key area of mathematical reasoning, which is widely involved in many important fields such as education, mathematical ability assessment of artificial intelligence, and multimodal ability assessment. In recent years, the rapid development of deep learning technology, especially the rise of multimodal large language models, has triggered a widespread research boom. This paper provides a survey of the applications of deep learning in geometry problem solving, including (i) a comprehensive summary of the relevant tasks in geometry problem solving; (ii) a thorough review of related deep learning methods; (iii) a detailed analysis of evaluation metrics and methods; and (iv) a critical discussion of the current challenges and future directions that can be explored. Our goal is to provide a comprehensive and practical reference of deep learning for geometry problem solving to promote further developments in this field. We create a continuously updated list of papers on GitHub: this https URL. 

---
# Marco-Bench-MIF: On Multilingual Instruction-Following Capability of Large Language Models 

**Authors**: Bo Zeng, Chenyang Lyu, Sinuo Liu, Mingyan Zeng, Minghao Wu, Xuanfan Ni, Tianqi Shi, Yu Zhao, Yefeng Liu, Chenyu Zhu, Ruizhe Li, Jiahui Geng, Qing Li, Yu Tong, Longyue Wang, Weihua Luo, Kaifu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.11882)  

**Abstract**: Instruction-following capability has become a major ability to be evaluated for Large Language Models (LLMs). However, existing datasets, such as IFEval, are either predominantly monolingual and centered on English or simply machine translated to other languages, limiting their applicability in multilingual contexts. In this paper, we present an carefully-curated extension of IFEval to a localized multilingual version named Marco-Bench-MIF, covering 30 languages with varying levels of localization. Our benchmark addresses linguistic constraints (e.g., modifying capitalization requirements for Chinese) and cultural references (e.g., substituting region-specific company names in prompts) via a hybrid pipeline combining translation with verification. Through comprehensive evaluation of 20+ LLMs on our Marco-Bench-MIF, we found that: (1) 25-35% accuracy gap between high/low-resource languages, (2) model scales largely impact performance by 45-60% yet persists script-specific challenges, and (3) machine-translated data underestimates accuracy by7-22% versus localized data. Our analysis identifies challenges in multilingual instruction following, including keyword consistency preservation and compositional constraint adherence across languages. Our Marco-Bench-MIF is available at this https URL. 

---
# LLMs Encode Harmfulness and Refusal Separately 

**Authors**: Jiachen Zhao, Jing Huang, Zhengxuan Wu, David Bau, Weiyan Shi  

**Link**: [PDF](https://arxiv.org/pdf/2507.11878)  

**Abstract**: LLMs are trained to refuse harmful instructions, but do they truly understand harmfulness beyond just refusing? Prior work has shown that LLMs' refusal behaviors can be mediated by a one-dimensional subspace, i.e., a refusal direction. In this work, we identify a new dimension to analyze safety mechanisms in LLMs, i.e., harmfulness, which is encoded internally as a separate concept from refusal. There exists a harmfulness direction that is distinct from the refusal direction. As causal evidence, steering along the harmfulness direction can lead LLMs to interpret harmless instructions as harmful, but steering along the refusal direction tends to elicit refusal responses directly without reversing the model's judgment on harmfulness. Furthermore, using our identified harmfulness concept, we find that certain jailbreak methods work by reducing the refusal signals without reversing the model's internal belief of harmfulness. We also find that adversarially finetuning models to accept harmful instructions has minimal impact on the model's internal belief of harmfulness. These insights lead to a practical safety application: The model's latent harmfulness representation can serve as an intrinsic safeguard (Latent Guard) for detecting unsafe inputs and reducing over-refusals that is robust to finetuning attacks. For instance, our Latent Guard achieves performance comparable to or better than Llama Guard 3 8B, a dedicated finetuned safeguard model, across different jailbreak methods. Our findings suggest that LLMs' internal understanding of harmfulness is more robust than their refusal decision to diverse input instructions, offering a new perspective to study AI safety 

---
# DualReward: A Dynamic Reinforcement Learning Framework for Cloze Tests Distractor Generation 

**Authors**: Tianyou Huang, Xinglu Chen, Jingshen Zhang, Xinying Qiu, Ruiying Niu  

**Link**: [PDF](https://arxiv.org/pdf/2507.11875)  

**Abstract**: This paper introduces DualReward, a novel reinforcement learning framework for automatic distractor generation in cloze tests. Unlike conventional approaches that rely primarily on supervised learning or static generative models, our method employs a dual reward structure with adaptive scaling that differentiates between human-created gold standard distractors and model-generated candidates. The framework dynamically adjusts reward signal intensity based on model performance and confidence. We evaluate our approach on both passage-level (CLOTH-F) and sentence-level (MCQ) cloze test datasets, demonstrating consistent improvements over state-of-the-art baselines. Experimental results show that our adaptive reward scaling mechanism provides modest but consistent benefits on homogeneous datasets (CLOTH-F) and more substantial improvements (3.48-3.86% in P@1) on diverse, cross-domain data (MCQ), suggesting its particular effectiveness for handling varied question types and domains. Our work offers a flexible framework that effectively balances learning from reliable human examples while exploring novel, high-quality distractors for automated test generation. 

---
# COLA-GEC: A Bidirectional Framework for Enhancing Grammatical Acceptability and Error Correction 

**Authors**: Xiangyu Yang, Xinying Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2507.11867)  

**Abstract**: Grammatical Error Correction (GEC) and grammatical acceptability judgment (COLA) are core tasks in natural language processing, sharing foundational grammatical knowledge yet typically evolving independently. This paper introduces COLA-GEC, a novel bidirectional framework that enhances both tasks through mutual knowledge transfer. First, we augment grammatical acceptability models using GEC datasets, significantly improving their performance across multiple languages. Second, we integrate grammatical acceptability signals into GEC model training via a dynamic loss function, effectively guiding corrections toward grammatically acceptable outputs. Our approach achieves state-of-the-art results on several multilingual benchmarks. Comprehensive error analysis highlights remaining challenges, particularly in punctuation error correction, providing insights for future improvements in grammatical modeling. 

---
# Cross-Domain Transfer and Few-Shot Learning for Personal Identifiable Information Recognition 

**Authors**: Junhong Ye, Xu Yuan, Xinying Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2507.11862)  

**Abstract**: Accurate recognition of personally identifiable information (PII) is central to automated text anonymization. This paper investigates the effectiveness of cross-domain model transfer, multi-domain data fusion, and sample-efficient learning for PII recognition. Using annotated corpora from healthcare (I2B2), legal (TAB), and biography (Wikipedia), we evaluate models across four dimensions: in-domain performance, cross-domain transferability, fusion, and few-shot learning. Results show legal-domain data transfers well to biographical texts, while medical domains resist incoming transfer. Fusion benefits are domain-specific, and high-quality recognition is achievable with only 10% of training data in low-specialization domains. 

---
# Your LLM Knows the Future: Uncovering Its Multi-Token Prediction Potential 

**Authors**: Mohammad Samragh, Arnav Kundu, David Harrison, Kumari Nishu, Devang Naik, Minsik Cho, Mehrdad Farajtabar  

**Link**: [PDF](https://arxiv.org/pdf/2507.11851)  

**Abstract**: Autoregressive language models are constrained by their inherently sequential nature, generating one token at a time. This paradigm limits inference speed and parallelism, especially during later stages of generation when the direction and semantics of text are relatively certain. In this work, we propose a novel framework that leverages the inherent knowledge of vanilla autoregressive language models about future tokens, combining techniques to realize this potential and enable simultaneous prediction of multiple subsequent tokens. Our approach introduces several key innovations: (1) a masked-input formulation where multiple future tokens are jointly predicted from a common prefix; (2) a gated LoRA formulation that preserves the original LLM's functionality, while equipping it for multi-token prediction; (3) a lightweight, learnable sampler module that generates coherent sequences from the predicted future tokens; (4) a set of auxiliary training losses, including a consistency loss, to enhance the coherence and accuracy of jointly generated tokens; and (5) a speculative generation strategy that expands tokens quadratically in the future while maintaining high fidelity. Our method achieves significant speedups through supervised fine-tuning on pretrained models. For example, it generates code and math nearly 5x faster, and improves general chat and knowledge tasks by almost 2.5x. These gains come without any loss in quality. 

---
# ILID: Native Script Language Identification for Indian Languages 

**Authors**: Yash Ingle, Pruthwik Mishra  

**Link**: [PDF](https://arxiv.org/pdf/2507.11832)  

**Abstract**: The language identification task is a crucial fundamental step in NLP. Often it serves as a pre-processing step for widely used NLP applications such as multilingual machine translation, information retrieval, question and answering, and text summarization. The core challenge of language identification lies in distinguishing languages in noisy, short, and code-mixed environments. This becomes even harder in case of diverse Indian languages that exhibit lexical and phonetic similarities, but have distinct differences. Many Indian languages share the same script making the task even more challenging. In this paper, we release a dataset of 230K sentences consisting of English and all 22 official Indian languages labeled with their language identifiers where data in most languages are newly created. We also develop and release robust baseline models using state-of-the-art approaches in machine learning and deep learning that can aid the research in this field. Our baseline models are comparable to the state-of-the-art models for the language identification task. 

---
# Tracing Facts or just Copies? A critical investigation of the Competitions of Mechanisms in Large Language Models 

**Authors**: Dante Campregher, Yanxu Chen, Sander Hoffman, Maria Heuss  

**Link**: [PDF](https://arxiv.org/pdf/2507.11809)  

**Abstract**: This paper presents a reproducibility study examining how Large Language Models (LLMs) manage competing factual and counterfactual information, focusing on the role of attention heads in this process. We attempt to reproduce and reconcile findings from three recent studies by Ortu et al., Yu, Merullo, and Pavlick and McDougall et al. that investigate the competition between model-learned facts and contradictory context information through Mechanistic Interpretability tools. Our study specifically examines the relationship between attention head strength and factual output ratios, evaluates competing hypotheses about attention heads' suppression mechanisms, and investigates the domain specificity of these attention patterns. Our findings suggest that attention heads promoting factual output do so via general copy suppression rather than selective counterfactual suppression, as strengthening them can also inhibit correct facts. Additionally, we show that attention head behavior is domain-dependent, with larger models exhibiting more specialized and category-sensitive patterns. 

---
# AI Wizards at CheckThat! 2025: Enhancing Transformer-Based Embeddings with Sentiment for Subjectivity Detection in News Articles 

**Authors**: Matteo Fasulo, Luca Babboni, Luca Tedeschini  

**Link**: [PDF](https://arxiv.org/pdf/2507.11764)  

**Abstract**: This paper presents AI Wizards' participation in the CLEF 2025 CheckThat! Lab Task 1: Subjectivity Detection in News Articles, classifying sentences as subjective/objective in monolingual, multilingual, and zero-shot settings. Training/development datasets were provided for Arabic, German, English, Italian, and Bulgarian; final evaluation included additional unseen languages (e.g., Greek, Romanian, Polish, Ukrainian) to assess generalization. Our primary strategy enhanced transformer-based classifiers by integrating sentiment scores, derived from an auxiliary model, with sentence representations, aiming to improve upon standard fine-tuning. We explored this sentiment-augmented architecture with mDeBERTaV3-base, ModernBERT-base (English), and Llama3.2-1B. To address class imbalance, prevalent across languages, we employed decision threshold calibration optimized on the development set. Our experiments show sentiment feature integration significantly boosts performance, especially subjective F1 score. This framework led to high rankings, notably 1st for Greek (Macro F1 = 0.51). 

---
# CRABS: A syntactic-semantic pincer strategy for bounding LLM interpretation of Python notebooks 

**Authors**: Meng Li, Timothy M. McPhillips, Dingmin Wang, Shin-Rong Tsai, Bertram Ludäscher  

**Link**: [PDF](https://arxiv.org/pdf/2507.11742)  

**Abstract**: Recognizing the information flows and operations comprising data science and machine learning Python notebooks is critical for evaluating, reusing, and adapting notebooks for new tasks. Investigating a notebook via re-execution often is impractical due to the challenges of resolving data and software dependencies. While Large Language Models (LLMs) pre-trained on large codebases have demonstrated effectiveness in understanding code without running it, we observe that they fail to understand some realistic notebooks due to hallucinations and long-context challenges. To address these issues, we propose a notebook understanding task yielding an information flow graph and corresponding cell execution dependency graph for a notebook, and demonstrate the effectiveness of a pincer strategy that uses limited syntactic analysis to assist full comprehension of the notebook using an LLM. Our Capture and Resolve Assisted Bounding Strategy (CRABS) employs shallow syntactic parsing and analysis of the abstract syntax tree (AST) to capture the correct interpretation of a notebook between lower and upper estimates of the inter-cell I/O sets, then uses an LLM to resolve remaining ambiguities via cell-by-cell zero-shot learning, thereby identifying the true data inputs and outputs of each cell. We evaluate and demonstrate the effectiveness of our approach using an annotated dataset of 50 representative, highly up-voted Kaggle notebooks that together represent 3454 actual cell inputs and outputs. The LLM correctly resolves 1397 of 1425 (98%) ambiguities left by analyzing the syntactic structure of these notebooks. Across 50 notebooks, CRABS achieves average F1 scores of 98% identifying cell-to-cell information flows and 99% identifying transitive cell execution dependencies. 

---
# ExpliCIT-QA: Explainable Code-Based Image Table Question Answering 

**Authors**: Maximiliano Hormazábal Lagos, Álvaro Bueno Sáez, Pedro Alonso Doval, Jorge Alcalde Vesteiro, Héctor Cerezo-Costas  

**Link**: [PDF](https://arxiv.org/pdf/2507.11694)  

**Abstract**: We present ExpliCIT-QA, a system that extends our previous MRT approach for tabular question answering into a multimodal pipeline capable of handling complex table images and providing explainable answers. ExpliCIT-QA follows a modular design, consisting of: (1) Multimodal Table Understanding, which uses a Chain-of-Thought approach to extract and transform content from table images; (2) Language-based Reasoning, where a step-by-step explanation in natural language is generated to solve the problem; (3) Automatic Code Generation, where Python/Pandas scripts are created based on the reasoning steps, with feedback for handling errors; (4) Code Execution to compute the final answer; and (5) Natural Language Explanation that describes how the answer was computed. The system is built for transparency and auditability: all intermediate outputs, parsed tables, reasoning steps, generated code, and final answers are available for inspection. This strategy works towards closing the explainability gap in end-to-end TableVQA systems. We evaluated ExpliCIT-QA on the TableVQA-Bench benchmark, comparing it with existing baselines. We demonstrated improvements in interpretability and transparency, which open the door for applications in sensitive domains like finance and healthcare where auditing results are critical. 

---
# Partitioner Guided Modal Learning Framework 

**Authors**: Guimin Hu, Yi Xin, Lijie Hu, Zhihong Zhu, Hasti Seifi  

**Link**: [PDF](https://arxiv.org/pdf/2507.11661)  

**Abstract**: Multimodal learning benefits from multiple modal information, and each learned modal representations can be divided into uni-modal that can be learned from uni-modal training and paired-modal features that can be learned from cross-modal interaction. Building on this perspective, we propose a partitioner-guided modal learning framework, PgM, which consists of the modal partitioner, uni-modal learner, paired-modal learner, and uni-paired modal decoder. Modal partitioner segments the learned modal representation into uni-modal and paired-modal features. Modal learner incorporates two dedicated components for uni-modal and paired-modal learning. Uni-paired modal decoder reconstructs modal representation based on uni-modal and paired-modal features. PgM offers three key benefits: 1) thorough learning of uni-modal and paired-modal features, 2) flexible distribution adjustment for uni-modal and paired-modal representations to suit diverse downstream tasks, and 3) different learning rates across modalities and partitions. Extensive experiments demonstrate the effectiveness of PgM across four multimodal tasks and further highlight its transferability to existing models. Additionally, we visualize the distribution of uni-modal and paired-modal features across modalities and tasks, offering insights into their respective contributions. 

---
# Cross-lingual Few-shot Learning for Persian Sentiment Analysis with Incremental Adaptation 

**Authors**: Farideh Majidi, Ziaeddin Beheshtifard  

**Link**: [PDF](https://arxiv.org/pdf/2507.11634)  

**Abstract**: This research examines cross-lingual sentiment analysis using few-shot learning and incremental learning methods in Persian. The main objective is to develop a model capable of performing sentiment analysis in Persian using limited data, while getting prior knowledge from high-resource languages. To achieve this, three pre-trained multilingual models (XLM-RoBERTa, mDeBERTa, and DistilBERT) were employed, which were fine-tuned using few-shot and incremental learning approaches on small samples of Persian data from diverse sources, including X, Instagram, Digikala, Snappfood, and Taaghche. This variety enabled the models to learn from a broad range of contexts. Experimental results show that the mDeBERTa and XLM-RoBERTa achieved high performances, reaching 96% accuracy on Persian sentiment analysis. These findings highlight the effectiveness of combining few-shot learning and incremental learning with multilingual pre-trained models. 

---
# MapIQ: Benchmarking Multimodal Large Language Models for Map Question Answering 

**Authors**: Varun Srivastava, Fan Lei, Srija Mukhopadhyay, Vivek Gupta, Ross Maciejewski  

**Link**: [PDF](https://arxiv.org/pdf/2507.11625)  

**Abstract**: Recent advancements in multimodal large language models (MLLMs) have driven researchers to explore how well these models read data visualizations, e.g., bar charts, scatter plots. More recently, attention has shifted to visual question answering with maps (Map-VQA). However, Map-VQA research has primarily focused on choropleth maps, which cover only a limited range of thematic categories and visual analytical tasks. To address these gaps, we introduce MapIQ, a benchmark dataset comprising 14,706 question-answer pairs across three map types: choropleth maps, cartograms, and proportional symbol maps spanning topics from six distinct themes (e.g., housing, crime). We evaluate multiple MLLMs using six visual analytical tasks, comparing their performance against one another and a human baseline. An additional experiment examining the impact of map design changes (e.g., altered color schemes, modified legend designs, and removal of map elements) provides insights into the robustness and sensitivity of MLLMs, their reliance on internal geographic knowledge, and potential avenues for improving Map-VQA performance. 

---
# Subjective Evaluation Profile Analysis of Science Fiction Short Stories and its Critical-Theoretical Significance 

**Authors**: Kazuyoshi Otsuka  

**Link**: [PDF](https://arxiv.org/pdf/2507.11582)  

**Abstract**: This study positions large language models (LLMs) as "subjective literary critics" to explore aesthetic preferences and evaluation patterns in literary assessment. Ten Japanese science fiction short stories were translated into English and evaluated by six state-of-the-art LLMs across seven independent sessions. Principal component analysis and clustering techniques revealed significant variations in evaluation consistency ({\alpha} ranging from 1.00 to 0.35) and five distinct evaluation patterns. Additionally, evaluation variance across stories differed by up to 4.5-fold, with TF-IDF analysis confirming distinctive evaluation vocabularies for each model. Our seven-session within-day protocol using an original Science Fiction corpus strategically minimizes external biases, allowing us to observe implicit value systems shaped by RLHF and their influence on literary judgment. These findings suggest that LLMs may possess individual evaluation characteristics similar to human critical schools, rather than functioning as neutral benchmarkers. 

---
# Developing Visual Augmented Q&A System using Scalable Vision Embedding Retrieval & Late Interaction Re-ranker 

**Authors**: Rachna Saxena, Abhijeet Kumar, Suresh Shanmugam  

**Link**: [PDF](https://arxiv.org/pdf/2507.12378)  

**Abstract**: Traditional information extraction systems face challenges with text only language models as it does not consider infographics (visual elements of information) such as tables, charts, images etc. often used to convey complex information to readers. Multimodal LLM (MLLM) face challenges of finding needle in the haystack problem i.e., either longer context length or substantial number of documents as search space. Late interaction mechanism over visual language models has shown state of the art performance in retrieval-based vision augmented Q&A tasks. There are yet few challenges using it for RAG based multi-modal Q&A. Firstly, many popular and widely adopted vector databases do not support native multi-vector retrieval. Secondly, late interaction requires computation which inflates space footprint and can hinder enterprise adoption. Lastly, the current state of late interaction mechanism does not leverage the approximate neighbor search indexing methods for large speed ups in retrieval process. This paper explores a pragmatic approach to make vision retrieval process scalable and efficient without compromising on performance quality. We propose multi-step custom implementation utilizing widely adopted hybrid search (metadata & embedding) and state of the art late interaction re-ranker to retrieve best matching pages. Finally, MLLM are prompted as reader to generate answers from contextualized best matching pages. Through experiments, we observe that the proposed design is scalable (significant speed up) and stable (without degrading performance quality), hence can be used as production systems at enterprises. 

---
# Nonlinear Concept Erasure: a Density Matching Approach 

**Authors**: Antoine Saillenfest, Pirmin Lemberger  

**Link**: [PDF](https://arxiv.org/pdf/2507.12341)  

**Abstract**: Ensuring that neural models used in real-world applications cannot infer sensitive information, such as demographic attributes like gender or race, from text representations is a critical challenge when fairness is a concern. We address this issue through concept erasure, a process that removes information related to a specific concept from distributed representations while preserving as much of the remaining semantic information as possible. Our approach involves learning an orthogonal projection in the embedding space, designed to make the class-conditional feature distributions of the discrete concept to erase indistinguishable after projection. By adjusting the rank of the projector, we control the extent of information removal, while its orthogonality ensures strict preservation of the local structure of the embeddings. Our method, termed $\overline{\mathrm{L}}$EOPARD, achieves state-of-the-art performance in nonlinear erasure of a discrete attribute on classic natural language processing benchmarks. Furthermore, we demonstrate that $\overline{\mathrm{L}}$EOPARD effectively mitigates bias in deep nonlinear classifiers, thereby promoting fairness. 

---
# MERA Code: A Unified Framework for Evaluating Code Generation Across Tasks 

**Authors**: Artem Chervyakov, Alexander Kharitonov, Pavel Zadorozhny, Adamenko Pavel, Rodion Levichev, Dmitrii Vorobev, Dmitrii Salikhov, Aidar Valeev, Alena Pestova, Maria Dziuba, Ilseyar Alimova, Artem Zavgorodnev, Aleksandr Medvedev, Stanislav Moiseev, Elena Bruches, Daniil Grebenkin, Roman Derunets, Vikulov Vladimir, Anton Emelyanov, Dmitrii Babaev, Vladimir V. Ivanov, Valentin Malykh, Alena Fenogenova  

**Link**: [PDF](https://arxiv.org/pdf/2507.12284)  

**Abstract**: Advancements in LLMs have enhanced task automation in software engineering; however, current evaluations primarily focus on natural language tasks, overlooking code quality. Most benchmarks prioritize high-level reasoning over executable code and real-world performance, leaving gaps in understanding true capabilities and risks associated with these models in production. To address this issue, we propose MERA Code, a new addition to the MERA benchmark family, specifically focused on evaluating code for the latest code generation LLMs in Russian. This benchmark includes 11 evaluation tasks that span 8 programming languages. Our proposed evaluation methodology features a taxonomy that outlines the practical coding skills necessary for models to complete these tasks. The benchmark comprises an open-source codebase for users to conduct MERA assessments, a scoring system compatible with various programming environments, and a platform featuring a leaderboard and submission system. We evaluate open LLMs and frontier API models, analyzing their limitations in terms of practical coding tasks in non-English languages. We are publicly releasing MERA to guide future research, anticipate groundbreaking features in model development, and standardize evaluation procedures. 

---
# RUMAA: Repeat-Aware Unified Music Audio Analysis for Score-Performance Alignment, Transcription, and Mistake Detection 

**Authors**: Sungkyun Chang, Simon Dixon, Emmanouil Benetos  

**Link**: [PDF](https://arxiv.org/pdf/2507.12175)  

**Abstract**: This study introduces RUMAA, a transformer-based framework for music performance analysis that unifies score-to-performance alignment, score-informed transcription, and mistake detection in a near end-to-end manner. Unlike prior methods addressing these tasks separately, RUMAA integrates them using pre-trained score and audio encoders and a novel tri-stream decoder capturing task interdependencies through proxy tasks. It aligns human-readable MusicXML scores with repeat symbols to full-length performance audio, overcoming traditional MIDI-based methods that rely on manually unfolded score-MIDI data with pre-specified repeat structures. RUMAA matches state-of-the-art alignment methods on non-repeated scores and outperforms them on scores with repeats in a public piano music dataset, while also delivering promising transcription and mistake detection results. 

---
# RiemannLoRA: A Unified Riemannian Framework for Ambiguity-Free LoRA Optimization 

**Authors**: Vladimir Bogachev, Vladimir Aletov, Alexander Molozhavenko, Denis Bobkov, Vera Soboleva, Aibek Alanov, Maxim Rakhuba  

**Link**: [PDF](https://arxiv.org/pdf/2507.12142)  

**Abstract**: Low-Rank Adaptation (LoRA) has become a widely adopted standard for parameter-efficient fine-tuning of large language models (LLMs), significantly reducing memory and computational demands. However, challenges remain, including finding optimal initialization strategies or mitigating overparametrization in low-rank matrix factorization. In this work, we propose a novel approach that addresses both of the challenges simultaneously within a unified framework. Our method treats a set of fixed-rank LoRA matrices as a smooth manifold. Considering adapters as elements on this manifold removes overparametrization, while determining the direction of the fastest loss decrease along the manifold provides initialization. Special care is taken to obtain numerically stable and computationally efficient implementation of our method, using best practices from numerical linear algebra and Riemannian optimization. Experimental results on LLM and diffusion model architectures demonstrate that RiemannLoRA consistently improves both convergence speed and final performance over standard LoRA and its state-of-the-art modifications. 

---
# Simulated Language Acquisition in a Biologically Realistic Model of the Brain 

**Authors**: Daniel Mitropolsky, Christos Papadimitriou  

**Link**: [PDF](https://arxiv.org/pdf/2507.11788)  

**Abstract**: Despite tremendous progress in neuroscience, we do not have a compelling narrative for the precise way whereby the spiking of neurons in our brain results in high-level cognitive phenomena such as planning and language. We introduce a simple mathematical formulation of six basic and broadly accepted principles of neuroscience: excitatory neurons, brain areas, random synapses, Hebbian plasticity, local inhibition, and inter-area inhibition. We implement a simulated neuromorphic system based on this formalism, which is capable of basic language acquisition: Starting from a tabula rasa, the system learns, in any language, the semantics of words, their syntactic role (verb versus noun), and the word order of the language, including the ability to generate novel sentences, through the exposure to a modest number of grounded sentences in the same language. We discuss several possible extensions and implications of this result. 

---
# MetaLint: Generalizable Idiomatic Code Quality Analysis through Instruction-Following and Easy-to-Hard Generalization 

**Authors**: Atharva Naik, Lawanya Baghel, Dhakshin Govindarajan, Darsh Agrawal, Daniel Fried, Carolyn Rose  

**Link**: [PDF](https://arxiv.org/pdf/2507.11687)  

**Abstract**: Large Language Models, though successful in code generation, struggle with code quality analysis because they are limited by static training data and can't easily adapt to evolving best practices. We introduce MetaLint, a new instruction-following framework that formulates code quality analysis as the task of detecting and fixing problematic semantic code fragments or code idioms based on high-level specifications. Unlike conventional approaches that train models on static, rule-based data, MetaLint employs instruction tuning on synthetic linter-generated data to support easy-to-hard generalization, enabling models to adapt to novel or complex code patterns without retraining. To evaluate this, we construct a benchmark of challenging idioms inspired by real-world coding standards such as Python Enhancement Proposals (PEPs) and assess whether MetaLint-trained models reason adaptively or simply memorize. Our results show that MetaLint improves generalization to unseen PEP idioms, achieving a 70.37% F-score on idiom detection with the highest recall (70.43%) among all evaluated models. It also achieves 26.73% on localization, competitive for its 4B parameter size and comparable to larger state-of-the-art models like o3-mini, highlighting its potential for future-proof code quality analysis. 

---
# Let's Think in Two Steps: Mitigating Agreement Bias in MLLMs with Self-Grounded Verification 

**Authors**: Moises Andrade, Joonhyuk Cha, Brandon Ho, Vriksha Srihari, Karmesh Yadav, Zsolt Kira  

**Link**: [PDF](https://arxiv.org/pdf/2507.11662)  

**Abstract**: Verifiers -- functions assigning rewards to agent behavior -- have been key for AI progress in domains like math and board games. However, extending these gains to domains without clear-cut success criteria (e.g.,computer use) remains a challenge: while humans can recognize suitable outcomes, translating this intuition into scalable rules is non-trivial. Multimodal Large Language Models(MLLMs) emerge as a promising solution, given their world knowledge, human-preference alignment, and reasoning skills. We evaluate MLLMs as verifiers of agent trajectories across web navigation, computer use, and robotic manipulation, and identify a critical limitation: agreement bias, a strong tendency for MLLMs to favor information in their context window, often generating chains of thought to rationalize flawed behavior. This bias is pervasive across models, resilient to test-time scaling, and can impact several methods using MLLMs as evaluators (e.g.,data filtering). Notably, it occurs despite MLLMs showing strong, human-aligned priors on desired behavior. To address this, we propose Self-Grounded Verification (SGV), a lightweight method that enables more effective use of MLLMs' knowledge and reasoning by harnessing their own sampling mechanisms via unconditional and conditional generation. SGV operates in two steps: first, the MLLM is elicited to retrieve broad priors about task completion, independent of the data under evaluation. Then, conditioned on self-generated priors, it reasons over and evaluates a candidate trajectory. Enhanced with SGV, MLLM verifiers show gains of up to 20 points in accuracy and failure detection rates, and can perform real-time supervision of heterogeneous agents, boosting task completion of a GUI specialist in OSWorld, a diffusion policy in robomimic, and a ReAct agent in VisualWebArena -- setting a new state of the art on the benchmark, surpassing the previous best by 48%. 

---
# Jailbreak-Tuning: Models Efficiently Learn Jailbreak Susceptibility 

**Authors**: Brendan Murphy, Dillon Bowen, Shahrad Mohammadzadeh, Julius Broomfield, Adam Gleave, Kellin Pelrine  

**Link**: [PDF](https://arxiv.org/pdf/2507.11630)  

**Abstract**: AI systems are rapidly advancing in capability, and frontier model developers broadly acknowledge the need for safeguards against serious misuse. However, this paper demonstrates that fine-tuning, whether via open weights or closed fine-tuning APIs, can produce helpful-only models. In contrast to prior work which is blocked by modern moderation systems or achieved only partial removal of safeguards or degraded output quality, our jailbreak-tuning method teaches models to generate detailed, high-quality responses to arbitrary harmful requests. For example, OpenAI, Google, and Anthropic models will fully comply with requests for CBRN assistance, executing cyberattacks, and other criminal activity. We further show that backdoors can increase not only the stealth but also the severity of attacks, while stronger jailbreak prompts become even more effective in fine-tuning attacks, linking attack and potentially defenses in the input and weight spaces. Not only are these models vulnerable, more recent ones also appear to be becoming even more vulnerable to these attacks, underscoring the urgent need for tamper-resistant safeguards. Until such safeguards are discovered, companies and policymakers should view the release of any fine-tunable model as simultaneously releasing its evil twin: equally capable as the original model, and usable for any malicious purpose within its capabilities. 

---
# Fairness Is Not Enough: Auditing Competence and Intersectional Bias in AI-powered Resume Screening 

**Authors**: Kevin T Webster  

**Link**: [PDF](https://arxiv.org/pdf/2507.11548)  

**Abstract**: The increasing use of generative AI for resume screening is predicated on the assumption that it offers an unbiased alternative to biased human decision-making. However, this belief fails to address a critical question: are these AI systems fundamentally competent at the evaluative tasks they are meant to perform? This study investigates the question of competence through a two-part audit of eight major AI platforms. Experiment 1 confirmed complex, contextual racial and gender biases, with some models penalizing candidates merely for the presence of demographic signals. Experiment 2, which evaluated core competence, provided a critical insight: some models that appeared unbiased were, in fact, incapable of performing a substantive evaluation, relying instead on superficial keyword matching. This paper introduces the "Illusion of Neutrality" to describe this phenomenon, where an apparent lack of bias is merely a symptom of a model's inability to make meaningful judgments. This study recommends that organizations and regulators adopt a dual-validation framework, auditing AI hiring tools for both demographic bias and demonstrable competence to ensure they are both equitable and effective. 

---
