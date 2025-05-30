# TRUST: An LLM-Based Dialogue System for Trauma Understanding and Structured Assessments 

**Authors**: Sichang Tu, Abigail Powers, Stephen Doogan, Jinho D. Choi  

**Link**: [PDF](https://arxiv.org/pdf/2504.21851)  

**Abstract**: Objectives: While Large Language Models (LLMs) have been widely used to assist clinicians and support patients, no existing work has explored dialogue systems for standard diagnostic interviews and assessments. This study aims to bridge the gap in mental healthcare accessibility by developing an LLM-powered dialogue system that replicates clinician behavior. Materials and Methods: We introduce TRUST, a framework of cooperative LLM modules capable of conducting formal diagnostic interviews and assessments for Post-Traumatic Stress Disorder (PTSD). To guide the generation of appropriate clinical responses, we propose a Dialogue Acts schema specifically designed for clinical interviews. Additionally, we develop a patient simulation approach based on real-life interview transcripts to replace time-consuming and costly manual testing by clinicians. Results: A comprehensive set of evaluation metrics is designed to assess the dialogue system from both the agent and patient simulation perspectives. Expert evaluations by conversation and clinical specialists show that TRUST performs comparably to real-life clinical interviews. Discussion: Our system performs at the level of average clinicians, with room for future enhancements in communication styles and response appropriateness. Conclusions: Our TRUST framework shows its potential to facilitate mental healthcare availability. 

---
# DeepSeek-Prover-V2: Advancing Formal Mathematical Reasoning via Reinforcement Learning for Subgoal Decomposition 

**Authors**: Z.Z. Ren, Zhihong Shao, Junxiao Song, Huajian Xin, Haocheng Wang, Wanjia Zhao, Liyue Zhang, Zhe Fu, Qihao Zhu, Dejian Yang, Z.F. Wu, Zhibin Gou, Shirong Ma, Hongxuan Tang, Yuxuan Liu, Wenjun Gao, Daya Guo, Chong Ruan  

**Link**: [PDF](https://arxiv.org/pdf/2504.21801)  

**Abstract**: We introduce DeepSeek-Prover-V2, an open-source large language model designed for formal theorem proving in Lean 4, with initialization data collected through a recursive theorem proving pipeline powered by DeepSeek-V3. The cold-start training procedure begins by prompting DeepSeek-V3 to decompose complex problems into a series of subgoals. The proofs of resolved subgoals are synthesized into a chain-of-thought process, combined with DeepSeek-V3's step-by-step reasoning, to create an initial cold start for reinforcement learning. This process enables us to integrate both informal and formal mathematical reasoning into a unified model. The resulting model, DeepSeek-Prover-V2-671B, achieves state-of-the-art performance in neural theorem proving, reaching 88.9% pass ratio on the MiniF2F-test and solving 49 out of 658 problems from PutnamBench. In addition to standard benchmarks, we introduce ProverBench, a collection of 325 formalized problems, to enrich our evaluation, including 15 selected problems from the recent AIME competitions (years 24-25). Further evaluation on these 15 AIME problems shows that the model successfully solves 6 of them. In comparison, DeepSeek-V3 solves 8 of these problems using majority voting, highlighting that the gap between formal and informal mathematical reasoning in large language models is substantially narrowing. 

---
# How Real Are Synthetic Therapy Conversations? Evaluating Fidelity in Prolonged Exposure Dialogues 

**Authors**: Suhas BN, Dominik Mattioli, Saeed Abdullah, Rosa I. Arriaga, Chris W. Wiese, Andrew M. Sherrill  

**Link**: [PDF](https://arxiv.org/pdf/2504.21800)  

**Abstract**: The growing adoption of synthetic data in healthcare is driven by privacy concerns, limited access to real-world data, and the high cost of annotation. This work explores the use of synthetic Prolonged Exposure (PE) therapeutic conversations for Post-Traumatic Stress Disorder (PTSD) as a scalable alternative for training and evaluating clinical models. We systematically compare real and synthetic dialogues using linguistic, structural, and protocol-specific metrics, including turn-taking patterns and treatment fidelity. We also introduce and evaluate PE-specific metrics derived from linguistic analysis and semantic modeling, offering a novel framework for assessing clinical fidelity beyond surface fluency. Our findings show that although synthetic data holds promise for mitigating data scarcity and protecting patient privacy, it can struggle to capture the subtle dynamics of therapeutic interactions. In our dataset, synthetic dialogues match structural features of real-world dialogues (e.g., speaker switch ratio: 0.98 vs. 0.99), however, synthetic interactions do not adequately reflect key fidelity markers (e.g., distress monitoring). We highlight gaps in existing evaluation frameworks and advocate for fidelity-aware metrics that go beyond surface fluency to uncover clinically significant failures. Our findings clarify where synthetic data can effectively complement real-world datasets -- and where critical limitations remain. 

---
# WebThinker: Empowering Large Reasoning Models with Deep Research Capability 

**Authors**: Xiaoxi Li, Jiajie Jin, Guanting Dong, Hongjin Qian, Yutao Zhu, Yongkang Wu, Ji-Rong Wen, Zhicheng Dou  

**Link**: [PDF](https://arxiv.org/pdf/2504.21776)  

**Abstract**: Large reasoning models (LRMs), such as OpenAI-o1 and DeepSeek-R1, demonstrate impressive long-horizon reasoning capabilities. However, their reliance on static internal knowledge limits their performance on complex, knowledge-intensive tasks and hinders their ability to produce comprehensive research reports requiring synthesis of diverse web information. To address this, we propose \textbf{WebThinker}, a deep research agent that empowers LRMs to autonomously search the web, navigate web pages, and draft research reports during the reasoning process. WebThinker integrates a \textbf{Deep Web Explorer} module, enabling LRMs to dynamically search, navigate, and extract information from the web when encountering knowledge gaps. It also employs an \textbf{Autonomous Think-Search-and-Draft strategy}, allowing the model to seamlessly interleave reasoning, information gathering, and report writing in real time. To further enhance research tool utilization, we introduce an \textbf{RL-based training strategy} via iterative online Direct Preference Optimization (DPO). Extensive experiments on complex reasoning benchmarks (GPQA, GAIA, WebWalkerQA, HLE) and scientific report generation tasks (Glaive) demonstrate that WebThinker significantly outperforms existing methods and strong proprietary systems. Our approach enhances LRM reliability and applicability in complex scenarios, paving the way for more capable and versatile deep research systems. The code is available at this https URL. 

---
# MAC-Tuning: LLM Multi-Compositional Problem Reasoning with Enhanced Knowledge Boundary Awareness 

**Authors**: Junsheng Huang, Zhitao He, Sandeep Polisetty, Qingyun Wang, May Fung  

**Link**: [PDF](https://arxiv.org/pdf/2504.21773)  

**Abstract**: With the widespread application of large language models (LLMs), the issue of generating non-existing facts, known as hallucination, has garnered increasing attention. Previous research in enhancing LLM confidence estimation mainly focuses on the single problem setting. However, LLM awareness of its internal parameterized knowledge boundary under the more challenging multi-problem setting, which requires answering multiple problems accurately simultaneously, remains underexplored. To bridge this gap, we introduce a novel method, Multiple Answers and Confidence Stepwise Tuning (MAC-Tuning), that separates the learning of answer prediction and confidence estimation during fine-tuning on instruction data. Extensive experiments demonstrate that our method outperforms baselines by up to 25% in average precision. 

---
# Improving Retrieval-Augmented Neural Machine Translation with Monolingual Data 

**Authors**: Maxime Bouthors, Josep Crego, François Yvon  

**Link**: [PDF](https://arxiv.org/pdf/2504.21747)  

**Abstract**: Conventional retrieval-augmented neural machine translation (RANMT) systems leverage bilingual corpora, e.g., translation memories (TMs). Yet, in many settings, in-domain monolingual target-side corpora are often available. This work explores ways to take advantage of such resources by retrieving relevant segments directly in the target language, based on a source-side query. For this, we design improved cross-lingual retrieval systems, trained with both sentence level and word-level matching objectives. In our experiments with two RANMT architectures, we first demonstrate the benefits of such cross-lingual objectives in a controlled setting, obtaining translation performances that surpass standard TM-based models. We then showcase our method on a real-world set-up, where the target monolingual resources far exceed the amount of parallel data and observe large improvements of our new techniques, which outperform both the baseline setting, and general-purpose cross-lingual retrievers. 

---
# Investigating Literary Motifs in Ancient and Medieval Novels with Large Language Models 

**Authors**: Emelie Hallenberg  

**Link**: [PDF](https://arxiv.org/pdf/2504.21742)  

**Abstract**: The Greek fictional narratives often termed love novels or romances, ranging from the first century CE to the middle of the 15th century, have long been considered as similar in many ways, not least in the use of particular literary motifs. By applying the use of fine-tuned large language models, this study aims to investigate which motifs exactly that the texts in this corpus have in common, and in which ways they differ from each other. The results show that while some motifs persist throughout the corpus, others fluctuate in frequency, indicating certain trends or external influences. Conclusively, the method proves to adequately extract literary motifs according to a set definition, providing data for both quantitative and qualitative analyses. 

---
# Enhancing Health Mention Classification Performance: A Study on Advancements in Parameter Efficient Tuning 

**Authors**: Reem Abdel-Salam, Mary Adewunmi  

**Link**: [PDF](https://arxiv.org/pdf/2504.21685)  

**Abstract**: Health Mention Classification (HMC) plays a critical role in leveraging social media posts for real-time tracking and public health monitoring. Nevertheless, the process of HMC presents significant challenges due to its intricate nature, primarily stemming from the contextual aspects of health mentions, such as figurative language and descriptive terminology, rather than explicitly reflecting a personal ailment. To address this problem, we argue that clearer mentions can be achieved through conventional fine-tuning with enhanced parameters of biomedical natural language methods (NLP). In this study, we explore different techniques such as the utilisation of part-of-speech (POS) tagger information, improving on PEFT techniques, and different combinations thereof. Extensive experiments are conducted on three widely used datasets: RHDM, PHM, and Illness. The results incorporated POS tagger information, and leveraging PEFT techniques significantly improves performance in terms of F1-score compared to state-of-the-art methods across all three datasets by utilising smaller models and efficient training. Furthermore, the findings highlight the effectiveness of incorporating POS tagger information and leveraging PEFT techniques for HMC. In conclusion, the proposed methodology presents a potentially effective approach to accurately classifying health mentions in social media posts while optimising the model size and training efficiency. 

---
# Investigating the Effect of Parallel Data in the Cross-Lingual Transfer for Vision-Language Encoders 

**Authors**: Andrei-Alexandru Manea, Jindřich Libovický  

**Link**: [PDF](https://arxiv.org/pdf/2504.21681)  

**Abstract**: Most pre-trained Vision-Language (VL) models and training data for the downstream tasks are only available in English. Therefore, multilingual VL tasks are solved using cross-lingual transfer: fine-tune a multilingual pre-trained model or transfer the text encoder using parallel data. We study the alternative approach: transferring an already trained encoder using parallel data. We investigate the effect of parallel data: domain and the number of languages, which were out of focus in previous work. Our results show that even machine-translated task data are the best on average, caption-like authentic parallel data outperformed it in some languages. Further, we show that most languages benefit from multilingual training. 

---
# 20min-XD: A Comparable Corpus of Swiss News Articles 

**Authors**: Michelle Wastl, Jannis Vamvas, Selena Calleri, Rico Sennrich  

**Link**: [PDF](https://arxiv.org/pdf/2504.21677)  

**Abstract**: We present 20min-XD (20 Minuten cross-lingual document-level), a French-German, document-level comparable corpus of news articles, sourced from the Swiss online news outlet 20 Minuten/20 minutes. Our dataset comprises around 15,000 article pairs spanning 2015 to 2024, automatically aligned based on semantic similarity. We detail the data collection process and alignment methodology. Furthermore, we provide a qualitative and quantitative analysis of the corpus. The resulting dataset exhibits a broad spectrum of cross-lingual similarity, ranging from near-translations to loosely related articles, making it valuable for various NLP applications and broad linguistically motivated studies. We publicly release the dataset in document- and sentence-aligned versions and code for the described experiments. 

---
# Sadeed: Advancing Arabic Diacritization Through Small Language Model 

**Authors**: Zeina Aldallal, Sara Chrouf, Khalil Hennara, Mohamed Motaism Hamed, Muhammad Hreden, Safwan AlModhayan  

**Link**: [PDF](https://arxiv.org/pdf/2504.21635)  

**Abstract**: Arabic text diacritization remains a persistent challenge in natural language processing due to the language's morphological richness. In this paper, we introduce Sadeed, a novel approach based on a fine-tuned decoder-only language model adapted from Kuwain 1.5B Hennara et al. [2025], a compact model originally trained on diverse Arabic corpora. Sadeed is fine-tuned on carefully curated, high-quality diacritized datasets, constructed through a rigorous data-cleaning and normalization pipeline. Despite utilizing modest computational resources, Sadeed achieves competitive results compared to proprietary large language models and outperforms traditional models trained on similar domains. Additionally, we highlight key limitations in current benchmarking practices for Arabic diacritization. To address these issues, we introduce SadeedDiac-25, a new benchmark designed to enable fairer and more comprehensive evaluation across diverse text genres and complexity levels. Together, Sadeed and SadeedDiac-25 provide a robust foundation for advancing Arabic NLP applications, including machine translation, text-to-speech, and language learning tools. 

---
# Meeseeks: An Iterative Benchmark Evaluating LLMs Multi-Turn Instruction-Following Ability 

**Authors**: Jiaming Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.21625)  

**Abstract**: The ability to follow instructions accurately is fundamental for Large Language Models (LLMs) to serve as reliable agents in real-world applications. While existing instruction-following benchmarks are either single-turn or introduce new requirements in each turn without allowing self-correction, Meeseeks simulates realistic human-LLM interactions through an iterative feedback process. This design enables models to self-correct based on specific requirement failures, better reflecting real-world user-end usage patterns. The benchmark implements a comprehensive evaluation system with 38 capability tags organized across three dimensions: Intent Recognition, Granular Content Validation, and Output Structure Validation. Through rigorous evaluation across LLMs, Meeseeks provides valuable insights into LLMs' instruction-following capabilities in practical applications. 

---
# RDF-Based Structured Quality Assessment Representation of Multilingual LLM Evaluations 

**Authors**: Jonas Gwozdz, Andreas Both  

**Link**: [PDF](https://arxiv.org/pdf/2504.21605)  

**Abstract**: Large Language Models (LLMs) increasingly serve as knowledge interfaces, yet systematically assessing their reliability with conflicting information remains difficult. We propose an RDF-based framework to assess multilingual LLM quality, focusing on knowledge conflicts. Our approach captures model responses across four distinct context conditions (complete, incomplete, conflicting, and no-context information) in German and English. This structured representation enables the comprehensive analysis of knowledge leakage-where models favor training data over provided context-error detection, and multilingual consistency. We demonstrate the framework through a fire safety domain experiment, revealing critical patterns in context prioritization and language-specific performance, and demonstrating that our vocabulary was sufficient to express every assessment facet encountered in the 28-question study. 

---
# Robust Misinformation Detection by Visiting Potential Commonsense Conflict 

**Authors**: Bing Wang, Ximing Li, Changchun Li, Bingrui Zhao, Bo Fu, Renchu Guan, Shengsheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.21604)  

**Abstract**: The development of Internet technology has led to an increased prevalence of misinformation, causing severe negative effects across diverse domains. To mitigate this challenge, Misinformation Detection (MD), aiming to detect online misinformation automatically, emerges as a rapidly growing research topic in the community. In this paper, we propose a novel plug-and-play augmentation method for the MD task, namely Misinformation Detection with Potential Commonsense Conflict (MD-PCC). We take inspiration from the prior studies indicating that fake articles are more likely to involve commonsense conflict. Accordingly, we construct commonsense expressions for articles, serving to express potential commonsense conflicts inferred by the difference between extracted commonsense triplet and golden ones inferred by the well-established commonsense reasoning tool COMET. These expressions are then specified for each article as augmentation. Any specific MD methods can be then trained on those commonsense-augmented articles. Besides, we also collect a novel commonsense-oriented dataset named CoMis, whose all fake articles are caused by commonsense conflict. We integrate MD-PCC with various existing MD backbones and compare them across both 4 public benchmark datasets and CoMis. Empirical results demonstrate that MD-PCC can consistently outperform the existing MD baselines. 

---
# DNB-AI-Project at SemEval-2025 Task 5: An LLM-Ensemble Approach for Automated Subject Indexing 

**Authors**: Lisa Kluge, Maximilian Kähler  

**Link**: [PDF](https://arxiv.org/pdf/2504.21589)  

**Abstract**: This paper presents our system developed for the SemEval-2025 Task 5: LLMs4Subjects: LLM-based Automated Subject Tagging for a National Technical Library's Open-Access Catalog. Our system relies on prompting a selection of LLMs with varying examples of intellectually annotated records and asking the LLMs to similarly suggest keywords for new records. This few-shot prompting technique is combined with a series of post-processing steps that map the generated keywords to the target vocabulary, aggregate the resulting subject terms to an ensemble vote and, finally, rank them as to their relevance to the record. Our system is fourth in the quantitative ranking in the all-subjects track, but achieves the best result in the qualitative ranking conducted by subject indexing experts. 

---
# Precision Where It Matters: A Novel Spike Aware Mixed-Precision Quantization Strategy for LLaMA-based Language Models 

**Authors**: Lucas Maisonnave, Cyril Moineau, Olivier Bichler, Fabrice Rastello  

**Link**: [PDF](https://arxiv.org/pdf/2504.21553)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities in various natural language processing tasks. However, their size presents significant challenges for deployment and inference. This paper investigates the quantization of LLMs, focusing on the LLaMA architecture and its derivatives. We challenge existing assumptions about activation outliers in LLMs and propose a novel mixed-precision quantization approach tailored for LLaMA-like models. Our method leverages the observation that activation spikes in LLaMA architectures are predominantly concentrated in specific projection layers. By applying higher precision (FP16 or FP8) to these layers while quantizing the rest of the model to lower bit-widths, we achieve superior performance compared to existing quantization techniques. Experimental results on LLaMA2, LLaMA3, and Mistral models demonstrate significant improvements in perplexity and zero-shot accuracy, particularly for 8-bit per-tensor quantization. Our approach outperforms general-purpose methods designed to handle outliers across all architecture types, highlighting the benefits of architecture-specific quantization strategies. This research contributes to the ongoing efforts to make LLMs more efficient and deployable, potentially enabling their use in resource-constrained environments. Our findings emphasize the importance of considering model-specific characteristics in developing effective quantization pipelines for state-of-the-art language models by identifying and targeting a small number of projections that concentrate activation spikes. 

---
# TartuNLP at SemEval-2025 Task 5: Subject Tagging as Two-Stage Information Retrieval 

**Authors**: Aleksei Dorkin, Kairit Sirts  

**Link**: [PDF](https://arxiv.org/pdf/2504.21547)  

**Abstract**: We present our submission to the Task 5 of SemEval-2025 that aims to aid librarians in assigning subject tags to the library records by producing a list of likely relevant tags for a given document. We frame the task as an information retrieval problem, where the document content is used to retrieve subject tags from a large subject taxonomy. We leverage two types of encoder models to build a two-stage information retrieval system -- a bi-encoder for coarse-grained candidate extraction at the first stage, and a cross-encoder for fine-grained re-ranking at the second stage. This approach proved effective, demonstrating significant improvements in recall compared to single-stage methods and showing competitive results according to qualitative evaluation. 

---
# Improving Informally Romanized Language Identification 

**Authors**: Adrian Benton, Alexander Gutkin, Christo Kirov, Brian Roark  

**Link**: [PDF](https://arxiv.org/pdf/2504.21540)  

**Abstract**: The Latin script is often used to informally write languages with non-Latin native scripts. In many cases (e.g., most languages in India), there is no conventional spelling of words in the Latin script, hence there will be high spelling variability in written text. Such romanization renders languages that are normally easily distinguished based on script highly confusable, such as Hindi and Urdu. In this work, we increase language identification (LID) accuracy for romanized text by improving the methods used to synthesize training sets. We find that training on synthetic samples which incorporate natural spelling variation yields higher LID system accuracy than including available naturally occurring examples in the training set, or even training higher capacity models. We demonstrate new state-of-the-art LID performance on romanized text from 20 Indic languages in the Bhasha-Abhijnaanam evaluation set (Madhani et al., 2023a), improving test F1 from the reported 74.7% (using a pretrained neural model) to 85.4% using a linear classifier trained solely on synthetic data and 88.2% when also training on available harvested text. 

---
# Advancing Arabic Reverse Dictionary Systems: A Transformer-Based Approach with Dataset Construction Guidelines 

**Authors**: Serry Sibaee, Samar Ahmed, Abdullah Al Harbi, Omer Nacar, Adel Ammar, Yasser Habashi, Wadii Boulila  

**Link**: [PDF](https://arxiv.org/pdf/2504.21475)  

**Abstract**: This study addresses the critical gap in Arabic natural language processing by developing an effective Arabic Reverse Dictionary (RD) system that enables users to find words based on their descriptions or meanings. We present a novel transformer-based approach with a semi-encoder neural network architecture featuring geometrically decreasing layers that achieves state-of-the-art results for Arabic RD tasks. Our methodology incorporates a comprehensive dataset construction process and establishes formal quality standards for Arabic lexicographic definitions. Experiments with various pre-trained models demonstrate that Arabic-specific models significantly outperform general multilingual embeddings, with ARBERTv2 achieving the best ranking score (0.0644). Additionally, we provide a formal abstraction of the reverse dictionary task that enhances theoretical understanding and develop a modular, extensible Python library (RDTL) with configurable training pipelines. Our analysis of dataset quality reveals important insights for improving Arabic definition construction, leading to eight specific standards for building high-quality reverse dictionary resources. This work contributes significantly to Arabic computational linguistics and provides valuable tools for language learning, academic writing, and professional communication in Arabic. 

---
# Homa at SemEval-2025 Task 5: Aligning Librarian Records with OntoAligner for Subject Tagging 

**Authors**: Hadi Bayrami Asl Tekanlou, Jafar Razmara, Mahsa Sanaei, Mostafa Rahgouy, Hamed Babaei Giglou  

**Link**: [PDF](https://arxiv.org/pdf/2504.21474)  

**Abstract**: This paper presents our system, Homa, for SemEval-2025 Task 5: Subject Tagging, which focuses on automatically assigning subject labels to technical records from TIBKAT using the Gemeinsame Normdatei (GND) taxonomy. We leverage OntoAligner, a modular ontology alignment toolkit, to address this task by integrating retrieval-augmented generation (RAG) techniques. Our approach formulates the subject tagging problem as an alignment task, where records are matched to GND categories based on semantic similarity. We evaluate OntoAligner's adaptability for subject indexing and analyze its effectiveness in handling multilingual records. Experimental results demonstrate the strengths and limitations of this method, highlighting the potential of alignment techniques for improving subject tagging in digital libraries. 

---
# RWKV-X: A Linear Complexity Hybrid Language Model 

**Authors**: Haowen Hou, Zhiyi Huang, Kaifeng Tan, Rongchang Lu, Fei Richard Yu  

**Link**: [PDF](https://arxiv.org/pdf/2504.21463)  

**Abstract**: In this paper, we introduce \textbf{RWKV-X}, a novel hybrid architecture that combines the efficiency of RWKV for short-range modeling with a sparse attention mechanism designed to capture long-range context. Unlike previous hybrid approaches that rely on full attention layers and retain quadratic complexity, RWKV-X achieves linear-time complexity in training and constant-time complexity in inference decoding. We demonstrate that RWKV-X, when continually pretrained on 64K-token sequences, achieves near-perfect accuracy on the 64K passkey retrieval benchmark. It consistently outperforms prior RWKV-7 models on long-context benchmarks, while maintaining strong performance on short-context tasks. These results highlight RWKV-X as a scalable and efficient backbone for general-purpose language modeling, capable of decoding sequences up to 1 million tokens with stable speed and memory usage. To facilitate further research and analysis, we have made the checkpoints and the associated code publicly accessible at: this https URL. 

---
# The Distribution of Dependency Distance and Hierarchical Distance in Contemporary Written Japanese and Its Influencing Factors 

**Authors**: Linxuan Wang, Shuiyuan Yu  

**Link**: [PDF](https://arxiv.org/pdf/2504.21421)  

**Abstract**: To explore the relationship between dependency distance (DD) and hierarchical distance (HD) in Japanese, we compared the probability distributions of DD and HD with and without sentence length fixed, and analyzed the changes in mean dependency distance (MDD) and mean hierarchical distance (MHD) as sentence length increases, along with their correlation coefficient based on the Balanced Corpus of Contemporary Written Japanese. It was found that the valency of the predicates is the underlying factor behind the trade-off relation between MDD and MHD in Japanese. Native speakers of Japanese regulate the linear complexity and hierarchical complexity through the valency of the predicates, and the relative sizes of MDD and MHD depend on whether the threshold of valency has been reached. Apart from the cognitive load, the valency of the predicates also affects the probability distributions of DD and HD. The effect of the valency of the predicates on the distribution of HD is greater than on that of DD, which leads to differences in their probability distributions and causes the mean of MDD to be lower than that of MHD. 

---
# Retrieval-Enhanced Few-Shot Prompting for Speech Event Extraction 

**Authors**: Máté Gedeon  

**Link**: [PDF](https://arxiv.org/pdf/2504.21372)  

**Abstract**: Speech Event Extraction (SpeechEE) is a challenging task that lies at the intersection of Automatic Speech Recognition (ASR) and Natural Language Processing (NLP), requiring the identification of structured event information from spoken language. In this work, we present a modular, pipeline-based SpeechEE framework that integrates high-performance ASR with semantic search-enhanced prompting of Large Language Models (LLMs). Our system first classifies speech segments likely to contain events using a hybrid filtering mechanism including rule-based, BERT-based, and LLM-based models. It then employs few-shot LLM prompting, dynamically enriched via semantic similarity retrieval, to identify event triggers and extract corresponding arguments. We evaluate the pipeline using multiple LLMs (Llama3-8B, GPT-4o-mini, and o1-mini) highlighting significant performance gains with o1-mini, which achieves 63.3% F1 on trigger classification and 27.8% F1 on argument classification, outperforming prior benchmarks. Our results demonstrate that pipeline approaches, when empowered by retrieval-augmented LLMs, can rival or exceed end-to-end systems while maintaining interpretability and modularity. This work provides practical insights into LLM-driven event extraction and opens pathways for future hybrid models combining textual and acoustic features. 

---
# Does the Prompt-based Large Language Model Recognize Students' Demographics and Introduce Bias in Essay Scoring? 

**Authors**: Kaixun Yang, Mladen Raković, Dragan Gašević, Guanliang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.21330)  

**Abstract**: Large Language Models (LLMs) are widely used in Automated Essay Scoring (AES) due to their ability to capture semantic meaning. Traditional fine-tuning approaches required technical expertise, limiting accessibility for educators with limited technical backgrounds. However, prompt-based tools like ChatGPT have made AES more accessible, enabling educators to obtain machine-generated scores using natural-language prompts (i.e., the prompt-based paradigm). Despite advancements, prior studies have shown bias in fine-tuned LLMs, particularly against disadvantaged groups. It remains unclear whether such biases persist or are amplified in the prompt-based paradigm with cutting-edge tools. Since such biases are believed to stem from the demographic information embedded in pre-trained models (i.e., the ability of LLMs' text embeddings to predict demographic attributes), this study explores the relationship between the model's predictive power of students' demographic attributes based on their written works and its predictive bias in the scoring task in the prompt-based paradigm. Using a publicly available dataset of over 25,000 students' argumentative essays, we designed prompts to elicit demographic inferences (i.e., gender, first-language background) from GPT-4o and assessed fairness in automated scoring. Then we conducted multivariate regression analysis to explore the impact of the model's ability to predict demographics on its scoring outcomes. Our findings revealed that (i) prompt-based LLMs can somewhat infer students' demographics, particularly their first-language backgrounds, from their essays; (ii) scoring biases are more pronounced when the LLM correctly predicts students' first-language background than when it does not; and (iii) scoring error for non-native English speakers increases when the LLM correctly identifies them as non-native. 

---
# Confidence in Large Language Model Evaluation: A Bayesian Approach to Limited-Sample Challenges 

**Authors**: Xiao Xiao, Yu Su, Sijing Zhang, Zhang Chen, Yadong Chen, Tian Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.21303)  

**Abstract**: Large language models (LLMs) exhibit probabilistic output characteristics, yet conventional evaluation frameworks rely on deterministic scalar metrics. This study introduces a Bayesian approach for LLM capability assessment that integrates prior knowledge through probabilistic inference, addressing limitations under limited-sample regimes. By treating model capabilities as latent variables and leveraging a curated query set to induce discriminative responses, we formalize model ranking as a Bayesian hypothesis testing problem over mutually exclusive capability intervals. Experimental evaluations with GPT-series models demonstrate that the proposed method achieves superior discrimination compared to conventional evaluation methods. Results indicate that even with reduced sample sizes, the approach maintains statistical robustness while providing actionable insights, such as probabilistic statements about a model's likelihood of surpassing specific baselines. This work advances LLM evaluation methodologies by bridging Bayesian inference with practical constraints in real-world deployment scenarios. 

---
# BiasGuard: A Reasoning-enhanced Bias Detection Tool For Large Language Models 

**Authors**: Zhiting Fan, Ruizhe Chen, Zuozhu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.21299)  

**Abstract**: Identifying bias in LLM-generated content is a crucial prerequisite for ensuring fairness in LLMs. Existing methods, such as fairness classifiers and LLM-based judges, face limitations related to difficulties in understanding underlying intentions and the lack of criteria for fairness judgment. In this paper, we introduce BiasGuard, a novel bias detection tool that explicitly analyzes inputs and reasons through fairness specifications to provide accurate judgments. BiasGuard is implemented through a two-stage approach: the first stage initializes the model to explicitly reason based on fairness specifications, while the second stage leverages reinforcement learning to enhance its reasoning and judgment capabilities. Our experiments, conducted across five datasets, demonstrate that BiasGuard outperforms existing tools, improving accuracy and reducing over-fairness misjudgments. We also highlight the importance of reasoning-enhanced decision-making and provide evidence for the effectiveness of our two-stage optimization pipeline. 

---
# Talk Before You Retrieve: Agent-Led Discussions for Better RAG in Medical QA 

**Authors**: Xuanzhao Dong, Wenhui Zhu, Hao Wang, Xiwen Chen, Peijie Qiu, Rui Yin, Yi Su, Yalin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.21252)  

**Abstract**: Medical question answering (QA) is a reasoning-intensive task that remains challenging for large language models (LLMs) due to hallucinations and outdated domain knowledge. Retrieval-Augmented Generation (RAG) provides a promising post-training solution by leveraging external knowledge. However, existing medical RAG systems suffer from two key limitations: (1) a lack of modeling for human-like reasoning behaviors during information retrieval, and (2) reliance on suboptimal medical corpora, which often results in the retrieval of irrelevant or noisy snippets. To overcome these challenges, we propose Discuss-RAG, a plug-and-play module designed to enhance the medical QA RAG system through collaborative agent-based reasoning. Our method introduces a summarizer agent that orchestrates a team of medical experts to emulate multi-turn brainstorming, thereby improving the relevance of retrieved content. Additionally, a decision-making agent evaluates the retrieved snippets before their final integration. Experimental results on four benchmark medical QA datasets show that Discuss-RAG consistently outperforms MedRAG, especially significantly improving answer accuracy by up to 16.67% on BioASQ and 12.20% on PubMedQA. The code is available at: this https URL. 

---
# Memorization and Knowledge Injection in Gated LLMs 

**Authors**: Xu Pan, Ely Hahami, Zechen Zhang, Haim Sompolinsky  

**Link**: [PDF](https://arxiv.org/pdf/2504.21239)  

**Abstract**: Large Language Models (LLMs) currently struggle to sequentially add new memories and integrate new knowledge. These limitations contrast with the human ability to continuously learn from new experiences and acquire knowledge throughout life. Most existing approaches add memories either through large context windows or external memory buffers (e.g., Retrieval-Augmented Generation), and studies on knowledge injection rarely test scenarios resembling everyday life events. In this work, we introduce a continual learning framework, Memory Embedded in Gated LLMs (MEGa), which injects event memories directly into the weights of LLMs. Each memory is stored in a dedicated set of gated low-rank weights. During inference, a gating mechanism activates relevant memory weights by matching query embeddings to stored memory embeddings. This enables the model to both recall entire memories and answer related questions. On two datasets - fictional characters and Wikipedia events - MEGa outperforms baseline approaches in mitigating catastrophic forgetting. Our model draws inspiration from the complementary memory system of the human brain. 

---
# Phi-4-Mini-Reasoning: Exploring the Limits of Small Reasoning Language Models in Math 

**Authors**: Haoran Xu, Baolin Peng, Hany Awadalla, Dongdong Chen, Yen-Chun Chen, Mei Gao, Young Jin Kim, Yunsheng Li, Liliang Ren, Yelong Shen, Shuohang Wang, Weijian Xu, Jianfeng Gao, Weizhu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.21233)  

**Abstract**: Chain-of-Thought (CoT) significantly enhances formal reasoning capabilities in Large Language Models (LLMs) by training them to explicitly generate intermediate reasoning steps. While LLMs readily benefit from such techniques, improving reasoning in Small Language Models (SLMs) remains challenging due to their limited model capacity. Recent work by Deepseek-R1 demonstrates that distillation from LLM-generated synthetic data can substantially improve the reasoning ability of SLM. However, the detailed modeling recipe is not disclosed. In this work, we present a systematic training recipe for SLMs that consists of four steps: (1) large-scale mid-training on diverse distilled long-CoT data, (2) supervised fine-tuning on high-quality long-CoT data, (3) Rollout DPO leveraging a carefully curated preference dataset, and (4) Reinforcement Learning (RL) with Verifiable Reward. We apply our method on Phi-4-Mini, a compact 3.8B-parameter model. The resulting Phi-4-Mini-Reasoning model exceeds, on math reasoning tasks, much larger reasoning models, e.g., outperforming DeepSeek-R1-Distill-Qwen-7B by 3.2 points and DeepSeek-R1-Distill-Llama-8B by 7.7 points on Math-500. Our results validate that a carefully designed training recipe, with large-scale high-quality CoT data, is effective to unlock strong reasoning capabilities even in resource-constrained small models. 

---
# Pretraining Large Brain Language Model for Active BCI: Silent Speech 

**Authors**: Jinzhao Zhou, Zehong Cao, Yiqun Duan, Connor Barkley, Daniel Leong, Xiaowei Jiang, Quoc-Toan Nguyen, Ziyi Zhao, Thomas Do, Yu-Cheng Chang, Sheng-Fu Liang, Chin-teng Lin  

**Link**: [PDF](https://arxiv.org/pdf/2504.21214)  

**Abstract**: This paper explores silent speech decoding in active brain-computer interface (BCI) systems, which offer more natural and flexible communication than traditional BCI applications. We collected a new silent speech dataset of over 120 hours of electroencephalogram (EEG) recordings from 12 subjects, capturing 24 commonly used English words for language model pretraining and decoding. Following the recent success of pretraining large models with self-supervised paradigms to enhance EEG classification performance, we propose Large Brain Language Model (LBLM) pretrained to decode silent speech for active BCI. To pretrain LBLM, we propose Future Spectro-Temporal Prediction (FSTP) pretraining paradigm to learn effective representations from unlabeled EEG data. Unlike existing EEG pretraining methods that mainly follow a masked-reconstruction paradigm, our proposed FSTP method employs autoregressive modeling in temporal and frequency domains to capture both temporal and spectral dependencies from EEG signals. After pretraining, we finetune our LBLM on downstream tasks, including word-level and semantic-level classification. Extensive experiments demonstrate significant performance gains of the LBLM over fully-supervised and pretrained baseline models. For instance, in the difficult cross-session setting, our model achieves 47.0\% accuracy on semantic-level classification and 39.6\% in word-level classification, outperforming baseline methods by 5.4\% and 7.3\%, respectively. Our research advances silent speech decoding in active BCI systems, offering an innovative solution for EEG language model pretraining and a new dataset for fundamental research. 

---
# Automatic Legal Writing Evaluation of LLMs 

**Authors**: Ramon Pires, Roseval Malaquias Junior, Rodrigo Nogueira  

**Link**: [PDF](https://arxiv.org/pdf/2504.21202)  

**Abstract**: Despite the recent advances in Large Language Models, benchmarks for evaluating legal writing remain scarce due to the inherent complexity of assessing open-ended responses in this domain. One of the key challenges in evaluating language models on domain-specific tasks is finding test datasets that are public, frequently updated, and contain comprehensive evaluation guidelines. The Brazilian Bar Examination meets these requirements. We introduce oab-bench, a benchmark comprising 105 questions across seven areas of law from recent editions of the exam. The benchmark includes comprehensive evaluation guidelines and reference materials used by human examiners to ensure consistent grading. We evaluate the performance of four LLMs on oab-bench, finding that Claude-3.5 Sonnet achieves the best results with an average score of 7.93 out of 10, passing all 21 exams. We also investigated whether LLMs can serve as reliable automated judges for evaluating legal writing. Our experiments show that frontier models like OpenAI's o1 achieve a strong correlation with human scores when evaluating approved exams, suggesting their potential as reliable automated evaluators despite the inherently subjective nature of legal writing assessment. The source code and the benchmark -- containing questions, evaluation guidelines, model-generated responses, and their respective automated evaluations -- are publicly available. 

---
# Small or Large? Zero-Shot or Finetuned? Guiding Language Model Choice for Specialized Applications in Healthcare 

**Authors**: Lovedeep Gondara, Jonathan Simkin, Graham Sayle, Shebnum Devji, Gregory Arbour, Raymond Ng  

**Link**: [PDF](https://arxiv.org/pdf/2504.21191)  

**Abstract**: This study aims to guide language model selection by investigating: 1) the necessity of finetuning versus zero-shot usage, 2) the benefits of domain-adjacent versus generic pretrained models, 3) the value of further domain-specific pretraining, and 4) the continued relevance of Small Language Models (SLMs) compared to Large Language Models (LLMs) for specific tasks. Using electronic pathology reports from the British Columbia Cancer Registry (BCCR), three classification scenarios with varying difficulty and data size are evaluated. Models include various SLMs and an LLM. SLMs are evaluated both zero-shot and finetuned; the LLM is evaluated zero-shot only. Finetuning significantly improved SLM performance across all scenarios compared to their zero-shot results. The zero-shot LLM outperformed zero-shot SLMs but was consistently outperformed by finetuned SLMs. Domain-adjacent SLMs generally performed better than the generic SLM after finetuning, especially on harder tasks. Further domain-specific pretraining yielded modest gains on easier tasks but significant improvements on the complex, data-scarce task. The results highlight the critical role of finetuning for SLMs in specialized domains, enabling them to surpass zero-shot LLM performance on targeted classification tasks. Pretraining on domain-adjacent or domain-specific data provides further advantages, particularly for complex problems or limited finetuning data. While LLMs offer strong zero-shot capabilities, their performance on these specific tasks did not match that of appropriately finetuned SLMs. In the era of LLMs, SLMs remain relevant and effective, offering a potentially superior performance-resource trade-off compared to LLMs. 

---
# Detecting Manipulated Contents Using Knowledge-Grounded Inference 

**Authors**: Mark Huasong Meng, Ruizhe Wang, Meng Xu, Chuan Yan, Guangdong Bai  

**Link**: [PDF](https://arxiv.org/pdf/2504.21165)  

**Abstract**: The detection of manipulated content, a prevalent form of fake news, has been widely studied in recent years. While existing solutions have been proven effective in fact-checking and analyzing fake news based on historical events, the reliance on either intrinsic knowledge obtained during training or manually curated context hinders them from tackling zero-day manipulated content, which can only be recognized with real-time contextual information. In this work, we propose Manicod, a tool designed for detecting zero-day manipulated content. Manicod first sources contextual information about the input claim from mainstream search engines, and subsequently vectorizes the context for the large language model (LLM) through retrieval-augmented generation (RAG). The LLM-based inference can produce a "truthful" or "manipulated" decision and offer a textual explanation for the decision. To validate the effectiveness of Manicod, we also propose a dataset comprising 4270 pieces of manipulated fake news derived from 2500 recent real-world news headlines. Manicod achieves an overall F1 score of 0.856 on this dataset and outperforms existing methods by up to 1.9x in F1 score on their benchmarks on fact-checking and claim verification. 

---
# LLM Enhancer: Merged Approach using Vector Embedding for Reducing Large Language Model Hallucinations with External Knowledge 

**Authors**: Naheed Rayhan, Md. Ashrafuzzaman  

**Link**: [PDF](https://arxiv.org/pdf/2504.21132)  

**Abstract**: Large Language Models (LLMs), such as ChatGPT, have demonstrated the capability to generate human like, natural responses across a range of tasks, including task oriented dialogue and question answering. However, their application in real world, critical scenarios is often hindered by a tendency to produce inaccurate information and a limited ability to leverage external knowledge sources. This paper introduces the LLM ENHANCER system, designed to integrate multiple online sources such as Google, Wikipedia, and DuckDuckGo to enhance data accuracy. The LLMs employed within this system are open source. The data acquisition process for the LLM ENHANCER system operates in parallel, utilizing custom agent tools to manage the flow of information. Vector embeddings are used to identify the most pertinent information, which is subsequently supplied to the LLM for user interaction. The LLM ENHANCER system mitigates hallucinations in chat based LLMs while preserving response naturalness and accuracy. 

---
# Beyond One-Size-Fits-All: Inversion Learning for Highly Effective NLG Evaluation Prompts 

**Authors**: Hanhua Hong, Chenghao Xiao, Yang Wang, Yiqi Liu, Wenge Rong, Chenghua Lin  

**Link**: [PDF](https://arxiv.org/pdf/2504.21117)  

**Abstract**: Evaluating natural language generation (NLG) systems is challenging due to the diversity of valid outputs. While human evaluation is the gold standard, it suffers from inconsistencies, lack of standardisation, and demographic biases, limiting reproducibility. LLM-based evaluation offers a scalable alternative but is highly sensitive to prompt design, where small variations can lead to significant discrepancies. In this work, we propose an inversion learning method that learns effective reverse mappings from model outputs back to their input instructions, enabling the automatic generation of highly effective, model-specific evaluation prompts. Our method requires only a single evaluation sample and eliminates the need for time-consuming manual prompt engineering, thereby improving both efficiency and robustness. Our work contributes toward a new direction for more robust and efficient LLM-based evaluation. 

---
# UrbanPlanBench: A Comprehensive Urban Planning Benchmark for Evaluating Large Language Models 

**Authors**: Yu Zheng, Longyi Liu, Yuming Lin, Jie Feng, Guozhen Zhang, Depeng Jin, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.21027)  

**Abstract**: The advent of Large Language Models (LLMs) holds promise for revolutionizing various fields traditionally dominated by human expertise. Urban planning, a professional discipline that fundamentally shapes our daily surroundings, is one such field heavily relying on multifaceted domain knowledge and experience of human experts. The extent to which LLMs can assist human practitioners in urban planning remains largely unexplored. In this paper, we introduce a comprehensive benchmark, UrbanPlanBench, tailored to evaluate the efficacy of LLMs in urban planning, which encompasses fundamental principles, professional knowledge, and management and regulations, aligning closely with the qualifications expected of human planners. Through extensive evaluation, we reveal a significant imbalance in the acquisition of planning knowledge among LLMs, with even the most proficient models falling short of meeting professional standards. For instance, we observe that 70% of LLMs achieve subpar performance in understanding planning regulations compared to other aspects. Besides the benchmark, we present the largest-ever supervised fine-tuning (SFT) dataset, UrbanPlanText, comprising over 30,000 instruction pairs sourced from urban planning exams and textbooks. Our findings demonstrate that fine-tuned models exhibit enhanced performance in memorization tests and comprehension of urban planning knowledge, while there exists significant room for improvement, particularly in tasks requiring domain-specific terminology and reasoning. By making our benchmark, dataset, and associated evaluation and fine-tuning toolsets publicly available at this https URL, we aim to catalyze the integration of LLMs into practical urban planning, fostering a symbiotic collaboration between human expertise and machine intelligence. 

---
# Creating and Evaluating Code-Mixed Nepali-English and Telugu-English Datasets for Abusive Language Detection Using Traditional and Deep Learning Models 

**Authors**: Manish Pandey, Nageshwar Prasad Yadav, Mokshada Adduru, Sawan Rai  

**Link**: [PDF](https://arxiv.org/pdf/2504.21026)  

**Abstract**: With the growing presence of multilingual users on social media, detecting abusive language in code-mixed text has become increasingly challenging. Code-mixed communication, where users seamlessly switch between English and their native languages, poses difficulties for traditional abuse detection models, as offensive content may be context-dependent or obscured by linguistic blending. While abusive language detection has been extensively explored for high-resource languages like English and Hindi, low-resource languages such as Telugu and Nepali remain underrepresented, leaving gaps in effective moderation. In this study, we introduce a novel, manually annotated dataset of 2 thousand Telugu-English and 5 Nepali-English code-mixed comments, categorized as abusive and non-abusive, collected from various social media platforms. The dataset undergoes rigorous preprocessing before being evaluated across multiple Machine Learning (ML), Deep Learning (DL), and Large Language Models (LLMs). We experimented with models including Logistic Regression, Random Forest, Support Vector Machines (SVM), Neural Networks (NN), LSTM, CNN, and LLMs, optimizing their performance through hyperparameter tuning, and evaluate it using 10-fold cross-validation and statistical significance testing (t-test). Our findings provide key insights into the challenges of detecting abusive language in code-mixed settings and offer a comparative analysis of computational approaches. This study contributes to advancing NLP for low-resource languages by establishing benchmarks for abusive language detection in Telugu-English and Nepali-English code-mixed text. The dataset and insights can aid in the development of more robust moderation strategies for multilingual social media environments. 

---
# Durghotona GPT: A Web Scraping and Large Language Model Based Framework to Generate Road Accident Dataset Automatically in Bangladesh 

**Authors**: MD Thamed Bin Zaman Chowdhury, Moazzem Hossain, Md. Ridwanul Islam  

**Link**: [PDF](https://arxiv.org/pdf/2504.21025)  

**Abstract**: Road accidents pose significant concerns globally. They lead to large financial losses, injuries, disabilities, and societal challenges. Accurate and timely accident data is essential for predicting and mitigating these events. This paper presents a novel framework named 'Durghotona GPT' that integrates web scraping and Large Language Models (LLMs) to automate the generation of comprehensive accident datasets from prominent national dailies in Bangladesh. The authors collected accident reports from three major newspapers: Prothom Alo, Dhaka Tribune, and The Daily Star. The collected news was then processed using the newest available LLMs: GPT-4, GPT-3.5, and Llama-3. The framework efficiently extracts relevant information, categorizes reports, and compiles detailed datasets. Thus, this framework overcomes limitations of manual data collection methods such as delays, errors, and communication gaps. The authors' evaluation demonstrates that Llama-3, an open-source model, performs comparably to GPT-4. It achieved 89% accuracy in the authors' evaluation. Therefore, it can be considered a cost-effective alternative for similar tasks. The results suggest that the framework developed by the authors can drastically enhance the quality and availability of accident data. As a result, it can support critical applications in traffic safety analysis, urban planning, and public health. The authors also developed an interface for 'Durghotona GPT' for ease of use as part of this paper. Future work will focus on expanding data collection methods and refining LLMs to further increase dataset accuracy and applicability. 

---
# WebEvolver: Enhancing Web Agent Self-Improvement with Coevolving World Model 

**Authors**: Tianqing Fang, Hongming Zhang, Zhisong Zhang, Kaixin Ma, Wenhao Yu, Haitao Mi, Dong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2504.21024)  

**Abstract**: Agent self-improvement, where the backbone Large Language Model (LLM) of the agent are trained on trajectories sampled autonomously based on their own policies, has emerged as a promising approach for enhancing performance. Recent advancements, particularly in web environments, face a critical limitation: their performance will reach a stagnation point during autonomous learning cycles, hindering further improvement. We argue that this stems from limited exploration of the web environment and insufficient exploitation of pre-trained web knowledge in LLMs. To improve the performance of self-improvement, we propose a novel framework that introduces a co-evolving World Model LLM. This world model predicts the next observation based on the current observation and action within the web environment. Leveraging LLMs' pretrained knowledge of abundant web content, the World Model serves dual roles: (1) as a virtual web server generating self-instructed training data to continuously refine the agent's policy, and (2) as an imagination engine during inference, enabling look-ahead simulation to guide action selection for the agent LLM. Experiments in real-world web environments (Mind2Web-Live, WebVoyager, and GAIA-web) show a 10% performance gain over existing self-evolving agents, demonstrating the efficacy and generalizability of our approach, without using any distillation from more powerful close-sourced models. Our work establishes the necessity of integrating world models into autonomous agent frameworks to unlock sustained adaptability. 

---
# Param$Δ$ for Direct Weight Mixing: Post-Train Large Language Model at Zero Cost 

**Authors**: Sheng Cao, Mingrui Wu, Karthik Prasad, Yuandong Tian, Zechun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.21023)  

**Abstract**: The post-training phase of large language models is essential for enhancing capabilities such as instruction-following, reasoning, and alignment with human preferences. However, it demands extensive high-quality data and poses risks like overfitting, alongside significant computational costs due to repeated post-training and evaluation after each base model update. This paper introduces $Param\Delta$, a novel method that streamlines post-training by transferring knowledge from an existing post-trained model to a newly updated base model with ZERO additional training. By computing the difference between post-trained model weights ($\Theta_\text{post}$) and base model weights ($\Theta_\text{base}$), and adding this to the updated base model ($\Theta'_\text{base}$), we define $Param\Delta$ Model as: $\Theta_{\text{Param}\Delta} = \Theta_\text{post} - \Theta_\text{base} + \Theta'_\text{base}$. This approach surprisingly equips the new base model with post-trained capabilities, achieving performance comparable to direct post-training. We did analysis on LLama3, Llama3.1, Qwen, and DeepSeek-distilled models. Results indicate $Param\Delta$ Model effectively replicates traditional post-training. For example, the $Param\Delta$ Model obtained from 70B Llama3-inst, Llama3-base, Llama3.1-base models attains approximately 95\% of Llama3.1-inst model's performance on average. $Param\Delta$ brings a new perspective on how to fully leverage models in the open-weight community, where checkpoints for base and instruct models are readily available and frequently updated, by providing a cost-free framework to accelerate the iterative cycle of model development. 

---
# ConformalNL2LTL: Translating Natural Language Instructions into Temporal Logic Formulas with Conformal Correctness Guarantees 

**Authors**: Jun Wang, David Smith Sundarsingh, Jyotirmoy V. Deshmukh, Yiannis Kantaros  

**Link**: [PDF](https://arxiv.org/pdf/2504.21022)  

**Abstract**: Linear Temporal Logic (LTL) has become a prevalent specification language for robotic tasks. To mitigate the significant manual effort and expertise required to define LTL-encoded tasks, several methods have been proposed for translating Natural Language (NL) instructions into LTL formulas, which, however, lack correctness guarantees. To address this, we introduce a new NL-to-LTL translation method, called ConformalNL2LTL, that can achieve user-defined translation success rates over unseen NL commands. Our method constructs LTL formulas iteratively by addressing a sequence of open-vocabulary Question-Answering (QA) problems with LLMs. To enable uncertainty-aware translation, we leverage conformal prediction (CP), a distribution-free uncertainty quantification tool for black-box models. CP enables our method to assess the uncertainty in LLM-generated answers, allowing it to proceed with translation when sufficiently confident and request help otherwise. We provide both theoretical and empirical results demonstrating that ConformalNL2LTL achieves user-specified translation accuracy while minimizing help rates. 

---
# Context-Enhanced Contrastive Search for Improved LLM Text Generation 

**Authors**: Jaydip Sen, Rohit Pandey, Hetvi Waghela  

**Link**: [PDF](https://arxiv.org/pdf/2504.21020)  

**Abstract**: Recently, Large Language Models (LLMs) have demonstrated remarkable advancements in Natural Language Processing (NLP). However, generating high-quality text that balances coherence, diversity, and relevance remains challenging. Traditional decoding methods, such as bean search and top-k sampling, often struggle with either repetitive or incoherent outputs, particularly in tasks that require long-form text generation. To address these limitations, the paper proposes a novel enhancement of the well-known Contrastive Search algorithm, Context-Enhanced Contrastive Search (CECS) with contextual calibration. The proposed scheme introduces several novelties including dynamic contextual importance weighting, multi-level Contrastive Search, and adaptive temperature control, to optimize the balance between fluency, creativity, and precision. The performance of CECS is evaluated using several standard metrics such as BLEU, ROUGE, and semantic similarity. Experimental results demonstrate significant improvements in both coherence and relevance of the generated texts by CECS outperforming the existing Contrastive Search techniques. The proposed algorithm has several potential applications in the real world including legal document drafting, customer service chatbots, and content marketing. 

---
# Kill two birds with one stone: generalized and robust AI-generated text detection via dynamic perturbations 

**Authors**: Yinghan Zhou, Juan Wen, Wanli Peng, Yiming Xue, Ziwei Zhang, Zhengxian Wu  

**Link**: [PDF](https://arxiv.org/pdf/2504.21019)  

**Abstract**: The growing popularity of large language models has raised concerns regarding the potential to misuse AI-generated text (AIGT). It becomes increasingly critical to establish an excellent AIGT detection method with high generalization and robustness. However, existing methods either focus on model generalization or concentrate on robustness. The unified mechanism, to simultaneously address the challenges of generalization and robustness, is less explored. In this paper, we argue that robustness can be view as a specific form of domain shift, and empirically reveal an intrinsic mechanism for model generalization of AIGT detection task. Then, we proposed a novel AIGT detection method (DP-Net) via dynamic perturbations introduced by a reinforcement learning with elaborated reward and action. Experimentally, extensive results show that the proposed DP-Net significantly outperforms some state-of-the-art AIGT detection methods for generalization capacity in three cross-domain scenarios. Meanwhile, the DP-Net achieves best robustness under two text adversarial attacks. The code is publicly available at this https URL. 

---
# HYPEROFA: Expanding LLM Vocabulary to New Languages via Hypernetwork-Based Embedding Initialization 

**Authors**: Enes Özeren, Yihong Liu, Hinrich Schütze  

**Link**: [PDF](https://arxiv.org/pdf/2504.21018)  

**Abstract**: Many pre-trained language models (PLMs) exhibit suboptimal performance on mid- and low-resource languages, largely due to limited exposure to these languages during pre-training. A common strategy to address this is to introduce new tokens specific to the target languages, initialize their embeddings, and apply continual pre-training on target-language data. Among such methods, OFA (Liu et al., 2024a) proposes a similarity-based subword embedding initialization heuristic that is both effective and efficient. However, OFA restricts target-language token embeddings to be convex combinations of a fixed number of source-language embeddings, which may limit expressiveness. To overcome this limitation, we propose HYPEROFA, a hypernetwork-based approach for more adaptive token embedding initialization. The hypernetwork is trained to map from an external multilingual word vector space to the PLMs token embedding space using source-language tokens. Once trained, it can generate flexible embeddings for target-language tokens, serving as a good starting point for continual pretraining. Experiments demonstrate that HYPEROFA consistently outperforms random initialization baseline and matches or exceeds the performance of OFA in both continual pre-training convergence and downstream task performance. We make the code publicly available. 

---
# ViQA-COVID: COVID-19 Machine Reading Comprehension Dataset for Vietnamese 

**Authors**: Hai-Chung Nguyen-Phung, Ngoc C. Lê, Van-Chien Nguyen, Hang Thi Nguyen, Thuy Phuong Thi Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2504.21017)  

**Abstract**: After two years of appearance, COVID-19 has negatively affected people and normal life around the world. As in May 2022, there are more than 522 million cases and six million deaths worldwide (including nearly ten million cases and over forty-three thousand deaths in Vietnam). Economy and society are both severely affected. The variant of COVID-19, Omicron, has broken disease prevention measures of countries and rapidly increased number of infections. Resources overloading in treatment and epidemics prevention is happening all over the world. It can be seen that, application of artificial intelligence (AI) to support people at this time is extremely necessary. There have been many studies applying AI to prevent COVID-19 which are extremely useful, and studies on machine reading comprehension (MRC) are also in it. Realizing that, we created the first MRC dataset about COVID-19 for Vietnamese: ViQA-COVID and can be used to build models and systems, contributing to disease prevention. Besides, ViQA-COVID is also the first multi-span extraction MRC dataset for Vietnamese, we hope that it can contribute to promoting MRC studies in Vietnamese and multilingual. 

---
# Nested Named-Entity Recognition on Vietnamese COVID-19: Dataset and Experiments 

**Authors**: Ngoc C.Lê, Hai-Chung Nguyen-Phung, Thu-Huong Pham Thi, Hue Vu, Phuong-Thao Nguyen Thi, Thu-Thuy Tran, Hong-Nhung Le Thi, Thuy-Duong Nguyen-Thi, Thanh-Huy Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2504.21016)  

**Abstract**: The COVID-19 pandemic caused great losses worldwide, efforts are taken place to prevent but many countries have failed. In Vietnam, the traceability, localization, and quarantine of people who contact with patients contribute to effective disease prevention. However, this is done by hand, and take a lot of work. In this research, we describe a named-entity recognition (NER) study that assists in the prevention of COVID-19 pandemic in Vietnam. We also present our manually annotated COVID-19 dataset with nested named entity recognition task for Vietnamese which be defined new entity types using for our system. 

---
# Analyzing Feedback Mechanisms in AI-Generated MCQs: Insights into Readability, Lexical Properties, and Levels of Challenge 

**Authors**: Antoun Yaacoub, Zainab Assaghir, Lionel Prevost, Jérôme Da-Rugna  

**Link**: [PDF](https://arxiv.org/pdf/2504.21013)  

**Abstract**: Artificial Intelligence (AI)-generated feedback in educational settings has garnered considerable attention due to its potential to enhance learning outcomes. However, a comprehensive understanding of the linguistic characteristics of AI-generated feedback, including readability, lexical richness, and adaptability across varying challenge levels, remains limited. This study delves into the linguistic and structural attributes of feedback generated by Google's Gemini 1.5-flash text model for computer science multiple-choice questions (MCQs). A dataset of over 1,200 MCQs was analyzed, considering three difficulty levels (easy, medium, hard) and three feedback tones (supportive, neutral, challenging). Key linguistic metrics, such as length, readability scores (Flesch-Kincaid Grade Level), vocabulary richness, and lexical density, were computed and examined. A fine-tuned RoBERTa-based multi-task learning (MTL) model was trained to predict these linguistic properties, achieving a Mean Absolute Error (MAE) of 2.0 for readability and 0.03 for vocabulary richness. The findings reveal significant interaction effects between feedback tone and question difficulty, demonstrating the dynamic adaptation of AI-generated feedback within diverse educational contexts. These insights contribute to the development of more personalized and effective AI-driven feedback mechanisms, highlighting the potential for improved learning outcomes while underscoring the importance of ethical considerations in their design and deployment. 

---
# Waking Up an AI: A Quantitative Framework for Prompt-Induced Phase Transition in Large Language Models 

**Authors**: Makoto Sato  

**Link**: [PDF](https://arxiv.org/pdf/2504.21012)  

**Abstract**: What underlies intuitive human thinking? One approach to this question is to compare the cognitive dynamics of humans and large language models (LLMs). However, such a comparison requires a method to quantitatively analyze AI cognitive behavior under controlled conditions. While anecdotal observations suggest that certain prompts can dramatically change LLM behavior, these observations have remained largely qualitative. Here, we propose a two-part framework to investigate this phenomenon: a Transition-Inducing Prompt (TIP) that triggers a rapid shift in LLM responsiveness, and a Transition Quantifying Prompt (TQP) that evaluates this change using a separate LLM. Through controlled experiments, we examined how LLMs react to prompts embedding two semantically distant concepts (e.g., mathematical aperiodicity and traditional crafts)--either fused together or presented separately--by changing their linguistic quality and affective tone. Whereas humans tend to experience heightened engagement when such concepts are meaningfully blended producing a novel concept--a form of conceptual fusion--current LLMs showed no significant difference in responsiveness between semantically fused and non-fused prompts. This suggests that LLMs may not yet replicate the conceptual integration processes seen in human intuition. Our method enables fine-grained, reproducible measurement of cognitive responsiveness, and may help illuminate key differences in how intuition and conceptual leaps emerge in artificial versus human minds. 

---
# SWE-smith: Scaling Data for Software Engineering Agents 

**Authors**: John Yang, Kilian Leret, Carlos E. Jimenez, Alexander Wettig, Kabir Khandpur, Yanzhe Zhang, Binyuan Hui, Ofir Press, Ludwig Schmidt, Diyi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.21798)  

**Abstract**: Despite recent progress in Language Models (LMs) for software engineering, collecting training data remains a significant pain point. Existing datasets are small, with at most 1,000s of training instances from 11 or fewer GitHub repositories. The procedures to curate such datasets are often complex, necessitating hundreds of hours of human labor; companion execution environments also take up several terabytes of storage, severely limiting their scalability and usability. To address this pain point, we introduce SWE-smith, a novel pipeline for generating software engineering training data at scale. Given any Python codebase, SWE-smith constructs a corresponding execution environment, then automatically synthesizes 100s to 1,000s of task instances that break existing test(s) in the codebase. Using SWE-smith, we create a dataset of 50k instances sourced from 128 GitHub repositories, an order of magnitude larger than all previous works. We train SWE-agent-LM-32B, achieving 40.2% Pass@1 resolve rate on the SWE-bench Verified benchmark, state of the art among open source models. We open source SWE-smith (collection procedure, task instances, trajectories, models) to lower the barrier of entry for research in LM systems for automated software engineering. All assets available at this https URL. 

---
# CodeFlowBench: A Multi-turn, Iterative Benchmark for Complex Code Generation 

**Authors**: Sizhe Wang, Zhengren Wang, Dongsheng Ma, Yongan Yu, Rui Ling, Zhiyu Li, Feiyu Xiong, Wentao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.21751)  

**Abstract**: Real world development demands code that is readable, extensible, and testable by organizing the implementation into modular components and iteratively reuse pre-implemented code. We term this iterative, multi-turn process codeflow and introduce CodeFlowBench, the first benchmark designed for comprehensively evaluating LLMs' ability to perform codeflow, namely to implement new functionality by reusing existing functions over multiple turns. CodeFlowBench comprises 5258 problems drawn from Codeforces and is continuously updated via an automated pipeline that decomposes each problem into a series of function-level subproblems based on its dependency tree and each subproblem is paired with unit tests. We further propose a novel evaluation framework with tasks and metrics tailored to multi-turn code reuse to assess model performance. In experiments across various LLMs under both multi-turn and single-turn patterns. We observe models' poor performance on CodeFlowBench, with a substantial performance drop in the iterative codeflow scenario. For instance, o1-mini achieves a pass@1 of 20.8% in multi-turn pattern versus 37.8% in single-turn pattern. Further analysis shows that different models excel at different dependency depths, yet all struggle to correctly solve structurally complex problems, highlighting challenges for current LLMs to serve as code generation tools when performing codeflow. Overall, CodeFlowBench offers a comprehensive benchmark and new insights into LLM capabilities for multi-turn, iterative code generation, guiding future advances in code generation tasks. 

---
# LLM-Empowered Embodied Agent for Memory-Augmented Task Planning in Household Robotics 

**Authors**: Marc Glocker, Peter Hönig, Matthias Hirschmanner, Markus Vincze  

**Link**: [PDF](https://arxiv.org/pdf/2504.21716)  

**Abstract**: We present an embodied robotic system with an LLM-driven agent-orchestration architecture for autonomous household object management. The system integrates memory-augmented task planning, enabling robots to execute high-level user commands while tracking past actions. It employs three specialized agents: a routing agent, a task planning agent, and a knowledge base agent, each powered by task-specific LLMs. By leveraging in-context learning, our system avoids the need for explicit model training. RAG enables the system to retrieve context from past interactions, enhancing long-term object tracking. A combination of Grounded SAM and LLaMa3.2-Vision provides robust object detection, facilitating semantic scene understanding for task planning. Evaluation across three household scenarios demonstrates high task planning accuracy and an improvement in memory recall due to RAG. Specifically, Qwen2.5 yields best performance for specialized agents, while LLaMA3.1 excels in routing tasks. The source code is available at: this https URL. 

---
# AdaR1: From Long-CoT to Hybrid-CoT via Bi-Level Adaptive Reasoning Optimization 

**Authors**: Haotian Luo, Haiying He, Yibo Wang, Jinluan Yang, Rui Liu, Naiqiang Tan, Xiaochun Cao, Dacheng Tao, Li Shen  

**Link**: [PDF](https://arxiv.org/pdf/2504.21659)  

**Abstract**: Recently, long-thought reasoning models achieve strong performance on complex reasoning tasks, but often incur substantial inference overhead, making efficiency a critical concern. Our empirical analysis reveals that the benefit of using Long-CoT varies across problems: while some problems require elaborate reasoning, others show no improvement, or even degraded accuracy. This motivates adaptive reasoning strategies that tailor reasoning depth to the input. However, prior work primarily reduces redundancy within long reasoning paths, limiting exploration of more efficient strategies beyond the Long-CoT paradigm. To address this, we propose a novel two-stage framework for adaptive and efficient reasoning. First, we construct a hybrid reasoning model by merging long and short CoT models to enable diverse reasoning styles. Second, we apply bi-level preference training to guide the model to select suitable reasoning styles (group-level), and prefer concise and correct reasoning within each style group (instance-level). Experiments demonstrate that our method significantly reduces inference costs compared to other baseline approaches, while maintaining performance. Notably, on five mathematical datasets, the average length of reasoning is reduced by more than 50%, highlighting the potential of adaptive strategies to optimize reasoning efficiency in large language models. Our code is coming soon at this https URL 

---
# Glucagon and insulin production in pancreatic cells modeled using Petri nets and Boolean networks 

**Authors**: Kamila Barylska, Frank Delaplace, Anna Gogolińska, Ewa Pańkowska  

**Link**: [PDF](https://arxiv.org/pdf/2504.21578)  

**Abstract**: Diabetes is a civilization chronic disease characterized by a constant elevated concentration of glucose in the blood. Many processes are involved in the glucose regulation, and their interactions are very complex. To better understand those processes we set ourselves a goal to create a Petri net model of the glucose regulation in the whole body. So far we have managed to create a model of glycolysis and synthesis of glucose in the liver, and the general overview models of the glucose regulation in a healthy and diabetic person. In this paper we introduce Petri nets models of insulin secretion in beta cell of the pancreas, and glucagon in the pancreas alpha cells. Those two hormones have mutually opposite effects: insulin preventing hyperglycemia, and glucagon preventing hypoglycemia. Understanding the mechanisms of insulin and glucagon secretion constitutes the basis for understanding diabetes. We also present a model in which both processes occur together, depending on the blood glucose level. The dynamics of each model is analysed. Additionally, we transform the overall insulin and glucagon secretion system to a Boolean network, following standard transformation rules. 

---
# Black-Box Visual Prompt Engineering for Mitigating Object Hallucination in Large Vision Language Models 

**Authors**: Sangmin Woo, Kang Zhou, Yun Zhou, Shuai Wang, Sheng Guan, Haibo Ding, Lin Lee Cheong  

**Link**: [PDF](https://arxiv.org/pdf/2504.21559)  

**Abstract**: Large Vision Language Models (LVLMs) often suffer from object hallucination, which undermines their reliability. Surprisingly, we find that simple object-based visual prompting -- overlaying visual cues (e.g., bounding box, circle) on images -- can significantly mitigate such hallucination; however, different visual prompts (VPs) vary in effectiveness. To address this, we propose Black-Box Visual Prompt Engineering (BBVPE), a framework to identify optimal VPs that enhance LVLM responses without needing access to model internals. Our approach employs a pool of candidate VPs and trains a router model to dynamically select the most effective VP for a given input image. This black-box approach is model-agnostic, making it applicable to both open-source and proprietary LVLMs. Evaluations on benchmarks such as POPE and CHAIR demonstrate that BBVPE effectively reduces object hallucination. 

---
# SeriesBench: A Benchmark for Narrative-Driven Drama Series Understanding 

**Authors**: Chenkai Zhang, Yiming Lei, Zeming Liu, Haitao Leng, ShaoGuo Liu, Tingting Gao, Qingjie Liu, Yunhong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.21435)  

**Abstract**: With the rapid development of Multi-modal Large Language Models (MLLMs), an increasing number of benchmarks have been established to evaluate the video understanding capabilities of these models. However, these benchmarks focus on \textbf{standalone} videos and mainly assess ``visual elements'' like human actions and object states. In reality, contemporary videos often encompass complex and continuous narratives, typically presented as a \textbf{series}. To address this challenge, we propose \textbf{SeriesBench}, a benchmark consisting of 105 carefully curated narrative-driven series, covering 28 specialized tasks that require deep narrative understanding. Specifically, we first select a diverse set of drama series spanning various genres. Then, we introduce a novel long-span narrative annotation method, combined with a full-information transformation approach to convert manual annotations into diverse task formats. To further enhance model capacity for detailed analysis of plot structures and character relationships within series, we propose a novel narrative reasoning framework, \textbf{PC-DCoT}. Extensive results on \textbf{SeriesBench} indicate that existing MLLMs still face significant challenges in understanding narrative-driven series, while \textbf{PC-DCoT} enables these MLLMs to achieve performance improvements. Overall, our \textbf{SeriesBench} and \textbf{PC-DCoT} highlight the critical necessity of advancing model capabilities to understand narrative-driven series, guiding the future development of MLLMs. SeriesBench is publicly available at this https URL. 

---
# Who Gets the Callback? Generative AI and Gender Bias 

**Authors**: Sugat Chaturvedi, Rochana Chaturvedi  

**Link**: [PDF](https://arxiv.org/pdf/2504.21400)  

**Abstract**: Generative artificial intelligence (AI), particularly large language models (LLMs), is being rapidly deployed in recruitment and for candidate shortlisting. We audit several mid-sized open-source LLMs for gender bias using a dataset of 332,044 real-world online job postings. For each posting, we prompt the model to recommend whether an equally qualified male or female candidate should receive an interview callback. We find that most models tend to favor men, especially for higher-wage roles. Mapping job descriptions to the Standard Occupational Classification system, we find lower callback rates for women in male-dominated occupations and higher rates in female-associated ones, indicating occupational segregation. A comprehensive analysis of linguistic features in job ads reveals strong alignment of model recommendations with traditional gender stereotypes. To examine the role of recruiter identity, we steer model behavior by infusing Big Five personality traits and simulating the perspectives of historical figures. We find that less agreeable personas reduce stereotyping, consistent with an agreeableness bias in LLMs. Our findings highlight how AI-driven hiring may perpetuate biases in the labor market and have implications for fairness and diversity within firms. 

---
# Phi-4-reasoning Technical Report 

**Authors**: Marah Abdin, Sahaj Agarwal, Ahmed Awadallah, Vidhisha Balachandran, Harkirat Behl, Lingjiao Chen, Gustavo de Rosa, Suriya Gunasekar, Mojan Javaheripi, Neel Joshi, Piero Kauffmann, Yash Lara, Caio César Teodoro Mendes, Arindam Mitra, Besmira Nushi, Dimitris Papailiopoulos, Olli Saarikivi, Shital Shah, Vaishnavi Shrivastava, Vibhav Vineet, Yue Wu, Safoora Yousefi, Guoqing Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2504.21318)  

**Abstract**: We introduce Phi-4-reasoning, a 14-billion parameter reasoning model that achieves strong performance on complex reasoning tasks. Trained via supervised fine-tuning of Phi-4 on carefully curated set of "teachable" prompts-selected for the right level of complexity and diversity-and reasoning demonstrations generated using o3-mini, Phi-4-reasoning generates detailed reasoning chains that effectively leverage inference-time compute. We further develop Phi-4-reasoning-plus, a variant enhanced through a short phase of outcome-based reinforcement learning that offers higher performance by generating longer reasoning traces. Across a wide range of reasoning tasks, both models outperform significantly larger open-weight models such as DeepSeek-R1-Distill-Llama-70B model and approach the performance levels of full DeepSeek-R1 model. Our comprehensive evaluations span benchmarks in math and scientific reasoning, coding, algorithmic problem solving, planning, and spatial understanding. Interestingly, we observe a non-trivial transfer of improvements to general-purpose benchmarks as well. In this report, we provide insights into our training data, our training methodologies, and our evaluations. We show that the benefit of careful data curation for supervised fine-tuning (SFT) extends to reasoning language models, and can be further amplified by reinforcement learning (RL). Finally, our evaluation points to opportunities for improving how we assess the performance and robustness of reasoning models. 

---
# Multimodal Large Language Models for Medicine: A Comprehensive Survey 

**Authors**: Jiarui Ye, Hao Tang  

**Link**: [PDF](https://arxiv.org/pdf/2504.21051)  

**Abstract**: MLLMs have recently become a focal point in the field of artificial intelligence research. Building on the strong capabilities of LLMs, MLLMs are adept at addressing complex multi-modal tasks. With the release of GPT-4, MLLMs have gained substantial attention from different domains. Researchers have begun to explore the potential of MLLMs in the medical and healthcare domain. In this paper, we first introduce the background and fundamental concepts related to LLMs and MLLMs, while emphasizing the working principles of MLLMs. Subsequently, we summarize three main directions of application within healthcare: medical reporting, medical diagnosis, and medical treatment. Our findings are based on a comprehensive review of 330 recent papers in this area. We illustrate the remarkable capabilities of MLLMs in these domains by providing specific examples. For data, we present six mainstream modes of data along with their corresponding evaluation benchmarks. At the end of the survey, we discuss the challenges faced by MLLMs in the medical and healthcare domain and propose feasible methods to mitigate or overcome these issues. 

---
# A False Sense of Privacy: Evaluating Textual Data Sanitization Beyond Surface-level Privacy Leakage 

**Authors**: Rui Xin, Niloofar Mireshghallah, Shuyue Stella Li, Michael Duan, Hyunwoo Kim, Yejin Choi, Yulia Tsvetkov, Sewoong Oh, Pang Wei Koh  

**Link**: [PDF](https://arxiv.org/pdf/2504.21035)  

**Abstract**: Sanitizing sensitive text data typically involves removing personally identifiable information (PII) or generating synthetic data under the assumption that these methods adequately protect privacy; however, their effectiveness is often only assessed by measuring the leakage of explicit identifiers but ignoring nuanced textual markers that can lead to re-identification. We challenge the above illusion of privacy by proposing a new framework that evaluates re-identification attacks to quantify individual privacy risks upon data release. Our approach shows that seemingly innocuous auxiliary information -- such as routine social activities -- can be used to infer sensitive attributes like age or substance use history from sanitized data. For instance, we demonstrate that Azure's commercial PII removal tool fails to protect 74\% of information in the MedQA dataset. Although differential privacy mitigates these risks to some extent, it significantly reduces the utility of the sanitized text for downstream tasks. Our findings indicate that current sanitization techniques offer a \textit{false sense of privacy}, highlighting the need for more robust methods that protect against semantic-level information leakage. 

---
# Don't Retrieve, Generate: Prompting LLMs for Synthetic Training Data in Dense Retrieval 

**Authors**: Aarush Sinha  

**Link**: [PDF](https://arxiv.org/pdf/2504.21015)  

**Abstract**: Training effective dense retrieval models often relies on hard negative (HN) examples mined from the document corpus via methods like BM25 or cross-encoders (CE), processes that can be computationally demanding and require full corpus access. This paper introduces a different approach, an end-to-end pipeline where a Large Language Model (LLM) first generates a query from a passage, and then generates a hard negative example using \emph{only} that query text. This corpus-free negative generation contrasts with standard mining techniques. We evaluated this \textsc{LLM Query $\rightarrow$ LLM HN} approach against traditional \textsc{LLM Query $\rightarrow$ BM25 HN} and \textsc{LLM Query $\rightarrow$ CE HN} pipelines using E5-Base and GTE-Base models on several BEIR benchmark datasets. Our results show the proposed all-LLM pipeline achieves performance identical to both the BM25 and the computationally intensive CE baselines across nDCG@10, Precision@10, and Recall@100 metrics. This demonstrates that our corpus-free negative generation method matches the effectiveness of complex, corpus-dependent mining techniques, offering a potentially simpler and more efficient pathway for training high-performance retrievers without sacrificing results. We make the dataset including the queries and the hard-negatives for all three methods publicly available this https URL. 

---
