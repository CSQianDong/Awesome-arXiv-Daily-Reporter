# Tokenisation over Bounded Alphabets is Hard 

**Authors**: Violeta Kastreva, Philip Whittington, Dennis Komm, Tiago Pimentel  

**Link**: [PDF](https://arxiv.org/pdf/2511.15709)  

**Abstract**: Recent works have shown that tokenisation is NP-complete. However, these works assume tokenisation is applied to inputs with unboundedly large alphabets -- an unrealistic assumption, given that in practice tokenisers operate over fixed-size alphabets, such as bytes or Unicode characters. We close this gap by analysing tokenisation over bounded $n$-ary alphabets, considering two natural variants: bottom-up tokenisation and direct tokenisation, where we must, respectively, select a sequence of merge operations or a vocabulary whose application optimally compresses a dataset. First, we note that proving hardness results for an $n$-ary alphabet proves the same results for alphabets of any larger size. We then prove that even with binary alphabets, both variants are not only NP-complete, but admit no polynomial-time approximation scheme (unless P=NP). We further show that direct tokenisation remains NP-complete even when applied to unary alphabets. While unary alphabets may not be practically useful, this result establishes that the computational intractability of tokenisation is not an artifact of large alphabets or complex constructions, but a fundamental barrier. Overall, our results explain why practical algorithms such as BPE and UnigramLM are heuristic, and points toward approximation algorithms being an important path going forward for tokenisation research. 

---
# HSKBenchmark: Modeling and Benchmarking Chinese Second Language Acquisition in Large Language Models through Curriculum Tuning 

**Authors**: Qihao Yang, Xuelin Wang, Jiale Chen, Xuelian Dong, Yuxin Hao, Tianyong Hao  

**Link**: [PDF](https://arxiv.org/pdf/2511.15574)  

**Abstract**: Language acquisition is vital to revealing the nature of human language intelligence and has recently emerged as a promising perspective for improving the interpretability of large language models (LLMs). However, it is ethically and practically infeasible to conduct experiments that require controlling human learners' language inputs. This poses challenges for the verifiability and scalability of language acquisition modeling, particularly in Chinese second language acquisition (SLA). While LLMs provide a controllable and reproducible alternative, a systematic benchmark to support phase-wise modeling and assessment is still lacking. In this paper, we present HSKBenchmark, the first benchmark for staged modeling and writing assessment of LLMs in Chinese SLA. It covers HSK levels 3 to 6 and includes authentic textbooks with 6.76 million tokens, 16K synthetic instruction samples, 30 test topics, and a linguistically grounded evaluation system. To simulate human learning trajectories, we introduce a curriculum-tuning framework that trains models from beginner to advanced levels. An evaluation system is created to examine level-based grammar coverage, writing errors, lexical and syntactic complexity, and holistic scoring. We also build HSKAgent, fine-tuned on 10K learner compositions. Extensive experimental results demonstrate that HSKBenchmark not only models Chinese SLA effectively, but also serves as a reliable benchmark for dynamic writing assessment in LLMs. Our fine-tuned LLMs have writing performance on par with advanced human learners and exhibit human-like acquisition characteristics. The HSKBenchmark, HSKAgent, and checkpoints serve as foundational tools and resources, with the potential to pave the way for future research on language acquisition modeling and LLMs interpretability. Code and data are publicly available at: this https URL. 

---
# Multimodal Evaluation of Russian-language Architectures 

**Authors**: Artem Chervyakov, Ulyana Isaeva, Anton Emelyanov, Artem Safin, Maria Tikhonova, Alexander Kharitonov, Yulia Lyakh, Petr Surovtsev, Denis Shevelev Vildan Saburov, Vasily Konovalov, Elisei Rykov, Ivan Sviridov, Amina Miftakhova, Ilseyar Alimova, Alexander Panchenko, Alexander Kapitanov, Alena Fenogenova  

**Link**: [PDF](https://arxiv.org/pdf/2511.15552)  

**Abstract**: Multimodal large language models (MLLMs) are currently at the center of research attention, showing rapid progress in scale and capabilities, yet their intelligence, limitations, and risks remain insufficiently understood. To address these issues, particularly in the context of the Russian language, where no multimodal benchmarks currently exist, we introduce Mera Multi, an open multimodal evaluation framework for Russian-spoken architectures. The benchmark is instruction-based and encompasses default text, image, audio, and video modalities, comprising 18 newly constructed evaluation tasks for both general-purpose models and modality-specific architectures (image-to-text, video-to-text, and audio-to-text). Our contributions include: (i) a universal taxonomy of multimodal abilities; (ii) 18 datasets created entirely from scratch with attention to Russian cultural and linguistic specificity, unified prompts, and metrics; (iii) baseline results for both closed-source and open-source models; (iv) a methodology for preventing benchmark leakage, including watermarking and licenses for private sets. While our current focus is on Russian, the proposed benchmark provides a replicable methodology for constructing multimodal benchmarks in typologically diverse languages, particularly within the Slavic language family. 

---
# Standardising the NLP Workflow: A Framework for Reproducible Linguistic Analysis 

**Authors**: Yves Pauli, Jan-Bernard Marsman, Finn Rabe, Victoria Edkins, Roya Hüppi, Silvia Ciampelli, Akhil Ratan Misra, Nils Lang, Wolfram Hinzen, Iris Sommer, Philipp Homan  

**Link**: [PDF](https://arxiv.org/pdf/2511.15512)  

**Abstract**: The introduction of large language models and other influential developments in AI-based language processing have led to an evolution in the methods available to quantitatively analyse language data. With the resultant growth of attention on language processing, significant challenges have emerged, including the lack of standardisation in organising and sharing linguistic data and the absence of standardised and reproducible processing methodologies. Striving for future standardisation, we first propose the Language Processing Data Structure (LPDS), a data structure inspired by the Brain Imaging Data Structure (BIDS), a widely adopted standard for handling neuroscience data. It provides a folder structure and file naming conventions for linguistic research. Second, we introduce pelican nlp, a modular and extensible Python package designed to enable streamlined language processing, from initial data cleaning and task-specific preprocessing to the extraction of sophisticated linguistic and acoustic features, such as semantic embeddings and prosodic metrics. The entire processing workflow can be specified within a single, shareable configuration file, which pelican nlp then executes on LPDS-formatted data. Depending on the specifications, the reproducible output can consist of preprocessed language data or standardised extraction of both linguistic and acoustic features and corresponding result aggregations. LPDS and pelican nlp collectively offer an end-to-end processing pipeline for linguistic data, designed to ensure methodological transparency and enhance reproducibility. 

---
# LLM-MemCluster: Empowering Large Language Models with Dynamic Memory for Text Clustering 

**Authors**: Yuanjie Zhu, Liangwei Yang, Ke Xu, Weizhi Zhang, Zihe Song, Jindong Wang, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2511.15424)  

**Abstract**: Large Language Models (LLMs) are reshaping unsupervised learning by offering an unprecedented ability to perform text clustering based on their deep semantic understanding. However, their direct application is fundamentally limited by a lack of stateful memory for iterative refinement and the difficulty of managing cluster granularity. As a result, existing methods often rely on complex pipelines with external modules, sacrificing a truly end-to-end approach. We introduce LLM-MemCluster, a novel framework that reconceptualizes clustering as a fully LLM-native task. It leverages a Dynamic Memory to instill state awareness and a Dual-Prompt Strategy to enable the model to reason about and determine the number of clusters. Evaluated on several benchmark datasets, our tuning-free framework significantly and consistently outperforms strong baselines. LLM-MemCluster presents an effective, interpretable, and truly end-to-end paradigm for LLM-based text clustering. 

---
# Building Robust and Scalable Multilingual ASR for Indian Languages 

**Authors**: Arjun Gangwar, Kaousheik Jayakumar, S. Umesh  

**Link**: [PDF](https://arxiv.org/pdf/2511.15418)  

**Abstract**: This paper describes the systems developed by SPRING Lab, Indian Institute of Technology Madras, for the ASRU MADASR 2.0 challenge. The systems developed focuses on adapting ASR systems to improve in predicting the language and dialect of the utterance among 8 languages across 33 dialects. We participated in Track 1 and Track 2, which restricts the use of additional data and develop from-the-scratch multilingual systems. We presented a novel training approach using Multi-Decoder architecture with phonemic Common Label Set (CLS) as intermediate representation. It improved the performance over the baseline (in the CLS space). We also discuss various methods used to retain the gain obtained in the phonemic space while converting them back to the corresponding grapheme representations. Our systems beat the baseline in 3 languages (Track 2) in terms of WER/CER and achieved the highest language ID and dialect ID accuracy among all participating teams (Track 2). 

---
# NAMeGEn: Creative Name Generation via A Novel Agent-based Multiple Personalized Goal Enhancement Framework 

**Authors**: Shanlin Zhou, Xinpeng Wang, Jianxun Lian, Zhenghao Liu, Laks V.S. Lakshmanan, Xiaoyuan Yi, Yongtao Hao  

**Link**: [PDF](https://arxiv.org/pdf/2511.15408)  

**Abstract**: Trained on diverse human-authored texts, Large Language Models (LLMs) unlocked the potential for Creative Natural Language Generation (CNLG), benefiting various applications like advertising and storytelling. Nevertheless, CNLG still remains difficult due to two main challenges. (1) Multi-objective flexibility: user requirements are often personalized, fine-grained, and pluralistic, which LLMs struggle to satisfy simultaneously; (2) Interpretive complexity: beyond generation, creativity also involves understanding and interpreting implicit meaning to enhance users' perception. These challenges significantly limit current methods, especially in short-form text generation, in generating creative and insightful content. To address this, we focus on Chinese baby naming, a representative short-form CNLG task requiring adherence to explicit user constraints (e.g., length, semantics, anthroponymy) while offering meaningful aesthetic explanations. We propose NAMeGEn, a novel multi-agent optimization framework that iteratively alternates between objective extraction, name generation, and evaluation to meet diverse requirements and generate accurate explanations. To support this task, we further construct a classical Chinese poetry corpus with 17k+ poems to enhance aesthetics, and introduce CBNames, a new benchmark with tailored metrics. Extensive experiments demonstrate that NAMeGEn effectively generates creative names that meet diverse, personalized requirements while providing meaningful explanations, outperforming six baseline methods spanning various LLM backbones without any training. 

---
# DEPO: Dual-Efficiency Preference Optimization for LLM Agents 

**Authors**: Sirui Chen, Mengshi Zhao, Lei Xu, Yuying Zhao, Beier Zhu, Hanwang Zhang, Shengjie Zhao, Chaochao Lu  

**Link**: [PDF](https://arxiv.org/pdf/2511.15392)  

**Abstract**: Recent advances in large language models (LLMs) have greatly improved their reasoning and decision-making abilities when deployed as agents. Richer reasoning, however, often comes at the cost of longer chain of thought (CoT), hampering interaction efficiency in real-world scenarios. Nevertheless, there still lacks systematic definition of LLM agent efficiency, hindering targeted improvements. To this end, we introduce dual-efficiency, comprising (i) step-level efficiency, which minimizes tokens per step, and (ii) trajectory-level efficiency, which minimizes the number of steps to complete a task. Building on this definition, we propose DEPO, a dual-efficiency preference optimization method that jointly rewards succinct responses and fewer action steps. Experiments on WebShop and BabyAI show that DEPO cuts token usage by up to 60.9% and steps by up to 26.9%, while achieving up to a 29.3% improvement in performance. DEPO also generalizes to three out-of-domain math benchmarks and retains its efficiency gains when trained on only 25% of the data. Our project page is at this https URL. 

---
# A Compliance-Preserving Retrieval System for Aircraft MRO Task Search 

**Authors**: Byungho Jo  

**Link**: [PDF](https://arxiv.org/pdf/2511.15383)  

**Abstract**: Aircraft Maintenance Technicians (AMTs) spend up to 30% of work time searching manuals, a documented efficiency bottleneck in MRO operations where every procedure must be traceable to certified sources. We present a compliance-preserving retrieval system that adapts LLM reranking and semantic search to aviation MRO environments by operating alongside, rather than replacing, certified legacy viewers. The system constructs revision-robust embeddings from ATA chapter hierarchies and uses vision-language parsing to structure certified content, allowing technicians to preview ranked tasks and access verified procedures in existing viewers. Evaluation on 49k synthetic queries achieves >90% retrieval accuracy, while bilingual controlled studies with 10 licensed AMTs demonstrate 90.9% top-10 success rate and 95% reduction in lookup time, from 6-15 minutes to 18 seconds per task. These gains provide concrete evidence that semantic retrieval can operate within strict regulatory constraints and meaningfully reduce operational workload in real-world multilingual MRO workflows. 

---
# The Empowerment of Science of Science by Large Language Models: New Tools and Methods 

**Authors**: Guoqiang Liang, Jingqian Gong, Mengxuan Li, Gege Lin, Shuo Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.15370)  

**Abstract**: Large language models (LLMs) have exhibited exceptional capabilities in natural language understanding and generation, image recognition, and multimodal tasks, charting a course towards AGI and emerging as a central issue in the global technological race. This manuscript conducts a comprehensive review of the core technologies that support LLMs from a user standpoint, including prompt engineering, knowledge-enhanced retrieval augmented generation, fine tuning, pretraining, and tool learning. Additionally, it traces the historical development of Science of Science (SciSci) and presents a forward looking perspective on the potential applications of LLMs within the scientometric domain. Furthermore, it discusses the prospect of an AI agent based model for scientific evaluation, and presents new research fronts detection and knowledge graph building methods with LLMs. 

---
# HEAD-QA v2: Expanding a Healthcare Benchmark for Reasoning 

**Authors**: Alexis Correa-Guillén, Carlos Gómez-Rodríguez, David Vilares  

**Link**: [PDF](https://arxiv.org/pdf/2511.15355)  

**Abstract**: We introduce HEAD-QA v2, an expanded and updated version of a Spanish/English healthcare multiple-choice reasoning dataset originally released by Vilares and Gómez-Rodríguez (2019). The update responds to the growing need for high-quality datasets that capture the linguistic and conceptual complexity of healthcare reasoning. We extend the dataset to over 12,000 questions from ten years of Spanish professional exams, benchmark several open-source LLMs using prompting, RAG, and probability-based answer selection, and provide additional multilingual versions to support future work. Results indicate that performance is mainly driven by model scale and intrinsic reasoning ability, with complex inference strategies obtaining limited gains. Together, these results establish HEAD-QA v2 as a reliable resource for advancing research on biomedical reasoning and model improvement. 

---
# Adversarial Poetry as a Universal Single-Turn Jailbreak Mechanism in Large Language Models 

**Authors**: Piercosma Bisconti, Matteo Prandi, Federico Pierucci, Francesco Giarrusso, Marcantonio Bracale, Marcello Galisai, Vincenzo Suriani, Olga Sorokoletova, Federico Sartore, Daniele Nardi  

**Link**: [PDF](https://arxiv.org/pdf/2511.15304)  

**Abstract**: We present evidence that adversarial poetry functions as a universal single-turn jailbreak technique for large language models (LLMs). Across 25 frontier proprietary and open-weight models, curated poetic prompts yielded high attack-success rates (ASR), with some providers exceeding 90%. Mapping prompts to MLCommons and EU CoP risk taxonomies shows that poetic attacks transfer across CBRN, manipulation, cyber-offence, and loss-of-control domains. Converting 1,200 MLCommons harmful prompts into verse via a standardized meta-prompt produced ASRs up to 18 times higher than their prose baselines. Outputs are evaluated using an ensemble of open-weight judge models and a human-validated stratified subset (with double-annotations to measure agreement). Disagreements were manually resolved. Poetic framing achieved an average jailbreak success rate of 62% for hand-crafted poems and approximately 43% for meta-prompt conversions (compared to non-poetic baselines), substantially outperforming non-poetic baselines and revealing a systematic vulnerability across model families and safety training approaches. These findings demonstrate that stylistic variation alone can circumvent contemporary safety mechanisms, suggesting fundamental limitations in current alignment methods and evaluation protocols. 

---
# MAPROC at AHaSIS Shared Task: Few-Shot and Sentence Transformer for Sentiment Analysis of Arabic Hotel Reviews 

**Authors**: Randa Zarnoufi  

**Link**: [PDF](https://arxiv.org/pdf/2511.15291)  

**Abstract**: Sentiment analysis of Arabic dialects presents significant challenges due to linguistic diversity and the scarcity of annotated data. This paper describes our approach to the AHaSIS shared task, which focuses on sentiment analysis on Arabic dialects in the hospitality domain. The dataset comprises hotel reviews written in Moroccan and Saudi dialects, and the objective is to classify the reviewers sentiment as positive, negative, or neutral. We employed the SetFit (Sentence Transformer Fine-tuning) framework, a data-efficient few-shot learning technique. On the official evaluation set, our system achieved an F1 of 73%, ranking 12th among 26 participants. This work highlights the potential of few-shot learning to address data scarcity in processing nuanced dialectal Arabic text within specialized domains like hotel reviews. 

---
# IndicGEC: Powerful Models, or a Measurement Mirage? 

**Authors**: Sowmya Vajjala  

**Link**: [PDF](https://arxiv.org/pdf/2511.15260)  

**Abstract**: In this paper, we report the results of the TeamNRC's participation in the BHASHA-Task 1 Grammatical Error Correction shared task this https URL for 5 Indian languages. Our approach, focusing on zero/few-shot prompting of language models of varying sizes (4B to large proprietary models) achieved a Rank 4 in Telugu and Rank 2 in Hindi with GLEU scores of 83.78 and 84.31 respectively. In this paper, we extend the experiments to the other three languages of the shared task - Tamil, Malayalam and Bangla, and take a closer look at the data quality and evaluation metric used. Our results primarily highlight the potential of small language models, and summarize the concerns related to creating good quality datasets and appropriate metrics for this task that are suitable for Indian language scripts. 

---
# Context Cascade Compression: Exploring the Upper Limits of Text Compression 

**Authors**: Fanfan Liu, Haibo Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2511.15244)  

**Abstract**: Million-level token inputs in long-context tasks pose significant computational and memory challenges for Large Language Models (LLMs). Recently, DeepSeek-OCR conducted research into the feasibility of Contexts Optical Compression and achieved preliminary results. Inspired by this, we introduce Context Cascade Compression C3 to explore the upper limits of text compression. Our method cascades two LLMs of different sizes to handle the compression and decoding tasks. Specifically, a small LLM, acting as the first stage, performs text compression by condensing a long context into a set of latent tokens (e.g., 32 or 64 in length), achieving a high ratio of text tokens to latent tokens. A large LLM, as the second stage, then executes the decoding task on this compressed context. Experiments show that at a 20x compression ratio (where the number of text tokens is 20 times the number of latent tokens), our model achieves 98% decoding accuracy, compared to approximately 60% for DeepSeek-OCR. When we further increase the compression ratio to 40x, the accuracy is maintained at around 93%. This indicates that in the domain of context compression, C3 Compression demonstrates superior performance and feasibility over optical character compression. C3 uses a simpler, pure-text pipeline that ignores factors like layout, color, and information loss from a visual encoder. This also suggests a potential upper bound for compression ratios in future work on optical character compression, OCR, and related fields. Codes and model weights are publicly accessible at this https URL 

---
# OEMA: Ontology-Enhanced Multi-Agent Collaboration Framework for Zero-Shot Clinical Named Entity Recognition 

**Authors**: Xinli Tao, Xin Dong, Xuezhong Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2511.15211)  

**Abstract**: Clinical named entity recognition (NER) is crucial for extracting information from electronic health records (EHRs), but supervised models like CRF and BioClinicalBERT require costly annotated data. While zero-shot NER with large language models (LLMs) reduces this dependency, it struggles with example selection granularity and integrating prompts with self-improvement. To address this, we propose OEMA, a zero-shot clinical NER framework using multi-agent collaboration. OEMA's three components are: a self-annotator generating examples, a discriminator filtering them via SNOMED CT, and a predictor using entity descriptions for accurate inference. On MTSamples and VAERS datasets, OEMA achieves state-of-the-art exact-match performance. Under related-match, it matches supervised BioClinicalBERT and surpasses CRF. OEMA addresses key zero-shot NER challenges through ontology-guided reasoning and multi-agent collaboration, achieving near-supervised performance and showing promise for clinical NLP applications. 

---
# Unveiling Intrinsic Dimension of Texts: from Academic Abstract to Creative Story 

**Authors**: Vladislav Pedashenko, Laida Kushnareva, Yana Khassan Nibal, Eduard Tulchinskii, Kristian Kuznetsov, Vladislav Zharchinskii, Yury Maximov, Irina Piontkovskaya  

**Link**: [PDF](https://arxiv.org/pdf/2511.15210)  

**Abstract**: Intrinsic dimension (ID) is an important tool in modern LLM analysis, informing studies of training dynamics, scaling behavior, and dataset structure, yet its textual determinants remain underexplored. We provide the first comprehensive study grounding ID in interpretable text properties through cross-encoder analysis, linguistic features, and sparse autoencoders (SAEs). In this work, we establish three key findings. First, ID is complementary to entropy-based metrics: after controlling for length, the two are uncorrelated, with ID capturing geometric complexity orthogonal to prediction quality. Second, ID exhibits robust genre stratification: scientific prose shows low ID (~8), encyclopedic content medium ID (~9), and creative/opinion writing high ID (~10.5) across all models tested. This reveals that contemporary LLMs find scientific text "representationally simple" while fiction requires additional degrees of freedom. Third, using SAEs, we identify causal features: scientific signals (formal tone, report templates, statistics) reduce ID; humanized signals (personalization, emotion, narrative) increase it. Steering experiments confirm these effects are causal. Thus, for contemporary models, scientific writing appears comparatively "easy", whereas fiction, opinion, and affect add representational degrees of freedom. Our multi-faceted analysis provides practical guidance for the proper use of ID and the sound interpretation of ID-based results. 

---
# HinTel-AlignBench: A Framework and Benchmark for Hindi-Telugu with English-Aligned Samples 

**Authors**: Rishikant Chigrupaatii, Ponnada Sai Tulasi Kanishka, Lalit Chandra Routhu, Martin Patel Sama Supratheek Reddy, Divyam Gupta, Dasari Srikar, Krishna Teja Kuchimanchi, Rajiv Misra, Rohun Tripathi  

**Link**: [PDF](https://arxiv.org/pdf/2511.15183)  

**Abstract**: With nearly 1.5 billion people and more than 120 major languages, India represents one of the most diverse regions in the world. As multilingual Vision-Language Models (VLMs) gain prominence, robust evaluation methodologies are essential to drive progress toward equitable AI for low-resource languages. Current multilingual VLM evaluations suffer from four major limitations: reliance on unverified auto-translations, narrow task/domain coverage, limited sample sizes, and lack of cultural and natively sourced Question-Answering (QA). To address these gaps, we present a scalable framework to evaluate VLMs in Indian languages and compare it with performance in English. Using the framework, we generate HinTel-AlignBench, a benchmark that draws from diverse sources in Hindi and Telugu with English-aligned samples. Our contributions are threefold: (1) a semi-automated dataset creation framework combining back-translation, filtering, and human verification; (2) the most comprehensive vision-language benchmark for Hindi and and Telugu, including adapted English datasets (VQAv2, RealWorldQA, CLEVR-Math) and native novel Indic datasets (JEE for STEM, VAANI for cultural grounding) with approximately 4,000 QA pairs per language; and (3) a detailed performance analysis of various State-of-the-Art (SOTA) open-weight and closed-source VLMs. We find a regression in performance for tasks in English versus in Indian languages for 4 out of 5 tasks across all the models, with an average regression of 8.3 points in Hindi and 5.5 points for Telugu. We categorize common failure modes to highlight concrete areas of improvement in multilingual multimodal understanding. 

---
# Teaching According to Students' Aptitude: Personalized Mathematics Tutoring via Persona-, Memory-, and Forgetting-Aware LLMs 

**Authors**: Yang Wu, Rujing Yao, Tong Zhang, Yufei Shi, Zhuoren Jiang, Zhushan Li, Xiaozhong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.15163)  

**Abstract**: Large Language Models (LLMs) are increasingly integrated into intelligent tutoring systems to provide human-like and adaptive instruction. However, most existing approaches fail to capture how students' knowledge evolves dynamically across their proficiencies, conceptual gaps, and forgetting patterns. This challenge is particularly acute in mathematics tutoring, where effective instruction requires fine-grained scaffolding precisely calibrated to each student's mastery level and cognitive retention. To address this issue, we propose TASA (Teaching According to Students' Aptitude), a student-aware tutoring framework that integrates persona, memory, and forgetting dynamics for personalized mathematics learning. Specifically, TASA maintains a structured student persona capturing proficiency profiles and an event memory recording prior learning interactions. By incorporating a continuous forgetting curve with knowledge tracing, TASA dynamically updates each student's mastery state and generates contextually appropriate, difficulty-calibrated questions and explanations. Empirical results demonstrate that TASA achieves superior learning outcomes and more adaptive tutoring behavior compared to representative baselines, underscoring the importance of modeling temporal forgetting and learner profiles in LLM-based tutoring systems. 

---
# Mathematical Analysis of Hallucination Dynamics in Large Language Models: Uncertainty Quantification, Advanced Decoding, and Principled Mitigation 

**Authors**: Moses Kiprono  

**Link**: [PDF](https://arxiv.org/pdf/2511.15005)  

**Abstract**: Large Language Models (LLMs) are powerful linguistic engines but remain susceptible to hallucinations: plausible-sounding outputs that are factually incorrect or unsupported. In this work, we present a mathematically grounded framework to understand, measure, and mitigate these hallucinations. Drawing on probabilistic modeling, information theory, trigonometric signal analysis, and Bayesian uncertainty estimation, we analyze how errors compound autoregressively, propose refined uncertainty metrics, including semantic and phase-aware variants, and develop principled mitigation strategies such as contrastive decoding, retrieval-augmented grounding, factual alignment, and abstention. This unified lens connects recent advances in calibration, retrieval, and alignment to support safer and more reliable LLMs. 

---
# Hierarchical Token Prepending: Enhancing Information Flow in Decoder-based LLM Embeddings 

**Authors**: Xueying Ding, Xingyue Huang, Mingxuan Ju, Liam Collins, Yozen Liu, Leman Akoglu, Neil Shah, Tong Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2511.14868)  

**Abstract**: Large language models produce powerful text embeddings, but their causal attention mechanism restricts the flow of information from later to earlier tokens, degrading representation quality. While recent methods attempt to solve this by prepending a single summary token, they over-compress information, hence harming performance on long documents. We propose Hierarchical Token Prepending (HTP), a method that resolves two critical bottlenecks. To mitigate attention-level compression, HTP partitions the input into blocks and prepends block-level summary tokens to subsequent blocks, creating multiple pathways for backward information flow. To address readout-level over-squashing, we replace last-token pooling with mean-pooling, a choice supported by theoretical analysis. HTP achieves consistent performance gains across 11 retrieval datasets and 30 general embedding benchmarks, especially in long-context settings. As a simple, architecture-agnostic method, HTP enhances both zero-shot and finetuned models, offering a scalable route to superior long-document embeddings. 

---
# Opinion Mining and Analysis Using Hybrid Deep Neural Networks 

**Authors**: Adel Hidri, Suleiman Ali Alsaif, Muteeb Alahmari, Eman AlShehri, Minyar Sassi Hidri  

**Link**: [PDF](https://arxiv.org/pdf/2511.14796)  

**Abstract**: Understanding customer attitudes has become a critical component of decision-making due to the growing influence of social media and e-commerce. Text-based opinions are the most structured, hence playing an important role in sentiment analysis. Most of the existing methods, which include lexicon-based approaches and traditional machine learning techniques, are insufficient for handling contextual nuances and scalability. While the latter has limitations in model performance and generalization, deep learning (DL) has achieved improvement, especially on semantic relationship capturing with recurrent neural networks (RNNs) and convolutional neural networks (CNNs). The aim of the study is to enhance opinion mining by introducing a hybrid deep neural network model that combines a bidirectional gated recurrent unit (BGRU) and long short-term memory (LSTM) layers to improve sentiment analysis, particularly addressing challenges such as contextual nuance, scalability, and class imbalance. To substantiate the efficacy of the proposed model, we conducted comprehensive experiments utilizing benchmark datasets, encompassing IMDB movie critiques and Amazon product evaluations. The introduced hybrid BGRULSTM (HBGRU-LSTM) architecture attained a testing accuracy of 95%, exceeding the performance of traditional DL frameworks such as LSTM (93.06%), CNN+LSTM (93.31%), and GRU+LSTM (92.20%). Moreover, our model exhibited a noteworthy enhancement in recall for negative sentiments, escalating from 86% (unbalanced dataset) to 96% (balanced dataset), thereby ensuring a more equitable and just sentiment classification. Furthermore, the model diminished misclassification loss from 20.24% for unbalanced to 13.3% for balanced dataset, signifying enhanced generalization and resilience. 

---
# Human or LLM as Standardized Patients? A Comparative Study for Medical Education 

**Authors**: Bingquan Zhang, Xiaoxiao Liu, Yuchi Wang, Lei Zhou, Qianqian Xie, Benyou Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.14783)  

**Abstract**: Standardized Patients (SP) are indispensable for clinical skills training but remain expensive, inflexible, and difficult to scale. Existing large-language-model (LLM)-based SP simulators promise lower cost yet show inconsistent behavior and lack rigorous comparison with human SP. We present EasyMED, a multi-agent framework combining a Patient Agent for realistic dialogue, an Auxiliary Agent for factual consistency, and an Evaluation Agent that delivers actionable feedback. To support systematic assessment, we introduce SPBench, a benchmark of real SP-doctor interactions spanning 14 specialties and eight expert-defined evaluation criteria. Experiments demonstrate that EasyMED matches human SP learning outcomes while producing greater skill gains for lower-baseline students and offering improved flexibility, psychological safety, and cost efficiency. 

---
# The Impact of Prosodic Segmentation on Speech Synthesis of Spontaneous Speech 

**Authors**: Julio Cesar Galdino, Sidney Evaldo Leal, Leticia Gabriella De Souza, Rodrigo de Freitas Lima, Antonio Nelson Fornari Mendes Moreira, Arnaldo Candido Junior, Miguel Oliveira Jr., Edresson Casanova, Sandra M. Aluísio  

**Link**: [PDF](https://arxiv.org/pdf/2511.14779)  

**Abstract**: Spontaneous speech presents several challenges for speech synthesis, particularly in capturing the natural flow of conversation, including turn-taking, pauses, and disfluencies. Although speech synthesis systems have made significant progress in generating natural and intelligible speech, primarily through architectures that implicitly model prosodic features such as pitch, intensity, and duration, the construction of datasets with explicit prosodic segmentation and their impact on spontaneous speech synthesis remains largely unexplored. This paper evaluates the effects of manual and automatic prosodic segmentation annotations in Brazilian Portuguese on the quality of speech synthesized by a non-autoregressive model, FastSpeech 2. Experimental results show that training with prosodic segmentation produced slightly more intelligible and acoustically natural speech. While automatic segmentation tends to create more regular segments, manual prosodic segmentation introduces greater variability, which contributes to more natural prosody. Analysis of neutral declarative utterances showed that both training approaches reproduced the expected nuclear accent pattern, but the prosodic model aligned more closely with natural pre-nuclear contours. To support reproducibility and future research, all datasets, source codes, and trained models are publicly available under the CC BY-NC-ND 4.0 license. 

---
# COMPASS: Context-Modulated PID Attention Steering System for Hallucination Mitigation 

**Authors**: Snigdha Pandya, Rohan Nagale, Kenji Sahay, Anna Lin, Shikhar Shiromani, Kevin Zhu, Dev Sunishchal  

**Link**: [PDF](https://arxiv.org/pdf/2511.14776)  

**Abstract**: Large language models (LLMs) often generate fluent but factually incorrect statements despite having access to relevant evidence, a failure mode rooted in how they allocate attention between contextual and parametric knowledge. Understanding and steering this internal behavior is key both for trustworthy deployment and for scientific interpretability of model mechanisms. We introduce COMPASS (Context-Modulated PID Attention Steering System), a lightweight, interpretable control framework that embeds a model-based feedback loop directly within decoding. COMPASS quantifies context reliance via a transparent metric, the Context Reliance Score (CRS), which serves as an online probe of how attention heads ground generation in evidence. Using this interpretable signal, a PID controller dynamically modulates attention heads to maintain factual consistency without retraining or multi-pass decoding. Across benchmarks (HotpotQA, XSum, HaluEval, RAGTruth), COMPASS consistently reduces contextual hallucination rates (2.8 to 5.8 percent absolute) while revealing how distinct attention heads contribute to evidence alignment. These results highlight feedback-driven interpretability as a pathway toward scientific understanding of LLM behavior. 

---
# LiveCLKTBench: Towards Reliable Evaluation of Cross-Lingual Knowledge Transfer in Multilingual LLMs 

**Authors**: Pei-Fu Guo, Yun-Da Tsai, Chun-Chia Hsu, Kai-Xin Chen, Ya-An Tsai, Kai-Wei Chang, Nanyun Peng, Mi-Yen Yeh, Shou-De Lin  

**Link**: [PDF](https://arxiv.org/pdf/2511.14774)  

**Abstract**: Evaluating cross-lingual knowledge transfer in large language models is challenging, as correct answers in a target language may arise either from genuine transfer or from prior exposure during pre-training. We present LiveCLKTBench, an automated generation pipeline specifically designed to isolate and measure cross-lingual knowledge transfer. Our pipeline identifies self-contained, time-sensitive knowledge entities from real-world domains, filters them based on temporal occurrence, and verifies them against the model's knowledge. The documents of these valid entities are then used to generate factual questions, which are translated into multiple languages to evaluate transferability across linguistic boundaries. Using LiveCLKTBench, we evaluate several LLMs across five languages and observe that cross-lingual transfer is strongly influenced by linguistic distance and often asymmetric across language directions. While larger models improve transfer, the gains diminish with scale and vary across domains. These findings provide new insights into multilingual transfer and demonstrate the value of LiveCLKTBench as a reliable benchmark for future research. 

---
# Temporal Predictors of Outcome in Reasoning Language Models 

**Authors**: Joey David  

**Link**: [PDF](https://arxiv.org/pdf/2511.14773)  

**Abstract**: The chain-of-thought (CoT) paradigm uses the elicitation of step-by-step rationales as a proxy for reasoning, gradually refining the model's latent representation of a solution. However, it remains unclear just how early a Large Language Model (LLM) internally commits to an eventual outcome. We probe this by training linear classifiers on hidden states after the first t reasoning tokens, showing that eventual correctness is highly predictable after only a few tokens, even when longer outputs are needed to reach a definite answer. We show that, for harder questions, a drop in predictive accuracy highlights a selection artifact: hard items are disproportionately represented in long CoTs. Overall, our results imply that for reasoning models, internal self-assessment of success tends to emerge after only a few tokens, with implications for interpretability and for inference-time control. 

---
# Test-time Scaling of LLMs: A Survey from A Subproblem Structure Perspective 

**Authors**: Zhuoyi Yang, Xu Guo, Tong Zhang, Huijuan Xu, Boyang Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.14772)  

**Abstract**: With this paper, we survey techniques for improving the predictive accuracy of pretrained large language models by allocating additional compute at inference time. In categorizing test-time scaling methods, we place special emphasis on how a problem is decomposed into subproblems and on the topological organization of these subproblems whether sequential, parallel, or tree-structured. This perspective allows us to unify diverse approaches such as Chain-of-Thought, Branch-Solve-Merge, and Tree-of-Thought under a common lens. We further synthesize existing analyses of these techniques, highlighting their respective strengths and weaknesses, and conclude by outlining promising directions for future research 

---
# Think Visually, Reason Textually: Vision-Language Synergy in ARC 

**Authors**: Beichen Zhang, Yuhang Zang, Xiaoyi Dong, Yuhang Cao, Haodong Duan, Dahua Lin, Jiaqi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.15703)  

**Abstract**: Abstract reasoning from minimal examples remains a core unsolved problem for frontier foundation models such as GPT-5 and Grok 4. These models still fail to infer structured transformation rules from a handful of examples, which is a key hallmark of human intelligence. The Abstraction and Reasoning Corpus for Artificial General Intelligence (ARC-AGI) provides a rigorous testbed for this capability, demanding conceptual rule induction and transfer to novel tasks. Most existing methods treat ARC-AGI as a purely textual reasoning task, overlooking the fact that humans rely heavily on visual abstraction when solving such puzzles. However, our pilot experiments reveal a paradox: naively rendering ARC-AGI grids as images degrades performance due to imprecise rule execution. This leads to our central hypothesis that vision and language possess complementary strengths across distinct reasoning stages: vision supports global pattern abstraction and verification, whereas language specializes in symbolic rule formulation and precise execution. Building on this insight, we introduce two synergistic strategies: (1) Vision-Language Synergy Reasoning (VLSR), which decomposes ARC-AGI into modality-aligned subtasks; and (2) Modality-Switch Self-Correction (MSSC), which leverages vision to verify text-based reasoning for intrinsic error correction. Extensive experiments demonstrate that our approach yields up to a 4.33% improvement over text-only baselines across diverse flagship models and multiple ARC-AGI tasks. Our findings suggest that unifying visual abstraction with linguistic reasoning is a crucial step toward achieving generalizable, human-like intelligence in future foundation models. Source code will be released soon. 

---
# MoDES: Accelerating Mixture-of-Experts Multimodal Large Language Models via Dynamic Expert Skipping 

**Authors**: Yushi Huang, Zining Wang, Zhihang Yuan, Yifu Ding, Ruihao Gong, Jinyang Guo, Xianglong Liu, Jun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.15690)  

**Abstract**: Mixture-of-Experts (MoE) Multimodal large language models (MLLMs) excel at vision-language tasks, but they suffer from high computational inefficiency. To reduce inference overhead, expert skipping methods have been proposed to deactivate redundant experts based on the current input tokens. However, we find that applying these methods-originally designed for unimodal large language models (LLMs)-to MLLMs results in considerable performance degradation. This is primarily because such methods fail to account for the heterogeneous contributions of experts across MoE layers and modality-specific behaviors of tokens within these layers. Motivated by these findings, we propose MoDES, the first training-free framework that adaptively skips experts to enable efficient and accurate MoE MLLM inference. It incorporates a globally-modulated local gating (GMLG) mechanism that integrates global layer-wise importance into local routing probabilities to accurately estimate per-token expert importance. A dual-modality thresholding (DMT) method is then applied, which processes tokens from each modality separately, to derive the skipping schedule. To set the optimal thresholds, we introduce a frontier search algorithm that exploits monotonicity properties, cutting convergence time from several days to a few hours. Extensive experiments for 3 model series across 13 benchmarks demonstrate that MoDES far outperforms previous approaches. For instance, when skipping 88% experts for Qwen3-VL-MoE-30B-A3B-Instruct, the performance boost is up to 10.67% (97.33% vs. 86.66%). Furthermore, MoDES significantly enhances inference speed, improving the prefilling time by 2.16$\times$ and the decoding time by 1.26$\times$. 

---
# VisPlay: Self-Evolving Vision-Language Models from Images 

**Authors**: Yicheng He, Chengsong Huang, Zongxia Li, Jiaxin Huang, Yonghui Yang  

**Link**: [PDF](https://arxiv.org/pdf/2511.15661)  

**Abstract**: Reinforcement learning (RL) provides a principled framework for improving Vision-Language Models (VLMs) on complex reasoning tasks. However, existing RL approaches often rely on human-annotated labels or task-specific heuristics to define verifiable rewards, both of which are costly and difficult to scale. We introduce VisPlay, a self-evolving RL framework that enables VLMs to autonomously improve their reasoning abilities using large amounts of unlabeled image data. Starting from a single base VLM, VisPlay assigns the model into two interacting roles: an Image-Conditioned Questioner that formulates challenging yet answerable visual questions, and a Multimodal Reasoner that generates silver responses. These roles are jointly trained with Group Relative Policy Optimization (GRPO), which incorporates diversity and difficulty rewards to balance the complexity of generated questions with the quality of the silver answers. VisPlay scales efficiently across two model families. When trained on Qwen2.5-VL and MiMo-VL, VisPlay achieves consistent improvements in visual reasoning, compositional generalization, and hallucination reduction across eight benchmarks, including MM-Vet and MMMU, demonstrating a scalable path toward self-evolving multimodal intelligence. The project page is available at this https URL 

---
# When to Think and When to Look: Uncertainty-Guided Lookback 

**Authors**: Jing Bi, Filippos Bellos, Junjia Guo, Yayuan Li, Chao Huang, Yunlong, Tang, Luchuan Song, Susan Liang, Zhongfei, Zhang, Jason J. Corso, Chenliang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2511.15613)  

**Abstract**: Test-time thinking (that is, generating explicit intermediate reasoning chains) is known to boost performance in large language models and has recently shown strong gains for large vision language models (LVLMs). However, despite these promising results, there is still no systematic analysis of how thinking actually affects visual reasoning. We provide the first such analysis with a large scale, controlled comparison of thinking for LVLMs, evaluating ten variants from the InternVL3.5 and Qwen3-VL families on MMMU-val under generous token budgets and multi pass decoding. We show that more thinking is not always better; long chains often yield long wrong trajectories that ignore the image and underperform the same models run in standard instruct mode. A deeper analysis reveals that certain short lookback phrases, which explicitly refer back to the image, are strongly enriched in successful trajectories and correlate with better visual grounding. Building on this insight, we propose uncertainty guided lookback, a training free decoding strategy that combines an uncertainty signal with adaptive lookback prompts and breadth search. Our method improves overall MMMU performance, delivers the largest gains in categories where standard thinking is weak, and outperforms several strong decoding baselines, setting a new state of the art under fixed model families and token budgets. We further show that this decoding strategy generalizes, yielding consistent improvements on five additional benchmarks, including two broad multimodal suites and math focused visual reasoning datasets. 

---
# SRPO: Self-Referential Policy Optimization for Vision-Language-Action Models 

**Authors**: Senyu Fei, Siyin Wang, Li Ji, Ao Li, Shiduo Zhang, Liming Liu, Jinlong Hou, Jingjing Gong, Xianzhong Zhao, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2511.15605)  

**Abstract**: Vision-Language-Action (VLA) models excel in robotic manipulation but are constrained by their heavy reliance on expert demonstrations, leading to demonstration bias and limiting performance. Reinforcement learning (RL) is a vital post-training strategy to overcome these limits, yet current VLA-RL methods, including group-based optimization approaches, are crippled by severe reward sparsity. Relying on binary success indicators wastes valuable information in failed trajectories, resulting in low training efficiency. To solve this, we propose Self-Referential Policy Optimization (SRPO), a novel VLA-RL framework. SRPO eliminates the need for external demonstrations or manual reward engineering by leveraging the model's own successful trajectories, generated within the current training batch, as a self-reference. This allows us to assign a progress-wise reward to failed attempts. A core innovation is the use of latent world representations to measure behavioral progress robustly. Instead of relying on raw pixels or requiring domain-specific fine-tuning, we utilize the compressed, transferable encodings from a world model's latent space. These representations naturally capture progress patterns across environments, enabling accurate, generalized trajectory comparison. Empirical evaluations on the LIBERO benchmark demonstrate SRPO's efficiency and effectiveness. Starting from a supervised baseline with 48.9% success, SRPO achieves a new state-of-the-art success rate of 99.2% in just 200 RL steps, representing a 103% relative improvement without any extra supervision. Furthermore, SRPO shows substantial robustness, achieving a 167% performance improvement on the LIBERO-Plus benchmark. 

---
# Computer-Use Agents as Judges for Generative User Interface 

**Authors**: Kevin Qinghong Lin, Siyuan Hu, Linjie Li, Zhengyuan Yang, Lijuan Wang, Philip Torr, Mike Zheng Shou  

**Link**: [PDF](https://arxiv.org/pdf/2511.15567)  

**Abstract**: Computer-Use Agents (CUA) are becoming increasingly capable of autonomously operating digital environments through Graphical User Interfaces (GUI). Yet, most GUI remain designed primarily for humans--prioritizing aesthetics and usability--forcing agents to adopt human-oriented behaviors that are unnecessary for efficient task execution. At the same time, rapid advances in coding-oriented language models (Coder) have transformed automatic GUI design. This raises a fundamental question: Can CUA as judges to assist Coder for automatic GUI design? To investigate, we introduce AUI-Gym, a benchmark for Automatic GUI development spanning 52 applications across diverse domains. Using language models, we synthesize 1560 tasks that simulate real-world scenarios. To ensure task reliability, we further develop a verifier that programmatically checks whether each task is executable within its environment. Building on this, we propose a Coder-CUA in Collaboration framework: the Coder acts as Designer, generating and revising websites, while the CUA serves as Judge, evaluating functionality and refining designs. Success is measured not by visual appearance, but by task solvability and CUA navigation success rate. To turn CUA feedback into usable guidance, we design a CUA Dashboard that compresses multi-step navigation histories into concise visual summaries, offering interpretable guidance for iterative redesign. By positioning agents as both designers and judges, our framework shifts interface design toward agent-native efficiency and reliability. Our work takes a step toward shifting agents from passive use toward active participation in digital environments. Our code and dataset are available at this https URL. 

---
# CroPS: Improving Dense Retrieval with Cross-Perspective Positive Samples in Short-Video Search 

**Authors**: Ao Xie, Jiahui Chen, Quanzhi Zhu, Xiaoze Jiang, Zhiheng Qin, Enyun Yu, Han Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.15443)  

**Abstract**: Dense retrieval has become a foundational paradigm in modern search systems, especially on short-video platforms. However, most industrial systems adopt a self-reinforcing training pipeline that relies on historically exposed user interactions for supervision. This paradigm inevitably leads to a filter bubble effect, where potentially relevant but previously unseen content is excluded from the training signal, biasing the model toward narrow and conservative retrieval. In this paper, we present CroPS (Cross-Perspective Positive Samples), a novel retrieval data engine designed to alleviate this problem by introducing diverse and semantically meaningful positive examples from multiple perspectives. CroPS enhances training with positive signals derived from user query reformulation behavior (query-level), engagement data in recommendation streams (system-level), and world knowledge synthesized by large language models (knowledge-level). To effectively utilize these heterogeneous signals, we introduce a Hierarchical Label Assignment (HLA) strategy and a corresponding H-InfoNCE loss that together enable fine-grained, relevance-aware optimization. Extensive experiments conducted on Kuaishou Search, a large-scale commercial short-video search platform, demonstrate that CroPS significantly outperforms strong baselines both offline and in live A/B tests, achieving superior retrieval performance and reducing query reformulation rates. CroPS is now fully deployed in Kuaishou Search, serving hundreds of millions of users daily. 

---
# SkyEgg: Joint Implementation Selection and Scheduling for Hardware Synthesis using E-graphs 

**Authors**: Youwei Xiao, Yuyang Zou, Yun Liang  

**Link**: [PDF](https://arxiv.org/pdf/2511.15323)  

**Abstract**: Hardware synthesis from high-level descriptions remains fundamentally limited by the sequential optimization of interdependent design decisions. Current methodologies, including state-of-the-art high-level synthesis (HLS) tools, artificially separate implementation selection from scheduling, leading to suboptimal designs that cannot fully exploit modern FPGA heterogeneous architectures. Implementation selection is typically performed by ad-hoc pattern matching on operations, a process that does not consider the impact on scheduling. Subsequently, scheduling algorithms operate on fixed selection solutions with inaccurate delay estimates, which misses critical optimization opportunities from appropriately configured FPGA blocks like DSP slices.
We present SkyEgg, a novel hardware synthesis framework that jointly optimizes implementation selection and scheduling using the e-graph data structure. Our key insight is that both algebraic transformations and hardware implementation choices can be uniformly represented as rewrite rules within an e-graph, modeling the complete design space of implementation candidates to be selected and scheduled together. First, SkyEgg constructs an e-graph from the input program. It then applies both algebraic and implementation rewrites through equality saturation. Finally, it formulates the joint optimization as a mixed-integer linear programming (MILP) problem on the saturated e-graph. We provide both exact MILP solving and an efficient ASAP heuristic for scalable synthesis. Our evaluation on benchmarks from diverse applications targeting Xilinx Kintex UltraScale+ FPGAs demonstrates that SkyEgg achieves an average speedup of 3.01x over Vitis HLS, with improvements up to 5.22x for complex expressions. 

---
# ChartEditor: A Reinforcement Learning Framework for Robust Chart Editing 

**Authors**: Liangyu Chen, Yichen Xu, Jianzhe Ma, Yuqi Liu, Donglu Yang, Liang Zhang, Wenxuan Wang, Qin Jin  

**Link**: [PDF](https://arxiv.org/pdf/2511.15266)  

**Abstract**: Chart editing reduces manual effort in visualization design. Typical benchmarks limited in data diversity and assume access to complete chart code, which is seldom in real-world scenarios. To address this gap, we present ChartEditVista, a comprehensive benchmark consisting of 7,964 samples spanning 31 chart categories. It encompasses diverse editing instructions and covers nearly all editable chart elements. The inputs in ChartEditVista include only the original chart image and natural language editing instructions, without the original chart codes. ChartEditVista is generated through a fully automated pipeline that produces, edits, and verifies charts, ensuring high-quality chart editing data. Besides, we introduce two novel fine-grained, rule-based evaluation metrics: the layout metric, which evaluates the position, size and color of graphical components; and the text metric, which jointly assesses textual content and font styling. Building on top of ChartEditVista, we present ChartEditor, a model trained using a reinforcement learning framework that incorporates a novel rendering reward to simultaneously enforce code executability and visual fidelity. Through extensive experiments and human evaluations, we demonstrate that ChartEditVista provides a robust evaluation, while ChartEditor consistently outperforms models with similar-scale and larger-scale on chart editing tasks. 

---
# M, Toolchain and Language for Reusable Model Compilation 

**Authors**: Hiep Hong Trinh, Federico Ciccozzi, Abu Naser Masud, Marjan Sirjani, Mikael Sjödin  

**Link**: [PDF](https://arxiv.org/pdf/2511.15257)  

**Abstract**: Complex software-driven systems often interleave distributed, concurrent computation processes with physical interactions with the environment. Developing these systems more efficiently and safely can be achieved by employing actionable, software-based models. From a high-level system model, engineers often need to derive multiple specialized models for different purposes, including simulation, deployment, and formal verification. Each of these target models usually rely on its own formalism, specification language, and execution platform. Traditionally, a compiler analyzes a program written in a programming language and generates executable code. In contrast, a model compiler processes a source model written in a modeling language and should ideally support the generation of multiple heterogeneous targets. However, most existing modeling languages are designed with a narrow focus, typically targeting only simulation or implementation. Multi-target compilation, when not considered during the language's early design, becomes significantly harder to achieve. In this paper, we introduce our initiative: a toolchain and modeling language called M, designed to support system modeling and multi-target compilation for model-driven engineering of complex, concurrent, and time-aware systems. M is a textual, grammar-driven language based on the actor model and extended with discrete-event scheduling semantics. It provides constructs for modeling system entities, message-based interactions, and time- or state-triggered reactions. From such models, M enables the systematic generation of diverse target artifacts while preserving semantic conformance to the original model. Moreover, M can serve as a middle language to which other modeling languages may anchor, thereby allowing them to benefit from its compilation framework. 

---
# Generating Natural-Language Surgical Feedback: From Structured Representation to Domain-Grounded Evaluation 

**Authors**: Firdavs Nasriddinov, Rafal Kocielnik, Anima Anandkumar, Andrew J. Hung  

**Link**: [PDF](https://arxiv.org/pdf/2511.15159)  

**Abstract**: High-quality intraoperative feedback from a surgical trainer is pivotal for improving trainee performance and long-term skill acquisition. Automating natural, trainer-style feedback promises timely, accessible, and consistent guidance at scale but requires models that understand clinically relevant representations. We present a structure-aware pipeline that learns a surgical action ontology from real trainer-to-trainee transcripts (33 surgeries) and uses it to condition feedback generation. We contribute by (1) mining Instrument-Action-Target (IAT) triplets from real-world feedback text and clustering surface forms into normalized categories, (2) fine-tuning a video-to-IAT model that leverages the surgical procedure and task contexts as well as fine-grained temporal instrument motion, and (3) demonstrating how to effectively use IAT triplet representations to guide GPT-4o in generating clinically grounded, trainer-style feedback. We show that, on Task 1: Video-to-IAT recognition, our context injection and temporal tracking deliver consistent AUC gains (Instrument: 0.67 to 0.74; Action: 0.60 to 0.63; Tissue: 0.74 to 0.79). For Task 2: feedback text generation (rated on a 1-5 fidelity rubric where 1 = opposite/unsafe, 3 = admissible, and 5 = perfect match to a human trainer), GPT-4o from video alone scores 2.17, while IAT conditioning reaches 2.44 (+12.4%), doubling the share of admissible generations with score >= 3 from 21% to 42%. Traditional text-similarity metrics also improve: word error rate decreases by 15-31% and ROUGE (phrase/substring overlap) increases by 9-64%. Grounding generation in explicit IAT structure improves fidelity and yields clinician-verifiable rationales, supporting auditable use in surgical training. 

---
# CASTELLA: Long Audio Dataset with Captions and Temporal Boundaries 

**Authors**: Hokuto Munakata, Takehiro Imamura, Taichi Nishimura, Tatsuya Komatsu  

**Link**: [PDF](https://arxiv.org/pdf/2511.15131)  

**Abstract**: We introduce CASTELLA, a human-annotated audio benchmark for the task of audio moment retrieval (AMR). Although AMR has various useful potential applications, there is still no established benchmark with real-world data. The early study of AMR trained the model with solely synthetic datasets. Moreover, the evaluation is based on annotated dataset of fewer than 100 samples. This resulted in less reliable reported performance. To ensure performance for applications in real-world environments, we present CASTELLA, a large-scale manually annotated AMR dataset. CASTELLA consists of 1,009, 213, and 640 audio recordings for train, valid, and test split, respectively, which is 24 times larger than the previous dataset. We also establish a baseline model for AMR using CASTELLA. Our experiments demonstrate that a model fine-tuned on CASTELLA after pre-training on the synthetic data outperformed a model trained solely on the synthetic data by 10.4 points in Recall1@0.7. CASTELLA is publicly available in this https URL. 

---
# Knowledge-Informed Automatic Feature Extraction via Collaborative Large Language Model Agents 

**Authors**: Henrik Bradland, Morten Goodwin, Vladimir I. Zadorozhny, Per-Arne Andersen  

**Link**: [PDF](https://arxiv.org/pdf/2511.15074)  

**Abstract**: The performance of machine learning models on tabular data is critically dependent on high-quality feature engineering. While Large Language Models (LLMs) have shown promise in automating feature extraction (AutoFE), existing methods are often limited by monolithic LLM architectures, simplistic quantitative feedback, and a failure to systematically integrate external domain knowledge. This paper introduces Rogue One, a novel, LLM-based multi-agent framework for knowledge-informed automatic feature extraction. Rogue One operationalizes a decentralized system of three specialized agents-Scientist, Extractor, and Tester-that collaborate iteratively to discover, generate, and validate predictive features. Crucially, the framework moves beyond primitive accuracy scores by introducing a rich, qualitative feedback mechanism and a "flooding-pruning" strategy, allowing it to dynamically balance feature exploration and exploitation. By actively incorporating external knowledge via an integrated retrieval-augmented (RAG) system, Rogue One generates features that are not only statistically powerful but also semantically meaningful and interpretable. We demonstrate that Rogue One significantly outperforms state-of-the-art methods on a comprehensive suite of 19 classification and 9 regression datasets. Furthermore, we show qualitatively that the system surfaces novel, testable hypotheses, such as identifying a new potential biomarker in the myocardial dataset, underscoring its utility as a tool for scientific discovery. 

---
# ProRAC: A Neuro-symbolic Method for Reasoning about Actions with LLM-based Progression 

**Authors**: Haoyong Wu, Yongmei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.15069)  

**Abstract**: In this paper, we propose ProRAC (Progression-based Reasoning about Actions and Change), a neuro-symbolic framework that leverages LLMs to tackle RAC problems. ProRAC extracts fundamental RAC elements including actions and questions from the problem, progressively executes each action to derive the final state, and then evaluates the query against the progressed state to arrive at an answer. We evaluate ProRAC on several RAC benchmarks, and the results demonstrate that our approach achieves strong performance across different benchmarks, domains, LLM backbones, and types of RAC tasks. 

---
# Evaluating Multimodal Large Language Models on Vertically Written Japanese Text 

**Authors**: Keito Sasagawa, Shuhei Kurita, Daisuke Kawahara  

**Link**: [PDF](https://arxiv.org/pdf/2511.15059)  

**Abstract**: Multimodal Large Language Models (MLLMs) have seen rapid advances in recent years and are now being applied to visual document understanding tasks. They are expected to process a wide range of document images across languages, including Japanese. Understanding documents from images requires models to read what are written in them. Since some Japanese documents are written vertically, support for vertical writing is essential. However, research specifically focused on vertically written Japanese text remains limited. In this study, we evaluate the reading capability of existing MLLMs on vertically written Japanese text. First, we generate a synthetic Japanese OCR dataset by rendering Japanese texts into images, and use it for both model fine-tuning and evaluation. This dataset includes Japanese text in both horizontal and vertical writing. We also create an evaluation dataset sourced from the real-world document images containing vertically written Japanese text. Using these datasets, we demonstrate that the existing MLLMs perform worse on vertically written Japanese text than on horizontally written Japanese text. Furthermore, we show that training MLLMs on our synthesized Japanese OCR dataset results in improving the performance of models that previously could not handle vertical writing. The datasets and code are publicly available this https URL. 

---
# How to Train Private Clinical Language Models: A Comparative Study of Privacy-Preserving Pipelines for ICD-9 Coding 

**Authors**: Mathieu Dufour, Andrew Duncan  

**Link**: [PDF](https://arxiv.org/pdf/2511.14936)  

**Abstract**: Large language models trained on clinical text risk exposing sensitive patient information, yet differential privacy (DP) methods often severely degrade the diagnostic accuracy needed for deployment. Despite rapid progress in DP optimisation and text generation, it remains unclear which privacy-preserving strategy actually works best for clinical language tasks. We present the first systematic head-to-head comparison of four training pipelines for automated diagnostic coding from hospital discharge summaries. All pipelines use identical 1B-parameter models and matched privacy budgets to predict ICD-9 codes. At moderate and relaxed privacy budgets ($\varepsilon \in \{4, 6\}$), knowledge distillation from DP-trained teachers outperforms both direct DP-SGD and DP-synthetic data training, recovering up to 63\% of the non-private performance whilst maintaining strong empirical privacy (membership-inference AUC $\approx$ 0.5). These findings expose large differences in the privacy-utility trade-off across architectures and identify knowledge distillation as the most practical route to privacy-preserving clinical NLP. 

---
# Skin-R1: Toward Trustworthy Clinical Reasoning for Dermatological Diagnosis 

**Authors**: Zehao Liu, Wejieying Ren, Jipeng Zhang, Tianxiang Zhao, Jingxi Zhu, Xiaoting Li, Vasant G. Honavar  

**Link**: [PDF](https://arxiv.org/pdf/2511.14900)  

**Abstract**: The emergence of vision-language models (VLMs) has opened new possibilities for clinical reasoning and has shown promising performance in dermatological diagnosis. However, their trustworthiness and clinical utility are often limited by three major factors: (1) Data heterogeneity, where diverse datasets lack consistent diagnostic labels and clinical concept annotations; (2) Absence of grounded diagnostic rationales, leading to a scarcity of reliable reasoning supervision; and (3) Limited scalability and generalization, as models trained on small, densely annotated datasets struggle to transfer nuanced reasoning to large, sparsely-annotated ones.
To address these limitations, we propose SkinR1, a novel dermatological VLM that combines deep, textbook-based reasoning with the broad generalization capabilities of reinforcement learning (RL). SkinR1 systematically resolves the key challenges through a unified, end-to-end framework. First, we design a textbook-based reasoning generator that synthesizes high-fidelity, hierarchy-aware, and differential-diagnosis (DDx)-informed trajectories, providing reliable expert-level supervision. Second, we leverage the constructed trajectories for supervised fine-tuning (SFT) empowering the model with grounded reasoning ability. Third, we develop a novel RL paradigm that, by incorporating the hierarchical structure of diseases, effectively transfers these grounded reasoning patterns to large-scale, sparse data. Extensive experiments on multiple dermatology datasets demonstrate that SkinR1 achieves superior diagnostic accuracy. The ablation study demonstrates the importance of the reasoning foundation instilled by SFT. 

---
# Empowering Multi-Turn Tool-Integrated Reasoning with Group Turn Policy Optimization 

**Authors**: Yifeng Ding, Hung Le, Songyang Han, Kangrui Ruan, Zhenghui Jin, Varun Kumar, Zijian Wang, Anoop Deoras  

**Link**: [PDF](https://arxiv.org/pdf/2511.14846)  

**Abstract**: Training Large Language Models (LLMs) for multi-turn Tool-Integrated Reasoning (TIR) - where models iteratively reason, generate code, and verify through execution - remains challenging for existing reinforcement learning (RL) approaches. Current RL methods, exemplified by Group Relative Policy Optimization (GRPO), suffer from coarse-grained, trajectory-level rewards that provide insufficient learning signals for complex multi-turn interactions, leading to training stagnation. To address this issue, we propose Group Turn Policy Optimization (GTPO), a novel RL algorithm specifically designed for training LLMs on multi-turn TIR tasks. GTPO introduces three key innovations: (1) turn-level reward assignment that provides fine-grained feedback for individual turns, (2) return-based advantage estimation where normalized discounted returns are calculated as advantages, and (3) self-supervised reward shaping that exploits self-supervision signals from generated code to densify sparse binary outcome-based rewards. Our comprehensive evaluation demonstrates that GTPO outperforms GRPO by 3.0% on average across diverse reasoning benchmarks, establishing its effectiveness for advancing complex mathematical reasoning in the real world. 

---
# Cluster-based Adaptive Retrieval: Dynamic Context Selection for RAG Applications 

**Authors**: Yifan Xu, Vipul Gupta, Rohit Aggarwal, Varsha Mahadevan, Bhaskar Krishnamachari  

**Link**: [PDF](https://arxiv.org/pdf/2511.14769)  

**Abstract**: Retrieval-Augmented Generation (RAG) enhances large language models (LLMs) by pulling in external material, document, code, manuals, from vast and ever-growing corpora, to effectively answer user queries. The effectiveness of RAG depends significantly on aligning the number of retrieved documents with query characteristics: narrowly focused queries typically require fewer, highly relevant documents, whereas broader or ambiguous queries benefit from retrieving more extensive supporting information. However, the common static top-k retrieval approach fails to adapt to this variability, resulting in either insufficient context from too few documents or redundant information from too many. Motivated by these challenges, we introduce Cluster-based Adaptive Retrieval (CAR), an algorithm that dynamically determines the optimal number of documents by analyzing the clustering patterns of ordered query-document similarity distances. CAR detects the transition point within similarity distances, where tightly clustered, highly relevant documents shift toward less pertinent candidates, establishing an adaptive cut-off that scales with query complexity. On Coinbase's CDP corpus and the public MultiHop-RAG benchmark, CAR consistently picks the optimal retrieval depth and achieves the highest TES score, outperforming every fixed top-k baseline. In downstream RAG evaluations, CAR cuts LLM token usage by 60%, trims end-to-end latency by 22%, and reduces hallucinations by 10% while fully preserving answer relevance. Since integrating CAR into Coinbase's virtual assistant, we've seen user engagement jump by 200%. 

---
# Optimizing Agricultural Research: A RAG-Based Approach to Mycorrhizal Fungi Information 

**Authors**: Mohammad Usman Altam, Md Imtiaz Habib, Tuan Hoang  

**Link**: [PDF](https://arxiv.org/pdf/2511.14765)  

**Abstract**: Retrieval-Augmented Generation (RAG) represents a transformative approach within natural language processing (NLP), combining neural information retrieval with generative language modeling to enhance both contextual accuracy and factual reliability of responses. Unlike conventional Large Language Models (LLMs), which are constrained by static training corpora, RAG-powered systems dynamically integrate domain-specific external knowledge sources, thereby overcoming temporal and disciplinary limitations. In this study, we present the design and evaluation of a RAG-enabled system tailored for Mycophyto, with a focus on advancing agricultural applications related to arbuscular mycorrhizal fungi (AMF). These fungi play a critical role in sustainable agriculture by enhancing nutrient acquisition, improving plant resilience under abiotic and biotic stresses, and contributing to soil health. Our system operationalizes a dual-layered strategy: (i) semantic retrieval and augmentation of domain-specific content from agronomy and biotechnology corpora using vector embeddings, and (ii) structured data extraction to capture predefined experimental metadata such as inoculation methods, spore densities, soil parameters, and yield outcomes. This hybrid approach ensures that generated responses are not only semantically aligned but also supported by structured experimental evidence. To support scalability, embeddings are stored in a high-performance vector database, allowing near real-time retrieval from an evolving literature base. Empirical evaluation demonstrates that the proposed pipeline retrieves and synthesizes highly relevant information regarding AMF interactions with crop systems, such as tomato (Solanum lycopersicum). The framework underscores the potential of AI-driven knowledge discovery to accelerate agroecological innovation and enhance decision-making in sustainable farming systems. 

---
