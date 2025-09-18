# Apertus: Democratizing Open and Compliant LLMs for Global Language Environments 

**Authors**: Alejandro Hernández-Cano, Alexander Hägele, Allen Hao Huang, Angelika Romanou, Antoni-Joan Solergibert, Barna Pasztor, Bettina Messmer, Dhia Garbaya, Eduard Frank Ďurech, Ido Hakimi, Juan García Giraldo, Mete Ismayilzada, Negar Foroutan, Skander Moalla, Tiancheng Chen, Vinko Sabolčec, Yixuan Xu, Michael Aerni, Badr AlKhamissi, Ines Altemir Marinas, Mohammad Hossein Amani, Matin Ansaripour, Ilia Badanin, Harold Benoit, Emanuela Boros, Nicholas Browning, Fabian Bösch, Maximilian Böther, Niklas Canova, Camille Challier, Clement Charmillot, Jonathan Coles, Jan Deriu, Arnout Devos, Lukas Drescher, Daniil Dzenhaliou, Maud Ehrmann, Dongyang Fan, Simin Fan, Silin Gao, Miguel Gila, María Grandury, Diba Hashemi, Alexander Hoyle, Jiaming Jiang, Mark Klein, Andrei Kucharavy, Anastasiia Kucherenko, Frederike Lübeck, Roman Machacek, Theofilos Manitaras, Andreas Marfurt, Kyle Matoba, Simon Matrenok, Henrique Mendoncça, Fawzi Roberto Mohamed, Syrielle Montariol, Luca Mouchel, Sven Najem-Meyer, Jingwei Ni, Gennaro Oliva, Matteo Pagliardini, Elia Palme, Andrei Panferov, Léo Paoletti, Marco Passerini, Ivan Pavlov, Auguste Poiroux, Kaustubh Ponkshe, Nathan Ranchin, Javi Rando, Mathieu Sauser, Jakhongir Saydaliev, Muhammad Ali Sayfiddinov, Marian Schneider, Stefano Schuppli, Marco Scialanga, Andrei Semenov, Kumar Shridhar, Raghav Singhal, Anna Sotnikova, Alexander Sternfeld, Ayush Kumar Tarun, Paul Teiletche, Jannis Vamvas, Xiaozhe Yao, Hao Zhao Alexander Ilic, Ana Klimovic, Andreas Krause, Caglar Gulcehre, David Rosenthal, Elliott Ash, Florian Tramèr, Joost VandeVondele, Livio Veraldi, Martin Rajman, Thomas Schulthess, Torsten Hoefler, Antoine Bosselut, Martin Jaggi, Imanol Schlag  

**Link**: [PDF](https://arxiv.org/pdf/2509.14233)  

**Abstract**: We present Apertus, a fully open suite of large language models (LLMs) designed to address two systemic shortcomings in today's open model ecosystem: data compliance and multilingual representation. Unlike many prior models that release weights without reproducible data pipelines or regard for content-owner rights, Apertus models are pretrained exclusively on openly available data, retroactively respecting this http URL exclusions and filtering for non-permissive, toxic, and personally identifiable content. To mitigate risks of memorization, we adopt the Goldfish objective during pretraining, strongly suppressing verbatim recall of data while retaining downstream task performance. The Apertus models also expand multilingual coverage, training on 15T tokens from over 1800 languages, with ~40% of pretraining data allocated to non-English content. Released at 8B and 70B scales, Apertus approaches state-of-the-art results among fully open models on multilingual benchmarks, rivalling or surpassing open-weight counterparts. Beyond model weights, we release all scientific artifacts from our development cycle with a permissive license, including data preparation scripts, checkpoints, evaluation suites, and training code, enabling transparent audit and extension. 

---
# Framing Migration: A Computational Analysis of UK Parliamentary Discourse 

**Authors**: Vahid Ghafouri, Robert McNeil, Teodor Yankov, Madeleine Sumption, Luc Rocher, Scott A. Hale, Adam Mahdi  

**Link**: [PDF](https://arxiv.org/pdf/2509.14197)  

**Abstract**: We present a large-scale computational analysis of migration-related discourse in UK parliamentary debates spanning over 75 years and compare it with US congressional discourse. Using open-weight LLMs, we annotate each statement with high-level stances toward migrants and track the net tone toward migrants across time and political parties. For the UK, we extend this with a semi-automated framework for extracting fine-grained narrative frames to capture nuances of migration discourse. Our findings show that, while US discourse has grown increasingly polarised, UK parliamentary attitudes remain relatively aligned across parties, with a persistent ideological gap between Labour and the Conservatives, reaching its most negative level in 2025. The analysis of narrative frames in the UK parliamentary statements reveals a shift toward securitised narratives such as border control and illegal immigration, while longer-term integration-oriented frames such as social integration have declined. Moreover, discussions of national law about immigration have been replaced over time by international law and human rights, revealing nuances in discourse trends. Taken together broadly, our findings demonstrate how LLMs can support scalable, fine-grained discourse analysis in political and historical contexts. 

---
# Synthesizing Behaviorally-Grounded Reasoning Chains: A Data-Generation Framework for Personal Finance LLMs 

**Authors**: Akhil Theerthala  

**Link**: [PDF](https://arxiv.org/pdf/2509.14180)  

**Abstract**: Personalized financial advice requires consideration of user goals, constraints, risk tolerance, and jurisdiction. Prior LLM work has focused on support systems for investors and financial planners. Simultaneously, numerous recent studies examine broader personal finance tasks, including budgeting, debt management, retirement, and estate planning, through agentic pipelines that incur high maintenance costs, yielding less than 25% of their expected financial returns. In this study, we introduce a novel and reproducible framework that integrates relevant financial context with behavioral finance studies to construct supervision data for end-to-end advisors. Using this framework, we create a 19k sample reasoning dataset and conduct a comprehensive fine-tuning of the Qwen-3-8B model on the dataset. Through a held-out test split and a blind LLM-jury study, we demonstrate that through careful data curation and behavioral integration, our 8B model achieves performance comparable to significantly larger baselines (14-32B parameters) across factual accuracy, fluency, and personalization metrics while incurring 80% lower costs than the larger counterparts. 

---
# AssoCiAm: A Benchmark for Evaluating Association Thinking while Circumventing Ambiguity 

**Authors**: Yifan Liu, Wenkuan Zhao, Shanshan Zhong, Jinghui Qin, Mingfu Liang, Zhongzhan Huang, Wushao Wen  

**Link**: [PDF](https://arxiv.org/pdf/2509.14171)  

**Abstract**: Recent advancements in multimodal large language models (MLLMs) have garnered significant attention, offering a promising pathway toward artificial general intelligence (AGI). Among the essential capabilities required for AGI, creativity has emerged as a critical trait for MLLMs, with association serving as its foundation. Association reflects a model' s ability to think creatively, making it vital to evaluate and understand. While several frameworks have been proposed to assess associative ability, they often overlook the inherent ambiguity in association tasks, which arises from the divergent nature of associations and undermines the reliability of evaluations. To address this issue, we decompose ambiguity into two types-internal ambiguity and external ambiguity-and introduce AssoCiAm, a benchmark designed to evaluate associative ability while circumventing the ambiguity through a hybrid computational method. We then conduct extensive experiments on MLLMs, revealing a strong positive correlation between cognition and association. Additionally, we observe that the presence of ambiguity in the evaluation process causes MLLMs' behavior to become more random-like. Finally, we validate the effectiveness of our method in ensuring more accurate and reliable evaluations. See Project Page for the data and codes. 

---
# CS-FLEURS: A Massively Multilingual and Code-Switched Speech Dataset 

**Authors**: Brian Yan, Injy Hamed, Shuichiro Shimizu, Vasista Lodagala, William Chen, Olga Iakovenko, Bashar Talafha, Amir Hussein, Alexander Polok, Kalvin Chang, Dominik Klement, Sara Althubaiti, Puyuan Peng, Matthew Wiesner, Thamar Solorio, Ahmed Ali, Sanjeev Khudanpur, Shinji Watanabe, Chih-Chen Chen, Zhen Wu, Karim Benharrak, Anuj Diwan, Samuele Cornell, Eunjung Yeo, Kwanghee Choi, Carlos Carvalho, Karen Rosero  

**Link**: [PDF](https://arxiv.org/pdf/2509.14161)  

**Abstract**: We present CS-FLEURS, a new dataset for developing and evaluating code-switched speech recognition and translation systems beyond high-resourced languages. CS-FLEURS consists of 4 test sets which cover in total 113 unique code-switched language pairs across 52 languages: 1) a 14 X-English language pair set with real voices reading synthetically generated code-switched sentences, 2) a 16 X-English language pair set with generative text-to-speech 3) a 60 {Arabic, Mandarin, Hindi, Spanish}-X language pair set with the generative text-to-speech, and 4) a 45 X-English lower-resourced language pair test set with concatenative text-to-speech. Besides the four test sets, CS-FLEURS also provides a training set with 128 hours of generative text-to-speech data across 16 X-English language pairs. Our hope is that CS-FLEURS helps to broaden the scope of future code-switched speech research. Dataset link: this https URL. 

---
# Canary-1B-v2 & Parakeet-TDT-0.6B-v3: Efficient and High-Performance Models for Multilingual ASR and AST 

**Authors**: Monica Sekoyan, Nithin Rao Koluguri, Nune Tadevosyan, Piotr Zelasko, Travis Bartley, Nick Karpov, Jagadeesh Balam, Boris Ginsburg  

**Link**: [PDF](https://arxiv.org/pdf/2509.14128)  

**Abstract**: This report introduces Canary-1B-v2, a fast, robust multilingual model for Automatic Speech Recognition (ASR) and Speech-to-Text Translation (AST). Built with a FastConformer encoder and Transformer decoder, it supports 25 languages primarily European. The model was trained on 1.7M hours of total data samples, including Granary and NeMo ASR Set 3.0, with non-speech audio added to reduce hallucinations for ASR and AST. We describe its two-stage pre-training and fine-tuning process with dynamic data balancing, as well as experiments with an nGPT encoder. Results show nGPT scales well with massive data, while FastConformer excels after fine-tuning. For timestamps, Canary-1B-v2 uses the NeMo Forced Aligner (NFA) with an auxiliary CTC model, providing reliable segment-level timestamps for ASR and AST. Evaluations show Canary-1B-v2 outperforms Whisper-large-v3 on English ASR while being 10x faster, and delivers competitive multilingual ASR and AST performance against larger models like Seamless-M4T-v2-large and LLM-based systems. We also release Parakeet-TDT-0.6B-v3, a successor to v2, offering multilingual ASR across the same 25 languages with just 600M parameters. 

---
# SSL-SSAW: Self-Supervised Learning with Sigmoid Self-Attention Weighting for Question-Based Sign Language Translation 

**Authors**: Zekang Liu, Wei Feng, Fanhua Shang, Lianyu Hu, Jichao Feng, Liqing Gao  

**Link**: [PDF](https://arxiv.org/pdf/2509.14036)  

**Abstract**: Sign Language Translation (SLT) bridges the communication gap between deaf people and hearing people, where dialogue provides crucial contextual cues to aid in translation. Building on this foundational concept, this paper proposes Question-based Sign Language Translation (QB-SLT), a novel task that explores the efficient integration of dialogue. Unlike gloss (sign language transcription) annotations, dialogue naturally occurs in communication and is easier to annotate. The key challenge lies in aligning multimodality features while leveraging the context of the question to improve translation. To address this issue, we propose a cross-modality Self-supervised Learning with Sigmoid Self-attention Weighting (SSL-SSAW) fusion method for sign language translation. Specifically, we employ contrastive learning to align multimodality features in QB-SLT, then introduce a Sigmoid Self-attention Weighting (SSAW) module for adaptive feature extraction from question and sign language sequences. Additionally, we leverage available question text through self-supervised learning to enhance representation and translation capabilities. We evaluated our approach on newly constructed CSL-Daily-QA and PHOENIX-2014T-QA datasets, where SSL-SSAW achieved SOTA performance. Notably, easily accessible question assistance can achieve or even surpass the performance of gloss assistance. Furthermore, visualization results demonstrate the effectiveness of incorporating dialogue in improving translation quality. 

---
# Enhancing Multi-Agent Debate System Performance via Confidence Expression 

**Authors**: Zijie Lin, Bryan Hooi  

**Link**: [PDF](https://arxiv.org/pdf/2509.14034)  

**Abstract**: Generative Large Language Models (LLMs) have demonstrated remarkable performance across a wide range of tasks. Recent research has introduced Multi-Agent Debate (MAD) systems, which leverage multiple LLMs to simulate human debate and thereby improve task performance. However, while some LLMs may possess superior knowledge or reasoning capabilities for specific tasks, they often struggle to clearly communicate this advantage during debates, in part due to a lack of confidence expression. Moreover, inappropriate confidence expression can cause agents in MAD systems to either stubbornly maintain incorrect beliefs or converge prematurely on suboptimal answers, ultimately reducing debate effectiveness and overall system performance. To address these challenges, we propose incorporating confidence expression into MAD systems to allow LLMs to explicitly communicate their confidence levels. To validate this approach, we develop ConfMAD, a MAD framework that integrates confidence expression throughout the debate process. Experimental results demonstrate the effectiveness of our method, and we further analyze how confidence influences debate dynamics, offering insights into the design of confidence-aware MAD systems. 

---
# You Are What You Train: Effects of Data Composition on Training Context-aware Machine Translation Models 

**Authors**: Paweł Mąka, Yusuf Can Semerci, Jan Scholtes, Gerasimos Spanakis  

**Link**: [PDF](https://arxiv.org/pdf/2509.14031)  

**Abstract**: Achieving human-level translations requires leveraging context to ensure coherence and handle complex phenomena like pronoun disambiguation. Sparsity of contextually rich examples in the standard training data has been hypothesized as the reason for the difficulty of context utilization. In this work, we systematically validate this claim in both single- and multilingual settings by constructing training datasets with a controlled proportions of contextually relevant examples. We demonstrate a strong association between training data sparsity and model performance confirming sparsity as a key bottleneck. Importantly, we reveal that improvements in one contextual phenomenon do no generalize to others. While we observe some cross-lingual transfer, it is not significantly higher between languages within the same sub-family. Finally, we propose and empirically evaluate two training strategies designed to leverage the available data. These strategies improve context utilization, resulting in accuracy gains of up to 6 and 8 percentage points on the ctxPro evaluation in single- and multilingual settings respectively. 

---
# Audio-Based Crowd-Sourced Evaluation of Machine Translation Quality 

**Authors**: Sami Ul Haq, Sheila Castilho, Yvette Graham  

**Link**: [PDF](https://arxiv.org/pdf/2509.14023)  

**Abstract**: Machine Translation (MT) has achieved remarkable performance, with growing interest in speech translation and multimodal approaches. However, despite these advancements, MT quality assessment remains largely text centric, typically relying on human experts who read and compare texts. Since many real-world MT applications (e.g Google Translate Voice Mode, iFLYTEK Translator) involve translation being spoken rather printed or read, a more natural way to assess translation quality would be through speech as opposed text-only evaluations. This study compares text-only and audio-based evaluations of 10 MT systems from the WMT General MT Shared Task, using crowd-sourced judgments collected via Amazon Mechanical Turk. We additionally, performed statistical significance testing and self-replication experiments to test reliability and consistency of audio-based approach. Crowd-sourced assessments based on audio yield rankings largely consistent with text only evaluations but, in some cases, identify significant differences between translation systems. We attribute this to speech richer, more natural modality and propose incorporating speech-based assessments into future MT evaluation frameworks. 

---
# Hala Technical Report: Building Arabic-Centric Instruction & Translation Models at Scale 

**Authors**: Hasan Abed Al Kader Hammoud, Mohammad Zbeeb, Bernard Ghanem  

**Link**: [PDF](https://arxiv.org/pdf/2509.14008)  

**Abstract**: We present Hala, a family of Arabic-centric instruction and translation models built with our translate-and-tune pipeline. We first compress a strong AR$\leftrightarrow$EN teacher to FP8 (yielding $\sim$2$\times$ higher throughput with no quality loss) and use it to create high-fidelity bilingual supervision. A lightweight language model LFM2-1.2B is then fine-tuned on this data and used to translate high-quality English instruction sets into Arabic, producing a million-scale corpus tailored to instruction following. We train Hala models at 350M, 700M, 1.2B, and 9B parameters, and apply slerp merging to balance Arabic specialization with base-model strengths. On Arabic-centric benchmarks, Hala achieves state-of-the-art results within both the "nano" ($\leq$2B) and "small" (7-9B) categories, outperforming their bases. We release models, data, evaluation, and recipes to accelerate research in Arabic NLP. 

---
# Early Stopping Chain-of-thoughts in Large Language Models 

**Authors**: Minjia Mao, Bowen Yin, Yu Zhu, Xiao Fang  

**Link**: [PDF](https://arxiv.org/pdf/2509.14004)  

**Abstract**: Reasoning large language models (LLMs) have demonstrated superior capacities in solving complicated problems by generating long chain-of-thoughts (CoT), but such a lengthy CoT incurs high inference costs. In this study, we introduce ES-CoT, an inference-time method that shortens CoT generation by detecting answer convergence and stopping early with minimal performance loss. At the end of each reasoning step, we prompt the LLM to output its current final answer, denoted as a step answer. We then track the run length of consecutive identical step answers as a measure of answer convergence. Once the run length exhibits a sharp increase and exceeds a minimum threshold, the generation is terminated. We provide both empirical and theoretical support for this heuristic: step answers steadily converge to the final answer, and large run-length jumps reliably mark this convergence. Experiments on five reasoning datasets across three LLMs show that ES-CoT reduces the number of inference tokens by about 41\% on average while maintaining accuracy comparable to standard CoT. Further, ES-CoT integrates seamlessly with self-consistency prompting and remains robust across hyperparameter choices, highlighting it as a practical and effective approach for efficient reasoning. 

---
# Slim-SC: Thought Pruning for Efficient Scaling with Self-Consistency 

**Authors**: Colin Hong, Xu Guo, Anand Chaanan Singh, Esha Choukse, Dmitrii Ustiugov  

**Link**: [PDF](https://arxiv.org/pdf/2509.13990)  

**Abstract**: Recently, Test-Time Scaling (TTS) has gained increasing attention for improving LLM reasoning performance at test time without retraining the model. A notable TTS technique is Self-Consistency (SC), which generates multiple reasoning chains in parallel and selects the final answer via majority voting. While effective, the order-of-magnitude computational overhead limits its broad deployment. Prior attempts to accelerate SC mainly rely on model-based confidence scores or heuristics with limited empirical support. For the first time, we theoretically and empirically analyze the inefficiencies of SC and reveal actionable opportunities for improvement. Building on these insights, we propose Slim-SC, a step-wise pruning strategy that identifies and removes redundant chains using inter-chain similarity at the thought level. Experiments on three STEM reasoning datasets and two recent LLM architectures show that Slim-SC reduces inference latency and KVC usage by up to 45% and 26%, respectively, with R1-Distill, while maintaining or improving accuracy, thus offering a simple yet efficient TTS alternative for SC. 

---
# Long-context Reference-based MT Quality Estimation 

**Authors**: Sami Ul Haq, Chinonso Cynthia Osuji, Sheila Castilho, Brian Davis  

**Link**: [PDF](https://arxiv.org/pdf/2509.13980)  

**Abstract**: In this paper, we present our submission to the Tenth Conference on Machine Translation (WMT25) Shared Task on Automated Translation Quality Evaluation.
Our systems are built upon the COMET framework and trained to predict segment-level Error Span Annotation (ESA) scores using augmented long-context data.
To construct long-context training data, we concatenate in-domain, human-annotated sentences and compute a weighted average of their scores.
We integrate multiple human judgment datasets (MQM, SQM, and DA) by normalising their scales and train multilingual regression models to predict quality scores from the source, hypothesis, and reference translations.
Experimental results show that incorporating long-context information improves correlations with human judgments compared to models trained only on short segments. 

---
# Linguistic Nepotism: Trading-off Quality for Language Preference in Multilingual RAG 

**Authors**: Dayeon Ki, Marine Carpuat, Paul McNamee, Daniel Khashabi, Eugene Yang, Dawn Lawrie, Kevin Duh  

**Link**: [PDF](https://arxiv.org/pdf/2509.13930)  

**Abstract**: Multilingual Retrieval-Augmented Generation (mRAG) systems enable language models to answer knowledge-intensive queries with citation-supported responses across languages. While such systems have been proposed, an open questions is whether the mixture of different document languages impacts generation and citation in unintended ways. To investigate, we introduce a controlled methodology using model internals to measure language preference while holding other factors such as document relevance constant. Across eight languages and six open-weight models, we find that models preferentially cite English sources when queries are in English, with this bias amplified for lower-resource languages and for documents positioned mid-context. Crucially, we find that models sometimes trade-off document relevance for language preference, indicating that citation choices are not always driven by informativeness alone. Our findings shed light on how language models leverage multilingual context and influence citation behavior. 

---
# Do Large Language Models Understand Word Senses? 

**Authors**: Domenico Meconi, Simone Stirpe, Federico Martelli, Leonardo Lavalle, Roberto Navigli  

**Link**: [PDF](https://arxiv.org/pdf/2509.13905)  

**Abstract**: Understanding the meaning of words in context is a fundamental capability for Large Language Models (LLMs). Despite extensive evaluation efforts, the extent to which LLMs show evidence that they truly grasp word senses remains underexplored. In this paper, we address this gap by evaluating both i) the Word Sense Disambiguation (WSD) capabilities of instruction-tuned LLMs, comparing their performance to state-of-the-art systems specifically designed for the task, and ii) the ability of two top-performing open- and closed-source LLMs to understand word senses in three generative settings: definition generation, free-form explanation, and example generation. Notably, we find that, in the WSD task, leading models such as GPT-4o and DeepSeek-V3 achieve performance on par with specialized WSD systems, while also demonstrating greater robustness across domains and levels of difficulty. In the generation tasks, results reveal that LLMs can explain the meaning of words in context up to 98\% accuracy, with the highest performance observed in the free-form explanation task, which best aligns with their generative capabilities. 

---
# Combating Biomedical Misinformation through Multi-modal Claim Detection and Evidence-based Verification 

**Authors**: Mariano Barone, Antonio Romano, Giuseppe Riccio, Marco Postiglione, Vincenzo Moscato  

**Link**: [PDF](https://arxiv.org/pdf/2509.13888)  

**Abstract**: Misinformation in healthcare, from vaccine hesitancy to unproven treatments, poses risks to public health and trust in medical systems. While machine learning and natural language processing have advanced automated fact-checking, validating biomedical claims remains uniquely challenging due to complex terminology, the need for domain expertise, and the critical importance of grounding in scientific evidence. We introduce CER (Combining Evidence and Reasoning), a novel framework for biomedical fact-checking that integrates scientific evidence retrieval, reasoning via large language models, and supervised veracity prediction. By integrating the text-generation capabilities of large language models with advanced retrieval techniques for high-quality biomedical scientific evidence, CER effectively mitigates the risk of hallucinations, ensuring that generated outputs are grounded in verifiable, evidence-based sources. Evaluations on expert-annotated datasets (HealthFC, BioASQ-7b, SciFact) demonstrate state-of-the-art performance and promising cross-dataset generalization. Code and data are released for transparency and reproducibility: this https URL 

---
# Combining Evidence and Reasoning for Biomedical Fact-Checking 

**Authors**: Mariano Barone, Antonio Romano, Giuseppe Riccio, Marco Postiglione, Vincenzo Moscato  

**Link**: [PDF](https://arxiv.org/pdf/2509.13879)  

**Abstract**: Misinformation in healthcare, from vaccine hesitancy to unproven treatments, poses risks to public health and trust in medical sys- tems. While machine learning and natural language processing have advanced automated fact-checking, validating biomedical claims remains uniquely challenging due to complex terminol- ogy, the need for domain expertise, and the critical importance of grounding in scientific evidence. We introduce CER (Combin- ing Evidence and Reasoning), a novel framework for biomedical fact-checking that integrates scientific evidence retrieval, reasoning via large language models, and supervised veracity prediction. By integrating the text-generation capabilities of large language mod- els with advanced retrieval techniques for high-quality biomedical scientific evidence, CER effectively mitigates the risk of halluci- nations, ensuring that generated outputs are grounded in veri- fiable, evidence-based sources. Evaluations on expert-annotated datasets (HealthFC, BioASQ-7b, SciFact) demonstrate state-of-the- art performance and promising cross-dataset generalization. Code and data are released for transparency and reproducibility: https: //github.com/PRAISELab-PicusLab/CER. 

---
# Do LLMs Align Human Values Regarding Social Biases? Judging and Explaining Social Biases with LLMs 

**Authors**: Yang Liu, Chenhui Chu  

**Link**: [PDF](https://arxiv.org/pdf/2509.13869)  

**Abstract**: Large language models (LLMs) can lead to undesired consequences when misaligned with human values, especially in scenarios involving complex and sensitive social biases. Previous studies have revealed the misalignment of LLMs with human values using expert-designed or agent-based emulated bias scenarios. However, it remains unclear whether the alignment of LLMs with human values differs across different types of scenarios (e.g., scenarios containing negative vs. non-negative questions). In this study, we investigate the alignment of LLMs with human values regarding social biases (HVSB) in different types of bias scenarios. Through extensive analysis of 12 LLMs from four model families and four datasets, we demonstrate that LLMs with large model parameter scales do not necessarily have lower misalignment rate and attack success rate. Moreover, LLMs show a certain degree of alignment preference for specific types of scenarios and the LLMs from the same model family tend to have higher judgment consistency. In addition, we study the understanding capacity of LLMs with their explanations of HVSB. We find no significant differences in the understanding of HVSB across LLMs. We also find LLMs prefer their own generated explanations. Additionally, we endow smaller language models (LMs) with the ability to explain HVSB. The generation results show that the explanations generated by the fine-tuned smaller LMs are more readable, but have a relatively lower model agreeability. 

---
# Large Language Models Discriminate Against Speakers of German Dialects 

**Authors**: Minh Duc Bui, Carolin Holtermann, Valentin Hofmann, Anne Lauscher, Katharina von der Wense  

**Link**: [PDF](https://arxiv.org/pdf/2509.13835)  

**Abstract**: Dialects represent a significant component of human culture and are found across all regions of the world. In Germany, more than 40% of the population speaks a regional dialect (Adler and Hansen, 2022). However, despite cultural importance, individuals speaking dialects often face negative societal stereotypes. We examine whether such stereotypes are mirrored by large language models (LLMs). We draw on the sociolinguistic literature on dialect perception to analyze traits commonly associated with dialect speakers. Based on these traits, we assess the dialect naming bias and dialect usage bias expressed by LLMs in two tasks: an association task and a decision task. To assess a model's dialect usage bias, we construct a novel evaluation corpus that pairs sentences from seven regional German dialects (e.g., Alemannic and Bavarian) with their standard German counterparts. We find that: (1) in the association task, all evaluated LLMs exhibit significant dialect naming and dialect usage bias against German dialect speakers, reflected in negative adjective associations; (2) all models reproduce these dialect naming and dialect usage biases in their decision making; and (3) contrary to prior work showing minimal bias with explicit demographic mentions, we find that explicitly labeling linguistic demographics--German dialect speakers--amplifies bias more than implicit cues like dialect usage. 

---
# Findings of the Third Automatic Minuting (AutoMin) Challenge 

**Authors**: Kartik Shinde, Laurent Besacier, Ondrej Bojar, Thibaut Thonet, Tirthankar Ghosal  

**Link**: [PDF](https://arxiv.org/pdf/2509.13814)  

**Abstract**: This paper presents the third edition of AutoMin, a shared task on automatic meeting summarization into minutes. In 2025, AutoMin featured the main task of minuting, the creation of structured meeting minutes, as well as a new task: question answering (QA) based on meeting transcripts.
The minuting task covered two languages, English and Czech, and two domains: project meetings and European Parliament sessions. The QA task focused solely on project meetings and was available in two settings: monolingual QA in English, and cross-lingual QA, where questions were asked and answered in Czech based on English meetings.
Participation in 2025 was more limited compared to previous years, with only one team joining the minuting task and two teams participating in QA. However, as organizers, we included multiple baseline systems to enable a comprehensive evaluation of current (2025) large language models (LLMs) on both tasks. 

---
# Geometric Uncertainty for Detecting and Correcting Hallucinations in LLMs 

**Authors**: Edward Phillips, Sean Wu, Soheila Molaei, Danielle Belgrave, Anshul Thakur, David Clifton  

**Link**: [PDF](https://arxiv.org/pdf/2509.13813)  

**Abstract**: Large language models demonstrate impressive results across diverse tasks but are still known to hallucinate, generating linguistically plausible but incorrect answers to questions. Uncertainty quantification has been proposed as a strategy for hallucination detection, but no existing black-box approach provides estimates for both global and local uncertainty. The former attributes uncertainty to a batch of responses, while the latter attributes uncertainty to individual responses. Current local methods typically rely on white-box access to internal model states, whilst black-box methods only provide global uncertainty estimates. We introduce a geometric framework to address this, based on archetypal analysis of batches of responses sampled with only black-box model access. At the global level, we propose Geometric Volume, which measures the convex hull volume of archetypes derived from response embeddings. At the local level, we propose Geometric Suspicion, which ranks responses by reliability and enables hallucination reduction through preferential response selection. Unlike prior dispersion methods which yield only a single global score, our approach provides semantic boundary points which have utility for attributing reliability to individual responses. Experiments show that our framework performs comparably to or better than prior methods on short form question-answering datasets, and achieves superior results on medical datasets where hallucinations carry particularly critical risks. We also provide theoretical justification by proving a link between convex hull volume and entropy. 

---
# Measuring Gender Bias in Job Title Matching for Grammatical Gender Languages 

**Authors**: Laura García-Sardiña, Hermenegildo Fabregat, Daniel Deniz, Rabih Zbib  

**Link**: [PDF](https://arxiv.org/pdf/2509.13803)  

**Abstract**: This work sets the ground for studying how explicit grammatical gender assignment in job titles can affect the results of automatic job ranking systems. We propose the usage of metrics for ranking comparison controlling for gender to evaluate gender bias in job title ranking systems, in particular RBO (Rank-Biased Overlap). We generate and share test sets for a job title matching task in four grammatical gender languages, including occupations in masculine and feminine form and annotated by gender and matching relevance. We use the new test sets and the proposed methodology to evaluate the gender bias of several out-of-the-box multilingual models to set as baselines, showing that all of them exhibit varying degrees of gender bias. 

---
# Teaching According to Talents! Instruction Tuning LLMs with Competence-Aware Curriculum Learning 

**Authors**: Yangning Li, Tingwei Lu, Yinghui Li, Yankai Chen, Wei-Chieh Huang, Wenhao Jiang, Hui Wang, Hai-Tao Zheng, Philip S.Yu  

**Link**: [PDF](https://arxiv.org/pdf/2509.13790)  

**Abstract**: Efficient instruction tuning aims to enhance the ultimate performance of large language models (LLMs) trained on a given instruction dataset. Curriculum learning as a typical data organization strategy has shown preliminary effectiveness in instruction tuning. However, current curriculum tuning methods suffer from the curriculum rigidity, since they rely solely on static heuristic difficulty metrics. These methods fail to adapt to the evolving capabilities of models during training, resulting in a fixed and potentially sub-optimal learning trajectory. To address the issue, Competence-Aware Multi-Perspective cUrriculum inStruction tuning framework termed CAMPUS is proposed. CAMPUS offers several advantages: (1) Dynamic selection for sub-curriculum. (2) Competency-aware adjustment to the curriculum schedule. (3) Multiple difficulty-based scheduling. Extensive experiments prove the superior performance of CAMPUS, compared to other state-of-the-art baselines for efficient instruction tuning. 

---
# Exploring Data and Parameter Efficient Strategies for Arabic Dialect Identifications 

**Authors**: Vani Kanjirangat, Ljiljana Dolamic, Fabio Rinaldi  

**Link**: [PDF](https://arxiv.org/pdf/2509.13775)  

**Abstract**: This paper discusses our exploration of different data-efficient and parameter-efficient approaches to Arabic Dialect Identification (ADI). In particular, we investigate various soft-prompting strategies, including prefix-tuning, prompt-tuning, P-tuning, and P-tuning V2, as well as LoRA reparameterizations. For the data-efficient strategy, we analyze hard prompting with zero-shot and few-shot inferences to analyze the dialect identification capabilities of Large Language Models (LLMs). For the parameter-efficient PEFT approaches, we conducted our experiments using Arabic-specific encoder models on several major datasets. We also analyzed the n-shot inferences on open-source decoder-only models, a general multilingual model (Phi-3.5), and an Arabic-specific one(SILMA). We observed that the LLMs generally struggle to differentiate the dialectal nuances in the few-shot or zero-shot setups. The soft-prompted encoder variants perform better, while the LoRA-based fine-tuned models perform best, even surpassing full fine-tuning. 

---
# Implementing a Logical Inference System for Japanese Comparatives 

**Authors**: Yosuke Mikami, Daiki Matsuoka, Hitomi Yanaka  

**Link**: [PDF](https://arxiv.org/pdf/2509.13734)  

**Abstract**: Natural Language Inference (NLI) involving comparatives is challenging because it requires understanding quantities and comparative relations expressed by sentences. While some approaches leverage Large Language Models (LLMs), we focus on logic-based approaches grounded in compositional semantics, which are promising for robust handling of numerical and logical expressions. Previous studies along these lines have proposed logical inference systems for English comparatives. However, it has been pointed out that there are several morphological and semantic differences between Japanese and English comparatives. These differences make it difficult to apply such systems directly to Japanese comparatives. To address this gap, this study proposes ccg-jcomp, a logical inference system for Japanese comparatives based on compositional semantics. We evaluate the proposed system on a Japanese NLI dataset containing comparative expressions. We demonstrate the effectiveness of our system by comparing its accuracy with that of existing LLMs. 

---
# DSPC: Dual-Stage Progressive Compression Framework for Efficient Long-Context Reasoning 

**Authors**: Yaxin Gao, Yao Lu, Zongfei Zhang, Jiaqi Nie, Shanqing Yu, Qi Xuan  

**Link**: [PDF](https://arxiv.org/pdf/2509.13723)  

**Abstract**: Large language models (LLMs) have achieved remarkable success in many natural language processing (NLP) tasks. To achieve more accurate output, the prompts used to drive LLMs have become increasingly longer, which incurs higher computational costs. To address this prompt inflation problem, prompt compression has been proposed. However, most existing methods require training a small auxiliary model for compression, incurring a significant amount of additional computation. To avoid this, we propose a two-stage, training-free approach, called Dual-Stage Progressive Compression (DSPC). In the coarse-grained stage, semantic-related sentence filtering removes sentences with low semantic value based on TF-IDF. In the fine-grained stage, token importance is assessed using attention contribution, cross-model loss difference, and positional importance, enabling the pruning of low-utility tokens while preserving semantics. We validate DSPC on LLaMA-3.1-8B-Instruct and GPT-3.5-Turbo under a constrained token budget and observe consistent improvements. For instance, in the FewShot task of the Longbench dataset, DSPC achieves a performance of 49.17 by using only 3x fewer tokens, outperforming the best state-of-the-art baseline LongLLMLingua by 7.76. 

---
# Automated Triaging and Transfer Learning of Incident Learning Safety Reports Using Large Language Representational Models 

**Authors**: Peter Beidler, Mark Nguyen, Kevin Lybarger, Ola Holmberg, Eric Ford, John Kang  

**Link**: [PDF](https://arxiv.org/pdf/2509.13706)  

**Abstract**: PURPOSE: Incident reports are an important tool for safety and quality improvement in healthcare, but manual review is time-consuming and requires subject matter expertise. Here we present a natural language processing (NLP) screening tool to detect high-severity incident reports in radiation oncology across two institutions.
METHODS AND MATERIALS: We used two text datasets to train and evaluate our NLP models: 7,094 reports from our institution (Inst.), and 571 from IAEA SAFRON (SF), all of which had severity scores labeled by clinical content experts. We trained and evaluated two types of models: baseline support vector machines (SVM) and BlueBERT which is a large language model pretrained on PubMed abstracts and hospitalized patient data. We assessed for generalizability of our model in two ways. First, we evaluated models trained using Inst.-train on SF-test. Second, we trained a BlueBERT_TRANSFER model that was first fine-tuned on Inst.-train then on SF-train before testing on SF-test set. To further analyze model performance, we also examined a subset of 59 reports from our Inst. dataset, which were manually edited for clarity.
RESULTS Classification performance on the Inst. test achieved AUROC 0.82 using SVM and 0.81 using BlueBERT. Without cross-institution transfer learning, performance on the SF test was limited to an AUROC of 0.42 using SVM and 0.56 using BlueBERT. BlueBERT_TRANSFER, which was fine-tuned on both datasets, improved the performance on SF test to AUROC 0.78. Performance of SVM, and BlueBERT_TRANSFER models on the manually curated Inst. reports (AUROC 0.85 and 0.74) was similar to human performance (AUROC 0.81).
CONCLUSION: In summary, we successfully developed cross-institution NLP models on incident report text from radiation oncology centers. These models were able to detect high-severity reports similarly to humans on a curated dataset. 

---
# DSCC-HS: A Dynamic Self-Reinforcing Framework for Hallucination Suppression in Large Language Models 

**Authors**: Xiao Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2509.13702)  

**Abstract**: Large Language Model (LLM) hallucination is a significant barrier to their reliable deployment. Current methods like Retrieval-Augmented Generation (RAG) are often reactive. We introduce **Dynamic Self-reinforcing Calibration for Hallucination Suppression (DSCC-HS)**, a novel, proactive framework that intervenes during autoregressive decoding. Inspired by dual-process cognitive theory, DSCC-HS uses a compact proxy model, trained in adversarial roles as a Factual Alignment Proxy (FAP) and a Hallucination Detection Proxy (HDP). During inference, these proxies dynamically steer a large target model by injecting a real-time steering vector, which is the difference between FAP and HDP logits, at each decoding step. This plug-and-play approach requires no modification to the target model. Our experiments on TruthfulQA and BioGEN show DSCC-HS achieves state-of-the-art performance. On TruthfulQA, it reached a 99.2% Factual Consistency Rate (FCR). On the long-form BioGEN benchmark, it attained the highest FActScore of 46.50. These results validate DSCC-HS as a principled and efficient solution for enhancing LLM factuality. 

---
# Integrating Text and Time-Series into (Large) Language Models to Predict Medical Outcomes 

**Authors**: Iyadh Ben Cheikh Larbi, Ajay Madhavan Ravichandran, Aljoscha Burchardt, Roland Roller  

**Link**: [PDF](https://arxiv.org/pdf/2509.13696)  

**Abstract**: Large language models (LLMs) excel at text generation, but their ability to handle clinical classification tasks involving structured data, such as time series, remains underexplored. In this work, we adapt instruction-tuned LLMs using DSPy-based prompt optimization to process clinical notes and structured EHR inputs jointly. Our results show that this approach achieves performance on par with specialized multimodal systems while requiring less complexity and offering greater adaptability across tasks. 

---
# Can Large Language Models Robustly Perform Natural Language Inference for Japanese Comparatives? 

**Authors**: Yosuke Mikami, Daiki Matsuoka, Hitomi Yanaka  

**Link**: [PDF](https://arxiv.org/pdf/2509.13695)  

**Abstract**: Large Language Models (LLMs) perform remarkably well in Natural Language Inference (NLI). However, NLI involving numerical and logical expressions remains challenging. Comparatives are a key linguistic phenomenon related to such inference, but the robustness of LLMs in handling them, especially in languages that are not dominant in the models' training data, such as Japanese, has not been sufficiently explored. To address this gap, we construct a Japanese NLI dataset that focuses on comparatives and evaluate various LLMs in zero-shot and few-shot settings. Our results show that the performance of the models is sensitive to the prompt formats in the zero-shot setting and influenced by the gold labels in the few-shot examples. The LLMs also struggle to handle linguistic phenomena unique to Japanese. Furthermore, we observe that prompts containing logical semantic representations help the models predict the correct labels for inference problems that they struggle to solve even with few-shot examples. 

---
# Improving Context Fidelity via Native Retrieval-Augmented Reasoning 

**Authors**: Suyuchen Wang, Jinlin Wang, Xinyu Wang, Shiqi Li, Xiangru Tang, Sirui Hong, Xiao-Wen Chang, Chenglin Wu, Bang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.13683)  

**Abstract**: Large language models (LLMs) often struggle with context fidelity, producing inconsistent answers when responding to questions based on provided information. Existing approaches either rely on expensive supervised fine-tuning to generate evidence post-answer or train models to perform web searches without necessarily improving utilization of the given context. We propose CARE, a novel native retrieval-augmented reasoning framework that teaches LLMs to explicitly integrate in-context evidence within their reasoning process with the model's own retrieval capabilities. Our method requires limited labeled evidence data while significantly enhancing both retrieval accuracy and answer generation performance through strategically retrieved in-context tokens in the reasoning chain. Extensive experiments on multiple real-world and counterfactual QA benchmarks demonstrate that our approach substantially outperforms supervised fine-tuning, traditional retrieval-augmented generation methods, and external retrieval solutions. This work represents a fundamental advancement in making LLMs more accurate, reliable, and efficient for knowledge-intensive tasks. 

---
# AgentCTG: Harnessing Multi-Agent Collaboration for Fine-Grained Precise Control in Text Generation 

**Authors**: Xinxu Zhou, Jiaqi Bai, Zhenqi Sun, Fanxiang Zeng, Yue Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.13677)  

**Abstract**: Although significant progress has been made in many tasks within the field of Natural Language Processing (NLP), Controlled Text Generation (CTG) continues to face numerous challenges, particularly in achieving fine-grained conditional control over generation. Additionally, in real scenario and online applications, cost considerations, scalability, domain knowledge learning and more precise control are required, presenting more challenge for CTG. This paper introduces a novel and scalable framework, AgentCTG, which aims to enhance precise and complex control over the text generation by simulating the control and regulation mechanisms in multi-agent workflows. We explore various collaboration methods among different agents and introduce an auto-prompt module to further enhance the generation effectiveness. AgentCTG achieves state-of-the-art results on multiple public datasets. To validate its effectiveness in practical applications, we propose a new challenging Character-Driven Rewriting task, which aims to convert the original text into new text that conform to specific character profiles and simultaneously preserve the domain knowledge. When applied to online navigation with role-playing, our approach significantly enhances the driving experience through improved content delivery. By optimizing the generation of contextually relevant text, we enable a more immersive interaction within online communities, fostering greater personalization and user engagement. 

---
# CL$^2$GEC: A Multi-Discipline Benchmark for Continual Learning in Chinese Literature Grammatical Error Correction 

**Authors**: Shang Qin, Jingheng Ye, Yinghui Li, Hai-Tao Zheng, Qi Li, Jinxiao Shan, Zhixing Li, Hong-Gee Kim  

**Link**: [PDF](https://arxiv.org/pdf/2509.13672)  

**Abstract**: The growing demand for automated writing assistance in diverse academic domains highlights the need for robust Chinese Grammatical Error Correction (CGEC) systems that can adapt across disciplines. However, existing CGEC research largely lacks dedicated benchmarks for multi-disciplinary academic writing, overlooking continual learning (CL) as a promising solution to handle domain-specific linguistic variation and prevent catastrophic forgetting. To fill this crucial gap, we introduce CL$^2$GEC, the first Continual Learning benchmark for Chinese Literature Grammatical Error Correction, designed to evaluate adaptive CGEC across multiple academic fields. Our benchmark includes 10,000 human-annotated sentences spanning 10 disciplines, each exhibiting distinct linguistic styles and error patterns. CL$^2$GEC focuses on evaluating grammatical error correction in a continual learning setting, simulating sequential exposure to diverse academic disciplines to reflect real-world editorial dynamics. We evaluate large language models under sequential tuning, parameter-efficient adaptation, and four representative CL algorithms, using both standard GEC metrics and continual learning metrics adapted to task-level variation. Experimental results reveal that regularization-based methods mitigate forgetting more effectively than replay-based or naive sequential approaches. Our benchmark provides a rigorous foundation for future research in adaptive grammatical error correction across diverse academic domains. 

---
# Sparse Neurons Carry Strong Signals of Question Ambiguity in LLMs 

**Authors**: Zhuoxuan Zhang, Jinhao Duan, Edward Kim, Kaidi Xu  

**Link**: [PDF](https://arxiv.org/pdf/2509.13664)  

**Abstract**: Ambiguity is pervasive in real-world questions, yet large language models (LLMs) often respond with confident answers rather than seeking clarification. In this work, we show that question ambiguity is linearly encoded in the internal representations of LLMs and can be both detected and controlled at the neuron level. During the model's pre-filling stage, we identify that a small number of neurons, as few as one, encode question ambiguity information. Probes trained on these Ambiguity-Encoding Neurons (AENs) achieve strong performance on ambiguity detection and generalize across datasets, outperforming prompting-based and representation-based baselines. Layerwise analysis reveals that AENs emerge from shallow layers, suggesting early encoding of ambiguity signals in the model's processing pipeline. Finally, we show that through manipulating AENs, we can control LLM's behavior from direct answering to abstention. Our findings reveal that LLMs form compact internal representations of question ambiguity, enabling interpretable and controllable behavior. 

---
# Latent Traits and Cross-Task Transfer: Deconstructing Dataset Interactions in LLM Fine-tuning 

**Authors**: Shambhavi Krishna, Atharva Naik, Chaitali Agarwal, Sudharshan Govindan, Taesung Lee, Haw-Shiuan Chang  

**Link**: [PDF](https://arxiv.org/pdf/2509.13624)  

**Abstract**: Large language models are increasingly deployed across diverse applications. This often includes tasks LLMs have not encountered during training. This implies that enumerating and obtaining the high-quality training data for all tasks is infeasible. Thus, we often need to rely on transfer learning using datasets with different characteristics, and anticipate out-of-distribution requests. Motivated by this practical need, we propose an analysis framework, building a transfer learning matrix and dimensionality reduction, to dissect these cross-task interactions. We train and analyze 10 models to identify latent abilities (e.g., Reasoning, Sentiment Classification, NLU, Arithmetic) and discover the side effects of the transfer learning. Our findings reveal that performance improvements often defy explanations based on surface-level dataset similarity or source data quality. Instead, hidden statistical factors of the source dataset, such as class distribution and generation length proclivities, alongside specific linguistic features, are actually more influential. This work offers insights into the complex dynamics of transfer learning, paving the way for more predictable and effective LLM adaptation. 

---
# Overview of Dialog System Evaluation Track: Dimensionality, Language, Culture and Safety at DSTC 12 

**Authors**: John Mendonça, Lining Zhang, Rahul Mallidi, Alon Lavie, Isabel Trancoso, Luis Fernando D'Haro, João Sedoc  

**Link**: [PDF](https://arxiv.org/pdf/2509.13569)  

**Abstract**: The rapid advancement of Large Language Models (LLMs) has intensified the need for robust dialogue system evaluation, yet comprehensive assessment remains challenging. Traditional metrics often prove insufficient, and safety considerations are frequently narrowly defined or culturally biased. The DSTC12 Track 1, "Dialog System Evaluation: Dimensionality, Language, Culture and Safety," is part of the ongoing effort to address these critical gaps. The track comprised two subtasks: (1) Dialogue-level, Multi-dimensional Automatic Evaluation Metrics, and (2) Multilingual and Multicultural Safety Detection. For Task 1, focused on 10 dialogue dimensions, a Llama-3-8B baseline achieved the highest average Spearman's correlation (0.1681), indicating substantial room for improvement. In Task 2, while participating teams significantly outperformed a Llama-Guard-3-1B baseline on the multilingual safety subset (top ROC-AUC 0.9648), the baseline proved superior on the cultural subset (0.5126 ROC-AUC), highlighting critical needs in culturally-aware safety. This paper describes the datasets and baselines provided to participants, as well as submission evaluation results for each of the two proposed subtasks. 

---
# Op-Fed: Opinion, Stance, and Monetary Policy Annotations on FOMC Transcripts Using Active Learning 

**Authors**: Alisa Kanganis, Katherine A. Keith  

**Link**: [PDF](https://arxiv.org/pdf/2509.13539)  

**Abstract**: The U.S. Federal Open Market Committee (FOMC) regularly discusses and sets monetary policy, affecting the borrowing and spending decisions of millions of people. In this work, we release Op-Fed, a dataset of 1044 human-annotated sentences and their contexts from FOMC transcripts. We faced two major technical challenges in dataset creation: imbalanced classes -- we estimate fewer than 8% of sentences express a non-neutral stance towards monetary policy -- and inter-sentence dependence -- 65% of instances require context beyond the sentence-level. To address these challenges, we developed a five-stage hierarchical schema to isolate aspects of opinion, monetary policy, and stance towards monetary policy as well as the level of context needed. Second, we selected instances to annotate using active learning, roughly doubling the number of positive instances across all schema aspects. Using Op-Fed, we found a top-performing, closed-weight LLM achieves 0.80 zero-shot accuracy in opinion classification but only 0.61 zero-shot accuracy classifying stance towards monetary policy -- below our human baseline of 0.89. We expect Op-Fed to be useful for future model training, confidence calibration, and as a seed dataset for future annotation efforts. 

---
# Gender-Neutral Rewriting in Italian: Models, Approaches, and Trade-offs 

**Authors**: Andrea Piergentili, Beatrice Savoldi, Matteo Negri, Luisa Bentivogli  

**Link**: [PDF](https://arxiv.org/pdf/2509.13480)  

**Abstract**: Gender-neutral rewriting (GNR) aims to reformulate text to eliminate unnecessary gender specifications while preserving meaning, a particularly challenging task in grammatical-gender languages like Italian. In this work, we conduct the first systematic evaluation of state-of-the-art large language models (LLMs) for Italian GNR, introducing a two-dimensional framework that measures both neutrality and semantic fidelity to the input. We compare few-shot prompting across multiple LLMs, fine-tune selected models, and apply targeted cleaning to boost task relevance. Our findings show that open-weight LLMs outperform the only existing model dedicated to GNR in Italian, whereas our fine-tuned models match or exceed the best open-weight LLM's performance at a fraction of its size. Finally, we discuss the trade-off between optimizing the training data for neutrality and meaning preservation. 

---
# Language models' activations linearly encode training-order recency 

**Authors**: Dmitrii Krasheninnikov, Richard E. Turner, David Krueger  

**Link**: [PDF](https://arxiv.org/pdf/2509.14223)  

**Abstract**: We show that language models' activations linearly encode when information was learned during training. Our setup involves creating a model with a known training order by sequentially fine-tuning Llama-3.2-1B on six disjoint but otherwise similar datasets about named entities. We find that the average activations of test samples for the six training datasets encode the training order: when projected into a 2D subspace, these centroids are arranged exactly in the order of training and lie on a straight line. Further, we show that linear probes can accurately (~90%) distinguish "early" vs. "late" entities, generalizing to entities unseen during the probes' own training. The model can also be fine-tuned to explicitly report an unseen entity's training stage (~80% accuracy). Interestingly, this temporal signal does not seem attributable to simple differences in activation magnitudes, losses, or model confidence. Our paper demonstrates that models are capable of differentiating information by its acquisition time, and carries significant implications for how they might manage conflicting data and respond to knowledge modifications. 

---
# GEM-Bench: A Benchmark for Ad-Injected Response Generation within Generative Engine Marketing 

**Authors**: Silan Hu, Shiqi Zhang, Yimin Shi, Xiaokui Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2509.14221)  

**Abstract**: Generative Engine Marketing (GEM) is an emerging ecosystem for monetizing generative engines, such as LLM-based chatbots, by seamlessly integrating relevant advertisements into their responses. At the core of GEM lies the generation and evaluation of ad-injected responses. However, existing benchmarks are not specifically designed for this purpose, which limits future research. To address this gap, we propose GEM-Bench, the first comprehensive benchmark for ad-injected response generation in GEM. GEM-Bench includes three curated datasets covering both chatbot and search scenarios, a metric ontology that captures multiple dimensions of user satisfaction and engagement, and several baseline solutions implemented within an extensible multi-agent framework. Our preliminary results indicate that, while simple prompt-based methods achieve reasonable engagement such as click-through rate, they often reduce user satisfaction. In contrast, approaches that insert ads based on pre-generated ad-free responses help mitigate this issue but introduce additional overhead. These findings highlight the need for future research on designing more effective and efficient solutions for generating ad-injected responses in GEM. 

---
# Dense Video Understanding with Gated Residual Tokenization 

**Authors**: Haichao Zhang, Wenhao Chai, Shwai He, Ang Li, Yun Fu  

**Link**: [PDF](https://arxiv.org/pdf/2509.14199)  

**Abstract**: High temporal resolution is essential for capturing fine-grained details in video understanding. However, current video large language models (VLLMs) and benchmarks mostly rely on low-frame-rate sampling, such as uniform sampling or keyframe selection, discarding dense temporal information. This compromise avoids the high cost of tokenizing every frame, which otherwise leads to redundant computation and linear token growth as video length increases. While this trade-off works for slowly changing content, it fails for tasks like lecture comprehension, where information appears in nearly every frame and requires precise temporal alignment. To address this gap, we introduce Dense Video Understanding (DVU), which enables high-FPS video comprehension by reducing both tokenization time and token overhead. Existing benchmarks are also limited, as their QA pairs focus on coarse content changes. We therefore propose DIVE (Dense Information Video Evaluation), the first benchmark designed for dense temporal reasoning. To make DVU practical, we present Gated Residual Tokenization (GRT), a two-stage framework: (1) Motion-Compensated Inter-Gated Tokenization uses pixel-level motion estimation to skip static regions during tokenization, achieving sub-linear growth in token count and compute. (2) Semantic-Scene Intra-Tokenization Merging fuses tokens across static regions within a scene, further reducing redundancy while preserving dynamic semantics. Experiments on DIVE show that GRT outperforms larger VLLM baselines and scales positively with FPS. These results highlight the importance of dense temporal information and demonstrate that GRT enables efficient, scalable high-FPS video understanding. 

---
# When Avatars Have Personality: Effects on Engagement and Communication in Immersive Medical Training 

**Authors**: Julia S. Dollis, Iago A. Brito, Fernanda B. Färber, Pedro S. F. B. Ribeiro, Rafael T. Sousa, Arlindo R. Galvão Filho  

**Link**: [PDF](https://arxiv.org/pdf/2509.14132)  

**Abstract**: While virtual reality (VR) excels at simulating physical environments, its effectiveness for training complex interpersonal skills is limited by a lack of psychologically plausible virtual humans. This is a critical gap in high-stakes domains like medical education, where communication is a core competency. This paper introduces a framework that integrates large language models (LLMs) into immersive VR to create medically coherent virtual patients with distinct, consistent personalities, built on a modular architecture that decouples personality from clinical data. We evaluated our system in a mixed-method, within-subjects study with licensed physicians who engaged in simulated consultations. Results demonstrate that the approach is not only feasible but is also perceived by physicians as a highly rewarding and effective training enhancement. Furthermore, our analysis uncovers critical design principles, including a ``realism-verbosity paradox" where less communicative agents can seem more artificial, and the need for challenges to be perceived as authentic to be instructive. This work provides a validated framework and key insights for developing the next generation of socially intelligent VR training environments. 

---
# Reasoning Efficiently Through Adaptive Chain-of-Thought Compression: A Self-Optimizing Framework 

**Authors**: Kerui Huang, Shuhan Liu, Xing Hu, Tongtong Xu, Lingfeng Bao, Xin Xia  

**Link**: [PDF](https://arxiv.org/pdf/2509.14093)  

**Abstract**: Chain-of-Thought (CoT) reasoning enhances Large Language Models (LLMs) by prompting intermediate steps, improving accuracy and robustness in arithmetic, logic, and commonsense tasks. However, this benefit comes with high computational costs: longer outputs increase latency, memory usage, and KV-cache demands. These issues are especially critical in software engineering tasks where concise and deterministic outputs are required. To investigate these trade-offs, we conduct an empirical study based on code generation benchmarks. The results reveal that longer CoT does not always help. Excessive reasoning often causes truncation, accuracy drops, and latency up to five times higher, with failed outputs consistently longer than successful ones. These findings challenge the assumption that longer reasoning is inherently better and highlight the need for adaptive CoT control. Motivated by this, we propose SEER (Self-Enhancing Efficient Reasoning), an adaptive framework that compresses CoT while preserving accuracy. SEER combines Best-of-N sampling with task-aware adaptive filtering, dynamically adjusting thresholds based on pre-inference outputs to reduce verbosity and computational overhead. We then evaluate SEER on three software engineering tasks and one math task. On average, SEER shortens CoT by 42.1%, improves accuracy by reducing truncation, and eliminates most infinite loops. These results demonstrate SEER as a practical method to make CoT-enhanced LLMs more efficient and robust, even under resource constraints. 

---
# A TRRIP Down Memory Lane: Temperature-Based Re-Reference Interval Prediction For Instruction Caching 

**Authors**: Henry Kao, Nikhil Sreekumar, Prabhdeep Singh Soni, Ali Sedaghati, Fang Su, Bryan Chan, Maziar Goudarzi, Reza Azimi  

**Link**: [PDF](https://arxiv.org/pdf/2509.14041)  

**Abstract**: Modern mobile CPU software pose challenges for conventional instruction cache replacement policies due to their complex runtime behavior causing high reuse distance between executions of the same instruction. Mobile code commonly suffers from large amounts of stalls in the CPU frontend and thus starvation of the rest of the CPU resources. Complexity of these applications and their code footprint are projected to grow at a rate faster than available on-chip memory due to power and area constraints, making conventional hardware-centric methods for managing instruction caches to be inadequate. We present a novel software-hardware co-design approach called TRRIP (Temperature-based Re-Reference Interval Prediction) that enables the compiler to analyze, classify, and transform code based on "temperature" (hot/cold), and to provide the hardware with a summary of code temperature information through a well-defined OS interface based on using code page attributes. TRRIP's lightweight hardware extension employs code temperature attributes to optimize the instruction cache replacement policy resulting in the eviction rate reduction of hot code. TRRIP is designed to be practical and adoptable in real mobile systems that have strict feature requirements on both the software and hardware components. TRRIP can reduce the L2 MPKI for instructions by 26.5% resulting in geomean speedup of 3.9%, on top of RRIP cache replacement running mobile code already optimized using PGO. 

---
# Exploring Major Transitions in the Evolution of Biological Cognition With Artificial Neural Networks 

**Authors**: Konstantinos Voudouris, Andrew Barron, Marta Halina, Colin Klein, Matishalin Patel  

**Link**: [PDF](https://arxiv.org/pdf/2509.13968)  

**Abstract**: Transitional accounts of evolution emphasise a few changes that shape what is evolvable, with dramatic consequences for derived lineages. More recently it has been proposed that cognition might also have evolved via a series of major transitions that manipulate the structure of biological neural networks, fundamentally changing the flow of information. We used idealised models of information flow, artificial neural networks (ANNs), to evaluate whether changes in information flow in a network can yield a transitional change in cognitive performance. We compared networks with feed-forward, recurrent and laminated topologies, and tested their performance learning artificial grammars that differed in complexity, controlling for network size and resources. We documented a qualitative expansion in the types of input that recurrent networks can process compared to feed-forward networks, and a related qualitative increase in performance for learning the most complex grammars. We also noted how the difficulty in training recurrent networks poses a form of transition barrier and contingent irreversibility -- other key features of evolutionary transitions. Not all changes in network topology confer a performance advantage in this task set. Laminated networks did not outperform non-laminated networks in grammar learning. Overall, our findings show how some changes in information flow can yield transitions in cognitive performance. 

---
# Enhancing Time Awareness in Generative Recommendation 

**Authors**: Sunkyung Lee, Seongmin Park, Jonghyo Kim, Mincheol Yoon, Jongwuk Lee  

**Link**: [PDF](https://arxiv.org/pdf/2509.13957)  

**Abstract**: Generative recommendation has emerged as a promising paradigm that formulates the recommendations into a text-to-text generation task, harnessing the vast knowledge of large language models. However, existing studies focus on considering the sequential order of items and neglect to handle the temporal dynamics across items, which can imply evolving user preferences. To address this limitation, we propose a novel model, Generative Recommender Using Time awareness (GRUT), effectively capturing hidden user preferences via various temporal signals. We first introduce Time-aware Prompting, consisting of two key contexts. The user-level temporal context models personalized temporal patterns across timestamps and time intervals, while the item-level transition context provides transition patterns across users. We also devise Trend-aware Inference, a training-free method that enhances rankings by incorporating trend information about items with generation likelihood. Extensive experiments demonstrate that GRUT outperforms state-of-the-art models, with gains of up to 15.4% and 14.3% in Recall@5 and NDCG@5 across four benchmark datasets. The source code is available at this https URL. 

---
# An Empirical Study on Failures in Automated Issue Solving 

**Authors**: Simiao Liu, Fang Liu, Liehao Li, Xin Tan, Yinghao Zhu, Xiaoli Lian, Li Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.13941)  

**Abstract**: Automated issue solving seeks to autonomously identify and repair defective code snippets across an entire codebase. SWE-Bench has emerged as the most widely adopted benchmark for evaluating progress in this area. While LLM-based agentic tools show great promise, they still fail on a substantial portion of tasks. Moreover, current evaluations primarily report aggregate issue-solving rates, which obscure the underlying causes of success and failure, making it challenging to diagnose model weaknesses or guide targeted improvements. To bridge this gap, we first analyze the performance and efficiency of three SOTA tools, spanning both pipeline-based and agentic architectures, in automated issue solving tasks of SWE-Bench-Verified under varying task characteristics. Furthermore, to move from high-level performance metrics to underlying cause analysis, we conducted a systematic manual analysis of 150 failed instances. From this analysis, we developed a comprehensive taxonomy of failure modes comprising 3 primary phases, 9 main categories, and 25 fine-grained subcategories. Then we systematically analyze the distribution of the identified failure modes, the results reveal distinct failure fingerprints between the two architectural paradigms, with the majority of agentic failures stemming from flawed reasoning and cognitive deadlocks. Motivated by these insights, we propose a collaborative Expert-Executor framework. It introduces a supervisory Expert agent tasked with providing strategic oversight and course-correction for a primary Executor agent. This architecture is designed to correct flawed reasoning and break the cognitive deadlocks that frequently lead to failure. Experiments show that our framework solves 22.2% of previously intractable issues for a leading single agent. These findings pave the way for building more robust agents through diagnostic evaluation and collaborative design. 

---
# Noise Supervised Contrastive Learning and Feature-Perturbed for Anomalous Sound Detection 

**Authors**: Shun Huang, Zhihua Fang, Liang He  

**Link**: [PDF](https://arxiv.org/pdf/2509.13853)  

**Abstract**: Unsupervised anomalous sound detection aims to detect unknown anomalous sounds by training a model using only normal audio data. Despite advancements in self-supervised methods, the issue of frequent false alarms when handling samples of the same type from different machines remains unresolved. This paper introduces a novel training technique called one-stage supervised contrastive learning (OS-SCL), which significantly addresses this problem by perturbing features in the embedding space and employing a one-stage noisy supervised contrastive learning approach. On the DCASE 2020 Challenge Task 2, it achieved 94.64\% AUC, 88.42\% pAUC, and 89.24\% mAUC using only Log-Mel features. Additionally, a time-frequency feature named TFgram is proposed, which is extracted from raw audio. This feature effectively captures critical information for anomalous sound detection, ultimately achieving 95.71\% AUC, 90.23\% pAUC, and 91.23\% mAUC. The source code is available at: \underline{this http URL}. 

---
# Diving into Mitigating Hallucinations from a Vision Perspective for Large Vision-Language Models 

**Authors**: Weihang Wang, Xinhao Li, Ziyue Wang, Yan Pang, Jielei Zhang, Peiyi Li, Qiang Zhang, Longwen Gao  

**Link**: [PDF](https://arxiv.org/pdf/2509.13836)  

**Abstract**: Object hallucination in Large Vision-Language Models (LVLMs) significantly impedes their real-world applicability. As the primary component for accurately interpreting visual information, the choice of visual encoder is pivotal. We hypothesize that the diverse training paradigms employed by different visual encoders instill them with distinct inductive biases, which leads to their diverse hallucination performances. Existing benchmarks typically focus on coarse-grained hallucination detection and fail to capture the diverse hallucinations elaborated in our hypothesis. To systematically analyze these effects, we introduce VHBench-10, a comprehensive benchmark with approximately 10,000 samples for evaluating LVLMs across ten fine-grained hallucination categories. Our evaluations confirm encoders exhibit unique hallucination characteristics. Building on these insights and the suboptimality of simple feature fusion, we propose VisionWeaver, a novel Context-Aware Routing Network. It employs global visual features to generate routing signals, dynamically aggregating visual features from multiple specialized experts. Comprehensive experiments confirm the effectiveness of VisionWeaver in significantly reducing hallucinations and improving overall model performance. 

---
# THOR: Tool-Integrated Hierarchical Optimization via RL for Mathematical Reasoning 

**Authors**: Qikai Chang, Zhenrong Zhang, Pengfei Hu, Jiefeng Ma, Yicheng Pan, Jianshu Zhang, Jun Du, Quan Liu, Jianqing Gao  

**Link**: [PDF](https://arxiv.org/pdf/2509.13761)  

**Abstract**: Large Language Models (LLMs) have made remarkable progress in mathematical reasoning, but still continue to struggle with high-precision tasks like numerical computation and formal symbolic manipulation. Integrating external tools has emerged as a promising approach to bridge this gap. Despite recent advances, existing methods struggle with three key challenges: constructing tool-integrated reasoning data, performing fine-grained optimization, and enhancing inference. To overcome these limitations, we propose THOR (Tool-Integrated Hierarchical Optimization via RL). First, we introduce TIRGen, a multi-agent actor-critic-based pipeline for constructing high-quality datasets of tool-integrated reasoning paths, aligning with the policy and generalizing well across diverse models. Second, to perform fine-grained hierarchical optimization, we introduce an RL strategy that jointly optimizes for both trajectory-level problem solving and step-level code generation. This is motivated by our key insight that the success of an intermediate tool call is a strong predictor of the final answer's correctness. Finally, THOR incorporates a self-correction mechanism that leverages immediate tool feedback to dynamically revise erroneous reasoning paths during inference. Our approach demonstrates strong generalization across diverse models, performing effectively in both reasoning and non-reasoning models. It further achieves state-of-the-art performance for models of a similar scale on multiple mathematical benchmarks, while also delivering consistent improvements on code benchmarks. Our code will be publicly available at this https URL. 

---
# Privacy-Aware In-Context Learning for Large Language Models 

**Authors**: Bishnu Bhusal, Manoj Acharya, Ramneet Kaur, Colin Samplawski, Anirban Roy, Adam D. Cobb, Rohit Chadha, Susmit Jha  

**Link**: [PDF](https://arxiv.org/pdf/2509.13625)  

**Abstract**: Large language models (LLMs) have significantly transformed natural language understanding and generation, but they raise privacy concerns due to potential exposure of sensitive information. Studies have highlighted the risk of information leakage, where adversaries can extract sensitive information embedded in the prompts. In this work, we introduce a novel private prediction framework for generating high-quality synthetic text with strong privacy guarantees. Our approach leverages the Differential Privacy (DP) framework to ensure worst-case theoretical bounds on information leakage without requiring any fine-tuning of the underlying this http URL proposed method performs inference on private records and aggregates the resulting per-token output distributions. This enables the generation of longer and coherent synthetic text while maintaining privacy guarantees. Additionally, we propose a simple blending operation that combines private and public inference to further enhance utility. Empirical evaluations demonstrate that our approach outperforms previous state-of-the-art methods on in-context-learning (ICL) tasks, making it a promising direction for privacy-preserving text generation while maintaining high utility. 

---
# See, Think, Act: Teaching Multimodal Agents to Effectively Interact with GUI by Identifying Toggles 

**Authors**: Zongru Wu, Rui Mao, Zhiyuan Tian, Pengzhou Cheng, Tianjie Ju, Zheng Wu, Lingzhong Dong, Haiyue Sheng, Zhuosheng Zhang, Gongshen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.13615)  

**Abstract**: The advent of multimodal agents facilitates effective interaction within graphical user interface (GUI), especially in ubiquitous GUI control. However, their inability to reliably execute toggle control instructions remains a key bottleneck. To investigate this, we construct a state control benchmark with binary toggle instructions from public datasets. Evaluations of existing agents demonstrate their unreliability, particularly when the current toggle state already matches the desired state. To address the challenge, we propose State-aware Reasoning (StaR), a training method that teaches agents to perceive the current toggle state, analyze the desired state from the instruction, and act accordingly. Experiments on three multimodal agents demonstrate that StaR can improve toggle instruction execution accuracy by over 30\%. Further evaluations on three public benchmarks show that StaR also enhances general task performance. Finally, evaluations on a dynamic environment highlight the potential of StaR for real-world applications. Code, benchmark, and StaR-enhanced agents are available at this https URL. 

---
# Annotating Satellite Images of Forests with Keywords from a Specialized Corpus in the Context of Change Detection 

**Authors**: Nathalie Neptune, Josiane Mothe  

**Link**: [PDF](https://arxiv.org/pdf/2509.13586)  

**Abstract**: The Amazon rain forest is a vital ecosystem that plays a crucial role in regulating the Earth's climate and providing habitat for countless species. Deforestation in the Amazon is a major concern as it has a significant impact on global carbon emissions and biodiversity. In this paper, we present a method for detecting deforestation in the Amazon using image pairs from Earth observation satellites. Our method leverages deep learning techniques to compare the images of the same area at different dates and identify changes in the forest cover. We also propose a visual semantic model that automatically annotates the detected changes with relevant keywords. The candidate annotation for images are extracted from scientific documents related to the Amazon region. We evaluate our approach on a dataset of Amazon image pairs and demonstrate its effectiveness in detecting deforestation and generating relevant annotations. Our method provides a useful tool for monitoring and studying the impact of deforestation in the Amazon. While we focus on environment applications of our work by using images of deforestation in the Amazon rain forest to demonstrate the effectiveness of our proposed approach, it is generic enough to be applied to other domains. 

---
# SteeringControl: Holistic Evaluation of Alignment Steering in LLMs 

**Authors**: Vincent Siu, Nicholas Crispino, David Park, Nathan W. Henry, Zhun Wang, Yang Liu, Dawn Song, Chenguang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.13450)  

**Abstract**: We introduce SteeringControl, a benchmark for evaluating representation steering methods across core alignment objectives--bias, harmful generation, and hallucination--and their effects on secondary behaviors such as sycophancy and commonsense morality. While prior alignment work often highlights truthfulness or reasoning ability to demonstrate the side effects of representation steering, we find there are many unexplored tradeoffs not yet understood in a systematic way. We collect a dataset of safety-relevant primary and secondary behaviors to evaluate steering effectiveness and behavioral entanglement centered around five popular steering methods. To enable this, we craft a modular steering framework based on unique components that serve as the building blocks of many existing methods. Our results on Qwen-2.5-7B and Llama-3.1-8B find that strong steering performance is dependent on the specific combination of steering method, model, and targeted behavior, and that severe concept entanglement can result from poor combinations of these three as well. We release our code here: this https URL. 

---
# TICL: Text-Embedding KNN For Speech In-Context Learning Unlocks Speech Recognition Abilities of Large Multimodal Models 

**Authors**: Haolong Zheng, Yekaterina Yegorova, Mark Hasegawa-Johnson  

**Link**: [PDF](https://arxiv.org/pdf/2509.13395)  

**Abstract**: Speech foundation models have recently demonstrated the ability to perform Speech In-Context Learning (SICL). Selecting effective in-context examples is crucial for SICL performance, yet selection methodologies remain underexplored. In this work, we propose Text-Embedding KNN for SICL (TICL), a simple pipeline that uses semantic context to enhance off-the-shelf large multimodal models' speech recognition ability without fine-tuning. Across challenging automatic speech recognition tasks, including accented English, multilingual speech, and children's speech, our method enables models to surpass zero-shot performance with up to 84.7% relative WER reduction. We conduct ablation studies to show the robustness and efficiency of our method. 

---
# CogniAlign: Survivability-Grounded Multi-Agent Moral Reasoning for Safe and Transparent AI 

**Authors**: Hasin Jawad Ali, Ilhamul Azam, Ajwad Abrar, Md. Kamrul Hasan, Hasan Mahmud  

**Link**: [PDF](https://arxiv.org/pdf/2509.13356)  

**Abstract**: The challenge of aligning artificial intelligence (AI) with human values persists due to the abstract and often conflicting nature of moral principles and the opacity of existing approaches. This paper introduces CogniAlign, a multi-agent deliberation framework based on naturalistic moral realism, that grounds moral reasoning in survivability, defined across individual and collective dimensions, and operationalizes it through structured deliberations among discipline-specific scientist agents. Each agent, representing neuroscience, psychology, sociology, and evolutionary biology, provides arguments and rebuttals that are synthesized by an arbiter into transparent and empirically anchored judgments. We evaluate CogniAlign on classic and novel moral questions and compare its outputs against GPT-4o using a five-part ethical audit framework. Results show that CogniAlign consistently outperforms the baseline across more than sixty moral questions, with average performance gains of 16.2 points in analytic quality, 14.3 points in breadth, and 28.4 points in depth of explanation. In the Heinz dilemma, for example, CogniAlign achieved an overall score of 89.2 compared to GPT-4o's 69.2, demonstrating a decisive advantage in handling moral reasoning. By reducing black-box reasoning and avoiding deceptive alignment, CogniAlign highlights the potential of interdisciplinary deliberation as a scalable pathway for safe and transparent AI alignment. 

---
# Teaching LLMs to Plan: Logical Chain-of-Thought Instruction Tuning for Symbolic Planning 

**Authors**: Pulkit Verma, Ngoc La, Anthony Favier, Swaroop Mishra, Julie A. Shah  

**Link**: [PDF](https://arxiv.org/pdf/2509.13351)  

**Abstract**: Large language models (LLMs) have demonstrated impressive capabilities across diverse tasks, yet their ability to perform structured symbolic planning remains limited, particularly in domains requiring formal representations like the Planning Domain Definition Language (PDDL). In this paper, we present a novel instruction tuning framework, PDDL-Instruct, designed to enhance LLMs' symbolic planning capabilities through logical chain-of-thought reasoning. Our approach focuses on teaching models to rigorously reason about action applicability, state transitions, and plan validity using explicit logical inference steps. By developing instruction prompts that guide models through the precise logical reasoning required to determine when actions can be applied in a given state, we enable LLMs to self-correct their planning processes through structured reflection. The framework systematically builds verification skills by decomposing the planning process into explicit reasoning chains about precondition satisfaction, effect application, and invariant preservation. Experimental results on multiple planning domains show that our chain-of-thought reasoning based instruction-tuned models are significantly better at planning, achieving planning accuracy of up to 94% on standard benchmarks, representing a 66% absolute improvement over baseline models. This work bridges the gap between the general reasoning capabilities of LLMs and the logical precision required for automated planning, offering a promising direction for developing better AI planning systems. 

---
# Accuracy Paradox in Large Language Models: Regulating Hallucination Risks in Generative AI 

**Authors**: Zihao Li, Weiwei Yi, Jiahong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.13345)  

**Abstract**: As Large Language Models (LLMs) permeate everyday decision-making, their epistemic and societal risks demand urgent scrutiny. Hallucinations, the generation of fabricated, misleading, oversimplified or untrustworthy outputs, has emerged as imperative challenges. While regulatory, academic, and technical discourse position accuracy as the principal benchmark for mitigating such harms, this article contends that overreliance on accuracy misdiagnoses the problem and has counterproductive effect: the accuracy paradox. Drawing on interdisciplinary literatures, this article develops a taxonomy of hallucination types and shows the paradox along three intertwining dimensions: outputs, individuals and society. First, accuracy functions as a superficial proxy for reliability, incentivising the optimisation of rhetorical fluency and surface-level correctness over epistemic trustworthiness. This encourages passive user trust in outputs that appear accurate but epistemically untenable. Second, accuracy as a singular metric fails to detect harms that are not factually false but are nonetheless misleading, value-laden, or socially distorting, including consensus illusions, sycophantic alignment, and subtle manipulation. Third, regulatory overemphasis on accuracy obscures the wider societal consequences of hallucination, including social sorting, privacy violations, equity harms, epistemic convergence that marginalises dissent, reduces pluralism, and causes social deskilling. By examining the EU AI Act, GDPR, and DSA, the article argues that current regulations are not yet structurally equipped to address these epistemic, relational, and systemic harms and exacerbated by the overreliance on accuracy. By exposing such conceptual and practical challenges, this article calls for a fundamental shift towards pluralistic, context-aware, and manipulation-resilient approaches to AI trustworthy governance. 

---
# Explicit Reasoning Makes Better Judges: A Systematic Study on Accuracy, Efficiency, and Robustness 

**Authors**: Pratik Jayarao, Himanshu Gupta, Neeraj Varshney, Chaitanya Dwivedi  

**Link**: [PDF](https://arxiv.org/pdf/2509.13332)  

**Abstract**: As Large Language Models (LLMs) are increasingly adopted as automated judges in benchmarking and reward modeling, ensuring their reliability, efficiency, and robustness has become critical. In this work, we present a systematic comparison of "thinking" and "non-thinking" LLMs in the LLM-as-a-judge paradigm using open-source Qwen 3 models of relatively small sizes (0.6B, 1.7B, and 4B parameters). We evaluate both accuracy and computational efficiency (FLOPs) on RewardBench tasks, and further examine augmentation strategies for non-thinking models, including in-context learning, rubric-guided judging, reference-based evaluation, and n-best aggregation. Our results show that despite these enhancements, non-thinking models generally fall short of their thinking counterparts. Our results show that thinking models achieve approximately 10% points higher accuracy with little overhead (under 2x), in contrast to augmentation strategies like few-shot learning, which deliver modest gains at a higher cost (>8x). Bias and robustness analyses further demonstrate that thinking models maintain significantly greater consistency under a variety of bias conditions such as positional, bandwagon, identity, diversity, and random biases (6% higher on average). We further extend our experiments to the multilingual setting and our results confirm that explicit reasoning extends its benefits beyond English. Overall, our work results in several important findings that provide systematic evidence that explicit reasoning offers clear advantages in the LLM-as-a-judge paradigm not only in accuracy and efficiency but also in robustness. 

---
# An AI-Powered Framework for Analyzing Collective Idea Evolution in Deliberative Assemblies 

**Authors**: Elinor Poole-Dayan, Deb Roy, Jad Kabbara  

**Link**: [PDF](https://arxiv.org/pdf/2509.12577)  

**Abstract**: In an era of increasing societal fragmentation, political polarization, and erosion of public trust in institutions, representative deliberative assemblies are emerging as a promising democratic forum for developing effective policy outcomes on complex global issues. Despite theoretical attention, there remains limited empirical work that systematically traces how specific ideas evolve, are prioritized, or are discarded during deliberation to form policy recommendations. Addressing these gaps, this work poses two central questions: (1) How might we trace the evolution and distillation of ideas into concrete recommendations within deliberative assemblies? (2) How does the deliberative process shape delegate perspectives and influence voting dynamics over the course of the assembly? To address these questions, we develop LLM-based methodologies for empirically analyzing transcripts from a tech-enhanced in-person deliberative assembly. The framework identifies and visualizes the space of expressed suggestions. We also empirically reconstruct each delegate's evolving perspective throughout the assembly. Our methods contribute novel empirical insights into deliberative processes and demonstrate how LLMs can surface high-resolution dynamics otherwise invisible in traditional assembly outputs. 

---
