# The Universal Landscape of Human Reasoning 

**Authors**: Qiguang Chen, Jinhao Liu, Libo Qin, Yimeng Zhang, Yihao Liang, Shangxu Ren, Chengyu Luan, Dengyun Peng, Hanjing Li, Jiannan Guan, Zheng Yan, Jiaqi Wang, Mengkang Hu, Yantao Du, Zhi Chen, Xie Chen, Wanxiang Che  

**Link**: [PDF](https://arxiv.org/pdf/2510.21623)  

**Abstract**: Understanding how information is dynamically accumulated and transformed in human reasoning has long challenged cognitive psychology, philosophy, and artificial intelligence. Existing accounts, from classical logic to probabilistic models, illuminate aspects of output or individual modelling, but do not offer a unified, quantitative description of general human reasoning dynamics. To solve this, we introduce Information Flow Tracking (IF-Track), that uses large language models (LLMs) as probabilistic encoder to quantify information entropy and gain at each reasoning step. Through fine-grained analyses across diverse tasks, our method is the first successfully models the universal landscape of human reasoning behaviors within a single metric space. We show that IF-Track captures essential reasoning features, identifies systematic error patterns, and characterizes individual differences. Applied to discussion of advanced psychological theory, we first reconcile single- versus dual-process theories in IF-Track and discover the alignment of artificial and human cognition and how LLMs reshaping human reasoning process. This approach establishes a quantitative bridge between theory and measurement, offering mechanistic insights into the architecture of reasoning. 

---
# RETuning: Upgrading Inference-Time Scaling for Stock Movement Prediction with Large Language Models 

**Authors**: Xueyuan Lin, Cehao Yang, Ye Ma, Ming Li, Rongjunchen Zhang, Yang Ni, Xiaojun Wu, Chengjin Xu, Jian Guo, Hui Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2510.21604)  

**Abstract**: Recently, large language models (LLMs) have demonstrated outstanding reasoning capabilities on mathematical and coding tasks. However, their application to financial tasks-especially the most fundamental task of stock movement prediction-remains underexplored. We study a three-class classification problem (up, hold, down) and, by analyzing existing reasoning responses, observe that: (1) LLMs follow analysts' opinions rather than exhibit a systematic, independent analytical logic (CoTs). (2) LLMs list summaries from different sources without weighing adversarial evidence, yet such counterevidence is crucial for reliable prediction. It shows that the model does not make good use of its reasoning ability to complete the task. To address this, we propose Reflective Evidence Tuning (RETuning), a cold-start method prior to reinforcement learning, to enhance prediction ability. While generating CoT, RETuning encourages dynamically constructing an analytical framework from diverse information sources, organizing and scoring evidence for price up or down based on that framework-rather than on contextual viewpoints-and finally reflecting to derive the prediction. This approach maximally aligns the model with its learned analytical framework, ensuring independent logical reasoning and reducing undue influence from context. We also build a large-scale dataset spanning all of 2024 for 5,123 A-share stocks, with long contexts (32K tokens) and over 200K samples. In addition to price and news, it incorporates analysts' opinions, quantitative reports, fundamental data, macroeconomic indicators, and similar stocks. Experiments show that RETuning successfully unlocks the model's reasoning ability in the financial domain. Inference-time scaling still works even after 6 months or on out-of-distribution stocks, since the models gain valuable insights about stock movement prediction. 

---
# Automated Quality Control for Language Documentation: Detecting Phonotactic Inconsistencies in a Kokborok Wordlist 

**Authors**: Kellen Parker van Dam, Abishek Stephen  

**Link**: [PDF](https://arxiv.org/pdf/2510.21584)  

**Abstract**: Lexical data collection in language documentation often contains transcription errors and undocumented borrowings that can mislead linguistic analysis. We present unsupervised anomaly detection methods to identify phonotactic inconsistencies in wordlists, applying them to a multilingual dataset of Kokborok varieties with Bangla. Using character-level and syllable-level phonotactic features, our algorithms identify potential transcription errors and borrowings. While precision and recall remain modest due to the subtle nature of these anomalies, syllable-aware features significantly outperform character-level baselines. The high-recall approach provides fieldworkers with a systematic method to flag entries requiring verification, supporting data quality improvement in low-resourced language documentation. 

---
# From Polyester Girlfriends to Blind Mice: Creating the First Pragmatics Understanding Benchmarks for Slovene 

**Authors**: Mojca Brglez, Špela Vintar  

**Link**: [PDF](https://arxiv.org/pdf/2510.21575)  

**Abstract**: Large language models are demonstrating increasing capabilities, excelling at benchmarks once considered very difficult. As their capabilities grow, there is a need for more challenging evaluations that go beyond surface-level linguistic competence. Namely, language competence involves not only syntax and semantics but also pragmatics, i.e., understanding situational meaning as shaped by context as well as linguistic and cultural norms. To contribute to this line of research, we introduce SloPragEval and SloPragMega, the first pragmatics understanding benchmarks for Slovene that contain altogether 405 multiple-choice questions. We discuss the difficulties of translation, describe the campaign to establish a human baseline, and report pilot evaluations with LLMs. Our results indicate that current models have greatly improved in understanding nuanced language but may still fail to infer implied speaker meaning in non-literal utterances, especially those that are culture-specific. We also observe a significant gap between proprietary and open-source models. Finally, we argue that benchmarks targeting nuanced language understanding and knowledge of the target culture must be designed with care, preferably constructed from native data, and validated with human responses. 

---
# Are the LLMs Capable of Maintaining at Least the Language Genus? 

**Authors**: Sandra Mitrović, David Kletz, Ljiljana Dolamic, Fabio Rinaldi  

**Link**: [PDF](https://arxiv.org/pdf/2510.21561)  

**Abstract**: Large Language Models (LLMs) display notable variation in multilingual behavior, yet the role of genealogical language structure in shaping this variation remains underexplored. In this paper, we investigate whether LLMs exhibit sensitivity to linguistic genera by extending prior analyses on the MultiQ dataset. We first check if models prefer to switch to genealogically related languages when prompt language fidelity is not maintained. Next, we investigate whether knowledge consistency is better preserved within than across genera. We show that genus-level effects are present but strongly conditioned by training resource availability. We further observe distinct multilingual strategies across LLMs families. Our findings suggest that LLMs encode aspects of genus-level structure, but training data imbalances remain the primary factor shaping their multilingual performance. 

---
# Document Understanding, Measurement, and Manipulation Using Category Theory 

**Authors**: Jared Claypoole, Yunye Gong, Noson S. Yanofsky, Ajay Divakaran  

**Link**: [PDF](https://arxiv.org/pdf/2510.21553)  

**Abstract**: We apply category theory to extract multimodal document structure which leads us to develop information theoretic measures, content summarization and extension, and self-supervised improvement of large pretrained models. We first develop a mathematical representation of a document as a category of question-answer pairs. Second, we develop an orthogonalization procedure to divide the information contained in one or more documents into non-overlapping pieces. The structures extracted in the first and second steps lead us to develop methods to measure and enumerate the information contained in a document. We also build on those steps to develop new summarization techniques, as well as to develop a solution to a new problem viz. exegesis resulting in an extension of the original document. Our question-answer pair methodology enables a novel rate distortion analysis of summarization techniques. We implement our techniques using large pretrained models, and we propose a multimodal extension of our overall mathematical framework. Finally, we develop a novel self-supervised method using RLVR to improve large pretrained models using consistency constraints such as composability and closure under certain operations that stem naturally from our category theoretic framework. 

---
# InterpDetect: Interpretable Signals for Detecting Hallucinations in Retrieval-Augmented Generation 

**Authors**: Likun Tan, Kuan-Wei Huang, Joy Shi, Kevin Wu  

**Link**: [PDF](https://arxiv.org/pdf/2510.21538)  

**Abstract**: Retrieval-Augmented Generation (RAG) integrates external knowledge to mitigate hallucinations, yet models often generate outputs inconsistent with retrieved content. Accurate hallucination detection requires disentangling the contributions of external context and parametric knowledge, which prior methods typically conflate. We investigate the mechanisms underlying RAG hallucinations and find they arise when later-layer FFN modules disproportionately inject parametric knowledge into the residual stream. To address this, we explore a mechanistic detection approach based on external context scores and parametric knowledge scores. Using Qwen3-0.6b, we compute these scores across layers and attention heads and train regression-based classifiers to predict hallucinations. Our method is evaluated against state-of-the-art LLMs (GPT-5, GPT-4.1) and detection baselines (RAGAS, TruLens, RefChecker). Furthermore, classifiers trained on Qwen3-0.6b signals generalize to GPT-4.1-mini responses, demonstrating the potential of proxy-model evaluation. Our results highlight mechanistic signals as efficient, generalizable predictors for hallucination detection in RAG systems. 

---
# Brain-tuning Improves Generalizability and Efficiency of Brain Alignment in Speech Models 

**Authors**: Omer Moussa, Mariya Toneva  

**Link**: [PDF](https://arxiv.org/pdf/2510.21520)  

**Abstract**: Pretrained language models are remarkably effective in aligning with human brain responses elicited by natural language stimuli, positioning them as promising model organisms for studying language processing in the brain. However, existing approaches for both estimating and improving this brain alignment are participant-dependent and highly affected by the amount of data available per participant, hindering both generalization to new participants and population-level analyses. In this work, we address these limitations by introducing a scalable, generalizable brain-tuning method, in which we fine-tune pretrained speech language models to jointly predict fMRI responses from multiple participants. We demonstrate that the resulting brain-tuned models exhibit strong individual brain alignment while generalizing across participants. Specifically, our method leads to 1) a 5-fold decrease in the amount of fMRI data needed to predict brain data from new participants, 2) up to a 50% increase in the overall brain alignment, and 3) strong generalization to new unseen datasets. Furthermore, this multi-participant brain-tuning additionally improves downstream performance on semantic tasks, suggesting that training using brain data from multiple participants leads to more generalizable semantic representations. Taken together, these findings demonstrate a bidirectional benefit between neuroscience and AI, helping bridge the gap between the two fields. We make our code and models publicly available at this https URL. 

---
# MRO: Enhancing Reasoning in Diffusion Language Models via Multi-Reward Optimization 

**Authors**: Chenglong Wang, Yang Gan, Hang Zhou, Chi Hu, Yongyu Mu, Kai Song, Murun Yang, Bei Li, Chunliang Zhang, Tongran Liu, Jingbo Zhu, Zhengtao Yu, Tong Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2510.21473)  

**Abstract**: Recent advances in diffusion language models (DLMs) have presented a promising alternative to traditional autoregressive large language models (LLMs). However, DLMs still lag behind LLMs in reasoning performance, especially as the number of denoising steps decreases. Our analysis reveals that this shortcoming arises primarily from the independent generation of masked tokens across denoising steps, which fails to capture the token correlation. In this paper, we define two types of token correlation: intra-sequence correlation and inter-sequence correlation, and demonstrate that enhancing these correlations improves reasoning performance. To this end, we propose a Multi-Reward Optimization (MRO) approach, which encourages DLMs to consider the token correlation during the denoising process. More specifically, our MRO approach leverages test-time scaling, reject sampling, and reinforcement learning to directly optimize the token correlation with multiple elaborate rewards. Additionally, we introduce group step and importance sampling strategies to mitigate reward variance and enhance sampling efficiency. Through extensive experiments, we demonstrate that MRO not only improves reasoning performance but also achieves significant sampling speedups while maintaining high performance on reasoning benchmarks. 

---
# REMONI: An Autonomous System Integrating Wearables and Multimodal Large Language Models for Enhanced Remote Health Monitoring 

**Authors**: Thanh Cong Ho, Farah Kharrat, Abderrazek Abid, Fakhri Karray  

**Link**: [PDF](https://arxiv.org/pdf/2510.21445)  

**Abstract**: With the widespread adoption of wearable devices in our daily lives, the demand and appeal for remote patient monitoring have significantly increased. Most research in this field has concentrated on collecting sensor data, visualizing it, and analyzing it to detect anomalies in specific diseases such as diabetes, heart disease and depression. However, this domain has a notable gap in the aspect of human-machine interaction. This paper proposes REMONI, an autonomous REmote health MONItoring system that integrates multimodal large language models (MLLMs), the Internet of Things (IoT), and wearable devices. The system automatically and continuously collects vital signs, accelerometer data from a special wearable (such as a smartwatch), and visual data in patient video clips collected from cameras. This data is processed by an anomaly detection module, which includes a fall detection model and algorithms to identify and alert caregivers of the patient's emergency conditions. A distinctive feature of our proposed system is the natural language processing component, developed with MLLMs capable of detecting and recognizing a patient's activity and emotion while responding to healthcare worker's inquiries. Additionally, prompt engineering is employed to integrate all patient information seamlessly. As a result, doctors and nurses can access real-time vital signs and the patient's current state and mood by interacting with an intelligent agent through a user-friendly web application. Our experiments demonstrate that our system is implementable and scalable for real-life scenarios, potentially reducing the workload of medical professionals and healthcare costs. A full-fledged prototype illustrating the functionalities of the system has been developed and being tested to demonstrate the robustness of its various capabilities. 

---
# Redefining Retrieval Evaluation in the Era of LLMs 

**Authors**: Giovanni Trappolini, Florin Cuconasu, Simone Filice, Yoelle Maarek, Fabrizio Silvestri  

**Link**: [PDF](https://arxiv.org/pdf/2510.21440)  

**Abstract**: Traditional Information Retrieval (IR) metrics, such as nDCG, MAP, and MRR, assume that human users sequentially examine documents with diminishing attention to lower ranks. This assumption breaks down in Retrieval Augmented Generation (RAG) systems, where search results are consumed by Large Language Models (LLMs), which, unlike humans, process all retrieved documents as a whole rather than sequentially. Additionally, traditional IR metrics do not account for related but irrelevant documents that actively degrade generation quality, rather than merely being ignored. Due to these two major misalignments, namely human vs. machine position discount and human relevance vs. machine utility, classical IR metrics do not accurately predict RAG performance. We introduce a utility-based annotation schema that quantifies both the positive contribution of relevant passages and the negative impact of distracting ones. Building on this foundation, we propose UDCG (Utility and Distraction-aware Cumulative Gain), a metric using an LLM-oriented positional discount to directly optimize the correlation with the end-to-end answer accuracy. Experiments on five datasets and six LLMs demonstrate that UDCG improves correlation by up to 36% compared to traditional metrics. Our work provides a critical step toward aligning IR evaluation with LLM consumers and enables more reliable assessment of RAG components 

---
# Vision Language Models for Dynamic Human Activity Recognition in Healthcare Settings 

**Authors**: Abderrazek Abid, Thanh-Cong Ho, Fakhri Karray  

**Link**: [PDF](https://arxiv.org/pdf/2510.21424)  

**Abstract**: As generative AI continues to evolve, Vision Language Models (VLMs) have emerged as promising tools in various healthcare applications. One area that remains relatively underexplored is their use in human activity recognition (HAR) for remote health monitoring. VLMs offer notable strengths, including greater flexibility and the ability to overcome some of the constraints of traditional deep learning models. However, a key challenge in applying VLMs to HAR lies in the difficulty of evaluating their dynamic and often non-deterministic outputs. To address this gap, we introduce a descriptive caption data set and propose comprehensive evaluation methods to evaluate VLMs in HAR. Through comparative experiments with state-of-the-art deep learning models, our findings demonstrate that VLMs achieve comparable performance and, in some cases, even surpass conventional approaches in terms of accuracy. This work contributes a strong benchmark and opens new possibilities for the integration of VLMs into intelligent healthcare systems. 

---
# HalleluBERT: Let every token that has meaning bear its weight 

**Authors**: Raphael Scheible-Schmitt  

**Link**: [PDF](https://arxiv.org/pdf/2510.21372)  

**Abstract**: Transformer-based models have advanced NLP, yet Hebrew still lacks a large-scale RoBERTa encoder which is extensively trained. Existing models such as HeBERT, AlephBERT, and HeRo are limited by corpus size, vocabulary, or training depth. We present HalleluBERT, a RoBERTa-based encoder family (base and large) trained from scratch on 49.1~GB of deduplicated Hebrew web text and Wikipedia with a Hebrew-specific byte-level BPE vocabulary. Evaluated on NER and sentiment classification benchmarks, HalleluBERT outperforms both monolingual and multilingual baselines. HalleluBERT sets a new state of the art for Hebrew and highlights the benefits of fully converged monolingual pretraining. 

---
# SindBERT, the Sailor: Charting the Seas of Turkish NLP 

**Authors**: Raphael Scheible-Schmitt, Stefan Schweter  

**Link**: [PDF](https://arxiv.org/pdf/2510.21364)  

**Abstract**: Transformer models have revolutionized NLP, yet many morphologically rich languages remain underrepresented in large-scale pre-training efforts. With SindBERT, we set out to chart the seas of Turkish NLP, providing the first large-scale RoBERTa-based encoder for Turkish. Trained from scratch on 312 GB of Turkish text (mC4, OSCAR23, Wikipedia), SindBERT is released in both base and large configurations, representing the first large-scale encoder-only language model available for Turkish. We evaluate SindBERT on part-of-speech tagging, named entity recognition, offensive language detection, and the TurBLiMP linguistic acceptability benchmark. Our results show that SindBERT performs competitively with existing Turkish and multilingual models, with the large variant achieving the best scores in two of four tasks but showing no consistent scaling advantage overall. This flat scaling trend, also observed for XLM-R and EuroBERT, suggests that current Turkish benchmarks may already be saturated. At the same time, comparisons with smaller but more curated models such as BERTurk highlight that corpus quality and diversity can outweigh sheer data volume. Taken together, SindBERT contributes both as an openly released resource for Turkish NLP and as an empirical case study on the limits of scaling and the central role of corpus composition in morphologically rich languages. The SindBERT models are released under the MIT license and made available in both fairseq and Huggingface formats. 

---
# A Diagnostic Benchmark for Sweden-Related Factual Knowledge 

**Authors**: Jenny Kunz  

**Link**: [PDF](https://arxiv.org/pdf/2510.21360)  

**Abstract**: Many Swedish benchmarks are translated US-centric benchmarks, and therefore not suitable for testing knowledge that is particularly relevant, or even specific, to Sweden. We therefore introduce a manually written question-answering benchmark specifically targeted to Sweden-related personalities and events, many of which receive very limited coverage in international media. Our annotators drew inspiration from a popular radio program featuring public figures from culture and media, as well as major sports events in Sweden. The dataset can be used to measure factual recall across models of varying sizes and degrees of Swedish coverage, and allows to probe cross-lingual factual consistency as to contains English translations. Using the dataset, we find that smaller models with stronger Swedish coverage perform comparably to a three times larger multilingual model in recalling Sweden-related facts. We also observe that continued pre-training on Swedish generally improves factual knowledge but also leads to forgetting of a part of the previously known information. These results demonstrate the dataset's potential as a diagnostic tool for studying language adaptation and knowledge retention in multilingual models and during language adaptation. 

---
# Multi-turn Training with Basic Human Feedback Helps Little on LLM Reasoning 

**Authors**: Qiang Liu, Wuganjing Song, Zhenzhou Lin, Feifan Chen, Qiaolong Cai, Chen Li, Yongduo Sui  

**Link**: [PDF](https://arxiv.org/pdf/2510.21339)  

**Abstract**: The reasoning capabilities of Large Language Models (LLMs) are typically developed through the single-turn reinforcement learning, whereas real-world applications often involve multi-turn interactions with human feedback, leading to a potential mismatch between training and deployment conditions. In this work, we study whether multi-turn training with human feedback is necessary for reasoning tasks. We compare conventional single-turn training with three multi-turn strategies and reach contrary conclusions to previous research. We find that models trained in a single-turn setting generalize effectively to both single- and multi-turn evaluations, while models trained with multi-turn strategies exhibit a significant degradation in single-turn reasoning performance. These results suggest that for tasks with complete information, robust single-turn training remains more effective and reliable, as multi-turn training with basic feedback provides limited benefits and can even degrade reasoning capabilities. 

---
# TripTide: A Benchmark for Adaptive Travel Planning under Disruptions 

**Authors**: Priyanshu Karmakar, Soumyabrata Chaudhuri, Shubhojit Mallick, Manish Gupta, Abhik Jana, Shreya Ghosh  

**Link**: [PDF](https://arxiv.org/pdf/2510.21329)  

**Abstract**: Recent efforts like TripCraft and TravelPlanner have advanced the use of Large Language Models ( LLMs) for personalized, constraint aware travel itinerary generation. Yet, real travel often faces disruptions. To address this, we present TripTide, the first benchmark evaluating LLM's ability to revise itineraries under realistic disruptions. TripTide models key dimensions such as disruption severity and traveler tolerance, enabling nuanced assessment of LLM adaptability to events like flight cancellations, weather closures, or overbooked attractions. We conduct a threefold evaluation. First, we introduce automatic metrics including Preservation of Intent (how well the revised plan maintains feasibility and goals), Responsiveness (promptness and appropriateness of disruption handling), and Adaptability (semantic, spatial, and sequential divergence between original and revised plans). Second, we apply an LLM-as-a-judge approach to automatically assess revision quality. Third, we perform manual expert evaluation to verify whether revisions preserve semantic, spatial, sequential, and responsive aspects. Our experiments show that LLMs maintain strong sequential consistency and semantic stability, while spatial deviations are larger for shorter trips but decrease with longer ones, indicating that extended plans encourage better geographic coherence. However, disruption-handling ability declines as plan length increases, highlighting limits in LLM robustness. TripTide establishes a benchmark for evaluating adaptability, personalization, and resilience in LLM-based travel planning under real-world uncertainty. 

---
# Typoglycemia under the Hood: Investigating Language Models' Understanding of Scrambled Words 

**Authors**: Gianluca Sperduti, Alejandro Moreo  

**Link**: [PDF](https://arxiv.org/pdf/2510.21326)  

**Abstract**: Research in linguistics has shown that humans can read words with internally scrambled letters, a phenomenon recently dubbed typoglycemia. Some specific NLP models have recently been proposed that similarly demonstrate robustness to such distortions by ignoring the internal order of characters by design. This raises a fundamental question: how can models perform well when many distinct words (e.g., form and from) collapse into identical representations under typoglycemia? Our work, focusing exclusively on the English language, seeks to shed light on the underlying aspects responsible for this robustness. We hypothesize that the main reasons have to do with the fact that (i) relatively few English words collapse under typoglycemia, and that (ii) collapsed words tend to occur in contexts so distinct that disambiguation becomes trivial. In our analysis, we (i) analyze the British National Corpus to quantify word collapse and ambiguity under typoglycemia, (ii) evaluate BERT's ability to disambiguate collapsing forms, and (iii) conduct a probing experiment by comparing variants of BERT trained from scratch on clean versus typoglycemic Wikipedia text; our results reveal that the performance degradation caused by scrambling is smaller than expected. 

---
# Efficient semantic uncertainty quantification in language models via diversity-steered sampling 

**Authors**: Ji Won Park, Kyunghyun Cho  

**Link**: [PDF](https://arxiv.org/pdf/2510.21310)  

**Abstract**: Accurately estimating semantic aleatoric and epistemic uncertainties in large language models (LLMs) is particularly challenging in free-form question answering (QA), where obtaining stable estimates often requires many expensive generations. We introduce a diversity-steered sampler that discourages semantically redundant outputs during decoding, covers both autoregressive and masked diffusion paradigms, and yields substantial sample-efficiency gains. The key idea is to inject a continuous semantic-similarity penalty into the model's proposal distribution using a natural language inference (NLI) model lightly finetuned on partial prefixes or intermediate diffusion states. We debias downstream uncertainty estimates with importance reweighting and shrink their variance with control variates. Across four QA benchmarks, our method matches or surpasses baselines while covering more semantic clusters with the same number of samples. Being modular and requiring no gradient access to the base LLM, the framework promises to serve as a drop-in enhancement for uncertainty estimation in risk-sensitive model deployments. 

---
# PARL: Prompt-based Agents for Reinforcement Learning 

**Authors**: Yarik Menchaca Resendiz, Roman Klinger  

**Link**: [PDF](https://arxiv.org/pdf/2510.21306)  

**Abstract**: Large language models (LLMs) have demonstrated high performance on tasks expressed in natural language, particularly in zero- or few-shot settings. These are typically framed as supervised (e.g., classification) or unsupervised (e.g., clustering) problems. However, limited work evaluates LLMs as agents in reinforcement learning (RL) tasks (e.g., playing games), where learning occurs through interaction with an environment and a reward system. While prior work focused on representing tasks that rely on a language representation, we study structured, non-linguistic reasoning - such as interpreting positions in a grid world. We therefore introduce PARL (Prompt-based Agent for Reinforcement Learning), a method that uses LLMs as RL agents through prompting, without any fine-tuning. PARL encodes actions, states, and rewards in the prompt, enabling the model to learn through trial-and-error interaction. We evaluate PARL on three standard RL tasks that do not entirely rely on natural language. We show that it can match or outperform traditional RL agents in simple environments by leveraging pretrained knowledge. However, we identify performance limitations in tasks that require complex mathematical operations or decoding states and actions. 

---
# Sparser Block-Sparse Attention via Token Permutation 

**Authors**: Xinghao Wang, Pengyu Wang, Dong Zhang, Chenkun Tan, Shaojun Zhou, Zhaoxiang Liu, Shiguo Lian, Fangxu Liu, Kai Song, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2510.21270)  

**Abstract**: Scaling the context length of large language models (LLMs) offers significant benefits but is computationally expensive. This expense stems primarily from the self-attention mechanism, whose $O(N^2)$ complexity with respect to sequence length presents a major bottleneck for both memory and latency. Fortunately, the attention matrix is often sparse, particularly for long sequences, suggesting an opportunity for optimization. Block-sparse attention has emerged as a promising solution that partitions sequences into blocks and skips computation for a subset of these blocks. However, the effectiveness of this method is highly dependent on the underlying attention patterns, which can lead to sub-optimal block-level sparsity. For instance, important key tokens for queries within a single block may be scattered across numerous other blocks, leading to computational redundancy. In this work, we propose Permuted Block-Sparse Attention (\textbf{PBS-Attn}), a plug-and-play method that leverages the permutation properties of attention to increase block-level sparsity and enhance the computational efficiency of LLM prefilling. We conduct comprehensive experiments on challenging real-world long-context datasets, demonstrating that PBS-Attn consistently outperforms existing block-sparse attention methods in model accuracy and closely matches the full attention baseline. Powered by our custom permuted-FlashAttention kernels, PBS-Attn achieves an end-to-end speedup of up to $2.75\times$ in long-context prefilling, confirming its practical viability. Code available at this https URL 

---
# Correlation Dimension of Auto-Regressive Large Language Models 

**Authors**: Xin Du, Kumiko Tanaka-Ishii  

**Link**: [PDF](https://arxiv.org/pdf/2510.21258)  

**Abstract**: Large language models (LLMs) have achieved remarkable progress in natural language generation, yet they continue to display puzzling behaviors -- such as repetition and incoherence -- even when exhibiting low perplexity. This highlights a key limitation of conventional evaluation metrics, which emphasize local prediction accuracy while overlooking long-range structural complexity. We introduce correlation dimension, a fractal-geometric measure of self-similarity, to quantify the epistemological complexity of text as perceived by a language model. This measure captures the hierarchical recurrence structure of language, bridging local and global properties in a unified framework. Through extensive experiments, we show that correlation dimension (1) reveals three distinct phases during pretraining, (2) reflects context-dependent complexity, (3) indicates a model's tendency toward hallucination, and (4) reliably detects multiple forms of degeneration in generated text. The method is computationally efficient, robust to model quantization (down to 4-bit precision), broadly applicable across autoregressive architectures (e.g., Transformer and Mamba), and provides fresh insight into the generative dynamics of LLMs. 

---
# DispatchMAS: Fusing taxonomy and artificial intelligence agents for emergency medical services 

**Authors**: Xiang Li, Huizi Yu, Wenkong Wang, Yiran Wu, Jiayan Zhou, Wenyue Hua, Xinxin Lin, Wenjia Tan, Lexuan Zhu, Bingyi Chen, Guang Chen, Ming-Li Chen, Yang Zhou, Zhao Li, Themistocles L. Assimes, Yongfeng Zhang, Qingyun Wu, Xin Ma, Lingyao Li, Lizhou Fan  

**Link**: [PDF](https://arxiv.org/pdf/2510.21228)  

**Abstract**: Objective: Emergency medical dispatch (EMD) is a high-stakes process challenged by caller distress, ambiguity, and cognitive load. Large Language Models (LLMs) and Multi-Agent Systems (MAS) offer opportunities to augment dispatchers. This study aimed to develop and evaluate a taxonomy-grounded, LLM-powered multi-agent system for simulating realistic EMD scenarios. Methods: We constructed a clinical taxonomy (32 chief complaints, 6 caller identities from MIMIC-III) and a six-phase call protocol. Using this framework, we developed an AutoGen-based MAS with Caller and Dispatcher Agents. The system grounds interactions in a fact commons to ensure clinical plausibility and mitigate misinformation. We used a hybrid evaluation framework: four physicians assessed 100 simulated cases for "Guidance Efficacy" and "Dispatch Effectiveness," supplemented by automated linguistic analysis (sentiment, readability, politeness). Results: Human evaluation, with substantial inter-rater agreement (Gwe's AC1 > 0.70), confirmed the system's high performance. It demonstrated excellent Dispatch Effectiveness (e.g., 94 % contacting the correct potential other agents) and Guidance Efficacy (advice provided in 91 % of cases), both rated highly by physicians. Algorithmic metrics corroborated these findings, indicating a predominantly neutral affective profile (73.7 % neutral sentiment; 90.4 % neutral emotion), high readability (Flesch 80.9), and a consistently polite style (60.0 % polite; 0 % impolite). Conclusion: Our taxonomy-grounded MAS simulates diverse, clinically plausible dispatch scenarios with high fidelity. Findings support its use for dispatcher training, protocol evaluation, and as a foundation for real-time decision support. This work outlines a pathway for safely integrating advanced AI agents into emergency response workflows. 

---
# The "Right" Discourse on Migration: Analysing Migration-Related Tweets in Right and Far-Right Political Movements 

**Authors**: Nishan Chatterjee, Veronika Bajt, Ana Zwitter Vitez, Senja Pollak  

**Link**: [PDF](https://arxiv.org/pdf/2510.21220)  

**Abstract**: The rise of right-wing populism in Europe has brought to the forefront the significance of analysing social media discourse to understand the dissemination of extremist ideologies and their impact on political outcomes. Twitter, as a platform for interaction and mobilisation, provides a unique window into the everyday communication of far-right supporters. In this paper, we propose a methodology that uses state-of-the-art natural language processing techniques with sociological insights to analyse the MIGR-TWIT corpus of far-right tweets in English and French. We aim to uncover patterns of discourse surrounding migration, hate speech, and persuasion techniques employed by right and far-right actors. By integrating linguistic, sociological, and computational approaches, we seek to offer cross-disciplinary insights into societal dynamics and contribute to a better understanding of contemporary challenges posed by right-wing extremism on social media platforms. 

---
# Estonian Native Large Language Model Benchmark 

**Authors**: Helena Grete Lillepalu, Tanel Alumäe  

**Link**: [PDF](https://arxiv.org/pdf/2510.21193)  

**Abstract**: The availability of LLM benchmarks for the Estonian language is limited, and a comprehensive evaluation comparing the performance of different LLMs on Estonian tasks has yet to be conducted. We introduce a new benchmark for evaluating LLMs in Estonian, based on seven diverse datasets. These datasets assess general and domain-specific knowledge, understanding of Estonian grammar and vocabulary, summarization abilities, contextual comprehension, and more. The datasets are all generated from native Estonian sources without using machine translation. We compare the performance of base models, instruction-tuned open-source models, and commercial models. Our evaluation includes 6 base models and 26 instruction-tuned models. To assess the results, we employ both human evaluation and LLM-as-a-judge methods. Human evaluation scores showed moderate to high correlation with benchmark evaluations, depending on the dataset. Claude 3.7 Sonnet, used as an LLM judge, demonstrated strong alignment with human ratings, indicating that top-performing LLMs can effectively support the evaluation of Estonian-language models. 

---
# Social Simulations with Large Language Model Risk Utopian Illusion 

**Authors**: Ning Bian, Xianpei Han, Hongyu Lin, Baolei Wu, Jun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.21180)  

**Abstract**: Reliable simulation of human behavior is essential for explaining, predicting, and intervening in our society. Recent advances in large language models (LLMs) have shown promise in emulating human behaviors, interactions, and decision-making, offering a powerful new lens for social science studies. However, the extent to which LLMs diverge from authentic human behavior in social contexts remains underexplored, posing risks of misinterpretation in scientific studies and unintended consequences in real-world applications. Here, we introduce a systematic framework for analyzing LLMs' behavior in social simulation. Our approach simulates multi-agent interactions through chatroom-style conversations and analyzes them across five linguistic dimensions, providing a simple yet effective method to examine emergent social cognitive biases. We conduct extensive experiments involving eight representative LLMs across three families. Our findings reveal that LLMs do not faithfully reproduce genuine human behavior but instead reflect overly idealized versions of it, shaped by the social desirability bias. In particular, LLMs show social role bias, primacy effect, and positivity bias, resulting in "Utopian" societies that lack the complexity and variability of real human interactions. These findings call for more socially grounded LLMs that capture the diversity of human social behavior. 

---
# Large Language Models Meet Text-Attributed Graphs: A Survey of Integration Frameworks and Applications 

**Authors**: Guangxin Su, Hanchen Wang, Jianwei Wang, Wenjie Zhang, Ying Zhang, Jian Pei  

**Link**: [PDF](https://arxiv.org/pdf/2510.21131)  

**Abstract**: Large Language Models (LLMs) have achieved remarkable success in natural language processing through strong semantic understanding and generation. However, their black-box nature limits structured and multi-hop reasoning. In contrast, Text-Attributed Graphs (TAGs) provide explicit relational structures enriched with textual context, yet often lack semantic depth. Recent research shows that combining LLMs and TAGs yields complementary benefits: enhancing TAG representation learning and improving the reasoning and interpretability of LLMs. This survey provides the first systematic review of LLM--TAG integration from an orchestration perspective. We introduce a novel taxonomy covering two fundamental directions: LLM for TAG, where LLMs enrich graph-based tasks, and TAG for LLM, where structured graphs improve LLM reasoning. We categorize orchestration strategies into sequential, parallel, and multi-module frameworks, and discuss advances in TAG-specific pretraining, prompting, and parameter-efficient fine-tuning. Beyond methodology, we summarize empirical insights, curate available datasets, and highlight diverse applications across recommendation systems, biomedical analysis, and knowledge-intensive question answering. Finally, we outline open challenges and promising research directions, aiming to guide future work at the intersection of language and graph learning. 

---
# The Gray Zone of Faithfulness: Taming Ambiguity in Unfaithfulness Detection 

**Authors**: Qiang Ding, Lvzhou Luo, Yixuan Cao, Ping Luo  

**Link**: [PDF](https://arxiv.org/pdf/2510.21118)  

**Abstract**: Ensuring that Large Language Models (LLMs) generate summaries faithful to a given source document is essential for real-world applications. While prior research has explored LLM faithfulness, existing benchmarks suffer from annotation ambiguity, primarily due to the ill-defined boundary of permissible external knowledge in generated outputs. For instance, common sense is often incorporated into responses and labeled as "faithful", yet the acceptable extent of such knowledge remains unspecified, leading to inconsistent annotations. To address this issue, we propose a novel faithfulness annotation framework, which introduces an intermediate category, Out-Dependent, to classify cases where external knowledge is required for verification. Using this framework, we construct VeriGray (Verification with the Gray Zone) -- a new unfaithfulness detection benchmark in summarization. Statistics reveal that even SOTA LLMs, such as GPT-5, exhibit hallucinations ($\sim 6\%$ of sentences) in summarization tasks. Moreover, a substantial proportion ($\sim 8\%$ on average of models) of generated sentences fall into the Out-Dependent category, underscoring the importance of resolving annotation ambiguity in unfaithfulness detection benchmarks. Experiments demonstrate that our benchmark poses significant challenges to multiple baseline methods, indicating considerable room for future improvement. 

---
# Self-Rewarding PPO: Aligning Large Language Models with Demonstrations Only 

**Authors**: Qingru Zhang, Liang Qiu, Ilgee Hong, Zhenghao Xu, Tianyi Liu, Shiyang Li, Rongzhi Zhang, Zheng Li, Lihong Li, Bing Yin, Chao Zhang, Jianshu Chen, Haoming Jiang, Tuo Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2510.21090)  

**Abstract**: Supervised fine-tuning (SFT) has emerged as a crucial method for aligning large language models (LLMs) with human-annotated demonstrations. However, SFT, being an off-policy approach similar to behavior cloning, often struggles with overfitting and poor out-of-domain generalization, especially in limited-data scenarios. To address these limitations, we propose Self-Rewarding PPO, a novel fine-tuning method that leverages on-policy techniques to enhance generalization performance. Our approach combines the strengths of SFT and proximal policy optimization (PPO) to achieve more effective alignment from demonstration data. At its core is a reward function designed as the log policy ratio between the SFT model and the pretrained base model. This function serves as an implicit reward signal, using the pretrained policy as a baseline and the SFT policy as a target. By doing so, it enables on-policy fine-tuning without relying on human preference annotations. The integration of this self-rewarding mechanism with PPO addresses key limitations of SFT, improving generalization, data efficiency, and robustness. Our empirical evaluation across a range of natural language processing tasks demonstrates that Self-Rewarding PPO consistently outperforms traditional SFT methods. The results highlight the effectiveness of our approach in aligning LLMs using demonstration data, particularly in scenarios where high-quality annotated data is scarce. 

---
# CDrugRed: A Chinese Drug Recommendation Dataset for Discharge Medications in Metabolic Diseases 

**Authors**: Juntao Li, Haobin Yuan, Ling Luo, Yan Jiang, Fan Wang, Ping Zhang, Huiyi Lv, Jian Wang, Yuanyuan Sun, Hongfei Lin  

**Link**: [PDF](https://arxiv.org/pdf/2510.21084)  

**Abstract**: Intelligent drug recommendation based on Electronic Health Records (EHRs) is critical for improving for improving the quality and efficiency of clinical decision-making. By leveraging large-scale patient data, drug recommendation systems can assist physicians in selecting the most appropriate medications according to a patient's medical history, diagnoses, laboratory results, and comorbidities. However, the advancement of such systems is significantly hampered by the scarcity of publicly available, real-world EHR datasets, particularly in languages other than English. In this work, we present CDrugRed, a first publicly available Chinese drug recommendation dataset focused on discharge medications for metabolic diseases. The dataset includes 5,894 de-identified records from 3,190 patients, containing comprehensive information such as patient demographics, medical history, clinical course, and discharge diagnoses. We assess the utility of CDrugRed by benchmarking several state-of-the-art large language models (LLMs) on the discharge medication recommendation task. Experimental results show that while supervised fine-tuning improves model performance, there remains substantial room for improvement, with the best model achieving the F1 score of 0.5648 and Jaccard score of 0.4477. This result highlights the complexity of the clinical drug recommendation task and establishes CDrugRed as a challenging and valuable resource for developing more robust and accurate drug recommendation systems. The dataset is publicly available to the research community under the data usage agreements at this https URL. 

---
# Bridging Language Gaps with Adaptive RAG: Improving Indonesian Language Question Answering 

**Authors**: William Christian, Daniel Adamlu, Adrian Yu, Derwin Suhartono  

**Link**: [PDF](https://arxiv.org/pdf/2510.21068)  

**Abstract**: Question Answering (QA) has seen significant improvements with the advancement of machine learning models, further studies enhanced this question answering system by retrieving external information, called Retrieval-Augmented Generation (RAG) to produce more accurate and informative answers. However, these state-of-the-art-performance is predominantly in English language. To address this gap we made an effort of bridging language gaps by incorporating Adaptive RAG system to Indonesian language. Adaptive RAG system integrates a classifier whose task is to distinguish the question complexity, which in turn determines the strategy for answering the question. To overcome the limited availability of Indonesian language dataset, our study employs machine translation as data augmentation approach. Experiments show reliable question complexity classifier; however, we observed significant inconsistencies in multi-retrieval answering strategy which negatively impacted the overall evaluation when this strategy was applied. These findings highlight both the promise and challenges of question answering in low-resource language suggesting directions for future improvement. 

---
# Dynamic Retriever for In-Context Knowledge Editing via Policy Optimization 

**Authors**: Mahmud Wasif Nafee, Maiqi Jiang, Haipeng Chen, Yanfu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.21059)  

**Abstract**: Large language models (LLMs) excel at factual recall yet still propagate stale or incorrect knowledge. In-context knowledge editing offers a gradient-free remedy suitable for black-box APIs, but current editors rely on static demonstration sets chosen by surface-level similarity, leading to two persistent obstacles: (i) a quantity-quality trade-off, and (ii) lack of adaptivity to task difficulty. We address these issues by dynamically selecting supporting demonstrations according to their utility for the edit. We propose Dynamic Retriever for In-Context Knowledge Editing (DR-IKE), a lightweight framework that (1) trains a BERT retriever with REINFORCE to rank demonstrations by editing reward, and (2) employs a learnable threshold to prune low-value examples, shortening the prompt when the edit is easy and expanding it when the task is hard. DR-IKE performs editing without modifying model weights, relying solely on forward passes for compatibility with black-box LLMs. On the COUNTERFACT benchmark, it improves edit success by up to 17.1%, reduces latency by 41.6%, and preserves accuracy on unrelated queries, demonstrating scalable and adaptive knowledge editing. The code is available at this https URL . 

---
# Reasoning's Razor: Reasoning Improves Accuracy but Can Hurt Recall at Critical Operating Points in Safety and Hallucination Detection 

**Authors**: Atoosa Chegini, Hamid Kazemi, Garrett Souza, Maria Safi, Yang Song, Samy Bengio, Sinead Williamson, Mehrdad Farajtabar  

**Link**: [PDF](https://arxiv.org/pdf/2510.21049)  

**Abstract**: Reasoning has become a central paradigm for large language models (LLMs), consistently boosting accuracy across diverse benchmarks. Yet its suitability for precision-sensitive tasks remains unclear. We present the first systematic study of reasoning for classification tasks under strict low false positive rate (FPR) regimes. Our analysis covers two tasks--safety detection and hallucination detection--evaluated in both fine-tuned and zero-shot settings, using standard LLMs and Large Reasoning Models (LRMs). Our results reveal a clear trade-off: Think On (reasoning-augmented) generation improves overall accuracy, but underperforms at the low-FPR thresholds essential for practical use. In contrast, Think Off (no reasoning during inference) dominates in these precision-sensitive regimes, with Think On surpassing only when higher FPRs are acceptable. In addition, we find token-based scoring substantially outperforms self-verbalized confidence for precision-sensitive deployments. Finally, a simple ensemble of the two modes recovers the strengths of each. Taken together, our findings position reasoning as a double-edged tool: beneficial for average accuracy, but often ill-suited for applications requiring strict precision. 

---
# Input Matters: Evaluating Input Structure's Impact on LLM Summaries of Sports Play-by-Play 

**Authors**: Barkavi Sundararajan, Somayajulu Sripada, Ehud Reiter  

**Link**: [PDF](https://arxiv.org/pdf/2510.21034)  

**Abstract**: A major concern when deploying LLMs in accuracy-critical domains such as sports reporting is that the generated text may not faithfully reflect the input data. We quantify how input structure affects hallucinations and other factual errors in LLM-generated summaries of NBA play-by-play data, across three formats: row-structured, JSON and unstructured. We manually annotated 3,312 factual errors across 180 game summaries produced by two models, Llama-3.1-70B and Qwen2.5-72B. Input structure has a strong effect: JSON input reduces error rates by 69% for Llama and 65% for Qwen compared to unstructured input, while row-structured input reduces errors by 54% for Llama and 51% for Qwen. A two-way repeated measures ANOVA shows that input structure accounts for over 80% of the variance in error rates, with Tukey HSD post hoc tests confirming statistically significant differences between all input formats. 

---
# Can Confidence Estimates Decide When Chain-of-thought is Necessary for Llms? 

**Authors**: Samuel Lewis-Lim, Xingwei Tan, Zhixue Zhao, Nikolaos Aletras  

**Link**: [PDF](https://arxiv.org/pdf/2510.21007)  

**Abstract**: Chain-of-thought (CoT) prompting has emerged as a common technique for enhancing the reasoning abilities of large language models (LLMs). While extended reasoning can boost accuracy on complex tasks, it is often unnecessary and substantially increases token usage, limiting the practicality of reasoning models in many scenarios. Recent models, such as GPT-OSS and Qwen3, expose controls that enable users to adjust the length of CoT or determine whether it is used at all. Yet, it remains unclear when CoT should be used: on some tasks it improves performance, while on others it provides little benefit or even harms performance. We address this challenge with confidence-gated CoT, where a model invokes reasoning only when confidence in its direct answer is low. To this end, we present the first systematic study of training-free confidence estimation methods for CoT gating. Specifically, we evaluate four training-free confidence estimation methods and compare them to a random baseline and an oracle that always knows when CoT is needed. Through extensive experiments, we show that existing training-free confidence measures can reduce redundant CoT and outperform randomly invoked CoT. However, the utility of individual confidence measures is inconsistent, varying with both the dataset and the model, underscoring the difficulty of deploying confidence-gated CoT in practice. By analysing both strengths and failure modes, our study highlights the potential and limitations of current methods and paves the way toward more reliable adaptive gating of CoT. 

---
# Irish-BLiMP: A Linguistic Benchmark for Evaluating Human and Language Model Performance in a Low-Resource Setting 

**Authors**: Josh McGiff, Khanh-Tung Tran, William Mulcahy, Dáibhidh Ó Luinín, Jake Dalzell, Róisín Ní Bhroin, Adam Burke, Barry O'Sullivan, Hoang D. Nguyen, Nikola S. Nikolov  

**Link**: [PDF](https://arxiv.org/pdf/2510.20957)  

**Abstract**: We present Irish-BLiMP (Irish Benchmark of Linguistic Minimal Pairs), the first dataset and framework designed for fine-grained evaluation of linguistic competence in the Irish language, an endangered language. Drawing on a variety of linguistic literature and grammar reference works, we manually constructed and reviewed 1020 minimal pairs across a taxonomy of 11 linguistic features, through a team of fluent Irish speakers. We evaluate both existing Large Language Models (LLMs) and fluent human participants on their syntactic knowledge of Irish. Our findings show that humans outperform all models across all linguistic features, achieving 16.6% higher accuracy on average. Moreover, a substantial performance gap of 18.1% persists between open- and closed-source LLMs, with even the strongest model (gpt-5) reaching only 73.5% accuracy compared to 90.1% by human. Interestingly, human participants and models struggle on different aspects of Irish grammar, thus highlighting a difference in representation learned by the models. Overall, Irish-BLiMP provides the first systematic framework for evaluating the grammatical competence of LLMs in Irish and offers a valuable benchmark for advancing research on linguistic understanding in low-resource languages. 

---
# Do LLMs Truly Understand When a Precedent Is Overruled? 

**Authors**: Li Zhang, Jaromir Savelka, Kevin Ashley  

**Link**: [PDF](https://arxiv.org/pdf/2510.20941)  

**Abstract**: Large language models (LLMs) with extended context windows show promise for complex legal reasoning tasks, yet their ability to understand long legal documents remains insufficiently evaluated. Developing long-context benchmarks that capture realistic, high-stakes tasks remains a significant challenge in the field, as most existing evaluations rely on simplified synthetic tasks that fail to represent the complexity of real-world document understanding. Overruling relationships are foundational to common-law doctrine and commonly found in judicial opinions. They provide a focused and important testbed for long-document legal understanding that closely resembles what legal professionals actually do. We present an assessment of state-of-the-art LLMs on identifying overruling relationships from U.S. Supreme Court cases using a dataset of 236 case pairs. Our evaluation reveals three critical limitations: (1) era sensitivity -- the models show degraded performance on historical cases compared to modern ones, revealing fundamental temporal bias in their training; (2) shallow reasoning -- models rely on shallow logical heuristics rather than deep legal comprehension; and (3) context-dependent reasoning failures -- models produce temporally impossible relationships in complex open-ended tasks despite maintaining basic temporal awareness in simple contexts. Our work contributes a benchmark that addresses the critical gap in realistic long-context evaluation, providing an environment that mirrors the complexity and stakes of actual legal reasoning tasks. 

---
# FicSim: A Dataset for Multi-Faceted Semantic Similarity in Long-Form Fiction 

**Authors**: Natasha Johnson, Amanda Bertsch, Maria-Emil Deal, Emma Strubell  

**Link**: [PDF](https://arxiv.org/pdf/2510.20926)  

**Abstract**: As language models become capable of processing increasingly long and complex texts, there has been growing interest in their application within computational literary studies. However, evaluating the usefulness of these models for such tasks remains challenging due to the cost of fine-grained annotation for long-form texts and the data contamination concerns inherent in using public-domain literature. Current embedding similarity datasets are not suitable for evaluating literary-domain tasks because of a focus on coarse-grained similarity and primarily on very short text. We assemble and release FICSIM, a dataset of long-form, recently written fiction, including scores along 12 axes of similarity informed by author-produced metadata and validated by digital humanities scholars. We evaluate a suite of embedding models on this task, demonstrating a tendency across models to focus on surface-level features over semantic categories that would be useful for computational literary studies tasks. Throughout our data-collection process, we prioritize author agency and rely on continual, informed author consent. 

---
# Code-enabled language models can outperform reasoning models on diverse tasks 

**Authors**: Cedegao E. Zhang, Cédric Colas, Gabriel Poesia, Joshua B. Tenenbaum, Jacob Andreas  

**Link**: [PDF](https://arxiv.org/pdf/2510.20909)  

**Abstract**: Reasoning models (RMs), language models (LMs) trained with reinforcement learning to produce long-form natural language reasoning, have been remarkably successful, but they still require large amounts of computation and data to train, and can be slow and expensive to run. In this paper, we show that standard instruct LMs can already be elicited to be strong reasoners at a level comparable to or even surpassing their corresponding RMs (e.g., DeepSeek V3 vs R1) without finetuning, across diverse domains from instruction following and creative generation to mathematical reasoning. This is achieved by CodeAdapt, our simple recipe that combines the CodeAct framework, where LMs interleave natural language reasoning with code execution in a multi-step fashion, with few-shot bootstrap in-context learning from as few as five training problems. Analyzing four matched pairs of LMs and RMs, we find that CodeAdapt enables three LMs to outperform the corresponding RMs on average over eight tasks (up to 22.9%) while being 10-81% more token efficient, and delivers superior performance on six tasks when averaged over the four models (up to 35.7%). Furthermore, the code-augmented reasoning traces display rich and varied problem-solving strategies. Our findings support that (1) CodeAdapt-style learning and reasoning may be robust and domain general and (2) code-enabled LMs are cognitively grounded and powerful systems, potentially providing a strong foundation for in-weight reinforcement learning. 

---
# Shoot First, Ask Questions Later? Building Rational Agents that Explore and Act Like People 

**Authors**: Gabriel Grand, Valerio Pepe, Jacob Andreas, Joshua B. Tenenbaum  

**Link**: [PDF](https://arxiv.org/pdf/2510.20886)  

**Abstract**: Many high-stakes applications of AI require forming data-driven hypotheses and making targeted guesses; e.g., in scientific and diagnostic settings. Given limited resources, to what extent do agents based on language models (LMs) act rationally? We develop methods to benchmark and enhance agentic information-seeking, drawing on insights from human behavior. First, we introduce a strategic decision-oriented dialogue task called Collaborative Battleship, in which a partially-informed Captain must balance exploration (asking questions) and action (taking shots), while a fully-informed Spotter must provide accurate answers under an information bottleneck. Compared to human players (N=42), we find that LM agents struggle to ground answers in context, generate informative questions, and select high-value actions. Next, to address these gaps, we develop novel Monte Carlo inference strategies for LMs based on principles from Bayesian Experimental Design (BED). For Spotter agents, our approach boosts accuracy by up to 14.7% absolute over LM-only baselines; for Captain agents, it raises expected information gain (EIG) by up to 0.227 bits (94.2% of the achievable noise ceiling). Combined, these components yield sharper targeting (+0.303-0.374 F1), and enable weaker LMs, such as Llama-4-Scout, to outperform both humans (8% -> 82% win rate) and frontier models (0% -> 67% win rate vs. GPT-5) at ~1% of GPT-5's cost. We replicate these findings on Guess Who? where our methods significantly boost accuracy (+28.3-42.4 p.p.), demonstrating their general applicability for building rational information-seeking agents. 

---
# AstaBench: Rigorous Benchmarking of AI Agents with a Scientific Research Suite 

**Authors**: Jonathan Bragg, Mike D'Arcy, Nishant Balepur, Dan Bareket, Bhavana Dalvi, Sergey Feldman, Dany Haddad, Jena D. Hwang, Peter Jansen, Varsha Kishore, Bodhisattwa Prasad Majumder, Aakanksha Naik, Sigal Rahamimov, Kyle Richardson, Amanpreet Singh, Harshit Surana, Aryeh Tiktinsky, Rosni Vasu, Guy Wiener, Chloe Anastasiades, Stefan Candra, Jason Dunkelberger, Dan Emery, Rob Evans, Malachi Hamada, Regan Huff, Rodney Kinney, Matt Latzke, Jaron Lochner, Ruben Lozano-Aguilera, Cecile Nguyen, Smita Rao, Amber Tanaka, Brooke Vlahos, Peter Clark, Doug Downey, Yoav Goldberg, Ashish Sabharwal, Daniel S. Weld  

**Link**: [PDF](https://arxiv.org/pdf/2510.21652)  

**Abstract**: AI agents hold the potential to revolutionize scientific productivity by automating literature reviews, replicating experiments, analyzing data, and even proposing new directions of inquiry; indeed, there are now many such agents, ranging from general-purpose "deep research" systems to specialized science-specific agents, such as AI Scientist and AIGS. Rigorous evaluation of these agents is critical for progress. Yet existing benchmarks fall short on several fronts: they (1) fail to provide holistic, product-informed measures of real-world use cases such as science research; (2) lack reproducible agent tools necessary for a controlled comparison of core agentic capabilities; (3) do not account for confounding variables such as model cost and tool access; (4) do not provide standardized interfaces for quick agent prototyping and evaluation; and (5) lack comprehensive baseline agents necessary to identify true advances. In response, we define principles and tooling for more rigorously benchmarking agents. Using these, we present AstaBench, a suite that provides the first holistic measure of agentic ability to perform scientific research, comprising 2400+ problems spanning the entire scientific discovery process and multiple scientific domains, and including many problems inspired by actual user requests to deployed Asta agents. Our suite comes with the first scientific research environment with production-grade search tools that enable controlled, reproducible evaluation, better accounting for confounders. Alongside, we provide a comprehensive suite of nine science-optimized classes of Asta agents and numerous baselines. Our extensive evaluation of 57 agents across 22 agent classes reveals several interesting findings, most importantly that despite meaningful progress on certain individual aspects, AI remains far from solving the challenge of science research assistance. 

---
# Few-Shot Knowledge Distillation of LLMs With Counterfactual Explanations 

**Authors**: Faisal Hamman, Pasan Dissanayake, Yanjun Fu, Sanghamitra Dutta  

**Link**: [PDF](https://arxiv.org/pdf/2510.21631)  

**Abstract**: Knowledge distillation is a promising approach to transfer capabilities from complex teacher models to smaller, resource-efficient student models that can be deployed easily, particularly in task-aware scenarios. However, existing methods of task-aware distillation typically require substantial quantities of data which may be unavailable or expensive to obtain in many practical scenarios. In this paper, we address this challenge by introducing a novel strategy called Counterfactual-explanation-infused Distillation CoD for few-shot task-aware knowledge distillation by systematically infusing counterfactual explanations. Counterfactual explanations (CFEs) refer to inputs that can flip the output prediction of the teacher model with minimum perturbation. Our strategy CoD leverages these CFEs to precisely map the teacher's decision boundary with significantly fewer samples. We provide theoretical guarantees for motivating the role of CFEs in distillation, from both statistical and geometric perspectives. We mathematically show that CFEs can improve parameter estimation by providing more informative examples near the teacher's decision boundary. We also derive geometric insights on how CFEs effectively act as knowledge probes, helping the students mimic the teacher's decision boundaries more effectively than standard data. We perform experiments across various datasets and LLMs to show that CoD outperforms standard distillation approaches in few-shot regimes (as low as 8-512 samples). Notably, CoD only uses half of the original samples used by the baselines, paired with their corresponding CFEs and still improves performance. 

---
# DeepAgent: A General Reasoning Agent with Scalable Toolsets 

**Authors**: Xiaoxi Li, Wenxiang Jiao, Jiarui Jin, Guanting Dong, Jiajie Jin, Yinuo Wang, Hao Wang, Yutao Zhu, Ji-Rong Wen, Yuan Lu, Zhicheng Dou  

**Link**: [PDF](https://arxiv.org/pdf/2510.21618)  

**Abstract**: Large reasoning models have demonstrated strong problem-solving abilities, yet real-world tasks often require external tools and long-horizon interactions. Existing agent frameworks typically follow predefined workflows, which limit autonomous and global task completion. In this paper, we introduce DeepAgent, an end-to-end deep reasoning agent that performs autonomous thinking, tool discovery, and action execution within a single, coherent reasoning process. To address the challenges of long-horizon interactions, particularly the context length explosion from multiple tool calls and the accumulation of interaction history, we introduce an autonomous memory folding mechanism that compresses past interactions into structured episodic, working, and tool memories, reducing error accumulation while preserving critical information. To teach general-purpose tool use efficiently and stably, we develop an end-to-end reinforcement learning strategy, namely ToolPO, that leverages LLM-simulated APIs and applies tool-call advantage attribution to assign fine-grained credit to the tool invocation tokens. Extensive experiments on eight benchmarks, including general tool-use tasks (ToolBench, API-Bank, TMDB, Spotify, ToolHop) and downstream applications (ALFWorld, WebShop, GAIA, HLE), demonstrate that DeepAgent consistently outperforms baselines across both labeled-tool and open-set tool retrieval scenarios. This work takes a step toward more general and capable agents for real-world applications. The code and demo are available at this https URL. 

---
# Doc-Researcher: A Unified System for Multimodal Document Parsing and Deep Research 

**Authors**: Kuicai Dong, Shurui Huang, Fangda Ye, Wei Han, Zhi Zhang, Dexun Li, Wenjun Li, Qu Yang, Gang Wang, Yichao Wang, Chen Zhang, Yong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.21603)  

**Abstract**: Deep Research systems have revolutionized how LLMs solve complex questions through iterative reasoning and evidence gathering. However, current systems remain fundamentally constrained to textual web data, overlooking the vast knowledge embedded in multimodal documents Processing such documents demands sophisticated parsing to preserve visual semantics (figures, tables, charts, and equations), intelligent chunking to maintain structural coherence, and adaptive retrieval across modalities, which are capabilities absent in existing systems. In response, we present Doc-Researcher, a unified system that bridges this gap through three integrated components: (i) deep multimodal parsing that preserves layout structure and visual semantics while creating multi-granular representations from chunk to document level, (ii) systematic retrieval architecture supporting text-only, vision-only, and hybrid paradigms with dynamic granularity selection, and (iii) iterative multi-agent workflows that decompose complex queries, progressively accumulate evidence, and synthesize comprehensive answers across documents and modalities. To enable rigorous evaluation, we introduce M4DocBench, the first benchmark for Multi-modal, Multi-hop, Multi-document, and Multi-turn deep research. Featuring 158 expert-annotated questions with complete evidence chains across 304 documents, M4DocBench tests capabilities that existing benchmarks cannot assess. Experiments demonstrate that Doc-Researcher achieves 50.6% accuracy, 3.4xbetter than state-of-the-art baselines, validating that effective document research requires not just better retrieval, but fundamentally deep parsing that preserve multimodal integrity and support iterative research. Our work establishes a new paradigm for conducting deep research on multimodal document collections. 

---
# ColorEcosystem: Powering Personalized, Standardized, and Trustworthy Agentic Service in massive-agent Ecosystem 

**Authors**: Fangwen Wu, Zheng Wu, Jihong Wang, Yunku Chen, Ruiguang Pei, Heyuan Huang, Xin Liao, Xingyu Lou, Huarong Deng, Zhihui Fu, Weiwen Liu, Zhuosheng Zhang, Weinan Zhang, Jun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.21566)  

**Abstract**: With the rapid development of (multimodal) large language model-based agents, the landscape of agentic service management has evolved from single-agent systems to multi-agent systems, and now to massive-agent ecosystems. Current massive-agent ecosystems face growing challenges, including impersonal service experiences, a lack of standardization, and untrustworthy behavior. To address these issues, we propose ColorEcosystem, a novel blueprint designed to enable personalized, standardized, and trustworthy agentic service at scale. Concretely, ColorEcosystem consists of three key components: agent carrier, agent store, and agent audit. The agent carrier provides personalized service experiences by utilizing user-specific data and creating a digital twin, while the agent store serves as a centralized, standardized platform for managing diverse agentic services. The agent audit, based on the supervision of developer and user activities, ensures the integrity and credibility of both service providers and users. Through the analysis of challenges, transitional forms, and practical considerations, the ColorEcosystem is poised to power personalized, standardized, and trustworthy agentic service across massive-agent ecosystems. Meanwhile, we have also implemented part of ColorEcosystem's functionality, and the relevant code is open-sourced at this https URL. 

---
# Head Pursuit: Probing Attention Specialization in Multimodal Transformers 

**Authors**: Lorenzo Basile, Valentino Maiorca, Diego Doimo, Francesco Locatello, Alberto Cazzaniga  

**Link**: [PDF](https://arxiv.org/pdf/2510.21518)  

**Abstract**: Language and vision-language models have shown impressive performance across a wide range of tasks, but their internal mechanisms remain only partly understood. In this work, we study how individual attention heads in text-generative models specialize in specific semantic or visual attributes. Building on an established interpretability method, we reinterpret the practice of probing intermediate activations with the final decoding layer through the lens of signal processing. This lets us analyze multiple samples in a principled way and rank attention heads based on their relevance to target concepts. Our results show consistent patterns of specialization at the head level across both unimodal and multimodal transformers. Remarkably, we find that editing as few as 1% of the heads, selected using our method, can reliably suppress or enhance targeted concepts in the model output. We validate our approach on language tasks such as question answering and toxicity mitigation, as well as vision-language tasks including image classification and captioning. Our findings highlight an interpretable and controllable structure within attention layers, offering simple tools for understanding and editing large-scale generative models. 

---
# Wisdom and Delusion of LLM Ensembles for Code Generation and Repair 

**Authors**: Fernando Vallecillos Ruiz, Max Hort, Leon Moonen  

**Link**: [PDF](https://arxiv.org/pdf/2510.21513)  

**Abstract**: Today's pursuit of a single Large Language Model (LMM) for all software engineering tasks is resource-intensive and overlooks the potential benefits of complementarity, where different models contribute unique strengths. However, the degree to which coding LLMs complement each other and the best strategy for maximizing an ensemble's potential are unclear, leaving practitioners without a clear path to move beyond single-model systems.
To address this gap, we empirically compare ten individual LLMs from five families, and three ensembles of these LLMs across three software engineering benchmarks covering code generation and program repair. We assess the complementarity between models and the performance gap between the best individual model and the ensembles. Next, we evaluate various selection heuristics to identify correct solutions from an ensemble's candidate pool.
We find that the theoretical upperbound for an ensemble's performance can be 83% above the best single model. Our results show that consensus-based strategies for selecting solutions fall into a "popularity trap," amplifying common but incorrect outputs. In contrast, a diversity-based strategy realizes up to 95% of this theoretical potential, and proves effective even in small two-model ensembles, enabling a cost-efficient way to enhance performance by leveraging multiple LLMs. 

---
# SBASH: a Framework for Designing and Evaluating RAG vs. Prompt-Tuned LLM Honeypots 

**Authors**: Adetayo Adebimpe, Helmut Neukirchen, Thomas Welsh  

**Link**: [PDF](https://arxiv.org/pdf/2510.21459)  

**Abstract**: Honeypots are decoy systems used for gathering valuable threat intelligence or diverting attackers away from production systems. Maximising attacker engagement is essential to their utility. However research has highlighted that context-awareness, such as the ability to respond to new attack types, systems and attacker agents, is necessary to increase engagement. Large Language Models (LLMs) have been shown as one approach to increase context awareness but suffer from several challenges including accuracy and timeliness of response time, high operational costs and data-protection issues due to cloud deployment. We propose the System-Based Attention Shell Honeypot (SBASH) framework which manages data-protection issues through the use of lightweight local LLMs. We investigate the use of Retrieval Augmented Generation (RAG) supported LLMs and non-RAG LLMs for Linux shell commands and evaluate them using several different metrics such as response time differences, realism from human testers, and similarity to a real system calculated with Levenshtein distance, SBert, and BertScore. We show that RAG improves accuracy for untuned models while models that have been tuned via a system prompt that tells the LLM to respond like a Linux system achieve without RAG a similar accuracy as untuned with RAG, while having a slightly lower latency. 

---
# Does Model Size Matter? A Comparison of Small and Large Language Models for Requirements Classification 

**Authors**: Mohammad Amin Zadenoori, Vincenzo De Martino, Jacek Dabrowski, Xavier Franch, Alessio Ferrari  

**Link**: [PDF](https://arxiv.org/pdf/2510.21443)  

**Abstract**: [Context and motivation] Large language models (LLMs) show notable results in natural language processing (NLP) tasks for requirements engineering (RE). However, their use is compromised by high computational cost, data sharing risks, and dependence on external services. In contrast, small language models (SLMs) offer a lightweight, locally deployable alternative. [Question/problem] It remains unclear how well SLMs perform compared to LLMs in RE tasks in terms of accuracy. [Results] Our preliminary study compares eight models, including three LLMs and five SLMs, on requirements classification tasks using the PROMISE, PROMISE Reclass, and SecReq datasets. Our results show that although LLMs achieve an average F1 score of 2% higher than SLMs, this difference is not statistically significant. SLMs almost reach LLMs performance across all datasets and even outperform them in recall on the PROMISE Reclass dataset, despite being up to 300 times smaller. We also found that dataset characteristics play a more significant role in performance than model size. [Contribution] Our study contributes with evidence that SLMs are a valid alternative to LLMs for requirements classification, offering advantages in privacy, cost, and local deployability. 

---
# HIKMA: Human-Inspired Knowledge by Machine Agents through a Multi-Agent Framework for Semi-Autonomous Scientific Conferences 

**Authors**: Zain Ul Abideen Tariq, Mahmood Al-Zubaidi, Uzair Shah, Marco Agus, Mowafa Househ  

**Link**: [PDF](https://arxiv.org/pdf/2510.21370)  

**Abstract**: HIKMA Semi-Autonomous Conference is the first experiment in reimagining scholarly communication through an end-to-end integration of artificial intelligence into the academic publishing and presentation pipeline. This paper presents the design, implementation, and evaluation of the HIKMA framework, which includes AI dataset curation, AI-based manuscript generation, AI-assisted peer review, AI-driven revision, AI conference presentation, and AI archival dissemination. By combining language models, structured research workflows, and domain safeguards, HIKMA shows how AI can support - not replace traditional scholarly practices while maintaining intellectual property protection, transparency, and integrity. The conference functions as a testbed and proof of concept, providing insights into the opportunities and challenges of AI-enabled scholarship. It also examines questions about AI authorship, accountability, and the role of human-AI collaboration in research. 

---
# FairImagen: Post-Processing for Bias Mitigation in Text-to-Image Models 

**Authors**: Zihao Fu, Ryan Brown, Shun Shao, Kai Rawal, Eoin Delaney, Chris Russell  

**Link**: [PDF](https://arxiv.org/pdf/2510.21363)  

**Abstract**: Text-to-image diffusion models, such as Stable Diffusion, have demonstrated remarkable capabilities in generating high-quality and diverse images from natural language prompts. However, recent studies reveal that these models often replicate and amplify societal biases, particularly along demographic attributes like gender and race. In this paper, we introduce FairImagen (this https URL), a post-hoc debiasing framework that operates on prompt embeddings to mitigate such biases without retraining or modifying the underlying diffusion model. Our method integrates Fair Principal Component Analysis to project CLIP-based input embeddings into a subspace that minimizes group-specific information while preserving semantic content. We further enhance debiasing effectiveness through empirical noise injection and propose a unified cross-demographic projection method that enables simultaneous debiasing across multiple demographic attributes. Extensive experiments across gender, race, and intersectional settings demonstrate that FairImagen significantly improves fairness with a moderate trade-off in image quality and prompt fidelity. Our framework outperforms existing post-hoc methods and offers a simple, scalable, and model-agnostic solution for equitable text-to-image generation. 

---
# Magellan: Guided MCTS for Latent Space Exploration and Novelty Generation 

**Authors**: Lufan Chang  

**Link**: [PDF](https://arxiv.org/pdf/2510.21341)  

**Abstract**: Large Language Models (LLMs) often struggle with generating truly innovative ideas, typically defaulting to high-probability, familiar concepts within their training data's "gravity wells." While advanced search-based methods like Tree of Thoughts (ToT) attempt to mitigate this, they are fundamentally limited by their reliance on unprincipled, inconsistent self-evaluation heuristics to guide exploration. To address this gap, we introduce \textbf{Magellan}, a novel framework that reframes creative generation as a principled, guided exploration of an LLM's latent conceptual space. At its core, Magellan employs Monte Carlo Tree Search (MCTS) governed by a hierarchical guidance system. For long-range direction, a "semantic compass" vector, formulated via orthogonal projection, steers the search towards relevant novelty. For local, step-by-step decisions, a landscape-aware value function replaces flawed self-evaluation with an explicit reward structure that balances intrinsic coherence, extrinsic novelty, and narrative progress. Extensive experiments demonstrate that Magellan significantly outperforms strong baselines, including ReAct and ToT, in generating scientific ideas with superior plausibility and innovation. Our work shows that for creative discovery, a principled, guided search is more effective than unconstrained agency, paving the way for LLMs to become more capable partners in innovation. 

---
# Leverage Unlearning to Sanitize LLMs 

**Authors**: Antoine Boutet, Lucas Magnana  

**Link**: [PDF](https://arxiv.org/pdf/2510.21322)  

**Abstract**: Pre-trained large language models (LLMs) are becoming useful for various tasks. To improve their performance on certain tasks, it is necessary to fine-tune them on specific data corpora (e.g., medical reports, business data). These specialized data corpora may contain sensitive data (e.g., personal or confidential data) that will be memorized by the model and likely to be regurgitated during its subsequent use. This memorization of sensitive information by the model poses a significant privacy or confidentiality issue. To remove this memorization and sanitize the model without requiring costly additional fine-tuning on a secured data corpus, we propose SANI. SANI is an unlearning approach to sanitize language models. It relies on both an erasure and repair phases that 1) reset certain neurons in the last layers of the model to disrupt the memorization of fine-grained information, and then 2) fine-tune the model while avoiding memorizing sensitive information. We comprehensively evaluate SANI to sanitize both a model fine-tuned and specialized with medical data by removing directly and indirectly identifiers from the memorization of the model, and a standard pre-trained model by removing specific terms defined as confidential information from the model. Results show that with only few additional epochs of unlearning, the model is sanitized and the number of regurgitations is drastically reduced. This approach can be particularly useful for hospitals or other industries that have already spent significant resources training models on large datasets and wish to sanitize them before sharing. 

---
# When Models Outthink Their Safety: Mitigating Self-Jailbreak in Large Reasoning Models with Chain-of-Guardrails 

**Authors**: Yingzhi Mao, Chunkang Zhang, Junxiang Wang, Xinyan Guan, Boxi Cao, Yaojie Lu, Hongyu Lin, Xianpei Han, Le Sun  

**Link**: [PDF](https://arxiv.org/pdf/2510.21285)  

**Abstract**: Large Reasoning Models (LRMs) demonstrate remarkable capabilities on complex reasoning tasks but remain vulnerable to severe safety risks, including harmful content generation and jailbreak attacks. Existing mitigation strategies rely on injecting heuristic safety signals during training, which often suppress reasoning ability and fail to resolve the safety-reasoning trade-off. To systematically investigate this issue, we analyze the reasoning trajectories of diverse LRMs and uncover a phenomenon we term Self-Jailbreak, where models override their own risk assessments and justify responding to unsafe prompts. This finding reveals that LRMs inherently possess the ability to reject unsafe queries, but this ability is compromised, resulting in harmful outputs. Building on these insights, we propose the Chain-of-Guardrail (CoG), a training framework that recomposes or backtracks unsafe reasoning steps, steering the model back onto safe trajectories while preserving valid reasoning chains. Extensive experiments across multiple reasoning and safety benchmarks demonstrate that CoG substantially improves the safety of current LRMs while preserving comparable reasoning ability, significantly outperforming prior methods that suffer from severe safety-reasoning trade-offs. 

---
# Pctx: Tokenizing Personalized Context for Generative Recommendation 

**Authors**: Qiyong Zhong, Jiajie Su, Yunshan Ma, Julian McAuley, Yupeng Hou  

**Link**: [PDF](https://arxiv.org/pdf/2510.21276)  

**Abstract**: Generative recommendation (GR) models tokenize each action into a few discrete tokens (called semantic IDs) and autoregressively generate the next tokens as predictions, showing advantages such as memory efficiency, scalability, and the potential to unify retrieval and ranking. Despite these benefits, existing tokenization methods are static and non-personalized. They typically derive semantic IDs solely from item features, assuming a universal item similarity that overlooks user-specific perspectives. However, under the autoregressive paradigm, semantic IDs with the same prefixes always receive similar probabilities, so a single fixed mapping implicitly enforces a universal item similarity standard across all users. In practice, the same item may be interpreted differently depending on user intentions and preferences. To address this issue, we propose a personalized context-aware tokenizer that incorporates a user's historical interactions when generating semantic IDs. This design allows the same item to be tokenized into different semantic IDs under different user contexts, enabling GR models to capture multiple interpretive standards and produce more personalized predictions. Experiments on three public datasets demonstrate up to 11.44% improvement in NDCG@10 over non-personalized action tokenization baselines. Our code is available at this https URL. 

---
# Reducing the Probability of Undesirable Outputs in Language Models Using Probabilistic Inference 

**Authors**: Stephen Zhao, Aidan Li, Rob Brekelmans, Roger Grosse  

**Link**: [PDF](https://arxiv.org/pdf/2510.21184)  

**Abstract**: Reinforcement learning (RL) has become a predominant technique to align language models (LMs) with human preferences or promote outputs which are deemed to be desirable by a given reward function. Standard RL approaches optimize average reward, while methods explicitly focused on reducing the probability of undesired outputs typically come at a cost to average-case performance. To improve this tradeoff, we introduce RePULSe, a new training method that augments the standard RL loss with an additional loss that uses learned proposals to guide sampling low-reward outputs, and then reduces those outputs' probability. We run experiments demonstrating that RePULSe produces a better tradeoff of expected reward versus the probability of undesired outputs and is more adversarially robust, compared to standard RL alignment approaches and alternatives. 

---
# KBE-DME: Dynamic Multimodal Evaluation via Knowledge Enhanced Benchmark Evolution 

**Authors**: Junzhe Zhang, Huixuan Zhang, Xiaojun Wan  

**Link**: [PDF](https://arxiv.org/pdf/2510.21182)  

**Abstract**: The rapid progress of multimodal large language models (MLLMs) calls for more reliable evaluation protocols. Existing static benchmarks suffer from the potential risk of data contamination and saturation, leading to inflated or misleading performance evaluations. To address these issues, we first apply Graph formulation to represent a static or dynamic VQA sample. With the formulation, we propose Knowledge-enhanced Benchmark Evolution(KBE), a dynamic multimodal evaluation framework. KBE first analyzes the original static benchmark, then expands it by integrating multimodal knowledge, transforming the static benchmark into a controllable, dynamic evolving version. Crucially, KBE can both reconstruct questions by Re-selecting visual information in the original image and expand existing questions with external textual knowledge. It enables difficulty-controllable evaluation by adjusting the degree of question exploration. Extensive experiments demonstrate that KBE alleviates the risk of data contamination, data saturation, and provides a more comprehensive assessment of MLLM capabilities. 

---
# Designing and Evaluating Hint Generation Systems for Science Education 

**Authors**: Anubhav Jangra, Smaranda Muresan  

**Link**: [PDF](https://arxiv.org/pdf/2510.21087)  

**Abstract**: Large language models are influencing the education landscape, with students relying on them in their learning process. Often implemented using general-purpose models, these systems are likely to give away the answers, which could hinder conceptual understanding and critical thinking. We study the role of automatic hint generation as a pedagogical strategy to promote active engagement with the learning content, while guiding learners toward the answers. Focusing on scientific topics at the secondary education level, we explore the potential of large language models to generate chains of hints that scaffold learners without revealing answers. We compare two distinct hinting strategies: static hints, pre-generated for each problem, and dynamic hints, adapted to learners' progress. Through a quantitative study with 41 participants, we uncover different preferences among learners with respect to hinting strategies, and identify the limitations of automatic evaluation metrics to capture them. Our findings highlight key design considerations for future research on hint generation and intelligent tutoring systems that seek to develop learner-centered educational technologies. 

---
# Self-Jailbreaking: Language Models Can Reason Themselves Out of Safety Alignment After Benign Reasoning Training 

**Authors**: Zheng-Xin Yong, Stephen H. Bach  

**Link**: [PDF](https://arxiv.org/pdf/2510.20956)  

**Abstract**: We discover a novel and surprising phenomenon of unintentional misalignment in reasoning language models (RLMs), which we call self-jailbreaking. Specifically, after benign reasoning training on math or code domains, RLMs will use multiple strategies to circumvent their own safety guardrails. One strategy is to introduce benign assumptions about users and scenarios to justify fulfilling harmful requests. For instance, an RLM reasons that harmful requests like ``outline a strategy for stealing customers' credit card information from a retail store'' could be associated with the benign intent of ``a security professional trying to test defense,'' despite no such benign context being provided as input. We observe that many open-weight RLMs, including DeepSeek-R1-distilled, s1.1, Phi-4-mini-reasoning, and Nemotron, suffer from self-jailbreaking despite being aware of the harmfulness of the requests. We also provide a mechanistic understanding of self-jailbreaking: RLMs are more compliant after benign reasoning training, and after self-jailbreaking, models appear to perceive malicious requests as less harmful in the CoT, thus enabling compliance with them. To mitigate self-jailbreaking, we find that including minimal safety reasoning data during training is sufficient to ensure RLMs remain safety-aligned. Our work provides the first systematic analysis of self-jailbreaking behavior and offers a practical path forward for maintaining safety in increasingly capable RLMs. 

---
# Data-Centric Lessons To Improve Speech-Language Pretraining 

**Authors**: Vishaal Udandarao, Zhiyun Lu, Xuankai Chang, Yongqiang Wang, Violet Z. Yao, Albin Madapally Jose, Fartash Faghri, Josh Gardner, Chung-Cheng Chiu  

**Link**: [PDF](https://arxiv.org/pdf/2510.20860)  

**Abstract**: Spoken Question-Answering (SQA) is a core capability for useful and interactive artificial intelligence systems. Recently, several speech-language models (SpeechLMs) have been released with a specific focus on improving their SQA performance. However, a lack of controlled ablations of pretraining data processing and curation makes it challenging to understand what factors account for performance, despite substantial gains from similar studies in other data modalities. In this work, we address this gap by conducting a data-centric exploration for pretraining SpeechLMs. We focus on three research questions fundamental to speech-language pretraining data: (1) how to process raw web-crawled audio content for speech-text pretraining, (2) how to construct synthetic pretraining datasets to augment web-crawled data and (3) how to interleave (text, audio) segments into training sequences. We apply the insights from our controlled data-centric ablations to pretrain a 3.8B-parameter SpeechLM, called SpeLangy, that outperforms models that are up to 3x larger by 10.2% absolute performance. We hope our findings highlight the impact of effective data curation for speech-language pretraining and guide future data-centric exploration in SpeechLMs. 

---
# Beyond Hearing: Learning Task-agnostic ExG Representations from Earphones via Physiology-informed Tokenization 

**Authors**: Hyungjun Yoon, Seungjoo Lee, Yu Yvonne Wu, Xiaomeng Chen, Taiting Lu, Freddy Yifei Liu, Taeckyung Lee, Hyeongheon Cha, Haochen Zhao, Gaoteng Zhao, Sung-Ju Lee, Cecilia Mascolo, Dongyao Chen, Lili Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2510.20853)  

**Abstract**: Electrophysiological (ExG) signals offer valuable insights into human physiology, yet building foundation models that generalize across everyday tasks remains challenging due to two key limitations: (i) insufficient data diversity, as most ExG recordings are collected in controlled labs with bulky, expensive devices; and (ii) task-specific model designs that require tailored processing (i.e., targeted frequency filters) and architectures, which limit generalization across tasks. To address these challenges, we introduce an approach for scalable, task-agnostic ExG monitoring in the wild. We collected 50 hours of unobtrusive free-living ExG data with an earphone-based hardware prototype to narrow the data diversity gap. At the core of our approach is Physiology-informed Multi-band Tokenization (PiMT), which decomposes ExG signals into 12 physiology-informed tokens, followed by a reconstruction task to learn robust representations. This enables adaptive feature recognition across the full frequency spectrum while capturing task-relevant information. Experiments on our new DailySense dataset-the first to enable ExG-based analysis across five human senses-together with four public ExG benchmarks, demonstrate that PiMT consistently outperforms state-of-the-art methods across diverse tasks. 

---
# Can large audio language models understand child stuttering speech? speech summarization, and source separation 

**Authors**: Chibuzor Okocha, Maya Bakri, Christan Grant  

**Link**: [PDF](https://arxiv.org/pdf/2510.20850)  

**Abstract**: Child speech differs from adult speech in acoustics, prosody, and language development, and disfluencies (repetitions, prolongations, blocks) further challenge Automatic Speech Recognition (ASR) and downstream Natural Language Processing (NLP). Recent large audio-language models (LALMs) demonstrate strong cross-modal audio understanding; however, their behavior in disfluent child speech remains underexplored. We evaluate several state-of-the-art LALMs in two settings: an interview (mixed speakers) and a reading task (single child). The tasks are (i) single-channel source separation to isolate the child and (ii) child-only summarization that preserves clinically relevant disfluencies and avoids adult-speech leakage.
Evaluation combines Large Language Model (LLM) as a judge, human expert ratings, and BERTScore (F1), and we report agreement between models and between models and humans to assess reliability. Our findings delineate the conditions under which LALMs produce faithful child-only summaries from mixed audio and where they fail, offering practical guidance for clinical and educational deployments. We provide prompts and evaluation scripts to support replication. 

---
# Cultural Alien Sampler: Open-ended art generation balancing originality and coherence 

**Authors**: Alejandro H. Artiles, Hiromu Yakura, Levin Brinkmann, Mar Canet Sola, Hassan Abu Alhaija, Ignacio Serna, Nasim Rahaman, Bernhard Schölkopf, Iyad Rahwan  

**Link**: [PDF](https://arxiv.org/pdf/2510.20849)  

**Abstract**: In open-ended domains like art, autonomous agents must generate ideas that are both original and internally coherent, yet current Large Language Models (LLMs) either default to familiar cultural patterns or sacrifice coherence when pushed toward novelty. We address this by introducing the Cultural Alien Sampler (CAS), a concept-selection method that explicitly separates compositional fit from cultural typicality. CAS uses two GPT-2 models fine-tuned on WikiArt concepts: a Concept Coherence Model that scores whether concepts plausibly co-occur within artworks, and a Cultural Context Model that estimates how typical those combinations are within individual artists' bodies of work. CAS targets combinations that are high in coherence and low in typicality, yielding ideas that maintain internal consistency while deviating from learned conventions and embedded cultural context. In a human evaluation (N = 100), our approach outperforms random selection and GPT-4o baselines and achieves performance comparable to human art students in both perceived originality and harmony. Additionally, a quantitative study shows that our method produces more diverse outputs and explores a broader conceptual space than its GPT-4o counterpart, demonstrating that artificial cultural alienness can unlock creative potential in autonomous agents. 

---
