# Answer Matching Outperforms Multiple Choice for Language Model Evaluation 

**Authors**: Nikhil Chandak, Shashwat Goel, Ameya Prabhu, Moritz Hardt, Jonas Geiping  

**Link**: [PDF](https://arxiv.org/pdf/2507.02856)  

**Abstract**: Multiple choice benchmarks have long been the workhorse of language model evaluation because grading multiple choice is objective and easy to automate. However, we show multiple choice questions from popular benchmarks can often be answered without even seeing the question. These shortcuts arise from a fundamental limitation of discriminative evaluation not shared by evaluations of the model's free-form, generative answers. Until recently, there appeared to be no viable, scalable alternative to multiple choice--but, we show that this has changed. We consider generative evaluation via what we call answer matching: Give the candidate model the question without the options, have it generate a free-form response, then use a modern language model with the reference answer to determine if the response matches the reference. To compare the validity of different evaluation strategies, we annotate MMLU-Pro and GPQA-Diamond to obtain human grading data, and measure the agreement of each evaluation approach. We find answer matching using recent models--even small ones--achieves near-perfect agreement, in the range of inter-annotator agreement. In contrast, both multiple choice evaluation and using LLM-as-a-judge without reference answers aligns poorly with human grading. Improving evaluations via answer matching is not merely a conceptual concern: the rankings of several models change significantly when evaluating their free-form responses with answer matching. In light of these findings, we discuss how to move the evaluation ecosystem from multiple choice to answer matching. 

---
# MOTIF: Modular Thinking via Reinforcement Fine-tuning in LLMs 

**Authors**: Purbesh Mitra, Sennur Ulukus  

**Link**: [PDF](https://arxiv.org/pdf/2507.02851)  

**Abstract**: Recent advancements in the reasoning capabilities of large language models (LLMs) show that employing group relative policy optimization (GRPO) algorithm for reinforcement learning (RL) training allows the models to use more thinking/reasoning tokens for generating better responses. However, LLMs can generate only a finite amount of tokens while maintaining attention to the previously generated tokens. This limit, also known as the context size of an LLM, is a bottleneck in LLM reasoning with arbitrarily large number of tokens. To think beyond the limit of context size, an LLM must employ a modular thinking strategy to reason over multiple rounds. In this work, we propose $\textbf{MOTIF: Modular Thinking via Reinforcement Finetuning}$ -- an RL training method for generating thinking tokens in multiple rounds, effectively allowing the model to think with additional context size. We trained the open-source model Qwen2.5-3B-Instruct on GSM8K dataset via parameter efficient fine-tuning and tested its accuracy on MATH500 and AIME2024 benchmarks. Our experiments show 3.8\% and 3.3\% improvements over vanilla GRPO based training in the respective benchmarks. Furthermore, this improvement was achieved with only 15\% of samples, thus demonstrating sample efficiency of MOTIF. Our code and models are available at this https URL and this https URL, respectively. 

---
# LLM Hypnosis: Exploiting User Feedback for Unauthorized Knowledge Injection to All Users 

**Authors**: Almog Hilel, Idan Shenfeld, Leshem Choshen, Jacob Andreas  

**Link**: [PDF](https://arxiv.org/pdf/2507.02850)  

**Abstract**: We describe a vulnerability in language models (LMs) trained with user feedback, whereby a single user can persistently alter LM knowledge and behavior given only the ability to provide prompts and upvote / downvote feedback on LM outputs. To implement the attack, the attacker prompts the LM to stochastically output either a "poisoned" or benign response, then upvotes the poisoned response or downvotes the benign one. When feedback signals are used in a subsequent preference tuning behavior, LMs exhibit increased probability of producing poisoned responses even in contexts without malicious prompts. We show that this attack can be used to (1) insert factual knowledge the model did not previously possess, (2) modify code generation patterns in ways that introduce exploitable security flaws, and (3) inject fake financial news. Our finding both identifies a new qualitative feature of language model preference tuning (showing that it even highly restricted forms of preference data can be used to exert fine-grained control over behavior), and a new attack mechanism for LMs trained with user feedback (extending work on pretraining-time data poisoning and deployment-time prompt injection). 

---
# Generalizing Verifiable Instruction Following 

**Authors**: Valentina Pyatkin, Saumya Malik, Victoria Graf, Hamish Ivison, Shengyi Huang, Pradeep Dasigi, Nathan Lambert, Hannaneh Hajishirzi  

**Link**: [PDF](https://arxiv.org/pdf/2507.02833)  

**Abstract**: A crucial factor for successful human and AI interaction is the ability of language models or chatbots to follow human instructions precisely. A common feature of instructions are output constraints like ``only answer with yes or no" or ``mention the word `abrakadabra' at least 3 times" that the user adds to craft a more useful answer. Even today's strongest models struggle with fulfilling such constraints. We find that most models strongly overfit on a small set of verifiable constraints from the benchmarks that test these abilities, a skill called precise instruction following, and are not able to generalize well to unseen output constraints. We introduce a new benchmark, IFBench, to evaluate precise instruction following generalization on 58 new, diverse, and challenging verifiable out-of-domain constraints. In addition, we perform an extensive analysis of how and on what data models can be trained to improve precise instruction following generalization. Specifically, we carefully design constraint verification modules and show that reinforcement learning with verifiable rewards (RLVR) significantly improves instruction following. In addition to IFBench, we release 29 additional new hand-annotated training constraints and verification functions, RLVR training prompts, and code. 

---
# SynapseRoute: An Auto-Route Switching Framework on Dual-State Large Language Model 

**Authors**: Wencheng Zhang, Shiqin Qiao, Lingjie Luo, Yinfeng Li, Chuanyang Zheng, Qian Xu, Meng Li, Yong Gui, Yijun He, Jianing Qiu, Jindong Hong, Jiankai Sun  

**Link**: [PDF](https://arxiv.org/pdf/2507.02822)  

**Abstract**: With the widespread adoption of large language models (LLMs) in practical applications, selecting an appropriate model requires balancing not only performance but also operational cost. The emergence of reasoning-capable models has further widened the cost gap between "thinking" (high reasoning) and "non-thinking" (fast, low-cost) modes. In this work, we reveal that approximately 58% of medical questions can be accurately answered by the non-thinking mode alone, without requiring the high-cost reasoning process. This highlights a clear dichotomy in problem complexity and suggests that dynamically routing queries to the appropriate mode based on complexity could optimize accuracy, cost-efficiency, and overall user experience. Based on this, we further propose SynapseRoute, a machine learning-based dynamic routing framework that intelligently assigns input queries to either thinking or non-thinking modes. Experimental results on several medical datasets demonstrate that SynapseRoute not only improves overall accuracy (0.8390 vs. 0.8272) compared to the thinking mode alone but also reduces inference time by 36.8% and token consumption by 39.66%. Importantly, qualitative analysis indicates that over-reasoning on simpler queries can lead to unnecessary delays and even decreased accuracy, a pitfall avoided by our adaptive routing. Finally, this work further introduces the Accuracy-Inference-Token (AIT) index to comprehensively evaluate the trade-offs among accuracy, latency, and token cost. 

---
# Multimodal Mathematical Reasoning with Diverse Solving Perspective 

**Authors**: Wenhao Shi, Zhiqiang Hu, Yi Bin, Yang Yang, See-Kiong Ng, Heng Tao Shen  

**Link**: [PDF](https://arxiv.org/pdf/2507.02804)  

**Abstract**: Recent progress in large-scale reinforcement learning (RL) has notably enhanced the reasoning capabilities of large language models (LLMs), especially in mathematical domains. However, current multimodal LLMs (MLLMs) for mathematical reasoning often rely on one-to-one image-text pairs and single-solution supervision, overlooking the diversity of valid reasoning perspectives and internal reflections. In this work, we introduce MathV-DP, a novel dataset that captures multiple diverse solution trajectories for each image-question pair, fostering richer reasoning supervision. We further propose Qwen-VL-DP, a model built upon Qwen-VL, fine-tuned with supervised learning and enhanced via group relative policy optimization (GRPO), a rule-based RL approach that integrates correctness discrimination and diversity-aware reward functions. Our method emphasizes learning from varied reasoning perspectives and distinguishing between correct yet distinct solutions. Extensive experiments on the MathVista's minitest and Math-V benchmarks demonstrate that Qwen-VL-DP significantly outperforms prior base MLLMs in both accuracy and generative diversity, highlighting the importance of incorporating diverse perspectives and reflective reasoning in multimodal mathematical reasoning. 

---
# Is Reasoning All You Need? Probing Bias in the Age of Reasoning Language Models 

**Authors**: Riccardo Cantini, Nicola Gabriele, Alessio Orsino, Domenico Talia  

**Link**: [PDF](https://arxiv.org/pdf/2507.02799)  

**Abstract**: Reasoning Language Models (RLMs) have gained traction for their ability to perform complex, multi-step reasoning tasks through mechanisms such as Chain-of-Thought (CoT) prompting or fine-tuned reasoning traces. While these capabilities promise improved reliability, their impact on robustness to social biases remains unclear. In this work, we leverage the CLEAR-Bias benchmark, originally designed for Large Language Models (LLMs), to investigate the adversarial robustness of RLMs to bias elicitation. We systematically evaluate state-of-the-art RLMs across diverse sociocultural dimensions, using an LLM-as-a-judge approach for automated safety scoring and leveraging jailbreak techniques to assess the strength of built-in safety mechanisms. Our evaluation addresses three key questions: (i) how the introduction of reasoning capabilities affects model fairness and robustness; (ii) whether models fine-tuned for reasoning exhibit greater safety than those relying on CoT prompting at inference time; and (iii) how the success rate of jailbreak attacks targeting bias elicitation varies with the reasoning mechanisms employed. Our findings reveal a nuanced relationship between reasoning capabilities and bias safety. Surprisingly, models with explicit reasoning, whether via CoT prompting or fine-tuned reasoning traces, are generally more vulnerable to bias elicitation than base models without such mechanisms, suggesting reasoning may unintentionally open new pathways for stereotype reinforcement. Reasoning-enabled models appear somewhat safer than those relying on CoT prompting, which are particularly prone to contextual reframing attacks through storytelling prompts, fictional personas, or reward-shaped instructions. These results challenge the assumption that reasoning inherently improves robustness and underscore the need for more bias-aware approaches to reasoning design. 

---
# Self-Correction Bench: Revealing and Addressing the Self-Correction Blind Spot in LLMs 

**Authors**: Ken Tsui  

**Link**: [PDF](https://arxiv.org/pdf/2507.02778)  

**Abstract**: Although large language models (LLMs) have become transformative, they still make mistakes and can explore unproductive reasoning paths. Self-correction is an important capability for a trustworthy LLM, particularly an autoregressive LLM. While LLMs can identify error in user input, they exhibit a systematic 'Self-Correction Blind Spot' - failing to correct identical error in their own outputs. To systematically study this phenomenon, we introduce Self-Correction Bench, a systematic framework to measure this phenomenon through controlled error injection at three complexity levels. Testing 14 models, we find an average 64.5% blind spot rate. We find multiple evidences that this limitation relates to training data composition: human training demonstrations predominantly show error-free responses rather than error-correction sequences, unlike RL-trained models that learn error correction through outcome feedback. Remarkably, simply appending "Wait" reduces blind spots by 89.3%, suggesting that the capability exists but requires activation. Our work highlights a critical limitation in current LLMs and offers potential avenues for improving their reliability and trustworthiness. 

---
# Measurement of the Granularity of Vowel Production Space By Just Producible Different (JPD) Limens 

**Authors**: Peter Viechnicki  

**Link**: [PDF](https://arxiv.org/pdf/2507.02744)  

**Abstract**: A body of work over the past several decades has demonstrated that the complex and coordinated articulatory movements of human vowel production are governed (at least in part)by control mechanisms whose targets are regions of auditory space. Within the target region control at the sub-phonemic level has also been demonstrated. But the degree of accuracy of that control is unknown. The current work investigates this question by asking how far apart must two vowel stimuli lie in auditory space in order to yield reliably different imitations? This distance is termed 'Just Producible Difference' (JPD). The current study uses a vowel mimicry paradigm to derive the first measurement of JPD among two sets of English speakers during front vowel production. JPD is estimated at between 14 and 51 mels in F1 X F2 space. This finding has implications for episodic theories of speech production. It also clarifies the possible structures of human vowel systems, by setting a theoretical lower bound for how close two vowel phonemes may be in a speaker's formant space, and hence a psychophysical explanation of observed trends in number and patterns of possible vowel phonemes. 

---
# Can LLMs Identify Critical Limitations within Scientific Research? A Systematic Evaluation on AI Research Papers 

**Authors**: Zhijian Xu, Yilun Zhao, Manasi Patwardhan, Lovekesh Vig, Arman Cohan  

**Link**: [PDF](https://arxiv.org/pdf/2507.02694)  

**Abstract**: Peer review is fundamental to scientific research, but the growing volume of publications has intensified the challenges of this expertise-intensive process. While LLMs show promise in various scientific tasks, their potential to assist with peer review, particularly in identifying paper limitations, remains understudied. We first present a comprehensive taxonomy of limitation types in scientific research, with a focus on AI. Guided by this taxonomy, for studying limitations, we present LimitGen, the first comprehensive benchmark for evaluating LLMs' capability to support early-stage feedback and complement human peer review. Our benchmark consists of two subsets: LimitGen-Syn, a synthetic dataset carefully created through controlled perturbations of high-quality papers, and LimitGen-Human, a collection of real human-written limitations. To improve the ability of LLM systems to identify limitations, we augment them with literature retrieval, which is essential for grounding identifying limitations in prior scientific findings. Our approach enhances the capabilities of LLM systems to generate limitations in research papers, enabling them to provide more concrete and constructive feedback. 

---
# Exploring Gender Bias Beyond Occupational Titles 

**Authors**: Ahmed Sabir, Rajesh Sharama  

**Link**: [PDF](https://arxiv.org/pdf/2507.02679)  

**Abstract**: In this work, we investigate the correlation between gender and contextual biases, focusing on elements such as action verbs, object nouns, and particularly on occupations. We introduce a novel dataset, GenderLexicon, and a framework that can estimate contextual bias and its related gender bias. Our model can interpret the bias with a score and thus improve the explainability of gender bias. Also, our findings confirm the existence of gender biases beyond occupational stereotypes. To validate our approach and demonstrate its effectiveness, we conduct evaluations on five diverse datasets, including a Japanese dataset. 

---
# MPF: Aligning and Debiasing Language Models post Deployment via Multi Perspective Fusion 

**Authors**: Xin Guan, PeiHsin Lin, Zekun Wu, Ze Wang, Ruibo Zhang, Emre Kazim, Adriano Koshiyama  

**Link**: [PDF](https://arxiv.org/pdf/2507.02595)  

**Abstract**: Multiperspective Fusion (MPF) is a novel posttraining alignment framework for large language models (LLMs) developed in response to the growing need for easy bias mitigation. Built on top of the SAGED pipeline, an automated system for constructing bias benchmarks and extracting interpretable baseline distributions, MPF leverages multiperspective generations to expose and align biases in LLM outputs with nuanced, humanlike baselines. By decomposing baseline, such as sentiment distributions from HR professionals, into interpretable perspective components, MPF guides generation through sampling and balancing of responses, weighted by the probabilities obtained in the decomposition. Empirically, we demonstrate its ability to align LLM sentiment distributions with both counterfactual baselines (absolute equality) and the HR baseline (biased for Top Univeristy), resulting in small KL divergence, reduction of calibration error and generalization to unseen questions. This shows that MPF offers a scalable and interpretable method for alignment and bias mitigation, compatible with deployed LLMs and requiring no extensive prompt engineering or finetuning. 

---
# Revisiting Active Learning under (Human) Label Variation 

**Authors**: Cornelia Gruber, Helen Alber, Bernd Bischl, Göran Kauermann, Barbara Plank, Matthias Aßenmacher  

**Link**: [PDF](https://arxiv.org/pdf/2507.02593)  

**Abstract**: Access to high-quality labeled data remains a limiting factor in applied supervised learning. While label variation (LV), i.e., differing labels for the same instance, is common, especially in natural language processing, annotation frameworks often still rest on the assumption of a single ground truth. This overlooks human label variation (HLV), the occurrence of plausible differences in annotations, as an informative signal. Similarly, active learning (AL), a popular approach to optimizing the use of limited annotation budgets in training ML models, often relies on at least one of several simplifying assumptions, which rarely hold in practice when acknowledging HLV. In this paper, we examine foundational assumptions about truth and label nature, highlighting the need to decompose observed LV into signal (e.g., HLV) and noise (e.g., annotation error). We survey how the AL and (H)LV communities have addressed -- or neglected -- these distinctions and propose a conceptual framework for incorporating HLV throughout the AL loop, including instance selection, annotator choice, and label representation. We further discuss the integration of large language models (LLM) as annotators. Our work aims to lay a conceptual foundation for HLV-aware active learning, better reflecting the complexities of real-world annotation. 

---
# WebSailor: Navigating Super-human Reasoning for Web Agent 

**Authors**: Kuan Li, Zhongwang Zhang, Huifeng Yin, Liwen Zhang, Litu Ou, Jialong Wu, Wenbiao Yin, Baixuan Li, Zhengwei Tao, Xinyu Wang, Weizhou Shen, Junkai Zhang, Dingchu Zhang, Xixi Wu, Yong Jiang, Ming Yan, Pengjun Xie, Fei Huang, Jingren Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2507.02592)  

**Abstract**: Transcending human cognitive limitations represents a critical frontier in LLM training. Proprietary agentic systems like DeepResearch have demonstrated superhuman capabilities on extremely complex information-seeking benchmarks such as BrowseComp, a feat previously unattainable. We posit that their success hinges on a sophisticated reasoning pattern absent in open-source models: the ability to systematically reduce extreme uncertainty when navigating vast information landscapes. Based on this insight, we introduce WebSailor, a complete post-training methodology designed to instill this crucial capability. Our approach involves generating novel, high-uncertainty tasks through structured sampling and information obfuscation, RFT cold start, and an efficient agentic RL training algorithm, Duplicating Sampling Policy Optimization (DUPO). With this integrated pipeline, WebSailor significantly outperforms all opensource agents in complex information-seeking tasks, matching proprietary agents' performance and closing the capability gap. 

---
# IndianBailJudgments-1200: A Multi-Attribute Dataset for Legal NLP on Indian Bail Orders 

**Authors**: Sneha Deshmukh, Prathmesh Kamble  

**Link**: [PDF](https://arxiv.org/pdf/2507.02506)  

**Abstract**: Legal NLP remains underdeveloped in regions like India due to the scarcity of structured datasets. We introduce IndianBailJudgments-1200, a new benchmark dataset comprising 1200 Indian court judgments on bail decisions, annotated across 20+ attributes including bail outcome, IPC sections, crime type, and legal reasoning. Annotations were generated using a prompt-engineered GPT-4o pipeline and verified for consistency. This resource supports a wide range of legal NLP tasks such as outcome prediction, summarization, and fairness analysis, and is the first publicly available dataset focused specifically on Indian bail jurisprudence. 

---
# A Cookbook for Community-driven Data Collection of Impaired Speech in LowResource Languages 

**Authors**: Sumaya Ahmed Salihs, Isaac Wiafe, Jamal-Deen Abdulai, Elikem Doe Atsakpo, Gifty Ayoka, Richard Cave, Akon Obu Ekpezu, Catherine Holloway, Katrin Tomanek, Fiifi Baffoe Payin Winful  

**Link**: [PDF](https://arxiv.org/pdf/2507.02428)  

**Abstract**: This study presents an approach for collecting speech samples to build Automatic Speech Recognition (ASR) models for impaired speech, particularly, low-resource languages. It aims to democratize ASR technology and data collection by developing a "cookbook" of best practices and training for community-driven data collection and ASR model building. As a proof-of-concept, this study curated the first open-source dataset of impaired speech in Akan: a widely spoken indigenous language in Ghana. The study involved participants from diverse backgrounds with speech impairments. The resulting dataset, along with the cookbook and open-source tools, are publicly available to enable researchers and practitioners to create inclusive ASR technologies tailored to the unique needs of speech impaired individuals. In addition, this study presents the initial results of fine-tuning open-source ASR models to better recognize impaired speech in Akan. 

---
# Benchmarking Akan ASR Models Across Domain-Specific Datasets: A Comparative Evaluation of Performance, Scalability, and Adaptability 

**Authors**: Mark Atta Mensah, Isaac Wiafe, Akon Ekpezu, Justice Kwame Appati, Jamal-Deen Abdulai, Akosua Nyarkoa Wiafe-Akenten, Frank Ernest Yeboah, Gifty Odame  

**Link**: [PDF](https://arxiv.org/pdf/2507.02407)  

**Abstract**: Most existing automatic speech recognition (ASR) research evaluate models using in-domain datasets. However, they seldom evaluate how they generalize across diverse speech contexts. This study addresses this gap by benchmarking seven Akan ASR models built on transformer architectures, such as Whisper and Wav2Vec2, using four Akan speech corpora to determine their performance. These datasets encompass various domains, including culturally relevant image descriptions, informal conversations, biblical scripture readings, and spontaneous financial dialogues. A comparison of the word error rate and character error rate highlighted domain dependency, with models performing optimally only within their training domains while showing marked accuracy degradation in mismatched scenarios. This study also identified distinct error behaviors between the Whisper and Wav2Vec2 architectures. Whereas fine-tuned Whisper Akan models led to more fluent but potentially misleading transcription errors, Wav2Vec2 produced more obvious yet less interpretable outputs when encountering unfamiliar inputs. This trade-off between readability and transparency in ASR errors should be considered when selecting architectures for low-resource language (LRL) applications. These findings highlight the need for targeted domain adaptation techniques, adaptive routing strategies, and multilingual training frameworks for Akan and other LRLs. 

---
# Efficient Code LLM Training via Distribution-Consistent and Diversity-Aware Data Selection 

**Authors**: Weijie Lyu, Sheng-Jun Huang, Xuan Xia  

**Link**: [PDF](https://arxiv.org/pdf/2507.02378)  

**Abstract**: Recent advancements in large language models (LLMs) have significantly improved code generation and program comprehension, accelerating the evolution of software engineering. Current methods primarily enhance model performance by leveraging vast amounts of data, focusing on data quantity while often overlooking data quality, thereby reducing training efficiency. To address this, we introduce an approach that utilizes a parametric model for code data selection, aimed at improving both training efficiency and model performance. Our method optimizes the parametric model to ensure distribution consistency and diversity within the selected subset, guaranteeing high-quality data. Experimental results demonstrate that using only 10K samples, our method achieves gains of 2.4% (HumanEval) and 2.3% (MBPP) over 92K full-sampled baseline, outperforming other sampling approaches in both performance and efficiency. This underscores that our method effectively boosts model performance while significantly reducing computational costs. 

---
# QFFN-BERT: An Empirical Study of Depth, Performance, and Data Efficiency in Hybrid Quantum-Classical Transformers 

**Authors**: Pilsung Kang  

**Link**: [PDF](https://arxiv.org/pdf/2507.02364)  

**Abstract**: Parameterized quantum circuits (PQCs) have recently emerged as promising components for enhancing the expressibility of neural architectures. In this work, we introduce QFFN-BERT, a hybrid quantum-classical transformer where the feedforward network (FFN) modules of a compact BERT variant are replaced by PQC-based layers. This design is motivated by the dominant parameter contribution of FFNs, which account for approximately two-thirds of the parameters within standard Transformer encoder blocks. While prior studies have primarily integrated PQCs into self-attention modules, our work focuses on the FFN and systematically investigates the trade-offs between PQC depth, expressibility, and trainability. Our final PQC architecture incorporates a residual connection, both $R_Y$ and $R_Z$ rotations, and an alternating entanglement strategy to ensure stable training and high expressibility. Our experiments, conducted on a classical simulator, on the SST-2 and DBpedia benchmarks demonstrate two key findings. First, a carefully configured QFFN-BERT achieves up to 102.0% of the baseline accuracy, surpassing its classical counterpart in a full-data setting while reducing FFN-specific parameters by over 99%. Second, our model exhibits a consistent and competitive edge in few-shot learning scenarios, confirming its potential for superior data efficiency. These results, supported by an ablation study on a non-optimized PQC that failed to learn, confirm that PQCs can serve as powerful and parameter-efficient alternatives to classical FFNs when co-designed with foundational deep learning principles. 

---
# Coling-UniA at SciVQA 2025: Few-Shot Example Retrieval and Confidence-Informed Ensembling for Multimodal Large Language Models 

**Authors**: Christian Jaumann, Annemarie Friedrich, Rainer Lienhart  

**Link**: [PDF](https://arxiv.org/pdf/2507.02357)  

**Abstract**: This paper describes our system for the SciVQA 2025 Shared Task on Scientific Visual Question Answering. Our system employs an ensemble of two Multimodal Large Language Models and various few-shot example retrieval strategies. The model and few-shot setting are selected based on the figure and question type. We also select answers based on the models' confidence levels. On the blind test data, our system ranks third out of seven with an average F1 score of 85.12 across ROUGE-1, ROUGE-L, and BERTS. Our code is publicly available. 

---
# DoMIX: An Efficient Framework for Exploiting Domain Knowledge in Fine-Tuning 

**Authors**: Dohoon Kim, Donghun Kang, Taesup Moon  

**Link**: [PDF](https://arxiv.org/pdf/2507.02302)  

**Abstract**: Domain-Adaptive Pre-training (DAP) has recently gained attention for its effectiveness in fine-tuning pre-trained models. Building on this, continual DAP has been explored to develop pre-trained models capable of incrementally incorporating different domain datasets. However, existing continual DAP methods face several limitations: (1) high computational cost and GPU memory usage during training; (2) sensitivity to incremental data order; and (3) providing a single, generalized model for all end tasks, which contradicts the essence of DAP. In this paper, we propose DoMIX, a novel approach that addresses these challenges by leveraging LoRA modules, a representative parameter-efficient fine-tuning (PEFT) method. Our approach enables efficient and parallel domain-adaptive pre-training that is robust to domain order and effectively utilizes accumulated knowledge to provide tailored pre-trained models for specific tasks. We also demonstrate that our method can be extended beyond the DAP setting to standard LLM fine-tuning scenarios. Code is available at this https URL. 

---
# MemAgent: Reshaping Long-Context LLM with Multi-Conv RL-based Memory Agent 

**Authors**: Hongli Yu, Tinghong Chen, Jiangtao Feng, Jiangjie Chen, Weinan Dai, Qiying Yu, Ya-Qin Zhang, Wei-Ying Ma, Jingjing Liu, Mingxuan Wang, Hao Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2507.02259)  

**Abstract**: Despite improvements by length extrapolation, efficient attention and memory modules, handling infinitely long documents with linear complexity without performance degradation during extrapolation remains the ultimate challenge in long-text processing. We directly optimize for long-text tasks in an end-to-end fashion and introduce a novel agent workflow, MemAgent, which reads text in segments and updates the memory using an overwrite strategy. We extend the DAPO algorithm to facilitate training via independent-context multi-conversation generation. MemAgent has demonstrated superb long-context capabilities, being able to extrapolate from an 8K context trained on 32K text to a 3.5M QA task with performance loss < 5% and achieves 95%+ in 512K RULER test. 

---
# GDC Cohort Copilot: An AI Copilot for Curating Cohorts from the Genomic Data Commons 

**Authors**: Steven Song, Anirudh Subramanyam, Zhenyu Zhang, Aarti Venkat, Robert L. Grossman  

**Link**: [PDF](https://arxiv.org/pdf/2507.02221)  

**Abstract**: Motivation: The Genomic Data Commons (GDC) provides access to high quality, harmonized cancer genomics data through a unified curation and analysis platform centered around patient cohorts. While GDC users can interactively create complex cohorts through the graphical Cohort Builder, users (especially new ones) may struggle to find specific cohort descriptors across hundreds of possible fields and properties. However, users may be better able to describe their desired cohort in free-text natural language.
Results: We introduce GDC Cohort Copilot, an open-source copilot tool for curating cohorts from the GDC. GDC Cohort Copilot automatically generates the GDC cohort filter corresponding to a user-input natural language description of their desired cohort, before exporting the cohort back to the GDC for further analysis. An interactive user interface allows users to further refine the generated cohort. We develop and evaluate multiple large language models (LLMs) for GDC Cohort Copilot and demonstrate that our locally-served, open-source GDC Cohort LLM achieves better results than GPT-4o prompting in generating GDC cohorts.
Availability and implementation: The standalone docker image for GDC Cohort Copilot is available at this https URL. Source code is available at this https URL. GDC Cohort LLM weights are available at this https URL. 

---
# Latent Chain-of-Thought? Decoding the Depth-Recurrent Transformer 

**Authors**: Wenquan Lu, Yuechuan Yang, Kyle Lee, Yanshu Li, Enqi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.02199)  

**Abstract**: Chain-of-thought (CoT) reasoning has enabled transformer-based language models to excel at complex mathematics and multi-step planning. However, in standard decoder-only architectures, these reasoning steps are externalized in natural language, improving interpretability at the cost of efficiency. To capture reasoning that is not easily represented in words, many works have explored recurrent architectures that aim to internalize reasoning in latent space, potentially supporting latent CoT. In this paper, we investigate whether such reasoning structures emerge in Huginn-3.5B, a depth-recurrent Transformer that reuses layers at inference time without increasing parameter count. We examine the model's internal behavior on arithmetic tasks using a suite of probing techniques including the Logit Lens and Coda Lens. Our findings reveal limited evidence of interpretable latent CoT by tracking rank trajectories of final and intermediate result tokens. Furthermore, we uncover significant probing inconsistencies across recurrent blocks, where the interpretability of hidden states depends heavily on both the layer index and the decoding method. Finally, we empirically show that increasing recurrence depth yields only marginal gains and falls well short of models that explicitly externalize reasoning steps. The code is available at this https URL. 

---
# Reasoning or Not? A Comprehensive Evaluation of Reasoning LLMs for Dialogue Summarization 

**Authors**: Keyan Jin, Yapeng Wang, Leonel Santos, Tao Fang, Xu Yang, Sio Kei Im, Hugo Gonçalo Oliveira  

**Link**: [PDF](https://arxiv.org/pdf/2507.02145)  

**Abstract**: Dialogue summarization is a challenging task with significant practical value in customer service, meeting analysis, and conversational AI. Although large language models (LLMs) have achieved substantial progress in summarization tasks, the performance of step-by-step reasoning architectures-specifically Long Chain-of-Thought (CoT) implementations such as OpenAI-o1 and DeepSeek-R1-remains unexplored for dialogue scenarios requiring concurrent abstraction and conciseness. In this work, we present the first comprehensive and systematic evaluation of state-of-the-art reasoning LLMs and non-reasoning LLMs across three major paradigms-generic, role-oriented, and query-oriented dialogue summarization. Our study spans diverse languages, domains, and summary lengths, leveraging strong benchmarks (SAMSum, DialogSum, CSDS, and QMSum) and advanced evaluation protocols that include both LLM-based automatic metrics and human-inspired criteria. Contrary to trends in other reasoning-intensive tasks, our findings show that explicit stepwise reasoning does not consistently improve dialogue summarization quality. Instead, reasoning LLMs are often prone to verbosity, factual inconsistencies, and less concise summaries compared to their non-reasoning counterparts. Through scenario-specific analyses and detailed case studies, we further identify when and why explicit reasoning may fail to benefit-or even hinder-summarization in complex dialogue contexts. Our work provides new insights into the limitations of current reasoning LLMs and highlights the need for targeted modeling and evaluation strategies for real-world dialogue summarization. 

---
# McBE: A Multi-task Chinese Bias Evaluation Benchmark for Large Language Models 

**Authors**: Tian Lan, Xiangdong Su, Xu Liu, Ruirui Wang, Ke Chang, Jiang Li, Guanglai Gao  

**Link**: [PDF](https://arxiv.org/pdf/2507.02088)  

**Abstract**: As large language models (LLMs) are increasingly applied to various NLP tasks, their inherent biases are gradually disclosed. Therefore, measuring biases in LLMs is crucial to mitigate its ethical risks. However, most existing bias evaluation datasets focus on English and North American culture, and their bias categories are not fully applicable to other cultures. The datasets grounded in the Chinese language and culture are scarce. More importantly, these datasets usually only support single evaluation tasks and cannot evaluate the bias from multiple aspects in LLMs. To address these issues, we present a Multi-task Chinese Bias Evaluation Benchmark (McBE) that includes 4,077 bias evaluation instances, covering 12 single bias categories, 82 subcategories and introducing 5 evaluation tasks, providing extensive category coverage, content diversity, and measuring comprehensiveness. Additionally, we evaluate several popular LLMs from different series and with parameter sizes. In general, all these LLMs demonstrated varying degrees of bias. We conduct an in-depth analysis of results, offering novel insights into bias in LLMs. 

---
# Requirements Elicitation Follow-Up Question Generation 

**Authors**: Yuchen Shen, Anmol Singhal, Travis Breaux  

**Link**: [PDF](https://arxiv.org/pdf/2507.02858)  

**Abstract**: Interviews are a widely used technique in eliciting requirements to gather stakeholder needs, preferences, and expectations for a software system. Effective interviewing requires skilled interviewers to formulate appropriate interview questions in real time while facing multiple challenges, including lack of familiarity with the domain, excessive cognitive load, and information overload that hinders how humans process stakeholders' speech. Recently, large language models (LLMs) have exhibited state-of-the-art performance in multiple natural language processing tasks, including text summarization and entailment. To support interviewers, we investigate the application of GPT-4o to generate follow-up interview questions during requirements elicitation by building on a framework of common interviewer mistake types. In addition, we describe methods to generate questions based on interviewee speech. We report a controlled experiment to evaluate LLM-generated and human-authored questions with minimal guidance, and a second controlled experiment to evaluate the LLM-generated questions when generation is guided by interviewer mistake types. Our findings demonstrate that, for both experiments, the LLM-generated questions are no worse than the human-authored questions with respect to clarity, relevancy, and informativeness. In addition, LLM-generated questions outperform human-authored questions when guided by common mistakes types. This highlights the potential of using LLMs to help interviewers improve the quality and ease of requirements elicitation interviews in real time. 

---
# Legal Requirements Translation from Law 

**Authors**: Anmol Singhal, Travis Breaux  

**Link**: [PDF](https://arxiv.org/pdf/2507.02846)  

**Abstract**: Software systems must comply with legal regulations, which is a resource-intensive task, particularly for small organizations and startups lacking dedicated legal expertise. Extracting metadata from regulations to elicit legal requirements for software is a critical step to ensure compliance. However, it is a cumbersome task due to the length and complex nature of legal text. Although prior work has pursued automated methods for extracting structural and semantic metadata from legal text, key limitations remain: they do not consider the interplay and interrelationships among attributes associated with these metadata types, and they rely on manual labeling or heuristic-driven machine learning, which does not generalize well to new documents. In this paper, we introduce an approach based on textual entailment and in-context learning for automatically generating a canonical representation of legal text, encodable and executable as Python code. Our representation is instantiated from a manually designed Python class structure that serves as a domain-specific metamodel, capturing both structural and semantic legal metadata and their interrelationships. This design choice reduces the need for large, manually labeled datasets and enhances applicability to unseen legislation. We evaluate our approach on 13 U.S. state data breach notification laws, demonstrating that our generated representations pass approximately 89.4% of test cases and achieve a precision and recall of 82.2 and 88.7, respectively. 

---
# Visual Contextual Attack: Jailbreaking MLLMs with Image-Driven Context Injection 

**Authors**: Ziqi Miao, Yi Ding, Lijun Li, Jing Shao  

**Link**: [PDF](https://arxiv.org/pdf/2507.02844)  

**Abstract**: With the emergence of strong visual-language capabilities, multimodal large language models (MLLMs) have demonstrated tremendous potential for real-world applications. However, the security vulnerabilities exhibited by the visual modality pose significant challenges to deploying such models in open-world environments. Recent studies have successfully induced harmful responses from target MLLMs by encoding harmful textual semantics directly into visual inputs. However, in these approaches, the visual modality primarily serves as a trigger for unsafe behavior, often exhibiting semantic ambiguity and lacking grounding in realistic scenarios. In this work, we define a novel setting: visual-centric jailbreak, where visual information serves as a necessary component in constructing a complete and realistic jailbreak context. Building on this setting, we propose the VisCo (Visual Contextual) Attack. VisCo fabricates contextual dialogue using four distinct visual-focused strategies, dynamically generating auxiliary images when necessary to construct a visual-centric jailbreak scenario. To maximize attack effectiveness, it incorporates automatic toxicity obfuscation and semantic refinement to produce a final attack prompt that reliably triggers harmful responses from the target black-box MLLMs. Specifically, VisCo achieves a toxicity score of 4.78 and an Attack Success Rate (ASR) of 85% on MM-SafetyBench against GPT-4o, significantly outperforming the baseline, which performs a toxicity score of 2.48 and an ASR of 22.2%. The code is available at this https URL. 

---
# StepHint: Multi-level Stepwise Hints Enhance Reinforcement Learning to Reason 

**Authors**: Kaiyi Zhang, Ang Lv, Jinpeng Li, Yongbo Wang, Feng Wang, Haoyuan Hu, Rui Yan  

**Link**: [PDF](https://arxiv.org/pdf/2507.02841)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) is a promising approach for improving the complex reasoning abilities of large language models (LLMs). However, current RLVR methods face two significant challenges: the near-miss reward problem, where a small mistake can invalidate an otherwise correct reasoning process, greatly hindering training efficiency; and exploration stagnation, where models tend to focus on solutions within their ``comfort zone,'' lacking the motivation to explore potentially more effective alternatives. To address these challenges, we propose StepHint, a novel RLVR algorithm that utilizes multi-level stepwise hints to help models explore the solution space more effectively. StepHint generates valid reasoning chains from stronger models and partitions these chains into reasoning steps using our proposed adaptive partitioning method. The initial few steps are used as hints, and simultaneously, multiple-level hints (each comprising a different number of steps) are provided to the model. This approach directs the model's exploration toward a promising solution subspace while preserving its flexibility for independent exploration. By providing hints, StepHint mitigates the near-miss reward problem, thereby improving training efficiency. Additionally, the external reasoning pathways help the model develop better reasoning abilities, enabling it to move beyond its ``comfort zone'' and mitigate exploration stagnation. StepHint outperforms competitive RLVR enhancement methods across six mathematical benchmarks, while also demonstrating superior generalization and excelling over baselines on out-of-domain benchmarks. 

---
# ExPO: Unlocking Hard Reasoning with Self-Explanation-Guided Reinforcement Learning 

**Authors**: Ruiyang Zhou, Shuozhe Li, Amy Zhang, Liu Leqi  

**Link**: [PDF](https://arxiv.org/pdf/2507.02834)  

**Abstract**: Recent advances in large language models have been driven by reinforcement learning (RL)-style post-training, which improves reasoning by optimizing model outputs based on reward or preference signals. GRPO-style approaches implement this by using self-generated samples labeled by an outcome-based verifier. However, these methods depend heavily on the model's initial ability to produce positive samples. They primarily refine what the model already knows (distribution sharpening) rather than enabling the model to solve problems where it initially fails. This limitation is especially problematic in early-stage RL training and on challenging reasoning tasks, where positive samples are unlikely to be generated. To unlock reasoning ability in such settings, the model must explore new reasoning trajectories beyond its current output distribution. Such exploration requires access to sufficiently good positive samples to guide the learning. While expert demonstrations seem like a natural solution, we find that they are often ineffective in RL post-training. Instead, we identify two key properties of effective positive samples: they should (1) be likely under the current policy, and (2) increase the model's likelihood of predicting the correct answer. Based on these insights, we propose $\textbf{Self-Explanation Policy Optimization (ExPO)}$-a simple and modular framework that generates such samples by conditioning on the ground-truth answer. ExPO enables efficient exploration and guides the model to produce reasoning trajectories more aligned with its policy than expert-written CoTs, while ensuring higher quality than its own (incorrect) samples. Experiments show that ExPO improves both learning efficiency and final performance on reasoning benchmarks, surpassing expert-demonstration-based methods in challenging settings such as MATH level-5, where the model initially struggles the most. 

---
# From Long Videos to Engaging Clips: A Human-Inspired Video Editing Framework with Multimodal Narrative Understanding 

**Authors**: Xiangfeng Wang, Xiao Li, Yadong Wei, Xueyu Song, Yang Song, Xiaoqiang Xia, Fangrui Zeng, Zaiyi Chen, Liu Liu, Gu Xu, Tong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2507.02790)  

**Abstract**: The rapid growth of online video content, especially on short video platforms, has created a growing demand for efficient video editing techniques that can condense long-form videos into concise and engaging clips. Existing automatic editing methods predominantly rely on textual cues from ASR transcripts and end-to-end segment selection, often neglecting the rich visual context and leading to incoherent outputs. In this paper, we propose a human-inspired automatic video editing framework (HIVE) that leverages multimodal narrative understanding to address these limitations. Our approach incorporates character extraction, dialogue analysis, and narrative summarization through multimodal large language models, enabling a holistic understanding of the video content. To further enhance coherence, we apply scene-level segmentation and decompose the editing process into three subtasks: highlight detection, opening/ending selection, and pruning of irrelevant content. To facilitate research in this area, we introduce DramaAD, a novel benchmark dataset comprising over 800 short drama episodes and 500 professionally edited advertisement clips. Experimental results demonstrate that our framework consistently outperforms existing baselines across both general and advertisement-oriented editing tasks, significantly narrowing the quality gap between automatic and human-edited videos. 

---
# DeSTA2.5-Audio: Toward General-Purpose Large Audio Language Model with Self-Generated Cross-Modal Alignment 

**Authors**: Ke-Han Lu, Zhehuai Chen, Szu-Wei Fu, Chao-Han Huck Yang, Sung-Feng Huang, Chih-Kai Yang, Chee-En Yu, Chun-Wei Chen, Wei-Chih Chen, Chien-yu Huang, Yi-Cheng Lin, Yu-Xiang Lin, Chi-An Fu, Chun-Yi Kuan, Wenze Ren, Xuanjun Chen, Wei-Ping Huang, En-Pei Hu, Tzu-Quan Lin, Yuan-Kuei Wu, Kuan-Po Huang, Hsiao-Ying Huang, Huang-Cheng Chou, Kai-Wei Chang, Cheng-Han Chiang, Boris Ginsburg, Yu-Chiang Frank Wang, Hung-yi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2507.02768)  

**Abstract**: We introduce DeSTA2.5-Audio, a general-purpose Large Audio Language Model (LALM) designed for robust auditory perception and instruction-following, without requiring task-specific audio instruction-tuning. Recent LALMs typically augment Large Language Models (LLMs) with auditory capabilities by training on large-scale, manually curated or LLM-synthesized audio-instruction datasets. However, these approaches have often suffered from the catastrophic forgetting of the LLM's original language abilities. To address this, we revisit the data construction pipeline and propose DeSTA, a self-generated cross-modal alignment strategy in which the backbone LLM generates its own training targets. This approach preserves the LLM's native language proficiency while establishing effective audio-text alignment, thereby enabling zero-shot generalization without task-specific tuning. Using DeSTA, we construct DeSTA-AQA5M, a large-scale, task-agnostic dataset containing 5 million training samples derived from 7,000 hours of audio spanning 50 diverse datasets, including speech, environmental sounds, and music. DeSTA2.5-Audio achieves state-of-the-art or competitive performance across a wide range of audio-language benchmarks, including Dynamic-SUPERB, MMAU, SAKURA, Speech-IFEval, and VoiceBench. Comprehensive comparative studies demonstrate that our self-generated strategy outperforms widely adopted data construction and training strategies in both auditory perception and instruction-following capabilities. Our findings underscore the importance of carefully designed data construction in LALM development and offer practical insights for building robust, general-purpose LALMs. 

---
# Early Signs of Steganographic Capabilities in Frontier LLMs 

**Authors**: Artur Zolkowski, Kei Nishimura-Gasparian, Robert McCarthy, Roland S. Zimmermann, David Lindner  

**Link**: [PDF](https://arxiv.org/pdf/2507.02737)  

**Abstract**: Monitoring Large Language Model (LLM) outputs is crucial for mitigating risks from misuse and misalignment. However, LLMs could evade monitoring through steganography: Encoding hidden information within seemingly benign generations. In this paper, we evaluate the steganography capabilities in frontier LLMs to better understand the risk they pose. We focus on two types of steganography: passing encoded messages and performing encoded reasoning. We find that current models are unable to encode short messages in their outputs without a monitor noticing under standard affordances. They can succeed, however, if given additional affordances such as using an unmonitored scratchpad and coordinating on what encoding scheme to use. We additionally find early signs that models can perform basic encoded reasoning in a simple state-tracking problem. This includes some ability to reason with their own and pre-defined schemes, including encoding schemes such as Hexadecimal. Despite this, they can rarely hide reasoning subtly within a cover task to fool a monitor. Overall, our results indicate that current LLMs exhibit nascent steganographic capabilities. While these capabilities are likely insufficient to bypass well-designed monitors at present, this could change in the future. 

---
# ASDA: Audio Spectrogram Differential Attention Mechanism for Self-Supervised Representation Learning 

**Authors**: Junyu Wang, Tianrui Wang, Meng Ge, Longbiao Wang, Jianwu Dang  

**Link**: [PDF](https://arxiv.org/pdf/2507.02666)  

**Abstract**: In recent advancements in audio self-supervised representation learning, the standard Transformer architecture has emerged as the predominant approach, yet its attention mechanism often allocates a portion of attention weights to irrelevant information, potentially impairing the model's discriminative ability. To address this, we introduce a differential attention mechanism, which effectively mitigates ineffective attention allocation through the integration of dual-softmax operations and appropriately tuned differential coefficients. Experimental results demonstrate that our ASDA model achieves state-of-the-art (SOTA) performance across multiple benchmarks, including audio classification (49.0% mAP on AS-2M, 41.5% mAP on AS20K), keyword spotting (98.3% accuracy on SPC-2), and environmental sound classification (96.1% accuracy on ESC-50). These results highlight ASDA's effectiveness in audio tasks, paving the way for broader applications. 

---
# OmniDraft: A Cross-vocabulary, Online Adaptive Drafter for On-device Speculative Decoding 

**Authors**: Ramchalam Kinattinkara Ramakrishnan, Zhaocong Yuan, Shaojie Zhuo, Chen Feng, Yicheng Lin, Chenzheng Su, Xiaopeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.02659)  

**Abstract**: Speculative decoding generally dictates having a small, efficient draft model that is either pretrained or distilled offline to a particular target model series, for instance, Llama or Qwen models. However, within online deployment settings, there are two major challenges: 1) usage of a target model that is incompatible with the draft model; 2) expectation of latency improvements over usage and time. In this work, we propose OmniDraft, a unified framework that enables a single draft model to operate with any target model and adapt dynamically to user data. We introduce an online n-gram cache with hybrid distillation fine-tuning to address the cross-vocabulary mismatch across draft and target models; and further improve decoding speed by leveraging adaptive drafting techniques. OmniDraft is particularly suitable for on-device LLM applications where model cost, efficiency and user customization are the major points of contention. This further highlights the need to tackle the above challenges and motivates the \textit{``one drafter for all''} paradigm. We showcase the proficiency of the OmniDraft framework by performing online learning on math reasoning, coding and text generation tasks. Notably, OmniDraft enables a single Llama-68M model to pair with various target models including Vicuna-7B, Qwen2-7B and Llama3-8B models for speculative decoding; and additionally provides up to 1.5-2x speedup. 

---
# Decoupled Planning and Execution: A Hierarchical Reasoning Framework for Deep Search 

**Authors**: Jiajie Jin, Xiaoxi Li, Guanting Dong, Yuyao Zhang, Yutao Zhu, Yang Zhao, Hongjin Qian, Zhicheng Dou  

**Link**: [PDF](https://arxiv.org/pdf/2507.02652)  

**Abstract**: Complex information needs in real-world search scenarios demand deep reasoning and knowledge synthesis across diverse sources, which traditional retrieval-augmented generation (RAG) pipelines struggle to address effectively. Current reasoning-based approaches suffer from a fundamental limitation: they use a single model to handle both high-level planning and detailed execution, leading to inefficient reasoning and limited scalability. In this paper, we introduce HiRA, a hierarchical framework that separates strategic planning from specialized execution. Our approach decomposes complex search tasks into focused subtasks, assigns each subtask to domain-specific agents equipped with external tools and reasoning capabilities, and coordinates the results through a structured integration mechanism. This separation prevents execution details from disrupting high-level reasoning while enabling the system to leverage specialized expertise for different types of information processing. Experiments on four complex, cross-modal deep search benchmarks demonstrate that HiRA significantly outperforms state-of-the-art RAG and agent-based systems. Our results show improvements in both answer quality and system efficiency, highlighting the effectiveness of decoupled planning and execution for multi-step information seeking tasks. Our code is available at this https URL. 

---
# Strategic Intelligence in Large Language Models: Evidence from evolutionary Game Theory 

**Authors**: Kenneth Payne, Baptiste Alloui-Cros  

**Link**: [PDF](https://arxiv.org/pdf/2507.02618)  

**Abstract**: Are Large Language Models (LLMs) a new form of strategic intelligence, able to reason about goals in competitive settings? We present compelling supporting evidence. The Iterated Prisoner's Dilemma (IPD) has long served as a model for studying decision-making. We conduct the first ever series of evolutionary IPD tournaments, pitting canonical strategies (e.g., Tit-for-Tat, Grim Trigger) against agents from the leading frontier AI companies OpenAI, Google, and Anthropic. By varying the termination probability in each tournament (the "shadow of the future"), we introduce complexity and chance, confounding memorisation.
Our results show that LLMs are highly competitive, consistently surviving and sometimes even proliferating in these complex ecosystems. Furthermore, they exhibit distinctive and persistent "strategic fingerprints": Google's Gemini models proved strategically ruthless, exploiting cooperative opponents and retaliating against defectors, while OpenAI's models remained highly cooperative, a trait that proved catastrophic in hostile environments. Anthropic's Claude emerged as the most forgiving reciprocator, showing remarkable willingness to restore cooperation even after being exploited or successfully defecting. Analysis of nearly 32,000 prose rationales provided by the models reveals that they actively reason about both the time horizon and their opponent's likely strategy, and we demonstrate that this reasoning is instrumental to their decisions. This work connects classic game theory with machine psychology, offering a rich and granular view of algorithmic decision-making under uncertainty. 

---
# JoyTTS: LLM-based Spoken Chatbot With Voice Cloning 

**Authors**: Fangru Zhou, Jun Zhao, Guoxin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.02380)  

**Abstract**: JoyTTS is an end-to-end spoken chatbot that combines large language models (LLM) with text-to-speech (TTS) technology, featuring voice cloning capabilities. This project is built upon the open-source MiniCPM-o and CosyVoice2 models and trained on 2000 hours of conversational data. We have also provided the complete training code to facilitate further development and optimization by the community. On the testing machine seed-tts-zh, it achieves a SS (speaker similarity) score of 0.73 and a WER (Word Error Rate) of 5.09. The code and models, along with training and inference scripts, are available at this https URL. 

---
# Seeing Through Green: Text-Based Classification and the Firm's Returns from Green Patents 

**Authors**: Lapo Santarlasci, Armando Rungi, Antonio Zinilli  

**Link**: [PDF](https://arxiv.org/pdf/2507.02287)  

**Abstract**: This paper introduces Natural Language Processing for identifying ``true'' green patents from official supporting documents. We start our training on about 12.4 million patents that had been classified as green from previous literature. Thus, we train a simple neural network to enlarge a baseline dictionary through vector representations of expressions related to environmental technologies. After testing, we find that ``true'' green patents represent about 20\% of the total of patents classified as green from previous literature. We show heterogeneity by technological classes, and then check that `true' green patents are about 1\% less cited by following inventions. In the second part of the paper, we test the relationship between patenting and a dashboard of firm-level financial accounts in the European Union. After controlling for reverse causality, we show that holding at least one ``true'' green patent raises sales, market shares, and productivity. If we restrict the analysis to high-novelty ``true'' green patents, we find that they also yield higher profits. Our findings underscore the importance of using text analyses to gauge finer-grained patent classifications that are useful for policymaking in different domains. 

---
# SciGA: A Comprehensive Dataset for Designing Graphical Abstracts in Academic Papers 

**Authors**: Takuro Kawada, Shunsuke Kitada, Sota Nemoto, Hitoshi Iyatomi  

**Link**: [PDF](https://arxiv.org/pdf/2507.02212)  

**Abstract**: Graphical Abstracts (GAs) play a crucial role in visually conveying the key findings of scientific papers. While recent research has increasingly incorporated visual materials such as Figure 1 as de facto GAs, their potential to enhance scientific communication remains largely unexplored. Moreover, designing effective GAs requires advanced visualization skills, creating a barrier to their widespread adoption. To tackle these challenges, we introduce SciGA-145k, a large-scale dataset comprising approximately 145,000 scientific papers and 1.14 million figures, explicitly designed for supporting GA selection and recommendation as well as facilitating research in automated GA generation. As a preliminary step toward GA design support, we define two tasks: 1) Intra-GA recommendation, which identifies figures within a given paper that are well-suited to serve as GAs, and 2) Inter-GA recommendation, which retrieves GAs from other papers to inspire the creation of new GAs. We provide reasonable baseline models for these tasks. Furthermore, we propose Confidence Adjusted top-1 ground truth Ratio (CAR), a novel recommendation metric that offers a fine-grained analysis of model behavior. CAR addresses limitations in traditional ranking-based metrics by considering cases where multiple figures within a paper, beyond the explicitly labeled GA, may also serve as GAs. By unifying these tasks and metrics, our SciGA-145k establishes a foundation for advancing visual scientific communication while contributing to the development of AI for Science. 

---
# ESTR-CoT: Towards Explainable and Accurate Event Stream based Scene Text Recognition with Chain-of-Thought Reasoning 

**Authors**: Xiao Wang, Jingtao Jiang, Qiang Chen, Lan Chen, Lin Zhu, Yaowei Wang, Yonghong Tian, Jin Tang  

**Link**: [PDF](https://arxiv.org/pdf/2507.02200)  

**Abstract**: Event stream based scene text recognition is a newly arising research topic in recent years which performs better than the widely used RGB cameras in extremely challenging scenarios, especially the low illumination, fast motion. Existing works either adopt end-to-end encoder-decoder framework or large language models for enhanced recognition, however, they are still limited by the challenges of insufficient interpretability and weak contextual logical reasoning. In this work, we propose a novel chain-of-thought reasoning based event stream scene text recognition framework, termed ESTR-CoT. Specifically, we first adopt the vision encoder EVA-CLIP (ViT-G/14) to transform the input event stream into tokens and utilize a Llama tokenizer to encode the given generation prompt. A Q-former is used to align the vision token to the pre-trained large language model Vicuna-7B and output both the answer and chain-of-thought (CoT) reasoning process simultaneously. Our framework can be optimized using supervised fine-tuning in an end-to-end manner. In addition, we also propose a large-scale CoT dataset to train our framework via a three stage processing (i.e., generation, polish, and expert verification). This dataset provides a solid data foundation for the development of subsequent reasoning-based large models. Extensive experiments on three event stream STR benchmark datasets (i.e., EventSTR, WordArt*, IC15*) fully validated the effectiveness and interpretability of our proposed framework. The source code and pre-trained models will be released on this https URL. 

---
# Analyzing and Improving Speaker Similarity Assessment for Speech Synthesis 

**Authors**: Marc-André Carbonneau, Benjamin van Niekerk, Hugo Seuté, Jean-Philippe Letendre, Herman Kamper, Julian Zaïdi  

**Link**: [PDF](https://arxiv.org/pdf/2507.02176)  

**Abstract**: Modeling voice identity is challenging due to its multifaceted nature. In generative speech systems, identity is often assessed using automatic speaker verification (ASV) embeddings, designed for discrimination rather than characterizing identity. This paper investigates which aspects of a voice are captured in such representations. We find that widely used ASV embeddings focus mainly on static features like timbre and pitch range, while neglecting dynamic elements such as rhythm. We also identify confounding factors that compromise speaker similarity measurements and suggest mitigation strategies. To address these gaps, we propose U3D, a metric that evaluates speakers' dynamic rhythm patterns. This work contributes to the ongoing challenge of assessing speaker identity consistency in the context of ever-better voice cloning systems. We publicly release our code. 

---
# Dissecting the Impact of Mobile DVFS Governors on LLM Inference Performance and Energy Efficiency 

**Authors**: Zongpu Zhang, Pranab Dash, Y. Charlie Hu, Qiang Xu, Jian Li, Haibing Guan  

**Link**: [PDF](https://arxiv.org/pdf/2507.02135)  

**Abstract**: Large Language Models (LLMs) are increasingly being integrated into various applications and services running on billions of mobile devices. However, deploying LLMs on resource-limited mobile devices faces a significant challenge due to their high demand for computation, memory, and ultimately energy. While current LLM frameworks for mobile use three power-hungry components-CPU, GPU, and Memory-even when running primarily-GPU LLM models, optimized DVFS governors for CPU, GPU, and memory featured in modern mobile devices operate independently and are oblivious of each other. Motivated by the above observation, in this work, we first measure the energy-efficiency of a SOTA LLM framework consisting of various LLM models on mobile phones which showed the triplet mobile governors result in up to 40.4% longer prefilling and decoding latency compared to optimal combinations of CPU, GPU, and memory frequencies with the same energy consumption for sampled prefill and decode lengths. Second, we conduct an in-depth measurement study to uncover how the intricate interplay (or lack of) among the mobile governors cause the above inefficiency in LLM inference. Finally, based on these insights, we design FUSE - a unified energy-aware governor for optimizing the energy efficiency of LLM inference on mobile devices. Our evaluation using a ShareGPT dataset shows FUSE reduces the time-to-first-token and time-per-output-token latencies by 7.0%-16.9% and 25.4%-36.8% on average with the same energy-per-token for various mobile LLM models. 

---
# Energy-Based Transformers are Scalable Learners and Thinkers 

**Authors**: Alexi Gladstone, Ganesh Nanduru, Md Mofijul Islam, Peixuan Han, Hyeonjeong Ha, Aman Chadha, Yilun Du, Heng Ji, Jundong Li, Tariq Iqbal  

**Link**: [PDF](https://arxiv.org/pdf/2507.02092)  

**Abstract**: Inference-time computation techniques, analogous to human System 2 Thinking, have recently become popular for improving model performances. However, most existing approaches suffer from several limitations: they are modality-specific (e.g., working only in text), problem-specific (e.g., verifiable domains like math and coding), or require additional supervision/training on top of unsupervised pretraining (e.g., verifiers or verifiable rewards). In this paper, we ask the question "Is it possible to generalize these System 2 Thinking approaches, and develop models that learn to think solely from unsupervised learning?" Interestingly, we find the answer is yes, by learning to explicitly verify the compatibility between inputs and candidate-predictions, and then re-framing prediction problems as optimization with respect to this verifier. Specifically, we train Energy-Based Transformers (EBTs) -- a new class of Energy-Based Models (EBMs) -- to assign an energy value to every input and candidate-prediction pair, enabling predictions through gradient descent-based energy minimization until convergence. Across both discrete (text) and continuous (visual) modalities, we find EBTs scale faster than the dominant Transformer++ approach during training, achieving an up to 35% higher scaling rate with respect to data, batch size, parameters, FLOPs, and depth. During inference, EBTs improve performance with System 2 Thinking by 29% more than the Transformer++ on language tasks, and EBTs outperform Diffusion Transformers on image denoising while using fewer forward passes. Further, we find that EBTs achieve better results than existing models on most downstream tasks given the same or worse pretraining performance, suggesting that EBTs generalize better than existing approaches. Consequently, EBTs are a promising new paradigm for scaling both the learning and thinking capabilities of models. 

---
# Evaluating the Promise and Pitfalls of LLMs in Hiring Decisions 

**Authors**: Eitan Anzenberg, Arunava Samajpati, Sivasankaran Chandrasekar, Varun Kacholia  

**Link**: [PDF](https://arxiv.org/pdf/2507.02087)  

**Abstract**: The use of large language models (LLMs) in hiring promises to streamline candidate screening, but it also raises serious concerns regarding accuracy and algorithmic bias where sufficient safeguards are not in place. In this work, we benchmark several state-of-the-art foundational LLMs - including models from OpenAI, Anthropic, Google, Meta, and Deepseek, and compare them with our proprietary domain-specific hiring model (Match Score) for job candidate matching. We evaluate each model's predictive accuracy (ROC AUC, Precision-Recall AUC, F1-score) and fairness (impact ratio of cut-off analysis across declared gender, race, and intersectional subgroups). Our experiments on a dataset of roughly 10,000 real-world recent candidate-job pairs show that Match Score outperforms the general-purpose LLMs on accuracy (ROC AUC 0.85 vs 0.77) and achieves significantly more equitable outcomes across demographic groups. Notably, Match Score attains a minimum race-wise impact ratio of 0.957 (near-parity), versus 0.809 or lower for the best LLMs, (0.906 vs 0.773 for the intersectionals, respectively). We discuss why pretraining biases may cause LLMs with insufficient safeguards to propagate societal biases in hiring scenarios, whereas a bespoke supervised model can more effectively mitigate these biases. Our findings highlight the importance of domain-specific modeling and bias auditing when deploying AI in high-stakes domains such as hiring, and caution against relying on off-the-shelf LLMs for such tasks without extensive fairness safeguards. Furthermore, we show with empirical evidence that there shouldn't be a dichotomy between choosing accuracy and fairness in hiring: a well-designed algorithm can achieve both accuracy in hiring and fairness in outcomes. 

---
# STELLA: Self-Evolving LLM Agent for Biomedical Research 

**Authors**: Ruofan Jin, Zaixi Zhang, Mengdi Wang, Le Cong  

**Link**: [PDF](https://arxiv.org/pdf/2507.02004)  

**Abstract**: The rapid growth of biomedical data, tools, and literature has created a fragmented research landscape that outpaces human expertise. While AI agents offer a solution, they typically rely on static, manually curated toolsets, limiting their ability to adapt and scale. Here, we introduce STELLA, a self-evolving AI agent designed to overcome these limitations. STELLA employs a multi-agent architecture that autonomously improves its own capabilities through two core mechanisms: an evolving Template Library for reasoning strategies and a dynamic Tool Ocean that expands as a Tool Creation Agent automatically discovers and integrates new bioinformatics tools. This allows STELLA to learn from experience. We demonstrate that STELLA achieves state-of-the-art accuracy on a suite of biomedical benchmarks, scoring approximately 26\% on Humanity's Last Exam: Biomedicine, 54\% on LAB-Bench: DBQA, and 63\% on LAB-Bench: LitQA, outperforming leading models by up to 6 percentage points. More importantly, we show that its performance systematically improves with experience; for instance, its accuracy on the Humanity's Last Exam benchmark almost doubles with increased trials. STELLA represents a significant advance towards AI Agent systems that can learn and grow, dynamically scaling their expertise to accelerate the pace of biomedical discovery. 

---
# Why Multi-Interest Fairness Matters: Hypergraph Contrastive Multi-Interest Learning for Fair Conversational Recommender System 

**Authors**: Yongsen Zheng, Zongxuan Xie, Guohua Wang, Ziyao Liu, Liang Lin, Kwok-Yan Lam  

**Link**: [PDF](https://arxiv.org/pdf/2507.02000)  

**Abstract**: Unfairness is a well-known challenge in Recommender Systems (RSs), often resulting in biased outcomes that disadvantage users or items based on attributes such as gender, race, age, or popularity. Although some approaches have started to improve fairness recommendation in offline or static contexts, the issue of unfairness often exacerbates over time, leading to significant problems like the Matthew effect, filter bubbles, and echo chambers. To address these challenges, we proposed a novel framework, Hypergraph Contrastive Multi-Interest Learning for Fair Conversational Recommender System (HyFairCRS), aiming to promote multi-interest diversity fairness in dynamic and interactive Conversational Recommender Systems (CRSs). HyFairCRS first captures a wide range of user interests by establishing diverse hypergraphs through contrastive learning. These interests are then utilized in conversations to generate informative responses and ensure fair item predictions within the dynamic user-system feedback loop. Experiments on two CRS-based datasets show that HyFairCRS achieves a new state-of-the-art performance while effectively alleviating unfairness. Our code is available at this https URL. 

---
# FinAI-BERT: A Transformer-Based Model for Sentence-Level Detection of AI Disclosures in Financial Reports 

**Authors**: Muhammad Bilal Zafar  

**Link**: [PDF](https://arxiv.org/pdf/2507.01991)  

**Abstract**: The proliferation of artificial intelligence (AI) in financial services has prompted growing demand for tools that can systematically detect AI-related disclosures in corporate filings. While prior approaches often rely on keyword expansion or document-level classification, they fall short in granularity, interpretability, and robustness. This study introduces FinAI-BERT, a domain-adapted transformer-based language model designed to classify AI-related content at the sentence level within financial texts. The model was fine-tuned on a manually curated and balanced dataset of 1,586 sentences drawn from 669 annual reports of U.S. banks (2015 to 2023). FinAI-BERT achieved near-perfect classification performance (accuracy of 99.37 percent, F1 score of 0.993), outperforming traditional baselines such as Logistic Regression, Naive Bayes, Random Forest, and XGBoost. Interpretability was ensured through SHAP-based token attribution, while bias analysis and robustness checks confirmed the model's stability across sentence lengths, adversarial inputs, and temporal samples. Theoretically, the study advances financial NLP by operationalizing fine-grained, theme-specific classification using transformer architectures. Practically, it offers a scalable, transparent solution for analysts, regulators, and scholars seeking to monitor the diffusion and framing of AI across financial institutions. 

---
# Multimodal Misinformation Detection Using Early Fusion of Linguistic, Visual, and Social Features 

**Authors**: Gautam Kishore Shahi  

**Link**: [PDF](https://arxiv.org/pdf/2507.01984)  

**Abstract**: Amid a tidal wave of misinformation flooding social media during elections and crises, extensive research has been conducted on misinformation detection, primarily focusing on text-based or image-based approaches. However, only a few studies have explored multimodal feature combinations, such as integrating text and images for building a classification model to detect misinformation. This study investigates the effectiveness of different multimodal feature combinations, incorporating text, images, and social features using an early fusion approach for the classification model. This study analyzed 1,529 tweets containing both text and images during the COVID-19 pandemic and election periods collected from Twitter (now X). A data enrichment process was applied to extract additional social features, as well as visual features, through techniques such as object detection and optical character recognition (OCR). The results show that combining unsupervised and supervised machine learning models improves classification performance by 15% compared to unimodal models and by 5% compared to bimodal models. Additionally, the study analyzes the propagation patterns of misinformation based on the characteristics of misinformation tweets and the users who disseminate them. 

---
