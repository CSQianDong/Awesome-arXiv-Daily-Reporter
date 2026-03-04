# Using Learning Progressions to Guide AI Feedback for Science Learning 

**Authors**: Xin Xia, Nejla Yuruk, Yun Wang, Xiaoming Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2603.03249)  

**Abstract**: Generative artificial intelligence (AI) offers scalable support for formative feedback, yet most AI-generated feedback relies on task-specific rubrics authored by domain experts. While effective, rubric authoring is time-consuming and limits scalability across instructional contexts. Learning progressions (LP) provide a theoretically grounded representation of students' developing understanding and may offer an alternative solution. This study examines whether an LP-driven rubric generation pipeline can produce AI-generated feedback comparable in quality to feedback guided by expert-authored task rubrics. We analyzed AI-generated feedback for written scientific explanations produced by 207 middle school students in a chemistry task. Two pipelines were compared: (a) feedback guided by a human expert-designed, task-specific rubric, and (b) feedback guided by a task-specific rubric automatically derived from a learning progression prior to grading and feedback generation. Two human coders evaluated feedback quality using a multi-dimensional rubric assessing Clarity, Accuracy, Relevance, Engagement and Motivation, and Reflectiveness (10 sub-dimensions). Inter-rater reliability was high, with percent agreement ranging from 89% to 100% and Cohen's kappa values for estimable dimensions (kappa = .66 to .88). Paired t-tests revealed no statistically significant differences between the two pipelines for Clarity (t1 = 0.00, p1 = 1.000; t2 = 0.84, p2 = .399), Relevance (t1 = 0.28, p1 = .782; t2 = -0.58, p2 = .565), Engagement and Motivation (t1 = 0.50, p1 = .618; t2 = -0.58, p2 = .565), or Reflectiveness (t = -0.45, p = .656). These findings suggest that the LP-driven rubric pipeline can serve as an alternative solution. 

---
# Learning When to Act or Refuse: Guarding Agentic Reasoning Models for Safe Multi-Step Tool Use 

**Authors**: Aradhye Agarwal, Gurdit Siyan, Yash Pandya, Joykirat Singh, Akshay Nambi, Ahmed Awadallah  

**Link**: [PDF](https://arxiv.org/pdf/2603.03205)  

**Abstract**: Agentic language models operate in a fundamentally different safety regime than chat models: they must plan, call tools, and execute long-horizon actions where a single misstep, such as accessing files or entering credentials, can cause irreversible harm. Existing alignment methods, largely optimized for static generation and task completion, break down in these settings due to sequential decision-making, adversarial tool feedback, and overconfident intermediate reasoning. We introduce MOSAIC, a post-training framework that aligns agents for safe multi-step tool use by making safety decisions explicit and learnable. MOSAIC structures inference as a plan, check, then act or refuse loop, with explicit safety reasoning and refusal as first-class actions. To train without trajectory-level labels, we use preference-based reinforcement learning with pairwise trajectory comparisons, which captures safety distinctions often missed by scalar rewards. We evaluate MOSAIC zero-shot across three model families, Qwen2.5-7B, Qwen3-4B-Thinking, and Phi-4, and across out-of-distribution benchmarks spanning harmful tasks, prompt injection, benign tool use, and cross-domain privacy leakage. MOSAIC reduces harmful behavior by up to 50%, increases harmful-task refusal by over 20% on injection attacks, cuts privacy leakage, and preserves or improves benign task performance, demonstrating robust generalization across models, domains, and agentic settings. 

---
# Code2Math: Can Your Code Agent Effectively Evolve Math Problems Through Exploration? 

**Authors**: Dadi Guo, Yuejin Xie, Qingyu Liu, Jiayu Liu, Zhiyuan Fan, Qihan Ren, Shuai Shao, Tianyi Zhou, Dongrui Liu, Yi R. Fung  

**Link**: [PDF](https://arxiv.org/pdf/2603.03202)  

**Abstract**: As large language models (LLMs) advance their mathematical capabilities toward the IMO level, the scarcity of challenging, high-quality problems for training and evaluation has become a significant bottleneck. Simultaneously, recent code agents have demonstrated sophisticated skills in agentic coding and reasoning, suggesting that code execution can serve as a scalable environment for mathematical experimentation. In this paper, we investigate the potential of code agents to autonomously evolve existing math problems into more complex variations. We introduce a multi-agent framework designed to perform problem evolution while validating the solvability and increased difficulty of the generated problems. Our experiments demonstrate that, given sufficient test-time exploration, code agents can synthesize new, solvable problems that are structurally distinct from and more challenging than the originals. This work provides empirical evidence that code-driven agents can serve as a viable mechanism for synthesizing high-difficulty mathematical reasoning problems within scalable computational environments. Our data is available at this https URL. 

---
# BeyondSWE: Can Current Code Agent Survive Beyond Single-Repo Bug Fixing? 

**Authors**: Guoxin Chen, Fanzhe Meng, Jiale Zhao, Minghao Li, Daixuan Cheng, Huatong Song, Jie Chen, Yuzhi Lin, Hui Chen, Xin Zhao, Ruihua Song, Chang Liu, Cheng Chen, Kai Jia, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2603.03194)  

**Abstract**: Current benchmarks for code agents primarily assess narrow, repository-specific fixes, overlooking critical real-world challenges such as cross-repository reasoning, domain-specialized problem solving, dependency-driven migration, and full-repository generation. To address this gap, we introduce BeyondSWE, a comprehensive benchmark that broadens existing evaluations along two axes - resolution scope and knowledge scope - using 500 real-world instances across four distinct settings. Experimental results reveal a significant capability gap: even frontier models plateau below 45% success, and no single model performs consistently across task types. To systematically investigate the role of external knowledge, we develop SearchSWE, a framework that integrates deep search with coding abilities. Our experiments show that search augmentation yields inconsistent gains and can in some cases degrade performance, highlighting the difficulty of emulating developer-like workflows that interleave search and reasoning during coding tasks. This work offers both a realistic, challenging evaluation benchmark and a flexible framework to advance research toward more capable code agents. 

---
# APRES: An Agentic Paper Revision and Evaluation System 

**Authors**: Bingchen Zhao, Jenny Zhang, Chenxi Whitehouse, Minqi Jiang, Michael Shvartsman, Abhishek Charnalia, Despoina Magka, Tatiana Shavrina, Derek Dunfield, Oisin Mac Aodha, Yoram Bachrach  

**Link**: [PDF](https://arxiv.org/pdf/2603.03142)  

**Abstract**: Scientific discoveries must be communicated clearly to realize their full potential. Without effective communication, even the most groundbreaking findings risk being overlooked or misunderstood. The primary way scientists communicate their work and receive feedback from the community is through peer review. However, the current system often provides inconsistent feedback between reviewers, ultimately hindering the improvement of a manuscript and limiting its potential impact. In this paper, we introduce a novel method APRES powered by Large Language Models (LLMs) to update a scientific papers text based on an evaluation rubric. Our automated method discovers a rubric that is highly predictive of future citation counts, and integrate it with APRES in an automated system that revises papers to enhance their quality and impact. Crucially, this objective should be met without altering the core scientific content. We demonstrate the success of APRES, which improves future citation prediction by 19.6% in mean averaged error over the next best baseline, and show that our paper revision process yields papers that are preferred over the originals by human expert evaluators 79% of the time. Our findings provide strong empirical support for using LLMs as a tool to help authors stress-test their manuscripts before submission. Ultimately, our work seeks to augment, not replace, the essential role of human expert reviewers, for it should be humans who discern which discoveries truly matter, guiding science toward advancing knowledge and enriching lives. 

---
# UniSkill: A Dataset for Matching University Curricula to Professional Competencies 

**Authors**: Nurlan Musazade, Joszef Mezei, Mike Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.03134)  

**Abstract**: Skill extraction and recommendation systems have been studied from recruiter, applicant, and education perspectives. While AI applications in job advertisements have received broad attention, deficiencies in the instructed skills side remain a challenge. In this work, we address the scarcity of publicly available datasets by releasing both manually annotated and synthetic datasets of skills from the European Skills, Competences, Qualifications and Occupations (ESCO) taxonomy and university course pairs and publishing corresponding annotation guidelines. Specifically, we match graduate-level university courses with skills from the Systems Analysts and Management and Organization Analyst ESCO occupation groups at two granularities: course title with a skill, and course sentence with a skill. We train language models on this dataset to serve as a baseline for retrieval and recommendation systems for course-to-skill and skill-to-course matching. We evaluate the models on a portion of the annotated data. Our BERT model achieves 87% F1-score, showing that course and skill matching is a feasible task. 

---
# Evaluating Performance Drift from Model Switching in Multi-Turn LLM Systems 

**Authors**: Raad Khraishi, Iman Zafar, Katie Myles, Greig A Cowan  

**Link**: [PDF](https://arxiv.org/pdf/2603.03111)  

**Abstract**: Deployed multi-turn LLM systems routinely switch models mid-interaction due to upgrades, cross-provider routing, and fallbacks. Such handoffs create a context mismatch: the model generating later turns must condition on a dialogue prefix authored by a different model, potentially inducing silent performance drift. We introduce a switch-matrix benchmark that measures this effect by running a prefix model for early turns and a suffix model for the final turn, and comparing against the no-switch baseline using paired episode-level bootstrap confidence intervals. Across CoQA conversational QA and Multi-IF benchmarks, even a single-turn handoff yields prevalent and statistically significant, directional effects and may swing outcomes by -8 to +13 percentage points in Multi-IF strict success rate and +/- 4 absolute F1 on CoQA, comparable to the no-switch gap between common model tiers (e.g., GPT-5-nano vs GPT-5-mini). We further find systematic compatibility patterns: some suffix models degrade under nearly any non-self dialogue history, while others improve under nearly any foreign prefix. To enable compressed handoff risk monitoring, we decompose switch-induced drift into per-model prefix influence and suffix susceptibility terms, accounting for ~70% of variance across benchmarks. These results position handoff robustness as an operational reliability dimension that single-model benchmarks miss, motivating explicit monitoring and handoff-aware mitigation in multi-turn systems. 

---
# Compact Prompting in Instruction-tuned LLMs for Joint Argumentative Component Detection 

**Authors**: Sofiane Elguendouze, Erwan Hain, Elena Cabrio, Serena Villata  

**Link**: [PDF](https://arxiv.org/pdf/2603.03095)  

**Abstract**: Argumentative component detection (ACD) is a core subtask of Argument(ation) Mining (AM) and one of its most challenging aspects, as it requires jointly delimiting argumentative spans and classifying them into components such as claims and premises. While research on this subtask remains relatively limited compared to other AM tasks, most existing approaches formulate it as a simplified sequence labeling problem, component classification, or a pipeline of component segmentation followed by classification. In this paper, we propose a novel approach based on instruction-tuned Large Language Models (LLMs) using compact instruction-based prompts, and reframe ACD as a language generation task, enabling arguments to be identified directly from plain text without relying on pre-segmented components. Experiments on standard benchmarks show that our approach achieves higher performance compared to state-of-the-art systems. To the best of our knowledge, this is one of the first attempts to fully model ACD as a generative task, highlighting the potential of instruction tuning for complex AM problems. 

---
# TAO-Attack: Toward Advanced Optimization-Based Jailbreak Attacks for Large Language Models 

**Authors**: Zhi Xu, Jiaqi Li, Xiaotong Zhang, Hong Yu, Han Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.03081)  

**Abstract**: Large language models (LLMs) have achieved remarkable success across diverse applications but remain vulnerable to jailbreak attacks, where attackers craft prompts that bypass safety alignment and elicit unsafe responses. Among existing approaches, optimization-based attacks have shown strong effectiveness, yet current methods often suffer from frequent refusals, pseudo-harmful outputs, and inefficient token-level updates. In this work, we propose TAO-Attack, a new optimization-based jailbreak method. TAO-Attack employs a two-stage loss function: the first stage suppresses refusals to ensure the model continues harmful prefixes, while the second stage penalizes pseudo-harmful outputs and encourages the model toward more harmful completions. In addition, we design a direction-priority token optimization (DPTO) strategy that improves efficiency by aligning candidates with the gradient direction before considering update magnitude. Extensive experiments on multiple LLMs demonstrate that TAO-Attack consistently outperforms state-of-the-art methods, achieving higher attack success rates and even reaching 100\% in certain scenarios. 

---
# PrivMedChat: End-to-End Differentially Private RLHF for Medical Dialogue Systems 

**Authors**: Sudip Bhujel  

**Link**: [PDF](https://arxiv.org/pdf/2603.03054)  

**Abstract**: Large language models are increasingly used for patient-facing medical assistance and clinical decision support, but adapting them to clinical dialogue often requires supervision derived from doctor-patient conversations that may contain sensitive information. Conventional supervised fine-tuning and reinforcement learning from human feedback (RLHF) can amplify memorization risks, enabling empirical membership inference and extraction of rare training-set content. We present PrivMedChat, an end-to-end framework for differentially private RLHF (DP-RLHF) for medical dialogue. Our design enforces differential privacy at every training stage that directly accesses dialogue-derived supervision: (i) Differential Private Stochastic Gradient Descent (DP-SGD) for medical SFT and (ii) DP-SGD for reward model learning from preference pairs. To limit additional privacy expenditure during alignment, we apply DP-SGD to the PPO actor and critic when operating on dialogue-derived prompts, while the reward model remains fixed after DP training.
We also introduce an annotation-free preference construction strategy that pairs physician responses with filtered non-expert generations to produce scalable preference data without clinician labeling. Experiments on medical dialogue benchmarks show that PrivMedChat at $\varepsilon=7$ achieves the highest ROUGE-L of 0.156 among all DP models, reduces clinical hallucinations to 1.4% and harmful advice to 0.4%, and obtains the highest overall score of 2.86 in a 3-model LLM-jury evaluation, while producing membership-inference signals that are near chance (AUC 0.510-0.555). We open-source our code at this https URL. 

---
# TrustMH-Bench: A Comprehensive Benchmark for Evaluating the Trustworthiness of Large Language Models in Mental Health 

**Authors**: Zixin Xiong, Ziteng Wang, Haotian Fan, Xinjie Zhang, Wenxuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.03047)  

**Abstract**: While Large Language Models (LLMs) demonstrate significant potential in providing accessible mental health support, their practical deployment raises critical trustworthiness concerns due to the domains high-stakes and safety-sensitive nature. Existing evaluation paradigms for general-purpose LLMs fail to capture mental health-specific requirements, highlighting an urgent need to prioritize and enhance their trustworthiness. To address this, we propose TrustMH-Bench, a holistic framework designed to systematically quantify the trustworthiness of mental health LLMs. By establishing a deep mapping from domain-specific norms to quantitative evaluation metrics, TrustMH-Bench evaluates models across eight core pillars: Reliability, Crisis Identification and Escalation, Safety, Fairness, Privacy, Robustness, Anti-sycophancy, and Ethics. We conduct extensive experiments across six general-purpose LLMs and six specialized mental health models. Experimental results indicate that the evaluated models underperform across various trustworthiness dimensions in mental health scenarios, revealing significant deficiencies. Notably, even generally powerful models (e.g., GPT-5.1) fail to maintain consistently high performance across all dimensions. Consequently, systematically improving the trustworthiness of LLMs has become a critical task. Our data and code are released. 

---
# MaBERT:A Padding Safe Interleaved Transformer Mamba Hybrid Encoder for Efficient Extended Context Masked Language Modeling 

**Authors**: Jinwoong Kim, Sangjin Park  

**Link**: [PDF](https://arxiv.org/pdf/2603.03001)  

**Abstract**: Self attention encoders such as Bidirectional Encoder Representations from Transformers(BERT) scale quadratically with sequence length, making long context modeling expensive. Linear time state space models, such as Mamba, are efficient; however, they show limitations in modeling global interactions and can suffer from padding induced state contamination. We propose MaBERT, a hybrid encoder that interleaves Transformer layers for global dependency modeling with Mamba layers for linear time state updates. This design alternates global contextual integration with fast state accumulation, enabling efficient training and inference on long inputs. To stabilize variable length batching, we introduce paddingsafe masking, which blocks state propagation through padded positions, and mask aware attention pooling, which aggregates information only from valid tokens. On GLUE, MaBERT achieves the best mean score on five of the eight tasks, with strong performance on the CoLA and sentence pair inference tasks. When extending the context from 512 to 4,096 tokens, MaBERT reduces training time and inference latency by 2.36x and 2.43x, respectively, relative to the average of encoder baselines, demonstrating a practical long context efficient encoder. 

---
# ACE-Merging: Data-Free Model Merging with Adaptive Covariance Estimation 

**Authors**: Bo Xu, Haotian Wu, Hehai Lin, Weiquan Huang, Beier Zhu, Yao Shu, Chengwei Qin  

**Link**: [PDF](https://arxiv.org/pdf/2603.02945)  

**Abstract**: Model merging aims to combine multiple task-specific expert models into a single model while preserving generalization across diverse tasks. However, interference among experts, especially when they are trained on different objectives, often leads to significant performance degradation. Despite recent progress, resolving this interference without data access, retraining, or architectural modification remains a fundamental challenge. This paper provides a theoretical analysis demonstrating that the input covariance of each task, which is a key factor for optimal merging, can be implicitly estimated from the parameter differences of its fine-tuned model, even in a fully data-free setting. Building on this insight, we introduce \acem, an Adaptive Covariance Estimation framework that effectively mitigates inter-task interference. Our approach features a principled, closed-form solution that contrasts with prior iterative or heuristic methods. Extensive experiments on both vision and language benchmarks demonstrate that \acem sets a new state-of-the-art among data-free methods. It consistently outperforms existing baselines; for example, \acem achieves an average absolute improvement of 4\% over the previous methods across seven tasks on GPT-2. Owing to its efficient closed-form formulation, \acem delivers superior performance with a modest computational cost, providing a practical and theoretically grounded solution for model merging. 

---
# Learning to Generate and Extract: A Multi-Agent Collaboration Framework For Zero-shot Document-level Event Arguments Extraction 

**Authors**: Guangjun Zhang, Hu Zhang, Yazhou Han, Yue Fan, Yuhang Shao, Ru Li, Hongye Tan  

**Link**: [PDF](https://arxiv.org/pdf/2603.02909)  

**Abstract**: Document-level event argument extraction (DEAE) is essential for knowledge acquisition, aiming to extract participants of events from this http URL the zero-shot setting, existing methods employ LLMs to generate synthetic data to address the challenge posed by the scarcity of annotated data. However, relying solely on Event-type-only prompts makes it difficult for the generated content to accurately capture the contextual and structural relationships of unseen events. Moreover, ensuring the reliability and usability of synthetic data remains a significant challenge due to the absence of quality evaluation mechanisms. To this end, we introduce a multi-agent collaboration framework for zero-shot document-level event argument extraction (ZS-DEAE), which simulates the human collaborative cognitive process of "Propose-Evaluate-Revise." Specifically, the framework comprises a generation agent and an evaluation agent. The generation agent synthesizes data for unseen events by leveraging knowledge from seen events, while the evaluation agent extracts arguments from the synthetic data and assesses their semantic consistency with the context. The evaluation results are subsequently converted into reward signals, with event structure constraints incorporated into the reward design to enable iterative optimization of both agents via reinforcement this http URL three zero-shot scenarios constructed from the RAMS and WikiEvents datasets, our method achieves improvements both in data generation quality and argument extraction performance, while the generated data also effectively enhances the zero-shot performance of other DEAE models. 

---
# Eval4Sim: An Evaluation Framework for Persona Simulation 

**Authors**: Eliseo Bao, Anxo Perez, Xi Wang, Javier Parapar  

**Link**: [PDF](https://arxiv.org/pdf/2603.02876)  

**Abstract**: Large Language Model (LLM) personas with explicit specifications of attributes, background, and behavioural tendencies are increasingly used to simulate human conversations for tasks such as user modeling, social reasoning, and behavioural analysis. Ensuring that persona-grounded simulations faithfully reflect human conversational behaviour is therefore critical. However, current evaluation practices largely rely on LLM-as-a-judge approaches, offering limited grounding in observable human behavior and producing opaque scalar scores. We address this gap by proposing Eval4Sim, an evaluation framework that measures how closely simulated conversations align with human conversational patterns across three complementary dimensions. Adherence captures how effectively persona backgrounds are implicitly encoded in generated utterances, assessed via dense retrieval with speaker-aware representations. Consistency evaluates whether a persona maintains a distinguishable identity across conversations, computed through authorship verification. Naturalness reflects whether conversations exhibit human-like flow rather than overly rigid or optimized structure, quantified through distributions derived from dialogue-focused Natural Language Inference. Unlike absolute or optimization-oriented metrics, Eval4Sim uses a human conversational corpus (i.e., PersonaChat) as a reference baseline and penalizes deviations in both directions, distinguishing insufficient persona encoding from over-optimized, unnatural behaviour. Although demonstrated on PersonaChat, the applicability of Eval4Sim extends to any conversational corpus containing speaker-level annotations. 

---
# LaTeX Compilation: Challenges in the Era of LLMs 

**Authors**: Tianyou Liu, Ziqiang Li, Yansong Li, Xurui Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.02873)  

**Abstract**: As large language models (LLMs) increasingly assist scientific writing, limitations and the significant token cost of TeX become more and more visible. This paper analyzes TeX's fundamental defects in compilation and user experience design to illustrate its limitations on compilation efficiency, generated semantics, error localization, and tool ecosystem in the era of LLMs. As an alternative, Mogan STEM, a WYSIWYG structured editor, is introduced. Mogan outperforms TeX in the above aspects by its efficient data structure, fast rendering, and on-demand plugin loading. Extensive experiments are conducted to verify the benefits on compilation/rendering time and performance in LLM tasks. What's more, we show that due to Mogan's lower information entropy, it is more efficient to use .tmu (the document format of Mogan) to fine-tune LLMs than TeX. Therefore, we launch an appeal for larger experiments on LLM training using the .tmu format. 

---
# Nodes Are Early, Edges Are Late: Probing Diagram Representations in Large Vision-Language Models 

**Authors**: Haruto Yoshida, Keito Kudo, Yoichi Aoki, Ryota Tanaka, Itsumi Saito, Keisuke Sakaguchi, Kentaro Inui  

**Link**: [PDF](https://arxiv.org/pdf/2603.02865)  

**Abstract**: Large vision-language models (LVLMs) demonstrate strong performance on diagram understanding benchmarks, yet they still struggle with understanding relationships between elements, particularly those represented by nodes and directed edges (e.g., arrows and lines). To investigate the underlying causes of this limitation, we probe the internal representation of LVLMs using a carefully constructed synthetic diagram dataset based on directed graphs. Our probing experiments reveal that edge information is not linearly separable in the vision encoder and becomes linearly encoded only in the text tokens in the language model. In contrast, node information and global structural features are already linearly encoded in individual hidden states of the vision encoder. These findings suggest that the stage at which linearly separable representations are formed varies depending on the type of visual information. In particular, the delayed emergence of edge representations may help explain why LVLMs struggle with relational understanding, such as interpreting edge directions, which require more abstract, compositionally integrated processes. 

---
# The Distribution of Phoneme Frequencies across the World's Languages: Macroscopic and Microscopic Information-Theoretic Models 

**Authors**: Fermín Moscoso del Prado Martín, Suchir Salhan  

**Link**: [PDF](https://arxiv.org/pdf/2603.02860)  

**Abstract**: We demonstrate that the frequency distribution of phonemes across languages can be explained at both macroscopic and microscopic levels. Macroscopically, phoneme rank-frequency distributions closely follow the order statistics of a symmetric Dirichlet distribution whose single concentration parameter scales systematically with phonemic inventory size, revealing a robust compensation effect whereby larger inventories exhibit lower relative entropy. Microscopically, a Maximum Entropy model incorporating constraints from articulatory, phonotactic, and lexical structure accurately predicts language-specific phoneme probabilities. Together, these findings provide a unified information-theoretic account of phoneme frequency structure. 

---
# A Browser-based Open Source Assistant for Multimodal Content Verification 

**Authors**: Rosanna Milner, Michael Foster, Olesya Razuvayevskaya, Ian Roberts, Valentin Porcellini, Denis Teyssou, Kalina Bontcheva  

**Link**: [PDF](https://arxiv.org/pdf/2603.02842)  

**Abstract**: Disinformation and false content produced by generative AI pose a significant challenge for journalists and fact-checkers who must rapidly verify digital media information. While there is an abundance of NLP models for detecting credibility signals such as persuasion techniques, subjectivity, or machine-generated text, such methods often remain inaccessible to non-expert users and are not integrated into their daily workflows as a unified framework. This paper demonstrates the VERIFICATION ASSISTANT, a browser-based tool designed to bridge this gap. The VERIFICATION ASSISTANT, a core component of the widely adopted VERIFICATION PLUGIN (140,000+ users), allows users to submit URLs or media files to a unified interface. It automatically extracts content and routes it to a suite of backend NLP classifiers, delivering actionable credibility signals, estimating AI-generated content, and providing other verification guidance in a clear, easy-to-digest format. This paper showcases the tool architecture, its integration of multiple NLP services, and its real-world application to detecting disinformation. 

---
# Faster, Cheaper, More Accurate: Specialised Knowledge Tracing Models Outperform LLMs 

**Authors**: Prarthana Bhattacharyya, Joshua Mitton, Ralph Abboud, Simon Woodhead  

**Link**: [PDF](https://arxiv.org/pdf/2603.02830)  

**Abstract**: Predicting future student responses to questions is particularly valuable for educational learning platforms where it enables effective interventions. One of the key approaches to do this has been through the use of knowledge tracing (KT) models. These are small, domain-specific, temporal models trained on student question-response data. KT models are optimised for high accuracy on specific educational domains and have fast inference and scalable deployments. The rise of Large Language Models (LLMs) motivates us to ask the following questions: (1) How well can LLMs perform at predicting students' future responses to questions? (2) Are LLMs scalable for this domain? (3) How do LLMs compare to KT models on this domain-specific task? In this paper, we compare multiple LLMs and KT models across predictive performance, deployment cost, and inference speed to answer the above questions. We show that KT models outperform LLMs with respect to accuracy and F1 scores on this domain-specific task. Further, we demonstrate that LLMs are orders of magnitude slower than KT models and cost orders of magnitude more to deploy. This highlights the importance of domain-specific models for education prediction tasks and the fact that current closed source LLMs should not be used as a universal solution for all tasks. 

---
# OCR or Not? Rethinking Document Information Extraction in the MLLMs Era with Real-World Large-Scale Datasets 

**Authors**: Jiyuan Shen, Peiyue Yuan, Atin Ghosh, Yifan Mai, Daniel Dahlmeier  

**Link**: [PDF](https://arxiv.org/pdf/2603.02789)  

**Abstract**: Multimodal Large Language Models (MLLMs) enhance the potential of natural language processing. However, their actual impact on document information extraction remains unclear. In particular, it is unclear whether an MLLM-only pipeline--while simpler--can truly match the performance of traditional OCR+MLLM setups. In this paper, we conduct a large-scale benchmarking study that evaluates various out-of-the-box MLLMs on business-document information extraction. To examine and explore failure modes, we propose an automated hierarchical error analysis framework that leverages large language models (LLMs) to diagnose error patterns systematically. Our findings suggest that OCR may not be necessary for powerful MLLMs, as image-only input can achieve comparable performance to OCR-enhanced approaches. Moreover, we demonstrate that carefully designed schema, exemplars, and instructions can further enhance MLLMs performance. We hope this work can offer practical guidance and valuable insight for advancing document information extraction. 

---
# From Solver to Tutor: Evaluating the Pedagogical Intelligence of LLMs with KMP-Bench 

**Authors**: Weikang Shi, Houxing Ren, Junting Pan, Aojun Zhou, Ke Wang, Zimu Lu, Yunqiao Yang, Yuxuan Hu, Linda Wei, Mingjie Zhan, Hongsheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.02775)  

**Abstract**: Large Language Models (LLMs) show significant potential in AI mathematical tutoring, yet current evaluations often rely on simplistic metrics or narrow pedagogical scenarios, failing to assess comprehensive, multi-turn teaching effectiveness. In this paper, we introduce KMP-Bench, a comprehensive K-8 Mathematical Pedagogical Benchmark designed to assess LLMs from two complementary perspectives. The first module, KMP-Dialogue, evaluates holistic pedagogical capabilities against six core principles (e.g., Challenge, Explanation, Feedback), leveraging a novel multi-turn dialogue dataset constructed by weaving together diverse pedagogical components. The second module, KMP-Skills, provides a granular assessment of foundational tutoring abilities, including multi-turn problem-solving, error detection and correction, and problem generation. Our evaluations on KMP-Bench reveal a key disparity: while leading LLMs excel at tasks with verifiable solutions, they struggle with the nuanced application of pedagogical principles. Additionally, we present KMP-Pile, a large-scale (150K) dialogue dataset. Models fine-tuned on KMP-Pile show substantial improvement on KMP-Bench, underscoring the value of pedagogically-rich training data for developing more effective AI math tutors. 

---
# Efficient Self-Evaluation for Diffusion Language Models via Sequence Regeneration 

**Authors**: Linhao Zhong, Linyu Wu, Wen Wang, Yuling Xi, Chenchen Jing, Jiaheng Zhang, Hao Chen, Chunhua Shen  

**Link**: [PDF](https://arxiv.org/pdf/2603.02760)  

**Abstract**: Diffusion large language models (dLLMs) have recently attracted significant attention for their ability to enhance diversity, controllability, and parallelism. However, their non-sequential, bidirectionally masked generation makes quality assessment difficult, underscoring the need for effective self-evaluation. In this work, we propose DiSE, a simple yet effective self-evaluation confidence quantification method for dLLMs. DiSE quantifies confidence by computing the probability of regenerating the tokens in the entire generated sequence, given the full context. This method enables more efficient and reliable quality assessment by leveraging token regeneration probabilities, facilitating both likelihood estimation and robust uncertainty quantification. Building upon DiSE, we further introduce a flexible-length generation framework, which adaptively controls the sequence length based on the model's self-assessment of its own output. We analyze and validate the feasibility of DiSE from the perspective of dLLM generalization, and empirically demonstrate that DiSE is positively correlated with both semantic coherence and answer accuracy. Extensive experiments on likelihood evaluation, uncertainty quantification, and flexible-length generation further confirm the effectiveness of the proposed DiSE. 

---
# Sensory-Aware Sequential Recommendation via Review-Distilled Representations 

**Authors**: Yeo Chan Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2603.02709)  

**Abstract**: We propose a novel framework for sensory-aware sequential recommendation that enriches item representations with linguistically extracted sensory attributes from product reviews. Our approach, \textsc{ASEGR} (Attribute-based Sensory Enhanced Generative Recommendation), introduces a two-stage pipeline in which a large language model is first fine-tuned as a teacher to extract structured sensory attribute--value pairs, such as \textit{color: matte black} and \textit{scent: vanilla}, from unstructured review text. The extracted structures are then distilled into a compact student transformer that produces fixed-dimensional sensory embeddings for each item. These embeddings encode experiential semantics in a reusable form and are incorporated into standard sequential recommender architectures as additional item-level representations. We evaluate our method on four Amazon domains and integrate the learned sensory embeddings into representative sequential recommendation models, including SASRec, BERT4Rec, and BSARec. Across domains, sensory-enhanced models consistently outperform their identifier-based counterparts, indicating that linguistically grounded sensory representations provide complementary signals to behavioral interaction patterns. Qualitative analysis further shows that the extracted attributes align closely with human perceptions of products, enabling interpretable connections between natural language descriptions and recommendation behavior. Overall, this work demonstrates that sensory attribute distillation offers a principled and scalable way to bridge information extraction and sequential recommendation through structured semantic representation learning. 

---
# Graph-GRPO: Stabilizing Multi-Agent Topology Learning via Group Relative Policy Optimization 

**Authors**: Yueyang Cang, Xiaoteng Zhang, Erlu Zhao, Zehua Ji, Yuhang Liu, Yuchen He, Zhiyuan Ning, Chen Yijun, Wenge Que, Li Shi  

**Link**: [PDF](https://arxiv.org/pdf/2603.02701)  

**Abstract**: Optimizing communication topology is fundamental to the efficiency and effectiveness of Large Language Model (LLM)-based Multi-Agent Systems (MAS). While recent approaches utilize reinforcement learning to dynamically construct task-specific graphs, they typically rely on single-sample policy gradients with absolute rewards (e.g., binary correctness). This paradigm suffers from severe gradient variance and the credit assignment problem: simple queries yield non-informative positive rewards for suboptimal structures, while difficult queries often result in failures that provide no learning signal. To address these challenges, we propose Graph-GRPO, a novel topology optimization framework that integrates Group Relative Policy Optimization. Instead of evaluating a single topology in isolation, Graph-GRPO samples a group of diverse communication graphs for each query and computes the advantage of specific edges based on their relative performance within the group. By normalizing rewards across the sampled group, our method effectively mitigates the noise derived from task difficulty variance and enables fine-grained credit assignment. Extensive experiments on reasoning and code generation benchmarks demonstrate that Graph-GRPO significantly outperforms state-of-the-art baselines, achieving superior training stability and identifying critical communication pathways previously obscured by reward noise. 

---
# HateMirage: An Explainable Multi-Dimensional Dataset for Decoding Faux Hate and Subtle Online Abuse 

**Authors**: Sai Kartheek Reddy Kasu, Shankar Biradar, Sunil Saumya, Md. Shad Akhtar  

**Link**: [PDF](https://arxiv.org/pdf/2603.02684)  

**Abstract**: Subtle and indirect hate speech remains an underexplored challenge in online safety research, particularly when harmful intent is embedded within misleading or manipulative narratives. Existing hate speech datasets primarily capture overt toxicity, underrepresenting the nuanced ways misinformation can incite or normalize hate. To address this gap, we present HateMirage, a novel dataset of Faux Hate comments designed to advance reasoning and explainability research on hate emerging from fake or distorted narratives. The dataset was constructed by identifying widely debunked misinformation claims from fact-checking sources and tracing related YouTube discussions, resulting in 4,530 user comments. Each comment is annotated along three interpretable dimensions: Target (who is affected), Intent (the underlying motivation or goal behind the comment), and Implication (its potential social impact). Unlike prior explainability datasets such as HateXplain and HARE, which offer token-level or single-dimensional reasoning, HateMirage introduces a multi-dimensional explanation framework that captures the interplay between misinformation, harm, and social consequence. We benchmark multiple open-source language models on HateMirage using ROUGE-L F1 and Sentence-BERT similarity to assess explanation coherence. Results suggest that explanation quality may depend more on pretraining diversity and reasoning-oriented data rather than on model scale alone. By coupling misinformation reasoning with harm attribution, HateMirage establishes a new benchmark for interpretable hate detection and responsible AI research. 

---
# ITLC at SemEval-2026 Task 11: Normalization and Deterministic Parsing for Formal Reasoning in LLMs 

**Authors**: Wicaksono Leksono Muhamad, Joanito Agili Lopo, Tack Hwa Wong, Muhammad Ravi Shulthan Habibi, Samuel Cahyawijaya  

**Link**: [PDF](https://arxiv.org/pdf/2603.02676)  

**Abstract**: Large language models suffer from content effects in reasoning tasks, particularly in multi-lingual contexts. We introduce a novel method that reduces these biases through explicit structural abstraction that transforms syllogisms into canonical logical representations and applies deterministic parsing to determine validity. Evaluated on the SemEval-2026 Task 11 multilingual benchmark, our approach achieves top-5 rankings across all subtasks while substantially reducing content effects and offering a competitive alternative to complex fine-tuning or activation-level interventions. 

---
# Evaluating Cross-Modal Reasoning Ability and Problem Characteristics with Multimodal Item Response Theory 

**Authors**: Shunki Uebayashi, Kento Masui, Kyohei Atarashi, Han Bao, Hisashi Kashima, Naoto Inoue, Mayu Otani, Koh Takeuchi  

**Link**: [PDF](https://arxiv.org/pdf/2603.02663)  

**Abstract**: Multimodal Large Language Models (MLLMs) have recently emerged as general architectures capable of reasoning over diverse modalities. Benchmarks for MLLMs should measure their ability for cross-modal integration. However, current benchmarks are filled with shortcut questions, which can be solved using only a single modality, thereby yielding unreliable rankings. For example, in vision-language cases, we can find the correct answer without either the image or the text. These low-quality questions unnecessarily increase the size and computational requirements of benchmarks. We introduce a multi-modal and multidimensional item response theory framework (M3IRT) that extends classical IRT by decomposing both model ability and item difficulty into image-only, text-only, and cross-modal components. M3IRT estimates cross-modal ability of MLLMs and each question's cross-modal difficulty, enabling compact, high-quality subsets that better reflect multimodal reasoning. Across 24 VLMs on three benchmarks, M3IRT prioritizes genuinely cross-modal questions over shortcuts and preserves ranking fidelity even when 50% of items are artificially generated low-quality questions, thereby reducing evaluation cost while improving reliability. M3IRT thus offers a practical tool for assessing cross-modal reasoning and refining multimodal benchmarks. 

---
# Real-Time Generation of Game Video Commentary with Multimodal LLMs: Pause-Aware Decoding Approaches 

**Authors**: Anum Afzal, Yuki Saito, Hiroya Takamura, Katsuhito Sudoh, Shinnosuke Takamichi, Graham Neubig, Florian Matthes, Tatsuya Ishigaki  

**Link**: [PDF](https://arxiv.org/pdf/2603.02655)  

**Abstract**: Real-time video commentary generation provides textual descriptions of ongoing events in videos. It supports accessibility and engagement in domains such as sports, esports, and livestreaming. Commentary generation involves two essential decisions: what to say and when to say it. While recent prompting-based approaches using multimodal large language models (MLLMs) have shown strong performance in content generation, they largely ignore the timing aspect. We investigate whether in-context prompting alone can support real-time commentary generation that is both semantically relevant and well-timed. We propose two prompting-based decoding strategies: 1) a fixed-interval approach, and 2) a novel dynamic interval-based decoding approach that adjusts the next prediction timing based on the estimated duration of the previous utterance. Both methods enable pause-aware generation without any fine-tuning. Experiments on Japanese and English datasets of racing and fighting games show that the dynamic interval-based decoding can generate commentary more closely aligned with human utterance timing and content using prompting alone. We release a multilingual benchmark dataset, trained models, and implementations to support future research on real-time video commentary generation. 

---
# Cross-Family Speculative Prefill: Training-Free Long-Context Compression with Small Draft Models 

**Authors**: Shubhangi Upasani, Ravi Shanker Raju, Bo Li, Mengmeing Ji, John Long, Chen Wu, Urmish Thakker, Guangtao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.02631)  

**Abstract**: Prompt length is a major bottleneck in agentic large language model (LLM) workloads, where repeated inference steps and multi-call loops incur substantial prefill cost. Recent work on speculative prefill demonstrates that attention-based token importance estimation can enable training-free prompt compression, but this assumes the existence of a draft model that shares the same tokenizer as the target model. In practice, however, agentic pipelines frequently employ models without any smaller in-family draft model. In this work, we study cross-family speculative prefill, where a lightweight draft model from one model family is used to perform prompt compression for a target model from a different family. Using the same speculative prefill mechanism as prior work, we evaluate a range of cross-family draft-target combinations, including Qwen, LLaMA, and DeepSeek models. Across a broad diversity of tasks, we find that attention-based token importance estimation transfers reliably across different model families despite differences in model architectures and tokenizers between draft and target models. Cross-model prompt compression largely retains 90~100% of full-prompt baseline performance and, in some cases, slightly improves accuracy due to denoising effects, while delivering substantial reductions in time to first token (TTFT). These results suggest that speculative prefill depends mainly on task priors and semantic structure, thus serving as a generalizable prompt compression primitive. We discuss the implications of our findings for agentic systems, where repeated long-context inference and heterogeneous model stacks make cross-model prompt compression both necessary and practical. 

---
# Think, But Don't Overthink: Reproducing Recursive Language Models 

**Authors**: Daren Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.02615)  

**Abstract**: This project reproduces and extends the recently proposed ``Recursive Language Models'' (RLMs) framework by Zhang et al. (2026). This framework enables Large Language Models (LLMs) to process near-infinite contexts by offloading the prompt into an external REPL environment. While the original paper relies on a default recursion depth of 1 and suggests deeper recursion as a future direction, this study specifically investigates the impact of scaling the recursion depth. Using state-of-the-art open-source agentic models (DeepSeek v3.2 and Kimi K2), I evaluated pure LLM, RLM (depth=1), and RLM (depth=2) on the S-NIAH and OOLONG benchmarks. The findings reveal a compelling phenomenon: Deeper recursion causes models to ``overthink''. While depth-1 RLMs effectively boost accuracy on complex reasoning tasks, applying deeper recursion (depth=2) or using RLMs on simple retrieval tasks paradoxically degrades performance and exponentially inflates execution time (e.g., from 3.6s to 344.5s) and token costs. Code and data are available at: this https URL 

---
# GPUTOK: GPU Accelerated Byte Level BPE Tokenization 

**Authors**: Venu Gopal Kadamba, Kanishkha Jaisankar  

**Link**: [PDF](https://arxiv.org/pdf/2603.02597)  

**Abstract**: As large language models move toward million-token context windows, CPU tokenizers become a major slowdown because they process text one step at a time while powerful GPUs sit unused. We built a GPU-based byte-level BPE tokenizer that follows GPT-2's merge rules. It includes a basic BlockBPE-style kernel and a faster, optimized version that uses cuCollections static map, CUB reductions, and a pybind11 interface for Python.
On WikiText103 sequences up to 131k tokens, the optimized GPU tokenizer produces the same tokens as a CPU version and, for the longest inputs, is about 1.7x faster than tiktoken and about 7.6x faster than the HuggingFace GPT-2 tokenizer. Nsight profiling shows that 70-80% of CUDA API time goes to memory allocation, so adding memory pooling should give the biggest speed boost next. Tests on generation tasks using WikiText103 prompts show that our GPU tokenizer's outputs stay within about one percentage point of tiktoken and HuggingFace GPT-2 on similarity and overlap metrics, meaning it keeps output quality while making long-context inference more practical. 

---
# ExpGuard: LLM Content Moderation in Specialized Domains 

**Authors**: Minseok Choi, Dongjin Kim, Seungbin Yang, Subin Kim, Youngjun Kwak, Juyoung Oh, Jaegul Choo, Jungmin Son  

**Link**: [PDF](https://arxiv.org/pdf/2603.02588)  

**Abstract**: With the growing deployment of large language models (LLMs) in real-world applications, establishing robust safety guardrails to moderate their inputs and outputs has become essential to ensure adherence to safety policies. Current guardrail models predominantly address general human-LLM interactions, rendering LLMs vulnerable to harmful and adversarial content within domain-specific contexts, particularly those rich in technical jargon and specialized concepts. To address this limitation, we introduce ExpGuard, a robust and specialized guardrail model designed to protect against harmful prompts and responses across financial, medical, and legal domains. In addition, we present ExpGuardMix, a meticulously curated dataset comprising 58,928 labeled prompts paired with corresponding refusal and compliant responses, from these specific sectors. This dataset is divided into two subsets: ExpGuardTrain, for model training, and ExpGuardTest, a high-quality test set annotated by domain experts to evaluate model robustness against technical and domain-specific content. Comprehensive evaluations conducted on ExpGuardTest and eight established public benchmarks reveal that ExpGuard delivers competitive performance across the board while demonstrating exceptional resilience to domain-specific adversarial attacks, surpassing state-of-the-art models such as WildGuard by up to 8.9% in prompt classification and 15.3% in response classification. To encourage further research and development, we open-source our code, data, and model, enabling adaptation to additional domains and supporting the creation of increasingly robust guardrail models. 

---
# How Controllable Are Large Language Models? A Unified Evaluation across Behavioral Granularities 

**Authors**: Ziwen Xu, Kewei Xu, Haoming Xu, Haiwen Hong, Longtao Huang, Hui Xue, Ningyu Zhang, Yongliang Shen, Guozhou Zheng, Huajun Chen, Shumin Deng  

**Link**: [PDF](https://arxiv.org/pdf/2603.02578)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed in socially sensitive domains, yet their unpredictable behaviors, ranging from misaligned intent to inconsistent personality, pose significant risks. We introduce SteerEval, a hierarchical benchmark for evaluating LLM controllability across three domains: language features, sentiment, and personality. Each domain is structured into three specification levels: L1 (what to express), L2 (how to express), and L3 (how to instantiate), connecting high-level behavioral intent to concrete textual output. Using SteerEval, we systematically evaluate contemporary steering methods, revealing that control often degrades at finer-grained levels. Our benchmark offers a principled and interpretable framework for safe and controllable LLM behavior, serving as a foundation for future research. 

---
# CoDAR: Continuous Diffusion Language Models are More Powerful Than You Think 

**Authors**: Junzhe Shen, Jieru Zhao, Ziwei He, Zhouhan Lin  

**Link**: [PDF](https://arxiv.org/pdf/2603.02547)  

**Abstract**: We study why continuous diffusion language models (DLMs) have lagged behind discrete diffusion approaches despite their appealing continuous generative dynamics. Under a controlled token--recovery study, we identify token rounding, the final projection from denoised embeddings to tokens, as a primary bottleneck. Building on these insights, we propose CoDAR (Continuous Diffusion with Contextual AutoRegressive Decoder), a two--stage framework that keeps diffusion entirely continuous in an embedding space while learning a strong, context--conditional discretizer: an autoregressive Transformer decoder that cross--attends to the denoised embedding sequence and performs contextualized rounding to tokens. Experiments on LM1B and OpenWebText demonstrate that CoDAR substantially improves generation quality over latent diffusion and becomes competitive with strong discrete DLMs, while exposing a simple decoder--temperature knob to navigate the fluency--diversity trade off. 

---
# GLoRIA: Gated Low-Rank Interpretable Adaptation for Dialectal ASR 

**Authors**: Pouya Mehralian, Melissa Farasyn, Anne Breitbarth, Anne-Sophie Ghyselen, Hugo Van hamme  

**Link**: [PDF](https://arxiv.org/pdf/2603.02464)  

**Abstract**: Automatic Speech Recognition (ASR) in dialect-heavy settings remains challenging due to strong regional variation and limited labeled data. We propose GLoRIA, a parameter-efficient adaptation framework that leverages metadata (e.g., coordinates) to modulate low-rank updates in a pre-trained encoder. GLoRIA injects low-rank matrices into each feed-forward layer, with a gating MLP determining the non-negative contribution of each LoRA rank-1 component based on location metadata. On the GCND corpus, GLoRIA outperforms geo-conditioned full fine-tuning, LoRA, and both dialect-specific and unified full fine-tuning, achieving state-of-the-art word error rates while updating under 10% of parameters. GLoRIA also generalizes well to unseen dialects, including in extrapolation scenarios, and enables interpretable adaptation patterns that can be visualized geospatially. These results show metadata-gated low-rank adaptation is an effective, interpretable, and efficient solution for dialectal ASR. 

---
# RO-N3WS: Enhancing Generalization in Low-Resource ASR with Diverse Romanian Speech Benchmarks 

**Authors**: Alexandra Diaconu, Mădălina Vînaga, Bogdan Alexe  

**Link**: [PDF](https://arxiv.org/pdf/2603.02368)  

**Abstract**: We introduce RO-N3WS, a benchmark Romanian speech dataset designed to improve generalization in automatic speech recognition (ASR), particularly in low-resource and out-of-distribution (OOD) conditions. RO-N3WS comprises over 126 hours of transcribed audio collected from broadcast news, literary audiobooks, film dialogue, children's stories, and conversational podcast speech. This diversity enables robust training and fine-tuning across stylistically distinct domains. We evaluate several state-of-the-art ASR systems (Whisper, Wav2Vec 2.0) in both zero-shot and fine-tuned settings, and conduct controlled comparisons using synthetic data generated with expressive TTS models. Our results show that even limited fine-tuning on real speech from RO-N3WS yields substantial WER improvements over zero-shot baselines. We will release all models, scripts, and data splits to support reproducible research in multilingual ASR, domain adaptation, and lightweight deployment. 

---
# Detecting AI-Generated Essays in Writing Assessment: Responsible Use and Generalizability Across LLMs 

**Authors**: Jiangang Hao  

**Link**: [PDF](https://arxiv.org/pdf/2603.02353)  

**Abstract**: Writing is a foundational literacy skill that underpins effective communication, fosters critical thinking, facilitates learning across disciplines, and enables individuals to organize and articulate complex ideas. Consequently, writing assessment plays a vital role in evaluating language proficiency, communicative effectiveness, and analytical reasoning. The rapid advancement of large language models (LLMs) has made it increasingly easy to generate coherent, high-quality essays, raising significant concerns about the authenticity of student-submitted work. This chapter first provides an overview of the current landscape of detectors for AI-generated and AI-assisted essays, along with guidelines for their responsible use. It then presents empirical analyses to evaluate how well detectors trained on essays from one LLM generalize to identifying essays produced by other LLMs, based on essays generated in response to public GRE writing prompts. These findings provide guidance for developing and retraining detectors for practical applications. 

---
# Characterizing Memorization in Diffusion Language Models: Generalized Extraction and Sampling Effects 

**Authors**: Xiaoyu Luo, Wenrui Yu, Qiongxiu Li, Johannes Bjerva  

**Link**: [PDF](https://arxiv.org/pdf/2603.02333)  

**Abstract**: Autoregressive language models (ARMs) have been shown to memorize and occasionally reproduce training data verbatim, raising concerns about privacy and copyright liability. Diffusion language models (DLMs) have recently emerged as a competitive alternative, yet their memorization behavior remains largely unexplored due to fundamental differences in generation dynamics. To address this gap, we present a systematic theoretical and empirical characterization of memorization in DLMs. We propose a generalized probabilistic extraction framework that unifies prefix-conditioned decoding and diffusion-based generation under arbitrary masking patterns and stochastic sampling trajectories. Theorem 4.3 establishes a monotonic relationship between sampling resolution and memorization: increasing resolution strictly increases the probability of exact training data extraction, implying that autoregressive decoding corresponds to a limiting case of diffusion-based generation by setting the sampling resolution maximal. Extensive experiments across model scales and sampling strategies validate our theoretical predictions. Under aligned prefix-conditioned evaluations, we further demonstrate that DLMs exhibit substantially lower memorization-based leakage of personally identifiable information (PII) compared to ARMs. 

---
# Universal Conceptual Structure in Neural Translation: Probing NLLB-200's Multilingual Geometry 

**Authors**: Kyle Elliott Mathewson  

**Link**: [PDF](https://arxiv.org/pdf/2603.02258)  

**Abstract**: Do neural machine translation models learn language-universal conceptual representations, or do they merely cluster languages by surface similarity? We investigate this question by probing the representation geometry of Meta's NLLB-200, a 200-language encoder-decoder Transformer, through six experiments that bridge NLP interpretability with cognitive science theories of multilingual lexical organization. Using the Swadesh core vocabulary list embedded across 135 languages, we find that the model's embedding distances significantly correlate with phylogenetic distances from the Automated Similarity Judgment Program ($\rho = 0.13$, $p = 0.020$), demonstrating that NLLB-200 has implicitly learned the genealogical structure of human languages. We show that frequently colexified concept pairs from the CLICS database exhibit significantly higher embedding similarity than non-colexified pairs ($U = 42656$, $p = 1.33 \times 10^{-11}$, $d = 0.96$), indicating that the model has internalized universal conceptual associations. Per-language mean-centering of embeddings improves the between-concept to within-concept distance ratio by a factor of 1.19, providing geometric evidence for a language-neutral conceptual store analogous to the anterior temporal lobe hub identified in bilingual neuroimaging. Semantic offset vectors between fundamental concept pairs (e.g., man to woman, big to small) show high cross-lingual consistency (mean cosine = 0.84), suggesting that second-order relational structure is preserved across typologically diverse languages. We release InterpretCognates, an open-source interactive toolkit for exploring these phenomena, alongside a fully reproducible analysis pipeline. 

---
# A Zipf-preserving, long-range correlated surrogate for written language and other symbolic sequences 

**Authors**: Marcelo A. Montemurro, Mirko Degli Esposti  

**Link**: [PDF](https://arxiv.org/pdf/2603.02213)  

**Abstract**: Symbolic sequences such as written language and genomic DNA display characteristic frequency distributions and long-range correlations extending over many symbols. In language, this takes the form of Zipf's law for word frequencies together with persistent correlations spanning hundreds or thousands of tokens, while in DNA it is reflected in nucleotide composition and long-memory walks under purine-pyrimidine mappings. Existing surrogate models usually preserve either the frequency distribution or the correlation properties, but not both simultaneously.
We introduce a surrogate model that retains both constraints: it preserves the empirical symbol frequencies of the original sequence and reproduces its long-range correlation structure, quantified by the detrended fluctuation analysis (DFA) exponent. Our method generates surrogates of symbolic sequences by mapping fractional Gaussian noise (FGN) onto the empirical histogram through a frequency-preserving assignment. The resulting surrogates match the original in first-order statistics and long-range scaling while randomising short-range dependencies.
We validate the model on representative texts in English and Latin, and illustrate its broader applicability with genomic DNA, showing that base composition and DFA scaling are reproduced. This approach provides a principled tool for disentangling structural features of symbolic systems and for testing hypotheses on the origin of scaling laws and memory effects across language, DNA, and other symbolic domains. 

---
# Density-Guided Response Optimization: Community-Grounded Alignment via Implicit Acceptance Signals 

**Authors**: Patrick Gerard, Svitlana Volkova  

**Link**: [PDF](https://arxiv.org/pdf/2603.03242)  

**Abstract**: Language models deployed in online communities must adapt to norms that vary across social, cultural, and domain-specific contexts. Prior alignment approaches rely on explicit preference supervision or predefined principles, which are effective for well-resourced settings but exclude most online communities -- particularly those without institutional backing, annotation infrastructure, or organized around sensitive topics -- where preference elicitation is costly, ethically fraught, or culturally misaligned.
We observe that communities already express preferences implicitly through what content they accept, engage with, and allow to persist. We show that this acceptance behavior induces measurable geometric structure in representation space: accepted responses occupy coherent, high-density regions that reflect community-specific norms, while rejected content falls in sparser or misaligned areas. We operationalize this structure as an implicit preference signal for alignment and introduce density-guided response optimization (DGRO), a method that aligns language models to community norms without requiring explicit preference labels.
Using labeled preference data, we demonstrate that local density recovers pairwise community judgments, indicating that geometric structure encodes meaningful preference signal. We then apply DGRO in annotation-scarce settings across diverse communities spanning platform, topic, and language. DGRO-aligned models consistently produce responses preferred by human annotators, domain experts, and model-based judges over supervised and prompt-based baselines. We position DGRO as a practical alignment alternative for communities where explicit preference supervision is unavailable or misaligned with situated practices, and discuss the implications and risks of learning from emergent acceptance behavior. 

---
# Understanding and Mitigating Dataset Corruption in LLM Steering 

**Authors**: Cullen Anderson, Narmeen Oozeer, Foad Namjoo, Remy Ogasawara, Amirali Abdullah, Jeff M. Phillips  

**Link**: [PDF](https://arxiv.org/pdf/2603.03206)  

**Abstract**: Contrastive steering has been shown as a simple and effective method to adjust the generative behavior of LLMs at inference time. It uses examples of prompt responses with and without a trait to identify a direction in an intermediate activation layer, and then shifts activations in this 1-dimensional subspace. However, despite its growing use in AI safety applications, the robustness of contrastive steering to noisy or adversarial data corruption is poorly understood. We initiate a study of the robustness of this process with respect to corruption of the dataset of examples used to train the steering direction. Our first observation is that contrastive steering is quite robust to a moderate amount of corruption, but unwanted side effects can be clearly and maliciously manifested when a non-trivial fraction of the training data is altered. Second, we analyze the geometry of various types of corruption, and identify some safeguards. Notably, a key step in learning the steering direction involves high-dimensional mean computation, and we show that replacing this step with a recently developed robust mean estimator often mitigates most of the unwanted effects of malicious corruption. 

---
# No Memorization, No Detection: Output Distribution-Based Contamination Detection in Small Language Models 

**Authors**: Omer Sela  

**Link**: [PDF](https://arxiv.org/pdf/2603.03203)  

**Abstract**: CDD, or Contamination Detection via output Distribution, identifies data contamination by measuring the peakedness of a model's sampled outputs. We study the conditions under which this approach succeeds and fails on small language models ranging from 70M to 410M parameters. Using controlled contamination experiments on GSM8K, HumanEval, and MATH, we find that CDD's effectiveness depends critically on whether fine-tuning produces verbatim memorization. With low-rank adaptation, models can learn from contaminated data without memorizing it, and CDD performs at chance level even when the data is verifiably contaminated. Only when fine-tuning capacity is sufficient to induce memorization does CDD recover strong detection accuracy. Our results characterize a memorization threshold that governs detectability and highlight a practical consideration: parameter-efficient fine-tuning can produce contamination that output-distribution methods do not detect. Our code is available at this https URL 

---
# ACE-Brain-0: Spatial Intelligence as a Shared Scaffold for Universal Embodiments 

**Authors**: Ziyang Gong, Zehang Luo, Anke Tang, Zhe Liu, Shi Fu, Zhi Hou, Ganlin Yang, Weiyun Wang, Xiaofeng Wang, Jianbo Liu, Gen Luo, Haolan Kang, Shuang Luo, Yue Zhou, Yong Luo, Li Shen, Xiaosong Jia, Yao Mu, Xue Yang, Chunxiao Liu, Junchi Yan, Hengshuang Zhao, Dacheng Tao, Xiaogang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.03198)  

**Abstract**: Universal embodied intelligence demands robust generalization across heterogeneous embodiments, such as autonomous driving, robotics, and unmanned aerial vehicles (UAVs). However, existing embodied brain in training a unified model over diverse embodiments frequently triggers long-tail data, gradient interference, and catastrophic forgetting, making it notoriously difficult to balance universal generalization with domain-specific proficiency. In this report, we introduce ACE-Brain-0, a generalist foundation brain that unifies spatial reasoning, autonomous driving, and embodied manipulation within a single multimodal large language model~(MLLM). Our key insight is that spatial intelligence serves as a universal scaffold across diverse physical embodiments: although vehicles, robots, and UAVs differ drastically in morphology, they share a common need for modeling 3D mental space, making spatial cognition a natural, domain-agnostic foundation for cross-embodiment transfer. Building on this insight, we propose the Scaffold-Specialize-Reconcile~(SSR) paradigm, which first establishes a shared spatial foundation, then cultivates domain-specialized experts, and finally harmonizes them through data-free model merging. Furthermore, we adopt Group Relative Policy Optimization~(GRPO) to strengthen the model's comprehensive capability. Extensive experiments demonstrate that ACE-Brain-0 achieves competitive and even state-of-the-art performance across 24 spatial and embodiment-related benchmarks. 

---
# MoD-DPO: Towards Mitigating Cross-modal Hallucinations in Omni LLMs using Modality Decoupled Preference Optimization 

**Authors**: Ashutosh Chaubey, Jiacheng Pang, Mohammad Soleymani  

**Link**: [PDF](https://arxiv.org/pdf/2603.03192)  

**Abstract**: Omni-modal large language models (omni LLMs) have recently achieved strong performance across audiovisual understanding tasks, yet they remain highly susceptible to cross-modal hallucinations arising from spurious correlations and dominant language priors. In this work, we propose Modality-Decoupled Direct Preference Optimization (MoD-DPO), a simple and effective framework for improving modality grounding in omni LLMs. MoD-DPO introduces modality-aware regularization terms that explicitly enforce invariance to corruptions in irrelevant modalities and sensitivity to perturbations in relevant modalities, thereby reducing unintended cross-modal interactions. To further mitigate over-reliance on textual priors, we incorporate a language-prior debiasing penalty that discourages hallucination-prone text-only responses. Extensive experiments across multiple audiovisual hallucination benchmarks demonstrate that MoD-DPO consistently improves perception accuracy and hallucination resistance, outperforming previous preference optimization baselines under similar training budgets. Our findings underscore the importance of modality-faithful alignment and demonstrate a scalable path toward more reliable and resilient multimodal foundation models. 

---
# Type-Aware Retrieval-Augmented Generation with Dependency Closure for Solver-Executable Industrial Optimization Modeling 

**Authors**: Y. Zhong, R. Huang, M. Wang, Z. Guo, YC. Li, M. Yu, Z. Jin  

**Link**: [PDF](https://arxiv.org/pdf/2603.03180)  

**Abstract**: Automated industrial optimization modeling requires reliable translation of natural-language requirements into solver-executable code. However, large language models often generate non-compilable models due to missing declarations, type inconsistencies, and incomplete dependency contexts. We propose a type-aware retrieval-augmented generation (RAG) method that enforces modeling entity types and minimal dependency closure to ensure executability. Unlike existing RAG approaches that index unstructured text, our method constructs a domain-specific typed knowledge base by parsing heterogeneous sources, such as academic papers and solver code, into typed units and encoding their mathematical dependencies in a knowledge graph. Given a natural-language instruction, it performs hybrid retrieval and computes a minimal dependency-closed context, the smallest set of typed symbols required for solver-executable code, via dependency propagation over the graph. We validate the method on two constraint-intensive industrial cases: demand response optimization in battery production and flexible job shop scheduling. In the first case, our method generates an executable model incorporating demand-response incentives and load-reduction constraints, achieving peak shaving while preserving profitability; conventional RAG baselines fail. In the second case, it consistently produces compilable models that reach known optimal solutions, demonstrating robust cross-domain generalization; baselines fail entirely. Ablation studies confirm that enforcing type-aware dependency closure is essential for avoiding structural hallucinations and ensuring executability, addressing a critical barrier to deploying large language models in complex engineering optimization tasks. 

---
# Interpreting Speaker Characteristics in the Dimensions of Self-Supervised Speech Features 

**Authors**: Kyle Janse van Rensburg, Benjamin van Niekerk, Herman Kamper  

**Link**: [PDF](https://arxiv.org/pdf/2603.03096)  

**Abstract**: How do speech models trained through self-supervised learning structure their representations? Previous studies have looked at how information is encoded in feature vectors across different layers. But few studies have considered whether speech characteristics are captured within individual dimensions of SSL features. In this paper we specifically look at speaker information using PCA on utterance-averaged representations. Using WavLM, we find that the principal dimension that explains most variance encodes pitch and associated characteristics like gender. Other individual principal dimensions correlate with intensity, noise levels, the second formant, and higher frequency characteristics. Finally, in synthesis experiments we show that most characteristics can be controlled by changing the corresponding dimensions. This provides a simple method to control characteristics of the output voice in synthesis applications. 

---
# TikZilla: Scaling Text-to-TikZ with High-Quality Data and Reinforcement Learning 

**Authors**: Christian Greisinger, Steffen Eger  

**Link**: [PDF](https://arxiv.org/pdf/2603.03072)  

**Abstract**: Large language models (LLMs) are increasingly used to assist scientists across diverse workflows. A key challenge is generating high-quality figures from textual descriptions, often represented as TikZ programs that can be rendered as scientific images. Prior research has proposed a variety of datasets and modeling approaches for this task. However, existing datasets for Text-to-TikZ are too small and noisy to capture the complexity of TikZ, causing mismatches between text and rendered figures. Moreover, prior approaches rely solely on supervised fine-tuning (SFT), which does not expose the model to the rendered semantics of the figure, often resulting in errors such as looping, irrelevant content, and incorrect spatial relations. To address these issues, we construct DaTikZ-V4, a dataset more than four times larger and substantially higher in quality than DaTikZ-V3, enriched with LLM-generated figure descriptions. Using this dataset, we train TikZilla, a family of small open-source Qwen models (3B and 8B) with a two-stage pipeline of SFT followed by reinforcement learning (RL). For RL, we leverage an image encoder trained via inverse graphics to provide semantically faithful reward signals. Extensive human evaluations with over 1,000 judgments show that TikZilla improves by 1.5-2 points over its base models on a 5-point scale, surpasses GPT-4o by 0.5 points, and matches GPT-5 in the image-based evaluation, while operating at much smaller model sizes. Code, data, and models will be made available. 

---
# Incremental Graph Construction Enables Robust Spectral Clustering of Texts 

**Authors**: Marko Pranjić, Boshko Koloski, Nada Lavrač, Senja Pollak, Marko Robnik-Šikonja  

**Link**: [PDF](https://arxiv.org/pdf/2603.03056)  

**Abstract**: Neighborhood graphs are a critical but often fragile step in spectral clustering of text embeddings. On realistic text datasets, standard $k$-NN graphs can contain many disconnected components at practical sparsity levels (small $k$), making spectral clustering degenerate and sensitive to hyperparameters. We introduce a simple incremental $k$-NN graph construction that preserves connectivity by design: each new node is linked to its $k$ nearest previously inserted nodes, which guarantees a connected graph for any $k$. We provide an inductive proof of connectedness and discuss implications for incremental updates when new documents arrive. We validate the approach on spectral clustering of SentenceTransformer embeddings using Laplacian eigenmaps across six clustering datasets from the Massive Text Embedding this http URL to standard $k$-NN graphs, our method outperforms in the low-$k$ regime where disconnected components are prevalent, and matches standard $k$-NN at larger $k$. 

---
# Contextualized Privacy Defense for LLM Agents 

**Authors**: Yule Wen, Yanzhe Zhang, Jianxun Lian, Xiaoyuan Yi, Xing Xie, Diyi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2603.02983)  

**Abstract**: LLM agents increasingly act on users' personal information, yet existing privacy defenses remain limited in both design and adaptability. Most prior approaches rely on static or passive defenses, such as prompting and guarding. These paradigms are insufficient for supporting contextual, proactive privacy decisions in multi-step agent execution. We propose Contextualized Defense Instructing (CDI), a new privacy defense paradigm in which an instructor model generates step-specific, context-aware privacy guidance during execution, proactively shaping actions rather than merely constraining or vetoing them. Crucially, CDI is paired with an experience-driven optimization framework that trains the instructor via reinforcement learning (RL), where we convert failure trajectories with privacy violations into learning environments. We formalize baseline defenses and CDI as distinct intervention points in a canonical agent loop, and compare their privacy-helpfulness trade-offs within a unified simulation framework. Results show that our CDI consistently achieves a better balance between privacy preservation (94.2%) and helpfulness (80.6%) than baselines, with superior robustness to adversarial conditions and generalization. 

---
# Guideline-Grounded Evidence Accumulation for High-Stakes Agent Verification 

**Authors**: Yichi Zhang, Nabeel Seedat, Yinpeng Dong, Peng Cui, Jun Zhu, Mihaela van de Schaar  

**Link**: [PDF](https://arxiv.org/pdf/2603.02798)  

**Abstract**: As LLM-powered agents have been used for high-stakes decision-making, such as clinical diagnosis, it becomes critical to develop reliable verification of their decisions to facilitate trustworthy deployment. Yet, existing verifiers usually underperform owing to a lack of domain knowledge and limited calibration. To address this, we establish GLEAN, an agent verification framework with Guideline-grounded Evidence Accumulation that compiles expert-curated protocols into trajectory-informed, well-calibrated correctness signals. GLEAN evaluates the step-wise alignment with domain guidelines and aggregates multi-guideline ratings into surrogate features, which are accumulated along the trajectory and calibrated into correctness probabilities using Bayesian logistic regression. Moreover, the estimated uncertainty triggers active verification, which selectively collects additional evidence for uncertain cases via expanding guideline coverage and performing differential checks. We empirically validate GLEAN with agentic clinical diagnosis across three diseases from the MIMIC-IV dataset, surpassing the best baseline by 12% in AUROC and 50% in Brier score reduction, which confirms the effectiveness in both discrimination and calibration. In addition, the expert study with clinicians recognizes GLEAN's utility in practice. 

---
# Credibility Governance: A Social Mechanism for Collective Self-Correction under Weak Truth Signals 

**Authors**: Wanying He, Yanxi Lin, Ziheng Zhou, Xue Feng, Min Peng, Qianqian Xie, Zilong Zheng, Yipeng Kang  

**Link**: [PDF](https://arxiv.org/pdf/2603.02640)  

**Abstract**: Online platforms increasingly rely on opinion aggregation to allocate real-world attention and resources, yet common signals such as engagement votes or capital-weighted commitments are easy to amplify and often track visibility rather than reliability. This makes collective judgments brittle under weak truth signals, noisy or delayed feedback, early popularity surges, and strategic manipulation. We propose Credibility Governance (CG), a mechanism that reallocates influence by learning which agents and viewpoints consistently track evolving public evidence. CG maintains dynamic credibility scores for both agents and opinions, updates opinion influence via credibility-weighted endorsements, and updates agent credibility based on the long-run performance of the opinions they support, rewarding early and persistent alignment with emerging evidence while filtering short-lived noise. We evaluate CG in POLIS, a socio-physical simulation environment that models coupled belief dynamics and downstream feedback under uncertainty. Across settings with initial majority misalignment, observation noise and contamination, and misinformation shocks, CG outperforms vote-based, stake-weighted, and no-governance baselines, yielding faster recovery to the true state, reduced lock-in and path dependence, and improved robustness under adversarial pressure. Our implementation and experimental scripts are publicly available at this https URL. 

---
# StitchCUDA: An Automated Multi-Agents End-to-End GPU Programing Framework with Rubric-based Agentic Reinforcement Learning 

**Authors**: Shiyang Li, Zijian Zhang, Winson Chen, Yuebo Luo, Mingyi Hong, Caiwen Ding  

**Link**: [PDF](https://arxiv.org/pdf/2603.02637)  

**Abstract**: Modern machine learning (ML) workloads increasingly rely on GPUs, yet achieving high end-to-end performance remains challenging due to dependencies on both GPU kernel efficiency and host-side settings. Although LLM-based methods show promise on automated GPU kernel generation, prior works mainly focus on single-kernel optimization and do not extend to end-to-end programs, hindering practical deployment.
To address the challenge, in this work, we propose StitchCUDA, a multi-agent framework for end-to-end GPU program generation, with three specialized agents: a Planner to orchestrate whole system design, a Coder dedicated to implementing it step-by-step, and a Verifier for correctness check and performance profiling using Nsys/NCU. To fundamentally improve the Coder's ability in end-to-end GPU programming, StitchCUDA integrates rubric-based agentic reinforcement learning over two atomic skills, task-to-code generation and feedback-driven code optimization, with combined rubric reward and rule-based reward from real executions. Therefore, the Coder learns how to implement advanced CUDA programming techniques (e.g., custom kernel fusion, cublas epilogue), and we also effectively prevent Coder's reward hacking (e.g., just copy PyTorch code or hardcoding output) during benchmarking. Experiments on KernelBench show that StitchCUDA achieves nearly 100% success rate on end-to-end GPU programming tasks, with 1.72x better speedup over the multi-agent baseline and 2.73x than the RL model baselines. 

---
# FlashEvaluator: Expanding Search Space with Parallel Evaluation 

**Authors**: Chao Feng, Yuanhao Pu, Chenghao Zhang, Shanqi Liu, Shuchang Liu, Xiang Li, Yongqi Liu, Lantao Hu, Kaiqiao Zhan, Han Li, Kun Gai  

**Link**: [PDF](https://arxiv.org/pdf/2603.02565)  

**Abstract**: The Generator-Evaluator (G-E) framework, i.e., evaluating K sequences from a generator and selecting the top-ranked one according to evaluator scores, is a foundational paradigm in tasks such as Recommender Systems (RecSys) and Natural Language Processing (NLP). Traditional evaluators process sequences independently, suffering from two major limitations: (1) lack of explicit cross-sequence comparison, leading to suboptimal accuracy; (2) poor parallelization with linear complexity of O(K), resulting in inefficient resource utilization and negative impact on both throughput and latency. To address these challenges, we propose FlashEvaluator, which enables cross-sequence token information sharing and processes all sequences in a single forward pass. This yields sublinear computational complexity that improves the system's efficiency and supports direct inter-sequence comparisons that improve selection accuracy. The paper also provides theoretical proofs and extensive experiments on recommendation and NLP tasks, demonstrating clear advantages over conventional methods. Notably, FlashEvaluator has been deployed in online recommender system of Kuaishou, delivering substantial and sustained revenue gains in practice. 

---
# Through the Lens of Contrast: Self-Improving Visual Reasoning in VLMs 

**Authors**: Zhiyu Pan, Yizheng Wu, Jiashen Hua, Junyi Feng, Shaotian Yan, Bing Deng, Zhiguo Cao, Jieping Ye  

**Link**: [PDF](https://arxiv.org/pdf/2603.02556)  

**Abstract**: Reasoning has emerged as a key capability of large language models. In linguistic tasks, this capability can be enhanced by self-improving techniques that refine reasoning paths for subsequent finetuning. However, extending these language-based self-improving approaches to vision language models (VLMs) presents a unique challenge:~visual hallucinations in reasoning paths cannot be effectively verified or rectified. Our solution starts with a key observation about visual contrast: when presented with a contrastive VQA pair, i.e., two visually similar images with synonymous questions, VLMs identify relevant visual cues more precisely. Motivated by this observation, we propose Visual Contrastive Self-Taught Reasoner (VC-STaR), a novel self-improving framework that leverages visual contrast to mitigate hallucinations in model-generated rationales. We collect a diverse suite of VQA datasets, curate contrastive pairs according to multi-modal similarity, and generate rationales using VC-STaR. Consequently, we obtain a new visual reasoning dataset, VisCoR-55K, which is then used to boost the reasoning capability of various VLMs through supervised finetuning. Extensive experiments show that VC-STaR not only outperforms existing self-improving approaches but also surpasses models finetuned on the SoTA visual reasoning datasets, demonstrating that the inherent contrastive ability of VLMs can bootstrap their own visual reasoning. Project at: this https URL. 

---
# MUSE: A Run-Centric Platform for Multimodal Unified Safety Evaluation of Large Language Models 

**Authors**: Zhongxi Wang, Yueqian Lin, Jingyang Zhang, Hai Helen Li, Yiran Chen  

**Link**: [PDF](https://arxiv.org/pdf/2603.02482)  

**Abstract**: Safety evaluation and red-teaming of large language models remain predominantly text-centric, and existing frameworks lack the infrastructure to systematically test whether alignment generalizes to audio, image, and video inputs. We present MUSE (Multimodal Unified Safety Evaluation), an open-source, run-centric platform that integrates automatic cross-modal payload generation, three multi-turn attack algorithms (Crescendo, PAIR, Violent Durian), provider-agnostic model routing, and an LLM judge with a five-level safety taxonomy into a single browser-based system. A dual-metric framework distinguishes hard Attack Success Rate (Compliance only) from soft ASR (including Partial Compliance), capturing partial information leakage that binary metrics miss. To probe whether alignment generalizes across modality boundaries, we introduce Inter-Turn Modality Switching (ITMS), which augments multi-turn attacks with per-turn modality rotation. Experiments across six multimodal LLMs from four providers show that multi-turn strategies can achieve up to 90-100% ASR against models with near-perfect single-turn refusal. ITMS does not uniformly raise final ASR on already-saturated baselines, but accelerates convergence by destabilizing early-turn defenses, and ablation reveals that the direction of modality effects is model-family-specific rather than universal, underscoring the need for provider-aware cross-modal safety testing. 

---
# A Directed Graph Model and Experimental Framework for Design and Study of Time-Dependent Text Visualisation 

**Authors**: Songhai Fan, Simon Angus, Tim Dwyer, Ying Yang, Sarah Goodwin, Helen Purchase  

**Link**: [PDF](https://arxiv.org/pdf/2603.02422)  

**Abstract**: Exponential growth in the quantity of digital news, social media, and other textual sources makes it difficult for humans to keep up with rapidly evolving narratives about world events. Various visualisation techniques have been touted to help people to understand such discourse by exposing relationships between texts (such as news articles) as topics and themes evolve over time. Arguably, the understandability of such visualisations hinges on the assumption that people will be able to easily interpret the relationships in such visual network structures. To test this assumption, we begin by defining an abstract model of time-dependent text visualisation based on directed graph structures. From this model we distill motifs that capture the set of possible ways that texts can be linked across changes in time. We also develop a controlled synthetic text generation methodology that leverages the power of modern LLMs to create fictional, yet structured sets of time-dependent texts that fit each of our patterns. Therefore, we create a clean user study environment (n=30) for participants to identify patterns that best represent a given set of synthetic articles. We find that it is a challenging task for the user to identify and recover the predefined motif. We analyse qualitative data to map an unexpectedly rich variety of user rationales when divergences from expected interpretation occur. A deeper analysis also points to unexpected complexities inherent in the formation of synthetic datasets with LLMs that undermine the study control in some cases. Furthermore, analysis of individual decision-making in our study hints at a future where text discourse visualisation may need to dispense with a one-size-fits-all approach and, instead, should be more adaptable to the specific user who is exploring the visualisation in front of them. 

---
# HELIOS: Harmonizing Early Fusion, Late Fusion, and LLM Reasoning for Multi-Granular Table-Text Retrieval 

**Authors**: Sungho Park, Joohyung Yun, Jongwuk Lee, Wook-Shin Han  

**Link**: [PDF](https://arxiv.org/pdf/2603.02248)  

**Abstract**: Table-text retrieval aims to retrieve relevant tables and text to support open-domain question answering. Existing studies use either early or late fusion, but face limitations. Early fusion pre-aligns a table row with its associated passages, forming "stars," which often include irrelevant contexts and miss query-dependent relationships. Late fusion retrieves individual nodes, dynamically aligning them, but it risks missing relevant contexts. Both approaches also struggle with advanced reasoning tasks, such as column-wise aggregation and multi-hop reasoning. To address these issues, we propose HELIOS, which combines the strengths of both approaches. First, the edge-based bipartite subgraph retrieval identifies finer-grained edges between table segments and passages, effectively avoiding the inclusion of irrelevant contexts. Then, the query-relevant node expansion identifies the most promising nodes, dynamically retrieving relevant edges to grow the bipartite subgraph, minimizing the risk of missing important contexts. Lastly, the star-based LLM refinement performs logical inference at the star graph level rather than the bipartite subgraph, supporting advanced reasoning tasks. Experimental results show that HELIOS outperforms state-of-the-art models with a significant improvement up to 42.6\% and 39.9\% in recall and nDCG, respectively, on the OTT-QA benchmark. 

---
# Safety Training Persists Through Helpfulness Optimization in LLM Agents 

**Authors**: Benjamin Plaut  

**Link**: [PDF](https://arxiv.org/pdf/2603.02229)  

**Abstract**: Safety post-training has been studied extensively in single-step "chat" settings where safety typically refers to refusing harmful requests. We study an "agentic" (i.e., multi-step, tool-use) setting where safety refers to harmful actions directly taken by the LLM. We compare the effects of running direct preference optimization (DPO) on safety or helpfulness alone vs both metrics sequentially. As expected, training on one metric alone results in an extreme point along this frontier. However, unlike prior work, we find that safety training persists through subsequent helpfulness training. We also find that all training configurations end up near a linear Pareto frontier with $R^2 = 0.77$. Even post-training on both metrics simultaneously simply results in another point on the frontier rather than finding a "best of both worlds" strategy, despite the presence of such strategies in our DPO dataset. Overall, our findings underscore the need for better understanding of post-training dynamics. 

---
# Routing Absorption in Sparse Attention: Why Random Gates Are Hard to Beat 

**Authors**: Keston Aquino-Michaels  

**Link**: [PDF](https://arxiv.org/pdf/2603.02227)  

**Abstract**: Can a transformer learn which attention entries matter during training? In principle, yes: attention distributions are highly concentrated, and a small gate network can identify the important entries post-hoc with near-perfect accuracy. In practice, barely. When sparse attention is trained end-to-end, the model's Q/K/V projections co-adapt to whatever mask is imposed, absorbing the routing signal until learned gates perform little better than frozen random gates. We call this routing absorption and present four independent lines of evidence for it in a controlled 31M-parameter transformer: (1) differentiable soft gating converges to nearly the same perplexity whether the gate is learned or random (48.73 +/- 0.60 vs. 49.83 +/- 0.04 over 3 seeds); (2) hard top-k gating receives exactly zero gradient through the mask; (3) a gate distilled onto co-adapted Q/K/V achieves high F1 against oracle masks but catastrophic perplexity when deployed (601.6 vs. 48.6 on mask-agnostic Q/K/V); and (4) stochastic mask randomization during training fails to prevent co-adaptation (78.2 ppl deployed dense vs. 37.3 baseline). We connect routing absorption to the same phenomenon in Mixture-of-Experts, where random routing matches learned routing because experts co-adapt to any router, but show that attention exhibits a structurally more severe form: shared Q/K/V parameters enable cross-layer compensation pathways absent in MoE, where experts are self-contained modules. The implication is that end-to-end sparse attention methods employing per-query token-level gating face absorption pressure proportional to the parameter asymmetry between the gate and the model, and that post-hoc approaches, which decouple representation learning from sparsification, sidestep this entirely. 

---
# Self-Play Only Evolves When Self-Synthetic Pipeline Ensures Learnable Information Gain 

**Authors**: Wei Liu, Siya Qi, Yali Du, Yulan He  

**Link**: [PDF](https://arxiv.org/pdf/2603.02218)  

**Abstract**: Large language models (LLMs) make it plausible to build systems that improve through self-evolving loops, but many existing proposals are better understood as self-play and often plateau quickly. A central failure mode is that the loop synthesises more data without increasing learnable information for the next iteration. Through experiments on a self-play coding task, we reveal that sustainable self-evolution requires a self-synthesised data pipeline with learnable information that increases across iterations. We identify triadic roles that self-evolving LLMs play: the Proposer, which generates tasks; the Solver, which attempts solutions; and the Verifier, which provides training signals, and we identify three system designs that jointly target learnable information gain from this triadic roles perspective. Asymmetric co-evolution closes a weak-to-strong-to-weak loop across roles. Capacity growth expands parameter and inference-time budgets to match rising learnable information. Proactive information seeking introduces external context and new task sources that prevent saturation. Together, these modules provide a measurable, system-level path from brittle self-play dynamics to sustained self-evolution. 

---
