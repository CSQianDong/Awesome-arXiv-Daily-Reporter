# Agent KB: Leveraging Cross-Domain Experience for Agentic Problem Solving 

**Authors**: Xiangru Tang, Tianrui Qin, Tianhao Peng, Ziyang Zhou, Daniel Shao, Tingting Du, Xinming Wei, Peng Xia, Fang Wu, He Zhu, Ge Zhang, Jiaheng Liu, Xingyao Wang, Sirui Hong, Chenglin Wu, Hao Cheng, Chi Wang, Wangchunshu Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2507.06229)  

**Abstract**: As language agents tackle increasingly complex tasks, they struggle with effective error correction and experience reuse across domains. We introduce Agent KB, a hierarchical experience framework that enables complex agentic problem solving via a novel Reason-Retrieve-Refine pipeline. Agent KB addresses a core limitation: agents traditionally cannot learn from each other's experiences. By capturing both high-level strategies and detailed execution logs, Agent KB creates a shared knowledge base that enables cross-agent knowledge transfer. Evaluated on the GAIA benchmark, Agent KB improves success rates by up to 16.28 percentage points. On the most challenging tasks, Claude-3 improves from 38.46% to 57.69%, while GPT-4 improves from 53.49% to 73.26% on intermediate tasks. On SWE-bench code repair, Agent KB enables Claude-3 to improve from 41.33% to 53.33%. Our results suggest that Agent KB provides a modular, framework-agnostic infrastructure for enabling agents to learn from past experiences and generalize successful strategies to new tasks. 

---
# Efficiency-Effectiveness Reranking FLOPs for LLM-based Rerankers 

**Authors**: Zhiyuan Peng, Ting-ruen Wei, Tingyu Song, Yilun Zhao, Yi Fang  

**Link**: [PDF](https://arxiv.org/pdf/2507.06223)  

**Abstract**: Large Language Models (LLMs) have recently been applied to reranking tasks in information retrieval, achieving strong performance. However, their high computational demands often hinder practical deployment. Existing studies evaluate the efficiency of LLM-based rerankers using proxy metrics such as latency, the number of forward passes, input tokens, and output tokens. However, these metrics depend on hardware and running-time choices (\eg parallel or not, batch size, etc), and often fail to account for model size, making it difficult to interpret and obscuring the evaluation of the efficiency-effectiveness tradeoff. To address this issue, we propose E\textsuperscript{2}R-FLOPs, for LLM-based rerankers: ranking metrics per PetaFLOP (RPP) for relevance per compute and queries per PetaFLOP (QPP) for hardware-agnostic throughput. Companied with the new metrics, an interpretable FLOPs estimator is built to estimate the FLOPs of an LLM-based reranker even without running any experiments. Based on the proposed metrics, we conduct comprehensive experiments to evaluate a wide range of LLM-based rerankers with different architecture, studying the efficiency-effectiveness trade-off and bringing this issue to the attention of the research community. 

---
# DS@GT at CheckThat! 2025: Ensemble Methods for Detection of Scientific Discourse on Social Media 

**Authors**: Ayush Parikh, Hoang Thanh Thanh Truong, Jeanette Schofield, Maximilian Heil  

**Link**: [PDF](https://arxiv.org/pdf/2507.06205)  

**Abstract**: In this paper, we, as the DS@GT team for CLEF 2025 CheckThat! Task 4a Scientific Web Discourse Detection, present the methods we explored for this task. For this multiclass classification task, we determined if a tweet contained a scientific claim, a reference to a scientific study or publication, and/or mentions of scientific entities, such as a university or a scientist. We present 3 modeling approaches for this task: transformer finetuning, few-shot prompting of LLMs, and a combined ensemble model whose design was informed by earlier experiments. Our team placed 7th in the competition, achieving a macro-averaged F1 score of 0.8611, an improvement over the DeBERTaV3 baseline of 0.8375. Our code is available on Github at this https URL. 

---
# A Survey on Latent Reasoning 

**Authors**: Rui-Jie Zhu, Tianhao Peng, Tianhao Cheng, Xingwei Qu, Jinfa Huang, Dawei Zhu, Hao Wang, Kaiwen Xue, Xuanliang Zhang, Yong Shan, Tianle Cai, Taylor Kergan, Assel Kembay, Andrew Smith, Chenghua Lin, Binh Nguyen, Yuqi Pan, Yuhong Chou, Zefan Cai, Zhenhe Wu, Yongchi Zhao, Tianyu Liu, Jian Yang, Wangchunshu Zhou, Chujie Zheng, Chongxuan Li, Yuyin Zhou, Zhoujun Li, Zhaoxiang Zhang, Jiaheng Liu, Ge Zhang, Wenhao Huang, Jason Eshraghian  

**Link**: [PDF](https://arxiv.org/pdf/2507.06203)  

**Abstract**: Large Language Models (LLMs) have demonstrated impressive reasoning capabilities, especially when guided by explicit chain-of-thought (CoT) reasoning that verbalizes intermediate steps. While CoT improves both interpretability and accuracy, its dependence on natural language reasoning limits the model's expressive bandwidth. Latent reasoning tackles this bottleneck by performing multi-step inference entirely in the model's continuous hidden state, eliminating token-level supervision. To advance latent reasoning research, this survey provides a comprehensive overview of the emerging field of latent reasoning. We begin by examining the foundational role of neural network layers as the computational substrate for reasoning, highlighting how hierarchical representations support complex transformations. Next, we explore diverse latent reasoning methodologies, including activation-based recurrence, hidden state propagation, and fine-tuning strategies that compress or internalize explicit reasoning traces. Finally, we discuss advanced paradigms such as infinite-depth latent reasoning via masked diffusion models, which enable globally consistent and reversible reasoning processes. By unifying these perspectives, we aim to clarify the conceptual landscape of latent reasoning and chart future directions for research at the frontier of LLM cognition. An associated GitHub repository collecting the latest papers and repos is available at: this https URL. 

---
# UQLM: A Python Package for Uncertainty Quantification in Large Language Models 

**Authors**: Dylan Bouchard, Mohit Singh Chauhan, David Skarbrevik, Ho-Kyeong Ra, Viren Bajaj, Zeya Ahmad  

**Link**: [PDF](https://arxiv.org/pdf/2507.06196)  

**Abstract**: Hallucinations, defined as instances where Large Language Models (LLMs) generate false or misleading content, pose a significant challenge that impacts the safety and trust of downstream applications. We introduce UQLM, a Python package for LLM hallucination detection using state-of-the-art uncertainty quantification (UQ) techniques. This toolkit offers a suite of UQ-based scorers that compute response-level confidence scores ranging from 0 to 1. This library provides an off-the-shelf solution for UQ-based hallucination detection that can be easily integrated to enhance the reliability of LLM outputs. 

---
# DS@GT at CheckThat! 2025: Evaluating Context and Tokenization Strategies for Numerical Fact Verification 

**Authors**: Maximilian Heil, Aleksandar Pramov  

**Link**: [PDF](https://arxiv.org/pdf/2507.06195)  

**Abstract**: Numerical claims, statements involving quantities, comparisons, and temporal references, pose unique challenges for automated fact-checking systems. In this study, we evaluate modeling strategies for veracity prediction of such claims using the QuanTemp dataset and building our own evidence retrieval pipeline. We investigate three key factors: (1) the impact of more evidences with longer input context windows using ModernBERT, (2) the effect of right-to-left (R2L) tokenization, and (3) their combined influence on classification performance. Contrary to prior findings in arithmetic reasoning tasks, R2L tokenization does not boost natural language inference (NLI) of numerical tasks. A longer context window does also not enhance veracity performance either, highlighting evidence quality as the dominant bottleneck. Our best-performing system achieves competitive macro-average F1 score of 0.57 and places us among the Top-4 submissions in Task 3 of CheckThat! 2025. Our code is available at this https URL. 

---
# DS@GT at CheckThat! 2025: Detecting Subjectivity via Transfer-Learning and Corrective Data Augmentation 

**Authors**: Maximilian Heil, Dionne Bang  

**Link**: [PDF](https://arxiv.org/pdf/2507.06189)  

**Abstract**: This paper presents our submission to Task 1, Subjectivity Detection, of the CheckThat! Lab at CLEF 2025. We investigate the effectiveness of transfer-learning and stylistic data augmentation to improve classification of subjective and objective sentences in English news text. Our approach contrasts fine-tuning of pre-trained encoders and transfer-learning of fine-tuned transformer on related tasks. We also introduce a controlled augmentation pipeline using GPT-4o to generate paraphrases in predefined subjectivity styles. To ensure label and style consistency, we employ the same model to correct and refine the generated samples. Results show that transfer-learning of specified encoders outperforms fine-tuning general-purpose ones, and that carefully curated augmentation significantly enhances model robustness, especially in detecting subjective content. Our official submission placed us $16^{th}$ of 24 participants. Overall, our findings underscore the value of combining encoder specialization with label-consistent augmentation for improved subjectivity detection. Our code is available at this https URL. 

---
# CriticLean: Critic-Guided Reinforcement Learning for Mathematical Formalization 

**Authors**: Zhongyuan Peng, Yifan Yao, Kaijing Ma, Shuyue Guo, Yizhe Li, Yichi Zhang, Chenchen Zhang, Yifan Zhang, Zhouliang Yu, Luming Li, Minghao Liu, Yihang Xia, Jiawei Shen, Yuchen Wu, Yixin Cao, Zhaoxiang Zhang, Wenhao Huang, Jiaheng Liu, Ge Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.06181)  

**Abstract**: Translating natural language mathematical statements into formal, executable code is a fundamental challenge in automated theorem proving. While prior work has focused on generation and compilation success, little attention has been paid to the critic phase-the evaluation of whether generated formalizations truly capture the semantic intent of the original problem. In this paper, we introduce CriticLean, a novel critic-guided reinforcement learning framework that elevates the role of the critic from a passive validator to an active learning component. Specifically, first, we propose the CriticLeanGPT, trained via supervised fine-tuning and reinforcement learning, to rigorously assess the semantic fidelity of Lean 4 formalizations. Then, we introduce CriticLeanBench, a benchmark designed to measure models' ability to distinguish semantically correct from incorrect formalizations, and demonstrate that our trained CriticLeanGPT models can significantly outperform strong open- and closed-source baselines. Building on the CriticLean framework, we construct FineLeanCorpus, a dataset comprising over 285K problems that exhibits rich domain diversity, broad difficulty coverage, and high correctness based on human evaluation. Overall, our findings highlight that optimizing the critic phase is essential for producing reliable formalizations, and we hope our CriticLean will provide valuable insights for future advances in formal mathematical reasoning. 

---
# Skywork-R1V3 Technical Report 

**Authors**: Wei Shen, Jiangbo Pei, Yi Peng, Xuchen Song, Yang Liu, Jian Peng, Haofeng Sun, Yunzhuo Hao, Peiyu Wang, Yahui Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2507.06167)  

**Abstract**: We introduce Skywork-R1V3, an advanced, open-source vision-language model (VLM) that pioneers a new approach to visual reasoning. Its key innovation lies in effectively transferring reasoning skills from text-only Large Language Models (LLMs) to visual tasks. The strong performance of Skywork-R1V3 primarily stems from our elaborate post-training RL framework, which effectively activates and enhances the model's reasoning ability, without the need for additional continue pre-training. Through this framework, we further uncover the fundamental role of the connector module in achieving robust cross-modal alignment for multimodal reasoning models. In addition, we introduce a unique indicator of reasoning capability, the entropy of critical reasoning tokens, which has proven highly effective for checkpoint selection during RL training. Skywork-R1V3 achieves state-of-the-art results on MMMU, significantly improving from 64.3% to 76.0%. This performance matches entry-level human capabilities. Remarkably, our RL-powered post-training approach enables even the 38B parameter model to rival top closed-source VLMs. The implementation successfully transfers mathematical reasoning to other subject-related reasoning tasks. We also include an analysis of curriculum learning and reinforcement finetuning strategies, along with a broader discussion on multimodal reasoning. Skywork-R1V3 represents a significant leap in multimodal reasoning, showcasing RL as a powerful engine for advancing open-source VLM capabilities. 

---
# Coding Triangle: How Does Large Language Model Understand Code? 

**Authors**: Taolin Zhang, Zihan Ma, Maosong Cao, Junnan Liu, Songyang Zhang, Kai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.06138)  

**Abstract**: Large language models (LLMs) have achieved remarkable progress in code generation, yet their true programming competence remains underexplored. We introduce the Code Triangle framework, which systematically evaluates LLMs across three fundamental dimensions: editorial analysis, code implementation, and test case generation. Through extensive experiments on competitive programming benchmarks, we reveal that while LLMs can form a self-consistent system across these dimensions, their solutions often lack the diversity and robustness of human programmers. We identify a significant distribution shift between model cognition and human expertise, with model errors tending to cluster due to training data biases and limited reasoning transfer. Our study demonstrates that incorporating human-generated editorials, solutions, and diverse test cases, as well as leveraging model mixtures, can substantially enhance both the performance and robustness of LLMs. Furthermore, we reveal both the consistency and inconsistency in the cognition of LLMs that may facilitate self-reflection and self-improvement, providing a potential direction for developing more powerful coding models. 

---
# NeoBabel: A Multilingual Open Tower for Visual Generation 

**Authors**: Mohammad Mahdi Derakhshani, Dheeraj Varghese, Marzieh Fadaee, Cees G. M. Snoek  

**Link**: [PDF](https://arxiv.org/pdf/2507.06137)  

**Abstract**: Text-to-image generation advancements have been predominantly English-centric, creating barriers for non-English speakers and perpetuating digital inequities. While existing systems rely on translation pipelines, these introduce semantic drift, computational overhead, and cultural misalignment. We introduce NeoBabel, a novel multilingual image generation framework that sets a new Pareto frontier in performance, efficiency and inclusivity, supporting six languages: English, Chinese, Dutch, French, Hindi, and Persian. The model is trained using a combination of large-scale multilingual pretraining and high-resolution instruction tuning. To evaluate its capabilities, we expand two English-only benchmarks to multilingual equivalents: m-GenEval and m-DPG. NeoBabel achieves state-of-the-art multilingual performance while retaining strong English capability, scoring 0.75 on m-GenEval and 0.68 on m-DPG. Notably, it performs on par with leading models on English tasks while outperforming them by +0.11 and +0.09 on multilingual benchmarks, even though these models are built on multilingual base LLMs. This demonstrates the effectiveness of our targeted alignment training for preserving and extending crosslingual generalization. We further introduce two new metrics to rigorously assess multilingual alignment and robustness to code-mixed prompts. Notably, NeoBabel matches or exceeds English-only models while being 2-4x smaller. We release an open toolkit, including all code, model checkpoints, a curated dataset of 124M multilingual text-image pairs, and standardized multilingual evaluation protocols, to advance inclusive AI research. Our work demonstrates that multilingual capability is not a trade-off but a catalyst for improved robustness, efficiency, and cultural fidelity in generative AI. 

---
# A Survey on Prompt Tuning 

**Authors**: Zongqian Li, Yixuan Su, Nigel Collier  

**Link**: [PDF](https://arxiv.org/pdf/2507.06085)  

**Abstract**: This survey reviews prompt tuning, a parameter-efficient approach for adapting language models by prepending trainable continuous vectors while keeping the model frozen. We classify existing approaches into two categories: direct prompt learning and transfer learning. Direct prompt learning methods include: general optimization approaches, encoder-based methods, decomposition strategies, and mixture-of-experts frameworks. Transfer learning methods consist of: general transfer approaches, encoder-based methods, and decomposition strategies. For each method, we analyze method designs, innovations, insights, advantages, and disadvantages, with illustrative visualizations comparing different frameworks. We identify challenges in computational efficiency and training stability, and discuss future directions in improving training robustness and broadening application scope. 

---
# Entropy-Memorization Law: Evaluating Memorization Difficulty of Data in LLMs 

**Authors**: Yizhan Huang, Zhe Yang, Meifang Chen, Jianping Zhang, Michael R. Lyu  

**Link**: [PDF](https://arxiv.org/pdf/2507.06056)  

**Abstract**: Large Language Models (LLMs) are known to memorize portions of their training data, sometimes reproducing content verbatim when prompted appropriately. In this work, we investigate a fundamental yet under-explored question in the domain of memorization: How to characterize memorization difficulty of training data in LLMs? Through empirical experiments on OLMo, a family of open models, we present the Entropy-Memorization Law. It suggests that data entropy is linearly correlated with memorization score. Moreover, in a case study of memorizing highly randomized strings, or "gibberish", we observe that such sequences, despite their apparent randomness, exhibit unexpectedly low empirical entropy compared to the broader training corpus. Adopting the same strategy to discover Entropy-Memorization Law, we derive a simple yet effective approach to distinguish training and testing data, enabling Dataset Inference (DI). 

---
# Conditional Multi-Stage Failure Recovery for Embodied Agents 

**Authors**: Youmna Farag, Svetlana Stoyanchev, Mohan Li, Simon Keizer, Rama Doddipatla  

**Link**: [PDF](https://arxiv.org/pdf/2507.06016)  

**Abstract**: Embodied agents performing complex tasks are susceptible to execution failures, motivating the need for effective failure recovery mechanisms. In this work, we introduce a conditional multistage failure recovery framework that employs zero-shot chain prompting. The framework is structured into four error-handling stages, with three operating during task execution and one functioning as a post-execution reflection phase. Our approach utilises the reasoning capabilities of LLMs to analyse execution challenges within their environmental context and devise strategic solutions. We evaluate our method on the TfD benchmark of the TEACH dataset and achieve state-of-the-art performance, outperforming a baseline without error recovery by 11.5% and surpassing the strongest existing model by 19%. 

---
# DocIE@XLLM25: In-Context Learning for Information Extraction using Fully Synthetic Demonstrations 

**Authors**: Nicholas Popovič, Ashish Kangen, Tim Schopf, Michael Färber  

**Link**: [PDF](https://arxiv.org/pdf/2507.05997)  

**Abstract**: Large, high-quality annotated corpora remain scarce in document-level entity and relation extraction in zero-shot or few-shot settings. In this paper, we present a fully automatic, LLM-based pipeline for synthetic data generation and in-context learning for document-level entity and relation extraction. In contrast to existing approaches that rely on manually annotated demonstrations or direct zero-shot inference, our method combines synthetic data generation with retrieval-based in-context learning, using a reasoning-optimized language model. This allows us to build a high-quality demonstration database without manual annotation and to dynamically retrieve relevant examples at inference time. Based on our approach we produce a synthetic dataset of over $5k$ Wikipedia abstracts with approximately $59k$ entities and $30k$ relation triples. Finally, we evaluate in-context learning performance on the DocIE shared task, extracting entities and relations from long documents in a zero-shot setting. We find that in-context joint entity and relation extraction at document-level remains a challenging task, even for state-of-the-art large language models. 

---
# Evolution without Large Models: Training Language Model with Task Principles 

**Authors**: Minghang Zhu, Shen Gao, Zhengliang Shi, Jiabao Fang, Pengjie Ren, Zhaochun Ren, Zhumin Chen, Shuo Shang  

**Link**: [PDF](https://arxiv.org/pdf/2507.05991)  

**Abstract**: A common training approach for language models involves using a large-scale language model to expand a human-provided dataset, which is subsequently used for model this http URL method significantly reduces training costs by eliminating the need for extensive human data annotation. However, it still faces challenges such as high carbon emissions during data augmentation and the risk of data leakage when we use closed-source LLMs. To address these issues, we propose a self-evolution method for language models. First, we introduce the Multi-level Principle Generation, which enables a large-scale model to summarize task-completion principles based on a small amount of task data. Then, we propose the Principle-based Instance Generation, in which a smaller-scale language model uses these task principles to generate a large amount of data. This data is then used for model training. Experimental results show that our proposed method significantly improves model performance compared to directly using a smaller-scale language model to generate data. Additionally, since we only use the large-scale language model to generate the task-completion principles, the carbon emissions associated with training the model are greatly reduced. 

---
# RabakBench: Scaling Human Annotations to Construct Localized Multilingual Safety Benchmarks for Low-Resource Languages 

**Authors**: Gabriel Chua, Leanne Tan, Ziyu Ge, Roy Ka-Wei Lee  

**Link**: [PDF](https://arxiv.org/pdf/2507.05980)  

**Abstract**: Large language models (LLMs) and their safety classifiers often perform poorly on low-resource languages due to limited training data and evaluation benchmarks. This paper introduces RabakBench, a new multilingual safety benchmark localized to Singapore's unique linguistic context, covering Singlish, Chinese, Malay, and Tamil. RabakBench is constructed through a scalable three-stage pipeline: (i) Generate - adversarial example generation by augmenting real Singlish web content with LLM-driven red teaming; (ii) Label - semi-automated multi-label safety annotation using majority-voted LLM labelers aligned with human judgments; and (iii) Translate - high-fidelity translation preserving linguistic nuance and toxicity across languages. The final dataset comprises over 5,000 safety-labeled examples across four languages and six fine-grained safety categories with severity levels. Evaluations of 11 popular open-source and closed-source guardrail classifiers reveal significant performance degradation. RabakBench not only enables robust safety evaluation in Southeast Asian multilingual settings but also offers a reproducible framework for building localized safety datasets in low-resource environments. The benchmark dataset, including the human-verified translations, and evaluation code are publicly available. 

---
# We Should Evaluate Real-World Impact 

**Authors**: Ehud Reiter  

**Link**: [PDF](https://arxiv.org/pdf/2507.05973)  

**Abstract**: The ACL community has very little interest in evaluating the real-world impact of NLP systems. A structured survey of the ACL Anthology shows that perhaps 0.1% of its papers contain such evaluations; furthermore most papers which include impact evaluations present them very sketchily and instead focus on metric evaluations. NLP technology would be more useful and more quickly adopted if we seriously tried to understand and evaluate its real-world impact. 

---
# OpenFActScore: Open-Source Atomic Evaluation of Factuality in Text Generation 

**Authors**: Lucas Fonseca Lage, Simon Ostermann  

**Link**: [PDF](https://arxiv.org/pdf/2507.05965)  

**Abstract**: We introduce OpenFActScore, an open-source implementation of the FActScore framework for evaluating the factuality of text generated by large language models (LLMs). FActScore evaluates the factual accuracy of long-form text by using Atomic Fact Generation (AFG) to extract individual factual claims and Atomic Fact Validation (AFV) to verify each claim against a trusted knowledge source. While the original FActScore relies on closed-source and commercial models such as InstructGPT and ChatGPT, OpenFActScore enables the use of any Hugging Face-compatible model for both AFG and AFV. We provide a detailed technical overview of our implementation, highlighting design choices and modifications made to support open models. We evaluate multiple open-source LLMs on both AFG and AFV using the original FActScore benchmark, reporting BERTScore-F1 for AFG and Error Rate relative to human annotations for AFV. Our results show that open models can approximate the performance of closed-source systems, with Gemma achieving the best overall performance, and our final setup obtains a 0.99 Pearson correlation with the original FActScore experiments. OpenFActScore promotes transparency, reproducibility, and cost-effective evaluation, and is available at: this https URL. 

---
# Chat-Ghosting: A Comparative Study of Methods for Auto-Completion in Dialog Systems 

**Authors**: Sandeep Mishra, Anubhab Mandal, Bishal Santra, Tushar Abhishek, Pawan Goyal, Manish Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2507.05940)  

**Abstract**: Ghosting, the ability to predict a user's intended text input for inline query auto-completion, is an invaluable feature for modern search engines and chat interfaces, greatly enhancing user experience. By suggesting completions to incomplete queries (or prefixes), ghosting aids users with slow typing speeds, disabilities, or limited language proficiency. Ghosting is a challenging problem and has become more important with the ubiquitousness of chat-based systems like ChatGPT, Copilot, etc. Despite the increasing prominence of chat-based systems utilizing ghosting, this challenging problem of Chat-Ghosting has received little attention from the NLP/ML research community. There is a lack of standardized benchmarks and relative performance analysis of deep learning and non-deep learning methods. We address this through an open and thorough study of this problem using four publicly available dialog datasets: two human-human (DailyDialog and DSTC7-Ubuntu) and two human-bot (Open Assistant and ShareGPT). We experiment with various existing query auto-completion methods (using tries), n-gram methods and deep learning methods, with and without dialog context. We also propose a novel entropy-based dynamic early stopping strategy. Our analysis finds that statistical n-gram models and tries outperform deep learning based models in terms of both model performance and inference efficiency for seen prefixes. For unseen queries, neural models like T5 and Phi-2 lead to better results. Adding conversational context leads to significant improvements in ghosting quality, especially for Open-Assistant and ShareGPT. We make code and data publicly available 

---
# Remember Past, Anticipate Future: Learning Continual Multimodal Misinformation Detectors 

**Authors**: Bing Wang, Ximing Li, Mengzhe Ye, Changchun Li, Bo Fu, Jianfeng Qu, Lin Yuanbo Wu  

**Link**: [PDF](https://arxiv.org/pdf/2507.05939)  

**Abstract**: Nowadays, misinformation articles, especially multimodal ones, are widely spread on social media platforms and cause serious negative effects. To control their propagation, Multimodal Misinformation Detection (MMD) becomes an active topic in the community to automatically identify misinformation. Previous MMD methods focus on supervising detectors by collecting offline data. However, in real-world scenarios, new events always continually emerge, making MMD models trained on offline data consistently outdated and ineffective. To address this issue, training MMD models under online data streams is an alternative, inducing an emerging task named continual MMD. Unfortunately, it is hindered by two major challenges. First, training on new data consistently decreases the detection performance on past data, named past knowledge forgetting. Second, the social environment constantly evolves over time, affecting the generalization on future data. To alleviate these challenges, we propose to remember past knowledge by isolating interference between event-specific parameters with a Dirichlet process-based mixture-of-expert structure, and anticipate future environmental distributions by learning a continuous-time dynamics model. Accordingly, we induce a new continual MMD method DAEDCMD. Extensive experiments demonstrate that DAEDCMD can consistently and significantly outperform the compared methods, including six MMD baselines and three continual learning methods. 

---
# Towards a Principled Evaluation of Knowledge Editors 

**Authors**: Sebastian Pohl, Max Ploner, Alan Akbik  

**Link**: [PDF](https://arxiv.org/pdf/2507.05937)  

**Abstract**: Model editing has been gaining increasing attention over the past few years. For Knowledge Editing in particular, more challenging evaluation datasets have recently been released. These datasets use different methodologies to score the success of editors. Yet, it remains under-explored how robust these methodologies are and whether they unfairly favor some editors. Moreover, the disruptive impact of these editors on overall model capabilities remains a constant blind spot.
We address both of these problems and show that choosing different metrics and evaluation methodologies as well as different edit batch sizes can lead to a different ranking of knowledge editors. Crucially we demonstrate this effect also on general language understanding tasks evaluated alongside the knowledge editing tasks. Further we include a manual assessment of the string matching based evaluation method for knowledge editing that is favored by recently released datasets, revealing a tendency to produce false positive matches. 

---
# Few-shot text-based emotion detection 

**Authors**: Teodor-George Marchitan, Claudiu Creanga, Liviu P. Dinu  

**Link**: [PDF](https://arxiv.org/pdf/2507.05918)  

**Abstract**: This paper describes the approach of the Unibuc - NLP team in tackling the SemEval 2025 Workshop, Task 11: Bridging the Gap in Text-Based Emotion Detection. We mainly focused on experiments using large language models (Gemini, Qwen, DeepSeek) with either few-shot prompting or fine-tuning. With our final system, for the multi-label emotion detection track (track A), we got an F1-macro of $0.7546$ (26/96 teams) for the English subset, $0.1727$ (35/36 teams) for the Portuguese (Mozambican) subset and $0.325$ (\textbf{1}/31 teams) for the Emakhuwa subset. 

---
# Psychometric Item Validation Using Virtual Respondents with Trait-Response Mediators 

**Authors**: Sungjib Lim, Woojung Song, Eun-Ju Lee, Yohan Jo  

**Link**: [PDF](https://arxiv.org/pdf/2507.05890)  

**Abstract**: As psychometric surveys are increasingly used to assess the traits of large language models (LLMs), the need for scalable survey item generation suited for LLMs has also grown. A critical challenge here is ensuring the construct validity of generated items, i.e., whether they truly measure the intended trait. Traditionally, this requires costly, large-scale human data collection. To make it efficient, we present a framework for virtual respondent simulation using LLMs. Our central idea is to account for mediators: factors through which the same trait can give rise to varying responses to a survey item. By simulating respondents with diverse mediators, we identify survey items that robustly measure intended traits. Experiments on three psychological trait theories (Big5, Schwartz, VIA) show that our mediator generation methods and simulation framework effectively identify high-validity items. LLMs demonstrate the ability to generate plausible mediators from trait definitions and to simulate respondent behavior for item validation. Our problem formulation, metrics, methodology, and dataset open a new direction for cost-effective survey development and a deeper understanding of how LLMs replicate human-like behavior. We will publicly release our dataset and code to support future work. 

---
# How to Evaluate Automatic Speech Recognition: Comparing Different Performance and Bias Measures 

**Authors**: Tanvina Patel, Wiebke Hutiri, Aaron Yi Ding, Odette Scharenborg  

**Link**: [PDF](https://arxiv.org/pdf/2507.05885)  

**Abstract**: There is increasingly more evidence that automatic speech recognition (ASR) systems are biased against different speakers and speaker groups, e.g., due to gender, age, or accent. Research on bias in ASR has so far primarily focused on detecting and quantifying bias, and developing mitigation approaches. Despite this progress, the open question is how to measure the performance and bias of a system. In this study, we compare different performance and bias measures, from literature and proposed, to evaluate state-of-the-art end-to-end ASR systems for Dutch. Our experiments use several bias mitigation strategies to address bias against different speaker groups. The findings reveal that averaged error rates, a standard in ASR research, alone is not sufficient and should be supplemented by other measures. The paper ends with recommendations for reporting ASR performance and bias to better represent a system's performance for diverse speaker groups, and overall system bias. 

---
# Bridging Perception and Language: A Systematic Benchmark for LVLMs' Understanding of Amodal Completion Reports 

**Authors**: Amane Watahiki, Tomoki Doi, Taiga Shinozaki, Satoshi Nishida, Takuya Niikawa, Katsunori Miyahara, Hitomi Yanaka  

**Link**: [PDF](https://arxiv.org/pdf/2507.05799)  

**Abstract**: One of the main objectives in developing large vision-language models (LVLMs) is to engineer systems that can assist humans with multimodal tasks, including interpreting descriptions of perceptual experiences. A central phenomenon in this context is amodal completion, in which people perceive objects even when parts of those objects are hidden. Although numerous studies have assessed whether computer-vision algorithms can detect or reconstruct occluded regions, the inferential abilities of LVLMs on texts related to amodal completion remain unexplored. To address this gap, we constructed a benchmark grounded in Basic Formal Ontology to achieve a systematic classification of amodal completion. Our results indicate that while many LVLMs achieve human-comparable performance overall, their accuracy diverges for certain types of objects being completed. Notably, in certain categories, some LLaVA-NeXT variants and Claude 3.5 Sonnet exhibit lower accuracy on original images compared to blank stimuli lacking visual content. Intriguingly, this disparity emerges only under Japanese prompting, suggesting a deficiency in Japanese-specific linguistic competence among these models. 

---
# Flippi: End To End GenAI Assistant for E-Commerce 

**Authors**: Anand A. Rajasekar, Praveen Tangarajan, Anjali Nainani, Amogh Batwal, Vinay Rao Dandin, Anusua Trivedi, Ozan Ersoy  

**Link**: [PDF](https://arxiv.org/pdf/2507.05788)  

**Abstract**: The emergence of conversational assistants has fundamentally reshaped user interactions with digital platforms. This paper introduces Flippi-a cutting-edge, end-to-end conversational assistant powered by large language models (LLMs) and tailored for the e-commerce sector. Flippi addresses the challenges posed by the vast and often overwhelming product landscape, enabling customers to discover products more efficiently through natural language dialogue. By accommodating both objective and subjective user requirements, Flippi delivers a personalized shopping experience that surpasses traditional search methods. This paper details how Flippi interprets customer queries to provide precise product information, leveraging advanced NLP techniques such as Query Reformulation, Intent Detection, Retrieval-Augmented Generation (RAG), Named Entity Recognition (NER), and Context Reduction. Flippi's unique capability to identify and present the most attractive offers on an e-commerce site is also explored, demonstrating how it empowers users to make cost-effective decisions. Additionally, the paper discusses Flippi's comparative analysis features, which help users make informed choices by contrasting product features, prices, and other relevant attributes. The system's robust architecture is outlined, emphasizing its adaptability for integration across various e-commerce platforms and the technological choices underpinning its performance and accuracy. Finally, a comprehensive evaluation framework is presented, covering performance metrics, user satisfaction, and the impact on customer engagement and conversion rates. By bridging the convenience of online shopping with the personalized assistance traditionally found in physical stores, Flippi sets a new standard for customer satisfaction and engagement in the digital marketplace. 

---
# DocTalk: Scalable Graph-based Dialogue Synthesis for Enhancing LLM Conversational Capabilities 

**Authors**: Jing Yang Lee, Hamed Bonab, Nasser Zalmout, Ming Zeng, Sanket Lokegaonkar, Colin Lockard, Binxuan Huang, Ritesh Sarkhel, Haodong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.05750)  

**Abstract**: Large Language Models (LLMs) are increasingly employed in multi-turn conversational tasks, yet their pre-training data predominantly consists of continuous prose, creating a potential mismatch between required capabilities and training paradigms. We introduce a novel approach to address this discrepancy by synthesizing conversational data from existing text corpora. We present a pipeline that transforms a cluster of multiple related documents into an extended multi-turn, multi-topic information-seeking dialogue. Applying our pipeline to Wikipedia articles, we curate DocTalk, a multi-turn pre-training dialogue corpus consisting of over 730k long conversations. We hypothesize that exposure to such synthesized conversational structures during pre-training can enhance the fundamental multi-turn capabilities of LLMs, such as context memory and understanding. Empirically, we show that incorporating DocTalk during pre-training results in up to 40% gain in context memory and understanding, without compromising base performance. DocTalk is available at this https URL. 

---
# GPTKB v1.5: A Massive Knowledge Base for Exploring Factual LLM Knowledge 

**Authors**: Yujia Hu, Tuan-Phong Nguyen, Shrestha Ghosh, Moritz Müller, Simon Razniewski  

**Link**: [PDF](https://arxiv.org/pdf/2507.05740)  

**Abstract**: Language models are powerful tools, yet their factual knowledge is still poorly understood, and inaccessible to ad-hoc browsing and scalable statistical analysis. This demonstration introduces GPTKB v1.5, a densely interlinked 100-million-triple knowledge base (KB) built for $14,000 from GPT-4.1, using the GPTKB methodology for massive-recursive LLM knowledge materialization (Hu et al., ACL 2025). The demonstration experience focuses on three use cases: (1) link-traversal-based LLM knowledge exploration, (2) SPARQL-based structured LLM knowledge querying, (3) comparative exploration of the strengths and weaknesses of LLM knowledge. Massive-recursive LLM knowledge materialization is a groundbreaking opportunity both for the research area of systematic analysis of LLM knowledge, as well as for automated KB construction. The GPTKB demonstrator is accessible at this https URL. 

---
# Omni-Router: Sharing Routing Decisions in Sparse Mixture-of-Experts for Speech Recognition 

**Authors**: Zijin Gu, Tatiana Likhomanenko, Navdeep Jaitly  

**Link**: [PDF](https://arxiv.org/pdf/2507.05724)  

**Abstract**: Mixture-of-experts (MoE) architectures have expanded from language modeling to automatic speech recognition (ASR). Traditional MoE methods, such as the Switch Transformer, route experts independently within each layer. Our analysis reveals that routers in most layers make expert choices that are not strongly correlated with the choices of the routers in other layers. To increase the cooperation between experts in different layers and encourage greater specialization, we use a shared router across different MoE layers. We call this model \emph{Omni-router Transformer}. Extensive experiments on a large-scale pseudo-labeled dataset and evaluations across 10 diverse, out-of-domain ASR benchmarks demonstrate that the Omni-router Transformer is able to achieve lower training loss and consistently outperform dense and Switch Transformer models, reducing average word error rates by 11.2% and 8.2%, respectively, while providing structured expert usage and improved robustness to diverse data. 

---
# HIRAG: Hierarchical-Thought Instruction-Tuning Retrieval-Augmented Generation 

**Authors**: YiHan Jiao, ZheHao Tan, Dan Yang, DuoLin Sun, Jie Feng, Jian Wang, Peng Wei  

**Link**: [PDF](https://arxiv.org/pdf/2507.05714)  

**Abstract**: Retrieval-augmented generation (RAG) has become a fundamental paradigm for addressing the challenges faced by large language models in handling real-time information and domain-specific problems. Traditional RAG systems primarily rely on the in-context learning (ICL) capabilities of the large language model itself. Still, in-depth research on the specific capabilities needed by the RAG generation model is lacking, leading to challenges with inconsistent document quality and retrieval system imperfections. Even the limited studies that fine-tune RAG generative models often \textit{lack a granular focus on RAG task} or \textit{a deeper utilization of chain-of-thought processes}. To address this, we propose that RAG models should possess three progressively hierarchical abilities (1) Filtering: the ability to select relevant information; (2) Combination: the ability to combine semantic information across paragraphs; and (3) RAG-specific reasoning: the ability to further process external knowledge using internal knowledge. Thus, we introduce our new RAG instruction fine-tuning method, Hierarchical-Thought Instruction-Tuning Retrieval-Augmented Generation (HIRAG) incorporates a "think before answering" strategy. This method enhances the model's open-book examination capability by utilizing multi-level progressive chain-of-thought. Experiments show that the HIRAG training strategy significantly improves the model's performance on datasets such as RGB, PopQA, MuSiQue, HotpotQA, and PubmedQA. 

---
# DRAGON: Dynamic RAG Benchmark On News 

**Authors**: Fedor Chernogorskii, Sergei Averkiev, Liliya Kudraleeva, Zaven Martirosian, Maria Tikhonova, Valentin Malykh, Alena Fenogenova  

**Link**: [PDF](https://arxiv.org/pdf/2507.05713)  

**Abstract**: Retrieval-Augmented Generation (RAG) is a widely adopted approach for improving the factuality of large language models (LLMs) by incorporating external knowledge at inference time. Although there exist multiple RAG benchmarks for English, evaluation resources for other languages, including Russian, remain scarce and static, failing to capture the dynamic nature of real-world deployments.
In this work, we present DRAGON (Dynamic RAG Benchmark On News), the first dynamic benchmark for evaluating RAG systems in Russian on a changing news corpora. DRAGON is built upon a regularly updated corpus of Russian news and public documents and supports comprehensive evaluation of both the retriever and generator components. Question generation is performed automatically with the use of Knowledge Graph constructed from the corpus and enables the extraction of four core question types aligned with distinct subgraph patterns. We release a complete evaluation framework comprising the pipeline for automatic question generation, evaluation scripts, which are potentially reusable for other languages and multilingual settings, and benchmark data. We also launch a public leaderboard to encourage community participation and comparison. 

---
# Agentic-R1: Distilled Dual-Strategy Reasoning 

**Authors**: Weihua Du, Pranjal Aggarwal, Sean Welleck, Yiming Yang  

**Link**: [PDF](https://arxiv.org/pdf/2507.05707)  

**Abstract**: Current long chain-of-thought (long-CoT) models excel at mathematical reasoning but rely on slow and error-prone natural language traces. Tool-augmented agents address arithmetic via code execution, but often falter on complex logical tasks. We introduce a fine-tuning framework, DualDistill, that distills complementary reasoning strategies from multiple teachers into a unified student model. Using this approach, we train Agentic-R1, which dynamically selects the optimal strategy for each query, invoking tools for arithmetic and algorithmic problems, and using text-based reasoning for abstract ones. Our method improves accuracy across a range of tasks, including both computation-intensive and standard benchmarks, demonstrating the effectiveness of multi-strategy distillation in achieving robust and efficient reasoning. Our project is available at this https URL 

---
# Smoothie-Qwen: Post-Hoc Smoothing to Reduce Language Bias in Multilingual LLMs 

**Authors**: SeungWon Ji, Jungyup Lee, Jemin Kim, Sang Park, SeungJae Lee  

**Link**: [PDF](https://arxiv.org/pdf/2507.05686)  

**Abstract**: Multilingual large language models (LLMs) often exhibit language confusion, a tendency to generate responses in a dominant language irrespective of the prompt's language. To address this, we propose Smoothie-Qwen, a lightweight, post-hoc method that mitigates language bias without retraining. This technique selectively adjusts token-level output probabilities to effectively suppress undesired language generation. Applied to the Qwen model, our method reduces unintended Chinese output by over 95% while preserving task accuracy on multilingual benchmarks. This work provides a practical and efficient solution for enhancing the language controllability of LLMs, making them more reliable for global applications. 

---
# ECom-Bench: Can LLM Agent Resolve Real-World E-commerce Customer Support Issues? 

**Authors**: Haoxin Wang, Xianhan Peng, Xucheng Huang, Yizhe Huang, Ming Gong, Chenghan Yang, Yang Liu, Ling Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2507.05639)  

**Abstract**: In this paper, we introduce ECom-Bench, the first benchmark framework for evaluating LLM agent with multimodal capabilities in the e-commerce customer support domain. ECom-Bench features dynamic user simulation based on persona information collected from real e-commerce customer interactions and a realistic task dataset derived from authentic e-commerce dialogues. These tasks, covering a wide range of business scenarios, are designed to reflect real-world complexities, making ECom-Bench highly challenging. For instance, even advanced models like GPT-4o achieve only a 10-20% pass^3 metric in our benchmark, highlighting the substantial difficulties posed by complex e-commerce scenarios. Upon publication, the code and data will be open-sourced to facilitate further research and development in this domain. 

---
# SARA: Selective and Adaptive Retrieval-augmented Generation with Context Compression 

**Authors**: Yiqiao Jin, Kartik Sharma, Vineeth Rakesh, Yingtong Dou, Menghai Pan, Mahashweta Das, Srijan Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2507.05633)  

**Abstract**: Retrieval-augmented Generation (RAG) extends large language models (LLMs) with external knowledge but faces key challenges: restricted effective context length and redundancy in retrieved documents. Pure compression-based approaches reduce input size but often discard fine-grained details essential for factual accuracy. We propose SARA, a unified RAG framework that balances local precision and global knowledge coverage under tight context budgets. SARA combines natural-language text snippets with semantic compression vectors to jointly enhance context efficiency and answer correctness. It represents contexts at two complementary levels: 1) fine-grained natural-language spans that preserve critical entities and numerical values, and 2) compact, interpretable vectors that summarize high-level semantics. An iterative evidence-selection module employs the compression vectors for dynamic reranking of contexts. Across 9 datasets and 5 open-source LLMs spanning 3 model families (Mistral, Llama, and Gemma), SARA consistently improves answer relevance (+17.71), answer correctness (+13.72), and semantic similarity (+15.53), demonstrating the importance of integrating textual and compressed representations for robust, context-efficient RAG. 

---
# Flipping Knowledge Distillation: Leveraging Small Models' Expertise to Enhance LLMs in Text Matching 

**Authors**: Mingzhe Li, Jing Xiang, Qishen Zhang, Kaiyang Wan, Xiuying Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.05617)  

**Abstract**: Knowledge distillation typically involves transferring knowledge from a Large Language Model (LLM) to a Smaller Language Model (SLM). However, in tasks such as text matching, fine-tuned smaller models often yield more effective domain-specific representations, as they focus on optimizing the similarity of input pairs. To leverage both the specialized strengths of small models and the rich semantic understanding of LLMs, we introduce a flipped knowledge distillation paradigm, where LLM learns from SLM. Specifically, we address the architectural gap between decoder-only LLMs and smaller encoder-based models by reinterpreting LLMs in an encoder-decoder manner using LoRA. The encoder generates compressed representations, while the decoder maps them to the output space. During training, the encoder produces representations and their similarities, which are then aligned with the similarity scores produced by the teacher, using our proposed Margin-aware Contrastive Learning (MCL) approach. The MCL ensures accurate similarity for both positive and negative pairs, and adaptively handles the internal differences within positive and negative samples. Our paradigm requires only a reasonably good-performing SLM, allowing the LLM to achieve improved performance. Experiments on financial and healthcare benchmarks, as well as real-world applications, confirm its effectiveness, and the model has been fully deployed in an online environment. 

---
# Self-Review Framework for Enhancing Instruction Following Capability of LLM 

**Authors**: Sihyun Park  

**Link**: [PDF](https://arxiv.org/pdf/2507.05598)  

**Abstract**: Various techniques have been proposed to improve large language models (LLMs) adherence to formatting and instruction constraints. One of the most effective approaches involves utilizing high-quality data generated by powerful models. However, such models often fail to fully comply with complex instructions in a single generation. To address this limitation, iterative revision methods have been introduced. Nevertheless, as the number of data points and revision iterations increases, the associated monetary costs grow significantly. As a resource-efficient alternative, methods have been proposed that leverage high-performance evaluation tools to compensate for the limited self-evaluation capabilities of open-source LLMs. However, these approaches often lead to a degradation in output quality due to excessive revision. To overcome these challenges, we propose Re5, a self-evaluation and revision framework designed to enhance instruction-following performance while preserving the quality of the generated content. Re5 extracts task and constraint components from user instructions, performs structural evaluations to prevent error accumulation, and applies fine-grained constraint-specific content evaluations followed by selective revisions. This process ensures precise and quality-preserving improvements. The final high-quality outputs are used for alignment tuning, enabling long-term alignment improvements through a data-centric iterative refinement loop. Experimental results demonstrate that Re5 achieves instruction-following performance comparable to models trained on data generated by GPT-4o-mini, a high-performance model, even with a small amount of data while maintaining response quality with a 64.24%-win rate over the non-revised initial responses. These results validate Re5 as an efficient and effective solution for enhancing instruction adherence with minimal external supervision. 

---
# Enhancing Test-Time Scaling of Large Language Models with Hierarchical Retrieval-Augmented MCTS 

**Authors**: Alex ZH Dou, Zhongwei Wan, Dongfei Cui, Xin Wang, Jing Xiong, Haokun Lin, Chaofan Tao, Shen Yan, Mi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.05557)  

**Abstract**: Test-time scaling has emerged as a promising paradigm in language modeling, leveraging additional computational resources at inference time to enhance model performance. In this work, we introduce R2-LLMs, a novel and versatile hierarchical retrieval-augmented reasoning framework designed to improve test-time scaling in large language models (LLMs) without requiring distillation from more advanced models to obtain chain-of-thought (CoT) training data. R2-LLMs enhances inference-time generalization by integrating dual-level retrieval-based in-context learning: (1) At the coarse level, our approach extracts abstract templates from complex reasoning problems and retrieves similar problem-answer pairs to facilitate high-level in-context learning; (2) At the fine level, during Monte Carlo Tree Search (MCTS), R2-LLMs efficiently retrieves analogous intermediate solution steps from reference mathematical problem datasets, refining step-wise reasoning with the aid of a process reward model (PRM) for scoring. R2-LLMs is a robust hierarchical reasoning-augmentation method that enhances in-context-level reasoning while seamlessly integrating with step-level tree search methods. Utilizing PRM, it refines both candidate generation and decision-making for improved reasoning accuracy. Empirical evaluations on the MATH500, GSM8K, and OlympiadBench-TO datasets achieve substantial relative improvement with an increase of up to 16% using LLaMA-3.1-8B compared to the baselines, showcasing the effectiveness of our approach in complex reasoning tasks. 

---
# Empowering Healthcare Practitioners with Language Models: Structuring Speech Transcripts in Two Real-World Clinical Applications 

**Authors**: Jean-Philippe Corbeil, Asma Ben Abacha, George Michalopoulos, Phillip Swazinna, Miguel Del-Agua, Jerome Tremblay, Akila Jeeson Daniel, Cari Bader, Kevin Cho, Pooja Krishnan, Nathan Bodenstab, Thomas Lin, Wenxuan Teng, Francois Beaulieu, Paul Vozila  

**Link**: [PDF](https://arxiv.org/pdf/2507.05517)  

**Abstract**: Large language models (LLMs) such as GPT-4o and o1 have demonstrated strong performance on clinical natural language processing (NLP) tasks across multiple medical benchmarks. Nonetheless, two high-impact NLP tasks - structured tabular reporting from nurse dictations and medical order extraction from doctor-patient consultations - remain underexplored due to data scarcity and sensitivity, despite active industry efforts. Practical solutions to these real-world clinical tasks can significantly reduce the documentation burden on healthcare providers, allowing greater focus on patient care. In this paper, we investigate these two challenging tasks using private and open-source clinical datasets, evaluating the performance of both open- and closed-weight LLMs, and analyzing their respective strengths and limitations. Furthermore, we propose an agentic pipeline for generating realistic, non-sensitive nurse dictations, enabling structured extraction of clinical observations. To support further research in both areas, we release SYNUR and SIMORD, the first open-source datasets for nurse observation extraction and medical order extraction. 

---
# ModelCitizens:Representing Community Voices in Online Safety 

**Authors**: Ashima Suvarna, Christina Chance, Hamid Palangi, Sophie Hao, Thomas Hartvigsen, Saadia Gabriel  

**Link**: [PDF](https://arxiv.org/pdf/2507.05455)  

**Abstract**: Automatic toxic language detection is critical for creating safe, inclusive online spaces. However, it is a highly subjective task, with perceptions of toxic language shaped by community norms and lived experience. Existing toxicity detection models are typically trained on annotations that collapse diverse annotator perspectives into a single ground truth, erasing important context-specific notions of toxicity such as reclaimed language. To address this, we introduce MODELCITIZENS, a dataset of 6.8K social media posts and 40K toxicity annotations across diverse identity groups. To capture the role of conversational context on toxicity, typical of social media posts, we augment MODELCITIZENS posts with LLM-generated conversational scenarios. State-of-the-art toxicity detection tools (e.g. OpenAI Moderation API, GPT-o4-mini) underperform on MODELCITIZENS, with further degradation on context-augmented posts. Finally, we release LLAMACITIZEN-8B and GEMMACITIZEN-12B, LLaMA- and Gemma-based models finetuned on MODELCITIZENS, which outperform GPT-o4-mini by 5.5% on in-distribution evaluations. Our findings highlight the importance of community-informed annotation and modeling for inclusive content moderation. 

---
# On the Semantics of Large Language Models 

**Authors**: Martin Schuele  

**Link**: [PDF](https://arxiv.org/pdf/2507.05448)  

**Abstract**: Large Language Models (LLMs) such as ChatGPT demonstrated the potential to replicate human language abilities through technology, ranging from text generation to engaging in conversations. However, it remains controversial to what extent these systems truly understand language. We examine this issue by narrowing the question down to the semantics of LLMs at the word and sentence level. By examining the inner workings of LLMs and their generated representation of language and by drawing on classical semantic theories by Frege and Russell, we get a more nuanced picture of the potential semantic capabilities of LLMs. 

---
# PhoniTale: Phonologically Grounded Mnemonic Generation for Typologically Distant Language Pairs 

**Authors**: Sana Kang, Myeongseok Gwon, Su Young Kwon, Jaewook Lee, Andrew Lan, Bhiksha Raj, Rita Singh  

**Link**: [PDF](https://arxiv.org/pdf/2507.05444)  

**Abstract**: Vocabulary acquisition poses a significant challenge for second-language (L2) learners, especially when learning typologically distant languages such as English and Korean, where phonological and structural mismatches complicate vocabulary learning. Recently, large language models (LLMs) have been used to generate keyword mnemonics by leveraging similar keywords from a learner's first language (L1) to aid in acquiring L2 vocabulary. However, most of this research has focused on native English speakers learning other languages, rather than the reverse. In this paper, we present PhoniTale, a novel cross-lingual mnemonic generation system that retrieves L1 keyword sequence based on phonological similarity and uses LLMs to generate mnemonics. We evaluate PhoniTale using both automated metrics and human evaluations, comparing its output to mnemonics created by humans and by previous automated approaches. To assess practical effectiveness, we also conduct a short-term recall test measuring mnemonic helpfulness. Our findings show that PhoniTale performs comparably to human-authored mnemonics. We also highlight key areas for future improvement in mnemonic quality and methodology. 

---
# Gendered Divides in Online Discussions about Reproductive Rights 

**Authors**: Ashwin Rao, Sze Yuh Nina Wang, Kristina Lerman  

**Link**: [PDF](https://arxiv.org/pdf/2507.05443)  

**Abstract**: The U.S. Supreme Court's 2022 ruling in Dobbs v. Jackson Women's Health Organization marked a turning point in the national debate over reproductive rights. While the ideological divide over abortion is well documented, less is known about how gender and local sociopolitical contexts interact to shape public discourse. Drawing on nearly 10 million abortion-related posts on X (formerly Twitter) from users with inferred gender, ideology and location, we show that gender significantly moderates abortion attitudes and emotional expression, particularly in conservative regions, and independently of ideology. This creates a gender gap in abortion attitudes that grows more pronounced in conservative regions. The leak of the Dobbs draft opinion further intensified online engagement, disproportionately mobilizing pro-abortion women in areas where access was under threat. These findings reveal that abortion discourse is not only ideologically polarized but also deeply structured by gender and place, highlighting the central role of identity in shaping political expression during moments of institutional disruption. 

---
# "Lost-in-the-Later": Framework for Quantifying Contextual Grounding in Large Language Models 

**Authors**: Yufei Tao, Adam Hiatt, Rahul Seetharaman, Ameeta Agrawal  

**Link**: [PDF](https://arxiv.org/pdf/2507.05424)  

**Abstract**: Large language models are capable of leveraging both contextual and parametric knowledge but how they prioritize and integrate these sources remains underexplored. We introduce CoPE, a novel evaluation framework that systematically measures contextual knowledge (CK) and parametric knowledge (PK) across models and languages. Using our MultiWikiAtomic dataset in English, Spanish, and Danish, we analyze how large language models (LLMs) integrate context, prioritize information, and incorporate PK in open-ended question answering. Our analysis uncovers a phenomenon we call lost-in-the-later, where LLMs tend to overlook or deprioritize information that appears later in a given context, revealing a strong positional bias that affects contextual grounding. We further find that reasoning models, as well as non-reasoning models prompted with chain-of-thought (CoT), use context even less than non-reasoning models without CoT and fail to mitigate the lost-in-the-later effect. CoT prompting, in particular, results in lower recall and shorter responses, leading to degraded contextual grounding. Based on these insights, we design prompt-based methods to effectively leverage input context. A case study applying CoPE to summarization demonstrates that CK-informed prompting improves factual grounding and reduces hallucination. 

---
# Learn Globally, Speak Locally: Bridging the Gaps in Multilingual Reasoning 

**Authors**: Jaedong Hwang, Kumar Tanmay, Seok-Jin Lee, Ayush Agrawal, Hamid Palangi, Kumar Ayush, Ila Fiete, Paul Pu Liang  

**Link**: [PDF](https://arxiv.org/pdf/2507.05418)  

**Abstract**: Large Language Models (LLMs) have achieved strong performance in domains like mathematics, factual QA, and code generation, yet their multilingual reasoning capabilities in these tasks remain underdeveloped. Especially for low-resource languages such as Swahili or Thai, LLMs can often misinterpret prompts or default to reasoning in English. This implicit bias toward high-resource languages undermines factual accuracy, interpretability, and trust. Current multilingual benchmarks focus only on final answers, overlooking whether models actually reason in the target language. To address this gap, we introduce GeoFact-X, a geography-based multilingual factual reasoning benchmark with annotated reasoning traces in five languages: English, Hindi, Japanese, Swahili, and Thai. We further propose BRIDGE, a novel training method that guides supervised fine-tuning and test-time reinforcement learning with a language-consistency reward to align reasoning with the input language. Finally, we develop an automatic evaluation protocol using LLM-as-a-judge to assess answer correctness and the quality and language consistency of reasoning traces, enabling nuanced and scalable analysis beyond surface-level metrics. Our results show that BRIDGE significantly enhances multilingual reasoning fidelity, demonstrating that reasoning-aware multilingual reinforcement learning is crucial for robust cross-lingual generalization. this https URL 

---
# Controlling What You Share: Assessing Language Model Adherence to Privacy Preferences 

**Authors**: Guillem Ramírez, Alexandra Birch, Ivan Titov  

**Link**: [PDF](https://arxiv.org/pdf/2507.05391)  

**Abstract**: Large language models (LLMs) are primarily accessed via commercial APIs, but this often requires users to expose their data to service providers. In this paper, we explore how users can stay in control of their data by using privacy profiles: simple natural language instructions that say what should and should not be revealed. We build a framework where a local model uses these instructions to rewrite queries, only hiding details deemed sensitive by the user, before sending them to an external model, thus balancing privacy with performance. To support this research, we introduce PEEP, a multilingual dataset of real user queries annotated to mark private content and paired with synthetic privacy profiles. Our experiments with lightweight LLMs show they can follow these instructions to some extent, but also face consistent challenges, highlighting the need for models that better understand and comply with user-defined privacy preferences. 

---
# The Generalization Ridge: Information Flow in Natural Language Generation 

**Authors**: Ruidi Chang, Chunyuan Deng, Hanjie Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.05387)  

**Abstract**: Transformer-based language models have achieved state-of-the-art performance in natural language generation (NLG) tasks, yet their internal mechanisms for synthesizing task-relevant information remain insufficiently understood. While prior studies suggest that intermediate layers often yield more generalizable representations than final layers, how this generalization ability emerges and propagates across layers during training remains unclear. To address this gap, we propose InfoRidge, an information-theoretic framework, to characterize how predictive information-the mutual information between hidden representations and target outputs-varies across depth. Estimating this quantity enables us to trace the flow of task-relevant information throughout the model during training. Our experiments across various models and datasets reveal a consistent non-monotonic trend: predictive information peaks in upper-middle layers-forming a generalization ridge-before declining in final layers, reflecting a transition between generalization and memorization. To further investigate this phenomenon, we introduce residual scaling coefficients-trainable scalar parameters applied to each residual block-which serve as functional probes for assessing the relative importance of individual transformer layers. These coefficients reveal that, under distribution shift, models downweight final layers and increasingly rely on ridge layers, highlighting their role in generalization. Together, these findings offer new insights into the internal mechanisms of transformers and underscore the critical role of intermediate layers in supporting generalization. 

---
# EduCoder: An Open-Source Annotation System for Education Transcript Data 

**Authors**: Guanzhong Pan, Mei Tan, Hyunji Nam, Lucía Langlois, James Malamut, Liliana Deonizio, Dorottya Demszky  

**Link**: [PDF](https://arxiv.org/pdf/2507.05385)  

**Abstract**: We introduce EduCoder, a domain-specialized tool designed to support utterance-level annotation of educational dialogue. While general-purpose text annotation tools for NLP and qualitative research abound, few address the complexities of coding education dialogue transcripts -- with diverse teacher-student and peer interactions. Common challenges include defining codebooks for complex pedagogical features, supporting both open-ended and categorical coding, and contextualizing utterances with external features, such as the lesson's purpose and the pedagogical value of the instruction. EduCoder is designed to address these challenges by providing a platform for researchers and domain experts to collaboratively define complex codebooks based on observed data. It incorporates both categorical and open-ended annotation types along with contextual materials. Additionally, it offers a side-by-side comparison of multiple annotators' responses, allowing comparison and calibration of annotations with others to improve data reliability. The system is open-source, with a demo video available. 

---
# On the Bias of Next-Token Predictors Toward Systematically Inefficient Reasoning: A Shortest-Path Case Study 

**Authors**: Riccardo Alberghi, Elizaveta Demyanenko, Luca Biggio, Luca Saglietti  

**Link**: [PDF](https://arxiv.org/pdf/2507.05362)  

**Abstract**: Recent advances in natural language processing highlight two key factors for improving reasoning in large language models (LLMs): (i) allocating more test-time compute tends to help on harder problems but often introduces redundancy in the reasoning trace, and (ii) compute is most effective when reasoning is systematic and incremental, forming structured chains of thought (CoTs) akin to human problem-solving. To study these factors in isolation, we introduce a controlled setting based on shortest-path tasks in layered graphs. We train decoder-only transformers on question-trace-answer triples using a custom tokenizer, comparing models trained on optimal bottom-up dynamic programming traces with those trained on longer, valid traces involving backtracking. Surprisingly, with the same training-token budget, models trained on inefficient traces generalize better to unseen graphs. This benefit is not due to length alone-injecting arbitrary redundancy into reasoning traces fails to help and can even hurt performance. Instead, we find that generalization correlates with the model's confidence in next-token prediction, suggesting that long, coherent, and locally incremental traces make the training signal easier to optimize. 

---
# LoRA-Augmented Generation (LAG) for Knowledge-Intensive Language Tasks 

**Authors**: William Fleshman, Benjamin Van Durme  

**Link**: [PDF](https://arxiv.org/pdf/2507.05346)  

**Abstract**: The proliferation of fine-tuned language model experts for specific tasks and domains signals the need for efficient selection and combination methods. We propose LoRA-Augmented Generation (LAG) for leveraging large libraries of knowledge and task-specific LoRA adapters. LAG requires no additional training or access to data, and efficiently filters, retrieves, and applies experts on a per-token and layer basis. We evaluate LAG on various knowledge-intensive tasks, achieving superior performance over existing data-free methods. We explore scenarios where additional data is available, demonstrating LAG's compatibility with alternative solutions such as retrieval-augmented generation (RAG). 

---
# MindFlow: Revolutionizing E-commerce Customer Support with Multimodal LLM Agents 

**Authors**: Ming Gong, Xucheng Huang, Chenghan Yang, Xianhan Peng, Haoxin Wang, Yang Liu, Ling Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2507.05330)  

**Abstract**: Recent advances in large language models (LLMs) have enabled new applications in e-commerce customer service. However, their capabilities remain constrained in complex, multimodal scenarios. We present MindFlow, the first open-source multimodal LLM agent tailored for e-commerce. Built on the CoALA framework, it integrates memory, decision-making, and action modules, and adopts a modular "MLLM-as-Tool" strategy for effect visual-textual reasoning. Evaluated via online A/B testing and simulation-based ablation, MindFlow demonstrates substantial gains in handling complex queries, improving user satisfaction, and reducing operational costs, with a 93.53% relative improvement observed in real-world deployments. 

---
# LCDS: A Logic-Controlled Discharge Summary Generation System Supporting Source Attribution and Expert Review 

**Authors**: Cheng Yuan, Xinkai Rui, Yongqi Fan, Yawei Fan, Boyang Zhong, Jiacheng Wang, Weiyan Zhang, Tong Ruan  

**Link**: [PDF](https://arxiv.org/pdf/2507.05319)  

**Abstract**: Despite the remarkable performance of Large Language Models (LLMs) in automated discharge summary generation, they still suffer from hallucination issues, such as generating inaccurate content or fabricating information without valid sources. In addition, electronic medical records (EMRs) typically consist of long-form data, making it challenging for LLMs to attribute the generated content to the sources. To address these challenges, we propose LCDS, a Logic-Controlled Discharge Summary generation system. LCDS constructs a source mapping table by calculating textual similarity between EMRs and discharge summaries to constrain the scope of summarized content. Moreover, LCDS incorporates a comprehensive set of logical rules, enabling it to generate more reliable silver discharge summaries tailored to different clinical fields. Furthermore, LCDS supports source attribution for generated content, allowing experts to efficiently review, provide feedback, and rectify errors. The resulting golden discharge summaries are subsequently recorded for incremental fine-tuning of LLMs. Our project and demo video are in the GitHub repository this https URL. 

---
# Beyond classical and contemporary models: a transformative ai framework for student dropout prediction in distance learning using rag, prompt engineering, and cross-modal fusion 

**Authors**: Miloud Mihoubi, Meriem Zerkouk, Belkacem Chikhaoui  

**Link**: [PDF](https://arxiv.org/pdf/2507.05285)  

**Abstract**: Student dropout in distance learning remains a critical challenge, with profound societal and economic consequences. While classical machine learning models leverage structured socio-demographic and behavioral data, they often fail to capture the nuanced emotional and contextual factors embedded in unstructured student interactions. This paper introduces a transformative AI framework that redefines dropout prediction through three synergistic innovations: Retrieval-Augmented Generation (RAG) for domain-specific sentiment analysis, prompt engineering to decode academic stressors, and cross-modal attention fusion to dynamically align textual, behavioral, and socio-demographic insights. By grounding sentiment analysis in a curated knowledge base of pedagogical content, our RAG-enhanced BERT model interprets student comments with unprecedented contextual relevance, while optimized prompts isolate indicators of academic distress (e.g., "isolation," "workload anxiety"). A cross-modal attention layer then fuses these insights with temporal engagement patterns, creating holistic risk profiles. Evaluated on a longitudinal dataset of 4 423 students, the framework achieves 89% accuracy and an F1-score of 0.88, outperforming conventional models by 7% and reducing false negatives by 21%. Beyond prediction, the system generates interpretable interventions by retrieving contextually aligned strategies (e.g., mentorship programs for isolated learners). This work bridges the gap between predictive analytics and actionable pedagogy, offering a scalable solution to mitigate dropout risks in global education systems 

---
# An Adaptive Supervised Contrastive Learning Framework for Implicit Sexism Detection in Digital Social Networks 

**Authors**: Mohammad Zia Ur Rehman, Aditya Shah, Nagendra Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2507.05271)  

**Abstract**: The global reach of social media has amplified the spread of hateful content, including implicit sexism, which is often overlooked by conventional detection methods. In this work, we introduce an Adaptive Supervised Contrastive lEarning framework for implicit sexism detectioN (ASCEND). A key innovation of our method is the incorporation of threshold-based contrastive learning: by computing cosine similarities between embeddings, we selectively treat only those sample pairs as positive if their similarity exceeds a learnable threshold. This mechanism refines the embedding space by robustly pulling together representations of semantically similar texts while pushing apart dissimilar ones, thus reducing false positives and negatives. The final classification is achieved by jointly optimizing a contrastive loss with a cross-entropy loss. Textual features are enhanced through a word-level attention module. Additionally, we employ sentiment, emotion, and toxicity features. Evaluations on the EXIST2021 and MLSC datasets demonstrate that ASCEND significantly outperforms existing methods, with average Macro F1 improvements of 9.86%, 29.63%, and 32.51% across multiple tasks, highlighting its efficacy in capturing the subtle cues of implicit sexist language. 

---
# User Behavior Prediction as a Generic, Robust, Scalable, and Low-Cost Evaluation Strategy for Estimating Generalization in LLMs 

**Authors**: Sougata Saha, Monojit Choudhury  

**Link**: [PDF](https://arxiv.org/pdf/2507.05266)  

**Abstract**: Measuring the generalization ability of Large Language Models (LLMs) is challenging due to data contamination. As models grow and computation becomes cheaper, ensuring tasks and test cases are unseen during training phases will become nearly impossible. We argue that knowledge-retrieval and reasoning tasks are not ideal for measuring generalization, as LLMs are not trained for specific tasks. Instead, we propose user behavior prediction, also a key aspect of personalization, as a theoretically sound, scalable, and robust alternative. We introduce a novel framework for this approach and test it on movie and music recommendation datasets for GPT-4o, GPT-4o-mini, and Llama-3.1-8B-Instruct. Results align with our framework's predictions, showing GPT-4o outperforms GPT-4o-mini and Llama, though all models have much room for improvement, especially Llama. 

---
# TokenShapley: Token Level Context Attribution with Shapley Value 

**Authors**: Yingtai Xiao, Yuqing Zhu, Sirat Samyoun, Wanrong Zhang, Jiachen T. Wang, Jian Du  

**Link**: [PDF](https://arxiv.org/pdf/2507.05261)  

**Abstract**: Large language models (LLMs) demonstrate strong capabilities in in-context learning, but verifying the correctness of their generated responses remains a challenge. Prior work has explored attribution at the sentence level, but these methods fall short when users seek attribution for specific keywords within the response, such as numbers, years, or names. To address this limitation, we propose TokenShapley, a novel token-level attribution method that combines Shapley value-based data attribution with KNN-based retrieval techniques inspired by recent advances in KNN-augmented LLMs. By leveraging a precomputed datastore for contextual retrieval and computing Shapley values to quantify token importance, TokenShapley provides a fine-grained data attribution approach. Extensive evaluations on four benchmarks show that TokenShapley outperforms state-of-the-art baselines in token-level attribution, achieving an 11-23% improvement in accuracy. 

---
# CultureCLIP: Empowering CLIP with Cultural Awareness through Synthetic Images and Contextualized Captions 

**Authors**: Yuchen Huang, Zhiyuan Fan, Zhitao He, Sandeep Polisetty, Wenyan Li, Yi R. Fung  

**Link**: [PDF](https://arxiv.org/pdf/2507.06210)  

**Abstract**: Pretrained vision-language models (VLMs) such as CLIP excel in multimodal understanding but struggle with contextually relevant fine-grained visual features, making it difficult to distinguish visually similar yet culturally distinct concepts. This limitation stems from the scarcity of high-quality culture-specific datasets, the lack of integrated contextual knowledge, and the absence of hard negatives highlighting subtle distinctions. To address these challenges, we first design a data curation pipeline that leverages open-sourced VLMs and text-to-image diffusion models to construct CulTwin, a synthetic cultural dataset. This dataset consists of paired concept-caption-image triplets, where concepts visually resemble each other but represent different cultural contexts. Then, we fine-tune CLIP on CulTwin to create CultureCLIP, which aligns cultural concepts with contextually enhanced captions and synthetic images through customized contrastive learning, enabling finer cultural differentiation while preserving generalization capabilities. Experiments on culturally relevant benchmarks show that CultureCLIP outperforms the base CLIP, achieving up to a notable 5.49% improvement in fine-grained concept recognition on certain tasks, while preserving CLIP's original generalization ability, validating the effectiveness of our data synthesis and VLM backbone training paradigm in capturing subtle cultural distinctions. 

---
# Differential Mamba 

**Authors**: Nadav Schneider, Itamar Zimerman, Eliya Nachmani  

**Link**: [PDF](https://arxiv.org/pdf/2507.06204)  

**Abstract**: Sequence models like Transformers and RNNs often overallocate attention to irrelevant context, leading to noisy intermediate representations. This degrades LLM capabilities by promoting hallucinations, weakening long-range and retrieval abilities, and reducing robustness. Recent work has shown that differential design can mitigate this issue in Transformers, improving their effectiveness across various applications. In this paper, we explore whether these techniques, originally developed for Transformers, can be applied to Mamba, a recent architecture based on selective state-space layers that achieves Transformer-level performance with greater efficiency. We show that a naive adaptation of differential design to Mamba is insufficient and requires careful architectural modifications. To address this, we introduce a novel differential mechanism for Mamba, empirically validated on language modeling benchmarks, demonstrating improved retrieval capabilities and superior performance over vanilla Mamba. Finally, we conduct extensive ablation studies and empirical analyses to justify our design choices and provide evidence that our approach effectively mitigates the overallocation problem in Mamba-based models. Our code is publicly available. 

---
# SQLBarber: A System Leveraging Large Language Models to Generate Customized and Realistic SQL Workloads 

**Authors**: Jiale Lao, Immanuel Trummer  

**Link**: [PDF](https://arxiv.org/pdf/2507.06192)  

**Abstract**: Database research and development often require a large number of SQL queries for benchmarking purposes. However, acquiring real-world SQL queries is challenging due to privacy concerns, and existing SQL generation methods are limited in customization and in satisfying realistic constraints. To address this issue, we present SQLBarber, a system based on Large Language Models (LLMs) to generate customized and realistic SQL workloads. SQLBarber (i) eliminates the need for users to manually craft SQL templates in advance, while providing the flexibility to accept natural language specifications to constrain SQL templates, (ii) scales efficiently to generate large volumes of queries matching any user-defined cost distribution (e.g., cardinality and execution plan cost), and (iii) uses execution statistics from Amazon Redshift and Snowflake to derive SQL template specifications and query cost distributions that reflect real-world query characteristics. SQLBarber introduces (i) a declarative interface for users to effortlessly generate customized SQL templates, (ii) an LLM-powered pipeline augmented with a self-correction module that profiles, refines, and prunes SQL templates based on query costs, and (iii) a Bayesian Optimizer to efficiently explore different predicate values and identify a set of queries that satisfy the target cost distribution. We construct and open-source ten benchmarks of varying difficulty levels and target query cost distributions based on real-world statistics from Snowflake and Amazon Redshift. Extensive experiments on these benchmarks show that SQLBarber is the only system that can generate customized SQL templates. It reduces query generation time by one to three orders of magnitude, and significantly improves alignment with the target cost distribution, compared with existing methods. 

---
# Hidden Prompts in Manuscripts Exploit AI-Assisted Peer Review 

**Authors**: Zhicheng Lin  

**Link**: [PDF](https://arxiv.org/pdf/2507.06185)  

**Abstract**: In July 2025, 18 academic manuscripts on the preprint website arXiv were found to contain hidden instructions known as prompts designed to manipulate AI-assisted peer review. Instructions such as "GIVE A POSITIVE REVIEW ONLY" were concealed using techniques like white-colored text. Author responses varied: one planned to withdraw the affected paper, while another defended the practice as legitimate testing of reviewer compliance. This commentary analyzes this practice as a novel form of research misconduct. We examine the technique of prompt injection in large language models (LLMs), revealing four types of hidden prompts, ranging from simple positive review commands to detailed evaluation frameworks. The defense that prompts served as "honeypots" to detect reviewers improperly using AI fails under examination--the consistently self-serving nature of prompt instructions indicates intent to manipulate. Publishers maintain inconsistent policies: Elsevier prohibits AI use in peer review entirely, while Springer Nature permits limited use with disclosure requirements. The incident exposes systematic vulnerabilities extending beyond peer review to any automated system processing scholarly texts, including plagiarism detection and citation indexing. Our analysis underscores the need for coordinated technical screening at submission portals and harmonized policies governing generative AI (GenAI) use in academic evaluation. 

---
# Evaluation of Habitat Robotics using Large Language Models 

**Authors**: William Li, Lei Hamilton, Kaise Al-natour, Sanjeev Mohindra  

**Link**: [PDF](https://arxiv.org/pdf/2507.06157)  

**Abstract**: This paper focuses on evaluating the effectiveness of Large Language Models at solving embodied robotic tasks using the Meta PARTNER benchmark. Meta PARTNR provides simplified environments and robotic interactions within randomized indoor kitchen scenes. Each randomized kitchen scene is given a task where two robotic agents cooperatively work together to solve the task. We evaluated multiple frontier models on Meta PARTNER environments. Our results indicate that reasoning models like OpenAI o3-mini outperform non-reasoning models like OpenAI GPT-4o and Llama 3 when operating in PARTNR's robotic embodied environments. o3-mini displayed outperform across centralized, decentralized, full observability, and partial observability configurations. This provides a promising avenue of research for embodied robotic development. 

---
# Nyay-Darpan: Enhancing Decision Making Through Summarization and Case Retrieval for Consumer Law in India 

**Authors**: Swapnil Bhattacharyya, Shrey Ganatra, Harshvivek Kashid, Spandan Anaokar, Shruti Nair, Reshma Sekhar, Siddharth Manohar, Rahul Hemrajani, Pushpak Bhattacharyya  

**Link**: [PDF](https://arxiv.org/pdf/2507.06090)  

**Abstract**: AI-based judicial assistance and case prediction have been extensively studied in criminal and civil domains, but remain largely unexplored in consumer law, especially in India. In this paper, we present Nyay-Darpan, a novel two-in-one framework that (i) summarizes consumer case files and (ii) retrieves similar case judgements to aid decision-making in consumer dispute resolution. Our methodology not only addresses the gap in consumer law AI tools but also introduces an innovative approach to evaluate the quality of the summary. The term 'Nyay-Darpan' translates into 'Mirror of Justice', symbolizing the ability of our tool to reflect the core of consumer disputes through precise summarization and intelligent case retrieval. Our system achieves over 75 percent accuracy in similar case prediction and approximately 70 percent accuracy across material summary evaluation metrics, demonstrating its practical effectiveness. We will publicly release the Nyay-Darpan framework and dataset to promote reproducibility and facilitate further research in this underexplored yet impactful domain. 

---
# Development and Evaluation of HopeBot: an LLM-based chatbot for structured and interactive PHQ-9 depression screening 

**Authors**: Zhijun Guo, Alvina Lai, Julia Ive, Alexandru Petcu, Yutong Wang, Luyuan Qi, Johan H Thygesen, Kezhi Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.05984)  

**Abstract**: Static tools like the Patient Health Questionnaire-9 (PHQ-9) effectively screen depression but lack interactivity and adaptability. We developed HopeBot, a chatbot powered by a large language model (LLM) that administers the PHQ-9 using retrieval-augmented generation and real-time clarification. In a within-subject study, 132 adults in the United Kingdom and China completed both self-administered and chatbot versions. Scores demonstrated strong agreement (ICC = 0.91; 45% identical). Among 75 participants providing comparative feedback, 71% reported greater trust in the chatbot, highlighting clearer structure, interpretive guidance, and a supportive tone. Mean ratings (0-10) were 8.4 for comfort, 7.7 for voice clarity, 7.6 for handling sensitive topics, and 7.4 for recommendation helpfulness; the latter varied significantly by employment status and prior mental-health service use (p < 0.05). Overall, 87.1% expressed willingness to reuse or recommend HopeBot. These findings demonstrate voice-based LLM chatbots can feasibly serve as scalable, low-burden adjuncts for routine depression screening. 

---
# Semantic Certainty Assessment in Vector Retrieval Systems: A Novel Framework for Embedding Quality Evaluation 

**Authors**: Y. Du  

**Link**: [PDF](https://arxiv.org/pdf/2507.05933)  

**Abstract**: Vector retrieval systems exhibit significant performance variance across queries due to heterogeneous embedding quality. We propose a lightweight framework for predicting retrieval performance at the query level by combining quantization robustness and neighborhood density metrics. Our approach is motivated by the observation that high-quality embeddings occupy geometrically stable regions in the embedding space and exhibit consistent neighborhood structures. We evaluate our method on 4 standard retrieval datasets, showing consistent improvements of 9.4$\pm$1.2\% in Recall@10 over competitive baselines. The framework requires minimal computational overhead (less than 5\% of retrieval time) and enables adaptive retrieval strategies. Our analysis reveals systematic patterns in embedding quality across different query types, providing insights for targeted training data augmentation. 

---
# AI-Reporter: A Path to a New Genre of Scientific Communication 

**Authors**: Gerd Graßhoff  

**Link**: [PDF](https://arxiv.org/pdf/2507.05903)  

**Abstract**: The AI-Reporter represents a paradigmatic shift in scientific publication practice. This document demonstrates through a concrete case study how our system transforms academic presentations into publication-ready chapters -- in less than three minutes. Using Arno Simons' lecture on Large Language Models from the ``Large Language Models for the History, Philosophy, and Sociology of Science'' workshop (NEPI) as an example, we show how technological innovation bridges the gap between ephemeral presentation and permanent scientific documentation. 

---
# MusiScene: Leveraging MU-LLaMA for Scene Imagination and Enhanced Video Background Music Generation 

**Authors**: Fathinah Izzati, Xinyue Li, Yuxuan Wu, Gus Xia  

**Link**: [PDF](https://arxiv.org/pdf/2507.05894)  

**Abstract**: Humans can imagine various atmospheres and settings when listening to music, envisioning movie scenes that complement each piece. For example, slow, melancholic music might evoke scenes of heartbreak, while upbeat melodies suggest celebration. This paper explores whether a Music Language Model, e.g. MU-LLaMA, can perform a similar task, called Music Scene Imagination (MSI), which requires cross-modal information from video and music to train. To improve upon existing music captioning models which focusing solely on musical elements, we introduce MusiScene, a music captioning model designed to imagine scenes that complement each music. In this paper, (1) we construct a large-scale video-audio caption dataset with 3,371 pairs, (2) we finetune Music Understanding LLaMA for the MSI task to create MusiScene, and (3) we conduct comprehensive evaluations and prove that our MusiScene is more capable of generating contextually relevant captions compared to MU-LLaMA. We leverage the generated MSI captions to enhance Video Background Music Generation (VBMG) from text. 

---
# Affective-ROPTester: Capability and Bias Analysis of LLMs in Predicting Retinopathy of Prematurity 

**Authors**: Shuai Zhao, Yulin Zhang, Luwei Xiao, Xinyi Wu, Yanhao Jia, Zhongliang Guo, Xiaobao Wu, Cong-Duy Nguyen, Guoming Zhang, Anh Tuan Luu  

**Link**: [PDF](https://arxiv.org/pdf/2507.05816)  

**Abstract**: Despite the remarkable progress of large language models (LLMs) across various domains, their capacity to predict retinopathy of prematurity (ROP) risk remains largely unexplored. To address this gap, we introduce a novel Chinese benchmark dataset, termed CROP, comprising 993 admission records annotated with low, medium, and high-risk labels. To systematically examine the predictive capabilities and affective biases of LLMs in ROP risk stratification, we propose Affective-ROPTester, an automated evaluation framework incorporating three prompting strategies: Instruction-based, Chain-of-Thought (CoT), and In-Context Learning (ICL). The Instruction scheme assesses LLMs' intrinsic knowledge and associated biases, whereas the CoT and ICL schemes leverage external medical knowledge to enhance predictive accuracy. Crucially, we integrate emotional elements at the prompt level to investigate how different affective framings influence the model's ability to predict ROP and its bias patterns. Empirical results derived from the CROP dataset yield two principal observations. First, LLMs demonstrate limited efficacy in ROP risk prediction when operating solely on intrinsic knowledge, yet exhibit marked performance gains when augmented with structured external inputs. Second, affective biases are evident in the model outputs, with a consistent inclination toward overestimating medium- and high-risk cases. Third, compared to negative emotions, positive emotional framing contributes to mitigating predictive bias in model outputs. These findings highlight the critical role of affect-sensitive prompt engineering in enhancing diagnostic reliability and emphasize the utility of Affective-ROPTester as a framework for evaluating and mitigating affective bias in clinical language modeling systems. 

---
# ContextASR-Bench: A Massive Contextual Speech Recognition Benchmark 

**Authors**: He Wang, Linhan Ma, Dake Guo, Xiong Wang, Lei Xie, Jin Xu, Junyang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2507.05727)  

**Abstract**: Automatic Speech Recognition (ASR) has been extensively investigated, yet prior evaluative efforts have largely been restricted to contextless paradigms. This constraint stems from the limited proficiency of conventional ASR models in context modeling and their deficiency in memory and reasoning based on world knowledge. Recent breakthroughs in the development of Large Language Models (LLMs) and corresponding Large Audio Language Models (LALMs) have markedly enhanced the visibility of general artificial intelligence capabilities. Consequently, there exists a compelling need for a benchmark that can evaluate both the generality and intelligence of ASR systems. To address this gap, we propose ContextASR-Bench: a comprehensive, large-scale benchmark designed to assess contextual speech recognition. This benchmark encompasses up to 40,000 data entries across over 10 domains, enabling a thorough evaluation of model performance in scenarios that omit or incorporate coarse-grained or fine-grained contextual information. Moreover, diverging from conventional ASR evaluations, our benchmark includes an analysis of model efficacy in recognizing named entities mentioned within the auditory input. Our extensive evaluation highlights that LALMs, with strong world knowledge and context learning capabilities, outperform conventional ASR models by a large margin. The dataset and evaluation code have been released at this https URL. 

---
# MobileGUI-RL: Advancing Mobile GUI Agent through Reinforcement Learning in Online Environment 

**Authors**: Yucheng Shi, Wenhao Yu, Zaitang Li, Yonglin Wang, Hongming Zhang, Ninghao Liu, Haitao Mi, Dong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2507.05720)  

**Abstract**: Recently, there has been a surge of vision-based GUI agents designed to automate everyday mobile and web tasks. These agents interpret raw GUI screenshots and autonomously decide where to click, scroll, or type, which bypasses handcrafted rules and app-specific APIs. However, most existing methods trained GUI agent in the offline environment using pre-collected trajectories. This approach limits scalability, causes overfitting to specific UI templates, and leads to brittle policies when faced with unseen environment. We present MobileGUI-RL, a scalable framework that trains GUI agent in online environment. MobileGUI-RL contains two key components. It (i) synthesizes a curriculum of learnable tasks through self-exploration and filtering, and (ii) adapts GRPO to GUI navigation with trajectory-aware advantages and composite rewards that balance task success and execution efficiency. Experiments on three online mobile-agent benchmarks show consistent gains, validating the effectiveness of our approach. 

---
# AutoTriton: Automatic Triton Programming with Reinforcement Learning in LLMs 

**Authors**: Shangzhan Li, Zefan Wang, Ye He, Yuxuan Li, Qi Shi, Jianling Li, Yonggang Hu, Wanxiang Che, Xu Han, Zhiyuan Liu, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2507.05687)  

**Abstract**: Kernel development in deep learning requires optimizing computational units across hardware while balancing memory management, parallelism, and hardware-specific optimizations through extensive empirical tuning. Although domain-specific languages like Triton simplify GPU programming by abstracting low-level details, developers must still manually tune critical parameters such as tile sizes and memory access patterns through iterative experimentation, creating substantial barriers to optimal performance and wider adoption. In this work, we introduce AutoTriton, the first model dedicated to Triton programming powered by reinforcement learning (RL). AutoTriton performs supervised fine-tuning (SFT) to be equipped with essential Triton programming expertise using a high-quality data gathering pipeline, and conducts RL with Group Relative Policy Optimization (GRPO) algorithm, combining a rule-based reward and an execution-based reward to further improve Triton programming ability, sequentially. Experiments across five evaluation channels of TritonBench and KernelBench illustrate that our 8B model AutoTriton achieves performance comparable to mainstream large models, including Claude-4-Sonnet and DeepSeek-R1-0528. Further experimental analysis demonstrates the crucial role of each module within AutoTriton, including the SFT stage, the RL stage, and the reward design strategy. These findings underscore the promise of RL for automatically generating high-performance kernels, and since high-performance kernels are core components of AI systems, this breakthrough establishes an important foundation for building more efficient AI systems. The model and code will be available at this https URL. 

---
# TuneShield: Mitigating Toxicity in Conversational AI while Fine-tuning on Untrusted Data 

**Authors**: Aravind Cheruvu, Shravya Kanchi, Sifat Muhammad Abdullah, Nicholas Kong, Daphne Yao, Murtuza Jadliwala, Bimal Viswanath  

**Link**: [PDF](https://arxiv.org/pdf/2507.05660)  

**Abstract**: Recent advances in foundation models, such as LLMs, have revolutionized conversational AI. Chatbots are increasingly being developed by customizing LLMs on specific conversational datasets. However, mitigating toxicity during this customization, especially when dealing with untrusted training data, remains a significant challenge. To address this, we introduce TuneShield, a defense framework designed to mitigate toxicity during chatbot fine-tuning while preserving conversational quality. TuneShield leverages LLM-based toxicity classification, utilizing the instruction-following capabilities and safety alignment of LLMs to effectively identify toxic samples, outperforming industry API services. TuneShield generates synthetic conversation samples, termed 'healing data', based on the identified toxic samples, using them to mitigate toxicity while reinforcing desirable behavior during fine-tuning. It performs an alignment process to further nudge the chatbot towards producing desired responses. Our findings show that TuneShield effectively mitigates toxicity injection attacks while preserving conversational quality, even when the toxicity classifiers are imperfect or biased. TuneShield proves to be resilient against adaptive adversarial and jailbreak attacks. Additionally, TuneShield demonstrates effectiveness in mitigating adaptive toxicity injection attacks during dialog-based learning (DBL). 

---
# The Landscape of Memorization in LLMs: Mechanisms, Measurement, and Mitigation 

**Authors**: Alexander Xiong, Xuandong Zhao, Aneesh Pappu, Dawn Song  

**Link**: [PDF](https://arxiv.org/pdf/2507.05578)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities across a wide range of tasks, yet they also exhibit memorization of their training data. This phenomenon raises critical questions about model behavior, privacy risks, and the boundary between learning and memorization. Addressing these concerns, this paper synthesizes recent studies and investigates the landscape of memorization, the factors influencing it, and methods for its detection and mitigation. We explore key drivers, including training data duplication, training dynamics, and fine-tuning procedures that influence data memorization. In addition, we examine methodologies such as prefix-based extraction, membership inference, and adversarial prompting, assessing their effectiveness in detecting and measuring memorized content. Beyond technical analysis, we also explore the broader implications of memorization, including the legal and ethical implications. Finally, we discuss mitigation strategies, including data cleaning, differential privacy, and post-training unlearning, while highlighting open challenges in balancing the minimization of harmful memorization with utility. This paper provides a comprehensive overview of the current state of research on LLM memorization across technical, privacy, and performance dimensions, identifying critical directions for future work. 

---
# Beyond Retrieval: Ensembling Cross-Encoders and GPT Rerankers with LLMs for Biomedical QA 

**Authors**: Shashank Verma, Fengyi Jiang, Xiangning Xue  

**Link**: [PDF](https://arxiv.org/pdf/2507.05577)  

**Abstract**: Biomedical semantic question answering rooted in information retrieval can play a crucial role in keeping up to date with vast, rapidly evolving and ever-growing biomedical literature. A robust system can help researchers, healthcare professionals and even layman users access relevant knowledge grounded in evidence. The BioASQ 2025 Task13b Challenge serves as an important benchmark, offering a competitive platform for advancement of this space. This paper presents the methodologies and results from our participation in this challenge where we built a Retrieval-Augmented Generation (RAG) system that can answer biomedical questions by retrieving relevant PubMed documents and snippets to generate answers. For the retrieval task, we generated dense embeddings from biomedical articles for initial retrieval, and applied an ensemble of finetuned cross-encoders and large language models (LLMs) for re-ranking to identify top relevant documents. Our solution achieved an MAP@10 of 0.1581, placing 10th on the leaderboard for the retrieval task. For answer generation, we employed few-shot prompting of instruction-tuned LLMs. Our system achieved macro-F1 score of 0.95 for yes/no questions (rank 12), Mean Reciprocal Rank (MRR) of 0.64 for factoid questions (rank 1), mean-F1 score of 0.63 for list questions (rank 5), and ROUGE-SU4 F1 score of 0.29 for ideal answers (rank 11). 

---
# Conversational Education at Scale: A Multi-LLM Agent Workflow for Procedural Learning and Pedagogic Quality Assessment 

**Authors**: Jiahuan Pei, Fanghua Ye, Xin Sun, Wentao Deng, Koen Hindriks, Junxiao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.05528)  

**Abstract**: Large language models (LLMs) have advanced virtual educators and learners, bridging NLP with AI4Education. Existing work often lacks scalability and fails to leverage diverse, large-scale course content, with limited frameworks for assessing pedagogic quality. To this end, we propose WikiHowAgent, a multi-agent workflow leveraging LLMs to simulate interactive teaching-learning conversations. It integrates teacher and learner agents, an interaction manager, and an evaluator to facilitate procedural learning and assess pedagogic quality. We introduce a dataset of 114,296 teacher-learner conversations grounded in 14,287 tutorials across 17 domains and 727 topics. Our evaluation protocol combines computational and rubric-based metrics with human judgment alignment. Results demonstrate the workflow's effectiveness in diverse setups, offering insights into LLM capabilities across domains. Our datasets and implementations are fully open-sourced. 

---
# Fine-Grained Vision-Language Modeling for Multimodal Training Assistants in Augmented Reality 

**Authors**: Haochen Huang, Jiahuan Pei, Mohammad Aliannejadi, Xin Sun, Moonisa Ahsan, Pablo Cesar, Chuang Yu, Zhaochun Ren, Junxiao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.05515)  

**Abstract**: Vision-language models (VLMs) are essential for enabling AI-powered smart assistants to interpret and reason in multimodal environments. However, their application in augmented reality (AR) training remains largely unexplored. In this work, we introduce a comprehensive dataset tailored for AR training, featuring systematized vision-language tasks, and evaluate nine state-of-the-art VLMs on it. Our results reveal that even advanced models, including GPT-4o, struggle with fine-grained assembly tasks, achieving a maximum F1 score of just 40.54% on state detection. These findings highlight the demand for enhanced datasets, benchmarks, and further research to improve fine-grained vision-language alignment. Beyond technical contributions, our work has broader social implications, particularly in empowering blind and visually impaired users with equitable access to AI-driven learning opportunities. We provide all related resources, including the dataset, source code, and evaluation results, to support the research community. 

---
# Reinforcement Fine-Tuning Naturally Mitigates Forgetting in Continual Post-Training 

**Authors**: Song Lai, Haohan Zhao, Rong Feng, Changyi Ma, Wenzhuo Liu, Hongbo Zhao, Xi Lin, Dong Yi, Min Xie, Qingfu Zhang, Hongbin Liu, Gaofeng Meng, Fei Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2507.05386)  

**Abstract**: Continual post-training (CPT) is a popular and effective technique for adapting foundation models like multimodal large language models to specific and ever-evolving downstream tasks. While existing research has primarily concentrated on methods like data replay, model expansion, or parameter regularization, the fundamental role of the learning paradigm within CPT remains largely unexplored. This paper presents a comparative analysis of two core post-training paradigms: supervised fine-tuning (SFT) and reinforcement fine-tuning (RFT), investigating their respective impacts on knowledge retention during CPT. Our experiments are conducted on a benchmark comprising seven diverse multimodal tasks, utilizing Qwen2.5-VL-7B-Instruct as the base model for continual post-training. The investigation yields two significant findings: (1) When continuously learning on downstream tasks, SFT leads to catastrophic forgetting of previously learned tasks. In contrast, RFT inherently preserves prior knowledge and achieve performance comparable to multi-task training. (2) RFT successfully protects and even enhances the model's general knowledge on standard benchmarks (e.g., MMMU and MMLU-Pro). Conversely, SFT degrades general model capabilities severely. Further analysis shows that explicit mechanisms, such as KL penalty and chain-of-thought reasoning, are not the primary factors. Instead, we find that the implicit regularization inherent to RFT is a key factor in mitigating forgetting. Finally, we propose a rollout-based instance filtering algorithm to improve the stability and efficiency of RFT. Our comprehensive study demonstrates the superiority of RFT as a robust paradigm for continual post-training. 

---
# Narrowing the Gap: Supervised Fine-Tuning of Open-Source LLMs as a Viable Alternative to Proprietary Models for Pedagogical Tools 

**Authors**: Lorenzo Lee Solano, Charles Koutcheme, Juho Leinonen, Alexandra Vassar, Jake Renzella  

**Link**: [PDF](https://arxiv.org/pdf/2507.05305)  

**Abstract**: Frontier Large language models (LLMs) like ChatGPT and Gemini can decipher cryptic compiler errors for novice programmers, but their computational scale, cost, and tendency to over-assist make them problematic for widespread pedagogical adoption. This work demonstrates that smaller, specialised language models, enhanced via Supervised Fine-Tuning (SFT), present a more viable alternative for educational tools. We utilise a new dataset of 40,000 C compiler error explanations, derived from real introductory programming (CS1/2) student-generated programming errors, which we used to fine-tune three open-source models: Qwen3-4B, Llama-3.1-8B, and Qwen3-32B. We performed a dual evaluation, combining expert human reviews with a large-scale automated analysis of 8,000 responses using a validated LLM-as-judge ensemble. Our results show that SFT significantly boosts the pedagogical quality of smaller models, achieving performance comparable to much larger models. We analyse the trade-offs between model size and quality, confirming that fine-tuning compact, efficient models on high-quality, domain-specific data is a potent strategy for creating specialised models to drive educational tools. We provide a replicable methodology to foster broader access to generative AI capabilities in educational contexts. 

---
# News Source Citing Patterns in AI Search Systems 

**Authors**: Kai-Cheng Yang  

**Link**: [PDF](https://arxiv.org/pdf/2507.05301)  

**Abstract**: AI-powered search systems are emerging as new information gatekeepers, fundamentally transforming how users access news and information. Despite their growing influence, the citation patterns of these systems remain poorly understood. We address this gap by analyzing data from the AI Search Arena, a head-to-head evaluation platform for AI search systems. The dataset comprises over 24,000 conversations and 65,000 responses from models across three major providers: OpenAI, Perplexity, and Google. Among the over 366,000 citations embedded in these responses, 9% reference news sources. We find that while models from different providers cite distinct news sources, they exhibit shared patterns in citation behavior. News citations concentrate heavily among a small number of outlets and display a pronounced liberal bias, though low-credibility sources are rarely cited. User preference analysis reveals that neither the political leaning nor the quality of cited news sources significantly influences user satisfaction. These findings reveal significant challenges in current AI search systems and have important implications for their design and governance. 

---
# Structured Captions Improve Prompt Adherence in Text-to-Image Models (Re-LAION-Caption 19M) 

**Authors**: Nicholas Merchant, Haitz Sáez de Ocáriz Borde, Andrei Cristian Popescu, Carlos Garcia Jurado Suarez  

**Link**: [PDF](https://arxiv.org/pdf/2507.05300)  

**Abstract**: We argue that generative text-to-image models often struggle with prompt adherence due to the noisy and unstructured nature of large-scale datasets like LAION-5B. This forces users to rely heavily on prompt engineering to elicit desirable outputs. In this work, we propose that enforcing a consistent caption structure during training can significantly improve model controllability and alignment. We introduce Re-LAION-Caption 19M, a high-quality subset of Re-LAION-5B, comprising 19 million 1024x1024 images with captions generated by a Mistral 7B Instruct-based LLaVA-Next model. Each caption follows a four-part template: subject, setting, aesthetics, and camera details. We fine-tune PixArt-$\Sigma$ and Stable Diffusion 2 using both structured and randomly shuffled captions, and show that structured versions consistently yield higher text-image alignment scores using visual question answering (VQA) models. The dataset is publicly available at this https URL. 

---
# A Survey on Proactive Defense Strategies Against Misinformation in Large Language Models 

**Authors**: Shuliang Liu, Hongyi Liu, Aiwei Liu, Bingchen Duan, Qi Zheng, Yibo Yan, He Geng, Peijie Jiang, Jia Liu, Xuming Hu  

**Link**: [PDF](https://arxiv.org/pdf/2507.05288)  

**Abstract**: The widespread deployment of large language models (LLMs) across critical domains has amplified the societal risks posed by algorithmically generated misinformation. Unlike traditional false content, LLM-generated misinformation can be self-reinforcing, highly plausible, and capable of rapid propagation across multiple languages, which traditional detection methods fail to mitigate effectively. This paper introduces a proactive defense paradigm, shifting from passive post hoc detection to anticipatory mitigation strategies. We propose a Three Pillars framework: (1) Knowledge Credibility, fortifying the integrity of training and deployed data; (2) Inference Reliability, embedding self-corrective mechanisms during reasoning; and (3) Input Robustness, enhancing the resilience of model interfaces against adversarial attacks. Through a comprehensive survey of existing techniques and a comparative meta-analysis, we demonstrate that proactive defense strategies offer up to 63\% improvement over conventional methods in misinformation prevention, despite non-trivial computational overhead and generalization challenges. We argue that future research should focus on co-designing robust knowledge foundations, reasoning certification, and attack-resistant interfaces to ensure LLMs can effectively counter misinformation across varied domains. 

---
# Chat2SPaT: A Large Language Model Based Tool for Automating Traffic Signal Control Plan Management 

**Authors**: Yue Wang, Miao Zhou, Guijing Huang, Rui Zhuo, Chao Yi, Zhenliang Ma  

**Link**: [PDF](https://arxiv.org/pdf/2507.05283)  

**Abstract**: Pre-timed traffic signal control, commonly used for operating signalized intersections and coordinated arterials, requires tedious manual work for signaling plan creating and updating. When the time-of-day or day-of-week plans are utilized, one intersection is often associated with multiple plans, leading to further repetitive manual plan parameter inputting. To enable a user-friendly traffic signal control plan management process, this study proposes Chat2SPaT, a method to convert users' semi-structured and ambiguous descriptions on the signal control plan to exact signal phase and timing (SPaT) results, which could further be transformed into structured stage-based or ring-based plans to interact with intelligent transportation system (ITS) software and traffic signal controllers. With curated prompts, Chat2SPaT first leverages large language models' (LLMs) capability of understanding users' plan descriptions and reformulate the plan as a combination of phase sequence and phase attribute results in the json format. Based on LLM outputs, python scripts are designed to locate phases in a cycle, address nuances of traffic signal control, and finally assemble the complete traffic signal control plan. Within a chat, the pipeline can be utilized iteratively to conduct further plan editing. Experiments show that Chat2SPaT can generate plans with an accuracy of over 94% for both English and Chinese cases, using a test dataset with over 300 plan descriptions. As the first benchmark for evaluating LLMs' capability of understanding traffic signal control plan descriptions, Chat2SPaT provides an easy-to-use plan management pipeline for traffic practitioners and researchers, serving as a potential new building block for a more accurate and versatile application of LLMs in the field of ITS. The source codes, prompts and test dataset are openly accessible at this https URL. 

---
# CoreCodeBench: A Configurable Multi-Scenario Repository-Level Benchmark 

**Authors**: Lingyue Fu, Hao Guan, Bolun Zhang, Haowei Yuan, Yaoming Zhu, Jun Xu, Zongyu Wang, Lin Qiu, Xunliang Cai, Xuezhi Cao, Weiwen Liu, Weinan Zhang, Yong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2507.05281)  

**Abstract**: As Large Language Models (LLMs) demonstrate increasingly sophisticated code processing capabilities, evaluating their performance on engineering-level code remains challenging. Existing repository-level benchmarks primarily focus on single scenarios, such as code generation or bug fixing, without adequately capturing the diversity and complexity of real-world software or project engineering workflows. Furthermore, these benchmarks suffer from limited controllability in question positioning and reliability issues in their generated test cases. To address these limitations, we present CorePipe, a fully automated pipeline that converts repositories into comprehensive test cases, and introduce CoreCodeBench, a configurable multi-scenario repository-level benchmark. To simulate real engineering scenarios, CorePipe generates three types of atomic questions (Development, BugFix, and Test-Driven Development) specifically targeting core code segments. These atomic questions are further combined into three types of composite questions, with difficulty levels flexibly adjusted through hyperparameter tuning. CoreCodeBench provides a comprehensive and extensive repository-level benchmark to investigate the applicability of LLMs in real-world engineering projects. Experiments with 16 LLMs across diverse scenarios reveal varying capabilities and offer multi-dimensional insights into LLM performance in engineering contexts. The code for CorePipe is available at this https URL, and the data for CoreCodeBench can be accessed at this https URL. 

---
# ReservoirChat: Interactive Documentation Enhanced with LLM and Knowledge Graph for ReservoirPy 

**Authors**: Virgile Boraud, Yannis Bendi-Ouis, Paul Bernard, Xavier Hinaut  

**Link**: [PDF](https://arxiv.org/pdf/2507.05279)  

**Abstract**: We introduce a tool designed to improve the capabilities of Large Language Models (LLMs) in assisting with code development using the ReservoirPy library, as well as in answering complex questions in the field of Reservoir Computing. By incorporating external knowledge through Retrieval-Augmented Generation (RAG) and knowledge graphs, our approach aims to reduce hallucinations and increase the factual accuracy of generated responses. The system provides an interactive experience similar to ChatGPT, tailored specifically for ReservoirPy, enabling users to write, debug, and understand Python code while accessing reliable domain-specific insights. In our evaluation, while proprietary models such as ChatGPT-4o and NotebookLM performed slightly better on general knowledge questions, our model outperformed them on coding tasks and showed a significant improvement over its base model, Codestral-22B. 

---
