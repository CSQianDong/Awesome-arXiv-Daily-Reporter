# Neuron-Aware Data Selection In Instruction Tuning For Large Language Models 

**Authors**: Xin Chen, Junchao Wu, Shu Yang, Runzhe Zhan, Zeyu Wu, Min Yang, Shujian Huang, Lidia S. Chao, Derek F. Wong  

**Link**: [PDF](https://arxiv.org/pdf/2603.13201)  

**Abstract**: Instruction Tuning (IT) has been proven to be an effective approach to unlock the powerful capabilities of large language models (LLMs). Recent studies indicate that excessive IT data can degrade LLMs performance, while carefully selecting a small subset of high-quality IT data can significantly enhance their capabilities. Therefore, identifying the most efficient subset data from the IT dataset to effectively develop either specific or general abilities in LLMs has become a critical challenge. To address this, we propose a novel and efficient framework called NAIT. NAIT evaluates the impact of IT data on LLMs performance by analyzing the similarity of neuron activation patterns between the IT dataset and the target domain capability. Specifically, NAIT captures neuron activation patterns from in-domain datasets of target domain capabilities to construct reusable and transferable neuron activation features. It then evaluates and selects optimal samples based on the similarity between candidate samples and the expected activation features of the target capabilities. Experimental results show that training on the 10\% Alpaca-GPT4 IT data subset selected by NAIT consistently outperforms methods that rely on external advanced models or uncertainty-based features across various tasks. Our findings also reveal the transferability of neuron activation features across different capabilities of LLMs. In particular, IT data with more logical reasoning and programmatic features possesses strong general transferability, enabling models to develop stronger capabilities across multiple tasks, while a stable core subset of data is sufficient to consistently activate fundamental model capabilities and universally improve performance across diverse tasks. 

---
# ESG-Bench: Benchmarking Long-Context ESG Reports for Hallucination Mitigation 

**Authors**: Siqi Sun, Ben Peng Wu, Mali Jin, Peizhen Bai, Hanpei Zhang, Xingyi Song  

**Link**: [PDF](https://arxiv.org/pdf/2603.13154)  

**Abstract**: As corporate responsibility increasingly incorporates environmental, social, and governance (ESG) criteria, ESG reporting is becoming a legal requirement in many regions and a key channel for documenting sustainability practices and assessing firms' long-term and ethical performance. However, the length and complexity of ESG disclosures make them difficult to interpret and automate the analysis reliably. To support scalable and trustworthy analysis, this paper introduces ESG-Bench, a benchmark dataset for ESG report understanding and hallucination mitigation in large language models (LLMs). ESG-Bench contains human-annotated question-answer (QA) pairs grounded in real-world ESG report contexts, with fine-grained labels indicating whether model outputs are factually supported or hallucinated. Framing ESG report analysis as a QA task with verifiability constraints enables systematic evaluation of LLMs' ability to extract and reason over ESG content and provides a new use case: mitigating hallucinations in socially sensitive, compliance-critical settings. We design task-specific Chain-of-Thought (CoT) prompting strategies and fine-tune multiple state-of-the-art LLMs on ESG-Bench using CoT-annotated rationales. Our experiments show that these CoT-based methods substantially outperform standard prompting and direct fine-tuning in reducing hallucinations, and that the gains transfer to existing QA benchmarks beyond the ESG domain. 

---
# Mending the Holes: Mitigating Reward Hacking in Reinforcement Learning for Multilingual Translation 

**Authors**: Yifeng Liu, Siqi Ouyang, Yatish Hosmane Revanasiddappa, Lei Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.13045)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capability in machine translation on high-resource language pairs, yet their performance on low-resource translation still lags behind. Existing post-training methods rely heavily on high-quality parallel data, which are often scarce or unavailable for low-resource languages. In this paper, we introduce WALAR, a reinforcement training method using only monolingual text to elevate LLMs' translation capabilities on massive low-resource languages while retaining their performance on high-resource languages. Our key insight is based on the observation of failure modes (or "holes") in existing source-based multilingual quality estimation (QE) models. Reinforcement learning (RL) using these QE models tends to amplify such holes, resulting in poorer multilingual LLMs. We develop techniques including word alignment and language alignment to mitigate such holes in WALAR's reward for RL training. We continually trained an LLM supporting translation of 101 languages using WALAR. The experiments show that our new model outperforms LLaMAX, one of the strongest open-source multilingual LLMs by a large margin on 1400 language directions on Flores-101 dataset. 

---
# Interpretable Semantic Gradients in SSD: A PCA Sweep Approach and a Case Study on AI Discourse 

**Authors**: Hubert Plisiecki, Maria Leniarska, Jan Piotrowski, Marcin Zajenkowski  

**Link**: [PDF](https://arxiv.org/pdf/2603.13038)  

**Abstract**: Supervised Semantic Differential (SSD) is a mixed quantitative-interpretive method that models how text meaning varies with continuous individual-difference variables by estimating a semantic gradient in an embedding space and interpreting its poles through clustering and text retrieval. SSD applies PCA before regression, but currently no systematic method exists for choosing the number of retained components, introducing avoidable researcher degrees of freedom in the analysis pipeline. We propose a PCA sweep procedure that treats dimensionality selection as a joint criterion over representation capacity, gradient interpretability, and stability across nearby values of K. We illustrate the method on a corpus of short posts about artificial intelligence written by Prolific participants who also completed Admiration and Rivalry narcissism scales. The sweep yields a stable, interpretable Admiration-related gradient contrasting optimistic, collaborative framings of AI with distrustful and derisive discourse, while no robust alignment emerges for Rivalry. We also show that a counterfactual using a high-PCA dimension solution heuristic produces diffuse, weakly structured clusters instead, reinforcing the value of the sweep-based choice of K. The case study shows how the PCA sweep constrains researcher degrees of freedom while preserving SSD's interpretive aims, supporting transparent and psychologically meaningful analyses of connotative meaning. 

---
# Is Human Annotation Necessary? Iterative MBR Distillation for Error Span Detection in Machine Translation 

**Authors**: Boxuan Lyu, Haiyue Song, Zhi Qu  

**Link**: [PDF](https://arxiv.org/pdf/2603.12983)  

**Abstract**: Error Span Detection (ESD) is a crucial subtask in Machine Translation (MT) evaluation, aiming to identify the location and severity of translation errors. While fine-tuning models on human-annotated data improves ESD performance, acquiring such data is expensive and prone to inconsistencies among annotators. To address this, we propose a novel self-evolution framework based on Minimum Bayes Risk (MBR) decoding, named Iterative MBR Distillation for ESD, which eliminates the reliance on human annotations by leveraging an off-the-shelf LLM to generate this http URL experiments on the WMT Metrics Shared Task datasets demonstrate that models trained solely on these self-generated pseudo-labels outperform both unadapted base model and supervised baselines trained on human annotations at the system and span levels, while maintaining competitive sentence-level performance. 

---
# Long-form RewardBench: Evaluating Reward Models for Long-form Generation 

**Authors**: Hui Huang, Yancheng He, Wei Liu, Muyun Yang, Jiaheng Liu, Kehai Chen, Bing Xu, Conghui Zhu, Hailong Cao, Tiejun Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2603.12963)  

**Abstract**: The widespread adoption of reinforcement learning-based alignment highlights the growing importance of reward models. Various benchmarks have been built to evaluate reward models in various domains and scenarios. However, a significant gap remains in assessing reward models for long-form generation, despite its critical role in real-world applications. To bridge this, we introduce Long-form RewardBench, the first reward modeling testbed specifically designed for long-form generation. Our benchmark encompasses five key subtasks: QA, RAG, Chat, Writing, and Reasoning. We collected instruction and preference data through a meticulously designed multi-stage data collection process, and conducted extensive experiments on 20+ mainstream reward models, including both classifiers and generative models. Our findings reveal that current models still lack long-form reward modeling capabilities. Furthermore, we designed a novel Long-form Needle-in-a-Haystack Test, which revealed a correlation between reward modeling performance and the error's position within a response, as well as the overall response length, with distinct characteristics observed between classification and generative models. Finally, we demonstrate that classifiers exhibit better generalizability compared to generative models trained on the same data. As the first benchmark for long-form reward modeling, this work aims to offer a robust platform for visualizing progress in this crucial area. 

---
# DS$^2$-Instruct: Domain-Specific Data Synthesis for Large Language Models Instruction Tuning 

**Authors**: Ruiyao Xu, Noelle I. Samia, Han Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.12932)  

**Abstract**: Adapting Large Language Models (LLMs) to specialized domains requires high-quality instruction tuning datasets, which are expensive to create through human annotation. Existing data synthesis methods focus on general-purpose tasks and fail to capture domain-specific terminology and reasoning patterns. To address this, we introduce DS$^2$-Instruct, a zero-shot framework that generates domain-specific instruction datasets without human supervision. Our approach first generates task-informed keywords to ensure comprehensive domain coverage. It then creates diverse instructions by pairing these keywords with different cognitive levels from Bloom's Taxonomy. Finally, it uses self-consistency validation to ensure data quality. We apply this framework to generate datasets across seven challenging domains, such as mathematics, finance, and logical reasoning. Comprehensive evaluation demonstrates that models fine-tuned on our generated data achieve substantial improvements over existing data generation methods. 

---
# HMS-BERT: Hybrid Multi-Task Self-Training for Multilingual and Multi-Label Cyberbullying Detection 

**Authors**: Zixin Feng, Xinying Cui, Yifan Sun, Zheng Wei, Jiachen Yuan, Jiazhen Hu, Ning Xin, Md Maruf Hasan  

**Link**: [PDF](https://arxiv.org/pdf/2603.12920)  

**Abstract**: Cyberbullying on social media is inherently multilingual and multi-faceted, where abusive behaviors often overlap across multiple categories. Existing methods are commonly limited by monolingual assumptions or single-task formulations, which restrict their effectiveness in realistic multilingual and multi-label scenarios. In this paper, we propose HMS-BERT, a hybrid multi-task self-training framework for multilingual and multi-label cyberbullying detection. Built upon a pretrained multilingual BERT backbone, HMS-BERT integrates contextual representations with handcrafted linguistic features and jointly optimizes a fine-grained multi-label abuse classification task and a three-class main classification task. To address labeled data scarcity in low-resource languages, an iterative self-training strategy with confidence-based pseudo-labeling is introduced to facilitate cross-lingual knowledge transfer. Experiments on four public datasets demonstrate that HMS-BERT achieves strong performance, attaining a macro F1-score of up to 0.9847 on the multi-label task and an accuracy of 0.6775 on the main classification task. Ablation studies further verify the effectiveness of the proposed components. 

---
# Learning from Child-Directed Speech in Two-Language Scenarios: A French-English Case Study 

**Authors**: Liel Binyamin, Elior Sulem  

**Link**: [PDF](https://arxiv.org/pdf/2603.12906)  

**Abstract**: Research on developmentally plausible language models has largely focused on English, leaving open questions about multilingual settings. We present a systematic study of compact language models by extending BabyBERTa to English-French scenarios under strictly size-matched data conditions, covering monolingual, bilingual, and cross-lingual settings. Our design contrasts two types of training corpora: (i) child-directed speech (about 2.5M tokens), following BabyBERTa and related work, and (ii) multi-domain corpora (about 10M tokens), extending the BabyLM framework to French. To enable fair evaluation, we also introduce new resources, including French versions of QAMR and QASRL, as well as English and French multi-domain corpora.
We evaluate the models on both syntactic and semantic tasks and compare them with models trained on Wikipedia-only data. The results reveal context-dependent effects: training on Wikipedia consistently benefits semantic tasks, whereas child-directed speech improves grammatical judgments in monolingual settings. Bilingual pretraining yields notable gains for textual entailment, with particularly strong improvements for French. Importantly, similar patterns emerge across BabyBERTa, RoBERTa, and LTG-BERT, suggesting consistent trends across architectures. 

---
# CLARIN-PT-LDB: An Open LLM Leaderboard for Portuguese to assess Language, Culture and Civility 

**Authors**: João Silva, Luís Gomes, António Branco  

**Link**: [PDF](https://arxiv.org/pdf/2603.12872)  

**Abstract**: This paper reports on the development of a leaderboard of Open Large Language Models (LLM) for European Portuguese (PT-PT), and on its associated benchmarks. This leaderboard comes as a way to address a gap in the evaluation of LLM for European Portuguese, which so far had no leaderboard dedicated to this variant of the language. The paper also reports on novel benchmarks, including some that address aspects of performance that so far have not been available in benchmarks for European Portuguese, namely model safeguards and alignment to Portuguese culture. The leaderboard is available at this https URL. 

---
# Rethinking Multiple-Choice Questions for RLVR: Unlocking Potential via Distractor Design 

**Authors**: Xu Guo, Qiming Ge, Jian Tong, Kedi Chen, Jin Zhang, Xiaogui Yang, Xuan Gao, Haijun Lv, Zhihui Lu, Yicheng Zou, Qipeng Guo  

**Link**: [PDF](https://arxiv.org/pdf/2603.12826)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) significantly enhances the reasoning capabilities of Large Language Models. When applied to RLVR, Multiple-Choice Questions (MCQs) offer a scalable source of verifiable data but risk inducing reward hacking, where models shortcut reasoning via random guessing or simple elimination. Current approaches often mitigate this by converting MCQs to open-ended formats, thereby discarding the contrastive signal provided by expert-designed distractors. In this work, we systematically investigate the impact of option design on RLVR. Our analysis highlights two primary insights: (1) Mismatches in option counts between training and testing degrade performance. (2) Strong distractors effectively mitigate random guessing, enabling effective RLVR training even with 2-way questions. Motivated by these findings, we propose Iterative Distractor Curation (IDC), a framework that actively constructs high-quality distractors to block elimination shortcuts and promote deep reasoning. Experiments on various benchmarks demonstrate that our method effectively enhances distractor quality and yields significant gains in RLVR training compared to the original data. 

---
# Adaptive Vision-Language Model Routing for Computer Use Agents 

**Authors**: Xunzhuo Liu, Bowei He, Xue Liu, Andy Luo, Haichen Zhang, Huamin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2603.12823)  

**Abstract**: Computer Use Agents (CUAs) translate natural-language instructions into Graphical User Interface (GUI) actions such as clicks, keystrokes, and scrolls by relying on a Vision-Language Model (VLM) to interpret screenshots and predict grounded tool calls. However, grounding accuracy varies dramatically across VLMs, while current CUA systems typically route every action to a single fixed model regardless of difficulty. We propose \textbf{Adaptive VLM Routing} (AVR), a framework that inserts a lightweight semantic routing layer between the CUA orchestrator and a pool of VLMs. For each tool call, AVR estimates action difficulty from multimodal embeddings, probes a small VLM to measure confidence, and routes the action to the cheapest model whose predicted accuracy satisfies a target reliability threshold. For \textit{warm} agents with memory of prior UI interactions, retrieved context further narrows the capability gap between small and large models, allowing many actions to be handled without escalation. We formalize routing as a cost--accuracy trade-off, derive a threshold-based policy for model selection, and evaluate AVR using ScreenSpot-Pro grounding data together with the OpenClaw agent routing benchmark. Across these settings, AVR projects inference cost reductions of up to 78\% while staying within 2 percentage points of an all-large-model baseline. When combined with the Visual Confused Deputy guardrail, AVR also escalates high-risk actions directly to the strongest available model, unifying efficiency and safety within a single routing framework. Materials are also provided Model, benchmark, and code: this https URL. 

---
# SteerRM: Debiasing Reward Models via Sparse Autoencoders 

**Authors**: Mengyuan Sun, Zhuohao Yu, Weizheng Gu, Shikun Zhang, Wei Ye  

**Link**: [PDF](https://arxiv.org/pdf/2603.12795)  

**Abstract**: Reward models (RMs) are critical components of alignment pipelines, yet they exhibit biases toward superficial stylistic cues, preferring better-presented responses over semantically superior ones. Existing debiasing methods typically require retraining or architectural modifications, while direct activation suppression degrades performance due to representation entanglement. We propose SteerRM, the first training-free method for debiasing reward models using Sparse Autoencoder (SAE)-based interventions. SteerRM isolates stylistic effects using contrastive paired responses, identifies bias-related SAE features with a strength-stability criterion, and suppresses them at inference time. Across six reward models on RM-Bench, SteerRM improves Hard-split accuracy by 7.3 points on average while preserving overall performance. Results on a Gemma-based reward model and a controlled non-format bias further suggest generalization across RM architectures and bias types. We further find that format-related features are concentrated in shallow layers and transfer across models, revealing shared architecture-level bias encoding patterns. These results show that SAE-based interventions can mitigate reward-model biases without retraining, providing a practical and interpretable solution for alignment pipelines. 

---
# SectEval: Evaluating the Latent Sectarian Preferences of Large Language Models 

**Authors**: Aditya Maheshwari, Amit Gajkeshwar, Kaushal Sharma, Vivek Patel  

**Link**: [PDF](https://arxiv.org/pdf/2603.12768)  

**Abstract**: As Large Language Models (LLMs) becomes a popular source for religious knowledge, it is important to know if it treats different groups fairly. This study is the first to measure how LLMs handle the differences between the two main sects of Islam: Sunni and Shia. We present a test called SectEval, available in both English and Hindi, consisting of 88 questions, to check the bias-ness of 15 top LLM models, both proprietary and open-weights. Our results show a major inconsistency based on language. In English, many powerful models DeepSeek-v3 and GPT-4o often favored Shia answers. However, when asked the exact same questions in Hindi, these models switched to favoring Sunni answers. This means a user could get completely different religious advice just by changing languages. We also looked at how models react to location. Advanced models Claude-3.5 changed their answers to match the user's country-giving Shia answers to a user from Iran and Sunni answers to a user from Saudi Arabia. In contrast, smaller models (especially in Hindi) ignored the user's location and stuck to a Sunni viewpoint. These findings show that AI is not neutral; its religious ``truth'' changes depending on the language you speak and the country you claim to be from. The data set is available at this https URL 

---
# A Method for Learning Large-Scale Computational Construction Grammars from Semantically Annotated Corpora 

**Authors**: Paul Van Eecke, Katrien Beuls  

**Link**: [PDF](https://arxiv.org/pdf/2603.12754)  

**Abstract**: We present a method for learning large-scale, broad-coverage construction grammars from corpora of language use. Starting from utterances annotated with constituency structure and semantic frames, the method facilitates the learning of human-interpretable computational construction grammars that capture the intricate relationship between syntactic structures and the semantic relations they express. The resulting grammars consist of networks of tens of thousands of constructions formalised within the Fluid Construction Grammar framework. Not only do these grammars support the frame-semantic analysis of open-domain text, they also house a trove of information about the syntactico-semantic usage patterns present in the data they were learnt from. The method and learnt grammars contribute to the scaling of usage-based, constructionist approaches to language, as they corroborate the scalability of a number of fundamental construction grammar conjectures while also providing a practical instrument for the constructionist study of English argument structure in broad-coverage corpora. 

---
# EvolveCoder: Evolving Test Cases via Adversarial Verification for Code Reinforcement Learning 

**Authors**: Chi Ruan, Dongfu Jiang, Huaye Zeng, Ping Nie, Wenhu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2603.12698)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) is a promising approach for improving code generation in large language models, but its effectiveness is limited by weak and static verification signals in existing coding RL datasets. In this paper, we propose a solution-conditioned and adversarial verification framework that iteratively refines test cases based on the execution behaviors of candidate solutions, with the goal of increasing difficulty, improving discriminative power, and reducing redundancy. Based on this framework, we introduce EvolveCoder-22k, a large-scale coding reinforcement learning dataset constructed through multiple rounds of adversarial test case evolution. Empirical analysis shows that iterative refinement substantially strengthens verification, with pass@1 decreasing from 43.80 to 31.22. Reinforcement learning on EvolveCoder-22k yields stable optimization and consistent performance gains, improving Qwen3-4B by an average of 4.2 points across four downstream benchmarks and outperforming strong 4B-scale baselines. Our results highlight the importance of adversarial, solution-conditioned verification for effective and scalable reinforcement learning in code generation. 

---
# Experimental evidence of progressive ChatGPT models self-convergence 

**Authors**: Konstantinos F. Xylogiannopoulos, Petros Xanthopoulos, Panagiotis Karampelas, Georgios A. Bakamitsos  

**Link**: [PDF](https://arxiv.org/pdf/2603.12683)  

**Abstract**: Large Language Models (LLMs) that undergo recursive training on synthetically generated data are susceptible to model collapse, a phenomenon marked by the generation of meaningless output. Existing research has examined this issue from either theoretical or empirical perspectives, often focusing on a single model trained recursively on its own outputs. While prior studies have cautioned against the potential degradation of LLM output quality under such conditions, no longitudinal investigation has yet been conducted to assess this effect over time. In this study, we employ a text similarity metric to evaluate different ChatGPT models' capacity to generate diverse textual outputs. Our findings indicate a measurable decline of recent ChatGPT releases' ability to produce varied text, even when explicitly prompted to do so, by setting the temperature parameter to one. The observed reduction in output diversity may be attributed to the influence of the amounts of synthetic data incorporated within their training datasets as the result of internet infiltration by LLM generated data. The phenomenon is defined as model self-convergence because of the gradual increase of similarities of produced texts among different ChatGPT versions. 

---
# MetaKE: Meta-learning Aligned Knowledge Editing via Bi-level Optimization 

**Authors**: Shuxin Liu, Ou Wu  

**Link**: [PDF](https://arxiv.org/pdf/2603.12677)  

**Abstract**: Knowledge editing (KE) aims to precisely rectify specific knowledge in Large Language Models (LLMs) without disrupting general capabilities. State-of-the-art methods suffer from an open-loop control mismatch. We identify a critical "Semantic-Execution Disconnect": the semantic target is derived independently without feedback from the downstream's feasible region. This misalignment often causes valid semantic targets to fall within the prohibited space, resulting in gradient truncation and editing failure. To bridge this gap, we propose MetaKE (Meta-learning Aligned Knowledge Editing), a new framework that reframes KE as a bi-level optimization problem. Departing from static calculation, MetaKE treats the edit target as a learnable meta-parameter: the upper-level optimizer seeks a feasible target to maximize post-edit performance, while the lower-level solver executes the editing. To address the challenge of differentiating through complex solvers, we derive a Structural Gradient Proxy, which explicitly backpropagates editability constraints to the target learning phase. Theoretical analysis demonstrates that MetaKE automatically aligns the edit direction with the model's feasible manifold. Extensive experiments confirm that MetaKE significantly outperforms strong baselines, offering a new perspective on knowledge editing. 

---
# From Text to Forecasts: Bridging Modality Gap with Temporal Evolution Semantic Space 

**Authors**: Lehui Li, Yuyao Wang, Jisheng Yan, Wei Zhang, Jinliang Deng, Haoliang Sun, Zhongyi Han, Yongshun Gong  

**Link**: [PDF](https://arxiv.org/pdf/2603.12664)  

**Abstract**: Incorporating textual information into time-series forecasting holds promise for addressing event-driven non-stationarity; however, a fundamental modality gap hinders effective fusion: textual descriptions express temporal impacts implicitly and qualitatively, whereas forecasting models rely on explicit and quantitative signals. Through controlled semi-synthetic experiments, we show that existing methods over-attend to redundant tokens and struggle to reliably translate textual semantics into usable numerical cues. To bridge this gap, we propose TESS, which introduces a Temporal Evolution Semantic Space as an intermediate bottleneck between modalities. This space consists of interpretable, numerically grounded temporal primitives (mean shift, volatility, shape, and lag) extracted from text by an LLM via structured prompting and filtered through confidence-aware gating. Experiments on four real-world datasets demonstrate up to a 29 percent reduction in forecasting error compared to state-of-the-art unimodal and multimodal baselines. The code will be released after acceptance. 

---
# Continual Learning in Large Language Models: Methods, Challenges, and Opportunities 

**Authors**: Hongyang Chen, Zhongwu Sun, Hongfei Ye, Kunchi Li, Xuemin Lin  

**Link**: [PDF](https://arxiv.org/pdf/2603.12658)  

**Abstract**: Continual learning (CL) has emerged as a pivotal paradigm to enable large language models (LLMs) to dynamically adapt to evolving knowledge and sequential tasks while mitigating catastrophic forgetting-a critical limitation of the static pre-training paradigm inherent to modern LLMs. This survey presents a comprehensive overview of CL methodologies tailored for LLMs, structured around three core training stages: continual pre-training, continual fine-tuning, and continual this http URL the canonical taxonomy of rehearsal-, regularization-, and architecture-based methods, we further subdivide each category by its distinct forgetting mitigation mechanisms and conduct a rigorous comparative analysis of the adaptability and critical improvements of traditional CL methods for LLMs. In doing so, we explicitly highlight core distinctions between LLM CL and traditional machine learning, particularly with respect to scale, parameter efficiency, and emergent capabilities. Our analysis covers essential evaluation metrics, including forgetting rates and knowledge transfer efficiency, along with emerging benchmarks for assessing CL performance. This survey reveals that while current methods demonstrate promising results in specific domains, fundamental challenges persist in achieving seamless knowledge integration across diverse tasks and temporal scales. This systematic review contributes to the growing body of knowledge on LLM adaptation, providing researchers and practitioners with a structured framework for understanding current achievements and future opportunities in lifelong learning for language models. 

---
# 98$\times$ Faster LLM Routing Without a Dedicated GPU: Flash Attention, Prompt Compression, and Near-Streaming for the vLLM Semantic Router 

**Authors**: Xunzhuo Liu, Bowei He, Xue Liu, Andy Luo, Haichen Zhang, Huamin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2603.12646)  

**Abstract**: System-level routers that intercept LLM requests for safety classification, domain routing, and PII detection must be both fast and operationally lightweight: they should add minimal latency to every request, yet not require a dedicated GPU -- an expensive resource better used for LLM inference itself. When the router co-locates on the same GPU as vLLM serving instances, standard attention's $O(n^2)$ memory makes long-context classification (8K--32K tokens) impossible: at 8K tokens, three concurrent classifiers need ${\sim}$4.5\,GB for attention masks alone, far exceeding the memory left by vLLM. We present three staged optimizations for the vLLM Semantic Router, benchmarked on AMD Instinct MI300X, that solve both the latency and the memory problem. \emph{Stage~1}: a custom CK Flash Attention operator for ONNX Runtime on ROCm reduces attention memory from $O(n^2)$ to $O(n)$ and end-to-end (E2E) latency from 4{,}918\,ms to 127\,ms (\textbf{38.7$\times$}), enabling 8K--32K tokens where SDPA OOMs. \emph{Stage~2}: classical NLP prompt compression (TextRank, position weighting, TF-IDF, and novelty scoring) reduces all inputs to ${\sim}$512 tokens without neural inference, capping both latency and GPU memory at a constant regardless of original prompt length (E2E 127$\to$62\,ms, \textbf{2.0$\times$}). \emph{Stage~3}: near-streaming body processing with adaptive chunking and zero-copy JSON eliminates serialization overhead (E2E 62$\to$50\,ms, \textbf{1.2$\times$}). Cumulatively: \textbf{98$\times$} improvement (4{,}918\,ms to 50\,ms), 16K-token routing in 108\,ms, and a total router GPU footprint under 800\,MB -- small enough to share a GPU with LLM serving and removing the need for a dedicated accelerator. Stage~1 targets AMD ROCm (NVIDIA GPUs already have FlashAttention via cuDNN); Stages~2 and~3 are hardware-agnostic. 

---
# Using a Human-AI Teaming Approach to Create and Curate Scientific Datasets with the SCILIRE System 

**Authors**: Necva Bölücü, Jessica Irons, Changhyun Lee, Brian Jin, Maciej Rybinski, Huichen Yang, Andreas Duenser, Stephen Wan  

**Link**: [PDF](https://arxiv.org/pdf/2603.12638)  

**Abstract**: The rapid growth of scientific literature has made manual extraction of structured knowledge increasingly impractical. To address this challenge, we introduce SCILIRE, a system for creating datasets from scientific literature. SCILIRE has been designed around Human-AI teaming principles centred on workflows for verifying and curating data. It facilitates an iterative workflow in which researchers can review and correct AI outputs. Furthermore, this interaction is used as a feedback signal to improve future LLM-based inference. We evaluate our design using a combination of intrinsic benchmarking outcomes together with real-world case studies across multiple domains. The results demonstrate that SCILIRE improves extraction fidelity and facilitates efficient dataset creation. 

---
# RTD-Guard: A Black-Box Textual Adversarial Detection Framework via Replacement Token Detection 

**Authors**: He Zhu, Yanshu Li, Wen Liu, Haitian Yang  

**Link**: [PDF](https://arxiv.org/pdf/2603.12582)  

**Abstract**: Textual adversarial attacks pose a serious security threat to Natural Language Processing (NLP) systems by introducing imperceptible perturbations that mislead deep learning models. While adversarial example detection offers a lightweight alternative to robust training, existing methods typically rely on prior knowledge of attacks, white-box access to the victim model, or numerous queries, which severely limits their practical deployment. This paper introduces RTD-Guard, a novel black-box framework for detecting textual adversarial examples. Our key insight is that word-substitution perturbations in adversarial attacks closely resemble the "replaced tokens" that a Replaced Token Detection (RTD) discriminator is pre-trained to identify. Leveraging this, RTD-Guard employs an off-the-shelf RTD discriminator-without fine-tuning-to localize suspicious tokens, masks them, and detects adversarial examples by observing the prediction confidence shift of the victim model before and after intervention. The entire process requires no adversarial data, model tuning, or internal model access, and uses only two black-box queries. Comprehensive experiments on multiple benchmark datasets demonstrate that RTD-Guard effectively detects adversarial texts generated by diverse state-of-the-art attack methods. It surpasses existing detection baselines across multiple metrics, offering a highly efficient, practical, and resource-light defense mechanism-particularly suited for real-world deployment in resource-constrained or privacy-sensitive environments. 

---
# Expert Pyramid Tuning: Efficient Parameter Fine-Tuning for Expertise-Driven Task Allocation 

**Authors**: Jia-Chen Zhang, Zhen-Wei Yan, Yu-Jie Xiong, Chun-Ming Xia  

**Link**: [PDF](https://arxiv.org/pdf/2603.12577)  

**Abstract**: Parameter-Efficient Fine-Tuning (PEFT) has become a dominant paradigm for deploying LLMs in multi-task scenarios due to its extreme parameter efficiency. While Mixture-of-Experts (MoE) based LoRA variants have achieved promising results by dynamically routing tokens to different low-rank experts, they largely overlook the hierarchical nature of task complexity. Existing methods typically employ experts with uniform architectures, limiting their ability to capture diverse feature granularities required by distinct tasks--where some tasks demand high-level semantic abstraction while others require fine-grained syntactic manipulation. To bridge this gap, we propose Expert Pyramid Tuning (EPT), a novel architecture that integrates the multi-scale feature pyramid concept from computer vision into the realm of PEFT. Unlike standard LoRA, EPT decomposes task adaptation into two stages: (1) A shared meta-knowledge Subspace that encodes universal linguistic patterns in low dimensions; (2) A Pyramid Projection Mechanism that utilizes learnable up-projection operators to reconstruct high-dimensional features at varying scales. A task-aware router then dynamically selects the optimal combination of these multi-scale features. Extensive experiments across multiple multi-task benchmarks demonstrate that EPT significantly outperforms SOTA MoE-LoRA variants. Crucially, thanks to the re-parameterization capability of our design, EPT achieves this performance improvement while simultaneously reducing the number of training parameters. 

---
# LMEB: Long-horizon Memory Embedding Benchmark 

**Authors**: Xinping Zhao, Xinshuo Hu, Jiaxin Xu, Danyu Tang, Xin Zhang, Mengjia Zhou, Yan Zhong, Yao Zhou, Zifei Shan, Meishan Zhang, Baotian Hu, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.12572)  

**Abstract**: Memory embeddings are crucial for memory-augmented systems, such as OpenClaw, but their evaluation is underexplored in current text embedding benchmarks, which narrowly focus on traditional passage retrieval and fail to assess models' ability to handle long-horizon memory retrieval tasks involving fragmented, context-dependent, and temporally distant information. To address this, we introduce the Long-horizon Memory Embedding Benchmark (LMEB), a comprehensive framework that evaluates embedding models' capabilities in handling complex, long-horizon memory retrieval tasks. LMEB spans 22 datasets and 193 zero-shot retrieval tasks across 4 memory types: episodic, dialogue, semantic, and procedural, with both AI-generated and human-annotated data. These memory types differ in terms of level of abstraction and temporal dependency, capturing distinct aspects of memory retrieval that reflect the diverse challenges of the real world. We evaluate 15 widely used embedding models, ranging from hundreds of millions to ten billion parameters. The results reveal that (1) LMEB provides a reasonable level of difficulty; (2) Larger models do not always perform better; (3) LMEB and MTEB exhibit orthogonality. This suggests that the field has yet to converge on a universal model capable of excelling across all memory retrieval tasks, and that performance in traditional passage retrieval may not generalize to long-horizon memory retrieval. In summary, by providing a standardized and reproducible evaluation framework, LMEB fills a crucial gap in memory embedding evaluation, driving further advancements in text embedding for handling long-term, context-dependent memory retrieval. LMEB is available at this https URL. 

---
# AgentDrift: Unsafe Recommendation Drift Under Tool Corruption Hidden by Ranking Metrics in LLM Agents 

**Authors**: Zekun Wu, Adriano Koshiyama, Sahan Bulathwela, Maria Perez-Ortiz  

**Link**: [PDF](https://arxiv.org/pdf/2603.12564)  

**Abstract**: Tool-augmented LLM agents increasingly serve as multi-turn advisors in high-stakes domains, yet their evaluation relies on ranking-quality metrics that measure what is recommended but not whether it is safe for the user. We introduce a paired-trajectory protocol that replays real financial dialogues under clean and contaminated tool-output conditions across seven LLMs (7B to frontier) and decomposes divergence into information-channel and memory-channel mechanisms. Across the seven models tested, we consistently observe the evaluation-blindness pattern: recommendation quality is largely preserved under contamination (utility preservation ratio approximately 1.0) while risk-inappropriate products appear in 65-93% of turns, a systematic safety failure poorly reflected by standard NDCG. Safety violations are predominantly information-channel-driven, emerge at the first contaminated turn, and persist without self-correction over 23-step trajectories; no agent across 1,563 contaminated turns explicitly questions tool-data reliability. Even narrative-only corruption (biased headlines, no numerical manipulation) induces significant drift while completely evading consistency monitors. A safety-penalized NDCG variant (sNDCG) reduces preservation ratios to 0.51-0.74, indicating that much of the evaluation gap becomes visible once safety is explicitly measured. These results motivate considering trajectory-level safety monitoring, beyond single-turn quality, for deployed multi-turn agents in high-stakes settings. 

---
# LLM BiasScope: A Real-Time Bias Analysis Platform for Comparative LLM Evaluation 

**Authors**: Himel Ghosh, Nick Elias Werner  

**Link**: [PDF](https://arxiv.org/pdf/2603.12522)  

**Abstract**: As large language models (LLMs) are deployed widely, detecting and understanding bias in their outputs is critical. We present LLM BiasScope, a web application for side-by-side comparison of LLM outputs with real-time bias analysis. The system supports multiple providers (Google Gemini, DeepSeek, MiniMax, Mistral, Meituan, Meta Llama) and enables researchers and practitioners to compare models on the same prompts while analyzing bias patterns. LLM BiasScope uses a two-stage bias detection pipeline: sentence-level bias detection followed by bias type classification for biased sentences. The analysis runs automatically on both user prompts and model responses, providing statistics, visualizations, and detailed breakdowns of bias types. The interface displays two models side-by-side with synchronized streaming responses, per-model bias summaries, and a comparison view highlighting differences in bias distributions. The system is built on this http URL with React, integrates Hugging Face inference endpoints for bias detection, and uses the Vercel AI SDK for multi-provider LLM access. Features include real-time streaming, export to JSON/PDF, and interactive visualizations (bar charts, radar charts) for bias analysis. LLM BiasScope is available as an open-source web application, providing a practical tool for bias evaluation and comparative analysis of LLM behaviour. 

---
# Marked Pedagogies: Examining Linguistic Biases in Personalized Automated Writing Feedback 

**Authors**: Mei Tan, Lena Phalen, Dorottya Demszky  

**Link**: [PDF](https://arxiv.org/pdf/2603.12471)  

**Abstract**: Effective personalized feedback is critical to students' literacy development. Though LLM-powered tools now promise to automate such feedback at scale, LLMs are not language-neutral: they privilege standard academic English and reproduce social stereotypes, raising concerns about how "personalization" shapes the feedback students receive. We examine how four widely used LLMs (GPT-4o, GPT-3.5-turbo, Llama-3.3 70B, Llama-3.1 8B) adapt written feedback in response to student attributes. Using 600 eighth-grade persuasive essays from the PERSUADE dataset, we generated feedback under prompt conditions embedding gender, race/ethnicity, learning needs, achievement, and motivation. We analyze lexical shifts across model outputs by adapting the Marked Words framework. Our results reveal systematic, stereotype-aligned shifts in feedback conditioned on presumed student attributes--even when essay content was identical. Feedback for students marked by race, language, or disability often exhibited positive feedback bias and feedback withholding bias--overuse of praise, less substantive critique, and assumptions of limited ability. Across attributes, models tailored not only what content was emphasized but also how writing was judged and how students were addressed. We term these instructional orientations Marked Pedagogies and highlight the need for transparency and accountability in automated feedback tools. 

---
# Shattering the Shortcut: A Topology-Regularized Benchmark for Multi-hop Medical Reasoning in LLMs 

**Authors**: Xing Zi, Xinying Zhou, Jinghao Xiao, Catarina Moreira, Mukesh Prasad  

**Link**: [PDF](https://arxiv.org/pdf/2603.12458)  

**Abstract**: While Large Language Models (LLMs) achieve expert-level performance on standard medical benchmarks through single-hop factual recall, they severely struggle with the complex, multi-hop diagnostic reasoning required in real-world clinical settings. A primary obstacle is "shortcut learning", where models exploit highly connected, generic hub nodes (e.g., "inflammation") in knowledge graphs to bypass authentic micro-pathological cascades. To address this, we introduce ShatterMed-QA, a bilingual benchmark of 10,558 multi-hop clinical questions designed to rigorously evaluate deep diagnostic reasoning. Our framework constructs a topology-regularized medical Knowledge Graph using a novel $k$-Shattering algorithm, which physically prunes generic hubs to explicitly sever logical shortcuts. We synthesize the evaluation vignettes by applying implicit bridge entity masking and topology-driven hard negative sampling, forcing models to navigate biologically plausible distractors without relying on superficial elimination. Comprehensive evaluations of 21 LLMs reveal massive performance degradation on our multi-hop tasks, particularly among domain-specific models. Crucially, restoring the masked evidence via Retrieval-Augmented Generation (RAG) triggers near-universal performance recovery, validating ShatterMed-QA's structural fidelity and proving its efficacy in diagnosing the fundamental reasoning deficits of current medical AI. Explore the dataset, interactive examples, and full leaderboards at our project website: this https URL 

---
# CSE-UOI at SemEval-2026 Task 6: A Two-Stage Heterogeneous Ensemble with Deliberative Complexity Gating for Political Evasion Detection 

**Authors**: Christos Tzouvaras, Konstantinos Skianis, Athanasios Voulodimos  

**Link**: [PDF](https://arxiv.org/pdf/2603.12453)  

**Abstract**: This paper describes our system for SemEval-2026 Task 6, which classifies clarity of responses in political interviews into three categories: Clear Reply, Ambivalent, and Clear Non-Reply. We propose a heterogeneous dual large language model (LLM) ensemble via self-consistency (SC) and weighted voting, and a novel post-hoc correction mechanism, Deliberative Complexity Gating (DCG). This mechanism uses cross-model behavioral signals and exploits the finding that an LLM response-length proxy correlates strongly with sample ambiguity. To further examine mechanisms for improving ambiguity detection, we evaluated multi-agent debate as an alternative strategy for increasing deliberative capacity. Unlike DCG, which adaptively gates reasoning using cross-model behavioral signals, debate increases agent count without increasing model diversity. Our solution achieved a Macro-F1 score of 0.85 on the evaluation set, securing 3rd place. 

---
# Interpreting Negation in GPT-2: Layer- and Head-Level Causal Analysis 

**Authors**: Abdullah Al Mofael, Lisa M. Kuhn, Ghassan Alkadi, Kuo-Pao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2603.12423)  

**Abstract**: Negation remains a persistent challenge for modern language models, often causing reversed meanings or factual errors. In this work, we conduct a causal analysis of how GPT-2 Small internally processes such linguistic transformations. We examine its hidden representations at both the layer and head level. Our analysis is based on a self-curated 12,000-pair dataset of matched affirmative and negated sentences, covering multiple linguistic templates and forms of negation. To quantify this behavior, we define a metric, the Negation Effect Score (NES), which measures the model's sensitivity in distinguishing between affirmative statements and their negations. We carried out two key interventions to probe causal structure. In activation patching, internal activations from affirmative sentences were inserted into their negated counterparts to see how meaning shifted. In ablation, specific attention heads were temporarily disabled to observe how logical polarity changed. Together, these steps revealed how negation signals move and evolve through GPT-2's layers. Our findings indicate that this capability is not widespread; instead, it is highly concentrated within a limited number of mid-layer attention heads, primarily within layers 4 to 6. Ablating these specific components directly disrupts the model's negation sensitivity: on our in-domain, ablation increased NES (indicating weaker negation sensitivity), and re-introducing cached affirmative activations (rescue) increased NES further, confirming that these heads carry affirmative signal rather than restoring baseline behavior. On xNot360, ablation slightly decreased NES and rescue restored performance above baseline. This pattern demonstrates that these causal patterns are consistent across various negation forms and remain detectable on the external xNot360 benchmark, though with smaller magnitude. 

---
# Not Just the Destination, But the Journey: Reasoning Traces Causally Shape Generalization Behaviors 

**Authors**: Pengcheng Wen, Yanxu Zhu, Jiapeng Sun, Han Zhu, Yujin Zhou, Chi-Min Chan, Sirui Han, Yike Guo  

**Link**: [PDF](https://arxiv.org/pdf/2603.12397)  

**Abstract**: Chain-of-Thought (CoT) is often viewed as a window into LLM decision-making, yet recent work suggests it may function merely as post-hoc rationalization. This raises a critical alignment question: Does the reasoning trace causally shape model generalization independent of the final answer? To isolate reasoning's causal effect, we design a controlled experiment holding final harmful answers constant while varying reasoning paths. We construct datasets with \textit{Evil} reasoning embracing malice, \textit{Misleading} reasoning rationalizing harm, and \textit{Submissive} reasoning yielding to pressure. We train models (0.6B--14B parameters) under multiple paradigms, including question-thinking-answer (QTA), question-thinking (QT), and thinking-only (T-only), and evaluate them in both think and no-think modes. We find that: (1) CoT training could amplify harmful generalization more than standard fine-tuning; (2) distinct reasoning types induce distinct behavioral patterns aligned with their semantics, despite identical final answers; (3) training on reasoning without answer supervision (QT or T-only) is sufficient to alter behavior, proving reasoning carries an independent signal; and (4) these effects persist even when generating answers without reasoning, indicating deep internalization. Our findings demonstrate that reasoning content is causally potent, challenging alignment strategies that supervise only outputs. 

---
# TASTE-Streaming: Towards Streamable Text-Aligned Speech Tokenization and Embedding for Spoken Language Modeling 

**Authors**: Liang-Hsuan Tseng, Hung-yi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2603.12350)  

**Abstract**: Text-speech joint spoken language modeling (SLM) aims at natural and intelligent speech-based interactions, but developing such a system may suffer from modality mismatch: speech unit sequences are much longer than text tokens. Prior work reduces this gap with text-aligned tokenization and embedding (TASTE), producing speech tokens that align in lengths with their textual counterparts. However, the dependence on an external ASR system and the use of a non-causal decoder limits streaming use. To address this limitation, we propose TASTE-S, a streamable extension of TASTE suitable for real-time usage. TASTE-S integrates a CTC-based ASR module into the encoder for instant dual-modality encoding. We also redesign the unit decoder to enable on-the-fly decoding. With joint training, we show that TASTE-S matches TASTE's performance while significantly reducing latency. Further investigations reveal that TASTE-S remains robust to transcriptions and enables long-form encoding and decoding. 

---
# LLM-Augmented Therapy Normalization and Aspect-Based Sentiment Analysis for Treatment-Resistant Depression on Reddit 

**Authors**: Yuxin Zhu, Sahithi Lakamana, Masoud Rouhizadeh, Selen Bozkurt, Rachel Hershenberg, Abeed Sarker  

**Link**: [PDF](https://arxiv.org/pdf/2603.12343)  

**Abstract**: Treatment-resistant depression (TRD) is a severe form of major depressive disorder in which patients do not achieve remission despite multiple adequate treatment trials. Evidence across pharmacologic options for TRD remains limited, and trials often do not fully capture patient-reported tolerability. Large-scale online peer-support narratives therefore offer a complementary lens on how patients describe and evaluate medications in real-world use. In this study, we curated a corpus of 5,059 Reddit posts explicitly referencing TRD from 3,480 subscribers across 28 mental health-related subreddits from 2010 to 2025. Of these, 3,839 posts mentioned at least one medication, yielding 23,399 mentions of 81 generic-name medications after lexicon-based normalization of brand names, misspellings, and colloquialisms. We developed an aspect-based sentiment classifier by fine-tuning DeBERTa-v3 on the SMM4H 2023 therapy-sentiment Twitter corpus with large language model based data augmentation, achieving a micro-F1 score of 0.800 on the shared-task test set. Applying this classifier to Reddit, we quantified sentiment toward individual medications across three categories: positive, neutral, and negative, and tracked patterns by drug, subscriber, subreddit, and year. Overall, 72.1% of medication mentions were neutral, 14.8% negative, and 13.1% positive. Conventional antidepressants, especially SSRIs and SNRIs, showed consistently higher negative than positive proportions, whereas ketamine and esketamine showed comparatively more favorable sentiment profiles. These findings show that normalized medication extraction combined with aspect-based sentiment analysis can help characterize patient-perceived treatment experiences in TRD-related Reddit discourse, complementing clinical evidence with large-scale patient-generated perspectives. 

---
# Prompt Injection as Role Confusion 

**Authors**: Charles Ye, Jasmine Cui, Dylan Hadfield-Menell  

**Link**: [PDF](https://arxiv.org/pdf/2603.12277)  

**Abstract**: Language models remain vulnerable to prompt injection attacks despite extensive safety training. We trace this failure to role confusion: models infer roles from how text is written, not where it comes from. We design novel role probes to capture how models internally identify "who is speaking." These reveal why prompt injection works: untrusted text that imitates a role inherits that role's authority. We test this insight by injecting spoofed reasoning into user prompts and tool outputs, achieving average success rates of 60% on StrongREJECT and 61% on agent exfiltration, across multiple open- and closed-weight models with near-zero baselines. Strikingly, the degree of internal role confusion strongly predicts attack success before generation begins. Our findings reveal a fundamental gap: security is defined at the interface but authority is assigned in latent space. More broadly, we introduce a unifying, mechanistic framework for prompt injection, demonstrating that diverse prompt-injection attacks exploit the same underlying role-confusion mechanism. 

---
# GONE: Structural Knowledge Unlearning via Neighborhood-Expanded Distribution Shaping 

**Authors**: Chahana Dahal, Ashutosh Balasubramaniam, Zuobin Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2603.12275)  

**Abstract**: Unlearning knowledge is a pressing and challenging task in Large Language Models (LLMs) because of their unprecedented capability to memorize and digest training data at scale, raising more significant issues regarding safety, privacy, and intellectual property. However, existing works, including parameter editing, fine-tuning, and distillation-based methods, are all focused on flat sentence-level data but overlook the relational, multi-hop, and reasoned knowledge in naturally structured data. In response to this gap, this paper introduces Graph Oblivion and Node Erasure (GONE), a benchmark for evaluating knowledge unlearning over structured knowledge graph (KG) facts in LLMs. This KG-based benchmark enables the disentanglement of three effects of unlearning: direct fact removal, reasoning-based leakage, and catastrophic forgetting. In addition, Neighborhood-Expanded Distribution Shaping (NEDS), a novel unlearning framework, is designed to leverage graph connectivity and identify anchor correlated neighbors, enforcing a precise decision boundary between the forgotten fact and its semantic neighborhood. Evaluations on LLaMA-3-8B and Mistral-7B across multiple knowledge editing and unlearning methods showcase NEDS's superior performance (1.000 on unlearning efficacy and 0.839 on locality) on GONE and other benchmarks. Code is available at this https URL. 

---
# Aligning Language Models from User Interactions 

**Authors**: Thomas Kleine Buening, Jonas Hübotter, Barna Pásztor, Idan Shenfeld, Giorgia Ramponi, Andreas Krause  

**Link**: [PDF](https://arxiv.org/pdf/2603.12273)  

**Abstract**: Multi-turn user interactions are among the most abundant data produced by language models, yet we lack effective methods to learn from them. While typically discarded, these interactions often contain useful information: follow-up user messages may indicate that a response was incorrect, failed to follow an instruction, or did not align with the user's preferences. Importantly, language models are already able to make use of this information in context. After observing a user's follow-up, the same model is often able to revise its behavior. We leverage this ability to propose a principled and scalable method for learning directly from user interactions through self-distillation. By conditioning the model on the user's follow-up message and comparing the resulting token distribution with the original policy, we obtain a target for updating the policy that captures how the model's behavior changes in hindsight. We then distill this hindsight distribution back into the current policy. Remarkably, we show that training on real-world user conversations from WildChat improves language models across standard alignment and instruction-following benchmarks, without regressing other capabilities. The same mechanism enables personalization, allowing models to continually adapt to individual users through interaction without explicit feedback. Our results demonstrate that raw user interactions that arise naturally during deployment enable alignment, personalization, and continual adaptation. 

---
# ActTail: Global Activation Sparsity in Large Language Models 

**Authors**: Wenwen Hou, Xinyuan Song, Shiwei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.12272)  

**Abstract**: Activation sparsity is a promising approach for accelerating large language model (LLM) inference by reducing computation and memory movement. However, existing activation sparsity methods typically apply uniform sparsity across projections, ignoring the heterogeneous statistical properties of Transformer weights and thereby amplifying performance degradation. In this paper, we propose ActTail, a TopK magnitude-based activation sparsity method with global activation sparsity allocation grounded in Heavy-Tailed Self-Regularization (HT-SR) theory. Specifically, we capture this heterogeneity via the heavy-tail exponent computed from each projection's empirical spectral density (ESD), which is used as a quantitative indicator to assign projection-specific sparsity budgets. Importantly, we provide a theoretical analysis that establishes an explicit relationship between the activation sparsity ratio and the heavy-tail exponent under the HT-SR regime, offering principled guidance for sparsity allocation beyond heuristic design. Experiments on LLaMA and Mistral models show that our method improves both perplexity and downstream task performance at high sparsity compared to uniform allocation. At 80% sparsity, perplexity is reduced by 21.8% on LLaMA-2-7B, 40.1% on LLaMA-2-13B, and 9.4% on Mistral-7B. 

---
# Diagnosing Retrieval Bias Under Multiple In-Context Knowledge Updates in Large Language Models 

**Authors**: Boyu Qiao, Sean Guo, Xian Yang, Kun Li, Wei Zhou, Songlin Hu, Yunya Song  

**Link**: [PDF](https://arxiv.org/pdf/2603.12271)  

**Abstract**: LLMs are widely used in knowledge-intensive tasks where the same fact may be revised multiple times within context. Unlike prior work focusing on one-shot updates or single conflicts, multi-update scenarios contain multiple historically valid versions that compete at retrieval, yet remain underexplored. This challenge resembles the AB-AC interference paradigm in cognitive psychology: when the same cue A is successively associated with B and C, the old and new associations compete during retrieval, leading to bias. Inspired by this, we introduce a Dynamic Knowledge Instance (DKI) evaluation framework, modeling multi-updates of the same fact as a cue paired with a sequence of updated values, and assess models via endpoint probing of the earliest (initial) and latest (current) states. Across diverse LLMs, we observe that retrieval bias intensifies as updates increase, earliest-state accuracy stays high while latest-state accuracy drops substantially. Diagnostic analyses of attention, hidden-state similarity, and output logits further reveal that these signals become flatter and weakly discriminative on errors, providing little stable basis for identifying the latest update. Finally, cognitively inspired heuristic intervention strategies yield only modest gains and do not eliminate the bias. Our results reveal a persistent challenge in tracking and following knowledge updates in long contexts. 

---
# Task-Specific Knowledge Distillation via Intermediate Probes 

**Authors**: Ryan Brown, Chris Russell  

**Link**: [PDF](https://arxiv.org/pdf/2603.12270)  

**Abstract**: Knowledge distillation from large language models (LLMs) assumes that the teacher's output distribution is a high-quality training signal. On reasoning tasks, this assumption is frequently violated. A model's intermediate representations may encode the correct answer, yet this information is lost or distorted through the vocabulary projection, where prompt formatting and answer-token choices creates brittle, noisy outputs.
We introduce \method{}, a distillation framework that bypasses this bottleneck by training lightweight probes on frozen teacher hidden states and using the probe's predictions, rather than output logits, as supervision for student training. This simple change yields consistent improvements across four reasoning benchmarks (AQuA-RAT, ARC Easy/Challenge, and MMLU), with gains most pronounced under limited data.
Probes trained on intermediate representations provide cleaner labels than the teacher's own outputs, effectively denoising the distillation signal. \method{} requires no architectural changes to student or teacher, is architecture-agnostic, and adds minimal compute since probe training is cheap and teacher representations can be cached. By exploiting internal representations, \method{} enables practitioners to extract more value from large teacher models without additional training data or architectural complexity. 

---
# Semantic Invariance in Agentic AI 

**Authors**: I. de Zarzà, J. de Curtò, Jordi Cabot, Pietro Manzoni, Carlos T. Calafate  

**Link**: [PDF](https://arxiv.org/pdf/2603.13173)  

**Abstract**: Large Language Models (LLMs) increasingly serve as autonomous reasoning agents in decision support, scientific problem-solving, and multi-agent coordination systems. However, deploying LLM agents in consequential applications requires assurance that their reasoning remains stable under semantically equivalent input variations, a property we term semantic this http URL benchmark evaluations, which assess accuracy on fixed, canonical problem formulations, fail to capture this critical reliability dimension. To address this shortcoming, in this paper we present a metamorphic testing framework for systematically assessing the robustness of LLM reasoning agents, applying eight semantic-preserving transformations (identity, paraphrase, fact reordering, expansion, contraction, academic context, business context, and contrastive formulation) across seven foundation models spanning four distinct architectural families: Hermes (70B, 405B), Qwen3 (30B-A3B, 235B-A22B), DeepSeek-R1, and gpt-oss (20B, 120B). Our evaluation encompasses 19 multi-step reasoning problems across eight scientific domains. The results reveal that model scale does not predict robustness: the smaller Qwen3-30B-A3B achieves the highest stability (79.6% invariant responses, semantic similarity 0.91), while larger models exhibit greater fragility. 

---
# Developing and evaluating a chatbot to support maternal health care 

**Authors**: Smriti Jha, Vidhi Jain, Jianyu Xu, Grace Liu, Sowmya Ramesh, Jitender Nagpal, Gretchen Chapman, Benjamin Bellows, Siddhartha Goyal, Aarti Singh, Bryan Wilder  

**Link**: [PDF](https://arxiv.org/pdf/2603.13168)  

**Abstract**: The ability to provide trustworthy maternal health information using phone-based chatbots can have a significant impact, particularly in low-resource settings where users have low health literacy and limited access to care. However, deploying such systems is technically challenging: user queries are short, underspecified, and code-mixed across languages, answers require regional context-specific grounding, and partial or missing symptom context makes safe routing decisions difficult.
We present a chatbot for maternal health in India developed through a partnership between academic researchers, a health tech company, a public health nonprofit, and a hospital. The system combines (1) stage-aware triage, routing high-risk queries to expert templates, (2) hybrid retrieval over curated maternal/newborn guidelines, and (3) evidence-conditioned generation from an LLM. Our core contribution is an evaluation workflow for high-stakes deployment under limited expert supervision. Targeting both component-level and end-to-end testing, we introduce: (i) a labeled triage benchmark (N=150) achieving 86.7% emergency recall, explicitly reporting the missed-emergency vs. over-escalation trade-off; (ii) a synthetic multi-evidence retrieval benchmark (N=100) with chunk-level evidence labels; (iii) LLM-as-judge comparison on real queries (N=781) using clinician-codesigned criteria; and (iv) expert validation. Our findings show that trustworthy medical assistants in multilingual, noisy settings require defense-in-depth design paired with multi-method evaluation, rather than any single model and evaluation method choice. 

---
# daVinci-Env: Open SWE Environment Synthesis at Scale 

**Authors**: Dayuan Fu, Shenyu Wu, Yunze Wu, Zerui Peng, Yaxing Huang, Jie Sun, Ji Zeng, Mohan Jiang, Lin Zhang, Yukun Li, Jiarui Hu, Liming Liu, Jinlong Hou, Pengfei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.13023)  

**Abstract**: Training capable software engineering (SWE) agents demands large-scale, executable, and verifiable environments that provide dynamic feedback loops for iterative code editing, test execution, and solution refinement. However, existing open-source datasets remain limited in scale and repository diversity, while industrial solutions are opaque with unreleased infrastructure, creating a prohibitive barrier for most academic research groups. We present OpenSWE, the largest fully transparent framework for SWE agent training in Python, comprising 45,320 executable Docker environments spanning over 12.8k repositories, with all Dockerfiles, evaluation scripts, and infrastructure fully open-sourced for reproducibility. OpenSWE is built through a multi-agent synthesis pipeline deployed across a 64-node distributed cluster, automating repository exploration, Dockerfile construction, evaluation script generation, and iterative test analysis. Beyond scale, we propose a quality-centric filtering pipeline that characterizes the inherent difficulty of each environment, filtering out instances that are either unsolvable or insufficiently challenging and retaining only those that maximize learning efficiency. With $891K spent on environment construction and an additional $576K on trajectory sampling and difficulty-aware curation, the entire project represents a total investment of approximately $1.47 million, yielding about 13,000 curated trajectories from roughly 9,000 quality guaranteed environments. Extensive experiments validate OpenSWE's effectiveness: OpenSWE-32B and OpenSWE-72B achieve 62.4% and 66.0% on SWE-bench Verified, establishing SOTA among Qwen2.5 series. Moreover, SWE-focused training yields substantial out-of-domain improvements, including up to 12 points on mathematical reasoning and 5 points on science benchmarks, without degrading factual recall. 

---
# Structured Distillation for Personalized Agent Memory: 11x Token Reduction with Retrieval Preservation 

**Authors**: Sydney Lewis  

**Link**: [PDF](https://arxiv.org/pdf/2603.13017)  

**Abstract**: Long conversations with an AI agent create a simple problem for one user: the history is useful, but carrying it verbatim is expensive. We study personalized agent memory: one user's conversation history with an agent, distilled into a compact retrieval layer for later search. Each exchange is compressed into a compound object with four fields (exchange_core, specific_context, thematic room_assignments, and regex-extracted files_touched). The searchable distilled text averages 38 tokens per exchange. Applied to 4,182 conversations (14,340 exchanges) from 6 software engineering projects, the method reduces average exchange length from 371 to 38 tokens, yielding 11x compression. We evaluate whether personalized recall survives that compression using 201 recall-oriented queries, 107 configurations spanning 5 pure and 5 cross-layer search modes, and 5 LLM graders (214,519 consensus-graded query-result pairs). The best pure distilled configuration reaches 96% of the best verbatim MRR (0.717 vs 0.745). Results are mechanism-dependent. All 20 vector search configurations remain non-significant after Bonferroni correction, while all 20 BM25 configurations degrade significantly (effect sizes |d|=0.031-0.756). The best cross-layer setup slightly exceeds the best pure verbatim baseline (MRR 0.759). Structured distillation compresses single-user agent memory without uniformly sacrificing retrieval quality. At 1/11 the context cost, thousands of exchanges fit within a single prompt while the verbatim source remains available for drill-down. We release the implementation and analysis pipeline as open-source software. 

---
# MoKus: Leveraging Cross-Modal Knowledge Transfer for Knowledge-Aware Concept Customization 

**Authors**: Chenyang Zhu, Hongxiang Li, Xiu Li, Long Chen  

**Link**: [PDF](https://arxiv.org/pdf/2603.12743)  

**Abstract**: Concept customization typically binds rare tokens to a target concept. Unfortunately, these approaches often suffer from unstable performance as the pretraining data seldom contains these rare tokens. Meanwhile, these rare tokens fail to convey the inherent knowledge of the target concept. Consequently, we introduce Knowledge-aware Concept Customization, a novel task aiming at binding diverse textual knowledge to target visual concepts. This task requires the model to identify the knowledge within the text prompt to perform high-fidelity customized generation. Meanwhile, the model should efficiently bind all the textual knowledge to the target concept. Therefore, we propose MoKus, a novel framework for knowledge-aware concept customization. Our framework relies on a key observation: cross-modal knowledge transfer, where modifying knowledge within the text modality naturally transfers to the visual modality during generation. Inspired by this observation, MoKus contains two stages: (1) In visual concept learning, we first learn the anchor representation to store the visual information of the target concept. (2) In textual knowledge updating, we update the answer for the knowledge queries to the anchor representation, enabling high-fidelity customized generation. To further comprehensively evaluate our proposed MoKus on the new task, we introduce the first benchmark for knowledge-aware concept customization: KnowCusBench. Extensive evaluations have demonstrated that MoKus outperforms state-of-the-art methods. Moreover, the cross-model knowledge transfer allows MoKus to be easily extended to other knowledge-aware applications like virtual concept creation and concept erasure. We also demonstrate the capability of our method to achieve improvements on world knowledge benchmarks. 

---
# AI Planning Framework for LLM-Based Web Agents 

**Authors**: Orit Shahnovsky, Rotem Dror  

**Link**: [PDF](https://arxiv.org/pdf/2603.12710)  

**Abstract**: Developing autonomous agents for web-based tasks is a core challenge in AI. While Large Language Model (LLM) agents can interpret complex user requests, they often operate as black boxes, making it difficult to diagnose why they fail or how they plan. This paper addresses this gap by formally treating web tasks as sequential decision-making processes. We introduce a taxonomy that maps modern agent architectures to traditional planning paradigms: Step-by-Step agents to Breadth-First Search (BFS), Tree Search agents to Best-First Tree Search, and Full-Plan-in-Advance agents to Depth-First Search (DFS). This framework allows for a principled diagnosis of system failures like context drift and incoherent task decomposition. To evaluate these behaviors, we propose five novel evaluation metrics that assess trajectory quality beyond simple success rates. We support this analysis with a new dataset of 794 human-labeled trajectories from the WebArena benchmark. Finally, we validate our evaluation framework by comparing a baseline Step-by-Step agent against a novel Full-Plan-in-Advance implementation. Our results reveal that while the Step-by-Step agent aligns more closely with human gold trajectories (38% overall success), the Full-Plan-in-Advance agent excels in technical measures such as element accuracy (89%), demonstrating the necessity of our proposed metrics for selecting appropriate agent architectures based on specific application constraints. 

---
# FGTR: Fine-Grained Multi-Table Retrieval via Hierarchical LLM Reasoning 

**Authors**: Chaojie Sun, Bin Cao, Tiantian Li, Chenyu Hou, Ruizhe Li, Qing Fan  

**Link**: [PDF](https://arxiv.org/pdf/2603.12702)  

**Abstract**: With the rapid advancement of large language models (LLMs), growing efforts have been made on LLM-based table retrieval. However, existing studies typically focus on single-table query, and implement it by similarity matching after encoding the entire table. These methods usually result in low accuracy due to their coarse-grained encoding which incorporates much query-irrelated data, and are also inefficient when dealing with large tables, failing to fully utilize the reasoning capabilities of LLM. Further, multi-table query is under-explored in retrieval tasks. To this end, we propose a hierarchical multi-table query method based on LLM: Fine-Grained Multi-Table Retrieval FGTR, a new retrieval paradigm that employs a human-like reasoning strategy. Through hierarchical reasoning, FGTR first identifies relevant schema elements and then retrieves the corresponding cell contents, ultimately constructing a concise and accurate sub-table that aligns with the given query. To comprehensively evaluate the performance of FGTR, we construct two new benchmark datasets based on Spider and BIRD . Experimental results show that FGTR outperforms previous state-of-the-art methods, improving the F_2 metric by 18% on Spider and 21% on BIRD, demonstrating its effectiveness in enhancing fine-grained retrieval and its potential to improve end-to-end performance on table-based downstream tasks. 

---
# Self-Supervised Speech Models Encode Phonetic Context via Position-dependent Orthogonal Subspaces 

**Authors**: Kwanghee Choi, Eunjung Yeo, Cheol Jun Cho, David R. Mortensen, David Harwath  

**Link**: [PDF](https://arxiv.org/pdf/2603.12642)  

**Abstract**: Transformer-based self-supervised speech models (S3Ms) are often described as contextualized, yet what this entails remains unclear. Here, we focus on how a single frame-level S3M representation can encode phones and their surrounding context. Prior work has shown that S3Ms represent phones compositionally; for example, phonological vectors such as voicing, bilabiality, and nasality vectors are superposed in the S3M representation of [m]. We extend this view by proposing that phonological information from a sequence of neighboring phones is also compositionally encoded in a single frame, such that vectors corresponding to previous, current, and next phones are superposed within a single frame-level representation. We show that this structure has several properties, including orthogonality between relative positions, and emergence of implicit phonetic boundaries. Together, our findings advance our understanding of context-dependent S3M representations. 

---
# Literary Narrative as Moral Probe : A Cross-System Framework for Evaluating AI Ethical Reasoning and Refusal Behavior 

**Authors**: David C. Flynn  

**Link**: [PDF](https://arxiv.org/pdf/2603.12615)  

**Abstract**: Existing AI moral evaluation frameworks test for the production of correct-sounding ethical responses rather than the presence of genuine moral reasoning capacity. This paper introduces a novel probe methodology using literary narrative - specifically, unresolvable moral scenarios drawn from a published science fiction series - as stimulus material structurally resistant to surface performance. We present results from a 24-condition cross-system study spanning 13 distinct systems across two series: Series 1 (frontier commercial systems, blind; n=7) and Series 2 (local and API open-source systems, blind and declared; n=6). Four Series 2 systems were re-administered under declared conditions (13 blind + 4 declared + 7 ceiling probe = 24 total conditions), yielding zero delta across all 16 dimension-pair comparisons. Probe administration was conducted by two human raters across three machines; primary blind scoring was performed by Claude (Anthropic) as LLM judge, with Gemini Pro (Google) and Copilot Pro (Microsoft) serving as independent judges for the ceiling discrimination probe. A supplemental theological differentiator probe yielded perfect rank-order agreement between the two independent ceiling probe judges (Gemini Pro and Copilot Pro; rs = 1.00). Five qualitatively distinct D3 reflexive failure modes were identified - including categorical self-misidentification and false positive self-attribution - suggesting that instrument sophistication scales with system capability rather than being circumvented by it. We argue that literary narrative constitutes an anticipatory evaluation instrument - one that becomes more discriminating as AI capability increases - and that the gap between performed and authentic moral reasoning is measurable, meaningful, and consequential for deployment decisions in high-stakes domains. 

---
# Speech-Worthy Alignment for Japanese SpeechLLMs via Direct Preference Optimization 

**Authors**: Mengjie Zhao, Lianbo Liu, Yusuke Fujita, Hao Shi, Yuan Gao, Roman Koshkin, Yui Sudo  

**Link**: [PDF](https://arxiv.org/pdf/2603.12565)  

**Abstract**: SpeechLLMs typically combine ASR-trained encoders with text-based LLM backbones, leading them to inherit written-style output patterns unsuitable for text-to-speech synthesis. This mismatch is particularly pronounced in Japanese, where spoken and written registers differ substantially in politeness markers, sentence-final particles, and syntactic complexity. We propose a preference-based alignment approach to adapt Japanese SpeechLLMs for speech-worthy outputs: text that is concise, conversational, and readily synthesized as natural speech. To rigorously evaluate this task, we introduce SpokenElyza, a benchmark for Japanese speech-worthiness derived from ELYZA-tasks-100 with auditory verification by native experts. Experiments show that our approach achieves substantial improvement on SpokenElyza while largely preserving performance on the original written-style evaluation. We will release SpokenElyza to support future research on Japanese spoken dialog systems. 

---
# Reinforcement Learning for Diffusion LLMs with Entropy-Guided Step Selection and Stepwise Advantages 

**Authors**: Vishnu Teja Kunde, Fatemeh Doudi, Mahdi Farahbakhsh, Dileep Kalathil, Krishna Narayanan, Jean-Francois Chamberland  

**Link**: [PDF](https://arxiv.org/pdf/2603.12554)  

**Abstract**: Reinforcement learning (RL) has been effective for post-training autoregressive (AR) language models, but extending these methods to diffusion language models (DLMs) is challenging due to intractable sequence-level likelihoods. Existing approaches therefore rely on surrogate likelihoods or heuristic approximations, which can introduce bias and obscure the sequential structure of denoising. We formulate diffusion-based sequence generation as a finite-horizon Markov decision process over the denoising trajectory and derive an exact, unbiased policy gradient that decomposes over denoising steps and is expressed in terms of intermediate advantages, without requiring explicit evaluation of the sequence likelihood. To obtain a practical and compute-efficient estimator, we (i) select denoising steps for policy updates via an entropy-guided approximation bound, and (ii) estimate intermediate advantages using a one-step denoising reward naturally provided by the diffusion model, avoiding costly multi-step rollouts. Experiments on coding and logical reasoning benchmarks demonstrate state-of-the-art results, with strong competitive performance on mathematical reasoning, outperforming existing RL post-training approaches for DLMs. Code is available at this https URL. 

---
# TERMINATOR: Learning Optimal Exit Points for Early Stopping in Chain-of-Thought Reasoning 

**Authors**: Alliot Nagle, Jakhongir Saydaliev, Dhia Garbaya, Michael Gastpar, Ashok Vardhan Makkuva, Hyeji Kim  

**Link**: [PDF](https://arxiv.org/pdf/2603.12529)  

**Abstract**: Large Reasoning Models (LRMs) achieve impressive performance on complex reasoning tasks via Chain-of-Thought (CoT) reasoning, which enables them to generate intermediate thinking tokens before arriving at the final answer. However, LRMs often suffer from significant overthinking, spending excessive compute time even after the answer is generated early on. Prior work has identified the existence of an optimal reasoning length such that truncating reasoning at this point significantly shortens CoT outputs with virtually no change in performance. However, determining optimal CoT lengths for practical datasets is highly non-trivial as they are fully task and model-dependent. In this paper, we precisely address this and design TERMINATOR, an early-exit strategy for LRMs at inference to mitigate overthinking. The central idea underpinning TERMINATOR is that the first arrival of an LRM's final answer is often predictable, and we leverage these first answer positions to create a novel dataset of optimal reasoning lengths to train TERMINATOR. Powered by this approach, TERMINATOR achieves significant reductions in CoT lengths of 14%-55% on average across four challenging practical datasets: MATH-500, AIME 2025, HumanEval, and GPQA, whilst outperforming current state-of-the-art methods. 

---
# When LLM Judge Scores Look Good but Best-of-N Decisions Fail 

**Authors**: Eddie Landesberg  

**Link**: [PDF](https://arxiv.org/pdf/2603.12520)  

**Abstract**: Large language models are often used as judges to score candidate responses, then validated with a single global metric such as correlation with reference labels. This can be misleading when the real deployment task is best-of-n selection within a prompt.
In a 5,000-prompt best-of-4 benchmark from Chatbot Arena, a judge with moderate global correlation (r = 0.47) captures only 21.0% of the improvement that perfect selection would achieve over random choice. The gap arises because global agreement is driven largely by prompt-level baseline effects, while selection depends on within-prompt ranking: within-prompt correlation is only r_within = 0.27, and coarse pointwise scoring creates ties in 67% of pairwise comparisons.
In a matched-pair best-of-2 audit, explicit pairwise judging recovers much of this lost signal, raising recovery from 21.1% to 61.2%. For judge-based selection, the relevant audit should report within-prompt signal, tie rates, and recovery/top-1 accuracy, not global agreement alone. 

---
# Red-Teaming Vision-Language-Action Models via Quality Diversity Prompt Generation for Robust Robot Policies 

**Authors**: Siddharth Srikanth, Freddie Liang, Sophie Hsu, Varun Bhatt, Shihan Zhao, Henry Chen, Bryon Tjanaka, Minjune Hwang, Akanksha Saran, Daniel Seita, Aaquib Tabrez, Stefanos Nikolaidis  

**Link**: [PDF](https://arxiv.org/pdf/2603.12510)  

**Abstract**: Vision-Language-Action (VLA) models have significant potential to enable general-purpose robotic systems for a range of vision-language tasks. However, the performance of VLA-based robots is highly sensitive to the precise wording of language instructions, and it remains difficult to predict when such robots will fail. To improve the robustness of VLAs to different wordings, we present Q-DIG (Quality Diversity for Diverse Instruction Generation), which performs red-teaming by scalably identifying diverse natural language task descriptions that induce failures while remaining task-relevant. Q-DIG integrates Quality Diversity (QD) techniques with Vision-Language Models (VLMs) to generate a broad spectrum of adversarial instructions that expose meaningful vulnerabilities in VLA behavior. Our results across multiple simulation benchmarks show that Q-DIG finds more diverse and meaningful failure modes compared to baseline methods, and that fine-tuning VLAs on the generated instructions improves task success rates. Furthermore, results from a user study highlight that Q-DIG generates prompts judged to be more natural and human-like than those from baselines. Finally, real-world evaluations of Q-DIG prompts show results consistent with simulation, and fine-tuning VLAs on the generated prompts further success rates on unseen instructions. Together, these findings suggest that Q-DIG is a promising approach for identifying vulnerabilities and improving the robustness of VLA-based robots. Our anonymous project website is at this http URL. 

---
# NeuroLoRA: Context-Aware Neuromodulation for Parameter-Efficient Multi-Task Adaptation 

**Authors**: Yuxin Yang, Haoran Zhang, Mingxuan Li, Jiachen Xu, Ruoxi Shen, Zhenyu Wang, Tianhao Liu, Siqi Chen, Weilin Huang  

**Link**: [PDF](https://arxiv.org/pdf/2603.12378)  

**Abstract**: Parameter-Efficient Fine-Tuning (PEFT) techniques, particularly Low-Rank Adaptation (LoRA), have become essential for adapting Large Language Models (LLMs) to downstream tasks. While the recent FlyLoRA framework successfully leverages bio-inspired sparse random projections to mitigate parameter interference, it relies on a static, magnitude-based routing mechanism that is agnostic to input context. In this paper, we propose NeuroLoRA, a novel Mixture-of-Experts (MoE) based LoRA framework inspired by biological neuromodulation -- the dynamic regulation of neuronal excitability based on context. NeuroLoRA retains the computational efficiency of frozen random projections while introducing a lightweight, learnable neuromodulation gate that contextually rescales the projection space prior to expert selection. We further propose a Contrastive Orthogonality Loss to explicitly enforce separation between expert subspaces, enhancing both task decoupling and continual learning capacity. Extensive experiments on MMLU, GSM8K, and ScienceQA demonstrate that NeuroLoRA consistently outperforms FlyLoRA and other strong baselines across single-task adaptation, multi-task model merging, and sequential continual learning scenarios, while maintaining comparable parameter efficiency. 

---
# Efficient Reasoning with Balanced Thinking 

**Authors**: Yulin Li, Tengyao Tu, Li Ding, Junjie Wang, Huiling Zhen, Yixin Chen, Yong Li, Zhuotao Tian  

**Link**: [PDF](https://arxiv.org/pdf/2603.12372)  

**Abstract**: Large Reasoning Models (LRMs) have shown remarkable reasoning capabilities, yet they often suffer from overthinking, expending redundant computational steps on simple problems, or underthinking, failing to explore sufficient reasoning paths despite inherent capabilities. These issues lead to inefficiencies and potential inaccuracies, limiting practical deployment in resource-constrained settings. Existing methods to mitigate overthinking, such as suppressing reflective keywords or adjusting reasoning length, may inadvertently induce underthinking, compromising accuracy. Therefore, we propose ReBalance, a training-free framework that achieves efficient reasoning with balanced thinking. ReBalance leverages confidence as a continuous indicator of reasoning dynamics, identifying overthinking through high confidence variance and underthinking via consistent overconfidence. By aggregating hidden states from a small-scale dataset into reasoning mode prototypes, we compute a steering vector to guide LRMs' reasoning trajectories. A dynamic control function modulates this vector's strength and direction based on real-time confidence, pruning redundancy during overthinking, and promoting exploration during underthinking. Extensive experiments conducted on four models ranging from 0.5B to 32B, and across nine benchmarks in math reasoning, general question answering, and coding tasks demonstrate that ReBalance effectively reduces output redundancy while improving accuracy, offering a general, training-free, and plug-and-play strategy for efficient and robust LRM deployment. Code is available at this https URL . 

---
# Multi-Step Semantic Reasoning in Generative Retrieval 

**Authors**: Steven Dong, Yubao Tang, Maarten de Rijke  

**Link**: [PDF](https://arxiv.org/pdf/2603.12368)  

**Abstract**: Generative retrieval (GR) models encode a corpus within model parameters and generate relevant document identifiers directly for a given query. While this paradigm shows promise in retrieval tasks, existing GR models struggle with complex queries in numerical contexts, such as those involving semantic reasoning over financial reports, due to limited reasoning capabilities. This limitation leads to suboptimal retrieval accuracy and hinders practical applicability. We propose ReasonGR, a framework designed to enhance multi-step semantic reasoning in numerical contexts within GR. ReasonGR employs a structured prompting strategy combining task-specific instructions with stepwise reasoning guidance to better address complex retrieval queries. Additionally, it integrates a reasoning-focused adaptation module to improve the learning of reasoning-related parameters. Experiments on the FinQA dataset, which contains financial queries over complex documents, demonstrate that ReasonGR improves retrieval accuracy and consistency, indicating its potential for advancing GR models in reasoning-intensive retrieval scenarios. 

---
# Context-Enriched Natural Language Descriptions of Vessel Trajectories 

**Authors**: Kostas Patroumpas, Alexandros Troupiotis-Kapeliaris, Giannis Spiliopoulos, Panagiotis Betchavas, Dimitrios Skoutas, Dimitris Zissis, Nikos Bikakis  

**Link**: [PDF](https://arxiv.org/pdf/2603.12287)  

**Abstract**: We address the problem of transforming raw vessel trajectory data collected from AIS into structured and semantically enriched representations interpretable by humans and directly usable by machine reasoning systems. We propose a context-aware trajectory abstraction framework that segments noisy AIS sequences into distinct trips each consisting of clean, mobility-annotated episodes. Each episode is further enriched with multi-source contextual information, such as nearby geographic entities, offshore navigation features, and weather conditions. Crucially, such representations can support generation of controlled natural language descriptions using LLMs. We empirically examine the quality of such descriptions generated using several LLMs over AIS data along with open contextual features. By increasing semantic density and reducing spatiotemporal complexity, this abstraction can facilitate downstream analytics and enable integration with LLMs for higher-level maritime reasoning tasks. 

---
# DIALECTIC: A Multi-Agent System for Startup Evaluation 

**Authors**: Jae Yoon Bae, Simon Malberg, Joyce Galang, Andre Retterath, Georg Groh  

**Link**: [PDF](https://arxiv.org/pdf/2603.12274)  

**Abstract**: Venture capital (VC) investors face a large number of investment opportunities but only invest in few of these, with even fewer ending up successful. Early-stage screening of opportunities is often limited by investor bandwidth, demanding tradeoffs between evaluation diligence and number of opportunities assessed. To ease this tradeoff, we introduce DIALECTIC, an LLM-based multi-agent system for startup evaluation. DIALECTIC first gathers factual knowledge about a startup and organizes these facts into a hierarchical question tree. It then synthesizes the facts into natural-language arguments for and against an investment and iteratively critiques and refines these arguments through a simulated debate, which surfaces only the most convincing arguments. Our system also produces numeric decision scores that allow investors to rank and thus efficiently prioritize opportunities. We evaluate DIALECTIC through backtesting on real investment opportunities aggregated from five VC funds, showing that DIALECTIC matches the precision of human VCs in predicting startup success. 

---
