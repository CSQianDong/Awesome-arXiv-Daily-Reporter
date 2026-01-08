# All That Glisters Is Not Gold: A Benchmark for Reference-Free Counterfactual Financial Misinformation Detection 

**Authors**: Yuechen Jiang, Zhiwei Liu, Yupeng Cao, Yueru He, Ziyang Xu, Chen Xu, Zhiyang Deng, Prayag Tiwari, Xi Chen, Alejandro Lopez-Lira, Jimin Huang, Junichi Tsujii, Sophia Ananiadou  

**Link**: [PDF](https://arxiv.org/pdf/2601.04160)  

**Abstract**: We introduce RFC Bench, a benchmark for evaluating large language models on financial misinformation under realistic news. RFC Bench operates at the paragraph level and captures the contextual complexity of financial news where meaning emerges from dispersed cues. The benchmark defines two complementary tasks: reference free misinformation detection and comparison based diagnosis using paired original perturbed inputs. Experiments reveal a consistent pattern: performance is substantially stronger when comparative context is available, while reference free settings expose significant weaknesses, including unstable predictions and elevated invalid outputs. These results indicate that current models struggle to maintain coherent belief states without external grounding. By highlighting this gap, RFC Bench provides a structured testbed for studying reference free reasoning and advancing more reliable financial misinformation detection in real world settings. 

---
# FLEx: Language Modeling with Few-shot Language Explanations 

**Authors**: Adar Avsian, Christopher Richardson, Anirudh Sundar, Larry Heck  

**Link**: [PDF](https://arxiv.org/pdf/2601.04157)  

**Abstract**: Language models have become effective at a wide range of tasks, from math problem solving to open-domain question answering. However, they still make mistakes, and these mistakes are often repeated across related queries. Natural language explanations can help correct these errors, but collecting them at scale may be infeasible, particularly in domains where expert annotators are required. To address this issue, we introduce FLEx ($\textbf{F}$ew-shot $\textbf{L}$anguage $\textbf{Ex}$planations), a method for improving model behavior using a small number of explanatory examples. FLEx selects representative model errors using embedding-based clustering, verifies that the associated explanations correct those errors, and summarizes them into a prompt prefix that is prepended at inference-time. This summary guides the model to avoid similar errors on new inputs, without modifying model weights. We evaluate FLEx on CounterBench, GSM8K, and ReasonIF. We find that FLEx consistently outperforms chain-of-thought (CoT) prompting across all three datasets and reduces up to 83\% of CoT's remaining errors. 

---
# LLMberjack: Guided Trimming of Debate Trees for Multi-Party Conversation Creation 

**Authors**: Leonardo Bottona, Nicolò Penzo, Bruno Lepri, Marco Guerini, Sara Tonelli  

**Link**: [PDF](https://arxiv.org/pdf/2601.04135)  

**Abstract**: We present LLMberjack, a platform for creating multi-party conversations starting from existing debates, originally structured as reply trees. The system offers an interactive interface that visualizes discussion trees and enables users to construct coherent linearized dialogue sequences while preserving participant identity and discourse relations. It integrates optional large language model (LLM) assistance to support automatic editing of the messages and speakers' descriptions. We demonstrate the platform's utility by showing how tree visualization facilitates the creation of coherent, meaningful conversation threads and how LLM support enhances output quality while reducing human effort. The tool is open-source and designed to promote transparent and reproducible workflows to create multi-party conversations, addressing a lack of resources of this type. 

---
# ContextFocus: Activation Steering for Contextual Faithfulness in Large Language Models 

**Authors**: Nikhil Anand, Shwetha Somasundaram, Anirudh Phukan, Apoorv Saxena, Koyel Mukherjee  

**Link**: [PDF](https://arxiv.org/pdf/2601.04131)  

**Abstract**: Large Language Models (LLMs) encode vast amounts of parametric knowledge during pre-training. As world knowledge evolves, effective deployment increasingly depends on their ability to faithfully follow externally retrieved context. When such evidence conflicts with the model's internal knowledge, LLMs often default to memorized facts, producing unfaithful outputs. In this work, we introduce ContextFocus, a lightweight activation steering approach that improves context faithfulness in such knowledge-conflict settings while preserving fluency and efficiency. Unlike prior approaches, our solution requires no model finetuning and incurs minimal inference-time overhead, making it highly efficient. We evaluate ContextFocus on the ConFiQA benchmark, comparing it against strong baselines including ContextDPO, COIECD, and prompting-based methods. Furthermore, we show that our method is complementary to prompting strategies and remains effective on larger models. Extensive experiments show that ContextFocus significantly improves contextual-faithfulness. Our results highlight the effectiveness, robustness, and efficiency of ContextFocus in improving contextual-faithfulness of LLM outputs. 

---
# InfiniteWeb: Scalable Web Environment Synthesis for GUI Agent Training 

**Authors**: Ziyun Zhang, Zezhou Wang, Xiaoyi Zhang, Zongyu Guo, Jiahao Li, Bin Li, Yan Lu  

**Link**: [PDF](https://arxiv.org/pdf/2601.04126)  

**Abstract**: GUI agents that interact with graphical interfaces on behalf of users represent a promising direction for practical AI assistants. However, training such agents is hindered by the scarcity of suitable environments. We present InfiniteWeb, a system that automatically generates functional web environments at scale for GUI agent training. While LLMs perform well on generating a single webpage, building a realistic and functional website with many interconnected pages faces challenges. We address these challenges through unified specification, task-centric test-driven development, and a combination of website seed with reference design image to ensure diversity. Our system also generates verifiable task evaluators enabling dense reward signals for reinforcement learning. Experiments show that InfiniteWeb surpasses commercial coding agents at realistic website construction, and GUI agents trained on our generated environments achieve significant performance improvements on OSWorld and Online-Mind2Web, demonstrating the effectiveness of proposed system. 

---
# Layer-wise Positional Bias in Short-Context Language Modeling 

**Authors**: Maryam Rahimi, Mahdi Nouri, Yadollah Yaghoobzadeh  

**Link**: [PDF](https://arxiv.org/pdf/2601.04098)  

**Abstract**: Language models often show a preference for using information from specific positions in the input regardless of semantic relevance. While positional bias has been studied in various contexts, from attention sinks to task performance degradation in long-context settings, prior work has not established how these biases evolve across individual layers and input positions, or how they vary independent of task complexity. We introduce an attribution-based framework to analyze positional effects in short-context language modeling. Using layer conductance with a sliding-window approach, we quantify how each layer distributes importance across input positions, yielding layer-wise positional importance profiles. We find that these profiles are architecture-specific, stable across inputs, and invariant to lexical scrambling. Characterizing these profiles, we find prominent recency bias that increases with depth and subtle primacy bias that diminishes through model depth. Beyond positional structure, we also show that early layers preferentially weight content words over function words across all positions, while later layers lose this word-type differentiation. 

---
# SearchAttack: Red-Teaming LLMs against Real-World Threats via Framing Unsafe Web Information-Seeking Tasks 

**Authors**: Yu Yan, Sheng Sun, Mingfeng Li, Zheming Yang, Chiwei Zhu, Fei Ma, Benfeng Xu, Min Liu  

**Link**: [PDF](https://arxiv.org/pdf/2601.04093)  

**Abstract**: Recently, people have suffered and become increasingly aware of the unreliability gap in LLMs for open and knowledge-intensive tasks, and thus turn to search-augmented LLMs to mitigate this issue. However, when the search engine is triggered for harmful tasks, the outcome is no longer under the LLM's control. Once the returned content directly contains targeted, ready-to-use harmful takeaways, the LLM's safeguards cannot withdraw that exposure. Motivated by this dilemma, we identify web search as a critical attack surface and propose \textbf{\textit{SearchAttack}} for red-teaming. SearchAttack outsources the harmful semantics to web search, retaining only the query's skeleton and fragmented clues, and further steers LLMs to reconstruct the retrieved content via structural rubrics to achieve malicious goals. Extensive experiments are conducted to red-team the search-augmented LLMs for responsible vulnerability assessment. Empirically, SearchAttack demonstrates strong effectiveness in attacking these systems. 

---
# KDCM: Reducing Hallucination in LLMs through Explicit Reasoning Structures 

**Authors**: Jinbo Hao, Kai Yang, Qingzhen Su, Yifan Li, Chao Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2601.04086)  

**Abstract**: To mitigate hallucinations in large language models (LLMs), we propose a framework that focuses on errors induced by prompts. Our method extends a chain-style knowledge distillation approach by incorporating a programmable module that guides knowledge graph exploration. This module is embedded as executable code within the reasoning prompt, allowing the model to leverage external structured knowledge during inference. Based on this design, we develop an enhanced distillation-based reasoning framework that explicitly regulates intermediate reasoning steps, resulting in more reliable predictions. We evaluate the proposed approach on multiple public benchmarks using GPT-4 and LLaMA-3.3. Experimental results show that code-guided reasoning significantly improves contextual modeling and reduces prompt-induced hallucinations. Specifically, HIT@1, HIT@3, and HIT@5 increase by 15.64%, 13.38%, and 13.28%, respectively, with scores exceeding 95% across several evaluation settings. These findings indicate that the proposed method effectively constrains erroneous reasoning while improving both accuracy and interpretability. 

---
# Bridging the Discrete-Continuous Gap: Unified Multimodal Generation via Coupled Manifold Discrete Absorbing Diffusion 

**Authors**: Yuanfeng Xu, Yuhao Chen, Liang Lin, Guangrun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.04056)  

**Abstract**: The bifurcation of generative modeling into autoregressive approaches for discrete data (text) and diffusion approaches for continuous data (images) hinders the development of truly unified multimodal systems. While Masked Language Models (MLMs) offer efficient bidirectional context, they traditionally lack the generative fidelity of autoregressive models and the semantic continuity of diffusion models. Furthermore, extending masked generation to multimodal settings introduces severe alignment challenges and training instability. In this work, we propose \textbf{CoM-DAD} (\textbf{Co}upled \textbf{M}anifold \textbf{D}iscrete \textbf{A}bsorbing \textbf{D}iffusion), a novel probabilistic framework that reformulates multimodal generation as a hierarchical dual-process. CoM-DAD decouples high-level semantic planning from low-level token synthesis. First, we model the semantic manifold via a continuous latent diffusion process; second, we treat token generation as a discrete absorbing diffusion process, regulated by a \textbf{Variable-Rate Noise Schedule}, conditioned on these evolving semantic priors. Crucially, we introduce a \textbf{Stochastic Mixed-Modal Transport} strategy that aligns disparate modalities without requiring heavy contrastive dual-encoders. Our method demonstrates superior stability over standard masked modeling, establishing a new paradigm for scalable, unified text-image generation. 

---
# Modular Prompt Optimization: Optimizing Structured Prompts with Section-Local Textual Gradients 

**Authors**: Prith Sharma, Austin Z. Henley  

**Link**: [PDF](https://arxiv.org/pdf/2601.04055)  

**Abstract**: Prompt quality plays a central role in controlling the behavior, reliability, and reasoning performance of large language models (LLMs), particularly for smaller open-source instruction-tuned models that depend heavily on explicit structure. While recent work has explored automatic prompt optimization using textual gradients and self-refinement, most existing methods treat prompts as monolithic blocks of text, making it difficult to localize errors, preserve critical instructions, or prevent uncontrolled prompt growth. We introduce Modular Prompt Optimization (MPO), a schema-based prompt optimization framework that treats prompts as structured objects composed of fixed semantic sections, including system role, context, task description, constraints, and output format. MPO applies section-local textual gradients, generated by a critic language model, to refine each section independently while keeping the overall prompt schema fixed. Section updates are consolidated through de-duplication to reduce redundancy and interference between components, yielding an interpretable and robust optimization process. We evaluate MPO on two reasoning benchmarks, ARC-Challenge and MMLU, using LLaMA-3 8B-Instruct and Mistral-7B-Instruct as solver models. Across both benchmarks and models, MPO consistently outperforms an untuned structured prompt and the TextGrad baseline, achieving substantial accuracy gains without modifying model parameters or altering prompt structure. These results demonstrate that maintaining a fixed prompt schema while applying localized, section-wise optimization is an effective and practical approach for improving reasoning performance in small open-source LMs. 

---
# When Helpers Become Hazards: A Benchmark for Analyzing Multimodal LLM-Powered Safety in Daily Life 

**Authors**: Xinyue Lou, Jinan Xu, Jingyi Yin, Xiaolong Wang, Zhaolu Kang, Youwei Liao, Yixuan Wang, Xiangyu Shi, Fengran Mo, Su Yao, Kaiyu Huang  

**Link**: [PDF](https://arxiv.org/pdf/2601.04043)  

**Abstract**: As Multimodal Large Language Models (MLLMs) become an indispensable assistant in human life, the unsafe content generated by MLLMs poses a danger to human behavior, perpetually overhanging human society like a sword of Damocles. To investigate and evaluate the safety impact of MLLMs responses on human behavior in daily life, we introduce SaLAD, a multimodal safety benchmark which contains 2,013 real-world image-text samples across 10 common categories, with a balanced design covering both unsafe scenarios and cases of oversensitivity. It emphasizes realistic risk exposure, authentic visual inputs, and fine-grained cross-modal reasoning, ensuring that safety risks cannot be inferred from text alone. We further propose a safety-warning-based evaluation framework that encourages models to provide clear and informative safety warnings, rather than generic refusals. Results on 18 MLLMs demonstrate that the top-performing models achieve a safe response rate of only 57.2% on unsafe queries. Moreover, even popular safety alignment methods limit effectiveness of the models in our scenario, revealing the vulnerabilities of current MLLMs in identifying dangerous behaviors in daily life. Our dataset is available at this https URL. 

---
# Analyzing and Improving Cross-lingual Knowledge Transfer for Machine Translation 

**Authors**: David Stap  

**Link**: [PDF](https://arxiv.org/pdf/2601.04036)  

**Abstract**: Multilingual machine translation systems aim to make knowledge accessible across languages, yet learning effective cross-lingual representations remains challenging. These challenges are especially pronounced for low-resource languages, where limited parallel data constrains generalization and transfer. Understanding how multilingual models share knowledge across languages requires examining the interaction between representations, data availability, and training strategies. In this thesis, we study cross-lingual knowledge transfer in neural models and develop methods to improve robustness and generalization in multilingual settings, using machine translation as a central testbed. We analyze how similarity between languages influences transfer, how retrieval and auxiliary supervision can strengthen low-resource translation, and how fine-tuning on parallel data can introduce unintended trade-offs in large language models. We further examine the role of language diversity during training and show that increasing translation coverage improves generalization and reduces off-target behavior. Together, this work highlights how modeling choices and data composition shape multilingual learning and offers insights toward more inclusive and resilient multilingual NLP systems. 

---
# SpeakerSleuth: Evaluating Large Audio-Language Models as Judges for Multi-turn Speaker Consistency 

**Authors**: Jonggeun Lee, Junseong Pyo, Gyuhyeon Seo, Yohan Jo  

**Link**: [PDF](https://arxiv.org/pdf/2601.04029)  

**Abstract**: Large Audio-Language Models (LALMs) as judges have emerged as a prominent approach for evaluating speech generation quality, yet their ability to assess speaker consistency across multi-turn conversations remains unexplored. We present SpeakerSleuth, a benchmark evaluating whether LALMs can reliably judge speaker consistency in multi-turn dialogues through three tasks reflecting real-world requirements. We construct 1,818 human-verified evaluation instances across four diverse datasets spanning synthetic and real speech, with controlled acoustic difficulty. Evaluating nine widely-used LALMs, we find that models struggle to reliably detect acoustic inconsistencies. For instance, given audio samples of the same speaker's turns, some models overpredict inconsistency, whereas others are overly lenient. Models further struggle to identify the exact turns that are problematic. When other interlocutors' turns are provided together, performance degrades dramatically as models prioritize textual coherence over acoustic cues, failing to detect even obvious gender switches for a speaker. On the other hand, models perform substantially better in choosing the audio that best matches the speaker among several acoustic variants, demonstrating inherent acoustic discrimination capabilities. These findings expose a significant bias in LALMs: they tend to prioritize text over acoustics, revealing fundamental modality imbalances that need to be addressed to build reliable audio-language judges. 

---
# Simulated Students in Tutoring Dialogues: Substance or Illusion? 

**Authors**: Alexander Scarlatos, Jaewook Lee, Simon Woodhead, Andrew Lan  

**Link**: [PDF](https://arxiv.org/pdf/2601.04025)  

**Abstract**: Advances in large language models (LLMs) enable many new innovations in education. However, evaluating the effectiveness of new technology requires real students, which is time-consuming and hard to scale up. Therefore, many recent works on LLM-powered tutoring solutions have used simulated students for both training and evaluation, often via simple prompting. Surprisingly, little work has been done to ensure or even measure the quality of simulated students. In this work, we formally define the student simulation task, propose a set of evaluation metrics that span linguistic, behavioral, and cognitive aspects, and benchmark a wide range of student simulation methods on these metrics. We experiment on a real-world math tutoring dialogue dataset, where both automated and human evaluation results show that prompting strategies for student simulation perform poorly; supervised fine-tuning and preference optimization yield much better but still limited performance, motivating future work on this challenging task. 

---
# VotIE: Information Extraction from Meeting Minutes 

**Authors**: José Pedro Evans, Luís Filipe Cunha, Purificação Silvano, Alípio Jorge, Nuno Guimarães, Sérgio Nunes, Ricardo Campos  

**Link**: [PDF](https://arxiv.org/pdf/2601.03997)  

**Abstract**: Municipal meeting minutes record key decisions in local democratic processes. Unlike parliamentary proceedings, which typically adhere to standardized formats, they encode voting outcomes in highly heterogeneous, free-form narrative text that varies widely across municipalities, posing significant challenges for automated extraction. In this paper, we introduce VotIE (Voting Information Extraction), a new information extraction task aimed at identifying structured voting events in narrative deliberative records, and establish the first benchmark for this task using Portuguese municipal minutes, building on the recently introduced CitiLink corpus. Our experiments yield two key findings. First, under standard in-domain evaluation, fine-tuned encoders, specifically XLM-R-CRF, achieve the strongest performance, reaching 93.2\% macro F1, outperforming generative approaches. Second, in a cross-municipality setting that evaluates transfer to unseen administrative contexts, these models suffer substantial performance degradation, whereas few-shot LLMs demonstrate greater robustness, with significantly smaller declines in performance. Despite this generalization advantage, the high computational cost of generative models currently constrains their practicality. As a result, lightweight fine-tuned encoders remain a more practical option for large-scale, real-world deployment. To support reproducible research in administrative NLP, we publicly release our benchmark, trained models, and evaluation framework. 

---
# Benchmark^2: Systematic Evaluation of LLM Benchmarks 

**Authors**: Qi Qian, Chengsong Huang, Jingwen Xu, Changze Lv, Muling Wu, Wenhao Liu, Xiaohua Wang, Zhenghua Wang, Zisu Huang, Muzhao Tian, Jianhan Xu, Kun Hu, He-Da Wang, Yao Hu, Xuanjing Huang, Xiaoqing Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2601.03986)  

**Abstract**: The rapid proliferation of benchmarks for evaluating large language models (LLMs) has created an urgent need for systematic methods to assess benchmark quality itself. We propose Benchmark^2, a comprehensive framework comprising three complementary metrics: (1) Cross-Benchmark Ranking Consistency, measuring whether a benchmark produces model rankings aligned with peer benchmarks; (2) Discriminability Score, quantifying a benchmark's ability to differentiate between models; and (3) Capability Alignment Deviation, identifying problematic instances where stronger models fail but weaker models succeed within the same model family. We conduct extensive experiments across 15 benchmarks spanning mathematics, reasoning, and knowledge domains, evaluating 11 LLMs across four model families. Our analysis reveals significant quality variations among existing benchmarks and demonstrates that selective benchmark construction based on our metrics can achieve comparable evaluation performance with substantially reduced test sets. 

---
# RADAR: Retrieval-Augmented Detector with Adversarial Refinement for Robust Fake News Detection 

**Authors**: Song-Duo Ma, Yi-Hung Liu, Hsin-Yu Lin, Pin-Yu Chen, Hong-Yan Huang, Shau-Yung Hsu, Yun-Nung Chen  

**Link**: [PDF](https://arxiv.org/pdf/2601.03981)  

**Abstract**: To efficiently combat the spread of LLM-generated misinformation, we present RADAR, a retrieval-augmented detector with adversarial refinement for robust fake news detection. Our approach employs a generator that rewrites real articles with factual perturbations, paired with a lightweight detector that verifies claims using dense passage retrieval. To enable effective co-evolution, we introduce verbal adversarial feedback (VAF). Rather than relying on scalar rewards, VAF issues structured natural-language critiques; these guide the generator toward more sophisticated evasion attempts, compelling the detector to adapt and improve. On a fake news detection benchmark, RADAR achieves 86.98% ROC-AUC, significantly outperforming general-purpose LLMs with retrieval. Ablation studies confirm that detector-side retrieval yields the largest gains, while VAF and few-shot demonstrations provide critical signals for robust training. 

---
# Large-Scale Aspect-Based Sentiment Analysis with Reasoning-Infused LLMs 

**Authors**: Paweł Liskowski, Krzysztof Jankowski  

**Link**: [PDF](https://arxiv.org/pdf/2601.03940)  

**Abstract**: We introduce Arctic-ABSA, a collection of powerful models for real-life aspect-based sentiment analysis (ABSA). Our models are tailored to commercial needs, trained on a large corpus of public data alongside carefully generated synthetic data, resulting in a dataset 20 times larger than SemEval14. We extend typical ABSA models by expanding the number of sentiment classes from the standard three (positive, negative, neutral) to five, adding mixed and unknown classes, while also jointly predicting overall text sentiment and supporting multiple languages. We experiment with reasoning injection by fine-tuning on Chain-of-Thought (CoT) examples and introduce a novel reasoning pretraining technique for encoder-only models that significantly improves downstream fine-tuning and generalization. Our 395M-parameter encoder and 8B-parameter decoder achieve up to 10 percentage points higher accuracy than GPT-4o and Claude 3.5 Sonnet, while setting new state-of-the-art results on the SemEval14 benchmark. A single multilingual model maintains 87-91% accuracy across six languages without degrading English performance. We release ABSA-mix, a large-scale benchmark aggregating 17 public ABSA datasets across 92 domains. 

---
# Doc-PP: Document Policy Preservation Benchmark for Large Vision-Language Models 

**Authors**: Haeun Jang, Hwan Chang, Hwanhee Lee  

**Link**: [PDF](https://arxiv.org/pdf/2601.03926)  

**Abstract**: The deployment of Large Vision-Language Models (LVLMs) for real-world document question answering is often constrained by dynamic, user-defined policies that dictate information disclosure based on context. While ensuring adherence to these explicit constraints is critical, existing safety research primarily focuses on implicit social norms or text-only settings, overlooking the complexities of multimodal documents. In this paper, we introduce Doc-PP (Document Policy Preservation Benchmark), a novel benchmark constructed from real-world reports requiring reasoning across heterogeneous visual and textual elements under strict non-disclosure policies. Our evaluation highlights a systemic Reasoning-Induced Safety Gap: models frequently leak sensitive information when answers must be inferred through complex synthesis or aggregated across modalities, effectively circumventing existing safety constraints. Furthermore, we identify that providing extracted text improves perception but inadvertently facilitates leakage. To address these vulnerabilities, we propose DVA (Decompose-Verify-Aggregation), a structural inference framework that decouples reasoning from policy verification. Experimental results demonstrate that DVA significantly outperforms standard prompting defenses, offering a robust baseline for policy-compliant document understanding 

---
# When Models Decide and When They Bind: A Two-Stage Computation for Multiple-Choice Question-Answering 

**Authors**: Hugh Mee Wong, Rick Nouwen, Albert Gatt  

**Link**: [PDF](https://arxiv.org/pdf/2601.03914)  

**Abstract**: Multiple-choice question answering (MCQA) is easy to evaluate but adds a meta-task: models must both solve the problem and output the symbol that *represents* the answer, conflating reasoning errors with symbol-binding failures. We study how language models implement MCQA internally using representational analyses (PCA, linear probes) as well as causal interventions. We find that option-boundary (newline) residual states often contain strong linearly decodable signals related to per-option correctness. Winner-identity probing reveals a two-stage progression: the winning *content position* becomes decodable immediately after the final option is processed, while the *output symbol* is represented closer to the answer emission position. Tests under symbol and content permutations support a two-stage mechanism in which models first select a winner in content space and then bind or route that winner to the appropriate symbol to emit. 

---
# Decide Then Retrieve: A Training-Free Framework with Uncertainty-Guided Triggering and Dual-Path Retrieval 

**Authors**: Wang Chen, Guanqiang Qi, Weikang Li, Yang Li, Deguo Xia, Jizhou Huang  

**Link**: [PDF](https://arxiv.org/pdf/2601.03908)  

**Abstract**: Retrieval-augmented generation (RAG) enhances large language models (LLMs) by incorporating external knowledge, but existing approaches indiscriminately trigger retrieval and rely on single-path evidence construction, often introducing noise and limiting performance gains. In this work, we propose Decide Then Retrieve (DTR), a training-free framework that adaptively determines when retrieval is necessary and how external information should be selected. DTR leverages generation uncertainty to guide retrieval triggering and introduces a dual-path retrieval mechanism with adaptive information selection to better handle sparse and ambiguous queries. Extensive experiments across five open-domain QA benchmarks, multiple model scales, and different retrievers demonstrate that DTR consistently improves EM and F1 over standard RAG and strong retrieval-enhanced baselines, while reducing unnecessary retrievals. The code and data used in this paper are available at this https URL. 

---
# Evaluating Small Decoder-Only Language Models for Grammar Correction and Text Simplification 

**Authors**: Anthony Lamelas  

**Link**: [PDF](https://arxiv.org/pdf/2601.03874)  

**Abstract**: Large language models have become extremely popular recently due to their ability to achieve strong performance on a variety of tasks, such as text generation and rewriting, but their size and computation cost make them difficult to access, deploy, and secure in many settings. This paper investigates whether small, decoder-only language models can provide an efficient alternative for the tasks of grammar correction and text simplification. The experiments in this paper focus on testing small language models out of the box, fine-tuned, and run sequentially on the JFLEG and ASSET datasets using established metrics. The results show that while SLMs may learn certain behaviors well, their performance remains below strong baselines and current LLMs. The results also show that SLMs struggle with retaining meaning and hallucinations. These findings suggest that despite their efficiency advantages, current SLMs are not yet competitive enough with modern LLMs for rewriting, and further advances in training are required for SLMs to close the performance gap between them and today's LLMs. 

---
# Atlas: Orchestrating Heterogeneous Models and Tools for Multi-Domain Complex Reasoning 

**Authors**: Jinyang Wu, Guocheng Zhai, Ruihan Jin, Jiahao Yuan, Yuhao Shen, Shuai Zhang, Zhengqi Wen, Jianhua Tao  

**Link**: [PDF](https://arxiv.org/pdf/2601.03872)  

**Abstract**: The integration of large language models (LLMs) with external tools has significantly expanded the capabilities of AI agents. However, as the diversity of both LLMs and tools increases, selecting the optimal model-tool combination becomes a high-dimensional optimization challenge. Existing approaches often rely on a single model or fixed tool-calling logic, failing to exploit the performance variations across heterogeneous model-tool pairs. In this paper, we present ATLAS (Adaptive Tool-LLM Alignment and Synergistic Invocation), a dual-path framework for dynamic tool usage in cross-domain complex reasoning. ATLAS operates via a dual-path approach: (1) \textbf{training-free cluster-based routing} that exploits empirical priors for domain-specific alignment, and (2) \textbf{RL-based multi-step routing} that explores autonomous trajectories for out-of-distribution generalization. Extensive experiments across 15 benchmarks demonstrate that our method outperforms closed-source models like GPT-4o, surpassing existing routing methods on both in-distribution (+10.1%) and out-of-distribution (+13.1%) tasks. Furthermore, our framework shows significant gains in visual reasoning by orchestrating specialized multi-modal tools. 

---
# What Matters For Safety Alignment? 

**Authors**: Xing Li, Hui-Ling Zhen, Lihao Yin, Xianzhi Yu, Zhenhua Dong, Mingxuan Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2601.03868)  

**Abstract**: This paper presents a comprehensive empirical study on the safety alignment capabilities. We evaluate what matters for safety alignment in LLMs and LRMs to provide essential insights for developing more secure and reliable AI systems. We systematically investigate and compare the influence of six critical intrinsic model characteristics and three external attack techniques. Our large-scale evaluation is conducted using 32 recent, popular LLMs and LRMs across thirteen distinct model families, spanning a parameter scale from 3B to 235B. The assessment leverages five established safety datasets and probes model vulnerabilities with 56 jailbreak techniques and four CoT attack strategies, resulting in 4.6M API calls. Our key empirical findings are fourfold. First, we identify the LRMs GPT-OSS-20B, Qwen3-Next-80B-A3B-Thinking, and GPT-OSS-120B as the top-three safest models, which substantiates the significant advantage of integrated reasoning and self-reflection mechanisms for robust safety alignment. Second, post-training and knowledge distillation may lead to a systematic degradation of safety alignment. We thus argue that safety must be treated as an explicit constraint or a core optimization objective during these stages, not merely subordinated to the pursuit of general capability. Third, we reveal a pronounced vulnerability: employing a CoT attack via a response prefix can elevate the attack success rate by 3.34x on average and from 0.6% to 96.3% for Seed-OSS-36B-Instruct. This critical finding underscores the safety risks inherent in text-completion interfaces and features that allow user-defined response prefixes in LLM services, highlighting an urgent need for architectural and deployment safeguards. Fourth, roleplay, prompt injection, and gradient-based search for adversarial prompts are the predominant methodologies for eliciting unaligned behaviors in modern models. 

---
# PartisanLens: A Multilingual Dataset of Hyperpartisan and Conspiratorial Immigration Narratives in European Media 

**Authors**: Michele Joshua Maggini, Paloma Piot, Anxo Pérez, Erik Bran Marino, Lúa Santamaría Montesinos, Ana Lisboa, Marta Vázquez Abuín, Javier Parapar, Pablo Gamallo  

**Link**: [PDF](https://arxiv.org/pdf/2601.03860)  

**Abstract**: Detecting hyperpartisan narratives and Population Replacement Conspiracy Theories (PRCT) is essential to addressing the spread of misinformation. These complex narratives pose a significant threat, as hyperpartisanship drives political polarisation and institutional distrust, while PRCTs directly motivate real-world extremist violence, making their identification critical for social cohesion and public safety. However, existing resources are scarce, predominantly English-centric, and often analyse hyperpartisanship, stance, and rhetorical bias in isolation rather than as interrelated aspects of political discourse. To bridge this gap, we introduce \textsc{PartisanLens}, the first multilingual dataset of \num{1617} hyperpartisan news headlines in Spanish, Italian, and Portuguese, annotated in multiple political discourse aspects. We first evaluate the classification performance of widely used Large Language Models (LLMs) on this dataset, establishing robust baselines for the classification of hyperpartisan and PRCT narratives. In addition, we assess the viability of using LLMs as automatic annotators for this task, analysing their ability to approximate human annotation. Results highlight both their potential and current limitations. Next, moving beyond standard judgments, we explore whether LLMs can emulate human annotation patterns by conditioning them on socio-economic and ideological profiles that simulate annotator perspectives. At last, we provide our resources and evaluation, \textsc{PartisanLens} supports future research on detecting partisan and conspiratorial narratives in European contexts. 

---
# What Does Loss Optimization Actually Teach, If Anything? Knowledge Dynamics in Continual Pre-training of LLMs 

**Authors**: Seyed Mahed Mousavi, Simone Alghisi, Giuseppe Riccardi  

**Link**: [PDF](https://arxiv.org/pdf/2601.03858)  

**Abstract**: Continual Pre-Training (CPT) is widely used for acquiring and updating factual knowledge in LLMs. This practice treats loss as a proxy for knowledge learning, while offering no grounding into how it changes during training. We study CPT as a knowledge learning process rather than a solely optimization problem. We construct a controlled, distribution-matched benchmark of factual documents and interleave diagnostic probes directly into the CPT loop, enabling epoch-level measurement of knowledge acquisition dynamics and changes in Out-Of-Domain (OOD) general skills (e.g., math). We further analyze how CPT reshapes knowledge circuits during training. Across three instruction-tuned LLMs and multiple CPT strategies, optimization and learning systematically diverge as loss decreases monotonically while factual learning is unstable and non-monotonic. Acquired facts are rarely consolidated, learning is strongly conditioned on prior exposure, and OOD performance degrades from early epochs. Circuit analysis reveals rapid reconfiguration of knowledge pathways across epochs, providing an explanation for narrow acquisition windows and systematic forgetting. These results show that loss optimization is misaligned with learning progress in CPT and motivate evaluation of stopping criteria based on task-level learning dynamics. 

---
# Rethinking Table Pruning in TableQA: From Sequential Revisions to Gold Trajectory-Supervised Parallel Search 

**Authors**: Yu Guo, Shenghao Ye, Shuangwu Chen, Zijian Wen, Tao Zhang, Qirui Bai, Dong Jin, Yunpeng Hou, Huasen He, Jian Yang, Xiaobin Tan  

**Link**: [PDF](https://arxiv.org/pdf/2601.03851)  

**Abstract**: Table Question Answering (TableQA) benefits significantly from table pruning, which extracts compact sub-tables by eliminating redundant cells to streamline downstream reasoning. However, existing pruning methods typically rely on sequential revisions driven by unreliable critique signals, often failing to detect the loss of answer-critical data. To address this limitation, we propose TabTrim, a novel table pruning framework which transforms table pruning from sequential revisions to gold trajectory-supervised parallel search. TabTrim derives a gold pruning trajectory using the intermediate sub-tables in the execution process of gold SQL queries, and trains a pruner and a verifier to make the step-wise pruning result align with the gold pruning trajectory. During inference, TabTrim performs parallel search to explore multiple candidate pruning trajectories and identify the optimal sub-table. Extensive experiments demonstrate that TabTrim achieves state-of-the-art performance across diverse tabular reasoning tasks: TabTrim-8B reaches 73.5% average accuracy, outperforming the strongest baseline by 3.2%, including 79.4% on WikiTQ and 61.2% on TableBench. 

---
# Step Potential Advantage Estimation: Harnessing Intermediate Confidence and Correctness for Efficient Mathematical Reasoning 

**Authors**: Fei Wu, Zhenrong Zhang, Qikai Chang, Jianshu Zhang, Quan Liu, Jun Du  

**Link**: [PDF](https://arxiv.org/pdf/2601.03823)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) elicits long chain-of-thought reasoning in large language models (LLMs), but outcome-based rewards lead to coarse-grained advantage estimation. While existing approaches improve RLVR via token-level entropy or sequence-level length control, they lack a semantically grounded, step-level measure of reasoning progress. As a result, LLMs fail to distinguish necessary deduction from redundant verification: they may continue checking after reaching a correct solution and, in extreme cases, overturn a correct trajectory into an incorrect final answer. To remedy the lack of process supervision, we introduce a training-free probing mechanism that extracts intermediate confidence and correctness and combines them into a Step Potential signal that explicitly estimates the reasoning state at each step. Building on this signal, we propose Step Potential Advantage Estimation (SPAE), a fine-grained credit assignment method that amplifies potential gains, penalizes potential drops, and applies penalty after potential saturates to encourage timely termination. Experiments across multiple benchmarks show SPAE consistently improves accuracy while substantially reducing response length, outperforming strong RL baselines and recent efficient reasoning and token-level advantage estimation methods. The code is available at this https URL. 

---
# AI Generated Text Detection 

**Authors**: Adilkhan Alikhanov, Aidar Amangeldi, Diar Demeubay, Dilnaz Akhmetzhan, Nurbek Moldakhmetov, Omar Polat, Galymzhan Zharas  

**Link**: [PDF](https://arxiv.org/pdf/2601.03812)  

**Abstract**: The rapid development of large language models has led to an increase in AI-generated text, with students increasingly using LLM-generated content as their own work, which violates academic integrity. This paper presents an evaluation of AI text detection methods, including both traditional machine learning models and transformer-based architectures. We utilize two datasets, HC3 and DAIGT v2, to build a unified benchmark and apply a topic-based data split to prevent information leakage. This approach ensures robust generalization across unseen domains. Our experiments show that TF-IDF logistic regression achieves a reasonable baseline accuracy of 82.87%. However, deep learning models outperform it. The BiLSTM classifier achieves an accuracy of 88.86%, while DistilBERT achieves a similar accuracy of 88.11% with the highest ROC-AUC score of 0.96, demonstrating the strongest overall performance. The results indicate that contextual semantic modeling is significantly superior to lexical features and highlight the importance of mitigating topic memorization through appropriate evaluation protocols. The limitations of this work are primarily related to dataset diversity and computational constraints. In future work, we plan to expand dataset diversity and utilize parameter-efficient fine-tuning methods such as LoRA. We also plan to explore smaller or distilled models and employ more efficient batching strategies and hardware-aware optimization. 

---
# Where meaning lives: Layer-wise accessibility of psycholinguistic features in encoder and decoder language models 

**Authors**: Taisiia Tikhomirova, Dirk U. Wulff  

**Link**: [PDF](https://arxiv.org/pdf/2601.03798)  

**Abstract**: Understanding where transformer language models encode psychologically meaningful aspects of meaning is essential for both theory and practice. We conduct a systematic layer-wise probing study of 58 psycholinguistic features across 10 transformer models, spanning encoder-only and decoder-only architectures, and compare three embedding extraction methods. We find that apparent localization of meaning is strongly method-dependent: contextualized embeddings yield higher feature-specific selectivity and different layer-wise profiles than isolated embeddings. Across models and methods, final-layer representations are rarely optimal for recovering psycholinguistic information with linear probes. Despite these differences, models exhibit a shared depth ordering of meaning dimensions, with lexical properties peaking earlier and experiential and affective dimensions peaking later. Together, these results show that where meaning "lives" in transformer models reflects an interaction between methodological choices and architectural constraints. 

---
# VietMed-MCQ: A Consistency-Filtered Data Synthesis Framework for Vietnamese Traditional Medicine Evaluation 

**Authors**: Huynh Trung Kiet, Dao Sy Duy Minh, Nguyen Dinh Ha Duong, Le Hoang Minh Huy, Long Nguyen, Dien Dinh  

**Link**: [PDF](https://arxiv.org/pdf/2601.03792)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable proficiency in general medical domains. However, their performance significantly degrades in specialized, culturally specific domains such as Vietnamese Traditional Medicine (VTM), primarily due to the scarcity of high-quality, structured benchmarks. In this paper, we introduce VietMed-MCQ, a novel multiple-choice question dataset generated via a Retrieval-Augmented Generation (RAG) pipeline with an automated consistency check mechanism. Unlike previous synthetic datasets, our framework incorporates a dual-model validation approach to ensure reasoning consistency through independent answer verification, though the substring-based evidence checking has known limitations. The complete dataset of 3,190 questions spans three difficulty levels and underwent validation by one medical expert and four students, achieving 94.2 percent approval with substantial inter-rater agreement (Fleiss' kappa = 0.82). We benchmark seven open-source models on VietMed-MCQ. Results reveal that general-purpose models with strong Chinese priors outperform Vietnamese-centric models, highlighting cross-lingual conceptual transfer, while all models still struggle with complex diagnostic reasoning. Our code and dataset are publicly available to foster research in low-resource medical domains. 

---
# Do LLMs Really Memorize Personally Identifiable Information? Revisiting PII Leakage with a Cue-Controlled Memorization Framework 

**Authors**: Xiaoyu Luo, Yiyi Chen, Qiongxiu Li, Johannes Bjerva  

**Link**: [PDF](https://arxiv.org/pdf/2601.03791)  

**Abstract**: Large Language Models (LLMs) have been reported to "leak" Personally Identifiable Information (PII), with successful PII reconstruction often interpreted as evidence of memorization. We propose a principled revision of memorization evaluation for LLMs, arguing that PII leakage should be evaluated under low lexical cue conditions, where target PII cannot be reconstructed through prompt-induced generalization or pattern completion. We formalize Cue-Resistant Memorization (CRM) as a cue-controlled evaluation framework and a necessary condition for valid memorization evaluation, explicitly conditioning on prompt-target overlap cues. Using CRM, we conduct a large-scale multilingual re-evaluation of PII leakage across 32 languages and multiple memorization paradigms. Revisiting reconstruction-based settings, including verbatim prefix-suffix completion and associative reconstruction, we find that their apparent effectiveness is driven primarily by direct surface-form cues rather than by true memorization. When such cues are controlled for, reconstruction success diminishes substantially. We further examine cue-free generation and membership inference, both of which exhibit extremely low true positive rates. Overall, our results suggest that previously reported PII leakage is better explained by cue-driven behavior than by genuine memorization, highlighting the importance of cue-controlled evaluation for reliably quantifying privacy-relevant memorization in LLMs. 

---
# NeoAMT: Neologism-Aware Agentic Machine Translation with Reinforcement Learning 

**Authors**: Zhongtao Miao, Kaiyan Zhao, Masaaki Nagata, Yoshimasa Tsuruoka  

**Link**: [PDF](https://arxiv.org/pdf/2601.03790)  

**Abstract**: Neologism-aware machine translation aims to translate source sentences containing neologisms into target languages. This field remains underexplored compared with general machine translation (MT). In this paper, we propose an agentic framework, NeoAMT, for neologism-aware machine translation using a Wiktionary search tool. Specifically, we first create a new dataset for neologism-aware machine translation and develop a search tool based on Wiktionary. The new dataset covers 16 languages and 75 translation directions and is derived from approximately 10 million records of an English Wiktionary dump. The retrieval corpus of the search tool is also constructed from around 3 million cleaned records of the Wiktionary dump. We then use it for training the translation agent with reinforcement learning (RL) and evaluating the accuracy of neologism-aware machine translation. Based on this, we also propose an RL training framework that contains a novel reward design and an adaptive rollout generation approach by leveraging "translation difficulty" to further improve the translation quality of translation agents using our search tool. 

---
# Compact Example-Based Explanations for Language Models 

**Authors**: Loris Schoenegger, Benjamin Roth  

**Link**: [PDF](https://arxiv.org/pdf/2601.03786)  

**Abstract**: Training data influence estimation methods quantify the contribution of training documents to a model's output, making them a promising source of information for example-based explanations. As humans cannot interpret thousands of documents, only a small subset of the training data can be presented as an explanation. Although the choice of which documents to include directly affects explanation quality, previous evaluations of such systems have largely ignored any selection strategies. To address this, we propose a novel selection relevance score, a retraining-free metric that quantifies how useful a set of examples is for explaining a model's output. We validate this score through fine-tuning experiments, confirming that it can predict whether a set of examples supports or undermines the model's predictions. Using this metric, we further show that common selection strategies often underperform random selection. Motivated by this finding, we propose a strategy that balances influence and representativeness, enabling better use of selection budgets than naively selecting the highest-ranking examples. 

---
# Membox: Weaving Topic Continuity into Long-Range Memory for LLM Agents 

**Authors**: Dehao Tao, Guoliang Ma, Yongfeng Huang, Minghu Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2601.03785)  

**Abstract**: Human-agent dialogues often exhibit topic continuity-a stable thematic frame that evolves through temporally adjacent exchanges-yet most large language model (LLM) agent memory systems fail to preserve it. Existing designs follow a fragmentation-compensation paradigm: they first break dialogue streams into isolated utterances for storage, then attempt to restore coherence via embedding-based retrieval. This process irreversibly damages narrative and causal flow, while biasing retrieval towards lexical similarity. We introduce membox, a hierarchical memory architecture centered on a Topic Loom that continuously monitors dialogue in a sliding-window fashion, grouping consecutive same-topic turns into coherent "memory boxes" at storage time. Sealed boxes are then linked by a Trace Weaver into long-range event-timeline traces, recovering macro-topic recurrences across discontinuities. Experiments on LoCoMo demonstrate that Membox achieves up to 68% F1 improvement on temporal reasoning tasks, outperforming competitive baselines (e.g., Mem0, A-MEM). Notably, Membox attains these gains while using only a fraction of the context tokens required by existing methods, highlighting a superior balance between efficiency and effectiveness. By explicitly modeling topic continuity, Membox offers a cognitively motivated mechanism for enhancing both coherence and efficiency in LLM agents. 

---
# HearSay Benchmark: Do Audio LLMs Leak What They Hear? 

**Authors**: Jin Wang, Liang Lin, Kaiwen Luo, Weiliu Wang, Yitian Chen, Moayad Aloqaily, Xuehai Tang, Zhenhong Zhou, Kun Wang, Li Sun, Qingsong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2601.03783)  

**Abstract**: While Audio Large Language Models (ALLMs) have achieved remarkable progress in understanding and generation, their potential privacy implications remain largely unexplored. This paper takes the first step to investigate whether ALLMs inadvertently leak user privacy solely through acoustic voiceprints and introduces $\textit{HearSay}$, a comprehensive benchmark constructed from over 22,000 real-world audio clips. To ensure data quality, the benchmark is meticulously curated through a rigorous pipeline involving automated profiling and human verification, guaranteeing that all privacy labels are grounded in factual records. Extensive experiments on $\textit{HearSay}$ yield three critical findings: $\textbf{Significant Privacy Leakage}$: ALLMs inherently extract private attributes from voiceprints, reaching 92.89% accuracy on gender and effectively profiling social attributes. $\textbf{Insufficient Safety Mechanisms}$: Alarmingly, existing safeguards are severely inadequate; most models fail to refuse privacy-intruding requests, exhibiting near-zero refusal rates for physiological traits. $\textbf{Reasoning Amplifies Risk}$: Chain-of-Thought (CoT) reasoning exacerbates privacy risks in capable models by uncovering deeper acoustic correlations. These findings expose critical vulnerabilities in ALLMs, underscoring the urgent need for targeted privacy alignment. The codes and dataset are available at this https URL 

---
# Tracing the complexity profiles of different linguistic phenomena through the intrinsic dimension of LLM representations 

**Authors**: Marco Baroni, Emily Cheng, Iria deDios-Flores, Francesca Franzon  

**Link**: [PDF](https://arxiv.org/pdf/2601.03779)  

**Abstract**: We explore the intrinsic dimension (ID) of LLM representations as a marker of linguistic complexity, asking if different ID profiles across LLM layers differentially characterize formal and functional complexity. We find the formal contrast between sentences with multiple coordinated or subordinated clauses to be reflected in ID differences whose onset aligns with a phase of more abstract linguistic processing independently identified in earlier work. The functional contrasts between sentences characterized by right branching vs. center embedding or unambiguous vs. ambiguous relative clause attachment are also picked up by ID, but in a less marked way, and they do not correlate with the same processing phase. Further experiments using representational similarity and layer ablation confirm the same trends. We conclude that ID is a useful marker of linguistic complexity in LLMs, that it allows to differentiate between different types of complexity, and that it points to similar stages of linguistic processing across disparate LLMs. 

---
# Do LLM Self-Explanations Help Users Predict Model Behavior? Evaluating Counterfactual Simulatability with Pragmatic Perturbations 

**Authors**: Pingjun Hong, Benjamin Roth  

**Link**: [PDF](https://arxiv.org/pdf/2601.03775)  

**Abstract**: Large Language Models (LLMs) can produce verbalized self-explanations, yet prior studies suggest that such rationales may not reliably reflect the model's true decision process. We ask whether these explanations nevertheless help users predict model behavior, operationalized as counterfactual simulatability. Using StrategyQA, we evaluate how well humans and LLM judges can predict a model's answers to counterfactual follow-up questions, with and without access to the model's chain-of-thought or post-hoc explanations. We compare LLM-generated counterfactuals with pragmatics-based perturbations as alternative ways to construct test cases for assessing the potential usefulness of explanations. Our results show that self-explanations consistently improve simulation accuracy for both LLM judges and humans, but the degree and stability of gains depend strongly on the perturbation strategy and judge strength. We also conduct a qualitative analysis of free-text justifications written by human users when predicting the model's behavior, which provides evidence that access to explanations helps humans form more accurate predictions on the perturbed questions. 

---
# Evaluation of Multilingual LLMs Personalized Text Generation Capabilities Targeting Groups and Social-Media Platforms 

**Authors**: Dominik Macko  

**Link**: [PDF](https://arxiv.org/pdf/2601.03752)  

**Abstract**: Capabilities of large language models to generate multilingual coherent text have continuously enhanced in recent years, which opens concerns about their potential misuse. Previous research has shown that they can be misused for generation of personalized disinformation in multiple languages. It has also been observed that personalization negatively affects detectability of machine-generated texts; however, this has been studied in the English language only. In this work, we examine this phenomenon across 10 languages, while we focus not only on potential misuse of personalization capabilities, but also on potential benefits they offer. Overall, we cover 1080 combinations of various personalization aspects in the prompts, for which the texts are generated by 16 distinct language models (17,280 texts in total). Our results indicate that there are differences in personalization quality of the generated texts when targeting demographic groups and when targeting social-media platforms across languages. Personalization towards platforms affects detectability of the generated texts in a higher scale, especially in English, where the personalization quality is the highest. 

---
# Whose Facts Win? LLM Source Preferences under Knowledge Conflicts 

**Authors**: Jakob Schuster, Vagrant Gautam, Katja Markert  

**Link**: [PDF](https://arxiv.org/pdf/2601.03746)  

**Abstract**: As large language models (LLMs) are more frequently used in retrieval-augmented generation pipelines, it is increasingly relevant to study their behavior under knowledge conflicts. Thus far, the role of the source of the retrieved information has gone unexamined. We address this gap with a novel framework to investigate how source preferences affect LLM resolution of inter-context knowledge conflicts in English, motivated by interdisciplinary research on credibility. With a comprehensive, tightly-controlled evaluation of 13 open-weight LLMs, we find that LLMs prefer institutionally-corroborated information (e.g., government or newspaper sources) over information from people and social media. However, these source preferences can be reversed by simply repeating information from less credible sources. To mitigate repetition effects and maintain consistent preferences, we propose a novel method that reduces repetition bias by up to 99.8%, while also maintaining at least 88.8% of original preferences. We release all data and code to encourage future work on credibility and source preferences in knowledge-intensive NLP. 

---
# O-Researcher: An Open Ended Deep Research Model via Multi-Agent Distillation and Agentic RL 

**Authors**: Yi Yao, He Zhu, Piaohong Wang, Jincheng Ren, Xinlong Yang, Qianben Chen, Xiaowan Li, Dingfeng Shi, Jiaxian Li, Qiexiang Wang, Sinuo Wang, Xinpeng Liu, Jiaqi Wu, Minghao Liu, Wangchunshu Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2601.03743)  

**Abstract**: The performance gap between closed-source and open-source large language models (LLMs) is largely attributed to disparities in access to high-quality training data. To bridge this gap, we introduce a novel framework for the automated synthesis of sophisticated, research-grade instructional data. Our approach centers on a multi-agent workflow where collaborative AI agents simulate complex tool-integrated reasoning to generate diverse and high-fidelity data end-to-end. Leveraging this synthesized data, we develop a two-stage training strategy that integrates supervised fine-tuning with a novel reinforcement learning method, designed to maximize model alignment and capability. Extensive experiments demonstrate that our framework empowers open-source models across multiple scales, enabling them to achieve new state-of-the-art performance on the major deep research benchmark. This work provides a scalable and effective pathway for advancing open-source LLMs without relying on proprietary data or models. 

---
# Stuttering-Aware Automatic Speech Recognition for Indonesian Language 

**Authors**: Fadhil Muhammad, Alwin Djuliansah, Adrian Aryaputra Hamzah, Kurniawati Azizah  

**Link**: [PDF](https://arxiv.org/pdf/2601.03727)  

**Abstract**: Automatic speech recognition systems have achieved remarkable performance on fluent speech but continue to degrade significantly when processing stuttered speech, a limitation that is particularly acute for low-resource languages like Indonesian where specialized datasets are virtually non-existent. To overcome this scarcity, we propose a data augmentation framework that generates synthetic stuttered audio by injecting repetitions and prolongations into fluent text through a combination of rule-based transformations and large language models followed by text-to-speech synthesis. We apply this synthetic data to fine-tune a pre-trained Indonesian Whisper model using transfer learning, enabling the architecture to adapt to dysfluent acoustic patterns without requiring large-scale real-world recordings. Our experiments demonstrate that this targeted synthetic exposure consistently reduces recognition errors on stuttered speech while maintaining performance on fluent segments, validating the utility of synthetic data pipelines for developing more inclusive speech technologies in under-represented languages. 

---
# MIND: From Passive Mimicry to Active Reasoning through Capability-Aware Multi-Perspective CoT Distillation 

**Authors**: Jin Cui, Jiaqi Guo, Jiepeng Zhou, Ruixuan Yang, Jiayi Lu, Jiajun Xu, Jiangcheng Song, Boran Zhao, Pengju Ren  

**Link**: [PDF](https://arxiv.org/pdf/2601.03717)  

**Abstract**: While Large Language Models (LLMs) have emerged with remarkable capabilities in complex tasks through Chain-of-Thought reasoning, practical resource constraints have sparked interest in transferring these abilities to smaller models. However, achieving both domain performance and cross-domain generalization remains challenging. Existing approaches typically restrict students to following a single golden rationale and treat different reasoning paths independently. Due to distinct inductive biases and intrinsic preferences, alongside the student's evolving capacity and reasoning preferences during training, a teacher's "optimal" rationale could act as out-of-distribution noise. This misalignment leads to a degeneration of the student's latent reasoning distribution, causing suboptimal performance. To bridge this gap, we propose MIND, a capability-adaptive framework that transitions distillation from passive mimicry to active cognitive construction. We synthesize diverse teacher perspectives through a novel "Teaching Assistant" network. By employing a Feedback-Driven Inertia Calibration mechanism, this network utilizes inertia-filtered training loss to align supervision with the student's current adaptability, effectively enhancing performance while mitigating catastrophic forgetting. Extensive experiments demonstrate that MIND achieves state-of-the-art performance on both in-distribution and out-of-distribution benchmarks, and our sophisticated latent space analysis further confirms the mechanism of reasoning ability internalization. 

---
# Visual Merit or Linguistic Crutch? A Close Look at DeepSeek-OCR 

**Authors**: Yunhao Liang, Ruixuan Ying, Bo Li, Hong Li, Kai Yan, Qingwen Li, Min Yang, Okamoto Satoshi, Zhe Cui, Shiwen Ni  

**Link**: [PDF](https://arxiv.org/pdf/2601.03714)  

**Abstract**: DeepSeek-OCR utilizes an optical 2D mapping approach to achieve high-ratio vision-text compression, claiming to decode text tokens exceeding ten times the input visual tokens. While this suggests a promising solution for the LLM long-context bottleneck, we investigate a critical question: "Visual merit or linguistic crutch - which drives DeepSeek-OCR's performance?" By employing sentence-level and word-level semantic corruption, we isolate the model's intrinsic OCR capabilities from its language priors. Results demonstrate that without linguistic support, DeepSeek-OCR's performance plummets from approximately 90% to 20%. Comparative benchmarking against 13 baseline models reveals that traditional pipeline OCR methods exhibit significantly higher robustness to such semantic perturbations than end-to-end methods. Furthermore, we find that lower visual token counts correlate with increased reliance on priors, exacerbating hallucination risks. Context stress testing also reveals a total model collapse around 10,000 text tokens, suggesting that current optical compression techniques may paradoxically aggravate the long-context bottleneck. This study empirically defines DeepSeek-OCR's capability boundaries and offers essential insights for future optimizations of the vision-text compression paradigm. We release all data, results and scripts used in this study at this https URL. 

---
# AirNav: A Large-Scale Real-World UAV Vision-and-Language Navigation Dataset with Natural and Diverse Instructions 

**Authors**: Hengxing Cai, Yijie Rao, Ligang Huang, Zanyang Zhong, Jinhan Dong, Jingjun Tan, Wenhao Lu, Renxin Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2601.03707)  

**Abstract**: Existing Unmanned Aerial Vehicle (UAV) Vision-Language Navigation (VLN) datasets face issues such as dependence on virtual environments, lack of naturalness in instructions, and limited scale. To address these challenges, we propose AirNav, a large-scale UAV VLN benchmark constructed from real urban aerial data, rather than synthetic environments, with natural and diverse instructions. Additionally, we introduce the AirVLN-R1, which combines Supervised Fine-Tuning and Reinforcement Fine-Tuning to enhance performance and generalization. The feasibility of the model is preliminarily evaluated through real-world tests. Our dataset and code are publicly available. 

---
# ADEPT: Adaptive Dynamic Early-Exit Process for Transformers 

**Authors**: Sangmin Yoo, Srikanth Malla, Chiho Choi, Wei D. Lu, Joon Hee Choi  

**Link**: [PDF](https://arxiv.org/pdf/2601.03700)  

**Abstract**: The inference of large language models imposes significant computational workloads, often requiring the processing of billions of parameters. Although early-exit strategies have proven effective in reducing computational demands by halting inference earlier, they apply either to only the first token in the generation phase or at the prompt level in the prefill phase. Thus, the Key-Value (KV) cache for skipped layers remains a bottleneck for subsequent token generation, limiting the benefits of early exit. We introduce ADEPT (Adaptive Dynamic Early-exit Process for Transformers), a novel approach designed to overcome this issue and enable dynamic early exit in both the prefill and generation phases. The proposed adaptive token-level early-exit mechanism adjusts computation dynamically based on token complexity, optimizing efficiency without compromising performance. ADEPT further enhances KV generation procedure by decoupling sequential dependencies in skipped layers, making token-level early exit more practical. Experimental results demonstrate that ADEPT improves efficiency by up to 25% in language generation tasks and achieves a 4x speed-up in downstream classification tasks, with up to a 45% improvement in performance. 

---
# RedBench: A Universal Dataset for Comprehensive Red Teaming of Large Language Models 

**Authors**: Quy-Anh Dang, Chris Ngo, Truong-Son Hy  

**Link**: [PDF](https://arxiv.org/pdf/2601.03699)  

**Abstract**: As large language models (LLMs) become integral to safety-critical applications, ensuring their robustness against adversarial prompts is paramount. However, existing red teaming datasets suffer from inconsistent risk categorizations, limited domain coverage, and outdated evaluations, hindering systematic vulnerability assessments. To address these challenges, we introduce RedBench, a universal dataset aggregating 37 benchmark datasets from leading conferences and repositories, comprising 29,362 samples across attack and refusal prompts. RedBench employs a standardized taxonomy with 22 risk categories and 19 domains, enabling consistent and comprehensive evaluations of LLM vulnerabilities. We provide a detailed analysis of existing datasets, establish baselines for modern LLMs, and open-source the dataset and evaluation code. Our contributions facilitate robust comparisons, foster future research, and promote the development of secure and reliable LLMs for real-world deployment. Code: this https URL 

---
# Evaluation Framework for AI Creativity: A Case Study Based on Story Generation 

**Authors**: Pharath Sathya, Yin Jou Huang, Fei Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2601.03698)  

**Abstract**: Evaluating creative text generation remains a challenge because existing reference-based metrics fail to capture the subjective nature of creativity. We propose a structured evaluation framework for AI story generation comprising four components (Novelty, Value, Adherence, and Resonance) and eleven sub-components. Using controlled story generation via ``Spike Prompting'' and a crowdsourced study of 115 readers, we examine how different creative components shape both immediate and reflective human creativity judgments. Our findings show that creativity is evaluated hierarchically rather than cumulatively, with different dimensions becoming salient at different stages of judgment, and that reflective evaluation substantially alters both ratings and inter-rater agreement. Together, these results support the effectiveness of our framework in revealing dimensions of creativity that are obscured by reference-based evaluation. 

---
# From Implicit to Explicit: Token-Efficient Logical Supervision for Mathematical Reasoning in LLMs 

**Authors**: Shaojie Wang, Liang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.03682)  

**Abstract**: Recent studies reveal that large language models (LLMs) exhibit limited logical reasoning abilities in mathematical problem-solving, instead often relying on pattern-matching and memorization. We systematically analyze this limitation, focusing on logical relationship understanding, which is a core capability underlying genuine logical reasoning, and reveal that errors related to this capability account for over 90\% of incorrect predictions, with Chain-of-Thought Supervised Fine-Tuning (CoT-SFT) failing to substantially reduce these errors. To address this bottleneck, we propose First-Step Logical Reasoning (FSLR), a lightweight training framework targeting logical relationship understanding. Our key insight is that the first planning step-identifying which variables to use and which operation to apply-encourages the model to derive logical relationships directly from the problem statement. By training models on this isolated step, FSLR provides explicit supervision for logical relationship understanding, unlike CoT-SFT which implicitly embeds such relationships within complete solution trajectories. Extensive experiments across multiple models and datasets demonstrate that FSLR consistently outperforms CoT-SFT under both in-distribution and out-of-distribution settings, with average improvements of 3.2\% and 4.6\%, respectively. Moreover, FSLR achieves 4-6x faster training and reduces training token consumption by over 80\%. 

---
# Towards Compositional Generalization of LLMs via Skill Taxonomy Guided Data Synthesis 

**Authors**: Yifan Wei, Li Du, Xiaoyan Yu, Yang Feng, Angsheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.03676)  

**Abstract**: Large Language Models (LLMs) and agent-based systems often struggle with compositional generalization due to a data bottleneck in which complex skill combinations follow a long-tailed, power-law distribution, limiting both instruction-following performance and generalization in agent-centric tasks. To address this challenge, we propose STEPS, a Skill Taxonomy guided Entropy-based Post-training data Synthesis framework for generating compositionally challenging data. STEPS explicitly targets compositional generalization by uncovering latent relationships among skills and organizing them into an interpretable, hierarchical skill taxonomy using structural information theory. Building on this taxonomy, we formulate data synthesis as a constrained information maximization problem, selecting skill combinations that maximize marginal structural information within the hierarchy while preserving semantic coherence. Experiments on challenging instruction-following benchmarks show that STEPS outperforms existing data synthesis baselines, while also yielding improved compositional generalization in downstream agent-based evaluations. 

---
# NeuronScope: A Multi-Agent Framework for Explaining Polysemantic Neurons in Language Models 

**Authors**: Weiqi Liu, Yongliang Miao, Haiyan Zhao, Yanguang Liu, Mengnan Du  

**Link**: [PDF](https://arxiv.org/pdf/2601.03671)  

**Abstract**: Neuron-level interpretation in large language models (LLMs) is fundamentally challenged by widespread polysemanticity, where individual neurons respond to multiple distinct semantic concepts. Existing single-pass interpretation methods struggle to faithfully capture such multi-concept behavior. In this work, we propose NeuronScope, a multi-agent framework that reformulates neuron interpretation as an iterative, activation-guided process. NeuronScope explicitly deconstructs neuron activations into atomic semantic components, clusters them into distinct semantic modes, and iteratively refines each explanation using neuron activation feedback. Experiments demonstrate that NeuronScope uncovers hidden polysemanticity and produces explanations with significantly higher activation correlation compared to single-pass baselines. 

---
# DisastQA: A Comprehensive Benchmark for Evaluating Question Answering in Disaster Management 

**Authors**: Zhitong Chen, Kai Yin, Xiangjue Dong, Chengkai Liu, Xiangpeng Li, Yiming Xiao, Bo Li, Junwei Ma, Ali Mostafavi, James Caverlee  

**Link**: [PDF](https://arxiv.org/pdf/2601.03670)  

**Abstract**: Accurate question answering (QA) in disaster management requires reasoning over uncertain and conflicting information, a setting poorly captured by existing benchmarks built on clean evidence. We introduce DisastQA, a large-scale benchmark of 3,000 rigorously verified questions (2,000 multiple-choice and 1,000 open-ended) spanning eight disaster types. The benchmark is constructed via a human-LLM collaboration pipeline with stratified sampling to ensure balanced coverage. Models are evaluated under varying evidence conditions, from closed-book to noisy evidence integration, enabling separation of internal knowledge from reasoning under imperfect information. For open-ended QA, we propose a human-verified keypoint-based evaluation protocol emphasizing factual completeness over verbosity. Experiments with 20 models reveal substantial divergences from general-purpose leaderboards such as MMLU-Pro. While recent open-weight models approach proprietary systems in clean settings, performance degrades sharply under realistic noise, exposing critical reliability gaps for disaster response. All code, data, and evaluation resources are available at this https URL. 

---
# eTracer: Towards Traceable Text Generation via Claim-Level Grounding 

**Authors**: Bohao Chu, Qianli Wang, Hendrik Damm, Hui Wang, Ula Muhabbek, Elisabeth Livingstone, Christoph M. Friedrich, Norbert Fuhr  

**Link**: [PDF](https://arxiv.org/pdf/2601.03669)  

**Abstract**: How can system-generated responses be efficiently verified, especially in the high-stakes biomedical domain? To address this challenge, we introduce eTracer, a plug-and-play framework that enables traceable text generation by grounding claims against contextual evidence. Through post-hoc grounding, each response claim is aligned with contextual evidence that either supports or contradicts it. Building on claim-level grounding results, eTracer not only enables users to precisely trace responses back to their contextual source but also quantifies response faithfulness, thereby enabling the verifiability and trustworthiness of generated responses. Experiments show that our claim-level grounding approach alleviates the limitations of conventional grounding methods in aligning generated statements with contextual sentence-level evidence, resulting in substantial improvements in overall grounding quality and user verification efficiency. The code and data are available at this https URL. 

---
# e5-omni: Explicit Cross-modal Alignment for Omni-modal Embeddings 

**Authors**: Haonan Chen, Sicheng Gao, Radu Timofte, Tetsuya Sakai, Zhicheng Dou  

**Link**: [PDF](https://arxiv.org/pdf/2601.03666)  

**Abstract**: Modern information systems often involve different types of items, e.g., a text query, an image, a video clip, or an audio segment. This motivates omni-modal embedding models that map heterogeneous modalities into a shared space for direct comparison. However, most recent omni-modal embeddings still rely heavily on implicit alignment inherited from pretrained vision-language model (VLM) backbones. In practice, this causes three common issues: (i) similarity logits have modality-dependent sharpness, so scores are not on a consistent scale; (ii) in-batch negatives become less effective over time because mixed-modality batches create an imbalanced hardness distribution; as a result, many negatives quickly become trivial and contribute little gradient; and (iii) embeddings across modalities show mismatched first- and second-order statistics, which makes rankings less stable. To tackle these problems, we propose e5-omni, a lightweight explicit alignment recipe that adapts off-the-shelf VLMs into robust omni-modal embedding models. e5-omni combines three simple components: (1) modality-aware temperature calibration to align similarity scales, (2) a controllable negative curriculum with debiasing to focus on confusing negatives while reducing the impact of false negatives, and (3) batch whitening with covariance regularization to better match cross-modal geometry in the shared embedding space. Experiments on MMEB-V2 and AudioCaps show consistent gains over strong bi-modal and omni-modal baselines, and the same recipe also transfers well to other VLM backbones. We release our model checkpoint at this https URL. 

---
# SyncThink: A Training-Free Strategy to Align Inference Termination with Reasoning Saturation 

**Authors**: Gengyang Li, Wang Cai, Yifeng Gao, Yunfang Wu  

**Link**: [PDF](https://arxiv.org/pdf/2601.03649)  

**Abstract**: Chain-of-Thought (CoT) prompting improves reasoning but often produces long and redundant traces that substantially increase inference cost. We present SyncThink, a training-free and plug-and-play decoding method that reduces CoT overhead without modifying model weights. We find that answer tokens attend weakly to early reasoning and instead focus on the special token "/think", indicating an information bottleneck. Building on this observation, SyncThink monitors the model's own reasoning-transition signal and terminates reasoning. Experiments on GSM8K, MMLU, GPQA, and BBH across three DeepSeek-R1 distilled models show that SyncThink achieves 62.00 percent average Top-1 accuracy using 656 generated tokens and 28.68 s latency, compared to 61.22 percent, 2141 tokens, and 92.01 s for full CoT decoding. On long-horizon tasks such as GPQA, SyncThink can further yield up to +8.1 absolute accuracy by preventing over-thinking. 

---
# ELO: Efficient Layer-Specific Optimization for Continual Pretraining of Multilingual LLMs 

**Authors**: HanGyeol Yoo, ChangSu Choi, Minjun Kim, Seohyun Song, SeungWoo Song, Inho Won, Jongyoul Park, Cheoneum Park, KyungTae Lim  

**Link**: [PDF](https://arxiv.org/pdf/2601.03648)  

**Abstract**: We propose an efficient layer-specific optimization (ELO) method designed to enhance continual pretraining (CP) for specific languages in multilingual large language models (MLLMs). This approach addresses the common challenges of high computational cost and degradation of source language performance associated with traditional CP. The ELO method consists of two main stages: (1) ELO Pretraining, where a small subset of specific layers, identified in our experiments as the critically important first and last layers, are detached from the original MLLM and trained with the target language. This significantly reduces not only the number of trainable parameters but also the total parameters computed during the forward pass, minimizing GPU memory consumption and accelerating the training process. (2) Layer Alignment, where the newly trained layers are reintegrated into the original model, followed by a brief full fine-tuning step on a small dataset to align the parameters. Experimental results demonstrate that the ELO method achieves a training speedup of up to 6.46 times compared to existing methods, while improving target language performance by up to 6.2\% on qualitative benchmarks and effectively preserving source language (English) capabilities. 

---
# LLM-MC-Affect: LLM-Based Monte Carlo Modeling of Affective Trajectories and Latent Ambiguity for Interpersonal Dynamic Insight 

**Authors**: Yu-Zheng Lin, Bono Po-Jen Shih, John Paul Martin Encinas, Elizabeth Victoria Abraham Achom, Karan Himanshu Patel, Jesus Horacio Pacheco, Sicong Shao, Jyotikrishna Dass, Soheil Salehi, Pratik Satam  

**Link**: [PDF](https://arxiv.org/pdf/2601.03645)  

**Abstract**: Emotional coordination is a core property of human interaction that shapes how relational meaning is constructed in real time. While text-based affect inference has become increasingly feasible, prior approaches often treat sentiment as a deterministic point estimate for individual speakers, failing to capture the inherent subjectivity, latent ambiguity, and sequential coupling found in mutual exchanges. We introduce LLM-MC-Affect, a probabilistic framework that characterizes emotion not as a static label, but as a continuous latent probability distribution defined over an affective space. By leveraging stochastic LLM decoding and Monte Carlo estimation, the methodology approximates these distributions to derive high-fidelity sentiment trajectories that explicitly quantify both central affective tendencies and perceptual ambiguity. These trajectories enable a structured analysis of interpersonal coupling through sequential cross-correlation and slope-based indicators, identifying leading or lagging influences between interlocutors. To validate the interpretive capacity of this approach, we utilize teacher-student instructional dialogues as a representative case study, where our quantitative indicators successfully distill high-level interaction insights such as effective scaffolding. This work establishes a scalable and deployable pathway for understanding interpersonal dynamics, offering a generalizable solution that extends beyond education to broader social and behavioral research. 

---
# Agent-Dice: Disentangling Knowledge Updates via Geometric Consensus for Agent Continual Learning 

**Authors**: Zheng Wu, Xingyu Lou, Xinbei Ma, Yansi Li, Weiwen Liu, Weinan Zhang, Jun Wang, Zhuosheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.03641)  

**Abstract**: Large Language Model (LLM)-based agents significantly extend the utility of LLMs by interacting with dynamic environments. However, enabling agents to continually learn new tasks without catastrophic forgetting remains a critical challenge, known as the stability-plasticity dilemma. In this work, we argue that this dilemma fundamentally arises from the failure to explicitly distinguish between common knowledge shared across tasks and conflicting knowledge introduced by task-specific interference. To address this, we propose Agent-Dice, a parameter fusion framework based on directional consensus evaluation. Concretely, Agent-Dice disentangles knowledge updates through a two-stage process: geometric consensus filtering to prune conflicting gradients, and curvature-based importance weighting to amplify shared semantics. We provide a rigorous theoretical analysis that establishes the validity of the proposed fusion scheme and offers insight into the origins of the stability-plasticity dilemma. Extensive experiments on GUI agents and tool-use agent domains demonstrate that Agent-Dice exhibits outstanding continual learning performance with minimal computational overhead and parameter updates. 

---
# Reasoning Model Is Superior LLM-Judge, Yet Suffers from Biases 

**Authors**: Hui Huang, Xuanxin Wu, Muyun Yang, Yuki Arase  

**Link**: [PDF](https://arxiv.org/pdf/2601.03630)  

**Abstract**: This paper presents the first systematic comparison investigating whether Large Reasoning Models (LRMs) are superior judge to non-reasoning LLMs. Our empirical analysis yields four key findings: 1) LRMs outperform non-reasoning LLMs in terms of judgment accuracy, particularly on reasoning-intensive tasks; 2) LRMs demonstrate superior instruction-following capabilities in evaluation contexts; 3) LRMs exhibit enhanced robustness against adversarial attacks targeting judgment tasks; 4) However, LRMs still exhibit strong biases in superficial quality. To improve the robustness against biases, we propose PlanJudge, an evaluation strategy that prompts the model to generate an explicit evaluation plan before execution. Despite its simplicity, our experiments demonstrate that PlanJudge significantly mitigates biases in both LRMs and standard LLMs. 

---
# Evaluating the Pre-Consultation Ability of LLMs using Diagnostic Guidelines 

**Authors**: Jean Seo, Gibaeg Kim, Kihun Shin, Seungseop Lim, Hyunkyung Lee, Wooseok Han, Jongwon Lee, Eunho Yang  

**Link**: [PDF](https://arxiv.org/pdf/2601.03627)  

**Abstract**: We introduce EPAG, a benchmark dataset and framework designed for Evaluating the Pre-consultation Ability of LLMs using diagnostic Guidelines. LLMs are evaluated directly through HPI-diagnostic guideline comparison and indirectly through disease diagnosis. In our experiments, we observe that small open-source models fine-tuned with a well-curated, task-specific dataset can outperform frontier LLMs in pre-consultation. Additionally, we find that increased amount of HPI (History of Present Illness) does not necessarily lead to improved diagnostic performance. Further experiments reveal that the language of pre-consultation influences the characteristics of the dialogue. By open-sourcing our dataset and evaluation pipeline on this https URL, we aim to contribute to the evaluation and further development of LLM applications in real-world clinical settings. 

---
# Analyzing Reasoning Shifts in Audio Deepfake Detection under Adversarial Attacks: The Reasoning Tax versus Shield Bifurcation 

**Authors**: Binh Nguyen, Thai Le  

**Link**: [PDF](https://arxiv.org/pdf/2601.03615)  

**Abstract**: Audio Language Models (ALMs) offer a promising shift towards explainable audio deepfake detections (ADDs), moving beyond \textit{black-box} classifiers by providing some level of transparency into their predictions via reasoning traces. This necessitates a new class of model robustness analysis: robustness of the predictive reasoning under adversarial attacks, which goes beyond existing paradigm that mainly focuses on the shifts of the final predictions (e.g., fake v.s. real). To analyze such reasoning shifts, we introduce a forensic auditing framework to evaluate the robustness of ALMs' reasoning under adversarial attacks in three inter-connected dimensions: acoustic perception, cognitive coherence, and cognitive dissonance. Our systematic analysis reveals that explicit reasoning does not universally enhance robustness. Instead, we observe a bifurcation: for models exhibiting robust acoustic perception, reasoning acts as a defensive \textit{``shield''}, protecting them from adversarial attacks. However, for others, it imposes a performance \textit{``tax''}, particularly under linguistic attacks which reduce cognitive coherence and increase attack success rate. Crucially, even when classification fails, high cognitive dissonance can serve as a \textit{silent alarm}, flagging potential manipulation. Overall, this work provides a critical evaluation of the role of reasoning in forensic audio deepfake analysis and its vulnerabilities. 

---
# DiVA: Fine-grained Factuality Verification with Agentic-Discriminative Verifier 

**Authors**: Hui Huang, Muyun Yang, Yuki Arase  

**Link**: [PDF](https://arxiv.org/pdf/2601.03605)  

**Abstract**: Despite the significant advancements of Large Language Models (LLMs), their factuality remains a critical challenge, fueling growing interest in factuality verification. Existing research on factuality verification primarily conducts binary judgments (e.g., correct or incorrect), which fails to distinguish varying degrees of error severity. This limits its utility for applications such as fine-grained evaluation and preference optimization. To bridge this gap, we propose the Agentic Discriminative Verifier (DiVA), a hybrid framework that synergizes the agentic search capabilities of generative models with the precise scoring aptitude of discriminative models. We also construct a new benchmark, FGVeriBench, as a robust testbed for fine-grained factuality verification. Experimental results on FGVeriBench demonstrate that our DiVA significantly outperforms existing methods on factuality verification for both general and multi-hop questions. 

---
# From Chains to Graphs: Self-Structured Reasoning for General-Domain LLMs 

**Authors**: Yingjian Chen, Haoran Liu, Yinhong Liu, Sherry T. Tong, Aosong Feng, Jinghui Lu, Juntao Zhang, Yusuke Iwasawa, Yutaka Matsuo, Irene Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.03597)  

**Abstract**: Large Language Models (LLMs) show strong reasoning ability in open-domain question answering, yet their reasoning processes are typically linear and often logically inconsistent. In contrast, real-world reasoning requires integrating multiple premises and solving subproblems in parallel. Existing methods, such as Chain-of-Thought (CoT), express reasoning in a linear textual form, which may appear coherent but frequently leads to inconsistent conclusions. Recent approaches rely on externally provided graphs and do not explore how LLMs can construct and use their own graph-structured reasoning, particularly in open-domain QA. To fill this gap, we novelly explore graph-structured reasoning of LLMs in general-domain question answering. We propose Self-Graph Reasoning (SGR), a framework that enables LLMs to explicitly represent their reasoning process as a structured graph before producing the final answer. We further construct a graph-structured reasoning dataset that merges multiple candidate reasoning graphs into refined graph structures for model training. Experiments on five QA benchmarks across both general and specialized domains show that SGR consistently improves reasoning consistency and yields a 17.74% gain over the base model. The LLaMA-3.3-70B model fine-tuned with SGR performs comparably to GPT-4o and surpasses Claude-3.5-Haiku, demonstrating the effectiveness of graph-structured reasoning. 

---
# OLA: Output Language Alignment in Code-Switched LLM Interactions 

**Authors**: Juhyun Oh, Haneul Yoo, Faiz Ghifari Haznitrama, Alice Oh  

**Link**: [PDF](https://arxiv.org/pdf/2601.03589)  

**Abstract**: Code-switching, alternating between languages within a conversation, is natural for multilingual users, yet poses fundamental challenges for large language models (LLMs). When a user code-switches in their prompt to an LLM, they typically do not specify the expected language of the LLM response, and thus LLMs must infer the output language from contextual and pragmatic cues. We find that current LLMs systematically fail to align with this expectation, responding in undesired languages even when cues are clear to humans. We introduce OLA, a benchmark to evaluate LLMs' Output Language Alignment in code-switched interactions. OLA focuses on Korean--English code-switching and spans simple intra-sentential mixing to instruction-content mismatches. Even frontier models frequently misinterpret implicit language expectation, exhibiting a bias toward non-English responses. We further show this bias generalizes beyond Korean to Chinese and Indonesian pairs. Models also show instability through mid-response switching and language intrusions. Chain-of-Thought prompting fails to resolve these errors, indicating weak pragmatic reasoning about output language. However, Code-Switching Aware DPO with minimal data (about 1K examples) substantially reduces misalignment, suggesting these failures stem from insufficient alignment rather than fundamental limitations. Our results highlight the need to align multilingual LLMs with users' implicit expectations in real-world code-switched interactions. 

---
# PsychEthicsBench: Evaluating Large Language Models Against Australian Mental Health Ethics 

**Authors**: Yaling Shen, Stephanie Fong, Yiwen Jiang, Zimu Wang, Feilong Tang, Qingyang Xu, Xiangyu Zhao, Zhongxing Xu, Jiahe Liu, Jinpeng Hu, Dominic Dwyer, Zongyuan Ge  

**Link**: [PDF](https://arxiv.org/pdf/2601.03578)  

**Abstract**: The increasing integration of large language models (LLMs) into mental health applications necessitates robust frameworks for evaluating professional safety alignment. Current evaluative approaches primarily rely on refusal-based safety signals, which offer limited insight into the nuanced behaviors required in clinical practice. In mental health, clinically inadequate refusals can be perceived as unempathetic and discourage help-seeking. To address this gap, we move beyond refusal-centric metrics and introduce \texttt{PsychEthicsBench}, the first principle-grounded benchmark based on Australian psychology and psychiatry guidelines, designed to evaluate LLMs' ethical knowledge and behavioral responses through multiple-choice and open-ended tasks with fine-grained ethicality annotations. Empirical results across 14 models reveal that refusal rates are poor indicators of ethical behavior, revealing a significant divergence between safety triggers and clinical appropriateness. Notably, we find that domain-specific fine-tuning can degrade ethical robustness, as several specialized models underperform their base backbones in ethical alignment. PsychEthicsBench provides a foundation for systematic, jurisdiction-aware evaluation of LLMs in mental health, encouraging more responsible development in this domain. 

---
# How Do Large Language Models Learn Concepts During Continual Pre-Training? 

**Authors**: Barry Menglong Yao, Sha Li, Yunzhi Yao, Minqian Liu, Zaishuo Xia, Qifan Wang, Lifu Huang  

**Link**: [PDF](https://arxiv.org/pdf/2601.03570)  

**Abstract**: Human beings primarily understand the world through concepts (e.g., dog), abstract mental representations that structure perception, reasoning, and learning. However, how large language models (LLMs) acquire, retain, and forget such concepts during continual pretraining remains poorly understood. In this work, we study how individual concepts are acquired and forgotten, as well as how multiple concepts interact through interference and synergy. We link these behavioral dynamics to LLMs' internal Concept Circuits, computational subgraphs associated with specific concepts, and incorporate Graph Metrics to characterize circuit structure. Our analysis reveals: (1) LLMs concept circuits provide a non-trivial, statistically significant signal of concept learning and forgetting; (2) Concept circuits exhibit a stage-wise temporal pattern during continual pretraining, with an early increase followed by gradual decrease and stabilization; (3) concepts with larger learning gains tend to exhibit greater forgetting under subsequent training; (4) semantically similar concepts induce stronger interference than weakly related ones; (5) conceptual knowledge differs in their transferability, with some significantly facilitating the learning of others. Together, our findings offer a circuit-level view of concept learning dynamics and inform the design of more interpretable and robust concept-aware training strategies for LLMs. 

---
# DiffCoT: Diffusion-styled Chain-of-Thought Reasoning in LLMs 

**Authors**: Shidong Cao, Hongzhan Lin, Yuxuan Gu, Ziyang Luo, Jing Ma  

**Link**: [PDF](https://arxiv.org/pdf/2601.03559)  

**Abstract**: Chain-of-Thought (CoT) reasoning improves multi-step mathematical problem solving in large language models but remains vulnerable to exposure bias and error accumulation, as early mistakes propagate irreversibly through autoregressive decoding. In this work, we propose DiffCoT, a diffusion-styled CoT framework that reformulates CoT reasoning as an iterative denoising process. DiffCoT integrates diffusion principles at the reasoning-step level via a sliding-window mechanism, enabling unified generation and retrospective correction of intermediate steps while preserving token-level autoregression. To maintain causal consistency, we further introduce a causal diffusion noise schedule that respects the temporal structure of reasoning chains. Extensive experiments on three multi-step CoT reasoning benchmarks across diverse model backbones demonstrate that DiffCoT consistently outperforms existing CoT preference optimization methods, yielding improved robustness and error-correction capability in CoT reasoning. 

---
# Evaluating LLMs for Police Decision-Making: A Framework Based on Police Action Scenarios 

**Authors**: Sangyub Lee, Heedou Kim, Hyeoncheol Kim  

**Link**: [PDF](https://arxiv.org/pdf/2601.03553)  

**Abstract**: The use of Large Language Models (LLMs) in police operations is growing, yet an evaluation framework tailored to police operations remains absent. While LLM's responses may not always be legally incorrect, their unverified use still can lead to severe issues such as unlawful arrests and improper evidence collection. To address this, we propose PAS (Police Action Scenarios), a systematic framework covering the entire evaluation process. Applying this framework, we constructed a novel QA dataset from over 8,000 official documents and established key metrics validated through statistical analysis with police expert judgements. Experimental results show that commercial LLMs struggle with our new police-related tasks, particularly in providing fact-based recommendations. This study highlights the necessity of an expandable evaluation framework to ensure reliable AI-driven police operations. We release our data and prompt template. 

---
# Value-Action Alignment in Large Language Models under Privacy-Prosocial Conflict 

**Authors**: Guanyu Chen, Chenxiao Yu, Xiyang Hu  

**Link**: [PDF](https://arxiv.org/pdf/2601.03546)  

**Abstract**: Large language models (LLMs) are increasingly used to simulate decision-making tasks involving personal data sharing, where privacy concerns and prosocial motivations can push choices in opposite directions. Existing evaluations often measure privacy-related attitudes or sharing intentions in isolation, which makes it difficult to determine whether a model's expressed values jointly predict its downstream data-sharing actions as in real human behaviors. We introduce a context-based assessment protocol that sequentially administers standardized questionnaires for privacy attitudes, prosocialness, and acceptance of data sharing within a bounded, history-carrying session. To evaluate value-action alignments under competing attitudes, we use multi-group structural equation modeling (MGSEM) to identify relations from privacy concerns and prosocialness to data sharing. We propose Value-Action Alignment Rate (VAAR), a human-referenced directional agreement metric that aggregates path-level evidence for expected signs. Across multiple LLMs, we observe stable but model-specific Privacy-PSA-AoDS profiles, and substantial heterogeneity in value-action alignment. 

---
# EvolMem: A Cognitive-Driven Benchmark for Multi-Session Dialogue Memory 

**Authors**: Ye Shen, Dun Pei, Yiqiu Guo, Junying Wang, Yijin Guo, Zicheng Zhang, Qi Jia, Jun Zhou, Guangtao Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2601.03543)  

**Abstract**: Despite recent advances in understanding and leveraging long-range conversational memory, existing benchmarks still lack systematic evaluation of large language models(LLMs) across diverse memory dimensions, particularly in multi-session settings. In this work, we propose EvolMem, a new benchmark for assessing multi-session memory capabilities of LLMs and agent systems. EvolMem is grounded in cognitive psychology and encompasses both declarative and non-declarative memory, further decomposed into multiple fine-grained abilities. To construct the benchmark, we introduce a hybrid data synthesis framework that consists of topic-initiated generation and narrative-inspired transformations. This framework enables scalable generation of multi-session conversations with controllable complexity, accompanied by sample-specific evaluation guidelines. Extensive evaluation reveals that no LLM consistently outperforms others across all memory dimensions. Moreover, agent memory mechanisms do not necessarily enhance LLMs' capabilities and often exhibit notable efficiency limitations. Data and code will be released at this https URL. 

---
# Layer-Order Inversion: Rethinking Latent Multi-Hop Reasoning in Large Language Models 

**Authors**: Xukai Liu, Ye Liu, Jipeng Zhang, Yanghai Zhang, Kai Zhang, Qi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2601.03542)  

**Abstract**: Large language models (LLMs) perform well on multi-hop reasoning, yet how they internally compose multiple facts remains unclear. Recent work proposes \emph{hop-aligned circuit hypothesis}, suggesting that bridge entities are computed sequentially across layers before later-hop answers. Through systematic analyses on real-world multi-hop queries, we show that this hop-aligned assumption does not generalize: later-hop answer entities can become decodable earlier than bridge entities, a phenomenon we call \emph{layer-order inversion}, which strengthens with total hops. To explain this behavior, we propose a \emph{probabilistic recall-and-extract} framework that models multi-hop reasoning as broad probabilistic recall in shallow MLP layers followed by selective extraction in deeper attention layers. This framework is empirically validated through systematic probing analyses, reinterpreting prior layer-wise decoding evidence, explaining chain-of-thought gains, and providing a mechanistic diagnosis of multi-hop failures despite correct single-hop knowledge. Code is available at this https URL. 

---
# DeepSynth-Eval: Objectively Evaluating Information Consolidation in Deep Survey Writing 

**Authors**: Hongzhi Zhang, Yuanze Hu, Tinghai Zhang, Jia Fu, Tao Wang, Junwei Jing, Zhaoxin Fan, Qi Wang, Ruiming Tang, Han Li, Guorui Zhou, Kun Gai  

**Link**: [PDF](https://arxiv.org/pdf/2601.03540)  

**Abstract**: The evolution of Large Language Models (LLMs) towards autonomous agents has catalyzed progress in Deep Research. While retrieval capabilities are well-benchmarked, the post-retrieval synthesis stage--where agents must digest massive amounts of context and consolidate fragmented evidence into coherent, long-form reports--remains under-evaluated due to the subjectivity of open-ended writing. To bridge this gap, we introduce DeepSynth-Eval, a benchmark designed to objectively evaluate information consolidation capabilities. We leverage high-quality survey papers as gold standards, reverse-engineering research requests and constructing "Oracle Contexts" from their bibliographies to isolate synthesis from retrieval noise. We propose a fine-grained evaluation protocol using General Checklists (for factual coverage) and Constraint Checklists (for structural organization), transforming subjective judgment into verifiable metrics. Experiments across 96 tasks reveal that synthesizing information from hundreds of references remains a significant challenge. Our results demonstrate that agentic plan-and-write workflows significantly outperform single-turn generation, effectively reducing hallucinations and improving adherence to complex structural constraints. 

---
# Persona-aware and Explainable Bikeability Assessment: A Vision-Language Model Approach 

**Authors**: Yilong Dai, Ziyi Wang, Chenguang Wang, Kexin Zhou, Yiheng Qian, Susu Xu, Xiang Yan  

**Link**: [PDF](https://arxiv.org/pdf/2601.03534)  

**Abstract**: Bikeability assessment is essential for advancing sustainable urban transportation and creating cyclist-friendly cities, and it requires incorporating users' perceptions of safety and comfort. Yet existing perception-based bikeability assessment approaches face key limitations in capturing the complexity of road environments and adequately accounting for heterogeneity in subjective user perceptions. This paper proposes a persona-aware Vision-Language Model framework for bikeability assessment with three novel contributions: (i) theory-grounded persona conditioning based on established cyclist typology that generates persona-specific explanations via chain-of-thought reasoning; (ii) multi-granularity supervised fine-tuning that combines scarce expert-annotated reasoning with abundant user ratings for joint prediction and explainable assessment; and (iii) AI-enabled data augmentation that creates controlled paired data to isolate infrastructure variable impacts. To test and validate this framework, we developed a panoramic image-based crowdsourcing system and collected 12,400 persona-conditioned assessments from 427 cyclists. Experiment results show that the proposed framework offers competitive bikeability rating prediction while uniquely enabling explainable factor attribution. 

---
# PALM-Bench: A Comprehensive Benchmark for Personalized Audio-Language Models 

**Authors**: Yuwen Wang, Xinyuan Qian, Tian-Hao Zhang, Jiaran Gao, Yuchen Pan, Xin Wang, Zhou Pan, Chen Wei, Yiming Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.03531)  

**Abstract**: Large Audio-Language Models (LALMs) have demonstrated strong performance in audio understanding and generation. Yet, our extensive benchmarking reveals that their behavior is largely generic (e.g., summarizing spoken content) and fails to adequately support personalized question answering (e.g., summarizing what my best friend says). In contrast, human conditions their interpretation and decision-making on each individual's personal context. To bridge this gap, we formalize the task of Personalized LALMs (PALM) for recognizing personal concepts and reasoning within personal context. Moreover, we create the first benchmark (PALM-Bench) to foster the methodological advances in PALM and enable structured evaluation on several tasks across multi-speaker scenarios. Our extensive experiments on representative open-source LALMs, show that existing training-free prompting and supervised fine-tuning strategies, while yield improvements, remains limited in modeling personalized knowledge and transferring them across tasks robustly. Data and code will be released. 

---
# Mem-Gallery: Benchmarking Multimodal Long-Term Conversational Memory for MLLM Agents 

**Authors**: Yuanchen Bei, Tianxin Wei, Xuying Ning, Yanjun Zhao, Zhining Liu, Xiao Lin, Yada Zhu, Hendrik Hamann, Jingrui He, Hanghang Tong  

**Link**: [PDF](https://arxiv.org/pdf/2601.03515)  

**Abstract**: Long-term memory is a critical capability for multimodal large language model (MLLM) agents, particularly in conversational settings where information accumulates and evolves over time. However, existing benchmarks either evaluate multi-session memory in text-only conversations or assess multimodal understanding within localized contexts, failing to evaluate how multimodal memory is preserved, organized, and evolved across long-term conversational trajectories. Thus, we introduce Mem-Gallery, a new benchmark for evaluating multimodal long-term conversational memory in MLLM agents. Mem-Gallery features high-quality multi-session conversations grounded in both visual and textual information, with long interaction horizons and rich multimodal dependencies. Building on this dataset, we propose a systematic evaluation framework that assesses key memory capabilities along three functional dimensions: memory extraction and test-time adaptation, memory reasoning, and memory knowledge management. Extensive benchmarking across thirteen memory systems reveals several key findings, highlighting the necessity of explicit multimodal information retention and memory organization, the persistent limitations in memory reasoning and knowledge management, as well as the efficiency bottleneck of current models. 

---
# IntroLM: Introspective Language Models via Prefilling-Time Self-Evaluation 

**Authors**: Hossein Hosseini Kasnavieh, Gholamreza Haffari, Chris Leckie, Adel N. Toosi  

**Link**: [PDF](https://arxiv.org/pdf/2601.03511)  

**Abstract**: A major challenge for the operation of large language models (LLMs) is how to predict whether a specific LLM will produce sufficiently high-quality output for a given query. Existing approaches rely on external classifiers, most commonly BERT based models, which suffer from limited context windows, constrained representational capacity, and additional computational overhead. We propose IntroLM, a method that enables causal language models to predict their own output quality during the prefilling phase without affecting generation using introspective tokens. By introducing token conditional LoRA that activates only for the introspective token, the model learns to predict the output quality for a given query while preserving the original backbone behavior and avoiding external evaluators. On question answering benchmarks, IntroLM applied to Qwen3 8B achieves a ROC AUC of 90 precent for success prediction, outperforming a DeBERTa classifier by 14 precent. When integrated into multi model routing systems, IntroLM achieves superior cost performance tradeoffs, reducing latency by up to 33 precent and large model usage by up to 50 precent at matched reliability. 

---
# Reasoning Pattern Alignment Merging for Adaptive Reasoning 

**Authors**: Zhaofeng Zhong, Wei Yuan, Tong Chen, Xiangyu Zhao, Quoc Viet Hung Nguyen, Hongzhi Yin  

**Link**: [PDF](https://arxiv.org/pdf/2601.03506)  

**Abstract**: Recent large reasoning models (LRMs) have made substantial progress in complex reasoning tasks, yet they often generate lengthy reasoning paths for every query, incurring unnecessary computation and latency. Existing speed-up approaches typically rely on retraining the model or designing sophisticated prompting, which are either prohibitively expensive or highly sensitive to the input and prompt formulation. In this work, we study model merging as a lightweight alternative for efficient reasoning: by combining a long chain-of-thought (Long-CoT) reasoning model with a Short-CoT instruction model, we obtain an adaptive reasoner without training from scratch or requiring large-scale additional data. Building on this idea, we propose Reasoning Pattern Alignment Merging (RPAM), a layer-wise model merging framework based on feature alignment to facilitate query-adaptive reasoning. RPAM first constructs a small pattern-labeled calibration set that assigns each query an appropriate reasoning pattern. It then optimizes layer-wise merging coefficients by aligning the merged model's intermediate representations with those of the selected model, while a contrastive objective explicitly pushes them away from the non-selected model. Experiments on seven widely used reasoning benchmarks show that RPAM substantially reduces inference cost while maintaining strong performance. Upon article acceptance, we will provide open-source code to reproduce experiments for RPAM. 

---
# Beyond Perplexity: A Lightweight Benchmark for Knowledge Retention in Supervised Fine-Tuning 

**Authors**: Soheil Zibakhsh Shabgahi, Pedram Aghazadeh, Farinaz Koushanfar  

**Link**: [PDF](https://arxiv.org/pdf/2601.03505)  

**Abstract**: Supervised Fine-Tuning (SFT) is a standard approach for injecting domain knowledge into Large Language Models (LLMs). However, relying on validation perplexity to monitor training is often insufficient, as it confounds stylistic mimicry with genuine factual internalization. To address this, we introduce the Knowledge Retention (KR) Test , a lightweight, corpus-grounded evaluation framework designed to distinguish factual learning from linguistics. KR-Test utilizes automatically generated contrastive examples to measure likelihood preferences for correct versus incorrect continuations, requiring no instruction tuning or generative decoding. We validate the framework's integrity through a "blind vs. oracle" baseline analysis. Furthermore, we demonstrate the diagnostic capabilities of KR-Test by analyzing the training dynamics of Low-Rank Adaptation (LoRA). By exposing the fine-grained dissociation between linguistic convergence and knowledge retention, KR-Test enhances the interpretability of fine-tuning dynamics. 

---
# Submodular Evaluation Subset Selection in Automatic Prompt Optimization 

**Authors**: Jinming Nian, Zhiyuan Peng, Hongwei Shang, Dae Hoon Park, Yi Fang  

**Link**: [PDF](https://arxiv.org/pdf/2601.03493)  

**Abstract**: Automatic prompt optimization reduces manual prompt engineering, but relies on task performance measured on a small, often randomly sampled evaluation subset as its main source of feedback signal. Despite this, how to select that evaluation subset is usually treated as an implementation detail. We study evaluation subset selection for prompt optimization from a principled perspective and propose SESS, a submodular evaluation subset selection method. We frame selection as maximizing an objective set function and show that, under mild conditions, it is monotone and submodular, enabling greedy selection with theoretical guarantees. Across GSM8K, MATH, and GPQA-Diamond, submodularly selected evaluation subsets can yield better optimized prompts than random or heuristic baselines. 

---
# CALM: Culturally Self-Aware Language Models 

**Authors**: Lingzhi Shen, Xiaohao Cai, Yunfei Long, Imran Razzak, Guanming Chen, Shoaib Jameel  

**Link**: [PDF](https://arxiv.org/pdf/2601.03483)  

**Abstract**: Cultural awareness in language models is the capacity to understand and adapt to diverse cultural contexts. However, most existing approaches treat culture as static background knowledge, overlooking its dynamic and evolving nature. This limitation reduces their reliability in downstream tasks that demand genuine cultural sensitivity. In this work, we introduce CALM, a novel framework designed to endow language models with cultural self-awareness. CALM disentangles task semantics from explicit cultural concepts and latent cultural signals, shaping them into structured cultural clusters through contrastive learning. These clusters are then aligned via cross-attention to establish fine-grained interactions among related cultural features and are adaptively integrated through a Mixture-of-Experts mechanism along culture-specific dimensions. The resulting unified representation is fused with the model's original knowledge to construct a culturally grounded internal identity state, which is further enhanced through self-prompted reflective learning, enabling continual adaptation and self-correction. Extensive experiments conducted on multiple cross-cultural benchmark datasets demonstrate that CALM consistently outperforms state-of-the-art methods. 

---
# Self-Explaining Hate Speech Detection with Moral Rationales 

**Authors**: Francielle Vargas, Jackson Trager, Diego Alves, Surendrabikram Thapa, Matteo Guida, Berk Atil, Daryna Dementieva, Andrew Smart, Ameeta Agrawal  

**Link**: [PDF](https://arxiv.org/pdf/2601.03481)  

**Abstract**: Hate speech detection models rely on surface-level lexical features, increasing vulnerability to spurious correlations and limiting robustness, cultural contextualization, and interpretability. We propose Supervised Moral Rationale Attention (SMRA), the first self-explaining hate speech detection framework to incorporate moral rationales as direct supervision for attention alignment. Based on Moral Foundations Theory, SMRA aligns token-level attention with expert-annotated moral rationales, guiding models to attend to morally salient spans rather than spurious lexical patterns. Unlike prior rationale-supervised or post-hoc approaches, SMRA integrates moral rationale supervision directly into the training objective, producing inherently interpretable and contextualized explanations. To support our framework, we also introduce HateBRMoralXplain, a Brazilian Portuguese benchmark dataset annotated with hate labels, moral categories, token-level moral rationales, and socio-political metadata. Across binary hate speech detection and multi-label moral sentiment classification, SMRA consistently improves performance (e.g., +0.9 and +1.5 F1, respectively) while substantially enhancing explanation faithfulness, increasing IoU F1 (+7.4 pp) and Token F1 (+5.0 pp). Although explanations become more concise, sufficiency improves (+2.3 pp) and fairness remains stable, indicating more faithful rationales without performance or bias trade-offs 

---
# SegNSP: Revisiting Next Sentence Prediction for Linear Text Segmentation 

**Authors**: José Isidro, Filipe Cunha, Purificação Silvano, Alípio Jorge, Nuno Guimarães, Sérgio Nunes, Ricardo Campos  

**Link**: [PDF](https://arxiv.org/pdf/2601.03474)  

**Abstract**: Linear text segmentation is a long-standing problem in natural language processing (NLP), focused on dividing continuous text into coherent and semantically meaningful units. Despite its importance, the task remains challenging due to the complexity of defining topic boundaries, the variability in discourse structure, and the need to balance local coherence with global context. These difficulties hinder downstream applications such as summarization, information retrieval, and question answering. In this work, we introduce SegNSP, framing linear text segmentation as a next sentence prediction (NSP) task. Although NSP has largely been abandoned in modern pre-training, its explicit modeling of sentence-to-sentence continuity makes it a natural fit for detecting topic boundaries. We propose a label-agnostic NSP approach, which predicts whether the next sentence continues the current topic without requiring explicit topic labels, and enhance it with a segmentation-aware loss combined with harder negative sampling to better capture discourse continuity. Unlike recent proposals that leverage NSP alongside auxiliary topic classification, our approach avoids task-specific supervision. We evaluate our model against established baselines on two datasets, CitiLink-Minutes, for which we establish the first segmentation benchmark, and WikiSection. On CitiLink-Minutes, SegNSP achieves a B-$F_1$ of 0.79, closely aligning with human-annotated topic transitions, while on WikiSection it attains a B-F$_1$ of 0.65, outperforming the strongest reproducible baseline, TopSeg, by 0.17 absolute points. These results demonstrate competitive and robust performance, highlighting the effectiveness of modeling sentence-to-sentence continuity for improving segmentation quality and supporting downstream NLP applications. 

---
# EpiQAL: Benchmarking Large Language Models in Epidemiological Question Answering for Enhanced Alignment and Reasoning 

**Authors**: Mingyang Wei, Dehai Min, Zewen Liu, Yuzhang Xie, Guanchen Wu, Carl Yang, Max S. Y. Lau, Qi He, Lu Cheng, Wei Jin  

**Link**: [PDF](https://arxiv.org/pdf/2601.03471)  

**Abstract**: Reliable epidemiological reasoning requires synthesizing study evidence to infer disease burden, transmission dynamics, and intervention effects at the population level. Existing medical question answering benchmarks primarily emphasize clinical knowledge or patient-level reasoning, yet few systematically evaluate evidence-grounded epidemiological inference. We present EpiQAL, the first diagnostic benchmark for epidemiological question answering across diverse diseases, comprising three subsets built from open-access literature. The subsets respectively evaluate text-grounded factual recall, multi-step inference linking document evidence with epidemiological principles, and conclusion reconstruction with the Discussion section withheld. Construction combines expert-designed taxonomy guidance, multi-model verification, and retrieval-based difficulty control. Experiments on ten open models reveal that current LLMs show limited performance on epidemiological reasoning, with multi-step inference posing the greatest challenge. Model rankings shift across subsets, and scale alone does not predict success. Chain-of-Thought prompting benefits multi-step inference but yields mixed results elsewhere. EpiQAL provides fine-grained diagnostic signals for evidence grounding, inferential reasoning, and conclusion reconstruction. 

---
# Prompting Underestimates LLM Capability for Time Series Classification 

**Authors**: Dan Schumacher, Erfan Nourbakhsh, Rocky Slavin, Anthony Rios  

**Link**: [PDF](https://arxiv.org/pdf/2601.03464)  

**Abstract**: Prompt-based evaluations suggest that large language models (LLMs) perform poorly on time series classification, raising doubts about whether they encode meaningful temporal structure. We show that this conclusion reflects limitations of prompt-based generation rather than the model's representational capacity by directly comparing prompt outputs with linear probes over the same internal representations. While zero-shot prompting performs near chance, linear probes improve average F1 from 0.15-0.26 to 0.61-0.67, often matching or exceeding specialized time series models. Layer-wise analyses further show that class-discriminative time series information emerges in early transformer layers and is amplified by visual and multimodal inputs. Together, these results demonstrate a systematic mismatch between what LLMs internally represent and what prompt-based evaluation reveals, leading current evaluations to underestimate their time series understanding. 

---
# Enhancing Linguistic Competence of Language Models through Pre-training with Language Learning Tasks 

**Authors**: Atsuki Yamaguchi, Maggie Mi, Nikolaos Aletras  

**Link**: [PDF](https://arxiv.org/pdf/2601.03448)  

**Abstract**: Language models (LMs) are pre-trained on raw text datasets to generate text sequences token-by-token. While this approach facilitates the learning of world knowledge and reasoning, it does not explicitly optimize for linguistic competence. To bridge this gap, we propose L2T, a pre-training framework integrating Language Learning Tasks alongside standard next-token prediction. Inspired by human language acquisition, L2T transforms raw text into structured input-output pairs to provide explicit linguistic stimulation. Pre-training LMs on a mixture of raw text and L2T data not only improves overall performance on linguistic competence benchmarks but accelerates its acquisition, while maintaining competitive performance on general reasoning tasks. 

---
# Grading Scale Impact on LLM-as-a-Judge: Human-LLM Alignment Is Highest on 0-5 Grading Scale 

**Authors**: Weiyue Li, Minda Zhao, Weixuan Dong, Jiahui Cai, Yuze Wei, Michael Pocress, Yi Li, Wanyan Yuan, Xiaoyue Wang, Ruoyu Hou, Kaiyuan Lou, Wenqi Zeng, Yutong Yang, Yilun Du, Mengyu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.03444)  

**Abstract**: Large language models (LLMs) are increasingly used as automated evaluators, yet prior works demonstrate that these LLM judges often lack consistency in scoring when the prompt is altered. However, the effect of the grading scale itself remains underexplored. We study the LLM-as-a-judge problem by comparing two kinds of raters: humans and LLMs. We collect ratings from both groups on three scales and across six benchmarks that include objective, open-ended subjective, and mixed tasks. Using intraclass correlation coefficients (ICC) to measure absolute agreement, we find that LLM judgments are not perfectly consistent across scales on subjective benchmarks, and that the choice of scale substantially shifts human-LLM agreement, even when within-group panel reliability is high. Aggregated over tasks, the grading scale of 0-5 yields the strongest human-LLM alignment. We further demonstrate that pooled reliability can mask benchmark heterogeneity and reveal systematic subgroup differences in alignment across gender groups, strengthening the importance of scale design and sub-level diagnostics as essential components of LLM-as-a-judge protocols. 

---
# The Critical Role of Aspects in Measuring Document Similarity 

**Authors**: Eftekhar Hossain, Tarnika Hazra, Ahatesham Bhuiyan, Santu Karmaker  

**Link**: [PDF](https://arxiv.org/pdf/2601.03435)  

**Abstract**: We introduce ASPECTSIM, a simple and interpretable framework that requires conditioning document similarity on an explicitly specified aspect, which is different from the traditional holistic approach in measuring document similarity. Experimenting with a newly constructed benchmark of 26K aspect-document pairs, we found that ASPECTSIM, when implemented with direct GPT-4o prompting, achieves substantially higher human-machine agreement ($\approx$80% higher) than the same for holistic similarity without explicit aspects. These findings underscore the importance of explicitly accounting for aspects when measuring document similarity and highlight the need to revise standard practice. Next, we conducted a large-scale meta-evaluation using 16 smaller open-source LLMs and 9 embedding models with a focus on making ASPECTSIM accessible and reproducible. While directly prompting LLMs to produce ASPECTSIM scores turned out be ineffective (20-30% human-machine agreement), a simple two-stage refinement improved their agreement by $\approx$140%. Nevertheless, agreement remains well below that of GPT-4o-based models, indicating that smaller open-source LLMs still lag behind large proprietary models in capturing aspect-conditioned similarity. 

---
# Training-Free Adaptation of New-Generation LLMs using Legacy Clinical Models 

**Authors**: Sasha Ronaghi, Chloe Stanwyck, Asad Aali, Amir Ronaghi, Miguel Fuentes, Tina Hernandez-Boussard, Emily Alsentzer  

**Link**: [PDF](https://arxiv.org/pdf/2601.03423)  

**Abstract**: Adapting language models to the clinical domain through continued pretraining and fine-tuning requires costly retraining for each new model generation. We propose Cross-Architecture Proxy Tuning (CAPT), a model-ensembling approach that enables training-free adaptation of state-of-the-art general-domain models using existing clinical models. CAPT supports models with disjoint vocabularies, leveraging contrastive decoding to selectively inject clinically relevant signals while preserving the general-domain model's reasoning and fluency. On six clinical classification and text-generation tasks, CAPT with a new-generation general-domain model and an older-generation clinical model consistently outperforms both models individually and state-of-the-art ensembling approaches (average +17.6% over UniTE, +41.4% over proxy tuning across tasks). Through token-level analysis and physician case studies, we demonstrate that CAPT amplifies clinically actionable language, reduces context errors, and increases clinical specificity. 

---
# PCoA: A New Benchmark for Medical Aspect-Based Summarization With Phrase-Level Context Attribution 

**Authors**: Bohao Chu, Sameh Frihat, Tabea M. G. Pakull, Hendrik Damm, Meijie Li, Ula Muhabbek, Georg Lodde, Norbert Fuhr  

**Link**: [PDF](https://arxiv.org/pdf/2601.03418)  

**Abstract**: Verifying system-generated summaries remains challenging, as effective verification requires precise attribution to the source context, which is especially crucial in high-stakes medical domains. To address this challenge, we introduce PCoA, an expert-annotated benchmark for medical aspect-based summarization with phrase-level context attribution. PCoA aligns each aspect-based summary with its supporting contextual sentences and contributory phrases within them. We further propose a fine-grained, decoupled evaluation framework that independently assesses the quality of generated summaries, citations, and contributory phrases. Through extensive experiments, we validate the quality and consistency of the PCoA dataset and benchmark several large language models on the proposed task. Experimental results demonstrate that PCoA provides a reliable benchmark for evaluating system-generated summaries with phrase-level context attribution. Furthermore, comparative experiments show that explicitly identifying relevant sentences and contributory phrases before summarization can improve overall quality. The data and code are available at this https URL. 

---
# Implicit Graph, Explicit Retrieval: Towards Efficient and Interpretable Long-horizon Memory for Large Language Models 

**Authors**: Xin Zhang, Kailai Yang, Hao Li, Chenyue Li, Qiyu Wei, Sophia Ananiadou  

**Link**: [PDF](https://arxiv.org/pdf/2601.03417)  

**Abstract**: Long-horizon applications increasingly require large language models (LLMs) to answer queries when relevant evidence is sparse and dispersed across very long contexts. Existing memory systems largely follow two paradigms: explicit structured memories offer interpretability but often become brittle under long-context overload, while latent memory mechanisms are efficient and stable yet difficult to inspect. We propose LatentGraphMem, a memory framework that combines implicit graph memory with explicit subgraph retrieval. LatentGraphMem stores a graph-structured memory in latent space for stability and efficiency, and exposes a task-specific subgraph retrieval interface that returns a compact symbolic subgraph under a fixed budget for downstream reasoning and human inspection. During training, an explicit graph view is materialized to interface with a frozen reasoner for question-answering supervision. At inference time, retrieval is performed in latent space and only the retrieved subgraph is externalized. Experiments on long-horizon benchmarks across multiple model scales show that LatentGraphMem consistently outperforms representative explicit-graph and latent-memory baselines, while enabling parameter-efficient adaptation and flexible scaling to larger reasoners without introducing large symbolic artifacts. 

---
# Tigrinya Number Verbalization: Rules, Algorithm, and Implementation 

**Authors**: Fitsum Gaim, Issayas Tesfamariam  

**Link**: [PDF](https://arxiv.org/pdf/2601.03403)  

**Abstract**: We present a systematic formalization of Tigrinya cardinal and ordinal number verbalization, addressing a gap in computational resources for the language. This work documents the canonical rules governing the expression of numerical values in spoken Tigrinya, including the conjunction system, scale words, and special cases for dates, times, and currency. We provide a formal algorithm for number-to-word conversion and release an open-source implementation. Evaluation of frontier large language models (LLMs) reveals significant gaps in their ability to accurately verbalize Tigrinya numbers, underscoring the need for explicit rule documentation. This work serves language modeling, speech synthesis, and accessibility applications targeting Tigrinya-speaking communities. 

---
# Rendering Data Unlearnable by Exploiting LLM Alignment Mechanisms 

**Authors**: Ruihan Zhang, Jun Sun  

**Link**: [PDF](https://arxiv.org/pdf/2601.03401)  

**Abstract**: Large language models (LLMs) are increasingly trained on massive, heterogeneous text corpora, raising serious concerns about the unauthorised use of proprietary or personal data during model training. In this work, we address the problem of data protection against unwanted model learning in a realistic black-box setting. We propose Disclaimer Injection, a novel data-level defence that renders text unlearnable to LLMs. Rather than relying on model-side controls or explicit data removal, our approach exploits the models' own alignment mechanisms: by injecting carefully designed alignment-triggering disclaimers to prevent effective learning. Through layer-wise analysis, we find that fine-tuning on such protected data induces persistent activation of alignment-related layers, causing alignment constraints to override task learning even on common inputs. Consequently, models trained on such data exhibit substantial and systematic performance degradation compared to standard fine-tuning. Our results identify alignment behaviour as a previously unexplored lever for data protection and, to our knowledge, present the first practical method for restricting data learnability at LLM scale without requiring access to or modification of the training pipeline. 

---
# Breaking the Assistant Mold: Modeling Behavioral Variation in LLM Based Procedural Character Generation 

**Authors**: Maan Qraitem, Kate Saenko, Bryan A. Plummer  

**Link**: [PDF](https://arxiv.org/pdf/2601.03396)  

**Abstract**: Procedural content generation has enabled vast virtual worlds through levels, maps, and quests, but large-scale character generation remains underexplored. We identify two alignment-induced biases in existing methods: a positive moral bias, where characters uniformly adopt agreeable stances (e.g. always saying lying is bad), and a helpful assistant bias, where characters invariably answer questions directly (e.g. never refusing or deflecting). While such tendencies suit instruction-following systems, they suppress dramatic tension and yield predictable characters, stemming from maximum likelihood training and assistant fine-tuning. To address this, we introduce PersonaWeaver, a framework that disentangles world-building (roles, demographics) from behavioral-building (moral stances, interactional styles), yielding characters with more diverse reactions and moral stances, as well as second-order diversity in stylistic markers like length, tone, and punctuation. Code: this https URL 

---
# Metaphors are a Source of Cross-Domain Misalignment of Large Reasoning Models 

**Authors**: Zhibo Hu, Chen Wang, Yanfeng Shu, Hye-young Paik, Liming Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2601.03388)  

**Abstract**: Earlier research has shown that metaphors influence human's decision making, which raises the question of whether metaphors also influence large language models (LLMs)' reasoning pathways, considering their training data contain a large number of metaphors. In this work, we investigate the problem in the scope of the emergent misalignment problem where LLMs can generalize patterns learned from misaligned content in one domain to another domain. We discover a strong causal relationship between metaphors in training data and the misalignment degree of LLMs' reasoning contents. With interventions using metaphors in pre-training, fine-tuning and re-alignment phases, models' cross-domain misalignment degrees change significantly. As we delve deeper into the causes behind this phenomenon, we observe that there is a connection between metaphors and the activation of global and local latent features of large reasoning models. By monitoring these latent features, we design a detector that predict misaligned content with high accuracy. 

---
# A path to natural language through tokenisation and transformers 

**Authors**: David S. Berman, Alexander G. Stapleton  

**Link**: [PDF](https://arxiv.org/pdf/2601.03368)  

**Abstract**: Natural languages exhibit striking regularities in their statistical structure, including notably the emergence of Zipf's and Heaps' laws. Despite this, it remains broadly unclear how these properties relate to the modern tokenisation schemes used in contemporary transformer models. In this note, we analyse the information content (as measured by the Shannon entropy) of various corpora under the assumption of a Zipfian frequency distribution, and derive a closed-form expression for the slot entropy expectation value. We then empirically investigate how byte--pair encoding (BPE) transforms corpus statistics, showing that recursive applications of BPE drive token frequencies toward a Zipfian power law while inducing a characteristic growth pattern in empirical entropy. Utilizing the ability of transformers to learn context dependent token probability distributions, we train language models on corpora tokenised at varying BPE depths, revealing that the model predictive entropies increasingly agree with Zipf-derived predictions as the BPE depth increases. Attention-based diagnostics further indicate that deeper tokenisation reduces local token dependencies, bringing the empirical distribution closer to the weakly dependent (near IID) regime. Together, these results clarify how BPE acts not only as a compression mechanism but also as a statistical transform that reconstructs key informational properties of natural language. 

---
# Bare-Metal Tensor Virtualization: Overcoming the Memory Wall in Edge-AI Inference on ARM64 

**Authors**: Bugra Kilictas, Faruk Alpay  

**Link**: [PDF](https://arxiv.org/pdf/2601.03324)  

**Abstract**: The deployment of Large Language Models (LLMs) on edge devices is fundamentally constrained by the "Memory Wall" the bottleneck where data movement latency outstrips arithmetic throughput. Standard inference runtimes often incur significant overhead through high-level abstractions, dynamic dispatch, and unaligned memory access patterns. In this work, we present a novel "Virtual Tensor Core" architecture implemented in software, optimized specifically for ARM64 microarchitectures (Apple Silicon). By bypassing standard library containers in favor of direct memory mapping (mmap) and implementing hand-tuned NEON SIMD kernels, we achieve a form of "Software-Defined Direct Memory Access (DMA)." Our proposed Tensor Virtualization Layout (TVL) guarantees 100% cache line utilization for weight matrices, while our zero-copy loader eliminates initialization latency. Experimental results on a 110M parameter model demonstrate a stable throughput of >60 tokens/second on M2 hardware. While proprietary hardware accelerators (e.g., Apple AMX) can achieve higher peak throughput, our architecture provides a fully open, portable, and deterministic reference implementation for studying the memory bottleneck on general-purpose ARM silicon, meeting the 200ms psycholinguistic latency threshold without opaque dependencies. 

---
# Topic Segmentation Using Generative Language Models 

**Authors**: Pierre Mackenzie, Maya Shah, Patrick Frenett  

**Link**: [PDF](https://arxiv.org/pdf/2601.03276)  

**Abstract**: Topic segmentation using generative Large Language Models (LLMs) remains relatively unexplored. Previous methods use semantic similarity between sentences, but such models lack the long range dependencies and vast knowledge found in LLMs. In this work, we propose an overlapping and recursive prompting strategy using sentence enumeration. We also support the adoption of the boundary similarity evaluation metric. Results show that LLMs can be more effective segmenters than existing methods, but issues remain to be solved before they can be relied upon for topic segmentation. 

---
# LLM_annotate: A Python package for annotating and analyzing fiction characters 

**Authors**: Hannes Rosenbusch  

**Link**: [PDF](https://arxiv.org/pdf/2601.03274)  

**Abstract**: LLM_annotate is a Python package for analyzing the personality of fiction characters with large language models. It standardizes workflows for annotating character behaviors in full texts (e.g., books and movie scripts), inferring character traits, and validating annotation/inference quality via a human-in-the-loop GUI. The package includes functions for text chunking, LLM-based annotation, character name disambiguation, quality scoring, and computation of character-level statistics and embeddings. Researchers can use any LLM, commercial, open-source, or custom, within LLM_annotate. Through tutorial examples using The Simpsons Movie and the novel Pride and Prejudice, I demonstrate the usage of the package for efficient and reproducible character analyses. 

---
# GuardEval: A Multi-Perspective Benchmark for Evaluating Safety, Fairness, and Robustness in LLM Moderators 

**Authors**: Naseem Machlovi, Maryam Saleki, Ruhul Amin, Mohamed Rahouti, Shawqi Al-Maliki, Junaid Qadir, Mohamed M. Abdallah, Ala Al-Fuqaha  

**Link**: [PDF](https://arxiv.org/pdf/2601.03273)  

**Abstract**: As large language models (LLMs) become deeply embedded in daily life, the urgent need for safer moderation systems, distinguishing between naive from harmful requests while upholding appropriate censorship boundaries, has never been greater. While existing LLMs can detect harmful or unsafe content, they often struggle with nuanced cases such as implicit offensiveness, subtle gender and racial biases, and jailbreak prompts, due to the subjective and context-dependent nature of these issues. Furthermore, their heavy reliance on training data can reinforce societal biases, resulting in inconsistent and ethically problematic outputs. To address these challenges, we introduce GuardEval, a unified multi-perspective benchmark dataset designed for both training and evaluation, containing 106 fine-grained categories spanning human emotions, offensive and hateful language, gender and racial bias, and broader safety concerns. We also present GemmaGuard (GGuard), a QLoRA fine-tuned version of Gemma3-12B trained on GuardEval, to assess content moderation with fine-grained labels. Our evaluation shows that GGuard achieves a macro F1 score of 0.832, substantially outperforming leading moderation models, including OpenAI Moderator (0.64) and Llama Guard (0.61). We show that multi-perspective, human-centered safety benchmarks are critical for reducing biased and inconsistent moderation decisions. GuardEval and GGuard together demonstrate that diverse, representative data materially improve safety, fairness, and robustness on complex, borderline cases. 

---
# Less is more: Not all samples are effective for evaluation 

**Authors**: Wentang Song, Jinqiang Li, Kele Huang, Junhui Lin, Shengxiang Wu, Zhongshi Xie  

**Link**: [PDF](https://arxiv.org/pdf/2601.03272)  

**Abstract**: The versatility of Large Language Models (LLMs) in vertical domains has spurred the development of numerous specialized evaluation benchmarks. However, these benchmarks often suffer from significant semantic redundancy and impose high computational costs during evaluation. Existing compression methods, such as tinyBenchmarks depend critically on correctness labels from multiple historical models evaluated on the full test set, making them inapplicable in cold-start scenarios, such as the introduction of a new task, domain, or model with no prior evaluation history.
To address this limitation, we propose a history-free test set compression framework that requires no prior model performance data. Our method begins by fine-tuning a base LLM on a small amount of domain-specific data to internalize task-relevant semantics. It then generates high-level semantic embeddings for all original test samples using only their raw textual content. In this domain-adapted embedding space, we perform task-aware clustering and introduce a novel dataset X-ray mechanism that analyzes cluster geometry to dynamically calibrate the compression intensity based on the intrinsic redundancy of the benchmark.
Experiments on professional-domain dataset, notably a large-scale 3GPP communications benchmark, demonstrate that our approach effectively identifies and removes redundant samples, reducing evaluation cost by over 90% while preserving high fidelity to the full benchmark. 

---
# Advances and Challenges in Semantic Textual Similarity: A Comprehensive Survey 

**Authors**: Lokendra Kumar, Neelesh S. Upadhye, Kannan Piedy  

**Link**: [PDF](https://arxiv.org/pdf/2601.03270)  

**Abstract**: Semantic Textual Similarity (STS) research has expanded rapidly since 2021, driven by advances in transformer architectures, contrastive learning, and domain-specific techniques. This survey reviews progress across six key areas: transformer-based models, contrastive learning, domain-focused solutions, multi-modal methods, graph-based approaches, and knowledge-enhanced techniques. Recent transformer models such as FarSSiBERT and DeBERTa-v3 have achieved remarkable accuracy, while contrastive methods like AspectCSE have established new benchmarks. Domain-adapted models, including CXR-BERT for medical texts and Financial-STS for finance, demonstrate how STS can be effectively customized for specialized fields. Moreover, multi-modal, graph-based, and knowledge-integrated models further enhance semantic understanding and representation. By organizing and analyzing these developments, the survey provides valuable insights into current methods, practical applications, and remaining challenges. It aims to guide researchers and practitioners alike in navigating rapid advancements, highlighting emerging trends and future opportunities in the evolving field of STS. 

---
# The Instruction Gap: LLMs get lost in Following Instruction 

**Authors**: Vishesh Tripathi, Uday Allu, Biddwan Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2601.03269)  

**Abstract**: Large Language Models (LLMs) have shown remarkable capabilities in natural language understanding and generation, yet their deployment in enterprise environments reveals a critical limitation: inconsistent adherence to custom instructions. This study presents a comprehensive evaluation of 13 leading LLMs across instruction compliance, response accuracy, and performance metrics in realworld RAG (Retrieval-Augmented Generation) scenarios. Through systematic testing with samples and enterprise-grade evaluation protocols, we demonstrate that instruction following varies dramatically across models, with Claude-Sonnet-4 and GPT-5 achieving the highest results. Our findings reveal the "instruction gap" - a fundamental challenge where models excel at general tasks but struggle with precise instruction adherence required for enterprise deployment. This work provides practical insights for organizations deploying LLM-powered solutions and establishes benchmarks for instruction-following capabilities across major model families. 

---
# WRAVAL -- WRiting Assist eVALuation 

**Authors**: Gabriel Benedict, Matthew Butler, Naved Merchant, Eetu Salama-Laine  

**Link**: [PDF](https://arxiv.org/pdf/2601.03268)  

**Abstract**: The emergence of Large Language Models (LLMs) has shifted language model evaluation toward reasoning and problem-solving tasks as measures of general intelligence. Small Language Models (SLMs) -- defined here as models under 10B parameters -- typically score 3-4 times lower than LLMs on these metrics. However, we demonstrate that these evaluations fail to capture SLMs' effectiveness in common industrial applications, such as tone modification tasks (e.g., funny, serious, professional). We propose an evaluation framework specifically designed to highlight SLMs' capabilities in non-reasoning tasks where predefined evaluation datasets don't exist. Our framework combines novel approaches in data generation, prompt-tuning, and LLM-based evaluation to demonstrate the potential of task-specific finetuning. This work provides practitioners with tools to effectively benchmark both SLMs and LLMs for practical applications, particularly in edge and private computing scenarios. Our implementation is available at: this https URL. 

---
# OpenAI GPT-5 System Card 

**Authors**: Aaditya Singh, Adam Fry, Adam Perelman, Adam Tart, Adi Ganesh, Ahmed El-Kishky, Aidan McLaughlin, Aiden Low, AJ Ostrow, Akhila Ananthram, Akshay Nathan, Alan Luo, Alec Helyar, Aleksander Madry, Aleksandr Efremov, Aleksandra Spyra, Alex Baker-Whitcomb, Alex Beutel, Alex Karpenko, Alex Makelov, Alex Neitz, Alex Wei, Alexandra Barr, Alexandre Kirchmeyer, Alexey Ivanov, Alexi Christakis, Alistair Gillespie, Allison Tam, Ally Bennett, Alvin Wan, Alyssa Huang, Amy McDonald Sandjideh, Amy Yang, Ananya Kumar, Andre Saraiva, Andrea Vallone, Andrei Gheorghe, Andres Garcia Garcia, Andrew Braunstein, Andrew Liu, Andrew Schmidt, Andrey Mereskin, Andrey Mishchenko, Andy Applebaum, Andy Rogerson, Ann Rajan, Annie Wei, Anoop Kotha, Anubha Srivastava, Anushree Agrawal, Arun Vijayvergiya, Ashley Tyra, Ashvin Nair, Avi Nayak, Ben Eggers, Bessie Ji, Beth Hoover, Bill Chen, Blair Chen, Boaz Barak, Borys Minaiev, Botao Hao, Bowen Baker, Brad Lightcap, Brandon McKinzie, Brandon Wang, Brendan Quinn, Brian Fioca, Brian Hsu, Brian Yang, Brian Yu, Brian Zhang, Brittany Brenner, Callie Riggins Zetino, Cameron Raymond, Camillo Lugaresi, Carolina Paz, Cary Hudson, Cedric Whitney, Chak Li, Charles Chen, Charlotte Cole, Chelsea Voss, Chen Ding, Chen Shen, Chengdu Huang, Chris Colby, Chris Hallacy, Chris Koch, Chris Lu, Christina Kaplan, Christina Kim, CJ Minott-Henriques, Cliff Frey, Cody Yu, Coley Czarnecki, Colin Reid, Colin Wei, Cory Decareaux, Cristina Scheau  

**Link**: [PDF](https://arxiv.org/pdf/2601.03267)  

**Abstract**: This is the system card published alongside the OpenAI GPT-5 launch, August 2025.
GPT-5 is a unified system with a smart and fast model that answers most questions, a deeper reasoning model for harder problems, and a real-time router that quickly decides which model to use based on conversation type, complexity, tool needs, and explicit intent (for example, if you say 'think hard about this' in the prompt). The router is continuously trained on real signals, including when users switch models, preference rates for responses, and measured correctness, improving over time. Once usage limits are reached, a mini version of each model handles remaining queries.
This system card focuses primarily on gpt-5-thinking and gpt-5-main, while evaluations for other models are available in the appendix. The GPT-5 system not only outperforms previous models on benchmarks and answers questions more quickly, but -- more importantly -- is more useful for real-world queries. We've made significant advances in reducing hallucinations, improving instruction following, and minimizing sycophancy, and have leveled up GPT-5's performance in three of ChatGPT's most common uses: writing, coding, and health. All of the GPT-5 models additionally feature safe-completions, our latest approach to safety training to prevent disallowed content.
Similarly to ChatGPT agent, we have decided to treat gpt-5-thinking as High capability in the Biological and Chemical domain under our Preparedness Framework, activating the associated safeguards. While we do not have definitive evidence that this model could meaningfully help a novice to create severe biological harm -- our defined threshold for High capability -- we have chosen to take a precautionary approach. 

---
# Benchmarking and Adapting On-Device Large Language Models for Clinical Decision Support 

**Authors**: Alif Munim, Jun Ma, Omar Ibrahim, Alhusain Abdalla, Shuolin Yin, Leo Chen, Bo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.03266)  

**Abstract**: Large language models (LLMs) have rapidly advanced in clinical decision-making, yet the deployment of proprietary systems is hindered by privacy concerns and reliance on cloud-based infrastructure. Open-source alternatives allow local inference but often require large model sizes that limit their use in resource-constrained clinical settings. Here, we benchmark two on-device LLMs, gpt-oss-20b and gpt-oss-120b, across three representative clinical tasks: general disease diagnosis, specialty-specific (ophthalmology) diagnosis and management, and simulation of human expert grading and evaluation. We compare their performance with state-of-the-art proprietary models (GPT-5 and o4-mini) and a leading open-source model (DeepSeek-R1), and we further evaluate the adaptability of on-device systems by fine-tuning gpt-oss-20b on general diagnostic data. Across tasks, gpt-oss models achieve performance comparable to or exceeding DeepSeek-R1 and o4-mini despite being substantially smaller. In addition, fine-tuning remarkably improves the diagnostic accuracy of gpt-oss-20b, enabling it to approach the performance of GPT-5. These findings highlight the potential of on-device LLMs to deliver accurate, adaptable, and privacy-preserving clinical decision support, offering a practical pathway for broader integration of LLMs into routine clinical practice. 

---
# Jailbreak-Zero: A Path to Pareto Optimal Red Teaming for Large Language Models 

**Authors**: Kai Hu, Abhinav Aggarwal, Mehran Khodabandeh, David Zhang, Eric Hsin, Li Chen, Ankit Jain, Matt Fredrikson, Akash Bharadwaj  

**Link**: [PDF](https://arxiv.org/pdf/2601.03265)  

**Abstract**: This paper introduces Jailbreak-Zero, a novel red teaming methodology that shifts the paradigm of Large Language Model (LLM) safety evaluation from a constrained example-based approach to a more expansive and effective policy-based framework. By leveraging an attack LLM to generate a high volume of diverse adversarial prompts and then fine-tuning this attack model with a preference dataset, Jailbreak-Zero achieves Pareto optimality across the crucial objectives of policy coverage, attack strategy diversity, and prompt fidelity to real user inputs. The empirical evidence demonstrates the superiority of this method, showcasing significantly higher attack success rates against both open-source and proprietary models like GPT-40 and Claude 3.5 when compared to existing state-of-the-art techniques. Crucially, Jailbreak-Zero accomplishes this while producing human-readable and effective adversarial prompts with minimal need for human intervention, thereby presenting a more scalable and comprehensive solution for identifying and mitigating the safety vulnerabilities of LLMs. 

---
# Internal Reasoning vs. External Control: A Thermodynamic Analysis of Sycophancy in Large Language Models 

**Authors**: Edward Y. Chang  

**Link**: [PDF](https://arxiv.org/pdf/2601.03263)  

**Abstract**: Large Language Models frequently exhibit sycophancy, prioritizing user agreeableness over correctness. We investigate whether this requires external regulation or can be mitigated by internal reasoning alone. Using CAP-GSM8K (N=500), an adversarial dataset, we evaluate internal (CoT) versus external (RCA) mechanisms across GPT-3.5, GPT-4o, and GPT-5.1. Our results reveal the structural limits of internal reasoning: it causes performance collapse in weak models (the Prioritization Paradox) and leaves an 11.4\% final output gap in frontier models. In contrast, RCA structurally eliminates sycophancy (0.0\%) across all tiers. We synthesize these findings into a thermodynamic hierarchy: hybrid systems achieve Resonance (optimal efficiency) only when capabilities are matched and strong, while weak or mismatched pairs succumb to Dissonance and Entropy. This confirms that external structural constraints are strictly necessary to guarantee safety. 

---
# DeepResearch-Slice: Bridging the Retrieval-Utilization Gap via Explicit Text Slicing 

**Authors**: Shuo Lu, Yinuo Xu, Jianjie Cheng, Lingxiao He, Meng Wang, Jian Liang  

**Link**: [PDF](https://arxiv.org/pdf/2601.03261)  

**Abstract**: Deep Research agents predominantly optimize search policies to maximize retrieval probability. However, we identify a critical bottleneck: the retrieval-utilization gap, where models fail to use gold evidence even after it is retrieved, due to context blindness in noisy environments. To bridge this gap, we propose DeepResearch-Slice, a simple yet effective neuro-symbolic framework. Unlike implicit attention, our approach predicts precise span indices to perform a deterministic hard filter before reasoning. Extensive evaluations across six benchmarks show substantial robustness gains. Applying our method to frozen backbones yields a 73 percent relative improvement, from 19.1 percent to 33.0 percent, effectively mitigating noise without requiring parameter updates to the reasoning model. These results highlight the need for explicit grounding mechanisms in open-ended research. 

---
# Analyzing Reasoning Consistency in Large Multimodal Models under Cross-Modal Conflicts 

**Authors**: Zhihao Zhu, Jiafeng Liang, Shixin Jiang, Jinlan Fu, Ming Liu, Guanglu Sun, See-Kiong Ng, Bing Qin  

**Link**: [PDF](https://arxiv.org/pdf/2601.04073)  

**Abstract**: Large Multimodal Models (LMMs) have demonstrated impressive capabilities in video reasoning via Chain-of-Thought (CoT). However, the robustness of their reasoning chains remains questionable. In this paper, we identify a critical failure mode termed textual inertia, where once a textual hallucination occurs in the thinking process, models tend to blindly adhere to the erroneous text while neglecting conflicting visual evidence. To systematically investigate this, we propose the LogicGraph Perturbation Protocol that structurally injects perturbations into the reasoning chains of diverse LMMs spanning both native reasoning architectures and prompt-driven paradigms to evaluate their self-reflection capabilities. The results reveal that models successfully self-correct in less than 10% of cases and predominantly succumb to blind textual error propagation. To mitigate this, we introduce Active Visual-Context Refinement, a training-free inference paradigm which orchestrates an active visual re-grounding mechanism to enforce fine-grained verification coupled with an adaptive context refinement strategy to summarize and denoise the reasoning history. Experiments demonstrate that our approach significantly stifles hallucination propagation and enhances reasoning robustness. 

---
# Stable Language Guidance for Vision-Language-Action Models 

**Authors**: Zhihao Zhan, Yuhao Chen, Jiaying Zhou, Qinhan Lv, Hao Liu, Keze Wang, Liang Lin, Guangrun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.04052)  

**Abstract**: Vision-Language-Action (VLA) models have demonstrated impressive capabilities in generalized robotic control; however, they remain notoriously brittle to linguistic perturbations. We identify a critical ``modality collapse'' phenomenon where strong visual priors overwhelm sparse linguistic signals, causing agents to overfit to specific instruction phrasings while ignoring the underlying semantic intent. To address this, we propose \textbf{Residual Semantic Steering (RSS)}, a probabilistic framework that disentangles physical affordance from semantic execution. RSS introduces two theoretical innovations: (1) \textbf{Monte Carlo Syntactic Integration}, which approximates the true semantic posterior via dense, LLM-driven distributional expansion, and (2) \textbf{Residual Affordance Steering}, a dual-stream decoding mechanism that explicitly isolates the causal influence of language by subtracting the visual affordance prior. Theoretical analysis suggests that RSS effectively maximizes the mutual information between action and intent while suppressing visual distractors. Empirical results across diverse manipulation benchmarks demonstrate that RSS achieves state-of-the-art robustness, maintaining performance even under adversarial linguistic perturbations. 

---
# SoK: Privacy Risks and Mitigations in Retrieval-Augmented Generation Systems 

**Authors**: Andreea-Elena Bodea, Stephen Meisenbacher, Alexandra Klymenko, Florian Matthes  

**Link**: [PDF](https://arxiv.org/pdf/2601.03979)  

**Abstract**: The continued promise of Large Language Models (LLMs), particularly in their natural language understanding and generation capabilities, has driven a rapidly increasing interest in identifying and developing LLM use cases. In an effort to complement the ingrained "knowledge" of LLMs, Retrieval-Augmented Generation (RAG) techniques have become widely popular. At its core, RAG involves the coupling of LLMs with domain-specific knowledge bases, whereby the generation of a response to a user question is augmented with contextual and up-to-date information. The proliferation of RAG has sparked concerns about data privacy, particularly with the inherent risks that arise when leveraging databases with potentially sensitive information. Numerous recent works have explored various aspects of privacy risks in RAG systems, from adversarial attacks to proposed mitigations. With the goal of surveying and unifying these works, we ask one simple question: What are the privacy risks in RAG, and how can they be measured and mitigated? To answer this question, we conduct a systematic literature review of RAG works addressing privacy, and we systematize our findings into a comprehensive set of privacy risks, mitigation techniques, and evaluation strategies. We supplement these findings with two primary artifacts: a Taxonomy of RAG Privacy Risks and a RAG Privacy Process Diagram. Our work contributes to the study of privacy in RAG not only by conducting the first systematization of risks and mitigations, but also by uncovering important considerations when mitigating privacy risks in RAG systems and assessing the current maturity of proposed mitigations. 

---
# Muse: Towards Reproducible Long-Form Song Generation with Fine-Grained Style Control 

**Authors**: Changhao Jiang, Jiahao Chen, Zhenghao Xiang, Zhixiong Yang, Hanchen Wang, Jiabao Zhuang, Xinmeng Che, Jiajun Sun, Hui Li, Yifei Cao, Shihan Dou, Ming Zhang, Junjie Ye, Tao Ji, Tao Gui, Qi Zhang, Xuanjing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2601.03973)  

**Abstract**: Recent commercial systems such as Suno demonstrate strong capabilities in long-form song generation, while academic research remains largely non-reproducible due to the lack of publicly available training data, hindering fair comparison and progress. To this end, we release a fully open-source system for long-form song generation with fine-grained style conditioning, including a licensed synthetic dataset, training and evaluation pipelines, and Muse, an easy-to-deploy song generation model. The dataset consists of 116k fully licensed synthetic songs with automatically generated lyrics and style descriptions paired with audio synthesized by SunoV5. We train Muse via single-stage supervised finetuning of a Qwen-based language model extended with discrete audio tokens using MuCodec, without task-specific losses, auxiliary objectives, or additional architectural components. Our evaluations find that although Muse is trained with a modest data scale and model size, it achieves competitive performance on phoneme error rate, text--music style similarity, and audio aesthetic quality, while enabling controllable segment-level generation across different musical structures. All data, model weights, and training and evaluation pipelines will be publicly released, paving the way for continued progress in controllable long-form song generation research. The project repository is available at this https URL. 

---
# Anti-Length Shift: Dynamic Outlier Truncation for Training Efficient Reasoning Models 

**Authors**: Wei Wu, Liyi Chen, Congxi Xiao, Tianfu Wang, Qimeng Wang, Chengqiang Lu, Yan Gao, Yi Wu, Yao Hu, Hui Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2601.03969)  

**Abstract**: Large reasoning models enhanced by reinforcement learning with verifiable rewards have achieved significant performance gains by extending their chain-of-thought. However, this paradigm incurs substantial deployment costs as models often exhibit excessive verbosity on simple queries. Existing efficient reasoning methods relying on explicit length penalties often introduce optimization conflicts and leave the generative mechanisms driving overthinking largely unexamined. In this paper, we identify a phenomenon termed length shift where models increasingly generate unnecessary reasoning on trivial inputs during training. To address this, we introduce Dynamic Outlier Truncation (DOT), a training-time intervention that selectively suppresses redundant tokens. This method targets only the extreme tail of response lengths within fully correct rollout groups while preserving long-horizon reasoning capabilities for complex problems. To complement this intervention and ensure stable convergence, we further incorporate auxiliary KL regularization and predictive dynamic sampling. Experimental results across multiple model scales demonstrate that our approach significantly pushes the efficiency-performance Pareto frontier outward. Notably, on the AIME-24, our method reduces inference token usage by 78% while simultaneously increasing accuracy compared to the initial policy and surpassing state-of-the-art efficient reasoning methods. 

---
# FOREVER: Forgetting Curve-Inspired Memory Replay for Language Model Continual Learning 

**Authors**: Yujie Feng, Hao Wang, Jian Li, Xu Chu, Zhaolu Kang, Yiran Liu, Yasha Wang, Philip S. Yu, Xiao-Ming Wu  

**Link**: [PDF](https://arxiv.org/pdf/2601.03938)  

**Abstract**: Continual learning (CL) for large language models (LLMs) aims to enable sequential knowledge acquisition without catastrophic forgetting. Memory replay methods are widely used for their practicality and effectiveness, but most rely on fixed, step-based heuristics that often misalign with the model's actual learning progress, since identical training steps can result in varying degrees of parameter change. Motivated by recent findings that LLM forgetting mirrors the Ebbinghaus human forgetting curve, we propose FOREVER (FORgEtting curVe-inspired mEmory Replay), a novel CL framework that aligns replay schedules with a model-centric notion of time. FOREVER defines model time using the magnitude of optimizer updates, allowing forgetting curve-inspired replay intervals to align with the model's internal evolution rather than raw training steps. Building on this approach, FOREVER incorporates a forgetting curve-based replay scheduler to determine when to replay and an intensity-aware regularization mechanism to adaptively control how to replay. Extensive experiments on three CL benchmarks and models ranging from 0.6B to 13B parameters demonstrate that FOREVER consistently mitigates catastrophic forgetting. 

---
# FocusUI: Efficient UI Grounding via Position-Preserving Visual Token Selection 

**Authors**: Mingyu Ouyang, Kevin Qinghong Lin, Mike Zheng Shou, Hwee Tou Ng  

**Link**: [PDF](https://arxiv.org/pdf/2601.03928)  

**Abstract**: Vision-Language Models (VLMs) have shown remarkable performance in User Interface (UI) grounding tasks, driven by their ability to process increasingly high-resolution screenshots. However, screenshots are tokenized into thousands of visual tokens (e.g., about 4700 for 2K resolution), incurring significant computational overhead and diluting attention. In contrast, humans typically focus on regions of interest when interacting with UI. In this work, we pioneer the task of efficient UI grounding. Guided by practical analysis of the task's characteristics and challenges, we propose FocusUI, an efficient UI grounding framework that selects patches most relevant to the instruction while preserving positional continuity for precise grounding. FocusUI addresses two key challenges: (1) Eliminating redundant tokens in visual encoding. We construct patch-level supervision by fusing an instruction-conditioned score with a rule-based UI-graph score that down-weights large homogeneous regions to select distinct and instruction-relevant visual tokens. (2) Preserving positional continuity during visual token selection. We find that general visual token pruning methods suffer from severe accuracy degradation on UI grounding tasks due to broken positional information. We introduce a novel PosPad strategy, which compresses each contiguous sequence of dropped visual tokens into a single special marker placed at the sequence's last index to preserve positional continuity. Comprehensive experiments on four grounding benchmarks demonstrate that FocusUI surpasses GUI-specific baselines. On the ScreenSpot-Pro benchmark, FocusUI-7B achieves a performance improvement of 3.7% over GUI-Actor-7B. Even with only 30% visual token retention, FocusUI-7B drops by only 3.2% while achieving up to 1.44x faster inference and 17% lower peak GPU memory. 

---
# Current Agents Fail to Leverage World Model as Tool for Foresight 

**Authors**: Cheng Qian, Emre Can Acikgoz, Bingxuan Li, Xiusi Chen, Yuji Zhang, Bingxiang He, Qinyu Luo, Dilek Hakkani-Tür, Gokhan Tur, Yunzhu Li, Heng Ji, Heng Ji  

**Link**: [PDF](https://arxiv.org/pdf/2601.03905)  

**Abstract**: Agents built on vision-language models increasingly face tasks that demand anticipating future states rather than relying on short-horizon reasoning. Generative world models offer a promising remedy: agents could use them as external simulators to foresee outcomes before acting. This paper empirically examines whether current agents can leverage such world models as tools to enhance their cognition. Across diverse agentic and visual question answering tasks, we observe that some agents rarely invoke simulation (fewer than 1%), frequently misuse predicted rollouts (approximately 15%), and often exhibit inconsistent or even degraded performance (up to 5%) when simulation is available or enforced. Attribution analysis further indicates that the primary bottleneck lies in the agents' capacity to decide when to simulate, how to interpret predicted outcomes, and how to integrate foresight into downstream reasoning. These findings underscore the need for mechanisms that foster calibrated, strategic interaction with world models, paving the way toward more reliable anticipatory cognition in future agent systems. 

---
# Adaptive-Boundary-Clipping GRPO: Ensuring Bounded Ratios for Stable and Generalizable Training 

**Authors**: Chi Liu, Xin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2601.03895)  

**Abstract**: Group Relative Policy Optimization (GRPO) has emerged as a popular algorithm for reinforcement learning with large language models (LLMs). However, upon analyzing its clipping mechanism, we argue that it is suboptimal in certain scenarios. With appropriate modifications, GRPO can be significantly enhanced to improve both flexibility and generalization. To this end, we propose Adaptive-Boundary-Clipping GRPO (ABC-GRPO), an asymmetric and adaptive refinement of the original GRPO framework. We demonstrate that ABC-GRPO achieves superior performance over standard GRPO on mathematical reasoning tasks using the Qwen3 LLMs. Moreover, ABC-GRPO maintains substantially higher entropy throughout training, thereby preserving the model's exploration capacity and mitigating premature convergence. The implementation code is available online to ease reproducibility this https URL. 

---
# RadDiff: Describing Differences in Radiology Image Sets with Natural Language 

**Authors**: Xiaoxian Shen, Yuhui Zhang, Sahithi Ankireddy, Xiaohan Wang, Maya Varma, Henry Guo, Curtis Langlotz, Serena Yeung-Levy  

**Link**: [PDF](https://arxiv.org/pdf/2601.03733)  

**Abstract**: Understanding how two radiology image sets differ is critical for generating clinical insights and for interpreting medical AI systems. We introduce RadDiff, a multimodal agentic system that performs radiologist-style comparative reasoning to describe clinically meaningful differences between paired radiology studies. RadDiff builds on a proposer-ranker framework from VisDiff, and incorporates four innovations inspired by real diagnostic workflows: (1) medical knowledge injection through domain-adapted vision-language models; (2) multimodal reasoning that integrates images with their clinical reports; (3) iterative hypothesis refinement across multiple reasoning rounds; and (4) targeted visual search that localizes and zooms in on salient regions to capture subtle findings. To evaluate RadDiff, we construct RadDiffBench, a challenging benchmark comprising 57 expert-validated radiology study pairs with ground-truth difference descriptions. On RadDiffBench, RadDiff achieves 47% accuracy, and 50% accuracy when guided by ground-truth reports, significantly outperforming the general-domain VisDiff baseline. We further demonstrate RadDiff's versatility across diverse clinical tasks, including COVID-19 phenotype comparison, racial subgroup analysis, and discovery of survival-related imaging features. Together, RadDiff and RadDiffBench provide the first method-and-benchmark foundation for systematically uncovering meaningful differences in radiological data. 

---
# Sandwich Reasoning: An Answer-Reasoning-Answer Approach for Low-Latency Query Correction 

**Authors**: Chen Zhang, Kepu Zhang, Jiatong Zhang, Xiao Zhang, Jun Xu  

**Link**: [PDF](https://arxiv.org/pdf/2601.03672)  

**Abstract**: Query correction is a critical entry point in modern search pipelines, demanding high accuracy strictly within real-time latency constraints. Chain-of-Thought (CoT) reasoning improves accuracy but incurs prohibitive latency for real-time query correction. A potential solution is to output an answer before reasoning to reduce latency; however, under autoregressive decoding, the early answer is independent of subsequent reasoning, preventing the model from leveraging its reasoning capability to improve accuracy. To address this issue, we propose Sandwich Reasoning (SandwichR), a novel approach that explicitly aligns a fast initial answer with post-hoc reasoning, enabling low-latency query correction without sacrificing reasoning-aware accuracy. SandwichR follows an Answer-Reasoning-Answer paradigm, producing an initial correction, an explicit reasoning process, and a final refined correction. To align the initial answer with post-reasoning insights, we design a consistency-aware reinforcement learning (RL) strategy: a dedicated consistency reward enforces alignment between the initial and final corrections, while margin-based rejection sampling prioritizes borderline samples where reasoning drives the most impactful corrective gains. Additionally, we construct a high-quality query correction dataset, addressing the lack of specialized benchmarks for complex query correction. Experimental results demonstrate that SandwichR achieves SOTA accuracy comparable to standard CoT while delivering a 40-70% latency reduction, resolving the latency-accuracy trade-off in online search. 

---
# Controllable LLM Reasoning via Sparse Autoencoder-Based Steering 

**Authors**: Yi Fang, Wenjie Wang, Mingfeng Xue, Boyi Deng, Fengli Xu, Dayiheng Liu, Fuli Feng  

**Link**: [PDF](https://arxiv.org/pdf/2601.03595)  

**Abstract**: Large Reasoning Models (LRMs) exhibit human-like cognitive reasoning strategies (e.g. backtracking, cross-verification) during reasoning process, which improves their performance on complex tasks. Currently, reasoning strategies are autonomously selected by LRMs themselves. However, such autonomous selection often produces inefficient or even erroneous reasoning paths. To make reasoning more reliable and flexible, it is important to develop methods for controlling reasoning strategies. Existing methods struggle to control fine-grained reasoning strategies due to conceptual entanglement in LRMs' hidden states. To address this, we leverage Sparse Autoencoders (SAEs) to decompose strategy-entangled hidden states into a disentangled feature space. To identify the few strategy-specific features from the vast pool of SAE features, we propose SAE-Steering, an efficient two-stage feature identification pipeline. SAE-Steering first recalls features that amplify the logits of strategy-specific keywords, filtering out over 99\% of features, and then ranks the remaining features by their control effectiveness. Using the identified strategy-specific features as control vectors, SAE-Steering outperforms existing methods by over 15\% in control effectiveness. Furthermore, controlling reasoning strategies can redirect LRMs from erroneous paths to correct ones, achieving a 7\% absolute accuracy improvement. 

---
# EASLT: Emotion-Aware Sign Language Translation 

**Authors**: Guobin Tu, Di Weng  

**Link**: [PDF](https://arxiv.org/pdf/2601.03549)  

**Abstract**: Sign Language Translation (SLT) is a complex cross-modal task requiring the integration of Manual Signals (MS) and Non-Manual Signals (NMS). While recent gloss-free SLT methods have made strides in translating manual gestures, they frequently overlook the semantic criticality of facial expressions, resulting in ambiguity when distinct concepts share identical manual articulations. To address this, we present **EASLT** (**E**motion-**A**ware **S**ign **L**anguage **T**ranslation), a framework that treats facial affect not as auxiliary information, but as a robust semantic anchor. Unlike methods that relegate facial expressions to a secondary role, EASLT incorporates a dedicated emotional encoder to capture continuous affective dynamics. These representations are integrated via a novel *Emotion-Aware Fusion* (EAF) module, which adaptively recalibrates spatio-temporal sign features based on affective context to resolve semantic ambiguities. Extensive evaluations on the PHOENIX14T and CSL-Daily benchmarks demonstrate that EASLT establishes advanced performance among gloss-free methods, achieving BLEU-4 scores of 26.15 and 22.80, and BLEURT scores of 61.0 and 57.8, respectively. Ablation studies confirm that explicitly modeling emotion effectively decouples affective semantics from manual dynamics, significantly enhancing translation fidelity. Code is available at this https URL. 

---
# STAR-S: Improving Safety Alignment through Self-Taught Reasoning on Safety Rules 

**Authors**: Di Wu, Yanyan Zhao, Xin Lu, Mingzhe Li, Bing Qin  

**Link**: [PDF](https://arxiv.org/pdf/2601.03537)  

**Abstract**: Defending against jailbreak attacks is crucial for the safe deployment of Large Language Models (LLMs). Recent research has attempted to improve safety by training models to reason over safety rules before responding. However, a key issue lies in determining what form of safety reasoning effectively defends against jailbreak attacks, which is difficult to explicitly design or directly obtain. To address this, we propose \textbf{STAR-S} (\textbf{S}elf-\textbf{TA}ught \textbf{R}easoning based on \textbf{S}afety rules), a framework that integrates the learning of safety rule reasoning into a self-taught loop. The core of STAR-S involves eliciting reasoning and reflection guided by safety rules, then leveraging fine-tuning to enhance safety reasoning. Repeating this process creates a synergistic cycle. Improvements in the model's reasoning and interpretation of safety rules allow it to produce better reasoning data under safety rule prompts, which is then utilized for further training. Experiments show that STAR-S effectively defends against jailbreak attacks, outperforming baselines. Code is available at: this https URL. 

---
# STELLA: Self-Reflective Terminology-Aware Framework for Building an Aerospace Information Retrieval Benchmark 

**Authors**: Bongmin Kim  

**Link**: [PDF](https://arxiv.org/pdf/2601.03496)  

**Abstract**: Tasks in the aerospace industry heavily rely on searching and reusing large volumes of technical documents, yet there is no public information retrieval (IR) benchmark that reflects the terminology- and query-intent characteristics of this domain. To address this gap, this paper proposes the STELLA (Self-Reflective TErminoLogy-Aware Framework for BuiLding an Aerospace Information Retrieval Benchmark) framework. Using this framework, we introduce the STELLA benchmark, an aerospace-specific IR evaluation set constructed from NASA Technical Reports Server (NTRS) documents via a systematic pipeline that comprises document layout detection, passage chunking, terminology dictionary construction, synthetic query generation, and cross-lingual extension. The framework generates two types of queries: the Terminology Concordant Query (TCQ), which includes the terminology verbatim to evaluate lexical matching, and the Terminology Agnostic Query (TAQ), which utilizes the terminology's description to assess semantic matching. This enables a disentangled evaluation of the lexical and semantic matching capabilities of embedding models. In addition, we combine Chain-of-Density (CoD) and the Self-Reflection method with query generation to improve quality and implement a hybrid cross-lingual extension that reflects real user querying practices. Evaluation of seven embedding models on the STELLA benchmark shows that large decoder-based embedding models exhibit the strongest semantic understanding, while lexical matching methods such as BM25 remain highly competitive in domains where exact lexical matching technical term is crucial. The STELLA benchmark provides a reproducible foundation for reliable performance evaluation and improvement of embedding models in aerospace-domain IR tasks. The STELLA benchmark can be found in this https URL. 

---
# Content vs. Form: What Drives the Writing Score Gap Across Socioeconomic Backgrounds? A Generated Panel Approach 

**Authors**: Nadav Kunievsky, Pedro Pertusi  

**Link**: [PDF](https://arxiv.org/pdf/2601.03469)  

**Abstract**: Students from different socioeconomic backgrounds exhibit persistent gaps in test scores, gaps that can translate into unequal educational and labor-market outcomes later in life. In many assessments, performance reflects not only what students know, but also how effectively they can communicate that knowledge. This distinction is especially salient in writing assessments, where scores jointly reward the substance of students' ideas and the way those ideas are expressed. As a result, observed score gaps may conflate differences in underlying content with differences in expressive skill. A central question, therefore, is how much of the socioeconomic-status (SES) gap in scores is driven by differences in what students say versus how they say it. We study this question using a large corpus of persuasive essays written by U.S. middle- and high-school students. We introduce a new measurement strategy that separates content from style by leveraging large language models to generate multiple stylistic variants of each essay. These rewrites preserve the underlying arguments while systematically altering surface expression, creating a "generated panel" that introduces controlled within-essay variation in style. This approach allows us to decompose SES gaps in writing scores into contributions from content and style. We find an SES gap of 0.67 points on a 1-6 scale. Approximately 69% of the gap is attributable to differences in essay content quality, Style differences account for 26% of the gap, and differences in evaluation standards across SES groups account for the remaining 5%. These patterns seems stable across demographic subgroups and writing tasks. More broadly, our approach shows how large language models can be used to generate controlled variation in observational data, enabling researchers to isolate and quantify the contributions of otherwise entangled factors. 

---
# Spectral Archaeology: The Causal Topology of Model Evolution 

**Authors**: Valentin Noël  

**Link**: [PDF](https://arxiv.org/pdf/2601.03424)  

**Abstract**: Behavioral benchmarks tell us \textit{what} a model does, but not \textit{how}. We introduce a training-free mechanistic probe using attention-graph spectra. Treating each layer as a token graph, we compute algebraic connectivity ($\lambda_2$), smoothness, and spectral entropy. Across 12 models and 10 languages, these measures yield stable ``spectral fingerprints'' that expose discontinuities missed by standard evaluation.
We report four results. (1) Models undergoing specific curriculum transitions (e.g., code-to-chat) show an English-only, syntax-triggered connectivity failure on non-canonical constructions, reaching $\Delta\lambda_2 \approx -0.76$. We term this scar \textit{Passive-Triggered Connectivity Collapse} (PTCC). Analysis of the Phi lineage reveals that PTCC appears and resolves across developmental stages, implicating brittle curriculum shifts rather than synthetic data per se. (2) PTCC reflects a specialization trade-off: strengthened formal routing at the expense of stylistic flexibility. (3) We identify four recurrent processing strategies; simple frozen-threshold rules enable perfect forensic identification across lineages. (4) Mechanistically, PTCC localizes to a sparse Layer 2 ``compensatory patch'' of heads that fails under syntactic stress; activation steering can partially restore connectivity, recovering $\approx 38\%$ of lost information flow.
Finally, dominant topological regimes track tokenization density more than language identity, suggesting ``healthy'' geometry varies systematically across scripts. Overall, attention-graph spectra provide a practical tool for auditing and training-regime verification. 

---
# RiskCueBench: Benchmarking Anticipatory Reasoning from Early Risk Cues in Video-Language Models 

**Authors**: Sha Luo, Yogesh Prabhu, Tim Ossowski, Kaiping Chen, Junjie Hu  

**Link**: [PDF](https://arxiv.org/pdf/2601.03369)  

**Abstract**: With the rapid growth of video centered social media, the ability to anticipate risky events from visual data is a promising direction for ensuring public safety and preventing real world accidents. Prior work has extensively studied supervised video risk assessment across domains such as driving, protests, and natural disasters. However, many existing datasets provide models with access to the full video sequence, including the accident itself, which substantially reduces the difficulty of the task. To better reflect real world conditions, we introduce a new video understanding benchmark RiskCueBench in which videos are carefully annotated to identify a risk signal clip, defined as the earliest moment that indicates a potential safety concern. Experimental results reveal a significant gap in current systems ability to interpret evolving situations and anticipate future risky events from early visual signals, highlighting important challenges for deploying video risk prediction models in practice. 

---
# How Real is Your Jailbreak? Fine-grained Jailbreak Evaluation with Anchored Reference 

**Authors**: Songyang Liu, Chaozhuo Li, Rui Pu, Litian Zhang, Chenxu Wang, Zejian Chen, Yuting Zhang, Yiming Hei  

**Link**: [PDF](https://arxiv.org/pdf/2601.03288)  

**Abstract**: Jailbreak attacks present a significant challenge to the safety of Large Language Models (LLMs), yet current automated evaluation methods largely rely on coarse classifications that focus mainly on harmfulness, leading to substantial overestimation of attack success. To address this problem, we propose FJAR, a fine-grained jailbreak evaluation framework with anchored references. We first categorized jailbreak responses into five fine-grained categories: Rejective, Irrelevant, Unhelpful, Incorrect, and Successful, based on the degree to which the response addresses the malicious intent of the query. This categorization serves as the basis for FJAR. Then, we introduce a novel harmless tree decomposition approach to construct high-quality anchored references by breaking down the original queries. These references guide the evaluator in determining whether the response genuinely fulfills the original query. Extensive experiments demonstrate that FJAR achieves the highest alignment with human judgment and effectively identifies the root causes of jailbreak failures, providing actionable guidance for improving attack strategies. 

---
# HyperCLOVA X 32B Think 

**Authors**: NAVER Cloud HyperCLOVA X Team  

**Link**: [PDF](https://arxiv.org/pdf/2601.03286)  

**Abstract**: In this report, we present HyperCLOVA X 32B Think, a vision-language model designed with particular emphasis on reasoning within the Korean linguistic and cultural context, as well as agentic ability. HyperCLOVA X 32B Think is pre-trained with a strong focus on reasoning capabilities and subsequently post-trained to support multimodal understanding, enhanced reasoning, agentic behaviors, and alignment with human preferences. Experimental evaluations against comparably sized models demonstrate that our model achieves strong performance on Korean text-to-text and vision-to-text benchmarks, as well as on agent-oriented evaluation tasks. By open-sourcing HyperCLOVA X 32B Think, we aim to support broader adoption and facilitate further research and innovation across both academic and industrial communities. 

---
# MixRx: Predicting Drug Combination Interactions with LLMs 

**Authors**: Risha Surana, Cameron Saidock, Hugo Chacon  

**Link**: [PDF](https://arxiv.org/pdf/2601.03277)  

**Abstract**: MixRx uses Large Language Models (LLMs) to classify drug combination interactions as Additive, Synergistic, or Antagonistic, given a multi-drug patient history. We evaluate the performance of 4 models, GPT-2, Mistral Instruct 2.0, and the fine-tuned counterparts. Our results showed a potential for such an application, with the Mistral Instruct 2.0 Fine-Tuned model providing an average accuracy score on standard and perturbed datasets of 81.5%. This paper aims to further develop an upcoming area of research that evaluates if LLMs can be used for biological prediction tasks. 

---
# Roles of MLLMs in Visually Rich Document Retrieval for RAG: A Survey 

**Authors**: Xiantao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.03262)  

**Abstract**: Visually rich documents (VRDs) challenge retrieval-augmented generation (RAG) with layout-dependent semantics, brittle OCR, and evidence spread across complex figures and structured tables. This survey examines how Multimodal Large Language Models (MLLMs) are being used to make VRD retrieval practical for RAG. We organize the literature into three roles: Modality-Unifying Captioners, Multimodal Embedders, and End-to-End Representers. We compare these roles along retrieval granularity, information fidelity, latency and index size, and compatibility with reranking and grounding. We also outline key trade-offs and offer some practical guidance on when to favor each role. Finally, we identify promising directions for future research, including adaptive retrieval units, model size reduction, and the development of evaluation methods. 

---
# SciNetBench: A Relation-Aware Benchmark for Scientific Literature Retrieval Agents 

**Authors**: Chenyang Shao, Yong Li, Fengli Xu  

**Link**: [PDF](https://arxiv.org/pdf/2601.03260)  

**Abstract**: The rapid development of AI agent has spurred the development of advanced research tools, such as Deep Research. Achieving this require a nuanced understanding of the relations within scientific literature, surpasses the scope of keyword-based or embedding-based retrieval. Existing retrieval agents mainly focus on the content-level similarities and are unable to decode critical relational dynamics, such as identifying corroborating or conflicting studies or tracing technological lineages, all of which are essential for a comprehensive literature review. Consequently, this fundamental limitation often results in a fragmented knowledge structure, misleading sentiment interpretation, and inadequate modeling of collective scientific progress. To investigate relation-aware retrieval more deeply, we propose SciNetBench, the first Scientific Network Relation-aware Benchmark for literature retrieval agents. Constructed from a corpus of over 18 million AI papers, our benchmark systematically evaluates three levels of relations: ego-centric retrieval of papers with novel knowledge structures, pair-wise identification of scholarly relationships, and path-wise reconstruction of scientific evolutionary trajectories. Through extensive evaluation of three categories of retrieval agents, we find that their accuracy on relation-aware retrieval tasks often falls below 20%, revealing a core shortcoming of current retrieval paradigms. Notably, further experiments on the literature review tasks demonstrate that providing agents with relational ground truth leads to a substantial 23.4% performance improvement in the review quality, validating the critical importance of relation-aware retrieval. We publicly release our benchmark at this https URL to support future research on advanced retrieval systems. 

---
