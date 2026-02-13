# Agentic Test-Time Scaling for WebAgents 

**Authors**: Nicholas Lee, Lutfi Eren Erdogan, Chris Joseph John, Surya Krishnapillai, Michael W. Mahoney, Kurt Keutzer, Amir Gholami  

**Link**: [PDF](https://arxiv.org/pdf/2602.12276)  

**Abstract**: Test-time scaling has become a standard way to improve performance and boost reliability of neural network models. However, its behavior on agentic, multi-step tasks remains less well-understood: small per-step errors can compound over long horizons; and we find that naive policies that uniformly increase sampling show diminishing returns. In this work, we present CATTS, a simple technique for dynamically allocating compute for multi-step agents. We first conduct an empirical study of inference-time scaling for web agents. We find that uniformly increasing per-step compute quickly saturates in long-horizon environments. We then investigate stronger aggregation strategies, including an LLM-based Arbiter that can outperform naive voting, but that can overrule high-consensus decisions. We show that uncertainty statistics derived from the agent's own vote distribution (entropy and top-1/top-2 margin) correlate with downstream success and provide a practical signal for dynamic compute allocation. Based on these findings, we introduce Confidence-Aware Test-Time Scaling (CATTS), which uses vote-derived uncertainty to allocate compute only when decisions are genuinely contentious. CATTS improves performance on WebArena-Lite and GoBrowse by up to 9.1% over React while using up to 2.3x fewer tokens than uniform scaling, providing both efficiency gains and an interpretable decision rule. 

---
# CM2: Reinforcement Learning with Checklist Rewards for Multi-Turn and Multi-Step Agentic Tool Use 

**Authors**: Zhen Zhang, Kaiqiang Song, Xun Wang, Yebowen Hu, Weixiang Yan, Chenyang Zhao, Henry Peng Zou, Haoyun Deng, Sathish Reddy Indurthi, Shujian Liu, Simin Ma, Xiaoyang Wang, Xin Eric Wang, Song Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.12268)  

**Abstract**: AI agents are increasingly used to solve real-world tasks by reasoning over multi-turn user interactions and invoking external tools. However, applying reinforcement learning to such settings remains difficult: realistic objectives often lack verifiable rewards and instead emphasize open-ended behaviors; moreover, RL for multi-turn, multi-step agentic tool use is still underexplored; and building and maintaining executable tool environments is costly, limiting scale and coverage. We propose CM2, an RL framework that replaces verifiable outcome rewards with checklist rewards. CM2 decomposes each turn's intended behavior into fine-grained binary criteria with explicit evidence grounding and structured metadata, turning open-ended judging into more stable classification-style decisions. To balance stability and informativeness, our method adopts a strategy of sparse reward assignment but dense evaluation criteria. Training is performed in a scalable LLM-simulated tool environment, avoiding heavy engineering for large tool sets. Experiments show that CM2 consistently improves over supervised fine-tuning. Starting from an 8B Base model and training on an 8k-example RL dataset, CM2 improves over the SFT counterpart by 8 points on tau^-Bench, by 10 points on BFCL-V4, and by 12 points on ToolSandbox. The results match or even outperform similarly sized open-source baselines, including the judging model. CM2 thus provides a scalable recipe for optimizing multi-turn, multi-step tool-using agents without relying on verifiable rewards. Code provided by the open-source community: this https URL. 

---
# Think like a Scientist: Physics-guided LLM Agent for Equation Discovery 

**Authors**: Jianke Yang, Ohm Venkatachalam, Mohammad Kianezhad, Sharvaree Vadgama, Rose Yu  

**Link**: [PDF](https://arxiv.org/pdf/2602.12259)  

**Abstract**: Explaining observed phenomena through symbolic, interpretable formulas is a fundamental goal of science. Recently, large language models (LLMs) have emerged as promising tools for symbolic equation discovery, owing to their broad domain knowledge and strong reasoning capabilities. However, most existing LLM-based systems try to guess equations directly from data, without modeling the multi-step reasoning process that scientists often follow: first inferring physical properties such as symmetries, then using these as priors to restrict the space of candidate equations. We introduce KeplerAgent, an agentic framework that explicitly follows this scientific reasoning process. The agent coordinates physics-based tools to extract intermediate structure and uses these results to configure symbolic regression engines such as PySINDy and PySR, including their function libraries and structural constraints. Across a suite of physical equation benchmarks, KeplerAgent achieves substantially higher symbolic accuracy and greater robustness to noisy data than both LLM and traditional baselines. 

---
# "Sorry, I Didn't Catch That": How Speech Models Miss What Matters Most 

**Authors**: Kaitlyn Zhou, Martijn Bartelds, Federico Bianchi, James Zou  

**Link**: [PDF](https://arxiv.org/pdf/2602.12249)  

**Abstract**: Despite speech recognition systems achieving low word error rates on standard benchmarks, they often fail on short, high-stakes utterances in real-world deployments. Here, we study this failure mode in a high-stakes task: the transcription of U.S. street names as spoken by U.S. participants. We evaluate 15 models from OpenAI, Deepgram, Google, and Microsoft on recordings from linguistically diverse U.S. speakers and find an average transcription error rate of 44%. We quantify the downstream impact of failed transcriptions by geographic locations and show that mis-transcriptions systematically cause errors for all speakers, but that routing distance errors are twice as large for non-English primary speakers compared to English primary speakers. To mitigate this harm, we introduce a synthetic data generation approach that produces diverse pronunciations of named entities using open-source text-to-speech models. Fine-tuning with less than 1,000 synthetic samples improves street name transcription accuracy by nearly 60% (relative to base models) for non-English primary speakers. Our results highlight a critical gap between benchmark performance and real-world reliability in speech systems and demonstrate a simple, scalable path to reducing high-stakes transcription errors. 

---
# SAM3-LiteText: An Anatomical Study of the SAM3 Text Encoder for Efficient Vision-Language Segmentation 

**Authors**: Chengxi Zeng, Yuxuan Jiang, Ge Gao, Shuai Wang, Duolikun Danier, Bin Zhu, Stevan Rudinac, David Bull, Fan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.12173)  

**Abstract**: Vision-language segmentation models such as SAM3 enable flexible, prompt-driven visual grounding, but inherit large, general-purpose text encoders originally designed for open-ended language understanding. In practice, segmentation prompts are short, structured, and semantically constrained, leading to substantial over-provisioning in text encoder capacity and persistent computational and memory overhead. In this paper, we perform a large-scale anatomical analysis of text prompting in vision-language segmentation, covering 404,796 real prompts across multiple benchmarks. Our analysis reveals severe redundancy: most context windows are underutilized, vocabulary usage is highly sparse, and text embeddings lie on low-dimensional manifold despite high-dimensional representations. Motivated by these findings, we propose SAM3-LiteText, a lightweight text encoding framework that replaces the original SAM3 text encoder with a compact MobileCLIP student that is optimized by knowledge distillation. Extensive experiments on image and video segmentation benchmarks show that SAM3-LiteText reduces text encoder parameters by up to 88%, substantially reducing static memory footprint, while maintaining segmentation performance comparable to the original model. Code: this https URL. 

---
# Pedagogically-Inspired Data Synthesis for Language Model Knowledge Distillation 

**Authors**: Bowei He, Yankai Chen, Xiaokun Zhang, Linghe Kong, Philip S. Yu, Xue Liu, Chen Ma  

**Link**: [PDF](https://arxiv.org/pdf/2602.12172)  

**Abstract**: Knowledge distillation from Large Language Models (LLMs) to smaller models has emerged as a critical technique for deploying efficient AI systems. However, current methods for distillation via synthetic data lack pedagogical awareness, treating knowledge transfer as a one-off data synthesis and training task rather than a systematic learning process. In this paper, we propose a novel pedagogically-inspired framework for LLM knowledge distillation that draws from fundamental educational principles. Our approach introduces a three-stage pipeline -- Knowledge Identifier, Organizer, and Adapter (IOA) -- that systematically identifies knowledge deficiencies in student models, organizes knowledge delivery through progressive curricula, and adapts representations to match the cognitive capacity of student models. We integrate Bloom's Mastery Learning Principles and Vygotsky's Zone of Proximal Development to create a dynamic distillation process where student models approach teacher model's performance on prerequisite knowledge before advancing, and new knowledge is introduced with controlled, gradual difficulty increments. Extensive experiments using LLaMA-3.1/3.2 and Qwen2.5 as student models demonstrate that IOA achieves significant improvements over baseline distillation methods, with student models retaining 94.7% of teacher performance on DollyEval while using less than 1/10th of the parameters. Our framework particularly excels in complex reasoning tasks, showing 19.2% improvement on MATH and 22.3% on HumanEval compared with state-of-the-art baselines. 

---
# Statistical Parsing for Logical Information Retrieval 

**Authors**: Greg Coppola  

**Link**: [PDF](https://arxiv.org/pdf/2602.12170)  

**Abstract**: In previous work (Coppola, 2024) we introduced the Quantified Boolean Bayesian Network (QBBN), a logical graphical model that implements the forward fragment of natural deduction (Prawitz, 1965) as a probabilistic factor graph. That work left two gaps: no negation/backward reasoning, and no parser for natural language.
This paper addresses both gaps across inference, semantics, and syntax. For inference, we extend the QBBN with NEG factors enforcing P(x) + P(neg x) = 1, enabling contrapositive reasoning (modus tollens) via backward lambda messages, completing Prawitz's simple elimination rules. The engine handles 44/44 test cases spanning 22 reasoning patterns. For semantics, we present a typed logical language with role-labeled predicates, modal quantifiers, and three tiers of expressiveness following Prawitz: first-order quantification, propositions as arguments, and predicate quantification via lambda abstraction. For syntax, we present a typed slot grammar that deterministically compiles sentences to logical form (33/33 correct, zero ambiguity). LLMs handle disambiguation (95% PP attachment accuracy) but cannot produce structured parses directly (12.4% UAS), confirming grammars are necessary. The architecture: LLM preprocesses, grammar parses, LLM reranks, QBBN infers.
We argue this reconciles formal semantics with Sutton's "bitter lesson" (2019): LLMs eliminate the annotation bottleneck that killed formal NLP, serving as annotator while the QBBN serves as verifier. Code: this https URL 

---
# Sci-CoE: Co-evolving Scientific Reasoning LLMs via Geometric Consensus with Sparse Supervision 

**Authors**: Xiaohan He, Shiyang Feng, Songtao Huang, Lei Bai, Bin Wang, Bo Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.12164)  

**Abstract**: Large language models (LLMs) have demonstrated exceptional reasoning capabilities, and co-evolving paradigms have shown promising results in domains such as code and math. However, in scientific reasoning tasks, these models remain fragile due to unreliable solution evaluation and limited diversity in verification strategies. In this work, we propose Sci-CoE, a two-stage scientific co-evolving framework that enables models to self-evolve as both solver and verifier through a transition from sparse supervision to unsupervised learning. In the first stage, the model uses a small set of annotated data to establish fundamental correctness judgment anchors for the Verifier. In the second stage, we introduce a geometric reward mechanism that jointly considers consensus, reliability, and diversity, driving large-scale self-iteration on unlabeled data. Experiments on several general scientific benchmarks demonstrate that Sci-CoE enhances complex reasoning capabilities and exhibits strong scalability, facilitating the construction of more robust and diverse evaluation systems. Codes are available at this https URL. 

---
# GPT-4o Lacks Core Features of Theory of Mind 

**Authors**: John Muchovej, Amanda Royka, Shane Lee, Julian Jara-Ettinger  

**Link**: [PDF](https://arxiv.org/pdf/2602.12150)  

**Abstract**: Do Large Language Models (LLMs) possess a Theory of Mind (ToM)? Research into this question has focused on evaluating LLMs against benchmarks and found success across a range of social tasks. However, these evaluations do not test for the actual representations posited by ToM: namely, a causal model of mental states and behavior. Here, we use a cognitively-grounded definition of ToM to develop and test a new evaluation framework. Specifically, our approach probes whether LLMs have a coherent, domain-general, and consistent model of how mental states cause behavior -- regardless of whether that model matches a human-like ToM. We find that even though LLMs succeed in approximating human judgments in a simple ToM paradigm, they fail at a logically equivalent task and exhibit low consistency between their action predictions and corresponding mental state inferences. As such, these findings suggest that the social proficiency exhibited by LLMs is not the result of an domain-general or consistent ToM. 

---
# Seq2Seq2Seq: Lossless Data Compression via Discrete Latent Transformers and Reinforcement Learning 

**Authors**: Mahdi Khodabandeh, Ghazal Shabani, Arash Yousefi Jordehi, Seyed Abolghasem Mirroshandel  

**Link**: [PDF](https://arxiv.org/pdf/2602.12146)  

**Abstract**: Efficient lossless compression is essential for minimizing storage costs and transmission overhead while preserving data integrity. Traditional compression techniques, such as dictionary-based and statistical methods, often struggle to optimally exploit the structure and redundancy in complex data formats. Recent advancements in deep learning have opened new avenues for compression; however, many existing approaches depend on dense vector representations that obscure the underlying token structure. To address these limitations, we propose a novel lossless compression method that leverages Reinforcement Learning applied to a T5 language model architecture. This approach enables the compression of data into sequences of tokens rather than traditional vector representations. Unlike auto-encoders, which typically encode information into continuous latent spaces, our method preserves the token-based structure, aligning more closely with the original data format. This preservation allows for higher compression ratios while maintaining semantic integrity. By training the model using an off-policy Reinforcement Learning algorithm, we optimize sequence length to minimize redundancy and enhance compression efficiency. Our method introduces an efficient and adaptive data compression system built upon advanced Reinforcement Learning techniques, functioning independently of external grammatical or world knowledge. This approach shows significant improvements in compression ratios compared to conventional methods. By leveraging the latent information within language models, our system effectively compresses data without requiring explicit content understanding, paving the way for more robust and practical compression solutions across various applications. 

---
# STAR : Bridging Statistical and Agentic Reasoning for Large Model Performance Prediction 

**Authors**: Xiaoxiao Wang, Chunxiao Li, Junying Wang, Yijin Guo, Zijian Chen, Chunyi Li, Xiaohong Liu, Zicheng Zhang, Guangtao Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2602.12143)  

**Abstract**: As comprehensive large model evaluation becomes prohibitively expensive, predicting model performance from limited observations has become essential. However, existing statistical methods struggle with pattern shifts, data sparsity, and lack of explanation, while pure LLM methods remain unreliable. We propose STAR, a framework that bridges data-driven STatistical expectations with knowledge-driven Agentic Reasoning. STAR leverages specialized retrievers to gather external knowledge and embeds semantic features into Constrained Probabilistic Matrix Factorization (CPMF) to generate statistical expectations with uncertainty. A reasoning module guided by Expectation Violation Theory (EVT) then refines predictions through intra-family analysis, cross-model comparison, and credibility-aware aggregation, producing adjustments with traceable explanations. Extensive experiments show that STAR consistently outperforms all baselines on both score-based and rank-based metrics, delivering a 14.46% gain in total score over the strongest statistical method under extreme sparsity, with only 1--2 observed scores per test model. 

---
# Value Alignment Tax: Measuring Value Trade-offs in LLM Alignment 

**Authors**: Jiajun Chen, Hua Shen  

**Link**: [PDF](https://arxiv.org/pdf/2602.12134)  

**Abstract**: Existing work on value alignment typically characterizes value relations statically, ignoring how interventions - such as prompting, fine-tuning, or preference optimization - reshape the broader value system. We introduce the Value Alignment Tax (VAT), a framework that measures how alignment-induced changes propagate across interconnected values relative to achieved on-target gain. VAT captures the dynamics of value expression under alignment pressure. Using a controlled scenario-action dataset grounded in Schwartz value theory, we collect paired pre-post normative judgments and analyze alignment effects across models, values, and alignment strategies. Our results show that alignment often produces uneven, structured co-movement among values. These effects are invisible under conventional target-only evaluation, revealing systemic, process-level alignment risks and offering new insights into the dynamics of value alignment in LLMs. 

---
# Neutral Prompts, Non-Neutral People: Quantifying Gender and Skin-Tone Bias in Gemini Flash 2.5 Image and GPT Image 1.5 

**Authors**: Roberto Balestri  

**Link**: [PDF](https://arxiv.org/pdf/2602.12133)  

**Abstract**: This study quantifies gender and skin-tone bias in two widely deployed commercial image generators - Gemini Flash 2.5 Image (NanoBanana) and GPT Image 1.5 - to test the assumption that neutral prompts yield demographically neutral outputs. We generated 3,200 photorealistic images using four semantically neutral prompts. The analysis employed a rigorous pipeline combining hybrid color normalization, facial landmark masking, and perceptually uniform skin tone quantification using the Monk (MST), PERLA, and Fitzpatrick scales. Neutral prompts produced highly polarized defaults. Both models exhibited a strong "default white" bias (>96% of outputs). However, they diverged sharply on gender: Gemini favored female-presenting subjects, while GPT favored male-presenting subjects with lighter skin tones. This research provides a large-scale, comparative audit of state-of-the-art models using an illumination-aware colorimetric methodology, distinguishing aesthetic rendering from underlying pigmentation in synthetic imagery. The study demonstrates that neutral prompts function as diagnostic probes rather than neutral instructions. It offers a robust framework for auditing algorithmic visual culture and challenges the sociolinguistic assumption that unmarked language results in inclusive representation. 

---
# HLA: Hadamard Linear Attention 

**Authors**: Hanno Ackermann, Hong Cai, Mohsen Ghafoorian, Amirhossein Habibian  

**Link**: [PDF](https://arxiv.org/pdf/2602.12128)  

**Abstract**: The attention mechanism is an important reason for the success of transformers. It relies on computing pairwise relations between tokens. To reduce the high computational cost of standard quadratic attention, linear attention has been proposed as an efficient approximation. It employs kernel functions that are applied independently to the inputs before the pairwise similarities are calculated. That allows for an efficient computational procedure which, however, amounts to a low-degree rational function approximating softmax.
We propose Hadamard Linear Attention (HLA). Unlike previous works on linear attention, the nonlinearity in HLA is not applied separately to queries and keys, but, analogously to standard softmax attention, after the pairwise similarities have been computed. It will be shown that the proposed nonlinearity amounts to a higher-degree rational function to approximate softmax. An efficient computational scheme for the proposed method is derived that is similar to that of standard linear attention. In contrast to other approaches, no time-consuming tensor reshaping is necessary to apply the proposed algorithm. The effectiveness of the approach is demonstrated by applying it to a large diffusion transformer model for video generation, an application that involves very large amounts of tokens. 

---
# Commencing-Student Enrolment Forecasting Under Data Sparsity with Time Series Foundation Models 

**Authors**: Jittarin Jetwiriyanon, Teo Susnjak, Surangika Ranathunga  

**Link**: [PDF](https://arxiv.org/pdf/2602.12120)  

**Abstract**: Many universities face increasing financial pressure and rely on accurate forecasts of commencing enrolments. However, enrolment forecasting in higher education is often data-sparse; annual series are short and affected by reporting changes and regime shifts. Popular classical approaches can be unreliable, as parameter estimation and model selection are unstable with short samples, and structural breaks degrade extrapolation. Recently, TSFMs have provided zero-shot priors, delivering strong gains in annual, data-sparse institutional forecasting under leakage-disciplined covariate construction. We benchmark multiple TSFM families in a zero-shot setting and test a compact, leakage-safe covariate set and introduce the Institutional Operating Conditions Index (IOCI), a transferable 0-100 regime covariate derived from time-stamped documentary evidence available at each forecast origin, alongside Google Trends demand proxies with stabilising feature engineering. Using an expanding-window backtest with strict vintage alignment, covariate-conditioned TSFMs perform on par with classical benchmarks without institution-specific training, with performance differences varying by cohort and model. 

---
# Stop Unnecessary Reflection: Training LRMs for Efficient Reasoning with Adaptive Reflection and Length Coordinated Penalty 

**Authors**: Zewei Yu, Lirong Gao, Yuke Zhu, Bo Zheng, Sheng Guo, Haobo Wang, Junbo Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2602.12113)  

**Abstract**: Large Reasoning Models (LRMs) have demonstrated remarkable performance on complex reasoning tasks by employing test-time scaling. However, they often generate over-long chains-of-thought that, driven by substantial reflections such as repetitive self-questioning and circular reasoning, lead to high token consumption, substantial computational overhead, and increased latency without improving accuracy, particularly in smaller models. Our observation reveals that increasing problem complexity induces more excessive and unnecessary reflection, which in turn reduces accuracy and increases token overhead. To address this challenge, we propose Adaptive Reflection and Length Coordinated Penalty (ARLCP), a novel reinforcement learning framework designed to dynamically balance reasoning efficiency and solution accuracy. ARLCP introduces two key innovations: (1) a reflection penalty that adaptively curtails unnecessary reflective steps while preserving essential reasoning, and (2) a length penalty calibrated to the estimated complexity of the problem. By coordinating these penalties, ARLCP encourages the model to generate more concise and effective reasoning paths. We evaluate our method on five mathematical reasoning benchmarks using DeepSeek-R1-Distill-Qwen-1.5B and DeepSeek-R1-Distill-Qwen-7B models. Experimental results show that ARLCP achieves a superior efficiency-accuracy trade-off compared to existing approaches. For the 1.5B model, it reduces the average response length by 53.1% while simultaneously improving accuracy by 5.8%. For the 7B model, it achieves a 35.0% reduction in length with a 2.7% accuracy gain. The code is released at this https URL . 

---
# The Pensieve Paradigm: Stateful Language Models Mastering Their Own Context 

**Authors**: Xiaoyuan Liu, Tian Liang, Dongyang Ma, Deyu Zhou, Haitao Mi, Pinjia He, Yan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.12108)  

**Abstract**: In the world of Harry Potter, when Dumbledore's mind is overburdened, he extracts memories into a Pensieve to be revisited later. In the world of AI, while we possess the Pensieve-mature databases and retrieval systems, our models inexplicably lack the "wand" to operate it. They remain like a Dumbledore without agency, passively accepting a manually engineered context as their entire memory. This work finally places the wand in the model's hand. We introduce StateLM, a new class of foundation models endowed with an internal reasoning loop to manage their own state. We equip our model with a suite of memory tools, such as context pruning, document indexing, and note-taking, and train it to actively manage these tools. By learning to dynamically engineering its own context, our model breaks free from the architectural prison of a fixed window. Experiments across various model sizes demonstrate StateLM's effectiveness across diverse scenarios. On long-document QA tasks, StateLMs consistently outperform standard LLMs across all model scales; on the chat memory task, they achieve absolute accuracy improvements of 10% to 20% over standard LLMs. On the deep research task BrowseComp-Plus, the performance gap becomes even more pronounced: StateLM achieves up to 52% accuracy, whereas standard LLM counterparts struggle around 5%. Ultimately, our approach shifts LLMs from passive predictors to state-aware agents where reasoning becomes a stateful and manageable process. 

---
# Differentiable Modal Logic for Multi-Agent Diagnosis, Orchestration and Communication 

**Authors**: Antonin Sulc  

**Link**: [PDF](https://arxiv.org/pdf/2602.12083)  

**Abstract**: As multi-agent AI systems evolve from simple chatbots to autonomous swarms, debugging semantic failures requires reasoning about knowledge, belief, causality, and obligation, precisely what modal logic was designed to formalize. However, traditional modal logic requires manual specification of relationship structures that are unknown or dynamic in real systems. This tutorial demonstrates differentiable modal logic (DML), implemented via Modal Logical Neural Networks (MLNNs), enabling systems to learn trust networks, causal chains, and regulatory boundaries from behavioral data alone.
We present a unified neurosymbolic debugging framework through four modalities: epistemic (who to trust), temporal (when events cause failures), deontic (what actions are permitted), and doxastic (how to interpret agent confidence). Each modality is demonstrated on concrete multi-agent scenarios, from discovering deceptive alliances in diplomacy games to detecting LLM hallucinations, with complete implementations showing how logical contradictions become learnable optimization objectives. Key contributions for the neurosymbolic community: (1) interpretable learned structures where trust and causality are explicit parameters, not opaque embeddings; (2) knowledge injection via differentiable axioms that guide learning with sparse data (3) compositional multi-modal reasoning that combines epistemic, temporal, and deontic constraints; and (4) practical deployment patterns for monitoring, active control and communication of multi-agent systems. All code provided as executable Jupyter notebooks. 

---
# Tiny Recursive Reasoning with Mamba-2 Attention Hybrid 

**Authors**: Wenlong Wang, Fergal Reid  

**Link**: [PDF](https://arxiv.org/pdf/2602.12078)  

**Abstract**: Recent work on recursive reasoning models like TRM demonstrates that tiny networks (7M parameters) can achieve strong performance on abstract reasoning tasks through latent recursion -- iterative refinement in hidden representation space without emitting intermediate tokens. This raises a natural question about operator choice: Mamba-2's state space recurrence is itself a form of iterative refinement, making it a natural candidate for recursive reasoning -- but does introducing Mamba-2 into the recursive scaffold preserve reasoning capability? We investigate this by replacing the Transformer blocks in TRM with Mamba-2 hybrid operators while maintaining parameter parity (6.83M vs 6.86M parameters). On ARC-AGI-1, we find that the hybrid improves pass@2 (the official metric) by +2.0\% (45.88\% vs 43.88\%) and consistently outperforms at higher K values (+4.75\% at pass@100), whilst maintaining pass@1 parity. This suggests improved candidate coverage -- the model generates correct solutions more reliably -- with similar top-1 selection. Our results validate that Mamba-2 hybrid operators preserve reasoning capability within the recursive scaffold, establishing SSM-based operators as viable candidates in the recursive operator design space and taking a first step towards understanding the best mixing strategies for recursive reasoning. 

---
# LawThinker: A Deep Research Legal Agent in Dynamic Environments 

**Authors**: Xinyu Yang, Chenlong Deng, Tongyu Wen, Binyu Xie, Zhicheng Dou  

**Link**: [PDF](https://arxiv.org/pdf/2602.12056)  

**Abstract**: Legal reasoning requires not only correct outcomes but also procedurally compliant reasoning processes. However, existing methods lack mechanisms to verify intermediate reasoning steps, allowing errors such as inapplicable statute citations to propagate undetected through the reasoning chain. To address this, we propose LawThinker, an autonomous legal research agent that adopts an Explore-Verify-Memorize strategy for dynamic judicial environments. The core idea is to enforce verification as an atomic operation after every knowledge exploration step. A DeepVerifier module examines each retrieval result along three dimensions of knowledge accuracy, fact-law relevance, and procedural compliance, with a memory module for cross-round knowledge reuse in long-horizon tasks. Experiments on the dynamic benchmark J1-EVAL show that LawThinker achieves a 24% improvement over direct reasoning and an 11% gain over workflow-based methods, with particularly strong improvements on process-oriented metrics. Evaluations on three static benchmarks further confirm its generalization capability. The code is available at this https URL . 

---
# Multi UAVs Preflight Planning in a Shared and Dynamic Airspace 

**Authors**: Amath Sow, Mauricio Rodriguez Cesen, Fabiola Martins Campos de Oliveira, Mariusz Wzorek, Daniel de Leng, Mattias Tiger, Fredrik Heintz, Christian Esteve Rothenberg  

**Link**: [PDF](https://arxiv.org/pdf/2602.12055)  

**Abstract**: Preflight planning for large-scale Unmanned Aerial Vehicle (UAV) fleets in dynamic, shared airspace presents significant challenges, including temporal No-Fly Zones (NFZs), heterogeneous vehicle profiles, and strict delivery deadlines. While Multi-Agent Path Finding (MAPF) provides a formal framework, existing methods often lack the scalability and flexibility required for real-world Unmanned Traffic Management (UTM). We propose DTAPP-IICR: a Delivery-Time Aware Prioritized Planning method with Incremental and Iterative Conflict Resolution. Our framework first generates an initial solution by prioritizing missions based on urgency. Secondly, it computes roundtrip trajectories using SFIPP-ST, a novel 4D single-agent planner (Safe Flight Interval Path Planning with Soft and Temporal Constraints). SFIPP-ST handles heterogeneous UAVs, strictly enforces temporal NFZs, and models inter-agent conflicts as soft constraints. Subsequently, an iterative Large Neighborhood Search, guided by a geometric conflict graph, efficiently resolves any residual conflicts. A completeness-preserving directional pruning technique further accelerates the 3D search. On benchmarks with temporal NFZs, DTAPP-IICR achieves near-100% success with fleets of up to 1,000 UAVs and gains up to 50% runtime reduction from pruning, outperforming batch Enhanced Conflict-Based Search in the UTM context. Scaling successfully in realistic city-scale operations where other priority-based methods fail even at moderate deployments, DTAPP-IICR is positioned as a practical and scalable solution for preflight planning in dense, dynamic urban airspace. 

---
# InjectRBP: Steering Large Language Model Reasoning Behavior via Pattern Injection 

**Authors**: Xiuping Wu, Zhao Yu, Yuxin Cheng, Ngai Wong, Liangjun Ke, Tapas Mishra, Konstantinos V.Katsikopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2602.12013)  

**Abstract**: Reasoning can significantly enhance the performance of Large Language Models. While recent studies have exploited behavior-related prompts adjustment to enhance reasoning, these designs remain largely intuitive and lack a systematic analysis of the underlying behavioral patterns. Motivated by this, we investigate how models' reasoning behaviors shape reasoning from the perspective of behavioral patterns. We observe that models exhibit adaptive distributions of reasoning behaviors when responding to specific types of questions, and that structurally injecting these patterns can substantially influence the quality of the models' reasoning processes and outcomes. Building on these findings, we propose two optimization methods that require no parameter updates: InjectCorrect and InjectRLOpt. InjectCorrect guides the model by imitating behavioral patterns derived from its own past correct answers. InjectRLOpt learns a value function from historical behavior-pattern data and, via our proposed Reliability-Aware Softmax Policy, generates behavioral injectant during inference to steer the reasoning process. Our experiments demonstrate that both methods can improve model performance across various reasoning tasks without requiring any modifications to model parameters, achieving gains of up to 5.34% and 8.67%, respectively. 

---
# CSEval: A Framework for Evaluating Clinical Semantics in Text-to-Image Generation 

**Authors**: Robert Cronshaw, Konstantinos Vilouras, Junyu Yan, Yuning Du, Feng Chen, Steven McDonagh, Sotirios A. Tsaftaris  

**Link**: [PDF](https://arxiv.org/pdf/2602.12004)  

**Abstract**: Text-to-image generation has been increasingly applied in medical domains for various purposes such as data augmentation and education. Evaluating the quality and clinical reliability of these generated images is essential. However, existing methods mainly assess image realism or diversity, while failing to capture whether the generated images reflect the intended clinical semantics, such as anatomical location and pathology. In this study, we propose the Clinical Semantics Evaluator (CSEval), a framework that leverages language models to assess clinical semantic alignment between the generated images and their conditioning prompts. Our experiments show that CSEval identifies semantic inconsistencies overlooked by other metrics and correlates with expert judgment. CSEval provides a scalable and clinically meaningful complement to existing evaluation methods, supporting the safe adoption of generative models in healthcare. 

---
# Gaia2: Benchmarking LLM Agents on Dynamic and Asynchronous Environments 

**Authors**: Romain Froger, Pierre Andrews, Matteo Bettini, Amar Budhiraja, Ricardo Silveira Cabral, Virginie Do, Emilien Garreau, Jean-Baptiste Gaya, Hugo Laurençon, Maxime Lecanu, Kunal Malkan, Dheeraj Mekala, Pierre Ménard, Gerard Moreno-Torres Bertran, Ulyana Piterbarg, Mikhail Plekhanov, Mathieu Rita, Andrey Rusakov, Vladislav Vorotilov, Mengjue Wang, Ian Yu, Amine Benhalloum, Grégoire Mialon, Thomas Scialom  

**Link**: [PDF](https://arxiv.org/pdf/2602.11964)  

**Abstract**: We introduce Gaia2, a benchmark for evaluating large language model agents in realistic, asynchronous environments. Unlike prior static or synchronous evaluations, Gaia2 introduces scenarios where environments evolve independently of agent actions, requiring agents to operate under temporal constraints, adapt to noisy and dynamic events, resolve ambiguity, and collaborate with other agents. Each scenario is paired with a write-action verifier, enabling fine-grained, action-level evaluation and making Gaia2 directly usable for reinforcement learning from verifiable rewards. Our evaluation of state-of-the-art proprietary and open-source models shows that no model dominates across capabilities: GPT-5 (high) reaches the strongest overall score of 42% pass@1 but fails on time-sensitive tasks, Claude-4 Sonnet trades accuracy and speed for cost, Kimi-K2 leads among open-source models with 21% pass@1. These results highlight fundamental trade-offs between reasoning, efficiency, robustness, and expose challenges in closing the "sim2real" gap. Gaia2 is built on a consumer environment with the open-source Agents Research Environments platform and designed to be easy to extend. By releasing Gaia2 alongside the foundational ARE framework, we aim to provide the community with a flexible infrastructure for developing, benchmarking, and training the next generation of practical agent systems. 

---
# MEME: Modeling the Evolutionary Modes of Financial Markets 

**Authors**: Taian Guo, Haiyang Shen, Junyu Luo, Zhongshi Xing, Hanchun Lian, Jinsheng Huang, Binqi Chen, Luchen Liu, Yun Ma, Ming Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.11918)  

**Abstract**: LLMs have demonstrated significant potential in quantitative finance by processing vast unstructured data to emulate human-like analytical workflows. However, current LLM-based methods primarily follow either an Asset-Centric paradigm focused on individual stock prediction or a Market-Centric approach for portfolio allocation, often remaining agnostic to the underlying reasoning that drives market movements. In this paper, we propose a Logic-Oriented perspective, modeling the financial market as a dynamic, evolutionary ecosystem of competing investment narratives, termed Modes of Thought. To operationalize this view, we introduce MEME (Modeling the Evolutionary Modes of Financial Markets), designed to reconstruct market dynamics through the lens of evolving logics. MEME employs a multi-agent extraction module to transform noisy data into high-fidelity Investment Arguments and utilizes Gaussian Mixture Modeling to uncover latent consensus within a semantic space. To model semantic drift among different market conditions, we also implement a temporal evaluation and alignment mechanism to track the lifecycle and historical profitability of these modes. By prioritizing enduring market wisdom over transient anomalies, MEME ensures that portfolio construction is guided by robust reasoning. Extensive experiments on three heterogeneous Chinese stock pools from 2023 to 2025 demonstrate that MEME consistently outperforms seven SOTA baselines. Further ablation studies, sensitivity analysis, lifecycle case study and cost analysis validate MEME's capacity to identify and adapt to the evolving consensus of financial markets. Our implementation can be found at this https URL. 

---
# AlphaPROBE: Alpha Mining via Principled Retrieval and On-graph biased evolution 

**Authors**: Taian Guo, Haiyang Shen, Junyu Luo, Binqi Chen, Hongjun Ding, Jinsheng Huang, Luchen Liu, Yun Ma, Ming Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.11917)  

**Abstract**: Extracting signals through alpha factor mining is a fundamental challenge in quantitative finance. Existing automated methods primarily follow two paradigms: Decoupled Factor Generation, which treats factor discovery as isolated events, and Iterative Factor Evolution, which focuses on local parent-child refinements. However, both paradigms lack a global structural view, often treating factor pools as unstructured collections or fragmented chains, which leads to redundant search and limited diversity. To address these limitations, we introduce AlphaPROBE (Alpha Mining via Principled Retrieval and On-graph Biased Evolution), a framework that reframes alpha mining as the strategic navigation of a Directed Acyclic Graph (DAG). By modeling factors as nodes and evolutionary links as edges, AlphaPROBE treats the factor pool as a dynamic, interconnected ecosystem. The framework consists of two core components: a Bayesian Factor Retriever that identifies high-potential seeds by balancing exploitation and exploration through a posterior probability model, and a DAG-aware Factor Generator that leverages the full ancestral trace of factors to produce context-aware, nonredundant optimizations. Extensive experiments on three major Chinese stock market datasets against 8 competitive baselines demonstrate that AlphaPROBE significantly gains enhanced performance in predictive accuracy, return stability and training efficiency. Our results confirm that leveraging global evolutionary topology is essential for efficient and robust automated alpha discovery. We have open-sourced our implementation at this https URL. 

---
# When Should LLMs Be Less Specific? Selective Abstraction for Reliable Long-Form Text Generation 

**Authors**: Shani Goren, Ido Galil, Ran El-Yaniv  

**Link**: [PDF](https://arxiv.org/pdf/2602.11908)  

**Abstract**: LLMs are widely used, yet they remain prone to factual errors that erode user trust and limit adoption in high-risk settings. One approach to mitigate this risk is to equip models with uncertainty estimation mechanisms that abstain when confidence is low. However, this binary "all-or-nothing" approach is excessively restrictive in long-form settings, often discarding valuable information. We introduce Selective Abstraction (SA), a framework that enables LLMs to trade specificity for reliability by selectively reducing the detail of uncertain content. We first formalize SA through the lenses of selective risk and coverage. We then propose Atom-wise Selective Abstraction, a claim-level instantiation that decomposes responses into atomic claims (short, self-contained statements each expressing a single fact) and replaces uncertain atoms with higher confidence, less specific abstractions. To evaluate this framework, we develop a novel end-to-end pipeline for open-ended generation that instantiates risk as factual correctness and measures coverage using an information-theoretic measure of retained information. Across six open-source models on the FactScore and LongFact-Objects benchmarks, atom-wise SA consistently outperforms existing baselines, improving the area under the risk-coverage curve (AURC) by up to 27.73% over claim removal, demonstrating that reducing specificity can boost accuracy and reliability while preserving most of their original meaning. 

---
# From Atoms to Trees: Building a Structured Feature Forest with Hierarchical Sparse Autoencoders 

**Authors**: Yifan Luo, Yang Zhan, Jiedong Jiang, Tianyang Liu, Mingrui Wu, Zhennan Zhou, Bin Dong  

**Link**: [PDF](https://arxiv.org/pdf/2602.11881)  

**Abstract**: Sparse autoencoders (SAEs) have proven effective for extracting monosemantic features from large language models (LLMs), yet these features are typically identified in isolation. However, broad evidence suggests that LLMs capture the intrinsic structure of natural language, where the phenomenon of "feature splitting" in particular indicates that such structure is hierarchical. To capture this, we propose the Hierarchical Sparse Autoencoder (HSAE), which jointly learns a series of SAEs and the parent-child relationships between their features. HSAE strengthens the alignment between parent and child features through two novel mechanisms: a structural constraint loss and a random feature perturbation mechanism. Extensive experiments across various LLMs and layers demonstrate that HSAE consistently recovers semantically meaningful hierarchies, supported by both qualitative case studies and rigorous quantitative metrics. At the same time, HSAE preserves the reconstruction fidelity and interpretability of standard SAEs across different dictionary sizes. Our work provides a powerful, scalable tool for discovering and analyzing the multi-scale conceptual structures embedded in LLM representations. 

---
# Intelligent AI Delegation 

**Authors**: Nenad Tomašev, Matija Franklin, Simon Osindero  

**Link**: [PDF](https://arxiv.org/pdf/2602.11865)  

**Abstract**: AI agents are able to tackle increasingly complex tasks. To achieve more ambitious goals, AI agents need to be able to meaningfully decompose problems into manageable sub-components, and safely delegate their completion across to other AI agents and humans alike. Yet, existing task decomposition and delegation methods rely on simple heuristics, and are not able to dynamically adapt to environmental changes and robustly handle unexpected failures. Here we propose an adaptive framework for intelligent AI delegation - a sequence of decisions involving task allocation, that also incorporates transfer of authority, responsibility, accountability, clear specifications regarding roles and boundaries, clarity of intent, and mechanisms for establishing trust between the two (or more) parties. The proposed framework is applicable to both human and AI delegators and delegatees in complex delegation networks, aiming to inform the development of protocols in the emerging agentic web. 

---
# Talk2DM: Enabling Natural Language Querying and Commonsense Reasoning for Vehicle-Road-Cloud Integrated Dynamic Maps with Large Language Models 

**Authors**: Lu Tao, Jinxuan Luo, Yousuke Watanabe, Zhengshu Zhou, Yuhuan Lu, Shen Ying, Pan Zhang, Fei Zhao, Hiroaki Takada  

**Link**: [PDF](https://arxiv.org/pdf/2602.11860)  

**Abstract**: Dynamic maps (DM) serve as the fundamental information infrastructure for vehicle-road-cloud (VRC) cooperative autonomous driving in China and Japan. By providing comprehensive traffic scene representations, DM overcome the limitations of standalone autonomous driving systems (ADS), such as physical occlusions. Although DM-enhanced ADS have been successfully deployed in real-world applications in Japan, existing DM systems still lack a natural-language-supported (NLS) human interface, which could substantially enhance human-DM interaction. To address this gap, this paper introduces VRCsim, a VRC cooperative perception (CP) simulation framework designed to generate streaming VRC-CP data. Based on VRCsim, we construct a question-answering data set, VRC-QA, focused on spatial querying and reasoning in mixed-traffic scenes. Building upon VRCsim and VRC-QA, we further propose Talk2DM, a plug-and-play module that extends VRC-DM systems with NLS querying and commonsense reasoning capabilities. Talk2DM is built upon a novel chain-of-prompt (CoP) mechanism that progressively integrates human-defined rules with the commonsense knowledge of large language models (LLMs). Experiments on VRC-QA show that Talk2DM can seamlessly switch across different LLMs while maintaining high NLS query accuracy, demonstrating strong generalization capability. Although larger models tend to achieve higher accuracy, they incur significant efficiency degradation. Our results reveal that Talk2DM, powered by Qwen3:8B, Gemma3:27B, and GPT-oss models, achieves over 93\% NLS query accuracy with an average response time of only 2-5 seconds, indicating strong practical potential. 

---
# Prototype Transformer: Towards Language Model Architectures Interpretable by Design 

**Authors**: Yordan Yordanov, Matteo Forasassi, Bayar Menzat, Ruizhi Wang, Chang Qi, Markus Kaltenberger, Amine M'Charrak, Tommaso Salvatori, Thomas Lukasiewicz  

**Link**: [PDF](https://arxiv.org/pdf/2602.11852)  

**Abstract**: While state-of-the-art language models (LMs) surpass the vast majority of humans in certain domains, their reasoning remains largely opaque, undermining trust in their output. Furthermore, while autoregressive LMs can output explicit reasoning, their true reasoning process is opaque, which introduces risks like deception and hallucination. In this work, we introduce the Prototype Transformer (ProtoT) -- an autoregressive LM architecture based on prototypes (parameter vectors), posed as an alternative to the standard self-attention-based transformers. ProtoT works by means of two-way communication between the input sequence and the prototypes, and we show that this leads to the prototypes automatically capturing nameable concepts (e.g. "woman") during training. They provide the potential to interpret the model's reasoning and allow for targeted edits of its behavior. Furthermore, by design, the prototypes create communication channels that aggregate contextual information at different time scales, aiding interpretability. In terms of computation scalability, ProtoT scales linearly with sequence length vs the quadratic scalability of SOTA self-attention transformers. Compared to baselines, ProtoT scales well with model and data size, and performs well on text generation and downstream tasks (GLUE). ProtoT exhibits robustness to input perturbations on par or better than some baselines, but differs from them by providing interpretable pathways showing how robustness and sensitivity arises. Reaching close to the performance of state-of-the-art architectures, ProtoT paves the way to creating well-performing autoregressive LMs interpretable by design. 

---
# Revis: Sparse Latent Steering to Mitigate Object Hallucination in Large Vision-Language Models 

**Authors**: Jialin Wu, Wei Shi, Han Shen, Peigui Qi, Kunsheng Tang, Zhicong Huang, Binghao Wang, Zhou Yang  

**Link**: [PDF](https://arxiv.org/pdf/2602.11824)  

**Abstract**: Despite the advanced capabilities of Large Vision-Language Models (LVLMs), they frequently suffer from object hallucination. One reason is that visual features and pretrained textual representations often become intertwined in the deeper network layers. To address this, we propose REVIS, a training-free framework designed to explicitly re-activate this suppressed visual information. Rooted in latent space geometry, REVIS extracts the pure visual information vector via orthogonal projection and employs a calibrated strategy to perform sparse intervention only at the precise depth where suppression occurs. This surgical approach effectively restores visual information with minimal computational cost. Empirical evaluations on standard benchmarks demonstrate that REVIS reduces object hallucination rates by approximately 19% compared to state-of-the-art baselines, while preserving general reasoning capabilities. 

---
# Predicting LLM Output Length via Entropy-Guided Representations 

**Authors**: Huanyi Xie, Yubin Chen, Liangyu Wang, Lijie Hu, Di Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.11812)  

**Abstract**: The long-tailed distribution of sequence lengths in LLM serving and reinforcement learning (RL) sampling causes significant computational waste due to excessive padding in batched inference. Existing methods rely on auxiliary models for static length prediction, but they incur high overhead, generalize poorly, and fail in stochastic "one-to-many" sampling scenarios. We introduce a lightweight framework that reuses the main model's internal hidden states for efficient length prediction. Our framework features two core components: 1) Entropy-Guided Token Pooling (EGTP), which uses on-the-fly activations and token entropy for highly accurate static prediction with negligible cost, and 2) Progressive Length Prediction (PLP), which dynamically estimates the remaining length at each decoding step to handle stochastic generation. To validate our approach, we build and release ForeLen, a comprehensive benchmark with long-sequence, Chain-of-Thought, and RL data. On ForeLen, EGTP achieves state-of-the-art accuracy, reducing MAE by 29.16\% over the best baseline. Integrating our methods with a length-aware scheduler yields significant end-to-end throughput gains. Our work provides a new technical and evaluation baseline for efficient LLM inference. 

---
# PuYun-LDM: A Latent Diffusion Model for High-Resolution Ensemble Weather Forecasts 

**Authors**: Lianjun Wu, Shengchen Zhu, Yuxuan Liu, Liuyu Kai, Xiaoduan Feng, Duomin Wang, Wenshuo Liu, Jingxuan Zhang, Kelvin Li, Bin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.11807)  

**Abstract**: Latent diffusion models (LDMs) suffer from limited diffusability in high-resolution (<=0.25°) ensemble weather forecasting, where diffusability characterizes how easily a latent data distribution can be modeled by a diffusion process. Unlike natural image fields, meteorological fields lack task-agnostic foundation models and explicit semantic structures, making VFM-based regularization inapplicable. Moreover, existing frequency-based approaches impose identical spectral regularization across channels under a homogeneity assumption, which leads to uneven regularization strength under the inter-variable spectral heterogeneity in multivariate meteorological data. To address these challenges, we propose a 3D Masked AutoEncoder (3D-MAE) that encodes weather-state evolution features as an additional conditioning for the diffusion model, together with a Variable-Aware Masked Frequency Modeling (VA-MFM) strategy that adaptively selects thresholds based on the spectral energy distribution of each variable. Together, we propose PuYun-LDM, which enhances latent diffusability and achieves superior performance to ENS at short lead times while remaining comparable to ENS at longer horizons. PuYun-LDM generates a 15-day global forecast with a 6-hour temporal resolution in five minutes on a single NVIDIA H200 GPU, while ensemble forecasts can be efficiently produced in parallel. 

---
# Hi-SAM: A Hierarchical Structure-Aware Multi-modal Framework for Large-Scale Recommendation 

**Authors**: Pingjun Pan, Tingting Zhou, Peiyao Lu, Tingting Fei, Hongxiang Chen, Chuanjiang Luo  

**Link**: [PDF](https://arxiv.org/pdf/2602.11799)  

**Abstract**: Multi-modal recommendation has gained traction as items possess rich attributes like text and images. Semantic ID-based approaches effectively discretize this information into compact tokens. However, two challenges persist: (1) Suboptimal Tokenization: existing methods (e.g., RQ-VAE) lack disentanglement between shared cross-modal semantics and modality-specific details, causing redundancy or collapse; (2) Architecture-Data Mismatch: vanilla Transformers treat semantic IDs as flat streams, ignoring the hierarchy of user interactions, items, and tokens. Expanding items into multiple tokens amplifies length and noise, biasing attention toward local details over holistic semantics. We propose Hi-SAM, a Hierarchical Structure-Aware Multi-modal framework with two designs: (1) Disentangled Semantic Tokenizer (DST): unifies modalities via geometry-aware alignment and quantizes them via a coarse-to-fine strategy. Shared codebooks distill consensus while modality-specific ones recover nuances from residuals, enforced by mutual information minimization; (2) Hierarchical Memory-Anchor Transformer (HMAT): splits positional encoding into inter- and intra-item subspaces via Hierarchical RoPE to restore hierarchy. It inserts Anchor Tokens to condense items into compact memory, retaining details for the current item while accessing history only through compressed summaries. Experiments on real-world datasets show consistent improvements over SOTA baselines, especially in cold-start scenarios. Deployed on a large-scale social platform serving millions of users, Hi-SAM achieved a 6.55% gain in the core online metric. 

---
# Detecting RLVR Training Data via Structural Convergence of Reasoning 

**Authors**: Hongbo Zhang, Yue Yang, Jianhao Yan, Guangsheng Bao, Yue Zhang, Yue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.11792)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) is central to training modern reasoning models, but the undisclosed training data raises concerns about benchmark contamination. Unlike pretraining methods, which optimize models using token-level probabilities, RLVR fine-tunes models based on reward feedback from self-generated reasoning trajectories, making conventional likelihood-based detection methods less effective. We show that RLVR induces a distinctive behavioral signature: prompts encountered during RLVR training result in more rigid and similar generations, while unseen prompts retain greater diversity. We introduce Min-$k$NN Distance, a simple black-box detector that quantifies this collapse by sampling multiple completions for a given prompt and computing the average of the $k$ smallest nearest-neighbor edit distances. Min-$k$NN Distance requires no access to the reference model or token probabilities. Experiments across multiple RLVR-trained reasoning models show that Min-$k$NN Distance reliably distinguishes RL-seen examples from unseen ones and outperforms existing membership inference and RL contamination detection baselines. 

---
# Beyond End-to-End Video Models: An LLM-Based Multi-Agent System for Educational Video Generation 

**Authors**: Lingyong Yan, Jiulong Wu, Dong Xie, Weixian Shi, Deguo Xia, Jizhou Huang  

**Link**: [PDF](https://arxiv.org/pdf/2602.11790)  

**Abstract**: Although recent end-to-end video generation models demonstrate impressive performance in visually oriented content creation, they remain limited in scenarios that require strict logical rigor and precise knowledge representation, such as instructional and educational media. To address this problem, we propose LAVES, a hierarchical LLM-based multi-agent system for generating high-quality instructional videos from educational problems. The LAVES formulates educational video generation as a multi-objective task that simultaneously demands correct step-by-step reasoning, pedagogically coherent narration, semantically faithful visual demonstrations, and precise audio--visual alignment. To address the limitations of prior approaches--including low procedural fidelity, high production cost, and limited controllability--LAVES decomposes the generation workflow into specialized agents coordinated by a central Orchestrating Agent with explicit quality gates and iterative critique mechanisms. Specifically, the Orchestrating Agent supervises a Solution Agent for rigorous problem solving, an Illustration Agent that produces executable visualization codes, and a Narration Agent for learner-oriented instructional scripts. In addition, all outputs from the working agents are subject to semantic critique, rule-based constraints, and tool-based compilation checks. Rather than directly synthesizing pixels, the system constructs a structured executable video script that is deterministically compiled into synchronized visuals and narration using template-driven assembly rules, enabling fully automated end-to-end production without manual editing. In large-scale deployments, LAVES achieves a throughput exceeding one million videos per day, delivering over a 95% reduction in cost compared to current industry-standard approaches while maintaining a high acceptance rate. 

---
# FlowMind: Execute-Summarize for Structured Workflow Generation from LLM Reasoning 

**Authors**: Yihao Liu, Ziyun Zhang, Zile He, Huaqian Cai  

**Link**: [PDF](https://arxiv.org/pdf/2602.11782)  

**Abstract**: LLMs can solve complex tasks through reasoning and tool use, but accurately translating these solutions into structured workflows remains challenging. We model workflows as sequences of tool use and reformulate the problem as designing a mechanism that can both solve tasks and reliably construct workflows. Prior approaches that build workflows during execution often suffer from inaccuracies due to interference between the two processes. We propose an Execute-Summarize(ES) framework that decouples task execution from workflow construction: the model first completes the task using available tools, then independently reconstructs a structured workflow from execution traces. This separation improves workflow accuracy and robustness. We introduce FlowBench and show through extensive experiments that our approach outperforms existing methods, providing a reliable paradigm for grounding free-form LLM reasoning into structured workflows. 

---
# RELATE: A Reinforcement Learning-Enhanced LLM Framework for Advertising Text Generation 

**Authors**: Jinfang Wang, Jiajie Liu, Jianwei Wu, Ziqin Luo, Zhen Chen, Chunlei Li, Biao Han, Tao Deng, Yi Li, Shuanglong Li, Lin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2602.11780)  

**Abstract**: In online advertising, advertising text plays a critical role in attracting user engagement and driving advertiser value. Existing industrial systems typically follow a two-stage paradigm, where candidate texts are first generated and subsequently aligned with online performance metrics such as click-through rate(CTR). This separation often leads to misaligned optimization objectives and low funnel efficiency, limiting global optimality.
To address these limitations, we propose RELATE, a reinforcement learning-based end-to-end framework that unifies generation and objective alignment within a single model. Instead of decoupling text generation from downstream metric alignment, RELATE integrates performance and compliance objectives directly into the generation process via policy learning. To better capture ultimate advertiser value beyond click-level signals, We incorporate conversion-oriented metrics into the objective and jointly model them with compliance constraints as multi-dimensional rewards, enabling the model to generate high-quality ad texts that improve conversion performance under policy constraints.
Extensive experiments on large-scale industrial datasets demonstrate that RELATE consistently outperforms baselines. Furthermore, online deployment on a production advertising platform yields statistically significant improvements in click-through conversion rate(CTCVR) under strict policy constraints, validating the robustness and real-world effectiveness of the proposed framework. 

---
# How to Optimize Multispecies Set Predictions in Presence-Absence Modeling ? 

**Authors**: Sébastien Gigot--Léandri, Gaétan Morand, Alexis Joly, François Munoz, David Mouillot, Christophe Botella, Maximilien Servajean  

**Link**: [PDF](https://arxiv.org/pdf/2602.11771)  

**Abstract**: Species distribution models (SDMs) commonly produce probabilistic occurrence predictions that must be converted into binary presence-absence maps for ecological inference and conservation planning. However, this binarization step is typically heuristic and can substantially distort estimates of species prevalence and community composition. We present MaxExp, a decision-driven binarization framework that selects the most probable species assemblage by directly maximizing a chosen evaluation metric. MaxExp requires no calibration data and is flexible across several scores. We also introduce the Set Size Expectation (SSE) method, a computationally efficient alternative that predicts assemblages based on expected species richness. Using three case studies spanning diverse taxa, species counts, and performance metrics, we show that MaxExp consistently matches or surpasses widely used thresholding and calibration methods, especially under strong class imbalance and high rarity. SSE offers a simpler yet competitive option. Together, these methods provide robust, reproducible tools for multispecies SDM binarization. 

---
# TSR: Trajectory-Search Rollouts for Multi-Turn RL of LLM Agents 

**Authors**: Aladin Djuhera, Swanand Ravindra Kadhe, Farhan Ahmed, Holger Boche  

**Link**: [PDF](https://arxiv.org/pdf/2602.11767)  

**Abstract**: Advances in large language models (LLMs) are driving a shift toward using reinforcement learning (RL) to train agents from iterative, multi-turn interactions across tasks. However, multi-turn RL remains challenging as rewards are often sparse or delayed, and environments can be stochastic. In this regime, naive trajectory sampling can hinder exploitation and induce mode collapse. We propose TSR (Trajectory-Search Rollouts), a training-time approach that repurposes test-time scaling ideas for improved per-turn rollout generation. TSR performs lightweight tree-style search to construct high-quality trajectories by selecting high-scoring actions at each turn using task-specific feedback. This improves rollout quality and stabilizes learning while leaving the underlying optimization objective unchanged, making TSR optimizer-agnostic. We instantiate TSR with best-of-N, beam, and shallow lookahead search, and pair it with PPO and GRPO, achieving up to 15% performance gains and more stable learning on Sokoban, FrozenLake, and WebShop tasks at a one-time increase in training compute. By moving search from inference time to the rollout stage of training, TSR provides a simple and general mechanism for stronger multi-turn agent learning, complementary to existing frameworks and rejection-sampling-style selection methods. 

---
# AIR: Improving Agent Safety through Incident Response 

**Authors**: Zibo Xiao, Jun Sun, Junjie Chen  

**Link**: [PDF](https://arxiv.org/pdf/2602.11749)  

**Abstract**: Large Language Model (LLM) agents are increasingly deployed in practice across a wide range of autonomous applications. Yet current safety mechanisms for LLM agents focus almost exclusively on preventing failures in advance, providing limited capabilities for responding to, containing, or recovering from incidents after they inevitably arise. In this work, we introduce AIR, the first incident response framework for LLM agent systems. AIR defines a domain-specific language for managing the incident response lifecycle autonomously in LLM agent systems, and integrates it into the agent's execution loop to (1) detect incidents via semantic checks grounded in the current environment state and recent context, (2) guide the agent to execute containment and recovery actions via its tools, and (3) synthesize guardrail rules during eradication to block similar incidents in future executions. We evaluate AIR on three representative agent types. Results show that AIR achieves detection, remediation, and eradication success rates all exceeding 90%. Extensive experiments further confirm the necessity of AIR's key design components, show the timeliness and moderate overhead of AIR, and demonstrate that LLM-generated rules can approach the effectiveness of developer-authored rules across domains. These results show that incident response is both feasible and essential as a first-class mechanism for improving agent safety. 

---
# Text2GQL-Bench: A Text to Graph Query Language Benchmark [Experiment, Analysis & Benchmark] 

**Authors**: Songlin Lyu, Lujie Ban, Zihang Wu, Tianqi Luo, Jirong Liu, Chenhao Ma, Yuyu Luo, Nan Tang, Shipeng Qi, Heng Lin, Yongchao Liu, Chuntao Hong  

**Link**: [PDF](https://arxiv.org/pdf/2602.11745)  

**Abstract**: Graph models are fundamental to data analysis in domains rich with complex relationships. Text-to-Graph-Query-Language (Text-to-GQL) systems act as a translator, converting natural language into executable graph queries. This capability allows Large Language Models (LLMs) to directly analyze and manipulate graph data, posi-tioning them as powerful agent infrastructures for Graph Database Management System (GDBMS). Despite recent progress, existing datasets are often limited in domain coverage, supported graph query languages, or evaluation scope. The advancement of Text-to-GQL systems is hindered by the lack of high-quality benchmark datasets and evaluation methods to systematically compare model capabilities across different graph query languages and domains. In this work, we present Text2GQL-Bench, a unified Text-to-GQL benchmark designed to address these limitations. Text2GQL-Bench couples a multi-GQL dataset that has 178,184 (Question, Query) pairs spanning 13 domains, with a scalable construction framework that generates datasets in different domains, question abstraction levels, and GQLs with heterogeneous resources. To support compre-hensive assessment, we introduce an evaluation method that goes beyond a single end-to-end metric by jointly reporting grammatical validity, similarity, semantic alignment, and execution accuracy. Our evaluation uncovers a stark dialect gap in ISO-GQL generation: even strong LLMs achieve only at most 4% execution accuracy (EX) in zero-shot settings, though a fixed 3-shot prompt raises accuracy to around 50%, the grammatical validity remains lower than 70%. Moreover, a fine-tuned 8B open-weight model reaches 45.1% EX, and 90.8% grammatical validity, demonstrating that most of the performance jump is unlocked by exposure to sufficient ISO-GQL examples. 

---
# Cross-Architecture Model Diffing with Crosscoders: Unsupervised Discovery of Differences Between LLMs 

**Authors**: Thomas Jiralerspong, Trenton Bricken  

**Link**: [PDF](https://arxiv.org/pdf/2602.11729)  

**Abstract**: Model diffing, the process of comparing models' internal representations to identify their differences, is a promising approach for uncovering safety-critical behaviors in new models. However, its application has so far been primarily focused on comparing a base model with its finetune. Since new LLM releases are often novel architectures, cross-architecture methods are essential to make model diffing widely applicable. Crosscoders are one solution capable of cross-architecture model diffing but have only ever been applied to base vs finetune comparisons. We provide the first application of crosscoders to cross-architecture model diffing and introduce Dedicated Feature Crosscoders (DFCs), an architectural modification designed to better isolate features unique to one model. Using this technique, we find in an unsupervised fashion features including Chinese Communist Party alignment in Qwen3-8B and Deepseek-R1-0528-Qwen3-8B, American exceptionalism in Llama3.1-8B-Instruct, and a copyright refusal mechanism in GPT-OSS-20B. Together, our results work towards establishing cross-architecture crosscoder model diffing as an effective method for identifying meaningful behavioral differences between AI models. 

---
# Beyond Parameter Arithmetic: Sparse Complementary Fusion for Distribution-Aware Model Merging 

**Authors**: Weihong Lin, Lin Sun, Qilong Shi, Aomufei Yuan, Yuxuan Tian, Zhengyang Wang, Guangxiang Zhao, Xiangzheng Zhang, Tong Yang  

**Link**: [PDF](https://arxiv.org/pdf/2602.11717)  

**Abstract**: Model merging has emerged as a promising paradigm for composing the capabilities of large language models by directly operating in weight space, enabling the integration of specialized models without costly retraining. However, existing merging methods largely rely on parameter-space heuristics, which often introduce severe interference, leading to degraded generalization and unstable generation behaviors such as repetition and incoherent outputs. In this work, we propose Sparse Complementary Fusion with reverse KL (SCF-RKL), a novel model merging framework that explicitly controls functional interference through sparse, distribution-aware updates. Instead of assuming linear additivity in parameter space, SCF-RKL measures the functional divergence between models using reverse Kullback-Leibler divergence and selectively incorporates complementary parameters. This mode-seeking, sparsity-inducing design effectively preserves stable representations while integrating new capabilities. We evaluate SCF-RKL across a wide range of model scales and architectures, covering both reasoning-focused and instruction-tuned models. Extensive experiments on 24 benchmarks spanning advanced reasoning, general reasoning and knowledge, instruction following, and safety demonstrate, vision classification that SCF-RKL consistently outperforms existing model merging methods while maintaining strong generalization and generation stability. 

---
# ThinkRouter: Efficient Reasoning via Routing Thinking between Latent and Discrete Spaces 

**Authors**: Xin Xu, Tong Yu, Xiang Chen, Haoliang Wang, Julian McAuley, Saayan Mitra  

**Link**: [PDF](https://arxiv.org/pdf/2602.11683)  

**Abstract**: Recent work explores latent reasoning to improve reasoning efficiency by replacing explicit reasoning trajectories with continuous representations in a latent space, yet its effectiveness varies across settings. Analysis of model confidence dynamics under latent reasoning reveals that thinking trajectories ending in incorrect answers contain fewer low-confidence steps than those ending in correct answers. Meanwhile, we suggest that soft embeddings aggregated by multiple low-confidence thinking alternatives may introduce and propagate noise, leading to high confidence in unreliable reasoning trajectories. Motivated by these observations, ThinkRouter, an inference-time confidence-aware routing mechanism is proposed to avoid high confidence and noise for efficient reasoning. ThinkRouter routes thinking to the discrete token space when model confidence is low, and to the latent space otherwise. Extensive experiments on STEM reasoning and coding benchmarks across diverse large reasoning models demonstrate that ThinkRouter outperforms explicit CoT, random routing, and latent reasoning baselines in terms of accuracy, achieving an average improvement of 19.70 points in Pass@1, while reducing generation length by up to 15.55%. Further comprehensive analysis reveals that ThinkRouter can calibrate errors arising from explicit CoT and latent reasoning, and accelerates end-of-thinking token generation by globally lowering model confidence. 

---
# Beyond Pixels: Vector-to-Graph Transformation for Reliable Schematic Auditing 

**Authors**: Chengwei Ma, Zhen Tian, Zhou Zhou, Zhixian Xu, Xiaowei Zhu, Xia Hua, Si Shi, F. Richard Yu  

**Link**: [PDF](https://arxiv.org/pdf/2602.11678)  

**Abstract**: Multimodal Large Language Models (MLLMs) have shown remarkable progress in visual understanding, yet they suffer from a critical limitation: structural blindness. Even state-of-the-art models fail to capture topology and symbolic logic in engineering schematics, as their pixel-driven paradigm discards the explicit vector-defined relations needed for reasoning. To overcome this, we propose a Vector-to-Graph (V2G) pipeline that converts CAD diagrams into property graphs where nodes represent components and edges encode connectivity, making structural dependencies explicit and machine-auditable. On a diagnostic benchmark of electrical compliance checks, V2G yields large accuracy gains across all error categories, while leading MLLMs remain near chance level. These results highlight the systemic inadequacy of pixel-based methods and demonstrate that structure-aware representations provide a reliable path toward practical deployment of multimodal AI in engineering domains. To facilitate further research, we release our benchmark and implementation at this https URL. 

---
# Right for the Wrong Reasons: Epistemic Regret Minimization for Causal Rung Collapse in LLMs 

**Authors**: Edward Y. Chang  

**Link**: [PDF](https://arxiv.org/pdf/2602.11675)  

**Abstract**: Machine learning systems that are "right for the wrong reasons" achieve high performance through shortcuts that collapse under distributional shift. We show this pathology has a precise causal origin: autoregressive training provides no gradient signal to distinguish association P(Y|X) from intervention P(Y|do(X)), a failure we formalize as Rung Collapse. When outcome-based learning reinforces correct answers obtained through incorrect causal models, the agent becomes entrenched in flawed reasoning, a phenomenon we term Aleatoric Entrenchment. We propose Epistemic Regret Minimization (ERM), a belief revision objective that penalizes errors in causal reasoning independently of task success, and embed it within a three-layer architecture with three contributions grounded in knowledge representation: (1) a Physical Grounding Theorem proving that actions satisfying actuator independence implement valid do-operations, bridging action languages and do-calculus; (2) ERM as a causal belief revision operator satisfying AGM postulates, preventing entrenchment even when the agent succeeds for the wrong reasons; and (3) a failure mode taxonomy that classifies recurring reasoning errors and injects domain-independent guards, enabling cross-domain transfer. We prove asymptotic recovery of the true interventional distribution with finite-sample bounds. Experiments on 1,360 causal trap scenarios across six frontier LLMs reveal that Rung Collapse persists even in reasoning-enhanced models (3.7% for GPT-5.2), that steerability exhibits inverse scaling where advanced models resist generic correction, and that targeted ERM feedback recovers 53-59% of entrenched errors where outcome-level feedback fails. 

---
# Benchmark Health Index: A Systematic Framework for Benchmarking the Benchmarks of LLMs 

**Authors**: Longyuan Zhu, Hairan Hua, Linlin Miao, Bing Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2602.11674)  

**Abstract**: Large Language Models (LLMs) are advancing rapidly, yet the benchmarks used to measure this progress are becoming increasingly unreliable. Score inflation and selective reporting have eroded the authority of standard benchmarks, leaving the community uncertain about which evaluation results remain trustworthy. We introduce the Benchmark Health Index (BHI), a pure data-driven framework for auditing evaluation sets along three orthogonal and complementary axes: (1) Capability Discrimination, measuring how sharply a benchmark separates model performance beyond noise; (2) Anti-Saturation, estimating remaining headroom before ceiling effects erode resolution and thus the benchmark's expected longevity; and (3) Impact, quantifying influence across academic and industrial ecosystems via adoption breadth and practice-shaping power. By distilling 106 validated benchmarks from the technical reports of 91 representative models in 2025, we systematically characterize the evaluation landscape. BHI is the first framework to quantify benchmark health at a macro level, providing a principled basis for benchmark selection and enabling dynamic lifecycle management for next-generation evaluation protocols. 

---
# PhyNiKCE: A Neurosymbolic Agentic Framework for Autonomous Computational Fluid Dynamics 

**Authors**: E Fan, Lisong Shi, Zhengtong Li, Chih-yung Wen  

**Link**: [PDF](https://arxiv.org/pdf/2602.11666)  

**Abstract**: The deployment of autonomous agents for Computational Fluid Dynamics (CFD), is critically limited by the probabilistic nature of Large Language Models (LLMs), which struggle to enforce the strict conservation laws and numerical stability required for physics-based simulations. Reliance on purely semantic Retrieval Augmented Generation (RAG) often leads to "context poisoning," where agents generate linguistically plausible but physically invalid configurations due to a fundamental Semantic-Physical Disconnect. To bridge this gap, this work introduces PhyNiKCE (Physical and Numerical Knowledgeable Context Engineering), a neurosymbolic agentic framework for trustworthy engineering. Unlike standard black-box agents, PhyNiKCE decouples neural planning from symbolic validation. It employs a Symbolic Knowledge Engine that treats simulation setup as a Constraint Satisfaction Problem, rigidly enforcing physical constraints via a Deterministic RAG Engine with specialized retrieval strategies for solvers, turbulence models, and boundary conditions. Validated through rigorous OpenFOAM experiments on practical, non-tutorial CFD tasks using Gemini-2.5-Pro/Flash, PhyNiKCE demonstrates a 96% relative improvement over state-of-the-art baselines. Furthermore, by replacing trial-and-error with knowledge-driven initialization, the framework reduced autonomous self-correction loops by 59% while simultaneously lowering LLM token consumption by 17%. These results demonstrate that decoupling neural generation from symbolic constraint enforcement significantly enhances robustness and efficiency. While validated on CFD, this architecture offers a scalable, auditable paradigm for Trustworthy Artificial Intelligence in broader industrial automation. 

---
# Quark Medical Alignment: A Holistic Multi-Dimensional Alignment and Collaborative Optimization Paradigm 

**Authors**: Tianxiang Xu, Jiayi Liu, Yixuan Tong, Jialu Xu, Yunqing Wei, Kaiwen Feng, PanPan Hou, Kangping Yin, Jiyuan Hu, Hao Zhou, Zhenxin Ma, Jian Xu, Guanjun Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2602.11661)  

**Abstract**: While reinforcement learning for large language model alignment has progressed rapidly in recent years, transferring these paradigms to high-stakes medical question answering reveals a fundamental paradigm mismatch. Reinforcement Learning from Human Feedback relies on preference annotations that are prohibitively expensive and often fail to reflect the absolute correctness of medical facts. Reinforcement Learning from Verifiable Rewards lacks effective automatic verifiers and struggles to handle complex clinical contexts. Meanwhile, medical alignment requires the simultaneous optimization of correctness, safety, and compliance, yet multi-objective heterogeneous reward signals are prone to scale mismatch and optimization this http URL address these challenges, we propose a robust medical alignment paradigm. We first construct a holistic multi-dimensional medical alignment matrix that decomposes alignment objectives into four categories: fundamental capabilities, expert knowledge, online feedback, and format specifications. Within each category, we establish a closed loop of where observable metrics inform attributable diagnosis, which in turn drives optimizable rewards, thereby providing fine-grained, high-resolution supervision signals for subsequent iterative optimization. To resolve gradient domination and optimization instability problem caused by heterogeneous signals, we further propose a unified optimization mechanism. This mechanism employs Reference-Frozen Normalization to align reward scales and implements a Tri-Factor Adaptive Dynamic Weighting strategy to achieve collaborative optimization that is weakness-oriented, risk-prioritized, and redundancy-reducing. Experimental results demonstrate the effectiveness of our proposed paradigm in real-world medical scenario evaluations, establishing a new paradigm for complex alignment in vertical domains. 

---
# Do MLLMs Really Understand Space? A Mathematical Reasoning Evaluation 

**Authors**: Shuo Lu, Jianjie Cheng, Yinuo Xu, Yongcan Yu, Lijun Sheng, Peijie Wang, Siru Jiang, Yongguan Hu, Run Ling, Yihua Shao, Ao Ma, Wei Feng, Lingxiao He, Meng Wang, Qianlong Xie, Xingxing Wang, Ran He, Jian Liang  

**Link**: [PDF](https://arxiv.org/pdf/2602.11635)  

**Abstract**: Multimodal large language models (MLLMs) have achieved strong performance on perception-oriented tasks, yet their ability to perform mathematical spatial reasoning, defined as the capacity to parse and manipulate two- and three-dimensional relations, remains unclear. Humans easily solve textbook-style spatial reasoning problems with over 95\% accuracy, but we find that most leading MLLMs fail to reach even 60\% on the same tasks. This striking gap highlights spatial reasoning as a fundamental weakness of current models. To investigate this gap, we present MathSpatial, a unified framework for evaluating and improving spatial reasoning in MLLMs. MathSpatial includes three complementary components: (i) MathSpatial-Bench, a benchmark of 2K problems across three categories and eleven subtypes, designed to isolate reasoning difficulty from perceptual noise; (ii) MathSpatial-Corpus, a training dataset of 8K additional problems with verified solutions; and (iii) MathSpatial-SRT, which models reasoning as structured traces composed of three atomic operations--Correlate, Constrain, and Infer. Experiments show that fine-tuning Qwen2.5-VL-7B on MathSpatial achieves competitive accuracy while reducing tokens by 25\%. MathSpatial provides the first large-scale resource that disentangles perception from reasoning, enabling precise measurement and comprehensive understanding of mathematical spatial reasoning in MLLMs. 

---
# Neuro-Symbolic Multitasking: A Unified Framework for Discovering Generalizable Solutions to PDE Families 

**Authors**: Yipeng Huang, Dejun Xu, Zexin Lin, Zhenzhong Wang, Min Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2602.11630)  

**Abstract**: Solving Partial Differential Equations (PDEs) is fundamental to numerous scientific and engineering disciplines. A common challenge arises from solving the PDE families, which are characterized by sharing an identical mathematical structure but varying in specific parameters. Traditional numerical methods, such as the finite element method, need to independently solve each instance within a PDE family, which incurs massive computational cost. On the other hand, while recent advancements in machine learning PDE solvers offer impressive computational speed and accuracy, their inherent ``black-box" nature presents a considerable limitation. These methods primarily yield numerical approximations, thereby lacking the crucial interpretability provided by analytical expressions, which are essential for deeper scientific insight. To address these limitations, we propose a neuro-assisted multitasking symbolic PDE solver framework for PDE family solving, dubbed NMIPS. In particular, we employ multifactorial optimization to simultaneously discover the analytical solutions of PDEs. To enhance computational efficiency, we devise an affine transfer method by transferring learned mathematical structures among PDEs in a family, avoiding solving each PDE from scratch. Experimental results across multiple cases demonstrate promising improvements over existing baselines, achieving up to a $\sim$35.7% increase in accuracy while providing interpretable analytical solutions. 

---
# When Agents Disagree With Themselves: Measuring Behavioral Consistency in LLM-Based Agents 

**Authors**: Aman Mehta  

**Link**: [PDF](https://arxiv.org/pdf/2602.11619)  

**Abstract**: Run the same LLM agent on the same task twice: do you get the same behavior? We find the answer is often no. In a study of 3,000 agent runs across three models (Llama 3.1 70B, GPT-4o, and Claude Sonnet 4.5) on HotpotQA, we observe that ReAct-style agents produce 2.0--4.2 distinct action sequences per 10 runs on average, even with identical inputs. More importantly, this variance predicts failure: tasks with consistent behavior ($\leq$2 unique paths) achieve 80--92% accuracy, while highly inconsistent tasks ($\geq$6 unique paths) achieve only 25--60%, a 32--55 percentage point gap depending on model. We trace variance to early decisions: 69% of divergence occurs at step 2, the first search query. Our results suggest that monitoring behavioral consistency during execution could enable early error detection and improve agent reliability. 

---
# scPilot: Large Language Model Reasoning Toward Automated Single-Cell Analysis and Discovery 

**Authors**: Yiming Gao, Zhen Wang, Jefferson Chen, Mark Antkowiak, Mengzhou Hu, JungHo Kong, Dexter Pratt, Jieyuan Liu, Enze Ma, Zhiting Hu, Eric P. Xing  

**Link**: [PDF](https://arxiv.org/pdf/2602.11609)  

**Abstract**: We present scPilot, the first systematic framework to practice omics-native reasoning: a large language model (LLM) converses in natural language while directly inspecting single-cell RNA-seq data and on-demand bioinformatics tools. scPilot converts core single-cell analyses, i.e., cell-type annotation, developmental-trajectory reconstruction, and transcription-factor targeting, into step-by-step reasoning problems that the model must solve, justify, and, when needed, revise with new evidence.
To measure progress, we release scBench, a suite of 9 expertly curated datasets and graders that faithfully evaluate the omics-native reasoning capability of scPilot w.r.t various LLMs. Experiments with o1 show that iterative omics-native reasoning lifts average accuracy by 11% for cell-type annotation and Gemini-2.5-Pro cuts trajectory graph-edit distance by 30% versus one-shot prompting, while generating transparent reasoning traces explain marker gene ambiguity and regulatory logic. By grounding LLMs in raw omics data, scPilot enables auditable, interpretable, and diagnostically informative single-cell analyses.
Code, data, and package are available at this https URL 

---
# MAPLE: Modality-Aware Post-training and Learning Ecosystem 

**Authors**: Nikhil Verma, Minjung Kim, JooYoung Yoo, Kyung-Min Jin, Manasa Bharadwaj, Kevin Ferreira, Ko Keun Kim, Youngjoon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2602.11596)  

**Abstract**: Multimodal language models now integrate text, audio, and video for unified reasoning. Yet existing RL post-training pipelines treat all input signals as equally relevant, ignoring which modalities each task actually requires. This modality-blind training inflates policy-gradient variance, slows convergence, and degrades robustness to real-world distribution shifts where signals may be missing, added, or reweighted. We introduce MAPLE, a complete modality-aware post-training and learning ecosystem comprising: (1) MAPLE-bench, the first benchmark explicitly annotating minimal signal combinations required per task; (2) MAPO, a modality-aware policy optimization framework that stratifies batches by modality requirement to reduce gradient variance from heterogeneous group advantages; (3) Adaptive weighting and curriculum scheduling that balances and prioritizes harder signal combinations. Systematic analysis across loss aggregation, clipping, sampling, and curriculum design establishes MAPO's optimal training strategy. Adaptive weighting and curriculum focused learning further boost performance across signal combinations. MAPLE narrows uni/multi-modal accuracy gaps by 30.24%, converges 3.18x faster, and maintains stability across all modality combinations under realistic reduced signal access. MAPLE constitutes a complete recipe for deployment-ready multimodal RL post-training. 

---
# The Five Ws of Multi-Agent Communication: Who Talks to Whom, When, What, and Why -- A Survey from MARL to Emergent Language and LLMs 

**Authors**: Jingdi Chen, Hanqing Yang, Zongjun Liu, Carlee Joe-Wong  

**Link**: [PDF](https://arxiv.org/pdf/2602.11583)  

**Abstract**: Multi-agent sequential decision-making powers many real-world systems, from autonomous vehicles and robotics to collaborative AI assistants. In dynamic, partially observable environments, communication is often what reduces uncertainty and makes collaboration possible. This survey reviews multi-agent communication (MA-Comm) through the Five Ws: who communicates with whom, what is communicated, when communication occurs, and why communication is beneficial. This framing offers a clean way to connect ideas across otherwise separate research threads. We trace how communication approaches have evolved across three major paradigms. In Multi-Agent Reinforcement Learning (MARL), early methods used hand-designed or implicit protocols, followed by end-to-end learned communication optimized for reward and control. While successful, these protocols are frequently task-specific and hard to interpret, motivating work on Emergent Language (EL), where agents can develop more structured or symbolic communication through interaction. EL methods, however, still struggle with grounding, generalization, and scalability, which has fueled recent interest in large language models (LLMs) that bring natural language priors for reasoning, planning, and collaboration in more open-ended settings. Across MARL, EL, and LLM-based systems, we highlight how different choices shape communication design, where the main trade-offs lie, and what remains unsolved. We distill practical design patterns and open challenges to support future hybrid systems that combine learning, language, and control for scalable and interpretable multi-agent collaboration. 

---
# Learning to Configure Agentic AI Systems 

**Authors**: Aditya Taparia, Som Sagar, Ransalu Senanayake  

**Link**: [PDF](https://arxiv.org/pdf/2602.11574)  

**Abstract**: Configuring LLM-based agent systems involves choosing workflows, tools, token budgets, and prompts from a large combinatorial design space, and is typically handled today by fixed large templates or hand-tuned heuristics. This leads to brittle behavior and unnecessary compute, since the same cumbersome configuration is often applied to both easy and hard input queries. We formulate agent configuration as a query-wise decision problem and introduce ARC (Agentic Resource & Configuration learner), which learns a light-weight hierarchical policy using reinforcement learning to dynamically tailor these configurations. Across multiple benchmarks spanning reasoning and tool-augmented question answering, the learned policy consistently outperforms strong hand-designed and other baselines, achieving up to 25% higher task accuracy while also reducing token and runtime costs. These results demonstrate that learning per-query agent configurations is a powerful alternative to "one size fits all" designs. 

---
# SemaPop: Semantic-Persona Conditioned Population Synthesis 

**Authors**: Zhenlin Qin, Yancheng Ling, Leizhen Wang, Francisco Câmara Pereira, Zhenliang Ma  

**Link**: [PDF](https://arxiv.org/pdf/2602.11569)  

**Abstract**: Population synthesis is a critical component of individual-level socio-economic simulation, yet remains challenging due to the need to jointly represent statistical structure and latent behavioral semantics. Existing population synthesis approaches predominantly rely on structured attributes and statistical constraints, leaving a gap in semantic-conditioned population generation that can capture abstract behavioral patterns implicitly in survey data. This study proposes SemaPop, a semantic-statistical population synthesis model that integrates large language models (LLMs) with generative population modeling. SemaPop derives high-level persona representations from individual survey records and incorporates them as semantic conditioning signals for population generation, while marginal regularization is introduced to enforce alignment with target population marginals. In this study, the framework is instantiated using a Wasserstein GAN with gradient penalty (WGAN-GP) backbone, referred to as SemaPop-GAN. Extensive experiments demonstrate that SemaPop-GAN achieves improved generative performance, yielding closer alignment with target marginal and joint distributions while maintaining sample-level feasibility and diversity under semantic conditioning. Ablation studies further confirm the contribution of semantic persona conditioning and architectural design choices to balancing marginal consistency and structural realism. These results demonstrate that SemaPop-GAN enables controllable and interpretable population synthesis through effective semantic-statistical information fusion. SemaPop-GAN also provides a promising modular foundation for developing generative population projection systems that integrate individual-level behavioral semantics with population-level statistical constraints. 

---
# Budget-Constrained Agentic Large Language Models: Intention-Based Planning for Costly Tool Use 

**Authors**: Hanbing Liu, Chunhao Tian, Nan An, Ziyuan Wang, Pinyan Lu, Changyuan Yu, Qi Qi  

**Link**: [PDF](https://arxiv.org/pdf/2602.11541)  

**Abstract**: We study budget-constrained tool-augmented agents, where a large language model must solve multi-step tasks by invoking external tools under a strict monetary budget. We formalize this setting as sequential decision making in context space with priced and stochastic tool executions, making direct planning intractable due to massive state-action spaces, high variance of outcomes and prohibitive exploration cost. To address these challenges, we propose INTENT, an inference-time planning framework that leverages an intention-aware hierarchical world model to anticipate future tool usage, risk-calibrated cost, and guide decisions online. Across cost-augmented StableToolBench, INTENT strictly enforces hard budget feasibility while substantially improving task success over baselines, and remains robust under dynamic market shifts such as tool price changes and varying budgets. 

---
# CausalAgent: A Conversational Multi-Agent System for End-to-End Causal Inference 

**Authors**: Jiawei Zhu, Wei Chen, Ruichu Cai  

**Link**: [PDF](https://arxiv.org/pdf/2602.11527)  

**Abstract**: Causal inference holds immense value in fields such as healthcare, economics, and social sciences. However, traditional causal analysis workflows impose significant technical barriers, requiring researchers to possess dual backgrounds in statistics and computer science, while manually selecting algorithms, handling data quality issues, and interpreting complex results. To address these challenges, we propose CausalAgent, a conversational multi-agent system for end-to-end causal inference. The system innovatively integrates Multi-Agent Systems (MAS), Retrieval-Augmented Generation (RAG), and the Model Context Protocol (MCP) to achieve automation from data cleaning and causal structure learning to bias correction and report generation through natural language interaction. Users need only upload a dataset and pose questions in natural language to receive a rigorous, interactive analysis report. As a novel user-centered human-AI collaboration paradigm, CausalAgent explicitly models the analysis workflow. By leveraging interactive visualizations, it significantly lowers the barrier to entry for causal analysis while ensuring the rigor and interpretability of the process. 

---
# Human-Inspired Continuous Learning of Internal Reasoning Processes: Learning How to Think for Adaptive AI Systems 

**Authors**: Hong Su  

**Link**: [PDF](https://arxiv.org/pdf/2602.11516)  

**Abstract**: Learning internal reasoning processes is crucial for developing AI systems capable of sustained adaptation in dynamic real-world environments. However, most existing approaches primarily emphasize learning task-specific outputs or static knowledge representations, while overlooking the continuous refinement of internal reasoning structures, action scheduling policies, and learning mechanisms themselves. In this paper, we propose a human-inspired continuous learning framework that unifies reasoning, action, reflection, and verification within a sequential reasoning model enhanced by parallel learning. The framework explicitly treats internal thinking processes as primary learning objects. It systematically records internal reasoning trajectories and environmental interactions as structured learning material, enabling the system to optimize not only task-level content but also the organization, scheduling, and evolution of reasoning activities. This design realizes learning alongside processing, allowing cognitive structures to improve during execution. Furthermore, the framework supports controlled replacement of predefined logic with learned procedures and introduces a hierarchical learning-to-learn mechanism that jointly adapts task-level parameters and learning strategies. As a result, the system progressively evolves its internal cognitive architecture while preserving operational stability. Experimental results on a temperature sensor abnormality detection task show that incorporating internal-process learning reduces average runtime by 23.9%. 

---
# AgentLeak: A Full-Stack Benchmark for Privacy Leakage in Multi-Agent LLM Systems 

**Authors**: Faouzi El Yagoubi, Ranwa Al Mallah, Godwin Badu-Marfo  

**Link**: [PDF](https://arxiv.org/pdf/2602.11510)  

**Abstract**: Multi-agent Large Language Model (LLM) systems create privacy risks that current benchmarks cannot measure. When agents coordinate on tasks, sensitive data passes through inter-agent messages, shared memory, and tool arguments; pathways that output-only audits never inspect. We introduce AgentLeak, to the best of our knowledge the first full-stack benchmark for privacy leakage covering internal channels, spanning 1,000 scenarios across healthcare, finance, legal, and corporate domains, paired with a 32-class attack taxonomy and three-tier detection pipeline. Testing GPT-4o, GPT-4o-mini, Claude 3.5 Sonnet, Mistral Large, and Llama 3.3 70B across 4,979 traces reveals that multi-agent configurations reduce per-channel output leakage (C1: 27.2% vs 43.2% in single-agent) but introduce unmonitored internal channels that raise total system exposure to 68.9% (OR-aggregated across C1, C2, C5). Internal channels account for most of this gap: inter-agent messages (C2) leak at 68.8%, compared to 27.2% on C1 (output channel). This means that output-only audits miss 41.7% of violations. Claude 3.5 Sonnet, which emphasizes safety alignment in its design, achieves the lowest leakage rates on both external (3.3%) and internal (28.1%) channels, suggesting that model-level safety training may transfer to internal channel protection. Across all five models and four domains, the pattern C2 > C1 holds consistently, confirming that inter-agent communication is the primary vulnerability. These findings underscore the need for coordination frameworks that incorporate internal-channel privacy protections and enforce privacy controls on inter-agent communication. 

---
# Credit Where It is Due: Cross-Modality Connectivity Drives Precise Reinforcement Learning for MLLM Reasoning 

**Authors**: Zhengbo Jiao, Shaobo Wang, Zifan Zhang, Wei Wang, Bing Zhao, Hu Wei, Linfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.11455)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has significantly advanced the reasoning capabilities of Multimodal Large Language Models (MLLMs), yet how visual evidence is integrated during reasoning remains poorly understood. We explore multimodal RLVR through the lens of cross-modal attention connectivity and find that only a small fraction of tokens (approximately 15%) exhibit strong visual-textual coupling. These high-connectivity tokens act as anchors that ground reasoning in the image, while the majority follow linguistic patterns. During RLVR training, credit assignment naturally concentrates on these anchors, sharpening their visual grounding over time. Building on this insight, we propose Anchor-Token Reinforcement Learning (AT-RL), a lightweight framework that selectively reinforces high-connectivity tokens via graph-based clustering of attention topology. Evaluated across the series (3B-32B), AT-RL introduces only 1.2% overhead yet enables the 32B model to surpass the 72B-Instruct baseline on MathVista (80.2), with consistent gains observed across STEM, video and general tasks. Conversely, training solely on low-connectivity tokens causes severe degradation, confirming that effective multimodal RL hinges on precise credit assignment to visual anchors. Our work reveals that reasoning quality is governed not by token quantity but by the fidelity of cross-modal anchoring. 

---
# Distributionally Robust Cooperative Multi-Agent Reinforcement Learning via Robust Value Factorization 

**Authors**: Chengrui Qu, Christopher Yeh, Kishan Panaganti, Eric Mazumdar, Adam Wierman  

**Link**: [PDF](https://arxiv.org/pdf/2602.11437)  

**Abstract**: Cooperative multi-agent reinforcement learning (MARL) commonly adopts centralized training with decentralized execution, where value-factorization methods enforce the individual-global-maximum (IGM) principle so that decentralized greedy actions recover the team-optimal joint action. However, the reliability of this recipe in real-world settings remains unreliable due to environmental uncertainties arising from the sim-to-real gap, model mismatch, and system noise. We address this gap by introducing Distributionally robust IGM (DrIGM), a principle that requires each agent's robust greedy action to align with the robust team-optimal joint action. We show that DrIGM holds for a novel definition of robust individual action values, which is compatible with decentralized greedy execution and yields a provable robustness guarantee for the whole system. Building on this foundation, we derive DrIGM-compliant robust variants of existing value-factorization architectures (e.g., VDN/QMIX/QTRAN) that (i) train on robust Q-targets, (ii) preserve scalability, and (iii) integrate seamlessly with existing codebases without bespoke per-agent reward shaping. Empirically, on high-fidelity SustainGym simulators and a StarCraft game environment, our methods consistently improve out-of-distribution performance. Code and data are available at this https URL. 

---
# TRACER: Trajectory Risk Aggregation for Critical Episodes in Agentic Reasoning 

**Authors**: Sina Tayebati, Divake Kumar, Nastaran Darabi, Davide Ettori, Ranganath Krishnan, Amit Ranjan Trivedi  

**Link**: [PDF](https://arxiv.org/pdf/2602.11409)  

**Abstract**: Estimating uncertainty for AI agents in real-world multi-turn tool-using interaction with humans is difficult because failures are often triggered by sparse critical episodes (e.g., looping, incoherent tool use, or user-agent miscoordination) even when local generation appears confident. Existing uncertainty proxies focus on single-shot text generation and therefore miss these trajectory-level breakdown signals. We introduce TRACER, a trajectory-level uncertainty metric for dual-control Tool-Agent-User interaction. TRACER combines content-aware surprisal with situational-awareness signals, semantic and lexical repetition, and tool-grounded coherence gaps, and aggregates them using a tail-focused risk functional with a MAX-composite step risk to surface decisive anomalies. We evaluate TRACER on $\tau^2$-bench by predicting task failure and selective task execution. To this end, TRACER improves AUROC by up to 37.1% and AUARC by up to 55% over baselines, enabling earlier and more accurate detection of uncertainty in complex conversational tool-use settings. Our code and benchmark are available at this https URL. 

---
# GHOST: Unmasking Phantom States in Mamba2 via Grouped Hidden-state Output-aware Selection & Truncation 

**Authors**: Michael Menezes, Anastasios Kyrillidis  

**Link**: [PDF](https://arxiv.org/pdf/2602.11408)  

**Abstract**: While Mamba2's expanded state dimension enhances temporal modeling, it incurs substantial inference overhead that saturates bandwidth during autoregressive generation. Standard pruning methods fail to address this bottleneck: unstructured sparsity leaves activations dense, magnitude-based selection ignores runtime dynamics, and gradient-based methods impose prohibitive costs. We introduce GHOST (Grouped Hidden-state Output-aware Selection and Truncation), a structured pruning framework that approximates control-theoretic balanced truncation using only forward-pass statistics. By jointly measuring controllability and observability, GHOST rivals the fidelity of gradient-based methods without requiring backpropagation. As a highlight, on models ranging from 130M to 2.7B parameters, our approach achieves a 50\% state-dimension reduction with approximately 1 perplexity point increase on WikiText-2. Code is available at this https URL. 

---
# Causal-JEPA: Learning World Models through Object-Level Latent Interventions 

**Authors**: Heejeong Nam, Quentin Le Lidec, Lucas Maes, Yann LeCun, Randall Balestriero  

**Link**: [PDF](https://arxiv.org/pdf/2602.11389)  

**Abstract**: World models require robust relational understanding to support prediction, reasoning, and control. While object-centric representations provide a useful abstraction, they are not sufficient to capture interaction-dependent dynamics. We therefore propose C-JEPA, a simple and flexible object-centric world model that extends masked joint embedding prediction from image patches to object-centric representations. By applying object-level masking that requires an object's state to be inferred from other objects, C-JEPA induces latent interventions with counterfactual-like effects and prevents shortcut solutions, making interaction reasoning essential. Empirically, C-JEPA leads to consistent gains in visual question answering, with an absolute improvement of about 20\% in counterfactual reasoning compared to the same architecture without object-level masking. On agent control tasks, C-JEPA enables substantially more efficient planning by using only 1\% of the total latent input features required by patch-based world models, while achieving comparable performance. Finally, we provide a formal analysis demonstrating that object-level masking induces a causal inductive bias via latent interventions. Our code is available at this https URL. 

---
# ReplicatorBench: Benchmarking LLM Agents for Replicability in Social and Behavioral Sciences 

**Authors**: Bang Nguyen, Dominik Soós, Qian Ma, Rochana R. Obadage, Zack Ranjan, Sai Koneru, Timothy M. Errington, Shakhlo Nematova, Sarah Rajtmajer, Jian Wu, Meng Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2602.11354)  

**Abstract**: The literature has witnessed an emerging interest in AI agents for automated assessment of scientific papers. Existing benchmarks focus primarily on the computational aspect of this task, testing agents' ability to reproduce or replicate research outcomes when having access to the code and data. This setting, while foundational, (1) fails to capture the inconsistent availability of new data for replication as opposed to reproduction, and (2) lacks ground-truth diversity by focusing only on reproducible papers, thereby failing to evaluate an agent's ability to identify non-replicable research. Furthermore, most benchmarks only evaluate outcomes rather than the replication process. In response, we introduce ReplicatorBench, an end-to-end benchmark, including human-verified replicable and non-replicable research claims in social and behavioral sciences for evaluating AI agents in research replication across three stages: (1) extraction and retrieval of replication data; (2) design and execution of computational experiments; and (3) interpretation of results, allowing a test of AI agents' capability to mimic the activities of human replicators in real world. To set a baseline of AI agents' capability, we develop ReplicatorAgent, an agentic framework equipped with necessary tools, like web search and iterative interaction with sandboxed environments, to accomplish tasks in ReplicatorBench. We evaluate ReplicatorAgent across four underlying large language models (LLMs), as well as different design choices of programming language and levels of code access. Our findings reveal that while current LLM agents are capable of effectively designing and executing computational experiments, they struggle with retrieving resources, such as new data, necessary to replicate a claim. All code and data are publicly available at this https URL. 

---
# Pushing Forward Pareto Frontiers of Proactive Agents with Behavioral Agentic Optimization 

**Authors**: Yihang Yao, Zhepeng Cen, Haohong Lin, Shiqi Liu, Zuxin Liu, Jiacheng Zhu, Zhang-Wei Hong, Laixi Shi, Ding Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2602.11351)  

**Abstract**: Proactive large language model (LLM) agents aim to actively plan, query, and interact over multiple turns, enabling efficient task completion beyond passive instruction following and making them essential for real-world, user-centric applications. Agentic reinforcement learning (RL) has recently emerged as a promising solution for training such agents in multi-turn settings, allowing interaction strategies to be learned from feedback. However, existing pipelines face a critical challenge in balancing task performance with user engagement, as passive agents can not efficiently adapt to users' intentions while overuse of human feedback reduces their satisfaction. To address this trade-off, we propose BAO, an agentic RL framework that combines behavior enhancement to enrich proactive reasoning and information-gathering capabilities with behavior regularization to suppress inefficient or redundant interactions and align agent behavior with user expectations. We evaluate BAO on multiple tasks from the UserRL benchmark suite, and demonstrate that it substantially outperforms proactive agentic RL baselines while achieving comparable or even superior performance to commercial LLM agents, highlighting its effectiveness for training proactive, user-aligned LLM agents in complex multi-turn scenarios. Our website: this https URL. 

---
# AgentNoiseBench: Benchmarking Robustness of Tool-Using LLM Agents Under Noisy Condition 

**Authors**: Ruipeng Wang, Yuxin Chen, Yukai Wang, Chang Wu, Junfeng Fang, Xiaodong Cai, Qi Gu, Hui Su, An Zhang, Xiang Wang, Xunliang Cai, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2602.11348)  

**Abstract**: Recent advances in large language models have enabled LLM-based agents to achieve strong performance on a variety of benchmarks. However, their performance in real-world deployments often that observed on benchmark settings, especially in complex and imperfect environments. This discrepancy largely arises because prevailing training and evaluation paradigms are typically built on idealized assumptions, overlooking the inherent stochasticity and noise present in real-world interactions. To bridge this gap, we introduce AgentNoiseBench, a framework for systematically evaluating the robustness of agentic models under noisy environments. We first conduct an in-depth analysis of biases and uncertainties in real-world scenarios and categorize environmental noise into two primary types: user-noise and tool-noise. Building on this analysis, we develop an automated pipeline that injects controllable noise into existing agent-centric benchmarks while preserving task solvability. Leveraging this pipeline, we perform extensive evaluations across a wide range of models with diverse architectures and parameter scales. Our results reveal consistent performance variations under different noise conditions, highlighting the sensitivity of current agentic models to realistic environmental perturbations. 

---
# Bi-Level Prompt Optimization for Multimodal LLM-as-a-Judge 

**Authors**: Bo Pan, Xuan Kan, Kaitai Zhang, Yan Yan, Shunwen Tan, Zihao He, Zixin Ding, Junjie Wu, Liang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2602.11340)  

**Abstract**: Large language models (LLMs) have become widely adopted as automated judges for evaluating AI-generated content. Despite their success, aligning LLM-based evaluations with human judgments remains challenging. While supervised fine-tuning on human-labeled data can improve alignment, it is costly and inflexible, requiring new training for each task or dataset. Recent progress in auto prompt optimization (APO) offers a more efficient alternative by automatically improving the instructions that guide LLM judges. However, existing APO methods primarily target text-only evaluations and remain underexplored in multimodal settings. In this work, we study auto prompt optimization for multimodal LLM-as-a-judge, particularly for evaluating AI-generated images. We identify a key bottleneck: multimodal models can only process a limited number of visual examples due to context window constraints, which hinders effective trial-and-error prompt refinement. To overcome this, we propose BLPO, a bi-level prompt optimization framework that converts images into textual representations while preserving evaluation-relevant visual cues. Our bi-level optimization approach jointly refines the judge prompt and the I2T prompt to maintain fidelity under limited context budgets. Experiments on four datasets and three LLM judges demonstrate the effectiveness of our method. 

---
# Dissecting Subjectivity and the "Ground Truth" Illusion in Data Annotation 

**Authors**: Sheza Munir, Benjamin Mah, Krisha Kalsi, Shivani Kapania, Julian Posada, Edith Law, Ding Wang, Syed Ishtiaque Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2602.11318)  

**Abstract**: In machine learning, "ground truth" refers to the assumed correct labels used to train and evaluate models. However, the foundational "ground truth" paradigm rests on a positivistic fallacy that treats human disagreement as technical noise rather than a vital sociotechnical signal. This systematic literature review analyzes research published between 2020 and 2025 across seven premier venues: ACL, AIES, CHI, CSCW, EAAMO, FAccT, and NeurIPS, investigating the mechanisms in data annotation practices that facilitate this "consensus trap". Our identification phase captured 30,897 records, which were refined via a tiered keyword filtration schema to a high-recall corpus of 3,042 records for manual screening, resulting in a final included corpus of 346 papers for qualitative synthesis. Our reflexive thematic analysis reveals that systemic failures in positional legibility, combined with the recent architectural shift toward human-as-verifier models, specifically the reliance on model-mediated annotations, introduce deep-seated anchoring bias and effectively remove human voices from the loop. We further demonstrate how geographic hegemony imposes Western norms as universal benchmarks, often enforced by the performative alignment of precarious data workers who prioritize requester compliance over honest subjectivity to avoid economic penalties. Critiquing the "noisy sensor" fallacy, where statistical models misdiagnose cultural pluralism as random error, we argue for reclaiming disagreement as a high-fidelity signal essential for building culturally competent models. To address these systemic tensions, we propose a roadmap for pluralistic annotation infrastructures that shift the objective from discovering a singular "right" answer to mapping the diversity of human experience. 

---
# The PBSAI Governance Ecosystem: A Multi-Agent AI Reference Architecture for Securing Enterprise AI Estates 

**Authors**: John M. Willis  

**Link**: [PDF](https://arxiv.org/pdf/2602.11301)  

**Abstract**: Enterprises are rapidly deploying large language models, retrieval augmented generation pipelines, and tool using agents into production, often on shared high performance computing clusters and cloud accelerator platforms that also support defensive analytics. These systems increasingly function not as isolated models but as AI estates: socio technical systems spanning models, agents, data pipelines, security tooling, human workflows, and hyperscale infrastructure. Existing governance and security frameworks, including the NIST AI Risk Management Framework and systems security engineering guidance, articulate principles and risk functions but do not provide implementable architectures for multi agent, AI enabled cyber defense.
This paper introduces the Practitioners Blueprint for Secure AI (PBSAI) Governance Ecosystem, a multi agent reference architecture for securing enterprise and hyperscale AI estates. PBSAI organizes responsibilities into a twelve domain taxonomy and defines bounded agent families that mediate between tools and policy through shared context envelopes and structured output contracts. The architecture assumes baseline enterprise security capabilities and encodes key systems security techniques, including analytic monitoring, coordinated defense, and adaptive response. A lightweight formal model of agents, context envelopes, and ecosystem level invariants clarifies the traceability, provenance, and human in the loop guarantees enforced across domains. We demonstrate alignment with NIST AI RMF functions and illustrate application in enterprise SOC and hyperscale defensive environments. PBSAI is proposed as a structured, evidence centric foundation for open ecosystem development and future empirical validation. 

---
# Voxtral Realtime 

**Authors**: Alexander H. Liu, Andy Ehrenberg, Andy Lo, Chen-Yo Sun, Guillaume Lample, Jean-Malo Delignon, Khyathi Raghavi Chandu, Patrick von Platen, Pavankumar Reddy Muddireddy, Rohin Arora, Sanchit Gandhi, Sandeep Subramanian, Soham Ghosh, Srijan Mishra, Abhinav Rastogi, Alan Jeffares, Albert Jiang, Alexandre Sablayrolles, Amélie Héliou, Andrew Bai, Angele Lenglemetz, Anmol Agarwal, Anton Eliseev, Antonia Calvi, Arjun Majumdar, Baptiste Bout, Baptiste Rozière, Baudouin De Monicault, Benjamin Tibi, Clémence Lanfranchi, Connor Chen, Corentin Barreau, Corentin Sautier, Cyprien Courtot, Darius Dabert, Diego de las Casas, Elliot Chane-Sane, Enguerrand Paquin, Faruk Ahmed, Federico Baldassarre, Gabrielle Berrada, Gaëtan Ecrepont, Gauthier Guinet, Genevieve Hayes, Georgii Novikov, Giada Pistilli, Guillaume Martin, Gunjan Dhanuka, Gunshi Gupta, Han Zhou, Indraneel Mukherjee, Irene Zhang, Jaeyoung Kim, Jan Ludziejewski, Jason Rute, Joachim Studnia, John Harvill, Jonas Amar, Josselin Somerville Roberts, Julien Tauran, Karmesh Yadav, Kartik Khandelwal, Kush Jain, Laurence Aitchison, Léonard Blier, Lingxiao Zhao, Louis Martin, Lucile Saulnier, Luyu Gao, Maarten Buyl, Manan Sharma, Margaret Jennings, Marie Pellat, Mark Prins, Mathieu Poirée, Mathilde Guillaumin, Matthieu Dinot, Matthieu Futeral, Maxime Darrin, Maximilian Augustin, Mert Unsal, Mia Chiquier, Nathan Grinsztajn, Neha Gupta, Olivier Bousquet, Olivier Duchenne, Patricia Wang, Paul Jacob, Paul Wambergue, Paula Kurylowicz, Philomène Chagniot, Pierre Stock, Piotr Miłoś, Prateek Gupta, Pravesh Agrawal, Quentin Torroba, Ram Ramrakhya, Rishi Shah, Romain Sauvestre, Roman Soletskyi  

**Link**: [PDF](https://arxiv.org/pdf/2602.11298)  

**Abstract**: We introduce Voxtral Realtime, a natively streaming automatic speech recognition model that matches offline transcription quality at sub-second latency. Unlike approaches that adapt offline models through chunking or sliding windows, Voxtral Realtime is trained end-to-end for streaming, with explicit alignment between audio and text streams. Our architecture builds on the Delayed Streams Modeling framework, introducing a new causal audio encoder and Ada RMS-Norm for improved delay conditioning. We scale pretraining to a large-scale dataset spanning 13 languages. At a delay of 480ms, Voxtral Realtime achieves performance on par with Whisper, the most widely deployed offline transcription system. We release the model weights under the Apache 2.0 license. 

---
# On Decision-Valued Maps and Representational Dependence 

**Authors**: Gil Raitses  

**Link**: [PDF](https://arxiv.org/pdf/2602.11295)  

**Abstract**: A computational engine applied to different representations of the same data can produce different discrete outcomes, with some representations preserving the result and others changing it entirely. A decision-valued map records which representations preserve the outcome and which change it, associating each member of a declared representation family with the discrete result it produces. This paper formalizes decision-valued maps and describes DecisionDB, an infrastructure that logs, replays and audits these relationships using identifiers computed from content and artifacts stored in write-once form. Deterministic replay recovers each recorded decision identifier exactly from stored artifacts, with all three identifying fields matching their persisted values. The contribution partitions representation space into persistence regions and boundaries, and treats decision reuse as a mechanically checkable condition. 

---
# Latent Generative Solvers for Generalizable Long-Term Physics Simulation 

**Authors**: Zituo Chen, Haixu Wu, Sili Deng  

**Link**: [PDF](https://arxiv.org/pdf/2602.11229)  

**Abstract**: We study long-horizon surrogate simulation across heterogeneous PDE systems. We introduce Latent Generative Solvers (LGS), a two-stage framework that (i) maps diverse PDE states into a shared latent physics space with a pretrained VAE, and (ii) learns probabilistic latent dynamics with a Transformer trained by flow matching. Our key mechanism is an uncertainty knob that perturbs latent inputs during training and inference, teaching the solver to correct off-manifold rollout drift and stabilizing autoregressive prediction. We further use flow forcing to update a system descriptor (context) from model-generated trajectories, aligning train/test conditioning and improving long-term stability. We pretrain on a curated corpus of $\sim$2.5M trajectories at $128^2$ resolution spanning 12 PDE families. LGS matches strong deterministic neural-operator baselines on short horizons while substantially reducing rollout drift on long horizons. Learning in latent space plus efficient architectural choices yields up to \textbf{70$\times$} lower FLOPs than non-generative baselines, enabling scalable pretraining. We also show efficient adaptation to an out-of-distribution $256^2$ Kolmogorov flow dataset under limited finetuning budgets. Overall, LGS provides a practical route toward generalizable, uncertainty-aware neural PDE solvers that are more reliable for long-term forecasting and downstream scientific workflows. 

---
# Explaining AI Without Code: A User Study on Explainable AI 

**Authors**: Natalia Abarca, Andrés Carvallo, Claudia López Moncada, Felipe Bravo-Marquez  

**Link**: [PDF](https://arxiv.org/pdf/2602.11159)  

**Abstract**: The increasing use of Machine Learning (ML) in sensitive domains such as healthcare, finance, and public policy has raised concerns about the transparency of automated decisions. Explainable AI (XAI) addresses this by clarifying how models generate predictions, yet most methods demand technical expertise, limiting their value for novices. This gap is especially critical in no-code ML platforms, which seek to democratize AI but rarely include explainability. We present a human-centered XAI module in DashAI, an open-source no-code ML platform. The module integrates three complementary techniques, which are Partial Dependence Plots (PDP), Permutation Feature Importance (PFI), and KernelSHAP, into DashAI's workflow for tabular classification. A user study (N = 20; ML novices and experts) evaluated usability and the impact of explanations. Results show: (i) high task success ($\geq80\%$) across all explainability tasks; (ii) novices rated explanations as useful, accurate, and trustworthy on the Explanation Satisfaction Scale (ESS, Cronbach's $\alpha$ = 0.74, a measure of internal consistency), while experts were more critical of sufficiency and completeness; and (iii) explanations improved perceived predictability and confidence on the Trust in Automation scale (TiA, $\alpha$ = 0.60), with novices showing higher trust than experts. These findings highlight a central challenge for XAI in no-code ML, making explanations both accessible to novices and sufficiently detailed for experts. 

---
# Scaling Verification Can Be More Effective than Scaling Policy Learning for Vision-Language-Action Alignment 

**Authors**: Jacky Kwok, Xilun Zhang, Mengdi Xu, Yuejiang Liu, Azalia Mirhoseini, Chelsea Finn, Marco Pavone  

**Link**: [PDF](https://arxiv.org/pdf/2602.12281)  

**Abstract**: The long-standing vision of general-purpose robots hinges on their ability to understand and act upon natural language instructions. Vision-Language-Action (VLA) models have made remarkable progress toward this goal, yet their generated actions can still misalign with the given instructions. In this paper, we investigate test-time verification as a means to shrink the "intention-action gap.'' We first characterize the test-time scaling law for embodied instruction following and demonstrate that jointly scaling the number of rephrased instructions and generated actions greatly increases test-time sample diversity, often recovering correct actions more efficiently than scaling each dimension independently. To capitalize on these scaling laws, we present CoVer, a contrastive verifier for vision-language-action alignment, and show that our architecture scales gracefully with additional computational resources and data. We then introduce "boot-time compute" and a hierarchical verification inference pipeline for VLAs. At deployment, our framework precomputes a diverse set of rephrased instructions from a Vision-Language-Model (VLM), repeatedly generates action candidates for each instruction, and then uses a verifier to select the optimal high-level prompt and low-level action chunks. Compared to scaling policy pre-training on the same data, our verification approach yields 22% gains in-distribution and 13% out-of-distribution on the SIMPLER benchmark, with a further 45% improvement in real-world experiments. On the PolaRiS benchmark, CoVer achieves 14% gains in task progress and 9% in success rate. 

---
# UniT: Unified Multimodal Chain-of-Thought Test-time Scaling 

**Authors**: Leon Liangyu Chen, Haoyu Ma, Zhipeng Fan, Ziqi Huang, Animesh Sinha, Xiaoliang Dai, Jialiang Wang, Zecheng He, Jianwei Yang, Chunyuan Li, Junzhe Sun, Chu Wang, Serena Yeung-Levy, Felix Juefei-Xu  

**Link**: [PDF](https://arxiv.org/pdf/2602.12279)  

**Abstract**: Unified models can handle both multimodal understanding and generation within a single architecture, yet they typically operate in a single pass without iteratively refining their outputs. Many multimodal tasks, especially those involving complex spatial compositions, multiple interacting objects, or evolving instructions, require decomposing instructions, verifying intermediate results, and making iterative corrections. While test-time scaling (TTS) has demonstrated that allocating additional inference compute for iterative reasoning substantially improves language model performance, extending this paradigm to unified multimodal models remains an open challenge. We introduce UniT, a framework for multimodal chain-of-thought test-time scaling that enables a single unified model to reason, verify, and refine across multiple rounds. UniT combines agentic data synthesis, unified model training, and flexible test-time inference to elicit cognitive behaviors including verification, subgoal decomposition, and content memory. Our key findings are: (1) unified models trained on short reasoning trajectories generalize to longer inference chains at test time; (2) sequential chain-of-thought reasoning provides a more scalable and compute-efficient TTS strategy than parallel sampling; (3) training on generation and editing trajectories improves out-of-distribution visual reasoning. These results establish multimodal test-time scaling as an effective paradigm for advancing both generation and understanding in unified models. 

---
# AttentionRetriever: Attention Layers are Secretly Long Document Retrievers 

**Authors**: David Jiahao Fu, Lam Thanh Do, Jiayu Li, Kevin Chen-Chuan Chang  

**Link**: [PDF](https://arxiv.org/pdf/2602.12278)  

**Abstract**: Retrieval augmented generation (RAG) has been widely adopted to help Large Language Models (LLMs) to process tasks involving long documents. However, existing retrieval models are not designed for long document retrieval and fail to address several key challenges of long document retrieval, including context-awareness, causal dependence, and scope of retrieval. In this paper, we proposed AttentionRetriever, a novel long document retrieval model that leverages attention mechanism and entity-based retrieval to build context-aware embeddings for long document and determine the scope of retrieval. With extensive experiments, we found AttentionRetriever is able to outperform existing retrieval models on long document retrieval datasets by a large margin while remaining as efficient as dense retrieval models. 

---
# Creative Ownership in the Age of AI 

**Authors**: Annie Liang, Jay Lu  

**Link**: [PDF](https://arxiv.org/pdf/2602.12270)  

**Abstract**: Copyright law focuses on whether a new work is "substantially similar" to an existing one, but generative AI can closely imitate style without copying content, a capability now central to ongoing litigation. We argue that existing definitions of infringement are ill-suited to this setting and propose a new criterion: a generative AI output infringes on an existing work if it could not have been generated without that work in its training corpus. To operationalize this definition, we model generative systems as closure operators mapping a corpus of existing works to an output of new works. AI generated outputs are \emph{permissible} if they do not infringe on any existing work according to our criterion. Our results characterize structural properties of permissible generation and reveal a sharp asymptotic dichotomy: when the process of organic creations is light-tailed, dependence on individual works eventually vanishes, so that regulation imposes no limits on AI generation; with heavy-tailed creations, regulation can be persistently constraining. 

---
# On the implicit regularization of Langevin dynamics with projected noise 

**Authors**: Govind Menon, Austin J. Stromme, Adrien Vacher  

**Link**: [PDF](https://arxiv.org/pdf/2602.12257)  

**Abstract**: We study Langevin dynamics with noise projected onto the directions orthogonal to an isometric group action. This mathematical model is introduced to shed new light on the effects of symmetry on stochastic gradient descent for over-parametrized models. Our main result identifies a novel form of implicit regularization: when the initial and target density are both invariant under the group action, Langevin dynamics with projected noise is equivalent in law to Langevin dynamics with isotropic diffusion but with an additional drift term proportional to the negative log volume of the group orbit. We prove this result by constructing a coupling of the two processes via a third process on the group itself, and identify the additional drift as the mean curvature of the orbits. 

---
# A technical curriculum on language-oriented artificial intelligence in translation and specialised communication 

**Authors**: Ralph Krüger  

**Link**: [PDF](https://arxiv.org/pdf/2602.12251)  

**Abstract**: This paper presents a technical curriculum on language-oriented artificial intelligence (AI) in the language and translation (L&T) industry. The curriculum aims to foster domain-specific technical AI literacy among stakeholders in the fields of translation and specialised communication by exposing them to the conceptual and technical/algorithmic foundations of modern language-oriented AI in an accessible way. The core curriculum focuses on 1) vector embeddings, 2) the technical foundations of neural networks, 3) tokenization and 4) transformer neural networks. It is intended to help users develop computational thinking as well as algorithmic awareness and algorithmic agency, ultimately contributing to their digital resilience in AI-driven work environments. The didactic suitability of the curriculum was tested in an AI-focused MA course at the Institute of Translation and Multilingual Communication at TH Koeln. Results suggest the didactic effectiveness of the curriculum, but participant feedback indicates that it should be embedded into higher-level didactic scaffolding - e.g., in the form of lecturer support - in order to enable optimal learning conditions. 

---
# ExtractBench: A Benchmark and Evaluation Methodology for Complex Structured Extraction 

**Authors**: Nick Ferguson, Josh Pennington, Narek Beghian, Aravind Mohan, Douwe Kiela, Sheshansh Agrawal, Thien Hang Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2602.12247)  

**Abstract**: Unstructured documents like PDFs contain valuable structured information, but downstream systems require this data in reliable, standardized formats. LLMs are increasingly deployed to automate this extraction, making accuracy and reliability paramount. However, progress is bottlenecked by two gaps. First, no end-to-end benchmark evaluates PDF-to-JSON extraction under enterprise-scale schema breadth. Second, no principled methodology captures the semantics of nested extraction, where fields demand different notions of correctness (exact match for identifiers, tolerance for quantities, semantic equivalence for names), arrays require alignment, and omission must be distinguished from hallucination. We address both gaps with ExtractBench, an open-source benchmark and evaluation framework for PDF-to-JSON structured extraction. The benchmark pairs 35 PDF documents with JSON Schemas and human-annotated gold labels across economically valuable domains, yielding 12,867 evaluatable fields spanning schema complexities from tens to hundreds of fields. The evaluation framework treats the schema as an executable specification: each field declares its scoring metric. Baseline evaluations reveal that frontier models (GPT-5/5.2, Gemini-3 Flash/Pro, Claude 4.5 Opus/Sonnet) remain unreliable on realistic schemas. Performance degrades sharply with schema breadth, culminating in 0% valid output on a 369-field financial reporting schema across all tested models. We release ExtractBench at this https URL. 

---
# Intrinsic-Energy Joint Embedding Predictive Architectures Induce Quasimetric Spaces 

**Authors**: Anthony Kobanda, Waris Radji  

**Link**: [PDF](https://arxiv.org/pdf/2602.12245)  

**Abstract**: Joint-Embedding Predictive Architectures (JEPAs) aim to learn representations by predicting target embeddings from context embeddings, inducing a scalar compatibility energy in a latent space. In contrast, Quasimetric Reinforcement Learning (QRL) studies goal-conditioned control through directed distance values (cost-to-go) that support reaching goals under asymmetric dynamics. In this short article, we connect these viewpoints by restricting attention to a principled class of JEPA energy functions : intrinsic (least-action) energies, defined as infima of accumulated local effort over admissible trajectories between two states. Under mild closure and additivity assumptions, any intrinsic energy is a quasimetric. In goal-reaching control, optimal cost-to-go functions admit exactly this intrinsic form ; inversely, JEPAs trained to model intrinsic energies lie in the quasimetric value class targeted by QRL. Moreover, we observe why symmetric finite energies are structurally mismatched with one-way reachability, motivating asymmetric (quasimetric) energies when directionality matters. 

---
# Olmix: A Framework for Data Mixing Throughout LM Development 

**Authors**: Mayee F. Chen, Tyler Murray, David Heineman, Matt Jordan, Hannaneh Hajishirzi, Christopher Ré, Luca Soldaini, Kyle Lo  

**Link**: [PDF](https://arxiv.org/pdf/2602.12237)  

**Abstract**: Data mixing -- determining the ratios of data from different domains -- is a first-order concern for training language models (LMs). While existing mixing methods show promise, they fall short when applied during real-world LM development. We present Olmix, a framework that addresses two such challenges. First, the configuration space for developing a mixing method is not well understood -- design choices across existing methods lack justification or consensus and overlook practical issues like data constraints. We conduct a comprehensive empirical study of this space, identifying which design choices lead to a strong mixing method. Second, in practice, the domain set evolves throughout LM development as datasets are added, removed, partitioned, and revised -- a problem setting largely unaddressed by existing works, which assume fixed domains. We study how to efficiently recompute the mixture after the domain set is updated, leveraging information from past mixtures. We introduce mixture reuse, a mechanism that reuses existing ratios and recomputes ratios only for domains affected by the update. Over a sequence of five domain-set updates mirroring real-world LM development, mixture reuse matches the performance of fully recomputing the mix after each update with 74% less compute and improves over training without mixing by 11.6% on downstream tasks. 

---
# Energy-Aware Spike Budgeting for Continual Learning in Spiking Neural Networks for Neuromorphic Vision 

**Authors**: Anika Tabassum Meem, Muntasir Hossain Nadid, Md Zesun Ahmed Mia  

**Link**: [PDF](https://arxiv.org/pdf/2602.12236)  

**Abstract**: Neuromorphic vision systems based on spiking neural networks (SNNs) offer ultra-low-power perception for event-based and frame-based cameras, yet catastrophic forgetting remains a critical barrier to deployment in continually evolving environments. Existing continual learning methods, developed primarily for artificial neural networks, seldom jointly optimize accuracy and energy efficiency, with particularly limited exploration on event-based datasets. We propose an energy-aware spike budgeting framework for continual SNN learning that integrates experience replay, learnable leaky integrate-and-fire neuron parameters, and an adaptive spike scheduler to enforce dataset-specific energy constraints during training. Our approach exhibits modality-dependent behavior: on frame-based datasets (MNIST, CIFAR-10), spike budgeting acts as a sparsity-inducing regularizer, improving accuracy while reducing spike rates by up to 47\%; on event-based datasets (DVS-Gesture, N-MNIST, CIFAR-10-DVS), controlled budget relaxation enables accuracy gains up to 17.45 percentage points with minimal computational overhead. Across five benchmarks spanning both modalities, our method demonstrates consistent performance improvements while minimizing dynamic power consumption, advancing the practical viability of continual learning in neuromorphic vision systems. 

---
# Bandit Learning in Matching Markets with Interviews 

**Authors**: Amirmahdi Mirfakhar, Xuchuang Wang, Mengfan Xu, Hedyeh Beyhaghi, Mohammad Hajiesmaili  

**Link**: [PDF](https://arxiv.org/pdf/2602.12224)  

**Abstract**: Two-sided matching markets rely on preferences from both sides, yet it is often impractical to evaluate preferences. Participants, therefore, conduct a limited number of interviews, which provide early, noisy impressions and shape final decisions. We study bandit learning in matching markets with interviews, modeling interviews as \textit{low-cost hints} that reveal partial preference information to both sides. Our framework departs from existing work by allowing firm-side uncertainty: firms, like agents, may be unsure of their own preferences and can make early hiring mistakes by hiring less preferred agents. To handle this, we extend the firm's action space to allow \emph{strategic deferral} (choosing not to hire in a round), enabling recovery from suboptimal hires and supporting decentralized learning without coordination. We design novel algorithms for (i) a centralized setting with an omniscient interview allocator and (ii) decentralized settings with two types of firm-side feedback. Across all settings, our algorithms achieve time-independent regret, a substantial improvement over the $O(\log T)$ regret bounds known for learning stable matchings without interviews. Also, under mild structured markets, decentralized performance matches the centralized counterpart up to polynomial factors in the number of agents and firms. 

---
# Towards On-Policy SFT: Distribution Discriminant Theory and its Applications in LLM Training 

**Authors**: Miaosen Zhang, Yishan Liu, Shuxia Lin, Xu Yang, Qi Dai, Chong Luo, Weihao Jiang, Peng Hou, Anxiang Zeng, Xin Geng, Baining Guo  

**Link**: [PDF](https://arxiv.org/pdf/2602.12222)  

**Abstract**: Supervised fine-tuning (SFT) is computationally efficient but often yields inferior generalization compared to reinforcement learning (RL). This gap is primarily driven by RL's use of on-policy data. We propose a framework to bridge this chasm by enabling On-Policy SFT. We first present \textbf{\textit{Distribution Discriminant Theory (DDT)}}, which explains and quantifies the alignment between data and the model-induced distribution. Leveraging DDT, we introduce two complementary techniques: (i) \textbf{\textit{In-Distribution Finetuning (IDFT)}}, a loss-level method to enhance generalization ability of SFT, and (ii) \textbf{\textit{Hinted Decoding}}, a data-level technique that can re-align the training corpus to the model's distribution. Extensive experiments demonstrate that our framework achieves generalization performance on par with prominent offline RL algorithms, including DPO and SimPO, while maintaining the efficiency of an SFT pipeline. The proposed framework thus offers a practical alternative in domains where RL is infeasible. We open-source the code here: this https URL 

---
# The Observer Effect in World Models: Invasive Adaptation Corrupts Latent Physics 

**Authors**: Christian Internò, Jumpei Yamaguchi, Loren Amdahl-Culleton, Markus Olhofer, David Klindt, Barbara Hammer  

**Link**: [PDF](https://arxiv.org/pdf/2602.12218)  

**Abstract**: Determining whether neural models internalize physical laws as world models, rather than exploiting statistical shortcuts, remains challenging, especially under out-of-distribution (OOD) shifts. Standard evaluations often test latent capability via downstream adaptation (e.g., fine-tuning or high-capacity probes), but such interventions can change the representations being measured and thus confound what was learned during self-supervised learning (SSL). We propose a non-invasive evaluation protocol, PhyIP. We test whether physical quantities are linearly decodable from frozen representations, motivated by the linear representation hypothesis. Across fluid dynamics and orbital mechanics, we find that when SSL achieves low error, latent structure becomes linearly accessible. PhyIP recovers internal energy and Newtonian inverse-square scaling on OOD tests (e.g., $\rho > 0.90$). In contrast, adaptation-based evaluations can collapse this structure ($\rho \approx 0.05$). These findings suggest that adaptation-based evaluation can obscure latent structures and that low-capacity probes offer a more accurate evaluation of physical world models. 

---
# VIRENA: Virtual Arena for Research, Education, and Democratic Innovation 

**Authors**: Emma Hoes, K. Jonathan Klueser, Fabrizio Gilardi  

**Link**: [PDF](https://arxiv.org/pdf/2602.12207)  

**Abstract**: Digital platforms shape how people communicate, deliberate, and form opinions. Studying these dynamics has become increasingly difficult due to restricted data access, ethical constraints on real-world experiments, and limitations of existing research tools. VIRENA (Virtual Arena) is a platform that enables controlled experimentation in realistic social media environments. Multiple participants interact simultaneously in realistic replicas of feed-based platforms (Instagram, Facebook, Reddit) and messaging apps (WhatsApp, Messenger). Large language model-powered AI agents participate alongside humans with configurable personas and realistic behavior. Researchers can manipulate content moderation approaches, pre-schedule stimulus content, and run experiments across conditions through a visual interface requiring no programming skills. VIRENA makes possible research designs that were previously impractical: studying human--AI interaction in realistic social contexts, experimentally comparing moderation interventions, and observing group deliberation as it unfolds. Built on open-source technologies that ensure data remain under institutional control and comply with data protection requirements, VIRENA is currently in use at the University of Zurich and available for pilot collaborations. Designed for researchers, educators, and public organizations alike, VIRENA's no-code interface makes controlled social media simulation accessible across disciplines and sectors. This paper documents its design, architecture, and capabilities. 

---
# DeepGen 1.0: A Lightweight Unified Multimodal Model for Advancing Image Generation and Editing 

**Authors**: Dianyi Wang, Ruihang Li, Feng Han, Chaofan Ma, Wei Song, Siyuan Wang, Yibin Wang, Yi Xin, Hongjian Liu, Zhixiong Zhang, Shengyuan Ding, Tianhang Wang, Zhenglin Cheng, Tao Lin, Cheng Jin, Kaicheng Yu, Jingjing Chen, Wenjie Wang, Zhongyu Wei, Jiaqi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.12205)  

**Abstract**: Current unified multimodal models for image generation and editing typically rely on massive parameter scales (e.g., >10B), entailing prohibitive training costs and deployment footprints. In this work, we present DeepGen 1.0, a lightweight 5B unified model that achieves comprehensive capabilities competitive with or surpassing much larger counterparts. To overcome the limitations of compact models in semantic understanding and fine-grained control, we introduce Stacked Channel Bridging (SCB), a deep alignment framework that extracts hierarchical features from multiple VLM layers and fuses them with learnable 'think tokens' to provide the generative backbone with structured, reasoning-rich guidance. We further design a data-centric training strategy spanning three progressive stages: (1) Alignment Pre-training on large-scale image-text pairs and editing triplets to synchronize VLM and DiT representations, (2) Joint Supervised Fine-tuning on a high-quality mixture of generation, editing, and reasoning tasks to foster omni-capabilities, and (3) Reinforcement Learning with MR-GRPO, which leverages a mixture of reward functions and supervision signals, resulting in substantial gains in generation quality and alignment with human preferences, while maintaining stable training progress and avoiding visual artifacts. Despite being trained on only ~50M samples, DeepGen 1.0 achieves leading performance across diverse benchmarks, surpassing the 80B HunyuanImage by 28% on WISE and the 27B Qwen-Image-Edit by 37% on UniREditBench. By open-sourcing our training code, weights, and datasets, we provide an efficient, high-performance alternative to democratize unified multimodal research. 

---
# Visual Reasoning Benchmark: Evaluating Multimodal LLMs on Classroom-Authentic Visual Problems from Primary Education 

**Authors**: Mohamed Huti, Alasdair Mackintosh, Amy Waldock, Dominic Andrews, Maxime Lelièvre, Moritz Boos, Tobias Murray, Paul Atherton, Robin A. A. Ince, Oliver G. B. Garrod  

**Link**: [PDF](https://arxiv.org/pdf/2602.12196)  

**Abstract**: AI models have achieved state-of-the-art results in textual reasoning; however, their ability to reason over spatial and relational structures remains a critical bottleneck -- particularly in early-grade maths, which relies heavily on visuals. This paper introduces the visual reasoning benchmark (VRB), a novel dataset designed to evaluate Multimodal Large Language Models (MLLMs) on their ability to solve authentic visual problems from classrooms. This benchmark is built on a set of 701 questions sourced from primary school examinations in Zambia and India, which cover a range of tasks such as reasoning by analogy, pattern completion, and spatial matching. We outline the methodology and development of the benchmark which intentionally uses unedited, minimal-text images to test if models can meet realistic needs of primary education. Our findings reveal a ``jagged frontier'' of capability where models demonstrate better proficiency in static skills such as counting and scaling, but reach a distinct ``spatial ceiling'' when faced with dynamic operations like folding, reflection, and rotation. These weaknesses pose a risk for classroom use on visual reasoning problems, with the potential for incorrect marking, false scaffolding, and reinforcing student misconceptions. Consequently, education-focused benchmarks like the VRB are essential for determining the functional boundaries of multimodal tools used in classrooms. 

---
# SAGEO Arena: A Realistic Environment for Evaluating Search-Augmented Generative Engine Optimization 

**Authors**: Sunghwan Kim, Wooseok Jeong, Serin Kim, Sangam Lee, Dongha Lee  

**Link**: [PDF](https://arxiv.org/pdf/2602.12187)  

**Abstract**: Search-Augmented Generative Engines (SAGE) have emerged as a new paradigm for information access, bridging web-scale retrieval with generative capabilities to deliver synthesized answers. This shift has fundamentally reshaped how web content gains exposure online, giving rise to Search-Augmented Generative Engine Optimization (SAGEO), the practice of optimizing web documents to improve their visibility in AI-generated responses. Despite growing interest, no evaluation environment currently supports comprehensive investigation of SAGEO. Specifically, existing benchmarks lack end-to-end visibility evaluation of optimization strategies, operating on pre-determined candidate documents that abstract away retrieval and reranking preceding generation. Moreover, existing benchmarks discard structural information (e.g., schema markup) present in real web documents, overlooking the rich signals that search systems actively leverage in practice. Motivated by these gaps, we introduce SAGEO Arena, a realistic and reproducible environment for stage-level SAGEO analysis. Our objective is to jointly target search-oriented optimization (SEO) and generation-centric optimization (GEO). To achieve this, we integrate a full generative search pipeline over a large-scale corpus of web documents with rich structural information. Our findings reveal that existing approaches remain largely impractical under realistic conditions and often degrade performance in retrieval and reranking. We also find that structural information helps mitigate these limitations, and that effective SAGEO requires tailoring optimization to each pipeline stage. Overall, our benchmark paves the way for realistic SAGEO evaluation and optimization beyond simplified settings. 

---
# 3DGSNav: Enhancing Vision-Language Model Reasoning for Object Navigation via Active 3D Gaussian Splatting 

**Authors**: Wancai Zheng, Hao Chen, Xianlong Lu, Linlin Ou, Xinyi Yu  

**Link**: [PDF](https://arxiv.org/pdf/2602.12159)  

**Abstract**: Object navigation is a core capability of embodied intelligence, enabling an agent to locate target objects in unknown environments. Recent advances in vision-language models (VLMs) have facilitated zero-shot object navigation (ZSON). However, existing methods often rely on scene abstractions that convert environments into semantic maps or textual representations, causing high-level decision making to be constrained by the accuracy of low-level perception. In this work, we present 3DGSNav, a novel ZSON framework that embeds 3D Gaussian Splatting (3DGS) as persistent memory for VLMs to enhance spatial reasoning. Through active perception, 3DGSNav incrementally constructs a 3DGS representation of the environment, enabling trajectory-guided free-viewpoint rendering of frontier-aware first-person views. Moreover, we design structured visual prompts and integrate them with Chain-of-Thought (CoT) prompting to further improve VLM reasoning. During navigation, a real-time object detector filters potential targets, while VLM-driven active viewpoint switching performs target re-verification, ensuring efficient and reliable recognition. Extensive evaluations across multiple benchmarks and real-world experiments on a quadruped robot demonstrate that our method achieves robust and competitive performance against state-of-the-art this http URL Project Page:this https URL 

---
# dVoting: Fast Voting for dLLMs 

**Authors**: Sicheng Feng, Zigeng Chen, Xinyin Ma, Gongfan Fang, Xinchao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.12153)  

**Abstract**: Diffusion Large Language Models (dLLMs) represent a new paradigm beyond autoregressive modeling, offering competitive performance while naturally enabling a flexible decoding process. Specifically, dLLMs can generate tokens at arbitrary positions in parallel, endowing them with significant potential for parallel test-time scaling, which was previously constrained by severe inefficiency in autoregressive modeling. In this work, we introduce dVoting, a fast voting technique that boosts reasoning capability without training, with only an acceptable extra computational overhead. dVoting is motivated by the observation that, across multiple samples for the same prompt, token predictions remain largely consistent, whereas performance is determined by a small subset of tokens exhibiting cross-sample variability. Leveraging the arbitrary-position generation capability of dLLMs, dVoting performs iterative refinement by sampling, identifying uncertain tokens via consistency analysis, regenerating them through voting, and repeating this process until convergence. Extensive evaluations demonstrate that dVoting consistently improves performance across various benchmarks. It achieves gains of 6.22%-7.66% on GSM8K, 4.40%-7.20% on MATH500, 3.16%-14.84% on ARC-C, and 4.83%-5.74% on MMLU. Our code is available at this https URL 

---
# On the Adoption of AI Coding Agents in Open-source Android and iOS Development 

**Authors**: Muhammad Ahmad Khan, Hasnain Ali, Muneeb Rana, Muhammad Saqib Ilyas, Abdul Ali Bangash  

**Link**: [PDF](https://arxiv.org/pdf/2602.12144)  

**Abstract**: AI coding agents are increasingly contributing to software development, yet their impact on mobile development has received little empirical attention. In this paper, we present the first category-level empirical study of agent-generated code in open-source mobile app projects. We analyzed PR acceptance behaviors across mobile platforms, agents, and task categories using 2,901 AI-authored pull requests (PRs) in 193 verified Android and iOS open-source GitHub repositories in the AIDev dataset. We find that Android projects have received 2x more AI-authored PRs and have achieved higher PR acceptance rate (71%) than iOS (63%), with significant agent-level variation on Android. Across task categories, PRs with routine tasks (feature, fix, and ui) achieve the highest acceptance, while structural changes like refactor and build achieve lower success and longer resolution times. Furthermore, our evolution analysis shows improvement in PR resolution time on Android through mid-2025 before it declined again. Our findings offer the first evidence-based characterization of AI agents effects on OSS mobile projects and establish empirical baselines for evaluating agent-generated contributions to design platform aware agentic systems. 

---
# Learning beyond Teacher: Generalized On-Policy Distillation with Reward Extrapolation 

**Authors**: Wenkai Yang, Weijie Liu, Ruobing Xie, Kai Yang, Saiyong Yang, Yankai Lin  

**Link**: [PDF](https://arxiv.org/pdf/2602.12125)  

**Abstract**: On-policy distillation (OPD), which aligns the student with the teacher's logit distribution on student-generated trajectories, has demonstrated strong empirical gains in improving student performance and often outperforms off-policy distillation and reinforcement learning (RL) paradigms. In this work, we first theoretically show that OPD is a special case of dense KL-constrained RL where the reward function and the KL regularization are always weighted equally and the reference model can by any model. Then, we propose the Generalized On-Policy Distillation (G-OPD) framework, which extends the standard OPD objective by introducing a flexible reference model and a reward scaling factor that controls the relative weight of the reward term against the KL regularization. Through comprehensive experiments on math reasoning and code generation tasks, we derive two novel insights: (1) Setting the reward scaling factor to be greater than 1 (i.e., reward extrapolation), which we term ExOPD, consistently improves over standard OPD across a range of teacher-student size pairings. In particular, in the setting where we merge the knowledge from different domain experts, obtained by applying domain-specific RL to the same student model, back into the original student, ExOPD enables the student to even surpass the teacher's performance boundary and outperform the domain teachers. (2) Building on ExOPD, we further find that in the strong-to-weak distillation setting (i.e., distilling a smaller student from a larger teacher), performing reward correction by choosing the reference model as the teacher's base model before RL yields a more accurate reward signal and further improves distillation performance. However, this choice assumes access to the teacher's pre-RL variant and incurs more computational overhead. We hope our work offers new insights for future research on OPD. 

---
# Meta-Sel: Efficient Demonstration Selection for In-Context Learning via Supervised Meta-Learning 

**Authors**: Xubin Wang, Weijia Jia  

**Link**: [PDF](https://arxiv.org/pdf/2602.12123)  

**Abstract**: Demonstration selection is a practical bottleneck in in-context learning (ICL): under a tight prompt budget, accuracy can change substantially depending on which few-shot examples are included, yet selection must remain cheap enough to run per query over large candidate pools. We propose Meta-Sel, a lightweight supervised meta-learning approach for intent classification that learns a fast, interpretable scoring function for (candidate, query) pairs from labeled training data.
Meta-Sel constructs a meta-dataset by sampling pairs from the training split and using class agreement as supervision, then trains a calibrated logistic regressor on two inexpensive meta-features: TF--IDF cosine similarity and a length-compatibility ratio. At inference time, the selector performs a single vectorized scoring pass over the full candidate pool and returns the top-k demonstrations, requiring no model fine-tuning, no online exploration, and no additional LLM calls. This yields deterministic rankings and makes the selection mechanism straightforward to audit via interpretable feature weights.
Beyond proposing Meta-Sel, we provide a broad empirical study of demonstration selection, benchmarking 12 methods -- spanning prompt engineering baselines, heuristic selection, reinforcement learning, and influence-based approaches -- across four intent datasets and five open-source LLMs. Across this benchmark, Meta-Sel consistently ranks among the top-performing methods, is particularly effective for smaller models where selection quality can partially compensate for limited model capacity, and maintains competitive selection-time overhead. 

---
# KAN-FIF: Spline-Parameterized Lightweight Physics-based Tropical Cyclone Estimation on Meteorological Satellite 

**Authors**: Jiakang Shen, Qinghui Chen, Runtong Wang, Chenrui Xu, Jinglin Zhang, Cong Bai, Feng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.12117)  

**Abstract**: Tropical cyclones (TC) are among the most destructive natural disasters, causing catastrophic damage to coastal regions through extreme winds, heavy rainfall, and storm surges. Timely monitoring of tropical cyclones is crucial for reducing loss of life and property, yet it is hindered by the computational inefficiency and high parameter counts of existing methods on resource-constrained edge devices. Current physics-guided models suffer from linear feature interactions that fail to capture high-order polynomial relationships between TC attributes, leading to inflated model sizes and hardware incompatibility. To overcome these challenges, this study introduces the Kolmogorov-Arnold Network-based Feature Interaction Framework (KAN-FIF), a lightweight multimodal architecture that integrates MLP and CNN layers with spline-parameterized KAN layers. For Maximum Sustained Wind (MSW) prediction, experiments demonstrate that the KAN-FIF framework achieves a $94.8\%$ reduction in parameters (0.99MB vs 19MB) and $68.7\%$ faster inference per sample (2.3ms vs 7.35ms) compared to baseline model Phy-CoCo, while maintaining superior accuracy with $32.5\%$ lower MAE. The offline deployment experiment of the FY-4 series meteorological satellite processor on the Qingyun-1000 development board achieved a 14.41ms per-sample inference latency with the KAN-FIF framework, demonstrating promising feasibility for operational TC monitoring and extending deployability to edge-device AI applications. The code is released at this https URL. 

---
# On the Complexity of Offline Reinforcement Learning with $Q^\star$-Approximation and Partial Coverage 

**Authors**: Haolin Liu, Braham Snyder, Chen-Yu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2602.12107)  

**Abstract**: We study offline reinforcement learning under $Q^\star$-approximation and partial coverage, a setting that motivates practical algorithms such as Conservative $Q$-Learning (CQL; Kumar et al., 2020) but has received limited theoretical attention. Our work is inspired by the following open question: "Are $Q^\star$-realizability and Bellman completeness sufficient for sample-efficient offline RL under partial coverage?"
We answer in the negative by establishing an information-theoretic lower bound. Going substantially beyond this, we introduce a general framework that characterizes the intrinsic complexity of a given $Q^\star$ function class, inspired by model-free decision-estimation coefficients (DEC) for online RL (Foster et al., 2023b; Liu et al., 2025b). This complexity recovers and improves the quantities underlying the guarantees of Chen and Jiang (2022) and Uehara et al. (2023), and extends to broader settings. Our decision-estimation decomposition can be combined with a wide range of $Q^\star$ estimation procedures, modularizing and generalizing existing approaches.
Beyond the general framework, we make further contributions: By developing a novel second-order performance difference lemma, we obtain the first $\epsilon^{-2}$ sample complexity under partial coverage for soft $Q$-learning, improving the $\epsilon^{-4}$ bound of Uehara et al. (2023). We remove Chen and Jiang's (2022) need for additional online interaction when the value gap of $Q^\star$ is unknown. We also give the first characterization of offline learnability for general low-Bellman-rank MDPs without Bellman completeness (Jiang et al., 2017; Du et al., 2021; Jin et al., 2021), a canonical setting in online RL that remains unexplored in offline RL except for special cases. Finally, we provide the first analysis for CQL under $Q^\star$-realizability and Bellman completeness beyond the tabular case. 

---
# Multi Graph Search for High-Dimensional Robot Motion Planning 

**Authors**: Itamar Mishani, Maxim Likhachev  

**Link**: [PDF](https://arxiv.org/pdf/2602.12096)  

**Abstract**: Efficient motion planning for high-dimensional robotic systems, such as manipulators and mobile manipulators, is critical for real-time operation and reliable deployment. Although advances in planning algorithms have enhanced scalability to high-dimensional state spaces, these improvements often come at the cost of generating unpredictable, inconsistent motions or requiring excessive computational resources and memory. In this work, we introduce Multi-Graph Search (MGS), a search-based motion planning algorithm that generalizes classical unidirectional and bidirectional search to a multi-graph setting. MGS maintains and incrementally expands multiple implicit graphs over the state space, focusing exploration on high-potential regions while allowing initially disconnected subgraphs to be merged through feasible transitions as the search progresses. We prove that MGS is complete and bounded-suboptimal, and empirically demonstrate its effectiveness on a range of manipulation and mobile manipulation tasks. Demonstrations, benchmarks and code are available at this https URL. 

---
# DeepSight: An All-in-One LM Safety Toolkit 

**Authors**: Bo Zhang, Jiaxuan Guo, Lijun Li, Dongrui Liu, Sujin Chen, Guanxu Chen, Zhijie Zheng, Qihao Lin, Lewen Yan, Chen Qian, Yijin Zhou, Yuyao Wu, Shaoxiong Guo, Tianyi Du, Jingyi Yang, Xuhao Hu, Ziqi Miao, Xiaoya Lu, Jing Shao, Xia Hu  

**Link**: [PDF](https://arxiv.org/pdf/2602.12092)  

**Abstract**: As the development of Large Models (LMs) progresses rapidly, their safety is also a priority. In current Large Language Models (LLMs) and Multimodal Large Language Models (MLLMs) safety workflow, evaluation, diagnosis, and alignment are often handled by separate tools. Specifically, safety evaluation can only locate external behavioral risks but cannot figure out internal root causes. Meanwhile, safety diagnosis often drifts from concrete risk scenarios and remains at the explainable level. In this way, safety alignment lack dedicated explanations of changes in internal mechanisms, potentially degrading general capabilities. To systematically address these issues, we propose an open-source project, namely DeepSight, to practice a new safety evaluation-diagnosis integrated paradigm. DeepSight is low-cost, reproducible, efficient, and highly scalable large-scale model safety evaluation project consisting of a evaluation toolkit DeepSafe and a diagnosis toolkit DeepScan. By unifying task and data protocols, we build a connection between the two stages and transform safety evaluation from black-box to white-box insight. Besides, DeepSight is the first open source toolkit that support the frontier AI risk evaluation and joint safety evaluation and diagnosis. 

---
# Choose Your Agent: Tradeoffs in Adopting AI Advisors, Coaches, and Delegates in Multi-Party Negotiation 

**Authors**: Kehang Zhu, Lithium Thain, Vivian Tsai, James Wexler, Crystal Qian  

**Link**: [PDF](https://arxiv.org/pdf/2602.12089)  

**Abstract**: As AI usage becomes more prevalent in social contexts, understanding agent-user interaction is critical to designing systems that improve both individual and group outcomes. We present an online behavioral experiment (N = 243) in which participants play three multi-turn bargaining games in groups of three. Each game, presented in randomized order, grants \textit{access to} a single LLM assistance modality: proactive recommendations from an \textit{Advisor}, reactive feedback from a \textit{Coach}, or autonomous execution by a \textit{Delegate}; all modalities are powered by an underlying LLM that achieves superhuman performance in an all-agent environment. On each turn, participants privately decide whether to act manually or use the AI modality available in that game. Despite preferring the \textit{Advisor} modality, participants achieve the highest mean individual gains with the \textit{Delegate}, demonstrating a preference-performance misalignment. Moreover, delegation generates positive externalities; even non-adopting users in \textit{access-to-delegate} treatment groups benefit by receiving higher-quality offers. Mechanism analysis reveals that the \textit{Delegate} agent acts as a market maker, injecting rational, Pareto-improving proposals that restructure the trading environment. Our research reveals a gap between agent capabilities and realized group welfare. While autonomous agents can exhibit super-human strategic performance, their impact on realized welfare gains can be constrained by interfaces, user perceptions, and adoption barriers. Assistance modalities should be designed as mechanisms with endogenous participation; adoption-compatible interaction rules are a prerequisite to improving human welfare with automated assistance. 

---
# ModelWisdom: An Integrated Toolkit for TLA+ Model Visualization, Digest and Repair 

**Authors**: Zhiyong Chen, Jialun Cao, Chang Xu, Shing-Chi Cheung  

**Link**: [PDF](https://arxiv.org/pdf/2602.12058)  

**Abstract**: Model checking in TLA+ provides strong correctness guarantees, yet practitioners continue to face significant challenges in interpreting counterexamples, understanding large state-transition graphs, and repairing faulty models. These difficulties stem from the limited explainability of raw model-checker output and the substantial manual effort required to trace violations back to source specifications. Although the TLA+ Toolbox includes a state diagram viewer, it offers only a static, fully expanded graph without folding, color highlighting, or semantic explanations, which limits its scalability and interpretability. We present ModelWisdom, an interactive environment that uses visualization and large language models to make TLA+ model checking more interpretable and actionable. ModelWisdom offers: (i) Model Visualization, with colorized violation highlighting, click-through links from transitions to TLA+ code, and mapping between violating states and broken properties; (ii) Graph Optimization, including tree-based structuring and node/edge folding to manage large models; (iii) Model Digest, which summarizes and explains subgraphs via large language models (LLMs) and performs preprocessing and partial explanations; and (iv) Model Repair, which extracts error information and supports iterative debugging. Together, these capabilities turn raw model-checker output into an interactive, explainable workflow, improving understanding and reducing debugging effort for nontrivial TLA+ specifications. The website to ModelWisdom is available: this https URL. A demonstrative video can be found at this https URL. 

---
# Fourier Transformers for Latent Crystallographic Diffusion and Generative Modeling 

**Authors**: Jed A. Duersch, Elohan Veillon, Astrid Klipfel, Adlane Sayede, Zied Bouraoui  

**Link**: [PDF](https://arxiv.org/pdf/2602.12045)  

**Abstract**: The discovery of new crystalline materials calls for generative models that handle periodic boundary conditions, crystallographic symmetries, and physical constraints, while scaling to large and structurally diverse unit cells. We propose a reciprocal-space generative pipeline that represents crystals through a truncated Fourier transform of the species-resolved unit-cell density, rather than modeling atomic coordinates directly. This representation is periodicity-native, admits simple algebraic actions of space-group symmetries, and naturally supports variable atomic multiplicities during generation, addressing a common limitation of particle-based approaches. Using only nine Fourier basis functions per spatial dimension, our approach reconstructs unit cells containing up to 108 atoms per chemical species. We instantiate this pipeline with a transformer variational autoencoder over complex-valued Fourier coefficients, and a latent diffusion model that generates in the compressed latent space. We evaluate reconstruction and latent diffusion on the LeMaterial benchmark and compare unconditional generation against coordinate-based baselines in the small-cell regime ($\leq 16$ atoms per unit cell). 

---
# An Empirical Study of the Imbalance Issue in Software Vulnerability Detection 

**Authors**: Yuejun Guo, Qiang Hu, Qiang Tang, Yves Le Traon  

**Link**: [PDF](https://arxiv.org/pdf/2602.12038)  

**Abstract**: Vulnerability detection is crucial to protect software security. Nowadays, deep learning (DL) is the most promising technique to automate this detection task, leveraging its superior ability to extract patterns and representations within extensive code volumes. Despite its promise, DL-based vulnerability detection remains in its early stages, with model performance exhibiting variability across datasets. Drawing insights from other well-explored application areas like computer vision, we conjecture that the imbalance issue (the number of vulnerable code is extremely small) is at the core of the phenomenon. To validate this, we conduct a comprehensive empirical study involving nine open-source datasets and two state-of-the-art DL models. The results confirm our conjecture. We also obtain insightful findings on how existing imbalance solutions perform in vulnerability detection. It turns out that these solutions perform differently as well across datasets and evaluation metrics. Specifically: 1) Focal loss is more suitable to improve the precision, 2) mean false error and class-balanced loss encourages the recall, and 3) random over-sampling facilitates the F1-measure. However, none of them excels across all metrics. To delve deeper, we explore external influences on these solutions and offer insights for developing new solutions. 

---
# On the Sensitivity of Firing Rate-Based Federated Spiking Neural Networks to Differential Privacy 

**Authors**: Luiz Pereira, Mirko Perkusich, Dalton Valadares, Kyller Gorgônio  

**Link**: [PDF](https://arxiv.org/pdf/2602.12009)  

**Abstract**: Federated Neuromorphic Learning (FNL) enables energy-efficient and privacy-preserving learning on devices without centralizing data. However, real-world deployments require additional privacy mechanisms that can significantly alter training signals. This paper analyzes how Differential Privacy (DP) mechanisms, specifically gradient clipping and noise injection, perturb firing-rate statistics in Spiking Neural Networks (SNNs) and how these perturbations are propagated to rate-based FNL coordination. On a speech recognition task under non-IID settings, ablations across privacy budgets and clipping bounds reveal systematic rate shifts, attenuated aggregation, and ranking instability during client selection. Moreover, we relate these shifts to sparsity and memory indicators. Our findings provide actionable guidance for privacy-preserving FNL, specifically regarding the balance between privacy strength and rate-dependent coordination. 

---
# Evaluating AGENTS.md: Are Repository-Level Context Files Helpful for Coding Agents? 

**Authors**: Thibaud Gloaguen, Niels Mündler, Mark Müller, Veselin Raychev, Martin Vechev  

**Link**: [PDF](https://arxiv.org/pdf/2602.11988)  

**Abstract**: A widespread practice in software development is to tailor coding agents to repositories using context files, such as this http URL, by either manually or automatically generating them. Although this practice is strongly encouraged by agent developers, there is currently no rigorous investigation into whether such context files are actually effective for real-world tasks. In this work, we study this question and evaluate coding agents' task completion performance in two complementary settings: established SWE-bench tasks from popular repositories, with LLM-generated context files following agent-developer recommendations, and a novel collection of issues from repositories containing developer-committed context files.
Across multiple coding agents and LLMs, we find that context files tend to reduce task success rates compared to providing no repository context, while also increasing inference cost by over 20%. Behaviorally, both LLM-generated and developer-provided context files encourage broader exploration (e.g., more thorough testing and file traversal), and coding agents tend to respect their instructions. Ultimately, we conclude that unnecessary requirements from context files make tasks harder, and human-written context files should describe only minimal requirements. 

---
# Accelerating Robotic Reinforcement Learning with Agent Guidance 

**Authors**: Haojun Chen, Zili Zou, Chengdong Ma, Yaoxiang Pu, Haotong Zhang, Yuanpei Chen, Yaodong Yang  

**Link**: [PDF](https://arxiv.org/pdf/2602.11978)  

**Abstract**: Reinforcement Learning (RL) offers a powerful paradigm for autonomous robots to master generalist manipulation skills through trial-and-error. However, its real-world application is stifled by severe sample inefficiency. Recent Human-in-the-Loop (HIL) methods accelerate training by using human corrections, yet this approach faces a scalability barrier. Reliance on human supervisors imposes a 1:1 supervision ratio that limits fleet expansion, suffers from operator fatigue over extended sessions, and introduces high variance due to inconsistent human proficiency. We present Agent-guided Policy Search (AGPS), a framework that automates the training pipeline by replacing human supervisors with a multimodal agent. Our key insight is that the agent can be viewed as a semantic world model, injecting intrinsic value priors to structure physical exploration. By using executable tools, the agent provides precise guidance via corrective waypoints and spatial constraints for exploration pruning. We validate our approach on two tasks, ranging from precision insertion to deformable object manipulation. Results demonstrate that AGPS outperforms HIL methods in sample efficiency. This automates the supervision pipeline, unlocking the path to labor-free and scalable robot learning. Project website: this https URL. 

---
# Manifold-Aware Temporal Domain Generalization for Large Language Models 

**Authors**: Yiheng Yao, Zekun Cai, Xinyuan Song, Hiroki Hill Kobayashi, Xuan Song, Ryosuke Shibasaki, Liang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2602.11965)  

**Abstract**: Temporal distribution shifts are pervasive in real-world deployments of Large Language Models (LLMs), where data evolves continuously over time. While Temporal Domain Generalization (TDG) seeks to model such structured evolution, existing approaches characterize model adaptation in the full parameter space. This formulation becomes computationally infeasible for modern LLMs. This paper introduces a geometric reformulation of TDG under parameter-efficient fine-tuning. We establish that the low-dimensional temporal structure underlying model evolution can be preserved under parameter-efficient reparameterization, enabling temporal modeling without operating in the ambient parameter space. Building on this principle, we propose Manifold-aware Temporal LoRA (MaT-LoRA), which constrains temporal updates to a shared low-dimensional manifold within a low-rank adaptation subspace, and models its evolution through a structured temporal core. This reparameterization dramatically reduces temporal modeling complexity while retaining expressive power. Extensive experiments on synthetic and real-world datasets, including scientific documents, news publishers, and review ratings, demonstrate that MaT-LoRA achieves superior temporal generalization performance with practical scalability for LLMs. 

---
# TAVAE: A VAE with Adaptable Priors Explains Contextual Modulation in the Visual Cortex 

**Authors**: Balázs Meszéna, Keith T. Murray, Julien Corbo, O. Batuhan Erkat, Márton A. Hajnal, Pierre-Olivier Polack, Gergő Orbán  

**Link**: [PDF](https://arxiv.org/pdf/2602.11956)  

**Abstract**: The brain interprets visual information through learned regularities, a computation formalized as probabilistic inference under a prior. The visual cortex establishes priors for this inference, some delivered through established top-down connections that inform low-level cortices about statistics represented at higher levels in the cortical hierarchy. While evidence shows that adaptation leads to priors reflecting the structure of natural images, it remains unclear whether similar priors can be flexibly acquired when learning a specific task. To investigate this, we built a generative model of V1 optimized for a simple discrimination task and analyzed it together with large-scale recordings from mice performing an analogous task. In line with recent approaches, we assumed that neuronal activity in V1 corresponds to latent posteriors in the generative model, enabling investigation of task-related priors in neuronal responses. To obtain a flexible test bed, we extended the VAE formalism so that a task can be acquired efficiently by reusing previously learned representations. Task-specific priors learned by this Task-Amortized VAE were used to investigate biases in mice and model when presenting stimuli that violated trained task statistics. Mismatch between learned task statistics and incoming sensory evidence produced signatures of uncertainty in stimulus category in the TAVAE posterior, reflecting properties of bimodal response profiles in V1 recordings. The task-optimized generative model accounted for key characteristics of V1 population activity, including within-day updates to population responses. Our results confirm that flexible task-specific contextual priors can be learned on demand by the visual system and deployed as early as the entry level of visual cortex. 

---
# Towards Performance-Enhanced Model-Contrastive Federated Learning using Historical Information in Heterogeneous Scenarios 

**Authors**: Hongliang Zhang, Jiguo Yu, Guijuan Wang, Wenshuo Ma, Tianqing He, Baobao Chai, Chunqiang Hu  

**Link**: [PDF](https://arxiv.org/pdf/2602.11945)  

**Abstract**: Federated Learning (FL) enables multiple nodes to collaboratively train a model without sharing raw data. However, FL systems are usually deployed in heterogeneous scenarios, where nodes differ in both data distributions and participation frequencies, which undermines the FL performance. To tackle the above issue, this paper proposes PMFL, a performance-enhanced model-contrastive federated learning framework using historical training information. Specifically, on the node side, we design a novel model-contrastive term into the node optimization objective by incorporating historical local models to capture stable contrastive points, thereby improving the consistency of model updates in heterogeneous data distributions.
On the server side, we utilize the cumulative participation count of each node to adaptively adjust its aggregation weight, thereby correcting the bias in the global objective caused by different node participation frequencies. Furthermore, the updated global model incorporates historical global models to reduce its fluctuations in performance between adjacent rounds. Extensive experiments demonstrate that PMFL achieves superior performance compared with existing FL methods in heterogeneous scenarios. 

---
# Synthesis of Late Gadolinium Enhancement Images via Implicit Neural Representations for Cardiac Scar Segmentation 

**Authors**: Soufiane Ben Haddou, Laura Alvarez-Florez, Erik J. Bekkers, Fleur V. Y. Tjong, Ahmad S. Amin, Connie R. Bezzina, Ivana Išgum  

**Link**: [PDF](https://arxiv.org/pdf/2602.11942)  

**Abstract**: Late gadolinium enhancement (LGE) imaging is the clinical standard for myocardial scar assessment, but limited annotated datasets hinder the development of automated segmentation methods. We propose a novel framework that synthesises both LGE images and their corresponding segmentation masks using implicit neural representations (INRs) combined with denoising diffusion models. Our approach first trains INRs to capture continuous spatial representations of LGE data and associated myocardium and fibrosis masks. These INRs are then compressed into compact latent embeddings, preserving essential anatomical information. A diffusion model operates on this latent space to generate new representations, which are decoded into synthetic LGE images with anatomically consistent segmentation masks. Experiments on 133 cardiac MRI scans suggest that augmenting training data with 200 synthetic volumes contributes to improved fibrosis segmentation performance, with the Dice score showing an increase from 0.509 to 0.524. Our approach provides an annotation-free method to help mitigate data this http URL code for this research is publicly available. 

---
# IncompeBench: A Permissively Licensed, Fine-Grained Benchmark for Music Information Retrieval 

**Authors**: Benjamin Clavié, Atoof Shakir, Jonah Turner, Sean Lee, Aamir Shakir, Makoto P. Kato  

**Link**: [PDF](https://arxiv.org/pdf/2602.11941)  

**Abstract**: Multimodal Information Retrieval has made significant progress in recent years, leveraging the increasingly strong multimodal abilities of deep pre-trained models to represent information across modalities. Music Information Retrieval (MIR), in particular, has considerably increased in quality, with neural representations of music even making its way into everyday life products. However, there is a lack of high-quality benchmarks for evaluating music retrieval performance. To address this issue, we introduce \textbf{IncompeBench}, a carefully annotated benchmark comprising $1,574$ permissively licensed, high-quality music snippets, $500$ diverse queries, and over $125,000$ individual relevance judgements. These annotations were created through the use of a multi-stage pipeline, resulting in high agreement between human annotators and the generated data. The resulting datasets are publicly available at this https URL and this https URL with the prompts available at this https URL. 

---
# AdaptEvolve: Improving Efficiency of Evolutionary AI Agents through Adaptive Model Selection 

**Authors**: Pretam Ray, Pratik Prabhanjan Brahma, Zicheng Liu, Emad Barsoum  

**Link**: [PDF](https://arxiv.org/pdf/2602.11931)  

**Abstract**: Evolutionary agentic systems intensify the trade-off between computational efficiency and reasoning capability by repeatedly invoking large language models (LLMs) during inference. This setting raises a central question: how can an agent dynamically select an LLM that is sufficiently capable for the current generation step while remaining computationally efficient? While model cascades offer a practical mechanism for balancing this trade-off, existing routing strategies typically rely on static heuristics or external controllers and do not explicitly account for model uncertainty. We introduce AdaptEvolve: Adaptive LLM Selection for Multi-LLM Evolutionary Refinement within an evolutionary sequential refinement framework that leverages intrinsic generation confidence to estimate real-time solvability. Empirical results show that confidence-driven selection yields a favourable Pareto frontier, reducing total inference cost by an average of 37.9% across benchmarks while retaining 97.5% of the upper-bound accuracy of static large-model baselines. Our code is available at this https URL. 

---
# Who Does What? Archetypes of Roles Assigned to LLMs During Human-AI Decision-Making 

**Authors**: Shreya Chappidi, Jatinder Singh, Andra V. Krauze  

**Link**: [PDF](https://arxiv.org/pdf/2602.11924)  

**Abstract**: LLMs are increasingly supporting decision-making across high-stakes domains, requiring critical reflection on the socio-technical factors that shape how humans and LLMs are assigned roles and interact during human-in-the-loop decision-making. This paper introduces the concept of human-LLM archetypes -- defined as re-curring socio-technical interaction patterns that structure the roles of humans and LLMs in collaborative decision-making. We describe 17 human-LLM archetypes derived from a scoping literature review and thematic analysis of 113 LLM-supported decision-making papers. Then, we evaluate these diverse archetypes across real-world clinical diagnostic cases to examine the potential effects of adopting distinct human-LLM archetypes on LLM outputs and decision outcomes. Finally, we present relevant tradeoffs and design choices across human-LLM archetypes, including decision control, social hierarchies, cognitive forcing strategies, and information requirements. Through our analysis, we show that selection of human-LLM interaction archetype can influence LLM outputs and decisions, bringing important risks and considerations for the designers of human-AI decision-making systems 

---
# DynaHOI: Benchmarking Hand-Object Interaction for Dynamic Target 

**Authors**: BoCheng Hu, Zhonghan Zhao, Kaiyue Zhou, Hongwei Wang, Gaoang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.11919)  

**Abstract**: Most existing hand motion generation benchmarks for hand-object interaction (HOI) focus on static objects, leaving dynamic scenarios with moving targets and time-critical coordination largely untested. To address this gap, we introduce the DynaHOI-Gym, a unified online closed-loop platform with parameterized motion generators and rollout-based metrics for dynamic capture evaluation. Built on DynaHOI-Gym, we release DynaHOI-10M, a large-scale benchmark with 10M frames and 180K hand capture trajectories, whose target motions are organized into 8 major categories and 22 fine-grained subcategories. We also provide a simple observe-before-act baseline (ObAct) that integrates short-term observations with the current frame via spatiotemporal attention to predict actions, achieving an 8.1% improvement in location success rate. 

---
# Leveraging LLMs to support co-evolution between definitions and instances of textual DSLs: A Systematic Evaluation 

**Authors**: Weixing Zhang, Bowen Jiang, Yuhong Fu, Anne Koziolek, Regina Hebig, Daniel Strüber  

**Link**: [PDF](https://arxiv.org/pdf/2602.11904)  

**Abstract**: Software languages evolve over time for reasons such as feature additions. When grammars evolve, textual instances that originally conformed to them may become outdated. While model-driven engineering provides many techniques for co-evolving models with metamodel changes, these approaches are not designed for textual DSLs and may lose human-relevant information such as layout and comments. This study systematically evaluates the potential of large language models (LLMs) for co-evolving grammars and instances of textual DSLs. Using Claude Sonnet 4.5 and GPT-5.2 across ten case languages with ten runs each, we assess both correctness and preservation of human-oriented information. Results show strong performance on small-scale cases ($\geq$94% precision and recall for instances requiring fewer than 20 modified lines), but performance degraded with scale: Claude maintains 85% recall at 40 lines, while GPT fails on the largest instances. Response time increases substantially with instance size, and grammar evolution complexity and deletion granularity affect performance more than change type. These findings clarify when LLM-based co-evolution is effective and where current limitations remain. 

---
# Mitigating Mismatch within Reference-based Preference Optimization 

**Authors**: Suqin Yuan, Xingrui Yu, Jiyang Zheng, Lei Feng, Dadong Wang, Ivor Tsang, Tongliang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2602.11902)  

**Abstract**: Direct Preference Optimization (DPO) has become the de facto standard for offline preference alignment of large language models, but its reliance on a reference policy introduces a critical tension. DPO weighs each update relative to a reference, which stabilizes the training by regularizing the updates within a trusted region. This reliance becomes problematic for pessimistic pairs, where the reference model prefers the rejected response. For these pairs, DPO prematurely attenuates the gradient as soon as the policy margin ($\Delta_\theta$) merely beats the reference margin ($\Delta_{\mathrm{ref}}$) even if the policy is still wrong ($\Delta_{\theta}<0$). We name this failure premature satisfaction, which is a concrete form of the training-inference mismatch. Reference-free objectives remove this mismatch by optimizing the absolute margin, but at the cost of discarding the stabilizing signal of the reference. We mitigate this tension with Hybrid-DPO (HyPO), a drop-in modification to DPO that applies reference conditionally: HyPO behaves exactly like DPO when the reference is optimistic or neutral, and it treats the reference as neutral when it is pessimistic by replacing $\Delta_\theta-\Delta_{\mathrm{ref}}$ with $\Delta_\theta-\max\{0,\Delta_{\mathrm{ref}}\}$. This one-line change strictly strengthens per-example learning signals on pessimistic pairs while preserving DPO's objective form and computational cost. By conditionally debiasing the pessimistic reference signal, HyPO mitigates premature satisfaction; empirically, across preference alignment, HyPO improves inference-aligned metrics and achieves higher pairwise win rates. Our results provide evidence that direct preference alignment could be enhanced by conditionally debiasing the reference signal, rather than discarding it. 

---
# Agentic AI for Cybersecurity: A Meta-Cognitive Architecture for Governable Autonomy 

**Authors**: Andrei Kojukhov, Arkady Bovshover  

**Link**: [PDF](https://arxiv.org/pdf/2602.11897)  

**Abstract**: Contemporary AI-driven cybersecurity systems are predominantly architected as model-centric detection and automation pipelines optimized for task-level performance metrics such as accuracy and response latency. While effective for bounded classification tasks, these architectures struggle to support accountable decision-making under adversarial uncertainty, where actions must be justified, governed, and aligned with organizational and regulatory constraints. This paper argues that cybersecurity orchestration should be reconceptualized as an agentic, multi-agent cognitive system, rather than a linear sequence of detection and response components. We introduce a conceptual architectural framework in which heterogeneous AI agents responsible for detection, hypothesis formation, contextual interpretation, explanation, and governance are coordinated through an explicit meta-cognitive judgement function. This function governs decision readiness and dynamically calibrates system autonomy when evidence is incomplete, conflicting, or operationally risky. By synthesizing distributed cognition theory, multi-agent systems research, and responsible AI governance frameworks, we demonstrate that modern security operations already function as distributed cognitive systems, albeit without an explicit organizing principle. Our contribution is to make this cognitive structure architecturally explicit and governable by embedding meta-cognitive judgement as a first-class system function. We discuss implications for security operations centers, accountable autonomy, and the design of next-generation AI-enabled cyber defence architectures. The proposed framework shifts the focus of AI in cybersecurity from optimizing isolated predictions to governing autonomy under uncertainty. 

---
# Where Bits Matter in World Model Planning: A Paired Mixed-Bit Study for Efficient Spatial Reasoning 

**Authors**: Suraj Ranganath, Anish Patnaik, Vaishak Menon  

**Link**: [PDF](https://arxiv.org/pdf/2602.11882)  

**Abstract**: Efficient spatial reasoning requires world models that remain reliable under tight precision budgets. We study whether low-bit planning behavior is determined mostly by total bitwidth or by where bits are allocated across modules. Using DINO-WM on the Wall planning task, we run a paired-goal mixed-bit evaluation across uniform, mixed, asymmetric, and layerwise variants under two planner budgets. We observe a consistent three-regime pattern: 8-bit and 6-bit settings remain close to FP16, 3-bit settings collapse, and 4-bit settings are allocation-sensitive. In that transition region, preserving encoder precision improves planning relative to uniform quantization, and near-size asymmetric variants show the same encoder-side direction. In a later strict 22-cell replication with smaller per-cell episode count, the mixed-versus-uniform INT4 sign becomes budget-conditioned, which further highlights the sensitivity of this transition regime. These findings motivate module-aware, budget-aware quantization policies as a broader research direction for efficient spatial reasoning. Code and run artifacts are available at this https URL. 

---
# SynthRAR: Ring Artifacts Reduction in CT with Unrolled Network and Synthetic Data Training 

**Authors**: Hongxu Yang, Levente Lippenszky, Edina Timko, Gopal Avinash  

**Link**: [PDF](https://arxiv.org/pdf/2602.11880)  

**Abstract**: Defective and inconsistent responses in CT detectors can cause ring and streak artifacts in the reconstructed images, making them unusable for clinical purposes. In recent years, several ring artifact reduction solutions have been proposed in the image domain or in the sinogram domain using supervised deep learning methods. However, these methods require dedicated datasets for training, leading to a high data collection cost. Furthermore, existing approaches focus exclusively on either image-space or sinogram-space correction, neglecting the intrinsic correlations from the forward operation of the CT geometry. Based on the theoretical analysis of non-ideal CT detector responses, the RAR problem is reformulated as an inverse problem by using an unrolled network, which considers non-ideal response together with linear forward-projection with CT geometry. Additionally, the intrinsic correlations of ring artifacts between the sinogram and image domains are leveraged through synthetic data derived from natural images, enabling the trained model to correct artifacts without requiring real-world clinical data. Extensive evaluations on diverse scanning geometries and anatomical regions demonstrate that the model trained on synthetic data consistently outperforms existing state-of-the-art methods. 

---
# Towards Fair and Comprehensive Evaluation of Routers in Collaborative LLM Systems 

**Authors**: Wanxing Wu, He Zhu, Yixia Li, Lei Yang, Jiehui Zhao, Hongru Wang, Jian Yang, Benyou Wang, Bingyi Jing, Guanhua Chen  

**Link**: [PDF](https://arxiv.org/pdf/2602.11877)  

**Abstract**: Large language models (LLMs) have achieved success, but cost and privacy constraints necessitate deploying smaller models locally while offloading complex queries to cloud-based models. Existing router evaluations are unsystematic, overlooking scenario-specific requirements and out-of-distribution robustness. We propose RouterXBench, a principled evaluation framework with three dimensions: router ability, scenario alignment, and cross-domain robustness. Unlike prior work that relies on output probabilities or external embeddings, we utilize internal hidden states that capture model uncertainty before answer generation. We introduce ProbeDirichlet, a lightweight router that aggregates cross-layer hidden states via learnable Dirichlet distributions with probabilistic training. Trained on multi-domain data, it generalizes robustly across in-domain and out-of-distribution scenarios. Our results show ProbeDirichlet achieves 16.68% and 18.86% relative improvements over the best baselines in router ability and high-accuracy scenarios, with consistent performance across model families, model scales, heterogeneous tasks, and agentic workflows. 

---
# Zooming without Zooming: Region-to-Image Distillation for Fine-Grained Multimodal Perception 

**Authors**: Lai Wei, Liangbo He, Jun Lan, Lingzhong Dong, Yutong Cai, Siyuan Li, Huijia Zhu, Weiqiang Wang, Linghe Kong, Yue Wang, Zhuosheng Zhang, Weiran Huang  

**Link**: [PDF](https://arxiv.org/pdf/2602.11858)  

**Abstract**: Multimodal Large Language Models (MLLMs) excel at broad visual understanding but still struggle with fine-grained perception, where decisive evidence is small and easily overwhelmed by global context. Recent "Thinking-with-Images" methods alleviate this by iteratively zooming in and out regions of interest during inference, but incur high latency due to repeated tool calls and visual re-encoding. To address this, we propose Region-to-Image Distillation, which transforms zooming from an inference-time tool into a training-time primitive, thereby internalizing the benefits of agentic zooming into a single forward pass of an MLLM. In particular, we first zoom in to micro-cropped regions to let strong teacher models generate high-quality VQA data, and then distill this region-grounded supervision back to the full image. After training on such data, the smaller student model improves "single-glance" fine-grained perception without tool use. To rigorously evaluate this capability, we further present ZoomBench, a hybrid-annotated benchmark of 845 VQA data spanning six fine-grained perceptual dimensions, together with a dual-view protocol that quantifies the global--regional "zooming gap". Experiments show that our models achieve leading performance across multiple fine-grained perception benchmarks, and also improve general multimodal cognition on benchmarks such as visual reasoning and GUI agents. We further discuss when "Thinking-with-Images" is necessary versus when its gains can be distilled into a single forward pass. Our code is available at this https URL. 

---
# Resource-Aware Deployment Optimization for Collaborative Intrusion Detection in Layered Networks 

**Authors**: André García Gómez, Ines Rieger, Wolfgang Hotwagner, Max Landauer, Markus Wurzenberger, Florian Skopik, Edgar Weippl  

**Link**: [PDF](https://arxiv.org/pdf/2602.11851)  

**Abstract**: Collaborative Intrusion Detection Systems (CIDS) are increasingly adopted to counter cyberattacks, as their collaborative nature enables them to adapt to diverse scenarios across heterogeneous environments. As distributed critical infrastructure operates in rapidly evolving environments, such as drones in both civil and military domains, there is a growing need for CIDS architectures that can flexibly accommodate these dynamic changes. In this study, we propose a novel CIDS framework designed for easy deployment across diverse distributed environments. The framework dynamically optimizes detector allocation per node based on available resources and data types, enabling rapid adaptation to new operational scenarios with minimal computational overhead. We first conducted a comprehensive literature review to identify key characteristics of existing CIDS architectures. Based on these insights and real-world use cases, we developed our CIDS framework, which we evaluated using several distributed datasets that feature different attack chains and network topologies. Notably, we introduce a public dataset based on a realistic cyberattack targeting a ground drone aimed at sabotaging critical infrastructure. Experimental results demonstrate that the proposed CIDS framework can achieve adaptive, efficient intrusion detection in distributed settings, automatically reconfiguring detectors to maintain an optimal configuration, without requiring heavy computation, since all experiments were conducted on edge devices. 

---
# Improving Neural Retrieval with Attribution-Guided Query Rewriting 

**Authors**: Moncef Garouani, Josiane Mothe  

**Link**: [PDF](https://arxiv.org/pdf/2602.11841)  

**Abstract**: Neural retrievers are effective but brittle: underspecified or ambiguous queries can misdirect ranking even when relevant documents exist. Existing approaches address this brittleness only partially: LLMs rewrite queries without retriever feedback, and explainability methods identify misleading tokens but are used for post-hoc analysis. We close this loop and propose an attribution-guided query rewriting method that uses token-level explanations to guide query rewriting. For each query, we compute gradient-based token attributions from the retriever and then use these scores as soft guidance in a structured prompt to an LLM that clarifies weak or misleading query components while preserving intent. Evaluated on BEIR collections, the resulting rewrites consistently improve retrieval effectiveness over strong baselines, with larger gains for implicit or ambiguous information needs. 

---
# ULTRA:Urdu Language Transformer-based Recommendation Architecture 

**Authors**: Alishbah Bashir, Fatima Qaiser, Ijaz Hussain  

**Link**: [PDF](https://arxiv.org/pdf/2602.11836)  

**Abstract**: Urdu, as a low-resource language, lacks effective semantic content recommendation systems, particularly in the domain of personalized news retrieval. Existing approaches largely rely on lexical matching or language-agnostic techniques, which struggle to capture semantic intent and perform poorly under varying query lengths and information needs. This limitation results in reduced relevance and adaptability in Urdu content recommendation. We propose ULTRA (Urdu Language Transformer-based Recommendation Architecture),an adaptive semantic recommendation framework designed to address these challenges. ULTRA introduces a dual-embedding architecture with a query-length aware routing mechanism that dynamically distinguishes between short, intent-focused queries and longer, context-rich queries. Based on a threshold-driven decision process, user queries are routed to specialized semantic pipelines optimized for either title/headline-level or full-content/document level representations, ensuring appropriate semantic granularity during retrieval. The proposed system leverages transformer-based embeddings and optimized pooling strategies to move beyond surface-level keyword matching and enable context-aware similarity search. Extensive experiments conducted on a large-scale Urdu news corpus demonstrate that the proposed architecture consistently improves recommendation relevance across diverse query types. Results show gains in precision above 90% compared to single-pipeline baselines, highlighting the effectiveness of query-adaptive semantic alignment for low-resource languages. The findings establish ULTRA as a robust and generalizable content recommendation architecture, offering practical design insights for semantic retrieval systems in low-resource language settings. 

---
# Evaluating LLM Safety Under Repeated Inference via Accelerated Prompt Stress Testing 

**Authors**: Keita Broadwater  

**Link**: [PDF](https://arxiv.org/pdf/2602.11786)  

**Abstract**: Traditional benchmarks for large language models (LLMs) primarily assess safety risk through breadth-oriented evaluation across diverse tasks. However, real-world deployment exposes a different class of risk: operational failures arising from repeated inference on identical or near-identical prompts rather than broad task generalization. In high-stakes settings, response consistency and safety under sustained use are critical. We introduce Accelerated Prompt Stress Testing (APST), a depth-oriented evaluation framework inspired by reliability engineering. APST repeatedly samples identical prompts under controlled operational conditions (e.g., decoding temperature) to surface latent failure modes including hallucinations, refusal inconsistency, and unsafe completions. Rather than treating failures as isolated events, APST models them as stochastic outcomes of independent inference events. We formalize safety failures using Bernoulli and binomial models to estimate per-inference failure probabilities, enabling quantitative comparison of reliability across models and decoding configurations. Applying APST to multiple instruction-tuned LLMs evaluated on AIR-BENCH-derived safety prompts, we find that models with similar benchmark-aligned scores can exhibit substantially different empirical failure rates under repeated sampling, particularly as temperature increases. These results demonstrate that shallow, single-sample evaluation can obscure meaningful reliability differences under sustained use. APST complements existing benchmarks by providing a practical framework for evaluating LLM safety and reliability under repeated inference, bridging benchmark alignment and deployment-oriented risk assessment. 

---
# Safe Fairness Guarantees Without Demographics in Classification: Spectral Uncertainty Set Perspective 

**Authors**: Ainhize Barrainkua, Santiago Mazuelas, Novi Quadrianto, Jose A. Lozano  

**Link**: [PDF](https://arxiv.org/pdf/2602.11785)  

**Abstract**: As automated classification systems become increasingly prevalent, concerns have emerged over their potential to reinforce and amplify existing societal biases. In the light of this issue, many methods have been proposed to enhance the fairness guarantees of classifiers. Most of the existing interventions assume access to group information for all instances, a requirement rarely met in practice. Fairness without access to demographic information has often been approached through robust optimization techniques,which target worst-case outcomes over a set of plausible distributions known as the uncertainty set. However, their effectiveness is strongly influenced by the chosen uncertainty set. In fact, existing approaches often overemphasize outliers or overly pessimistic scenarios, compromising both overall performance and fairness. To overcome these limitations, we introduce SPECTRE, a minimax-fair method that adjusts the spectrum of a simple Fourier feature mapping and constrains the extent to which the worst-case distribution can deviate from the empirical distribution. We perform extensive experiments on the American Community Survey datasets involving 20 states. The safeness of SPECTRE comes as it provides the highest average values on fairness guarantees together with the smallest interquartile range in comparison to state-of-the-art approaches, even compared to those with access to demographic group information. In addition, we provide a theoretical analysis that derives computable bounds on the worst-case error for both individual groups and the overall population, as well as characterizes the worst-case distributions responsible for these extremal performances 

---
# MiniCPM-SALA: Hybridizing Sparse and Linear Attention for Efficient Long-Context Modeling 

**Authors**: MiniCPM Team, Wenhao An, Yingfa Chen, Yewei Fang, Jiayi Li, Xin Li, Yaohui Li, Yishan Li, Yuxuan Li, Biyuan Lin, Chuan Liu, Hezi Liu, Siyuan Liu, Hongya Lyu, Yinxu Pan, Shixin Ren, Xingyu Shen, Zhou Su, Haojun Sun, Yangang Sun, Zhen Leng Thai, Xin Tian, Rui Wang, Xiaorong Wang, Yudong Wang, Bo Wu, Xiaoyue Xu, Dong Xu, Shuaikang Xue, Jiawei Yang, Bowen Zhang, Jinqian Zhang, Letian Zhang, Shengnan Zhang, Xinyu Zhang, Xinyuan Zhang, Zhu Zhang, Hengyu Zhao, Jiacheng Zhao, Jie Zhou, Zihan Zhou, Shuo Wang, Chaojun Xiao, Xu Han, Zhiyuan Liu, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2602.11761)  

**Abstract**: The evolution of large language models (LLMs) towards applications with ultra-long contexts faces challenges posed by the high computational and memory costs of the Transformer architecture. While existing sparse and linear attention mechanisms attempt to mitigate these issues, they typically involve a trade-off between memory efficiency and model performance. This paper introduces MiniCPM-SALA, a 9B-parameter hybrid architecture that integrates the high-fidelity long-context modeling of sparse attention (InfLLM-V2) with the global efficiency of linear attention (Lightning Attention). By employing a layer selection algorithm to integrate these mechanisms in a 1:3 ratio and utilizing a hybrid positional encoding (HyPE), the model maintains efficiency and performance for long-context tasks. Furthermore, we introduce a cost-effective continual training framework that transforms pre-trained Transformer-based models into hybrid models, which reduces training costs by approximately 75% compared to training from scratch. Extensive experiments show that MiniCPM-SALA maintains general capabilities comparable to full-attention models while offering improved efficiency. On a single NVIDIA A6000D GPU, the model achieves up to 3.5x the inference speed of the full-attention model at the sequence length of 256K tokens and supports context lengths of up to 1M tokens, a scale where traditional full-attention 8B models fail because of memory constraints. 

---
# Cooperation Breakdown in LLM Agents Under Communication Delays 

**Authors**: Keita Nishimoto, Kimitaka Asatani, Ichiro Sakata  

**Link**: [PDF](https://arxiv.org/pdf/2602.11754)  

**Abstract**: LLM-based multi-agent systems (LLM-MAS), in which autonomous AI agents cooperate to solve tasks, are gaining increasing attention. For such systems to be deployed in society, agents must be able to establish cooperation and coordination under real-world computational and communication constraints. We propose the FLCOA framework (Five Layers for Cooperation/Coordination among Autonomous Agents) to conceptualize how cooperation and coordination emerge in groups of autonomous agents, and highlight that the influence of lower-layer factors - especially computational and communication resources - has been largely overlooked. To examine the effect of communication delay, we introduce a Continuous Prisoner's Dilemma with Communication Delay and conduct simulations with LLM-based agents. As delay increases, agents begin to exploit slower responses even without explicit instructions. Interestingly, excessive delay reduces cycles of exploitation, yielding a U-shaped relationship between delay magnitude and mutual cooperation. These results suggest that fostering cooperation requires attention not only to high-level institutional design but also to lower-layer factors such as communication delay and resource allocation, pointing to new directions for MAS research. 

---
# AmbiBench: Benchmarking Mobile GUI Agents Beyond One-Shot Instructions in the Wild 

**Authors**: Jiazheng Sun, Mingxuan Li, Yingying Zhang, Jiayang Niu, Yachen Wu, Ruihan Jin, Shuyu Lei, Pengrongrui Tan, Zongyu Zhang, Ruoyi Wang, Jiachen Yang, Boyu Yang, Jiacheng Liu, Xin Peng  

**Link**: [PDF](https://arxiv.org/pdf/2602.11750)  

**Abstract**: Benchmarks are paramount for gauging progress in the domain of Mobile GUI Agents. In practical scenarios, users frequently fail to articulate precise directives containing full task details at the onset, and their expressions are typically ambiguous. Consequently, agents are required to converge on the user's true intent via active clarification and interaction during execution. However, existing benchmarks predominantly operate under the idealized assumption that user-issued instructions are complete and unequivocal. This paradigm focuses exclusively on assessing single-turn execution while overlooking the alignment capability of the agent. To address this limitation, we introduce AmbiBench, the first benchmark incorporating a taxonomy of instruction clarity to shift evaluation from unidirectional instruction following to bidirectional intent alignment. Grounded in Cognitive Gap theory, we propose a taxonomy of four clarity levels: Detailed, Standard, Incomplete, and Ambiguous. We construct a rigorous dataset of 240 ecologically valid tasks across 25 applications, subject to strict review protocols. Furthermore, targeting evaluation in dynamic environments, we develop MUSE (Mobile User Satisfaction Evaluator), an automated framework utilizing an MLLM-as-a-judge multi-agent architecture. MUSE performs fine-grained auditing across three dimensions: Outcome Effectiveness, Execution Quality, and Interaction Quality. Empirical results on AmbiBench reveal the performance boundaries of SoTA agents across different clarity levels, quantify the gains derived from active interaction, and validate the strong correlation between MUSE and human judgment. This work redefines evaluation standards, laying the foundation for next-generation agents capable of truly understanding user intent. 

---
# Adapting Vision-Language Models for E-commerce Understanding at Scale 

**Authors**: Matteo Nulli, Vladimir Orshulevich, Tala Bazazo, Christian Herold, Michael Kozielski, Marcin Mazur, Szymon Tuzel, Cees G. M. Snoek, Seyyed Hadi Hashemi, Omar Javed, Yannick Versley, Shahram Khadivi  

**Link**: [PDF](https://arxiv.org/pdf/2602.11733)  

**Abstract**: E-commerce product understanding demands by nature, strong multimodal comprehension from text, images, and structured attributes. General-purpose Vision-Language Models (VLMs) enable generalizable multimodal latent modelling, yet there is no documented, well-known strategy for adapting them to the attribute-centric, multi-image, and noisy nature of e-commerce data, without sacrificing general performance. In this work, we show through a large-scale experimental study, how targeted adaptation of general VLMs can substantially improve e-commerce performance while preserving broad multimodal capabilities. Furthermore, we propose a novel extensive evaluation suite covering deep product understanding, strict instruction following, and dynamic attribute extraction. 

---
# LLM-Driven 3D Scene Generation of Agricultural Simulation Environments 

**Authors**: Arafa Yoncalik, Wouter Jansen, Nico Huebel, Mohammad Hasan Rahmani, Jan Steckel  

**Link**: [PDF](https://arxiv.org/pdf/2602.11706)  

**Abstract**: Procedural generation techniques in 3D rendering engines have revolutionized the creation of complex environments, reducing reliance on manual design. Recent approaches using Large Language Models (LLMs) for 3D scene generation show promise but often lack domain-specific reasoning, verification mechanisms, and modular design. These limitations lead to reduced control and poor scalability. This paper investigates the use of LLMs to generate agricultural synthetic simulation environments from natural language prompts, specifically to address the limitations of lacking domain-specific reasoning, verification mechanisms, and modular design. A modular multi-LLM pipeline was developed, integrating 3D asset retrieval, domain knowledge injection, and code generation for the Unreal rendering engine using its API. This results in a 3D environment with realistic planting layouts and environmental context, all based on the input prompt and the domain knowledge. To enhance accuracy and scalability, the system employs a hybrid strategy combining LLM optimization techniques such as few-shot prompting, Retrieval-Augmented Generation (RAG), finetuning, and validation. Unlike monolithic models, the modular architecture enables structured data handling, intermediate verification, and flexible expansion. The system was evaluated using structured prompts and semantic accuracy metrics. A user study assessed realism and familiarity against real-world images, while an expert comparison demonstrated significant time savings over manual scene design. The results confirm the effectiveness of multi-LLM pipelines in automating domain-specific 3D scene generation with improved reliability and precision. Future work will explore expanding the asset hierarchy, incorporating real-time generation, and adapting the pipeline to other simulation domains beyond agriculture. 

---
# Semantically Conditioned Diffusion Models for Cerebral DSA Synthesis 

**Authors**: Qiwen Xu, David Rügamer, Holger Wenz, Johann Fontana, Nora Meggyeshazi, Andreas Bender, Máté E. Maros  

**Link**: [PDF](https://arxiv.org/pdf/2602.11703)  

**Abstract**: Digital subtraction angiography (DSA) plays a central role in the diagnosis and treatment of cerebrovascular disease, yet its invasive nature and high acquisition cost severely limit large-scale data collection and public data sharing. Therefore, we developed a semantically conditioned latent diffusion model (LDM) that synthesizes arterial-phase cerebral DSA frames under explicit control of anatomical circulation (anterior vs.\ posterior) and canonical C-arm positions. We curated a large single-centre DSA dataset of 99,349 frames and trained a conditional LDM using text embeddings that encoded anatomy and acquisition geometry. To assess clinical realism, four medical experts, including two neuroradiologists, one neurosurgeon, and one internal medicine expert, systematically rated 400 synthetic DSA images using a 5-grade Likert scale for evaluating proximal large, medium, and small peripheral vessels. The generated images achieved image-wise overall Likert scores ranging from 3.1 to 3.3, with high inter-rater reliability (ICC(2,k) = 0.80--0.87). Distributional similarity to real DSA frames was supported by a low median Fréchet inception distance (FID) of 15.27. Our results indicate that semantically controlled LDMs can produce realistic synthetic DSAs suitable for downstream algorithm development, research, and training. 

---
# TabSieve: Explicit In-Table Evidence Selection for Tabular Prediction 

**Authors**: Yongyao Wang, Ziqi Miao, Lu Yang, Haonan Jia, Wenting Yan, Chen Qian, Lijun Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.11700)  

**Abstract**: Tabular prediction can benefit from in-table rows as few-shot evidence, yet existing tabular models typically perform instance-wise inference and LLM-based prompting is often brittle. Models do not consistently leverage relevant rows, and noisy context can degrade performance. To address this challenge, we propose TabSieve, a select-then-predict framework that makes evidence usage explicit and auditable. Given a table and a query row, TabSieve first selects a small set of informative rows as evidence and then predicts the missing target conditioned on the selected evidence. To enable this capability, we construct TabSieve-SFT-40K by synthesizing high-quality reasoning trajectories from 331 real tables using a strong teacher model with strict filtering. Furthermore, we introduce TAB-GRPO, a reinforcement learning recipe that jointly optimizes evidence selection and prediction correctness with separate rewards, and stabilizes mixed regression and classification training via dynamic task-advantage balancing. Experiments on a held-out benchmark of 75 classification and 52 regression tables show that TabSieve consistently improves performance across shot budgets, with average gains of 2.92% on classification and 4.45% on regression over the second-best baseline. Further analysis indicates that TabSieve concentrates more attention on the selected evidence, which improves robustness to noisy context. 

---
# OMEGA-Avatar: One-shot Modeling of 360° Gaussian Avatars 

**Authors**: Zehao Xia, Yiqun Wang, Zhengda Lu, Kai Liu, Jun Xiao, Peter Wonka  

**Link**: [PDF](https://arxiv.org/pdf/2602.11693)  

**Abstract**: Creating high-fidelity, animatable 3D avatars from a single image remains a formidable challenge. We identified three desirable attributes of avatar generation: 1) the method should be feed-forward, 2) model a 360° full-head, and 3) should be animation-ready. However, current work addresses only two of the three points simultaneously. To address these limitations, we propose OMEGA-Avatar, the first feed-forward framework that simultaneously generates a generalizable, 360°-complete, and animatable 3D Gaussian head from a single image. Starting from a feed-forward and animatable framework, we address the 360° full-head avatar generation problem with two novel components. First, to overcome poor hair modeling in full-head avatar generation, we introduce a semantic-aware mesh deformation module that integrates multi-view normals to optimize a FLAME head with hair while preserving its topology structure. Second, to enable effective feed-forward decoding of full-head features, we propose a multi-view feature splatting module that constructs a shared canonical UV representation from features across multiple views through differentiable bilinear splatting, hierarchical UV mapping, and visibility-aware fusion. This approach preserves both global structural coherence and local high-frequency details across all viewpoints, ensuring 360° consistency without per-instance optimization. Extensive experiments demonstrate that OMEGA-Avatar achieves state-of-the-art performance, significantly outperforming existing baselines in 360° full-head completeness while robustly preserving identity across different viewpoints. 

---
# ANML: Attribution-Native Machine Learning with Guaranteed Robustness 

**Authors**: Oliver Zahn, Matt Beton, Simran Chana  

**Link**: [PDF](https://arxiv.org/pdf/2602.11690)  

**Abstract**: Frontier AI systems increasingly train on specialized expert data, from clinical records to proprietary research to curated datasets, yet current training pipelines treat all samples identically. A Nobel laureate's contribution receives the same weight as an unverified submission. We introduce ANML (Attribution-Native Machine Learning), a framework that weights training samples by four quality factors: gradient-based consistency (q), verification status (v), contributor reputation (r), and temporal relevance (T). By combining what the model observes (gradient signals) with what the system knows about data provenance (external signals), ANML produces per-contributor quality weights that simultaneously improve model performance and enable downstream attribution. Across 5 datasets (178-32,561 samples), ANML achieves 33-72% error reduction over gradient-only baselines. Quality-weighted training is data-efficient: 20% high-quality data outperforms 100% uniformly weighted data by 47%. A Two-Stage Adaptive gating mechanism guarantees that ANML never underperforms the best available baseline, including under strategic joint attacks combining credential faking with gradient alignment. When per-sample detection fails against subtle corruption, contributor-level attribution provides 1.3-5.3x greater improvement than sample-level methods, with the advantage growing as corruption becomes harder to detect. 

---
# DRACO: a Cross-Domain Benchmark for Deep Research Accuracy, Completeness, and Objectivity 

**Authors**: Joey Zhong, Hao Zhang, Clare Southern, Jeremy Yang, Thomas Wang, Kate Jung, Shu Zhang, Denis Yarats, Johnny Ho, Jerry Ma  

**Link**: [PDF](https://arxiv.org/pdf/2602.11685)  

**Abstract**: We present DRACO (Deep Research Accuracy, Completeness, and Objectivity), a benchmark of complex deep research tasks. These tasks, which span 10 domains and draw on information sources from 40 countries, originate from anonymized real-world usage patterns within a large-scale deep research system. Tasks are sampled from a de-identified dataset of Perplexity Deep Research requests, then filtered and augmented to ensure that the tasks are anonymized, open-ended and complex, objectively evaluable, and representative of the broad scope of real-world deep research use cases. Outputs are graded against task-specific rubrics along four dimensions: factual accuracy (accuracy), breadth and depth of analysis (including completeness), presentation quality (including objectivity), and citation quality. DRACO is publicly available at this https URL. 

---
# PatientHub: A Unified Framework for Patient Simulation 

**Authors**: Sahand Sabour, TszYam NG, Minlie Huang  

**Link**: [PDF](https://arxiv.org/pdf/2602.11684)  

**Abstract**: As Large Language Models increasingly power role-playing applications, simulating patients has become a valuable tool for training counselors and scaling therapeutic assessment. However, prior work is fragmented: existing approaches rely on incompatible, non-standardized data formats, prompts, and evaluation metrics, hindering reproducibility and fair comparison. In this paper, we introduce PatientHub, a unified and modular framework that standardizes the definition, composition, and deployment of simulated patients. To demonstrate PatientHub's utility, we implement several representative patient simulation methods as case studies, showcasing how our framework supports standardized cross-method evaluation and the seamless integration of custom evaluation metrics. We further demonstrate PatientHub's extensibility by prototyping two new simulator variants, highlighting how PatientHub accelerates method development by eliminating infrastructure overhead. By consolidating existing work into a single reproducible pipeline, PatientHub lowers the barrier to developing new simulation methods and facilitates cross-method and cross-model benchmarking. Our framework provides a practical foundation for future datasets, methods, and benchmarks in patient-centered dialogue, and the code is publicly available via this https URL. 

---
# Provable Offline Reinforcement Learning for Structured Cyclic MDPs 

**Authors**: Kyungbok Lee, Angelica Cristello Sarteau, Michael R. Kosorok  

**Link**: [PDF](https://arxiv.org/pdf/2602.11679)  

**Abstract**: We introduce a novel cyclic Markov decision process (MDP) framework for multi-step decision problems with heterogeneous stage-specific dynamics, transitions, and discount factors across the cycle. In this setting, offline learning is challenging: optimizing a policy at any stage shifts the state distributions of subsequent stages, propagating mismatch across the cycle. To address this, we propose a modular structural framework that decomposes the cyclic process into stage-wise sub-problems. While generally applicable, we instantiate this principle as CycleFQI, an extension of fitted Q-iteration enabling theoretical analysis and interpretation. It uses a vector of stage-specific Q-functions, tailored to each stage, to capture within-stage sequences and transitions between stages. This modular design enables partial control, allowing some stages to be optimized while others follow predefined policies. We establish finite-sample suboptimality error bounds and derive global convergence rates under Besov regularity, demonstrating that CycleFQI mitigates the curse of dimensionality compared to monolithic baselines. Additionally, we propose a sieve-based method for asymptotic inference of optimal policy values under a margin condition. Experiments on simulated and real-world Type 1 Diabetes data sets demonstrate CycleFQI's effectiveness. 

---
# SToRM: Supervised Token Reduction for Multi-modal LLMs toward efficient end-to-end autonomous driving 

**Authors**: Seo Hyun Kim, Jin Bok Park, Do Yeon Koo, Ho Gun Park, Il Yong Chun  

**Link**: [PDF](https://arxiv.org/pdf/2602.11656)  

**Abstract**: In autonomous driving, end-to-end (E2E) driving systems that predict control commands directly from sensor data have achieved significant advancements. For safe driving in unexpected scenarios, these systems may additionally rely on human interventions such as natural language instructions. Using a multi-modal large language model (MLLM) facilitates human-vehicle interaction and can improve performance in such scenarios. However, this approach requires substantial computational resources due to its reliance on an LLM and numerous visual tokens from sensor inputs, which are limited in autonomous vehicles. Many MLLM studies have explored reducing visual tokens, but often suffer end-task performance degradation compared to using all tokens.
To enable efficient E2E driving while maintaining performance comparable to using all tokens, this paper proposes the first Supervised Token Reduction framework for multi-modal LLMs (SToRM). The proposed framework consists of three key elements. First, a lightweight importance predictor with short-term sliding windows estimates token importance scores. Second, a supervised training approach uses an auxiliary path to obtain pseudo-supervision signals from an all-token LLM pass. Third, an anchor-context merging module partitions tokens into anchors and context tokens, and merges context tokens into relevant anchors to reduce redundancy while minimizing information loss. Experiments on the LangAuto benchmark show that SToRM outperforms state-of-the-art E2E driving MLLMs under the same reduced-token budget, maintaining all-token performance while reducing computational cost by up to 30x. 

---
# LoRA-based Parameter-Efficient LLMs for Continuous Learning in Edge-based Malware Detection 

**Authors**: Christian Rondanini, Barbara Carminati, Elena Ferrari, Niccolò Lardo, Ashish Kundu  

**Link**: [PDF](https://arxiv.org/pdf/2602.11655)  

**Abstract**: The proliferation of edge devices has created an urgent need for security solutions capable of detecting malware in real time while operating under strict computational and memory constraints. Recently, Large Language Models (LLMs) have demonstrated remarkable capabilities in recognizing complex patterns, yet their deployment on edge devices remains impractical due to their resource demands. However, in edge malware detection, static or centrally retrained models degrade under evolving threats and heterogeneous traffic; locally trained models become siloed and fail to transfer across domains. To overcome these limitations, in this paper, we present a continuous learning architecture for edge-based malware detection that combines local adaptation on each device with global knowledge sharing through parameter-efficient LoRA adapters. Lightweight transformer models (DistilBERT, DistilGPT-2, TinyT5) run on edge nodes and are incrementally fine-tuned on device-specific traffic; only the resulting LoRA modules are aggregated by a lightweight coordinator and redistributed, enabling cross-device generalization without exchanging raw data. We evaluate on two public IoT security datasets, Edge-IIoTset and TON-IoT, under multi-round learning to simulate evolving threats. Compared to isolated fine-tuning, the LoRA-based exchange yields up to 20-25% accuracy gains when models encounter previously unseen attacks from another domain, while maintaining stable loss and F1 across rounds. LoRA adds less than 1% to model size (~0.6-1.8 MB), making updates practical for constrained edge hardware. 

---
# DMind-3: A Sovereign Edge--Local--Cloud AI System with Controlled Deliberation and Correction-Based Tuning for Safe, Low-Latency Transaction Execution 

**Authors**: Enhao Huang, Frank Li, Tony Lin, Lowes Yang  

**Link**: [PDF](https://arxiv.org/pdf/2602.11651)  

**Abstract**: This paper introduces DMind-3, a sovereign Edge-Local-Cloud intelligence stack designed to secure irreversible financial execution in Web3 environments against adversarial risks and strict latency constraints. While existing cloud-centric assistants compromise privacy and fail under network congestion, and purely local solutions lack global ecosystem context, DMind-3 resolves these tensions by decomposing capability into three cooperating layers: a deterministic signing-time intent firewall at the edge, a private high-fidelity reasoning engine on user hardware, and a policy-governed global context synthesizer in the cloud. We propose policy-driven selective offloading to route computation based on privacy sensitivity and uncertainty, supported by two novel training objectives: Hierarchical Predictive Synthesis (HPS) for fusing time-varying macro signals, and Contrastive Chain-of-Correction Supervised Fine-Tuning (C$^3$-SFT) to enhance local verification reliability. Extensive evaluations demonstrate that DMind-3 achieves a 93.7% multi-turn success rate in protocol-constrained tasks and superior domain reasoning compared to general-purpose baselines, providing a scalable framework where safety is bound to the edge execution primitive while maintaining sovereignty over sensitive user intent. 

---
# Brain Tumor Classifiers Under Attack: Robustness of ResNet Variants Against Transferable FGSM and PGD Attacks 

**Authors**: Ryan Deem, Garrett Goodman, Waqas Majeed, Md Abdullah Al Hafiz Khan, Michail S. Alexiou  

**Link**: [PDF](https://arxiv.org/pdf/2602.11646)  

**Abstract**: Adversarial robustness in deep learning models for brain tumor classification remains an underexplored yet critical challenge, particularly for clinical deployment scenarios involving MRI data. In this work, we investigate the susceptibility and resilience of several ResNet-based architectures, referred to as BrainNet, BrainNeXt and DilationNet, against gradient-based adversarial attacks, namely FGSM and PGD. These models, based on ResNet, ResNeXt, and dilated ResNet variants respectively, are evaluated across three preprocessing configurations (i) full-sized augmented, (ii) shrunk augmented and (iii) shrunk non-augmented MRI datasets. Our experiments reveal that BrainNeXt models exhibit the highest robustness to black-box attacks, likely due to their increased cardinality, though they produce weaker transferable adversarial samples. In contrast, BrainNet and Dilation models are more vulnerable to attacks from each other, especially under PGD with higher iteration steps and $\alpha$ values. Notably, shrunk and non-augmented data significantly reduce model resilience, even when the untampered test accuracy remains high, highlighting a key trade-off between input resolution and adversarial vulnerability. These results underscore the importance of jointly evaluating classification performance and adversarial robustness for reliable real-world deployment in brain MRI analysis. 

---
# ViTaS: Visual Tactile Soft Fusion Contrastive Learning for Visuomotor Learning 

**Authors**: Yufeng Tian, Shuiqi Cheng, Tianming Wei, Tianxing Zhou, Yuanhang Zhang, Zixian Liu, Qianwei Han, Zhecheng Yuan, Huazhe Xu  

**Link**: [PDF](https://arxiv.org/pdf/2602.11643)  

**Abstract**: Tactile information plays a crucial role in human manipulation tasks and has recently garnered increasing attention in robotic manipulation. However, existing approaches mostly focus on the alignment of visual and tactile features and the integration mechanism tends to be direct concatenation. Consequently, they struggle to effectively cope with occluded scenarios due to neglecting the inherent complementary nature of both modalities and the alignment may not be exploited enough, limiting the potential of their real-world deployment. In this paper, we present ViTaS, a simple yet effective framework that incorporates both visual and tactile information to guide the behavior of an agent. We introduce Soft Fusion Contrastive Learning, an advanced version of conventional contrastive learning method and a CVAE module to utilize the alignment and complementarity within visuo-tactile representations. We demonstrate the effectiveness of our method in 12 simulated and 3 real-world environments, and our experiments show that ViTaS significantly outperforms existing baselines. Project page: this https URL. 

---
# Variation-aware Flexible 3D Gaussian Editing 

**Authors**: Hao Qin, Yukai Sun, Meng Wang, Ming Kong, Mengxu Lu, Qiang Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2602.11638)  

**Abstract**: Indirect editing methods for 3D Gaussian Splatting (3DGS) have recently witnessed significant advancements. These approaches operate by first applying edits in the rendered 2D space and subsequently projecting the modifications back into 3D. However, this paradigm inevitably introduces cross-view inconsistencies and constrains both the flexibility and efficiency of the editing process. To address these challenges, we present VF-Editor, which enables native editing of Gaussian primitives by predicting attribute variations in a feedforward manner. To accurately and efficiently estimate these variations, we design a novel variation predictor distilled from 2D editing knowledge. The predictor encodes the input to generate a variation field and employs two learnable, parallel decoding functions to iteratively infer attribute changes for each 3D Gaussian. Thanks to its unified design, VF-Editor can seamlessly distill editing knowledge from diverse 2D editors and strategies into a single predictor, allowing for flexible and effective knowledge transfer into the 3D domain. Extensive experiments on both public and private datasets reveal the inherent limitations of indirect editing pipelines and validate the effectiveness and flexibility of our approach. 

---
# ScalSelect: Scalable Training-Free Multimodal Data Selection for Efficient Visual Instruction Tuning 

**Authors**: Changti Wu, Jiahuai Mao, Yuzhuo Miao, Shijie Lian, Bin Yu, Xiaopeng Lin, Cong Huang, Lei Zhang, Kai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2602.11636)  

**Abstract**: Large-scale Visual Instruction Tuning (VIT) has become a key paradigm for advancing the performance of vision-language models (VLMs) across various multimodal tasks. However, training on the large-scale datasets is computationally expensive and inefficient due to redundancy in the data, which motivates the need for multimodal data selection to improve training efficiency. Existing data selection methods for VIT either require costly training or gradient computation. Training-free alternatives often depend on proxy models or datasets, instruction-agnostic representations, and pairwise similarity with quadratic complexity, limiting scalability and representation fidelity. In this work, we propose ScalSelect, a scalable training-free multimodal data selection method with linear-time complexity with respect to the number of samples, eliminating the need for external models or auxiliary datasets. ScalSelect first constructs sample representations by extracting visual features most attended by instruction tokens in the target VLM, capturing instruction-relevant information. It then identifies samples whose representations best approximate the dominant subspace of the full dataset representations, enabling scalable importance scoring without pairwise comparisons. Extensive experiments across multiple VLMs, datasets, and selection budgets demonstrate that ScalSelect achieves over 97.5% of the performance of training on the full dataset using only 16% of the data, and even outperforms full-data training in some settings. The code is available at \href{this https URL}{ScalSelect}. 

---
# ArGEnT: Arbitrary Geometry-encoded Transformer for Operator Learning 

**Authors**: Wenqian Chen, Yucheng Fu, Michael Penwarden, Pratanu Roy, Panos Stinis  

**Link**: [PDF](https://arxiv.org/pdf/2602.11626)  

**Abstract**: Learning solution operators for systems with complex, varying geometries and parametric physical settings is a central challenge in scientific machine learning. In many-query regimes such as design optimization, control and inverse problems, surrogate modeling must generalize across geometries while allowing flexible evaluation at arbitrary spatial locations. In this work, we propose Arbitrary Geometry-encoded Transformer (ArGEnT), a geometry-aware attention-based architecture for operator learning on arbitrary domains. ArGEnT employs Transformer attention mechanisms to encode geometric information directly from point-cloud representations with three variants-self-attention, cross-attention, and hybrid-attention-that incorporates different strategies for incorporating geometric features. By integrating ArGEnT into DeepONet as the trunk network, we develop a surrogate modeling framework capable of learning operator mappings that depend on both geometric and non-geometric inputs without the need to explicitly parametrize geometry as a branch network input. Evaluation on benchmark problems spanning fluid dynamics, solid mechanics and electrochemical systems, we demonstrate significantly improved prediction accuracy and generalization performance compared with the standard DeepONet and other existing geometry-aware saurrogates. In particular, the cross-attention transformer variant enables accurate geometry-conditioned predictions with reduced reliance on signed distance functions. By combining flexible geometry encoding with operator-learning capabilities, ArGEnT provides a scalable surrogate modeling framework for optimization, uncertainty quantification, and data-driven modeling of complex physical systems. 

---
# PLOT-CT: Pre-log Voronoi Decomposition Assisted Generation for Low-dose CT Reconstruction 

**Authors**: Bin Huang, Xun Yu, Yikun Zhang, Yi Zhang, Yang Chen, Qiegen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2602.11625)  

**Abstract**: Low-dose computed tomography (LDCT) reconstruction is fundamentally challenged by severe noise and compromised data fidelity under reduced radiation exposure. Most existing methods operate either in the image or post-log projection domain, which fails to fully exploit the rich structural information in pre-log measurements while being highly susceptible to noise. The requisite logarithmic transformation critically amplifies noise within these data, imposing exceptional demands on reconstruction precision. To overcome these challenges, we propose PLOT-CT, a novel framework for Pre-Log vOronoi decomposiTion-assisted CT generation. Our method begins by applying Voronoi decomposition to pre-log sinograms, disentangling the data into distinct underlying components, which are embedded in separate latent spaces. This explicit decomposition significantly enhances the model's capacity to learn discriminative features, directly improving reconstruction accuracy by mitigating noise and preserving information inherent in the pre-log domain. Extensive experiments demonstrate that PLOT-CT achieves state-of-the-art performance, attaining a 2.36dB PSNR improvement over traditional methods at the 1e4 incident photon level in the pre-log domain. 

---
# ABot-N0: Technical Report on the VLA Foundation Model for Versatile Embodied Navigation 

**Authors**: Zedong Chu, Shichao Xie, Xiaolong Wu, Yanfen Shen, Minghua Luo, Zhengbo Wang, Fei Liu, Xiaoxu Leng, Junjun Hu, Mingyang Yin, Jia Lu, Yingnan Guo, Kai Yang, Jiawei Han, Xu Chen, Yanqing Zhu, Yuxiang Zhao, Xin Liu, Yirong Yang, Ye He, Jiahang Wang, Yang Cai, Tianlin Zhang, Li Gao, Liu Liu, Mingchao Sun, Fan Jiang, Chiyu Wang, Zhicheng Liu, Hongyu Pan, Honglin Han, Zhining Gu, Kuan Yang, Jianfang Zhang, Di Jing, Zihao Guan, Wei Guo, Guoqing Liu, Di Yang, Xiangpo Yang, Menglin Yang, Hongguang Xing, Weiguo Li, Mu Xu  

**Link**: [PDF](https://arxiv.org/pdf/2602.11598)  

**Abstract**: Embodied navigation has long been fragmented by task-specific architectures. We introduce ABot-N0, a unified Vision-Language-Action (VLA) foundation model that achieves a ``Grand Unification'' across 5 core tasks: Point-Goal, Object-Goal, Instruction-Following, POI-Goal, and Person-Following. ABot-N0 utilizes a hierarchical ``Brain-Action'' architecture, pairing an LLM-based Cognitive Brain for semantic reasoning with a Flow Matching-based Action Expert for precise, continuous trajectory generation.
To support large-scale learning, we developed the ABot-N0 Data Engine, curating 16.9M expert trajectories and 5.0M reasoning samples across 7,802 high-fidelity 3D scenes (10.7 $\text{km}^2$). ABot-N0 achieves new SOTA performance across 7 benchmarks, significantly outperforming specialized models. Furthermore, our Agentic Navigation System integrates a planner with hierarchical topological memory, enabling robust, long-horizon missions in dynamic real-world environments. 

---
# Gradient Compression May Hurt Generalization: A Remedy by Synthetic Data Guided Sharpness Aware Minimization 

**Authors**: Yujie Gu, Richeng Jin, Zhaoyang Zhang, Huaiyu Dai  

**Link**: [PDF](https://arxiv.org/pdf/2602.11584)  

**Abstract**: It is commonly believed that gradient compression in federated learning (FL) enjoys significant improvement in communication efficiency with negligible performance degradation. In this paper, we find that gradient compression induces sharper loss landscapes in federated learning, particularly under non-IID data distributions, which suggests hindered generalization capability. The recently emerging Sharpness Aware Minimization (SAM) effectively searches for a flat minima by incorporating a gradient ascent step (i.e., perturbing the model with gradients) before the celebrated stochastic gradient descent. Nonetheless, the direct application of SAM in FL suffers from inaccurate estimation of the global perturbation due to data heterogeneity. Existing approaches propose to utilize the model update from the previous communication round as a rough estimate. However, its effectiveness is hindered when model update compression is incorporated. In this paper, we propose FedSynSAM, which leverages the global model trajectory to construct synthetic data and facilitates an accurate estimation of the global perturbation. The convergence of the proposed algorithm is established, and extensive experiments are conducted to validate its effectiveness. 

---
# Analytical Search 

**Authors**: Yiteng Tu, Shuo Miao, Weihang Su, Yiqun Liu, Qingyao Ai  

**Link**: [PDF](https://arxiv.org/pdf/2602.11581)  

**Abstract**: Analytical information needs, such as trend analysis and causal impact assessment, are prevalent across various domains including law, finance, science, and much more. However, existing information retrieval paradigms, whether based on relevance-oriented document ranking or retrieval-augmented generation (RAG) with large language models (LLMs), often struggle to meet the end-to-end requirements of such tasks at the corpus scale. They either emphasize information finding rather than end-to-end problem solving, or simply treat everything as naive question answering, offering limited control over reasoning, evidence usage, and verifiability. As a result, they struggle to support analytical queries that have diverse utility concepts and high accountability requirements.
In this paper, we propose analytical search as a distinct and emerging search paradigm designed to fulfill these analytical information needs. Analytical search reframes search as an evidence-governed, process-oriented analytical workflow that explicitly models analytical intent, retrieves evidence for fusion, and produces verifiable conclusions through structured, multi-step inference. We position analytical search in contrast to existing paradigms, and present a unified system framework that integrates query understanding, recall-oriented retrieval, reasoning-aware fusion, and adaptive verification. We also discuss potential research directions for the construction of analytical search engines. In this way, we highlight the conceptual significance and practical importance of analytical search and call on efforts toward the next generation of search engines that support analytical information needs. 

---
# ReaDy-Go: Real-to-Sim Dynamic 3D Gaussian Splatting Simulation for Environment-Specific Visual Navigation with Moving Obstacles 

**Authors**: Seungyeon Yoo, Youngseok Jang, Dabin Kim, Youngsoo Han, Seungwoo Jung, H. Jin Kim  

**Link**: [PDF](https://arxiv.org/pdf/2602.11575)  

**Abstract**: Visual navigation models often struggle in real-world dynamic environments due to limited robustness to the sim-to-real gap and the difficulty of training policies tailored to target deployment environments (e.g., households, restaurants, and factories). Although real-to-sim navigation simulation using 3D Gaussian Splatting (GS) can mitigate this gap, prior works have assumed only static scenes or unrealistic dynamic obstacles, despite the importance of safe navigation in dynamic environments. To address these issues, we propose ReaDy-Go, a novel real-to-sim simulation pipeline that synthesizes photorealistic dynamic scenarios for target environments. ReaDy-Go generates photorealistic navigation datasets for dynamic environments by combining a reconstructed static GS scene with dynamic human GS obstacles, and trains policies robust to both the sim-to-real gap and moving obstacles. The pipeline consists of three components: (1) a dynamic GS simulator that integrates scene GS with a human animation module, enabling the insertion of animatable human GS avatars and the synthesis of plausible human motions from 2D trajectories, (2) navigation dataset generation for dynamic environments that leverages the simulator, a robot expert planner designed for dynamic GS representations, and a human planner, and (3) policy learning using the generated datasets. ReaDy-Go outperforms baselines across target environments in both simulation and real-world experiments, demonstrating improved navigation performance even after sim-to-real transfer and in the presence of moving obstacles. Moreover, zero-shot sim-to-real deployment in an unseen environment indicates its generalization potential. Project page: this https URL. 

---
# Perception-based Image Denoising via Generative Compression 

**Authors**: Nam Nguyen, Thinh Nguyen, Bella Bose  

**Link**: [PDF](https://arxiv.org/pdf/2602.11553)  

**Abstract**: Image denoising aims to remove noise while preserving structural details and perceptual realism, yet distortion-driven methods often produce over-smoothed reconstructions, especially under strong noise and distribution shift. This paper proposes a generative compression framework for perception-based denoising, where restoration is achieved by reconstructing from entropy-coded latent representations that enforce low-complexity structure, while generative decoders recover realistic textures via perceptual measures such as learned perceptual image patch similarity (LPIPS) loss and Wasserstein distance. Two complementary instantiations are introduced: (i) a conditional Wasserstein GAN (WGAN)-based compression denoiser that explicitly controls the rate-distortion-perception (RDP) trade-off, and (ii) a conditional diffusion-based reconstruction strategy that performs iterative denoising guided by compressed latents. We further establish non-asymptotic guarantees for the compression-based maximum-likelihood denoiser under additive Gaussian noise, including bounds on reconstruction error and decoding error probability. Experiments on synthetic and real-noise benchmarks demonstrate consistent perceptual improvements while maintaining competitive distortion performance. 

---
# TS-Memory: Plug-and-Play Memory for Time Series Foundation Models 

**Authors**: Sisuo Lyu, Siru Zhong, Tiegang Chen, Weilin Ruan, Qingxiang Liu, Taiqiang Lv, Qingsong Wen, Raymond Chi-Wing Wong, Yuxuan Liang  

**Link**: [PDF](https://arxiv.org/pdf/2602.11550)  

**Abstract**: Time Series Foundation Models (TSFMs) achieve strong zero-shot forecasting through large-scale pre-training, but adapting them to downstream domains under distribution shift remains challenging. Existing solutions face a trade-off: Parametric Adaptation can cause catastrophic forgetting and requires costly multi-domain maintenance, while Non-Parametric Retrieval improves forecasts but incurs high inference latency due to datastore search. We propose Parametric Memory Distillation and implement it as TS-Memory, a lightweight memory adapter that augments frozen TSFMs. TS-Memory is trained in two stages. First, we construct an offline, leakage-safe kNN teacher that synthesizes confidence-aware quantile targets from retrieved futures. Second, we distill this retrieval-induced distributional correction into a lightweight memory adapter via confidence-gated supervision. During inference, TS-Memory fuses memory and backbone predictions with constant-time overhead, enabling retrieval-free deployment. Experiments across diverse TSFMs and benchmarks demonstrate consistent improvements in both point and probabilistic forecasting over representative adaptation methods, with efficiency comparable to the frozen backbone. 

---
# Native Reasoning Models: Training Language Models to Reason on Unverifiable Data 

**Authors**: Yuanfu Wang, Zhixuan Liu, Xiangtian Li, Chaochao Lu, Chao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2602.11549)  

**Abstract**: The prevailing paradigm for training large reasoning models--combining Supervised Fine-Tuning (SFT) with Reinforcement Learning with Verifiable Rewards (RLVR)--is fundamentally constrained by its reliance on high-quality, human-annotated reasoning data and external verifiers. This dependency incurs significant data-collection costs, risks embedding human cognitive biases, and confines the reinforcement learning stage to objectively assessable domains like mathematics and coding, leaving a wide range of unverifiable tasks beyond its scope. To overcome these limitations, we introduce NRT (Native Reasoning Training), a novel framework that cultivates complex reasoning by having the model generate its own reasoning traces using only standard question-answer pairs, thereby obviating the need for expert-written demonstrations. NRT reframes the training problem by treating the reasoning process as a latent variable. It employs a unified training objective that models reasoning as an optimization problem, intrinsically rewarding paths that increase the model's likelihood of producing the ground-truth answer. This unified perspective allows us to analyze intrinsic failure modes of prior methods, such as policy collapse, and systematically design more robust reward aggregation functions, creating a self-reinforcing feedback loop where the model learns to think in ways that resolve its own uncertainty. Empirical evaluation on Llama and Mistral model families demonstrates that NRT achieves state-of-the-art performance among verifier-free methods, significantly outperforming standard SFT baselines and prior verifier-free RL methods. Our approach yields particularly strong performance gains in complex reasoning domains and exhibits high robustness to policy collapse, offering a general, scalable path toward building more powerful and broadly applicable reasoning systems. 

---
# Krause Synchronization Transformers 

**Authors**: Jingkun Liu, Yisong Yue, Max Welling, Yue Song  

**Link**: [PDF](https://arxiv.org/pdf/2602.11534)  

**Abstract**: Self-attention in Transformers relies on globally normalized softmax weights, causing all tokens to compete for influence at every layer. When composed across depth, this interaction pattern induces strong synchronization dynamics that favor convergence toward a dominant mode, a behavior associated with representation collapse and attention sink phenomena. We introduce Krause Attention, a principled attention mechanism inspired by bounded-confidence consensus dynamics. Krause Attention replaces similarity-based global aggregation with distance-based, localized, and selectively sparse interactions, promoting structured local synchronization instead of global mixing. We relate this behavior to recent theory modeling Transformer dynamics as interacting particle systems, and show how bounded-confidence interactions naturally moderate attention concentration and alleviate attention sinks. Restricting interactions to local neighborhoods also reduces runtime complexity from quadratic to linear in sequence length. Experiments across vision (ViT on CIFAR/ImageNet), autoregressive generation (MNIST/CIFAR-10), and large language models (Llama/Qwen) demonstrate consistent gains with substantially reduced computation, highlighting bounded-confidence dynamics as a scalable and effective inductive bias for attention. 

---
# AltTS: A Dual-Path Framework with Alternating Optimization for Multivariate Time Series Forecasting 

**Authors**: Zhihang Yuan, Zhiyuan Liu, Mahesh K. Marina  

**Link**: [PDF](https://arxiv.org/pdf/2602.11533)  

**Abstract**: Multivariate time series forecasting involves two qualitatively distinct factors: (i) stable within-series autoregressive (AR) dynamics, and (ii) intermittent cross-dimension interactions that can become spurious over long horizons. We argue that fitting a single model to capture both effects creates an optimization conflict: the high-variance updates needed for cross-dimension modeling can corrupt the gradients that support autoregression, resulting in brittle training and degraded long-horizon accuracy. To address this, we propose ALTTS, a dual-path framework that explicitly decouples autoregression and cross-relation (CR) modeling. In ALTTS, the AR path is instantiated with a linear predictor, while the CR path uses a Transformer equipped with Cross-Relation Self-Attention (CRSA); the two branches are coordinated via alternating optimization to isolate gradient noise and reduce cross-block interference. Extensive experiments on multiple benchmarks show that ALTTS consistently outperforms prior methods, with the most pronounced improvements on long-horizon forecasting. Overall, our results suggest that carefully designed optimization strategies, rather than ever more complex architectures, can be a key driver of progress in multivariate time series forecasting. 

---
# Stop Tracking Me! Proactive Defense Against Attribute Inference Attack in LLMs 

**Authors**: Dong Yan, Jian Liang, Ran He, Tieniu Tan  

**Link**: [PDF](https://arxiv.org/pdf/2602.11528)  

**Abstract**: Recent studies have shown that large language models (LLMs) can infer private user attributes (e.g., age, location, gender) from user-generated text shared online, enabling rapid and large-scale privacy breaches. Existing anonymization-based defenses are coarse-grained, lacking word-level precision in anonymizing privacy-leaking elements. Moreover, they are inherently limited as altering user text to hide sensitive cues still allows attribute inference to occur through models' reasoning capabilities. To address these limitations, we propose a unified defense framework that combines fine-grained anonymization (TRACE) with inference-preventing optimization (RPS). TRACE leverages attention mechanisms and inference chain generation to identify and anonymize privacy-leaking textual elements, while RPS employs a lightweight two-stage optimization strategy to induce model rejection behaviors, thereby preventing attribute inference. Evaluations across diverse LLMs show that TRACE-RPS reduces attribute inference accuracy from around 50\% to below 5\% on open-source models. In addition, our approach offers strong cross-model generalization, prompt-variation robustness, and utility-privacy tradeoffs. Our code is available at this https URL. 

---
# Adaptive Milestone Reward for GUI Agents 

**Authors**: Congmin Zheng, Xiaoyun Mo, Xinbei Ma, Qiqiang Lin, Yin Zhao, Jiachen Zhu, Xingyu Lou, Jun Wang, Zhaoxiang Wang, Weiwen Liu, Zhuosheng Zhang, Yong Yu, Weinan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.11524)  

**Abstract**: Reinforcement Learning (RL) has emerged as a mainstream paradigm for training Mobile GUI Agents, yet it struggles with the temporal credit assignment problem inherent in long-horizon tasks. A primary challenge lies in the trade-off between reward fidelity and density: outcome reward offers high fidelity but suffers from signal sparsity, while process reward provides dense supervision but remains prone to bias and reward hacking. To resolve this conflict, we propose the Adaptive Milestone Reward (ADMIRE) mechanism. ADMIRE constructs a verifiable, adaptive reward system by anchoring trajectory to milestones, which are dynamically distilled from successful explorations. Crucially, ADMIRE integrates an asymmetric credit assignment strategy that denoises successful trajectories and scaffolds failed trajectories. Extensive experiments demonstrate that ADMIRE consistently yields over 10% absolute improvement in success rate across different base models on AndroidWorld. Moreover, the method exhibits robust generalizability, achieving strong performance across diverse RL algorithms and heterogeneous environments such as web navigation and embodied tasks. 

---
# Locally Interpretable Individualized Treatment Rules for Black-Box Decision Models 

**Authors**: Yasin Khadem Charvadeh, Katherine S. Panageas, Yuan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2602.11520)  

**Abstract**: Individualized treatment rules (ITRs) aim to optimize healthcare by tailoring treatment decisions to patient-specific characteristics. Existing methods typically rely on either interpretable but inflexible models or highly flexible black-box approaches that sacrifice interpretability; moreover, most impose a single global decision rule across patients. We introduce the Locally Interpretable Individualized Treatment Rule (LI-ITR) method, which combines flexible machine learning models to accurately learn complex treatment outcomes with locally interpretable approximations to construct subject-specific treatment rules. LI-ITR employs variational autoencoders to generate realistic local synthetic samples and learns individualized decision rules through a mixture of interpretable experts. Simulation studies show that LI-ITR accurately recovers true subject-specific local coefficients and optimal treatment strategies. An application to precision side-effect management in breast cancer illustrates the necessity of flexible predictive modeling and highlights the practical utility of LI-ITR in estimating optimal treatment rules while providing transparent, clinically interpretable explanations. 

---
# How Smart Is Your GUI Agent? A Framework for the Future of Software Interaction 

**Authors**: Sidong Feng, Chunyang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2602.11514)  

**Abstract**: GUI agents are rapidly becoming a new interaction to software, allowing people to navigate web, desktop and mobile rather than execute them click by click. Yet ``agent'' is described with radically different degrees of autonomy, obscuring capability, responsibility and risk. We call for conceptual clarity through GUI Agent Autonomy Levels (GAL), a six-level framework that makes autonomy explicit and helps benchmark progress toward trustworthy software interaction. 

---
# Differentially Private and Communication Efficient Large Language Model Split Inference via Stochastic Quantization and Soft Prompt 

**Authors**: Yujie Gu, Richeng Jin, Xiaoyu Ji, Yier Jin, Wenyuan Xu  

**Link**: [PDF](https://arxiv.org/pdf/2602.11513)  

**Abstract**: Large Language Models (LLMs) have achieved remarkable performance and received significant research interest. The enormous computational demands, however, hinder the local deployment on devices with limited resources. The current prevalent LLM inference paradigms require users to send queries to the service providers for processing, which raises critical privacy concerns. Existing approaches propose to allow the users to obfuscate the token embeddings before transmission and utilize local models for denoising. Nonetheless, transmitting the token embeddings and deploying local models may result in excessive communication and computation overhead, preventing practical implementation. In this work, we propose \textbf{DEL}, a framework for \textbf{D}ifferentially private and communication \textbf{E}fficient \textbf{L}LM split inference. More specifically, an embedding projection module and a differentially private stochastic quantization mechanism are proposed to reduce the communication overhead in a privacy-preserving manner. To eliminate the need for local models, we adapt soft prompt at the server side to compensate for the utility degradation caused by privacy. To the best of our knowledge, this is the first work that utilizes soft prompt to improve the trade-off between privacy and utility in LLM inference, and extensive experiments on text generation and natural language understanding benchmarks demonstrate the effectiveness of the proposed method. 

---
# Multimodal Fact-Level Attribution for Verifiable Reasoning 

**Authors**: David Wan, Han Wang, Ziyang Wang, Elias Stengel-Eskin, Hyunji Lee, Mohit Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2602.11509)  

**Abstract**: Multimodal large language models (MLLMs) are increasingly used for real-world tasks involving multi-step reasoning and long-form generation, where reliability requires grounding model outputs in heterogeneous input sources and verifying individual factual claims. However, existing multimodal grounding benchmarks and evaluation methods focus on simplified, observation-based scenarios or limited modalities and fail to assess attribution in complex multimodal reasoning. We introduce MuRGAt (Multimodal Reasoning with Grounded Attribution), a benchmark for evaluating fact-level multimodal attribution in settings that require reasoning beyond direct observation. Given inputs spanning video, audio, and other modalities, MuRGAt requires models to generate answers with explicit reasoning and precise citations, where each citation specifies both modality and temporal segments. To enable reliable assessment, we introduce an automatic evaluation framework that strongly correlates with human judgments. Benchmarking with human and automated scores reveals that even strong MLLMs frequently hallucinate citations despite correct reasoning. Moreover, we observe a key trade-off: increasing reasoning depth or enforcing structured grounding often degrades accuracy, highlighting a significant gap between internal reasoning and verifiable attribution. 

---
# RooflineBench: A Benchmarking Framework for On-Device LLMs via Roofline Analysis 

**Authors**: Zhen Bi, Xueshu Chen, Luoyang Sun, Yuhang Yao, Qing Shen, Jungang Lou, Cheng Deng  

**Link**: [PDF](https://arxiv.org/pdf/2602.11506)  

**Abstract**: The transition toward localized intelligence through Small Language Models (SLMs) has intensified the need for rigorous performance characterization on resource-constrained edge hardware. However, objectively measuring the theoretical performance ceilings of diverse architectures across heterogeneous platforms remains a formidable challenge. In this work, we propose a systematic framework based on the Roofline model that unifies architectural primitives and hardware constraints through the lens of operational intensity (OI). By defining an inference-potential region, we introduce the Relative Inference Potential as a novel metric to compare efficiency differences between Large Language Models (LLMs) on the same hardware substrate. Extensive empirical analysis across diverse compute tiers reveals that variations in performance and OI are significantly influenced by sequence length. We further identify a critical regression in OI as model depth increases. Additionally, our findings highlight an efficiency trap induced by hardware heterogeneity and demonstrate how structural refinements, such as Multi-head Latent Attention (M LA), can effectively unlock latent inference potential across various hardware substrates. These insights provide actionable directions for hardware-software co-design to align neural structures with physical constraints in on-device intelligence. The released code is available in the Appendix C. 

---
# Understanding Persuasive Interactions between Generative Social Agents and Humans: The Knowledge-based Persuasion Model (KPM) 

**Authors**: Stephan Vonschallen, Friederike Eyssel, Theresa Schmiedel  

**Link**: [PDF](https://arxiv.org/pdf/2602.11483)  

**Abstract**: Generative social agents (GSAs) use artificial intelligence to autonomously communicate with human users in a natural and adaptive manner. Currently, there is a lack of theorizing regarding interactions with GSAs, and likewise, few guidelines exist for studying how they influence user attitudes and behaviors. Consequently, we propose the Knowledge-based Persuasion Model (KPM) as a novel theoretical framework. According to the KPM, a GSA's self, user, and context-related knowledge drives its persuasive behavior, which in turn shapes the attitudes and behaviors of a responding human user. By synthesizing existing research, the model offers a structured approach to studying interactions with GSAs, supporting the development of agents that motivate rather than manipulate humans. Accordingly, the KPM encourages the integration of responsible GSAs that adhere to social norms and ethical standards with the goal of increasing user wellbeing. Implications of the KPM for research and application domains such as healthcare and education are discussed. 

---
# Compiler-Guided Inference-Time Adaptation: Improving GPT-5 Programming Performance in Idris 

**Authors**: Minda Li, Bhaskar Krishnamachari  

**Link**: [PDF](https://arxiv.org/pdf/2602.11481)  

**Abstract**: GPT-5, a state of the art large language model from OpenAI, demonstrates strong performance in widely used programming languages such as Python, C++, and Java; however, its ability to operate in low resource or less commonly used languages remains underexplored. This work investigates whether GPT-5 can effectively acquire proficiency in an unfamiliar functional programming language, Idris, through iterative, feedback driven prompting. We first establish a baseline showing that with zero shot prompting the model solves only 22 out of 56 Idris exercises using the platform Exercism, substantially underperforming relative to higher resource languages (45 out of 50 in Python and 35 out of 47 in Erlang). We then evaluate several refinement strategies, including iterative prompting based on platform feedback, augmenting prompts with documentation and error classification guides, and iterative prompting using local compilation errors and failed test cases. Among these approaches, incorporating local compilation errors yields the most substantial improvements. Using this structured, error guided refinement loop, GPT-5 performance increased to an impressive 54 solved problems out of 56. These results suggest that while large language models may initially struggle in low resource settings, structured compiler level feedback can play a critical role in unlocking their capabilities. 

---
# EM-Aware Physical Synthesis: Neural Inductor Modeling and Intelligent Placement & Routing for RF Circuits 

**Authors**: Yilun Huang, Asal Mehradfar, Salman Avestimehr, Hamidreza Aghasi  

**Link**: [PDF](https://arxiv.org/pdf/2602.11461)  

**Abstract**: This paper presents an ML-driven framework for automated RF physical synthesis that transforms circuit netlists into manufacturable GDSII layouts. While recent ML approaches demonstrate success in topology selection and parameter optimization, they fail to produce manufacturable layouts due to oversimplified component models and lack of routing capabilities. Our framework addresses these limitations through three key innovations: (1) a neural network framework trained on 18,210 inductor geometries with frequency sweeps from 1-100 GHz, generating 7.5 million training samples, that predicts inductor Q-factor with less than 2% error and enables fast gradient-based layout optimization with a 93.77% success rate in producing high-Q layouts; (2) an intelligent P-Cell optimizer that reduces layout area while maintaining design-rule-check (DRC) compliance; and (3) a complete placement and routing engine with frequency-dependent EM spacing rules and DRC-aware synthesis. The neural inductor model demonstrates superior accuracy across 1-100 GHz, enabling EM-accurate component synthesis with real-time inference. The framework successfully generates DRC-aware GDSII layouts for RF circuits, representing a significant step toward automated RF physical design. 

---
# From Noise to Order: Learning to Rank via Denoising Diffusion 

**Authors**: Sajad Ebrahimi, Bhaskar Mitra, Negar Arabzadeh, Ye Yuan, Haolun Wu, Fattane Zarrinkalam, Ebrahim Bagheri  

**Link**: [PDF](https://arxiv.org/pdf/2602.11453)  

**Abstract**: In information retrieval (IR), learning-to-rank (LTR) methods have traditionally limited themselves to discriminative machine learning approaches that model the probability of the document being relevant to the query given some feature representation of the query-document pair. In this work, we propose an alternative denoising diffusion-based deep generative approach to LTR that instead models the full joint distribution over feature vectors and relevance labels. While in the discriminative setting, an over-parameterized ranking model may find different ways to fit the training data, we hypothesize that candidate solutions that can explain the full data distribution under the generative setting produce more robust ranking models. With this motivation, we propose DiffusionRank that extends TabDiff, an existing denoising diffusion-based generative model for tabular datasets, to create generative equivalents of classical discriminative pointwise and pairwise LTR objectives. Our empirical results demonstrate significant improvements from DiffusionRank models over their discriminative counterparts. Our work points to a rich space for future research exploration on how we can leverage ongoing advancements in deep generative modeling approaches, such as diffusion, for learning-to-rank in IR. 

---
# Enhanced Portable Ultra Low-Field Diffusion Tensor Imaging with Bayesian Artifact Correction and Deep Learning-Based Super-Resolution 

**Authors**: Mark D. Olchanyi, Annabel Sorby-Adams, John Kirsch, Brian L. Edlow, Ava Farnan, Renfei Liu, Matthew S. Rosen, Emery N. Brown, W. Taylor Kimberly, Juan Eugenio Iglesias  

**Link**: [PDF](https://arxiv.org/pdf/2602.11446)  

**Abstract**: Portable, ultra-low-field (ULF) magnetic resonance imaging has the potential to expand access to neuroimaging but currently suffers from coarse spatial and angular resolutions and low signal-to-noise ratios. Diffusion tensor imaging (DTI), a sequence tailored to detect and reconstruct white matter tracts within the brain, is particularly prone to such imaging degradation due to inherent sequence design coupled with prolonged scan times. In addition, ULF DTI scans exhibit artifacting that spans both the space and angular domains, requiring a custom modelling algorithm for subsequent correction. We introduce a nine-direction, single-shell ULF DTI sequence, as well as a companion Bayesian bias field correction algorithm that possesses angular dependence and convolutional neural network-based superresolution algorithm that is generalizable across DTI datasets and does not require re-training (''DiffSR''). We show through a synthetic downsampling experiment and white matter assessment in real, matched ULF and high-field DTI scans that these algorithms can recover microstructural and volumetric white matter information at ULF. We also show that DiffSR can be directly applied to white matter-based Alzheimers disease classification in synthetically degraded scans, with notable improvements in agreement between DTI metrics, as compared to un-degraded scans. We freely disseminate the Bayesian bias correction algorithm and DiffSR with the goal of furthering progress on both ULF reconstruction methods and general DTI sequence harmonization. We release all code related to DiffSR for $\href{this https URL}{public \space use}$. 

---
# Towards Reliable Machine Translation: Scaling LLMs for Critical Error Detection and Safety 

**Authors**: Muskaan Chopra, Lorenz Sparrenberg, Rafet Sifa  

**Link**: [PDF](https://arxiv.org/pdf/2602.11444)  

**Abstract**: Machine Translation (MT) plays a pivotal role in cross-lingual information access, public policy communication, and equitable knowledge dissemination. However, critical meaning errors, such as factual distortions, intent reversals, or biased translations, can undermine the reliability, fairness, and safety of multilingual systems. In this work, we explore the capacity of instruction-tuned Large Language Models (LLMs) to detect such critical errors, evaluating models across a range of parameters using the publicly accessible data sets. Our findings show that model scaling and adaptation strategies (zero-shot, few-shot, fine-tuning) yield consistent improvements, outperforming encoder-only baselines like XLM-R and ModernBERT. We argue that improving critical error detection in MT contributes to safer, more trustworthy, and socially accountable information systems by reducing the risk of disinformation, miscommunication, and linguistic harm, especially in high-stakes or underrepresented contexts. This work positions error detection not merely as a technical challenge, but as a necessary safeguard in the pursuit of just and responsible multilingual AI. The code will be made available at GitHub. 

---
# Fighting MRI Anisotropy: Learning Multiple Cardiac Shapes From a Single Implicit Neural Representation 

**Authors**: Carolina Brás, Soufiane Ben Haddou, Thijs P. Kuipers, Laura Alvarez-Florez, R. Nils Planken, Fleur V. Y. Tjong, Connie Bezzina, Ivana Išgum  

**Link**: [PDF](https://arxiv.org/pdf/2602.11436)  

**Abstract**: The anisotropic nature of short-axis (SAX) cardiovascular magnetic resonance imaging (CMRI) limits cardiac shape analysis. To address this, we propose to leverage near-isotropic, higher resolution computed tomography angiography (CTA) data of the heart. We use this data to train a single neural implicit function to jointly represent cardiac shapes from CMRI at any resolution. We evaluate the method for the reconstruction of right ventricle (RV) and myocardium (MYO), where MYO simultaneously models endocardial and epicardial left-ventricle surfaces. Since high-resolution SAX reference segmentations are unavailable, we evaluate performance by extracting a 4-chamber (4CH) slice of RV and MYO from their reconstructed shapes. When compared with the reference 4CH segmentation masks from CMRI, our method achieved a Dice similarity coefficient of 0.91 $\pm$ 0.07 and 0.75 $\pm$ 0.13, and a Hausdorff distance of 6.21 $\pm$ 3.97 mm and 7.53 $\pm$ 5.13 mm for RV and MYO, respectively. Quantitative and qualitative assessment demonstrate the model's ability to reconstruct accurate, smooth and anatomically plausible shapes, supporting improvements in cardiac shape analysis. 

---
# Gradients Must Earn Their Influence: Unifying SFT with Generalized Entropic Objectives 

**Authors**: Zecheng Wang, Deyuan Liu, Chunshan Li, Yupeng Zhang, Zhengyun Zhao, Dianhui Chu, Bingning Wang, Dianbo Sui  

**Link**: [PDF](https://arxiv.org/pdf/2602.11424)  

**Abstract**: Standard negative log-likelihood (NLL) for Supervised Fine-Tuning (SFT) applies uniform token-level weighting. This rigidity creates a two-fold failure mode: (i) overemphasizing low-probability targets can amplify gradients on noisy supervision and disrupt robust priors, and (ii) uniform weighting provides weak sharpening when the model is already confident. Existing methods fail to resolve the resulting plasticity--stability dilemma, often suppressing necessary learning signals alongside harmful ones. To address this issue, we unify token-level SFT objectives within a generalized deformed-log family and expose a universal gate $\times$ error gradient structure, where the gate controls how much the model trusts its current prediction. By employing the Cayley transform, we map the model's continuously evolving uncertainty onto a continuous focus trajectory, which enables seamless interpolation between scenarios involving uncertain novel concepts and those involving well-established knowledge. We then introduce Dynamic Entropy Fine-Tuning (DEFT), a parameter-free objective that modulates the trust gate using distribution concentration (Rényi-2 entropy) as a practical proxy for the model's predictive state. Extensive experiments and analyses demonstrate that DEFT achieves a better balance between exploration and exploitation, leading to improved overall performance. 

---
# When Visibility Outpaces Verification: Delayed Verification and Narrative Lock-in in Agentic AI Discourse 

**Authors**: Hanjing Shi, Dominic DiFranzo  

**Link**: [PDF](https://arxiv.org/pdf/2602.11412)  

**Abstract**: Agentic AI systems-autonomous entities capable of independent planning and execution-reshape the landscape of human-AI trust. Long before direct system exposure, user expectations are mediated through high-stakes public discourse on social platforms. However, platform-mediated engagement signals (e.g., upvotes) may inadvertently function as a ``credibility proxy,'' potentially stifling critical evaluation.
This paper investigates the interplay between social proof and verification timing in online discussions of agentic AI. Analyzing a longitudinal dataset from two distinct Reddit communities with contrasting interaction cultures-r/OpenClaw and r/Moltbook-we operationalize verification cues via reproducible lexical rules and model the ``time-to-first-verification'' using a right-censored survival analysis framework.
Our findings reveal a systemic ``Popularity Paradox'': high-visibility discussions in both subreddits experience significantly delayed or entirely absent verification cues compared to low-visibility threads. This temporal lag creates a critical window for ``Narrative Lock-in,'' where early, unverified claims crystallize into collective cognitive biases before evidence-seeking behaviors emerge. We discuss the implications of this ``credibility-by-visibility'' effect for AI safety and propose ``epistemic friction'' as a design intervention to rebalance engagement-driven platforms. 

---
# Can We Really Learn One Representation to Optimize All Rewards? 

**Authors**: Chongyi Zheng, Royina Karegoudra Jayanth, Benjamin Eysenbach  

**Link**: [PDF](https://arxiv.org/pdf/2602.11399)  

**Abstract**: As machine learning has moved towards leveraging large models as priors for downstream tasks, the community has debated the right form of prior for solving reinforcement learning (RL) problems. If one were to try to prefetch as much computation as possible, they would attempt to learn a prior over the policies for some yet-to-be-determined reward function. Recent work (forward-backward (FB) representation learning) has tried this, arguing that an unsupervised representation learning procedure can enable optimal control over arbitrary rewards without further fine-tuning. However, FB's training objective and learning behavior remain mysterious. In this paper, we demystify FB by clarifying when such representations can exist, what its objective optimizes, and how it converges in practice. We draw connections with rank matching, fitted Q-evaluation, and contraction mapping. Our analysis suggests a simplified unsupervised pre-training method for RL that, instead of enabling optimal control, performs one step of policy improvement. We call our proposed method $\textbf{one-step forward-backward representation learning (one-step FB)}$. Experiments in didactic settings, as well as in $10$ state-based and image-based continuous control domains, demonstrate that one-step FB converges to errors $10^5$ smaller and improves zero-shot performance by $+24\%$ on average. Our project website is available at this https URL. 

---
# General and Efficient Steering of Unconditional Diffusion 

**Authors**: Qingsong Wang, Mikhail Belkin, Yusu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.11395)  

**Abstract**: Guiding unconditional diffusion models typically requires either retraining with conditional inputs or per-step gradient computations (e.g., classifier-based guidance), both of which incur substantial computational overhead. We present a general recipe for efficiently steering unconditional diffusion {without gradient guidance during inference}, enabling fast controllable generation. Our approach is built on two observations about diffusion model structure: Noise Alignment: even in early, highly corrupted stages, coarse semantic steering is possible using a lightweight, offline-computed guidance signal, avoiding any per-step or per-sample gradients. Transferable concept vectors: a concept direction in activation space once learned transfers across both {timesteps} and {samples}; the same fixed steering vector learned near low noise level remains effective when injected at intermediate noise levels for every generation trajectory, providing refined conditional control with efficiency. Such concept directions can be efficiently and reliably identified via Recursive Feature Machine (RFM), a light-weight backpropagation-free feature learning method. Experiments on CIFAR-10, ImageNet, and CelebA demonstrate improved accuracy/quality over gradient-based guidance, while achieving significant inference speedups. 

---
# Retrieval-Aware Distillation for Transformer-SSM Hybrids 

**Authors**: Aviv Bick, Eric P. Xing, Albert Gu  

**Link**: [PDF](https://arxiv.org/pdf/2602.11374)  

**Abstract**: State-space models (SSMs) offer efficient sequence modeling but lag behind Transformers on benchmarks that require in-context retrieval. Prior work links this gap to a small set of attention heads, termed Gather-and-Aggregate (G&A), which SSMs struggle to reproduce. We propose *retrieval-aware distillation*, which converts a pretrained Transformer into a hybrid student by preserving only these retrieval-critical heads and distilling the rest into recurrent heads. We identify the essential heads via ablation on a synthetic retrieval task, producing a hybrid with sparse, non-uniform attention placement. We show that preserving **just 2% of attention heads recovers over 95% of teacher performance on retrieval-heavy tasks** (10 heads in a 1B model), requiring far fewer heads than hybrids that retain at least 25%. We further find that large recurrent states often compensate for missing retrieval: once retrieval is handled by these heads, the SSM backbone can be simplified with limited loss, even with an $8\times$ reduction in state dimension. By reducing both the attention cache and the SSM state, the resulting hybrid is $5$--$6\times$ more memory-efficient than comparable hybrids, closing the Transformer--SSM gap at a fraction of the memory cost. 

---
# The Manifold of the Absolute: Religious Perennialism as Generative Inference 

**Authors**: Arthur Juliani  

**Link**: [PDF](https://arxiv.org/pdf/2602.11368)  

**Abstract**: This paper formalizes religious epistemology through the mathematics of Variational Autoencoders. We model religious traditions as distinct generative mappings from a shared, low-dimensional latent space to the high-dimensional space of observable cultural forms, and define three competing generative configurations corresponding to exclusivism, universalism, and perennialism, alongside syncretism as direct mixing in observable space. Through abductive comparison, we argue that exclusivism cannot parsimoniously account for cross-traditional contemplative convergence, that syncretism fails because combining the outputs of distinct generative processes produces incoherent artifacts, and that universalism suffers from posterior collapse: stripping traditions to a common core discards the structural information necessary for inference. The perennialist configuration provides the best explanatory fit. Within this framework, strict orthodoxy emerges not as a cultural constraint but as a structural necessity: the contemplative practices that recover the latent source must be matched to the specific tradition whose forms they take as input. The unity of religions, if it exists, is real but inaccessible by shortcut: one must go deep rather than wide. 

---
# The Energy of Falsehood: Detecting Hallucinations via Diffusion Model Likelihoods 

**Authors**: Arpit Singh Gautam, Kailash Talreja, Saurabh Jha  

**Link**: [PDF](https://arxiv.org/pdf/2602.11364)  

**Abstract**: Large Language Models (LLMs) frequently hallucinate plausible but incorrect assertions, a vulnerability often missed by uncertainty metrics when models are confidently wrong. We propose DiffuTruth, an unsupervised framework that reconceptualizes fact verification via non equilibrium thermodynamics, positing that factual truths act as stable attractors on a generative manifold while hallucinations are unstable. We introduce the Generative Stress Test, claims are corrupted with noise and reconstructed using a discrete text diffusion model. We define Semantic Energy, a metric measuring the semantic divergence between the original claim and its reconstruction using an NLI critic. Unlike vector space errors, Semantic Energy isolates deep factual contradictions. We further propose a Hybrid Calibration fusing this stability signal with discriminative confidence. Extensive experiments on FEVER demonstrate DiffuTruth achieves a state of the art unsupervised AUROC of 0.725, outperforming baselines by 1.5 percent through the correction of overconfident predictions. Furthermore, we show superior zero shot generalization on the multi hop HOVER dataset, outperforming baselines by over 4 percent, confirming the robustness of thermodynamic truth properties to distribution shifts. 

---
# Finding the Cracks: Improving LLMs Reasoning with Paraphrastic Probing and Consistency Verification 

**Authors**: Weili Shi, Dongliang Guo, Lehan Yang, Tianlong Wang, Hanzhang Yuan, Sheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.11361)  

**Abstract**: Large language models have demonstrated impressive performance across a variety of reasoning tasks. However, their problem-solving ability often declines on more complex tasks due to hallucinations and the accumulation of errors within these intermediate steps. Recent work has introduced the notion of critical tokens--tokens in the reasoning process that exert significant influence on subsequent steps. Prior studies suggest that replacing critical tokens can refine reasoning trajectories. Nonetheless, reliably identifying and exploiting critical tokens remains challenging. To address this, we propose the Paraphrastic Probing and Consistency Verification~(PPCV) framework. PPCV operates in two stages. In the first stage, we roll out an initial reasoning path from the original question and then concatenate paraphrased versions of the question with this reasoning path. And we identify critical tokens based on mismatches between the predicted top-1 token and the expected token in the reasoning path. A criterion is employed to confirm the final critical token. In the second stage, we substitute critical tokens with candidate alternatives and roll out new reasoning paths for both the original and paraphrased questions. The final answer is determined by checking the consistency of outputs across these parallel reasoning processes. We evaluate PPCV on mainstream LLMs across multiple benchmarks. Extensive experiments demonstrate PPCV substantially enhances the reasoning performance of LLMs compared to baselines. 

---
# Bootstrapping-based Regularisation for Reducing Individual Prediction Instability in Clinical Risk Prediction Models 

**Authors**: Sara Matijevic, Christopher Yau  

**Link**: [PDF](https://arxiv.org/pdf/2602.11360)  

**Abstract**: Clinical prediction models are increasingly used to support patient care, yet many deep learning-based approaches remain unstable, as their predictions can vary substantially when trained on different samples from the same population. Such instability undermines reliability and limits clinical adoption. In this study, we propose a novel bootstrapping-based regularisation framework that embeds the bootstrapping process directly into the training of deep neural networks. This approach constrains prediction variability across resampled datasets, producing a single model with inherent stability properties. We evaluated models constructed using the proposed regularisation approach against conventional and ensemble models using simulated data and three clinical datasets: GUSTO-I, Framingham, and SUPPORT. Across all datasets, our model exhibited improved prediction stability, with lower mean absolute differences (e.g., 0.019 vs. 0.059 in GUSTO-I; 0.057 vs. 0.088 in Framingham) and markedly fewer significantly deviating predictions. Importantly, discriminative performance and feature importance consistency were maintained, with high SHAP correlations between models (e.g., 0.894 for GUSTO-I; 0.965 for Framingham). While ensemble models achieved greater stability, we show that this came at the expense of interpretability, as each constituent model used predictors in different ways. By regularising predictions to align with bootstrapped distributions, our approach allows prediction models to be developed that achieve greater robustness and reproducibility without sacrificing interpretability. This method provides a practical route toward more reliable and clinically trustworthy deep learning models, particularly valuable in data-limited healthcare settings. 

---
# When Models Examine Themselves: Vocabulary-Activation Correspondence in Self-Referential Processing 

**Authors**: Zachary Pedram Dadfar  

**Link**: [PDF](https://arxiv.org/pdf/2602.11358)  

**Abstract**: Large language models produce rich introspective language when prompted for self-examination, but whether this language reflects internal computation or sophisticated confabulation has remained unclear. We show that self-referential vocabulary tracks concurrent activation dynamics, and that this correspondence is specific to self-referential processing. We introduce the Pull Methodology, a protocol that elicits extended self-examination through format engineering, and use it to identify a direction in activation space that distinguishes self-referential from descriptive processing in Llama 3.1. The direction is orthogonal to the known refusal direction, localised at 6.25% of model depth, and causally influences introspective output when used for steering. When models produce "loop" vocabulary, their activations exhibit higher autocorrelation (r = 0.44, p = 0.002); when they produce "shimmer" vocabulary under steering, activation variability increases (r = 0.36, p = 0.002). Critically, the same vocabulary in non-self-referential contexts shows no activation correspondence despite nine-fold higher frequency. Qwen 2.5-32B, with no shared training, independently develops different introspective vocabulary tracking different activation metrics, all absent in descriptive controls. The findings indicate that self-report in transformer models can, under appropriate conditions, reliably track internal computational states. 

---
# Divide and Learn: Multi-Objective Combinatorial Optimization at Scale 

**Authors**: Esha Singh, Dongxia Wu, Chien-Yi Yang, Tajana Rosing, Rose Yu, Yi-An Ma  

**Link**: [PDF](https://arxiv.org/pdf/2602.11346)  

**Abstract**: Multi-objective combinatorial optimization seeks Pareto-optimal solutions over exponentially large discrete spaces, yet existing methods sacrifice generality, scalability, or theoretical guarantees. We reformulate it as an online learning problem over a decomposed decision space, solving position-wise bandit subproblems via adaptive expert-guided sequential construction. This formulation admits regret bounds of $O(d\sqrt{T \log T})$ depending on subproblem dimensionality \(d\) rather than combinatorial space size. On standard benchmarks, our method achieves 80--98\% of specialized solvers performance while achieving two to three orders of magnitude improvement in sample and computational efficiency over Bayesian optimization methods. On real-world hardware-software co-design for AI accelerators with expensive simulations, we outperform competing methods under fixed evaluation budgets. The advantage grows with problem scale and objective count, establishing bandit optimization over decomposed decision spaces as a principled alternative to surrogate modeling or offline training for multi-objective optimization. 

---
# Situated, Dynamic, and Subjective: Envisioning the Design of Theory-of-Mind-Enabled Everyday AI with Industry Practitioners 

**Authors**: Qiaosi Wang, Jini Kim, Avanita Sharma, Alicia, Jodi Forlizzi, Hong Shen  

**Link**: [PDF](https://arxiv.org/pdf/2602.11342)  

**Abstract**: Theory of Mind (ToM) -- the ability to infer what others are thinking (e.g., intentions) from observable cues -- is traditionally considered fundamental to human social interactions. This has sparked growing efforts in building and benchmarking AI's ToM capability, yet little is known about how such capability could translate into the design and experience of everyday user-facing AI products and services. We conducted 13 co-design sessions with 26 U.S.-based AI practitioners to envision, reflect, and distill design recommendations for ToM-enabled everyday AI products and services that are both future-looking and grounded in the realities of AI design and development practices. Analysis revealed three interrelated design recommendations: ToM-enabled AI should 1) be situated in the social context that shape users' mental states, 2) be responsive to the dynamic nature of mental states, and 3) be attuned to subjective individual differences. We surface design tensions within each recommendation that reveal a broader gap between practitioners' envisioned futures of ToM-enabled AI and the realities of current AI design and development practices. These findings point toward the need to move beyond static, inference-driven approach to ToM and toward designing ToM as a pervasive capability that supports continuous human-AI interaction loops. 

---
# MolmoSpaces: A Large-Scale Open Ecosystem for Robot Navigation and Manipulation 

**Authors**: Yejin Kim, Wilbert Pumacay, Omar Rayyan, Max Argus, Winson Han, Eli VanderBilt, Jordi Salvador, Abhay Deshpande, Rose Hendrix, Snehal Jauhri, Shuo Liu, Nur Muhammad Mahi Shafiullah, Maya Guru, Ainaz Eftekhar, Karen Farley, Donovan Clay, Jiafei Duan, Arjun Guru, Piper Wolters, Alvaro Herrasti, Ying-Chun Lee, Georgia Chalvatzaki, Yuchen Cui, Ali Farhadi, Dieter Fox, Ranjay Krishna  

**Link**: [PDF](https://arxiv.org/pdf/2602.11337)  

**Abstract**: Deploying robots at scale demands robustness to the long tail of everyday situations. The countless variations in scene layout, object geometry, and task specifications that characterize real environments are vast and underrepresented in existing robot benchmarks. Measuring this level of generalization requires infrastructure at a scale and diversity that physical evaluation alone cannot provide. We introduce MolmoSpaces, a fully open ecosystem to support large-scale benchmarking of robot policies. MolmoSpaces consists of over 230k diverse indoor environments, ranging from handcrafted household scenes to procedurally generated multiroom houses, populated with 130k richly annotated object assets, including 48k manipulable objects with 42M stable grasps. Crucially, these environments are simulator-agnostic, supporting popular options such as MuJoCo, Isaac, and ManiSkill. The ecosystem supports the full spectrum of embodied tasks: static and mobile manipulation, navigation, and multiroom long-horizon tasks requiring coordinated perception, planning, and interaction across entire indoor environments. We also design MolmoSpaces-Bench, a benchmark suite of 8 tasks in which robots interact with our diverse scenes and richly annotated objects. Our experiments show MolmoSpaces-Bench exhibits strong sim-to-real correlation (R = 0.96, \r{ho} = 0.98), confirm newer and stronger zero-shot policies outperform earlier versions in our benchmarks, and identify key sensitivities to prompt phrasing, initial joint positions, and camera occlusion. Through MolmoSpaces and its open-source assets and tooling, we provide a foundation for scalable data generation, policy training, and benchmark creation for robot learning research. 

---
# Security Threat Modeling for Emerging AI-Agent Protocols: A Comparative Analysis of MCP, A2A, Agora, and ANP 

**Authors**: Zeynab Anbiaee, Mahdi Rabbani, Mansur Mirani, Gunjan Piya, Igor Opushnyev, Ali Ghorbani, Sajjad Dadkhah  

**Link**: [PDF](https://arxiv.org/pdf/2602.11327)  

**Abstract**: The rapid development of the AI agent communication protocols, including the Model Context Protocol (MCP), Agent2Agent (A2A), Agora, and Agent Network Protocol (ANP), is reshaping how AI agents communicate with tools, services, and each other. While these protocols support scalable multi-agent interaction and cross-organizational interoperability, their security principles remain understudied, and standardized threat modeling is limited; no protocol-centric risk assessment framework has been established yet. This paper presents a systematic security analysis of four emerging AI agent communication protocols. First, we develop a structured threat modeling analysis that examines protocol architectures, trust assumptions, interaction patterns, and lifecycle behaviors to identify protocol-specific and cross-protocol risk surfaces. Second, we introduce a qualitative risk assessment framework that identifies twelve protocol-level risks and evaluates security posture across the creation, operation, and update phases through systematic assessment of likelihood, impact, and overall protocol risk, with implications for secure deployment and future standardization. Third, we provide a measurement-driven case study on MCP that formalizes the risk of missing mandatory validation/attestation for executable components as a falsifiable security claim by quantifying wrong-provider tool execution under multi-server composition across representative resolver policies. Collectively, our results highlight key design-induced risk surfaces and provide actionable guidance for secure deployment and future standardization of agent communication ecosystems. 

---
# Predictive Associative Memory: Retrieval Beyond Similarity Through Temporal Co-occurrence 

**Authors**: Jason Dury  

**Link**: [PDF](https://arxiv.org/pdf/2602.11322)  

**Abstract**: Current approaches to memory in neural systems rely on similarity-based retrieval: given a query, find the most representationally similar stored state. This assumption -- that useful memories are similar memories -- fails to capture a fundamental property of biological memory: association through temporal co-occurrence. We propose Predictive Associative Memory (PAM), an architecture in which a JEPA-style predictor, trained on temporal co-occurrence within a continuous experience stream, learns to navigate the associative structure of an embedding space. We introduce an Inward JEPA that operates over stored experience (predicting associatively reachable past states) as the complement to the standard Outward JEPA that operates over incoming sensory data (predicting future states). We evaluate PAM as an associative recall system -- testing faithfulness of recall for experienced associations -- rather than as a retrieval system evaluated on generalisation to unseen associations. On a synthetic benchmark, the predictor's top retrieval is a true temporal associate 97% of the time (Association Precision@1 = 0.970); it achieves cross-boundary Recall@20 = 0.421 where cosine similarity scores zero; and it separates experienced-together from never-experienced-together states with a discrimination AUC of 0.916 (cosine: 0.789). Even restricted to cross-room pairs where embedding similarity is uninformative, the predictor achieves AUC = 0.849 (cosine: 0.503, chance). A temporal shuffle control confirms the signal is genuine temporal co-occurrence structure, not embedding geometry: shuffling collapses cross-boundary recall by 90%, replicated across training seeds. All results are stable across seeds (SD < 0.006) and query selections (SD $\leq$ 0.012). 

---
# CryptoAnalystBench: Failures in Multi-Tool Long-Form LLM Analysis 

**Authors**: Anushri Eswaran, Oleg Golev, Darshan Tank, Sidhant Rahi, Himanshu Tyagi  

**Link**: [PDF](https://arxiv.org/pdf/2602.11304)  

**Abstract**: Modern analyst agents must reason over complex, high token inputs, including dozens of retrieved documents, tool outputs, and time sensitive data. While prior work has produced tool calling benchmarks and examined factuality in knowledge augmented systems, relatively little work studies their intersection: settings where LLMs must integrate large volumes of dynamic, structured and unstructured multi tool outputs. We investigate LLM failure modes in this regime using crypto as a representative high data density domain. We introduce (1) CryptoAnalystBench, an analyst aligned benchmark of 198 production crypto and DeFi queries spanning 11 categories; (2) an agentic harness equipped with relevant crypto and DeFi tools to generate responses across multiple frontier LLMs; and (3) an evaluation pipeline with citation verification and an LLM as a judge rubric spanning four user defined success dimensions: relevance, temporal relevance, depth, and data consistency. Using human annotation, we develop a taxonomy of seven higher order error types that are not reliably captured by factuality checks or LLM based quality scoring. We find that these failures persist even in state of the art systems and can compromise high stakes decisions. Based on this taxonomy, we refine the judge rubric to better capture these errors. While the judge does not align with human annotators on precise scoring across rubric iterations, it reliably identifies critical failure modes, enabling scalable feedback for developers and researchers studying analyst style agents. We release CryptoAnalystBench with annotated queries, the evaluation pipeline, judge rubrics, and the error taxonomy, and outline mitigation strategies and open challenges in evaluating long form, multi tool augmented systems. 

---
# HiFloat4 Format for Language Model Inference 

**Authors**: Yuanyong Luo, Jing Huang, Yu Cheng, Ziwei Yu, Kaihua Zhang, Kehong Hong, Xinda Ma, Xin Wang, Anping Tong, Guipeng Hu, Yun Xu, Mehran Taghian, Peng Wu, Guanglin Li, Yunke Peng, Tianchi Hu, Minqi Chen, Michael Bi Mi, Hu Liu, Xiping Zhou, Junsong Wang, Qiang Lin, Heng Liao  

**Link**: [PDF](https://arxiv.org/pdf/2602.11287)  

**Abstract**: This paper introduces HiFloat4 (HiF4), a block floating-point data format tailored for deep learning. Each HiF4 unit packs 64 4-bit elements with 32 bits of shared scaling metadata, averaging 4.5 bits per value. The metadata specifies a three-level scaling hierarchy, capturing inter- and intra-group dynamic range while improving the utilization of the representational space. In addition, the large 64-element group size enables matrix multiplications to be executed in a highly fixed-point manner, significantly reducing hardware area and power consumption. To evaluate the proposed format, we conducted inference experiments on several language models, including LLaMA, Qwen, Mistral, DeepSeek-V3.1 and LongCat. Results show that HiF4 achieves higher average accuracy than the state-of-the-art NVFP4 format across multiple models and diverse downstream tasks. 

---
# DeepRed: an architecture for redshift estimation 

**Authors**: Alessandro Meroni, Nicolò Oreste Pinciroli Vago, Piero Fraternali  

**Link**: [PDF](https://arxiv.org/pdf/2602.11281)  

**Abstract**: Estimating redshift is a central task in astrophysics, but its measurement is costly and time-consuming. In addition, current image-based methods are often validated on homogeneous datasets. The development and comparison of networks able generalize across different morphologies, ranging from galaxies to gravitationally-lensed transients, and observational conditions, remain an open challenge. This work proposes DeepRed, a deep learning pipeline that demonstrates how modern computer vision architectures, including ResNet, EfficientNet, Swin Transformer, and MLP-Mixer, can estimate redshifts from images of galaxies, gravitational lenses, and gravitationally-lensed supernovae. We compare these architectures and their ensemble to both neural networks (A1, A3, NetZ, and PhotoZ) and a feature-based method (HOG+SVR) on simulated (DeepGraviLens) and real (KiDS, SDSS) datasets. Our approach achieves state-of-the-art results on all datasets. On DeepGraviLens, DeepRed achieves a significant improvement in the Normalized Mean Absolute Deviation compared to the best baseline (PhotoZ): 55% on DES-deep (using EfficientNet), 51% on DES-wide (Ensemble), 52% on DESI-DOT (Ensemble), and 46% on LSST-wide (Ensemble). On real observations from the KiDS survey, the pipeline outperforms the best baseline (NetZ), improving NMAD by 16% on a general test set without high-probability lenses (Ensemble) and 27% on high-probability lenses (Ensemble). For non-lensed galaxies in the SDSS dataset, the MLP-Mixer architecture achieves a 5% improvement over the best baselines (A3 and NetZ). SHAP shows that the models correctly focus on the objects of interest with over 95% localization accuracy on high-quality images, validating the reliability of the predictions. These findings suggest that deep learning is a scalable, robust, and interpretable solution for redshift estimation in large-scale surveys. 

---
# How Many Features Can a Language Model Store Under the Linear Representation Hypothesis? 

**Authors**: Nikhil Garg, Jon Kleinberg, Kenny Peng  

**Link**: [PDF](https://arxiv.org/pdf/2602.11246)  

**Abstract**: We introduce a mathematical framework for the linear representation hypothesis (LRH), which asserts that intermediate layers of language models store features linearly. We separate the hypothesis into two claims: linear representation (features are linearly embedded in neuron activations) and linear accessibility (features can be linearly decoded). We then ask: How many neurons $d$ suffice to both linearly represent and linearly access $m$ features? Classical results in compressed sensing imply that for $k$-sparse inputs, $d = O(k\log (m/k))$ suffices if we allow non-linear decoding algorithms (Candes and Tao, 2006; Candes et al., 2006; Donoho, 2006). However, the additional requirement of linear decoding takes the problem out of the classical compressed sensing, into linear compressed sensing.
Our main theoretical result establishes nearly-matching upper and lower bounds for linear compressed sensing. We prove that $d = \Omega_\epsilon(\frac{k^2}{\log k}\log (m/k))$ is required while $d = O_\epsilon(k^2\log m)$ suffices. The lower bound establishes a quantitative gap between classical and linear compressed setting, illustrating how linear accessibility is a meaningfully stronger hypothesis than linear representation alone. The upper bound confirms that neurons can store an exponential number of features under the LRH, giving theoretical evidence for the "superposition hypothesis" (Elhage et al., 2022).
The upper bound proof uses standard random constructions of matrices with approximately orthogonal columns. The lower bound proof uses rank bounds for near-identity matrices (Alon, 2003) together with Turán's theorem (bounding the number of edges in clique-free graphs). We also show how our results do and do not constrain the geometry of feature representations and extend our results to allow decoders with an activation function and bias. 

---
# Toward Reliable Tea Leaf Disease Diagnosis Using Deep Learning Model: Enhancing Robustness With Explainable AI and Adversarial Training 

**Authors**: Samanta Ghosh, Jannatul Adan Mahi, Shayan Abrar, Md Parvez Mia, Asaduzzaman Rayhan, Abdul Awal Yasir, Asaduzzaman Hridoy  

**Link**: [PDF](https://arxiv.org/pdf/2602.11239)  

**Abstract**: Tea is a valuable asset for the economy of Bangladesh. So, tea cultivation plays an important role to boost the economy. These valuable plants are vulnerable to various kinds of leaf infections which may cause less production and low quality. It is not so easy to detect these diseases manually. It may take time and there could be some errors in the this http URL, the purpose of the study is to develop an automated deep learning model for tea leaf disease classification based on the teaLeafBD dataset so that anyone can detect the diseases more easily and efficiently. There are 5,278 high-resolution images in this dataset. The images are classified into seven categories. Six of them represents various diseases and the rest one represents healthy leaves. The proposed pipeline contains data preprocessing, data splitting, adversarial training, augmentation, model training, evaluation, and comprehension made possible with Explainable AI strategies. DenseNet201 and EfficientNetB3 were employed to perform the classification task. To prepare the model more robustly, we applied adversarial training so it can operate effectively even with noisy or disturbed inputs. In addition, Grad-CAM visualization was executed to analyze the model's predictions by identifying the most influential regions of each image. Our experimental outcomes revealed that EfficientNetB3 achieved the highest classification accuracy of 93%, while DenseNet201 reached 91%. The outcomes prove that the effectiveness of the proposed approach can accurately detect tea leaf diseases and provide a practical solution for advanced agricultural management. 

---
# AI-Driven Clinical Decision Support System for Enhanced Diabetes Diagnosis and Management 

**Authors**: Mujeeb Ur Rehman, Imran Rehan, Sohail Khalid  

**Link**: [PDF](https://arxiv.org/pdf/2602.11237)  

**Abstract**: Identifying type 2 diabetes mellitus can be challenging, particularly for primary care physicians. Clinical decision support systems incorporating artificial intelligence (AI-CDSS) can assist medical professionals in diagnosing type 2 diabetes with high accuracy. This study aimed to assess an AI-CDSS specifically developed for the diagnosis of type 2 diabetes by employing a hybrid approach that integrates expert-driven insights with machine learning techniques. The AI-CDSS was developed (training dataset: n = 650) and tested (test dataset: n = 648) using a dataset of 1298 patients with and without type 2 diabetes. To generate predictions, the algorithm utilized key features such as body mass index, plasma fasting glucose, and hemoglobin A1C. Furthermore, a clinical pilot study involving 105 patients was conducted to assess the diagnostic accuracy of the system in comparison to non-endocrinology specialists. The AI-CDSS showed a high degree of accuracy, with 99.8% accuracy in predicting diabetes, 99.3% in predicting prediabetes, 99.2% in identifying at-risk individuals, and 98.8% in predicting no diabetes. The test dataset revealed a 98.8% agreement between endocrinology specialists and the AI-CDSS. Type 2 diabetes was identified in 45% of 105 individuals in the pilot study. Compared with diabetes specialists, the AI-CDSS scored a 98.5% concordance rate, greatly exceeding that of nonendocrinology specialists, who had an 85% agreement rate. These findings indicate that the AI-CDSS has the potential to be a useful tool for accurately identifying type 2 diabetes, especially in situations in which diabetes specialists are not readily available. 

---
# Credal Concept Bottleneck Models: Structural Separation of Epistemic and Aleatoric Uncertainty 

**Authors**: Tanmoy Mukherjee, Marius Kloft, Pierre Marquis, Zied Bouraoui  

**Link**: [PDF](https://arxiv.org/pdf/2602.11219)  

**Abstract**: Decomposing predictive uncertainty into epistemic (model ignorance) and aleatoric (data ambiguity) components is central to reliable decision making, yet most methods estimate both from the same predictive distribution. Recent empirical and theoretical results show these estimates are typically strongly correlated, so changes in predictive spread simultaneously affect both components and blur their semantics. We propose a credal-set formulation in which uncertainty is represented as a set of predictive distributions, so that epistemic and aleatoric uncertainty correspond to distinct geometric properties: the size of the set versus the noise within its elements. We instantiate this idea in a Variational Credal Concept Bottleneck Model with two disjoint uncertainty heads trained by disjoint objectives and non-overlapping gradient paths, yielding separation by construction rather than post hoc decomposition. Across multi-annotator benchmarks, our approach reduces the correlation between epistemic and aleatoric uncertainty by over an order of magnitude compared to standard methods, while improving the alignment of epistemic uncertainty with prediction error and aleatoric uncertainty with ground-truth ambiguity. 

---
# SWE-MiniSandbox: Container-Free Reinforcement Learning for Building Software Engineering Agents 

**Authors**: Danlong Yuan, Wei Wu, Zhengren Wang, Xueliang Zhao, Huishuai Zhang, Dongyan Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2602.11210)  

**Abstract**: Reinforcement learning (RL) has become a key paradigm for training software engineering (SWE) agents, but existing pipelines typically rely on per-task containers for isolation. At scale, pre-built container images incur substantial storage overhead, slow environment setup, and require container-management privileges. We propose SWE-MiniSandbox, a lightweight, container-free method that enables scalable RL training of SWE agents without sacrificing isolation. Instead of relying on per-instance containers, SWE-MiniSandbox executes each task in an isolated workspace backed by kernel-level mechanisms, substantially reducing system overhead. It leverages lightweight environment pre-caching techniques to eliminate the need for bulky container images. As a result, our approach lowers disk usage to approximately 5\% of that required by container-based pipelines and reduces environment preparation time to about 25\% of the container baseline. Empirical results demonstrate that SWE-MiniSandbox achieves evaluation performance comparable to standard container-based pipelines. By removing the dependency on heavy container infrastructure, SWE-MiniSandbox offers a practical and accessible foundation for scaling RL-based SWE agents, particularly in resource-constrained research environments. 

---
# UltraLIF: Fully Differentiable Spiking Neural Networks via Ultradiscretization and Max-Plus Algebra 

**Authors**: Jose Marie Antonio Miñoza  

**Link**: [PDF](https://arxiv.org/pdf/2602.11206)  

**Abstract**: Spiking Neural Networks (SNNs) offer energy-efficient, biologically plausible computation but suffer from non-differentiable spike generation, necessitating reliance on heuristic surrogate gradients. This paper introduces UltraLIF, a principled framework that replaces surrogate gradients with ultradiscretization, a mathematical formalism from tropical geometry providing continuous relaxations of discrete dynamics. The central insight is that the max-plus semiring underlying ultradiscretization naturally models neural threshold dynamics: the log-sum-exp function serves as a differentiable soft-maximum that converges to hard thresholding as a learnable temperature parameter $\eps \to 0$. Two neuron models are derived from distinct dynamical systems: UltraLIF from the LIF ordinary differential equation (temporal dynamics) and UltraDLIF from the diffusion equation modeling gap junction coupling across neuronal populations (spatial dynamics). Both yield fully differentiable SNNs trainable via standard backpropagation with no forward-backward mismatch. Theoretical analysis establishes pointwise convergence to classical LIF dynamics with quantitative error bounds and bounded non-vanishing gradients. Experiments on six benchmarks spanning static images, neuromorphic vision, and audio demonstrate improvements over surrogate gradient baselines, with gains most pronounced in single-timestep ($T{=}1$) settings on neuromorphic and temporal datasets. An optional sparsity penalty enables significant energy reduction while maintaining competitive accuracy. 

---
# Zero-Sacrifice Persistent-Robustness Adversarial Defense for Pre-Trained Encoders 

**Authors**: Zhuxin Lei, Ziyuan Yang, Yi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.11204)  

**Abstract**: The widespread use of publicly available pre-trained encoders from self-supervised learning (SSL) has exposed a critical vulnerability: their susceptibility to downstream-agnostic adversarial examples (DAEs), which are crafted without knowledge of the downstream tasks but capable of misleading downstream models. While several defense methods have been explored recently, they rely primarily on task-specific adversarial fine-tuning, which inevitably limits generalizability and causes catastrophic forgetting and deteriorates benign performance. Different with previous works, we propose a more rigorous defense goal that requires only a single tuning for diverse downstream tasks to defend against DAEs and preserve benign performance. To achieve this defense goal, we introduce Zero-Sacrifice Persistent-Robustness Adversarial Defense (ZePAD), which is inspired by the inherent sensitivity of neural networks to data characteristics. Specifically, ZePAD is a dual-branch structure, which consists of a Multi-Pattern Adversarial Enhancement Branch (MPAE-Branch) that uses two adversarially fine-tuned encoders to strengthen adversarial resistance. The Benign Memory Preservation Branch (BMP-Branch) is trained on local data to ensure adversarial robustness does not compromise benign performance. Surprisingly, we find that ZePAD can directly detect DAEs by evaluating branch confidence, without introducing any adversarial exsample identification task during training. Notably, by enriching feature diversity, our method enables a single adversarial fine-tuning to defend against DAEs across downstream tasks, thereby achieving persistent robustness. Extensive experiments on 11 SSL methods and 6 datasets validate its effectiveness. In certain cases, it achieves a 29.20% improvement in benign performance and a 73.86% gain in adversarial robustness, highlighting its zero-sacrifice property. 

---
# interwhen: A Generalizable Framework for Verifiable Reasoning with Test-time Monitors 

**Authors**: Vishak K Bhat, Prateek Chanda, Ashmit Khandelwal, Maitreyi Swaroop, Vineeth N. Balasubramanian, Subbarao Kambhampati, Nagarajan Natarajan, Amit Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2602.11202)  

**Abstract**: We present a test-time verification framework, interwhen, that ensures that the output of a reasoning model is valid wrt. a given set of verifiers. Verified reasoning is an important goal in high-stakes scenarios such as deploying agents in the physical world or in domains such as law and finance. However, current techniques either rely on the generate-test paradigm that verifies only after the final answer is produced, or verify partial output through a step-extraction paradigm where the task execution is externally broken down into structured steps. The former is inefficient while the latter artificially restricts a model's problem solving strategies. Instead, we propose to verify a model's reasoning trace as-is, taking full advantage of a model's reasoning capabilities while verifying and steering the model's output only when needed. The key idea is meta-prompting, identifying the verifiable properties that any partial solution should satisfy and then prompting the model to follow a custom format in its trace such that partial outputs can be easily parsed and checked. We consider both self-verification and external verification and find that interwhen provides a useful abstraction to provide feedback and steer reasoning models in each case. Using self-verification, interwhen obtains state-of-the-art results on early stopping reasoning models, without any loss in accuracy. Using external verifiers, interwhen obtains 10 p.p. improvement in accuracy over test-time scaling methods, while ensuring 100% soundness and being 4x more efficient. The code for interwhen is available at this https URL 

---
# DDL2PropBank Agent: Benchmarking Multi-Agent Frameworks' Developer Experience Through a Novel Relational Schema Mapping Task 

**Authors**: Shafiuddin Rehan Ahmed, Wei Wei  

**Link**: [PDF](https://arxiv.org/pdf/2602.11198)  

**Abstract**: Multi-agent frameworks promise to simplify LLM-driven software development, yet there is no principled way to evaluate their developer experience in a controlled setting. We introduce DDL2PropBank, a novel benchmark task that maps relational database schemas to PropBank rolesets, requiring autonomous retrieval of candidate frames and fine-grained linguistic reasoning over table names, columns, and relations. Using the Agent-as-a-Tool pattern, we implement identical agent logic across 10 frameworks and evaluate along two dimensions: (i) code complexity via static analysis, and (ii) AI-assistability -- the extent to which LLMs can autonomously generate correct, framework-specific code. Our results reveal a threefold complexity spectrum, with Pydantic AI and Agno requiring the least implementation overhead. For AI-assistability, structural alignment scores reliably proxy runtime success for frameworks with single canonical patterns, but overestimate correctness for multi-pattern frameworks. Agno emerges as the strongest overall performer, combining lowest complexity with highest structural alignment and 83% pass@1. 

---
# Hybrid operator learning of wave scattering maps in high-contrast media 

**Authors**: Advait Balaji, Trevor Teolis, S. David Mis, Jose Antonio Lara Benitez, Chao Wang, Maarten V. de Hoop  

**Link**: [PDF](https://arxiv.org/pdf/2602.11197)  

**Abstract**: Surrogate modeling of wave propagation and scattering (i.e. the wave speed and source to wave field map) in heterogeneous media has significant potential in applications such as seismic imaging and inversion. High-contrast settings, such as subsurface models with salt bodies, exhibit strong scattering and phase sensitivity that challenge existing neural operators. We propose a hybrid architecture that decomposes the scattering operator into two separate contributions: a smooth background propagation and a high-contrast scattering correction. The smooth component is learned with a Fourier Neural Operator (FNO), which produces globally coupled feature tokens encoding background wave propagation; these tokens are then passed to a vision transformer, where attention is used to model the high-contrast scattering correction dominated by strong, spatial interactions. Evaluated on high-frequency Helmholtz problems with strong contrasts, the hybrid model achieves substantially improved phase and amplitude accuracy compared to standalone FNOs or transformers, with favorable accuracy-parameter scaling. 

---
# Position-Aware Self-supervised Representation Learning for Cross-mode Radar Signal Recognition 

**Authors**: Hongyang Zhang, Haitao Zhang, Yinhao Liu, Kunjie Lin, Yue Huang, Xinghao Ding  

**Link**: [PDF](https://arxiv.org/pdf/2602.11196)  

**Abstract**: Radar signal recognition in open electromagnetic environments is challenging due to diverse operating modes and unseen radar types. Existing methods often overlook position relations in pulse sequences, limiting their ability to capture semantic dependencies over time. We propose RadarPos, a position-aware self-supervised framework that leverages pulse-level temporal dynamics without complex augmentations or masking, providing improved position relation modeling over contrastive learning or masked reconstruction. Using this framework, we evaluate cross-mode radar signal recognition under the long-tailed setting to assess adaptability and generalization. Experimental results demonstrate enhanced discriminability and robustness, highlighting practical applicability in real-world electromagnetic environments. 

---
# MELINOE: Fine-Tuning Enables Memory-Efficient Inference for Mixture-of-Experts Models 

**Authors**: Arian Raje, Anupam Nayak, Gauri Joshi  

**Link**: [PDF](https://arxiv.org/pdf/2602.11192)  

**Abstract**: Mixture-of-Experts (MoE) model architectures can significantly reduce the number of activated parameters per token, enabling computationally efficient training and inference. However, their large overall parameter counts and model sizes have precluded their widespread usage in resource-constrained settings as all of the parameters must still be loaded into GPU memory. Prior works aim to address this memory bottleneck by offloading certain experts into CPU memory and porting them to GPU memory only when they are activated. In practice, these methods suffer from the significant I/O latency incurred by expert transfer. We present MELINOE, a method that fine-tunes an MoE model to more strongly prefer activating a smaller number of experts per sequence. Caching these preferred experts in GPU memory reduces expert churn and CPU-GPU transfer overhead. MELINOE increases throughput by $1.2-3\times$ over efficient baselines and up to $14.7\times$ over transfer-heavy baselines while retaining or even improving the performance of the model on a downstream task, making it a reliable method for improving MoE inference efficiency. 

---
# Time-TK: A Multi-Offset Temporal Interaction Framework Combining Transformer and Kolmogorov-Arnold Networks for Time Series Forecasting 

**Authors**: Fan Zhang, Shiming Fan, Hua Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.11190)  

**Abstract**: Time series forecasting is crucial for the World Wide Web and represents a core technical challenge in ensuring the stable and efficient operation of modern web services, such as intelligent transportation and website throughput. However, we have found that existing methods typically employ a strategy of embedding each time step as an independent token. This paradigm introduces a fundamental information bottleneck when processing long sequences, the root cause of which is that independent token embedding destroys a crucial structure within the sequence - what we term as multi-offset temporal correlation. This refers to the fine-grained dependencies embedded within the sequence that span across different time steps, which is especially prevalent in regular Web data. To fundamentally address this issue, we propose a new perspective on time series embedding. We provide an upper bound on the approximate reconstruction performance of token embedding, which guides our design of a concise yet effective Multi-Offset Time Embedding method to mitigate the performance degradation caused by standard token embedding. Furthermore, our MOTE can be integrated into various existing models and serve as a universal building block. Based on this paradigm, we further design a novel forecasting architecture named Time-TK. This architecture first utilizes a Multi-Offset Interactive KAN to learn and represent specific temporal patterns among multiple offset sub-sequences. Subsequently, it employs an efficient Multi-Offset Temporal Interaction mechanism to effectively capture the complex dependencies between these sub-sequences, achieving global information integration. Extensive experiments on 14 real-world benchmark datasets, covering domains such as traffic flow and BTC/USDT throughput, demonstrate that Time-TK significantly outperforms all baseline models, achieving state-of-the-art forecasting accuracy. 

---
# MuCO: Generative Peptide Cyclization Empowered by Multi-stage Conformation Optimization 

**Authors**: Yitian Wang, Fanmeng Wang, Angxiao Yue, Wentao Guo, Yaning Cui, Hongteng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2602.11189)  

**Abstract**: Modeling peptide cyclization is critical for the virtual screening of candidate peptides with desirable physical and pharmaceutical properties. This task is challenging because a cyclic peptide often exhibits diverse, ring-shaped conformations, which cannot be well captured by deterministic prediction models derived from linear peptide folding. In this study, we propose MuCO (Multi-stage Conformation Optimization), a generative peptide cyclization method that models the distribution of cyclic peptide conformations conditioned on the corresponding linear peptide. In principle, MuCO decouples the peptide cyclization task into three stages: topology-aware backbone design, generative side-chain packing, and physics-aware all-atom optimization, thereby generating and optimizing conformations of cyclic peptides in a coarse-to-fine manner. This multi-stage framework enables an efficient parallel sampling strategy for conformation generation and allows for rapid exploration of diverse, low-energy conformations. Experiments on the large-scale CPSea dataset demonstrate that MuCO significantly outperforms state-of-the-art methods in consistently in physical stability, structural diversity, secondary structure recovery, and computational efficiency, making it a promising computational tool for exploring and designing cyclic peptides. 

---
# TDPNavigator-Placer: Thermal- and Wirelength-Aware Chiplet Placement in 2.5D Systems Through Multi-Agent Reinforcement Learning 

**Authors**: Yubo Hou, Furen Zhuang, Partha Pratim Kundu, Sezin Ata Kircali, Jie Wang, Mihai Dragos Rotaru, Dutta Rahul, Ashish James  

**Link**: [PDF](https://arxiv.org/pdf/2602.11187)  

**Abstract**: The rapid growth of electronics has accelerated the adoption of 2.5D integrated circuits, where effective automated chiplet placement is essential as systems scale to larger and more heterogeneous chiplet assemblies. Existing placement methods typically focus on minimizing wirelength or transforming multi-objective optimization into a single objective through weighted sum, which limits their ability to handle competing design requirements. Wirelength reduction and thermal management are inherently conflicting objectives, making prior approaches inadequate for practical deployment. To address this challenge, we propose TDPNavigator-Placer, a novel multi-agent reinforcement learning framework that dynamically optimizes placement based on chiplet's thermal design power (TDP). This approach explicitly assigns these inherently conflicting objectives to specialized agents, each operating under distinct reward mechanisms and environmental constraints within a unified placement paradigm. Experimental results demonstrate that TDPNavigator-Placer delivers a significantly improved Pareto front over state-of-the-art methods, enabling more balanced trade-offs between wirelength and thermal performance. 

---
# Spectra: Rethinking Optimizers for LLMs Under Spectral Anisotropy 

**Authors**: Zhendong Huang, Hengjie Cao, Fang Dong, Ruijun Huang, Mengyi Chen, Yifeng Yang, Xin Zhang, Anrui Chen, Mingzhi Dong, Yujiang Wang, Jinlong Hou, Qin Lv, Robert P. Dick, Yuan Cheng, Fan Yang, Tun Lu, Li Shang  

**Link**: [PDF](https://arxiv.org/pdf/2602.11185)  

**Abstract**: Gradient signals in LLM training are highly anisotropic: recurrent linguistic structure concentrates energy into a small set of dominant spectral directions, while context specific information resides in a long tail. We show that this spike tail separation persists throughout training, with the spike occupying only about 1.5% of directions yet dominating optimizer statistics. This dominance suppresses tail learning by contracting tail updates through second moment normalization and tightening the globally stable learning rate bound. Motivated by this analysis, we propose Spectra, a spike aware optimizer that suppresses the dominant low rank spike subspace without amplifying the noise sensitive spectral tail. Spectra tracks the spike subspace via cached, warm started power iteration and applies low rank spectral shaping with negligible overhead and substantially reduced optimizer state memory. On LLaMA3 8B trained on 50B tokens, Spectra reaches the same target loss 30% faster than AdamW, reduces per step end to end overhead by 0.7%, cuts optimizer state memory by 49.25%, and improves average downstream accuracy by 1.62%. Compared to Muon, Spectra is 5.1x faster in optimizer processing time, achieves a lower final loss, and improves average accuracy by 0.66%. 

---
# KBVQ-MoE: KLT-guided SVD with Bias-Corrected Vector Quantization for MoE Large Language Models 

**Authors**: Zukang Xu, Zhixiong Zhao, Xing Hu, Zhixuan Chen, Dawei Yang  

**Link**: [PDF](https://arxiv.org/pdf/2602.11184)  

**Abstract**: Mixture of Experts (MoE) models have achieved great success by significantly improving performance while maintaining computational efficiency through sparse expert activation. However, their enormous parameter sizes and memory demands pose major challenges for deployment in resource-constrained environments. Vector Quantization (VQ) offers a promising approach for ultra-low-bit compression in Large Language Models (LLMs) by leveraging a codebook, where weight vectors are mapped to the most similar discrete codewords. Yet, directly applying VQ to MoEs often leads to substantial performance degradation due to two critical obstacles: (1) redundant representations among experts cause VQ to repeatedly quantize similar representations for each expert, resulting in inefficient use of limited codebook capacity; and (2) cumulative output bias is amplified by expert aggregation in MoE layers, leading to distributional shifts in the quantized outputs. To address these issues, we propose KBVQ-MoE, a novel VQ framework to enhance extremely low-bit quantization for MoE-based LLMs. KBVQ-MoE integrates two techniques: (1) input-driven redundancy elimination, where a Karhunen-Loeve Transform (KLT) guided singular value decomposition (SVD) extracts dominant weight components and shares them across experts; and (2) bias-corrected output stabilization, where vector quantization is applied only to expert-specific (non-redundant) representations and the quantized outputs are corrected via channel-wise affine compensation. Experiments on various MoE LLMs demonstrate that KBVQ-MoE preserves accuracy substantially better than existing quantization methods. For example, 3-bit quantization of Qwen1.5-MoE-A2.7B achieves an average accuracy of 67.99, nearly identical to the FP16 baseline of 68.07, underscoring KBVQ-MoE's potential for efficient deployment on edge devices and other resource-constrained platforms. 

---
# From Instruction to Output: The Role of Prompting in Modern NLG 

**Authors**: Munazza Zaib, Elaf Alhazmi  

**Link**: [PDF](https://arxiv.org/pdf/2602.11179)  

**Abstract**: Prompt engineering has emerged as an integral technique for extending the strengths and abilities of Large Language Models (LLMs) to gain significant performance gains in various Natural Language Processing (NLP) tasks. This approach, which requires instructions to be composed in natural language to bring out the knowledge from LLMs in a structured way, has driven breakthroughs in various NLP tasks. Yet there is still no structured framework or coherent understanding of the varied prompt engineering methods and techniques, particularly in the field of Natural Language Generation (NLG).
This survey aims to help fill that gap by outlining recent developments in prompt engineering, and their effect on different NLG tasks. It reviews recent advances in prompting methods and their impact on NLG tasks, presenting prompt design as an input-level control mechanism that complements fine-tuning and decoding approaches. The paper introduces a taxonomy of prompting paradigms, a decision framework for prompt selection based on varying factors for the practitioners, outlines emerging trends and challenges, and proposes a framework that links design, optimization, and evaluation to support more controllable and generalizable NLG. 

---
# What Do LLMs Know About Alzheimer's Disease? Fine-Tuning, Probing, and Data Synthesis for AD Detection 

**Authors**: Lei Jiang, Yue Zhou, Natalie Parde  

**Link**: [PDF](https://arxiv.org/pdf/2602.11177)  

**Abstract**: Reliable early detection of Alzheimer's disease (AD) is challenging, particularly due to limited availability of labeled data. While large language models (LLMs) have shown strong transfer capabilities across domains, adapting them to the AD domain through supervised fine-tuning remains largely unexplored. In this work, we fine-tune an LLM for AD detection and investigate how task-relevant information is encoded within its internal representations. We employ probing techniques to analyze intermediate activations across transformer layers, and we observe that, after fine-tuning, the probing values of specific words and special markers change substantially, indicating that these elements assume a crucial role in the model's improved detection performance. Guided by this insight, we design a curated set of task-aware special markers and train a sequence-to-sequence model as a data-synthesis tool that leverages these markers to generate structurally consistent and diagnostically informative synthetic samples. We evaluate the synthesized data both intrinsically and by incorporating it into downstream training pipelines. 

---
# Evaluating Few-Shot Temporal Reasoning of LLMs for Human Activity Prediction in Smart Environments 

**Authors**: Maral Doctorarastoo, Katherine A. Flanigan, Mario Bergés, Christopher McComb  

**Link**: [PDF](https://arxiv.org/pdf/2602.11176)  

**Abstract**: Anticipating human activities and their durations is essential in applications such as smart-home automation, simulation-based architectural and urban design, activity-based transportation system simulation, and human-robot collaboration, where adaptive systems must respond to human activities. Existing data-driven agent-based models--from rule-based to deep learning--struggle in low-data environments, limiting their practicality. This paper investigates whether large language models, pre-trained on broad human knowledge, can fill this gap by reasoning about everyday activities from compact contextual cues. We adopt a retrieval-augmented prompting strategy that integrates four sources of context--temporal, spatial, behavioral history, and persona--and evaluate it on the CASAS Aruba smart-home dataset. The evaluation spans two complementary tasks: next-activity prediction with duration estimation, and multi-step daily sequence generation, each tested with various numbers of few-shot examples provided in the prompt. Analyzing few-shot effects reveals how much contextual supervision is sufficient to balance data efficiency and predictive accuracy, particularly in low-data environments. Results show that large language models exhibit strong inherent temporal understanding of human behavior: even in zero-shot settings, they produce coherent daily activity predictions, while adding one or two demonstrations further refines duration calibration and categorical accuracy. Beyond a few examples, performance saturates, indicating diminishing returns. Sequence-level evaluation confirms consistent temporal alignment across few-shot conditions. These findings suggest that pre-trained language models can serve as promising temporal reasoners, capturing both recurring routines and context-dependent behavioral variations, thereby strengthening the behavioral modules of agent-based models. 

---
# The Script Tax: Measuring Tokenization-Driven Efficiency and Latency Disparities in Multilingual Language Models 

**Authors**: Aradhya Dixit, Shreem Dixit  

**Link**: [PDF](https://arxiv.org/pdf/2602.11174)  

**Abstract**: Pretrained multilingual language models are often assumed to be script-agnostic, yet their tokenizers can impose systematic costs on certain writing systems. We quantify this script tax by comparing two orthographic variants with identical linguistic content. Across mBERT and XLM-R, the higher-fragmentation orthography shows a ~3.4x increase in fertility (6.73-6.85 vs. 2.10-2.35 tokens/word), leading to a 16.5x inference slowdown (0.23 vs. 3.8 sentences/second) on identical hardware. Using bits per character (BPC) to avoid the "NLL paradox" from subword fragmentation, we find a substantial increase in information cost: +19.7% for mBERT (8.06->9.65) and +47.1% for XLM-R (12.19->17.94). A round-trip conversion check (CER_rt=0.31) suggests these gaps reflect orthography-conditioned processing rather than mapping noise. Our results highlight tokenization as a key source of inequity in multilingual NLP and motivate script-aware tokenization and pretraining. 

---
# Efficient Hyper-Parameter Search for LoRA via Language-aided Bayesian Optimization 

**Authors**: Baek Seong-Eun, Lee Jung-Mok, Kim Sung-Bin, Tae-Hyun Oh  

**Link**: [PDF](https://arxiv.org/pdf/2602.11171)  

**Abstract**: Fine-tuning Large Language Models (LLMs) with Low-Rank Adaptation (LoRA) enables resource-efficient personalization or specialization, but it comes at the expense of additional hyperparameter tuning. Although LoRA makes fine-tuning efficient, it is highly sensitive to the choice of hyperparameters, and exhaustive hyperparameter search is still computationally very demanding. To address these challenges, we propose a framework that integrates the domain knowledge of pre-trained LLMs into Bayesian Optimization (BO) to efficiently search for LoRA hyperparameters. To leverage the informed knowledge of LLMs, we repurpose LLMs as a discrete-to-continuous mapping to link the hyperparameters and their domain knowledge with a continuous vector space, where BO is conducted. We design and control the mapping by language prompting, where we provide a domain-aware textual prompt describing the relationships among hyperparameters and their respective roles; thereby, we explicitly inject domain knowledge about LoRA into the LLM in natural language. Also, we model the residual information that is hard to linguistically describe in the prompt with an additional learnable token. This aids BO to sample more high-performing hyperparameters. In addition, by leveraging the observation of the strong correlation between the respective performance obtained from full and subset training datasets in LoRA training regimes, we introduce proxy training and evaluation with a data subset. This further increases the efficiency of our method. We demonstrate that our hyperparameter found with only about 30 iterations achieves more than 20% performance improvement over standard hyperparameters found from about 45,000 combinations. 

---
# Disentangling Direction and Magnitude in Transformer Representations: A Double Dissociation Through L2-Matched Perturbation Analysis 

**Authors**: Mangadoddi Srikar Vardhan, Lekkala Sai Teja  

**Link**: [PDF](https://arxiv.org/pdf/2602.11169)  

**Abstract**: Transformer hidden states encode information as high-dimensional vectors, yet whether direction (orientation in representational space) and magnitude (vector norm) serve distinct functional roles remains unclear. Studying Pythia-family models, we discover a striking cross-over dissociation: angular perturbations cause up to 42.9 more damage to language modeling loss, while magnitude perturbations cause disproportionately more damage to syntactic processing (20.4% vs.1.6% accuracy drop on subject-verb agreement).This finding is enabled by L2-matched perturbation analysis, a methodology ensuring that an gular and magnitude perturbations achieve identical Euclidean displacements. Causal intervention reveals that angular damage flows substantially through the attention pathways (28.4% loss recovery via attention repair), while magnitude damage flows partly through the LayerNorm pathways(29.9% recovery via LayerNorm repair). These patterns replicate across scales within the Pythia architecture family. These findings provide evidence that direction and magnitude support partially distinct computational roles in LayerNorm based architectures. The direction preferentially affects attentional routing, while magnitude modulates processing intensity for fine-grained syntactic judgments. We find different patterns in RMSNorm-based architectures, suggesting that the dissociation depends on architectural choices. Our results refine the linear representation hypothesis and have implications for model editing and interpretability research 

---
# Enhancing SDG-Text Classification with Combinatorial Fusion Analysis and Generative AI 

**Authors**: Jingyan Xu, Marcelo L. LaFleur, Christina Schweikert, D. Frank Hsu  

**Link**: [PDF](https://arxiv.org/pdf/2602.11168)  

**Abstract**: (Natural Language Processing) NLP techniques such as text classification and topic discovery are very useful in many application areas including information retrieval, knowledge discovery, policy formulation, and decision-making. However, it remains a challenging problem in cases where the categories are unavailable, difficult to differentiate, or are interrelated. Social analysis with human context is an area that can benefit from text classification, as it relies substantially on text data. The focus of this paper is to enhance the classification of text according to the UN's Sustainable Development Goals (SDGs) by collecting and combining intelligence from multiple models. Combinatorial Fusion Analysis (CFA), a system fusion paradigm using a rank-score characteristic (RSC) function and cognitive diversity (CD), has been used to enhance classifier methods by combining a set of relatively good and mutually diverse classification models. We use a generative AI model to generate synthetic data for model training and then apply CFA to this classification task. The CFA technique achieves 96.73% performance, outperforming the best individual model. We compare the outcomes with those obtained from human domain experts. It is demonstrated that combining intelligence from multiple ML/AI models using CFA and getting input from human experts can, not only complement, but also enhance each other. 

---
# Visualizing and Benchmarking LLM Factual Hallucination Tendencies via Internal State Analysis and Clustering 

**Authors**: Nathan Mao, Varun Kaushik, Shreya Shivkumar, Parham Sharafoleslami, Kevin Zhu, Sunishchal Dev  

**Link**: [PDF](https://arxiv.org/pdf/2602.11167)  

**Abstract**: Large Language Models (LLMs) often hallucinate, generating nonsensical or false information that can be especially harmful in sensitive fields such as medicine or law. To study this phenomenon systematically, we introduce FalseCite, a curated dataset designed to capture and benchmark hallucinated responses induced by misleading or fabricated citations. Running GPT-4o-mini, Falcon-7B, and Mistral 7-B through FalseCite, we observed a noticeable increase in hallucination activity for false claims with deceptive citations, especially in GPT-4o-mini. Using the responses from FalseCite, we can also analyze the internal states of hallucinating models, visualizing and clustering the hidden state vectors. From this analysis, we noticed that the hidden state vectors, regardless of hallucination or non-hallucination, tend to trace out a distinct horn-like shape. Our work underscores FalseCite's potential as a foundation for evaluating and mitigating hallucinations in future LLM research. 

---
# Small Updates, Big Doubts: Does Parameter-Efficient Fine-tuning Enhance Hallucination Detection ? 

**Authors**: Xu Hu, Yifan Zhang, Songtao Wei, Chen Zhao, Qiannan Li, Bingzhe Li, Feng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2602.11166)  

**Abstract**: Parameter-efficient fine-tuning (PEFT) methods are widely used to adapt large language models (LLMs) to downstream tasks and are often assumed to improve factual correctness. However, how the parameter-efficient fine-tuning methods affect hallucination behavior remains insufficiently understood, especially on QA datasets. In this work, we systematically investigate the impact of PEFT on hallucination detection through a comprehensive empirical study across three open-weight LLM backbones and three fact-seeking QA benchmarks. For each model, we evaluate performance using seven unsupervised hallucination detection methods spanning three complementary approaches: semantic consistency based detectors, confidence based detectors, and entropy based detectors. This multifaceted evaluation enables us to characterize how PEFT reshapes uncertainty across different detection paradigms. In conclusion, our experimental results show that PEFT consistently strengthens hallucination detection ability, substantially improving AUROC across a wide range of hallucination detectors. Besides, further analyses using linear probes and representation diagnostics indicate that PEFT methods primarily reshapes how uncertainty is encoded and surfaced, comparing with injecting new factual knowledge into the models. 

---
# Assessing LLM Reliability on Temporally Recent Open-Domain Questions 

**Authors**: Pushwitha Krishnappa, Amit Das, Vinija Jain, Tathagata Mukherjee, Aman Chadha  

**Link**: [PDF](https://arxiv.org/pdf/2602.11165)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed for open-domain question answering, yet their alignment with human perspectives on temporally recent information remains underexplored. We introduce RECOM (Reddit Evaluation for Correspondence of Models), a benchmark dataset of 15,000 recent Reddit questions from September 2025 paired with community-derived reference answers. We investigate how four open-source LLMs (Llama3.1-8B, Mistral-7B, Gemma-2-9B, and GPT-OSS-20B) respond to these questions, evaluating alignment using lexical metrics (BLEU, ROUGE), semantic similarity (BERTScore, MoverScore, cosine similarity), and logical inference (NLI). Our central finding is a striking semantic-lexical paradox: all models achieve over 99% cosine similarity with references despite less than 8% BLEU-1 overlap, a 90+ percentage point gap indicating that models preserve meaning through extensive paraphrasing rather than lexical reproduction. MoverScore (51-53%) confirms this pattern, occupying an intermediate position that reflects the optimal transport cost of semantic alignment. Furthermore, model scale does not predict performance: Mistral-7B (7B parameters) outperforms GPT-OSS-20B (20B parameters) across all metrics. NLI analysis reveals that contradiction rates remain below 7%, suggesting models rarely generate content that directly conflicts with human consensus. These findings challenge the reliability of lexical metrics for evaluating abstractive generation and argue for multi-dimensional evaluation frameworks that capture semantic fidelity beyond surface-level text matching. The RECOM dataset is publicly available at this https URL 

---
# Automated Optimization Modeling via a Localizable Error-Driven Perspective 

**Authors**: Weiting Liu, Han Wu, Yufei Kuang, Xiongwei Han, Tao Zhong, Jianfeng Feng, Wenlian Lu  

**Link**: [PDF](https://arxiv.org/pdf/2602.11164)  

**Abstract**: Automated optimization modeling via Large Language Models (LLMs) has emerged as a promising approach to assist complex human decision-making. While post-training has become a pivotal technique to enhance LLMs' capabilities in this domain, its effectiveness is severely constrained by the scarcity and underutilization of high-quality training data. However, through a detailed profiling of error patterns across various problem-response pairs drawn from post-training, we identify two fundamental limitations of existing automated optimization modeling approaches: (L1) the sparsity of error-specific problems and (L2) the sparse rewards associated with difficult problems. We demonstrate that these limitations can result in suboptimal performance in domain-specific post-training for LLMs. To tackle the above two limitations, we propose a novel error-driven learning framework -- namely, auto\textbf{m}ated opt\textbf{i}mization modeli\textbf{n}g via a localizable error-\textbf{d}riven perspective (MIND) -- that customizes the whole model training framework from data synthesis to post-training. MIND is based on our key observation of the unique localizable patterns in error propagation of optimization modelings, that is, modeling errors may remain localized to specific semantic segments and do not propagate throughout the entire solution. Thus, in contrast to holistic reasoning tasks such as mathematical proofs, MIND leverages the construction of a focused, high-density training corpus and proposes \textbf{D}ynamic Supervised \textbf{F}ine-Tuning \textbf{P}olicy \textbf{O}ptimization (DFPO) to tackle difficult problems through localized refinement. Experiments on six benchmarks demonstrate that MIND consistently outperforms all the state-of-the-art automated optimization modeling approaches. 

---
# Nested Named Entity Recognition in Plasma Physics Research Articles 

**Authors**: Muhammad Haris, Hans Höft, Markus M. Becker, Markus Stocker  

**Link**: [PDF](https://arxiv.org/pdf/2602.11163)  

**Abstract**: Named Entity Recognition (NER) is an important task in natural language processing that aims to identify and extract key entities from unstructured text. We present a novel application of NER in plasma physics research articles and address the challenges of extracting specialized entities from scientific text in this domain. Research articles in plasma physics often contain highly complex and context-rich content that must be extracted to enable, e.g., advanced search. We propose a lightweight approach based on encoder-transformers and conditional random fields to extract (nested) named entities from plasma physics research articles. First, we annotate a plasma physics corpus with 16 classes specifically designed for the nested NER task. Second, we evaluate an entity-specific model specialization approach, where independent BERT-CRF models are trained to recognize individual entity types in plasma physics text. Third, we integrate an optimization process to systematically fine-tune hyperparameters and enhance model performance. Our work contributes to the advancement of entity recognition in plasma physics and also provides a foundation to support researchers in navigating and analyzing scientific literature. 

---
# BIRD: A Museum Open Dataset Combining Behavior Patterns and Identity Types to Better Model Visitors' Experience 

**Authors**: Alexanne Worm, Florian Marchal, Sylvain Castagnos  

**Link**: [PDF](https://arxiv.org/pdf/2602.11160)  

**Abstract**: Lack of data is a recurring problem in Artificial Intelligence, as it is essential for training and validating models. This is particularly true in the field of cultural heritage, where the number of open datasets is relatively limited and where the data collected does not always allow for holistic modeling of visitors' experience due to the fact that data are ad hoc (i.e. restricted to the sole characteristics required for the evaluation of a specific model). To overcome this lack, we conducted a study between February and March 2019 aimed at obtaining comprehensive and detailed information about visitors, their visit experience and their feedback. We equipped 51 participants with eye-tracking glasses, leaving them free to explore the 3 floors of the museum for an average of 57 minutes, and to discover an exhibition of more than 400 artworks. On this basis, we built an open dataset combining contextual data (demographic data, preferences, visiting habits, motivations, social context. . . ), behavioral data (spatiotemporal trajectories, gaze data) and feedback (satisfaction, fatigue, liked artworks, verbatim. . . ). Our analysis made it possible to re-enact visitor identities combining the majority of characteristics found in the literature and to reproduce the Veron and Levasseur profiles. This dataset will ultimately make it possible to improve the quality of recommended paths in museums by personalizing the number of points of interest (POIs), the time spent at these different POIs, and the amount of information to be provided to each visitor based on their level of interest. 

---
# Methodological Variation in Studying Staff and Student Perceptions of AI 

**Authors**: Juliana Gerard, Morgan Macleod, Kelly Norwood, Aisling Reid, Muskaan Singh  

**Link**: [PDF](https://arxiv.org/pdf/2602.11158)  

**Abstract**: In this paper, we compare methodological approaches for comparing student and staff perceptions, and ask: how much do these measures vary across different approaches? We focus on the case of AI perceptions, which are generally assessed via a single quantitative or qualitative measure, or with a mixed methods approach that compares two distinct data sources - e.g. a quantitative questionnaire with qualitative comments. To compare different approaches, we collect two forms of qualitative data: standalone comments and structured focus groups. We conduct two analyses for each data source: with a sentiment and stance analysis, we measure overall negativity/positivity of the comments and focus group conversations, respectively. Meanwhile, word clouds from the comments and a thematic analysis of the focus groups provide further detail on the content of this qualitative data - particularly the thematic analysis, which includes both similarities and differences between students and staff. We show that different analyses can produce different results - for a single data source. This variation stems from the construct being evaluated - an overall measure of positivity/negativity can produce a different picture from more detailed content-based analyses. We discuss the implications of this variation for institutional contexts, and for the comparisons from previous studies. 

---
# HybridRAG: A Practical LLM-based ChatBot Framework based on Pre-Generated Q&A over Raw Unstructured Documents 

**Authors**: Sungmoon Kim, Hyuna Jeon, Dahye Kim, Mingyu Kim, Dong-Kyu Chae, Jiwoong Kim  

**Link**: [PDF](https://arxiv.org/pdf/2602.11156)  

**Abstract**: Retrieval-Augmented Generation (RAG) has emerged as a powerful approach for grounding Large Language Model (LLM)-based chatbot responses on external knowledge. However, existing RAG studies typically assume well-structured textual sources (e.g. Wikipedia or curated datasets) and perform retrieval and generation at query time, which can limit their applicability in real-world chatbot scenarios. In this paper, we present HybridRAG, a novel and practical RAG framework towards more accurate and faster chatbot responses. First, HybridRAG ingests raw, unstructured PDF documents containing complex layouts (text, tables, figures) via Optical Character Recognition (OCR) and layout analysis, and convert them into hierarchical text chunks. Then, it pre-generates a plausible question-answer (QA) knowledge base from the organized chunks using an LLM. At query time, user questions are matched against this QA bank to retrieve immediate answers when possible, and only if no suitable QA match is found does our framework fall back to an on-the-fly response generation. Experiments on OHRBench demonstrate that our HybridRAG provides higher answer quality and lower latency compared to a standard RAG baseline. We believe that HybridRAG could be a practical solution for real-world chatbot applications that must handle large volumes of unstructured documents and lots of users under limited computational resources. 

---
# Improving Medical Visual Reinforcement Fine-Tuning via Perception and Reasoning Augmentation 

**Authors**: Guangjing Yang, ZhangYuan Yu, Ziyuan Qin, Xinyuan Song, Huahui Yi, Qingbo Kang, Jun Gao, Yiyue Li, Chenlin Du, Qicheng Lao  

**Link**: [PDF](https://arxiv.org/pdf/2602.10619)  

**Abstract**: While recent advances in Reinforcement Fine-Tuning (RFT) have shown that rule-based reward schemes can enable effective post-training for large language models, their extension to cross-modal, vision-centric domains remains largely underexplored. This limitation is especially pronounced in the medical imaging domain, where effective performance requires both robust visual perception and structured reasoning. In this work, we address this gap by proposing VRFT-Aug, a visual reinforcement fine-tuning framework tailored for the medical domain. VRFT-Aug introduces a series of training strategies designed to augment both perception and reasoning, including prior knowledge injection, perception-driven policy refinement, medically informed reward shaping, and behavioral imitation. Together, these methods aim to stabilize and improve the RFT process.
Through extensive experiments across multiple medical datasets, we show that our approaches consistently outperform both standard supervised fine-tuning and RFT baselines. Moreover, we provide empirically grounded insights and practical training heuristics that can be generalized to other medical image tasks. We hope this work contributes actionable guidance and fresh inspiration for the ongoing effort to develop reliable, reasoning-capable models for high-stakes medical applications. 

---
