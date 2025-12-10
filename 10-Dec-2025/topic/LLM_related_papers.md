# Principles2Plan: LLM-Guided System for Operationalising Ethical Principles into Plans 

**Authors**: Tammy Zhong, Yang Song, Maurice Pagnucco  

**Link**: [PDF](https://arxiv.org/pdf/2512.08536)  

**Abstract**: Ethical awareness is critical for robots operating in human environments, yet existing automated planning tools provide little support. Manually specifying ethical rules is labour-intensive and highly context-specific. We present Principles2Plan, an interactive research prototype demonstrating how a human and a Large Language Model (LLM) can collaborate to produce context-sensitive ethical rules and guide automated planning. A domain expert provides the planning domain, problem details, and relevant high-level principles such as beneficence and privacy. The system generates operationalisable ethical rules consistent with these principles, which the user can review, prioritise, and supply to a planner to produce ethically-informed plans. To our knowledge, no prior system supports users in generating principle-grounded rules for classical planning contexts. Principles2Plan showcases the potential of human-LLM collaboration for making ethical automated planning more practical and feasible. 

---
# CogMCTS: A Novel Cognitive-Guided Monte Carlo Tree Search Framework for Iterative Heuristic Evolution with Large Language Models 

**Authors**: Hui Wang, Yang Liu, Xiaoyu Zhang, Chaoxu Mu  

**Link**: [PDF](https://arxiv.org/pdf/2512.08609)  

**Abstract**: Automatic Heuristic Design (AHD) is an effective1 framework for solving complex optimization prob-2 lems. The development of large language mod-3 els (LLMs) enables the automated generation of4 heuristics. Existing LLM-based evolutionary meth-5 ods rely on population strategies and are prone6 to local optima. Integrating LLMs with Monte7 Carlo Tree Search (MCTS) improves the trade-off8 between exploration and exploitation, but multi-9 round cognitive integration remains limited and10 search diversity is constrained. To overcome these11 limitations, this paper proposes a novel cognitive-12 guided MCTS framework (CogMCTS). CogMCTS13 tightly integrates the cognitive guidance mecha-14 nism of LLMs with MCTS to achieve efficient au-15 tomated heuristic optimization. The framework16 employs multi-round cognitive feedback to incor-17 porate historical experience, node information, and18 negative outcomes, dynamically improving heuris-19 tic generation. Dual-track node expansion com-20 bined with elite heuristic management balances the21 exploration of diverse heuristics and the exploita-22 tion of high-quality experience. In addition, strate-23 gic mutation modifies the heuristic forms and pa-24 rameters to further enhance the diversity of the so-25 lution and the overall optimization performance.26 The experimental results indicate that CogMCTS27 outperforms existing LLM-based AHD methods in28 stability, efficiency, and solution quality. 

---
# Reflecting with Two Voices: A Co-Adaptive Dual-Strategy Framework for LLM-Based Agent Decision Making 

**Authors**: Wentao Zhang, Qunbo Wang, Tao Zhang, Junsheng Wu, Hongping Gan, Yang Liu, Ling Dai, Shizhuang Deng, Shuntong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2512.08366)  

**Abstract**: Large language model (LLM) agents often rely on external demonstrations or retrieval-augmented planning, leading to brittleness, poor generalization, and high computational overhead. Inspired by human problem-solving, we propose DuSAR (Dual-Strategy Agent with Reflecting) - a demonstration-free framework that enables a single frozen LLM to perform co-adaptive reasoning via two complementary strategies: a high-level holistic plan and a context-grounded local policy. These strategies interact through a lightweight reflection mechanism, where the agent continuously assesses progress via a Strategy Fitness Score and dynamically revises its global plan when stuck or refines it upon meaningful advancement, mimicking human metacognitive behavior. On ALFWorld and Mind2Web, DuSAR achieves state-of-the-art performance with open-source LLMs (7B-70B), reaching 37.1% success on ALFWorld (Llama3.1-70B) - more than doubling the best prior result (13.0%) - and 4.02% on Mind2Web, also more than doubling the strongest baseline. Remarkably, it reduces per-step token consumption by 3-9X while maintaining strong performance. Ablation studies confirm the necessity of dual-strategy coordination. Moreover, optional integration of expert demonstrations further boosts results, highlighting DuSAR's flexibility and compatibility with external knowledge. 

---
# The High Cost of Incivility: Quantifying Interaction Inefficiency via Multi-Agent Monte Carlo Simulations 

**Authors**: Benedikt Mangold  

**Link**: [PDF](https://arxiv.org/pdf/2512.08345)  

**Abstract**: Workplace toxicity is widely recognized as detrimental to organizational culture, yet quantifying its direct impact on operational efficiency remains methodologically challenging due to the ethical and practical difficulties of reproducing conflict in human subjects. This study leverages Large Language Model (LLM) based Multi-Agent Systems to simulate 1-on-1 adversarial debates, creating a controlled "sociological sandbox". We employ a Monte Carlo method to simulate hundrets of discussions, measuring the convergence time (defined as the number of arguments required to reach a conclusion) between a baseline control group and treatment groups involving agents with "toxic" system prompts. Our results demonstrate a statistically significant increase of approximately 25\% in the duration of conversations involving toxic participants. We propose that this "latency of toxicity" serves as a proxy for financial damage in corporate and academic settings. Furthermore, we demonstrate that agent-based modeling provides a reproducible, ethical alternative to human-subject research for measuring the mechanics of social friction. 

---
# rSIM: Incentivizing Reasoning Capabilities of LLMs via Reinforced Strategy Injection 

**Authors**: Sijia Chen, Baochun Li, Di Niu  

**Link**: [PDF](https://arxiv.org/pdf/2512.08300)  

**Abstract**: Large language models (LLMs) are post-trained through reinforcement learning (RL) to evolve into Reasoning Language Models (RLMs), where the hallmark of this advanced reasoning is ``aha'' moments when they start to perform strategies, such as self-reflection and deep thinking, within chain of thoughts (CoTs). Motivated by this, this paper proposes a novel reinforced strategy injection mechanism (rSIM), that enables any LLM to become an RLM by employing a small planner to guide the LLM's CoT through the adaptive injection of reasoning strategies. To achieve this, the planner (leader agent) is jointly trained with an LLM (follower agent) using multi-agent RL (MARL), based on a leader-follower framework and straightforward rule-based rewards. Experimental results show that rSIM enables Qwen2.5-0.5B to become an RLM and significantly outperform Qwen2.5-14B. Moreover, the planner is generalizable: it only needs to be trained once and can be applied as a plug-in to substantially improve the reasoning capabilities of existing LLMs. In addition, the planner supports continual learning across various tasks, allowing its planning abilities to gradually improve and generalize to a wider range of problems. 

---
# Reasoning Models Ace the CFA Exams 

**Authors**: Jaisal Patel, Yunzhe Chen, Kaiwen He, Keyi Wang, David Li, Kairong Xiao, Xiao-Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2512.08270)  

**Abstract**: Previous research has reported that large language models (LLMs) demonstrate poor performance on the Chartered Financial Analyst (CFA) exams. However, recent reasoning models have achieved strong results on graduate-level academic and professional examinations across various disciplines. In this paper, we evaluate state-of-the-art reasoning models on a set of mock CFA exams consisting of 980 questions across three Level I exams, two Level II exams, and three Level III exams. Using the same pass/fail criteria from prior studies, we find that most models clear all three levels. The models that pass, ordered by overall performance, are Gemini 3.0 Pro, Gemini 2.5 Pro, GPT-5, Grok 4, Claude Opus 4.1, and DeepSeek-V3.1. Specifically, Gemini 3.0 Pro achieves a record score of 97.6% on Level I. Performance is also strong on Level II, led by GPT-5 at 94.3%. On Level III, Gemini 2.5 Pro attains the highest score with 86.4% on multiple-choice questions while Gemini 3.0 Pro achieves 92.0% on constructed-response questions. 

---
# Toward an AI Reasoning-Enabled System for Patient-Clinical Trial Matching 

**Authors**: Caroline N. Leach, Mitchell A. Klusty, Samuel E. Armstrong, Justine C. Pickarski, Kristen L. Hankins, Emily B. Collier, Maya Shah, Aaron D. Mullen, V. K. Cody Bumgardner  

**Link**: [PDF](https://arxiv.org/pdf/2512.08026)  

**Abstract**: Screening patients for clinical trial eligibility remains a manual, time-consuming, and resource-intensive process. We present a secure, scalable proof-of-concept system for Artificial Intelligence (AI)-augmented patient-trial matching that addresses key implementation challenges: integrating heterogeneous electronic health record (EHR) data, facilitating expert review, and maintaining rigorous security standards. Leveraging open-source, reasoning-enabled large language models (LLMs), the system moves beyond binary classification to generate structured eligibility assessments with interpretable reasoning chains that support human-in-the-loop review. This decision support tool represents eligibility as a dynamic state rather than a fixed determination, identifying matches when available and offering actionable recommendations that could render a patient eligible in the future. The system aims to reduce coordinator burden, intelligently broaden the set of trials considered for each patient and guarantee comprehensive auditability of all AI-generated outputs. 

---
# AgentEval: Generative Agents as Reliable Proxies for Human Evaluation of AI-Generated Content 

**Authors**: Thanh Vu, Richi Nayak, Thiru Balasubramaniam  

**Link**: [PDF](https://arxiv.org/pdf/2512.08273)  

**Abstract**: Modern businesses are increasingly challenged by the time and expense required to generate and assess high-quality content. Human writers face time constraints, and extrinsic evaluations can be costly. While Large Language Models (LLMs) offer potential in content creation, concerns about the quality of AI-generated content persist. Traditional evaluation methods, like human surveys, further add operational costs, highlighting the need for efficient, automated solutions. This research introduces Generative Agents as a means to tackle these challenges. These agents can rapidly and cost-effectively evaluate AI-generated content, simulating human judgment by rating aspects such as coherence, interestingness, clarity, fairness, and relevance. By incorporating these agents, businesses can streamline content generation and ensure consistent, high-quality output while minimizing reliance on costly human evaluations. The study provides critical insights into enhancing LLMs for producing business-aligned, high-quality content, offering significant advancements in automated content generation and evaluation. 

---
# DeepFeature: Iterative Context-aware Feature Generation for Wearable Biosignals 

**Authors**: Kaiwei Liu, Yuting He, Bufang Yang, Mu Yuan, Chun Man Victor Wong, Ho Pong Andrew Sze, Zhenyu Yan, Hongkai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2512.08379)  

**Abstract**: Biosignals collected from wearable devices are widely utilized in healthcare applications. Machine learning models used in these applications often rely on features extracted from biosignals due to their effectiveness, lower data dimensionality, and wide compatibility across various model architectures. However, existing feature extraction methods often lack task-specific contextual knowledge, struggle to identify optimal feature extraction settings in high-dimensional feature space, and are prone to code generation and automation errors. In this paper, we propose DeepFeature, the first LLM-empowered, context-aware feature generation framework for wearable biosignals. DeepFeature introduces a multi-source feature generation mechanism that integrates expert knowledge with task settings. It also employs an iterative feature refinement process that uses feature assessment-based feedback for feature re-selection. Additionally, DeepFeature utilizes a robust multi-layer filtering and verification approach for robust feature-to-code translation to ensure that the extraction functions run without crashing. Experimental evaluation results show that DeepFeature achieves an average AUROC improvement of 4.21-9.67% across eight diverse tasks compared to baseline methods. It outperforms state-of-the-art approaches on five tasks while maintaining comparable performance on the remaining tasks. 

---
# Beyond Traditional Diagnostics: Transforming Patient-Side Information into Predictive Insights with Knowledge Graphs and Prototypes 

**Authors**: Yibowen Zhao, Yinan Zhang, Zhixiang Su, Lizhen Cui, Chunyan Miao  

**Link**: [PDF](https://arxiv.org/pdf/2512.08261)  

**Abstract**: Predicting diseases solely from patient-side information, such as demographics and self-reported symptoms, has attracted significant research attention due to its potential to enhance patient awareness, facilitate early healthcare engagement, and improve healthcare system efficiency. However, existing approaches encounter critical challenges, including imbalanced disease distributions and a lack of interpretability, resulting in biased or unreliable predictions. To address these issues, we propose the Knowledge graph-enhanced, Prototype-aware, and Interpretable (KPI) framework. KPI systematically integrates structured and trusted medical knowledge into a unified disease knowledge graph, constructs clinically meaningful disease prototypes, and employs contrastive learning to enhance predictive accuracy, which is particularly important for long-tailed diseases. Additionally, KPI utilizes large language models (LLMs) to generate patient-specific, medically relevant explanations, thereby improving interpretability and reliability. Extensive experiments on real-world datasets demonstrate that KPI outperforms state-of-the-art methods in predictive accuracy and provides clinically valid explanations that closely align with patient narratives, highlighting its practical value for patient-centered healthcare delivery. 

---
# No Labels, No Problem: Training Visual Reasoners with Multimodal Verifiers 

**Authors**: Damiano Marsili, Georgia Gkioxari  

**Link**: [PDF](https://arxiv.org/pdf/2512.08889)  

**Abstract**: Visual reasoning is challenging, requiring both precise object grounding and understanding complex spatial relationships. Existing methods fall into two camps: language-only chain-of-thought approaches, which demand large-scale (image, query, answer) supervision, and program-synthesis approaches which use pre-trained models and avoid training, but suffer from flawed logic and erroneous grounding. We propose an annotation-free training framework that improves both reasoning and grounding. Our framework uses AI-powered verifiers: an LLM verifier refines LLM reasoning via reinforcement learning, while a VLM verifier strengthens visual grounding through automated hard-negative mining, eliminating the need for ground truth labels. This design combines the strengths of modern AI systems: advanced language-only reasoning models for decomposing spatial queries into simpler subtasks, and strong vision specialist models improved via performant VLM critics. We evaluate our approach across diverse spatial reasoning tasks, and show that our method improves visual reasoning and surpasses open-source and proprietary models, while with our improved visual grounding model we further outperform recent text-only visual reasoning methods. Project webpage: this https URL 

---
# SkipKV: Selective Skipping of KV Generation and Storage for Efficient Inference with Large Reasoning Models 

**Authors**: Jiayi Tian, Seyedarmin Azizi, Yequan Zhao, Erfan Baghaei Potraghloo, Sean McPherson, Sharath Nittur Sridhar, Zhengyang Wang, Zheng Zhang, Massoud Pedram, Souvik Kundu  

**Link**: [PDF](https://arxiv.org/pdf/2512.07993)  

**Abstract**: Large reasoning models (LRMs) often cost significant key-value (KV) cache overhead, due to their linear growth with the verbose chain-of-thought (CoT) reasoning process. This costs both memory and throughput bottleneck limiting their efficient deployment. Towards reducing KV cache size during inference, we first investigate the effectiveness of existing KV cache eviction methods for CoT reasoning. Interestingly, we find that due to unstable token-wise scoring and the reduced effective KV budget caused by padding tokens, state-of-the-art (SoTA) eviction methods fail to maintain accuracy in the multi-batch setting. Additionally, these methods often generate longer sequences than the original model, as semantic-unaware token-wise eviction leads to repeated revalidation during reasoning. To address these issues, we present \textbf{SkipKV}, a \textbf{\textit{training-free}} KV compression method for selective \textit{eviction} and \textit{generation} operating at a coarse-grained sentence-level sequence removal for efficient CoT reasoning. In specific, it introduces a \textit{sentence-scoring metric} to identify and remove highly similar sentences while maintaining semantic coherence. To suppress redundant generation, SkipKV dynamically adjusts a steering vector to update the hidden activation states during inference enforcing the LRM to generate concise response. Extensive evaluations on multiple reasoning benchmarks demonstrate the effectiveness of SkipKV in maintaining up to $\mathbf{26.7}\%$ improved accuracy compared to the alternatives, at a similar compression budget. Additionally, compared to SoTA, SkipKV yields up to $\mathbf{1.6}\times$ fewer generation length while improving throughput up to $\mathbf{1.7}\times$. 

---
# Large Language Models for Education and Research: An Empirical and User Survey-based Analysis 

**Authors**: Md Mostafizer Rahman, Ariful Islam Shiplu, Md Faizul Ibne Amin, Yutaka Watanobe, Lu Peng  

**Link**: [PDF](https://arxiv.org/pdf/2512.08057)  

**Abstract**: Pretrained Large Language Models (LLMs) have achieved remarkable success across diverse domains, with education and research emerging as particularly impactful areas. Among current state-of-the-art LLMs, ChatGPT and DeepSeek exhibit strong capabilities in mathematics, science, medicine, literature, and programming. In this study, we present a comprehensive evaluation of these two LLMs through background technology analysis, empirical experiments, and a real-world user survey. The evaluation explores trade-offs among model accuracy, computational efficiency, and user experience in educational and research affairs. We benchmarked these LLMs performance in text generation, programming, and specialized problem-solving. Experimental results show that ChatGPT excels in general language understanding and text generation, while DeepSeek demonstrates superior performance in programming tasks due to its efficiency- focused design. Moreover, both models deliver medically accurate diagnostic outputs and effectively solve complex mathematical problems. Complementing these quantitative findings, a survey of students, educators, and researchers highlights the practical benefits and limitations of these models, offering deeper insights into their role in advancing education and research. 

---
# When Tables Leak: Attacking String Memorization in LLM-Based Tabular Data Generation 

**Authors**: Joshua Ward, Bochao Gu, Chi-Hua Wang, Guang Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2512.08875)  

**Abstract**: Large Language Models (LLMs) have recently demonstrated remarkable performance in generating high-quality tabular synthetic data. In practice, two primary approaches have emerged for adapting LLMs to tabular data generation: (i) fine-tuning smaller models directly on tabular datasets, and (ii) prompting larger models with examples provided in context. In this work, we show that popular implementations from both regimes exhibit a tendency to compromise privacy by reproducing memorized patterns of numeric digits from their training data. To systematically analyze this risk, we introduce a simple No-box Membership Inference Attack (MIA) called LevAtt that assumes adversarial access to only the generated synthetic data and targets the string sequences of numeric digits in synthetic observations. Using this approach, our attack exposes substantial privacy leakage across a wide range of models and datasets, and in some cases, is even a perfect membership classifier on state-of-the-art models. Our findings highlight a unique privacy vulnerability of LLM-based synthetic data generation and the need for effective defenses. To this end, we propose two methods, including a novel sampling strategy that strategically perturbs digits during generation. Our evaluation demonstrates that this approach can defeat these attacks with minimal loss of fidelity and utility of the synthetic data. 

---
# Multicalibration for LLM-based Code Generation 

**Authors**: Viola Campos, Robin Kuschnereit, Adrian Ulges  

**Link**: [PDF](https://arxiv.org/pdf/2512.08810)  

**Abstract**: As AI-based code generation becomes widespread, researchers are investigating the calibration of code LLMs - ensuring their confidence scores faithfully represent the true likelihood of code correctness. To do so, we investigate multicalibration, which can capture additional factors about a coding problem, such as complexity, code length, or programming language used. We study four multicalibration approaches on three function synthesis benchmarks, using latest-generation code LLMs (Qwen3 Coder, GPT-OSS, DeepSeek-R1-Distill). Our results demonstrate that multicalibration can yield distinct improvements over both uncalibrated token likelihoods (+1.03 in skill score) and baseline calibrations (+0.37 in skill score). We study the influence of the aforementioned factors in ablations, and make our dataset (consisting of code generations, likelihoods, and correctness labels) available for future research on code LLM calibration. 

---
# Multi-Agent Intelligence for Multidisciplinary Decision-Making in Gastrointestinal Oncology 

**Authors**: Rongzhao Zhang, Junqiao Wang, Shuyun Yang, Mouxiao Bian, Chao Ding, Yuwei Bai, Chihao Zhang, Yuguang Shen, Lei Wang, Lei Zheng, Qiujuan Yan, Yun Zhong, Meiling Liu, Jiwei Yu, Zheng Wang, Jie Xu, Meng Luo  

**Link**: [PDF](https://arxiv.org/pdf/2512.08674)  

**Abstract**: Multimodal clinical reasoning in the field of gastrointestinal (GI) oncology necessitates the integrated interpretation of endoscopic imagery, radiological data, and biochemical markers. Despite the evident potential exhibited by Multimodal Large Language Models (MLLMs), they frequently encounter challenges such as context dilution and hallucination when confronted with intricate, heterogeneous medical histories. In order to address these limitations, a hierarchical Multi-Agent Framework is proposed, which emulates the collaborative workflow of a human Multidisciplinary Team (MDT). The system attained a composite expert evaluation score of 4.60/5.00, thereby demonstrating a substantial improvement over the monolithic baseline. It is noteworthy that the agent-based architecture yielded the most substantial enhancements in reasoning logic and medical accuracy. The findings indicate that mimetic, agent-based collaboration provides a scalable, interpretable, and clinically robust paradigm for automated decision support in oncology. 

---
# Biothreat Benchmark Generation Framework for Evaluating Frontier AI Models III: Implementing the Bacterial Biothreat Benchmark (B3) Dataset 

**Authors**: Gary Ackerman, Theodore Wilson, Zachary Kallenborn, Olivia Shoemaker, Anna Wetzel, Hayley Peterson, Abigail Danfora, Jenna LaTourette, Brandon Behlendorf, Douglas Clifford  

**Link**: [PDF](https://arxiv.org/pdf/2512.08459)  

**Abstract**: The potential for rapidly-evolving frontier artificial intelligence (AI) models, especially large language models (LLMs), to facilitate bioterrorism or access to biological weapons has generated significant policy, academic, and public concern. Both model developers and policymakers seek to quantify and mitigate any risk, with an important element of such efforts being the development of model benchmarks that can assess the biosecurity risk posed by a particular model. This paper discusses the pilot implementation of the Bacterial Biothreat Benchmark (B3) dataset. It is the third in a series of three papers describing an overall Biothreat Benchmark Generation (BBG) framework, with previous papers detailing the development of the B3 dataset. The pilot involved running the benchmarks through a sample frontier AI model, followed by human evaluation of model responses, and an applied risk analysis of the results along several dimensions. Overall, the pilot demonstrated that the B3 dataset offers a viable, nuanced method for rapidly assessing the biosecurity risk posed by a LLM, identifying the key sources of that risk and providing guidance for priority areas of mitigation priority. 

---
# LLM-based Vulnerable Code Augmentation: Generate or Refactor? 

**Authors**: Dyna Soumhane Ouchebara, Stéphane Dupont  

**Link**: [PDF](https://arxiv.org/pdf/2512.08493)  

**Abstract**: Vulnerability code-bases often suffer from severe imbalance, limiting the effectiveness of Deep Learning-based vulnerability classifiers. Data Augmentation could help solve this by mitigating the scarcity of under-represented CWEs. In this context, we investigate LLM-based augmentation for vulnerable functions, comparing controlled generation of new vulnerable samples with semantics-preserving refactoring of existing ones. Using Qwen2.5-Coder to produce augmented data and CodeBERT as a vulnerability classifier on the SVEN dataset, we find that our approaches are indeed effective in enriching vulnerable code-bases through a simple process and with reasonable quality, and that a hybrid strategy best boosts vulnerability classifiers' performance. 

---
# Are generative AI text annotations systematically biased? 

**Authors**: Sjoerd B. Stolwijk, Mark Boukes, Damian Trilling  

**Link**: [PDF](https://arxiv.org/pdf/2512.08404)  

**Abstract**: This paper investigates bias in GLLM annotations by conceptually replicating manual annotations of Boukes (2024). Using various GLLMs (Llama3.1:8b, Llama3.3:70b, GPT4o, Qwen2.5:72b) in combination with five different prompts for five concepts (political content, interactivity, rationality, incivility, and ideology). We find GLLMs perform adequate in terms of F1 scores, but differ from manual annotations in terms of prevalence, yield substantively different downstream results, and display systematic bias in that they overlap more with each other than with manual annotations. Differences in F1 scores fail to account for the degree of bias. 

---
# Systematization of Knowledge: Security and Safety in the Model Context Protocol Ecosystem 

**Authors**: Shiva Gaire, Srijan Gyawali, Saroj Mishra, Suman Niroula, Dilip Thakur, Umesh Yadav  

**Link**: [PDF](https://arxiv.org/pdf/2512.08290)  

**Abstract**: The Model Context Protocol (MCP) has emerged as the de facto standard for connecting Large Language Models (LLMs) to external data and tools, effectively functioning as the "USB-C for Agentic AI." While this decoupling of context and execution solves critical interoperability challenges, it introduces a profound new threat landscape where the boundary between epistemic errors (hallucinations) and security breaches (unauthorized actions) dissolves. This Systematization of Knowledge (SoK) aims to provide a comprehensive taxonomy of risks in the MCP ecosystem, distinguishing between adversarial security threats (e.g., indirect prompt injection, tool poisoning) and epistemic safety hazards (e.g., alignment failures in distributed tool delegation). We analyze the structural vulnerabilities of MCP primitives, specifically Resources, Prompts, and Tools, and demonstrate how "context" can be weaponized to trigger unauthorized operations in multi-agent environments. Furthermore, we survey state-of-the-art defenses, ranging from cryptographic provenance (ETDI) to runtime intent verification, and conclude with a roadmap for securing the transition from conversational chatbots to autonomous agentic operating systems. 

---
# Empowering smart app development with SolidGPT: an edge-cloud hybrid AI agent framework 

**Authors**: Liao Hu, Qiteng Wu, Ruoyu Qi  

**Link**: [PDF](https://arxiv.org/pdf/2512.08286)  

**Abstract**: The integration of Large Language Models (LLMs) into mobile and software development workflows faces a persistent tension among three demands: semantic awareness, developer productivity, and data privacy. Traditional cloud-based tools offer strong reasoning but risk data exposure and latency, while on-device solutions lack full-context understanding across codebase and developer tooling. We introduce SolidGPT, an open-source, edge-cloud hybrid developer assistant built on GitHub, designed to enhance code and workspace semantic search. SolidGPT enables developers to: talk to your codebase: interactively query code and project structure, discovering the right methods and modules without manual searching. Automate software project workflows: generate PRDs, task breakdowns, Kanban boards, and even scaffold web app beginnings, with deep integration via VSCode and Notion. Configure private, extensible agents: onboard private code folders (up to approximately 500 files), connect Notion, customize AI agent personas via embedding and in-context training, and deploy via Docker, CLI, or VSCode extension. In practice, SolidGPT empowers developer productivity through: Semantic-rich code navigation: no more hunting through files or wondering where a feature lives. Integrated documentation and task management: seamlessly sync generated PRD content and task boards into developer workflows. Privacy-first design: running locally via Docker or VSCode, with full control over code and data, while optionally reaching out to LLM APIs as needed. By combining interactive code querying, automated project scaffolding, and human-AI collaboration, SolidGPT provides a practical, privacy-respecting edge assistant that accelerates real-world development workflows, ideal for intelligent mobile and software engineering contexts. 

---
# SpeechQualityLLM: LLM-Based Multimodal Assessment of Speech Quality 

**Authors**: Mahathir Monjur, Shahriar Nirjon  

**Link**: [PDF](https://arxiv.org/pdf/2512.08238)  

**Abstract**: Objective speech quality assessment is central to telephony, VoIP, and streaming systems, where large volumes of degraded audio must be monitored and optimized at scale. Classical metrics such as PESQ and POLQA approximate human mean opinion scores (MOS) but require carefully controlled conditions and expensive listening tests, while learning-based models such as NISQA regress MOS and multiple perceptual dimensions from waveforms or spectrograms, achieving high correlation with subjective ratings yet remaining rigid: they do not support interactive, natural-language queries and do not natively provide textual rationales. In this work, we introduce SpeechQualityLLM, a multimodal speech quality question-answering (QA) system that couples an audio encoder with a language model and is trained on the NISQA corpus using template-based question-answer pairs covering overall MOS and four perceptual dimensions (noisiness, coloration, discontinuity, and loudness) in both single-ended (degraded only) and double-ended (degraded plus clean reference) setups. Instead of directly regressing scores, our system is supervised to generate textual answers from which numeric predictions are parsed and evaluated with standard regression and ranking metrics; on held-out NISQA clips, the double-ended model attains a MOS mean absolute error (MAE) of 0.41 with Pearson correlation of 0.86, with competitive performance on dimension-wise tasks. Beyond these quantitative gains, it offers a flexible natural-language interface in which the language model acts as an audio quality expert: practitioners can query arbitrary aspects of degradations, prompt the model to emulate different listener profiles to capture human variability and produce diverse but plausible judgments rather than a single deterministic score, and thereby reduce reliance on large-scale crowdsourced tests and their monetary cost. 

---
# Chat with UAV -- Human-UAV Interaction Based on Large Language Models 

**Authors**: Haoran Wang, Zhuohang Chen, Guang Li, Bo Ma, Chuanghuang Li  

**Link**: [PDF](https://arxiv.org/pdf/2512.08145)  

**Abstract**: The future of UAV interaction systems is evolving from engineer-driven to user-driven, aiming to replace traditional predefined Human-UAV Interaction designs. This shift focuses on enabling more personalized task planning and design, thereby achieving a higher quality of interaction experience and greater flexibility, which can be used in many fileds, such as agriculture, aerial photography, logistics, and environmental monitoring. However, due to the lack of a common language between users and the UAVs, such interactions are often difficult to be achieved. The developments of Large Language Models possess the ability to understand nature languages and Robots' (UAVs') behaviors, marking the possibility of personalized Human-UAV Interaction. Recently, some HUI frameworks based on LLMs have been proposed, but they commonly suffer from difficulties in mixed task planning and execution, leading to low adaptability in complex scenarios. In this paper, we propose a novel dual-agent HUI framework. This framework constructs two independent LLM agents (a task planning agent, and an execution agent) and applies different Prompt Engineering to separately handle the understanding, planning, and execution of tasks. To verify the effectiveness and performance of the framework, we have built a task database covering four typical application scenarios of UAVs and quantified the performance of the HUI framework using three independent metrics. Meanwhile different LLM models are selected to control the UAVs with compared performance. Our user study experimental results demonstrate that the framework improves the smoothness of HUI and the flexibility of task execution in the tasks scenario we set up, effectively meeting users' personalized needs. 

---
# Training LLMs for Honesty via Confessions 

**Authors**: Manas Joglekar, Jeremy Chen, Gabriel Wu, Jason Yosinski, Jasmine Wang, Boaz Barak, Amelia Glaese  

**Link**: [PDF](https://arxiv.org/pdf/2512.08093)  

**Abstract**: Large language models (LLMs) can be dishonest when reporting on their actions and beliefs -- for example, they may overstate their confidence in factual claims or cover up evidence of covert actions. Such dishonesty may arise due to the effects of reinforcement learning (RL), where challenges with reward shaping can result in a training process that inadvertently incentivizes the model to lie or misrepresent its actions.
In this work we propose a method for eliciting an honest expression of an LLM's shortcomings via a self-reported *confession*. A confession is an output, provided upon request after a model's original answer, that is meant to serve as a full account of the model's compliance with the letter and spirit of its policies and instructions. The reward assigned to a confession during training is solely based on its honesty, and does not impact positively or negatively the main answer's reward. As long as the "path of least resistance" for maximizing confession reward is to surface misbehavior rather than covering it up, this incentivizes models to be honest in their confessions. Our findings provide some justification this empirical assumption, especially in the case of egregious model misbehavior.
To demonstrate the viability of our approach, we train GPT-5-Thinking to produce confessions, and we evaluate its honesty in out-of-distribution scenarios measuring hallucination, instruction following, scheming, and reward hacking. We find that when the model lies or omits shortcomings in its "main" answer, it often confesses to these behaviors honestly, and this confession honesty modestly improves with training. Confessions can enable a number of inference-time interventions including monitoring, rejection sampling, and surfacing issues to the user. 

---
# DeepCode: Open Agentic Coding 

**Authors**: Zongwei Li, Zhonghang Li, Zirui Guo, Xubin Ren, Chao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2512.07921)  

**Abstract**: Recent advances in large language models (LLMs) have given rise to powerful coding agents, making it possible for code assistants to evolve into code engineers. However, existing methods still face significant challenges in achieving high-fidelity document-to-codebase synthesis--such as scientific papers to code--primarily due to a fundamental conflict between information overload and the context bottlenecks of LLMs. In this work, we introduce DeepCode, a fully autonomous framework that fundamentally addresses this challenge through principled information-flow management. By treating repository synthesis as a channel optimization problem, DeepCode seamlessly orchestrates four information operations to maximize task-relevant signals under finite context budgets: source compression via blueprint distillation, structured indexing using stateful code memory, conditional knowledge injection via retrieval-augmented generation, and closed-loop error correction. Extensive evaluations on the PaperBench benchmark demonstrate that DeepCode achieves state-of-the-art performance, decisively outperforming leading commercial agents such as Cursor and Claude Code, and crucially, surpassing PhD-level human experts from top institutes on key reproduction metrics. By systematically transforming paper specifications into production-grade implementations comparable to human expert quality, this work establishes new foundations for autonomous scientific reproduction that can accelerate research evaluation and discovery. 

---
# MARINE: Theoretical Optimization and Design for Multi-Agent Recursive IN-context Enhancement 

**Authors**: Hongwei Zhang, Ji Lu, Yongsheng Du, Yanqin Gao, Lingjun Huang, Baoli Wang, Fang Tan, Peng Zou  

**Link**: [PDF](https://arxiv.org/pdf/2512.07898)  

**Abstract**: Large Language Model (LLM)-based agents demonstrate advanced reasoning capabilities, yet practical constraints frequently limit outputs to single responses, leaving significant performance potential unrealized. This paper introduces MARINE (Multi-Agent Recursive IN-context Enhancement), a theoretically grounded framework that reconceptualizes test-time reasoning as iterative refinement of a persistent reference trajectory, fundamentally departing from conventional one-shot or multi-sample paradigms. The MARINE refinement operator systematically converts a base model's pass@N capabilities into near-optimal pass@1 performance. Rigorous theoretical analysis establishes that minimal feasible batches maximize expected performance gains under fixed invocation budgets, while logarithmically growing batch schedules ensure continuous improvement without computational constraints. Comprehensive evaluation on the BrowserComp-ZH benchmark demonstrates state-of-the-art results, with a 685B-parameter implementation achieving 46.0% pass@1 accuracy. Meanwhile, MARINE establishes a new paradigm for parameter-efficient reasoning: an 80B-parameter model augmented with MARINE matches the performance of standalone 1000B-parameter agents, reducing parameter requirements by over an order of magnitude. Notably, within a fixed computational budget, the proposed MARINE delivers higher-quality samples to alignment and optimization processes than traditional sampling-and-ranking strategies. Consequently, it has great potential to boost post-training efficiency. 

---
# CFD-copilot: leveraging domain-adapted large language model and model context protocol to enhance simulation automation 

**Authors**: Zhehao Dong, Shanghai Du, Zhen Lu, Yue Yang  

**Link**: [PDF](https://arxiv.org/pdf/2512.07917)  

**Abstract**: Configuring computational fluid dynamics (CFD) simulations requires significant expertise in physics modeling and numerical methods, posing a barrier to non-specialists. Although automating scientific tasks with large language models (LLMs) has attracted attention, applying them to the complete, end-to-end CFD workflow remains a challenge due to its stringent domain-specific requirements. We introduce CFD-copilot, a domain-specialized LLM framework designed to facilitate natural language-driven CFD simulation from setup to post-processing. The framework employs a fine-tuned LLM to directly translate user descriptions into executable CFD setups. A multi-agent system integrates the LLM with simulation execution, automatic error correction, and result analysis. For post-processing, the framework utilizes the model context protocol (MCP), an open standard that decouples LLM reasoning from external tool execution. This modular design allows the LLM to interact with numerous specialized post-processing functions through a unified and scalable interface, improving the automation of data extraction and analysis. The framework was evaluated on benchmarks including the NACA~0012 airfoil and the three-element 30P-30N airfoil. The results indicate that domain-specific adaptation and the incorporation of the MCP jointly enhance the reliability and efficiency of LLM-driven engineering workflows. 

---
# LLM-Generated Counterfactual Stress Scenarios for Portfolio Risk Simulation via Hybrid Prompt-RAG Pipeline 

**Authors**: Masoud Soleimani  

**Link**: [PDF](https://arxiv.org/pdf/2512.07867)  

**Abstract**: We develop a transparent and fully auditable LLM-based pipeline for macro-financial stress testing, combining structured prompting with optional retrieval of country fundamentals and news. The system generates machine-readable macroeconomic scenarios for the G7, which cover GDP growth, inflation, and policy rates, and are translated into portfolio losses through a factor-based mapping that enables Value-at-Risk and Expected Shortfall assessment relative to classical econometric baselines. Across models, countries, and retrieval settings, the LLMs produce coherent and country-specific stress narratives, yielding stable tail-risk amplification with limited sensitivity to retrieval choices. Comprehensive plausibility checks, scenario diagnostics, and ANOVA-based variance decomposition show that risk variation is driven primarily by portfolio composition and prompt design rather than by the retrieval mechanism. The pipeline incorporates snapshotting, deterministic modes, and hash-verified artifacts to ensure reproducibility and auditability. Overall, the results demonstrate that LLM-generated macro scenarios, when paired with transparent structure and rigorous validation, can provide a scalable and interpretable complement to traditional stress-testing frameworks. 

---
# MixLM: High-Throughput and Effective LLM Ranking via Text-Embedding Mix-Interaction 

**Authors**: Guoyao Li, Ran He, Shusen Jing, Kayhan Behdin, Yubo Wang, Sundara Raman Ramachandran, Chanh Nguyen, Jian Sheng, Xiaojing Ma, Chuanrui Zhu, Sriram Vasudevan, Muchen Wu, Sayan Ghosh, Lin Su, Qingquan Song, Xiaoqing Wang, Zhipeng Wang, Qing Lan, Yanning Chen, Jingwei Wu, Luke Simon, Wenjing Zhang, Qi Guo, Fedor Borisyuk  

**Link**: [PDF](https://arxiv.org/pdf/2512.07846)  

**Abstract**: Large language models (LLMs) excel at capturing semantic nuances and therefore show impressive relevance ranking performance in modern recommendation and search systems. However, they suffer from high computational overhead under industrial latency and throughput requirements. In particular, cross-encoder ranking systems often create long context prefill-heavy workloads, as the model has to be presented with the user, query and item information. To this end, we propose MixLM, a novel LLM-based ranking framework, which significantly improves the system throughput via reducing the input context length, while preserving the semantic strength of cross-encoder rankers. In contrast to a standard ranking system where the context is presented to the model as pure text, we propose to use mix-interaction, a mixture of text and embedding tokens to represent the input. Specifically, MixLM encodes all items in the catalog into a few embedding tokens and stores in a nearline cache. The encoded item descriptions are used during online inference, effectively reducing the item length from a few thousand text tokens to a few embedding tokens. We share insights from deploying our MixLM framework to a real-world search application at LinkedIn, including a detailed discussion of our training pipelines, as well as a thorough analysis of our online serving infrastructure optimization. Comparing with strong baselines, MixLM increased throughput by 10.0x under the same latency budget, while maintaining relevance metrics. The efficiency gains delivered by MixLM enabled full-traffic deployment of LLM-powered search, which resulted in a significant 0.47% increase in Daily Active Users (DAU) in online A/B tests. 

---
# AudioScene: Integrating Object-Event Audio into 3D Scenes 

**Authors**: Shuaihang Yuan, Congcong Wen, Muhammad Shafique, Anthony Tzes, Yi Fang  

**Link**: [PDF](https://arxiv.org/pdf/2512.07845)  

**Abstract**: The rapid advances in audio analysis underscore its vast potential for humancomputer interaction, environmental monitoring, and public safety; yet, existing audioonly datasets often lack spatial context. To address this gap, we present two novel audiospatial scene datasets, AudioScanNet and AudioRoboTHOR, designed to explore audioconditioned tasks within 3D environments. By integrating audio clips with spatially aligned 3D scenes, our datasets enable research on how audio signals interact with spatial context. To associate audio events with corresponding spatial information, we leverage the common sense reasoning ability of large language models and supplement them with rigorous human verification, This approach offers greater scalability compared to purely manual annotation while maintaining high standards of accuracy, completeness, and diversity, quantified through inter annotator agreement and performance on two benchmark tasks audio based 3D visual grounding and audio based robotic zeroshot navigation. The results highlight the limitations of current audiocentric methods and underscore the practical challenges and significance of our datasets in advancing audio guided spatial learning. 

---
# ThreadWeaver: Adaptive Threading for Efficient Parallel Reasoning in Language Models 

**Authors**: Long Lian, Sida Wang, Felix Juefei-Xu, Tsu-Jui Fu, Xiuyu Li, Adam Yala, Trevor Darrell, Alane Suhr, Yuandong Tian, Xi Victoria Lin  

**Link**: [PDF](https://arxiv.org/pdf/2512.07843)  

**Abstract**: Scaling inference-time computation has enabled Large Language Models (LLMs) to achieve strong reasoning performance, but inherently sequential decoding leads to substantial latency, especially on complex tasks. Recent work on adaptive parallel reasoning aims to improve inference efficiency by decomposing the problem-solving process into concurrent reasoning threads when beneficial. However, existing methods on realistic tasks are either limited to supervised behavior cloning or exhibit significant accuracy drops compared to widely-used sequential long chain-of-thought (CoT) baselines. Moreover, many require customized inference engines, complicating deployment. We introduce ThreadWeaver, a framework for adaptive parallel reasoning that achieves accuracy on par with popular sequential reasoning models of comparable size while significantly reducing inference latency. ThreadWeaver's performance stems from three key innovations: 1) a two-stage parallel trajectory generator that produces large-scale, high-quality CoT data with parallel annotations for supervised fine-tuning; 2) a trie-based training-inference co-design that enables parallel reasoning on any off-the-shelf autoregressive inference engine without modifying position embeddings or KV caches; and 3) a parallelization-aware reinforcement learning framework that teaches the model to balance accuracy with effective parallelization. Across six challenging mathematical reasoning benchmarks, ThreadWeaver trained atop Qwen3-8B achieves accuracy comparable to cutting-edge sequential reasoning models (71.9% on average and 79.9% on AIME24) while delivering up to 1.53x average speedup in token latency, establishing a new Pareto frontier between accuracy and efficiency. 

---
# Automating High Energy Physics Data Analysis with LLM-Powered Agents 

**Authors**: Eli Gendreau-Distler, Joshua Ho, Dongwon Kim, Luc Tomas Le Pottier, Haichen Wang, Chengxi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2512.07785)  

**Abstract**: We present a proof-of-principle study demonstrating the use of large language model (LLM) agents to automate a representative high energy physics (HEP) analysis. Using the Higgs boson diphoton cross-section measurement as a case study with ATLAS Open Data, we design a hybrid system that combines an LLM-based supervisor-coder agent with the Snakemake workflow manager. In this architecture, the workflow manager enforces reproducibility and determinism, while the agent autonomously generates, executes, and iteratively corrects analysis code in response to user instructions. We define quantitative evaluation metrics including success rate, error distribution, costs per specific task, and average number of API calls, to assess agent performance across multi-stage workflows. To characterize variability across architectures, we benchmark a representative selection of state-of-the-art LLMs spanning the Gemini and GPT-5 series, the Claude family, and leading open-weight models. While the workflow manager ensures deterministic execution of all analysis steps, the final outputs still show stochastic variation. Although we set the temperature to zero, other sampling parameters (e.g., top-p, top-k) remained at their defaults, and some reasoning-oriented models internally adjust these settings. Consequently, the models do not produce fully deterministic results. This study establishes the first LLM-agent-driven automated data-analysis framework in HEP, enabling systematic benchmarking of model capabilities, stability, and limitations in real-world scientific computing environments. The baseline code used in this work is available at this https URL. This work was accepted as a poster at the Machine Learning and the Physical Sciences (ML4PS) workshop at NeurIPS 2025. The initial submission was made on August 30, 2025. 

---
# Exploiting the Randomness of Large Language Models (LLM) in Text Classification Tasks: Locating Privileged Documents in Legal Matters 

**Authors**: Keith Huffman, Jianping Zhang, Nathaniel Huber-Fliflet, Fusheng Wei, Peter Gronvall  

**Link**: [PDF](https://arxiv.org/pdf/2512.08083)  

**Abstract**: In legal matters, text classification models are most often used to filter through large datasets in search of documents that meet certain pre-selected criteria like relevance to a certain subject matter, such as legally privileged communications and attorney-directed documents. In this context, large language models have demonstrated strong performance. This paper presents an empirical study investigating the role of randomness in LLM-based classification for attorney-client privileged document detection, focusing on four key dimensions: (1) the effectiveness of LLMs in identifying legally privileged documents, (2) the influence of randomness control parameters on classification outputs, (3) their impact on overall classification performance, and (4) a methodology for leveraging randomness to enhance accuracy. Experimental results showed that LLMs can identify privileged documents effectively, randomness control parameters have minimal impact on classification performance, and importantly, our developed methodology for leveraging randomness can have a significant impact on improving accuracy. Notably, this methodology that leverages randomness could also enhance a corporation's confidence in an LLM's output when incorporated into its sanctions-compliance processes. As organizations increasingly rely on LLMs to augment compliance workflows, reducing output variability helps build internal and regulatory confidence in LLM-derived sanctions-screening decisions. 

---
# Leveraging Machine Learning and Large Language Models for Automated Image Clustering and Description in Legal Discovery 

**Authors**: Qiang Mao, Fusheng Wei, Robert Neary, Charles Wang, Han Qin, Jianping Zhang, Nathaniel Huber-Fliflet  

**Link**: [PDF](https://arxiv.org/pdf/2512.08079)  

**Abstract**: The rapid increase in digital image creation and retention presents substantial challenges during legal discovery, digital archive, and content management. Corporations and legal teams must organize, analyze, and extract meaningful insights from large image collections under strict time pressures, making manual review impractical and costly. These demands have intensified interest in automated methods that can efficiently organize and describe large-scale image datasets. This paper presents a systematic investigation of automated cluster description generation through the integration of image clustering, image captioning, and large language models (LLMs). We apply K-means clustering to group images into 20 visually coherent clusters and generate base captions using the Azure AI Vision API. We then evaluate three critical dimensions of the cluster description process: (1) image sampling strategies, comparing random, centroid-based, stratified, hybrid, and density-based sampling against using all cluster images; (2) prompting techniques, contrasting standard prompting with chain-of-thought prompting; and (3) description generation methods, comparing LLM-based generation with traditional TF-IDF and template-based approaches. We assess description quality using semantic similarity and coverage metrics. Results show that strategic sampling with 20 images per cluster performs comparably to exhaustive inclusion while significantly reducing computational cost, with only stratified sampling showing modest degradation. LLM-based methods consistently outperform TF-IDF baselines, and standard prompts outperform chain-of-thought prompts for this task. These findings provide practical guidance for deploying scalable, accurate cluster description systems that support high-volume workflows in legal discovery and other domains requiring automated organization of large image collections. 

---
# Ontology-Based Knowledge Graph Framework for Industrial Standard Documents via Hierarchical and Propositional Structuring 

**Authors**: Jiin Park, Hyuna Jeon, Yoonseo Lee, Jisu Hong, Misuk Kim  

**Link**: [PDF](https://arxiv.org/pdf/2512.08398)  

**Abstract**: Ontology-based knowledge graph (KG) construction is a core technology that enables multidimensional understanding and advanced reasoning over domain knowledge. Industrial standards, in particular, contain extensive technical information and complex rules presented in highly structured formats that combine tables, scopes of application, constraints, exceptions, and numerical calculations, making KG construction especially challenging. In this study, we propose a method that organizes such documents into a hierarchical semantic structure, decomposes sentences and tables into atomic propositions derived from conditional and numerical rules, and integrates them into an ontology-knowledge graph through LLM-based triple extraction. Our approach captures both the hierarchical and logical structures of documents, effectively representing domain-specific semantics that conventional methods fail to reflect. To verify its effectiveness, we constructed rule, table, and multi-hop QA datasets, as well as a toxic clause detection dataset, from industrial standards, and implemented an ontology-aware KG-RAG framework for comparative evaluation. Experimental results show that our method achieves significant performance improvements across all QA types compared to existing KG-RAG approaches. This study demonstrates that reliable and scalable knowledge representation is feasible even for industrial documents with intertwined conditions, constraints, and scopes, contributing to future domain-specific RAG development and intelligent document management. 

---
# QSTN: A Modular Framework for Robust Questionnaire Inference with Large Language Models 

**Authors**: Maximilian Kreutner, Jens Rupprecht, Georg Ahnert, Ahmed Salem, Markus Strohmaier  

**Link**: [PDF](https://arxiv.org/pdf/2512.08646)  

**Abstract**: We introduce QSTN, an open-source Python framework for systematically generating responses from questionnaire-style prompts to support in-silico surveys and annotation tasks with large language models (LLMs). QSTN enables robust evaluation of questionnaire presentation, prompt perturbations, and response generation methods. Our extensive evaluation ($>40 $ million survey responses) shows that question structure and response generation methods have a significant impact on the alignment of generated survey responses with human answers, and can be obtained for a fraction of the compute cost. In addition, we offer a no-code user interface that allows researchers to set up robust experiments with LLMs without coding knowledge. We hope that QSTN will support the reproducibility and reliability of LLM-based research in the future. 

---
# What Triggers my Model? Contrastive Explanations Inform Gender Choices by Translation Models 

**Authors**: Janiça Hackenbuchner, Arda Tezcan, Joke Daems  

**Link**: [PDF](https://arxiv.org/pdf/2512.08440)  

**Abstract**: Interpretability can be implemented as a means to understand decisions taken by (black box) models, such as machine translation (MT) or large language models (LLMs). Yet, research in this area has been limited in relation to a manifested problem in these models: gender bias. With this research, we aim to move away from simply measuring bias to exploring its origins. Working with gender-ambiguous natural source data, this study examines which context, in the form of input tokens in the source sentence, influences (or triggers) the translation model choice of a certain gender inflection in the target language. To analyse this, we use contrastive explanations and compute saliency attribution. We first address the challenge of a lacking scoring threshold and specifically examine different attribution levels of source words on the model gender decisions in the translation. We compare salient source words with human perceptions of gender and demonstrate a noticeable overlap between human perceptions and model attribution. Additionally, we provide a linguistic analysis of salient words. Our work showcases the relevance of understanding model translation decisions in terms of gender, how this compares to human decisions and that this information should be leveraged to mitigate gender bias. 

---
# Soft Inductive Bias Approach via Explicit Reasoning Perspectives in Inappropriate Utterance Detection Using Large Language Models 

**Authors**: Ju-Young Kim, Ji-Hong Park, Se-Yeon Lee, Sujin Park, Gun-Woo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2512.08480)  

**Abstract**: Recent incidents in certain online games and communities, where anonymity is guaranteed, show that unchecked inappropriate remarks frequently escalate into verbal abuse and even criminal behavior, raising significant social concerns. Consequently, there is a growing need for research on techniques that can detect inappropriate utterances within conversational texts to help build a safer communication environment. Although large-scale language models trained on Korean corpora and chain-of-thought reasoning have recently gained attention, research applying these approaches to inappropriate utterance detection remains limited. In this study, we propose a soft inductive bias approach that explicitly defines reasoning perspectives to guide the inference process, thereby promoting rational decision-making and preventing errors that may arise during reasoning. We fine-tune a Korean large language model using the proposed method and conduct both quantitative performance comparisons and qualitative evaluations across different training strategies. Experimental results show that the Kanana-1.5 model achieves an average accuracy of 87.0046, improving by approximately 3.89 percent over standard supervised learning. These findings indicate that the proposed method goes beyond simple knowledge imitation by large language models and enables more precise and consistent judgments through constrained reasoning perspectives, demonstrating its effectiveness for inappropriate utterance detection. 

---
# Adaptation of Embedding Models to Financial Filings via LLM Distillation 

**Authors**: Eliot Brenner, Dominic Seyler, Manjunath Hegde, Andrei Simion, Koustuv Dasgupta, Bing Xiang  

**Link**: [PDF](https://arxiv.org/pdf/2512.08088)  

**Abstract**: Despite advances in generative large language models (LLMs), practical application of specialized conversational AI agents remains constrained by computation costs, latency requirements, and the need for precise domain-specific relevance measures. While existing embedding models address the first two constraints, they underperform on information retrieval in specialized domains like finance. This paper introduces a scalable pipeline that trains specialized models from an unlabeled corpus using a general purpose retrieval embedding model as foundation. Our method yields an average of 27.7% improvement in MRR$\texttt{@}$5, 44.6% improvement in mean DCG$\texttt{@}$5 across 14 financial filing types measured over 21,800 query-document pairs, and improved NDCG on 3 of 4 document classes in FinanceBench. We adapt retrieval embeddings (bi-encoder) for RAG, not LLM generators, using LLM-judged relevance to distill domain knowledge into a compact retriever. There are prior works which pair synthetically generated queries with real passages to directly fine-tune the retrieval model. Our pipeline differs from these by introducing interaction between student and teacher models that interleaves retrieval-based mining of hard positive/negative examples from the unlabeled corpus with iterative retraining of the student model's weights using these examples. Each retrieval iteration uses the refined student model to mine the corpus for progressively harder training examples for the subsequent training iteration. The methodology provides a cost-effective solution to bridging the gap between general-purpose models and specialized domains without requiring labor-intensive human annotation. 

---
# Ask, Answer, and Detect: Role-Playing LLMs for Personality Detection with Question-Conditioned Mixture-of-Experts 

**Authors**: Yifan Lyu, Liang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2512.08814)  

**Abstract**: Understanding human personality is crucial for web applications such as personalized recommendation and mental health assessment. Existing studies on personality detection predominantly adopt a "posts -> user vector -> labels" modeling paradigm, which encodes social media posts into user representations for predicting personality labels (e.g., MBTI labels). While recent advances in large language models (LLMs) have improved text encoding capacities, these approaches remain constrained by limited supervision signals due to label scarcity, and under-specified semantic mappings between user language and abstract psychological constructs. We address these challenges by proposing ROME, a novel framework that explicitly injects psychological knowledge into personality detection. Inspired by standardized self-assessment tests, ROME leverages LLMs' role-play capability to simulate user responses to validated psychometric questionnaires. These generated question-level answers transform free-form user posts into interpretable, questionnaire-grounded evidence linking linguistic cues to personality labels, thereby providing rich intermediate supervision to mitigate label scarcity while offering a semantic reasoning chain that guides and simplifies the text-to-personality mapping learning. A question-conditioned Mixture-of-Experts module then jointly routes over post and question representations, learning to answer questionnaire items under explicit supervision. The predicted answers are summarized into an interpretable answer vector and fused with the user representation for final prediction within a multi-task learning framework, where question answering serves as a powerful auxiliary task for personality detection. Extensive experiments on two real-world datasets demonstrate that ROME consistently outperforms state-of-the-art baselines, achieving improvements (15.41% on Kaggle dataset). 

---
# Balanced Accuracy: The Right Metric for Evaluating LLM Judges - Explained through Youden's J statistic 

**Authors**: Stephane Collot, Colin Fraser, Justin Zhao, William F. Shen, Timon Willi, Ilias Leontiadis  

**Link**: [PDF](https://arxiv.org/pdf/2512.08121)  

**Abstract**: Rigorous evaluation of large language models (LLMs) relies on comparing models by the prevalence of desirable or undesirable behaviors, such as task pass rates or policy violations. These prevalence estimates are produced by a classifier, either an LLM-as-a-judge or human annotators, making the choice of classifier central to trustworthy evaluation. Common metrics used for this choice, such as Accuracy, Precision, and F1, are sensitive to class imbalance and to arbitrary choices of positive class, and can favor judges that distort prevalence estimates. We show that Youden's $J$ statistic is theoretically aligned with choosing the best judge to compare models, and that Balanced Accuracy is an equivalent linear transformation of $J$. Through both analytical arguments and empirical examples and simulations, we demonstrate how selecting judges using Balanced Accuracy leads to better, more robust classifier selection. 

---
