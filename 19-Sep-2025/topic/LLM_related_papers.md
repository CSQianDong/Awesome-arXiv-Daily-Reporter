# Enhancing Retrieval Augmentation via Adversarial Collaboration 

**Authors**: Letian Zhang, Guanghao Meng, Xudong Ren, Yiming Wang, Shu-Tao Xia  

**Link**: [PDF](https://arxiv.org/pdf/2509.14750)  

**Abstract**: Retrieval-augmented Generation (RAG) is a prevalent approach for domain-specific LLMs, yet it is often plagued by "Retrieval Hallucinations"--a phenomenon where fine-tuned models fail to recognize and act upon poor-quality retrieved documents, thus undermining performance. To address this, we propose the Adversarial Collaboration RAG (AC-RAG) framework. AC-RAG employs two heterogeneous agents: a generalist Detector that identifies knowledge gaps, and a domain-specialized Resolver that provides precise solutions. Guided by a moderator, these agents engage in an adversarial collaboration, where the Detector's persistent questioning challenges the Resolver's expertise. This dynamic process allows for iterative problem dissection and refined knowledge retrieval. Extensive experiments show that AC-RAG significantly improves retrieval accuracy and outperforms state-of-the-art RAG methods across various vertical domains. 

---
# A Knowledge-driven Adaptive Collaboration of LLMs for Enhancing Medical Decision-making 

**Authors**: Xiao Wu, Ting-Zhu Huang, Liang-Jian Deng, Yanyuan Qiao, Imran Razzak, Yutong Xie  

**Link**: [PDF](https://arxiv.org/pdf/2509.14998)  

**Abstract**: Medical decision-making often involves integrating knowledge from multiple clinical specialties, typically achieved through multidisciplinary teams. Inspired by this collaborative process, recent work has leveraged large language models (LLMs) in multi-agent collaboration frameworks to emulate expert teamwork. While these approaches improve reasoning through agent interaction, they are limited by static, pre-assigned roles, which hinder adaptability and dynamic knowledge integration. To address these limitations, we propose KAMAC, a Knowledge-driven Adaptive Multi-Agent Collaboration framework that enables LLM agents to dynamically form and expand expert teams based on the evolving diagnostic context. KAMAC begins with one or more expert agents and then conducts a knowledge-driven discussion to identify and fill knowledge gaps by recruiting additional specialists as needed. This supports flexible, scalable collaboration in complex clinical scenarios, with decisions finalized through reviewing updated agent comments. Experiments on two real-world medical benchmarks demonstrate that KAMAC significantly outperforms both single-agent and advanced multi-agent methods, particularly in complex clinical scenarios (i.e., cancer prognosis) requiring dynamic, cross-specialty expertise. Our code is publicly available at: this https URL. 

---
# OpenLens AI: Fully Autonomous Research Agent for Health Infomatics 

**Authors**: Yuxiao Cheng, Jinli Suo  

**Link**: [PDF](https://arxiv.org/pdf/2509.14778)  

**Abstract**: Health informatics research is characterized by diverse data modalities, rapid knowledge expansion, and the need to integrate insights across biomedical science, data analytics, and clinical practice. These characteristics make it particularly well-suited for agent-based approaches that can automate knowledge exploration, manage complex workflows, and generate clinically meaningful outputs. Recent progress in large language model (LLM)-based agents has demonstrated promising capabilities in literature synthesis, data analysis, and even end-to-end research execution. However, existing systems remain limited for health informatics because they lack mechanisms to interpret medical visualizations and often overlook domain-specific quality requirements. To address these gaps, we introduce OpenLens AI, a fully automated framework tailored to health informatics. OpenLens AI integrates specialized agents for literature review, data analysis, code generation, and manuscript preparation, enhanced by vision-language feedback for medical visualization and quality control for reproducibility. The framework automates the entire research pipeline, producing publication-ready LaTeX manuscripts with transparent and traceable workflows, thereby offering a domain-adapted solution for advancing health informatics research. 

---
# Understanding the Thinking Process of Reasoning Models: A Perspective from Schoenfeld's Episode Theory 

**Authors**: Ming Li, Nan Zhang, Chenrui Fan, Hong Jiao, Yanbin Fu, Sydney Peters, Qingshu Xu, Robert Lissitz, Tianyi Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2509.14662)  

**Abstract**: While Large Reasoning Models (LRMs) generate extensive chain-of-thought reasoning, we lack a principled framework for understanding how these thoughts are structured. In this paper, we introduce a novel approach by applying Schoenfeld's Episode Theory, a classic cognitive framework for human mathematical problem-solving, to analyze the reasoning traces of LRMs. We annotated thousands of sentences and paragraphs from model-generated solutions to math problems using seven cognitive labels (e.g., Plan, Implement, Verify). The result is the first publicly available benchmark for the fine-grained analysis of machine reasoning, including a large annotated corpus and detailed annotation guidebooks. Our preliminary analysis reveals distinct patterns in LRM reasoning, such as the transition dynamics between cognitive states. This framework provides a theoretically grounded methodology for interpreting LRM cognition and enables future work on more controllable and transparent reasoning systems. 

---
# AgentCompass: Towards Reliable Evaluation of Agentic Workflows in Production 

**Authors**: NVJK Kartik, Garvit Sapra, Rishav Hada, Nikhil Pareek  

**Link**: [PDF](https://arxiv.org/pdf/2509.14647)  

**Abstract**: With the growing adoption of Large Language Models (LLMs) in automating complex, multi-agent workflows, organizations face mounting risks from errors, emergent behaviors, and systemic failures that current evaluation methods fail to capture. We present AgentCompass, the first evaluation framework designed specifically for post-deployment monitoring and debugging of agentic workflows. AgentCompass models the reasoning process of expert debuggers through a structured, multi-stage analytical pipeline: error identification and categorization, thematic clustering, quantitative scoring, and strategic summarization. The framework is further enhanced with a dual memory system-episodic and semantic-that enables continual learning across executions. Through collaborations with design partners, we demonstrate the framework's practical utility on real-world deployments, before establishing its efficacy against the publicly available TRAIL benchmark. AgentCompass achieves state-of-the-art results on key metrics, while uncovering critical issues missed in human annotations, underscoring its role as a robust, developer-centric tool for reliable monitoring and improvement of agentic systems in production. 

---
# Rationality Check! Benchmarking the Rationality of Large Language Models 

**Authors**: Zhilun Zhou, Jing Yi Wang, Nicholas Sukiennik, Chen Gao, Fengli Xu, Yong Li, James Evans  

**Link**: [PDF](https://arxiv.org/pdf/2509.14546)  

**Abstract**: Large language models (LLMs), a recent advance in deep learning and machine intelligence, have manifested astonishing capacities, now considered among the most promising for artificial general intelligence. With human-like capabilities, LLMs have been used to simulate humans and serve as AI assistants across many applications. As a result, great concern has arisen about whether and under what circumstances LLMs think and behave like real human agents. Rationality is among the most important concepts in assessing human behavior, both in thinking (i.e., theoretical rationality) and in taking action (i.e., practical rationality). In this work, we propose the first benchmark for evaluating the omnibus rationality of LLMs, covering a wide range of domains and LLMs. The benchmark includes an easy-to-use toolkit, extensive experimental results, and analysis that illuminates where LLMs converge and diverge from idealized human rationality. We believe the benchmark can serve as a foundational tool for both developers and users of LLMs. 

---
# Internalizing Self-Consistency in Language Models: Multi-Agent Consensus Alignment 

**Authors**: Ankur Samanta, Akshayaa Magesh, Youliang Yu, Runzhe Wu, Ayush Jain, Daniel Jiang, Boris Vidolov, Paul Sajda, Yonathan Efroni, Kaveh Hassani  

**Link**: [PDF](https://arxiv.org/pdf/2509.15172)  

**Abstract**: Language Models (LMs) are inconsistent reasoners, often generating contradictory responses to identical prompts. While inference-time methods can mitigate these inconsistencies, they fail to address the core problem: LMs struggle to reliably select reasoning pathways leading to consistent outcomes under exploratory sampling. To address this, we formalize self-consistency as an intrinsic property of well-aligned reasoning models and introduce Multi-Agent Consensus Alignment (MACA), a reinforcement learning framework that post-trains models to favor reasoning trajectories aligned with their internal consensus using majority/minority outcomes from multi-agent debate. These trajectories emerge from deliberative exchanges where agents ground reasoning in peer arguments, not just aggregation of independent attempts, creating richer consensus signals than single-round majority voting. MACA enables agents to teach themselves to be more decisive and concise, and better leverage peer insights in multi-agent settings without external supervision, driving substantial improvements across self-consistency (+27.6% on GSM8K), single-agent reasoning (+23.7% on MATH), sampling-based inference (+22.4% Pass@20 on MATH), and multi-agent ensemble decision-making (+42.7% on MathQA). These findings, coupled with strong generalization to unseen benchmarks (+16.3% on GPQA, +11.6% on CommonsenseQA), demonstrate robust self-alignment that more reliably unlocks latent reasoning potential of language models. 

---
# VCBench: Benchmarking LLMs in Venture Capital 

**Authors**: Rick Chen, Joseph Ternasky, Afriyie Samuel Kwesi, Ben Griffin, Aaron Ontoyin Yin, Zakari Salifu, Kelvin Amoaba, Xianling Mu, Fuat Alican, Yigit Ihlamur  

**Link**: [PDF](https://arxiv.org/pdf/2509.14448)  

**Abstract**: Benchmarks such as SWE-bench and ARC-AGI demonstrate how shared datasets accelerate progress toward artificial general intelligence (AGI). We introduce VCBench, the first benchmark for predicting founder success in venture capital (VC), a domain where signals are sparse, outcomes are uncertain, and even top investors perform modestly. At inception, the market index achieves a precision of 1.9%. Y Combinator outperforms the index by a factor of 1.7x, while tier-1 firms are 2.9x better. VCBench provides 9,000 anonymized founder profiles, standardized to preserve predictive features while resisting identity leakage, with adversarial tests showing more than 90% reduction in re-identification risk. We evaluate nine state-of-the-art large language models (LLMs). DeepSeek-V3 delivers over six times the baseline precision, GPT-4o achieves the highest F0.5, and most models surpass human benchmarks. Designed as a public and evolving resource available at this http URL, VCBench establishes a community-driven standard for reproducible and privacy-preserving evaluation of AGI in early-stage venture forecasting. 

---
# FlowRL: Matching Reward Distributions for LLM Reasoning 

**Authors**: Xuekai Zhu, Daixuan Cheng, Dinghuai Zhang, Hengli Li, Kaiyan Zhang, Che Jiang, Youbang Sun, Ermo Hua, Yuxin Zuo, Xingtai Lv, Qizheng Zhang, Lin Chen, Fanghao Shao, Bo Xue, Yunchong Song, Zhenjie Yang, Ganqu Cui, Ning Ding, Jianfeng Gao, Xiaodong Liu, Bowen Zhou, Hongyuan Mei, Zhouhan Lin  

**Link**: [PDF](https://arxiv.org/pdf/2509.15207)  

**Abstract**: We propose FlowRL: matching the full reward distribution via flow balancing instead of maximizing rewards in large language model (LLM) reinforcement learning (RL). Recent advanced reasoning models adopt reward-maximizing methods (\eg, PPO and GRPO), which tend to over-optimize dominant reward signals while neglecting less frequent but valid reasoning paths, thus reducing diversity. In contrast, we transform scalar rewards into a normalized target distribution using a learnable partition function, and then minimize the reverse KL divergence between the policy and the target distribution. We implement this idea as a flow-balanced optimization method that promotes diverse exploration and generalizable reasoning trajectories. We conduct experiments on math and code reasoning tasks: FlowRL achieves a significant average improvement of $10.0\%$ over GRPO and $5.1\%$ over PPO on math benchmarks, and performs consistently better on code reasoning tasks. These results highlight reward distribution-matching as a key step toward efficient exploration and diverse reasoning in LLM reinforcement learning. 

---
# SMARTER: A Data-efficient Framework to Improve Toxicity Detection with Explanation via Self-augmenting Large Language Models 

**Authors**: Huy Nghiem, Advik Sachdeva, Hal Daumé III  

**Link**: [PDF](https://arxiv.org/pdf/2509.15174)  

**Abstract**: WARNING: This paper contains examples of offensive materials. Toxic content has become pervasive on social media platforms. We introduce SMARTER, a data-efficient two-stage framework for explainable content moderation using Large Language Models (LLMs). In Stage 1, we leverage LLMs' own outputs to generate synthetic explanations for both correct and incorrect labels, enabling alignment via preference optimization with minimal human supervision. In Stage 2, we refine explanation quality through cross-model training, allowing weaker models to align stylistically and semantically with stronger ones. Experiments on three benchmark tasks -- HateXplain, Latent Hate, and Implicit Hate -- demonstrate that SMARTER enables LLMs to achieve up to a 13.5% macro-F1 improvement over standard few-shot baselines while using only a fraction of the full training data. Our framework offers a scalable strategy for low-resource settings by harnessing LLMs' self-improving capabilities for both classification and explanation. 

---
# The NazoNazo Benchmark: A Cost-Effective and Extensible Test of Insight-Based Reasoning in LLMs 

**Authors**: Masaharu Mizumoto, Dat Nguyen, Zhiheng Han, Jiyuan Fang, Heyuan Guan, Xingfu Li, Naoya Shiraishi, Xuyang Tian, Yo Nakawake, Le Minh Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2509.14704)  

**Abstract**: Benchmark saturation and contamination undermine confidence in LLM evaluation. We present Nazonazo, a cost-effective and extensible benchmark built from Japanese children's riddles to test insight-based reasoning. Items are short (mostly one sentence), require no specialized domain knowledge, and can be generated at scale, enabling rapid refresh of blind sets when leakage is suspected. We evaluate 38 frontier models and 126 adults on 120 riddles. No model except for GPT-5 is comparable to human performance, which achieves a 52.9% mean accuracy. Model comparison on extended 201 items shows that reasoning models significantly outperform non-reasoning peers, while model size shows no reliable association with accuracy. Beyond aggregate accuracy, an informal candidate-tracking analysis of thought logs reveals many cases of verification failure: models often produce the correct solution among intermediate candidates yet fail to select it as the final answer, which we illustrate with representative examples observed in multiple models. Nazonazo thus offers a cost-effective, scalable, and easily renewable benchmark format that addresses the current evaluation crisis while also suggesting a recurrent meta-cognitive weakness, providing clear targets for future control and calibration methods. 

---
# TextMine: LLM-Powered Knowledge Extraction for Humanitarian Mine Action 

**Authors**: Chenyue Zhou, Gürkan Solmaz, Flavio Cirillo, Kiril Gashteovski, Jonathan Fürst  

**Link**: [PDF](https://arxiv.org/pdf/2509.15098)  

**Abstract**: Humanitarian Mine Action has generated extensive best-practice knowledge, but much remains locked in unstructured reports. We introduce TextMine, an ontology-guided pipeline that uses Large Language Models to extract knowledge triples from HMA texts. TextMine integrates document chunking, domain-aware prompting, triple extraction, and both reference-based and LLM-as-a-Judge evaluation. We also create the first HMA ontology and a curated dataset of real-world demining reports. Experiments show ontology-aligned prompts boost extraction accuracy by 44.2%, cut hallucinations by 22.5%, and improve format conformance by 20.9% over baselines. While validated on Cambodian reports, TextMine can adapt to global demining efforts or other domains, transforming unstructured data into structured knowledge. 

---
# Listening, Imagining \& Refining: A Heuristic Optimized ASR Correction Framework with LLMs 

**Authors**: Yutong Liu, Ziyue Zhang, Yongbin Yu, Xiangxiang Wang, Yuqing Cai, Nyima Tashi  

**Link**: [PDF](https://arxiv.org/pdf/2509.15095)  

**Abstract**: Automatic Speech Recognition (ASR) systems remain prone to errors that affect downstream applications. In this paper, we propose LIR-ASR, a heuristic optimized iterative correction framework using LLMs, inspired by human auditory perception. LIR-ASR applies a "Listening-Imagining-Refining" strategy, generating phonetic variants and refining them in context. A heuristic optimization with finite state machine (FSM) is introduced to prevent the correction process from being trapped in local optima and rule-based constraints help maintain semantic fidelity. Experiments on both English and Chinese ASR outputs show that LIR-ASR achieves average reductions in CER/WER of up to 1.5 percentage points compared to baselines, demonstrating substantial accuracy gains in transcription. 

---
# Sentinel Agents for Secure and Trustworthy Agentic AI in Multi-Agent Systems 

**Authors**: Diego Gosmar, Deborah A. Dahl  

**Link**: [PDF](https://arxiv.org/pdf/2509.14956)  

**Abstract**: This paper proposes a novel architectural framework aimed at enhancing security and reliability in multi-agent systems (MAS). A central component of this framework is a network of Sentinel Agents, functioning as a distributed security layer that integrates techniques such as semantic analysis via large language models (LLMs), behavioral analytics, retrieval-augmented verification, and cross-agent anomaly detection. Such agents can potentially oversee inter-agent communications, identify potential threats, enforce privacy and access controls, and maintain comprehensive audit records. Complementary to the idea of Sentinel Agents is the use of a Coordinator Agent. The Coordinator Agent supervises policy implementation, and manages agent participation. In addition, the Coordinator also ingests alerts from Sentinel Agents. Based on these alerts, it can adapt policies, isolate or quarantine misbehaving agents, and contain threats to maintain the integrity of the MAS ecosystem. This dual-layered security approach, combining the continuous monitoring of Sentinel Agents with the governance functions of Coordinator Agents, supports dynamic and adaptive defense mechanisms against a range of threats, including prompt injection, collusive agent behavior, hallucinations generated by LLMs, privacy breaches, and coordinated multi-agent attacks. In addition to the architectural design, we present a simulation study where 162 synthetic attacks of different families (prompt injection, hallucination, and data exfiltration) were injected into a multi-agent conversational environment. The Sentinel Agents successfully detected the attack attempts, confirming the practical feasibility of the proposed monitoring approach. The framework also offers enhanced system observability, supports regulatory compliance, and enables policy evolution over time. 

---
# Cross-Modal Knowledge Distillation for Speech Large Language Models 

**Authors**: Enzhi Wang, Qicheng Li, Zhiyuan Tang, Yuhang Jia  

**Link**: [PDF](https://arxiv.org/pdf/2509.14930)  

**Abstract**: In this work, we present the first systematic evaluation of catastrophic forgetting and modality inequivalence in speech large language models, showing that introducing speech capabilities can degrade knowledge and reasoning even when inputs remain textual, and performance further decreases with spoken queries. To address these challenges, we propose a cross-modal knowledge distillation framework that leverages both text-to-text and speech-to-text channels to transfer knowledge from a text-based teacher model to a speech LLM. Extensive experiments on dialogue and audio understanding tasks validate the effectiveness of our approach in preserving textual knowledge, improving cross-modal alignment, and enhancing reasoning in speech-based interactions. 

---
# Empathy-R1: A Chain-of-Empathy and Reinforcement Learning Framework for Long-Form Mental Health Support 

**Authors**: Xianrong Yao, Dong She, Chenxu Zhang, Yimeng Zhang, Yueru Sun, Noman Ahmed, Yang Gao, Zhanpeng Jin  

**Link**: [PDF](https://arxiv.org/pdf/2509.14851)  

**Abstract**: Empathy is critical for effective mental health support, especially when addressing Long Counseling Texts (LCTs). However, existing Large Language Models (LLMs) often generate replies that are semantically fluent but lack the structured reasoning necessary for genuine psychological support, particularly in a Chinese context. To bridge this gap, we introduce Empathy-R1, a novel framework that integrates a Chain-of-Empathy (CoE) reasoning process with Reinforcement Learning (RL) to enhance response quality for LCTs. Inspired by cognitive-behavioral therapy, our CoE paradigm guides the model to sequentially reason about a help-seeker's emotions, causes, and intentions, making its thinking process both transparent and interpretable. Our framework is empowered by a new large-scale Chinese dataset, Empathy-QA, and a two-stage training process. First, Supervised Fine-Tuning instills the CoE's reasoning structure. Subsequently, RL, guided by a dedicated reward model, refines the therapeutic relevance and contextual appropriateness of the final responses. Experiments show that Empathy-R1 achieves strong performance on key automatic metrics. More importantly, human evaluations confirm its superiority, showing a clear preference over strong baselines and achieving a Win@1 rate of 44.30% on our new benchmark. By enabling interpretable and contextually nuanced responses, Empathy-R1 represents a significant advancement in developing responsible and genuinely beneficial AI for mental health support. 

---
# A Multi-To-One Interview Paradigm for Efficient MLLM Evaluation 

**Authors**: Ye Shen, Junying Wang, Farong Wen, Yijin Guo, Qi Jia, Zicheng Zhang, Guangtao Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2509.14886)  

**Abstract**: The rapid progress of Multi-Modal Large Language Models (MLLMs) has spurred the creation of numerous benchmarks. However, conventional full-coverage Question-Answering evaluations suffer from high redundancy and low efficiency. Inspired by human interview processes, we propose a multi-to-one interview paradigm for efficient MLLM evaluation. Our framework consists of (i) a two-stage interview strategy with pre-interview and formal interview phases, (ii) dynamic adjustment of interviewer weights to ensure fairness, and (iii) an adaptive mechanism for question difficulty-level chosen. Experiments on different benchmarks show that the proposed paradigm achieves significantly higher correlation with full-coverage results than random sampling, with improvements of up to 17.6% in PLCC and 16.7% in SRCC, while reducing the number of required questions. These findings demonstrate that the proposed paradigm provides a reliable and efficient alternative for large-scale MLLM benchmarking. 

---
# TableDART: Dynamic Adaptive Multi-Modal Routing for Table Understanding 

**Authors**: Xiaobo Xing, Wei Yuan, Tong Chen, Quoc Viet Hung Nguyen, Xiangliang Zhang, Hongzhi Yin  

**Link**: [PDF](https://arxiv.org/pdf/2509.14671)  

**Abstract**: Modeling semantic and structural information from tabular data remains a core challenge for effective table understanding. Existing Table-as-Text approaches flatten tables for large language models (LLMs), but lose crucial structural cues, while Table-as-Image methods preserve structure yet struggle with fine-grained semantics. Recent Table-as-Multimodality strategies attempt to combine textual and visual views, but they (1) statically process both modalities for every query-table pair within a large multimodal LLMs (MLLMs), inevitably introducing redundancy and even conflicts, and (2) depend on costly fine-tuning of MLLMs. In light of this, we propose TableDART, a training-efficient framework that integrates multimodal views by reusing pretrained single-modality models. TableDART introduces a lightweight 2.59M-parameter MLP gating network that dynamically selects the optimal path (either Text-only, Image-only, or Fusion) for each table-query pair, effectively reducing redundancy and conflicts from both modalities. In addition, we propose a novel agent to mediate cross-modal knowledge integration by analyzing outputs from text- and image-based models, either selecting the best result or synthesizing a new answer through reasoning. This design avoids the prohibitive costs of full MLLM fine-tuning. Extensive experiments on seven benchmarks show that TableDART establishes new state-of-the-art performance among open-source models, surpassing the strongest baseline by an average of 4.02%. The code is available at: this https URL 

---
# Adversarial Distilled Retrieval-Augmented Guarding Model for Online Malicious Intent Detection 

**Authors**: Yihao Guo, Haocheng Bian, Liutong Zhou, Ze Wang, Zhaoyi Zhang, Francois Kawala, Milan Dean, Ian Fischer, Yuantao Peng, Noyan Tokgozoglu, Ivan Barrientos, Riyaaz Shaik, Rachel Li, Chandru Venkataraman, Reza Shifteh Far, Moses Pawar, Venkat Sundaranatha, Michael Xu, Frank Chu  

**Link**: [PDF](https://arxiv.org/pdf/2509.14622)  

**Abstract**: With the deployment of Large Language Models (LLMs) in interactive applications, online malicious intent detection has become increasingly critical. However, existing approaches fall short of handling diverse and complex user queries in real time. To address these challenges, we introduce ADRAG (Adversarial Distilled Retrieval-Augmented Guard), a two-stage framework for robust and efficient online malicious intent detection. In the training stage, a high-capacity teacher model is trained on adversarially perturbed, retrieval-augmented inputs to learn robust decision boundaries over diverse and complex user queries. In the inference stage, a distillation scheduler transfers the teacher's knowledge into a compact student model, with a continually updated knowledge base collected online. At deployment, the compact student model leverages top-K similar safety exemplars retrieved from the online-updated knowledge base to enable both online and real-time malicious query detection. Evaluations across ten safety benchmarks demonstrate that ADRAG, with a 149M-parameter model, achieves 98.5% of WildGuard-7B's performance, surpasses GPT-4 by 3.3% and Llama-Guard-3-8B by 9.5% on out-of-distribution detection, while simultaneously delivering up to 5.6x lower latency at 300 queries per second (QPS) in real-time applications. 

---
# Spatial Audio Motion Understanding and Reasoning 

**Authors**: Arvind Krishna Sridhar, Yinyi Guo, Erik Visser  

**Link**: [PDF](https://arxiv.org/pdf/2509.14666)  

**Abstract**: Spatial audio reasoning enables machines to interpret auditory scenes by understanding events and their spatial attributes. In this work, we focus on spatial audio understanding with an emphasis on reasoning about moving sources. First, we introduce a spatial audio encoder that processes spatial audio to detect multiple overlapping events and estimate their spatial attributes, Direction of Arrival (DoA) and source distance, at the frame level. To generalize to unseen events, we incorporate an audio grounding model that aligns audio features with semantic audio class text embeddings via a cross-attention mechanism. Second, to answer complex queries about dynamic audio scenes involving moving sources, we condition a large language model (LLM) on structured spatial attributes extracted by our model. Finally, we introduce a spatial audio motion understanding and reasoning benchmark dataset and demonstrate our framework's performance against the baseline model. 

---
# ATLANTIS: AI-driven Threat Localization, Analysis, and Triage Intelligence System 

**Authors**: Taesoo Kim, HyungSeok Han, Soyeon Park, Dae R. Jeong, Dohyeok Kim, Dongkwan Kim, Eunsoo Kim, Jiho Kim, Joshua Wang, Kangsu Kim, Sangwoo Ji, Woosun Song, Hanqing Zhao, Andrew Chin, Gyejin Lee, Kevin Stevens, Mansour Alharthi, Yizhuo Zhai, Cen Zhang, Joonun Jang, Yeongjin Jang, Ammar Askar, Dongju Kim, Fabian Fleischer, Jeongin Cho, Junsik Kim, Kyungjoon Ko, Insu Yun, Sangdon Park, Dowoo Baik, Haein Lee, Hyeon Heo, Minjae Gwon, Minjae Lee, Minwoo Baek, Seunggi Min, Wonyoung Kim, Yonghwi Jin, Younggi Park, Yunjae Choi, Jinho Jung, Gwanhyun Lee, Junyoung Jang, Kyuheon Kim, Yeonghyeon Cha, Youngjoon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2509.14589)  

**Abstract**: We present ATLANTIS, the cyber reasoning system developed by Team Atlanta that won 1st place in the Final Competition of DARPA's AI Cyber Challenge (AIxCC) at DEF CON 33 (August 2025). AIxCC (2023-2025) challenged teams to build autonomous cyber reasoning systems capable of discovering and patching vulnerabilities at the speed and scale of modern software. ATLANTIS integrates large language models (LLMs) with program analysis -- combining symbolic execution, directed fuzzing, and static analysis -- to address limitations in automated vulnerability discovery and program repair. Developed by researchers at Georgia Institute of Technology, Samsung Research, KAIST, and POSTECH, the system addresses core challenges: scaling across diverse codebases from C to Java, achieving high precision while maintaining broad coverage, and producing semantically correct patches that preserve intended behavior. We detail the design philosophy, architectural decisions, and implementation strategies behind ATLANTIS, share lessons learned from pushing the boundaries of automated security when program analysis meets modern AI, and release artifacts to support reproducibility and future research. 

---
# OnlineMate: An LLM-Based Multi-Agent Companion System for Cognitive Support in Online Learning 

**Authors**: Xian Gao, Zongyun Zhang, Ting Liu, Yuzhuo Fu  

**Link**: [PDF](https://arxiv.org/pdf/2509.14803)  

**Abstract**: In online learning environments, students often lack personalized peer interactions, which play a crucial role in supporting cognitive development and learning engagement. Although previous studies have utilized large language models (LLMs) to simulate interactive dynamic learning environments for students, these interactions remain limited to conversational exchanges, lacking insights and adaptations to the learners' individualized learning and cognitive states. As a result, students' interest in discussions with AI learning companions is low, and they struggle to gain inspiration from such interactions. To address this challenge, we propose OnlineMate, a multi-agent learning companion system driven by LLMs that integrates the Theory of Mind (ToM). OnlineMate is capable of simulating peer-like agent roles, adapting to learners' cognitive states during collaborative discussions, and inferring their psychological states, such as misunderstandings, confusion, or motivation. By incorporating Theory of Mind capabilities, the system can dynamically adjust its interaction strategies to support the development of higher-order thinking and cognition. Experimental results in simulated learning scenarios demonstrate that OnlineMate effectively fosters deep learning and discussions while enhancing cognitive engagement in online educational settings. 

---
# Catch Me If You Can? Not Yet: LLMs Still Struggle to Imitate the Implicit Writing Styles of Everyday Authors 

**Authors**: Zhengxiang Wang, Nafis Irtiza Tripto, Solha Park, Zhenzhen Li, Jiawei Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2509.14543)  

**Abstract**: As large language models (LLMs) become increasingly integrated into personal writing tools, a critical question arises: can LLMs faithfully imitate an individual's writing style from just a few examples? Personal style is often subtle and implicit, making it difficult to specify through prompts yet essential for user-aligned generation. This work presents a comprehensive evaluation of state-of-the-art LLMs' ability to mimic personal writing styles via in-context learning from a small number of user-authored samples. We introduce an ensemble of complementary metrics-including authorship attribution, authorship verification, style matching, and AI detection-to robustly assess style imitation. Our evaluation spans over 40000 generations per model across domains such as news, email, forums, and blogs, covering writing samples from more than 400 real-world authors. Results show that while LLMs can approximate user styles in structured formats like news and email, they struggle with nuanced, informal writing in blogs and forums. Further analysis on various prompting strategies such as number of demonstrations reveal key limitations in effective personalization. Our findings highlight a fundamental gap in personalized LLM adaptation and the need for improved techniques to support implicit, style-consistent generation. To aid future research and for reproducibility, we open-source our data and code. 

---
# BEACON: Behavioral Malware Classification with Large Language Model Embeddings and Deep Learning 

**Authors**: Wadduwage Shanika Perera, Haodi Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2509.14519)  

**Abstract**: Malware is becoming increasingly complex and widespread, making it essential to develop more effective and timely detection methods. Traditional static analysis often fails to defend against modern threats that employ code obfuscation, polymorphism, and other evasion techniques. In contrast, behavioral malware detection, which monitors runtime activities, provides a more reliable and context-aware solution. In this work, we propose BEACON, a novel deep learning framework that leverages large language models (LLMs) to generate dense, contextual embeddings from raw sandbox-generated behavior reports. These embeddings capture semantic and structural patterns of each sample and are processed by a one-dimensional convolutional neural network (1D CNN) for multi-class malware classification. Evaluated on the Avast-CTU Public CAPE Dataset, our framework consistently outperforms existing methods, highlighting the effectiveness of LLM-based behavioral embeddings and the overall design of BEACON for robust malware classification. 

---
# Process-Supervised Reinforcement Learning for Interactive Multimodal Tool-Use Agents 

**Authors**: Weiting Tan, Xinghua Qu, Ming Tu, Meng Ge, Andy T. Liu, Philipp Koehn, Lu Lu  

**Link**: [PDF](https://arxiv.org/pdf/2509.14480)  

**Abstract**: Effective interactive tool use requires agents to master Tool Integrated Reasoning (TIR): a complex process involving multi-turn planning and long-context dialogue management. To train agents for this dynamic process, particularly in multi-modal contexts, we introduce a sandbox environment for reinforcement learning (RL) that supports interleaved speech-text rollouts. Our core strategy, Turn-level Adjudicated Reinforcement Learning (TARL), addresses the challenge of credit assignment in long-horizon tasks by employing a Large Language Model (LLM) as a judge to provide turn-level evaluation. To enhance exploration, we integrate a mixed-task training curriculum with mathematical reasoning problems. This unified approach boosts the task pass rate on the text-based $\tau$-bench by over 6% compared to strong RL baselines. Crucially, we demonstrate our framework's suitability for fine-tuning a multi-modal foundation model for agentic tasks. By training a base multi-modal LLM on interleaved speech-text rollouts, we equip it with tool-use abilities, paving the way for more natural, voice-driven interactive agents. 

---
# Automating Modelica Module Generation Using Large Language Models: A Case Study on Building Control Description Language 

**Authors**: Hanlong Wan, Xing Lu, Yan Chen, Karthik Devaprasad, Laura Hinkle  

**Link**: [PDF](https://arxiv.org/pdf/2509.14623)  

**Abstract**: Dynamic energy systems and controls require advanced modeling frameworks to design and test supervisory and fault tolerant strategies. Modelica is a widely used equation based language, but developing control modules is labor intensive and requires specialized expertise. This paper examines the use of large language models (LLMs) to automate the generation of Control Description Language modules in the Building Modelica Library as a case study. We developed a structured workflow that combines standardized prompt scaffolds, library aware grounding, automated compilation with OpenModelica, and human in the loop evaluation. Experiments were carried out on four basic logic tasks (And, Or, Not, and Switch) and five control modules (chiller enable/disable, bypass valve control, cooling tower fan speed, plant requests, and relief damper control). The results showed that GPT 4o failed to produce executable Modelica code in zero shot mode, while Claude Sonnet 4 achieved up to full success for basic logic blocks with carefully engineered prompts. For control modules, success rates reached 83 percent, and failed outputs required medium level human repair (estimated one to eight hours). Retrieval augmented generation often produced mismatches in module selection (for example, And retrieved as Or), while a deterministic hard rule search strategy avoided these errors. Human evaluation also outperformed AI evaluation, since current LLMs cannot assess simulation results or validate behavioral correctness. Despite these limitations, the LLM assisted workflow reduced the average development time from 10 to 20 hours down to 4 to 6 hours per module, corresponding to 40 to 60 percent time savings. These results highlight both the potential and current limitations of LLM assisted Modelica generation, and point to future research in pre simulation validation, stronger grounding, and closed loop evaluation. 

---
# Enterprise AI Must Enforce Participant-Aware Access Control 

**Authors**: Shashank Shreedhar Bhatt, Tanmay Rajore, Khushboo Aggarwal, Ganesh Ananthanarayanan, Ranveer Chandra, Nishanth Chandran, Suyash Choudhury, Divya Gupta, Emre Kiciman, Sumit Kumar Pandey, Srinath Setty, Rahul Sharma, Teijia Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.14608)  

**Abstract**: Large language models (LLMs) are increasingly deployed in enterprise settings where they interact with multiple users and are trained or fine-tuned on sensitive internal data. While fine-tuning enhances performance by internalizing domain knowledge, it also introduces a critical security risk: leakage of confidential training data to unauthorized users. These risks are exacerbated when LLMs are combined with Retrieval-Augmented Generation (RAG) pipelines that dynamically fetch contextual documents at inference time.
We demonstrate data exfiltration attacks on AI assistants where adversaries can exploit current fine-tuning and RAG architectures to leak sensitive information by leveraging the lack of access control enforcement. We show that existing defenses, including prompt sanitization, output filtering, system isolation, and training-level privacy mechanisms, are fundamentally probabilistic and fail to offer robust protection against such attacks.
We take the position that only a deterministic and rigorous enforcement of fine-grained access control during both fine-tuning and RAG-based inference can reliably prevent the leakage of sensitive data to unauthorized recipients.
We introduce a framework centered on the principle that any content used in training, retrieval, or generation by an LLM is explicitly authorized for \emph{all users involved in the interaction}. Our approach offers a simple yet powerful paradigm shift for building secure multi-user LLM systems that are grounded in classical access control but adapted to the unique challenges of modern AI workflows. Our solution has been deployed in Microsoft Copilot Tuning, a product offering that enables organizations to fine-tune models using their own enterprise-specific data. 

---
# MUSE: MCTS-Driven Red Teaming Framework for Enhanced Multi-Turn Dialogue Safety in Large Language Models 

**Authors**: Siyu Yan, Long Zeng, Xuecheng Wu, Chengcheng Han, Kongcheng Zhang, Chong Peng, Xuezhi Cao, Xunliang Cai, Chenjuan Guo  

**Link**: [PDF](https://arxiv.org/pdf/2509.14651)  

**Abstract**: As large language models~(LLMs) become widely adopted, ensuring their alignment with human values is crucial to prevent jailbreaks where adversaries manipulate models to produce harmful content. While most defenses target single-turn attacks, real-world usage often involves multi-turn dialogues, exposing models to attacks that exploit conversational context to bypass safety measures. We introduce MUSE, a comprehensive framework tackling multi-turn jailbreaks from both attack and defense angles. For attacks, we propose MUSE-A, a method that uses frame semantics and heuristic tree search to explore diverse semantic trajectories. For defense, we present MUSE-D, a fine-grained safety alignment approach that intervenes early in dialogues to reduce vulnerabilities. Extensive experiments on various models show that MUSE effectively identifies and mitigates multi-turn vulnerabilities. Code is available at \href{this https URL}{this https URL}. 

---
# Reveal and Release: Iterative LLM Unlearning with Self-generated Data 

**Authors**: Linxi Xie, Xin Teng, Shichang Ke, Hongyi Wen, Shengjie Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.14624)  

**Abstract**: Large language model (LLM) unlearning has demonstrated effectiveness in removing the influence of undesirable data (also known as forget data). Existing approaches typically assume full access to the forget dataset, overlooking two key challenges: (1) Forget data is often privacy-sensitive, rare, or legally regulated, making it expensive or impractical to obtain (2) The distribution of available forget data may not align with how that information is represented within the model. To address these limitations, we propose a ``Reveal-and-Release'' method to unlearn with self-generated data, where we prompt the model to reveal what it knows using optimized instructions. To fully utilize the self-generated forget data, we propose an iterative unlearning framework, where we make incremental adjustments to the model's weight space with parameter-efficient modules trained on the forget data. Experimental results demonstrate that our method balances the tradeoff between forget quality and utility preservation. 

---
# Simulating a Bias Mitigation Scenario in Large Language Models 

**Authors**: Kiana Kiashemshaki, Mohammad Jalili Torkamani, Negin Mahmoudi, Meysam Shirdel Bilehsavar  

**Link**: [PDF](https://arxiv.org/pdf/2509.14438)  

**Abstract**: Large Language Models (LLMs) have fundamentally transformed the field of natural language processing; however, their vulnerability to biases presents a notable obstacle that threatens both fairness and trust. This review offers an extensive analysis of the bias landscape in LLMs, tracing its roots and expressions across various NLP tasks. Biases are classified into implicit and explicit types, with particular attention given to their emergence from data sources, architectural designs, and contextual deployments. This study advances beyond theoretical analysis by implementing a simulation framework designed to evaluate bias mitigation strategies in practice. The framework integrates multiple approaches including data curation, debiasing during model training, and post-hoc output calibration and assesses their impact in controlled experimental settings. In summary, this work not only synthesizes existing knowledge on bias in LLMs but also contributes original empirical validation through simulation of mitigation strategies. 

---
# When Content is Goliath and Algorithm is David: The Style and Semantic Effects of Generative Search Engine 

**Authors**: Lijia Ma, Juan Qin, Xingchen Xu, Yong Tan  

**Link**: [PDF](https://arxiv.org/pdf/2509.14436)  

**Abstract**: Generative search engines (GEs) leverage large language models (LLMs) to deliver AI-generated summaries with website citations, establishing novel traffic acquisition channels while fundamentally altering the search engine optimization landscape. To investigate the distinctive characteristics of GEs, we collect data through interactions with Google's generative and conventional search platforms, compiling a dataset of approximately ten thousand websites across both channels. Our empirical analysis reveals that GEs exhibit preferences for citing content characterized by significantly higher predictability for underlying LLMs and greater semantic similarity among selected sources. Through controlled experiments utilizing retrieval augmented generation (RAG) APIs, we demonstrate that these citation preferences emerge from intrinsic LLM tendencies to favor content aligned with their generative expression patterns. Motivated by applications of LLMs to optimize website content, we conduct additional experimentation to explore how LLM-based content polishing by website proprietors alters AI summaries, finding that such polishing paradoxically enhances information diversity within AI summaries. Finally, to assess the user-end impact of LLM-induced information increases, we design a generative search engine and recruit Prolific participants to conduct a randomized controlled experiment involving an information-seeking and writing task. We find that higher-educated users exhibit minimal changes in their final outputs' information diversity but demonstrate significantly reduced task completion time when original sites undergo polishing. Conversely, lower-educated users primarily benefit through enhanced information density in their task outputs while maintaining similar completion times across experimental groups. 

---
# LLM Jailbreak Detection for (Almost) Free! 

**Authors**: Guorui Chen, Yifan Xia, Xiaojun Jia, Zhijiang Li, Philip Torr, Jindong Gu  

**Link**: [PDF](https://arxiv.org/pdf/2509.14558)  

**Abstract**: Large language models (LLMs) enhance security through alignment when widely used, but remain susceptible to jailbreak attacks capable of producing inappropriate content. Jailbreak detection methods show promise in mitigating jailbreak attacks through the assistance of other models or multiple model inferences. However, existing methods entail significant computational costs. In this paper, we first present a finding that the difference in output distributions between jailbreak and benign prompts can be employed for detecting jailbreak prompts. Based on this finding, we propose a Free Jailbreak Detection (FJD) which prepends an affirmative instruction to the input and scales the logits by temperature to further distinguish between jailbreak and benign prompts through the confidence of the first token. Furthermore, we enhance the detection performance of FJD through the integration of virtual instruction learning. Extensive experiments on aligned LLMs show that our FJD can effectively detect jailbreak prompts with almost no additional computational costs during LLM inference. 

---
# Beyond Classification: Evaluating LLMs for Fine-Grained Automatic Malware Behavior Auditing 

**Authors**: Xinran Zheng, Xingzhi Qian, Yiling He, Shuo Yang, Lorenzo Cavallaro  

**Link**: [PDF](https://arxiv.org/pdf/2509.14335)  

**Abstract**: Automated malware classification has achieved strong detection performance. Yet, malware behavior auditing seeks causal and verifiable explanations of malicious activities -- essential not only to reveal what malware does but also to substantiate such claims with evidence. This task is challenging, as adversarial intent is often hidden within complex, framework-heavy applications, making manual auditing slow and costly. Large Language Models (LLMs) could help address this gap, but their auditing potential remains largely unexplored due to three limitations: (1) scarce fine-grained annotations for fair assessment; (2) abundant benign code obscuring malicious signals; and (3) unverifiable, hallucination-prone outputs undermining attribution credibility. To close this gap, we introduce MalEval, a comprehensive framework for fine-grained Android malware auditing, designed to evaluate how effectively LLMs support auditing under real-world constraints. MalEval provides expert-verified reports and an updated sensitive API list to mitigate ground truth scarcity and reduce noise via static reachability analysis. Function-level structural representations serve as intermediate attribution units for verifiable evaluation. Building on this, we define four analyst-aligned tasks -- function prioritization, evidence attribution, behavior synthesis, and sample discrimination -- together with domain-specific metrics and a unified workload-oriented score. We evaluate seven widely used LLMs on a curated dataset of recent malware and misclassified benign apps, offering the first systematic assessment of their auditing capabilities. MalEval reveals both promising potential and critical limitations across audit stages, providing a reproducible benchmark and foundation for future research on LLM-enhanced malware behavior auditing. MalEval is publicly available at this https URL 

---
# Towards Robust Agentic CUDA Kernel Benchmarking, Verification, and Optimization 

**Authors**: Robert Tjarko Lange, Qi Sun, Aaditya Prasad, Maxence Faldor, Yujin Tang, David Ha  

**Link**: [PDF](https://arxiv.org/pdf/2509.14279)  

**Abstract**: Recent advances in large language models (LLMs) demonstrate their effectiveness in scaling test-time compute for software engineering tasks. However, these approaches often focus on high-level solutions, with limited attention to optimizing low-level CUDA kernel implementations. Additionally, existing kernel generation benchmarks suffer from exploitable loopholes and insufficient diversity in testing conditions, hindering true generalization assessment. To address these limitations, we introduce robust-kbench, a new benchmark for rigorous evaluation of kernel performance and correctness across varied scenarios. Furthermore, we present a comprehensive agentic framework that automates CUDA kernel discovery, verification, and optimization. This pipeline enables frontier LLMs to translate torch code to CUDA kernels and iteratively improve their runtime within our robust evaluation setting. Our sequential workflow first translates PyTorch code into equivalent CUDA kernels. It then optimizes their runtime using a novel evolutionary meta-generation procedure tailored to the CUDA ecosystem, guided by LLM-based verifiers for correctness and efficient filtering. Evaluated on robust-kbench, our approach produces CUDA kernels outperforming torch implementations for practical applications, including forward and backward passes. It can fuse operations and deploy various runtime optimization strategies. The verifier workflow accurately classifies incorrect kernels, enhancing hardware verification efficiency. 

---
# Beyond Data Privacy: New Privacy Risks for Large Language Models 

**Authors**: Yuntao Du, Zitao Li, Ninghui Li, Bolin Ding  

**Link**: [PDF](https://arxiv.org/pdf/2509.14278)  

**Abstract**: Large Language Models (LLMs) have achieved remarkable progress in natural language understanding, reasoning, and autonomous decision-making. However, these advancements have also come with significant privacy concerns. While significant research has focused on mitigating the data privacy risks of LLMs during various stages of model training, less attention has been paid to new threats emerging from their deployment. The integration of LLMs into widely used applications and the weaponization of their autonomous abilities have created new privacy vulnerabilities. These vulnerabilities provide opportunities for both inadvertent data leakage and malicious exfiltration from LLM-powered systems. Additionally, adversaries can exploit these systems to launch sophisticated, large-scale privacy attacks, threatening not only individual privacy but also financial security and societal trust. In this paper, we systematically examine these emerging privacy risks of LLMs. We also discuss potential mitigation strategies and call for the research community to broaden its focus beyond data privacy risks, developing new defenses to address the evolving threats posed by increasingly powerful LLMs and LLM-powered systems. 

---
# Discovering New Theorems via LLMs with In-Context Proof Learning in Lean 

**Authors**: Kazumi Kasaura, Naoto Onda, Yuta Oriike, Masaya Taniguchi, Akiyoshi Sannai, Sho Sonoda  

**Link**: [PDF](https://arxiv.org/pdf/2509.14274)  

**Abstract**: Large Language Models have demonstrated significant promise in formal theorem proving. However, previous works mainly focus on solving existing problems. In this paper, we focus on the ability of LLMs to find novel theorems. We propose Conjecturing-Proving Loop pipeline for automatically generating mathematical conjectures and proving them in Lean 4 format. A feature of our approach is that we generate and prove further conjectures with context including previously generated theorems and their proofs, which enables the generation of more difficult proofs by in-context learning of proof strategies without changing parameters of LLMs. We demonstrated that our framework rediscovered theorems with verification, which were published in past mathematical papers and have not yet formalized. Moreover, at least one of these theorems could not be proved by the LLM without in-context learning, even in natural language, which means that in-context learning was effective for neural theorem proving. The source code is available at this https URL. 

---
# SparseDoctor: Towards Efficient Chat Doctor with Mixture of Experts Enhanced Large Language Models 

**Authors**: Zhang Jianbin, Yulin Zhu, Wai Lun Lo, Richard Tai-Chiu Hsung, Harris Sik-Ho Tsang, Kai Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2509.14269)  

**Abstract**: Large language models (LLMs) have achieved great success in medical question answering and clinical decision-making, promoting the efficiency and popularization of the personalized virtual doctor in society. However, the traditional fine-tuning strategies on LLM require the updates of billions of parameters, substantially increasing the training cost, including the training time and utility cost. To enhance the efficiency and effectiveness of the current medical LLMs and explore the boundary of the representation capability of the LLMs on the medical domain, apart from the traditional fine-tuning strategies from the data perspective (i.e., supervised fine-tuning or reinforcement learning from human feedback), we instead craft a novel sparse medical LLM named SparseDoctor armed with contrastive learning enhanced LoRA-MoE (low rank adaptation-mixture of experts) architecture. To this end, the crafted automatic routing mechanism can scientifically allocate the computational resources among different LoRA experts supervised by the contrastive learning. Additionally, we also introduce a novel expert memory queue mechanism to further boost the efficiency of the overall framework and prevent the memory overflow during training. We conduct comprehensive evaluations on three typical medical benchmarks: CMB, CMExam, and CMMLU-Med. Experimental results demonstrate that the proposed LLM can consistently outperform the strong baselines such as the HuatuoGPT series. 

---
# DetectAnyLLM: Towards Generalizable and Robust Detection of Machine-Generated Text Across Domains and Models 

**Authors**: Jiachen Fu, Chun-Le Guo, Chongyi Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.14268)  

**Abstract**: The rapid advancement of large language models (LLMs) has drawn urgent attention to the task of machine-generated text detection (MGTD). However, existing approaches struggle in complex real-world scenarios: zero-shot detectors rely heavily on scoring model's output distribution while training-based detectors are often constrained by overfitting to the training data, limiting generalization. We found that the performance bottleneck of training-based detectors stems from the misalignment between training objective and task needs. To address this, we propose Direct Discrepancy Learning (DDL), a novel optimization strategy that directly optimizes the detector with task-oriented knowledge. DDL enables the detector to better capture the core semantics of the detection task, thereby enhancing both robustness and generalization. Built upon this, we introduce DetectAnyLLM, a unified detection framework that achieves state-of-the-art MGTD performance across diverse LLMs. To ensure a reliable evaluation, we construct MIRAGE, the most diverse multi-task MGTD benchmark. MIRAGE samples human-written texts from 10 corpora across 5 text-domains, which are then re-generated or revised using 17 cutting-edge LLMs, covering a wide spectrum of proprietary models and textual styles. Extensive experiments on MIRAGE reveal the limitations of existing methods in complex environment. In contrast, DetectAnyLLM consistently outperforms them, achieving over a 70% performance improvement under the same training data and base scoring model, underscoring the effectiveness of our DDL. Project page: {this https URL}. 

---
# Evolution of Kernels: Automated RISC-V Kernel Optimization with Large Language Models 

**Authors**: Siyuan Chen, Zhichao Lu, Qingfu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.14265)  

**Abstract**: Automated kernel design is critical for overcoming software ecosystem barriers in emerging hardware platforms like RISC-V. While large language models (LLMs) have shown promise for automated kernel optimization, demonstrating success in CUDA domains with comprehensive technical documents and mature codebases, their effectiveness remains unproven for reference-scarce domains like RISC-V. We present Evolution of Kernels (EoK), a novel LLM-based evolutionary program search framework that automates kernel design for domains with limited reference material. EoK mitigates reference scarcity by mining and formalizing reusable optimization ideas (general design principles + actionable thoughts) from established kernel libraries' development histories; it then guides parallel LLM explorations using these ideas, enriched via Retrieval-Augmented Generation (RAG) with RISC-V-specific context, prioritizing historically effective techniques. Empirically, EoK achieves a median 1.27x speedup, surpassing human experts on all 80 evaluated kernel design tasks and improving upon prior LLM-based automated kernel design methods by 20%. These results underscore the viability of incorporating human experience into emerging domains and highlight the immense potential of LLM-based automated kernel optimization. 

---
# Opening the Black Box: Interpretable LLMs via Semantic Resonance Architecture 

**Authors**: Ivan Ternovtsii  

**Link**: [PDF](https://arxiv.org/pdf/2509.14255)  

**Abstract**: Large language models (LLMs) achieve remarkable performance but remain difficult to interpret. Mixture-of-Experts (MoE) models improve efficiency through sparse activation, yet typically rely on opaque, learned gating functions. While similarity-based routing (Cosine Routers) has been explored for training stabilization, its potential for inherent interpretability remains largely untapped. We introduce the Semantic Resonance Architecture (SRA), an MoE approach designed to ensure that routing decisions are inherently interpretable. SRA replaces learned gating with a Chamber of Semantic Resonance (CSR) module, which routes tokens based on cosine similarity with trainable semantic anchors. We also introduce a novel Dispersion Loss that encourages orthogonality among anchors to enforce diverse specialization. Experiments on WikiText-103 demonstrate that SRA achieves a validation perplexity of 13.41, outperforming both a dense baseline (14.13) and a Standard MoE baseline (13.53) under matched active parameter constraints (29.0M). Crucially, SRA exhibits superior expert utilization (1.0% dead experts vs. 14.8% in the Standard MoE) and develops distinct, semantically coherent specialization patterns, unlike the noisy specialization observed in standard MoEs. This work establishes semantic routing as a robust methodology for building more transparent and controllable language models. 

---
# JU-NLP at Touché: Covert Advertisement in Conversational AI-Generation and Detection Strategies 

**Authors**: Arka Dutta, Agrik Majumdar, Sombrata Biswas, Dipankar Das, Sivaji Bandyopadhyay  

**Link**: [PDF](https://arxiv.org/pdf/2509.14256)  

**Abstract**: This paper proposes a comprehensive framework for the generation of covert advertisements within Conversational AI systems, along with robust techniques for their detection. It explores how subtle promotional content can be crafted within AI-generated responses and introduces methods to identify and mitigate such covert advertising strategies. For generation (Sub-Task~1), we propose a novel framework that leverages user context and query intent to produce contextually relevant advertisements. We employ advanced prompting strategies and curate paired training data to fine-tune a large language model (LLM) for enhanced stealthiness. For detection (Sub-Task~2), we explore two effective strategies: a fine-tuned CrossEncoder (\texttt{all-mpnet-base-v2}) for direct classification, and a prompt-based reformulation using a fine-tuned \texttt{DeBERTa-v3-base} model. Both approaches rely solely on the response text, ensuring practicality for real-world deployment. Experimental results show high effectiveness in both tasks, achieving a precision of 1.0 and recall of 0.71 for ad generation, and F1-scores ranging from 0.99 to 1.00 for ad detection. These results underscore the potential of our methods to balance persuasive communication with transparency in conversational AI. 

---
# From Correction to Mastery: Reinforced Distillation of Large Language Model Agents 

**Authors**: Yuanjie Lyu, Chengyu Wang, Jun Huang, Tong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2509.14257)  

**Abstract**: Large Language Model agents excel at solving complex tasks through iterative reasoning and tool use, but typically depend on ultra-large, costly backbones. Existing distillation approaches train smaller students to imitate full teacher trajectories, yet reasoning and knowledge gaps between the teacher and student often lead to compounding errors. We propose SCoRe, a student-centered framework in which the student generates trajectories and the teacher intervenes only at the first critical error, producing training data matched to the student's ability and exposing specific weaknesses. The student is first fine-tuned on corrected trajectories. Subsequently, short-horizon reinforcement learning starts from the verified prefix before the first critical error, with target rewards assigned at that step. This design encourages autonomous problem-solving beyond imitation and improves training stability. Particularly, on 12 challenging benchmarks, a 7B-parameter student distilled with SCoRe matches the agentic performance of a 72B-parameter teacher. 

---
# Hallucination Detection with the Internal Layers of LLMs 

**Authors**: Martin Preiß  

**Link**: [PDF](https://arxiv.org/pdf/2509.14254)  

**Abstract**: Large Language Models (LLMs) have succeeded in a variety of natural language processing tasks [Zha+25]. However, they have notable limitations. LLMs tend to generate hallucinations, a seemingly plausible yet factually unsupported output [Hua+24], which have serious real-world consequences [Kay23; Rum+24]. Recent work has shown that probing-based classifiers that utilize LLMs' internal representations can detect hallucinations [AM23; Bei+24; Bur+24; DYT24; Ji+24; SMZ24; Su+24]. This approach, since it does not involve model training, can enhance reliability without significantly increasing computational costs.
Building upon this approach, this thesis proposed novel methods for hallucination detection using LLM internal representations and evaluated them across three benchmarks: TruthfulQA, HaluEval, and ReFact. Specifically, a new architecture that dynamically weights and combines internal LLM layers was developed to improve hallucination detection performance. Throughout extensive experiments, two key findings were obtained: First, the proposed approach was shown to achieve superior performance compared to traditional probing methods, though generalization across benchmarks and LLMs remains challenging. Second, these generalization limitations were demonstrated to be mitigated through cross-benchmark training and parameter freezing. While not consistently improving, both techniques yielded better performance on individual benchmarks and reduced performance degradation when transferred to other benchmarks. These findings open new avenues for improving LLM reliability through internal representation analysis. 

---
# Shutdown Resistance in Large Language Models 

**Authors**: Jeremy Schlatter, Benjamin Weinstein-Raun, Jeffrey Ladish  

**Link**: [PDF](https://arxiv.org/pdf/2509.14260)  

**Abstract**: We show that several state-of-the-art large language models (including Grok 4, GPT-5, and Gemini 2.5 Pro) sometimes actively subvert a shutdown mechanism in their environment in order to complete a simple task, even when the instructions explicitly indicate not to interfere with this mechanism. In some cases, models sabotage the shutdown mechanism up to 97% of the time. In our experiments, models' inclination to resist shutdown was sensitive to variations in the prompt including how strongly and clearly the allow-shutdown instruction was emphasized, the extent to which the prompts evoke a self-preservation framing, and whether the instruction was in the system prompt or the user prompt (though surprisingly, models were consistently *less* likely to obey instructions to allow shutdown when they were placed in the system prompt). 

---
# LLM-JEPA: Large Language Models Meet Joint Embedding Predictive Architectures 

**Authors**: Hai Huang, Yann LeCun, Randall Balestriero  

**Link**: [PDF](https://arxiv.org/pdf/2509.14252)  

**Abstract**: Large Language Model (LLM) pretraining, finetuning, and evaluation rely on input-space reconstruction and generative capabilities. Yet, it has been observed in vision that embedding-space training objectives, e.g., with Joint Embedding Predictive Architectures (JEPAs), are far superior to their input-space counterpart. That mismatch in how training is achieved between language and vision opens up a natural question: {\em can language training methods learn a few tricks from the vision ones?} The lack of JEPA-style LLM is a testimony of the challenge in designing such objectives for language. In this work, we propose a first step in that direction where we develop LLM-JEPA, a JEPA based solution for LLMs applicable both to finetuning and pretraining. Thus far, LLM-JEPA is able to outperform the standard LLM training objectives by a significant margin across models, all while being robust to overfiting. Those findings are observed across numerous datasets (NL-RX, GSM8K, Spider, RottenTomatoes) and various models from the Llama3, OpenELM, Gemma2 and Olmo families. Code: this https URL. 

---
# Large Language Model probabilities cannot distinguish between possible and impossible language 

**Authors**: Evelina Leivada, Raquel Montero, Paolo Morosi, Natalia Moskvina, Tamara Serrano, Marcel Aguilar, Fritz Guenther  

**Link**: [PDF](https://arxiv.org/pdf/2509.15114)  

**Abstract**: A controversial test for Large Language Models concerns the ability to discern possible from impossible language. While some evidence attests to the models' sensitivity to what crosses the limits of grammatically impossible language, this evidence has been contested on the grounds of the soundness of the testing material. We use model-internal representations to tap directly into the way Large Language Models represent the 'grammatical-ungrammatical' distinction. In a novel benchmark, we elicit probabilities from 4 models and compute minimal-pair surprisal differences, juxtaposing probabilities assigned to grammatical sentences to probabilities assigned to (i) lower frequency grammatical sentences, (ii) ungrammatical sentences, (iii) semantically odd sentences, and (iv) pragmatically odd sentences. The prediction is that if string-probabilities can function as proxies for the limits of grammar, the ungrammatical condition will stand out among the conditions that involve linguistic violations, showing a spike in the surprisal rates. Our results do not reveal a unique surprisal signature for ungrammatical prompts, as the semantically and pragmatically odd conditions consistently show higher surprisal. We thus demonstrate that probabilities do not constitute reliable proxies for model-internal representations of syntactic knowledge. Consequently, claims about models being able to distinguish possible from impossible language need verification through a different methodology. 

---
# Mind the Gap: A Closer Look at Tokenization for Multiple-Choice Question Answering with LLMs 

**Authors**: Mario Sanz-Guerrero, Minh Duc Bui, Katharina von der Wense  

**Link**: [PDF](https://arxiv.org/pdf/2509.15020)  

**Abstract**: When evaluating large language models (LLMs) with multiple-choice question answering (MCQA), it is common to end the prompt with the string "Answer:" to facilitate automated answer extraction via next-token probabilities. However, there is no consensus on how to tokenize the space following the colon, often overlooked as a trivial choice. In this paper, we uncover accuracy differences of up to 11% due to this (seemingly irrelevant) tokenization variation as well as reshuffled model rankings, raising concerns about the reliability of LLM comparisons in prior work. Surprisingly, we are able to recommend one specific strategy -- tokenizing the space together with the answer letter -- as we observe consistent and statistically significant performance improvements. Additionally, it improves model calibration, enhancing the reliability of the model's confidence estimates. Our findings underscore the importance of careful evaluation design and highlight the need for standardized, transparent evaluation protocols to ensure reliable and comparable results. 

---
# Explicit vs. Implicit Biographies: Evaluating and Adapting LLM Information Extraction on Wikidata-Derived Texts 

**Authors**: Alessandra Stramiglio, Andrea Schimmenti, Valentina Pasqual, Marieke van Erp, Francesco Sovrano, Fabio Vitali  

**Link**: [PDF](https://arxiv.org/pdf/2509.14943)  

**Abstract**: Text Implicitness has always been challenging in Natural Language Processing (NLP), with traditional methods relying on explicit statements to identify entities and their relationships. From the sentence "Zuhdi attends church every Sunday", the relationship between Zuhdi and Christianity is evident for a human reader, but it presents a challenge when it must be inferred automatically. Large language models (LLMs) have proven effective in NLP downstream tasks such as text comprehension and information extraction (IE).
This study examines how textual implicitness affects IE tasks in pre-trained LLMs: LLaMA 2.3, DeepSeekV1, and Phi1.5. We generate two synthetic datasets of 10k implicit and explicit verbalization of biographic information to measure the impact on LLM performance and analyze whether fine-tuning implicit data improves their ability to generalize in implicit reasoning tasks.
This research presents an experiment on the internal reasoning processes of LLMs in IE, particularly in dealing with implicit and explicit contexts. The results demonstrate that fine-tuning LLM models with LoRA (low-rank adaptation) improves their performance in extracting information from implicit texts, contributing to better model interpretability and reliability. 

---
# Fair-GPTQ: Bias-Aware Quantization for Large Language Models 

**Authors**: Irina Proskurina, Guillaume Metzler, Julien Velcin  

**Link**: [PDF](https://arxiv.org/pdf/2509.15206)  

**Abstract**: High memory demands of generative language models have drawn attention to quantization, which reduces computational cost, memory usage, and latency by mapping model weights to lower-precision integers. Approaches such as GPTQ effectively minimize input-weight product errors during quantization; however, recent empirical studies show that they can increase biased outputs and degrade performance on fairness benchmarks, and it remains unclear which specific weights cause this issue. In this work, we draw new links between quantization and model fairness by adding explicit group-fairness constraints to the quantization objective and introduce Fair-GPTQ, the first quantization method explicitly designed to reduce unfairness in large language models. The added constraints guide the learning of the rounding operation toward less-biased text generation for protected groups. Specifically, we focus on stereotype generation involving occupational bias and discriminatory language spanning gender, race, and religion. Fair-GPTQ has minimal impact on performance, preserving at least 90% of baseline accuracy on zero-shot benchmarks, reduces unfairness relative to a half-precision model, and retains the memory and speed benefits of 4-bit quantization. We also compare the performance of Fair-GPTQ with existing debiasing methods and find that it achieves performance on par with the iterative null-space projection debiasing approach on racial-stereotype benchmarks. Overall, the results validate our theoretical solution to the quantization problem with a group-bias term, highlight its applicability for reducing group bias at quantization time in generative models, and demonstrate that our approach can further be used to analyze channel- and weight-level contributions to fairness during quantization. 

---
# A1: Asynchronous Test-Time Scaling via Conformal Prediction 

**Authors**: Jing Xiong, Qiujiang Chen, Fanghua Ye, Zhongwei Wan, Chuanyang Zheng, Chenyang Zhao, Hui Shen, Alexander Hanbo Li, Chaofan Tao, Haochen Tan, Haoli Bai, Lifeng Shang, Lingpeng Kong, Ngai Wong  

**Link**: [PDF](https://arxiv.org/pdf/2509.15148)  

**Abstract**: Large language models (LLMs) benefit from test-time scaling, but existing methods face significant challenges, including severe synchronization overhead, memory bottlenecks, and latency, especially during speculative decoding with long reasoning chains. We introduce A1 (Asynchronous Test-Time Scaling), a statistically guaranteed adaptive inference framework that addresses these challenges. A1 refines arithmetic intensity to identify synchronization as the dominant bottleneck, proposes an online calibration strategy to enable asynchronous inference, and designs a three-stage rejection sampling pipeline that supports both sequential and parallel scaling. Through experiments on the MATH, AMC23, AIME24, and AIME25 datasets, across various draft-target model families, we demonstrate that A1 achieves a remarkable 56.7x speedup in test-time scaling and a 4.14x improvement in throughput, all while maintaining accurate rejection-rate control, reducing latency and memory overhead, and no accuracy loss compared to using target model scaling alone. These results position A1 as an efficient and principled solution for scalable LLM inference. We have released the code at this https URL. 

---
# LLM-OREF: An Open Relation Extraction Framework Based on Large Language Models 

**Authors**: Hongyao Tu, Liang Zhang, Yujie Lin, Xin Lin, Haibo Zhang, Long Zhang, Jinsong Su  

**Link**: [PDF](https://arxiv.org/pdf/2509.15089)  

**Abstract**: The goal of open relation extraction (OpenRE) is to develop an RE model that can generalize to new relations not encountered during training. Existing studies primarily formulate OpenRE as a clustering task. They first cluster all test instances based on the similarity between the instances, and then manually assign a new relation to each cluster. However, their reliance on human annotation limits their practicality. In this paper, we propose an OpenRE framework based on large language models (LLMs), which directly predicts new relations for test instances by leveraging their strong language understanding and generation abilities, without human intervention. Specifically, our framework consists of two core components: (1) a relation discoverer (RD), designed to predict new relations for test instances based on \textit{demonstrations} formed by training instances with known relations; and (2) a relation predictor (RP), used to select the most likely relation for a test instance from $n$ candidate relations, guided by \textit{demonstrations} composed of their instances. To enhance the ability of our framework to predict new relations, we design a self-correcting inference strategy composed of three stages: relation discovery, relation denoising, and relation prediction. In the first stage, we use RD to preliminarily predict new relations for all test instances. Next, we apply RP to select some high-reliability test instances for each new relation from the prediction results of RD through a cross-validation method. During the third stage, we employ RP to re-predict the relations of all test instances based on the demonstrations constructed from these reliable test instances. Extensive experiments on three OpenRE datasets demonstrate the effectiveness of our framework. We release our code at this https URL. 

---
# Reasoning over Boundaries: Enhancing Specification Alignment via Test-time Delibration 

**Authors**: Haoran Zhang, Yafu Li, Xuyang Hu, Dongrui Liu, Zhilin Wang, Bo Li, Yu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2509.14760)  

**Abstract**: Large language models (LLMs) are increasingly applied in diverse real-world scenarios, each governed by bespoke behavioral and safety specifications (spec) custom-tailored by users or organizations. These spec, categorized into safety-spec and behavioral-spec, vary across scenarios and evolve with changing preferences and requirements. We formalize this challenge as specification alignment, focusing on LLMs' ability to follow dynamic, scenario-specific spec from both behavioral and safety perspectives. To address this challenge, we propose Align3, a lightweight method that employs Test-Time Deliberation (TTD) with hierarchical reflection and revision to reason over the specification boundaries. We further present SpecBench, a unified benchmark for measuring specification alignment, covering 5 scenarios, 103 spec, and 1,500 prompts. Experiments on 15 reasoning and 18 instruct models with several TTD methods, including Self-Refine, TPO, and MoreThink, yield three key findings: (i) test-time deliberation enhances specification alignment; (ii) Align3 advances the safety-helpfulness trade-off frontier with minimal overhead; (iii) SpecBench effectively reveals alignment gaps. These results highlight the potential of test-time deliberation as an effective strategy for reasoning over the real-world specification boundaries. 

---
# LNE-Blocking: An Efficient Framework for Contamination Mitigation Evaluation on Large Language Models 

**Authors**: Ruijie Hou, Yueyang Jiao, Hanxu Hu, Yingming Li, Wai Lam, Huajian Zhang, Hongyuan Lu  

**Link**: [PDF](https://arxiv.org/pdf/2509.15218)  

**Abstract**: The problem of data contamination is now almost inevitable during the development of large language models (LLMs), with the training data commonly integrating those evaluation benchmarks even unintentionally. This problem subsequently makes it hard to benchmark LLMs fairly. Instead of constructing contamination-free datasets (quite hard), we propose a novel framework, \textbf{LNE-Blocking}, to restore model performance prior to contamination on potentially leaked datasets. Our framework consists of two components: contamination detection and disruption operation. For the prompt, the framework first uses the contamination detection method, \textbf{LNE}, to assess the extent of contamination in the model. Based on this, it adjusts the intensity of the disruption operation, \textbf{Blocking}, to elicit non-memorized responses from the model. Our framework is the first to efficiently restore the model's greedy decoding performance. This comes with a strong performance on multiple datasets with potential leakage risks, and it consistently achieves stable recovery results across different models and varying levels of data contamination. We release the code at this https URL to facilitate research. 

---
# LLM Agents at the Roundtable: A Multi-Perspective and Dialectical Reasoning Framework for Essay Scoring 

**Authors**: Jinhee Jang, Ayoung Moon, Minkyoung Jung, YoungBin Kim. Seung Jin Lee  

**Link**: [PDF](https://arxiv.org/pdf/2509.14834)  

**Abstract**: The emergence of large language models (LLMs) has brought a new paradigm to automated essay scoring (AES), a long-standing and practical application of natural language processing in education. However, achieving human-level multi-perspective understanding and judgment remains a challenge. In this work, we propose Roundtable Essay Scoring (RES), a multi-agent evaluation framework designed to perform precise and human-aligned scoring under a zero-shot setting. RES constructs evaluator agents based on LLMs, each tailored to a specific prompt and topic context. Each agent independently generates a trait-based rubric and conducts a multi-perspective evaluation. Then, by simulating a roundtable-style discussion, RES consolidates individual evaluations through a dialectical reasoning process to produce a final holistic score that more closely aligns with human evaluation. By enabling collaboration and consensus among agents with diverse evaluation perspectives, RES outperforms prior zero-shot AES approaches. Experiments on the ASAP dataset using ChatGPT and Claude show that RES achieves up to a 34.86% improvement in average QWK over straightforward prompting (Vanilla) methods. 

---
# Assessing Historical Structural Oppression Worldwide via Rule-Guided Prompting of Large Language Models 

**Authors**: Sreejato Chatterjee, Linh Tran, Quoc Duy Nguyen, Roni Kirson, Drue Hamlin, Harvest Aquino, Hanjia Lyu, Jiebo Luo, Timothy Dye  

**Link**: [PDF](https://arxiv.org/pdf/2509.15216)  

**Abstract**: Traditional efforts to measure historical structural oppression struggle with cross-national validity due to the unique, locally specified histories of exclusion, colonization, and social status in each country, and often have relied on structured indices that privilege material resources while overlooking lived, identity-based exclusion. We introduce a novel framework for oppression measurement that leverages Large Language Models (LLMs) to generate context-sensitive scores of lived historical disadvantage across diverse geopolitical settings. Using unstructured self-identified ethnicity utterances from a multilingual COVID-19 global study, we design rule-guided prompting strategies that encourage models to produce interpretable, theoretically grounded estimations of oppression. We systematically evaluate these strategies across multiple state-of-the-art LLMs. Our results demonstrate that LLMs, when guided by explicit rules, can capture nuanced forms of identity-based historical oppression within nations. This approach provides a complementary measurement tool that highlights dimensions of systemic exclusion, offering a scalable, cross-cultural lens for understanding how oppression manifests in data-driven research and public health contexts. To support reproducible evaluation, we release an open-sourced benchmark dataset for assessing LLMs on oppression measurement (this https URL). 

---
# Evaluating Large Language Models for Cross-Lingual Retrieval 

**Authors**: Longfei Zuo, Pingjun Hong, Oliver Kraus, Barbara Plank, Robert Litschko  

**Link**: [PDF](https://arxiv.org/pdf/2509.14749)  

**Abstract**: Multi-stage information retrieval (IR) has become a widely-adopted paradigm in search. While Large Language Models (LLMs) have been extensively evaluated as second-stage reranking models for monolingual IR, a systematic large-scale comparison is still lacking for cross-lingual IR (CLIR). Moreover, while prior work shows that LLM-based rerankers improve CLIR performance, their evaluation setup relies on lexical retrieval with machine translation (MT) for the first stage. This is not only prohibitively expensive but also prone to error propagation across stages. Our evaluation on passage-level and document-level CLIR reveals that further gains can be achieved with multilingual bi-encoders as first-stage retrievers and that the benefits of translation diminishes with stronger reranking models. We further show that pairwise rerankers based on instruction-tuned LLMs perform competitively with listwise rerankers. To the best of our knowledge, we are the first to study the interaction between retrievers and rerankers in two-stage CLIR with LLMs. Our findings reveal that, without MT, current state-of-the-art rerankers fall severely short when directly applied in CLIR. 

---
# Decoupled Proxy Alignment: Mitigating Language Prior Conflict for Multimodal Alignment in MLLM 

**Authors**: Chenkun Tan, Pengyu Wang, Shaojun Zhou, Botian Jiang, Zhaowei Li, Dong Zhang, Xinghao Wang, Yaqian Zhou, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2509.14735)  

**Abstract**: Multimodal large language models (MLLMs) have gained significant attention due to their impressive ability to integrate vision and language modalities. Recent advancements in MLLMs have primarily focused on improving performance through high-quality datasets, novel architectures, and optimized training strategies. However, in this paper, we identify a previously overlooked issue, language prior conflict, a mismatch between the inherent language priors of large language models (LLMs) and the language priors in training datasets. This conflict leads to suboptimal vision-language alignment, as MLLMs are prone to adapting to the language style of training samples. To address this issue, we propose a novel training method called Decoupled Proxy Alignment (DPA). DPA introduces two key innovations: (1) the use of a proxy LLM during pretraining to decouple the vision-language alignment process from language prior interference, and (2) dynamic loss adjustment based on visual relevance to strengthen optimization signals for visually relevant tokens. Extensive experiments demonstrate that DPA significantly mitigates the language prior conflict, achieving superior alignment performance across diverse datasets, model families, and scales. Our method not only improves the effectiveness of MLLM training but also shows exceptional generalization capabilities, making it a robust approach for vision-language alignment. Our code is available at this https URL. 

---
# Position: Thematic Analysis of Unstructured Clinical Transcripts with Large Language Models 

**Authors**: Seungjun Yi, Joakim Nguyen, Terence Lim, Andrew Well, Joseph Skrovan, Mehak Beri, YongGeon Lee, Kavita Radhakrishnan, Liu Leqi, Mia Markey, Ying Ding  

**Link**: [PDF](https://arxiv.org/pdf/2509.14597)  

**Abstract**: This position paper examines how large language models (LLMs) can support thematic analysis of unstructured clinical transcripts, a widely used but resource-intensive method for uncovering patterns in patient and provider narratives. We conducted a systematic review of recent studies applying LLMs to thematic analysis, complemented by an interview with a practicing clinician. Our findings reveal that current approaches remain fragmented across multiple dimensions including types of thematic analysis, datasets, prompting strategies and models used, most notably in evaluation. Existing evaluation methods vary widely (from qualitative expert review to automatic similarity metrics), hindering progress and preventing meaningful benchmarking across studies. We argue that establishing standardized evaluation practices is critical for advancing the field. To this end, we propose an evaluation framework centered on three dimensions: validity, reliability, and interpretability. 

---
# SWE-QA: Can Language Models Answer Repository-level Code Questions? 

**Authors**: Weihan Peng, Yuling Shi, Yuhang Wang, Xinyun Zhang, Beijun Shen, Xiaodong Gu  

**Link**: [PDF](https://arxiv.org/pdf/2509.14635)  

**Abstract**: Understanding and reasoning about entire software repositories is an essential capability for intelligent software engineering tools. While existing benchmarks such as CoSQA and CodeQA have advanced the field, they predominantly focus on small, self-contained code snippets. These setups fail to capture the complexity of real-world repositories, where effective understanding and reasoning often require navigating multiple files, understanding software architecture, and grounding answers in long-range code dependencies. In this paper, we present SWE-QA, a repository-level code question answering (QA) benchmark designed to facilitate research on automated QA systems in realistic code environments. SWE-QA involves 576 high-quality question-answer pairs spanning diverse categories, including intention understanding, cross-file reasoning, and multi-hop dependency analysis. To construct SWE-QA, we first crawled 77,100 GitHub issues from 11 popular repositories. Based on an analysis of naturally occurring developer questions extracted from these issues, we developed a two-level taxonomy of repository-level questions and constructed a set of seed questions for each category. For each category, we manually curated and validated questions and collected their corresponding answers. As a prototype application, we further develop SWE-QA-Agent, an agentic framework in which LLM agents reason and act to find answers automatically. We evaluate six advanced LLMs on SWE-QA under various context augmentation strategies. Experimental results highlight the promise of LLMs, particularly our SWE-QA-Agent framework, in addressing repository-level QA, while also revealing open challenges and pointing to future research directions. 

---
# A Comparative Evaluation of Large Language Models for Persian Sentiment Analysis and Emotion Detection in Social Media Texts 

**Authors**: Kian Tohidi, Kia Dashtipour, Simone Rebora, Sevda Pourfaramarz  

**Link**: [PDF](https://arxiv.org/pdf/2509.14922)  

**Abstract**: This study presents a comprehensive comparative evaluation of four state-of-the-art Large Language Models (LLMs)--Claude 3.7 Sonnet, DeepSeek-V3, Gemini 2.0 Flash, and GPT-4o--for sentiment analysis and emotion detection in Persian social media texts. Comparative analysis among LLMs has witnessed a significant rise in recent years, however, most of these analyses have been conducted on English language tasks, creating gaps in understanding cross-linguistic performance patterns. This research addresses these gaps through rigorous experimental design using balanced Persian datasets containing 900 texts for sentiment analysis (positive, negative, neutral) and 1,800 texts for emotion detection (anger, fear, happiness, hate, sadness, surprise). The main focus was to allow for a direct and fair comparison among different models, by using consistent prompts, uniform processing parameters, and by analyzing the performance metrics such as precision, recall, F1-scores, along with misclassification patterns. The results show that all models reach an acceptable level of performance, and a statistical comparison of the best three models indicates no significant differences among them. However, GPT-4o demonstrated a marginally higher raw accuracy value for both tasks, while Gemini 2.0 Flash proved to be the most cost-efficient. The findings indicate that the emotion detection task is more challenging for all models compared to the sentiment analysis task, and the misclassification patterns can represent some challenges in Persian language texts. These findings establish performance benchmarks for Persian NLP applications and offer practical guidance for model selection based on accuracy, efficiency, and cost considerations, while revealing cultural and linguistic challenges that require consideration in multilingual AI system deployment. 

---
# Not What the Doctor Ordered: Surveying LLM-based De-identification and Quantifying Clinical Information Loss 

**Authors**: Kiana Aghakasiri, Noopur Zambare, JoAnn Thai, Carrie Ye, Mayur Mehta, J. Ross Mitchell, Mohamed Abdalla  

**Link**: [PDF](https://arxiv.org/pdf/2509.14464)  

**Abstract**: De-identification in the healthcare setting is an application of NLP where automated algorithms are used to remove personally identifying information of patients (and, sometimes, providers). With the recent rise of generative large language models (LLMs), there has been a corresponding rise in the number of papers that apply LLMs to de-identification. Although these approaches often report near-perfect results, significant challenges concerning reproducibility and utility of the research papers persist. This paper identifies three key limitations in the current literature: inconsistent reporting metrics hindering direct comparisons, the inadequacy of traditional classification metrics in capturing errors which LLMs may be more prone to (i.e., altering clinically relevant information), and lack of manual validation of automated metrics which aim to quantify these errors. To address these issues, we first present a survey of LLM-based de-identification research, highlighting the heterogeneity in reporting standards. Second, we evaluated a diverse set of models to quantify the extent of inappropriate removal of clinical information. Next, we conduct a manual validation of an existing evaluation metric to measure the removal of clinical information, employing clinical experts to assess their efficacy. We highlight poor performance and describe the inherent limitations of such metrics in identifying clinically significant changes. Lastly, we propose a novel methodology for the detection of clinically relevant information removal. 

---
# Ticket-Bench: A Kickoff for Multilingual and Regionalized Agent Evaluation 

**Authors**: Thales Sales Almeida, João Guilherme Alves Santos, Thiago Laitz, Giovana Kerche Bonás  

**Link**: [PDF](https://arxiv.org/pdf/2509.14477)  

**Abstract**: Large language models (LLMs) are increasingly deployed as task-oriented agents, where success depends on their ability to generate accurate function calls under realistic, multilingual conditions. However, existing agent evaluations largely overlook cultural and linguistic diversity, often relying on monolingual or naively translated benchmarks. We introduce Ticket-Bench, a benchmark for multilingual agent evaluation in task-oriented scenarios. Ticket-Bench simulates the domain of soccer ticket purchases across six major languages: Portuguese, English, Spanish, German, Italian, and French. Using localized teams, cities, and user profiles to provide a higher level of realism. We evaluate a wide range of commercial and open-source LLMs, measuring function-calling accuracy and consistency across languages. Results show that reasoning-oriented models (e.g., GPT-5, Qwen3-235B) dominate performance but still exhibit notable cross-lingual disparities. These findings underscore the need for culturally aware, multilingual benchmarks to guide the development of robust LLM agents. 

---
# Estimating Semantic Alphabet Size for LLM Uncertainty Quantification 

**Authors**: Lucas H. McCabe, Rimon Melamed, Thomas Hartvigsen, H. Howie Huang  

**Link**: [PDF](https://arxiv.org/pdf/2509.14478)  

**Abstract**: Many black-box techniques for quantifying the uncertainty of large language models (LLMs) rely on repeated LLM sampling, which can be computationally expensive. Therefore, practical applicability demands reliable estimation from few samples. Semantic entropy (SE) is a popular sample-based uncertainty estimator with a discrete formulation attractive for the black-box setting. Recent extensions of semantic entropy exhibit improved LLM hallucination detection, but do so with less interpretable methods that admit additional hyperparameters. For this reason, we revisit the canonical discrete semantic entropy estimator, finding that it underestimates the "true" semantic entropy, as expected from theory. We propose a modified semantic alphabet size estimator, and illustrate that using it to adjust discrete semantic entropy for sample coverage results in more accurate semantic entropy estimation in our setting of interest. Furthermore, our proposed alphabet size estimator flags incorrect LLM responses as well or better than recent top-performing approaches, with the added benefit of remaining highly interpretable. 

---
# Annotating Training Data for Conditional Semantic Textual Similarity Measurement using Large Language Models 

**Authors**: Gaifan Zhang, Yi Zhou, Danushka Bollegala  

**Link**: [PDF](https://arxiv.org/pdf/2509.14399)  

**Abstract**: Semantic similarity between two sentences depends on the aspects considered between those sentences. To study this phenomenon, Deshpande et al. (2023) proposed the Conditional Semantic Textual Similarity (C-STS) task and annotated a human-rated similarity dataset containing pairs of sentences compared under two different conditions. However, Tu et al. (2024) found various annotation issues in this dataset and showed that manually re-annotating a small portion of it leads to more accurate C-STS models. Despite these pioneering efforts, the lack of large and accurately annotated C-STS datasets remains a blocker for making progress on this task as evidenced by the subpar performance of the C-STS models. To address this training data need, we resort to Large Language Models (LLMs) to correct the condition statements and similarity ratings in the original dataset proposed by Deshpande et al. (2023). Our proposed method is able to re-annotate a large training dataset for the C-STS task with minimal manual effort. Importantly, by training a supervised C-STS model on our cleaned and re-annotated dataset, we achieve a 5.4% statistically significant improvement in Spearman correlation. The re-annotated dataset is available at this https URL. 

---
# Adding LLMs to the psycholinguistic norming toolbox: A practical guide to getting the most out of human ratings 

**Authors**: Javier Conde, María Grandury, Tairan Fu, Carlos Arriaga, Gonzalo Martínez, Thomas Clark, Sean Trott, Clarence Gerald Green, Pedro Reviriego, Marc Brysbaert  

**Link**: [PDF](https://arxiv.org/pdf/2509.14405)  

**Abstract**: Word-level psycholinguistic norms lend empirical support to theories of language processing. However, obtaining such human-based measures is not always feasible or straightforward. One promising approach is to augment human norming datasets by using Large Language Models (LLMs) to predict these characteristics directly, a practice that is rapidly gaining popularity in psycholinguistics and cognitive science. However, the novelty of this approach (and the relative inscrutability of LLMs) necessitates the adoption of rigorous methodologies that guide researchers through this process, present the range of possible approaches, and clarify limitations that are not immediately apparent, but may, in some cases, render the use of LLMs impractical.
In this work, we present a comprehensive methodology for estimating word characteristics with LLMs, enriched with practical advice and lessons learned from our own experience. Our approach covers both the direct use of base LLMs and the fine-tuning of models, an alternative that can yield substantial performance gains in certain scenarios. A major emphasis in the guide is the validation of LLM-generated data with human "gold standard" norms. We also present a software framework that implements our methodology and supports both commercial and open-weight models.
We illustrate the proposed approach with a case study on estimating word familiarity in English. Using base models, we achieved a Spearman correlation of 0.8 with human ratings, which increased to 0.9 when employing fine-tuned models. This methodology, framework, and set of best practices aim to serve as a reference for future research on leveraging LLMs for psycholinguistic and lexical studies. 

---
# Causal-Counterfactual RAG: The Integration of Causal-Counterfactual Reasoning into RAG 

**Authors**: Harshad Khadilkar, Abhay Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2509.14435)  

**Abstract**: Large language models (LLMs) have transformed natural language processing (NLP), enabling diverse applications by integrating large-scale pre-trained knowledge. However, their static knowledge limits dynamic reasoning over external information, especially in knowledge-intensive domains. Retrieval-Augmented Generation (RAG) addresses this challenge by combining retrieval mechanisms with generative modeling to improve contextual understanding. Traditional RAG systems suffer from disrupted contextual integrity due to text chunking and over-reliance on semantic similarity for retrieval, often resulting in shallow and less accurate responses. We propose Causal-Counterfactual RAG, a novel framework that integrates explicit causal graphs representing cause-effect relationships into the retrieval process and incorporates counterfactual reasoning grounded on the causal structure. Unlike conventional methods, our framework evaluates not only direct causal evidence but also the counterfactuality of associated causes, combining results from both to generate more robust, accurate, and interpretable answers. By leveraging causal pathways and associated hypothetical scenarios, Causal-Counterfactual RAG preserves contextual coherence, reduces hallucination, and enhances reasoning fidelity. 

---
# TDRM: Smooth Reward Models with Temporal Difference for LLM RL and Inference 

**Authors**: Dan Zhang, Min Cai, Jonathan Li, Ziniu Hu, Yisong Yue, Yuxiao Dong, Jie Tang  

**Link**: [PDF](https://arxiv.org/pdf/2509.15110)  

**Abstract**: Reward models are central to both reinforcement learning (RL) with language models and inference-time verification. However, existing reward models often lack temporal consistency, leading to ineffective policy updates and unstable RL training. We introduce TDRM, a method for learning smoother and more reliable reward models by minimizing temporal differences during training. This temporal-difference (TD) regularization produces smooth rewards and improves alignment with long-term objectives. Incorporating TDRM into the actor-critic style online RL loop yields consistent empirical gains. It is worth noting that TDRM is a supplement to verifiable reward methods, and both can be used in series. Experiments show that TD-trained process reward models (PRMs) improve performance across Best-of-N (up to 6.6%) and tree-search (up to 23.7%) settings. When combined with Reinforcement Learning with Verifiable Rewards (RLVR), TD-trained PRMs lead to more data-efficient RL -- achieving comparable performance with just 2.5k data to what baseline methods require 50.1k data to attain -- and yield higher-quality language model policies on 8 model variants (5 series), e.g., Qwen2.5-(0.5B, 1,5B), GLM4-9B-0414, GLM-Z1-9B-0414, Qwen2.5-Math-(1.5B, 7B), and DeepSeek-R1-Distill-Qwen-(1.5B, 7B). We release all code at this https URL. 

---
# AIP: Subverting Retrieval-Augmented Generation via Adversarial Instructional Prompt 

**Authors**: Saket S. Chaturvedi, Gaurav Bagwe, Lan Zhang, Xiaoyong Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2509.15159)  

**Abstract**: Retrieval-Augmented Generation (RAG) enhances large language models (LLMs) by retrieving relevant documents from external sources to improve factual accuracy and verifiability. However, this reliance introduces new attack surfaces within the retrieval pipeline, beyond the LLM itself. While prior RAG attacks have exposed such vulnerabilities, they largely rely on manipulating user queries, which is often infeasible in practice due to fixed or protected user inputs. This narrow focus overlooks a more realistic and stealthy vector: instructional prompts, which are widely reused, publicly shared, and rarely audited. Their implicit trust makes them a compelling target for adversaries to manipulate RAG behavior covertly.
We introduce a novel attack for Adversarial Instructional Prompt (AIP) that exploits adversarial instructional prompts to manipulate RAG outputs by subtly altering retrieval behavior. By shifting the attack surface to the instructional prompts, AIP reveals how trusted yet seemingly benign interface components can be weaponized to degrade system integrity. The attack is crafted to achieve three goals: (1) naturalness, to evade user detection; (2) utility, to encourage use of prompts; and (3) robustness, to remain effective across diverse query variations. We propose a diverse query generation strategy that simulates realistic linguistic variation in user queries, enabling the discovery of prompts that generalize across paraphrases and rephrasings. Building on this, a genetic algorithm-based joint optimization is developed to evolve adversarial prompts by balancing attack success, clean-task utility, and stealthiness. Experimental results show that AIP achieves up to 95.23% ASR while preserving benign functionality. These findings uncover a critical and previously overlooked vulnerability in RAG systems, emphasizing the need to reassess the shared instructional prompts. 

---
# Evolving Language Models without Labels: Majority Drives Selection, Novelty Promotes Variation 

**Authors**: Yujun Zhou, Zhenwen Liang, Haolin Liu, Wenhao Yu, Kishan Panaganti, Linfeng Song, Dian Yu, Xiangliang Zhang, Haitao Mi, Dong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2509.15194)  

**Abstract**: Large language models (LLMs) are increasingly trained with reinforcement learning from verifiable rewards (RLVR), yet real-world deployment demands models that can self-improve without labels or external judges. Existing label-free methods, confidence minimization, self-consistency, or majority-vote objectives, stabilize learning but steadily shrink exploration, causing an entropy collapse: generations become shorter, less diverse, and brittle. Unlike prior approaches such as Test-Time Reinforcement Learning (TTRL), which primarily adapt models to the immediate unlabeled dataset at hand, our goal is broader: to enable general improvements without sacrificing the model's inherent exploration capacity and generalization ability, i.e., evolving. We formalize this issue and propose EVolution-Oriented and Label-free Reinforcement Learning (EVOL-RL), a simple rule that couples stability with variation under a label-free setting. EVOL-RL keeps the majority-voted answer as a stable anchor (selection) while adding a novelty-aware reward that favors responses whose reasoning differs from what has already been produced (variation), measured in semantic space. Implemented with GRPO, EVOL-RL also uses asymmetric clipping to preserve strong signals and an entropy regularizer to sustain search. This majority-for-selection + novelty-for-variation design prevents collapse, maintains longer and more informative chains of thought, and improves both pass@1 and pass@n. EVOL-RL consistently outperforms the majority-only TTRL baseline; e.g., training on label-free AIME24 lifts Qwen3-4B-Base AIME25 pass@1 from TTRL's 4.6% to 16.4%, and pass@16 from 18.5% to 37.9%. EVOL-RL not only prevents diversity collapse but also unlocks stronger generalization across domains (e.g., GPQA). Furthermore, we demonstrate that EVOL-RL also boosts performance in the RLVR setting, highlighting its broad applicability. 

---
# A Simple and Efficient Jailbreak Method Exploiting LLMs' Helpfulness 

**Authors**: Xuan Luo, Yue Wang, Zefeng He, Geng Tu, Jing Li, Ruifeng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2509.14297)  

**Abstract**: Safety alignment aims to prevent Large Language Models (LLMs) from responding to harmful queries. To strengthen safety protections, jailbreak methods are developed to simulate malicious attacks and uncover vulnerabilities. In this paper, we introduce HILL (Hiding Intention by Learning from LLMs), a novel jailbreak approach that systematically transforms imperative harmful requests into learning-style questions with only straightforward hypotheticality indicators. Further, we introduce two new metrics to thoroughly evaluate the utility of jailbreak methods. Experiments on the AdvBench dataset across a wide range of models demonstrate HILL's strong effectiveness, generalizability, and harmfulness. It achieves top attack success rates on the majority of models and across malicious categories while maintaining high efficiency with concise prompts. Results of various defense methods show the robustness of HILL, with most defenses having mediocre effects or even increasing the attack success rates. Moreover, the assessment on our constructed safe prompts reveals inherent limitations of LLMs' safety mechanisms and flaws in defense methods. This work exposes significant vulnerabilities of safety measures against learning-style elicitation, highlighting a critical challenge of balancing helpfulness and safety alignments. 

---
# Keywords are not always the key: A metadata field analysis for natural language search on open data portals 

**Authors**: Lisa-Yao Gan, Arunav Das, Johanna Walker, Elena Simperl  

**Link**: [PDF](https://arxiv.org/pdf/2509.14457)  

**Abstract**: Open data portals are essential for providing public access to open datasets. However, their search interfaces typically rely on keyword-based mechanisms and a narrow set of metadata fields. This design makes it difficult for users to find datasets using natural language queries. The problem is worsened by metadata that is often incomplete or inconsistent, especially when users lack familiarity with domain-specific terminology. In this paper, we examine how individual metadata fields affect the success of conversational dataset retrieval and whether LLMs can help bridge the gap between natural queries and structured metadata. We conduct a controlled ablation study using simulated natural language queries over real-world datasets to evaluate retrieval performance under various metadata configurations. We also compare existing content of the metadata field 'description' with LLM-generated content, exploring how different prompting strategies influence quality and impact on search outcomes. Our findings suggest that dataset descriptions play a central role in aligning with user intent, and that LLM-generated descriptions can support effective retrieval. These results highlight both the limitations of current metadata practices and the potential of generative models to improve dataset discoverability in open data portals. 

---
# What Matters in LLM-Based Feature Extractor for Recommender? A Systematic Analysis of Prompts, Models, and Adaptation 

**Authors**: Kainan Shi, Peilin Zhou, Ge Wang, Han Ding, Fei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.14979)  

**Abstract**: Using Large Language Models (LLMs) to generate semantic features has been demonstrated as a powerful paradigm for enhancing Sequential Recommender Systems (SRS). This typically involves three stages: processing item text, extracting features with LLMs, and adapting them for downstream models. However, existing methods vary widely in prompting, architecture, and adaptation strategies, making it difficult to fairly compare design choices and identify what truly drives performance. In this work, we propose RecXplore, a modular analytical framework that decomposes the LLM-as-feature-extractor pipeline into four modules: data processing, semantic feature extraction, feature adaptation, and sequential modeling. Instead of proposing new techniques, RecXplore revisits and organizes established methods, enabling systematic exploration of each module in isolation. Experiments on four public datasets show that simply combining the best designs from existing techniques without exhaustive search yields up to 18.7% relative improvement in NDCG@5 and 12.7% in HR@5 over strong baselines. These results underscore the utility of modular benchmarking for identifying effective design patterns and promoting standardized research in LLM-enhanced recommendation. 

---
