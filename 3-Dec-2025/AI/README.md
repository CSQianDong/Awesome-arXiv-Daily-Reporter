# From Moderation to Mediation: Can LLMs Serve as Mediators in Online Flame Wars? 

**Authors**: Dawei Li, Abdullah Alnaibari, Arslan Bisharat, Manny Sandoval, Deborah Hall, Yasin Silva, Huan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2512.03005)  

**Abstract**: The rapid advancement of large language models (LLMs) has opened new possibilities for AI for good applications. As LLMs increasingly mediate online communication, their potential to foster empathy and constructive dialogue becomes an important frontier for responsible AI research. This work explores whether LLMs can serve not only as moderators that detect harmful content, but as mediators capable of understanding and de-escalating online conflicts. Our framework decomposes mediation into two subtasks: judgment, where an LLM evaluates the fairness and emotional dynamics of a conversation, and steering, where it generates empathetic, de-escalatory messages to guide participants toward resolution. To assess mediation quality, we construct a large Reddit-based dataset and propose a multi-stage evaluation pipeline combining principle-based scoring, user simulation, and human comparison. Experiments show that API-based models outperform open-source counterparts in both reasoning and intervention alignment when doing mediation. Our findings highlight both the promise and limitations of current LLMs as emerging agents for online social mediation. 

---
# Invasive Context Engineering to Control Large Language Models 

**Authors**: Thomas Rivasseau  

**Link**: [PDF](https://arxiv.org/pdf/2512.03001)  

**Abstract**: Current research on operator control of Large Language Models improves model robustness against adversarial attacks and misbehavior by training on preference examples, prompting, and input/output filtering. Despite good results, LLMs remain susceptible to abuse, and jailbreak probability increases with context length. There is a need for robust LLM security guarantees in long-context situations. We propose control sentences inserted into the LLM context as invasive context engineering to partially solve the problem. We suggest this technique can be generalized to the Chain-of-Thought process to prevent scheming. Invasive Context Engineering does not rely on LLM training, avoiding data shortage pitfalls which arise in training models for long context situations. 

---
# Martingale Score: An Unsupervised Metric for Bayesian Rationality in LLM Reasoning 

**Authors**: Zhonghao He, Tianyi Qiu, Hirokazu Shirado, Maarten Sap  

**Link**: [PDF](https://arxiv.org/pdf/2512.02914)  

**Abstract**: Recent advances in reasoning techniques have substantially improved the performance of large language models (LLMs), raising expectations for their ability to provide accurate, truthful, and reliable information. However, emerging evidence suggests that iterative reasoning may foster belief entrenchment and confirmation bias, rather than enhancing truth-seeking behavior. In this study, we propose a systematic evaluation framework for belief entrenchment in LLM reasoning by leveraging the Martingale property from Bayesian statistics. This property implies that, under rational belief updating, the expected value of future beliefs should remain equal to the current belief, i.e., belief updates are unpredictable from the current belief. We propose the unsupervised, regression-based Martingale Score to measure violations of this property, which signal deviation from the Bayesian ability of updating on new evidence. In open-ended problem domains including event forecasting, value-laden questions, and academic paper review, we find such violations to be widespread across models and setups, where the current belief positively predicts future belief updates, a phenomenon which we term belief entrenchment. We identify the models, reasoning techniques, and domains more prone to belief entrenchment. Finally, we validate the Martingale Score by showing that it predicts ground-truth accuracy on problem domains where ground truth labels are available. This indicates that, while designed as an unsupervised metric that operates even in domains without access to ground truth, the Martingale Score is a useful proxy of the truth-seeking ability of a reasoning process. 

---
# The future of AI in critical mineral exploration 

**Authors**: Jef Caers  

**Link**: [PDF](https://arxiv.org/pdf/2512.02879)  

**Abstract**: The energy transition through increased electrification has put the worlds attention on critical mineral exploration Even with increased investments a decrease in new discoveries has taken place over the last two decades Here I propose a solution to this problem where AI is implemented as the enabler of a rigorous scientific method for mineral exploration that aims to reduce cognitive bias and false positives drive down the cost of exploration I propose a new scientific method that is based on a philosophical approach founded on the principles of Bayesianism and falsification In this approach data acquisition is in the first place seen as a means to falsify human generated hypothesis Decision of what data to acquire next is quantified with verifiable metrics and based on rational decision making A practical protocol is provided that can be used as a template in any exploration campaign However in order to make this protocol practical various form of artificial intelligence are needed I will argue that the most important form are one novel unsupervised learning methods that collaborate with domain experts to better understand data and generate multiple competing geological hypotheses and two humanintheloop AI algorithms that can optimally plan various geological geophysical geochemical and drilling data acquisition where uncertainty reduction of geological hypothesis precedes the uncertainty reduction on grade and tonnage 

---
# Radiologist Copilot: An Agentic Assistant with Orchestrated Tools for Radiology Reporting with Quality Control 

**Authors**: Yongrui Yu, Zhongzhen Huang, Linjie Mu, Shaoting Zhang, Xiaofan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2512.02814)  

**Abstract**: Radiology reporting is an essential yet time-consuming and error-prone task for radiologists in clinical examinations, especially for volumetric medical images. Rigorous quality control is also critical but tedious, ensuring that the final report meets clinical standards. Existing automated approaches, including radiology report generation methods and medical vision-language models, focus mainly on the report generation phase and neglect the crucial quality control procedure, limiting their capability to provide comprehensive support to radiologists. We propose Radiologist Copilot, an agentic AI assistant equipped with orchestrated tools designed for automated radiology reporting with quality control. Leveraging large language models as the reasoning backbone, the agentic system autonomously selects tools, plans, and executes actions, emulating the behavior of radiologists throughout the holistic radiology reporting process. The orchestrated tools include region localization, think with image paradigm directed region analysis planning, strategic template selection for report generation, quality assessment and feedback-driven adaptive refinement for quality control. Therefore, Radiologist Copilot facilitates accurate, complete, and efficient radiology reporting, assisting radiologists and improving clinical efficiency. Experimental results demonstrate that Radiologist Copilot significantly surpasses other state-of-the-art methods in radiology reporting. The source code will be released upon acceptance. 

---
# Enhancing Automated Paper Reproduction via Prompt-Free Collaborative Agents 

**Authors**: Zijie Lin, Qilin Cai, Liang Shen, Mingjun Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2512.02812)  

**Abstract**: Automated paper reproduction has emerged as a promising approach to accelerate scientific research, employing multi-step workflow frameworks to systematically convert academic papers into executable code. However, existing frameworks often lack mechanisms to verify and refine the outputs at each generation step, or rely heavily on manually designed prompts for self-refinement, which limits their adaptability and scalability. To address these limitations, we propose a prompt-free collaborative agent framework that automatically enhances the quality of paper-to-code generation. Our approach employs two collaborative agents: a verification agent that examines whether the outputs at each step satisfy the requirements specified in the corresponding system prompt, and a refinement agent that revises the outputs based on the identified issues. Unlike previous methods that require human experts to craft specific refinement prompts for each step, our framework achieves automatic verification and improvement by leveraging only the original system prompts. We integrate our collaborative agents into the Paper2Code framework and conduct comprehensive experiments on PaperBench Code-Dev and Paper2CodeBench datasets. Experimental results demonstrate that our approach significantly improves the accuracy and completeness of reproduced code, achieving performance gains of approximately 15\% and 13\%, respectively, compared to the baseline without our agents. Furthermore, comparative experiments against Self-Refine validate the robustness and consistency of our prompt-free approach across different datasets. 

---
# A Framework for Causal Concept-based Model Explanations 

**Authors**: Anna Rodum Bjøru, Jacob Lysnæs-Larsen, Oskar Jørgensen, Inga Strümke, Helge Langseth  

**Link**: [PDF](https://arxiv.org/pdf/2512.02735)  

**Abstract**: This work presents a conceptual framework for causal concept-based post-hoc Explainable Artificial Intelligence (XAI), based on the requirements that explanations for non-interpretable models should be understandable as well as faithful to the model being explained. Local and global explanations are generated by calculating the probability of sufficiency of concept interventions. Example explanations are presented, generated with a proof-of-concept model made to explain classifiers trained on the CelebA dataset. Understandability is demonstrated through a clear concept-based vocabulary, subject to an implicit causal interpretation. Fidelity is addressed by highlighting important framework assumptions, stressing that the context of explanation interpretation must align with the context of explanation generation. 

---
# Self-Improving AI Agents through Self-Play 

**Authors**: Przemyslaw Chojecki  

**Link**: [PDF](https://arxiv.org/pdf/2512.02731)  

**Abstract**: We extend the moduli-theoretic framework of psychometric batteries to the domain of dynamical systems. While previous work established the AAI capability score as a static functional on the space of agent representations, this paper formalizes the agent as a flow $\nu_r$ parameterized by computational resource $r$, governed by a recursive Generator-Verifier-Updater (GVU) operator. We prove that this operator generates a vector field on the parameter manifold $\Theta$, and we identify the coefficient of self-improvement $\kappa$ as the Lie derivative of the capability functional along this flow.
The central contribution of this work is the derivation of the Variance Inequality, a spectral condition that is sufficient (under mild regularity) for the stability of self-improvement. We show that a sufficient condition for $\kappa > 0$ is that, up to curvature and step-size effects, the combined noise of generation and verification must be small enough.
We then apply this formalism to unify the recent literature on Language Self-Play (LSP), Self-Correction, and Synthetic Data bootstrapping. We demonstrate that architectures such as STaR, SPIN, Reflexion, GANs and AlphaZero are specific topological realizations of the GVU operator that satisfy the Variance Inequality through filtration, adversarial discrimination, or grounding in formal systems. 

---
# AuditCopilot: Leveraging LLMs for Fraud Detection in Double-Entry Bookkeeping 

**Authors**: Md Abdul Kadir, Sai Suresh Macharla Vasu, Sidharth S. Nair, Daniel Sonntag  

**Link**: [PDF](https://arxiv.org/pdf/2512.02726)  

**Abstract**: Auditors rely on Journal Entry Tests (JETs) to detect anomalies in tax-related ledger records, but rule-based methods generate overwhelming false positives and struggle with subtle irregularities. We investigate whether large language models (LLMs) can serve as anomaly detectors in double-entry bookkeeping. Benchmarking SoTA LLMs such as LLaMA and Gemma on both synthetic and real-world anonymized ledgers, we compare them against JETs and machine learning baselines. Our results show that LLMs consistently outperform traditional rule-based JETs and classical ML baselines, while also providing natural-language explanations that enhance interpretability. These results highlight the potential of \textbf{AI-augmented auditing}, where human auditors collaborate with foundation models to strengthen financial integrity. 

---
# StockMem: An Event-Reflection Memory Framework for Stock Forecasting 

**Authors**: He Wang, Wenyilin Xiao, Songqiao Han, Hailiang Huang  

**Link**: [PDF](https://arxiv.org/pdf/2512.02720)  

**Abstract**: Stock price prediction is challenging due to market volatility and its sensitivity to real-time events. While large language models (LLMs) offer new avenues for text-based forecasting, their application in finance is hindered by noisy news data and the lack of explicit answers in text. General-purpose memory architectures struggle to identify the key drivers of price movements. To address this, we propose StockMem, an event-reflection dual-layer memory framework. It structures news into events and mines them along two dimensions: horizontal consolidation integrates daily events, while longitudinal tracking captures event evolution to extract incremental information reflecting market expectation discrepancies. This builds a temporal event knowledge base. By analyzing event-price dynamics, the framework further forms a reflection knowledge base of causal experiences. For prediction, it retrieves analogous historical scenarios and reasons with current events, incremental data, and past experiences. Experiments show StockMem outperforms existing memory architectures and provides superior, explainable reasoning by tracing the information chain affecting prices, enhancing decision transparency in financial forecasting. 

---
# Menta: A Small Language Model for On-Device Mental Health Prediction 

**Authors**: Tianyi Zhang, Xiangyuan Xue, Lingyan Ruan, Shiya Fu, Feng Xia, Simon D'Alfonso, Vassilis Kostakos, Hong Jia  

**Link**: [PDF](https://arxiv.org/pdf/2512.02716)  

**Abstract**: Mental health conditions affect hundreds of millions globally, yet early detection remains limited. While large language models (LLMs) have shown promise in mental health applications, their size and computational demands hinder practical deployment. Small language models (SLMs) offer a lightweight alternative, but their use for social media--based mental health prediction remains largely underexplored. In this study, we introduce Menta, the first optimized SLM fine-tuned specifically for multi-task mental health prediction from social media data. Menta is jointly trained across six classification tasks using a LoRA-based framework, a cross-dataset strategy, and a balanced accuracy--oriented loss. Evaluated against nine state-of-the-art SLM baselines, Menta achieves an average improvement of 15.2\% across tasks covering depression, stress, and suicidality compared with the best-performing non--fine-tuned SLMs. It also achieves higher accuracy on depression and stress classification tasks compared to 13B-parameter LLMs, while being approximately 3.25x smaller. Moreover, we demonstrate real-time, on-device deployment of Menta on an iPhone 15 Pro Max, requiring only approximately 3GB RAM. Supported by a comprehensive benchmark against existing SLMs and LLMs, Menta highlights the potential for scalable, privacy-preserving mental health monitoring. Code is available at: this https URL 

---
# Training Data Attribution for Image Generation using Ontology-Aligned Knowledge Graphs 

**Authors**: Theodoros Aivalis, Iraklis A. Klampanos, Antonis Troumpoukis, Joemon M. Jose  

**Link**: [PDF](https://arxiv.org/pdf/2512.02713)  

**Abstract**: As generative models become powerful, concerns around transparency, accountability, and copyright violations have intensified. Understanding how specific training data contributes to a model's output is critical. We introduce a framework for interpreting generative outputs through the automatic construction of ontologyaligned knowledge graphs (KGs). While automatic KG construction from natural text has advanced, extracting structured and ontology-consistent representations from visual content remains challenging -- due to the richness and multi-object nature of images. Leveraging multimodal large language models (LLMs), our method extracts structured triples from images, aligned with a domain-specific ontology. By comparing the KGs of generated and training images, we can trace potential influences, enabling copyright analysis, dataset transparency, and interpretable AI. We validate our method through experiments on locally trained models via unlearning, and on large-scale models through a style-specific experiment. Our framework supports the development of AI systems that foster human collaboration, creativity and stimulate curiosity. 

---
# Learning What to Attend First: Modality-Importance-Guided Reasoning for Reliable Multimodal Emotion Understanding 

**Authors**: Hyeongseop Rha, Jeong Hun Yeo, Junil Won, Se Jin Park, Yong Man Ro  

**Link**: [PDF](https://arxiv.org/pdf/2512.02699)  

**Abstract**: In this paper, we present Modality-Importance-Guided Reasoning (MIGR), a framework designed to improve the reliability of reasoning-based multimodal emotion understanding in multimodal large language models. Although existing methods have advanced emotion understanding, they often suffer from reasoning drift: models gradually rely on their own generated text instead of multimodal evidence, and their explanations are overly shaped by visually initiated reasoning paths. To address these issues, we introduce Modality Importance (MI), a simple yet effective mechanism for identifying the emotion-dominant modality. Using MI, MIGR reorganizes reasoning sequences so that explanations begin from the modality most critical to the target emotion, preventing early reasoning from being misled by less informative cues. Our two-stage framework-comprising modality-aligned supervised fine-tuning and modality-aware reward optimization-encourages models to generate emotionally grounded, causally relevant, and coherence-preserving explanations. Experimental results on the DFEW benchmark show that MIGR substantially improves reasoning reliability, decreasing instances of correct predictions accompanied by emotionally inconsistent explanations from 18.10% to 7.37%. These results confirm the benefit of initiating reasoning from the emotion-dominant modality. 

---
# Exploring Depth Generalization in Large Language Models for Solving Recursive Logic Tasks 

**Authors**: Zhiyuan He  

**Link**: [PDF](https://arxiv.org/pdf/2512.02677)  

**Abstract**: Large language models have demonstrated remarkable capabilities across many tasks, yet face significant challenges when dealing with recursive reasoning problems, those requiring the resolution of nested hierarchical structures. While prior research has extensively studied length generalization (a model's ability to handle longer sequences than seen during training), we investigate a distinct and underexplored limitation: depth generalization. Here, depth refers to the number of nested levels in a hierarchical problem, such as the layers of parentheses in a mathematical expression or the nesting of logical clauses in a Boolean formula. Our work reveals that standard transformer architectures struggle with problems involving deeper recursion than encountered during training, even when they perform well on longer but non-nested sequences. This limitation stems from their inability to maintain stack-like behavior, the capacity to track and resolve multiple levels of nested dependencies. Through systematic analysis, we demonstrate how this architectural constraint leads to rapid performance decay as the depth of the recursion increases. To address this challenge, we develop a novel looped locate-and-replace pipeline that decomposes recursive problems into manageable subcomponents. The approach employs two specialized models: a locator that identifies solvable subexpressions and a replacer that evaluates these components while preserving the overall structure. We evaluated this method in three carefully designed domains: Boolean algebra, recursive arithmetic, and propositional logic, each with a controllable depth of recursion. We show that our method effectively alleviates the performance decay when tested on out-of-distribution recursion depth. 

---
# Zero-Shot Instruction Following in RL via Structured LTL Representations 

**Authors**: Mattia Giuri, Mathias Jackermeier, Alessandro Abate  

**Link**: [PDF](https://arxiv.org/pdf/2512.02633)  

**Abstract**: Linear temporal logic (LTL) is a compelling framework for specifying complex, structured tasks for reinforcement learning (RL) agents. Recent work has shown that interpreting LTL instructions as finite automata, which can be seen as high-level programs monitoring task progress, enables learning a single generalist policy capable of executing arbitrary instructions at test time. However, existing approaches fall short in environments where multiple high-level events (i.e., atomic propositions) can be true at the same time and potentially interact in complicated ways. In this work, we propose a novel approach to learning a multi-task policy for following arbitrary LTL instructions that addresses this shortcoming. Our method conditions the policy on sequences of simple Boolean formulae, which directly align with transitions in the automaton, and are encoded via a graph neural network (GNN) to yield structured task representations. Experiments in a complex chess-based environment demonstrate the advantages of our approach. 

---
# Target-specific Adaptation and Consistent Degradation Alignment for Cross-Domain Remaining Useful Life Prediction 

**Authors**: Yubo Hou, Mohamed Ragab, Min Wu, Chee-Keong Kwoh, Xiaoli Li, Zhenghua Chen  

**Link**: [PDF](https://arxiv.org/pdf/2512.02610)  

**Abstract**: Accurate prediction of the Remaining Useful Life (RUL) in machinery can significantly diminish maintenance costs, enhance equipment up-time, and mitigate adverse outcomes. Data-driven RUL prediction techniques have demonstrated commendable performance. However, their efficacy often relies on the assumption that training and testing data are drawn from the same distribution or domain, which does not hold in real industrial settings. To mitigate this domain discrepancy issue, prior adversarial domain adaptation methods focused on deriving domain-invariant features. Nevertheless, they overlook target-specific information and inconsistency characteristics pertinent to the degradation stages, resulting in suboptimal performance. To tackle these issues, we propose a novel domain adaptation approach for cross-domain RUL prediction named TACDA. Specifically, we propose a target domain reconstruction strategy within the adversarial adaptation process, thereby retaining target-specific information while learning domain-invariant features. Furthermore, we develop a novel clustering and pairing strategy for consistent alignment between similar degradation stages. Through extensive experiments, our results demonstrate the remarkable performance of our proposed TACDA method, surpassing state-of-the-art approaches with regard to two different evaluation metrics. Our code is available at this https URL. 

---
# IACT: A Self-Organizing Recursive Model for General AI Agents: A Technical White Paper on the Architecture Behind kragent.ai 

**Authors**: Pengju Lu  

**Link**: [PDF](https://arxiv.org/pdf/2512.02605)  

**Abstract**: This technical white paper introduces the Interactive Agents Call Tree (IACT), a computational model designed to address the limitations of static, hard-coded agent workflows. Unlike traditional systems that require pre-defined graphs or specialized programming, IACT operates as a general-purpose autonomous system driven purely by user dialogue. Given a high-level objective, the system autonomously grows a dynamic, recursive agent topology incrementally tailored to the problem's structure. This allows it to scale its organizational complexity to match open-ended tasks. To mitigate the error propagation inherent in unidirectional function calls, IACT introduces interactional redundancy by replacing rigid invocations with bidirectional, stateful dialogues. This mechanism enables runtime error correction and ambiguity resolution. We describe the architecture, design principles, and practical lessons behind the production deployment of this model in the this http URL system, presenting qualitative evidence from real-world workflows rather than exhaustive benchmark results. 

---
# PaperDebugger: A Plugin-Based Multi-Agent System for In-Editor Academic Writing, Review, and Editing 

**Authors**: Junyi Hou, Andre Lin Huikai, Nuo Chen, Yiwei Gong, Bingsheng He  

**Link**: [PDF](https://arxiv.org/pdf/2512.02589)  

**Abstract**: Large language models are increasingly embedded into academic writing workflows, yet existing assistants remain external to the editor, preventing deep interaction with document state, structure, and revision history. This separation makes it impossible to support agentic, context-aware operations directly within LaTeX editors such as Overleaf. We present PaperDebugger, an in-editor, multi-agent, and plugin-based academic writing assistant that brings LLM-driven reasoning directly into the writing environment. Enabling such in-editor interaction is technically non-trivial: it requires reliable bidirectional synchronization with the editor, fine-grained version control and patching, secure state management, multi-agent scheduling, and extensible communication with external tools. PaperDebugger addresses these challenges through a Chrome-approved extension, a Kubernetes-native orchestration layer, and a Model Context Protocol (MCP) toolchain that integrates literature search, reference lookup, document scoring, and revision pipelines. Our demo showcases a fully integrated workflow, including localized edits, structured reviews, parallel agent execution, and diff-based updates, encapsulated within a minimal-intrusion user interface (UI). Early aggregated analytics demonstrate active user engagement and validate the practicality of an editor-native, agentic writing assistant. More details about this demo and video could be found at this https URL. 

---
# Empathy Level Prediction in Multi-Modal Scenario with Supervisory Documentation Assistance 

**Authors**: Yufei Xiao, Shangfei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.02558)  

**Abstract**: Prevalent empathy prediction techniques primarily concentrate on a singular modality, typically textual, thus neglecting multi-modal processing capabilities. They also overlook the utilization of certain privileged information, which may encompass additional empathetic content. In response, we introduce an advanced multi-modal empathy prediction method integrating video, audio, and text information. The method comprises the Multi-Modal Empathy Prediction and Supervisory Documentation Assisted Training. We use pre-trained networks in the empathy prediction network to extract features from various modalities, followed by a cross-modal fusion. This process yields a multi-modal feature representation, which is employed to predict empathy labels. To enhance the extraction of text features, we incorporate supervisory documents as privileged information during the assisted training phase. Specifically, we apply the Latent Dirichlet Allocation model to identify potential topic distributions to constrain text features. These supervisory documents, created by supervisors, focus on the counseling topics and the counselor's display of empathy. Notably, this privileged information is only available during training and is not accessible during the prediction phase. Experimental results on the multi-modal and dialogue empathy datasets demonstrate that our approach is superior to the existing methods. 

---
# Aetheria: A multimodal interpretable content safety framework based on multi-agent debate and collaboration 

**Authors**: Yuxiang He, Jian Zhao, Yuchen Yuan, Tianle Zhang, Wei Cai, Haojie Cheng, Ziyan Shi, Ming Zhu, Haichuan Tang, Chi Zhang, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2512.02530)  

**Abstract**: The exponential growth of digital content presents significant challenges for content safety. Current moderation systems, often based on single models or fixed pipelines, exhibit limitations in identifying implicit risks and providing interpretable judgment processes. To address these issues, we propose Aetheria, a multimodal interpretable content safety framework based on multi-agent debate and this http URL a collaborative architecture of five core agents, Aetheria conducts in-depth analysis and adjudication of multimodal content through a dynamic, mutually persuasive debate mechanism, which is grounded by RAG-based knowledge this http URL experiments on our proposed benchmark (AIR-Bench) validate that Aetheria not only generates detailed and traceable audit reports but also demonstrates significant advantages over baselines in overall content safety accuracy, especially in the identification of implicit risks. This framework establishes a transparent and interpretable paradigm, significantly advancing the field of trustworthy AI content moderation. 

---
# COPE: Chain-Of-Thought Prediction Engine for Open-Source Large Language Model Based Stroke Outcome Prediction from Clinical Notes 

**Authors**: Yongkai Liu, Helena Feng, Bin Jiang, Yixin Wang, Max Wintermark, David S. Liebeskind, Michael Moseley, Maarten Lansberg, Gregory Albers, Jeremy Heit, Greg Zaharchuk  

**Link**: [PDF](https://arxiv.org/pdf/2512.02499)  

**Abstract**: Predicting outcomes in acute ischemic stroke (AIS) guides clinical decision-making, patient counseling, and resource allocation. Clinical notes contain rich contextual information, but their unstructured nature limits their use in traditional predictive models. We developed and evaluated the Chain-of-Thought (CoT) Outcome Prediction Engine (COPE), a reasoning-enhanced large language model framework, for predicting 90-day functional outcomes after AIS from unstructured clinical notes. This study included 464 AIS patients with discharge summaries and 90-day modified Rankin Scale (mRS) scores. COPE uses a two-step CoT framework based on sequential open-source LLaMA-3-8B models: the first generates clinical reasoning, and the second outputs an mRS prediction. We compared COPE with GPT-4.1, ClinicalBERT, a structured variable-based machine learning model (Clinical ML), and a single-step LLM without CoT. Performance was evaluated using mean absolute error (MAE), accuracy within +/-1 mRS point, and exact accuracy. COPE achieved an MAE of 1.01 (95% CI 0.92-1.11), +/-1 accuracy of 74.4% (69.9, 78.8%), and exact accuracy of 32.8% (28.0, 37.6%), comparable to GPT-4.1 and superior to ClinicalBERT [MAE 1.24 (1.13-1.36)], Clinical ML [1.28 (1.18-1.39)], and the single-step LLM [1.20 (1.09-1.33)]. Subgroup analyses showed consistent performance across sex and age, with slightly higher error among older patients, those undergoing thrombectomy, and those with longer summaries. These findings demonstrate that COPE, a lightweight, interpretable, and privacy-preserving open-source framework, provides an accurate and practical solution for outcome prediction from unstructured clinical text. 

---
# Guided Self-Evolving LLMs with Minimal Human Supervision 

**Authors**: Wenhao Yu, Zhenwen Liang, Chengsong Huang, Kishan Panaganti, Tianqing Fang, Haitao Mi, Dong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2512.02472)  

**Abstract**: AI self-evolution has long been envisioned as a path toward superintelligence, where models autonomously acquire, refine, and internalize knowledge from their own learning experiences. Yet in practice, unguided self-evolving systems often plateau quickly or even degrade as training progresses. These failures arise from issues such as concept drift, diversity collapse, and mis-evolution, as models reinforce their own biases and converge toward low-entropy behaviors. To enable models to self-evolve in a stable and controllable manner while minimizing reliance on human supervision, we introduce R-Few, a guided Self-Play Challenger-Solver framework that incorporates lightweight human oversight through in-context grounding and mixed training. At each iteration, the Challenger samples a small set of human-labeled examples to guide synthetic question generation, while the Solver jointly trains on human and synthetic examples under an online, difficulty-based curriculum. Across math and general reasoning benchmarks, R-Few achieves consistent and iterative improvements. For example, Qwen3-8B-Base improves by +3.0 points over R-Zero on math tasks and achieves performance on par with General-Reasoner, despite the latter being trained on 20 times more human data. Ablation studies confirm the complementary contributions of grounded challenger training and curriculum-based solver training, and further analysis shows that R-Few mitigates drift, yielding more stable and controllable co-evolutionary dynamics. 

---
# Semantic Trading: Agentic AI for Clustering and Relationship Discovery in Prediction Markets 

**Authors**: Agostino Capponi, Alfio Gliozzo, Brian Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2512.02436)  

**Abstract**: Prediction markets allow users to trade on outcomes of real-world events, but are prone to fragmentation through overlapping questions, implicit equivalences, and hidden contradictions across markets. We present an agentic AI pipeline that autonomously (i) clusters markets into coherent topical groups using natural-language understanding over contract text and metadata, and (ii) identifies within-cluster market pairs whose resolved outcomes exhibit strong dependence, including same-outcome (correlated) and different-outcome (anti-correlated) relationships. Using a historical dataset of resolved markets on Polymarket, we evaluate the accuracy of the agent's relational predictions. We then translate discovered relationships into a simple trading strategy to quantify how these relationships map to actionable signals. Results show that agent-identified relationships achieve roughly 60-70% accuracy, and their induced trading strategies earn about 20% average returns over week-long horizons, highlighting the ability of agentic AI and large language models to uncover latent semantic structure in prediction markets. 

---
# Synthetic Error Injection Fails to Elicit Self-Correction In Language Models 

**Authors**: David X. Wu, Shreyas Kapur, Anant Sahai, Stuart Russell  

**Link**: [PDF](https://arxiv.org/pdf/2512.02389)  

**Abstract**: Reinforcement learning has become the dominant paradigm for eliciting reasoning and self-correction capabilities in large language models, but its computational expense motivates exploration of alternatives. Inspired by techniques from autonomous driving and robotics, we investigate whether supervised learning with synthetic error injection can induce self-correction abilities in language models. Our approach inserts artificial errors into reasoning chains, masks them, and supervises the model to recognize and correct these mistakes. Despite the intuitive appeal of this method, we find that it fails to significantly improve performance even on simple synthetic tasks across multiple models. Moreover, even when the model catches its own error, it often parrots the original mistake. We find that the distribution shift of synthetic errors to on-policy errors significantly degrades the error-correction capabilities of the fine-tuned model, even with good synthetic coverage of on-policy errors. Our results help explain why on-policy reinforcement learning methods have proven uniquely effective for eliciting self-correction. 

---
# Beyond Playtesting: A Generative Multi-Agent Simulation System for Massively Multiplayer Online Games 

**Authors**: Ran Zhang, Kun Ouyang, Tiancheng Ma, Yida Yang, Dong Fang  

**Link**: [PDF](https://arxiv.org/pdf/2512.02358)  

**Abstract**: Optimizing numerical systems and mechanism design is crucial for enhancing player experience in Massively Multiplayer Online (MMO) games. Traditional optimization approaches rely on large-scale online experiments or parameter tuning over predefined statistical models, which are costly, time-consuming, and may disrupt player experience. Although simplified offline simulation systems are often adopted as alternatives, their limited fidelity prevents agents from accurately mimicking real player reasoning and reactions to interventions. To address these limitations, we propose a generative agent-based MMO simulation system empowered by Large Language Models (LLMs). By applying Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) on large-scale real player behavioral data, we adapt LLMs from general priors to game-specific domains, enabling realistic and interpretable player decision-making. In parallel, a data-driven environment model trained on real gameplay logs reconstructs dynamic in-game systems. Experiments demonstrate strong consistency with real-world player behaviors and plausible causal responses under interventions, providing a reliable, interpretable, and cost-efficient framework for data-driven numerical design optimization. 

---
# Reasoning Path and Latent State Analysis for Multi-view Visual Spatial Reasoning: A Cognitive Science Perspective 

**Authors**: Qiyao Xue, Weichen Liu, Shiqi Wang, Haoming Wang, Yuyang Wu, Wei Gao  

**Link**: [PDF](https://arxiv.org/pdf/2512.02340)  

**Abstract**: Spatial reasoning is a core aspect of human intelligence that allows perception, inference and planning in 3D environments. However, current vision-language models (VLMs) struggle to maintain geometric coherence and cross-view consistency for spatial reasoning in multi-view settings. We attribute this gap to the lack of fine-grained benchmarks that isolate multi-view reasoning from single-view perception and temporal factors. To address this, we present ReMindView-Bench, a cognitively grounded benchmark for evaluating how VLMs construct, align and maintain spatial mental models across complementary viewpoints. ReMindView-Bench systematically varies viewpoint spatial pattern and query type to probe key factors of spatial cognition. Evaluations of 15 current VLMs reveals consistent failures in cross-view alignment and perspective-taking in multi-view spatial reasoning, motivating deeper analysis on the reasoning process. Explicit phase-wise analysis using LLM-as-a-judge and self-consistency prompting shows that VLMs perform well on in-frame perception but degrade sharply when integrating information across views. Implicit analysis, including linear probing and entropy dynamics, further show progressive loss of task-relevant information and uncertainty separation between correct and incorrect trajectories. These results provide a cognitively grounded diagnosis of VLM spatial reasoning and reveal how multi-view spatial mental models are formed, degraded and destabilized across reasoning phases. The ReMindView-Bench benchmark is available at this https URL, and the source codes of benchmark construction and VLM reasoning analysis are available at this https URL. 

---
# OmniGuard: Unified Omni-Modal Guardrails with Deliberate Reasoning 

**Authors**: Boyu Zhu, Xiaofei Wen, Wenjie Jacky Mo, Tinghui Zhu, Yanan Xie, Peng Qi, Muhao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2512.02306)  

**Abstract**: Omni-modal Large Language Models (OLLMs) that process text, images, videos, and audio introduce new challenges for safety and value guardrails in human-AI interaction. Prior guardrail research largely targets unimodal settings and typically frames safeguarding as binary classification, which limits robustness across diverse modalities and tasks. To address this gap, we propose OmniGuard, the first family of omni-modal guardrails that performs safeguarding across all modalities with deliberate reasoning ability. To support the training of OMNIGUARD, we curate a large, comprehensive omni-modal safety dataset comprising over 210K diverse samples, with inputs that cover all modalities through both unimodal and cross-modal samples. Each sample is annotated with structured safety labels and carefully curated safety critiques from expert models through targeted distillation. Extensive experiments on 15 benchmarks show that OmniGuard achieves strong effectiveness and generalization across a wide range of multimodal safety scenarios. Importantly, OmniGuard provides a unified framework that enforces policies and mitigates risks in omni-modalities, paving the way toward building more robust and capable omnimodal safeguarding systems. 

---
# Breast Cell Segmentation Under Extreme Data Constraints: Quantum Enhancement Meets Adaptive Loss Stabilization 

**Authors**: Varun Kumar Dasoju, Qingsu Cheng, Zeyun Yu  

**Link**: [PDF](https://arxiv.org/pdf/2512.02302)  

**Abstract**: Annotating medical images demands significant time and expertise, often requiring pathologists to invest hundreds of hours in labeling mammary epithelial nuclei datasets. We address this critical challenge by achieving 95.5% Dice score using just 599 training images for breast cell segmentation, where just 4% of pixels represent breast tissue and 60% of images contain no breast regions. Our framework uses quantum-inspired edge enhancement via multi-scale Gabor filters creating a fourth input channel, enhancing boundary detection where inter-annotator variations reach +/- 3 pixels. We present a stabilized multi-component loss function that integrates adaptive Dice loss with boundary-aware terms and automatic positive weighting to effectively address severe class imbalance, where mammary epithelial cell regions comprise only 0.1%-20% of the total image area. Additionally, a complexity-based weighted sampling strategy is introduced to prioritize the challenging mammary epithelial cell regions. The model employs an EfficientNet-B7/UNet++ architecture with a 4-to-3 channel projection, enabling the use of pretrained weights despite limited medical imaging data. Finally, robust validation is achieved through exponential moving averaging and statistical outlier detection, ensuring reliable performance estimates on a small validation set (129 images). Our framework achieves a Dice score of 95.5% +/- 0.3% and an IoU of 91.2% +/- 0.4%. Notably, quantum-based enhancement contributes to a 2.1% improvement in boundary accuracy, while weighted sampling increases small lesion detection by 3.8%. By achieving groundbreaking performance with limited annotations, our approach significantly reduces the medical expert time required for dataset creation, addressing a fundamental bottleneck in clinical perception AI development. 

---
# Model Recovery at the Edge under Resource Constraints for Physical AI 

**Authors**: Bin Xu, Ayan Banerjee, Sandeep K.S. Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2512.02283)  

**Abstract**: Model Recovery (MR) enables safe, explainable decision making in mission-critical autonomous systems (MCAS) by learning governing dynamical equations, but its deployment on edge devices is hindered by the iterative nature of neural ordinary differential equations (NODEs), which are inefficient on FPGAs. Memory and energy consumption are the main concerns when applying MR on edge devices for real-time operation. We propose MERINDA, a novel FPGA-accelerated MR framework that replaces iterative solvers with a parallelizable neural architecture equivalent to NODEs. MERINDA achieves nearly 11x lower DRAM usage and 2.2x faster runtime compared to mobile GPUs. Experiments reveal an inverse relationship between memory and energy at fixed accuracy, highlighting MERINDA's suitability for resource-constrained, real-time MCAS. 

---
# DialogGuard: Multi-Agent Psychosocial Safety Evaluation of Sensitive LLM Responses 

**Authors**: Han Luo, Guy Laban  

**Link**: [PDF](https://arxiv.org/pdf/2512.02282)  

**Abstract**: Large language models (LLMs) now mediate many web-based mental- health, crisis, and other emotionally sensitive services, yet their psychosocial safety in these settings remains poorly understood and weakly evaluated. We present DialogGuard, a multi-agent frame- work for assessing psychosocial risks in LLM-generated responses along five high-severity dimensions: privacy violations, discrimi- natory behaviour, mental manipulation, psychological harm, and insulting behaviour. DialogGuard can be applied to diverse gen- erative models through four LLM-as-a-judge pipelines, including single-agent scoring, dual-agent correction, multi-agent debate, and stochastic majority voting, grounded in a shared three-level rubric usable by both human annotators and LLM judges. Using PKU-SafeRLHF with human safety annotations, we show that multi- agent mechanisms detect psychosocial risks more accurately than non-LLM baselines and single-agent judging; dual-agent correction and majority voting provide the best trade-off between accuracy, alignment with human ratings, and robustness, while debate attains higher recall but over-flags borderline cases. We release Dialog- Guard as open-source software with a web interface that provides per-dimension risk scores and explainable natural-language ratio- nales. A formative study with 12 practitioners illustrates how it supports prompt design, auditing, and supervision of web-facing applications for vulnerable users. 

---
# Bridging the Gap: Toward Cognitive Autonomy in Artificial Intelligence 

**Authors**: Noorbakhsh Amiri Golilarz, Sindhuja Penchala, Shahram Rahimi  

**Link**: [PDF](https://arxiv.org/pdf/2512.02280)  

**Abstract**: Artificial intelligence has advanced rapidly across perception, language, reasoning, and multimodal domains. Yet despite these achievements, modern AI systems remain fun- damentally limited in their ability to self-monitor, self-correct, and regulate their behavior autonomously in dynamic contexts. This paper identifies and analyzes seven core deficiencies that constrain contemporary AI models: the absence of intrinsic self- monitoring, lack of meta-cognitive awareness, fixed and non- adaptive learning mechanisms, inability to restructure goals, lack of representational maintenance, insufficient embodied feedback, and the absence of intrinsic agency. Alongside identifying these limitations, we also outline a forward-looking perspective on how AI may evolve beyond them through architectures that mirror neurocognitive principles. We argue that these structural limitations prevent current architectures, including deep learning and transformer-based systems, from achieving robust general- ization, lifelong adaptability, and real-world autonomy. Drawing on a comparative analysis of artificial systems and biological cognition [7], and integrating insights from AI research, cognitive science, and neuroscience, we outline how these capabilities are absent in current models and why scaling alone cannot resolve them. We conclude by advocating for a paradigmatic shift toward cognitively grounded AI (cognitive autonomy) capable of self-directed adaptation, dynamic representation management, and intentional, goal-oriented behavior, paired with reformative oversight mechanisms [8] that ensure autonomous systems remain interpretable, governable, and aligned with human values. 

---
# TradeTrap: Are LLM-based Trading Agents Truly Reliable and Faithful? 

**Authors**: Lewen Yan, Jilin Mei, Tianyi Zhou, Lige Huang, Jie Zhang, Dongrui Liu, Jing Shao  

**Link**: [PDF](https://arxiv.org/pdf/2512.02261)  

**Abstract**: LLM-based trading agents are increasingly deployed in real-world financial markets to perform autonomous analysis and execution. However, their reliability and robustness under adversarial or faulty conditions remain largely unexamined, despite operating in high-risk, irreversible financial environments. We propose TradeTrap, a unified evaluation framework for systematically stress-testing both adaptive and procedural autonomous trading agents. TradeTrap targets four core components of autonomous trading agents: market intelligence, strategy formulation, portfolio and ledger handling, and trade execution, and evaluates their robustness under controlled system-level perturbations. All evaluations are conducted in a closed-loop historical backtesting setting on real US equity market data with identical initial conditions, enabling fair and reproducible comparisons across agents and attacks. Extensive experiments show that small perturbations at a single component can propagate through the agent decision loop and induce extreme concentration, runaway exposure, and large portfolio drawdowns across both agent types, demonstrating that current autonomous trading agents can be systematically misled at the system level. Our code is available at this https URL. 

---
# Benchmarking LLM Agents for Wealth-Management Workflows 

**Authors**: Rory Milsom  

**Link**: [PDF](https://arxiv.org/pdf/2512.02230)  

**Abstract**: Modern work relies on an assortment of digital collaboration tools, yet routine processes continue to suffer from human error and delay. To address this gap, this dissertation extends TheAgentCompany with a finance-focused environment and investigates whether a general purpose LLM agent can complete representative wealth-management tasks both accurately and economically. This study introduces synthetic domain data, enriches colleague simulations, and prototypes an automatic task-generation pipeline. The study aims to create and assess an evaluation set that can meaningfully measure an agent's fitness for assistant-level wealth management work. We construct a benchmark of 12 task-pairs for wealth management assistants spanning retrieval, analysis, and synthesis/communication, with explicit acceptance criteria and deterministic graders. We seeded a set of new finance-specific data and introduced a high vs. low-autonomy variant of every task. The paper concluded that agents are limited less by mathematical reasoning and more so by end-to-end workflow reliability, and meaningfully affected by autonomy level, and that incorrect evaluation of models have hindered benchmarking. 

---
# STRIDE: A Systematic Framework for Selecting AI Modalities - Agentic AI, AI Assistants, or LLM Calls 

**Authors**: Shubhi Asthana, Bing Zhang, Chad DeLuca, Ruchi Mahindru, Hima Patel  

**Link**: [PDF](https://arxiv.org/pdf/2512.02228)  

**Abstract**: The rapid shift from stateless large language models (LLMs) to autonomous, goal-driven agents raises a central question: When is agentic AI truly necessary? While agents enable multi-step reasoning, persistent memory, and tool orchestration, deploying them indiscriminately leads to higher cost, complexity, and risk.
We present STRIDE (Systematic Task Reasoning Intelligence Deployment Evaluator), a framework that provides principled recommendations for selecting between three modalities: (i) direct LLM calls, (ii) guided AI assistants, and (iii) fully autonomous agentic AI. STRIDE integrates structured task decomposition, dynamism attribution, and self-reflection requirement analysis to produce an Agentic Suitability Score, ensuring that full agentic autonomy is reserved for tasks with inherent dynamism or evolving context.
Evaluated across 30 real-world tasks spanning SRE, compliance, and enterprise automation, STRIDE achieved 92% accuracy in modality selection, reduced unnecessary agent deployments by 45%, and cut resource costs by 37%. Expert validation over six months in SRE and compliance domains confirmed its practical utility, with domain specialists agreeing that STRIDE effectively distinguishes between tasks requiring simple LLM calls, guided assistants, or full agentic autonomy. This work reframes agent adoption as a necessity-driven design decision, ensuring autonomy is applied only when its benefits justify the costs. 

---
# From monoliths to modules: Decomposing transducers for efficient world modelling 

**Authors**: Alexander Boyd, Franz Nowak, David Hyland, Manuel Baltieri, Fernando E. Rosas  

**Link**: [PDF](https://arxiv.org/pdf/2512.02193)  

**Abstract**: World models have been recently proposed as sandbox environments in which AI agents can be trained and evaluated before deployment. Although realistic world models often have high computational demands, efficient modelling is usually possible by exploiting the fact that real-world scenarios tend to involve subcomponents that interact in a modular manner. In this paper, we explore this idea by developing a framework for decomposing complex world models represented by transducers, a class of models generalising POMDPs. Whereas the composition of transducers is well understood, our results clarify how to invert this process, deriving sub-transducers operating on distinct input-output subspaces, enabling parallelizable and interpretable alternatives to monolithic world modelling that can support distributed inference. Overall, these results lay a groundwork for bridging the structural transparency demanded by AI safety and the computational efficiency required for real-world inference. 

---
# Flowchart2Mermaid: A Vision-Language Model Powered System for Converting Flowcharts into Editable Diagram Code 

**Authors**: Pritam Deka, Barry Devereux  

**Link**: [PDF](https://arxiv.org/pdf/2512.02170)  

**Abstract**: Flowcharts are common tools for communicating processes but are often shared as static images that cannot be easily edited or reused. We present \textsc{Flowchart2Mermaid}, a lightweight web system that converts flowchart images into editable this http URL code which is a markup language for visual workflows, using a detailed system prompt and vision-language models. The interface supports mixed-initiative refinement through inline text editing, drag-and-drop node insertion, and natural-language commands interpreted by an integrated AI assistant. Unlike prior image-to-diagram tools, our approach produces a structured, version-controllable textual representation that remains synchronized with the rendered diagram. We further introduce evaluation metrics to assess structural accuracy, flow correctness, syntax validity, and completeness across multiple models. 

---
# The 4/$δ$ Bound: Designing Predictable LLM-Verifier Systems for Formal Method Guarantee 

**Authors**: PIerre Dantas, Lucas Cordeiro, Youcheng Sun, Waldir Junior  

**Link**: [PDF](https://arxiv.org/pdf/2512.02080)  

**Abstract**: The idea of using Formal Verification tools with large language models (LLMs) has enabled scaling software verification beyond manual workflows. However, current methods remain unreliable. Without a solid theoretical footing, the refinement process can wander; sometimes it settles, sometimes it loops back, and sometimes it breaks away from any stable trajectory. This work bridges this critical gap by developing an LLM-Verifier Convergence Theorem, providing the first formal framework with provable guarantees for termination and convergence. We model the interaction between the LLM and the verifier as a discrete-time Markov Chain, with state transitions determined by a key parameter: the error-reduction probability ($\delta$). The procedure reaching the Verified state almost surely demonstrates that the program terminates for any $\delta > 0$, with an expected iteration count bounded by $\mathbb{E}[n] \leq 4/\delta$. We then stress-tested this prediction in an extensive empirical campaign comprising more than 90,000 trials. The empirical results match the theory with striking consistency. Every single run reached verification, and the convergence factor clustered tightly around $C_f\approx$ 1.0. Consequently, the bound mirrors the system's actual behavior. The evidence is sufficiently robust to support dividing the workflow into three distinct operating zones: marginal, practical, and high-performance. Consequently, we establish the design thresholds with absolute confidence. Together, the theoretical guarantee and the experimental evidence provide a clearer architectural foundation for LLM-assisted verification. Heuristic tuning no longer has to be carried out by the system. Engineers gain a framework that supports predictable resource planning and performance budgeting, precisely what is needed before deploying these pipelines into safety-critical software environments. 

---
# PPTArena: A Benchmark for Agentic PowerPoint Editing 

**Authors**: Michael Ofengenden, Yunze Man, Ziqi Pang, Yu-Xiong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.03042)  

**Abstract**: We introduce PPTArena, a benchmark for PowerPoint editing that measures reliable modifications to real slides under natural-language instructions. In contrast to image-PDF renderings or text-to-slide generation, PPTArena focuses on in-place editing across 100 decks, 2125 slides, and over 800 targeted edits covering text, charts, tables, animations, and master-level styles. Each case includes a ground-truth deck, a fully specified target outcome, and a dual VLM-as-judge pipeline that separately scores instruction following and visual quality using both structural diffs and slide images. Building on this setting, we propose PPTPilot, a structure-aware slide-editing agent that plans semantic edit sequences, routes between high-level programmatic tools and deterministic XML operations for precise control, and verifies outputs through an iterative plan-edit-check loop against task-specific constraints. In our experiments, PPTPilot outperforms strong proprietary agents and frontier VLM systems by over 10 percentage points on compound, layout-sensitive, and cross-slide edits, with particularly large gains in visual fidelity and deck-wide consistency. Despite these improvements, existing agents still underperform on long-horizon, document-scale tasks in PPTArena, highlighting the remaining challenges in reliable PPT editing. 

---
# Video4Spatial: Towards Visuospatial Intelligence with Context-Guided Video Generation 

**Authors**: Zeqi Xiao, Yiwei Zhao, Lingxiao Li, Yushi Lan, Yu Ning, Rahul Garg, Roshni Cooper, Mohammad H. Taghavi, Xingang Pan  

**Link**: [PDF](https://arxiv.org/pdf/2512.03040)  

**Abstract**: We investigate whether video generative models can exhibit visuospatial intelligence, a capability central to human cognition, using only visual data. To this end, we present Video4Spatial, a framework showing that video diffusion models conditioned solely on video-based scene context can perform complex spatial tasks. We validate on two tasks: scene navigation - following camera-pose instructions while remaining consistent with 3D geometry of the scene, and object grounding - which requires semantic localization, instruction following, and planning. Both tasks use video-only inputs, without auxiliary modalities such as depth or poses. With simple yet effective design choices in the framework and data curation, Video4Spatial demonstrates strong spatial understanding from video context: it plans navigation and grounds target objects end-to-end, follows camera-pose instructions while maintaining spatial consistency, and generalizes to long contexts and out-of-domain environments. Taken together, these results advance video generative models toward general visuospatial reasoning. 

---
# ViSAudio: End-to-End Video-Driven Binaural Spatial Audio Generation 

**Authors**: Mengchen Zhang, Qi Chen, Tong Wu, Zihan Liu, Dahua Lin  

**Link**: [PDF](https://arxiv.org/pdf/2512.03036)  

**Abstract**: Despite progress in video-to-audio generation, the field focuses predominantly on mono output, lacking spatial immersion. Existing binaural approaches remain constrained by a two-stage pipeline that first generates mono audio and then performs spatialization, often resulting in error accumulation and spatio-temporal inconsistencies. To address this limitation, we introduce the task of end-to-end binaural spatial audio generation directly from silent video. To support this task, we present the BiAudio dataset, comprising approximately 97K video-binaural audio pairs spanning diverse real-world scenes and camera rotation trajectories, constructed through a semi-automated pipeline. Furthermore, we propose ViSAudio, an end-to-end framework that employs conditional flow matching with a dual-branch audio generation architecture, where two dedicated branches model the audio latent flows. Integrated with a conditional spacetime module, it balances consistency between channels while preserving distinctive spatial characteristics, ensuring precise spatio-temporal alignment between audio and the input video. Comprehensive experiments demonstrate that ViSAudio outperforms existing state-of-the-art methods across both objective metrics and subjective evaluations, generating high-quality binaural audio with spatial immersion that adapts effectively to viewpoint changes, sound-source motion, and diverse acoustic environments. Project website: this https URL. 

---
# SMP: Reusable Score-Matching Motion Priors for Physics-Based Character Control 

**Authors**: Yuxuan Mu, Ziyu Zhang, Yi Shi, Minami Matsumoto, Kotaro Imamura, Guy Tevet, Chuan Guo, Michael Taylor, Chang Shu, Pengcheng Xi, Xue Bin Peng  

**Link**: [PDF](https://arxiv.org/pdf/2512.03028)  

**Abstract**: Data-driven motion priors that can guide agents toward producing naturalistic behaviors play a pivotal role in creating life-like virtual characters. Adversarial imitation learning has been a highly effective method for learning motion priors from reference motion data. However, adversarial priors, with few exceptions, need to be retrained for each new controller, thereby limiting their reusability and necessitating the retention of the reference motion data when training on downstream tasks. In this work, we present Score-Matching Motion Priors (SMP), which leverages pre-trained motion diffusion models and score distillation sampling (SDS) to create reusable task-agnostic motion priors. SMPs can be pre-trained on a motion dataset, independent of any control policy or task. Once trained, SMPs can be kept frozen and reused as general-purpose reward functions to train policies to produce naturalistic behaviors for downstream tasks. We show that a general motion prior trained on large-scale datasets can be repurposed into a variety of style-specific priors. Furthermore SMP can compose different styles to synthesize new styles not present in the original dataset. Our method produces high-quality motion comparable to state-of-the-art adversarial imitation learning methods through reusable and modular motion priors. We demonstrate the effectiveness of SMP across a diverse suite of control tasks with physically simulated humanoid characters. Video demo available at this https URL 

---
# The Moral Consistency Pipeline: Continuous Ethical Evaluation for Large Language Models 

**Authors**: Saeid Jamshidi, Kawser Wazed Nafi, Arghavan Moradi Dakhel, Negar Shahabi, Foutse Khomh  

**Link**: [PDF](https://arxiv.org/pdf/2512.03026)  

**Abstract**: The rapid advancement and adaptability of Large Language Models (LLMs) highlight the need for moral consistency, the capacity to maintain ethically coherent reasoning across varied contexts. Existing alignment frameworks, structured approaches designed to align model behavior with human ethical and social norms, often rely on static datasets and post-hoc evaluations, offering limited insight into how ethical reasoning may evolve across different contexts or temporal scales. This study presents the Moral Consistency Pipeline (MoCoP), a dataset-free, closed-loop framework for continuously evaluating and interpreting the moral stability of LLMs. MoCoP combines three supporting layers: (i) lexical integrity analysis, (ii) semantic risk estimation, and (iii) reasoning-based judgment modeling within a self-sustaining architecture that autonomously generates, evaluates, and refines ethical scenarios without external supervision. Our empirical results on GPT-4-Turbo and DeepSeek suggest that MoCoP effectively captures longitudinal ethical behavior, revealing a strong inverse relationship between ethical and toxicity dimensions (correlation rET = -0.81, p value less than 0.001) and a near-zero association with response latency (correlation rEL approximately equal to 0). These findings demonstrate that moral coherence and linguistic safety tend to emerge as stable and interpretable characteristics of model behavior rather than short-term fluctuations. Furthermore, by reframing ethical evaluation as a dynamic, model-agnostic form of moral introspection, MoCoP offers a reproducible foundation for scalable, continuous auditing and advances the study of computational morality in autonomous AI systems. 

---
# LORE: A Large Generative Model for Search Relevance 

**Authors**: Chenji Lu, Zhuo Chen, Hui Zhao, Zhiyuan Zeng, Gang Zhao, Junjie Ren, Ruicong Xu, Haoran Li, Songyan Liu, Pengjie Wang, Jian Xu, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2512.03025)  

**Abstract**: Achievement. We introduce LORE, a systematic framework for Large Generative Model-based relevance in e-commerce search. Deployed and iterated over three years, LORE achieves a cumulative +27\% improvement in online GoodRate metrics. This report shares the valuable experience gained throughout its development lifecycle, spanning data, features, training, evaluation, and deployment. Insight. While existing works apply Chain-of-Thought (CoT) to enhance relevance, they often hit a performance ceiling. We argue this stems from treating relevance as a monolithic task, lacking principled deconstruction. Our key insight is that relevance comprises distinct capabilities: knowledge and reasoning, multi-modal matching, and rule adherence. We contend that a qualitative-driven decomposition is essential for breaking through current performance bottlenecks. Contributions. LORE provides a complete blueprint for the LLM relevance lifecycle. Key contributions include: (1) A two-stage training paradigm combining progressive CoT synthesis via SFT with human preference alignment via RL. (2) A comprehensive benchmark, RAIR, designed to evaluate these core capabilities. (3) A query frequency-stratified deployment strategy that efficiently transfers offline LLM capabilities to the online system. LORE serves as both a practical solution and a methodological reference for other vertical domains. 

---
# TokenPowerBench: Benchmarking the Power Consumption of LLM Inference 

**Authors**: Chenxu Niu, Wei Zhang, Jie Li, Yongjian Zhao, Tongyang Wang, Xi Wang, Yong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2512.03024)  

**Abstract**: Large language model (LLM) services now answer billions of queries per day, and industry reports show that inference, not training, accounts for more than 90% of total power consumption. However, existing benchmarks focus on either training/fine-tuning or performance of inference and provide little support for power consumption measurement and analysis of inference. We introduce TokenPowerBench, the first lightweight and extensible benchmark designed for LLM-inference power consumption studies. The benchmark combines (i) a declarative configuration interface covering model choice, prompt set, and inference engine, (ii) a measurement layer that captures GPU-, node-, and system-level power without specialized power meters, and (iii) a phase-aligned metrics pipeline that attributes energy to the prefill and decode stages of every request. These elements make it straight-forward to explore the power consumed by an LLM inference run; furthermore, by varying batch size, context length, parallelism strategy and quantization, users can quickly assess how each setting affects joules per token and other energy-efficiency metrics. We evaluate TokenPowerBench on four of the most widely used model series (Llama, Falcon, Qwen, and Mistral). Our experiments cover from 1 billion parameters up to the frontier-scale Llama3-405B model. Furthermore, we release TokenPowerBench as open source to help users to measure power consumption, forecast operating expenses, and meet sustainability targets when deploying LLM services. 

---
# Distribution-Calibrated Inference time compute for Thinking LLM-as-a-Judge 

**Authors**: Hamid Dadkhahi, Firas Trabelsi, Parker Riley, Juraj Juraska, Mehdi Mirzazadeh  

**Link**: [PDF](https://arxiv.org/pdf/2512.03019)  

**Abstract**: Thinking Large Language Models (LLMs) used as judges for pairwise preferences remain noisy at the single-sample level, and common aggregation rules (majority vote, soft self-consistency, or instruction-based self-aggregation) are inconsistent when ties are allowed. We study inference-time compute (ITC) for evaluators that generate n independent thinking-rating samples per item, and propose a principled, distribution-calibrated aggregation scheme. Our method models three-way preferences with a Bradley-Terry-Davidson formulation on rating counts, leveraging both polarity (margin among non-ties) and decisiveness (non-tie rate) to distinguish narrow margins from strong consensus. Across various evaluation benchmarks, our approach consistently reduces MAE and increases pairwise accuracy versus standard baselines, and when evaluated against human-consensus meta-labels, matches or exceeds individual human raters. These results show that carefully allocating ITC and aggregating with distribution-aware methods turns noisy individual model judgments into reliable ratings for evaluation. 

---
# In-Context Sync-LoRA for Portrait Video Editing 

**Authors**: Sagi Polaczek, Or Patashnik, Ali Mahdavi-Amiri, Daniel Cohen-Or  

**Link**: [PDF](https://arxiv.org/pdf/2512.03013)  

**Abstract**: Editing portrait videos is a challenging task that requires flexible yet precise control over a wide range of modifications, such as appearance changes, expression edits, or the addition of objects. The key difficulty lies in preserving the subject's original temporal behavior, demanding that every edited frame remains precisely synchronized with the corresponding source frame. We present Sync-LoRA, a method for editing portrait videos that achieves high-quality visual modifications while maintaining frame-accurate synchronization and identity consistency. Our approach uses an image-to-video diffusion model, where the edit is defined by modifying the first frame and then propagated to the entire sequence. To enable accurate synchronization, we train an in-context LoRA using paired videos that depict identical motion trajectories but differ in appearance. These pairs are automatically generated and curated through a synchronization-based filtering process that selects only the most temporally aligned examples for training. This training setup teaches the model to combine motion cues from the source video with the visual changes introduced in the edited first frame. Trained on a compact, highly curated set of synchronized human portraits, Sync-LoRA generalizes to unseen identities and diverse edits (e.g., modifying appearance, adding objects, or changing backgrounds), robustly handling variations in pose and expression. Our results demonstrate high visual fidelity and strong temporal coherence, achieving a robust balance between edit fidelity and precise motion preservation. 

---
# Fine-Tuned Large Language Models for Logical Translation: Reducing Hallucinations with Lang2Logic 

**Authors**: Muyu Pan, Dheeraj Kodakandla, Mahfuza Farooque  

**Link**: [PDF](https://arxiv.org/pdf/2512.02987)  

**Abstract**: Recent advances in natural language processing (NLP), particularly large language models (LLMs), have motivated the automatic translation of natural language statements into formal logic without human intervention. This enables automated reasoning and facilitates debugging, finding loop invariants, and adhering to specifications in software systems. However, hallucinations-incorrect outputs generated by LLMs are challenging, particularly for logical translation tasks requiring precision. This work introduces a novel framework that inputs English sentences, converts them into logical expressions, and then translates them into Conjunctive Normal Form (CNF) for satisfiability solving. It employs classical NLP techniques with self-defined grammar, symbolic computation libraries, and a fine-tuned language model to reduce hallucinations. In the early experiments, we observed that the fine-tuned model, trained on different grammar settings, could intentionally correct the same types of hallucinations made by the original model. Thus, it provides reliable CNF generation. 

---
# Rethinking Generalized BCIs: Benchmarking 340,000+ Unique Algorithmic Configurations for EEG Mental Command Decoding 

**Authors**: Paul Barbaste, Olivier Oullier, Xavier Vasques  

**Link**: [PDF](https://arxiv.org/pdf/2512.02978)  

**Abstract**: Robust decoding and classification of brain patterns measured with electroencephalography (EEG) remains a major challenge for real-world (i.e. outside scientific lab and medical facilities) brain-computer interface (BCI) applications due to well documented inter- and intra-participant variability. Here, we present a large-scale benchmark evaluating over 340,000+ unique combinations of spatial and nonlinear EEG classification. Our methodological pipeline consists in combinations of Common Spatial Patterns (CSP), Riemannian geometry, functional connectivity, and fractal- or entropy-based features across three open-access EEG datasets. Unlike prior studies, our analysis operates at the per-participant level and across multiple frequency bands (8-15 Hz and 8-30 Hz), enabling direct assessment of both group-level performance and individual variability. Covariance tangent space projection (cov-tgsp) and CSP consistently achieved the highest average classification accuracies. However, their effectiveness was strongly dataset-dependent, and marked participant-level differences persisted, particularly in the most heterogeneous of the datasets. Importantly, nonlinear methods outperformed spatial approaches for specific individuals, underscoring the need for personalized pipeline selection. Our findings highlight that no universal 'one-size-fits-all' method can optimally decode EEG motor imagery patterns across all users or datasets. Future work will require adaptive, multimodal, and possibly novel approaches to fully address neurophysiological variability in practical BCI applications where the system can automatically adapt to what makes each user unique. 

---
# Lumos: Let there be Language Model System Certification 

**Authors**: Isha Chaudhary, Vedaant Jain, Avaljot Singh, Kavya Sachdeva, Sayan Ranu, Gagandeep Singh  

**Link**: [PDF](https://arxiv.org/pdf/2512.02966)  

**Abstract**: We introduce the first principled framework, Lumos, for specifying and formally certifying Language Model System (LMS) behaviors. Lumos is an imperative probabilistic programming DSL over graphs, with constructs to generate independent and identically distributed prompts for LMS. It offers a structured view of prompt distributions via graphs, forming random prompts from sampled subgraphs. Lumos supports certifying LMS for arbitrary prompt distributions via integration with statistical certifiers. We provide hybrid (operational and denotational) semantics for Lumos, providing a rigorous way to interpret the specifications. Using only a small set of composable constructs, Lumos can encode existing LMS specifications, including complex relational and temporal specifications. It also facilitates specifying new properties - we present the first safety specifications for vision-language models (VLMs) in autonomous driving scenarios developed with Lumos. Using these, we show that the state-of-the-art VLM Qwen-VL exhibits critical safety failures, producing incorrect and unsafe responses with at least 90% probability in right-turn scenarios under rainy driving conditions, revealing substantial safety risks. Lumos's modular structure allows easy modification of the specifications, enabling LMS certification to stay abreast with the rapidly evolving threat landscape. We further demonstrate that specification programs written in Lumos enable finding specific failure cases exhibited by state-of-the-art LMS. Lumos is the first systematic and extensible language-based framework for specifying and certifying LMS behaviors, paving the way for a wider adoption of LMS certification. 

---
# Benchmarking Scientific Understanding and Reasoning for Video Generation using VideoScience-Bench 

**Authors**: Lanxiang Hu, Abhilash Shankarampeta, Yixin Huang, Zilin Dai, Haoyang Yu, Yujie Zhao, Haoqiang Kang, Daniel Zhao, Tajana Rosing, Hao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2512.02942)  

**Abstract**: The next frontier for video generation lies in developing models capable of zero-shot reasoning, where understanding real-world scientific laws is crucial for accurate physical outcome modeling under diverse conditions. However, existing video benchmarks are physical commonsense-based, offering limited insight into video models' scientific reasoning capability. We introduce VideoScience-Bench, a benchmark designed to evaluate undergraduate-level scientific understanding in video models. Each prompt encodes a composite scientific scenario that requires understanding and reasoning across multiple scientific concepts to generate the correct phenomenon. The benchmark comprises 200 carefully curated prompts spanning 14 topics and 103 concepts in physics and chemistry. We conduct expert-annotated evaluations across seven state-of-the-art video models in T2V and I2V settings along five dimensions: Prompt Consistency, Phenomenon Congruency, Correct Dynamism, Immutability, and Spatio-Temporal Continuity. Using a VLM-as-a-Judge to assess video generations, we observe strong correlation with human assessments. To the best of our knowledge, VideoScience-Bench is the first benchmark to evaluate video models not only as generators but also as reasoners, requiring their generations to demonstrate scientific understanding consistent with expected physical and chemical phenomena. Our data and evaluation code are available at: \href{this https URL}{this http URL}. 

---
# EGGS: Exchangeable 2D/3D Gaussian Splatting for Geometry-Appearance Balanced Novel View Synthesis 

**Authors**: Yancheng Zhang, Guangyu Sun, Chen Chen  

**Link**: [PDF](https://arxiv.org/pdf/2512.02932)  

**Abstract**: Novel view synthesis (NVS) is crucial in computer vision and graphics, with wide applications in AR, VR, and autonomous driving. While 3D Gaussian Splatting (3DGS) enables real-time rendering with high appearance fidelity, it suffers from multi-view inconsistencies, limiting geometric accuracy. In contrast, 2D Gaussian Splatting (2DGS) enforces multi-view consistency but compromises texture details. To address these limitations, we propose Exchangeable Gaussian Splatting (EGGS), a hybrid representation that integrates 2D and 3D Gaussians to balance appearance and geometry. To achieve this, we introduce Hybrid Gaussian Rasterization for unified rendering, Adaptive Type Exchange for dynamic adaptation between 2D and 3D Gaussians, and Frequency-Decoupled Optimization that effectively exploits the strengths of each type of Gaussian representation. Our CUDA-accelerated implementation ensures efficient training and inference. Extensive experiments demonstrate that EGGS outperforms existing methods in rendering quality, geometric accuracy, and efficiency, providing a practical solution for high-quality NVS. 

---
# In Silico Development of Psychometric Scales: Feasibility of Representative Population Data Simulation with LLMs 

**Authors**: Enrico Cipriani, Pavel Okopnyi, Danilo Menicucci, Simone Grassini  

**Link**: [PDF](https://arxiv.org/pdf/2512.02910)  

**Abstract**: Developing and validating psychometric scales requires large samples, multiple testing phases, and substantial resources. Recent advances in Large Language Models (LLMs) enable the generation of synthetic participant data by prompting models to answer items while impersonating individuals of specific demographic profiles, potentially allowing in silico piloting before real data collection. Across four preregistered studies (N = circa 300 each), we tested whether LLM-simulated datasets can reproduce the latent structures and measurement properties of human responses. In Studies 1-2, we compared LLM-generated data with real datasets for two validated scales; in Studies 3-4, we created new scales using EFA on simulated data and then examined whether these structures generalized to newly collected human samples. Simulated datasets replicated the intended factor structures in three of four studies and showed consistent configural and metric invariance, with scalar invariance achieved for the two newly developed scales. However, correlation-based tests revealed substantial differences between real and synthetic datasets, and notable discrepancies appeared in score distributions and variances. Thus, while LLMs capture group-level latent structures, they do not approximate individual-level data properties. Simulated datasets also showed full internal invariance across gender. Overall, LLM-generated data appear useful for early-stage, group-level psychometric prototyping, but not as substitutes for individual-level validation. We discuss methodological limitations, risks of bias and data pollution, and ethical considerations related to in silico psychometric simulations. 

---
# MRD: Multi-resolution Retrieval-Detection Fusion for High-Resolution Image Understanding 

**Authors**: Fan Yang, Kaihao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2512.02906)  

**Abstract**: Understanding high-resolution images remains a significant challenge for multimodal large language models (MLLMs). Recent study address this issue by dividing the image into smaller crops and computing the semantic similarity between each crop and a query using a pretrained retrieval-augmented generation (RAG) model. The most relevant crops are then selected to localize the target object and suppress irrelevant information. However, such crop-based processing can fragment complete objects across multiple crops, thereby disrupting the computation of semantic similarity. In our experiments, we find that image crops of objects with different sizes are better handled at different resolutions. Based on this observation, we propose Multi-resolution Retrieval-Detection (MRD), a training-free framework for high-resolution image understanding. To address the issue of semantic similarity bias caused by objects being split across different image crops, we propose a multi-resolution semantic fusion method, which integrates semantic similarity maps obtained at different resolutions to produce more accurate semantic information and preserve the integrity of target objects. Furthermore, to achieve direct localization of target objects at a global scale, we introduce an open-vocalbulary object detection (OVD) model that identifies object regions using a sliding-window this http URL on high-resolution image understanding benchmarks using different MLLMs demonstrate the effectiveness of our approach. 

---
# Towards a fully differentiable digital twin for solar cells 

**Authors**: Marie Louise Schubert, Houssam Metni, Jan David Fischbach, Benedikt Zerulla, Marjan Krstić, Ulrich W. Paetzold, Seyedamir Orooji, Olivier J. J. Ronsin, Yasin Ameslon, Jens Harting, Thomas Kirchartz, Sandheep Ravishankar, Chris Dreessen, Eunchi Kim, Christian Sprau, Mohamed Hussein, Alexander Colsmann, Karen Forberich, Klaus Jäger, Pascal Friederich, Carsten Rockstuhl  

**Link**: [PDF](https://arxiv.org/pdf/2512.02904)  

**Abstract**: Maximizing energy yield (EY) - the total electric energy generated by a solar cell within a year at a specific location - is crucial in photovoltaics (PV), especially for emerging technologies. Computational methods provide the necessary insights and guidance for future research. However, existing simulations typically focus on only isolated aspects of solar cells. This lack of consistency highlights the need for a framework unifying all computational levels, from material to cell properties, for accurate prediction and optimization of EY prediction. To address this challenge, a differentiable digital twin, Sol(Di)$^2$T, is introduced to enable comprehensive end-to-end optimization of solar cells. The workflow starts with material properties and morphological processing parameters, followed by optical and electrical simulations. Finally, climatic conditions and geographic location are incorporated to predict the EY. Each step is either intrinsically differentiable or replaced with a machine-learned surrogate model, enabling not only accurate EY prediction but also gradient-based optimization with respect to input parameters. Consequently, Sol(Di)$^2$T extends EY predictions to previously unexplored conditions. Demonstrated for an organic solar cell, the proposed framework marks a significant step towards tailoring solar cells for specific applications while ensuring maximal performance. 

---
# VLA Models Are More Generalizable Than You Think: Revisiting Physical and Spatial Modeling 

**Authors**: Weiqi Li, Quande Zhang, Ruifeng Zhai, Liang Lin, Guangrun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.02902)  

**Abstract**: Vision-language-action (VLA) models achieve strong in-distribution performance but degrade sharply under novel camera viewpoints and visual perturbations. We show that this brittleness primarily arises from misalignment in Spatial Modeling, rather than Physical Modeling. To address this, we propose a one-shot adaptation framework that recalibrates visual representations through lightweight, learnable updates. Our first method, Feature Token Modulation (FTM), applies a global affine transformation to visual tokens and improves Libero viewpoint accuracy from 48.5% to 87.1% with only 4K parameters. Building on this, Feature Linear Adaptation (FLA) introduces low-rank updates to the ViT encoder, achieving 90.8% success with 4.7M parameters -- matching LoRA-scale finetuning at far lower cost. Together, these results reveal substantial untapped robustness in pretrained VLA models and demonstrate that targeted, minimal visual adaptation is sufficient to restore viewpoint generalization. 

---
# FAIRY2I: Universal Extremely-Low Bit QAT framework via Widely-Linear Representation and Phase-Aware Quantization 

**Authors**: Feiyu Wang, Xinyu Tan, Bokai Huang, Yihao Zhang, Guoan Wang, Peizhuang Cong, Tong Yang  

**Link**: [PDF](https://arxiv.org/pdf/2512.02901)  

**Abstract**: Large language models (LLMs) have revolutionized artificial intelligence, yet their massive memory and computational demands necessitate aggressive quantization, increasingly pushing representations toward the theoretical limit of a single bit. While complex-valued LLMs, such as iFairy, offer a superior chance for low-bit representation compared to real-valued counterparts, they require training from scratch, preventing the utilization of the vast ecosystem of pre-trained real-valued foundation models. Here we present Fairy2i, a universal framework that transforms pre-trained real-valued layers into an equivalent widely-linear complex form, enabling extremely low-bit quantization while reusing existing checkpoints. By proving a lossless mathematical equivalence between real and widely-linear maps, we convert standard Transformers into the complex domain and employ a phase-aware quantization scheme with a highly efficient codebook of fourth roots of unity. Furthermore, we introduce a recursive residual quantization mechanism that iteratively minimizes quantization error, allowing inference to proceed via efficient multiplication-free accumulation. We demonstrate that Fairy2i restores the performance of LLaMA-2 7B at an effective 2-bit precision to levels nearly comparable with full-precision baselines, significantly outperforming state-of-the-art real-valued binary and ternary quantization methods. This work bridges the gap between the representational efficiency of complex-valued arithmetic and the practical utility of pre-trained models, paving a new way for efficient inference on commodity hardware. 

---
# Model-Based Diagnosis with Multiple Observations: A Unified Approach for C Software and Boolean Circuits 

**Authors**: Pedro Orvalho, Marta Kwiatkowska, Mikoláš Janota, Vasco Manquinho  

**Link**: [PDF](https://arxiv.org/pdf/2512.02898)  

**Abstract**: Debugging is one of the most time-consuming and expensive tasks in software development and circuit design. Several formula-based fault localisation (FBFL) methods have been proposed, but they fail to guarantee a set of diagnoses across all failing tests or may produce redundant diagnoses that are not subset-minimal, particularly for programs/circuits with multiple faults.
This paper introduces CFaults, a novel fault localisation tool for C software and Boolean circuits with multiple faults. CFaults leverages Model-Based Diagnosis (MBD) with multiple observations and aggregates all failing test cases into a unified Maximum Satisfiability (MaxSAT) formula. Consequently, our method guarantees consistency across observations and simplifies the fault localisation procedure. Experimental results on three benchmark sets, two of C programs, TCAS and C-Pack-IPAs, and one of Boolean circuits, ISCAS85, show that CFaults is faster at localising faults in C software than other FBFL approaches such as BugAssist, SNIPER, and HSD. On the ISCAS85 benchmark, CFaults is generally slower than HSD; however, it localises faults in only 6% fewer circuits, demonstrating that it remains competitive in this domain. Furthermore, CFaults produces only subset-minimal diagnoses of faulty statements, whereas the other approaches tend to enumerate redundant diagnoses (e.g., BugAssist and SNIPER). 

---
# OptPO: Optimal Rollout Allocation for Test-time Policy Optimization 

**Authors**: Youkang Wang, Jian Wang, Rubing Chen, Tianyi Zeng, Xiao-Yong Wei, Qing Li  

**Link**: [PDF](https://arxiv.org/pdf/2512.02882)  

**Abstract**: Test-time policy optimization enables large language models (LLMs) to adapt to distribution shifts by leveraging feedback from self-generated rollouts. However, existing methods rely on fixed-budget majority voting to estimate rewards, incurring substantial computational redundancy. We propose Optimal Rollout Allocation for Test-time Policy Optimization (OptPO), a principled framework that adaptively allocates inference budgets. By formulating the voting process as a Bayesian sequential probability ratio test, OptPO dynamically halts sampling once the posterior confidence in a consensus answer exceeds a specified threshold. Crucially, it utilizes the retained rollouts for on-policy updates, seamlessly integrating with algorithms like PPO or GRPO without requiring ground-truth labels. Across diverse reasoning benchmarks, OptPO significantly reduces rollout overhead compared to fixed-sample baselines while preserving or improving accuracy. By unifying statistically optimal stopping with test-time learning, OptPO offers a computationally efficient paradigm for test-time adaptation. The source code will be open upon acceptance at https://open-upon-acceptance. 

---
# GraphMatch: Fusing Language and Graph Representations in a Dynamic Two-Sided Work Marketplace 

**Authors**: Mikołaj Sacha, Hammad Jafri, Mattie Terzolo, Ayan Sinha, Andrew Rabinovich  

**Link**: [PDF](https://arxiv.org/pdf/2512.02849)  

**Abstract**: Recommending matches in a text-rich, dynamic two-sided marketplace presents unique challenges due to evolving content and interaction graphs. We introduce GraphMatch, a new large-scale recommendation framework that fuses pre-trained language models with graph neural networks to overcome these challenges. Unlike prior approaches centered on standalone models, GraphMatch is a comprehensive recipe built on powerful text encoders and GNNs working in tandem. It employs adversarial negative sampling alongside point-in-time subgraph training to learn representations that capture both the fine-grained semantics of evolving text and the time-sensitive structure of the graph. We evaluated extensively on interaction data from Upwork, a leading labor marketplace, at large scale, and discuss our approach towards low-latency inference suitable for real-time use. In our experiments, GraphMatch outperforms language-only and graph-only baselines on matching tasks while being efficient at runtime. These results demonstrate that unifying language and graph representations yields a highly effective solution to text-rich, dynamic two-sided recommendations, bridging the gap between powerful pretrained LMs and large-scale graphs in practice. 

---
# Cross-Lingual Prompt Steerability: Towards Accurate and Robust LLM Behavior across Languages 

**Authors**: Lechen Zhang, Yusheng Zhou, Tolga Ergen, Lajanugen Logeswaran, Moontae Lee, David Jurgens  

**Link**: [PDF](https://arxiv.org/pdf/2512.02841)  

**Abstract**: System prompts provide a lightweight yet powerful mechanism for conditioning large language models (LLMs) at inference time. While prior work has focused on English-only settings, real-world deployments benefit from having a single prompt to operate reliably across languages. This paper presents a comprehensive study of how different system prompts steer models toward accurate and robust cross-lingual behavior. We propose a unified four-dimensional evaluation framework to assess system prompts in multilingual environments. Through large-scale experiments on five languages, three LLMs, and three benchmarks, we uncover that certain prompt components, such as CoT, emotion, and scenario, correlate with robust multilingual behavior. We develop a prompt optimization framework for multilingual settings and show it can automatically discover prompts that improve all metrics by 5-10%. Finally, we analyze over 10 million reasoning units and find that more performant system prompts induce more structured and consistent reasoning patterns, while reducing unnecessary language-switching. Together, we highlight system prompt optimization as a scalable path to accurate and robust multilingual LLM behavior. 

---
# ReVSeg: Incentivizing the Reasoning Chain for Video Segmentation with Reinforcement Learning 

**Authors**: Yifan Li, Yingda Yin, Lingting Zhu, Weikai Chen, Shengju Qian, Xin Wang, Yanwei Fu  

**Link**: [PDF](https://arxiv.org/pdf/2512.02835)  

**Abstract**: Reasoning-centric video object segmentation is an inherently complex task: the query often refers to dynamics, causality, and temporal interactions, rather than static appearances. Yet existing solutions generally collapse these factors into simplified reasoning with latent embeddings, rendering the reasoning chain opaque and essentially intractable. We therefore adopt an explicit decomposition perspective and introduce ReVSeg, which executes reasoning as sequential decisions in the native interface of pretrained vision language models (VLMs). Rather than folding all reasoning into a single-step prediction, ReVSeg executes three explicit operations -- semantics interpretation, temporal evidence selection, and spatial grounding -- aligning pretrained capabilities. We further employ reinforcement learning to optimize the multi-step reasoning chain, enabling the model to self-refine its decision quality from outcome-driven signals. Experimental results demonstrate that ReVSeg attains state-of-the-art performances on standard video object segmentation benchmarks and yields interpretable reasoning trajectories. Project page is available at this https URL . 

---
# Steering Vision-Language-Action Models as Anti-Exploration: A Test-Time Scaling Approach 

**Authors**: Siyuan Yang, Yang Zhang, Haoran He, Ling Pan, Xiu Li, Chenjia Bai, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2512.02834)  

**Abstract**: Vision-Language-Action (VLA) models, trained via flow-matching or diffusion objectives, excel at learning complex behaviors from large-scale, multi-modal datasets (e.g., human teleoperation, scripted policies). However, since VLAs incorporate diverse data modes in the pre-training stage, and the finetuning dataset often contains demonstration data collected in a kinematically suboptimal or undesirable way, it exists redundant action modes that are irrelevant to the success action modes of the downstream task. Specifically, we observe a critical inference-time fragility among various sampled noises after supervised finetuning of pre-trained VLAs. In this paper, we attribute this instability to the distribution shift between the VLA policy and the policy induced by stable success modes of the downstream task dataset. Thus, we propose \textbf{TACO}, a test-time-scaling (TTS) framework that applies a lightweight pseudo-count estimator as a high-fidelity verifier of action chunks. The VLA models integrated with TACO can execute the actions with maximum pseudo-count from all sampled action chunks, thereby preventing distribution shifts while preserving the generalization ability of VLAs since the constraint is applied only during inference. Our method resembles the classical anti-exploration principle in offline reinforcement learning (RL), and being gradient-free, it incurs significant computational benefits compared to RL update, especially for flow or diffusion-based VLAs which are difficult to perform RL update due to denoising process. Extensive experiments across four simulation benchmarks (RoboTwin2.0, Robotwin, LIBERO, SimplerEnv) and a dual-arm platform demonstrate that our method significantly improves the inference stability and success rates in downstream-task adaptations. 

---
# A Comparative Study on How Data Normalization Affects Zero-Shot Generalization in Time Series Foundation Models 

**Authors**: Ihab Ahmed, Denis Krompaß, Cheng Feng, Volker Tresp  

**Link**: [PDF](https://arxiv.org/pdf/2512.02833)  

**Abstract**: We investigate input normalization methods for Time-Series Foundation Models (TSFMs). While normalization is well-studied in dataset-specific time-series models, it remains overlooked in TSFMs where generalization is critical. Time-series data, unlike text or images, exhibits significant scale variation across domains and channels, coupled with non-stationarity, can undermine TSFM performance regardless of architectural complexity. Through systematic evaluation across four architecturally diverse TSFMs, we empirically establish REVIN as the most efficient approach, reducing zero-shot MASE by 89\% relative to an un-normalized baseline and by 44\% versus other normalization methods, while matching the best in-domain accuracy (0.84 MASE) without any dataset-level preprocessing -- yielding the highest accuracy-efficiency trade-off. Yet its effect utilization depends on architectural design choices and optimization objective, particularly with respect to training loss scale sensitivity and model type (probabilistic, point-forecast, or LLM-based models). 

---
# Defense That Attacks: How Robust Models Become Better Attackers 

**Authors**: Mohamed Awad, Mahmoud Akrm, Walid Gomaa  

**Link**: [PDF](https://arxiv.org/pdf/2512.02830)  

**Abstract**: Deep learning has achieved great success in computer vision, but remains vulnerable to adversarial attacks. Adversarial training is the leading defense designed to improve model robustness. However, its effect on the transferability of attacks is underexplored. In this work, we ask whether adversarial training unintentionally increases the transferability of adversarial examples. To answer this, we trained a diverse zoo of 36 models, including CNNs and ViTs, and conducted comprehensive transferability experiments. Our results reveal a clear paradox: adversarially trained (AT) models produce perturbations that transfer more effectively than those from standard models, which introduce a new ecosystem risk. To enable reproducibility and further study, we release all models, code, and experimental scripts. Furthermore, we argue that robustness evaluations should assess not only the resistance of a model to transferred attacks but also its propensity to produce transferable adversarial examples. 

---
# From Navigation to Refinement: Revealing the Two-Stage Nature of Flow-based Diffusion Models through Oracle Velocity 

**Authors**: Haoming Liu, Jinnuo Liu, Yanhao Li, Liuyang Bai, Yunkai Ji, Yuanhe Guo, Shenji Wan, Hongyi Wen  

**Link**: [PDF](https://arxiv.org/pdf/2512.02826)  

**Abstract**: Flow-based diffusion models have emerged as a leading paradigm for training generative models across images and videos. However, their memorization-generalization behavior remains poorly understood. In this work, we revisit the flow matching (FM) objective and study its marginal velocity field, which admits a closed-form expression, allowing exact computation of the oracle FM target. Analyzing this oracle velocity field reveals that flow-based diffusion models inherently formulate a two-stage training target: an early stage guided by a mixture of data modes, and a later stage dominated by the nearest data sample. The two-stage objective leads to distinct learning behaviors: the early navigation stage generalizes across data modes to form global layouts, whereas the later refinement stage increasingly memorizes fine-grained details. Leveraging these insights, we explain the effectiveness of practical techniques such as timestep-shifted schedules, classifier-free guidance intervals, and latent space design choices. Our study deepens the understanding of diffusion model training dynamics and offers principles for guiding future architectural and algorithmic improvements. 

---
# Phase-Adaptive LLM Framework with Multi-Stage Validation for Construction Robot Task Allocation: A Systematic Benchmark Against Traditional Optimization Algorithms 

**Authors**: Shyam prasad reddy Kaitha, Hongrui Yu  

**Link**: [PDF](https://arxiv.org/pdf/2512.02810)  

**Abstract**: Multi-robot task allocation in construction automation has traditionally relied on optimization methods such as Dynamic Programming and Reinforcement Learning. This research introduces the LangGraph-based Task Allocation Agent (LTAA), an LLM-driven framework that integrates phase-adaptive allocation strategies, multi-stage validation with hierarchical retries, and dynamic prompting for efficient robot coordination. Although recent LLM approaches show potential for construction robotics, they largely lack rigorous validation and benchmarking against established algorithms. This paper presents the first systematic comparison of LLM-based task allocation with traditional methods in construction this http URL study validates LLM feasibility through SMART-LLM replication and addresses implementation challenges using a Self-Corrective Agent Architecture. LTAA leverages natural-language reasoning combined with structured validation mechanisms, achieving major computational gains reducing token usage by 94.6% and allocation time by 86% through dynamic prompting. The framework adjusts its strategy across phases: emphasizing execution feasibility early and workload balance in later this http URL authors evaluate LTAA against Dynamic Programming, Q-learning, and Deep Q-Network (DQN) baselines using construction operations from the TEACh human-robot collaboration dataset. In the Heavy Excels setting, where robots have strong task specializations, LTAA achieves 77% task completion with superior workload balance, outperforming all traditional methods. These findings show that LLM-based reasoning with structured validation can match established optimization algorithms while offering additional advantages such as interpretability, adaptability, and the ability to update task logic without retraining. 

---
# Perception of AI-Generated Music - The Role of Composer Identity, Personality Traits, Music Preferences, and Perceived Humanness 

**Authors**: David Stammer, Hannah Strauss, Peter Knees  

**Link**: [PDF](https://arxiv.org/pdf/2512.02785)  

**Abstract**: The rapid rise of AI-generated art has sparked debate about potential biases in how audiences perceive and evaluate such works. This study investigates how composer information and listener characteristics shape the perception of AI-generated music, adopting a mixed-method approach. Using a diverse set of stimuli across various genres from two AI music models, we examine effects of perceived authorship on liking and emotional responses, and explore how attitudes toward AI, personality traits, and music-related variables influence evaluations. We further assess the influence of perceived humanness and analyze open-ended responses to uncover listener criteria for judging AI-generated music. Attitudes toward AI proved to be the best predictor of both liking and emotional intensity of AI-generated music. This quantitative finding was complemented by qualitative themes from our thematic analysis, which identified ethical, cultural, and contextual considerations as important criteria in listeners' evaluations of AI-generated music. Our results offer a nuanced view of how people experience music created by AI tools and point to key factors and methodological considerations for future research on music perception in human-AI interaction. 

---
# SurveyEval: Towards Comprehensive Evaluation of LLM-Generated Academic Surveys 

**Authors**: Jiahao Zhao, Shuaixing Zhang, Nan Xu, Lei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.02763)  

**Abstract**: LLM-based automatic survey systems are transforming how users acquire information from the web by integrating retrieval, organization, and content synthesis into end-to-end generation pipelines. While recent works focus on developing new generation pipelines, how to evaluate such complex systems remains a significant challenge. To this end, we introduce SurveyEval, a comprehensive benchmark that evaluates automatically generated surveys across three dimensions: overall quality, outline coherence, and reference accuracy. We extend the evaluation across 7 subjects and augment the LLM-as-a-Judge framework with human references to strengthen evaluation-human alignment. Evaluation results show that while general long-text or paper-writing systems tend to produce lower-quality surveys, specialized survey-generation systems are able to deliver substantially higher-quality results. We envision SurveyEval as a scalable testbed to understand and improve automatic survey systems across diverse subjects and evaluation criteria. 

---
# Reasoning-Aware Multimodal Fusion for Hateful Video Detection 

**Authors**: Shuonan Yang, Tailin Chen, Jiangbei Yue, Guangliang Cheng, Jianbo Jiao, Zeyu Fu  

**Link**: [PDF](https://arxiv.org/pdf/2512.02743)  

**Abstract**: Hate speech in online videos is posing an increasingly serious threat to digital platforms, especially as video content becomes increasingly multimodal and context-dependent. Existing methods often struggle to effectively fuse the complex semantic relationships between modalities and lack the ability to understand nuanced hateful content. To address these issues, we propose an innovative Reasoning-Aware Multimodal Fusion (RAMF) framework. To tackle the first challenge, we design Local-Global Context Fusion (LGCF) to capture both local salient cues and global temporal structures, and propose Semantic Cross Attention (SCA) to enable fine-grained multimodal semantic interaction. To tackle the second challenge, we introduce adversarial reasoning-a structured three-stage process where a vision-language model generates (i) objective descriptions, (ii) hate-assumed inferences, and (iii) non-hate-assumed inferences-providing complementary semantic perspectives that enrich the model's contextual understanding of nuanced hateful intent. Evaluations on two real-world hateful video datasets demonstrate that our method achieves robust generalisation performance, improving upon state-of-the-art methods by 3% and 7% in Macro-F1 and hate class recall, respectively. We will release the code after the anonymity period ends. 

---
# DF-Mamba: Deformable State Space Modeling for 3D Hand Pose Estimation in Interactions 

**Authors**: Yifan Zhou, Takehiko Ohkawa, Guwenxiao Zhou, Kanoko Goto, Takumi Hirose, Yusuke Sekikawa, Nakamasa Inoue  

**Link**: [PDF](https://arxiv.org/pdf/2512.02727)  

**Abstract**: Modeling daily hand interactions often struggles with severe occlusions, such as when two hands overlap, which highlights the need for robust feature learning in 3D hand pose estimation (HPE). To handle such occluded hand images, it is vital to effectively learn the relationship between local image features (e.g., for occluded joints) and global context (e.g., cues from inter-joints, inter-hands, or the scene). However, most current 3D HPE methods still rely on ResNet for feature extraction, and such CNN's inductive bias may not be optimal for 3D HPE due to its limited capability to model the global context. To address this limitation, we propose an effective and efficient framework for visual feature extraction in 3D HPE using recent state space modeling (i.e., Mamba), dubbed Deformable Mamba (DF-Mamba). DF-Mamba is designed to capture global context cues beyond standard convolution through Mamba's selective state modeling and the proposed deformable state scanning. Specifically, for local features after convolution, our deformable scanning aggregates these features within an image while selectively preserving useful cues that represent the global context. This approach significantly improves the accuracy of structured 3D HPE, with comparable inference speed to ResNet-50. Our experiments involve extensive evaluations on five divergent datasets including single-hand and two-hand scenarios, hand-only and hand-object interactions, as well as RGB and depth-based estimation. DF-Mamba outperforms the latest image backbones, including VMamba and Spatial-Mamba, on all datasets and achieves state-of-the-art performance. 

---
# Emergent Bayesian Behaviour and Optimal Cue Combination in LLMs 

**Authors**: Julian Ma, Jun Wang, Zafeirios Fountas  

**Link**: [PDF](https://arxiv.org/pdf/2512.02719)  

**Abstract**: Large language models (LLMs) excel at explicit reasoning, but their implicit computational strategies remain underexplored. Decades of psychophysics research show that humans intuitively process and integrate noisy signals using near-optimal Bayesian strategies in perceptual tasks. We ask whether LLMs exhibit similar behaviour and perform optimal multimodal integration without explicit training or instruction. Adopting the psychophysics paradigm, we infer computational principles of LLMs from systematic behavioural studies. We introduce a behavioural benchmark - BayesBench: four magnitude estimation tasks (length, location, distance, and duration) over text and image, inspired by classic psychophysics, and evaluate a diverse set of nine LLMs alongside human judgments for calibration. Through controlled ablations of noise, context, and instruction prompts, we measure performance, behaviour and efficiency in multimodal cue-combination. Beyond accuracy and efficiency metrics, we introduce a Bayesian Consistency Score that detects Bayes-consistent behavioural shifts even when accuracy saturates. Our results show that while capable models often adapt in Bayes-consistent ways, accuracy does not guarantee robustness. Notably, GPT-5 Mini achieves perfect text accuracy but fails to integrate visual cues efficiently. This reveals a critical dissociation between capability and strategy, suggesting accuracy-centric benchmarks may over-index on performance while missing brittle uncertainty handling. These findings reveal emergent principled handling of uncertainty and highlight the correlation between accuracy and Bayesian tendencies. We release our psychophysics benchmark and consistency metric (this https URL) as evaluation tools and to inform future multimodal architecture designs. 

---
# Empirical Assessment of the Perception of Software Product Line Engineering by an SME before Migrating its Code Base 

**Authors**: Thomas Georges, Marianne Huchard, Mélanie König, Clémentine Nebut, Chouki Tibermacine  

**Link**: [PDF](https://arxiv.org/pdf/2512.02707)  

**Abstract**: Migrating a set of software variants into a software product line (SPL) is an expensive and potentially challenging endeavor. Indeed, SPL engineering can significantly impact a company's development process and often requires changes to established developer practices. The work presented in this paper stems from a collaboration with a Small and Medium-sized Enterprise (SME) that decided to migrate its existing code base into an SPL. In this study, we conducted an in-depth evaluation of the company's current development processes and practices, as well as the anticipated benefits and risks associated with the migration. Key stakeholders involved in software development participated in this evaluation to provide insight into their perceptions of the migration and their potential resistance to change. This paper describes the design of the interviews conducted with these stakeholders and presents an analysis of the results. Among the qualitative findings, we observed that all participants, regardless of their role in the development process, identified benefits of the migration relevant to their own activities. Furthermore, our results suggest that an effective risk mitigation strategy involves keeping stakeholders informed and engaged throughout the process, preserving as many good practices as possible, and actively involving them in the migration to ensure a smooth transition and minimize potential challenges. 

---
# An Empirical Survey of Model Merging Algorithms for Social Bias Mitigation 

**Authors**: Daiki Shirafuji, Tatsuhiko Saito, Yasutomo Kimura  

**Link**: [PDF](https://arxiv.org/pdf/2512.02689)  

**Abstract**: Large language models (LLMs) are known to inherit and even amplify societal biases present in their pre-training corpora, threatening fairness and social trust. To address this issue, recent work has explored ``editing'' LLM parameters to mitigate social bias with model merging approaches; however, there is no empirical comparison. In this work, we empirically survey seven algorithms: Linear, Karcher Mean, SLERP, NuSLERP, TIES, DELLA, and Nearswap, applying 13 open weight models in the GPT, LLaMA, and Qwen families. We perform a comprehensive evaluation using three bias datasets (BBQ, BOLD, and HONEST) and measure the impact of these techniques on LLM performance in downstream tasks of the SuperGLUE benchmark. We find a trade-off between bias reduction and downstream performance: methods achieving greater bias mitigation degrade accuracy, particularly on tasks requiring reading comprehension and commonsense and causal reasoning. Among the merging algorithms, Linear, SLERP, and Nearswap consistently reduce bias while maintaining overall performance, with SLERP at moderate interpolation weights emerging as the most balanced choice. These results highlight the potential of model merging algorithms for bias mitigation, while indicating that excessive debiasing or inappropriate merging methods may lead to the degradation of important linguistic abilities. 

---
# Beyond Single-Agent Safety: A Taxonomy of Risks in LLM-to-LLM Interactions 

**Authors**: Piercosma Bisconti, Marcello Galisai, Federico Pierucci, Marcantonio Bracale, Matteo Prandi  

**Link**: [PDF](https://arxiv.org/pdf/2512.02682)  

**Abstract**: This paper examines why safety mechanisms designed for human-model interaction do not scale to environments where large language models (LLMs) interact with each other. Most current governance practices still rely on single-agent safety containment, prompts, fine-tuning, and moderation layers that constrain individual model behavior but leave the dynamics of multi-model interaction ungoverned. These mechanisms assume a dyadic setting: one model responding to one user under stable oversight. Yet research and industrial development are rapidly shifting toward LLM-to-LLM ecosystems, where outputs are recursively reused as inputs across chains of agents. In such systems, local compliance can aggregate into collective failure even when every model is individually aligned. We propose a conceptual transition from model-level safety to system-level safety, introducing the framework of the Emergent Systemic Risk Horizon (ESRH) to formalize how instability arises from interaction structure rather than from isolated misbehavior. The paper contributes (i) a theoretical account of collective risk in interacting LLMs, (ii) a taxonomy connecting micro, meso, and macro-level failure modes, and (iii) a design proposal for InstitutionalAI, an architecture for embedding adaptive oversight within multi-agent systems. 

---
# SAND Challenge: Four Approaches for Dysartria Severity Classification 

**Authors**: Gauri Deshpande, Harish Battula, Ashish Panda, Sunil Kumar Kopparapu  

**Link**: [PDF](https://arxiv.org/pdf/2512.02669)  

**Abstract**: This paper presents a unified study of four distinct modeling approaches for classifying dysarthria severity in the Speech Analysis for Neurodegenerative Diseases (SAND) challenge. All models tackle the same five class classification task using a common dataset of speech recordings. We investigate: (1) a ViT-OF method leveraging a Vision Transformer on spectrogram images, (2) a 1D-CNN approach using eight 1-D CNN's with majority-vote fusion, (3) a BiLSTM-OF approach using nine BiLSTM models with majority vote fusion, and (4) a Hierarchical XGBoost ensemble that combines glottal and formant features through a two stage learning framework. Each method is described, and their performances on a validation set of 53 speakers are compared. Results show that while the feature-engineered XGBoost ensemble achieves the highest macro-F1 (0.86), the deep learning models (ViT, CNN, BiLSTM) attain competitive F1-scores (0.70) and offer complementary insights into the problem. 

---
# Graph VQ-Transformer (GVT): Fast and Accurate Molecular Generation via High-Fidelity Discrete Latents 

**Authors**: Haozhuo Zheng, Cheng Wang, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2512.02667)  

**Abstract**: The de novo generation of molecules with desirable properties is a critical challenge, where diffusion models are computationally intensive and autoregressive models struggle with error propagation. In this work, we introduce the Graph VQ-Transformer (GVT), a two-stage generative framework that achieves both high accuracy and efficiency. The core of our approach is a novel Graph Vector Quantized Variational Autoencoder (VQ-VAE) that compresses molecular graphs into high-fidelity discrete latent sequences. By synergistically combining a Graph Transformer with canonical Reverse Cuthill-McKee (RCM) node ordering and Rotary Positional Embeddings (RoPE), our VQ-VAE achieves near-perfect reconstruction rates. An autoregressive Transformer is then trained on these discrete latents, effectively converting graph generation into a well-structured sequence modeling problem. Crucially, this mapping of complex graphs to high-fidelity discrete sequences bridges molecular design with the powerful paradigm of large-scale sequence modeling, unlocking potential synergies with Large Language Models (LLMs). Extensive experiments show that GVT achieves state-of-the-art or highly competitive performance across major benchmarks like ZINC250k, MOSES, and GuacaMol, and notably outperforms leading diffusion models on key distribution similarity metrics such as FCD and KL Divergence. With its superior performance, efficiency, and architectural novelty, GVT not only presents a compelling alternative to diffusion models but also establishes a strong new baseline for the field, paving the way for future research in discrete latent-space molecular generation. 

---
# Distill, Forget, Repeat: A Framework for Continual Unlearning in Text-to-Image Diffusion Models 

**Authors**: Naveen George, Naoki Murata, Yuhta Takida, Konda Reddy Mopuri, Yuki Mitsufuji  

**Link**: [PDF](https://arxiv.org/pdf/2512.02657)  

**Abstract**: The recent rapid growth of visual generative models trained on vast web-scale datasets has created significant tension with data privacy regulations and copyright laws, such as GDPR's ``Right to be Forgotten.'' This necessitates machine unlearning (MU) to remove specific concepts without the prohibitive cost of retraining. However, existing MU techniques are fundamentally ill-equipped for real-world scenarios where deletion requests arrive sequentially, a setting known as continual unlearning (CUL). Naively applying one-shot methods in a continual setting triggers a stability crisis, leading to a cascade of degradation characterized by retention collapse, compounding collateral damage to related concepts, and a sharp decline in generative quality. To address this critical challenge, we introduce a novel generative distillation based continual unlearning framework that ensures targeted and stable unlearning under sequences of deletion requests. By reframing each unlearning step as a multi-objective, teacher-student distillation process, the framework leverages principles from continual learning to maintain model integrity. Experiments on a 10-step sequential benchmark demonstrate that our method unlearns forget concepts with better fidelity and achieves this without significant interference to the performance on retain concepts or the overall image quality, substantially outperforming baselines. This framework provides a viable pathway for the responsible deployment and maintenance of large-scale generative models, enabling industries to comply with ongoing data removal requests in a practical and effective manner. 

---
# Pianist Transformer: Towards Expressive Piano Performance Rendering via Scalable Self-Supervised Pre-Training 

**Authors**: Hong-Jie You, Jie-Jing Shao, Xiao-Wen Yang, Lin-Han Jia, Lan-Zhe Guo, Yu-Feng Li  

**Link**: [PDF](https://arxiv.org/pdf/2512.02652)  

**Abstract**: Existing methods for expressive music performance rendering rely on supervised learning over small labeled datasets, which limits scaling of both data volume and model size, despite the availability of vast unlabeled music, as in vision and language. To address this gap, we introduce Pianist Transformer, with four key contributions: 1) a unified Musical Instrument Digital Interface (MIDI) data representation for learning the shared principles of musical structure and expression without explicit annotation; 2) an efficient asymmetric architecture, enabling longer contexts and faster inference without sacrificing rendering quality; 3) a self-supervised pre-training pipeline with 10B tokens and 135M-parameter model, unlocking data and model scaling advantages for expressive performance rendering; 4) a state-of-the-art performance model, which achieves strong objective metrics and human-level subjective ratings. Overall, Pianist Transformer establishes a scalable path toward human-like performance synthesis in the music domain. 

---
# CryptoQA: A Large-scale Question-answering Dataset for AI-assisted Cryptography 

**Authors**: Mayar Elfares, Pascal Reisert, Tilman Dietz, Manpa Barman, Ahmed Zaki, Ralf Küsters, Andreas Bulling  

**Link**: [PDF](https://arxiv.org/pdf/2512.02625)  

**Abstract**: Large language models (LLMs) excel at many general-purpose natural language processing tasks. However, their ability to perform deep reasoning and mathematical analysis, particularly for complex tasks as required in cryptography, remains poorly understood, largely due to the lack of suitable data for evaluation and training. To address this gap, we present CryptoQA, the first large-scale question-answering (QA) dataset specifically designed for cryptography. CryptoQA contains over two million QA pairs drawn from curated academic sources, along with contextual metadata that can be used to test the cryptographic capabilities of LLMs and to train new LLMs on cryptographic tasks. We benchmark 15 state-of-the-art LLMs on CryptoQA, evaluating their factual accuracy, mathematical reasoning, consistency, referencing, backward reasoning, and robustness to adversarial samples. In addition to quantitative metrics, we provide expert reviews that qualitatively assess model outputs and establish a gold-standard baseline. Our results reveal significant performance deficits of LLMs, particularly on tasks that require formal reasoning and precise mathematical knowledge. This shows the urgent need for LLM assistants tailored to cryptography research and development. We demonstrate that, by using CryptoQA, LLMs can be fine-tuned to exhibit better performance on cryptographic tasks. 

---
# Feedback Loops and Code Perturbations in LLM-based Software Engineering: A Case Study on a C-to-Rust Translation System 

**Authors**: Martin Weiss, Jesko Hecking-Harbusch, Jochen Quante, Matthias Woehrle  

**Link**: [PDF](https://arxiv.org/pdf/2512.02567)  

**Abstract**: The advent of strong generative AI has a considerable impact on various software engineering tasks such as code repair, test generation, or language translation. While tools like GitHub Copilot are already in widespread use in interactive settings, automated approaches require a higher level of reliability before being usable in industrial practice. In this paper, we focus on three aspects that directly influence the quality of the results: a) the effect of automated feedback loops, b) the choice of Large Language Model (LLM), and c) the influence of behavior-preserving code changes.
We study the effect of these three variables on an automated C-to-Rust translation system. Code translation from C to Rust is an attractive use case in industry due to Rust's safety guarantees. The translation system is based on a generate-and-check pattern, in which Rust code generated by the LLM is automatically checked for compilability and behavioral equivalence with the original C code. For negative checking results, the LLM is re-prompted in a feedback loop to repair its output. These checks also allow us to evaluate and compare the respective success rates of the translation system when varying the three variables.
Our results show that without feedback loops LLM selection has a large effect on translation success. However, when the translation system uses feedback loops the differences across models diminish. We observe this for the average performance of the system as well as its robustness under code perturbations. Finally, we also identify that diversity provided by code perturbations can even result in improved system performance. 

---
# From Panel to Pixel: Zoom-In Vision-Language Pretraining from Biomedical Scientific Literature 

**Authors**: Kun Yuan, Min Woo Sun, Zhen Chen, Alejandro Lozano, Xiangteng He, Shi Li, Nassir Navab, Xiaoxiao Sun, Nicolas Padoy, Serena Yeung-Levy  

**Link**: [PDF](https://arxiv.org/pdf/2512.02566)  

**Abstract**: There is a growing interest in developing strong biomedical vision-language models. A popular approach to achieve robust representations is to use web-scale scientific data. However, current biomedical vision-language pretraining typically compresses rich scientific figures and text into coarse figure-level pairs, discarding the fine-grained correspondences that clinicians actually rely on when zooming into local structures. To tackle this issue, we introduce Panel2Patch, a novel data pipeline that mines hierarchical structure from existing biomedical scientific literature, i.e., multi-panel, marker-heavy figures and their surrounding text, and converts them into multi-granular supervision. Given scientific figures and captions, Panel2Patch parses layouts, panels, and visual markers, then constructs hierarchical aligned vision-language pairs at the figure, panel, and patch levels, preserving local semantics instead of treating each figure as a single data sample. Built on this hierarchical corpus, we develop a granularity-aware pretraining strategy that unifies heterogeneous objectives from coarse didactic descriptions to fine region-focused phrases. By applying Panel2Patch to only a small set of the literature figures, we extract far more effective supervision than prior pipelines, enabling substantially better performance with less pretraining data. 

---
# EZYer: A simulacrum of high school with generative agent 

**Authors**: Jinming Yang, Zimu Ji, Weiqi Luo, Gaoxi Wang, Bin Ma, Yueling Deng  

**Link**: [PDF](https://arxiv.org/pdf/2512.02561)  

**Abstract**: With the rapid development of the online education and large language model, the existing educational tools still suffer from incomplete service, insufficient performance and weak interactivity in terms of courseware generation, interactive notes and quality assurance of content. In particular, the proposed generative agent EZYer : 1) Teacher Module: Integrating the Text Corpus retrieval and in-depth generation technologies, it automatically generates structured teaching materials and LaTeX Beamer courseware in line with the high school mathematics syllabus and supports user-defined image insertion. 2) Student Module: Throughout the collaborative interaction of the four roles of Teacher, Assistant, Top Student and Struggling Student, Note Taker summarizes and generates academic notes to enhance the depth and interest of learning. 3) Controller: set up keyword filtering system, content scoring system, role co-validation system, and dynamic content correction system. This ensure academic strictness and pedagogical propriety of EZYer inputs and outputs. In order to evaluate EZYer, this paper designs five-dimensional evaluation indexes of content accuracy, knowledge coverage, usability, formatting correctness and visual design and appeal, and scores 100 Beamer and Notes generated by EZYer by five large language models, separately, and the results show that the quality of EZYer-generated content is excellent and has a good application prospect. 

---
# ADORE: Autonomous Domain-Oriented Relevance Engine for E-commerce 

**Authors**: Zheng Fang, Donghao Xie, Ming Pang, Chunyuan Yuan, Xue Jiang, Changping Peng, Zhangang Lin, Zheng Luo  

**Link**: [PDF](https://arxiv.org/pdf/2512.02555)  

**Abstract**: Relevance modeling in e-commerce search remains challenged by semantic gaps in term-matching methods (e.g., BM25) and neural models' reliance on the scarcity of domain-specific hard samples. We propose ADORE, a self-sustaining framework that synergizes three innovations: (1) A Rule-aware Relevance Discrimination module, where a Chain-of-Thought LLM generates intent-aligned training data, refined via Kahneman-Tversky Optimization (KTO) to align with user behavior; (2) An Error-type-aware Data Synthesis module that auto-generates adversarial examples to harden robustness; and (3) A Key-attribute-enhanced Knowledge Distillation module that injects domain-specific attribute hierarchies into a deployable student model. ADORE automates annotation, adversarial generation, and distillation, overcoming data scarcity while enhancing reasoning. Large-scale experiments and online A/B testing verify the effectiveness of ADORE. The framework establishes a new paradigm for resource-efficient, cognitively aligned relevance modeling in industrial applications. 

---
# CUDA-L2: Surpassing cuBLAS Performance for Matrix Multiplication through Reinforcement Learning 

**Authors**: Songqiao Su, Xiaofei Sun, Xiaoya Li, Albert Wang, Jiwei Li, Chris Shum  

**Link**: [PDF](https://arxiv.org/pdf/2512.02551)  

**Abstract**: In this paper, we propose CUDA-L2, a system that combines large language models (LLMs) and reinforcement learning (RL) to automatically optimize Half-precision General Matrix Multiply (HGEMM) CUDA kernels. Using CUDA execution speed as the RL reward, CUDA-L2 automatically optimizes HGEMM kernels across 1,000 configurations. CUDA-L2 systematically outperforms major matmul baselines to date, from the widely-used {\it this http URL} to state-of-the-art Nvidia's closed-source libraries, i.e., {\it cuBLAS}, {\it cuBLASLt}. In offline mode, where kernels are executed consecutively without time intervals, CUDA-L2 yields +22.0\% over {\it this http URL} on average; +19.2\% over {\it cuBLAS} using the optimal layout configuration (normal-normal NN and transposed-normal TN); +16.8\% over {\it cuBLASLt-heuristic}, which queries {\it cuBLASLt} library and selects the algorithm based on the heuristic's suggestion; and +11.4\% over the most competitive {\it cuBLASLt-AutoTuning} model, which selects the fastest algorithm from up to 100 candidates from {\it cuBLASLt}'s suggestions. In server mode, where kernels are executed at random intervals simulating real-time inference, the speedups further increase to +28.7\%, +26.0\%, +22.4\%, and +15.9\% for {\it this http URL}, {\it cuBLAS}, {\it cuBLASLt-heuristic}, and {\it cuBLASLt-AutoTuning} respectively. CUDA-L2 shows that even the most performance-critical, heavily-optimized kernels like HGEMM can be improved through LLM-guided RL automation by systematically exploring configuration spaces at scales impractical for humans. Project and code can be found at this http URL 

---
# Sparse Computations in Deep Learning Inference 

**Authors**: Ioanna Tasou, Panagiotis Mpakos, Angelos Vlachos, Dionysios Adamopoulos, Georgios Giannakopoulos, Konstantinos Katsikopoulos, Ioannis Karaparisis, Maria Lazou, Spyridon Loukovitis, Areti Mei, Anastasia Poulopoulou, Angeliki Dimitriou, Giorgos Filandrianos, Dimitrios Galanopoulos, Vasileios Karampinis, Ilias Mitsouras, Nikolaos Spanos, Petros Anastasiadis, Ioannis Doudalis, Konstantinos Nikas, George Retsinas, Paraskevi Tzouveli, Christina Giannoula, Nectarios Koziris, Nikela Papadopoulou, Giorgos Stamou, Athanasios Voulodimos, Georgios Goumas  

**Link**: [PDF](https://arxiv.org/pdf/2512.02550)  

**Abstract**: The computational demands of modern Deep Neural Networks (DNNs) are immense and constantly growing. While training costs usually capture public attention, inference demands are also contributing in significant computational, energy and environmental footprints. Sparsity stands out as a critical mechanism for drastically reducing these resource demands. However, its potential remains largely untapped and is not yet fully incorporated in production AI systems. To bridge this gap, this work provides the necessary knowledge and insights for performance engineers keen to get involved in deep learning inference optimization. In particular, in this work we: a) discuss the various forms of sparsity that can be utilized in DNN inference, b) explain how the original dense computations translate to sparse kernels, c) provide an extensive bibliographic review of the state-of-the-art in the implementation of these kernels for CPUs and GPUs, d) discuss the availability of sparse datasets in support of sparsity-related research and development, e) explore the current software tools and frameworks that provide robust sparsity support, and f) present evaluation results of different implementations of the key SpMM and SDDMM kernels on CPU and GPU platforms. Ultimately, this paper aims to serve as a resource for performance engineers seeking to develop and deploy highly efficient sparse deep learning models in productions. 

---
# AskNearby: An LLM-Based Application for Neighborhood Information Retrieval and Personalized Cognitive-Map Recommendations 

**Authors**: Luyao Niu, Zhicheng Deng, Boyang Li, Nuoxian Huang, Ruiqi Liu, Wenjia Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2512.02502)  

**Abstract**: The "15-minute city" envisions neighborhoods where residents can meet daily needs via a short walk or bike ride. Realizing this vision requires not only physical proximity but also efficient and reliable access to information about nearby places, services, and events. Existing location-based systems, however, focus mainly on city-level tasks and neglect the spatial, temporal, and cognitive factors that shape localized decision-making. We conceptualize this gap as the Local Life Information Accessibility (LLIA) problem and introduce AskNearby, an AI-driven community application that unifies retrieval and recommendation within the 15-minute life circle. AskNearby integrates (i) a three-layer Retrieval-Augmented Generation (RAG) pipeline that synergizes graph-based, semantic-vector, and geographic retrieval with (ii) a cognitive-map model that encodes each user's neighborhood familiarity and preferences. Experiments on real-world community datasets demonstrate that AskNearby significantly outperforms LLM-based and map-based baselines in retrieval accuracy and recommendation quality, achieving robust performance in spatiotemporal grounding and cognitive-aware ranking. Real-world deployments further validate its effectiveness. By addressing the LLIA challenge, AskNearby empowers residents to more effectively discover local resources, plan daily activities, and engage in community life. 

---
# Masking Matters: Unlocking the Spatial Reasoning Capabilities of LLMs for 3D Scene-Language Understanding 

**Authors**: Yerim Jeon, Miso Lee, WonJun Moon, Jae-Pil Heo  

**Link**: [PDF](https://arxiv.org/pdf/2512.02487)  

**Abstract**: Recent advances in 3D scene-language understanding have leveraged Large Language Models (LLMs) for 3D reasoning by transferring their general reasoning ability to 3D multi-modal contexts. However, existing methods typically adopt standard decoders from language modeling, which rely on a causal attention mask. This design introduces two fundamental conflicts in 3D scene understanding: sequential bias among order-agnostic 3D objects and restricted object-instruction attention, hindering task-specific reasoning. To overcome these limitations, we propose 3D Spatial Language Instruction Mask (3D-SLIM), an effective masking strategy that replaces the causal mask with an adaptive attention mask tailored to the spatial structure of 3D scenes. Our 3D-SLIM introduces two key components: a Geometry-adaptive Mask that constrains attention based on spatial density rather than token order, and an Instruction-aware Mask that enables object tokens to directly access instruction context. This design allows the model to process objects based on their spatial relationships while being guided by the user's task. 3D-SLIM is simple, requires no architectural modifications, and adds no extra parameters, yet it yields substantial performance improvements across diverse 3D scene-language tasks. Extensive experiments across multiple benchmarks and LLM baselines validate its effectiveness and underscore the critical role of decoder design in 3D multi-modal reasoning. 

---
# UCAgents: Unidirectional Convergence for Visual Evidence Anchored Multi-Agent Medical Decision-Making 

**Authors**: Qianhan Feng, Zhongzhen Huang, Yakun Zhu, Xiaofan Zhang, Qi Dou  

**Link**: [PDF](https://arxiv.org/pdf/2512.02485)  

**Abstract**: Vision-Language Models (VLMs) show promise in medical diagnosis, yet suffer from reasoning detachment, where linguistically fluent explanations drift from verifiable image evidence, undermining clinical trust. Recent multi-agent frameworks simulate Multidisciplinary Team (MDT) debates to mitigate single-model bias, but open-ended discussions amplify textual noise and computational cost while failing to anchor reasoning to visual evidence, the cornerstone of medical decision-making. We propose UCAgents, a hierarchical multi-agent framework enforcing unidirectional convergence through structured evidence auditing. Inspired by clinical workflows, UCAgents forbids position changes and limits agent interactions to targeted evidence verification, suppressing rhetorical drift while amplifying visual signal extraction. In UCAgents, a one-round inquiry discussion is introduced to uncover potential risks of visual-textual misalignment. This design jointly constrains visual ambiguity and textual noise, a dual-noise bottleneck that we formalize via information theory. Extensive experiments on four medical VQA benchmarks show UCAgents achieves superior accuracy (71.3% on PathVQA, +6.0% over state-of-the-art) with 87.7% lower token cost, the evaluation results further confirm that UCAgents strikes a balance between uncovering more visual evidence and avoiding confusing textual interference. These results demonstrate that UCAgents exhibits both diagnostic reliability and computational efficiency critical for real-world clinical deployment. Code is available at this https URL. 

---
# Q-BERT4Rec: Quantized Semantic-ID Representation Learning for Multimodal Recommendation 

**Authors**: Haofeng Huang, Ling Gai  

**Link**: [PDF](https://arxiv.org/pdf/2512.02474)  

**Abstract**: Sequential recommendation plays a critical role in modern online platforms such as e-commerce, advertising, and content streaming, where accurately predicting users' next interactions is essential for personalization. Recent Transformer-based methods like BERT4Rec have shown strong modeling capability, yet they still rely on discrete item IDs that lack semantic meaning and ignore rich multimodal information (e.g., text and image). This leads to weak generalization and limited interpretability. To address these challenges, we propose Q-Bert4Rec, a multimodal sequential recommendation framework that unifies semantic representation and quantized modeling. Specifically, Q-Bert4Rec consists of three stages: (1) cross-modal semantic injection, which enriches randomly initialized ID embeddings through a dynamic transformer that fuses textual, visual, and structural features; (2) semantic quantization, which discretizes fused representations into meaningful tokens via residual vector quantization; and (3) multi-mask pretraining and fine-tuning, which leverage diverse masking strategies -- span, tail, and multi-region -- to improve sequential understanding. We validate our model on public Amazon benchmarks and demonstrate that Q-Bert4Rec significantly outperforms many strong existing methods, confirming the effectiveness of semantic tokenization for multimodal sequential recommendation. Our source code will be publicly available on GitHub after publishing. 

---
# scCluBench: Comprehensive Benchmarking of Clustering Algorithms for Single-Cell RNA Sequencing 

**Authors**: Ping Xu, Zaitian Wang, Zhirui Wang, Pengjiang Li, Jiajia Wang, Ran Zhang, Pengfei Wang, Yuanchun Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2512.02471)  

**Abstract**: Cell clustering is crucial for uncovering cellular heterogeneity in single-cell RNA sequencing (scRNA-seq) data by identifying cell types and marker genes. Despite its importance, benchmarks for scRNA-seq clustering methods remain fragmented, often lacking standardized protocols and failing to incorporate recent advances in artificial intelligence. To fill these gaps, we present scCluBench, a comprehensive benchmark of clustering algorithms for scRNA-seq data. First, scCluBench provides 36 scRNA-seq datasets collected from diverse public sources, covering multiple tissues, which are uniformly processed and standardized to ensure consistency for systematic evaluation and downstream analyses. To evaluate performance, we collect and reproduce a range of scRNA-seq clustering methods, including traditional, deep learning-based, graph-based, and biological foundation models. We comprehensively evaluate each method both quantitatively and qualitatively, using core performance metrics as well as visualization analyses. Furthermore, we construct representative downstream biological tasks, such as marker gene identification and cell type annotation, to further assess the practical utility. scCluBench then investigates the performance differences and applicability boundaries of various clustering models across diverse analytical tasks, systematically assessing their robustness and scalability in real-world scenarios. Overall, scCluBench offers a standardized and user-friendly benchmark for scRNA-seq clustering, with curated datasets, unified evaluation protocols, and transparent analyses, facilitating informed method selection and providing valuable insights into model generalizability and application scope. 

---
# TabGRU: An Enhanced Design for Urban Rainfall Intensity Estimation Using Commercial Microwave Links 

**Authors**: Xingwang Li, Mengyun Chen, Jiamou Liu, Sijie Wang, Shuanggen Jin, Jafet C. M. Andersson, Jonas Olsson, Remco, van de Beek, Hai Victor Habi, Congzheng Han  

**Link**: [PDF](https://arxiv.org/pdf/2512.02465)  

**Abstract**: In the face of accelerating global urbanization and the increasing frequency of extreme weather events, highresolution urban rainfall monitoring is crucial for building resilient smart cities. Commercial Microwave Links (CMLs) are an emerging data source with great potential for this this http URL traditional rainfall retrieval from CMLs relies on physicsbased models, these often struggle with real-world complexities like signal noise and nonlinear attenuation. To address these limitations, this paper proposes a novel hybrid deep learning architecture based on the Transformer and a Bidirectional Gated Recurrent Unit (BiGRU), which we name TabGRU. This design synergistically captures both long-term dependencies and local sequential features in the CML signal data. The model is further enhanced by a learnable positional embedding and an attention pooling mechanism to improve its dynamic feature extraction and generalization capabilities. The model was validated on a public benchmark dataset from Gothenburg, Sweden (June-September 2015). The evaluation used 12 sub-links from two rain gauges (Torp and Barl) over a test period (August 22-31) covering approximately 10 distinct rainfall events. The proposed TabGRU model demonstrated consistent advantages, outperforming deep learning baselines and achieving high coefficients of determination (R2) at both the Torp site (0.91) and the Barl site (0.96). Furthermore, compared to the physics-based approach, TabGRU maintained higher accuracy and was particularly effective in mitigating the significant overestimation problem observed in the PL model during peak rainfall events. This evaluation confirms that the TabGRU model can effectively overcome the limitations of traditional methods, providing a robust and accurate solution for CML-based urban rainfall monitoring under the tested conditions. 

---
# HouseLayout3D: A Benchmark and Training-Free Baseline for 3D Layout Estimation in the Wild 

**Authors**: Valentin Bieri, Marie-Julie Rakotosaona, Keisuke Tateno, Francis Engelmann, Leonidas Guibas  

**Link**: [PDF](https://arxiv.org/pdf/2512.02450)  

**Abstract**: Current 3D layout estimation models are primarily trained on synthetic datasets containing simple single room or single floor environments. As a consequence, they cannot natively handle large multi floor buildings and require scenes to be split into individual floors before processing, which removes global spatial context that is essential for reasoning about structures such as staircases that connect multiple levels. In this work, we introduce HouseLayout3D, a real world benchmark designed to support progress toward full building scale layout estimation, including multiple floors and architecturally intricate spaces. We also present MultiFloor3D, a simple training free baseline that leverages recent scene understanding methods and already outperforms existing 3D layout estimation models on both our benchmark and prior datasets, highlighting the need for further research in this direction. Data and code are available at: this https URL. 

---
# When Refusals Fail: Unstable Safety Mechanisms in Long-Context LLM Agents 

**Authors**: Tsimur Hadeliya, Mohammad Ali Jauhar, Nidhi Sakpal, Diogo Cruz  

**Link**: [PDF](https://arxiv.org/pdf/2512.02445)  

**Abstract**: Solving complex or long-horizon problems often requires large language models (LLMs) to use external tools and operate over a significantly longer context window. New LLMs enable longer context windows and support tool calling capabilities. Prior works have focused mainly on evaluation of LLMs on long-context prompts, leaving agentic setup relatively unexplored, both from capability and safety perspectives. Our work addresses this gap. We find that LLM agents could be sensitive to length, type, and placement of the context, exhibiting unexpected and inconsistent shifts in task performance and in refusals to execute harmful requests. Models with 1M-2M token context windows show severe degradation already at 100K tokens, with performance drops exceeding 50\% for both benign and harmful tasks. Refusal rates shift unpredictably: GPT-4.1-nano increases from $\sim$5\% to $\sim$40\% while Grok 4 Fast decreases from $\sim$80\% to $\sim$10\% at 200K tokens. Our work shows potential safety issues with agents operating on longer context and opens additional questions on the current metrics and paradigm for evaluating LLM agent safety on long multi-step tasks. In particular, our results on LLM agents reveal a notable divergence in both capability and safety performance compared to prior evaluations of LLMs on similar criteria. 

---
# Boosting Medical Vision-Language Pretraining via Momentum Self-Distillation under Limited Computing Resources 

**Authors**: Phuc Pham, Nhu Pham, Ngoc Quoc Ly  

**Link**: [PDF](https://arxiv.org/pdf/2512.02438)  

**Abstract**: In medical healthcare, obtaining detailed annotations is challenging, highlighting the need for robust Vision-Language Models (VLMs). Pretrained VLMs enable fine-tuning on small datasets or zero-shot inference, achieving performance comparable to task-specific models. Contrastive learning (CL) is a key paradigm for training VLMs but inherently requires large batch sizes for effective learning, making it computationally demanding and often limited to well-resourced institutions. Moreover, with limited data in healthcare, it is important to prioritize knowledge extraction from both data and models during training to improve performance. Therefore, we focus on leveraging the momentum method combined with distillation to simultaneously address computational efficiency and knowledge exploitation. Our contributions can be summarized as follows: (1) leveraging momentum self-distillation to enhance multimodal learning, and (2) integrating momentum mechanisms with gradient accumulation to enlarge the effective batch size without increasing resource consumption. Our method attains competitive performance with state-of-the-art (SOTA) approaches in zero-shot classification, while providing a substantial boost in the few-shot adaption, achieving over 90% AUC-ROC and improving retrieval tasks by 2-3%. Importantly, our method achieves high training efficiency with a single GPU while maintaining reasonable training time. Our approach aims to advance efficient multimodal learning by reducing resource requirements while improving performance over SOTA methods. The implementation of our method is available at this https URL . 

---
# LightHCG: a Lightweight yet powerful HSIC Disentanglement based Causal Glaucoma Detection Model framework 

**Authors**: Daeyoung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2512.02437)  

**Abstract**: As a representative optic degenerative condition, glaucoma has been a threat to millions due to its irreversibility and severe impact on human vision fields. Mainly characterized by dimmed and blurred visions, or peripheral vision loss, glaucoma is well known to occur due to damages in the optic nerve from increased intraocular pressure (IOP) or neovascularization within the retina. Traditionally, most glaucoma related works and clinical diagnosis focused on detecting these damages in the optic nerve by using patient data from perimetry tests, optic papilla inspections and tonometer-based IOP measurements. Recently, with advancements in computer vision AI models, such as VGG16 or Vision Transformers (ViT), AI-automatized glaucoma detection and optic cup segmentation based on retinal fundus images or OCT recently exhibited significant performance in aiding conventional diagnosis with high performance. However, current AI-driven glaucoma detection approaches still have significant room for improvement in terms of reliability, excessive parameter usage, possibility of spurious correlation within detection, and limitations in applications to intervention analysis or clinical simulations. Thus, this research introduced a novel causal representation driven glaucoma detection model: LightHCG, an extremely lightweight Convolutional VAE-based latent glaucoma representation model that can consider the true causality among glaucoma-related physical factors within the optic nerve region. Using HSIC-based latent space disentanglement and Graph Autoencoder based unsupervised causal representation learning, LightHCG not only exhibits higher performance in classifying glaucoma with 93~99% less weights, but also enhances the possibility of AI-driven intervention analysis, compared to existing advanced vision models such as InceptionV3, MobileNetV2 or VGG16. 

---
# WorldMM: Dynamic Multimodal Memory Agent for Long Video Reasoning 

**Authors**: Woongyeong Yeo, Kangsan Kim, Jaehong Yoon, Sung Ju Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2512.02425)  

**Abstract**: Recent advances in video large language models have demonstrated strong capabilities in understanding short clips. However, scaling them to hours- or days-long videos remains highly challenging due to limited context capacity and the loss of critical visual details during abstraction. Existing memory-augmented methods mitigate this by leveraging textual summaries of video segments, yet they heavily rely on text and fail to utilize visual evidence when reasoning over complex scenes. Moreover, retrieving from fixed temporal scales further limits their flexibility in capturing events that span variable durations. To address this, we introduce WorldMM, a novel multimodal memory agent that constructs and retrieves from multiple complementary memories, encompassing both textual and visual representations. WorldMM comprises three types of memory: episodic memory indexes factual events across multiple temporal scales, semantic memory continuously updates high-level conceptual knowledge, and visual memory preserves detailed information about scenes. During inference, an adaptive retrieval agent iteratively selects the most relevant memory source and leverages multiple temporal granularities based on the query, continuing until it determines that sufficient information has been gathered. WorldMM significantly outperforms existing baselines across five long video question-answering benchmarks, achieving an average 8.4% performance gain over previous state-of-the-art methods, showing its effectiveness on long video reasoning. 

---
# Quantum feature encoding optimization 

**Authors**: Tommaso Fioravanti, Brian Quanz, Gabriele Agliardi, Edgar Andres Ruiz Guzman, Ginés Carrascal, Jae-Eun Park  

**Link**: [PDF](https://arxiv.org/pdf/2512.02422)  

**Abstract**: Quantum Machine Learning (QML) holds the promise of enhancing machine learning modeling in terms of both complexity and accuracy. A key challenge in this domain is the encoding of input data, which plays a pivotal role in determining the performance of QML models. In this work, we tackle a largely unaddressed aspect of encoding that is unique to QML modeling -- rather than adjusting the ansatz used for encoding, we consider adjusting how data is conveyed to the ansatz. We specifically implement QML pipelines that leverage classical data manipulation (i.e., ordering, selecting, and weighting features) as a preprocessing step, and evaluate if these aspects of encoding can have a significant impact on QML model performance, and if they can be effectively optimized to improve performance. Our experimental results, applied across a wide variety of data sets, ansatz, and circuit sizes, with a representative QML approach, demonstrate that by optimizing how features are encoded in an ansatz we can substantially and consistently improve the performance of QML models, making a compelling case for integrating these techniques in future QML applications. Finally we demonstrate the practical feasibility of this approach by running it using real quantum hardware with 100 qubit circuits and successfully achieving improved QML modeling performance in this case as well. 

---
# The brain-AI convergence: Predictive and generative world models for general-purpose computation 

**Authors**: Shogo Ohmae, Keiko Ohmae  

**Link**: [PDF](https://arxiv.org/pdf/2512.02419)  

**Abstract**: Recent advances in general-purpose AI systems with attention-based transformers offer a potential window into how the neocortex and cerebellum, despite their relatively uniform circuit architectures, give rise to diverse functions and, ultimately, to human intelligence. This Perspective provides a cross-domain comparison between the brain and AI that goes beyond the traditional focus on visual processing, adopting the emerging perspecive of world-model-based computation. Here, we identify shared computational mechanisms in the attention-based neocortex and the non-attentional cerebellum: both predict future world events from past inputs and construct internal world models through prediction-error learning. These predictive world models are repurposed for seemingly distinct functions--understanding in sensory processing and generation in motor processing-- enabling the brain to achieve multi-domain capabilities and human-like adaptive intelligence. Notably, attention-based AI has independently converged on a similar learning paradigm and world-model-based computation. We conclude that these shared mechanisms in both biological and artificial systems constitute a core computational foundation for realizing diverse functions including high-level intelligence, despite their relatively uniform circuit structures. Our theoretical insights bridge neuroscience and AI, advancing our understanding of the computational essence of intelligence. 

---
# Vehicle Dynamics Embedded World Models for Autonomous Driving 

**Authors**: Huiqian Li, Wei Pan, Haodong Zhang, Jin Huang, Zhihua Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2512.02417)  

**Abstract**: World models have gained significant attention as a promising approach for autonomous driving. By emulating human-like perception and decision-making processes, these models can predict and adapt to dynamic environments. Existing methods typically map high-dimensional observations into compact latent spaces and learn optimal policies within these latent representations. However, prior work usually jointly learns ego-vehicle dynamics and environmental transition dynamics from the image input, leading to inefficiencies and a lack of robustness to variations in vehicle dynamics. To address these issues, we propose the Vehicle Dynamics embedded Dreamer (VDD) method, which decouples the modeling of ego-vehicle dynamics from environmental transition dynamics. This separation allows the world model to generalize effectively across vehicles with diverse parameters. Additionally, we introduce two strategies to further enhance the robustness of the learned policy: Policy Adjustment during Deployment (PAD) and Policy Augmentation during Training (PAT). Comprehensive experiments in simulated environments demonstrate that the proposed model significantly improves both driving performance and robustness to variations in vehicle dynamics, outperforming existing approaches. 

---
# MitUNet: Enhancing Floor Plan Recognition using a Hybrid Mix-Transformer and U-Net Architecture 

**Authors**: Dmitriy Parashchuk, Alexey Kapshitskiy, Yuriy Karyakin  

**Link**: [PDF](https://arxiv.org/pdf/2512.02413)  

**Abstract**: Automatic 3D reconstruction of indoor spaces from 2D floor plans requires high-precision semantic segmentation of structural elements, particularly walls. However, existing methods optimized for standard metrics often struggle to detect thin structural components and yield masks with irregular boundaries, lacking the geometric precision required for subsequent vectorization. To address this issue, we introduce MitUNet, a hybrid neural network architecture specifically designed for wall segmentation tasks in the context of 3D modeling. In MitUNet, we utilize a hierarchical Mix-Transformer encoder to capture global context and a U-Net decoder enhanced with scSE attention blocks for precise boundary recovery. Furthermore, we propose an optimization strategy based on the Tversky loss function to effectively balance precision and recall. By fine-tuning the hyperparameters of the loss function, we prioritize the suppression of false positive noise along wall boundaries while maintaining high sensitivity to thin structures. Our experiments on the public CubiCasa5k dataset and a proprietary regional dataset demonstrate that the proposed approach ensures the generation of structurally correct masks with high boundary accuracy, outperforming standard single-task models. MitUNet provides a robust tool for data preparation in automated 3D reconstruction pipelines. 

---
# Data Curation Through the Lens of Spectral Dynamics: Static Limits, Dynamic Acceleration, and Practical Oracles 

**Authors**: Yizhou Zhang, Lun Du  

**Link**: [PDF](https://arxiv.org/pdf/2512.02409)  

**Abstract**: Large-scale neural models are increasingly trained with data pruning, synthetic data generation, cross-model distillation, reinforcement learning from human feedback (RLHF), and difficulty-based sampling. While several of these data-centric strategies reliably improve training efficiency and downstream performance, others fail to provide meaningful gains -- most notably self-generated synthetic data, which often increases dataset volume without enhancing model capability.
We formalize data curation as reweighting the sampling distribution and map its effect onto the eigenstructure of the data-induced operator. Our first main result shows that \textbf{static pruning induces a bounded operator and therefore cannot change the spectral tail exponent}; it provides at most finite-region improvements and cannot alter asymptotic neural scaling. Our second result analyzes \textbf{time-dependent data curation}, showing that an ideal oracle capable of tracking spectral residuals and continuously re-normalizing the tail can provably accelerate learning -- although practical systems can only approximate this behavior. 

---
# WISE: Weighted Iterative Society-of-Experts for Robust Multimodal Multi-Agent Debate 

**Authors**: Anoop Cherian, River Doyle, Eyal Ben-Dov, Suhas Lohit, Kuan-Chuan Peng  

**Link**: [PDF](https://arxiv.org/pdf/2512.02405)  

**Abstract**: Recent large language models (LLMs) are trained on diverse corpora and tasks, leading them to develop complementary strengths. Multi-agent debate (MAD) has emerged as a popular way to leverage these strengths for robust reasoning, though it has mostly been applied to language-only tasks, leaving its efficacy on multimodal problems underexplored. In this paper, we study MAD for solving vision-and-language reasoning problems. Our setup enables generalizing the debate protocol with heterogeneous experts that possess single- and multi-modal capabilities. To this end, we present Weighted Iterative Society-of-Experts (WISE), a generalized and modular MAD framework that partitions the agents into Solvers, that generate solutions, and Reflectors, that verify correctness, assign weights, and provide natural language feedback. To aggregate the agents' solutions across debate rounds, while accounting for variance in their responses and the feedback weights, we present a modified Dawid-Skene algorithm for post-processing that integrates our two-stage debate model. We evaluate WISE on SMART-840, VisualPuzzles, EvoChart-QA, and a new SMART-840++ dataset with programmatically generated problem instances of controlled difficulty. Our results show that WISE consistently improves accuracy by 2-7% over the state-of-the-art MAD setups and aggregation methods across diverse multimodal tasks and LLM configurations. 

---
# Process-Centric Analysis of Agentic Software Systems 

**Authors**: Shuyang Liu, Yang Chen, Rahul Krishna, Saurabh Sinha, Jatin Ganhotra, Reyhan Jabbarvand  

**Link**: [PDF](https://arxiv.org/pdf/2512.02393)  

**Abstract**: Agentic systems are modern software systems: they consist of orchestrated modules, expose interfaces, and are deployed in software pipelines. Unlike conventional programs, their execution (i.e., trajectories) is inherently stochastic and adaptive to the problem they are solving. Evaluation of such systems is often outcome-centric, judging their performance based on success or failure at the final step. This narrow focus overlooks detailed insights about such systems, failing to explain how agents reason, plan, act, or change their strategies over time. Inspired by the structured representation of conventional software systems as graphs, we introduce Graphectory to systematically encode the temporal and semantic relations in such software systems. Graphectory facilitates the design of process-centric metrics and analyses to assess the quality of agentic workflows independent of final success.
Using Graphectory, we analyze 4000 trajectories of two dominant agentic programming workflows, namely SWE-agent and OpenHands, with a combination of four backbone Large Language Models (LLMs), attempting to resolve SWE-bench Verified issues. Our fully automated analyses reveal that: (1) agents using richer prompts or stronger LLMs exhibit more complex Graphectory, reflecting deeper exploration, broader context gathering, and more thorough validation before patch submission; (2) agents' problem-solving strategies vary with both problem difficulty and the underlying LLM -- for resolved issues, the strategies often follow coherent localization-patching-validation steps, while unresolved ones exhibit chaotic, repetitive, or backtracking behaviors; (3) even when successful, agentic programming systems often display inefficient processes, leading to unnecessarily prolonged trajectories. 

---
# Multi-Domain Enhanced Map-Free Trajectory Prediction with Selective Attention 

**Authors**: Wenyi Xiong, Jian Chen  

**Link**: [PDF](https://arxiv.org/pdf/2512.02368)  

**Abstract**: Trajectory prediction is crucial for the reliability and safety of autonomous driving systems, yet it remains a challenging task in complex interactive scenarios. Existing methods often struggle to efficiently extract valuable scene information from redundant data, thereby reducing computational efficiency and prediction accuracy, especially when dealing with intricate agent interactions. To address these challenges, we propose a novel map-free trajectory prediction algorithm that achieves trajectory prediction across the temporal, spatial, and frequency domains. Specifically, in temporal information processing, We utilize a Mixture of Experts (MoE) mechanism to adaptively select critical frequency components. Concurrently, we extract these components and integrate multi-scale temporal features. Subsequently, a selective attention module is proposed to filter out redundant information in both temporal sequences and spatial interactions. Finally, we design a multimodal decoder. Under the supervision of patch-level and point-level losses, we obtain reasonable trajectory results. Experiments on Nuscences datasets demonstrate the superiority of our algorithm, validating its effectiveness in handling complex interactive scenarios. 

---
# Tackling Tuberculosis: A Comparative Dive into Machine Learning for Tuberculosis Detection 

**Authors**: Daanish Hindustani, Sanober Hindustani, Preston Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2512.02364)  

**Abstract**: This study explores the application of machine learning models, specifically a pretrained ResNet-50 model and a general SqueezeNet model, in diagnosing tuberculosis (TB) using chest X-ray images. TB, a persistent infectious disease affecting humanity for millennia, poses challenges in diagnosis, especially in resource-limited settings. Traditional methods, such as sputum smear microscopy and culture, are inefficient, prompting the exploration of advanced technologies like deep learning and computer vision. The study utilized a dataset from Kaggle, consisting of 4,200 chest X-rays, to develop and compare the performance of the two machine learning models. Preprocessing involved data splitting, augmentation, and resizing to enhance training efficiency. Evaluation metrics, including accuracy, precision, recall, and confusion matrix, were employed to assess model performance. Results showcase that the SqueezeNet achieved a loss of 32%, accuracy of 89%, precision of 98%, recall of 80%, and an F1 score of 87%. In contrast, the ResNet-50 model exhibited a loss of 54%, accuracy of 73%, precision of 88%, recall of 52%, and an F1 score of 65%. This study emphasizes the potential of machine learning in TB detection and possible implications for early identification and treatment initiation. The possibility of integrating such models into mobile devices expands their utility in areas lacking TB detection resources. However, despite promising results, the need for continued development of faster, smaller, and more accurate TB detection models remains crucial in contributing to the global efforts in combating TB. 

---
# Memory-Augmented Knowledge Fusion with Safety-Aware Decoding for Domain-Adaptive Question Answering 

**Authors**: Lei Fu, Xiang Chen, Kaige Gao Xinyue Huang, Kejian Tong  

**Link**: [PDF](https://arxiv.org/pdf/2512.02363)  

**Abstract**: Domain-specific question answering (QA) systems for services face unique challenges in integrating heterogeneous knowledge sources while ensuring both accuracy and safety. Existing large language models often struggle with factual consistency and context alignment in sensitive domains such as healthcare policies and government welfare. In this work, we introduce Knowledge-Aware Reasoning and Memory-Augmented Adaptation (KARMA), a novel framework designed to enhance QA performance in care scenarios. KARMA incorporates a dual-encoder architecture to fuse structured and unstructured knowledge sources, a gated memory unit to dynamically regulate external knowledge integration, and a safety-aware controllable decoder that mitigates unsafe outputs using safety classification and guided generation techniques. Extensive experiments on a proprietary QA dataset demonstrate that KARMA outperforms strong baselines in both answer quality and safety. This study offers a comprehensive solution for building trustworthy and adaptive QA systems in service contexts. 

---
# VACoT: Rethinking Visual Data Augmentation with VLMs 

**Authors**: Zhengzhuo Xu, Chong Sun, SiNan Du, Chen Li, Jing Lyu, Chun Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2512.02361)  

**Abstract**: While visual data augmentation remains a cornerstone for training robust vision models, it has received limited attention in visual language models (VLMs), which predominantly rely on large-scale real data acquisition or synthetic diversity. Consequently, they may struggle with basic perception tasks that conventional models handle reliably. Given the substantial cost of pre-training and fine-tuning VLMs, continue training on augmented data yields limited and diminishing returns. In this paper, we present Visual Augmentation Chain-of-Thought (VACoT), a framework that dynamically invokes image augmentations during model inference. By incorporating post-hoc transformations such as denoising, VACoT substantially improves robustness on challenging and out-of-distribution inputs, especially in OCR-related adversarial scenarios. Distinct from prior approaches limited to local cropping, VACoT integrates a structured collection of general visual augmentations, broadening the query image views while reducing training complexity and computational overhead with efficient agentic reinforcement learning. We propose a conditional reward scheme that encourages necessary augmentation while penalizing verbose responses, ensuring concise and effective reasoning in perception tasks. We demonstrate the superiority of VACoT with extensive experiments on 13 perception benchmarks and further introduce AdvOCR to highlight the generalization benefits of post-hoc visual augmentations in adversarial scenarios. 

---
# Understanding and Harnessing Sparsity in Unified Multimodal Models 

**Authors**: Shwai He, Chaorui Deng, Ang Li, Shen Yan  

**Link**: [PDF](https://arxiv.org/pdf/2512.02351)  

**Abstract**: Large multimodal models have achieved remarkable progress in both understanding and generation. Recent efforts pursue unified multimodal models that integrate heterogeneous components to support both capabilities within a single framework. However, such unification introduces inference inefficiencies, e.g., specific tasks or samples may not require the full knowledge or capacity of the unified model. Yet, a systematic understanding of how these inefficiencies manifest across different components remains limited. In this work, we first conduct a systematic analysis of unified multimodal model components using training-free pruning as a probing methodology, considering both depth pruning and width reduction. Our study reveals that the understanding component exhibits notable compressibility in both understanding and generation tasks, which is more pronounced in the latter. In contrast, the generation components are highly sensitive to compression, with performance deteriorating sharply even under moderate compression ratios. To address this limitation, we propose the Mixture-of-Experts (MoE) Adaptation, inspired by the dynamic activation patterns observed across different samples. This approach partitions the generation module into multiple experts and enables sparse activation to restore generation quality. We validate the effectiveness of sparse activation through expert-frozen tuning and further demonstrate that a fully trainable adaptation delivers additional gains. As a result, the adapted BAGEL model achieves performance comparable to the full model while activating only about half of its parameters. The code is released at \href{this https URL}{this link}. 

---
# FOVA: Offline Federated Reinforcement Learning with Mixed-Quality Data 

**Authors**: Nan Qiao, Sheng Yue, Ju Ren, Yaoxue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2512.02350)  

**Abstract**: Offline Federated Reinforcement Learning (FRL), a marriage of federated learning and offline reinforcement learning, has attracted increasing interest recently. Albeit with some advancement, we find that the performance of most existing offline FRL methods drops dramatically when provided with mixed-quality data, that is, the logging behaviors (offline data) are collected by policies with varying qualities across clients. To overcome this limitation, this paper introduces a new vote-based offline FRL framework, named FOVA. It exploits a \emph{vote mechanism} to identify high-return actions during local policy evaluation, alleviating the negative effect of low-quality behaviors from diverse local learning policies. Besides, building on advantage-weighted regression (AWR), we construct consistent local and global training objectives, significantly enhancing the efficiency and stability of FOVA. Further, we conduct an extensive theoretical analysis and rigorously show that the policy learned by FOVA enjoys strict policy improvement over the behavioral policy. Extensive experiments corroborate the significant performance gains of our proposed algorithm over existing baselines on widely used benchmarks. 

---
# Video Diffusion Models Excel at Tracking Similar-Looking Objects Without Supervision 

**Authors**: Chenshuang Zhang, Kang Zhang, Joon Son Chung, In So Kweon, Junmo Kim, Chengzhi Mao  

**Link**: [PDF](https://arxiv.org/pdf/2512.02339)  

**Abstract**: Distinguishing visually similar objects by their motion remains a critical challenge in computer vision. Although supervised trackers show promise, contemporary self-supervised trackers struggle when visual cues become ambiguous, limiting their scalability and generalization without extensive labeled data. We find that pre-trained video diffusion models inherently learn motion representations suitable for tracking without task-specific training. This ability arises because their denoising process isolates motion in early, high-noise stages, distinct from later appearance refinement. Capitalizing on this discovery, our self-supervised tracker significantly improves performance in distinguishing visually similar objects, an underexplored failure point for existing methods. Our method achieves up to a 6-point improvement over recent self-supervised approaches on established benchmarks and our newly introduced tests focused on tracking visually similar items. Visualizations confirm that these diffusion-derived motion representations enable robust tracking of even identical objects across challenging viewpoint changes and deformations. 

---
# COGNITION: From Evaluation to Defense against Multimodal LLM CAPTCHA Solvers 

**Authors**: Junyu Wang, Changjia Zhu, Yuanbo Zhou, Lingyao Li, Xu He, Junjie Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2512.02318)  

**Abstract**: This paper studies how multimodal large language models (MLLMs) undermine the security guarantees of visual CAPTCHA. We identify the attack surface where an adversary can cheaply automate CAPTCHA solving using off-the-shelf models. We evaluate 7 leading commercial and open-source MLLMs across 18 real-world CAPTCHA task types, measuring single-shot accuracy, success under limited retries, end-to-end latency, and per-solve cost. We further analyze the impact of task-specific prompt engineering and few-shot demonstrations on solver effectiveness. We reveal that MLLMs can reliably solve recognition-oriented and low-interaction CAPTCHA tasks at human-like cost and latency, whereas tasks requiring fine-grained localization, multi-step spatial reasoning, or cross-frame consistency remain significantly harder for current models. By examining the reasoning traces of such MLLMs, we investigate the underlying mechanisms of why models succeed/fail on specific CAPTCHA puzzles and use these insights to derive defense-oriented guidelines for selecting and strengthening CAPTCHA tasks. We conclude by discussing implications for platform operators deploying CAPTCHA as part of their abuse-mitigation this http URL Availability (this https URL). 

---
# HealthContradict: Evaluating Biomedical Knowledge Conflicts in Language Models 

**Authors**: Boya Zhang, Alban Bornet, Rui Yang, Nan Liu, Douglas Teodoro  

**Link**: [PDF](https://arxiv.org/pdf/2512.02299)  

**Abstract**: How do language models use contextual information to answer health questions? How are their responses impacted by conflicting contexts? We assess the ability of language models to reason over long, conflicting biomedical contexts using HealthContradict, an expert-verified dataset comprising 920 unique instances, each consisting of a health-related question, a factual answer supported by scientific evidence, and two documents presenting contradictory stances. We consider several prompt settings, including correct, incorrect or contradictory context, and measure their impact on model outputs. Compared to existing medical question-answering evaluation benchmarks, HealthContradict provides greater distinctions of language models' contextual reasoning capabilities. Our experiments show that the strength of fine-tuned biomedical language models lies not only in their parametric knowledge from pretraining, but also in their ability to exploit correct context while resisting incorrect context. 

---
# Enhancing Cross Domain SAR Oil Spill Segmentation via Morphological Region Perturbation and Synthetic Label-to-SAR Generation 

**Authors**: Andre Juarez, Luis Salsavilca, Frida Coaquira, Celso Gonzales  

**Link**: [PDF](https://arxiv.org/pdf/2512.02290)  

**Abstract**: Deep learning models for SAR oil spill segmentation often fail to generalize across regions due to differences in sea-state, backscatter statistics, and slick morphology, a limitation that is particularly severe along the Peruvian coast where labeled Sentinel-1 data remain scarce. To address this problem, we propose \textbf{MORP--Synth}, a two-stage synthetic augmentation framework designed to improve transfer from Mediterranean to Peruvian conditions. Stage~A applies Morphological Region Perturbation, a curvature guided label space method that generates realistic geometric variations of oil and look-alike regions. Stage~B renders SAR-like textures from the edited masks using a conditional generative INADE model. We compile a Peruvian dataset of 2112 labeled 512$\times$512 patches from 40 Sentinel-1 scenes (2014--2024), harmonized with the Mediterranean CleanSeaNet benchmark, and evaluate seven segmentation architectures. Models pretrained on Mediterranean data degrade from 67.8\% to 51.8\% mIoU on the Peruvian domain; MORP--Synth improves performance up to +6 mIoU and boosts minority-class IoU (+10.8 oil, +14.6 look-alike). 

---
# Progressive Image Restoration via Text-Conditioned Video Generation 

**Authors**: Peng Kang, Xijun Wang, Yu Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2512.02273)  

**Abstract**: Recent text-to-video models have demonstrated strong temporal generation capabilities, yet their potential for image restoration remains underexplored. In this work, we repurpose CogVideo for progressive visual restoration tasks by fine-tuning it to generate restoration trajectories rather than natural video motion. Specifically, we construct synthetic datasets for super-resolution, deblurring, and low-light enhancement, where each sample depicts a gradual transition from degraded to clean frames. Two prompting strategies are compared: a uniform text prompt shared across all samples, and a scene-specific prompting scheme generated via LLaVA multi-modal LLM and refined with ChatGPT. Our fine-tuned model learns to associate temporal progression with restoration quality, producing sequences that improve perceptual metrics such as PSNR, SSIM, and LPIPS across frames. Extensive experiments show that CogVideo effectively restores spatial detail and illumination consistency while maintaining temporal coherence. Moreover, the model generalizes to real-world scenarios on the ReLoBlur dataset without additional training, demonstrating strong zero-shot robustness and interpretability through temporal restoration. 

---
# Spatiotemporal Pyramid Flow Matching for Climate Emulation 

**Authors**: Jeremy Andrew Irvin, Jiaqi Han, Zikui Wang, Abdulaziz Alharbi, Yufei Zhao, Nomin-Erdene Bayarsaikhan, Daniele Visioni, Andrew Y. Ng, Duncan Watson-Parris  

**Link**: [PDF](https://arxiv.org/pdf/2512.02268)  

**Abstract**: Generative models have the potential to transform the way we emulate Earth's changing climate. Previous generative approaches rely on weather-scale autoregression for climate emulation, but this is inherently slow for long climate horizons and has yet to demonstrate stable rollouts under nonstationary forcings. Here, we introduce Spatiotemporal Pyramid Flows (SPF), a new class of flow matching approaches that model data hierarchically across spatial and temporal scales. Inspired by cascaded video models, SPF partitions the generative trajectory into a spatiotemporal pyramid, progressively increasing spatial resolution to reduce computation and coupling each stage with an associated timescale to enable direct sampling at any temporal level in the pyramid. This design, together with conditioning each stage on prescribed physical forcings (e.g., greenhouse gases or aerosols), enables efficient, parallel climate emulation at multiple timescales. On ClimateBench, SPF outperforms strong flow matching baselines and pre-trained models at yearly and monthly timescales while offering fast sampling, especially at coarser temporal levels. To scale SPF, we curate ClimateSuite, the largest collection of Earth system simulations to date, comprising over 33,000 simulation-years across ten climate models and the first dataset to include simulations of climate interventions. We find that the scaled SPF model demonstrates good generalization to held-out scenarios across climate models. Together, SPF and ClimateSuite provide a foundation for accurate, efficient, probabilistic climate emulation across temporal scales and realistic future scenarios. Data and code is publicly available at this https URL . 

---
# DETAIL Matters: Measuring the Impact of Prompt Specificity on Reasoning in Large Language Models 

**Authors**: Olivia Kim  

**Link**: [PDF](https://arxiv.org/pdf/2512.02246)  

**Abstract**: Prompt design plays a critical role in the reasoning performance of large language models (LLMs), yet the impact of prompt specificity - how detailed or vague a prompt is - remains understudied. This paper introduces DETAIL, a framework for evaluating LLM performance across varying levels of prompt specificity. We generate multi-level prompts using GPT-4, quantify specificity via perplexity, and assess correctness using GPT-based semantic equivalence. Experiments on 30 novel reasoning tasks across GPT-4 and O3-mini reveal that specificity improves accuracy, especially for smaller models and procedural tasks. Our results highlight the need for adaptive prompting strategies and provide tools and data to support further research. 

---
# See, Hear, and Understand: Benchmarking Audiovisual Human Speech Understanding in Multimodal Large Language Models 

**Authors**: Le Thien Phuc Nguyen, Zhuoran Yu, Samuel Low Yu Hang, Subin An, Jeongik Lee, Yohan Ban, SeungEun Chung, Thanh-Huy Nguyen, JuWan Maeng, Soochahn Lee, Yong Jae Lee  

**Link**: [PDF](https://arxiv.org/pdf/2512.02231)  

**Abstract**: Multimodal large language models (MLLMs) are expected to jointly interpret vision, audio, and language, yet existing video benchmarks rarely assess fine-grained reasoning about human speech. Many tasks remain visually solvable or only coarsely evaluate speech, offering limited insight into whether models can align who speaks, what is said, and when it occurs. We introduce AV-SpeakerBench, a curated benchmark of 3,212 multiple-choice questions focused on speaker-centric audiovisual reasoning in real-world videos. It features: (1) a speaker-centered formulation that treats speakers-not scenes-as the core reasoning unit; (2) fusion-grounded question design embedding audiovisual dependencies into question semantics; and (3) expert-curated annotations ensuring temporal precision and cross-modal validity. Comprehensive evaluations show that the Gemini family consistently outperforms open-source systems, with Gemini 2.5 Pro achieving the best results. Among open models, Qwen3-Omni-30B approaches Gemini 2.0 Flash but remains far behind Gemini 2.5 Pro, primarily due to weaker audiovisual fusion rather than visual perception. We believe AV-SpeakerBench establishes a rigorous foundation for advancing fine-grained audiovisual reasoning in future multimodal systems. 

---
# Orchestration Framework for Financial Agents: From Algorithmic Trading to Agentic Trading 

**Authors**: Jifeng Li, Arnav Grover, Abraham Alpuerto, Yupeng Cao, Xiao-Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2512.02227)  

**Abstract**: The financial market is a mission-critical playground for AI agents due to its temporal dynamics and low signal-to-noise ratio. Building an effective algorithmic trading system may require a professional team to develop and test over the years. In this paper, we propose an orchestration framework for financial agents, which aims to democratize financial intelligence to the general public. We map each component of the traditional algorithmic trading system to agents, including planner, orchestrator, alpha agents, risk agents, portfolio agents, backtest agents, execution agents, audit agents, and memory agent. We present two in-house trading examples. For the stock trading task (hourly data from 04/2024 to 12/2024), our approach achieved a return of $20.42\%$, a Sharpe ratio of 2.63, and a maximum drawdown of $-3.59\%$, while the S&P 500 index yielded a return of $15.97\%$. For the BTC trading task (minute data from 27/07/2025 to 13/08/2025), our approach achieved a return of $8.39\%$, a Sharpe ratio of $0.38$, and a maximum drawdown of $-2.80\%$, whereas the BTC price increased by $3.80\%$. Our code is available on \href{this https URL}{GitHub}. 

---
# Improved Training Mechanism for Reinforcement Learning via Online Model Selection 

**Authors**: Aida Afshar, Aldo Pacchiano  

**Link**: [PDF](https://arxiv.org/pdf/2512.02214)  

**Abstract**: We study the problem of online model selection in reinforcement learning, where the selector has access to a class of reinforcement learning agents and learns to adaptively select the agent with the right configuration. Our goal is to establish the improved efficiency and performance gains achieved by integrating online model selection methods into reinforcement learning training procedures. We examine the theoretical characterizations that are effective for identifying the right configuration in practice, and address three practical criteria from a theoretical perspective: 1) Efficient resource allocation, 2) Adaptation under non-stationary dynamics, and 3) Training stability across different seeds. Our theoretical results are accompanied by empirical evidence from various model selection tasks in reinforcement learning, including neural architecture selection, step-size selection, and self model selection. 

---
# Multifractal Recalibration of Neural Networks for Medical Imaging Segmentation 

**Authors**: Miguel L. Martins, Miguel T. Coimbra, Francesco Renna  

**Link**: [PDF](https://arxiv.org/pdf/2512.02198)  

**Abstract**: Multifractal analysis has revealed regularities in many self-seeding phenomena, yet its use in modern deep learning remains limited. Existing end-to-end multifractal methods rely on heavy pooling or strong feature-space decimation, which constrain tasks such as semantic segmentation. Motivated by these limitations, we introduce two inductive priors: Monofractal and Multifractal Recalibration. These methods leverage relationships between the probability mass of the exponents and the multifractal spectrum to form statistical descriptions of encoder embeddings, implemented as channel-attention functions in convolutional networks.
Using a U-Net-based framework, we show that multifractal recalibration yields substantial gains over a baseline equipped with other channel-attention mechanisms that also use higher-order statistics. Given the proven ability of multifractal analysis to capture pathological regularities, we validate our approach on three public medical-imaging datasets: ISIC18 (dermoscopy), Kvasir-SEG (endoscopy), and BUSI (ultrasound).
Our empirical analysis also provides insights into the behavior of these attention layers. We find that excitation responses do not become increasingly specialized with encoder depth in U-Net architectures due to skip connections, and that their effectiveness may relate to global statistics of instance variability. 

---
# Bin2Vec: Interpretable and Auditable Multi-View Binary Analysis for Code Plagiarism Detection 

**Authors**: Moussa Moussaoui, Tarik Houichime, Abdelalim Sadiq  

**Link**: [PDF](https://arxiv.org/pdf/2512.02197)  

**Abstract**: We introduce Bin2Vec, a new framework that helps compare software programs in a clear and explainable way. Instead of focusing only on one type of information, Bin2Vec combines what a program looks like (its built-in functions, imports, and exports) with how it behaves when it runs (its instructions and memory usage). This gives a more complete picture when deciding whether two programs are similar or not. Bin2Vec represents these different types of information as views that can be inspected separately using easy-to-read charts, and then brings them together into an overall similarity score. Bin2Vec acts as a bridge between binary representations and machine learning techniques by generating feature representations that can be efficiently processed by machine-learning models. We tested Bin2Vec on multiple versions of two well-known Windows programs, PuTTY and 7-Zip. The primary results strongly confirmed that our method compute an optimal and visualization-friendly representation of the analyzed software. For example, PuTTY versions showed more complex behavior and memory activity, while 7-Zip versions focused more on performance-related patterns. Overall, Bin2Vec provides decisions that are both reliable and explainable to humans. Because it is modular and easy to extend, it can be applied to tasks like auditing, verifying software origins, or quickly screening large numbers of programs in cybersecurity and reverse-engineering work. 

---
# A Knowledge-Based Language Model: Deducing Grammatical Knowledge in a Multi-Agent Language Acquisition Simulation 

**Authors**: David Ph. Shakouri, Crit Cremers, Niels O. Schiller  

**Link**: [PDF](https://arxiv.org/pdf/2512.02195)  

**Abstract**: This paper presents an initial study performed by the MODOMA system. The MODOMA is a computational multi-agent laboratory environment for unsupervised language acquisition experiments such that acquisition is based on the interaction between two language models, an adult and a child agent. Although this framework employs statistical as well as rule-based procedures, the result of language acquisition is a knowledge-based language model, which can be used to generate and parse new utterances of the target language. This system is fully parametrized and researchers can control all aspects of the experiments while the results of language acquisition, that is, the acquired grammatical knowledge, are explicitly represented and can be consulted. Thus, this system introduces novel possibilities for conducting computational language acquisition experiments. The experiments presented by this paper demonstrate that functional and content categories can be acquired and represented by the daughter agent based on training and test data containing different amounts of exemplars generated by the adult agent. Interestingly, similar patterns, which are well-established for human-generated data, are also found for these machine-generated data. As the procedures resulted in the successful acquisition of discrete grammatical categories by the child agent, these experiments substantiate the validity of the MODOMA approach to modelling language acquisition. 

---
# Enforcing Orderedness to Improve Feature Consistency 

**Authors**: Sophie L. Wang, Alex Quach, Nithin Parsan, John J. Yang  

**Link**: [PDF](https://arxiv.org/pdf/2512.02194)  

**Abstract**: Sparse autoencoders (SAEs) have been widely used for interpretability of neural networks, but their learned features often vary across seeds and hyperparameter settings. We introduce Ordered Sparse Autoencoders (OSAE), which extend Matryoshka SAEs by (1) establishing a strict ordering of latent features and (2) deterministically using every feature dimension, avoiding the sampling-based approximations of prior nested SAE methods. Theoretically, we show that OSAEs resolve permutation non-identifiability in settings of sparse dictionary learning where solutions are unique (up to natural symmetries). Empirically on Gemma2-2B and Pythia-70M, we show that OSAEs can help improve consistency compared to Matryoshka baselines. 

---
# Story2MIDI: Emotionally Aligned Music Generation from Text 

**Authors**: Mohammad Shokri, Alexandra C. Salem, Gabriel Levine, Johanna Devaney, Sarah Ita Levitan  

**Link**: [PDF](https://arxiv.org/pdf/2512.02192)  

**Abstract**: In this paper, we introduce Story2MIDI, a sequence-to-sequence Transformer-based model for generating emotion-aligned music from a given piece of text. To develop this model, we construct the Story2MIDI dataset by merging existing datasets for sentiment analysis from text and emotion classification in music. The resulting dataset contains pairs of text blurbs and music pieces that evoke the same emotions in the reader or listener. Despite the small scale of our dataset and limited computational resources, our results indicate that our model effectively learns emotion-relevant features in music and incorporates them into its generation process, producing samples with diverse emotional responses. We evaluate the generated outputs using objective musical metrics and a human listening study, confirming the model's ability to capture intended emotional cues. 

---
# Think Before You Prune: Self-Reflective Structured Pruning for Reasoning Language Models 

**Authors**: Ziyan Wang, Enmao Diao, Qi Le, Pu Wang, Guanchu Wang, Minwoo Lee, Shu-ping Yeh, Li Yang  

**Link**: [PDF](https://arxiv.org/pdf/2512.02185)  

**Abstract**: Reasoning LLMs (RLMs) such as OpenAI o1, DeepSeek-R1, and Qwen3 deliver strong multi-step reasoning through chain-of-thought generation, but their large model sizes and lengthy decode-time outputs make them costly to deploy and unsuitable for resource-constrained settings. To reduce computing and memory cost, pruning offers a promising solution by removing unimportant parameters. However, despite their success on standard LLMs, existing pruning methods severely damage RLMs, as even moderate sparsity (e.g., 20%) can collapse accuracy and completely disrupt the model's reasoning coherence. We begin by analyzing why existing pruning pipelines fail on reasoning LLMs and find that their brittleness largely stems from a mismatch between the calibration data, the pruning objective, and the model's decode-time reasoning behavior. Our study further shows that the most reliable calibration signal comes not from human-written labels but from the model's own self-generated reasoning traces, which more accurately reflect its inference distribution. Guided by these insights, we introduce RESP, a self-reflective structured pruning framework that aligns pruning decisions with the model's reasoning dynamics through self-generated calibration, decode-only gradient-based importance estimation, and progressive regeneration that maintains calibration fidelity as sparsity increases. Experiments on Qwen3-8B demonstrate that RESP markedly outperforms existing structured pruning methods on both GSM8K and MathQA, preserving near-dense accuracy at 20-30% sparsity and substantially mitigating performance collapse at higher sparsity levels. At 40% sparsity, RESP attains 81.3% accuracy on GSM8K and 59.6% on MathQA, surpassing the strongest baselines by 66.87% and 47%, respectively. 

---
# CLEF: Clinically-Guided Contrastive Learning for Electrocardiogram Foundation Models 

**Authors**: Yuxuan Shu, Peter H. Charlton, Fahim Kawsar, Jussi Hernesniemi, Mohammad Malekzadeh  

**Link**: [PDF](https://arxiv.org/pdf/2512.02180)  

**Abstract**: The electrocardiogram (ECG) is a key diagnostic tool in cardiovascular health. Single-lead ECG recording is integrated into both clinical-grade and consumer wearables. While self-supervised pretraining of foundation models on unlabeled ECGs improves diagnostic performance, existing approaches do not incorporate domain knowledge from clinical metadata. We introduce a novel contrastive learning approach that utilizes an established clinical risk score to adaptively weight negative pairs: clinically-guided contrastive learning. It aligns the similarities of ECG embeddings with clinically meaningful differences between subjects, with an explicit mechanism to handle missing metadata. On 12-lead ECGs from 161K patients in the MIMIC-IV dataset, we pretrain single-lead ECG foundation models at three scales, collectively called CLEF, using only routinely collected metadata without requiring per-sample ECG annotations. We evaluate CLEF on 18 clinical classification and regression tasks across 7 held-out datasets, and benchmark against 5 foundation model baselines and 3 self-supervised algorithms. When pretrained on 12-lead ECG data and tested on lead-I data, CLEF outperforms self-supervised foundation model baselines: the medium-sized CLEF achieves average AUROC improvements of at least 2.6% in classification and average reductions in MAEs of at least 3.2% in regression. Comparing with existing self-supervised learning algorithms, CLEF improves the average AUROC by at least 1.8%. Moreover, when pretrained only on lead-I data for classification tasks, CLEF performs comparably to the state-of-the-art ECGFounder, which was trained in a supervised manner. Overall, CLEF enables more accurate and scalable single-lead ECG analysis, advancing remote health monitoring. Code and pretrained CLEF models are available at: this http URL. 

---
# Young Children's Anthropomorphism of AI Chatbots and the Role of Parent Co-Presence 

**Authors**: Pilyoung Kim, Jenna H. Chin, Yun Xie, Nolan Brady, Tom Yeh, Sujin Yang  

**Link**: [PDF](https://arxiv.org/pdf/2512.02179)  

**Abstract**: Artificial Intelligence (AI) chatbots powered by a large language model (LLM) are entering young children's learning and play, yet little is known about how young children construe these agents or how such construals relate to engagement. We examined anthropomorphism of a social AI chatbot during collaborative storytelling and asked how children's attributions related to their behavior and prefrontal activation. Children at ages 5-6 (N = 23) completed three storytelling sessions: interacting with (1) an AI chatbot only, (2) a parent only, and (3) the AI and a parent together. After the sessions, children completed an interview assessing anthropomorphism toward both the AI chatbot and the parent. Behavioral engagement was indexed by the conversational turn count (CTC) ratio, and concurrent fNIRS measured oxygenated hemoglobin in bilateral vmPFC and dmPFC regions. Children reported higher anthropomorphism for parents than for the AI chatbot overall, although AI ratings were relatively high for perceptive abilities and epistemic states. Anthropomorphism was not associated with CTC. In the right dmPFC, higher perceptive scores were associated with greater activation during the AI-only condition and with lower activation during the AI+Parent condition. Exploratory analyses indicated that higher dmPFC activation during the AI-only condition correlated with higher end-of-session "scared" mood ratings. Findings suggest that stronger perceptive anthropomorphism can be associated with greater brain activation related to interpreting the AI's mental states, whereas parent co-presence may help some children interpret and regulate novel AI interactions. These results may have design implications for encouraging parent-AI co-use in early childhood. 

---
# Feature Selection Empowered BERT for Detection of Hate Speech with Vocabulary Augmentation 

**Authors**: Pritish N. Desai, Tanay Kewalramani, Srimanta Mandal  

**Link**: [PDF](https://arxiv.org/pdf/2512.02141)  

**Abstract**: Abusive speech on social media poses a persistent and evolving challenge, driven by the continuous emergence of novel slang and obfuscated terms designed to circumvent detection systems. In this work, we present a data efficient strategy for fine tuning BERT on hate speech classification by significantly reducing training set size without compromising performance. Our approach employs a TF IDF-based sample selection mechanism to retain only the most informative 75 percent of examples, thereby minimizing training overhead. To address the limitations of BERT's native vocabulary in capturing evolving hate speech terminology, we augment the tokenizer with domain-specific slang and lexical variants commonly found in abusive contexts. Experimental results on a widely used hate speech dataset demonstrate that our method achieves competitive performance while improving computational efficiency, highlighting its potential for scalable and adaptive abusive content moderation. 

---
# Comparing Baseline and Day-1 Diffusion MRI Using Multimodal Deep Embeddings for Stroke Outcome Prediction 

**Authors**: Sina Raeisadigh, Myles Joshua Toledo Tan, Henning Müller, Abderrahmane Hedjoudje  

**Link**: [PDF](https://arxiv.org/pdf/2512.02088)  

**Abstract**: This study compares baseline (J0) and 24-hour (J1) diffusion magnetic resonance imaging (MRI) for predicting three-month functional outcomes after acute ischemic stroke (AIS). Seventy-four AIS patients with paired apparent diffusion coefficient (ADC) scans and clinical data were analyzed. Three-dimensional ResNet-50 embeddings were fused with structured clinical variables, reduced via principal component analysis (<=12 components), and classified using linear support vector machines with eight-fold stratified group cross-validation. J1 multimodal models achieved the highest predictive performance (AUC = 0.923 +/- 0.085), outperforming J0-based configurations (AUC <= 0.86). Incorporating lesion-volume features further improved model stability and interpretability. These findings demonstrate that early post-treatment diffusion MRI provides superior prognostic value to pre-treatment imaging and that combining MRI, clinical, and lesion-volume features produces a robust and interpretable framework for predicting three-month functional outcomes in AIS patients. 

---
# FDRMFL:Multi-modal Federated Feature Extraction Model Based on Information Maximization and Contrastive Learning 

**Authors**: Haozhe Wu  

**Link**: [PDF](https://arxiv.org/pdf/2512.02076)  

**Abstract**: This study focuses on the feature extraction problem in multi-modal data regression. To address three core challenges in real-world scenarios: limited and non-IID data, effective extraction and fusion of multi-modal information, and susceptibility to catastrophic forgetting in model learning, a task-driven supervised multi-modal federated feature extraction method is proposed. The method integrates multi-modal information extraction and contrastive learning mechanisms, and can adapt to different neural network structures as the latent mapping functions for data of each modality. It supports each client to independently learn low-dimensional representations of multi-modal data, and can flexibly control the degree of retention of effective information about the response variable in the predictive variables within the low-dimensional features through parameter tuning. The multi-constraint learning framework constructed by the method guarantees regression accuracy using Mean Squared Error loss. Through the synergistic effect of mutual information preservation constraint, symmetric Kullback-Leibler divergence constraint, and inter-model contrastive constraint, it achieves the retention of task-related information, the extraction, fusion, and alignment of multi-modal features, and the mitigation of representation drift and catastrophic forgetting in non-IID scenarios, respectively. This ensures that the feature extraction process always centers on improving the performance of downstream regression tasks. Experimental results from simulations and real-world data analysis demonstrate that the proposed method achieves more significant performance improvement on downstream regression tasks compared with classical feature extraction techniques. 

---
# HTG-GCL: Leveraging Hierarchical Topological Granularity from Cellular Complexes for Graph Contrastive Learning 

**Authors**: Qirui Ji, Bin Qin, Yifan Jin, Yunze Zhao, Chuxiong Sun, Changwen Zheng, Jianwen Cao, Jiangmeng Li  

**Link**: [PDF](https://arxiv.org/pdf/2512.02073)  

**Abstract**: Graph contrastive learning (GCL) aims to learn discriminative semantic invariance by contrasting different views of the same graph that share critical topological patterns. However, existing GCL approaches with structural augmentations often struggle to identify task-relevant topological structures, let alone adapt to the varying coarse-to-fine topological granularities required across different downstream tasks. To remedy this issue, we introduce Hierarchical Topological Granularity Graph Contrastive Learning (HTG-GCL), a novel framework that leverages transformations of the same graph to generate multi-scale ring-based cellular complexes, embodying the concept of topological granularity, thereby generating diverse topological views. Recognizing that a certain granularity may contain misleading semantics, we propose a multi-granularity decoupled contrast and apply a granularity-specific weighting mechanism based on uncertainty estimation. Comprehensive experiments on various benchmarks demonstrate the effectiveness of HTG-GCL, highlighting its superior performance in capturing meaningful graph representations through hierarchical topological information. 

---
# DPWMixer: Dual-Path Wavelet Mixer for Long-Term Time Series Forecasting 

**Authors**: Li Qianyang, Zhang Xingjun, Wang Shaoxun, Wei Jia  

**Link**: [PDF](https://arxiv.org/pdf/2512.02070)  

**Abstract**: Long-term time series forecasting (LTSF) is a critical task in computational intelligence. While Transformer-based models effectively capture long-range dependencies, they often suffer from quadratic complexity and overfitting due to data sparsity. Conversely, efficient linear models struggle to depict complex non-linear local dynamics. Furthermore, existing multi-scale frameworks typically rely on average pooling, which acts as a non-ideal low-pass filter, leading to spectral aliasing and the irreversible loss of high-frequency transients. In response, this paper proposes DPWMixer, a computationally efficient Dual-Path architecture. The framework is built upon a Lossless Haar Wavelet Pyramid that replaces traditional pooling, utilizing orthogonal decomposition to explicitly disentangle trends and local fluctuations without information loss. To process these components, we design a Dual-Path Trend Mixer that integrates a global linear mapping for macro-trend anchoring and a flexible patch-based MLP-Mixer for micro-dynamic evolution. Finally, An adaptive multi-scale fusion module then integrates predictions from diverse scales, weighted by channel stationarity to optimize synthesis. Extensive experiments on eight public benchmarks demonstrate that our method achieves a consistent improvement over state-of-the-art baselines. The code is available at this https URL. 

---
# Large Language Model based Smart Contract Auditing with LLMBugScanner 

**Authors**: Yining Yuan, Yifei Wang, Yichang Xu, Zachary Yahn, Sihao Hu, Ling Liu  

**Link**: [PDF](https://arxiv.org/pdf/2512.02069)  

**Abstract**: This paper presents LLMBugScanner, a large language model (LLM) based framework for smart contract vulnerability detection using fine-tuning and ensemble learning. Smart contract auditing presents several challenges for LLMs: different pretrained models exhibit varying reasoning abilities, and no single model performs consistently well across all vulnerability types or contract structures. These limitations persist even after fine-tuning individual LLMs.
To address these challenges, LLMBugScanner combines domain knowledge adaptation with ensemble reasoning to improve robustness and generalization. Through domain knowledge adaptation, we fine-tune LLMs on complementary datasets to capture both general code semantics and instruction-guided vulnerability reasoning, using parameter-efficient tuning to reduce computational cost. Through ensemble reasoning, we leverage the complementary strengths of multiple LLMs and apply a consensus-based conflict resolution strategy to produce more reliable vulnerability assessments.
We conduct extensive experiments across multiple popular LLMs and compare LLMBugScanner with both pretrained and fine-tuned individual models. Results show that LLMBugScanner achieves consistent accuracy improvements and stronger generalization, demonstrating that it provides a principled, cost-effective, and extensible framework for smart contract auditing. 

---
# Parallel Multi-Circuit Quantum Feature Fusion in Hybrid Quantum-Classical Convolutional Neural Networks for Breast Tumor Classification 

**Authors**: Ece Yurtseven  

**Link**: [PDF](https://arxiv.org/pdf/2512.02066)  

**Abstract**: Quantum machine learning has emerged as a promising approach to improve feature extraction and classification tasks in high-dimensional data domains such as medical imaging. In this work, we present a hybrid Quantum-Classical Convolutional Neural Network (QCNN) architecture designed for the binary classification of the BreastMNIST dataset, a standardized benchmark for distinguishing between benign and malignant breast tumors. Our architecture integrates classical convolutional feature extraction with two distinct quantum circuits: an amplitude-encoding variational quantum circuit (VQC) and an angle-encoding VQC circuit with circular entanglement, both implemented on four qubits. These circuits generate quantum feature embeddings that are fused with classical features to form a joint feature space, which is subsequently processed by a fully connected classifier. To ensure fairness, the hybrid QCNN is parameter-matched against a baseline classical CNN, allowing us to isolate the contribution of quantum layers. Both models are trained under identical conditions using the Adam optimizer and binary cross-entropy loss. Experimental evaluation in five independent runs demonstrates that the hybrid QCNN achieves statistically significant improvements in classification accuracy compared to the classical CNN, as validated by a one-sided Wilcoxon signed rank test (p = 0.03125) and supported by large effect size of Cohen's d = 2.14. Our results indicate that hybrid QCNN architectures can leverage entanglement and quantum feature fusion to enhance medical image classification tasks. This work establishes a statistical validation framework for assessing hybrid quantum models in biomedical applications and highlights pathways for scaling to larger datasets and deployment on near-term quantum hardware. 

---
# Superpixel Attack: Enhancing Black-box Adversarial Attack with Image-driven Division Areas 

**Authors**: Issa Oe, Keiichiro Yamamura, Hiroki Ishikura, Ryo Hamahira, Katsuki Fujisawa  

**Link**: [PDF](https://arxiv.org/pdf/2512.02062)  

**Abstract**: Deep learning models are used in safety-critical tasks such as automated driving and face recognition. However, small perturbations in the model input can significantly change the predictions. Adversarial attacks are used to identify small perturbations that can lead to misclassifications. More powerful black-box adversarial attacks are required to develop more effective defenses. A promising approach to black-box adversarial attacks is to repeat the process of extracting a specific image area and changing the perturbations added to it. Existing attacks adopt simple rectangles as the areas where perturbations are changed in a single iteration. We propose applying superpixels instead, which achieve a good balance between color variance and compactness. We also propose a new search method, versatile search, and a novel attack method, Superpixel Attack, which applies superpixels and performs versatile search. Superpixel Attack improves attack success rates by an average of 2.10% compared with existing attacks. Most models used in this study are robust against adversarial attacks, and this improvement is significant for black-box adversarial attacks. The code is avilable at this https URL. 

---
# Ada-MoGE: Adaptive Mixture of Gaussian Expert Model for Time Series Forecasting 

**Authors**: Zhenliang Ni, Xiaowen Ma, Zhenkai Wu, Shuai Xiao, Han Shu, Xinghao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2512.02061)  

**Abstract**: Multivariate time series forecasts are widely used, such as industrial, transportation and financial forecasts. However, the dominant frequencies in time series may shift with the evolving spectral distribution of the data. Traditional Mixture of Experts (MoE) models, which employ a fixed number of experts, struggle to adapt to these changes, resulting in frequency coverage imbalance issue. Specifically, too few experts can lead to the overlooking of critical information, while too many can introduce noise. To this end, we propose Ada-MoGE, an adaptive Gaussian Mixture of Experts model. Ada-MoGE integrates spectral intensity and frequency response to adaptively determine the number of experts, ensuring alignment with the input data's frequency distribution. This approach prevents both information loss due to an insufficient number of experts and noise contamination from an excess of experts. Additionally, to prevent noise introduction from direct band truncation, we employ Gaussian band-pass filtering to smoothly decompose the frequency domain features, further optimizing the feature representation. The experimental results show that our model achieves state-of-the-art performance on six public benchmarks with only 0.2 million parameters. 

---
# Opening the Black Box: An Explainable, Few-shot AI4E Framework Informed by Physics and Expert Knowledge for Materials Engineering 

**Authors**: Haoxiang Zhang, Ruihao Yuan, Lihui Zhang, Yushi Luo, Qiang Zhang, Pan Ding, Xiaodong Ren, Weijie Xing, Niu Gao, Jishan Chen, Chubo Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2512.02057)  

**Abstract**: The industrial adoption of Artificial Intelligence for Engineering (AI4E) faces two fundamental bottlenecks: scarce high-quality data and the lack of interpretability in black-box models-particularly critical in safety-sensitive sectors like aerospace. We present an explainable, few-shot AI4E framework that is systematically informed by physics and expert knowledge throughout its architecture. Starting from only 32 experimental samples in an aerial K439B superalloy castings repair welding case, we first augment physically plausible synthetic data through a three-stage protocol: differentiated noise injection calibrated to process variabilities, enforcement of hard physical constraints, and preservation of inter-parameter relationships. We then employ a nested optimization strategy for constitutive model discovery, where symbolic regression explores equation structures while differential evolution optimizes parameters, followed by intensive parameter refinement using hybrid global-local optimization. The resulting interpretable constitutive equation achieves 88% accuracy in predicting hot-cracking tendency. This equation not only provides quantitative predictions but also delivers explicit physical insight, revealing how thermal, geometric, and metallurgical mechanisms couple to drive cracking-thereby advancing engineers' cognitive understanding of the process. Furthermore, the constitutive equation serves as a multi-functional tool for process optimization and high-fidelity virtual data generation, enabling accuracy improvements in other data-driven models. Our approach provides a general blueprint for developing trustworthy AI systems that embed engineering domain knowledge directly into their architecture, enabling reliable adoption in high-stakes industrial applications where data is limited but physical understanding is available. 

---
# Reversing Large Language Models for Efficient Training and Fine-Tuning 

**Authors**: Eshed Gal, Moshe Eliasof, Javier Turek, Uri Ascher, Eran Treister, Eldad Haber  

**Link**: [PDF](https://arxiv.org/pdf/2512.02056)  

**Abstract**: Large Language Models (LLMs) are known for their expensive and time-consuming training. Thus, oftentimes, LLMs are fine-tuned to address a specific task, given the pretrained weights of a pre-trained LLM considered a foundation model. In this work, we introduce memory-efficient, reversible architectures for LLMs, inspired by symmetric and symplectic differential equations, and investigate their theoretical properties. Different from standard, baseline architectures that store all intermediate activations, the proposed models use time-reversible dynamics to retrieve hidden states during backpropagation, relieving the need to store activations. This property allows for a drastic reduction in memory consumption, allowing for the processing of larger batch sizes for the same available memory, thereby offering improved throughput. In addition, we propose an efficient method for converting existing, non-reversible LLMs into reversible architectures through fine-tuning, rendering our approach practical for exploiting existing pre-trained models. Our results show comparable or improved performance on several datasets and benchmarks, on several LLMs, building a scalable and efficient path towards reducing the memory and computational costs associated with both training from scratch and fine-tuning of LLMs. 

---
# Leveraging AI multimodal geospatial foundation models for improved near-real-time flood mapping at a global scale 

**Authors**: Mirela G. Tulbure, Julio Caineta, Mark Broich, Mollie D. Gaines, Philippe Rufin, Leon-Friedrich Thomas, Hamed Alemohammad, Jan Hemmerling, Patrick Hostert  

**Link**: [PDF](https://arxiv.org/pdf/2512.02055)  

**Abstract**: Floods are among the most damaging weather-related hazards, and in 2024, the warmest year on record, extreme flood events affected communities across five continents. Earth observation (EO) satellites provide critical, frequent coverage for mapping inundation, yet operational accuracy depends heavily on labeled datasets and model generalization. Recent Geospatial Foundation Models (GFMs), such as ESA-IBM's TerraMind, offer improved generalizability through large-scale self-supervised pretraining, but their performance on diverse global flood events remains poorly understood.
We fine-tune TerraMind for flood extent mapping using FloodsNet, a harmonized multimodal dataset containing co-located Sentinel-1 (Synthetic Aperture Radar, SAR data) and Sentinel-2 (optical) imagery for 85 flood events worldwide. We tested four configurations (base vs. large models; frozen vs. unfrozen backbones) and compared against the TerraMind Sen1Floods11 example and a U-Net trained on both FloodsNet and Sen1Floods11. The base-unfrozen configuration provided the best balance of accuracy, precision, and recall at substantially lower computational cost than the large model. The large unfrozen model achieved the highest recall. Models trained on FloodsNet outperformed the Sen1Floods11-trained example in recall with similar overall accuracy. U-Net achieved higher recall than all GFM configurations, though with slightly lower accuracy and precision.
Our results demonstrate that integrating multimodal optical and SAR data and fine-tuning a GFM can enhance near-real-time flood mapping. This study provides one of the first global-scale evaluations of a GFM for flood segmentation, highlighting both its potential and current limitations for climate adaptation and disaster resilience. 

---
# The Impact of Artificial Intelligence on Enterprise Decision-Making Process 

**Authors**: Ernest Górka, Dariusz Baran, Gabriela Wojak, Michał Ćwiąkała, Sebastian Zupok, Dariusz Starkowski, Dariusz Reśko, Oliwia Okrasa  

**Link**: [PDF](https://arxiv.org/pdf/2512.02048)  

**Abstract**: Artificial intelligence improves enterprise decision-making by accelerating data analysis, reducing human error, and supporting evidence-based choices. A quantitative survey of 92 companies across multiple industries examines how AI adoption influences managerial performance, decision efficiency, and organizational barriers. Results show that 93 percent of firms use AI, primarily in customer service, data forecasting, and decision support. AI systems increase the speed and clarity of managerial decisions, yet implementation faces challenges. The most frequent barriers include employee resistance, high costs, and regulatory ambiguity. Respondents indicate that organizational factors are more significant than technological limitations. Critical competencies for successful AI use include understanding algorithmic mechanisms and change management. Technical skills such as programming play a smaller role. Employees report difficulties in adapting to AI tools, especially when formulating prompts or accepting system outputs. The study highlights the importance of integrating AI with human judgment and communication practices. When supported by adaptive leadership and transparent processes, AI adoption enhances organizational agility and strengthens decision-making performance. These findings contribute to ongoing research on how digital technologies reshape management and the evolution of hybrid human-machine decision environments. 

---
# Deep Research: A Systematic Survey 

**Authors**: Zhengliang Shi, Yiqun Chen, Haitao Li, Weiwei Sun, Shiyu Ni, Yougang Lyu, Run-Ze Fan, Bowen Jin, Yixuan Weng, Minjun Zhu, Qiujie Xie, Xinyu Guo, Qu Yang, Jiayi Wu, Jujia Zhao, Xiaqiang Tang, Xinbei Ma, Cunxiang Wang, Jiaxin Mao, Qingyao Ai, Jen-Tse Huang, Wenxuan Wang, Yue Zhang, Yiming Yang, Zhaopeng Tu, Zhaochun Ren  

**Link**: [PDF](https://arxiv.org/pdf/2512.02038)  

**Abstract**: Large language models (LLMs) have rapidly evolved from text generators into powerful problem solvers. Yet, many open tasks demand critical thinking, multi-source, and verifiable outputs, which are beyond single-shot prompting or standard retrieval-augmented generation. Recently, numerous studies have explored Deep Research (DR), which aims to combine the reasoning capabilities of LLMs with external tools, such as search engines, thereby empowering LLMs to act as research agents capable of completing complex, open-ended tasks. This survey presents a comprehensive and systematic overview of deep research systems, including a clear roadmap, foundational components, practical implementation techniques, important challenges, and future directions. Specifically, our main contributions are as follows: (i) we formalize a three-stage roadmap and distinguish deep research from related paradigms; (ii) we introduce four key components: query planning, information acquisition, memory management, and answer generation, each paired with fine-grained sub-taxonomies; (iii) we summarize optimization techniques, including prompting, supervised fine-tuning, and agentic reinforcement learning; and (iv) we consolidate evaluation criteria and open challenges, aiming to guide and facilitate future development. As the field of deep research continues to evolve rapidly, we are committed to continuously updating this survey to reflect the latest progress in this area. 

---
# Statistical Arbitrage in Polish Equities Market Using Deep Learning Techniques 

**Authors**: Marek Adamczyk, Michał Dąbrowski  

**Link**: [PDF](https://arxiv.org/pdf/2512.02037)  

**Abstract**: We study a systematic approach to a popular Statistical Arbitrage technique: Pairs Trading. Instead of relying on two highly correlated assets, we replace the second asset with a replication of the first using risk factor representations. These factors are obtained through Principal Components Analysis (PCA), exchange traded funds (ETFs), and, as our main contribution, Long Short Term Memory networks (LSTMs). Residuals between the main asset and its replication are examined for mean reversion properties, and trading signals are generated for sufficiently fast mean reverting portfolios.
Beyond introducing a deep learning based replication method, we adapt the framework of Avellaneda and Lee (2008) to the Polish market. Accordingly, components of WIG20, mWIG40, and selected sector indices replace the original S&P500 universe, and market parameters such as the risk free rate and transaction costs are updated to reflect local conditions.
We outline the full strategy pipeline: risk factor construction, residual modeling via the Ornstein Uhlenbeck process, and signal generation. Each replication technique is described together with its practical implementation. Strategy performance is evaluated over two periods: 2017-2019 and the recessive year 2020.
All methods yield profits in 2017-2019, with PCA achieving roughly 20 percent cumulative return and an annualized Sharpe ratio of up to 2.63. Despite multiple adaptations, our conclusions remain consistent with those of the original paper. During the COVID-19 recession, only the ETF based approach remains profitable (about 5 percent annual return), while PCA and LSTM methods underperform. LSTM results, although negative, are promising and indicate potential for future optimization. 

---
# Integration of LSTM Networks in Random Forest Algorithms for Stock Market Trading Predictions 

**Authors**: Juan C. King, Jose M. Amigo  

**Link**: [PDF](https://arxiv.org/pdf/2512.02036)  

**Abstract**: The aim of this paper is the analysis and selection of stock trading systems that combine different models with data of different nature, such as financial and microeconomic information. Specifically, based on previous work by the authors and applying advanced techniques of Machine Learning and Deep Learning, our objective is to formulate trading algorithms for the stock market with empirically tested statistical advantages, thus improving results published in the literature. Our approach integrates Long Short-Term Memory (LSTM) networks with algorithms based on decision trees, such as Random Forest and Gradient Boosting. While the former analyze price patterns of financial assets, the latter are fed with economic data of companies. Numerical simulations of algorithmic trading with data from international companies and 10-weekday predictions confirm that an approach based on both fundamental and technical variables can outperform the usual approaches, which do not combine those two types of variables. In doing so, Random Forest turned out to be the best performer among the decision trees. We also discuss how the prediction performance of such a hybrid approach can be boosted by selecting the technical variables. 

---
# CONFIDE: Hallucination Assessment for Reliable Biomolecular Structure Prediction and Design 

**Authors**: Zijun Gao, Mutian He, Shijia Sun, Hanqun Cao, Jingjie Zhang, Zihao Luo, Xiaorui Wang, Xiaojun Yao, Chang-Yu Hsieh, Chunbin Gu, Pheng Ann Heng  

**Link**: [PDF](https://arxiv.org/pdf/2512.02033)  

**Abstract**: Reliable evaluation of protein structure predictions remains challenging, as metrics like pLDDT capture energetic stability but often miss subtle errors such as atomic clashes or conformational traps reflecting topological frustration within the protein folding energy landscape. We present CODE (Chain of Diffusion Embeddings), a self evaluating metric empirically found to quantify topological frustration directly from the latent diffusion embeddings of the AlphaFold3 series of structure predictors in a fully unsupervised manner. Integrating this with pLDDT, we propose CONFIDE, a unified evaluation framework that combines energetic and topological perspectives to improve the reliability of AlphaFold3 and related models. CODE strongly correlates with protein folding rates driven by topological frustration, achieving a correlation of 0.82 compared to pLDDT's 0.33 (a relative improvement of 148\%). CONFIDE significantly enhances the reliability of quality evaluation in molecular glue structure prediction benchmarks, achieving a Spearman correlation of 0.73 with RMSD, compared to pLDDT's correlation of 0.42, a relative improvement of 73.8\%. Beyond quality assessment, our approach applies to diverse drug design tasks, including all-atom binder design, enzymatic active site mapping, mutation induced binding affinity prediction, nucleic acid aptamer screening, and flexible protein modeling. By combining data driven embeddings with theoretical insight, CODE and CONFIDE outperform existing metrics across a wide range of biomolecular systems, offering robust and versatile tools to refine structure predictions, advance structural biology, and accelerate drug discovery. 

---
# Characterizing Continuous and Discrete Hybrid Latent Spaces for Structural Connectomes 

**Authors**: Gaurav Rudravaram, Lianrui Zuo, Adam M. Saunders, Michael E. Kim, Praitayini Kanakaraj, Nancy R. Newlin, Aravind R. Krishnan, Elyssa M. McMaster, Chloe Cho, Susan M. Resnick, Lori L. Beason Held, Derek Archer, Timothy J. Hohman, Daniel C. Moyer, Bennett A. Landman  

**Link**: [PDF](https://arxiv.org/pdf/2512.02032)  

**Abstract**: Structural connectomes are detailed graphs that map how different brain regions are physically connected, offering critical insight into aging, cognition, and neurodegenerative diseases. However, these connectomes are high-dimensional and densely interconnected, which makes them difficult to interpret and analyze at scale. While low-dimensional spaces like PCA and autoencoders are often used to capture major sources of variation, their latent spaces are generally continuous and cannot fully reflect the mixed nature of variability in connectomes, which include both continuous (e.g., connectivity strength) and discrete factors (e.g., imaging site). Motivated by this, we propose a variational autoencoder (VAE) with a hybrid latent space that jointly models the discrete and continuous components. We analyze a large dataset of 5,761 connectomes from six Alzheimer's disease studies with ten acquisition protocols. Each connectome represents a single scan from a unique subject (3579 females, 2182 males), aged 22 to 102, with 4338 cognitively normal, 809 with mild cognitive impairment (MCI), and 614 with Alzheimer's disease (AD). Each connectome contains 121 brain regions defined by the BrainCOLOR atlas. We train our hybrid VAE in an unsupervised way and characterize what each latent component captures. We find that the discrete space is particularly effective at capturing subtle site-related differences, achieving an Adjusted Rand Index (ARI) of 0.65 with site labels, significantly outperforming PCA and a standard VAE followed by clustering (p < 0.05). These results demonstrate that the hybrid latent space can disentangle distinct sources of variability in connectomes in an unsupervised manner, offering potential for large-scale connectome analysis. 

---
# On the Difficulty of Token-Level Modeling of Dysfluency and Fluency Shaping Artifacts 

**Authors**: Kashaf Gulzar, Dominik Wagner, Sebastian P. Bayerl, Florian Hönig, Tobias Bocklet, Korbinian Riedhammer  

**Link**: [PDF](https://arxiv.org/pdf/2512.02027)  

**Abstract**: Automatic transcription of stuttered speech remains a challenge, even for modern end-to-end (E2E) automatic speech recognition (ASR) frameworks. Dysfluencies and fluency-shaping artifacts are often overlooked, resulting in non-verbatim transcriptions with limited clinical and research value. We propose a parameter-efficient adaptation method to decode dysfluencies and fluency modifications as special tokens within transcriptions, evaluated on simulated (LibriStutter, English) and natural (KSoF, German) stuttered speech datasets. To mitigate ASR performance disparities and bias towards English, we introduce a multi-step fine-tuning strategy with language-adaptive pretraining. Tokenization analysis further highlights the tokenizer's English-centric bias, which poses challenges for improving performance on German data. Our findings demonstrate the effectiveness of lightweight adaptation techniques for dysfluency-aware ASR while exposing key limitations in multilingual E2E systems. 

---
# Towards Sustainable Precision: Machine Learning for Laser Micromachining Optimization 

**Authors**: Luis Correas-Naranjo, Miguel Camacho-Sánchez, Laëtitia Launet, Milena Zuric, Valery Naranjo  

**Link**: [PDF](https://arxiv.org/pdf/2512.02026)  

**Abstract**: In the pursuit of sustainable manufacturing, ultra-short pulse laser micromachining stands out as a promising solution while also offering high-precision and qualitative laser processing. However, unlocking the full potential of ultra-short pulse lasers requires an optimized monitoring system capable of early detection of defective workpieces, regardless of the preprocessing technique employed. While advances in machine learning can help predict process quality features, the complexity of monitoring data necessitates reducing both model size and data dimensionality to enable real-time analysis. To address these challenges, this paper introduces a machine learning framework designed to enhance surface quality assessment across diverse preprocessing techniques. To facilitate real-time laser processing monitoring, our solution aims to optimize the computational requirements of the machine learning model. Experimental results show that the proposed model not only outperforms the generalizability achieved by previous works across diverse preprocessing techniques but also significantly reduces the computational requirements for training. Through these advancements, we aim to establish the baseline for a more sustainable manufacturing process. 

---
# DySTAN: Joint Modeling of Sedentary Activity and Social Context from Smartphone Sensors 

**Authors**: Aditya Sneh, Nilesh Kumar Sahu, Snehil Gupta, Haroon R. Lone  

**Link**: [PDF](https://arxiv.org/pdf/2512.02025)  

**Abstract**: Accurately recognizing human context from smartphone sensor data remains a significant challenge, especially in sedentary settings where activities such as studying, attending lectures, relaxing, and eating exhibit highly similar inertial patterns. Furthermore, social context plays a critical role in understanding user behavior, yet is often overlooked in mobile sensing research. To address these gaps, we introduce LogMe, a mobile sensing application that passively collects smartphone sensor data (accelerometer, gyroscope, magnetometer, and rotation vector) and prompts users for hourly self-reports capturing both sedentary activity and social context. Using this dual-label dataset, we propose DySTAN (Dynamic Cross-Stitch with Task Attention Network), a multi-task learning framework that jointly classifies both context dimensions from shared sensor inputs. It integrates task-specific layers with cross-task attention to model subtle distinctions effectively. DySTAN improves sedentary activity macro F1 scores by 21.8% over a single-task CNN-BiLSTM-GRU (CBG) model and by 8.2% over the strongest multi-task baseline, Sluice Network (SN). These results demonstrate the importance of modeling multiple, co-occurring context dimensions to improve the accuracy and robustness of mobile context recognition. 

---
# Do Large Language Models Walk Their Talk? Measuring the Gap Between Implicit Associations, Self-Report, and Behavioral Altruism 

**Authors**: Sandro Andric  

**Link**: [PDF](https://arxiv.org/pdf/2512.01568)  

**Abstract**: We investigate whether Large Language Models (LLMs) exhibit altruistic tendencies, and critically, whether their implicit associations and self-reports predict actual altruistic behavior. Using a multi-method approach inspired by human social psychology, we tested 24 frontier LLMs across three paradigms: (1) an Implicit Association Test (IAT) measuring implicit altruism bias, (2) a forced binary choice task measuring behavioral altruism, and (3) a self-assessment scale measuring explicit altruism beliefs. Our key findings are: (1) All models show strong implicit pro-altruism bias (mean IAT = 0.87, p < .0001), confirming models "know" altruism is good. (2) Models behave more altruistically than chance (65.6% vs. 50%, p < .0001), but with substantial variation (48-85%). (3) Implicit associations do not predict behavior (r = .22, p = .29). (4) Most critically, models systematically overestimate their own altruism, claiming 77.5% altruism while acting at 65.6% (p < .0001, Cohen's d = 1.08). This "virtue signaling gap" affects 75% of models tested. Based on these findings, we recommend the Calibration Gap (the discrepancy between self-reported and behavioral values) as a standardized alignment metric. Well-calibrated models are more predictable and behaviorally consistent; only 12.5% of models achieve the ideal combination of high prosocial behavior and accurate self-knowledge. 

---
# Graphing the Truth: Structured Visualizations for Automated Hallucination Detection in LLMs 

**Authors**: Tanmay Agrawal  

**Link**: [PDF](https://arxiv.org/pdf/2512.00663)  

**Abstract**: Large Language Models have rapidly advanced in their ability to interpret and generate natural language. In enterprise settings, they are frequently augmented with closed-source domain knowledge to deliver more contextually informed responses. However, operational constraints such as limited context windows and inconsistencies between pre-training data and supplied knowledge often lead to hallucinations, some of which appear highly credible and escape routine human review. Current mitigation strategies either depend on costly, large-scale gold-standard Q\&A curation or rely on secondary model verification, neither of which offers deterministic assurance. This paper introduces a framework that organizes proprietary knowledge and model-generated content into interactive visual knowledge graphs. The objective is to provide end users with a clear, intuitive view of potential hallucination zones by linking model assertions to underlying sources of truth and indicating confidence levels. Through this visual interface, users can diagnose inconsistencies, identify weak reasoning chains, and supply corrective feedback. The resulting human-in-the-loop workflow creates a structured feedback loop that can enhance model reliability and continuously improve response quality. 

---
# Mixed precision accumulation for neural network inference guided by componentwise forward error analysis 

**Authors**: El-Mehdi El Arar, Silviu-Ioan Filip, Theo Mary, Elisa Riccietti  

**Link**: [PDF](https://arxiv.org/pdf/2503.15568)  

**Abstract**: This work proposes a mathematically founded mixed precision accumulation strategy for the inference of neural networks. Our strategy is based on a new componentwise forward error analysis that explains the propagation of errors in the forward pass of neural networks. Specifically, our analysis shows that the error in each component of the output of a linear layer is proportional to the condition number of the inner product between the weights and the input, multiplied by the condition number of the activation function. These condition numbers can vary widely from one component to the other, thus creating a significant opportunity to introduce mixed precision: each component should be accumulated in a precision inversely proportional to the product of these condition numbers. We propose a numerical algorithm that exploits this observation: it first computes all components in low precision, uses this output to estimate the condition numbers, and recomputes in higher precision only the components associated with large condition numbers. We test our algorithm on various networks and datasets and confirm experimentally that it can significantly improve the cost--accuracy tradeoff compared with uniform precision accumulation baselines. 

---
