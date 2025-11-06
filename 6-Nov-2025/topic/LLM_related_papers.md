# Toward Autonomous Engineering Design: A Knowledge-Guided Multi-Agent Framework 

**Authors**: Varun Kumar, George Em Karniadakis  

**Link**: [PDF](https://arxiv.org/pdf/2511.03179)  

**Abstract**: The engineering design process often demands expertise from multiple domains, leading to complex collaborations and iterative refinements. Traditional methods can be resource-intensive and prone to inefficiencies. To address this, we formalize the engineering design process through a multi-agent AI framework that integrates structured design and review loops. The framework introduces specialized knowledge-driven agents that collaborate to generate and refine design candidates. As an exemplar, we demonstrate its application to the aerodynamic optimization of 4-digit NACA airfoils. The framework consists of three key AI agents: a Graph Ontologist, a Design Engineer, and a Systems Engineer. The Graph Ontologist employs a Large Language Model (LLM) to construct two domain-specific knowledge graphs from airfoil design literature. The Systems Engineer, informed by a human manager, formulates technical requirements that guide design generation and evaluation. The Design Engineer leverages the design knowledge graph and computational tools to propose candidate airfoils meeting these requirements. The Systems Engineer reviews and provides feedback both qualitative and quantitative using its own knowledge graph, forming an iterative feedback loop until a design is validated by the manager. The final design is then optimized to maximize performance metrics such as the lift-to-drag ratio. Overall, this work demonstrates how collaborative AI agents equipped with structured knowledge representations can enhance efficiency, consistency, and quality in the engineering design process. 

---
# Using Multi-modal Large Language Model to Boost Fireworks Algorithm's Ability in Settling Challenging Optimization Tasks 

**Authors**: Shipeng Cen, Ying Tan  

**Link**: [PDF](https://arxiv.org/pdf/2511.03137)  

**Abstract**: As optimization problems grow increasingly complex and diverse, advancements in optimization techniques and paradigm innovations hold significant importance. The challenges posed by optimization problems are primarily manifested in their non-convexity, high-dimensionality, black-box nature, and other unfavorable characteristics. Traditional zero-order or first-order methods, which are often characterized by low efficiency, inaccurate gradient information, and insufficient utilization of optimization information, are ill-equipped to address these challenges effectively. In recent years, the rapid development of large language models (LLM) has led to substantial improvements in their language understanding and code generation capabilities. Consequently, the design of optimization algorithms leveraging large language models has garnered increasing attention from researchers. In this study, we choose the fireworks algorithm(FWA) as the basic optimizer and propose a novel approach to assist the design of the FWA by incorporating multi-modal large language model(MLLM). To put it simply, we propose the concept of Critical Part(CP), which extends FWA to complex high-dimensional tasks, and further utilizes the information in the optimization process with the help of the multi-modal characteristics of large language models. We focus on two specific tasks: the \textit{traveling salesman problem }(TSP) and \textit{electronic design automation problem} (EDA). The experimental results show that FWAs generated under our new framework have achieved or surpassed SOTA results on many problem instances. 

---
# Towards Scalable Web Accessibility Audit with MLLMs as Copilots 

**Authors**: Ming Gu, Ziwei Wang, Sicen Lai, Zirui Gao, Sheng Zhou, Jiajun Bu  

**Link**: [PDF](https://arxiv.org/pdf/2511.03471)  

**Abstract**: Ensuring web accessibility is crucial for advancing social welfare, justice, and equality in digital spaces, yet the vast majority of website user interfaces remain non-compliant, due in part to the resource-intensive and unscalable nature of current auditing practices. While WCAG-EM offers a structured methodology for site-wise conformance evaluation, it involves great human efforts and lacks practical support for execution at scale. In this work, we present an auditing framework, AAA, which operationalizes WCAG-EM through a human-AI partnership model. AAA is anchored by two key innovations: GRASP, a graph-based multimodal sampling method that ensures representative page coverage via learned embeddings of visual, textual, and relational cues; and MaC, a multimodal large language model-based copilot that supports auditors through cross-modal reasoning and intelligent assistance in high-effort tasks. Together, these components enable scalable, end-to-end web accessibility auditing, empowering human auditors with AI-enhanced assistance for real-world impact. We further contribute four novel datasets designed for benchmarking core stages of the audit pipeline. Extensive experiments demonstrate the effectiveness of our methods, providing insights that small-scale language models can serve as capable experts when fine-tuned. 

---
# From Five Dimensions to Many: Large Language Models as Precise and Interpretable Psychological Profilers 

**Authors**: Yi-Fei Liu, Yi-Long Lu, Di He, Hang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.03235)  

**Abstract**: Psychological constructs within individuals are widely believed to be interconnected. We investigated whether and how Large Language Models (LLMs) can model the correlational structure of human psychological traits from minimal quantitative inputs. We prompted various LLMs with Big Five Personality Scale responses from 816 human individuals to role-play their responses on nine other psychological scales. LLMs demonstrated remarkable accuracy in capturing human psychological structure, with the inter-scale correlation patterns from LLM-generated responses strongly aligning with those from human data $(R^2 > 0.89)$. This zero-shot performance substantially exceeded predictions based on semantic similarity and approached the accuracy of machine learning algorithms trained directly on the dataset. Analysis of reasoning traces revealed that LLMs use a systematic two-stage process: First, they transform raw Big Five responses into natural language personality summaries through information selection and compression, analogous to generating sufficient statistics. Second, they generate target scale responses based on reasoning from these summaries. For information selection, LLMs identify the same key personality factors as trained algorithms, though they fail to differentiate item importance within factors. The resulting compressed summaries are not merely redundant representations but capture synergistic information--adding them to original scores enhances prediction alignment, suggesting they encode emergent, second-order patterns of trait interplay. Our findings demonstrate that LLMs can precisely predict individual participants' psychological traits from minimal data through a process of abstraction and reasoning, offering both a powerful tool for psychological simulation and valuable insights into their emergent reasoning capabilities. 

---
# A Proprietary Model-Based Safety Response Framework for AI Agents 

**Authors**: Qi Li, Jianjun Xu, Pingtao Wei, Jiu Li, Peiqiang Zhao, Jiwei Shi, Xuan Zhang, Yanhui Yang, Xiaodong Hui, Peng Xu, Wenqin Shao  

**Link**: [PDF](https://arxiv.org/pdf/2511.03138)  

**Abstract**: With the widespread application of Large Language Models (LLMs), their associated security issues have become increasingly prominent, severely constraining their trustworthy deployment in critical domains. This paper proposes a novel safety response framework designed to systematically safeguard LLMs at both the input and output levels. At the input level, the framework employs a supervised fine-tuning-based safety classification model. Through a fine-grained four-tier taxonomy (Safe, Unsafe, Conditionally Safe, Focused Attention), it performs precise risk identification and differentiated handling of user queries, significantly enhancing risk coverage and business scenario adaptability, and achieving a risk recall rate of 99.3%. At the output level, the framework integrates Retrieval-Augmented Generation (RAG) with a specifically fine-tuned interpretation model, ensuring all responses are grounded in a real-time, trustworthy knowledge base. This approach eliminates information fabrication and enables result traceability. Experimental results demonstrate that our proposed safety control model achieves a significantly higher safety score on public safety evaluation benchmarks compared to the baseline model, TinyR1-Safety-8B. Furthermore, on our proprietary high-risk test set, the framework's components attained a perfect 100% safety score, validating their exceptional protective capabilities in complex risk scenarios. This research provides an effective engineering pathway for building high-security, high-trust LLM applications. 

---
# Large language models require a new form of oversight: capability-based monitoring 

**Authors**: Katherine C. Kellogg, Bingyang Ye, Yifan Hu, Guergana K. Savova, Byron Wallace, Danielle S. Bitterman  

**Link**: [PDF](https://arxiv.org/pdf/2511.03106)  

**Abstract**: The rapid adoption of large language models (LLMs) in healthcare has been accompanied by scrutiny of their oversight. Existing monitoring approaches, inherited from traditional machine learning (ML), are task-based and founded on assumed performance degradation arising from dataset drift. In contrast, with LLMs, inevitable model degradation due to changes in populations compared to the training dataset cannot be assumed, because LLMs were not trained for any specific task in any given population. We therefore propose a new organizing principle guiding generalist LLM monitoring that is scalable and grounded in how these models are developed and used in practice: capability-based monitoring. Capability-based monitoring is motivated by the fact that LLMs are generalist systems whose overlapping internal capabilities are reused across numerous downstream tasks. Instead of evaluating each downstream task independently, this approach organizes monitoring around shared model capabilities, such as summarization, reasoning, translation, or safety guardrails, in order to enable cross-task detection of systemic weaknesses, long-tail errors, and emergent behaviors that task-based monitoring may miss. We describe considerations for developers, organizational leaders, and professional societies for implementing a capability-based monitoring approach. Ultimately, capability-based monitoring will provide a scalable foundation for safe, adaptive, and collaborative monitoring of LLMs and future generalist artificial intelligence models in healthcare. 

---
# No-Human in the Loop: Agentic Evaluation at Scale for Recommendation 

**Authors**: Tao Zhang, Kehui Yao, Luyi Ma, Jiao Chen, Reza Yousefi Maragheh, Kai Zhao, Jianpeng Xu, Evren Korpeoglu, Sushant Kumar, Kannan Achan  

**Link**: [PDF](https://arxiv.org/pdf/2511.03051)  

**Abstract**: Evaluating large language models (LLMs) as judges is increasingly critical for building scalable and trustworthy evaluation pipelines. We present ScalingEval, a large-scale benchmarking study that systematically compares 36 LLMs, including GPT, Gemini, Claude, and Llama, across multiple product categories using a consensus-driven evaluation protocol. Our multi-agent framework aggregates pattern audits and issue codes into ground-truth labels via scalable majority voting, enabling reproducible comparison of LLM evaluators without human annotation. Applied to large-scale complementary-item recommendation, the benchmark reports four key findings: (i) Anthropic Claude 3.5 Sonnet achieves the highest decision confidence; (ii) Gemini 1.5 Pro offers the best overall performance across categories; (iii) GPT-4o provides the most favorable latency-accuracy-cost tradeoff; and (iv) GPT-OSS 20B leads among open-source models. Category-level analysis shows strong consensus in structured domains (Electronics, Sports) but persistent disagreement in lifestyle categories (Clothing, Food). These results establish ScalingEval as a reproducible benchmark and evaluation protocol for LLMs as judges, with actionable guidance on scaling, reliability, and model family tradeoffs. 

---
# Grounded Misunderstandings in Asymmetric Dialogue: A Perspectivist Annotation Scheme for MapTask 

**Authors**: Nan Li, Albert Gatt, Massimo Poesio  

**Link**: [PDF](https://arxiv.org/pdf/2511.03718)  

**Abstract**: Collaborative dialogue relies on participants incrementally establishing common ground, yet in asymmetric settings they may believe they agree while referring to different entities. We introduce a perspectivist annotation scheme for the HCRC MapTask corpus (Anderson et al., 1991) that separately captures speaker and addressee grounded interpretations for each reference expression, enabling us to trace how understanding emerges, diverges, and repairs over time. Using a scheme-constrained LLM annotation pipeline, we obtain 13k annotated reference expressions with reliability estimates and analyze the resulting understanding states. The results show that full misunderstandings are rare once lexical variants are unified, but multiplicity discrepancies systematically induce divergences, revealing how apparent grounding can mask referential misalignment. Our framework provides both a resource and an analytic lens for studying grounded misunderstanding and for evaluating (V)LLMs' capacity to model perspective-dependent grounding in collaborative dialogue. 

---
# AnaFlow: Agentic LLM-based Workflow for Reasoning-Driven Explainable and Sample-Efficient Analog Circuit Sizing 

**Authors**: Mohsen Ahmadzadeh, Kaichang Chen, Georges Gielen  

**Link**: [PDF](https://arxiv.org/pdf/2511.03697)  

**Abstract**: Analog/mixed-signal circuits are key for interfacing electronics with the physical world. Their design, however, remains a largely handcrafted process, resulting in long and error-prone design cycles. While the recent rise of AI-based reinforcement learning and generative AI has created new techniques to automate this task, the need for many time-consuming simulations is a critical bottleneck hindering the overall efficiency. Furthermore, the lack of explainability of the resulting design solutions hampers widespread adoption of the tools. To address these issues, a novel agentic AI framework for sample-efficient and explainable analog circuit sizing is presented. It employs a multi-agent workflow where specialized Large Language Model (LLM)-based agents collaborate to interpret the circuit topology, to understand the design goals, and to iteratively refine the circuit's design parameters towards the target goals with human-interpretable reasoning. The adaptive simulation strategy creates an intelligent control that yields a high sample efficiency. The AnaFlow framework is demonstrated for two circuits of varying complexity and is able to complete the sizing task fully automatically, differently from pure Bayesian optimization and reinforcement learning approaches. The system learns from its optimization history to avoid past mistakes and to accelerate convergence. The inherent explainability makes this a powerful tool for analog design space exploration and a new paradigm in analog EDA, where AI agents serve as transparent design assistants. 

---
# Whisper Leak: a side-channel attack on Large Language Models 

**Authors**: Geoff McDonald, Jonathan Bar Or  

**Link**: [PDF](https://arxiv.org/pdf/2511.03675)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed in sensitive domains including healthcare, legal services, and confidential communications, where privacy is paramount. This paper introduces Whisper Leak, a side-channel attack that infers user prompt topics from encrypted LLM traffic by analyzing packet size and timing patterns in streaming responses. Despite TLS encryption protecting content, these metadata patterns leak sufficient information to enable topic classification. We demonstrate the attack across 28 popular LLMs from major providers, achieving near-perfect classification (often >98% AUPRC) and high precision even at extreme class imbalance (10,000:1 noise-to-target ratio). For many models, we achieve 100% precision in identifying sensitive topics like "money laundering" while recovering 5-20% of target conversations. This industry-wide vulnerability poses significant risks for users under network surveillance by ISPs, governments, or local adversaries. We evaluate three mitigation strategies - random padding, token batching, and packet injection - finding that while each reduces attack effectiveness, none provides complete protection. Through responsible disclosure, we have collaborated with providers to implement initial countermeasures. Our findings underscore the need for LLM providers to address metadata leakage as AI systems handle increasingly sensitive information. 

---
# PerfDojo: Automated ML Library Generation for Heterogeneous Architectures 

**Authors**: Andrei Ivanov, Siyuan Shen, Gioele Gottardo, Marcin Chrapek, Afif Boudaoud, Timo Schneider, Luca Benini, Torsten Hoefler  

**Link**: [PDF](https://arxiv.org/pdf/2511.03586)  

**Abstract**: The increasing complexity of machine learning models and the proliferation of diverse hardware architectures (CPUs, GPUs, accelerators) make achieving optimal performance a significant challenge. Heterogeneity in instruction sets, specialized kernel requirements for different data types and model features (e.g., sparsity, quantization), and architecture-specific optimizations complicate performance tuning. Manual optimization is resource-intensive, while existing automatic approaches often rely on complex hardware-specific heuristics and uninterpretable intermediate representations, hindering performance portability. We introduce PerfLLM, a novel automatic optimization methodology leveraging Large Language Models (LLMs) and Reinforcement Learning (RL). Central to this is PerfDojo, an environment framing optimization as an RL game using a human-readable, mathematically-inspired code representation that guarantees semantic validity through transformations. This allows effective optimization without prior hardware knowledge, facilitating both human analysis and RL agent training. We demonstrate PerfLLM's ability to achieve significant performance gains across diverse CPU (x86, Arm, RISC-V) and GPU architectures. 

---
# PublicAgent: Multi-Agent Design Principles From an LLM-Based Open Data Analysis Framework 

**Authors**: Sina Montazeri, Yunhe Feng, Kewei Sha  

**Link**: [PDF](https://arxiv.org/pdf/2511.03023)  

**Abstract**: Open data repositories hold potential for evidence-based decision-making, yet are inaccessible to non-experts lacking expertise in dataset discovery, schema mapping, and statistical analysis. Large language models show promise for individual tasks, but end-to-end analytical workflows expose fundamental limitations: attention dilutes across growing contexts, specialized reasoning patterns interfere, and errors propagate undetected. We present PublicAgent, a multi-agent framework that addresses these limitations through decomposition into specialized agents for intent clarification, dataset discovery, analysis, and reporting. This architecture maintains focused attention within agent contexts and enables validation at each stage. Evaluation across five models and 50 queries derives five design principles for multi-agent LLM systems. First, specialization provides value independent of model strength--even the strongest model shows 97.5% agent win rates, with benefits orthogonal to model scale. Second, agents divide into universal (discovery, analysis) and conditional (report, intent) categories. Universal agents show consistent effectiveness (std dev 12.4%) while conditional agents vary by model (std dev 20.5%). Third, agents mitigate distinct failure modes--removing discovery or analysis causes catastrophic failures (243-280 instances), while removing report or intent causes quality degradation. Fourth, architectural benefits persist across task complexity with stable win rates (86-92% analysis, 84-94% discovery), indicating workflow management value rather than reasoning enhancement. Fifth, wide variance in agent effectiveness across models (42-96% for analysis) requires model-aware architecture design. These principles guide when and why specialization is necessary for complex analytical workflows while enabling broader access to public data through natural language interfaces. 

---
# LiveTradeBench: Seeking Real-World Alpha with Large Language Models 

**Authors**: Haofei Yu, Fenghai Li, Jiaxuan You  

**Link**: [PDF](https://arxiv.org/pdf/2511.03628)  

**Abstract**: Large language models (LLMs) achieve strong performance across benchmarks--from knowledge quizzes and math reasoning to web-agent tasks--but these tests occur in static settings, lacking real dynamics and uncertainty. Consequently, they evaluate isolated reasoning or problem-solving rather than decision-making under uncertainty. To address this, we introduce LiveTradeBench, a live trading environment for evaluating LLM agents in realistic and evolving markets. LiveTradeBench follows three design principles: (i) Live data streaming of market prices and news, eliminating dependence on offline backtesting and preventing information leakage while capturing real-time uncertainty; (ii) a portfolio-management abstraction that extends control from single-asset actions to multi-asset allocation, integrating risk management and cross-asset reasoning; and (iii) multi-market evaluation across structurally distinct environments--U.S. stocks and Polymarket prediction markets--differing in volatility, liquidity, and information flow. At each step, an agent observes prices, news, and its portfolio, then outputs percentage allocations that balance risk and return. Using LiveTradeBench, we run 50-day live evaluations of 21 LLMs across families. Results show that (1) high LMArena scores do not imply superior trading outcomes; (2) models display distinct portfolio styles reflecting risk appetite and reasoning dynamics; and (3) some LLMs effectively leverage live signals to adapt decisions. These findings expose a gap between static evaluation and real-world competence, motivating benchmarks that test sequential decision making and consistency under live uncertainty. 

---
# Uncovering Code Insights: Leveraging GitHub Artifacts for Deeper Code Understanding 

**Authors**: Ziv Nevo, Orna Raz, Karen Yorav  

**Link**: [PDF](https://arxiv.org/pdf/2511.03549)  

**Abstract**: Understanding the purpose of source code is a critical task in software maintenance, onboarding, and modernization. While large language models (LLMs) have shown promise in generating code explanations, they often lack grounding in the broader software engineering context. We propose a novel approach that leverages natural language artifacts from GitHub -- such as pull request descriptions, issue descriptions and discussions, and commit messages -- to enhance LLM-based code understanding. Our system consists of three components: one that extracts and structures relevant GitHub context, another that uses this context to generate high-level explanations of the code's purpose, and a third that validates the explanation. We implemented this as a standalone tool, as well as a server within the Model Context Protocol (MCP), enabling integration with other AI-assisted development tools. Our main use case is that of enhancing a standard LLM-based code explanation with code insights that our system generates. To evaluate explanations' quality, we conducted a small scale user study, with developers of several open projects, as well as developers of proprietary projects. Our user study indicates that when insights are generated they often are helpful and non trivial, and are free from hallucinations. 

---
# Visualization Biases MLLM's Decision Making in Network Data Tasks 

**Authors**: Timo Brand, Henry Förster, Stephen G. Kobourov, Jacob Miller  

**Link**: [PDF](https://arxiv.org/pdf/2511.03617)  

**Abstract**: We evaluate how visualizations can influence the judgment of MLLMs about the presence or absence of bridges in a network. We show that the inclusion of visualization improves confidence over a structured text-based input that could theoretically be helpful for answering the question. On the other hand, we observe that standard visualization techniques create a strong bias towards accepting or refuting the presence of a bridge -- independently of whether or not a bridge actually exists in the network. While our results indicate that the inclusion of visualization techniques can effectively influence the MLLM's judgment without compromising its self-reported confidence, they also imply that practitioners must be careful of allowing users to include visualizations in generative AI applications so as to avoid undesired hallucinations. 

---
# CareMedEval dataset: Evaluating Critical Appraisal and Reasoning in the Biomedical Field 

**Authors**: Doria Bonzi, Alexandre Guiggi, Frédéric Béchet, Carlos Ramisch, Benoit Favre  

**Link**: [PDF](https://arxiv.org/pdf/2511.03441)  

**Abstract**: Critical appraisal of scientific literature is an essential skill in the biomedical field. While large language models (LLMs) can offer promising support in this task, their reliability remains limited, particularly for critical reasoning in specialized domains. We introduce CareMedEval, an original dataset designed to evaluate LLMs on biomedical critical appraisal and reasoning tasks. Derived from authentic exams taken by French medical students, the dataset contains 534 questions based on 37 scientific articles. Unlike existing benchmarks, CareMedEval explicitly evaluates critical reading and reasoning grounded in scientific papers. Benchmarking state-of-the-art generalist and biomedical-specialized LLMs under various context conditions reveals the difficulty of the task: open and commercial models fail to exceed an Exact Match Rate of 0.5 even though generating intermediate reasoning tokens considerably improves the results. Yet, models remain challenged especially on questions about study limitations and statistical analysis. CareMedEval provides a challenging benchmark for grounded reasoning, exposing current LLM limitations and paving the way for future development of automated support for critical appraisal. 

---
# Comparing the Performance of LLMs in RAG-based Question-Answering: A Case Study in Computer Science Literature 

**Authors**: Ranul Dayarathne, Uvini Ranaweera, Upeksha Ganegoda  

**Link**: [PDF](https://arxiv.org/pdf/2511.03261)  

**Abstract**: Retrieval Augmented Generation (RAG) is emerging as a powerful technique to enhance the capabilities of Generative AI models by reducing hallucination. Thus, the increasing prominence of RAG alongside Large Language Models (LLMs) has sparked interest in comparing the performance of different LLMs in question-answering (QA) in diverse domains. This study compares the performance of four open-source LLMs, Mistral-7b-instruct, LLaMa2-7b-chat, Falcon-7b-instruct and Orca-mini-v3-7b, and OpenAI's trending GPT-3.5 over QA tasks within the computer science literature leveraging RAG support. Evaluation metrics employed in the study include accuracy and precision for binary questions and ranking by a human expert, ranking by Google's AI model Gemini, alongside cosine similarity for long-answer questions. GPT-3.5, when paired with RAG, effectively answers binary and long-answer questions, reaffirming its status as an advanced LLM. Regarding open-source LLMs, Mistral AI's Mistral-7b-instruct paired with RAG surpasses the rest in answering both binary and long-answer questions. However, among the open-source LLMs, Orca-mini-v3-7b reports the shortest average latency in generating responses, whereas LLaMa2-7b-chat by Meta reports the highest average latency. This research underscores the fact that open-source LLMs, too, can go hand in hand with proprietary models like GPT-3.5 with better infrastructure. 

---
# Benchmarking the Thinking Mode of Multimodal Large Language Models in Clinical Tasks 

**Authors**: Jindong Hong, Tianjie Chen, Lingjie Luo, Chuanyang Zheng, Ting Xu, Haibao Yu, Jianing Qiu, Qianzhong Chen, Suning Huang, Yan Xu, Yong Gui, Yijun He, Jiankai Sun  

**Link**: [PDF](https://arxiv.org/pdf/2511.03328)  

**Abstract**: A recent advancement in Multimodal Large Language Models (MLLMs) research is the emergence of "reasoning MLLMs" that offer explicit control over their internal thinking processes (normally referred as the "thinking mode") alongside the standard "non-thinking mode". This capability allows these models to engage in a step-by-step process of internal deliberation before generating a final response. With the rapid transition to and adoption of these "dual-state" MLLMs, this work rigorously evaluated how the enhanced reasoning processes of these MLLMs impact model performance and reliability in clinical tasks. This paper evaluates the active "thinking mode" capabilities of two leading MLLMs, Seed1.5-VL and Gemini-2.5-Flash, for medical applications. We assessed their performance on four visual medical tasks using VQA-RAD and ROCOv2 datasets. Our findings reveal that the improvement from activating the thinking mode remains marginal compared to the standard non-thinking mode for the majority of the tasks. Their performance on complex medical tasks such as open-ended VQA and medical image interpretation remains suboptimal, highlighting the need for domain-specific medical data and more advanced methods for medical knowledge integration. 

---
# Hybrid Fact-Checking that Integrates Knowledge Graphs, Large Language Models, and Search-Based Retrieval Agents Improves Interpretable Claim Verification 

**Authors**: Shaghayegh Kolli, Richard Rosenbaum, Timo Cavelius, Lasse Strothe, Andrii Lata, Jana Diesner  

**Link**: [PDF](https://arxiv.org/pdf/2511.03217)  

**Abstract**: Large language models (LLMs) excel in generating fluent utterances but can lack reliable grounding in verified information. At the same time, knowledge-graph-based fact-checkers deliver precise and interpretable evidence, yet suffer from limited coverage or latency. By integrating LLMs with knowledge graphs and real-time search agents, we introduce a hybrid fact-checking approach that leverages the individual strengths of each component. Our system comprises three autonomous steps: 1) a Knowledge Graph (KG) Retrieval for rapid one - hop lookups in DBpedia, 2) an LM-based classification guided by a task-specific labeling prompt, producing outputs with internal rule-based logic, and 3) a Web Search Agent invoked only when KG coverage is insufficient. Our pipeline achieves an F1 score of 0.93 on the FEVER benchmark on the Supported/Refuted split without task- specific fine - tuning. To address Not enough information cases, we conduct a targeted reannotation study showing that our approach frequently uncovers valid evidence for claims originally labeled as Not Enough Information (NEI), as confirmed by both expert annotators and LLM reviewers. With this paper, we present a modular, opensource fact-checking pipeline with fallback strategies and generalization across datasets. 

---
# LGM: Enhancing Large Language Models with Conceptual Meta-Relations and Iterative Retrieval 

**Authors**: Wenchang Lei, Ping Zou, Yue Wang, Feng Sun, Lei Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2511.03214)  

**Abstract**: Large language models (LLMs) exhibit strong semantic understanding, yet struggle when user instructions involve ambiguous or conceptually misaligned terms. We propose the Language Graph Model (LGM) to enhance conceptual clarity by extracting meta-relations-inheritance, alias, and composition-from natural language. The model further employs a reflection mechanism to validate these meta-relations. Leveraging a Concept Iterative Retrieval Algorithm, these relations and related descriptions are dynamically supplied to the LLM, improving its ability to interpret concepts and generate accurate responses. Unlike conventional Retrieval-Augmented Generation (RAG) approaches that rely on extended context windows, our method enables large language models to process texts of any length without the need for truncation. Experiments on standard benchmarks demonstrate that the LGM consistently outperforms existing RAG baselines. 

---
# RefAgent: A Multi-agent LLM-based Framework for Automatic Software Refactoring 

**Authors**: Khouloud Oueslati, Maxime Lamothe, Foutse Khomh  

**Link**: [PDF](https://arxiv.org/pdf/2511.03153)  

**Abstract**: Large Language Models (LLMs) have substantially influenced various software engineering tasks. Indeed, in the case of software refactoring, traditional LLMs have shown the ability to reduce development time and enhance code quality. However, these LLMs often rely on static, detailed instructions for specific tasks. In contrast, LLM-based agents can dynamically adapt to evolving contexts and autonomously make decisions by interacting with software tools and executing workflows. In this paper, we explore the potential of LLM-based agents in supporting refactoring activities. Specifically, we introduce RefAgent, a multi-agent LLM-based framework for end-to-end software refactoring. RefAgent consists of specialized agents responsible for planning, executing, testing, and iteratively refining refactorings using self-reflection and tool-calling capabilities. We evaluate RefAgent on eight open-source Java projects, comparing its effectiveness against a single-agent approach, a search-based refactoring tool, and historical developer refactorings. Our assessment focuses on: (1) the impact of generated refactorings on software quality, (2) the ability to identify refactoring opportunities, and (3) the contribution of each LLM agent through an ablation study. Our results show that RefAgent achieves a median unit test pass rate of 90%, reduces code smells by a median of 52.5%, and improves key quality attributes (e.g., reusability) by a median of 8.6%. Additionally, it closely aligns with developer refactorings and the search-based tool in identifying refactoring opportunities, attaining a median F1-score of 79.15% and 72.7%, respectively. Compared to single-agent approaches, RefAgent improves the median unit test pass rate by 64.7% and the median compilation success rate by 40.1%. These findings highlight the promise of multi-agent architectures in advancing automated software refactoring. 

---
# Who Sees the Risk? Stakeholder Conflicts and Explanatory Policies in LLM-based Risk Assessment 

**Authors**: Srishti Yadav, Jasmina Gajcin, Erik Miehling, Elizabeth Daly  

**Link**: [PDF](https://arxiv.org/pdf/2511.03152)  

**Abstract**: Understanding how different stakeholders perceive risks in AI systems is essential for their responsible deployment. This paper presents a framework for stakeholder-grounded risk assessment by using LLMs, acting as judges to predict and explain risks. Using the Risk Atlas Nexus and GloVE explanation method, our framework generates stakeholder-specific, interpretable policies that shows how different stakeholders agree or disagree about the same risks. We demonstrate our method using three real-world AI use cases of medical AI, autonomous vehicles, and fraud detection domain. We further propose an interactive visualization that reveals how and why conflicts emerge across stakeholder perspectives, enhancing transparency in conflict reasoning. Our results show that stakeholder perspectives significantly influence risk perception and conflict patterns. Our work emphasizes the importance of these stakeholder-aware explanations needed to make LLM-based evaluations more transparent, interpretable, and aligned with human-centered AI governance goals. 

---
# From Measurement to Expertise: Empathetic Expert Adapters for Context-Based Empathy in Conversational AI Agents 

**Authors**: Erfan Shayegani, Jina Suh, Andy Wilson, Nagu Rangan, Javier Hernandez  

**Link**: [PDF](https://arxiv.org/pdf/2511.03143)  

**Abstract**: Empathy is a critical factor in fostering positive user experiences in conversational AI. While models can display empathy, it is often generic rather than tailored to specific tasks and contexts. In this work, we introduce a novel framework for developing and evaluating context-specific empathetic large language models (LLMs). We first analyze a real-world conversational dataset consisting of 672 multi-turn conversations across 8 tasks, revealing significant differences in terms of expected and experienced empathy before and after the conversations, respectively. To help minimize this gap, we develop a synthetic multi-turn conversational generation pipeline and steer responses toward our defined empathy patterns based on the context that more closely matches users' expectations. We then train empathetic expert adapters for context-specific empathy that specialize in varying empathy levels based on the recognized task. Our empirical results demonstrate a significant gap reduction of 72.66% between perceived and desired empathy with scores increasing by an average factor of 2.43 as measured by our metrics and reward models. Additionally, our trained empathetic expert adapters demonstrate superior effectiveness in preserving empathy patterns throughout conversation turns, outperforming system prompts, which tend to dramatically diminish in impact as conversations lengthen. 

---
# QG-CoC: Question-Guided Chain-of-Captions for Large Multimodal Models 

**Authors**: Kuei-Chun Kao, Hsu Tzu-Yin, Yunqi Hong, Ruochen Wang, Cho-Jui Hsieh  

**Link**: [PDF](https://arxiv.org/pdf/2511.03206)  

**Abstract**: Recently, Multimodal Large Language Models (MLLMs) encounter two key issues in multi-image contexts: (1) a lack of fine-grained perception across disparate images, and (2) a diminished capability to effectively reason over and synthesize information from multiple visual inputs. However, while various prompting methods aim to describe visual content, many existing studies focus primarily on single-image settings or specific, constrained scenarios. This leaves a critical gap in understanding and addressing how MLLMs tackle more general and complex multi-image reasoning tasks. Thus, we first extensively investigate how current prompting methods perceive fine-grained visual details and process visual information when dealing with multiple images. Our findings reveal that existing prompting methods fall short in attending to needed clues and seamlessly integrating perception and reasoning. Inspired by the findings, we propose a new zero-shot prompting method, Question-Guided Chain-of-Captions (QG-CoC), a generalized prompting approach that effectively handles problems with an arbitrary number of images. We evaluate our method on various open-source and closed-source MLLMs for multi-image and single-image benchmarks. Experimental results indicate that QG-CoC demonstrates competitive performance across tasks and exhibits robust improvements in the challenging scenarios where existing prompting methods fail. 

---
# Control Barrier Function for Aligning Large Language Models 

**Authors**: Yuya Miyaoka, Masaki Inoue  

**Link**: [PDF](https://arxiv.org/pdf/2511.03121)  

**Abstract**: This paper proposes a control-based framework for aligning large language models (LLMs) by leveraging a control barrier function (CBF) to ensure user-desirable text generation. The presented framework applies the CBF safety filter to the predicted token generated from the baseline LLM, to intervene in the generated text. The safety filter includes two significant advantages: this safety filter is an add-on type, allowing it to be used for alignment purposes without fine-tuning the baseline LLM, and if there is an evaluation model regarding the desired alignment, it can be directly applied to the filter design. The overall text-generation system is implemented with open-source language models, aiming to generate positive text. 

---
# Zero-shot data citation function classification using transformer-based large language models (LLMs) 

**Authors**: Neil Byers, Ali Zaidi, Valerie Skye, Chris Beecroft, Kjiersten Fagnan  

**Link**: [PDF](https://arxiv.org/pdf/2511.02936)  

**Abstract**: Efforts have increased in recent years to identify associations between specific datasets and the scientific literature that incorporates them. Knowing that a given publication cites a given dataset, the next logical step is to explore how or why that data was used. Advances in recent years with pretrained, transformer-based large language models (LLMs) offer potential means for scaling the description of data use cases in the published literature. This avoids expensive manual labeling and the development of training datasets for classical machine-learning (ML) systems. In this work we apply an open-source LLM, Llama 3.1-405B, to generate structured data use case labels for publications known to incorporate specific genomic datasets. We also introduce a novel evaluation framework for determining the efficacy of our methods. Our results demonstrate that the stock model can achieve an F1 score of .674 on a zero-shot data citation classification task with no previously defined categories. While promising, our results are qualified by barriers related to data availability, prompt overfitting, computational infrastructure, and the expense required to conduct responsible performance evaluation. 

---
# FATE: A Formal Benchmark Series for Frontier Algebra of Multiple Difficulty Levels 

**Authors**: Jiedong Jiang, Wanyi He, Yuefeng Wang, Guoxiong Gao, Yongle Hu, Jingting Wang, Nailing Guan, Peihao Wu, Chunbo Dai, Liang Xiao, Bin Dong  

**Link**: [PDF](https://arxiv.org/pdf/2511.02872)  

**Abstract**: Recent advances in large language models (LLMs) have demonstrated impressive capabilities in formal theorem proving, particularly on contest-based mathematical benchmarks like the IMO. However, these contests do not reflect the depth, breadth, and abstraction of modern mathematical research. To bridge this gap, we introduce FATE (Formal Algebra Theorem Evaluation), a new benchmark series in formal algebra designed to chart a course toward advanced mathematical reasoning. We present two new components, FATE-H and FATE-X, each with 100 problems in abstract and commutative algebra. The FATE series spans a difficulty spectrum from undergraduate exercises to problems exceeding PhD qualifying exams. Notably, FATE-X is the first formal benchmark to surpass both PhD-level exam difficulty and the coverage of the Mathlib library. Our evaluations of state-of-the-art LLM provers on this new benchmark reveal a stark performance gap compared to contest math: the best model achieves only 3% (pass@64) accuracy on FATE-H and 0% on FATE-X. Our two-stage evaluation reveals that models' natural-language reasoning is notably more accurate than their ability to formalize this reasoning. We systematically classify the common errors that arise during this formalization process. Furthermore, a comparative study shows that a specialized prover can exhibit less effective reflection than general-purpose models, reducing its accuracy at the natural-language stage. We believe FATE provides a robust and challenging benchmark that establishes essential checkpoints on the path toward research-level formal mathematical reasoning. 

---
# Mathematical exploration and discovery at scale 

**Authors**: Bogdan Georgiev, Javier Gómez-Serrano, Terence Tao, Adam Zsolt Wagner  

**Link**: [PDF](https://arxiv.org/pdf/2511.02864)  

**Abstract**: AlphaEvolve is a generic evolutionary coding agent that combines the generative capabilities of LLMs with automated evaluation in an iterative evolutionary framework that proposes, tests, and refines algorithmic solutions to challenging scientific and practical problems. In this paper we showcase AlphaEvolve as a tool for autonomously discovering novel mathematical constructions and advancing our understanding of long-standing open problems.
To demonstrate its breadth, we considered a list of 67 problems spanning mathematical analysis, combinatorics, geometry, and number theory. The system rediscovered the best known solutions in most of the cases and discovered improved solutions in several. In some instances, AlphaEvolve is also able to generalize results for a finite number of input values into a formula valid for all input values. Furthermore, we are able to combine this methodology with Deep Think and AlphaProof in a broader framework where the additional proof-assistants and reasoning systems provide automated proof generation and further mathematical insights.
These results demonstrate that large language model-guided evolutionary search can autonomously discover mathematical constructions that complement human intuition, at times matching or even improving the best known results, highlighting the potential for significant new ways of interaction between mathematicians and AI systems. We present AlphaEvolve as a powerful new tool for mathematical discovery, capable of exploring vast search spaces to solve complex optimization problems at scale, often with significantly reduced requirements on preparation and computation time. 

---
# SELF-REDRAFT: Eliciting Intrinsic Exploration-Exploitation Balance in Test-Time Scaling for Code Generation 

**Authors**: Yixiang Chen, Tianshi Zheng, Shijue Huang, Zhitao He, Yi R. Fung  

**Link**: [PDF](https://arxiv.org/pdf/2511.02854)  

**Abstract**: Test-time scaling without interpreter feedback is essential for real-world code generation scenarios where test cases are not readily available. While existing paradigms often rely on either greedy exploitation (i.e., iterative refinement) or stochastic exploration (i.e., relying on sample-based voting or reranking mechanisms), the balance between these two dimensions remains underexplored. To investigate the LLM's intrinsic ability to balance exploitation and exploration, we introduce SELF-REDRAFT, a framework built upon Self-Refine that encourages the model to propose new drafts for solutions that are fundamentally flawed. Our results show that SELF-REDRAFT consistently achieves better performance than Self-Refine when converged under the same maximum number of iterations. Still, we observe that significant room for improvement remains, largely due to two core aspects of current self-redraft capabilities: constrained capacity for generating instructive feedback and fragile discriminative judgment. We also find that balancing strategies vary notably across different LLMs, reflecting distinct, model-specific behaviors. Overall, our study establishes a baseline for intrinsic exploration-exploitation balancing in test-time scaling and identifies feedback and discrimination as key areas with potential for future advances. 

---
# LM-Fix: Lightweight Bit-Flip Detection and Rapid Recovery Framework for Language Models 

**Authors**: Ahmad Tahmasivand, Noureldin Zahran, Saba Al-Sayouri, Mohammed Fouda, Khaled N. Khasawneh  

**Link**: [PDF](https://arxiv.org/pdf/2511.02866)  

**Abstract**: This paper presents LM-Fix, a lightweight detection and rapid recovery framework for faults in large language models (LLMs). Existing integrity approaches are often heavy or slow for modern LLMs. LM-Fix runs a short test-vector pass and uses hash-guided checks to detect bit-flip faults, then repairs them locally without a full reload. Across multiple models, it detects over 94% of single-bit flips at TVL=200 and nearly 100% of multi-bit flips with approximately 1% to 7.7% runtime overhead; recovery is more than 100x faster than reloading. These results show a practical, low-overhead solution to keep LLMs reliable in production 

---
# Digital Transformation Chatbot (DTchatbot): Integrating Large Language Model-based Chatbot in Acquiring Digital Transformation Needs 

**Authors**: Jiawei Zheng, Gokcen Yilmaz, Ji Han, Saeema Ahmed-Kristensen  

**Link**: [PDF](https://arxiv.org/pdf/2511.02842)  

**Abstract**: Many organisations pursue digital transformation to enhance operational efficiency, reduce manual efforts, and optimise processes by automation and digital tools. To achieve this, a comprehensive understanding of their unique needs is required. However, traditional methods, such as expert interviews, while effective, face several challenges, including scheduling conflicts, resource constraints, inconsistency, etc. To tackle these issues, we investigate the use of a Large Language Model (LLM)-powered chatbot to acquire organisations' digital transformation needs. Specifically, the chatbot integrates workflow-based instruction with LLM's planning and reasoning capabilities, enabling it to function as a virtual expert and conduct interviews. We detail the chatbot's features and its implementation. Our preliminary evaluation indicates that the chatbot performs as designed, effectively following predefined workflows and supporting user interactions with areas for improvement. We conclude by discussing the implications of employing chatbots to elicit user information, emphasizing their potential and limitations. 

---
# ASVRI-Legal: Fine-Tuning LLMs with Retrieval Augmented Generation for Enhanced Legal Regulation 

**Authors**: One Octadion, Bondan Sapta Prakoso, Nanang Yudi Setiawan, Novanto Yudistira  

**Link**: [PDF](https://arxiv.org/pdf/2511.03563)  

**Abstract**: In this study, we explore the fine-tuning of Large Language Models (LLMs) to better support policymakers in their crucial work of understanding, analyzing, and crafting legal regulations. To equip the model with a deep understanding of legal texts, we curated a supervised dataset tailored to the specific needs of the legal domain. Additionally, we integrated the Retrieval-Augmented Generation (RAG) method, enabling the LLM to access and incorporate up-to-date legal knowledge from external sources. This combination of fine-tuning and RAG-based augmentation results in a tool that not only processes legal information but actively assists policymakers in interpreting regulations and drafting new ones that align with current needs. The results demonstrate that this approach can significantly enhance the effectiveness of legal research and regulation development, offering a valuable resource in the ever-evolving field of law. 

---
# Do Androids Dream of Unseen Puppeteers? Probing for a Conspiracy Mindset in Large Language Models 

**Authors**: Francesco Corso, Francesco Pierri, Gianmarco De Francisci Morales  

**Link**: [PDF](https://arxiv.org/pdf/2511.03699)  

**Abstract**: In this paper, we investigate whether Large Language Models (LLMs) exhibit conspiratorial tendencies, whether they display sociodemographic biases in this domain, and how easily they can be conditioned into adopting conspiratorial perspectives. Conspiracy beliefs play a central role in the spread of misinformation and in shaping distrust toward institutions, making them a critical testbed for evaluating the social fidelity of LLMs. LLMs are increasingly used as proxies for studying human behavior, yet little is known about whether they reproduce higher-order psychological constructs such as a conspiratorial mindset. To bridge this research gap, we administer validated psychometric surveys measuring conspiracy mindset to multiple models under different prompting and conditioning strategies. Our findings reveal that LLMs show partial agreement with elements of conspiracy belief, and conditioning with socio-demographic attributes produces uneven effects, exposing latent demographic biases. Moreover, targeted prompts can easily shift model responses toward conspiratorial directions, underscoring both the susceptibility of LLMs to manipulation and the potential risks of their deployment in sensitive contexts. These results highlight the importance of critically evaluating the psychological dimensions embedded in LLMs, both to advance computational social science and to inform possible mitigation strategies against harmful uses. 

---
# Towards Transparent Stance Detection: A Zero-Shot Approach Using Implicit and Explicit Interpretability 

**Authors**: Apoorva Upadhyaya, Wolfgang Nejdl, Marco Fisichella  

**Link**: [PDF](https://arxiv.org/pdf/2511.03635)  

**Abstract**: Zero-Shot Stance Detection (ZSSD) identifies the attitude of the post toward unseen targets. Existing research using contrastive, meta-learning, or data augmentation suffers from generalizability issues or lack of coherence between text and target. Recent works leveraging large language models (LLMs) for ZSSD focus either on improving unseen target-specific knowledge or generating explanations for stance analysis. However, most of these works are limited by their over-reliance on explicit reasoning, provide coarse explanations that lack nuance, and do not explicitly model the reasoning process, making it difficult to interpret the model's predictions. To address these issues, in our study, we develop a novel interpretable ZSSD framework, IRIS. We provide an interpretable understanding of the attitude of the input towards the target implicitly based on sequences within the text (implicit rationales) and explicitly based on linguistic measures (explicit rationales). IRIS considers stance detection as an information retrieval ranking task, understanding the relevance of implicit rationales for different stances to guide the model towards correct predictions without requiring the ground-truth of rationales, thus providing inherent interpretability. In addition, explicit rationales based on communicative features help decode the emotional and cognitive dimensions of stance, offering an interpretable understanding of the author's attitude towards the given target. Extensive experiments on the benchmark datasets of VAST, EZ-STANCE, P-Stance, and RFD using 50%, 30%, and even 10% training data prove the generalizability of our model, benefiting from the proposed architecture and interpretable design. 

---
# One Battle After Another: Probing LLMs' Limits on Multi-Turn Instruction Following with a Benchmark Evolving Framework 

**Authors**: Qi Jia, Kaiwei Zhang, Xiujie Song, Ye Shen, Xiangyang Zhu, Guangtao Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2511.03508)  

**Abstract**: Understanding how well large language models can follow users' instructions throughout a dialogue spanning multiple topics is of great importance for data-intensive conversational applications. Existing benchmarks are often limited to a fixed number of turns, making them susceptible to saturation and failing to account for the user's interactive experience. In this work, we propose an extensible framework for assessing multi-turn instruction-following ability. At its core, our framework decouples linguistic surface forms from user intent simulation through a three-layer mechanism that tracks constraints, instructions, and topics. This framework mimics User-LLM interaction by enabling the dynamic construction of benchmarks with state changes and tracebacks, terminating a conversation only when the model exhausts a simulated user's patience. We define a suite of metrics capturing the quality of the interaction process. Using this framework, we construct EvolIF, an evolving instruction-following benchmark incorporating nine distinct constraint types. Our results indicate that GPT-5 exhibits superior instruction-following performance. It sustains an average of 18.54 conversational turns and demonstrates 70.31% robustness, outperforming Gemini-2.5-Pro by a significant margin of 11.41%, while other models lag far behind. All of the data and code will be made publicly available online. 

---
# Knowledge-Augmented Question Error Correction for Chinese Question Answer System with QuestionRAG 

**Authors**: Longpeng Qiu, Ting Li, Shuai Mao, Nan Yang, Xiaohui Yan  

**Link**: [PDF](https://arxiv.org/pdf/2511.03410)  

**Abstract**: Input errors in question-answering (QA) systems often lead to incorrect responses. Large language models (LLMs) struggle with this task, frequently failing to interpret user intent (misinterpretation) or unnecessarily altering the original question's structure (over-correction). We propose QuestionRAG, a framework that tackles these problems. To address misinterpretation, it enriches the input with external knowledge (e.g., search results, related entities). To prevent over-correction, it uses reinforcement learning (RL) to align the model's objective with precise correction, not just paraphrasing. Our results demonstrate that knowledge augmentation is critical for understanding faulty questions. Furthermore, RL-based alignment proves significantly more effective than traditional supervised fine-tuning (SFT), boosting the model's ability to follow instructions and generalize. By integrating these two strategies, QuestionRAG unlocks the full potential of LLMs for the question correction task. 

---
# Efficient Reasoning via Thought-Training and Thought-Free Inference 

**Authors**: Canhui Wu, Qiong Cao, Chao Xue, Wei Xi, Xiaodong He  

**Link**: [PDF](https://arxiv.org/pdf/2511.03408)  

**Abstract**: Recent advances in large language models (LLMs) have leveraged explicit Chain-of-Thought (CoT) prompting to improve reasoning accuracy. However, most existing methods primarily compress verbose reasoning outputs. These Long-to-Short transformations aim to improve efficiency, but still rely on explicit reasoning during inference. In this work, we introduce \textbf{3TF} (\textbf{T}hought-\textbf{T}raining and \textbf{T}hought-\textbf{F}ree inference), a framework for efficient reasoning that takes a Short-to-Long perspective. We first train a hybrid model that can operate in both reasoning and non-reasoning modes, and then further train it on CoT-annotated data to internalize structured reasoning, while enforcing concise, thought-free outputs at inference time using the no-reasoning mode. Unlike compression-based approaches, 3TF improves the reasoning quality of non-reasoning outputs, enabling models to perform rich internal reasoning implicitly while keeping external outputs short. Empirically, 3TF-trained models obtain large improvements on reasoning benchmarks under thought-free inference, demonstrating that high quality reasoning can be learned and executed implicitly without explicit step-by-step generation. 

---
# Silenced Biases: The Dark Side LLMs Learned to Refuse 

**Authors**: Rom Himelstein, Amit LeVi, Brit Youngmann, Yaniv Nemcovsky, Avi Mendelson  

**Link**: [PDF](https://arxiv.org/pdf/2511.03369)  

**Abstract**: Safety-aligned large language models (LLMs) are becoming increasingly widespread, especially in sensitive applications where fairness is essential and biased outputs can cause significant harm. However, evaluating the fairness of models is a complex challenge, and approaches that do so typically utilize standard question-answer (QA) styled schemes. Such methods often overlook deeper issues by interpreting the model's refusal responses as positive fairness measurements, which creates a false sense of fairness. In this work, we introduce the concept of silenced biases, which are unfair preferences encoded within models' latent space and are effectively concealed by safety-alignment. Previous approaches that considered similar indirect biases often relied on prompt manipulation or handcrafted implicit queries, which present limited scalability and risk contaminating the evaluation process with additional biases. We propose the Silenced Bias Benchmark (SBB), which aims to uncover these biases by employing activation steering to reduce model refusals during QA. SBB supports easy expansion to new demographic groups and subjects, presenting a fairness evaluation framework that encourages the future development of fair models and tools beyond the masking effects of alignment training. We demonstrate our approach over multiple LLMs, where our findings expose an alarming distinction between models' direct responses and their underlying fairness issues. 

---
# LFC-DA: Logical Formula-Controlled Data Augmentation for Enhanced Logical Reasoning 

**Authors**: Shenghao Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.03372)  

**Abstract**: For complex logical data augmentation, heavy reliance on human annotation is costly, whereas direct generation with large language models yields uninterpretable and logically homogeneous examples. To address this, we present LFC-DA, a symbolic-logic-controlled pipeline: logical text is first mapped to propositional expressions, a compact rule library is compiled, and a bounded state-space search systematically discovers valid formulas that are then verbalized back into natural-language questions, ensuring both diversity and logical rigor under propositional logic. Experiments on ReClor and LogiQA show significant improvements in the logical-reasoning accuracy of pretrained models, confirming the effectiveness of LFC-DA for LLM-guided logical data augmentation. 

---
# Measuring Aleatoric and Epistemic Uncertainty in LLMs: Empirical Evaluation on ID and OOD QA Tasks 

**Authors**: Kevin Wang, Subre Abdoul Moktar, Jia Li, Kangshuo Li, Feng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2511.03166)  

**Abstract**: Large Language Models (LLMs) have become increasingly pervasive, finding applications across many industries and disciplines. Ensuring the trustworthiness of LLM outputs is paramount, where Uncertainty Estimation (UE) plays a key role. In this work, a comprehensive empirical study is conducted to examine the robustness and effectiveness of diverse UE measures regarding aleatoric and epistemic uncertainty in LLMs. It involves twelve different UE methods and four generation quality metrics including LLMScore from LLM criticizers to evaluate the uncertainty of LLM-generated answers in Question-Answering (QA) tasks on both in-distribution (ID) and out-of-distribution (OOD) datasets. Our analysis reveals that information-based methods, which leverage token and sequence probabilities, perform exceptionally well in ID settings due to their alignment with the model's understanding of the data. Conversely, density-based methods and the P(True) metric exhibit superior performance in OOD contexts, highlighting their effectiveness in capturing the model's epistemic uncertainty. Semantic consistency methods, which assess variability in generated answers, show reliable performance across different datasets and generation metrics. These methods generally perform well but may not be optimal for every situation. 

---
# EQ-Negotiator: Dynamic Emotional Personas Empower Small Language Models for Edge-Deployable Credit Negotiation 

**Authors**: Yunbo Long, Yuhan Liu, Alexandra Brintrup  

**Link**: [PDF](https://arxiv.org/pdf/2511.03370)  

**Abstract**: The deployment of large language models (LLMs) in automated negotiation has set a high performance benchmark, but their computational cost and data privacy requirements render them unsuitable for many privacy-sensitive, on-device applications such as mobile assistants, embodied AI agents or private client interactions. While small language models (SLMs) offer a practical alternative, they suffer from a significant performance gap compared to LLMs in playing emotionally charged complex personas, especially for credit negotiation. This paper introduces EQ-Negotiator, a novel framework that bridges this capability gap using emotional personas. Its core is a reasoning system that integrates game theory with a Hidden Markov Model(HMM) to learn and track debtor emotional states online, without pre-training. This allows EQ-Negotiator to equip SLMs with the strategic intelligence to counter manipulation while de-escalating conflict and upholding ethical standards. Through extensive agent-to-agent simulations across diverse credit negotiation scenarios, including adversarial debtor strategies like cheating, threatening, and playing the victim, we show that a 7B parameter language model with EQ-Negotiator achieves better debt recovery and negotiation efficiency than baseline LLMs more than 10 times its size. This work advances persona modeling from descriptive character profiles to dynamic emotional architectures that operate within privacy constraints. Besides, this paper establishes that strategic emotional intelligence, not raw model scale, is the critical factor for success in automated negotiation, paving the way for effective, ethical, and privacy-preserving AI negotiators that can operate on the edge. 

---
# PolyNorm: Few-Shot LLM-Based Text Normalization for Text-to-Speech 

**Authors**: Michel Wong, Ali Alshehri, Sophia Kao, Haotian He  

**Link**: [PDF](https://arxiv.org/pdf/2511.03080)  

**Abstract**: Text Normalization (TN) is a key preprocessing step in Text-to-Speech (TTS) systems, converting written forms into their canonical spoken equivalents. Traditional TN systems can exhibit high accuracy, but involve substantial engineering effort, are difficult to scale, and pose challenges to language coverage, particularly in low-resource settings. We propose PolyNorm, a prompt-based approach to TN using Large Language Models (LLMs), aiming to reduce the reliance on manually crafted rules and enable broader linguistic applicability with minimal human intervention. Additionally, we present a language-agnostic pipeline for automatic data curation and evaluation, designed to facilitate scalable experimentation across diverse languages. Experiments across eight languages show consistent reductions in the word error rate (WER) compared to a production-grade-based system. To support further research, we release PolyNorm-Benchmark, a multilingual data set covering a diverse range of text normalization phenomena. 

---
# ROBoto2: An Interactive System and Dataset for LLM-assisted Clinical Trial Risk of Bias Assessment 

**Authors**: Anthony Hevia, Sanjana Chintalapati, Veronica Ka Wai Lai, Thanh Tam Nguyen, Wai-Tat Wong, Terry Klassen, Lucy Lu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.03048)  

**Abstract**: We present ROBOTO2, an open-source, web-based platform for large language model (LLM)-assisted risk of bias (ROB) assessment of clinical trials. ROBOTO2 streamlines the traditionally labor-intensive ROB v2 (ROB2) annotation process via an interactive interface that combines PDF parsing, retrieval-augmented LLM prompting, and human-in-the-loop review. Users can upload clinical trial reports, receive preliminary answers and supporting evidence for ROB2 signaling questions, and provide real-time feedback or corrections to system suggestions. ROBOTO2 is publicly available at this https URL, with code and data released to foster reproducibility and adoption. We construct and release a dataset of 521 pediatric clinical trial reports (8954 signaling questions with 1202 evidence passages), annotated using both manually and LLM-assisted methods, serving as a benchmark and enabling future research. Using this dataset, we benchmark ROB2 performance for 4 LLMs and provide an analysis into current model capabilities and ongoing challenges in automating this critical aspect of systematic review. 

---
# BengaliMoralBench: A Benchmark for Auditing Moral Reasoning in Large Language Models within Bengali Language and Culture 

**Authors**: Shahriyar Zaman Ridoy, Azmine Toushik Wasi, Koushik Ahamed Tonmoy  

**Link**: [PDF](https://arxiv.org/pdf/2511.03180)  

**Abstract**: As multilingual Large Language Models (LLMs) gain traction across South Asia, their alignment with local ethical norms, particularly for Bengali, which is spoken by over 285 million people and ranked 6th globally, remains underexplored. Existing ethics benchmarks are largely English-centric and shaped by Western frameworks, overlooking cultural nuances critical for real-world deployment. To address this, we introduce BengaliMoralBench, the first large-scale ethics benchmark for the Bengali language and socio-cultural contexts. It covers five moral domains, Daily Activities, Habits, Parenting, Family Relationships, and Religious Activities, subdivided into 50 culturally relevant subtopics. Each scenario is annotated via native-speaker consensus using three ethical lenses: Virtue, Commonsense, and Justice ethics. We conduct systematic zero-shot evaluation of prominent multilingual LLMs, including Llama, Gemma, Qwen, and DeepSeek, using a unified prompting protocol and standard metrics. Performance varies widely (50-91% accuracy), with qualitative analysis revealing consistent weaknesses in cultural grounding, commonsense reasoning, and moral fairness. BengaliMoralBench provides a foundation for responsible localization, enabling culturally aligned evaluation and supporting the deployment of ethically robust AI in diverse, low-resource multilingual settings such as Bangladesh. 

---
# Watermarking Large Language Models in Europe: Interpreting the AI Act in Light of Technology 

**Authors**: Thomas Souverain  

**Link**: [PDF](https://arxiv.org/pdf/2511.03641)  

**Abstract**: To foster trustworthy Artificial Intelligence (AI) within the European Union, the AI Act requires providers to mark and detect the outputs of their general-purpose models. The Article 50 and Recital 133 call for marking methods that are ''sufficiently reliable, interoperable, effective and robust''. Yet, the rapidly evolving and heterogeneous landscape of watermarks for Large Language Models (LLMs) makes it difficult to determine how these four standards can be translated into concrete and measurable evaluations. Our paper addresses this challenge, anchoring the normativity of European requirements in the multiplicity of watermarking techniques. Introducing clear and distinct concepts on LLM watermarking, our contribution is threefold. (1) Watermarking Categorisation: We propose an accessible taxonomy of watermarking methods according to the stage of the LLM lifecycle at which they are applied - before, during, or after training, and during next-token distribution or sampling. (2) Watermarking Evaluation: We interpret the EU AI Act's requirements by mapping each criterion with state-of-the-art evaluations on robustness and detectability of the watermark, and of quality of the LLM. Since interoperability remains largely untheorised in LLM watermarking research, we propose three normative dimensions to frame its assessment. (3) Watermarking Comparison: We compare current watermarking methods for LLMs against the operationalised European criteria and show that no approach yet satisfies all four standards. Encouraged by emerging empirical tests, we recommend further research into watermarking directly embedded within the low-level architecture of LLMs. 

---
# From Insight to Exploit: Leveraging LLM Collaboration for Adaptive Adversarial Text Generation 

**Authors**: Najrin Sultana, Md Rafi Ur Rashid, Kang Gu, Shagufta Mehnaz  

**Link**: [PDF](https://arxiv.org/pdf/2511.03128)  

**Abstract**: LLMs can provide substantial zero-shot performance on diverse tasks using a simple task prompt, eliminating the need for training or fine-tuning. However, when applying these models to sensitive tasks, it is crucial to thoroughly assess their robustness against adversarial inputs. In this work, we introduce Static Deceptor (StaDec) and Dynamic Deceptor (DyDec), two innovative attack frameworks designed to systematically generate dynamic and adaptive adversarial examples by leveraging the understanding of the LLMs. We produce subtle and natural-looking adversarial inputs that preserve semantic similarity to the original text while effectively deceiving the target LLM. By utilizing an automated, LLM-driven pipeline, we eliminate the dependence on external heuristics. Our attacks evolve with the advancements in LLMs and demonstrate strong transferability across models unknown to the attacker. Overall, this work provides a systematic approach for the self-assessment of an LLM's robustness. We release our code and data at this https URL. 

---
# Targeted Error Correction in Knowledge Distillation: Small Language Models Surpass GPT 

**Authors**: Hee-Jin Lee, Zhen Guo, Luchao Jin, Morteza Moazami Goudarzi  

**Link**: [PDF](https://arxiv.org/pdf/2511.03005)  

**Abstract**: We introduce an Analyze-Revise-Finetune (ARF) pipeline that enables smaller open-source language models (LLMs) to surpass substantially larger proprietary models in customer service summarization tasks. The pipeline first analyzes and categorizes common errors in summaries produced by a teacher model (GPT-3.5), then performs a targeted revision using a compact editor model (Llama 3.1 70B) to generate high-quality, refined training data. Fine-tuning a smaller student model (Llama 3.1 8B) on this refined data resulted in superior summarization performance compared to GPT-3.5. The ARF pipeline improves cost efficiency and data privacy while maintaining competitive accuracy, illustrating a generalizable framework for enhancing open-source LLMs across diverse downstream applications. 

---
