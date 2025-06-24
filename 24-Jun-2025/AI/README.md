# jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval 

**Authors**: Michael Günther, Saba Sturua, Mohammad Kalim Akram, Isabelle Mohr, Andrei Ungureanu, Sedigheh Eslami, Scott Martens, Bo Wang, Nan Wang, Han Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2506.18902)  

**Abstract**: We introduce jina-embeddings-v4, a 3.8 billion parameter multimodal embedding model that unifies text and image representations through a novel architecture supporting both single-vector and multi-vector embeddings in the late interaction style. The model incorporates task-specific Low-Rank Adaptation (LoRA) adapters to optimize performance across diverse retrieval scenarios, including query-based information retrieval, cross-modal semantic similarity, and programming code search. Comprehensive evaluations demonstrate that jina-embeddings-v4 achieves state-of-the-art performance on both single- modal and cross-modal retrieval tasks, with particular strength in processing visually rich content such as tables, charts, diagrams, and mixed-media formats. To facilitate evaluation of this capability, we also introduce Jina-VDR, a novel benchmark specifically designed for visually rich image retrieval. 

---
# Steering Conceptual Bias via Transformer Latent-Subspace Activation 

**Authors**: Vansh Sharma, Venkat Raman  

**Link**: [PDF](https://arxiv.org/pdf/2506.18887)  

**Abstract**: This work examines whether activating latent subspaces in language models (LLMs) can steer scientific code generation toward a specific programming language. Five causal LLMs were first evaluated on scientific coding prompts to quantify their baseline bias among four programming languages. A static neuron-attribution method, perturbing the highest activated MLP weight for a C++ or CPP token, proved brittle and exhibited limited generalization across prompt styles and model scales. To address these limitations, a gradient-refined adaptive activation steering framework (G-ACT) was developed: per-prompt activation differences are clustered into a small set of steering directions, and lightweight per-layer probes are trained and refined online to select the appropriate steering vector. In LLaMA-3.2 3B, this approach reliably biases generation towards the CPP language by increasing the average probe classification accuracy by 15% and the early layers (0-6) improving the probe classification accuracy by 61.5% compared to the standard ACT framework. For LLaMA-3.3 70B, where attention-head signals become more diffuse, targeted injections at key layers still improve language selection. Although per-layer probing introduces a modest inference overhead, it remains practical by steering only a subset of layers and enables reproducible model behavior. These results demonstrate a scalable, interpretable and efficient mechanism for concept-level control for practical agentic systems. 

---
# ConciseHint: Boosting Efficient Reasoning via Continuous Concise Hints during Generation 

**Authors**: Siao Tang, Xinyin Ma, Gongfan Fang, Xinchao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.18810)  

**Abstract**: Recent advancements in large reasoning models (LRMs) like DeepSeek-R1 and OpenAI o1 series have achieved notable performance enhancements on complex reasoning tasks by scaling up the generation length by Chain-of-Thought (CoT). However, an emerging issue is their inclination to produce excessively verbose reasoning processes, leading to the inefficiency problem. Existing literature on improving efficiency mainly adheres to the before-reasoning paradigms such as prompting and reasoning or fine-tuning and reasoning, but ignores the promising direction of directly encouraging the model to speak concisely by intervening during the generation of reasoning. In order to fill the blank, we propose a framework dubbed ConciseHint, which continuously encourages the reasoning model to speak concisely by injecting the textual hint (manually designed or trained on the concise data) during the token generation of the reasoning process. Besides, ConciseHint is adaptive to the complexity of the query by adaptively adjusting the hint intensity, which ensures it will not undermine model performance. Experiments on the state-of-the-art LRMs, including DeepSeek-R1 and Qwen-3 series, demonstrate that our method can effectively produce concise reasoning processes while maintaining performance well. For instance, we achieve a reduction ratio of 65\% for the reasoning length on GSM8K benchmark with Qwen-3 4B with nearly no accuracy loss. 

---
# TRIZ Agents: A Multi-Agent LLM Approach for TRIZ-Based Innovation 

**Authors**: Kamil Szczepanik, Jarosław A. Chudziak  

**Link**: [PDF](https://arxiv.org/pdf/2506.18783)  

**Abstract**: TRIZ, the Theory of Inventive Problem Solving, is a structured, knowledge-based framework for innovation and abstracting problems to find inventive solutions. However, its application is often limited by the complexity and deep interdisciplinary knowledge required. Advancements in Large Language Models (LLMs) have revealed new possibilities for automating parts of this process. While previous studies have explored single LLMs in TRIZ applications, this paper introduces a multi-agent approach. We propose an LLM-based multi-agent system, called TRIZ agents, each with specialized capabilities and tool access, collaboratively solving inventive problems based on the TRIZ methodology. This multi-agent system leverages agents with various domain expertise to efficiently navigate TRIZ steps. The aim is to model and simulate an inventive process with language agents. We assess the effectiveness of this team of agents in addressing complex innovation challenges based on a selected case study in engineering. We demonstrate the potential of agent collaboration to produce diverse, inventive solutions. This research contributes to the future of AI-driven innovation, showcasing the advantages of decentralized problem-solving in complex ideation tasks. 

---
# Programming by Backprop: LLMs Acquire Reusable Algorithmic Abstractions During Code Training 

**Authors**: Jonathan Cook, Silvia Sapora, Arash Ahmadian, Akbir Khan, Tim Rocktaschel, Jakob Foerster, Laura Ruis  

**Link**: [PDF](https://arxiv.org/pdf/2506.18777)  

**Abstract**: Training large language models (LLMs) on source code significantly enhances their general-purpose reasoning abilities, but the mechanisms underlying this generalisation are poorly understood. In this paper, we propose Programming by Backprop (PBB) as a potential driver of this effect - teaching a model to evaluate a program for inputs by training on its source code alone, without ever seeing I/O examples. To explore this idea, we finetune LLMs on two sets of programs representing simple maths problems and algorithms: one with source code and I/O examples (w/ IO), the other with source code only (w/o IO). We find evidence that LLMs have some ability to evaluate w/o IO programs for inputs in a range of experimental settings, and make several observations. Firstly, PBB works significantly better when programs are provided as code rather than semantically equivalent language descriptions. Secondly, LLMs can produce outputs for w/o IO programs directly, by implicitly evaluating the program within the forward pass, and more reliably when stepping through the program in-context via chain-of-thought. We further show that PBB leads to more robust evaluation of programs across inputs than training on I/O pairs drawn from a distribution that mirrors naturally occurring data. Our findings suggest a mechanism for enhanced reasoning through code training: it allows LLMs to internalise reusable algorithmic abstractions. Significant scope remains for future work to enable LLMs to more effectively learn from symbolic procedures, and progress in this direction opens other avenues like model alignment by training on formal constitutional principles. 

---
# Dual-level Behavioral Consistency for Inter-group and Intra-group Coordination in Multi-Agent Systems 

**Authors**: Shuocun Yang, Huawen Hu, Enze Shi, Shu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.18651)  

**Abstract**: Behavioral diversity in Multi-agent reinforcement learning(MARL) represents an emerging and promising research area. Prior work has largely centered on intra-group behavioral consistency in multi-agent systems, with limited attention given to behavioral consistency in multi-agent grouping scenarios. In this paper, we introduce Dual-Level Behavioral Consistency (DLBC), a novel MARL control method designed to explicitly regulate agent behaviors at both intra-group and inter-group levels. DLBC partitions agents into distinct groups and dynamically modulates behavioral diversity both within and between these groups. By dynamically modulating behavioral diversity within and between these groups, DLBC achieves enhanced division of labor through inter-group consistency, which constrains behavioral strategies across different groups. Simultaneously, intra-group consistency, achieved by aligning behavioral strategies within each group, fosters stronger intra-group cooperation. Crucially, DLBC's direct constraint of agent policy functions ensures its broad applicability across various algorithmic frameworks. Experimental results in various grouping cooperation scenarios demonstrate that DLBC significantly enhances both intra-group cooperative performance and inter-group task specialization, yielding substantial performance improvements. DLBC provides new ideas for behavioral consistency control of multi-intelligent body systems, and its potential for application in more complex tasks and dynamic environments can be further explored in the future. 

---
# AggTruth: Contextual Hallucination Detection using Aggregated Attention Scores in LLMs 

**Authors**: Piotr Matys, Jan Eliasz, Konrad Kiełczyński, Mikołaj Langner, Teddy Ferdinan, Jan Kocoń, Przemysław Kazienko  

**Link**: [PDF](https://arxiv.org/pdf/2506.18628)  

**Abstract**: In real-world applications, Large Language Models (LLMs) often hallucinate, even in Retrieval-Augmented Generation (RAG) settings, which poses a significant challenge to their deployment. In this paper, we introduce AggTruth, a method for online detection of contextual hallucinations by analyzing the distribution of internal attention scores in the provided context (passage). Specifically, we propose four different variants of the method, each varying in the aggregation technique used to calculate attention scores. Across all LLMs examined, AggTruth demonstrated stable performance in both same-task and cross-task setups, outperforming the current SOTA in multiple scenarios. Furthermore, we conducted an in-depth analysis of feature selection techniques and examined how the number of selected attention heads impacts detection performance, demonstrating that careful selection of heads is essential to achieve optimal results. 

---
# Airalogy: AI-empowered universal data digitization for research automation 

**Authors**: Zijie Yang, Qiji Zhou, Fang Guo, Sijie Zhang, Yexun Xi, Jinglei Nie, Yudian Zhu, Liping Huang, Chou Wu, Yonghe Xia, Xiaoyu Ma, Yingming Pu, Panzhong Lu, Junshu Pan, Mingtao Chen, Tiannan Guo, Yanmei Dou, Hongyu Chen, Anping Zeng, Jiaxing Huang, Tian Xu, Yue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.18586)  

**Abstract**: Research data are the foundation of Artificial Intelligence (AI)-driven science, yet current AI applications remain limited to a few fields with readily available, well-structured, digitized datasets. Achieving comprehensive AI empowerment across multiple disciplines is still out of reach. Present-day research data collection is often fragmented, lacking unified standards, inefficiently managed, and difficult to share. Creating a single platform for standardized data digitization needs to overcome the inherent challenge of balancing between universality (supporting the diverse, ever-evolving needs of various disciplines) and standardization (enforcing consistent formats to fully enable AI). No existing platform accommodates both facets. Building a truly multidisciplinary platform requires integrating scientific domain knowledge with sophisticated computing skills. Researchers often lack the computational expertise to design customized and standardized data recording methods, whereas platform developers rarely grasp the intricate needs of multiple scientific domains. These gaps impede research data standardization and hamper AI-driven progress. In this study, we address these challenges by developing Airalogy (this https URL), the world's first AI- and community-driven platform that balances universality and standardization for digitizing research data across multiple disciplines. Airalogy represents entire research workflows using customizable, standardized data records and offers an advanced AI research copilot for intelligent Q&A, automated data entry, analysis, and research automation. Already deployed in laboratories across all four schools of Westlake University, Airalogy has the potential to accelerate and automate scientific innovation in universities, industry, and the global research community-ultimately benefiting humanity as a whole. 

---
# T-CPDL: A Temporal Causal Probabilistic Description Logic for Developing Logic-RAG Agent 

**Authors**: Hong Qing Yu  

**Link**: [PDF](https://arxiv.org/pdf/2506.18559)  

**Abstract**: Large language models excel at generating fluent text but frequently struggle with structured reasoning involving temporal constraints, causal relationships, and probabilistic reasoning. To address these limitations, we propose Temporal Causal Probabilistic Description Logic (T-CPDL), an integrated framework that extends traditional Description Logic with temporal interval operators, explicit causal relationships, and probabilistic annotations. We present two distinct variants of T-CPDL: one capturing qualitative temporal relationships through Allen's interval algebra, and another variant enriched with explicit timestamped causal assertions. Both variants share a unified logical structure, enabling complex reasoning tasks ranging from simple temporal ordering to nuanced probabilistic causation. Empirical evaluations on temporal reasoning and causal inference benchmarks confirm that T-CPDL substantially improves inference accuracy, interpretability, and confidence calibration of language model outputs. By delivering transparent reasoning paths and fine-grained temporal and causal semantics, T-CPDL significantly enhances the capability of language models to support robust, explainable, and trustworthy decision-making. This work also lays the groundwork for developing advanced Logic-Retrieval-Augmented Generation (Logic-RAG) frameworks, potentially boosting the reasoning capabilities and efficiency of knowledge graph-enhanced RAG systems. 

---
# A Question Bank to Assess AI Inclusivity: Mapping out the Journey from Diversity Errors to Inclusion Excellence 

**Authors**: Rifat Ara Shams, Didar Zowghi, Muneera Bano  

**Link**: [PDF](https://arxiv.org/pdf/2506.18538)  

**Abstract**: Ensuring diversity and inclusion (D&I) in artificial intelligence (AI) is crucial for mitigating biases and promoting equitable decision-making. However, existing AI risk assessment frameworks often overlook inclusivity, lacking standardized tools to measure an AI system's alignment with D&I principles. This paper introduces a structured AI inclusivity question bank, a comprehensive set of 253 questions designed to evaluate AI inclusivity across five pillars: Humans, Data, Process, System, and Governance. The development of the question bank involved an iterative, multi-source approach, incorporating insights from literature reviews, D&I guidelines, Responsible AI frameworks, and a simulated user study. The simulated evaluation, conducted with 70 AI-generated personas related to different AI jobs, assessed the question bank's relevance and effectiveness for AI inclusivity across diverse roles and application domains. The findings highlight the importance of integrating D&I principles into AI development workflows and governance structures. The question bank provides an actionable tool for researchers, practitioners, and policymakers to systematically assess and enhance the inclusivity of AI systems, paving the way for more equitable and responsible AI technologies. 

---
# Standard Applicability Judgment and Cross-jurisdictional Reasoning: A RAG-based Framework for Medical Device Compliance 

**Authors**: Yu Han, Aaron Ceross, Jeroen H.M. Bergmann  

**Link**: [PDF](https://arxiv.org/pdf/2506.18511)  

**Abstract**: Identifying the appropriate regulatory standard applicability remains a critical yet understudied challenge in medical device compliance, frequently necessitating expert interpretation of fragmented and heterogeneous documentation across different jurisdictions. To address this challenge, we introduce a modular AI system that leverages a retrieval-augmented generation (RAG) pipeline to automate standard applicability determination. Given a free-text device description, our system retrieves candidate standards from a curated corpus and uses large language models to infer jurisdiction-specific applicability, classified as Mandatory, Recommended, or Not Applicable, with traceable justifications. We construct an international benchmark dataset of medical device descriptions with expert-annotated standard mappings, and evaluate our system against retrieval-only, zero-shot, and rule-based baselines. The proposed approach attains a classification accuracy of 73% and a Top-5 retrieval recall of 87%, demonstrating its effectiveness in identifying relevant regulatory standards. We introduce the first end-to-end system for standard applicability reasoning, enabling scalable and interpretable AI-supported regulatory science. Notably, our region-aware RAG agent performs cross-jurisdictional reasoning between Chinese and U.S. standards, supporting conflict resolution and applicability justification across regulatory frameworks. 

---
# How Robust is Model Editing after Fine-Tuning? An Empirical Study on Text-to-Image Diffusion Models 

**Authors**: Feng He, Zhenyang Liu, Marco Valentino, Zhixue Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.18428)  

**Abstract**: Model editing offers a low-cost technique to inject or correct a particular behavior in a pre-trained model without extensive retraining, supporting applications such as factual correction and bias mitigation. Despite this common practice, it remains unknown whether edits persist after fine-tuning or whether they are inadvertently reversed. This question has fundamental practical implications. For example, if fine-tuning removes prior edits, it could serve as a defence mechanism against hidden malicious edits. Vice versa, the unintended removal of edits related to bias mitigation could pose serious safety concerns. We systematically investigate the interaction between model editing and fine-tuning in the context of T2I diffusion models, which are known to exhibit biases and generate inappropriate content. Our study spans two T2I model families (Stable Diffusion and FLUX), two sota editing techniques, and three fine-tuning methods (DreamBooth, LoRA, and DoRA). Through an extensive empirical analysis across diverse editing tasks and evaluation metrics, our findings reveal a trend: edits generally fail to persist through fine-tuning, even when fine-tuning is tangential or unrelated to the edits. Notably, we observe that DoRA exhibits the strongest edit reversal effect. At the same time, among editing methods, UCE demonstrates greater robustness, retaining significantly higher efficacy post-fine-tuning compared to ReFACT. These findings highlight a crucial limitation in current editing methodologies, emphasizing the need for more robust techniques to ensure reliable long-term control and alignment of deployed AI systems. These findings have dual implications for AI safety: they suggest that fine-tuning could serve as a remediation mechanism for malicious edits while simultaneously highlighting the need for re-editing after fine-tuning to maintain beneficial safety and alignment properties. 

---
# A Large Language Model-based Multi-Agent Framework for Analog Circuits' Sizing Relationships Extraction 

**Authors**: Chengjie Liu, Weiyu Chen, Huiyao Xu, Yuan Du, Jun Yang, Li Du  

**Link**: [PDF](https://arxiv.org/pdf/2506.18424)  

**Abstract**: In the design process of the analog circuit pre-layout phase, device sizing is an important step in determining whether an analog circuit can meet the required performance metrics. Many existing techniques extract the circuit sizing task as a mathematical optimization problem to solve and continuously improve the optimization efficiency from a mathematical perspective. But they ignore the automatic introduction of prior knowledge, fail to achieve effective pruning of the search space, which thereby leads to a considerable compression margin remaining in the search space. To alleviate this problem, we propose a large language model (LLM)-based multi-agent framework for analog circuits' sizing relationships extraction from academic papers. The search space in the sizing process can be effectively pruned based on the sizing relationship extracted by this framework. Eventually, we conducted tests on 3 types of circuits, and the optimization efficiency was improved by $2.32 \sim 26.6 \times$. This work demonstrates that the LLM can effectively prune the search space for analog circuit sizing, providing a new solution for the combination of LLMs and conventional analog circuit design automation methods. 

---
# Dynamic Knowledge Exchange and Dual-diversity Review: Concisely Unleashing the Potential of a Multi-Agent Research Team 

**Authors**: Weilun Yu, Shixiang Tang, Yonggui Huang, Nanqing Dong, Li Fan, Honggang Qi, Wei Liu, Xiaoli Diao, Xi Chen, Wanli Ouyang  

**Link**: [PDF](https://arxiv.org/pdf/2506.18348)  

**Abstract**: Scientific progress increasingly relies on effective collaboration among researchers, a dynamic that large language models (LLMs) have only begun to emulate. While recent LLM-based scientist agents show promise in autonomous scientific discovery, they often lack the interactive reasoning and evaluation mechanisms essential to real-world research. We propose IDVSCI (Internal Discussion and Vote SCIentists), a multi-agent framework built on LLMs that incorporates two key innovations: a Dynamic Knowledge Exchange mechanism enabling iterative feedback among agents, and a Dual-Diversity Review paradigm that simulates heterogeneous expert evaluation. These components jointly promote deeper reasoning and the generation of more creative and impactful scientific ideas. To evaluate the effectiveness and generalizability of our approach, we conduct experiments on two datasets: a widely used benchmark in computer science and a new dataset we introduce in the health sciences domain. Results show that IDVSCI consistently achieves the best performance across both datasets, outperforming existing systems such as AI Scientist and VIRSCI. These findings highlight the value of modeling interaction and peer review dynamics in LLM-based autonomous research. 

---
# Advanced For-Loop for QML algorithm search 

**Authors**: FuTe Wong  

**Link**: [PDF](https://arxiv.org/pdf/2506.18260)  

**Abstract**: This paper introduces an advanced framework leveraging Large Language Model-based Multi-Agent Systems (LLMMA) for the automated search and optimization of Quantum Machine Learning (QML) algorithms. Inspired by Google DeepMind's FunSearch, the proposed system works on abstract level to iteratively generates and refines quantum transformations of classical machine learning algorithms (concepts), such as the Multi-Layer Perceptron, forward-forward and backpropagation algorithms. As a proof of concept, this work highlights the potential of agentic frameworks to systematically explore classical machine learning concepts and adapt them for quantum computing, paving the way for efficient and automated development of QML algorithms. Future directions include incorporating planning mechanisms and optimizing strategy in the search space for broader applications in quantum-enhanced machine learning. 

---
# The 4th Dimension for Scaling Model Size 

**Authors**: Ruike Zhu, Hanwen Zhang, Tianyu Shi, Chi Wang, Tianyi Zhou, Zengyi Qin  

**Link**: [PDF](https://arxiv.org/pdf/2506.18233)  

**Abstract**: Scaling the size of large language models typically involves three dimensions: depth, width, and the number of parameters. In this work, we explore a fourth dimension, virtual logical depth (VLD), which increases the effective algorithmic depth without changing the overall parameter count by reusing parameters within the model. Although parameter reuse is not a new concept, its potential and characteristics in model scaling have not been thoroughly studied. Through carefully designed controlled experiments, we make the following key discoveries regarding VLD scaling:
VLD scaling forces the knowledge capacity of the model to remain almost constant, with only minor variations.
VLD scaling enables a significant improvement in reasoning capability, provided the scaling method is properly implemented.
The number of parameters correlates with knowledge capacity, but not with reasoning capability. Under certain conditions, it is not necessary to increase the parameter count to enhance reasoning.
These findings are consistent across various model configurations and are likely to be generally valid within the scope of our experiments. 

---
# A Conceptual Framework for AI Capability Evaluations 

**Authors**: María Victoria Carro, Denise Alejandra Mester, Francisca Gauna Selasco, Luca Nicolás Forziati Gangi, Matheo Sandleris Musa, Lola Ramos Pereyra, Mario Leiva, Juan Gustavo Corvalan, María Vanina Martinez, Gerardo Simari  

**Link**: [PDF](https://arxiv.org/pdf/2506.18213)  

**Abstract**: As AI systems advance and integrate into society, well-designed and transparent evaluations are becoming essential tools in AI governance, informing decisions by providing evidence about system capabilities and risks. Yet there remains a lack of clarity on how to perform these assessments both comprehensively and reliably. To address this gap, we propose a conceptual framework for analyzing AI capability evaluations, offering a structured, descriptive approach that systematizes the analysis of widely used methods and terminology without imposing new taxonomies or rigid formats. This framework supports transparency, comparability, and interpretability across diverse evaluations. It also enables researchers to identify methodological weaknesses, assists practitioners in designing evaluations, and provides policymakers with an accessible tool to scrutinize, compare, and navigate complex evaluation landscapes. 

---
# The Impact of Medication Non-adherence on Adverse Outcomes: Evidence from Schizophrenia Patients via Survival Analysis 

**Authors**: Shahriar Noroozizadeh, Pim Welle, Jeremy C. Weiss, George H. Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.18187)  

**Abstract**: This study quantifies the association between non-adherence to antipsychotic medications and adverse outcomes in individuals with schizophrenia. We frame the problem using survival analysis, focusing on the time to the earliest of several adverse events (early death, involuntary hospitalization, jail booking). We extend standard causal inference methods (T-learner, S-learner, nearest neighbor matching) to utilize various survival models to estimate individual and average treatment effects, where treatment corresponds to medication non-adherence. Analyses are repeated using different amounts of longitudinal information (3, 6, 9, and 12 months). Using data from Allegheny County in western Pennsylvania, we find strong evidence that non-adherence advances adverse outcomes by approximately 1 to 4 months. Ablation studies confirm that county-provided risk scores adjust for key confounders, as their removal amplifies the estimated effects. Subgroup analyses by medication formulation (injectable vs. oral) and medication type consistently show that non-adherence is associated with earlier adverse events. These findings highlight the clinical importance of adherence in delaying psychiatric crises and show that integrating survival analysis with causal inference tools can yield policy-relevant insights. We caution that although we apply causal inference, we only make associative claims and discuss assumptions needed for causal interpretation. 

---
# Reasoning about Uncertainty: Do Reasoning Models Know When They Don't Know? 

**Authors**: Zhiting Mei, Christina Zhang, Tenny Yin, Justin Lidard, Ola Shorinwa, Anirudha Majumdar  

**Link**: [PDF](https://arxiv.org/pdf/2506.18183)  

**Abstract**: Reasoning language models have set state-of-the-art (SOTA) records on many challenging benchmarks, enabled by multi-step reasoning induced using reinforcement learning. However, like previous language models, reasoning models are prone to generating confident, plausible responses that are incorrect (hallucinations). Knowing when and how much to trust these models is critical to the safe deployment of reasoning models in real-world applications. To this end, we explore uncertainty quantification of reasoning models in this work. Specifically, we ask three fundamental questions: First, are reasoning models well-calibrated? Second, does deeper reasoning improve model calibration? Finally, inspired by humans' innate ability to double-check their thought processes to verify the validity of their answers and their confidence, we ask: can reasoning models improve their calibration by explicitly reasoning about their chain-of-thought traces? We introduce introspective uncertainty quantification (UQ) to explore this direction. In extensive evaluations on SOTA reasoning models across a broad range of benchmarks, we find that reasoning models: (i) are typically overconfident, with self-verbalized confidence estimates often greater than 85% particularly for incorrect responses, (ii) become even more overconfident with deeper reasoning, and (iii) can become better calibrated through introspection (e.g., o3-Mini and DeepSeek R1) but not uniformly (e.g., Claude 3.7 Sonnet becomes more poorly calibrated). Lastly, we conclude with important research directions to design necessary UQ benchmarks and improve the calibration of reasoning models. 

---
# Chain-of-Memory: Enhancing GUI Agents for Cross-Application Navigation 

**Authors**: Xinzge Gao, Chuanrui Hu, Bin Chen, Teng Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.18158)  

**Abstract**: Multimodal large language models (MLLMs) are attracting growing attention in the development of Graphical User Interface (GUI) agents. Existing approaches often rely on historical screenshots or actions to implicitly represent the task state. This reliance poses challenges for GUI agents in accurately understanding task states and underscores the absence of effective mechanisms to store critical information in complex and lengthy cross-app tasks. To address these challenges, we propose Chain-of-Memory (CoM), a novel approach for explicitly modeling short-term and long-term memory in GUI agents. CoM achieves this by capturing action descriptions, integrating task-relevant screen information, and maintaining a dedicated memory module to store and manage this information. By leveraging explicit memory representations, CoM enables GUI agents to better understand task states and retain critical historical information persistently. To equip GUI agents with memory management capabilities and evaluate the effectiveness of CoM, we developed the GUI Odyssey-CoM, a dataset comprising 111k screen-action pairs annotated with Chain-of-Memory. Experimental results demonstrate that CoM significantly improves GUI agents' performance in cross-application tasks. Additionally, GUI Odyssey-CoM enables 7B models to achieve memory management capabilities comparable to 72B models. The dataset and code will be open-sourced. 

---
# AI Through the Human Lens: Investigating Cognitive Theories in Machine Psychology 

**Authors**: Akash Kundu, Rishika Goswami  

**Link**: [PDF](https://arxiv.org/pdf/2506.18156)  

**Abstract**: We investigate whether Large Language Models (LLMs) exhibit human-like cognitive patterns under four established frameworks from psychology: Thematic Apperception Test (TAT), Framing Bias, Moral Foundations Theory (MFT), and Cognitive Dissonance. We evaluated several proprietary and open-source models using structured prompts and automated scoring. Our findings reveal that these models often produce coherent narratives, show susceptibility to positive framing, exhibit moral judgments aligned with Liberty/Oppression concerns, and demonstrate self-contradictions tempered by extensive rationalization. Such behaviors mirror human cognitive tendencies yet are shaped by their training data and alignment methods. We discuss the implications for AI transparency, ethical deployment, and future work that bridges cognitive psychology and AI safety 

---
# CoachGPT: A Scaffolding-based Academic Writing Assistant 

**Authors**: Fumian Chen, Sotheara Veng, Joshua Wilson, Xiaoming Li, Hui Fang  

**Link**: [PDF](https://arxiv.org/pdf/2506.18149)  

**Abstract**: Academic writing skills are crucial for students' success, but can feel overwhelming without proper guidance and practice, particularly when writing in a second language. Traditionally, students ask instructors or search dictionaries, which are not universally accessible. Early writing assistants emerged as rule-based systems that focused on detecting misspellings, subject-verb disagreements, and basic punctuation errors; however, they are inaccurate and lack contextual understanding. Machine learning-based assistants demonstrate a strong ability for language understanding but are expensive to train. Large language models (LLMs) have shown remarkable capabilities in generating responses in natural languages based on given prompts. Still, they have a fundamental limitation in education: they generate essays without teaching, which can have detrimental effects on learning when misused. To address this limitation, we develop CoachGPT, which leverages large language models (LLMs) to assist individuals with limited educational resources and those who prefer self-paced learning in academic writing. CoachGPT is an AI agent-based web application that (1) takes instructions from experienced educators, (2) converts instructions into sub-tasks, and (3) provides real-time feedback and suggestions using large language models. This unique scaffolding structure makes CoachGPT unique among existing writing assistants. Compared to existing writing assistants, CoachGPT provides a more immersive writing experience with personalized feedback and guidance. Our user studies prove the usefulness of CoachGPT and the potential of large language models for academic writing. 

---
# SE-Merging: A Self-Enhanced Approach for Dynamic Model Merging 

**Authors**: Zijun Chen, Zhanpeng Zhou, Bo Zhang, Weinan Zhang, Xi Sun, Junchi Yan  

**Link**: [PDF](https://arxiv.org/pdf/2506.18135)  

**Abstract**: Model merging has gained increasing attention due to its intriguing property: interpolating the parameters of different task-specific fine-tuned models leads to multi-task abilities. However, despite its empirical success, the underlying mechanisms of model merging remain poorly understood. In this work, we delve into the mechanism behind model merging from a representation perspective. Our analysis reveals that model merging achieves multi-task abilities through two key capabilities: i) distinguishing samples from different tasks, and ii) adapting to the corresponding expert model for each sample. These two capabilities allow the merged model to retain task-specific expertise, enabling efficient multi-task adaptation. Building on these insights, we propose \texttt{SE-Merging}, a self-enhanced model merging framework that leverages these two characteristics to dynamically identify the corresponding task for each sample and then adaptively rescales the merging coefficients to further enhance task-specific expertise in the merged model. Notably, \texttt{SE-Merging} achieves dynamic model merging without additional training. Extensive experiments demonstrate that \texttt{SE-Merging} achieves significant performance improvements while remaining compatible with existing model merging techniques. 

---
# Decentralized Consensus Inference-based Hierarchical Reinforcement Learning for Multi-Constrained UAV Pursuit-Evasion Game 

**Authors**: Xiang Yuming, Li Sizhao, Li Rongpeng, Zhao Zhifeng, Zhang Honggang  

**Link**: [PDF](https://arxiv.org/pdf/2506.18126)  

**Abstract**: Multiple quadrotor unmanned aerial vehicle (UAV) systems have garnered widespread research interest and fostered tremendous interesting applications, especially in multi-constrained pursuit-evasion games (MC-PEG). The Cooperative Evasion and Formation Coverage (CEFC) task, where the UAV swarm aims to maximize formation coverage across multiple target zones while collaboratively evading predators, belongs to one of the most challenging issues in MC-PEG, especially under communication-limited constraints. This multifaceted problem, which intertwines responses to obstacles, adversaries, target zones, and formation dynamics, brings up significant high-dimensional complications in locating a solution. In this paper, we propose a novel two-level framework (i.e., Consensus Inference-based Hierarchical Reinforcement Learning (CI-HRL)), which delegates target localization to a high-level policy, while adopting a low-level policy to manage obstacle avoidance, navigation, and formation. Specifically, in the high-level policy, we develop a novel multi-agent reinforcement learning module, Consensus-oriented Multi-Agent Communication (ConsMAC), to enable agents to perceive global information and establish consensus from local states by effectively aggregating neighbor messages. Meanwhile, we leverage an Alternative Training-based Multi-agent proximal policy optimization (AT-M) and policy distillation to accomplish the low-level control. The experimental results, including the high-fidelity software-in-the-loop (SITL) simulations, validate that CI-HRL provides a superior solution with enhanced swarm's collaborative evasion and task completion capabilities. 

---
# Deep Research Agents: A Systematic Examination And Roadmap 

**Authors**: Yuxuan Huang, Yihang Chen, Haozheng Zhang, Kang Li, Meng Fang, Linyi Yang, Xiaoguang Li, Lifeng Shang, Songcen Xu, Jianye Hao, Kun Shao, Jun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.18096)  

**Abstract**: The rapid progress of Large Language Models (LLMs) has given rise to a new category of autonomous AI systems, referred to as Deep Research (DR) agents. These agents are designed to tackle complex, multi-turn informational research tasks by leveraging a combination of dynamic reasoning, adaptive long-horizon planning, multi-hop information retrieval, iterative tool use, and the generation of structured analytical reports. In this paper, we conduct a detailed analysis of the foundational technologies and architectural components that constitute Deep Research agents. We begin by reviewing information acquisition strategies, contrasting API-based retrieval methods with browser-based exploration. We then examine modular tool-use frameworks, including code execution, multimodal input processing, and the integration of Model Context Protocols (MCPs) to support extensibility and ecosystem development. To systematize existing approaches, we propose a taxonomy that differentiates between static and dynamic workflows, and we classify agent architectures based on planning strategies and agent composition, including single-agent and multi-agent configurations. We also provide a critical evaluation of current benchmarks, highlighting key limitations such as restricted access to external knowledge, sequential execution inefficiencies, and misalignment between evaluation metrics and the practical objectives of DR agents. Finally, we outline open challenges and promising directions for future research. A curated and continuously updated repository of DR agent research is available at: {this https URL}. 

---
# Weighted Assumption Based Argumentation to reason about ethical principles and actions 

**Authors**: Paolo Baldi, Fabio Aurelio D'Asaro, Abeer Dyoub, Francesca Alessandra Lisi  

**Link**: [PDF](https://arxiv.org/pdf/2506.18056)  

**Abstract**: We augment Assumption Based Argumentation (ABA for short) with weighted argumentation. In a nutshell, we assign weights to arguments and then derive the weight of attacks between ABA arguments. We illustrate our proposal through running examples in the field of ethical reasoning, and present an implementation based on Answer Set Programming. 

---
# Action Language BC+ 

**Authors**: Joseph Babb, Joohyung Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.18044)  

**Abstract**: Action languages are formal models of parts of natural language that are designed to describe effects of actions. Many of these languages can be viewed as high level notations of answer set programs structured to represent transition systems. However, the form of answer set programs considered in the earlier work is quite limited in comparison with the modern Answer Set Programming (ASP) language, which allows several useful constructs for knowledge representation, such as choice rules, aggregates, and abstract constraint atoms. We propose a new action language called BC+, which closes the gap between action languages and the modern ASP language. The main idea is to define the semantics of BC+ in terms of general stable model semantics for propositional formulas, under which many modern ASP language constructs can be identified with shorthands for propositional formulas. Language BC+ turns out to be sufficiently expressive to encompass the best features of other action languages, such as languages B, C, C+, and BC. Computational methods available in ASP solvers are readily applicable to compute BC+, which led to an implementation of the language by extending system cplus2asp. 

---
# Graphs Meet AI Agents: Taxonomy, Progress, and Future Opportunities 

**Authors**: Yuanchen Bei, Weizhi Zhang, Siwen Wang, Weizhi Chen, Sheng Zhou, Hao Chen, Yong Li, Jiajun Bu, Shirui Pan, Yizhou Yu, Irwin King, Fakhri Karray, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2506.18019)  

**Abstract**: AI agents have experienced a paradigm shift, from early dominance by reinforcement learning (RL) to the rise of agents powered by large language models (LLMs), and now further advancing towards a synergistic fusion of RL and LLM capabilities. This progression has endowed AI agents with increasingly strong abilities. Despite these advances, to accomplish complex real-world tasks, agents are required to plan and execute effectively, maintain reliable memory, and coordinate smoothly with other agents. Achieving these capabilities involves contending with ever-present intricate information, operations, and interactions. In light of this challenge, data structurization can play a promising role by transforming intricate and disorganized data into well-structured forms that agents can more effectively understand and process. In this context, graphs, with their natural advantage in organizing, managing, and harnessing intricate data relationships, present a powerful data paradigm for structurization to support the capabilities demanded by advanced AI agents. To this end, this survey presents a first systematic review of how graphs can empower AI agents. Specifically, we explore the integration of graph techniques with core agent functionalities, highlight notable applications, and identify prospective avenues for future research. By comprehensively surveying this burgeoning intersection, we hope to inspire the development of next-generation AI agents equipped to tackle increasingly sophisticated challenges with graphs. Related resources are collected and continuously updated for the community in the Github link. 

---
# medicX-KG: A Knowledge Graph for Pharmacists' Drug Information Needs 

**Authors**: Lizzy Farrugia, Lilian M. Azzopardi, Jeremy Debattista, Charlie Abela  

**Link**: [PDF](https://arxiv.org/pdf/2506.17959)  

**Abstract**: The role of pharmacists is evolving from medicine dispensing to delivering comprehensive pharmaceutical services within multidisciplinary healthcare teams. Central to this shift is access to accurate, up-to-date medicinal product information supported by robust data integration. Leveraging artificial intelligence and semantic technologies, Knowledge Graphs (KGs) uncover hidden relationships and enable data-driven decision-making. This paper presents medicX-KG, a pharmacist-oriented knowledge graph supporting clinical and regulatory decisions. It forms the semantic layer of the broader medicX platform, powering predictive and explainable pharmacy services. medicX-KG integrates data from three sources, including, the British National Formulary (BNF), DrugBank, and the Malta Medicines Authority (MMA) that addresses Malta's regulatory landscape and combines European Medicines Agency alignment with partial UK supply dependence. The KG tackles the absence of a unified national drug repository, reducing pharmacists' reliance on fragmented sources. Its design was informed by interviews with practicing pharmacists to ensure real-world applicability. We detail the KG's construction, including data extraction, ontology design, and semantic mapping. Evaluation demonstrates that medicX-KG effectively supports queries about drug availability, interactions, adverse reactions, and therapeutic classes. Limitations, including missing detailed dosage encoding and real-time updates, are discussed alongside directions for future enhancements. 

---
# Evolving Prompts In-Context: An Open-ended, Self-replicating Perspective 

**Authors**: Jianyu Wang, Zhiqiang Hu, Lidong Bing  

**Link**: [PDF](https://arxiv.org/pdf/2506.17930)  

**Abstract**: We propose a novel prompt design paradigm that challenges conventional wisdom in large language model (LLM) prompting. While conventional wisdom prioritizes well-crafted instructions and demonstrations for in-context learning (ICL), we show that pruning random demonstrations into seemingly incoherent "gibberish" can remarkably improve performance across diverse tasks. Notably, the "gibberish" always matches or surpasses state-of-the-art automatic prompt optimization techniques, achieving substantial gains regardless of LLM alignment. Nevertheless, discovering an effective pruning strategy is non-trivial, as existing attribution methods and prompt compression algorithms fail to deliver robust results, let alone human intuition. In terms of this, we propose a self-discover prompt optimization framework, PromptQuine, an evolutionary search framework that automatically searches for the pruning strategy by itself using only low-data regimes. Much like the emergent complexity in nature--such as symbiosis and self-organization--arising in response to resource constraints, our framework evolves and refines unconventional yet highly effective prompts by leveraging only the tokens present within the context. We demonstrate its effectiveness across classification, multi-choice question answering, generation and math reasoning tasks across LLMs, while achieving decent runtime efficiency. We hope our findings can guide mechanistic studies on in-context learning, and provide a call to action, to pave the way for more open-ended search algorithms for more effective LLM prompting. 

---
# Learning, Reasoning, Refinement: A Framework for Kahneman's Dual-System Intelligence in GUI Agents 

**Authors**: Jinjie Wei, Jiyao Liu, Lihao Liu, Ming Hu, Junzhi Ning, Mingcheng Li, Weijie Yin, Junjun He, Xiao Liang, Chao Feng, Dingkang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.17913)  

**Abstract**: Graphical User Interface (GUI) agents have made significant progress in automating digital tasks through the utilization of computer vision and language models. Nevertheless, existing agent systems encounter notable limitations. Firstly, they predominantly depend on trial and error decision making rather than progressive reasoning, thereby lacking the capability to learn and adapt from interactive encounters. Secondly, these systems are assessed using overly simplistic single step accuracy metrics, which do not adequately reflect the intricate nature of real world GUI interactions. In this paper, we present CogniGUI, a cognitive framework developed to overcome these limitations by enabling adaptive learning for GUI automation resembling human-like behavior. Inspired by Kahneman's Dual Process Theory, our approach combines two main components: (1) an omni parser engine that conducts immediate hierarchical parsing of GUI elements through quick visual semantic analysis to identify actionable components, and (2) a Group based Relative Policy Optimization (GRPO) grounding agent that assesses multiple interaction paths using a unique relative reward system, promoting minimal and efficient operational routes. This dual-system design facilitates iterative ''exploration learning mastery'' cycles, enabling the agent to enhance its strategies over time based on accumulated experience. Moreover, to assess the generalization and adaptability of agent systems, we introduce ScreenSeek, a comprehensive benchmark that includes multi application navigation, dynamic state transitions, and cross interface coherence, which are often overlooked challenges in current benchmarks. Experimental results demonstrate that CogniGUI surpasses state-of-the-art methods in both the current GUI grounding benchmarks and our newly proposed benchmark. 

---
# Leveraging Large Language Model for Intelligent Log Processing and Autonomous Debugging in Cloud AI Platforms 

**Authors**: Cheng Ji, Huaiying Luo  

**Link**: [PDF](https://arxiv.org/pdf/2506.17900)  

**Abstract**: With the increasing complexity and rapid expansion of the scale of AI systems in cloud platforms, the log data generated during system operation is massive, unstructured, and semantically ambiguous, which brings great challenges to fault location and system self-repair. In order to solve this problem, this paper proposes an intelligent log processing and automatic debugging framework based on Large Language Model (LLM), named Intelligent Debugger (LLM-ID). This method is extended on the basis of the existing pre-trained Transformer model, and integrates a multi-stage semantic inference mechanism to realize the context understanding of system logs and the automatic reconstruction of fault chains. Firstly, the system log is dynamically structured, and the unsupervised clustering and embedding mechanism is used to extract the event template and semantic schema. Subsequently, the fine-tuned LLM combined with the multi-round attention mechanism to perform contextual reasoning on the log sequence to generate potential fault assumptions and root cause paths. Furthermore, this paper introduces a reinforcement learning-based policy-guided recovery planner, which is driven by the remediation strategy generated by LLM to support dynamic decision-making and adaptive debugging in the cloud environment. Compared with the existing rule engine or traditional log analysis system, the proposed model has stronger semantic understanding ability, continuous learning ability and heterogeneous environment adaptability. Experiments on the cloud platform log dataset show that LLM-ID improves the fault location accuracy by 16.2%, which is significantly better than the current mainstream methods 

---
# Towards Robust Fact-Checking: A Multi-Agent System with Advanced Evidence Retrieval 

**Authors**: Tam Trinh, Manh Nguyen, Truong-Son Hy  

**Link**: [PDF](https://arxiv.org/pdf/2506.17878)  

**Abstract**: The rapid spread of misinformation in the digital era poses significant challenges to public discourse, necessitating robust and scalable fact-checking solutions. Traditional human-led fact-checking methods, while credible, struggle with the volume and velocity of online content, prompting the integration of automated systems powered by Large Language Models (LLMs). However, existing automated approaches often face limitations, such as handling complex claims, ensuring source credibility, and maintaining transparency. This paper proposes a novel multi-agent system for automated fact-checking that enhances accuracy, efficiency, and explainability. The system comprises four specialized agents: an Input Ingestion Agent for claim decomposition, a Query Generation Agent for formulating targeted subqueries, an Evidence Retrieval Agent for sourcing credible evidence, and a Verdict Prediction Agent for synthesizing veracity judgments with human-interpretable explanations. Evaluated on benchmark datasets (FEVEROUS, HOVER, SciFact), the proposed system achieves a 12.3% improvement in Macro F1-score over baseline methods. The system effectively decomposes complex claims, retrieves reliable evidence from trusted sources, and generates transparent explanations for verification decisions. Our approach contributes to the growing field of automated fact-checking by providing a more accurate, efficient, and transparent verification methodology that aligns with human fact-checking practices while maintaining scalability for real-world applications. Our source code is available at this https URL 

---
# Out of Control -- Why Alignment Needs Formal Control Theory (and an Alignment Control Stack) 

**Authors**: Elija Perrier  

**Link**: [PDF](https://arxiv.org/pdf/2506.17846)  

**Abstract**: This position paper argues that formal optimal control theory should be central to AI alignment research, offering a distinct perspective from prevailing AI safety and security approaches. While recent work in AI safety and mechanistic interpretability has advanced formal methods for alignment, they often fall short of the generalisation required of control frameworks for other technologies. There is also a lack of research into how to render different alignment/control protocols interoperable. We argue that by recasting alignment through principles of formal optimal control and framing alignment in terms of hierarchical stack from physical to socio-technical layers according to which controls may be applied we can develop a better understanding of the potential and limitations for controlling frontier models and agentic AI systems. To this end, we introduce an Alignment Control Stack which sets out a hierarchical layered alignment stack, identifying measurement and control characteristics at each layer and how different layers are formally interoperable. We argue that such analysis is also key to the assurances that will be needed by governments and regulators in order to see AI technologies sustainably benefit the community. Our position is that doing so will bridge the well-established and empirically validated methods of optimal control with practical deployment considerations to create a more comprehensive alignment framework, enhancing how we approach safety and reliability for advanced AI systems. 

---
# Reflective Verbal Reward Design for Pluralistic Alignment 

**Authors**: Carter Blair, Kate Larson, Edith Law  

**Link**: [PDF](https://arxiv.org/pdf/2506.17834)  

**Abstract**: AI agents are commonly aligned with "human values" through reinforcement learning from human feedback (RLHF), where a single reward model is learned from aggregated human feedback and used to align an agent's behavior. However, human values are not homogeneous--different people hold distinct and sometimes conflicting values. Aggregating feedback into a single reward model risks disproportionately suppressing minority preferences. To address this, we present a novel reward modeling approach for learning individualized reward models. Our approach uses a language model to guide users through reflective dialogues where they critique agent behavior and construct their preferences. This personalized dialogue history, containing the user's reflections and critiqued examples, is then used as context for another language model that serves as an individualized reward function (what we call a "verbal reward model") for evaluating new trajectories. In studies with 30 participants, our method achieved a 9-12% improvement in accuracy over non-reflective verbal reward models while being more sample efficient than traditional supervised learning methods. 

---
# Efficient Strategy Synthesis for MDPs via Hierarchical Block Decomposition 

**Authors**: Alexandros Evangelidis, Gricel Vázquez, Simos Gerasimou  

**Link**: [PDF](https://arxiv.org/pdf/2506.17792)  

**Abstract**: Software-intensive systems, such as software product lines and robotics, utilise Markov decision processes (MDPs) to capture uncertainty and analyse sequential decision-making problems. Despite the usefulness of conventional policy synthesis methods, they fail to scale to large state spaces. Our approach addresses this issue and accelerates policy synthesis in large MDPs by dynamically refining the MDP and iteratively selecting the most fragile MDP regions for refinement. This iterative procedure offers a balance between accuracy and efficiency, as refinement occurs only when necessary. Through a comprehensive empirical evaluation comprising diverse case studies and MDPs up to 1M states, we demonstrate significant performance improvements yielded by our approach compared to the leading probabilistic model checker PRISM (up to 2x), thus offering a very competitive solution for real-world policy synthesis tasks in larger MDPs. 

---
# Bayesian Social Deduction with Graph-Informed Language Models 

**Authors**: Shahab Rahimirad, Guven Gergerli, Lucia Romero, Angela Qian, Matthew Lyle Olson, Simon Stepputtis, Joseph Campbell  

**Link**: [PDF](https://arxiv.org/pdf/2506.17788)  

**Abstract**: Social reasoning - inferring unobservable beliefs and intentions from partial observations of other agents - remains a challenging task for large language models (LLMs). We evaluate the limits of current reasoning language models in the social deduction game Avalon and find that while the largest models demonstrate strong performance, they require extensive test-time inference and degrade sharply when distilled to smaller, real-time-capable variants. To address this, we introduce a hybrid reasoning framework that externalizes belief inference to a structured probabilistic model, while using an LLM for language understanding and interaction. Our approach achieves competitive performance with much larger models in Agent-Agent play and, notably, is the first language agent to defeat human players in a controlled study - achieving a 67% win rate and receiving higher qualitative ratings than both reasoning baselines and human teammates. We release code, models, and a dataset to support future work on social reasoning in LLM agents, which can be found at this https URL 

---
# AnyMAC: Cascading Flexible Multi-Agent Collaboration via Next-Agent Prediction 

**Authors**: Song Wang, Zhen Tan, Zihan Chen, Shuang Zhou, Tianlong Chen, Jundong Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.17784)  

**Abstract**: Recent progress in large language model (LLM)-based multi-agent collaboration highlights the power of structured communication in enabling collective intelligence. However, existing methods largely rely on static or graph-based inter-agent topologies, lacking the potential adaptability and flexibility in communication. In this work, we propose a new framework that rethinks multi-agent coordination through a sequential structure rather than a graph structure, offering a significantly larger topology space for multi-agent communication. Our method focuses on two key directions: (1) Next-Agent Prediction, which selects the most suitable agent role at each step, and (2) Next-Context Selection (NCS), which enables each agent to selectively access relevant information from any previous step. Together, these components construct task-adaptive communication pipelines that support both role flexibility and global information flow. Extensive evaluations across multiple benchmarks demonstrate that our approach achieves superior performance while substantially reducing communication overhead. 

---
# Beyond Syntax: Action Semantics Learning for App Agents 

**Authors**: Bohan Tang, Dezhao Luo, Jingxuan Chen, Shaogang Gong, Jianye Hao, Jun Wang, Kun Shao  

**Link**: [PDF](https://arxiv.org/pdf/2506.17697)  

**Abstract**: The advent of Large Language Models (LLMs) enables the rise of App agents that interpret user intent and operate smartphone Apps through actions such as clicking and scrolling. While prompt-based solutions with closed LLM APIs show promising ability, they incur heavy compute costs and external API dependency. Fine-tuning smaller open-source LLMs solves these limitations. However, current fine-tuning methods use a syntax learning paradigm that forces agents to reproduce exactly the ground truth action strings, leading to out-of-distribution (OOD) vulnerability. To fill this gap, we propose Action Semantics Learning (ASL), a novel learning framework, where the learning objective is capturing the semantics of the ground truth actions. Specifically, inspired by the programming language theory, we define the action semantics for App agents as the state transition induced by the action in the user interface. With this insight, ASL employs a novel SEmantic Estimator (SEE) to compute a semantic reward to train the App agents in generating actions aligned with the semantics of ground truth actions, even when the syntactic forms differ. To support the effectiveness of ASL, we theoretically demonstrate the superior robustness of ASL for the OOD problem compared with the existing syntax learning paradigm. Extensive experiments on offline and online smartphone App operation benchmarks show that ASL significantly improves the accuracy and generalisation of App agents over existing methods. 

---
# PhysUniBench: An Undergraduate-Level Physics Reasoning Benchmark for Multimodal Models 

**Authors**: Lintao Wang, Encheng Su, Jiaqi Liu, Pengze Li, Peng Xia, Jiabei Xiao, Wenlong Zhang, Xinnan Dai, Xi Chen, Yuan Meng, Mingyu Ding, Lei Bai, Wanli Ouyang, Shixiang Tang, Aoran Wang, Xinzhu Ma  

**Link**: [PDF](https://arxiv.org/pdf/2506.17667)  

**Abstract**: Physics problem-solving is a challenging domain for large AI models, requiring integration of conceptual understanding, mathematical reasoning, and interpretation of physical diagrams. Current evaluation methodologies show notable limitations in capturing the breadth and complexity of undergraduate-level physics, underscoring the need for more rigorous assessments. To this end, we present PhysUniBench, a large-scale multimodal benchmark designed to evaluate and improve the reasoning capabilities of multimodal large language models (MLLMs) specifically on undergraduate-level physics problems. PhysUniBench consists of 3,304 physics questions spanning 8 major sub-disciplines of physics, each accompanied by one visual diagrams. The benchmark includes both open-ended and multiple-choice questions, systematically curated and difficulty-rated through an iterative model-in-the-loop process. The benchmark's construction involved a rigorous multi-stage process, including multiple roll-outs, expert-level evaluation, automated filtering of easily solved problems, and a nuanced difficulty grading system with five levels. Through extensive experiments, we observe that current state-of-the-art models encounter substantial challenges in physics reasoning. For example, GPT-4o mini achieves only about 34.2\% accuracy in the proposed PhysUniBench. These results highlight that current MLLMs struggle with advanced physics reasoning, especially on multi-step problems and those requiring precise diagram interpretation. By providing a broad and rigorous assessment tool, PhysUniBench aims to drive progress in AI for Science, encouraging the development of models with stronger physical reasoning, problem-solving skills, and multimodal understanding. The benchmark and evaluation scripts are available at this https URL. 

---
# Measuring and Augmenting Large Language Models for Solving Capture-the-Flag Challenges 

**Authors**: Zimo Ji, Daoyuan Wu, Wenyuan Jiang, Pingchuan Ma, Zongjie Li, Shuai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.17644)  

**Abstract**: Capture-the-Flag (CTF) competitions are crucial for cybersecurity education and training. As large language models (LLMs) evolve, there is increasing interest in their ability to automate CTF challenge solving. For example, DARPA has organized the AIxCC competition since 2023 to advance AI-powered automated offense and defense. However, this demands a combination of multiple abilities, from knowledge to reasoning and further to actions. In this paper, we highlight the importance of technical knowledge in solving CTF problems and deliberately construct a focused benchmark, CTFKnow, with 3,992 questions to measure LLMs' performance in this core aspect. Our study offers a focused and innovative measurement of LLMs' capability in understanding CTF knowledge and applying it to solve CTF challenges. Our key findings reveal that while LLMs possess substantial technical knowledge, they falter in accurately applying this knowledge to specific scenarios and adapting their strategies based on feedback from the CTF environment.
Based on insights derived from this measurement study, we propose CTFAgent, a novel LLM-driven framework for advancing CTF problem-solving. CTFAgent introduces two new modules: two-stage Retrieval Augmented Generation (RAG) and interactive Environmental Augmentation, which enhance LLMs' technical knowledge and vulnerability exploitation on CTF, respectively. Our experimental results show that, on two popular CTF datasets, CTFAgent both achieves over 80% performance improvement. Moreover, in the recent picoCTF2024 hosted by CMU, CTFAgent ranked in the top 23.6% of nearly 7,000 participating teams. This reflects the benefit of our measurement study and the potential of our framework in advancing LLMs' capabilities in CTF problem-solving. 

---
# Taming the Untamed: Graph-Based Knowledge Retrieval and Reasoning for MLLMs to Conquer the Unknown 

**Authors**: Bowen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.17589)  

**Abstract**: The real value of knowledge lies not just in its accumulation, but in its potential to be harnessed effectively to conquer the unknown. Although recent multimodal large language models (MLLMs) exhibit impressing multimodal capabilities, they often fail in rarely encountered domain-specific tasks due to limited relevant knowledge. To explore this, we adopt visual game cognition as a testbed and select Monster Hunter: World as the target to construct a multimodal knowledge graph (MH-MMKG), which incorporates multi-modalities and intricate entity relations. We also design a series of challenging queries based on MH-MMKG to evaluate the models' ability for complex knowledge retrieval and reasoning. Furthermore, we propose a multi-agent retriever that enables a model to autonomously search relevant knowledge without additional training. Experimental results show that our approach significantly enhances the performance of MLLMs, providing a new perspective on multimodal knowledge-augmented reasoning and laying a solid foundation for future research. 

---
# Cite Pretrain: Retrieval-Free Knowledge Attribution for Large Language Models 

**Authors**: Yukun Huang, Sanxing Chen, Jian Pei, Manzil Zaheer, Bhuwan Dhingra  

**Link**: [PDF](https://arxiv.org/pdf/2506.17585)  

**Abstract**: Trustworthy language models should provide both correct and verifiable answers. While language models can sometimes attribute their outputs to pretraining data, their citations are often unreliable due to hallucination. As a result, current systems insert citations by querying an external retriever at inference time, introducing latency, infrastructure dependence, and vulnerability to retrieval noise. We explore whether LLMs can be made to reliably attribute to the documents seen during (continual) pretraining--without test-time retrieval--by revising the training process. To evaluate this, we release CitePretrainBench, a benchmark that mixes real-world corpora (Wikipedia, Common Crawl, arXiv) with novel, unseen documents and probes both short-form (single fact) and long-form (multi-fact) citation tasks. Our approach follows a two-stage process: (1) continual pretraining to bind facts to persistent document identifiers, and (2) instruction tuning to elicit citation behavior. We find that simple Passive Indexing, which appends an identifier to each document, helps memorize verbatim text but fails on paraphrased or compositional facts. Instead, we propose Active Indexing, which continually pretrains on synthetic QA pairs that (1) restate each fact in diverse compositional forms, and (2) require bidirectional source-to-fact and fact-to-source generation, jointly teaching the model to generate content from a cited source and to attribute its own answers. Experiments with Qwen2.5-7B and 3B show that Active Indexing consistently outperforms Passive Indexing across all tasks and models, with citation precision gains up to 30.2 percent. Our ablation studies reveal that performance continues to improve as we scale the amount of augmented data, showing a clear upward trend even at 16 times the original token count. 

---
# Kaleidoscopic Teaming in Multi Agent Simulations 

**Authors**: Ninareh Mehrabi, Tharindu Kumarage, Kai-Wei Chang, Aram Galstyan, Rahul Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2506.17514)  

**Abstract**: Warning: This paper contains content that may be inappropriate or offensive.
AI agents have gained significant recent attention due to their autonomous tool usage capabilities and their integration in various real-world applications. This autonomy poses novel challenges for the safety of such systems, both in single- and multi-agent scenarios. We argue that existing red teaming or safety evaluation frameworks fall short in evaluating safety risks in complex behaviors, thought processes and actions taken by agents. Moreover, they fail to consider risks in multi-agent setups where various vulnerabilities can be exposed when agents engage in complex behaviors and interactions with each other. To address this shortcoming, we introduce the term kaleidoscopic teaming which seeks to capture complex and wide range of vulnerabilities that can happen in agents both in single-agent and multi-agent scenarios. We also present a new kaleidoscopic teaming framework that generates a diverse array of scenarios modeling real-world human societies. Our framework evaluates safety of agents in both single-agent and multi-agent setups. In single-agent setup, an agent is given a scenario that it needs to complete using the tools it has access to. In multi-agent setup, multiple agents either compete against or cooperate together to complete a task in the scenario through which we capture existing safety vulnerabilities in agents. We introduce new in-context optimization techniques that can be used in our kaleidoscopic teaming framework to generate better scenarios for safety analysis. Lastly, we present appropriate metrics that can be used along with our framework to measure safety of agents. Utilizing our kaleidoscopic teaming framework, we identify vulnerabilities in various models with respect to their safety in agentic use-cases. 

---
# From Unstructured Communication to Intelligent RAG: Multi-Agent Automation for Supply Chain Knowledge Bases 

**Authors**: Yao Zhang, Zaixi Shang, Silpan Patel, Mikel Zuniga  

**Link**: [PDF](https://arxiv.org/pdf/2506.17484)  

**Abstract**: Supply chain operations generate vast amounts of operational data; however, critical knowledge such as system usage practices, troubleshooting workflows, and resolution techniques often remains buried within unstructured communications like support tickets, emails, and chat logs. While RAG systems aim to leverage such communications as a knowledge base, their effectiveness is limited by raw data challenges: support tickets are typically noisy, inconsistent, and incomplete, making direct retrieval suboptimal. Unlike existing RAG approaches that focus on runtime optimization, we introduce a novel offline-first methodology that transforms these communications into a structured knowledge base. Our key innovation is a LLMs-based multi-agent system orchestrating three specialized agents: Category Discovery for taxonomy creation, Categorization for ticket grouping, and Knowledge Synthesis for article generation. Applying our methodology to real-world support tickets with resolution notes and comments, our system creates a compact knowledge base - reducing total volume to just 3.4% of original ticket data while improving quality. Experiments demonstrate that our prebuilt knowledge base in RAG systems significantly outperforms traditional RAG implementations (48.74% vs. 38.60% helpful answers) and achieves a 77.4% reduction in unhelpful responses. By automating institutional knowledge capture that typically remains siloed in experts' heads, our solution translates to substantial operational efficiency: reducing support workload, accelerating resolution times, and creating self-improving systems that automatically resolve approximately 50% of future supply chain tickets. Our approach addresses a key gap in knowledge management by transforming transient communications into structured, reusable knowledge through intelligent offline processing rather than latency-inducing runtime architectures. 

---
# OmniReflect: Discovering Transferable Constitutions for LLM agents via Neuro-Symbolic Reflections 

**Authors**: Manasa Bharadwaj, Nikhil Verma, Kevin Ferreira  

**Link**: [PDF](https://arxiv.org/pdf/2506.17449)  

**Abstract**: Efforts to improve Large Language Model (LLM) agent performance on complex tasks have largely focused on fine-tuning and iterative self-correction. However, these approaches often lack generalizable mechanisms for longterm learning and remain inefficient in dynamic environments. We introduce OmniReflect, a hierarchical, reflection-driven framework that constructs a constitution, a compact set of guiding principles distilled from task experiences, to enhance the effectiveness and efficiency of an LLM agent. OmniReflect operates in two modes: Self-sustaining, where a single agent periodically curates its own reflections during task execution, and Co-operative, where a Meta-advisor derives a constitution from a small calibration set to guide another agent. To construct these constitutional principles, we employ Neural, Symbolic, and NeuroSymbolic techniques, offering a balance between contextual adaptability and computational efficiency. Empirical results averaged across models show major improvements in task success, with absolute gains of +10.3% on ALFWorld, +23.8% on BabyAI, and +8.3% on PDDL in the Self-sustaining mode. Similar gains are seen in the Co-operative mode, where a lightweight Qwen3-4B ReAct agent outperforms all Reflexion baselines on BabyAI. These findings highlight the robustness and effectiveness of OmniReflect across environments and backbones. 

---
# Keeping Medical AI Healthy: A Review of Detection and Correction Methods for System Degradation 

**Authors**: Hao Guan, David Bates, Li Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.17442)  

**Abstract**: Artificial intelligence (AI) is increasingly integrated into modern healthcare, offering powerful support for clinical decision-making. However, in real-world settings, AI systems may experience performance degradation over time, due to factors such as shifting data distributions, changes in patient characteristics, evolving clinical protocols, and variations in data quality. These factors can compromise model reliability, posing safety concerns and increasing the likelihood of inaccurate predictions or adverse outcomes. This review presents a forward-looking perspective on monitoring and maintaining the "health" of AI systems in healthcare. We highlight the urgent need for continuous performance monitoring, early degradation detection, and effective self-correction mechanisms. The paper begins by reviewing common causes of performance degradation at both data and model levels. We then summarize key techniques for detecting data and model drift, followed by an in-depth look at root cause analysis. Correction strategies are further reviewed, ranging from model retraining to test-time adaptation. Our survey spans both traditional machine learning models and state-of-the-art large language models (LLMs), offering insights into their strengths and limitations. Finally, we discuss ongoing technical challenges and propose future research directions. This work aims to guide the development of reliable, robust medical AI systems capable of sustaining safe, long-term deployment in dynamic clinical settings. 

---
# Resource Rational Contractualism Should Guide AI Alignment 

**Authors**: Sydney Levine, Matija Franklin, Tan Zhi-Xuan, Secil Yanik Guyot, Lionel Wong, Daniel Kilov, Yejin Choi, Joshua B. Tenenbaum, Noah Goodman, Seth Lazar, Iason Gabriel  

**Link**: [PDF](https://arxiv.org/pdf/2506.17434)  

**Abstract**: AI systems will soon have to navigate human environments and make decisions that affect people and other AI agents whose goals and values diverge. Contractualist alignment proposes grounding those decisions in agreements that diverse stakeholders would endorse under the right conditions, yet securing such agreement at scale remains costly and slow -- even for advanced AI. We therefore propose Resource-Rational Contractualism (RRC): a framework where AI systems approximate the agreements rational parties would form by drawing on a toolbox of normatively-grounded, cognitively-inspired heuristics that trade effort for accuracy. An RRC-aligned agent would not only operate efficiently, but also be equipped to dynamically adapt to and interpret the ever-changing human social world. 

---
# Individual Causal Inference with Structural Causal Model 

**Authors**: Daniel T. Chang  

**Link**: [PDF](https://arxiv.org/pdf/2506.17300)  

**Abstract**: Individual causal inference (ICI) uses causal inference methods to understand and predict the effects of interventions on individuals, considering their specific characteristics / facts. It aims to estimate individual causal effect (ICE), which varies across individuals. Estimating ICE can be challenging due to the limited data available for individuals, and the fact that most causal inference methods are population-based. Structural Causal Model (SCM) is fundamentally population-based. Therefore, causal discovery (structural learning and parameter learning), association queries and intervention queries are all naturally population-based. However, exogenous variables (U) in SCM can encode individual variations and thus provide the mechanism for individualized population per specific individual characteristics / facts. Based on this, we propose ICI with SCM as a "rung 3" causal inference, because it involves "imagining" what would be the causal effect of a hypothetical intervention on an individual, given the individual's observed characteristics / facts. Specifically, we propose the indiv-operator, indiv(W), to formalize/represent the population individualization process, and the individual causal query, P(Y | indiv(W), do(X), Z), to formalize/represent ICI. We show and argue that ICI with SCM is inference on individual alternatives (possible), not individual counterfactuals (non-actual). 

---
# Evaluating Generalization and Representation Stability in Small LMs via Prompting 

**Authors**: Rahul Raja, Arpita Vats  

**Link**: [PDF](https://arxiv.org/pdf/2506.17289)  

**Abstract**: We investigate the generalization capabilities of small language models under two popular adaptation paradigms: few-shot prompting and supervised fine-tuning. While prompting is often favored for its parameter efficiency and flexibility, it remains unclear how robust this approach is in low-resource settings and under distributional shifts. This paper presents a comparative study of prompting and fine-tuning across task formats, prompt styles, and model scales, with a focus on their behavior in both in-distribution and out-of-distribution (OOD) settings.
Beyond accuracy, we analyze the internal representations learned by each approach to assess the stability and abstraction of task-specific features. Our findings highlight critical differences in how small models internalize and generalize knowledge under different adaptation strategies. This work offers practical guidance for model selection in low-data regimes and contributes empirical insight into the ongoing debate over prompting versus fine-tuning. Code for the experiments is available at the following 

---
# Vision as a Dialect: Unifying Visual Understanding and Generation via Text-Aligned Representations 

**Authors**: Jiaming Han, Hao Chen, Yang Zhao, Hanyu Wang, Qi Zhao, Ziyan Yang, Hao He, Xiangyu Yue, Lu Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2506.18898)  

**Abstract**: This paper presents a multimodal framework that attempts to unify visual understanding and generation within a shared discrete semantic representation. At its core is the Text-Aligned Tokenizer (TA-Tok), which converts images into discrete tokens using a text-aligned codebook projected from a large language model's (LLM) vocabulary. By integrating vision and text into a unified space with an expanded vocabulary, our multimodal LLM, Tar, enables cross-modal input and output through a shared interface, without the need for modality-specific designs. Additionally, we propose scale-adaptive encoding and decoding to balance efficiency and visual detail, along with a generative de-tokenizer to produce high-fidelity visual outputs. To address diverse decoding needs, we utilize two complementary de-tokenizers: a fast autoregressive model and a diffusion-based model. To enhance modality fusion, we investigate advanced pre-training tasks, demonstrating improvements in both visual understanding and generation. Experiments across benchmarks show that Tar matches or surpasses existing multimodal LLM methods, achieving faster convergence and greater training efficiency. Code, models, and data are available at this https URL 

---
# MinD: Unified Visual Imagination and Control via Hierarchical World Models 

**Authors**: Xiaowei Chi, Kuangzhi Ge, Jiaming Liu, Siyuan Zhou, Peidong Jia, Zichen He, Yuzhen Liu, Tingguang Li, Lei Han, Sirui Han, Shanghang Zhang, Yike Guo  

**Link**: [PDF](https://arxiv.org/pdf/2506.18897)  

**Abstract**: Video generation models (VGMs) offer a promising pathway for unified world modeling in robotics by integrating simulation, prediction, and manipulation. However, their practical application remains limited due to (1) slowgeneration speed, which limits real-time interaction, and (2) poor consistency between imagined videos and executable actions. To address these challenges, we propose Manipulate in Dream (MinD), a hierarchical diffusion-based world model framework that employs a dual-system design for vision-language manipulation. MinD executes VGM at low frequencies to extract video prediction features, while leveraging a high-frequency diffusion policy for real-time interaction. This architecture enables low-latency, closed-loop control in manipulation with coherent visual guidance. To better coordinate the two systems, we introduce a video-action diffusion matching module (DiffMatcher), with a novel co-training strategy that uses separate schedulers for each diffusion model. Specifically, we introduce a diffusion-forcing mechanism to DiffMatcher that aligns their intermediate representations during training, helping the fast action model better understand video-based predictions. Beyond manipulation, MinD also functions as a world simulator, reliably predicting task success or failure in latent space before execution. Trustworthy analysis further shows that VGMs can preemptively evaluate task feasibility and mitigate risks. Extensive experiments across multiple benchmarks demonstrate that MinD achieves state-of-the-art manipulation (63%+) in RL-Bench, advancing the frontier of unified world modeling in robotics. 

---
# OMEGA: Can LLMs Reason Outside the Box in Math? Evaluating Exploratory, Compositional, and Transformative Generalization 

**Authors**: Yiyou Sun, Shawn Hu, Georgia Zhou, Ken Zheng, Hannaneh Hajishirzi, Nouha Dziri, Dawn Song  

**Link**: [PDF](https://arxiv.org/pdf/2506.18880)  

**Abstract**: Recent large-scale language models (LLMs) with long Chain-of-Thought reasoning-such as DeepSeek-R1-have achieved impressive results on Olympiad-level mathematics benchmarks. However, they often rely on a narrow set of strategies and struggle with problems that require a novel way of thinking. To systematically investigate these limitations, we introduce OMEGA-Out-of-distribution Math Problems Evaluation with 3 Generalization Axes-a controlled yet diverse benchmark designed to evaluate three axes of out-of-distribution generalization, inspired by Boden's typology of creativity: (1) Exploratory-applying known problem solving skills to more complex instances within the same problem domain; (2) Compositional-combining distinct reasoning skills, previously learned in isolation, to solve novel problems that require integrating these skills in new and coherent ways; and (3) Transformative-adopting novel, often unconventional strategies by moving beyond familiar approaches to solve problems more effectively. OMEGA consists of programmatically generated training-test pairs derived from templated problem generators across geometry, number theory, algebra, combinatorics, logic, and puzzles, with solutions verified using symbolic, numerical, or graphical methods. We evaluate frontier (or top-tier) LLMs and observe sharp performance degradation as problem complexity increases. Moreover, we fine-tune the Qwen-series models across all generalization settings and observe notable improvements in exploratory generalization, while compositional generalization remains limited and transformative reasoning shows little to no improvement. By isolating and quantifying these fine-grained failures, OMEGA lays the groundwork for advancing LLMs toward genuine mathematical creativity beyond mechanical proficiency. 

---
# CommVQ: Commutative Vector Quantization for KV Cache Compression 

**Authors**: Junyan Li, Yang Zhang, Muhammad Yusuf Hassan, Talha Chafekar, Tianle Cai, Zhile Ren, Pengsheng Guo, Foroozan Karimzadeh, Colorado Reed, Chong Wang, Chuang Gan  

**Link**: [PDF](https://arxiv.org/pdf/2506.18879)  

**Abstract**: Large Language Models (LLMs) are increasingly used in applications requiring long context lengths, but the key-value (KV) cache often becomes a memory bottleneck on GPUs as context grows. To address this, we propose Commutative Vector Quantization (CommVQ) to significantly reduce memory usage for long-context LLM inference. We first introduce additive quantization with a lightweight encoder and codebook to compress the KV cache, which can be decoded via simple matrix multiplication. To further reduce computational costs during decoding, we design the codebook to be commutative with Rotary Position Embedding (RoPE) and train it using an Expectation-Maximization (EM) algorithm. This enables efficient integration of decoding into the self-attention mechanism. Our approach achieves high accuracy with additive quantization and low overhead via the RoPE-commutative codebook. Experiments on long-context benchmarks and GSM8K show that our method reduces FP16 KV cache size by 87.5% with 2-bit quantization, while outperforming state-of-the-art KV cache quantization methods. Notably, it enables 1-bit KV cache quantization with minimal accuracy loss, allowing a LLaMA-3.1 8B model to run with a 128K context length on a single RTX 4090 GPU. The source code is available at: this https URL. 

---
# OmniGen2: Exploration to Advanced Multimodal Generation 

**Authors**: Chenyuan Wu, Pengfei Zheng, Ruiran Yan, Shitao Xiao, Xin Luo, Yueze Wang, Wanli Li, Xiyan Jiang, Yexin Liu, Junjie Zhou, Ze Liu, Ziyi Xia, Chaofan Li, Haoge Deng, Jiahao Wang, Kun Luo, Bo Zhang, Defu Lian, Xinlong Wang, Zhongyuan Wang, Tiejun Huang, Zheng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.18871)  

**Abstract**: In this work, we introduce OmniGen2, a versatile and open-source generative model designed to provide a unified solution for diverse generation tasks, including text-to-image, image editing, and in-context generation. Unlike OmniGen v1, OmniGen2 features two distinct decoding pathways for text and image modalities, utilizing unshared parameters and a decoupled image tokenizer. This design enables OmniGen2 to build upon existing multimodal understanding models without the need to re-adapt VAE inputs, thereby preserving the original text generation capabilities. To facilitate the training of OmniGen2, we developed comprehensive data construction pipelines, encompassing image editing and in-context generation data. Additionally, we introduce a reflection mechanism tailored for image generation tasks and curate a dedicated reflection dataset based on OmniGen2. Despite its relatively modest parameter size, OmniGen2 achieves competitive results on multiple task benchmarks, including text-to-image and image editing. To further evaluate in-context generation, also referred to as subject-driven tasks, we introduce a new benchmark named OmniContext. OmniGen2 achieves state-of-the-art performance among open-source models in terms of consistency. We will release our models, training code, datasets, and data construction pipeline to support future research in this field. Project Page: this https URL GitHub Link: this https URL 

---
# OmniAvatar: Efficient Audio-Driven Avatar Video Generation with Adaptive Body Animation 

**Authors**: Qijun Gan, Ruizi Yang, Jianke Zhu, Shaofei Xue, Steven Hoi  

**Link**: [PDF](https://arxiv.org/pdf/2506.18866)  

**Abstract**: Significant progress has been made in audio-driven human animation, while most existing methods focus mainly on facial movements, limiting their ability to create full-body animations with natural synchronization and fluidity. They also struggle with precise prompt control for fine-grained generation. To tackle these challenges, we introduce OmniAvatar, an innovative audio-driven full-body video generation model that enhances human animation with improved lip-sync accuracy and natural movements. OmniAvatar introduces a pixel-wise multi-hierarchical audio embedding strategy to better capture audio features in the latent space, enhancing lip-syncing across diverse scenes. To preserve the capability for prompt-driven control of foundation models while effectively incorporating audio features, we employ a LoRA-based training approach. Extensive experiments show that OmniAvatar surpasses existing models in both facial and semi-body video generation, offering precise text-based control for creating videos in various domains, such as podcasts, human interactions, dynamic scenes, and singing. Our project page is this https URL. 

---
# TAMMs: Temporal-Aware Multimodal Model for Satellite Image Change Understanding and Forecasting 

**Authors**: Zhongbin Guo, Yuhao Wang, Ping Jian, Xinyue Chen, Wei Peng, Ertai E  

**Link**: [PDF](https://arxiv.org/pdf/2506.18862)  

**Abstract**: Satellite image time-series analysis demands fine-grained spatial-temporal reasoning, which remains a challenge for existing multimodal large language models (MLLMs). In this work, we study the capabilities of MLLMs on a novel task that jointly targets temporal change understanding and future scene generation, aiming to assess their potential for modeling complex multimodal dynamics over time. We propose TAMMs, a Temporal-Aware Multimodal Model for satellite image change understanding and forecasting, which enhances frozen MLLMs with lightweight temporal modules for structured sequence encoding and contextual prompting. To guide future image generation, TAMMs introduces a Semantic-Fused Control Injection (SFCI) mechanism that adaptively combines high-level semantic reasoning and structural priors within an enhanced ControlNet. This dual-path conditioning enables temporally consistent and semantically grounded image synthesis. Experiments demonstrate that TAMMs outperforms strong MLLM baselines in both temporal change understanding and future image forecasting tasks, highlighting how carefully designed temporal reasoning and semantic fusion can unlock the full potential of MLLMs for spatio-temporal understanding. 

---
# Mechanistic Interpretability Needs Philosophy 

**Authors**: Iwan Williams, Ninell Oldenburg, Ruchira Dhar, Joshua Hatherley, Constanza Fierro, Nina Rajcic, Sandrine R. Schiller, Filippos Stamatiou, Anders Søgaard  

**Link**: [PDF](https://arxiv.org/pdf/2506.18852)  

**Abstract**: Mechanistic interpretability (MI) aims to explain how neural networks work by uncovering their underlying causal mechanisms. As the field grows in influence, it is increasingly important to examine not just models themselves, but the assumptions, concepts and explanatory strategies implicit in MI research. We argue that mechanistic interpretability needs philosophy: not as an afterthought, but as an ongoing partner in clarifying its concepts, refining its methods, and assessing the epistemic and ethical stakes of interpreting AI systems. Taking three open problems from the MI literature as examples, this position paper illustrates the value philosophy can add to MI research, and outlines a path toward deeper interdisciplinary dialogue. 

---
# LongWriter-Zero: Mastering Ultra-Long Text Generation via Reinforcement Learning 

**Authors**: Yuhao Wu, Yushi Bai, Zhiqiang Hu, Roy Ka-Wei Lee, Juanzi Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.18841)  

**Abstract**: Ultra-long generation by large language models (LLMs) is a widely demanded scenario, yet it remains a significant challenge due to their maximum generation length limit and overall quality degradation as sequence length increases. Previous approaches, exemplified by LongWriter, typically rely on ''teaching'', which involves supervised fine-tuning (SFT) on synthetic long-form outputs. However, this strategy heavily depends on synthetic SFT data, which is difficult and costly to construct, often lacks coherence and consistency, and tends to be overly artificial and structurally monotonous. In this work, we propose an incentivization-based approach that, starting entirely from scratch and without relying on any annotated or synthetic data, leverages reinforcement learning (RL) to foster the emergence of ultra-long, high-quality text generation capabilities in LLMs. We perform RL training starting from a base model, similar to R1-Zero, guiding it to engage in reasoning that facilitates planning and refinement during the writing process. To support this, we employ specialized reward models that steer the LLM towards improved length control, writing quality, and structural formatting. Experimental evaluations show that our LongWriter-Zero model, trained from Qwen2.5-32B, consistently outperforms traditional SFT methods on long-form writing tasks, achieving state-of-the-art results across all metrics on WritingBench and Arena-Write, and even surpassing 100B+ models such as DeepSeek R1 and Qwen3-235B. We open-source our data and model checkpoints under this https URL 

---
# Understanding Software Engineering Agents: A Study of Thought-Action-Result Trajectories 

**Authors**: Islem Bouzenia, Michael Pradel  

**Link**: [PDF](https://arxiv.org/pdf/2506.18824)  

**Abstract**: Large Language Model (LLM)-based agents are increasingly employed to automate complex software engineering tasks such as program repair and issue resolution. These agents operate by autonomously generating natural language thoughts, invoking external tools, and iteratively refining their solutions. Despite their widespread adoption, the internal decision-making processes of these agents remain largely unexplored, limiting our understanding of their operational dynamics and failure modes. In this paper, we present a large-scale empirical study of the thought-action-result trajectories of three state-of-the-art LLM-based agents: \textsc{RepairAgent}, \textsc{AutoCodeRover}, and \textsc{OpenHands}. We unify their interaction logs into a common format, capturing 120 trajectories and 2822 LLM interactions focused on program repair and issue resolution. Our study combines quantitative analyses of structural properties, action patterns, and token usage with qualitative assessments of reasoning coherence and feedback integration. We identify key trajectory characteristics such as iteration counts and token consumption, recurring action sequences, and the semantic coherence linking thoughts, actions, and their results. Our findings reveal behavioral motifs and anti-patterns that distinguish successful from failed executions, providing actionable insights for improving agent design, including prompting strategies, failure diagnosis, and anti-pattern detection. We release our dataset and annotation framework to support further research on transparent and robust autonomous software engineering agents. 

---
# RWESummary: A Framework and Test for Choosing Large Language Models to Summarize Real-World Evidence (RWE) Studies 

**Authors**: Arjun Mukerji, Michael L. Jackson, Jason Jones, Neil Sanghavi  

**Link**: [PDF](https://arxiv.org/pdf/2506.18819)  

**Abstract**: Large Language Models (LLMs) have been extensively evaluated for general summarization tasks as well as medical research assistance, but they have not been specifically evaluated for the task of summarizing real-world evidence (RWE) from structured output of RWE studies. We introduce RWESummary, a proposed addition to the MedHELM framework (Bedi, Cui, Fuentes, Unell et al., 2025) to enable benchmarking of LLMs for this task. RWESummary includes one scenario and three evaluations covering major types of errors observed in summarization of medical research studies and was developed using Atropos Health proprietary data. Additionally, we use RWESummary to compare the performance of different LLMs in our internal RWE summarization tool. At the time of publication, with 13 distinct RWE studies, we found the Gemini 2.5 models performed best overall (both Flash and Pro). We suggest RWESummary as a novel and useful foundation model benchmark for real-world evidence study summarization. 

---
# OC-SOP: Enhancing Vision-Based 3D Semantic Occupancy Prediction by Object-Centric Awareness 

**Authors**: Helin Cao, Sven Behnke  

**Link**: [PDF](https://arxiv.org/pdf/2506.18798)  

**Abstract**: Autonomous driving perception faces significant challenges due to occlusions and incomplete scene data in the environment. To overcome these issues, the task of semantic occupancy prediction (SOP) is proposed, which aims to jointly infer both the geometry and semantic labels of a scene from images. However, conventional camera-based methods typically treat all categories equally and primarily rely on local features, leading to suboptimal predictions, especially for dynamic foreground objects. To address this, we propose Object-Centric SOP (OC-SOP), a framework that integrates high-level object-centric cues extracted via a detection branch into the semantic occupancy prediction pipeline. This object-centric integration significantly enhances the prediction accuracy for foreground objects and achieves state-of-the-art performance among all categories on SemanticKITTI. 

---
# Shift Happens: Mixture of Experts based Continual Adaptation in Federated Learning 

**Authors**: Rahul Atul Bhope, K.R. Jayaram, Praveen Venkateswaran, Nalini Venkatasubramanian  

**Link**: [PDF](https://arxiv.org/pdf/2506.18789)  

**Abstract**: Federated Learning (FL) enables collaborative model training across decentralized clients without sharing raw data, yet faces significant challenges in real-world settings where client data distributions evolve dynamically over time. This paper tackles the critical problem of covariate and label shifts in streaming FL environments, where non-stationary data distributions degrade model performance and require adaptive middleware solutions. We introduce ShiftEx, a shift-aware mixture of experts framework that dynamically creates and trains specialized global models in response to detected distribution shifts using Maximum Mean Discrepancy for covariate shifts. The framework employs a latent memory mechanism for expert reuse and implements facility location-based optimization to jointly minimize covariate mismatch, expert creation costs, and label imbalance. Through theoretical analysis and comprehensive experiments on benchmark datasets, we demonstrate 5.5-12.9 percentage point accuracy improvements and 22-95 % faster adaptation compared to state-of-the-art FL baselines across diverse shift scenarios. The proposed approach offers a scalable, privacy-preserving middleware solution for FL systems operating in non-stationary, real-world conditions while minimizing communication and computational overhead. 

---
# SWA-SOP: Spatially-aware Window Attention for Semantic Occupancy Prediction in Autonomous Driving 

**Authors**: Helin Cao, Rafael Materla, Sven Behnke  

**Link**: [PDF](https://arxiv.org/pdf/2506.18785)  

**Abstract**: Perception systems in autonomous driving rely on sensors such as LiDAR and cameras to perceive the 3D environment. However, due to occlusions and data sparsity, these sensors often fail to capture complete information. Semantic Occupancy Prediction (SOP) addresses this challenge by inferring both occupancy and semantics of unobserved regions. Existing transformer-based SOP methods lack explicit modeling of spatial structure in attention computation, resulting in limited geometric awareness and poor performance in sparse or occluded areas. To this end, we propose Spatially-aware Window Attention (SWA), a novel mechanism that incorporates local spatial context into attention. SWA significantly improves scene completion and achieves state-of-the-art results on LiDAR-based SOP benchmarks. We further validate its generality by integrating SWA into a camera-based SOP pipeline, where it also yields consistent gains across modalities. 

---
# Sensitivity Analysis of Image Classification Models using Generalized Polynomial Chaos 

**Authors**: Lukas Bahr, Lucas Poßner, Konstantin Weise, Sophie Gröger, Rüdiger Daub  

**Link**: [PDF](https://arxiv.org/pdf/2506.18751)  

**Abstract**: Integrating advanced communication protocols in production has accelerated the adoption of data-driven predictive quality methods, notably machine learning (ML) models. However, ML models in image classification often face significant uncertainties arising from model, data, and domain shifts. These uncertainties lead to overconfidence in the classification model's output. To better understand these models, sensitivity analysis can help to analyze the relative influence of input parameters on the output. This work investigates the sensitivity of image classification models used for predictive quality. We propose modeling the distributional domain shifts of inputs with random variables and quantifying their impact on the model's outputs using Sobol indices computed via generalized polynomial chaos (GPC). This approach is validated through a case study involving a welding defect classification problem, utilizing a fine-tuned ResNet18 model and an emblem classification model used in BMW Group production facilities. 

---
# BRAVE: Brain-Controlled Prosthetic Arm with Voice Integration and Embodied Learning for Enhanced Mobility 

**Authors**: Abdul Basit, Maha Nawaz, Muhammad Shafique  

**Link**: [PDF](https://arxiv.org/pdf/2506.18749)  

**Abstract**: Non-invasive brain-computer interfaces (BCIs) have the potential to enable intuitive control of prosthetic limbs for individuals with upper limb amputations. However, existing EEG-based control systems face challenges related to signal noise, classification accuracy, and real-time adaptability. In this work, we present BRAVE, a hybrid EEG and voice-controlled prosthetic system that integrates ensemble learning-based EEG classification with a human-in-the-loop (HITL) correction framework for enhanced responsiveness. Unlike traditional electromyography (EMG)-based prosthetic control, BRAVE aims to interpret EEG-driven motor intent, enabling movement control without reliance on residual muscle activity. To improve classification robustness, BRAVE combines LSTM, CNN, and Random Forest models in an ensemble framework, achieving a classification accuracy of 96% across test subjects. EEG signals are preprocessed using a bandpass filter (0.5-45 Hz), Independent Component Analysis (ICA) for artifact removal, and Common Spatial Pattern (CSP) feature extraction to minimize contamination from electromyographic (EMG) and electrooculographic (EOG) signals. Additionally, BRAVE incorporates automatic speech recognition (ASR) to facilitate intuitive mode switching between different degrees of freedom (DOF) in the prosthetic arm. The system operates in real time, with a response latency of 150 ms, leveraging Lab Streaming Layer (LSL) networking for synchronized data acquisition. The system is evaluated on an in-house fabricated prosthetic arm and on multiple participants highlighting the generalizability across users. The system is optimized for low-power embedded deployment, ensuring practical real-world application beyond high-performance computing environments. Our results indicate that BRAVE offers a promising step towards robust, real-time, non-invasive prosthetic control. 

---
# ContinualFlow: Learning and Unlearning with Neural Flow Matching 

**Authors**: Lorenzo Simone, Davide Bacciu, Shuangge Ma  

**Link**: [PDF](https://arxiv.org/pdf/2506.18747)  

**Abstract**: We introduce ContinualFlow, a principled framework for targeted unlearning in generative models via Flow Matching. Our method leverages an energy-based reweighting loss to softly subtract undesired regions of the data distribution without retraining from scratch or requiring direct access to the samples to be unlearned. Instead, it relies on energy-based proxies to guide the unlearning process. We prove that this induces gradients equivalent to Flow Matching toward a soft mass-subtracted target, and validate the framework through experiments on 2D and image domains, supported by interpretable visualizations and quantitative evaluations. 

---
# On the Existence of Universal Simulators of Attention 

**Authors**: Debanjan Dutta, Faizanuddin Ansari, Anish Chakrabarty, Swagatam Das  

**Link**: [PDF](https://arxiv.org/pdf/2506.18739)  

**Abstract**: Prior work on the learnability of transformers has established its capacity to approximate specific algorithmic patterns through training under restrictive architectural assumptions. Fundamentally, these arguments remain data-driven and therefore can only provide a probabilistic guarantee. Expressivity, on the contrary, has theoretically been explored to address the problems \emph{computable} by such architecture. These results proved the Turing-completeness of transformers, investigated bounds focused on circuit complexity, and formal logic. Being at the crossroad between learnability and expressivity, the question remains: \emph{can transformer architectures exactly simulate an arbitrary attention mechanism, or in particular, the underlying operations?} In this study, we investigate the transformer encoder's ability to simulate a vanilla attention mechanism. By constructing a universal simulator $\mathcal{U}$ composed of transformer encoders, we present algorithmic solutions to identically replicate attention outputs and the underlying elementary matrix and activation operations via RASP, a formal framework for transformer computation. Our proofs, for the first time, show the existence of an algorithmically achievable data-agnostic solution, previously known to be approximated only by learning. 

---
# Deep CNN Face Matchers Inherently Support Revocable Biometric Templates 

**Authors**: Aman Bhatta, Michael C. King, Kevin W. Bowyer  

**Link**: [PDF](https://arxiv.org/pdf/2506.18731)  

**Abstract**: One common critique of biometric authentication is that if an individual's biometric is compromised, then the individual has no recourse. The concept of revocable biometrics was developed to address this concern. A biometric scheme is revocable if an individual can have their current enrollment in the scheme revoked, so that the compromised biometric template becomes worthless, and the individual can re-enroll with a new template that has similar recognition power. We show that modern deep CNN face matchers inherently allow for a robust revocable biometric scheme. For a given state-of-the-art deep CNN backbone and training set, it is possible to generate an unlimited number of distinct face matcher models that have both (1) equivalent recognition power, and (2) strongly incompatible biometric templates. The equivalent recognition power extends to the point of generating impostor and genuine distributions that have the same shape and placement on the similarity dimension, meaning that the models can share a similarity threshold for a 1-in-10,000 false match rate. The biometric templates from different model instances are so strongly incompatible that the cross-instance similarity score for images of the same person is typically lower than the same-instance similarity score for images of different persons. That is, a stolen biometric template that is revoked is of less value in attempting to match the re-enrolled identity than the average impostor template. We also explore the feasibility of using a Vision Transformer (ViT) backbone-based face matcher in the revocable biometric system proposed in this work and demonstrate that it is less suitable compared to typical ResNet-based deep CNN backbones. 

---
# MuseControlLite: Multifunctional Music Generation with Lightweight Conditioners 

**Authors**: Fang-Duo Tsai, Shih-Lun Wu, Weijaw Lee, Sheng-Ping Yang, Bo-Rui Chen, Hao-Chung Cheng, Yi-Hsuan Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.18729)  

**Abstract**: We propose MuseControlLite, a lightweight mechanism designed to fine-tune text-to-music generation models for precise conditioning using various time-varying musical attributes and reference audio signals. The key finding is that positional embeddings, which have been seldom used by text-to-music generation models in the conditioner for text conditions, are critical when the condition of interest is a function of time. Using melody control as an example, our experiments show that simply adding rotary positional embeddings to the decoupled cross-attention layers increases control accuracy from 56.6% to 61.1%, while requiring 6.75 times fewer trainable parameters than state-of-the-art fine-tuning mechanisms, using the same pre-trained diffusion Transformer model of Stable Audio Open. We evaluate various forms of musical attribute control, audio inpainting, and audio outpainting, demonstrating improved controllability over MusicGen-Large and Stable Audio Open ControlNet at a significantly lower fine-tuning cost, with only 85M trainble parameters. Source code, model checkpoints, and demo examples are available at: https: //MuseControlLite.this http URL. 

---
# A Study of Dynamic Stock Relationship Modeling and S&P500 Price Forecasting Based on Differential Graph Transformer 

**Authors**: Linyue Hu, Qi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.18717)  

**Abstract**: Stock price prediction is vital for investment decisions and risk management, yet remains challenging due to markets' nonlinear dynamics and time-varying inter-stock correlations. Traditional static-correlation models fail to capture evolving stock relationships. To address this, we propose a Differential Graph Transformer (DGT) framework for dynamic relationship modeling and price prediction. Our DGT integrates sequential graph structure changes into multi-head self-attention via a differential graph mechanism, adaptively preserving high-value connections while suppressing noise. Causal temporal attention captures global/local dependencies in price sequences. We further evaluate correlation metrics (Pearson, Mutual Information, Spearman, Kendall's Tau) across global/local/dual scopes as spatial-attention priors. Using 10 years of S&P 500 closing prices (z-score normalized; 64-day sliding windows), DGT with spatial priors outperformed GRU baselines (RMSE: 0.24 vs. 0.87). Kendall's Tau global matrices yielded optimal results (MAE: 0.11). K-means clustering revealed "high-volatility growth" and "defensive blue-chip" stocks, with the latter showing lower errors (RMSE: 0.13) due to stable correlations. Kendall's Tau and Mutual Information excelled in volatile sectors. This study innovatively combines differential graph structures with Transformers, validating dynamic relationship modeling and identifying optimal correlation metrics/scopes. Clustering analysis supports tailored quantitative strategies. Our framework advances financial time-series prediction through dynamic modeling and cross-asset interaction analysis. 

---
# Frequency-Weighted Training Losses for Phoneme-Level DNN-based Speech Enhancement 

**Authors**: Nasser-Eddine Monir, Paul Magron, Romain Serizel  

**Link**: [PDF](https://arxiv.org/pdf/2506.18714)  

**Abstract**: Recent advances in deep learning have significantly improved multichannel speech enhancement algorithms, yet conventional training loss functions such as the scale-invariant signal-to-distortion ratio (SDR) may fail to preserve fine-grained spectral cues essential for phoneme intelligibility. In this work, we propose perceptually-informed variants of the SDR loss, formulated in the time-frequency domain and modulated by frequency-dependent weighting schemes. These weights are designed to emphasize time-frequency regions where speech is prominent or where the interfering noise is particularly strong. We investigate both fixed and adaptive strategies, including ANSI band-importance weights, spectral magnitude-based weighting, and dynamic weighting based on the relative amount of speech and noise. We train the FaSNet multichannel speech enhancement model using these various losses. Experimental results show that while standard metrics such as the SDR are only marginally improved, their perceptual frequency-weighted counterparts exhibit a more substantial improvement. Besides, spectral and phoneme-level analysis indicates better consonant reconstruction, which points to a better preservation of certain acoustic cues. 

---
# Benchmarking the Pedagogical Knowledge of Large Language Models 

**Authors**: Maxime Lelièvre, Amy Waldock, Meng Liu, Natalia Valdés Aspillaga, Alasdair Mackintosh, María José Ogando Portelo, Jared Lee, Paul Atherton, Robin A. A. Ince, Oliver G. B. Garrod  

**Link**: [PDF](https://arxiv.org/pdf/2506.18710)  

**Abstract**: Benchmarks like Massive Multitask Language Understanding (MMLU) have played a pivotal role in evaluating AI's knowledge and abilities across diverse domains. However, existing benchmarks predominantly focus on content knowledge, leaving a critical gap in assessing models' understanding of pedagogy - the method and practice of teaching. This paper introduces The Pedagogy Benchmark, a novel dataset designed to evaluate large language models on their Cross-Domain Pedagogical Knowledge (CDPK) and Special Education Needs and Disability (SEND) pedagogical knowledge. These benchmarks are built on a carefully curated set of questions sourced from professional development exams for teachers, which cover a range of pedagogical subdomains such as teaching strategies and assessment methods. Here we outline the methodology and development of these benchmarks. We report results for 97 models, with accuracies spanning a range from 28% to 89% on the pedagogical knowledge questions. We consider the relationship between cost and accuracy and chart the progression of the Pareto value frontier over time. We provide online leaderboards at this https URL which are updated with new models and allow interactive exploration and filtering based on various model properties, such as cost per token and open-vs-closed weights, as well as looking at performance in different subjects. LLMs and generative AI have tremendous potential to influence education and help to address the global learning crisis. Education-focused benchmarks are crucial to measure models' capacities to understand pedagogical concepts, respond appropriately to learners' needs, and support effective teaching practices across diverse contexts. They are needed for informing the responsible and evidence-based deployment of LLMs and LLM-based tools in educational settings, and for guiding both development and policy decisions. 

---
# Matrix-Game: Interactive World Foundation Model 

**Authors**: Yifan Zhang, Chunli Peng, Boyang Wang, Puyi Wang, Qingcheng Zhu, Fei Kang, Biao Jiang, Zedong Gao, Eric Li, Yang Liu, Yahui Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.18701)  

**Abstract**: We introduce Matrix-Game, an interactive world foundation model for controllable game world generation. Matrix-Game is trained using a two-stage pipeline that first performs large-scale unlabeled pretraining for environment understanding, followed by action-labeled training for interactive video generation. To support this, we curate Matrix-Game-MC, a comprehensive Minecraft dataset comprising over 2,700 hours of unlabeled gameplay video clips and over 1,000 hours of high-quality labeled clips with fine-grained keyboard and mouse action annotations. Our model adopts a controllable image-to-world generation paradigm, conditioned on a reference image, motion context, and user actions. With over 17 billion parameters, Matrix-Game enables precise control over character actions and camera movements, while maintaining high visual quality and temporal coherence. To evaluate performance, we develop GameWorld Score, a unified benchmark measuring visual quality, temporal quality, action controllability, and physical rule understanding for Minecraft world generation. Extensive experiments show that Matrix-Game consistently outperforms prior open-source Minecraft world models (including Oasis and MineWorld) across all metrics, with particularly strong gains in controllability and physical consistency. Double-blind human evaluations further confirm the superiority of Matrix-Game, highlighting its ability to generate perceptually realistic and precisely controllable videos across diverse game scenarios. To facilitate future research on interactive image-to-world generation, we will open-source the Matrix-Game model weights and the GameWorld Score benchmark at this https URL. 

---
# NOVA: Navigation via Object-Centric Visual Autonomy for High-Speed Target Tracking in Unstructured GPS-Denied Environments 

**Authors**: Alessandro Saviolo, Giuseppe Loianno  

**Link**: [PDF](https://arxiv.org/pdf/2506.18689)  

**Abstract**: Autonomous aerial target tracking in unstructured and GPS-denied environments remains a fundamental challenge in robotics. Many existing methods rely on motion capture systems, pre-mapped scenes, or feature-based localization to ensure safety and control, limiting their deployment in real-world conditions. We introduce NOVA, a fully onboard, object-centric framework that enables robust target tracking and collision-aware navigation using only a stereo camera and an IMU. Rather than constructing a global map or relying on absolute localization, NOVA formulates perception, estimation, and control entirely in the target's reference frame. A tightly integrated stack combines a lightweight object detector with stereo depth completion, followed by histogram-based filtering to infer robust target distances under occlusion and noise. These measurements feed a visual-inertial state estimator that recovers the full 6-DoF pose of the robot relative to the target. A nonlinear model predictive controller (NMPC) plans dynamically feasible trajectories in the target frame. To ensure safety, high-order control barrier functions are constructed online from a compact set of high-risk collision points extracted from depth, enabling real-time obstacle avoidance without maps or dense representations. We validate NOVA across challenging real-world scenarios, including urban mazes, forest trails, and repeated transitions through buildings with intermittent GPS loss and severe lighting changes that disrupt feature-based localization. Each experiment is repeated multiple times under similar conditions to assess resilience, showing consistent and reliable performance. NOVA achieves agile target following at speeds exceeding 50 km/h. These results show that high-speed vision-based tracking is possible in the wild using only onboard sensing, with no reliance on external localization or environment assumptions. 

---
# SIM-Net: A Multimodal Fusion Network Using Inferred 3D Object Shape Point Clouds from RGB Images for 2D Classification 

**Authors**: Youcef Sklab, Hanane Ariouat, Eric Chenin, Edi Prifti, Jean-Daniel Zucker  

**Link**: [PDF](https://arxiv.org/pdf/2506.18683)  

**Abstract**: We introduce the Shape-Image Multimodal Network (SIM-Net), a novel 2D image classification architecture that integrates 3D point cloud representations inferred directly from RGB images. Our key contribution lies in a pixel-to-point transformation that converts 2D object masks into 3D point clouds, enabling the fusion of texture-based and geometric features for enhanced classification performance. SIM-Net is particularly well-suited for the classification of digitized herbarium specimens (a task made challenging by heterogeneous backgrounds), non-plant elements, and occlusions that compromise conventional image-based models. To address these issues, SIM-Net employs a segmentation-based preprocessing step to extract object masks prior to 3D point cloud generation. The architecture comprises a CNN encoder for 2D image features and a PointNet-based encoder for geometric features, which are fused into a unified latent space. Experimental evaluations on herbarium datasets demonstrate that SIM-Net consistently outperforms ResNet101, achieving gains of up to 9.9% in accuracy and 12.3% in F-score. It also surpasses several transformer-based state-of-the-art architectures, highlighting the benefits of incorporating 3D structural reasoning into 2D image classification tasks. 

---
# Multi-Scale Spectral Attention Module-based Hyperspectral Segmentation in Autonomous Driving Scenarios 

**Authors**: Imad Ali Shah, Jiarong Li, Tim Brophy, Martin Glavin, Edward Jones, Enda Ward, Brian Deegan  

**Link**: [PDF](https://arxiv.org/pdf/2506.18682)  

**Abstract**: Recent advances in autonomous driving (AD) have highlighted the potential of Hyperspectral Imaging (HSI) for enhanced environmental perception, particularly in challenging weather and lighting conditions. However, efficiently processing its high-dimensional spectral data remains a significant challenge. This paper introduces a Multi-scale Spectral Attention Module (MSAM) that enhances spectral feature extraction through three parallel 1D convolutions with varying kernel sizes between 1 to 11, coupled with an adaptive feature aggregation mechanism. By integrating MSAM into UNet's skip connections (UNet-SC), our proposed UNet-MSAM achieves significant improvements in semantic segmentation performance across multiple HSI datasets: HyKo-VIS v2, HSI-Drive v2, and Hyperspectral City v2. Our comprehensive experiments demonstrate that with minimal computational overhead (on average 0.02% in parameters and 0.82% GFLOPS), UNet-MSAM consistently outperforms UNet-SC, achieving average improvements of 3.61% in mean IoU and 3.80% in mF1 across the three datasets. Through extensive ablation studies, we have established that multi-scale kernel combinations perform better than single-scale configurations. These findings demonstrate the potential of HSI processing for AD and provide valuable insights into designing robust, multi-scale spectral feature extractors for real-world applications. 

---
# Is There a Case for Conversation Optimized Tokenizers in Large Language Models? 

**Authors**: Raquel Ferrando, Javier Conde, Gonzalo Martínez, Pedro Reviriego  

**Link**: [PDF](https://arxiv.org/pdf/2506.18674)  

**Abstract**: The computational and energy costs of Large Language Models (LLMs) have increased exponentially driven by the growing model sizes and the massive adoption of LLMs by hundreds of millions of users. The unit cost of an LLM is the computation of a token. Therefore, the tokenizer plays an important role in the efficiency of a model, and they are carefully optimized to minimize the number of tokens for the text in their training corpus. One of the most popular applications of LLMs are chatbots that interact with users. A key observation is that, for those chatbots, what is important is the performance of the tokenizer in the user text input and the chatbot responses. Those are most likely different from the text in the training corpus. So, a question that immediately arises is whether there is a potential benefit in optimizing tokenizers for chatbot conversations. In this paper, this idea is explored for different tokenizers by using a publicly available corpus of chatbot conversations to redesign their vocabularies and evaluate their performance in this domain. The results show that conversation-optimized tokenizers consistently reduce the number of tokens in chatbot dialogues, which can lead to meaningful energy savings, in the range of 5% to 10% while having minimal or even slightly positive impact on tokenization efficiency for the original training corpus. 

---
# Benchmarking histopathology foundation models in a multi-center dataset for skin cancer subtyping 

**Authors**: Pablo Meseguer, Rocío del Amor, Valery Naranjo  

**Link**: [PDF](https://arxiv.org/pdf/2506.18668)  

**Abstract**: Pretraining on large-scale, in-domain datasets grants histopathology foundation models (FM) the ability to learn task-agnostic data representations, enhancing transfer learning on downstream tasks. In computational pathology, automated whole slide image analysis requires multiple instance learning (MIL) frameworks due to the gigapixel scale of the slides. The diversity among histopathology FMs has highlighted the need to design real-world challenges for evaluating their effectiveness. To bridge this gap, our work presents a novel benchmark for evaluating histopathology FMs as patch-level feature extractors within a MIL classification framework. For that purpose, we leverage the AI4SkIN dataset, a multi-center cohort encompassing slides with challenging cutaneous spindle cell neoplasm subtypes. We also define the Foundation Model - Silhouette Index (FM-SI), a novel metric to measure model consistency against distribution shifts. Our experimentation shows that extracting less biased features enhances classification performance, especially in similarity-based MIL classifiers. 

---
# Historical Report Guided Bi-modal Concurrent Learning for Pathology Report Generation 

**Authors**: Ling Zhang, Boxiang Yun, Qingli Li, Yan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.18658)  

**Abstract**: Automated pathology report generation from Whole Slide Images (WSIs) faces two key challenges: (1) lack of semantic content in visual features and (2) inherent information redundancy in WSIs. To address these issues, we propose a novel Historical Report Guided \textbf{Bi}-modal Concurrent Learning Framework for Pathology Report \textbf{Gen}eration (BiGen) emulating pathologists' diagnostic reasoning, consisting of: (1) A knowledge retrieval mechanism to provide rich semantic content, which retrieves WSI-relevant knowledge from pre-built medical knowledge bank by matching high-attention patches and (2) A bi-modal concurrent learning strategy instantiated via a learnable visual token and a learnable textual token to dynamically extract key visual features and retrieved knowledge, where weight-shared layers enable cross-modal alignment between visual features and knowledge features. Our multi-modal decoder integrates both modals for comprehensive diagnostic reports generation. Experiments on the PathText (BRCA) dataset demonstrate our framework's superiority, achieving state-of-the-art performance with 7.4\% relative improvement in NLP metrics and 19.1\% enhancement in classification metrics for Her-2 prediction versus existing methods. Ablation studies validate the necessity of our proposed modules, highlighting our method's ability to provide WSI-relevant rich semantic content and suppress information redundancy in WSIs. Code is publicly available at this https URL. 

---
# Federated Loss Exploration for Improved Convergence on Non-IID Data 

**Authors**: Christian Internò, Markus Olhofer, Yaochu Jin, Barbara Hammer  

**Link**: [PDF](https://arxiv.org/pdf/2506.18640)  

**Abstract**: Federated learning (FL) has emerged as a groundbreaking paradigm in machine learning (ML), offering privacy-preserving collaborative model training across diverse datasets. Despite its promise, FL faces significant hurdles in non-identically and independently distributed (non-IID) data scenarios, where most existing methods often struggle with data heterogeneity and lack robustness in performance. This paper introduces Federated Loss Exploration (FedLEx), an innovative approach specifically designed to tackle these challenges. FedLEx distinctively addresses the shortcomings of existing FL methods in non-IID settings by optimizing its learning behavior for scenarios in which assumptions about data heterogeneity are impractical or unknown. It employs a federated loss exploration technique, where clients contribute to a global guidance matrix by calculating gradient deviations for model parameters. This matrix serves as a strategic compass to guide clients' gradient updates in subsequent FL rounds, thereby fostering optimal parameter updates for the global model. FedLEx effectively navigates the complex loss surfaces inherent in non-IID data, enhancing knowledge transfer in an efficient manner, since only a small number of epochs and small amount of data are required to build a strong global guidance matrix that can achieve model convergence without the need for additional data sharing or data distribution statics in a large client scenario. Our extensive experiments with state-of-the art FL algorithms demonstrate significant improvements in performance, particularly under realistic non-IID conditions, thus highlighting FedLEx's potential to overcome critical barriers in diverse FL applications. 

---
# Granular-Ball-Induced Multiple Kernel K-Means 

**Authors**: Shuyin Xia, Yifan Wang, Lifeng Shen, Guoyin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.18637)  

**Abstract**: Most existing multi-kernel clustering algorithms, such as multi-kernel K-means, often struggle with computational efficiency and robustness when faced with complex data distributions. These challenges stem from their dependence on point-to-point relationships for optimization, which can lead to difficulty in accurately capturing data sets' inherent structure and diversity. Additionally, the intricate interplay between multiple kernels in such algorithms can further exacerbate these issues, effectively impacting their ability to cluster data points in high-dimensional spaces. In this paper, we leverage granular-ball computing to improve the multi-kernel clustering framework. The core of granular-ball computing is to adaptively fit data distribution by balls from coarse to acceptable levels. Each ball can enclose data points based on a density consistency measurement. Such ball-based data description thus improves the computational efficiency and the robustness to unknown noises. Specifically, based on granular-ball representations, we introduce the granular-ball kernel (GBK) and its corresponding granular-ball multi-kernel K-means framework (GB-MKKM) for efficient clustering. Using granular-ball relationships in multiple kernel spaces, the proposed GB-MKKM framework shows its superiority in efficiency and clustering performance in the empirical evaluation of various clustering tasks. 

---
# ReDit: Reward Dithering for Improved LLM Policy Optimization 

**Authors**: Chenxing Wei, Jiarui Yu, Ying Tiffany He, Hande Dong, Yao Shu, Fei Yu  

**Link**: [PDF](https://arxiv.org/pdf/2506.18631)  

**Abstract**: DeepSeek-R1 has successfully enhanced Large Language Model (LLM) reasoning capabilities through its rule-based reward system. While it's a ''perfect'' reward system that effectively mitigates reward hacking, such reward functions are often discrete. Our experimental observations suggest that discrete rewards can lead to gradient anomaly, unstable optimization, and slow convergence. To address this issue, we propose ReDit (Reward Dithering), a method that dithers the discrete reward signal by adding simple random noise. With this perturbed reward, exploratory gradients are continuously provided throughout the learning process, enabling smoother gradient updates and accelerating convergence. The injected noise also introduces stochasticity into flat reward regions, encouraging the model to explore novel policies and escape local optima. Experiments across diverse tasks demonstrate the effectiveness and efficiency of ReDit. On average, ReDit achieves performance comparable to vanilla GRPO with only approximately 10% the training steps, and furthermore, still exhibits a 4% performance improvement over vanilla GRPO when trained for a similar duration. Visualizations confirm significant mitigation of gradient issues with ReDit. Moreover, theoretical analyses are provided to further validate these advantages. 

---
# Multi-Agent Reinforcement Learning for Inverse Design in Photonic Integrated Circuits 

**Authors**: Yannik Mahlau, Maximilian Schier, Christoph Reinders, Frederik Schubert, Marco Bügling, Bodo Rosenhahn  

**Link**: [PDF](https://arxiv.org/pdf/2506.18627)  

**Abstract**: Inverse design of photonic integrated circuits (PICs) has traditionally relied on gradientbased optimization. However, this approach is prone to end up in local minima, which results in suboptimal design functionality. As interest in PICs increases due to their potential for addressing modern hardware demands through optical computing, more adaptive optimization algorithms are needed. We present a reinforcement learning (RL) environment as well as multi-agent RL algorithms for the design of PICs. By discretizing the design space into a grid, we formulate the design task as an optimization problem with thousands of binary variables. We consider multiple two- and three-dimensional design tasks that represent PIC components for an optical computing system. By decomposing the design space into thousands of individual agents, our algorithms are able to optimize designs with only a few thousand environment samples. They outperform previous state-of-the-art gradient-based optimization in both twoand three-dimensional design tasks. Our work may also serve as a benchmark for further exploration of sample-efficient RL for inverse design in photonics. 

---
# Frequency Control in Microgrids: An Adaptive Fuzzy-Neural-Network Virtual Synchronous Generator 

**Authors**: Waleed Breesam, Rezvan Alamian, Nima Tashakor, Brahim Elkhalil Youcefa, Stefan M. Goetz  

**Link**: [PDF](https://arxiv.org/pdf/2506.18611)  

**Abstract**: The reliance on distributed renewable energy has increased recently. As a result, power electronic-based distributed generators replaced synchronous generators which led to a change in the dynamic characteristics of the microgrid. Most critically, they reduced system inertia and damping. Virtual synchronous generators emulated in power electronics, which mimic the dynamic behaviour of synchronous generators, are meant to fix this problem. However, fixed virtual synchronous generator parameters cannot guarantee a frequency regulation within the acceptable tolerance range. Conversely, a dynamic adjustment of these virtual parameters promises robust solution with stable frequency. This paper proposes a method to adapt the inertia, damping, and droop parameters dynamically through a fuzzy neural network controller. This controller trains itself online to choose appropriate values for these virtual parameters. The proposed method can be applied to a typical AC microgrid by considering the penetration and impact of renewable energy sources. We study the system in a MATLAB/Simulink model and validate it experimentally in real time using hardware-in-the-loop based on an embedded ARM system (SAM3X8E, Cortex-M3). Compared to traditional and fuzzy logic controller methods, the results demonstrate that the proposed method significantly reduces the frequency deviation to less than 0.03 Hz and shortens the stabilizing/recovery time. 

---
# Simulation-Free Differential Dynamics through Neural Conservation Laws 

**Authors**: Mengjian Hua, Eric Vanden-Eijnden, Ricky T.Q. Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.18604)  

**Abstract**: We present a novel simulation-free framework for training continuous-time diffusion processes over very general objective functions. Existing methods typically involve either prescribing the optimal diffusion process -- which only works for heavily restricted problem formulations -- or require expensive simulation to numerically obtain the time-dependent densities and sample from the diffusion process. In contrast, we propose a coupled parameterization which jointly models a time-dependent density function, or probability path, and the dynamics of a diffusion process that generates this probability path. To accomplish this, our approach directly bakes in the Fokker-Planck equation and density function requirements as hard constraints, by extending and greatly simplifying the construction of Neural Conservation Laws. This enables simulation-free training for a large variety of problem formulations, from data-driven objectives as in generative modeling and dynamical optimal transport, to optimality-based objectives as in stochastic optimal control, with straightforward extensions to mean-field objectives due to the ease of accessing exact density functions. We validate our method in a diverse range of application domains from modeling spatio-temporal events to learning optimal dynamics from population data. 

---
# BulletGen: Improving 4D Reconstruction with Bullet-Time Generation 

**Authors**: Denys Rozumnyi, Jonathon Luiten, Numair Khan, Johannes Schönberger, Peter Kontschieder  

**Link**: [PDF](https://arxiv.org/pdf/2506.18601)  

**Abstract**: Transforming casually captured, monocular videos into fully immersive dynamic experiences is a highly ill-posed task, and comes with significant challenges, e.g., reconstructing unseen regions, and dealing with the ambiguity in monocular depth estimation. In this work we introduce BulletGen, an approach that takes advantage of generative models to correct errors and complete missing information in a Gaussian-based dynamic scene representation. This is done by aligning the output of a diffusion-based video generation model with the 4D reconstruction at a single frozen "bullet-time" step. The generated frames are then used to supervise the optimization of the 4D Gaussian model. Our method seamlessly blends generative content with both static and dynamic scene components, achieving state-of-the-art results on both novel-view synthesis, and 2D/3D tracking tasks. 

---
# Optimization-Induced Dynamics of Lipschitz Continuity in Neural Networks 

**Authors**: Róisín Luo, James McDermott, Christian Gagné, Qiang Sun, Colm O'Riordan  

**Link**: [PDF](https://arxiv.org/pdf/2506.18588)  

**Abstract**: Lipschitz continuity characterizes the worst-case sensitivity of neural networks to small input perturbations; yet its dynamics (i.e. temporal evolution) during training remains under-explored. We present a rigorous mathematical framework to model the temporal evolution of Lipschitz continuity during training with stochastic gradient descent (SGD). This framework leverages a system of stochastic differential equations (SDEs) to capture both deterministic and stochastic forces. Our theoretical analysis identifies three principal factors driving the evolution: (i) the projection of gradient flows, induced by the optimization dynamics, onto the operator-norm Jacobian of parameter matrices; (ii) the projection of gradient noise, arising from the randomness in mini-batch sampling, onto the operator-norm Jacobian; and (iii) the projection of the gradient noise onto the operator-norm Hessian of parameter matrices. Furthermore, our theoretical framework sheds light on such as how noisy supervision, parameter initialization, batch size, and mini-batch sampling trajectories, among other factors, shape the evolution of the Lipschitz continuity of neural networks. Our experimental results demonstrate strong agreement between the theoretical implications and the observed behaviors. 

---
# Security Assessment of DeepSeek and GPT Series Models against Jailbreak Attacks 

**Authors**: Xiaodong Wu, Xiangman Li, Jianbing Ni  

**Link**: [PDF](https://arxiv.org/pdf/2506.18543)  

**Abstract**: The widespread deployment of large language models (LLMs) has raised critical concerns over their vulnerability to jailbreak attacks, i.e., adversarial prompts that bypass alignment mechanisms and elicit harmful or policy-violating outputs. While proprietary models like GPT-4 have undergone extensive evaluation, the robustness of emerging open-source alternatives such as DeepSeek remains largely underexplored, despite their growing adoption in real-world applications. In this paper, we present the first systematic jailbreak evaluation of DeepSeek-series models, comparing them with GPT-3.5 and GPT-4 using the HarmBench benchmark. We evaluate seven representative attack strategies across 510 harmful behaviors categorized by both function and semantic domain. Our analysis reveals that DeepSeek's Mixture-of-Experts (MoE) architecture introduces routing sparsity that offers selective robustness against optimization-based attacks such as TAP-T, but leads to significantly higher vulnerability under prompt-based and manually engineered attacks. In contrast, GPT-4 Turbo demonstrates stronger and more consistent safety alignment across diverse behaviors, likely due to its dense Transformer design and reinforcement learning from human feedback. Fine-grained behavioral analysis and case studies further show that DeepSeek often routes adversarial prompts to under-aligned expert modules, resulting in inconsistent refusal behaviors. These findings highlight a fundamental trade-off between architectural efficiency and alignment generalization, emphasizing the need for targeted safety tuning and modular alignment strategies to ensure secure deployment of open-source LLMs. 

---
# Embedded FPGA Acceleration of Brain-Like Neural Networks: Online Learning to Scalable Inference 

**Authors**: Muhammad Ihsan Al Hafiz, Naresh Ravichandran, Anders Lansner, Pawel Herman, Artur Podobas  

**Link**: [PDF](https://arxiv.org/pdf/2506.18530)  

**Abstract**: Edge AI applications increasingly require models that can learn and adapt on-device with minimal energy budget. Traditional deep learning models, while powerful, are often overparameterized, energy-hungry, and dependent on cloud connectivity. Brain-Like Neural Networks (BLNNs), such as the Bayesian Confidence Propagation Neural Network (BCPNN), propose a neuromorphic alternative by mimicking cortical architecture and biologically-constrained learning. They offer sparse architectures with local learning rules and unsupervised/semi-supervised learning, making them well-suited for low-power edge intelligence. However, existing BCPNN implementations rely on GPUs or datacenter FPGAs, limiting their applicability to embedded systems. This work presents the first embedded FPGA accelerator for BCPNN on a Zynq UltraScale+ SoC using High-Level Synthesis. We implement both online learning and inference-only kernels with support for variable and mixed precision. Evaluated on MNIST, Pneumonia, and Breast Cancer datasets, our accelerator achieves up to 17.5x latency and 94% energy savings over ARM baselines, without sacrificing accuracy. This work enables practical neuromorphic computing on edge devices, bridging the gap between brain-like learning and real-world deployment. 

---
# Smooth Operators: LLMs Translating Imperfect Hints into Disfluency-Rich Transcripts 

**Authors**: Duygu Altinok  

**Link**: [PDF](https://arxiv.org/pdf/2506.18510)  

**Abstract**: Accurate detection of disfluencies in spoken language is crucial for enhancing the performance of automatic speech and language processing systems, as well as fostering the development of more inclusive speech and language technologies. Leveraging the growing trend of large language models (LLMs) as versatile learners capable of processing both lexical and non-lexical inputs (e.g., audio and video), we propose a novel approach to transcribing disfluencies as explicit tokens with timestamps, enabling the generation of fully annotated disfluency-rich transcripts. Our method integrates acoustic representations extracted from an audio encoder with textual inputs of varying quality: clean transcriptions without disfluencies, time-aligned transcriptions from aligners, or outputs from phoneme-based ASR models -- all of which may contain imperfections. Importantly, our experiments demonstrate that textual inputs do not need to be flawless. As long as they include timestamp-related cues, LLMs can effectively smooth the input and produce fully disfluency-annotated transcripts, underscoring their robustness in handling imperfect hints. 

---
# Generalizing Vision-Language Models to Novel Domains: A Comprehensive Survey 

**Authors**: Xinyao Li, Jingjing Li, Fengling Li, Lei Zhu, Yang Yang, Heng Tao Shen  

**Link**: [PDF](https://arxiv.org/pdf/2506.18504)  

**Abstract**: Recently, vision-language pretraining has emerged as a transformative technique that integrates the strengths of both visual and textual modalities, resulting in powerful vision-language models (VLMs). Leveraging web-scale pretraining data, these models exhibit strong zero-shot capabilities. However, their performance often deteriorates when confronted with domain-specific or specialized generalization tasks. To address this, a growing body of research focuses on transferring or generalizing the rich knowledge embedded in VLMs to various downstream applications. This survey aims to comprehensively summarize the generalization settings, methodologies, benchmarking and results in VLM literatures. Delving into the typical VLM structures, current literatures are categorized into prompt-based, parameter-based and feature-based methods according to the transferred modules. The differences and characteristics in each category are furthered summarized and discussed by revisiting the typical transfer learning (TL) settings, providing novel interpretations for TL in the era of VLMs. Popular benchmarks for VLM generalization are further introduced with thorough performance comparisons among the reviewed methods. Following the advances in large-scale generalizable pretraining, this survey also discusses the relations and differences between VLMs and up-to-date multimodal large language models (MLLM), e.g., DeepSeek-VL. By systematically reviewing the surging literatures in vision-language research from a novel and practical generalization prospective, this survey contributes to a clear landscape of current and future multimodal researches. 

---
# Comparative Evaluation of ChatGPT and DeepSeek Across Key NLP Tasks: Strengths, Weaknesses, and Domain-Specific Performance 

**Authors**: Wael Etaiwi, Bushra Alhijawi  

**Link**: [PDF](https://arxiv.org/pdf/2506.18501)  

**Abstract**: The increasing use of large language models (LLMs) in natural language processing (NLP) tasks has sparked significant interest in evaluating their effectiveness across diverse applications. While models like ChatGPT and DeepSeek have shown strong results in many NLP domains, a comprehensive evaluation is needed to understand their strengths, weaknesses, and domain-specific abilities. This is critical as these models are applied to various tasks, from sentiment analysis to more nuanced tasks like textual entailment and translation. This study aims to evaluate ChatGPT and DeepSeek across five key NLP tasks: sentiment analysis, topic classification, text summarization, machine translation, and textual entailment. A structured experimental protocol is used to ensure fairness and minimize variability. Both models are tested with identical, neutral prompts and evaluated on two benchmark datasets per task, covering domains like news, reviews, and formal/informal texts. The results show that DeepSeek excels in classification stability and logical reasoning, while ChatGPT performs better in tasks requiring nuanced understanding and flexibility. These findings provide valuable insights for selecting the appropriate LLM based on task requirements. 

---
# PuckTrick: A Library for Making Synthetic Data More Realistic 

**Authors**: Alessandra Agostini, Andrea Maurino, Blerina Spahiu  

**Link**: [PDF](https://arxiv.org/pdf/2506.18499)  

**Abstract**: The increasing reliance on machine learning (ML) models for decision-making requires high-quality training data. However, access to real-world datasets is often restricted due to privacy concerns, proprietary restrictions, and incomplete data availability. As a result, synthetic data generation (SDG) has emerged as a viable alternative, enabling the creation of artificial datasets that preserve the statistical properties of real data while ensuring privacy compliance. Despite its advantages, synthetic data is often overly clean and lacks real-world imperfections, such as missing values, noise, outliers, and misclassified labels, which can significantly impact model generalization and robustness. To address this limitation, we introduce Pucktrick, a Python library designed to systematically contaminate synthetic datasets by introducing controlled errors. The library supports multiple error types, including missing data, noisy values, outliers, label misclassification, duplication, and class imbalance, offering a structured approach to evaluating ML model resilience under real-world data imperfections. Pucktrick provides two contamination modes: one for injecting errors into clean datasets and another for further corrupting already contaminated datasets. Through extensive experiments on real-world financial datasets, we evaluate the impact of systematic data contamination on model performance. Our findings demonstrate that ML models trained on contaminated synthetic data outperform those trained on purely synthetic, error-free data, particularly for tree-based and linear models such as SVMs and Extra Trees. 

---
# AI-Generated Song Detection via Lyrics Transcripts 

**Authors**: Markus Frohmann, Elena V. Epure, Gabriel Meseguer-Brocal, Markus Schedl, Romain Hennequin  

**Link**: [PDF](https://arxiv.org/pdf/2506.18488)  

**Abstract**: The recent rise in capabilities of AI-based music generation tools has created an upheaval in the music industry, necessitating the creation of accurate methods to detect such AI-generated content. This can be done using audio-based detectors; however, it has been shown that they struggle to generalize to unseen generators or when the audio is perturbed. Furthermore, recent work used accurate and cleanly formatted lyrics sourced from a lyrics provider database to detect AI-generated music. However, in practice, such perfect lyrics are not available (only the audio is); this leaves a substantial gap in applicability in real-life use cases. In this work, we instead propose solving this gap by transcribing songs using general automatic speech recognition (ASR) models. We do this using several detectors. The results on diverse, multi-genre, and multi-lingual lyrics show generally strong detection performance across languages and genres, particularly for our best-performing model using Whisper large-v2 and LLM2Vec embeddings. In addition, we show that our method is more robust than state-of-the-art audio-based ones when the audio is perturbed in different ways and when evaluated on different music generators. Our code is available at this https URL. 

---
# MeRF: Motivation-enhanced Reinforcement Finetuning for Large Reasoning Models 

**Authors**: Junjie Zhang, Guozheng Ma, Shunyu Liu, Haoyu Wang, Jiaxing Huang, Ting-En Lin, Fei Huang, Yongbin Li, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2506.18485)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as a powerful learn-to-reason paradigm for Large Language Models (LLMs) to tackle complex reasoning tasks. However, existing RLVR methods overlook one of the most distinctive capabilities of LLMs, their in-context learning ability, as prominently demonstrated by the success of Chain-of-Thought (CoT) prompting. This motivates us to explore how reinforcement learning can be effectively combined with in-context learning to better improve the reasoning capabilities of LLMs. In this paper, we introduce Motivation-enhanced Reinforcement Finetuning} (MeRF), an intuitive yet effective method enhancing reinforcement learning of LLMs by involving ``telling LLMs the rules of the game''. Specifically, MeRF directly injects the reward specification into the prompt, which serves as an in-context motivation for model to improve its responses with awareness of the optimization objective. This simple modification leverages the in-context learning ability of LLMs aligning generation with optimization, thereby incentivizing the model to generate desired outputs from both inner motivation and external reward. Empirical evaluations on the Knights and Knaves~(K&K) logic puzzle reasoning benchmark demonstrate that \texttt{MeRF} achieves substantial performance gains over baselines. Moreover, ablation studies show that performance improves with greater consistency between the in-context motivation and the external reward function, while the model also demonstrates an ability to adapt to misleading motivations through reinforcement learning. 

---
# A Deep Convolutional Neural Network-Based Novel Class Balancing for Imbalance Data Segmentation 

**Authors**: Atifa Kalsoom, M.A. Iftikhar, Amjad Ali, Zubair Shah, Shidin Balakrishnan, Hazrat Ali  

**Link**: [PDF](https://arxiv.org/pdf/2506.18474)  

**Abstract**: Retinal fundus images provide valuable insights into the human eye's interior structure and crucial features, such as blood vessels, optic disk, macula, and fovea. However, accurate segmentation of retinal blood vessels can be challenging due to imbalanced data distribution and varying vessel thickness. In this paper, we propose BLCB-CNN, a novel pipeline based on deep learning and bi-level class balancing scheme to achieve vessel segmentation in retinal fundus images. The BLCB-CNN scheme uses a Convolutional Neural Network (CNN) architecture and an empirical approach to balance the distribution of pixels across vessel and non-vessel classes and within thin and thick vessels. Level-I is used for vessel/non-vessel balancing and Level-II is used for thick/thin vessel balancing. Additionally, pre-processing of the input retinal fundus image is performed by Global Contrast Normalization (GCN), Contrast Limited Adaptive Histogram Equalization (CLAHE), and gamma corrections to increase intensity uniformity as well as to enhance the contrast between vessels and background pixels. The resulting balanced dataset is used for classification-based segmentation of the retinal vascular tree. We evaluate the proposed scheme on standard retinal fundus images and achieve superior performance measures, including an area under the ROC curve of 98.23%, Accuracy of 96.22%, Sensitivity of 81.57%, and Specificity of 97.65%. We also demonstrate the method's efficacy through external cross-validation on STARE images, confirming its generalization ability. 

---
# Benchmarking Foundation Models and Parameter-Efficient Fine-Tuning for Prognosis Prediction in Medical Imaging 

**Authors**: Filippo Ruffini, Elena Mulero Ayllon, Linlin Shen, Paolo Soda, Valerio Guarrasi  

**Link**: [PDF](https://arxiv.org/pdf/2506.18434)  

**Abstract**: Artificial Intelligence (AI) holds significant promise for improving prognosis prediction in medical imaging, yet its effective application remains challenging. In this work, we introduce a structured benchmark explicitly designed to evaluate and compare the transferability of Convolutional Neural Networks and Foundation Models in predicting clinical outcomes in COVID-19 patients, leveraging diverse publicly available Chest X-ray datasets. Our experimental methodology extensively explores a wide set of fine-tuning strategies, encompassing traditional approaches such as Full Fine-Tuning and Linear Probing, as well as advanced Parameter-Efficient Fine-Tuning methods including Low-Rank Adaptation, BitFit, VeRA, and IA3. The evaluations were conducted across multiple learning paradigms, including both extensive full-data scenarios and more clinically realistic Few-Shot Learning settings, which are critical for modeling rare disease outcomes and rapidly emerging health threats. By implementing a large-scale comparative analysis involving a diverse selection of pretrained models, including general-purpose architectures pretrained on large-scale datasets such as CLIP and DINOv2, to biomedical-specific models like MedCLIP, BioMedCLIP, and PubMedCLIP, we rigorously assess each model's capacity to effectively adapt and generalize to prognosis tasks, particularly under conditions of severe data scarcity and pronounced class imbalance. The benchmark was designed to capture critical conditions common in prognosis tasks, including variations in dataset size and class distribution, providing detailed insights into the strengths and limitations of each fine-tuning strategy. This extensive and structured evaluation aims to inform the practical deployment and adoption of robust, efficient, and generalizable AI-driven solutions in real-world clinical prognosis prediction workflows. 

---
# TReB: A Comprehensive Benchmark for Evaluating Table Reasoning Capabilities of Large Language Models 

**Authors**: Ce Li, Xiaofan Liu, Zhiyan Song, Ce Chi, Chen Zhao, Jingjing Yang, Zhendong Wang, Kexin Yang, Boshen Shi, Xing Wang, Chao Deng, Junlan Feng  

**Link**: [PDF](https://arxiv.org/pdf/2506.18421)  

**Abstract**: The majority of data in businesses and industries is stored in tables, databases, and data warehouses. Reasoning with table-structured data poses significant challenges for large language models (LLMs) due to its hidden semantics, inherent complexity, and structured nature. One of these challenges is lacking an effective evaluation benchmark fairly reflecting the performances of LLMs on broad table reasoning abilities. In this paper, we fill in this gap, presenting a comprehensive table reasoning evolution benchmark, TReB, which measures both shallow table understanding abilities and deep table reasoning abilities, a total of 26 sub-tasks. We construct a high quality dataset through an iterative data processing procedure. We create an evaluation framework to robustly measure table reasoning capabilities with three distinct inference modes, TCoT, PoT and ICoT. Further, we benchmark over 20 state-of-the-art LLMs using this frame work and prove its effectiveness. Experimental results reveal that existing LLMs still have significant room for improvement in addressing the complex and real world Table related tasks. Both the dataset and evaluation framework are publicly available, with the dataset hosted on [HuggingFace] and the framework on [GitHub]. 

---
# Latent Space Analysis for Melanoma Prevention 

**Authors**: Ciro Listone, Aniello Murano  

**Link**: [PDF](https://arxiv.org/pdf/2506.18414)  

**Abstract**: Melanoma represents a critical health risk due to its aggressive progression and high mortality, underscoring the need for early, interpretable diagnostic tools. While deep learning has advanced in skin lesion classification, most existing models provide only binary outputs, offering limited clinical insight. This work introduces a novel approach that extends beyond classification, enabling interpretable risk modelling through a Conditional Variational Autoencoder. The proposed method learns a structured latent space that captures semantic relationships among lesions, allowing for a nuanced, continuous assessment of morphological differences. An SVM is also trained on this representation effectively differentiating between benign nevi and melanomas, demonstrating strong and consistent performance. More importantly, the learned latent space supports visual and geometric interpretation of malignancy, with the spatial proximity of a lesion to known melanomas serving as a meaningful indicator of risk. This approach bridges predictive performance with clinical applicability, fostering early detection, highlighting ambiguous cases, and enhancing trust in AI-assisted diagnosis through transparent and interpretable decision-making. 

---
# The Debugging Decay Index: Rethinking Debugging Strategies for Code LLMs 

**Authors**: Muntasir Adnan, Carlos C. N. Kuhn  

**Link**: [PDF](https://arxiv.org/pdf/2506.18403)  

**Abstract**: The effectiveness of AI debugging follows a predictable exponential decay pattern; most models lose 60-80% of their debugging capability within just 2-3 attempts, despite iterative debugging being a critical capability for practical code generation systems. We introduce the Debugging Decay Index (DDI), a mathematical framework that quantifies when debugging becomes ineffective and predicts intervention points. Our strategic fresh start approach shifts from exploitation to exploration at strategic points in the debugging process, demonstrating that well-timed interventions can rescue the effectiveness of debugging. DDI reveals a fundamental limitation in current AI debugging and provides the first quantitative framework for optimising iterative code generation strategies. 

---
# ADNF-Clustering: An Adaptive and Dynamic Neuro-Fuzzy Clustering for Leukemia Prediction 

**Authors**: Marco Aruta, Ciro Listone, Giuseppe Murano, Aniello Murano  

**Link**: [PDF](https://arxiv.org/pdf/2506.18396)  

**Abstract**: Leukemia diagnosis and monitoring rely increasingly on high-throughput image data, yet conventional clustering methods lack the flexibility to accommodate evolving cellular patterns and quantify uncertainty in real time. We introduce Adaptive and Dynamic Neuro-Fuzzy Clustering, a novel streaming-capable framework that combines Convolutional Neural Network-based feature extraction with an online fuzzy clustering engine. ADNF initializes soft partitions via Fuzzy C-Means, then continuously updates micro-cluster centers, densities, and fuzziness parameters using a Fuzzy Temporal Index (FTI) that measures entropy evolution. A topology refinement stage performs density-weighted merging and entropy-guided splitting to guard against over- and under-segmentation. On the C-NMC leukemia microscopy dataset, our tool achieves a silhouette score of 0.51, demonstrating superior cohesion and separation over static baselines. The method's adaptive uncertainty modeling and label-free operation hold immediate potential for integration within the INFANT pediatric oncology network, enabling scalable, up-to-date support for personalized leukemia management. 

---
# Evaluating Causal Explanation in Medical Reports with LLM-Based and Human-Aligned Metrics 

**Authors**: Yousang Cho, Key-Sun Choi  

**Link**: [PDF](https://arxiv.org/pdf/2506.18387)  

**Abstract**: This study investigates how accurately different evaluation metrics capture the quality of causal explanations in automatically generated diagnostic reports. We compare six metrics: BERTScore, Cosine Similarity, BioSentVec, GPT-White, GPT-Black, and expert qualitative assessment across two input types: observation-based and multiple-choice-based report generation. Two weighting strategies are applied: one reflecting task-specific priorities, and the other assigning equal weights to all metrics. Our results show that GPT-Black demonstrates the strongest discriminative power in identifying logically coherent and clinically valid causal narratives. GPT-White also aligns well with expert evaluations, while similarity-based metrics diverge from clinical reasoning quality. These findings emphasize the impact of metric selection and weighting on evaluation outcomes, supporting the use of LLM-based evaluation for tasks requiring interpretability and causal reasoning. 

---
# LOGICPO: Efficient Translation of NL-based Logical Problems to FOL using LLMs and Preference Optimization 

**Authors**: Koushik Viswanadha, Deepanway Ghosal, Somak Aditya  

**Link**: [PDF](https://arxiv.org/pdf/2506.18383)  

**Abstract**: Logical reasoning is a key task for artificial intelligence due to it's role in major downstream tasks such as Question Answering, Summarization. Recent methods in improving the reasoning ability of LLMs fall short in correctly converting a natural language reasoning problem to an equivalent logical formulation, which hinders the framework's overall ability to reason. Towards this, we propose to use finetuning on a preference optimization dataset to learn to parse and represent a natural language problem as a whole to a consistent logical program by 1) introducing a new supervised and preference optimization dataset LogicPO, and 2) adopting popular techniques such as Direct Preference Optimization (DPO), Kahneman-Tversky optimization (KTO) to finetune open-source LLMs. Our best model with Phi-3.5 consistently outperforms GPT-3.5-turbo's (8-shot) by producing 10% more logically correct and with 14% less syntax errors. Through the framework and our improved evaluation metrics, we offer a promising direction in improving the logical reasoning of LLMs by better representing them in their logical formulations. 

---
# PERSCEN: Learning Personalized Interaction Pattern and Scenario Preference for Multi-Scenario Matching 

**Authors**: Haotong Du, Yaqing Wang, Fei Xiong, Lei Shao, Ming Liu, Hao Gu, Quanming Yao, Zhen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.18382)  

**Abstract**: With the expansion of business scales and scopes on online platforms, multi-scenario matching has become a mainstream solution to reduce maintenance costs and alleviate data sparsity. The key to effective multi-scenario recommendation lies in capturing both user preferences shared across all scenarios and scenario-aware preferences specific to each scenario. However, existing methods often overlook user-specific modeling, limiting the generation of personalized user representations. To address this, we propose PERSCEN, an innovative approach that incorporates user-specific modeling into multi-scenario matching. PERSCEN constructs a user-specific feature graph based on user characteristics and employs a lightweight graph neural network to capture higher-order interaction patterns, enabling personalized extraction of preferences shared across scenarios. Additionally, we leverage vector quantization techniques to distil scenario-aware preferences from users' behavior sequence within individual scenarios, facilitating user-specific and scenario-aware preference modeling. To enhance efficient and flexible information transfer, we introduce a progressive scenario-aware gated linear unit that allows fine-grained, low-latency fusion. Extensive experiments demonstrate that PERSCEN outperforms existing methods. Further efficiency analysis confirms that PERSCEN effectively balances performance with computational cost, ensuring its practicality for real-world industrial systems. 

---
# Robots and Children that Learn Together : Improving Knowledge Retention by Teaching Peer-Like Interactive Robots 

**Authors**: Imene Tarakli, Samuele Vinanzi, Richard Moore, Alessandro Di Nuovo  

**Link**: [PDF](https://arxiv.org/pdf/2506.18365)  

**Abstract**: Despite growing interest in Learning-by-Teaching (LbT), few studies have explored how this paradigm can be implemented with autonomous, peer-like social robots in real classrooms. Most prior work has relied on scripted or Wizard-of-Oz behaviors, limiting our understanding of how real-time, interactive learning can be supported by artificial agents. This study addresses this gap by introducing Interactive Reinforcement Learning (RL) as a cognitive model for teachable social robots. We conducted two between-subject experiments with 58 primary school children, who either taught a robot or practiced independently on a tablet while learning French vocabulary (memorization) and grammatical rules (inference). The robot, powered by Interactive RL, learned from the child's evaluative feedback. Children in the LbT condition achieved significantly higher retention gains compared to those in the self-practice condition, especially on the grammar task. Learners with lower prior knowledge benefited most from teaching the robot. Behavioural metrics revealed that children adapted their teaching strategies over time and engaged more deeply during inference tasks. This work makes two contributions: (1) it introduces Interactive RL as a pedagogically effective and scalable model for peer-robot learning, and (2) it demonstrates, for the first time, the feasibility of deploying multiple autonomous robots simultaneously in real classrooms. These findings extend theoretical understanding of LbT by showing that social robots can function not only as passive tutees but as adaptive partners that enhance meta-cognitive engagement and long-term learning outcomes. 

---
# Controlled Generation with Equivariant Variational Flow Matching 

**Authors**: Floor Eijkelboom, Heiko Zimmermann, Sharvaree Vadgama, Erik J Bekkers, Max Welling, Christian A. Naesseth, Jan-Willem van de Meent  

**Link**: [PDF](https://arxiv.org/pdf/2506.18340)  

**Abstract**: We derive a controlled generation objective within the framework of Variational Flow Matching (VFM), which casts flow matching as a variational inference problem. We demonstrate that controlled generation can be implemented two ways: (1) by way of end-to-end training of conditional generative models, or (2) as a Bayesian inference problem, enabling post hoc control of unconditional models without retraining. Furthermore, we establish the conditions required for equivariant generation and provide an equivariant formulation of VFM tailored for molecular generation, ensuring invariance to rotations, translations, and permutations. We evaluate our approach on both uncontrolled and controlled molecular generation, achieving state-of-the-art performance on uncontrolled generation and outperforming state-of-the-art models in controlled generation, both with end-to-end training and in the Bayesian inference setting. This work strengthens the connection between flow-based generative modeling and Bayesian inference, offering a scalable and principled framework for constraint-driven and symmetry-aware generation. 

---
# Structured Kolmogorov-Arnold Neural ODEs for Interpretable Learning and Symbolic Discovery of Nonlinear Dynamics 

**Authors**: Wei Liu, Kiran Bacsa, Loon Ching Tang, Eleni Chatzi  

**Link**: [PDF](https://arxiv.org/pdf/2506.18339)  

**Abstract**: Understanding and modeling nonlinear dynamical systems is a fundamental problem across scientific and engineering domains. While deep learning has demonstrated remarkable potential for learning complex system behavior, achieving models that are both highly accurate and physically interpretable remains a major challenge. To address this, we propose Structured Kolmogorov-Arnold Neural ODEs (SKANODEs), a novel framework that integrates structured state-space modeling with the Kolmogorov-Arnold Network (KAN). SKANODE first employs a fully trainable KAN as a universal function approximator within a structured Neural ODE framework to perform virtual sensing, recovering latent states that correspond to physically interpretable quantities such as positions and velocities. Once this structured latent representation is established, we exploit the symbolic regression capability of KAN to extract compact and interpretable expressions for the system's governing dynamics. The resulting symbolic expression is then substituted back into the Neural ODE framework and further calibrated through continued training to refine its coefficients, enhancing both the precision of the discovered equations and the predictive accuracy of system responses. Extensive experiments on both simulated and real-world systems demonstrate that SKANODE achieves superior performance while offering interpretable, physics-consistent models that uncover the underlying mechanisms of nonlinear dynamical systems. 

---
# Confucius3-Math: A Lightweight High-Performance Reasoning LLM for Chinese K-12 Mathematics Learning 

**Authors**: Lixin Wu, Na Cai, Qiao Cheng, Jiachen Wang, Yitao Duan  

**Link**: [PDF](https://arxiv.org/pdf/2506.18330)  

**Abstract**: We introduce Confucius3-Math, an open-source large language model with 14B parameters that (1) runs efficiently on a single consumer-grade GPU; (2) achieves SOTA performances on a range of mathematical reasoning tasks, outperforming many models with significantly larger sizes. In particular, as part of our mission to enhancing education and knowledge dissemination with AI, Confucius3-Math is specifically committed to mathematics learning for Chinese K-12 students and educators. Built via post-training with large-scale reinforcement learning (RL), Confucius3-Math aligns with national curriculum and excels at solving main-stream Chinese K-12 mathematical problems with low cost. In this report we share our development recipe, the challenges we encounter and the techniques we develop to overcome them. In particular, we introduce three technical innovations: Targeted Entropy Regularization, Recent Sample Recovery and Policy-Specific Hardness Weighting. These innovations encompass a new entropy regularization, a novel data scheduling policy, and an improved group-relative advantage estimator. Collectively, they significantly stabilize the RL training, improve data efficiency, and boost performance. Our work demonstrates the feasibility of building strong reasoning models in a particular domain at low cost. We open-source our model and code at this https URL. 

---
# Bias vs Bias -- Dawn of Justice: A Fair Fight in Recommendation Systems 

**Authors**: Tahsin Alamgir Kheya, Mohamed Reda Bouadjenek, Sunil Aryal  

**Link**: [PDF](https://arxiv.org/pdf/2506.18327)  

**Abstract**: Recommendation systems play a crucial role in our daily lives by impacting user experience across various domains, including e-commerce, job advertisements, entertainment, etc. Given the vital role of such systems in our lives, practitioners must ensure they do not produce unfair and imbalanced recommendations. Previous work addressing bias in recommendations overlooked bias in certain item categories, potentially leaving some biases unaddressed. Additionally, most previous work on fair re-ranking focused on binary-sensitive attributes. In this paper, we address these issues by proposing a fairness-aware re-ranking approach that helps mitigate bias in different categories of items. This re-ranking approach leverages existing biases to correct disparities in recommendations across various demographic groups. We show how our approach can mitigate bias on multiple sensitive attributes, including gender, age, and occupation. We experimented on three real-world datasets to evaluate the effectiveness of our re-ranking scheme in mitigating bias in recommendations. Our results show how this approach helps mitigate social bias with little to no degradation in performance. 

---
# A Multi-Scale Spatial Attention-Based Zero-Shot Learning Framework for Low-Light Image Enhancement 

**Authors**: Muhammad Azeem Aslam, Hassan Khalid, Nisar Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2506.18323)  

**Abstract**: Low-light image enhancement remains a challenging task, particularly in the absence of paired training data. In this study, we present LucentVisionNet, a novel zero-shot learning framework that addresses the limitations of traditional and deep learning-based enhancement methods. The proposed approach integrates multi-scale spatial attention with a deep curve estimation network, enabling fine-grained enhancement while preserving semantic and perceptual fidelity. To further improve generalization, we adopt a recurrent enhancement strategy and optimize the model using a composite loss function comprising six tailored components, including a novel no-reference image quality loss inspired by human visual perception. Extensive experiments on both paired and unpaired benchmark datasets demonstrate that LucentVisionNet consistently outperforms state-of-the-art supervised, unsupervised, and zero-shot methods across multiple full-reference and no-reference image quality metrics. Our framework achieves high visual quality, structural consistency, and computational efficiency, making it well-suited for deployment in real-world applications such as mobile photography, surveillance, and autonomous navigation. 

---
# Use Property-Based Testing to Bridge LLM Code Generation and Validation 

**Authors**: Lehan He, Zeren Chen, Zhe Zhang, Jing Shao, Xiang Gao, Lu Sheng  

**Link**: [PDF](https://arxiv.org/pdf/2506.18315)  

**Abstract**: Large Language Models (LLMs) excel at code generation, but ensuring their outputs to be functionally correct, especially in complex programming tasks, is a persistent challenge. While traditional Test-Driven Development (TDD) offers a path for code refinement, its efficacy with LLMs is often undermined by the scarcity of high-quality test cases or the pitfalls of automated test generation, including biased tests or inaccurate output predictions that can misdirect the correction process. This paper introduces Property-Generated Solver, a novel framework that leverages Property-Based Testing (PBT) to validate high-level program properties or invariants, instead of relying on specific input-output examples. These properties are often simpler to define and verify than directly predicting exhaustive test oracles, breaking the "cycle of self-deception" where tests might share flaws with the code they are meant to validate. Property-Generated Solver employs two collaborative LLM-based agents: a Generator dedicated to code generation and iterative refinement, and a Tester that manages the PBT life-cycle and formulate semantically rich feedback from property violations. The resulting comprehensive and actionable feedback then guides the Generator in its refinement efforts. By establishing PBT as the core validation engine within this iterative, closed-loop paradigm, Property-Generated Solver provides a robust mechanism for steering LLMs towards more correct and generalizable code. Extensive experimental results on multiple code generation benchmarks demonstrate that Property-Generated Solver achieves substantial pass@1 improvements, ranging from 23.1% to 37.3% relative gains over established TDD methods. 

---
# LettinGo: Explore User Profile Generation for Recommendation System 

**Authors**: Lu Wang, Di Zhang, Fangkai Yang, Pu Zhao, Jianfeng Liu, Yuefeng Zhan, Hao Sun, Qingwei Lin, Weiwei Deng, Dongmei Zhang, Feng Sun, Qi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.18309)  

**Abstract**: User profiling is pivotal for recommendation systems, as it transforms raw user interaction data into concise and structured representations that drive personalized recommendations. While traditional embedding-based profiles lack interpretability and adaptability, recent advances with large language models (LLMs) enable text-based profiles that are semantically richer and more transparent. However, existing methods often adhere to fixed formats that limit their ability to capture the full diversity of user behaviors. In this paper, we introduce LettinGo, a novel framework for generating diverse and adaptive user profiles. By leveraging the expressive power of LLMs and incorporating direct feedback from downstream recommendation tasks, our approach avoids the rigid constraints imposed by supervised fine-tuning (SFT). Instead, we employ Direct Preference Optimization (DPO) to align the profile generator with task-specific performance, ensuring that the profiles remain adaptive and effective. LettinGo operates in three stages: (1) exploring diverse user profiles via multiple LLMs, (2) evaluating profile quality based on their impact in recommendation systems, and (3) aligning the profile generation through pairwise preference data derived from task performance. Experimental results demonstrate that our framework significantly enhances recommendation accuracy, flexibility, and contextual awareness. This work enhances profile generation as a key innovation for next-generation recommendation systems. 

---
# Spiffy: Efficient Implementation of CoLaNET for Raspberry Pi 

**Authors**: Andrey Derzhavin, Denis Larionov  

**Link**: [PDF](https://arxiv.org/pdf/2506.18306)  

**Abstract**: This paper presents a lightweight software-based approach for running spiking neural networks (SNNs) without relying on specialized neuromorphic hardware or frameworks. Instead, we implement a specific SNN architecture (CoLaNET) in Rust and optimize it for common computing platforms. As a case study, we demonstrate our implementation, called Spiffy, on a Raspberry Pi using the MNIST dataset. Spiffy achieves 92% accuracy with low latency - just 0.9 ms per training step and 0.45 ms per inference step. The code is open-source. 

---
# Sharpening the Spear: Adaptive Expert-Guided Adversarial Attack Against DRL-based Autonomous Driving Policies 

**Authors**: Junchao Fan, Xuyang Lei, Xiaolin Chang  

**Link**: [PDF](https://arxiv.org/pdf/2506.18304)  

**Abstract**: Deep reinforcement learning (DRL) has emerged as a promising paradigm for autonomous driving. However, despite their advanced capabilities, DRL-based policies remain highly vulnerable to adversarial attacks, posing serious safety risks in real-world deployments. Investigating such attacks is crucial for revealing policy vulnerabilities and guiding the development of more robust autonomous systems. While prior attack methods have made notable progress, they still face several challenges: 1) they often rely on high-frequency attacks, yet critical attack opportunities are typically context-dependent and temporally sparse, resulting in inefficient attack patterns; 2) restricting attack frequency can improve efficiency but often results in unstable training due to the adversary's limited exploration. To address these challenges, we propose an adaptive expert-guided adversarial attack method that enhances both the stability and efficiency of attack policy training. Our method first derives an expert policy from successful attack demonstrations using imitation learning, strengthened by an ensemble Mixture-of-Experts architecture for robust generalization across scenarios. This expert policy then guides a DRL-based adversary through a KL-divergence regularization term. Due to the diversity of scenarios, expert policies may be imperfect. To address this, we further introduce a performance-aware annealing strategy that gradually reduces reliance on the expert as the adversary improves. Extensive experiments demonstrate that our method achieves outperforms existing approaches in terms of collision rate, attack efficiency, and training stability, especially in cases where the expert policy is sub-optimal. 

---
# GeNeRT: A Physics-Informed Approach to Intelligent Wireless Channel Modeling via Generalizable Neural Ray Tracing 

**Authors**: Kejia Bian, Meixia Tao, Shu Sun, Jun Yu  

**Link**: [PDF](https://arxiv.org/pdf/2506.18295)  

**Abstract**: Neural ray tracing (RT) has emerged as a promising paradigm for channel modeling by combining physical propagation principles with neural networks. It enables high modeling accuracy and efficiency. However, current neural RT methods face two key limitations: constrained generalization capability due to strong spatial dependence, and weak adherence to electromagnetic laws. In this paper, we propose GeNeRT, a Generalizable Neural RT framework with enhanced generalization, accuracy and efficiency. GeNeRT supports both intra-scenario spatial transferability and inter-scenario zero-shot generalization. By incorporating Fresnel-inspired neural network design, it also achieves higher accuracy in multipath component (MPC) prediction. Furthermore, a GPU-tensorized acceleration strategy is introduced to improve runtime efficiency. Extensive experiments conducted in outdoor scenarios demonstrate that GeNeRT generalizes well across untrained regions within a scenario and entirely unseen environments, and achieves superior accuracy in MPC prediction compared to baselines. Moreover, it outperforms Wireless Insite in runtime efficiency, particularly in multi-transmitter settings. Ablation experiments validate the effectiveness of the network architecture and training strategy in capturing physical principles of ray-surface interactions. 

---
# Selective Social-Interaction via Individual Importance for Fast Human Trajectory Prediction 

**Authors**: Yota Urano, Hiromu Taketsugu, Norimichi Ukita  

**Link**: [PDF](https://arxiv.org/pdf/2506.18291)  

**Abstract**: This paper presents an architecture for selecting important neighboring people to predict the primary person's trajectory. To achieve effective neighboring people selection, we propose a people selection module called the Importance Estimator which outputs the importance of each neighboring person for predicting the primary person's future trajectory. To prevent gradients from being blocked by non-differentiable operations when sampling surrounding people based on their importance, we employ the Gumbel Softmax for training. Experiments conducted on the JRDB dataset show that our method speeds up the process with competitive prediction accuracy. 

---
# Tu(r)ning AI Green: Exploring Energy Efficiency Cascading with Orthogonal Optimizations 

**Authors**: Saurabhsingh Rajput, Mootez Saad, Tushar Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2506.18289)  

**Abstract**: AI's exponential growth intensifies computational demands and energy challenges. While practitioners employ various optimization techniques, that we refer as "knobs" in this paper, to tune model efficiency, these are typically afterthoughts and reactive ad-hoc changes applied in isolation without understanding their combinatorial effects on energy efficiency. This paper emphasizes on treating energy efficiency as the first-class citizen and as a fundamental design consideration for a compute-intensive pipeline. We show that strategic selection across five AI pipeline phases (data, model, training, system, inference) creates cascading efficiency. Experimental validation shows orthogonal combinations reduce energy consumption by up to $94.6$% while preserving $95.95$% of the original F1 score of non-optimized pipelines. This curated approach provides actionable frameworks for informed sustainable AI that balance efficiency, performance, and environmental responsibility. 

---
# Learning Causal Graphs at Scale: A Foundation Model Approach 

**Authors**: Naiyu Yin, Tian Gao, Yue Yu  

**Link**: [PDF](https://arxiv.org/pdf/2506.18285)  

**Abstract**: Due to its human-interpretability and invariance properties, Directed Acyclic Graph (DAG) has been a foundational tool across various areas of AI research, leading to significant advancements. However, DAG learning remains highly challenging, due to its super-exponential growth in computational cost and identifiability issues, particularly in small-sample regimes. To address these two challenges, in this work we leverage the recent success of linear transformers and develop a foundation model approach for discovering multiple order-consistent DAGs across tasks. In particular, we propose Attention-DAG (ADAG), a novel attention-mechanism-based architecture for learning multiple linear Structural Equation Models (SEMs). ADAG learns the mapping from observed data to both graph structure and parameters via a nonlinear attention-based kernel, enabling efficient multi-task estimation of the underlying linear SEMs. By formulating the learning process across multiple tasks as a continuous optimization problem, the pre-trained ADAG model captures the common structural properties as a shared low-dimensional prior, thereby reducing the ill-posedness of downstream DAG learning tasks in small-sample regimes. We evaluate our proposed approach on benchmark synthetic datasets and find that ADAG achieves substantial improvements in both DAG learning accuracy and zero-shot inference efficiency. To the best of our knowledge, this is the first practical approach for pre-training a foundation model specifically designed for DAG learning, representing a step toward more efficient and generalizable down-stream applications in causal discovery. 

---
# Open Set Recognition for Endoscopic Image Classification: A Deep Learning Approach on the Kvasir Dataset 

**Authors**: Kasra Moazzami, Seoyoun Son, John Lin, Sun Min Lee, Daniel Son, Hayeon Lee, Jeongho Lee, Seongji Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.18284)  

**Abstract**: Endoscopic image classification plays a pivotal role in medical diagnostics by identifying anatomical landmarks and pathological findings. However, conventional closed-set classification frameworks are inherently limited in open-world clinical settings, where previously unseen conditions can arise andcompromise model reliability. To address this, we explore the application of Open Set Recognition (OSR) techniques on the Kvasir dataset, a publicly available and diverse endoscopic image collection. In this study, we evaluate and compare the OSR capabilities of several representative deep learning architectures, including ResNet-50, Swin Transformer, and a hybrid ResNet-Transformer model, under both closed-set and open-set conditions. OpenMax is adopted as a baseline OSR method to assess the ability of these models to distinguish known classes from previously unseen categories. This work represents one of the first efforts to apply open set recognition to the Kvasir dataset and provides a foundational benchmark for evaluating OSR performance in medical image analysis. Our results offer practical insights into model behavior in clinically realistic settings and highlight the importance of OSR techniques for the safe deployment of AI systems in endoscopy. 

---
# ARD-LoRA: Dynamic Rank Allocation for Parameter-Efficient Fine-Tuning of Foundation Models with Heterogeneous Adaptation Needs 

**Authors**: Haseeb Ullah Khan Shinwari, Muhammad Usama  

**Link**: [PDF](https://arxiv.org/pdf/2506.18267)  

**Abstract**: Conventional Low-Rank Adaptation (LoRA) methods employ a fixed rank, imposing uniform adaptation across transformer layers and attention heads despite their heterogeneous learning dynamics. This paper introduces Adaptive Rank Dynamic LoRA (ARD-LoRA), a novel framework that automates rank allocation through learnable scaling factors. These factors are optimized via a meta-objective balancing task performance and parameter efficiency, incorporating $\ell_1$ sparsity for minimal rank and Total Variation regularization for stable rank transitions. ARD-LoRA enables continuous, differentiable, per-head rank adaptation. Experiments on LLAMA-3.1-70B and PaliGemma-2 demonstrate ARD-LoRA's efficacy, achieving up to 99.3% of full fine-tuning performance with only 0.32% trainable parameters, outperforming strong baselines like DoRA and AdaLoRA. Furthermore, it reduces multimodal adaptation memory by 41%. These results establish dynamic, fine-grained rank allocation as a critical paradigm for efficient foundation model adaptation. 

---
# RLPR: Extrapolating RLVR to General Domains without Verifiers 

**Authors**: Tianyu Yu, Bo Ji, Shouli Wang, Shu Yao, Zefan Wang, Ganqu Cui, Lifan Yuan, Ning Ding, Yuan Yao, Zhiyuan Liu, Maosong Sun, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2506.18254)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) demonstrates promising potential in advancing the reasoning capabilities of LLMs. However, its success remains largely confined to mathematical and code domains. This primary limitation stems from the heavy reliance on domain-specific verifiers, which results in prohibitive complexity and limited scalability. To address the challenge, our key observation is that LLM's intrinsic probability of generating a correct free-form answer directly indicates its own evaluation of the reasoning reward (i.e., how well the reasoning process leads to the correct answer). Building on this insight, we propose RLPR, a simple verifier-free framework that extrapolates RLVR to broader general domains. RLPR uses the LLM's own token probability scores for reference answers as the reward signal and maximizes the expected reward during training. We find that addressing the high variance of this noisy probability reward is crucial to make it work, and propose prob-to-reward and stabilizing methods to ensure a precise and stable reward from LLM intrinsic probabilities. Comprehensive experiments in four general-domain benchmarks and three mathematical benchmarks show that RLPR consistently improves reasoning capabilities in both areas for Gemma, Llama, and Qwen based models. Notably, RLPR outperforms concurrent VeriFree by 7.6 points on TheoremQA and 7.5 points on Minerva, and even surpasses strong verifier-model-dependent approaches General-Reasoner by 1.6 average points across seven benchmarks. 

---
# Morse: Dual-Sampling for Lossless Acceleration of Diffusion Models 

**Authors**: Chao Li, Jiawei Fan, Anbang Yao  

**Link**: [PDF](https://arxiv.org/pdf/2506.18251)  

**Abstract**: In this paper, we present Morse, a simple dual-sampling framework for accelerating diffusion models losslessly. The key insight of Morse is to reformulate the iterative generation (from noise to data) process via taking advantage of fast jump sampling and adaptive residual feedback strategies. Specifically, Morse involves two models called Dash and Dot that interact with each other. The Dash model is just the pre-trained diffusion model of any type, but operates in a jump sampling regime, creating sufficient space for sampling efficiency improvement. The Dot model is significantly faster than the Dash model, which is learnt to generate residual feedback conditioned on the observations at the current jump sampling point on the trajectory of the Dash model, lifting the noise estimate to easily match the next-step estimate of the Dash model without jump sampling. By chaining the outputs of the Dash and Dot models run in a time-interleaved fashion, Morse exhibits the merit of flexibly attaining desired image generation performance while improving overall runtime efficiency. With our proposed weight sharing strategy between the Dash and Dot models, Morse is efficient for training and inference. Our method shows a lossless speedup of 1.78X to 3.31X on average over a wide range of sampling step budgets relative to 9 baseline diffusion models on 6 image generation tasks. Furthermore, we show that our method can be also generalized to improve the Latent Consistency Model (LCM-SDXL, which is already accelerated with consistency distillation technique) tailored for few-step text-to-image synthesis. The code and models are available at this https URL. 

---
# Semantic Structure-Aware Generative Attacks for Enhanced Adversarial Transferability 

**Authors**: Jongoh Jeong, Hunmin Yang, Jaeseok Jeong, Kuk-Jin Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2506.18248)  

**Abstract**: Generative adversarial attacks train a perturbation generator on a white-box surrogate model and subsequently apply the crafted perturbations to unseen black-box victim models. In contrast to iterative attacks, these methods deliver superior inference-time efficiency, scalability, and transferability; however, up until now, existing studies have not fully exploited the representational capacity of generative models to preserve and harness semantic information. Specifically, the intermediate activations of the generator encode rich semantic features--object boundaries and coarse shapes--that remain under-exploited, thereby limiting the alignment of perturbations with object-salient regions which are critical for adversarial transferability. To remedy this, we introduce a semantic structure-aware attack framework based on the Mean Teacher, which serves as a temporally smoothed feature reference. With this smoothed reference, we further direct semantic consistency between the early-layer activations in the student and those of the semantically rich teacher by feature distillation. By anchoring perturbation synthesis to the semantically salient early intermediate blocks within the generator based on empirical findings, our method guides progressive adversarial perturbation on regions that substantially enhance adversarial transferability. We conduct extensive experiments over diverse models, domains and tasks to demonstrate consistent improvements relative to state-of-the-art generative attacks, comprehensively evaluated using conventional metrics and our newly proposed Accidental Correction Rate (ACR). 

---
# Smart-LLaMA-DPO: Reinforced Large Language Model for Explainable Smart Contract Vulnerability Detection 

**Authors**: Lei Yu, Zhirong Huang, Hang Yuan, Shiqi Cheng, Li Yang, Fengjun Zhang, Chenjie Shen, Jiajia Ma, Jingyuan Zhang, Junyi Lu, Chun Zuo  

**Link**: [PDF](https://arxiv.org/pdf/2506.18245)  

**Abstract**: Smart contract vulnerability detection remains a major challenge in blockchain security. Existing vulnerability detection methods face two main issues: (1) Existing datasets lack comprehensive coverage and high-quality explanations for preference learning. (2) Large language models (LLMs) often struggle with accurately interpreting specific concepts in smart contract security. Empirical analysis shows that even after continual pre-training (CPT) and supervised fine-tuning (SFT), LLMs may misinterpret the execution order of state changes, resulting in incorrect explanations despite making correct detection decisions. To address these challenges, we propose Smart-LLaMA-DPO based on LLaMA-3.1-8B. We construct a comprehensive dataset covering four major vulnerability types and machine-unauditable vulnerabilities, including precise labels, explanations, and locations for SFT, as well as high-quality and low-quality output pairs for Direct Preference Optimization (DPO). Second, we perform CPT using large-scale smart contract to enhance the LLM's understanding of specific security practices in smart contracts. Futhermore, we conduct SFT with our comprehensive dataset. Finally, we apply DPO, leveraging human feedback and a specially designed loss function that increases the probability of preferred explanations while reducing the likelihood of non-preferred outputs. We evaluate Smart-LLaMA-DPO on four major vulnerability types: reentrancy, timestamp dependence, integer overflow/underflow, and delegatecall, as well as machine-unauditable vulnerabilities. Our method significantly outperforms state-of-the-art baselines, with average improvements of 10.43% in F1 score and 7.87% in accuracy. Moreover, both LLM evaluation and human evaluation confirm that our method generates more correct, thorough, and clear explanations. 

---
# Quantum-Classical Hybrid Quantized Neural Network 

**Authors**: Wenxin Li, Chuan Wang, Hongdong Zhu, Qi Gao, Yin Ma, Hai Wei, Kai Wen  

**Link**: [PDF](https://arxiv.org/pdf/2506.18240)  

**Abstract**: Here in this work, we present a novel Quadratic Binary Optimization (QBO) model for quantized neural network training, enabling the use of arbitrary activation and loss functions through spline interpolation. We introduce Forward Interval Propagation (FIP), a method designed to tackle the challenges of non-linearity and the multi-layer composite structure in neural networks by discretizing activation functions into linear subintervals. This approach preserves the universal approximation properties of neural networks while allowing complex nonlinear functions to be optimized using quantum computers, thus broadening their applicability in artificial intelligence. We provide theoretical upper bounds on the approximation error and the number of Ising spins required, by deriving the sample complexity of the empirical risk minimization problem, from an optimization perspective. A significant challenge in solving the associated Quadratic Constrained Binary Optimization (QCBO) model on a large scale is the presence of numerous constraints. When employing the penalty method to handle these constraints, tuning a large number of penalty coefficients becomes a critical hyperparameter optimization problem, increasing computational complexity and potentially affecting solution quality. To address this, we employ the Quantum Conditional Gradient Descent (QCGD) algorithm, which leverages quantum computing to directly solve the QCBO problem. We prove the convergence of QCGD under a quantum oracle with randomness and bounded variance in objective value, as well as under limited precision constraints in the coefficient matrix. Additionally, we provide an upper bound on the Time-To-Solution for the QCBO solving process. Experimental results using a coherent Ising machine (CIM) demonstrate a 94.95% accuracy on the Fashion MNIST classification task, with only 1.1-bit precision. 

---
# AdapThink: Adaptive Thinking Preferences for Reasoning Language Model 

**Authors**: Xu Wan, Wei Wang, Wenyue Xu, Wotao Yin, Jie Song, Mingyang Sun  

**Link**: [PDF](https://arxiv.org/pdf/2506.18237)  

**Abstract**: Reinforcement Learning (RL)-based post-training has significantly advanced the complex reasoning capabilities of language models, fostering sophisticated self-reflection processes. However, this ``slow thinking'' paradigm presents a critical challenge to reasoning efficiency: models may expend excessive computation on simple questions and shift reasoning prematurely for complex ones. Previous mechanisms typically rely on static length budgets or predefined rules, lacking the adaptability for varying question complexities and models' evolving capabilities. To this end, we propose AdapThink, an adaptive post-training framework designed to induce more efficient thinking while maintaining the performance of reasoning language models. Specifically, AdapThink incorporates two key mechanisms: 1) A group-relative reward function that leverages model confidence and response's characteristic to dynamically adjust the preference of reflection-related transition words without resorting to a fixed length preference. 2) A diversity-aware sampling mechanism that balances the training group's solution accuracy with reasoning diversity via an entropy-guided score. Experiments on several mathematical reasoning datasets with DeepSeek-distilled models demonstrate AdapThink's advantages in enabling adaptive reasoning patterns and mitigating the inefficiencies. 

---
# Make It Efficient: Dynamic Sparse Attention for Autoregressive Image Generation 

**Authors**: Xunzhi Xiang, Qi Fan  

**Link**: [PDF](https://arxiv.org/pdf/2506.18226)  

**Abstract**: Autoregressive conditional image generation models have emerged as a dominant paradigm in text-to-image synthesis. These methods typically convert images into one-dimensional token sequences and leverage the self-attention mechanism, which has achieved remarkable success in natural language processing, to capture long-range dependencies, model global context, and ensure semantic coherence. However, excessively long contexts during inference lead to significant memory overhead caused by KV-cache and computational delays. To alleviate these challenges, we systematically analyze how global semantics, spatial layouts, and fine-grained textures are formed during inference, and propose a novel training-free context optimization method called Adaptive Dynamic Sparse Attention (ADSA). Conceptually, ADSA dynamically identifies historical tokens crucial for maintaining local texture consistency and those essential for ensuring global semantic coherence, thereby efficiently streamlining attention computation. Additionally, we introduce a dynamic KV-cache update mechanism tailored for ADSA, reducing GPU memory consumption during inference by approximately $50\%$. Extensive qualitative and quantitative experiments demonstrate the effectiveness and superiority of our approach in terms of both generation quality and resource efficiency. 

---
# These are Not All the Features You are Looking For: A Fundamental Bottleneck In Supervised Pretraining 

**Authors**: Xingyu Alice Yang, Jianyu Zhang, Léon Bottou  

**Link**: [PDF](https://arxiv.org/pdf/2506.18221)  

**Abstract**: Transfer learning is a cornerstone of modern machine learning, promising a way to adapt models pretrained on a broad mix of data to new tasks with minimal new data. However, a significant challenge remains in ensuring that transferred features are sufficient to handle unseen datasets, amplified by the difficulty of quantifying whether two tasks are "related". To address these challenges, we evaluate model transfer from a pretraining mixture to each of its component tasks, assessing whether pretrained features can match the performance of task-specific direct training. We identify a fundamental limitation in deep learning models -- an "information saturation bottleneck" -- where networks fail to learn new features once they encode similar competing features during training. When restricted to learning only a subset of key features during pretraining, models will permanently lose critical features for transfer and perform inconsistently on data distributions, even components of the training mixture. Empirical evidence from published studies suggests that this phenomenon is pervasive in deep learning architectures -- factors such as data distribution or ordering affect the features that current representation learning methods can learn over time. This study suggests that relying solely on large-scale networks may not be as effective as focusing on task-specific training, when available. We propose richer feature representations as a potential solution to better generalize across new datasets and, specifically, present existing methods alongside a novel approach, the initial steps towards addressing this challenge. 

---
# Cross-Architecture Knowledge Distillation (KD) for Retinal Fundus Image Anomaly Detection on NVIDIA Jetson Nano 

**Authors**: Berk Yilmaz, Aniruddh Aiyengar  

**Link**: [PDF](https://arxiv.org/pdf/2506.18220)  

**Abstract**: Early and accurate identification of retinal ailments is crucial for averting ocular decline; however, access to dependable diagnostic devices is not often available in low-resourced settings. This project proposes to solve that by developing a lightweight, edge-device deployable disease classifier using cross-architecture knowledge distilling. We first train a high-capacity vision transformer (ViT) teacher model, pre-trained using I-JEPA self-supervised learning, to classify fundus images into four classes: Normal, Diabetic Retinopathy, Glaucoma, and Cataract. We kept an Internet of Things (IoT) focus when compressing to a CNN-based student model for deployment in resource-limited conditions, such as the NVIDIA Jetson Nano. This was accomplished using a novel framework which included a Partitioned Cross-Attention (PCA) projector, a Group-Wise Linear (GL) projector, and a multi-view robust training method. The teacher model has 97.4 percent more parameters than the student model, with it achieving 89 percent classification with a roughly 93 percent retention of the teacher model's diagnostic performance. The retention of clinical classification behavior supports our method's initial aim: compression of the ViT while retaining accuracy. Our work serves as an example of a scalable, AI-driven triage solution for retinal disorders in under-resourced areas. 

---
# Deep Learning-based Alignment Measurement in Knee Radiographs 

**Authors**: Zhisen Hu, Dominic Cullen, Peter Thompson, David Johnson, Chang Bian, Aleksei Tiulpin, Timothy Cootes, Claudia Lindner  

**Link**: [PDF](https://arxiv.org/pdf/2506.18209)  

**Abstract**: Radiographic knee alignment (KA) measurement is important for predicting joint health and surgical outcomes after total knee replacement. Traditional methods for KA measurements are manual, time-consuming and require long-leg radiographs. This study proposes a deep learning-based method to measure KA in anteroposterior knee radiographs via automatically localized knee anatomical landmarks. Our method builds on hourglass networks and incorporates an attention gate structure to enhance robustness and focus on key anatomical features. To our knowledge, this is the first deep learning-based method to localize over 100 knee anatomical landmarks to fully outline the knee shape while integrating KA measurements on both pre-operative and post-operative images. It provides highly accurate and reliable anatomical varus/valgus KA measurements using the anatomical tibiofemoral angle, achieving mean absolute differences ~1° when compared to clinical ground truth measurements. Agreement between automated and clinical measurements was excellent pre-operatively (intra-class correlation coefficient (ICC) = 0.97) and good post-operatively (ICC = 0.86). Our findings demonstrate that KA assessment can be automated with high accuracy, creating opportunities for digitally enhanced clinical workflows. 

---
# Multimodal Fusion SLAM with Fourier Attention 

**Authors**: Youjie Zhou, Guofeng Mei, Yiming Wang, Yi Wan, Fabio Poiesi  

**Link**: [PDF](https://arxiv.org/pdf/2506.18204)  

**Abstract**: Visual SLAM is particularly challenging in environments affected by noise, varying lighting conditions, and darkness. Learning-based optical flow algorithms can leverage multiple modalities to address these challenges, but traditional optical flow-based visual SLAM approaches often require significant computational this http URL overcome this limitation, we propose FMF-SLAM, an efficient multimodal fusion SLAM method that utilizes fast Fourier transform (FFT) to enhance the algorithm efficiency. Specifically, we introduce a novel Fourier-based self-attention and cross-attention mechanism to extract features from RGB and depth signals. We further enhance the interaction of multimodal features by incorporating multi-scale knowledge distillation across modalities. We also demonstrate the practical feasibility of FMF-SLAM in real-world scenarios with real time performance by integrating it with a security robot by fusing with a global positioning module GNSS-RTK and global Bundle Adjustment. Our approach is validated using video sequences from TUM, TartanAir, and our real-world datasets, showcasing state-of-the-art performance under noisy, varying lighting, and dark this http URL code and datasets are available at this https URL. 

---
# Prompt Engineering Techniques for Mitigating Cultural Bias Against Arabs and Muslims in Large Language Models: A Systematic Review 

**Authors**: Bushra Asseri, Estabrag Abdelaziz, Areej Al-Wabil  

**Link**: [PDF](https://arxiv.org/pdf/2506.18199)  

**Abstract**: Large language models have demonstrated remarkable capabilities across various domains, yet concerns about cultural bias - particularly towards Arabs and Muslims - pose significant ethical challenges by perpetuating harmful stereotypes and marginalization. Despite growing recognition of bias in LLMs, prompt engineering strategies specifically addressing Arab and Muslim representation remain understudied. This mixed-methods systematic review examines such techniques, offering evidence-based guidance for researchers and practitioners. Following PRISMA guidelines and Kitchenham's systematic review methodology, we analyzed 8 empirical studies published between 2021-2024 investigating bias mitigation strategies. Our findings reveal five primary prompt engineering approaches: cultural prompting, affective priming, self-debiasing techniques, structured multi-step pipelines, and parameter-optimized continuous prompts. Although all approaches show potential for reducing bias, effectiveness varied substantially across studies and bias types. Evidence suggests that certain bias types may be more resistant to prompt-based mitigation than others. Structured multi-step pipelines demonstrated the highest overall effectiveness, achieving up to 87.7% reduction in bias, though they require greater technical expertise. Cultural prompting offers broader accessibility with substantial effectiveness. These results underscore the accessibility of prompt engineering for mitigating cultural bias without requiring access to model parameters. The limited number of studies identified highlights a significant research gap in this critical area. Future research should focus on developing culturally adaptive prompting techniques, creating Arab and Muslim-specific evaluation resources, and integrating prompt engineering with complementary debiasing methods to address deeper stereotypes while maintaining model utility. 

---
# Two Sonification Methods for the MindCube 

**Authors**: Fangzheng Liu, Lancelot Blanchard, Don D. Haddad, Joseph A. Paradiso  

**Link**: [PDF](https://arxiv.org/pdf/2506.18196)  

**Abstract**: In this work, we explore the musical interface potential of the MindCube, an interactive device designed to study emotions. Embedding diverse sensors and input devices, this interface resembles a fidget cube toy commonly used to help users relieve their stress and anxiety. As such, it is a particularly well-suited controller for musical systems that aim to help with emotion regulation. In this regard, we present two different mappings for the MindCube, with and without AI. With our generative AI mapping, we propose a way to infuse meaning within a latent space and techniques to navigate through it with an external controller. We discuss our results and propose directions for future work. 

---
# Wisdom of Crowds Through Myopic Self-Confidence Adaptation 

**Authors**: Giacomo Como, Fabio Fagnani, Anton Proskurnikov  

**Link**: [PDF](https://arxiv.org/pdf/2506.18195)  

**Abstract**: The wisdom of crowds is an umbrella term for phenomena suggesting that the collective judgment or decision of a large group can be more accurate than the individual judgments or decisions of the group members. A well-known example illustrating this concept is the competition at a country fair described by Galton, where the median value of the individual guesses about the weight of an ox resulted in an astonishingly accurate estimate of the actual weight. This phenomenon resembles classical results in probability theory and relies on independent decision-making. The accuracy of the group's final decision can be significantly reduced if the final agents' opinions are driven by a few influential agents.
In this paper, we consider a group of agents who initially possess uncorrelated and unbiased noisy measurements of a common state of the world. Assume these agents iteratively update their estimates according to a simple non-Bayesian learning rule, commonly known in mathematical sociology as the French-DeGroot dynamics or iterative opinion pooling. As a result of this iterative distributed averaging process, each agent arrives at an asymptotic estimate of the state of the world, with the variance of this estimate determined by the matrix of weights the agents assign to each other. Every agent aims at minimizing the variance of her asymptotic estimate of the state of the world; however, such variance is also influenced by the weights allocated by other agents. To achieve the best possible estimate, the agents must then solve a game-theoretic, multi-objective optimization problem defined by the available sets of influence weights. We characterize both the Pareto frontier and the set of Nash equilibria in the resulting game. Additionally, we examine asynchronous best-response dynamics for the group of agents and prove their convergence to the set of strict Nash equilibria. 

---
# DeInfoReg: A Decoupled Learning Framework for Better Training Throughput 

**Authors**: Zih-Hao Huang, You-Teng Lin, Hung-Hsuan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.18193)  

**Abstract**: This paper introduces Decoupled Supervised Learning with Information Regularization (DeInfoReg), a novel approach that transforms a long gradient flow into multiple shorter ones, thereby mitigating the vanishing gradient problem. Integrating a pipeline strategy, DeInfoReg enables model parallelization across multiple GPUs, significantly improving training throughput. We compare our proposed method with standard backpropagation and other gradient flow decomposition techniques. Extensive experiments on diverse tasks and datasets demonstrate that DeInfoReg achieves superior performance and better noise resistance than traditional BP models and efficiently utilizes parallel computing resources. The code for reproducibility is available at: this https URL. 

---
# Call Me Maybe: Enhancing JavaScript Call Graph Construction using Graph Neural Networks 

**Authors**: Masudul Hasan Masud Bhuiyan, Gianluca De Stefano, Giancarlo Pellegrino, Cristian-Alexandru Staicu  

**Link**: [PDF](https://arxiv.org/pdf/2506.18191)  

**Abstract**: Static analysis plays a key role in finding bugs, including security issues. A critical step in static analysis is building accurate call graphs that model function calls in a program. However, due to hard-to-analyze language features, existing call graph construction algorithms for JavaScript are neither sound nor complete. Prior work shows that even advanced solutions produce false edges and miss valid ones. In this work, we assist these tools by identifying missed call edges. Our main idea is to frame the problem as link prediction on full program graphs, using a rich representation with multiple edge types. Our approach, GRAPHIA, leverages recent advances in graph neural networks to model non-local relationships between code elements. Concretely, we propose representing JavaScript programs using a combination of syntactic- and semantic-based edges. GRAPHIA can learn from imperfect labels, including static call edges from existing tools and dynamic edges from tests, either from the same or different projects. Because call graphs are sparse, standard machine learning metrics like ROC are not suitable. Instead, we evaluate GRAPHIA by ranking function definitions for each unresolved call site. We conduct a large-scale evaluation on 50 popular JavaScript libraries with 163K call edges (150K static and 13K dynamic). GRAPHIA builds program graphs with 6.6M structural and 386K semantic edges. It ranks the correct target as the top candidate in over 42% of unresolved cases and within the top 5 in 72% of cases, reducing the manual effort needed for analysis. Our results show that learning-based methods can improve the recall of JavaScript call graph construction. To our knowledge, this is the first work to apply GNN-based link prediction to full multi-file program graphs for interprocedural analysis. 

---
# CareLab at #SMM4H-HeaRD 2025: Insomnia Detection and Food Safety Event Extraction with Domain-Aware Transformers 

**Authors**: Zihan Liang, Ziwen Pan, Sumon Kanti Dey, Azra Ismail  

**Link**: [PDF](https://arxiv.org/pdf/2506.18185)  

**Abstract**: This paper presents our system for the SMM4H-HeaRD 2025 shared tasks, specifically Task 4 (Subtasks 1, 2a, and 2b) and Task 5 (Subtasks 1 and 2). Task 4 focused on detecting mentions of insomnia in clinical notes, while Task 5 addressed the extraction of food safety events from news articles. We participated in all subtasks and report key findings across them, with particular emphasis on Task 5 Subtask 1, where our system achieved strong performance-securing first place with an F1 score of 0.958 on the test set. To attain this result, we employed encoder-based models (e.g., RoBERTa), alongside GPT-4 for data augmentation. This paper outlines our approach, including preprocessing, model architecture, and subtask-specific adaptations 

---
# STACT-Time: Spatio-Temporal Cross Attention for Cine Thyroid Ultrasound Time Series Classification 

**Authors**: Irsyad Adam, Tengyue Zhang, Shrayes Raman, Zhuyu Qiu, Brandon Taraku, Hexiang Feng, Sile Wang, Ashwath Radhachandran, Shreeram Athreya, Vedrana Ivezic, Peipei Ping, Corey Arnold, William Speier  

**Link**: [PDF](https://arxiv.org/pdf/2506.18172)  

**Abstract**: Thyroid cancer is among the most common cancers in the United States. Thyroid nodules are frequently detected through ultrasound (US) imaging, and some require further evaluation via fine-needle aspiration (FNA) biopsy. Despite its effectiveness, FNA often leads to unnecessary biopsies of benign nodules, causing patient discomfort and anxiety. To address this, the American College of Radiology Thyroid Imaging Reporting and Data System (TI-RADS) has been developed to reduce benign biopsies. However, such systems are limited by interobserver variability. Recent deep learning approaches have sought to improve risk stratification, but they often fail to utilize the rich temporal and spatial context provided by US cine clips, which contain dynamic global information and surrounding structural changes across various views. In this work, we propose the Spatio-Temporal Cross Attention for Cine Thyroid Ultrasound Time Series Classification (STACT-Time) model, a novel representation learning framework that integrates imaging features from US cine clips with features from segmentation masks automatically generated by a pretrained model. By leveraging self-attention and cross-attention mechanisms, our model captures the rich temporal and spatial context of US cine clips while enhancing feature representation through segmentation-guided learning. Our model improves malignancy prediction compared to state-of-the-art models, achieving a cross-validation precision of 0.91 (plus or minus 0.02) and an F1 score of 0.89 (plus or minus 0.02). By reducing unnecessary biopsies of benign nodules while maintaining high sensitivity for malignancy detection, our model has the potential to enhance clinical decision-making and improve patient outcomes. 

---
# Understanding Reasoning in Thinking Language Models via Steering Vectors 

**Authors**: Constantin Venhoff, Iván Arcuschin, Philip Torr, Arthur Conmy, Neel Nanda  

**Link**: [PDF](https://arxiv.org/pdf/2506.18167)  

**Abstract**: Recent advances in large language models (LLMs) have led to the development of thinking language models that generate extensive internal reasoning chains before producing responses. While these models achieve improved performance, controlling their reasoning processes remains challenging. This work presents a steering approach for thinking LLMs by analyzing and manipulating specific reasoning behaviors in DeepSeek-R1-Distill models. Through a systematic experiment on 500 tasks across 10 diverse categories, we identify several reasoning behaviors exhibited by thinking models, including expressing uncertainty, generating examples for hypothesis validation, and backtracking in reasoning chains. We demonstrate that these behaviors are mediated by linear directions in the model's activation space and can be controlled using steering vectors. By extracting and applying these vectors, we provide a method to modulate specific aspects of the model's reasoning process, such as its tendency to backtrack or express uncertainty. Our approach offers practical tools for steering reasoning processes in thinking models in a controlled and interpretable manner. We validate our steering method using two DeepSeek-R1-Distill models, demonstrating consistent control across different model architectures. 

---
# Non-equilibrium Annealed Adjoint Sampler 

**Authors**: Jaemoo Choi, Yongxin Chen, Molei Tao, Guan-Horng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.18165)  

**Abstract**: Recently, there has been significant progress in learning-based diffusion samplers, which aim to sample from a given unnormalized density. These methods typically follow one of two paradigms: (i) formulating sampling as an unbiased stochastic optimal control (SOC) problem using a canonical reference process, or (ii) refining annealed path measures through importance-weighted sampling. Although annealing approaches have advantages in guiding samples toward high-density regions, reliance on importance sampling leads to high variance and limited scalability in practice. In this paper, we introduce the \textbf{Non-equilibrium Annealed Adjoint Sampler (NAAS)}, a novel SOC-based diffusion sampler that leverages annealed reference dynamics without resorting to importance sampling. NAAS employs a lean adjoint system inspired by adjoint matching, enabling efficient and scalable training. We demonstrate the effectiveness of our approach across a range of tasks, including sampling from classical energy landscapes and molecular Boltzmann distribution. 

---
# QuranMorph: Morphologically Annotated Quranic Corpus 

**Authors**: Diyam Akra, Tymaa Hammouda, Mustafa Jarrar  

**Link**: [PDF](https://arxiv.org/pdf/2506.18148)  

**Abstract**: We present the QuranMorph corpus, a morphologically annotated corpus for the Quran (77,429 tokens). Each token in the QuranMorph was manually lemmatized and tagged with its part-of-speech by three expert linguists. The lemmatization process utilized lemmas from Qabas, an Arabic lexicographic database linked with 110 lexicons and corpora of 2 million tokens. The part-of-speech tagging was performed using the fine-grained SAMA/Qabas tagset, which encompasses 40 tags. As shown in this paper, this rich lemmatization and POS tagset enabled the QuranMorph corpus to be inter-linked with many linguistic resources. The corpus is open-source and publicly available as part of the SinaLab resources at (this https URL) 

---
# Routing Mamba: Scaling State Space Models with Mixture-of-Experts Projection 

**Authors**: Zheng Zhan, Liliang Ren, Shuohang Wang, Liyuan Liu, Yang Liu, Yeyun Gong, Yanzhi Wang, Yelong Shen  

**Link**: [PDF](https://arxiv.org/pdf/2506.18145)  

**Abstract**: Linear State Space Models (SSMs) offer remarkable performance gains in efficient sequence modeling, with constant inference-time computation and memory complexity. Recent advances, such as Mamba, further enhance SSMs with input-dependent gating and hardware-aware implementations, positioning them as strong alternatives to Transformers for long sequence modeling. However, efficiently scaling the expressive power of SSMs, particularly with Mixture of Experts (MoE), remains challenging, as naive integration attempts often falter or degrade performance. In this work, we introduce Routing Mamba (RoM), a novel approach that scales SSM parameters using sparse mixtures of linear projection experts. By sharing routing decisions between projection layers and lightweight sub-modules within Mamba across experts, RoM leverages synergies among linear projection experts for effective and efficient sparse scaling of Mamba layers. At a scale of 1.3B active parameters (10B total) and 16K training sequence length, RoM achieves language modeling performance equivalent to a dense Mamba model requiring over 2.3x more active parameters, and demonstrates consistent perplexity across context lengths. Experimental results further show RoM effectively scales hybrid language models, yielding a 23% FLOPS saving compared to dense Mamba scaling for similar performance. 

---
# AI Harmonizer: Expanding Vocal Expression with a Generative Neurosymbolic Music AI System 

**Authors**: Lancelot Blanchard, Cameron Holt, Joseph A. Paradiso  

**Link**: [PDF](https://arxiv.org/pdf/2506.18143)  

**Abstract**: Vocals harmonizers are powerful tools to help solo vocalists enrich their melodies with harmonically supportive voices. These tools exist in various forms, from commercially available pedals and software to custom-built systems, each employing different methods to generate harmonies. Traditional harmonizers often require users to manually specify a key or tonal center, while others allow pitch selection via an external keyboard-both approaches demanding some degree of musical expertise. The AI Harmonizer introduces a novel approach by autonomously generating musically coherent four-part harmonies without requiring prior harmonic input from the user. By integrating state-of-the-art generative AI techniques for pitch detection and voice modeling with custom-trained symbolic music models, our system arranges any vocal melody into rich choral textures. In this paper, we present our methods, explore potential applications in performance and composition, and discuss future directions for real-time implementations. While our system currently operates offline, we believe it represents a significant step toward AI-assisted vocal performance and expressive musical augmentation. We release our implementation on GitHub. 

---
# Sparse Feature Coactivation Reveals Composable Semantic Modules in Large Language Models 

**Authors**: Ruixuan Deng, Xiaoyang Hu, Miles Gilberti, Shane Storks, Aman Taxali, Mike Angstadt, Chandra Sripada, Joyce Chai  

**Link**: [PDF](https://arxiv.org/pdf/2506.18141)  

**Abstract**: We identify semantically coherent, context-consistent network components in large language models (LLMs) using coactivation of sparse autoencoder (SAE) features collected from just a handful of prompts. Focusing on country-relation tasks, we show that ablating semantic components for countries and relations changes model outputs in predictable ways, while amplifying these components induces counterfactual responses. Notably, composing relation and country components yields compound counterfactual outputs. We find that, whereas most country components emerge from the very first layer, the more abstract relation components are concentrated in later layers. Furthermore, within relation components themselves, nodes from later layers tend to have a stronger causal impact on model outputs. Overall, these findings suggest a modular organization of knowledge within LLMs and advance methods for efficient, targeted model manipulation. 

---
# $ϕ^{\infty}$: Clause Purification, Embedding Realignment, and the Total Suppression of the Em Dash in Autoregressive Language Models 

**Authors**: Bugra Kilictas, Faruk Alpay  

**Link**: [PDF](https://arxiv.org/pdf/2506.18129)  

**Abstract**: We identify a critical vulnerability in autoregressive transformer language models where the em dash token induces recursive semantic drift, leading to clause boundary hallucination and embedding space entanglement. Through formal analysis of token-level perturbations in semantic lattices, we demonstrate that em dash insertion fundamentally alters the model's latent representations, causing compounding errors in long-form generation. We propose a novel solution combining symbolic clause purification via the phi-infinity operator with targeted embedding matrix realignment. Our approach enables total suppression of problematic tokens without requiring model retraining, while preserving semantic coherence through fixed-point convergence guarantees. Experimental validation shows significant improvements in generation consistency and topic maintenance. This work establishes a general framework for identifying and mitigating token-level vulnerabilities in foundation models, with immediate implications for AI safety, model alignment, and robust deployment of large language models in production environments. The methodology extends beyond punctuation to address broader classes of recursive instabilities in neural text generation systems. 

---
# Conceptualization, Operationalization, and Measurement of Machine Companionship: A Scoping Review 

**Authors**: Jaime Banks, Zhixin Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.18119)  

**Abstract**: The notion of machine companions has long been embedded in social-technological imaginaries. Recent advances in AI have moved those media musings into believable sociality manifested in interfaces, robotic bodies, and devices. Those machines are often referred to colloquially as "companions" yet there is little careful engagement of machine companionship (MC) as a formal concept or measured variable. This PRISMA-guided scoping review systematically samples, surveys, and synthesizes current scholarly works on MC (N = 71; 2017-2025), to that end. Works varied widely in considerations of MC according to guiding theories, dimensions of a-priori specified properties (subjectively positive, sustained over time, co-active, autotelic), and in measured concepts (with more than 50 distinct measured variables). WE ultimately offer a literature-guided definition of MC as an autotelic, coordinated connection between human and machine that unfolds over time and is subjectively positive. 

---
# Mental Health Equity in LLMs: Leveraging Multi-Hop Question Answering to Detect Amplified and Silenced Perspectives 

**Authors**: Batool Haider, Atmika Gorti, Aman Chadha, Manas Gaur  

**Link**: [PDF](https://arxiv.org/pdf/2506.18116)  

**Abstract**: Large Language Models (LLMs) in mental healthcare risk propagating biases that reinforce stigma and harm marginalized groups. While previous research identified concerning trends, systematic methods for detecting intersectional biases remain limited. This work introduces a multi-hop question answering (MHQA) framework to explore LLM response biases in mental health discourse. We analyze content from the Interpretable Mental Health Instruction (IMHI) dataset across symptom presentation, coping mechanisms, and treatment approaches. Using systematic tagging across age, race, gender, and socioeconomic status, we investigate bias patterns at demographic intersections. We evaluate four LLMs: Claude 3.5 Sonnet, Jamba 1.6, Gemma 3, and Llama 4, revealing systematic disparities across sentiment, demographics, and mental health conditions. Our MHQA approach demonstrates superior detection compared to conventional methods, identifying amplification points where biases magnify through sequential reasoning. We implement two debiasing techniques: Roleplay Simulation and Explicit Bias Reduction, achieving 66-94% bias reductions through few-shot prompting with BBQ dataset examples. These findings highlight critical areas where LLMs reproduce mental healthcare biases, providing actionable insights for equitable AI development. 

---
# RL for Reasoning by Adaptively Revealing Rationales 

**Authors**: Mohammad Hossein Amani, Aryo Lotfi, Nicolas Mario Baldwin, Samy Bengio, Mehrdad Farajtabar, Emmanuel Abbe, Robert West  

**Link**: [PDF](https://arxiv.org/pdf/2506.18110)  

**Abstract**: We propose that reinforcement learning (RL) from partial expert demonstrations is not merely a training heuristic, but a promising framework for solving complex sequence generation tasks. Supervised fine-tuning (SFT) relies on dense ground-truth labels, which become increasingly costly as sequence length grows. RL, on the other hand, struggles with sparse rewards and a combinatorially large output space. We address this by introducing adaptive backtracking (AdaBack), a per-sample curriculum learning algorithm that reveals only a partial prefix of the target output during training. The supervision length is adjusted dynamically for each sample based on the model's past reward signal, allowing it to incrementally learn to complete reasoning chains by conditioning on correct partial solutions. We investigate this intermediate regime between SFT and RL and argue that per-sample curriculum learning is more than a trade-off between efficiency and generality, it can succeed in tasks with long sequences of latent dependencies where SFT and RL both fail to generalize. Using a synthetic task with latent parity constraints, we show that our adaptive curriculum over partial answers reliably solves problems that are otherwise intractable. On mathematical reasoning benchmarks (MATH, GSM8k), we find that curriculum learning enables models to solve problems that RL alone cannot, acquiring new reasoning capabilities through incremental exposure to partial solutions. 

---
# ShareGPT-4o-Image: Aligning Multimodal Models with GPT-4o-Level Image Generation 

**Authors**: Junying Chen, Zhenyang Cai, Pengcheng Chen, Shunian Chen, Ke Ji, Xidong Wang, Yunjin Yang, Benyou Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.18095)  

**Abstract**: Recent advances in multimodal generative models have unlocked photorealistic, instruction-aligned image generation, yet leading systems like GPT-4o-Image remain proprietary and inaccessible. To democratize these capabilities, we present ShareGPT-4o-Image, the first dataset comprising 45K text-to-image and 46K text-and-image-to-image data, all synthesized using GPT-4o's image generation capabilities for distilling its advanced image generation abilities. Leveraging this dataset, we develop Janus-4o, a multimodal large language model capable of both text-to-image and text-and-image-to-image generation. Janus-4o not only significantly improves text-to-image generation over its predecessor, Janus-Pro, but also newly supports text-and-image-to-image generation. Notably, it achieves impressive performance in text-and-image-to-image generation from scratch, using only 91K synthetic samples and 6 hours of training on an 8 A800-GPU machine. We hope the release of ShareGPT-4o-Image and Janus-4o will foster open research in photorealistic, instruction-aligned image generation. 

---
# RoboTwin 2.0: A Scalable Data Generator and Benchmark with Strong Domain Randomization for Robust Bimanual Robotic Manipulation 

**Authors**: Tianxing Chen, Zanxin Chen, Baijun Chen, Zijian Cai, Yibin Liu, Qiwei Liang, Zixuan Li, Xianliang Lin, Yiheng Ge, Zhenyu Gu, Weiliang Deng, Yubin Guo, Tian Nian, Xuanbing Xie, Qiangyu Chen, Kailun Su, Tianling Xu, Guodong Liu, Mengkang Hu, Huan-ang Gao, Kaixuan Wang, Zhixuan Liang, Yusen Qin, Xiaokang Yang, Ping Luo, Yao Mu  

**Link**: [PDF](https://arxiv.org/pdf/2506.18088)  

**Abstract**: Simulation-based data synthesis has emerged as a powerful paradigm for enhancing real-world robotic manipulation. However, existing synthetic datasets remain insufficient for robust bimanual manipulation due to two challenges: (1) the lack of an efficient, scalable data generation method for novel tasks, and (2) oversimplified simulation environments that fail to capture real-world complexity. We present RoboTwin 2.0, a scalable simulation framework that enables automated, large-scale generation of diverse and realistic data, along with unified evaluation protocols for dual-arm manipulation. We first construct RoboTwin-OD, a large-scale object library comprising 731 instances across 147 categories, each annotated with semantic and manipulation-relevant labels. Building on this foundation, we develop an expert data synthesis pipeline that combines multimodal large language models (MLLMs) with simulation-in-the-loop refinement to generate task-level execution code automatically. To improve sim-to-real transfer, RoboTwin 2.0 incorporates structured domain randomization along five axes: clutter, lighting, background, tabletop height and language instructions, thereby enhancing data diversity and policy robustness. We instantiate this framework across 50 dual-arm tasks spanning five robot embodiments, and pre-collect over 100,000 domain-randomized expert trajectories. Empirical results show a 10.9% gain in code generation success and improved generalization to novel real-world scenarios. A VLA model fine-tuned on our dataset achieves a 367% relative improvement (42.0% vs. 9.0%) on unseen scene real-world tasks, while zero-shot models trained solely on our synthetic data achieve a 228% relative gain, highlighting strong generalization without real-world supervision. We release the data generator, benchmark, dataset, and code to support scalable research in robust bimanual manipulation. 

---
# Federated Learning-Based Data Collaboration Method for Enhancing Edge Cloud AI System Security Using Large Language Models 

**Authors**: Huaiying Luo, Cheng Ji  

**Link**: [PDF](https://arxiv.org/pdf/2506.18087)  

**Abstract**: With the widespread application of edge computing and cloud systems in AI-driven applications, how to maintain efficient performance while ensuring data privacy has become an urgent security issue. This paper proposes a federated learning-based data collaboration method to improve the security of edge cloud AI systems, and use large-scale language models (LLMs) to enhance data privacy protection and system robustness. Based on the existing federated learning framework, this method introduces a secure multi-party computation protocol, which optimizes the data aggregation and encryption process between distributed nodes by using LLM to ensure data privacy and improve system efficiency. By combining advanced adversarial training techniques, the model enhances the resistance of edge cloud AI systems to security threats such as data leakage and model poisoning. Experimental results show that the proposed method is 15% better than the traditional federated learning method in terms of data protection and model robustness. 

---
# Distributionally robust minimization in meta-learning for system identification 

**Authors**: Matteo Rufolo, Dario Piga, Marco Forgione  

**Link**: [PDF](https://arxiv.org/pdf/2506.18074)  

**Abstract**: Meta learning aims at learning how to solve tasks, and thus it allows to estimate models that can be quickly adapted to new scenarios. This work explores distributionally robust minimization in meta learning for system identification. Standard meta learning approaches optimize the expected loss, overlooking task variability. We use an alternative approach, adopting a distributionally robust optimization paradigm that prioritizes high-loss tasks, enhancing performance in worst-case scenarios. Evaluated on a meta model trained on a class of synthetic dynamical systems and tested in both in-distribution and out-of-distribution settings, the proposed approach allows to reduce failures in safety-critical applications. 

---
# Multimodal Medical Image Binding via Shared Text Embeddings 

**Authors**: Yunhao Liu, Suyang Xi, Shiqi Liu, Hong Ding, Chicheng Jin, Chenxi Yang, Junjun He, Yiqing Shen  

**Link**: [PDF](https://arxiv.org/pdf/2506.18072)  

**Abstract**: Medical image analysis increasingly relies on the integration of multiple imaging modalities to capture complementary anatomical and functional information, enabling more accurate diagnosis and treatment planning. Achieving aligned feature representations across these diverse modalities is therefore important for effective multimodal analysis. While contrastive language-image pre-training (CLIP) and its variant have enabled image-text alignments, they require explicitly paired data between arbitrary two modalities, which is difficult to acquire in medical contexts. To address the gap, we present Multimodal Medical Image Binding with Text (M\textsuperscript{3}Bind), a novel pre-training framework that enables seamless alignment of multiple medical imaging modalities through a shared text representation space without requiring explicit paired data between any two medical image modalities. Specifically, based on the insight that different images can naturally bind with text, M\textsuperscript{3}Bind first fine-tunes pre-trained CLIP-like image-text models to align their modality-specific text embedding space while preserving their original image-text alignments. Subsequently, we distill these modality-specific text encoders into a unified model, creating a shared text embedding space. Experiments on X-ray, CT, retina, ECG, and pathological images on multiple downstream tasks demonstrate that M\textsuperscript{3}Bind achieves state-of-the-art performance in zero-shot, few-shot classification and cross-modal retrieval tasks compared to its CLIP-like counterparts. These results validate M\textsuperscript{3}Bind's effectiveness in achieving cross-image-modal alignment for medical analysis. 

---
# MUPA: Towards Multi-Path Agentic Reasoning for Grounded Video Question Answering 

**Authors**: Jisheng Dang, Huilin Song, Junbin Xiao, Bimei Wang, Han Peng, Haoxuan Li, Xun Yang, Meng Wang, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2506.18071)  

**Abstract**: Grounded Video Question Answering (Grounded VideoQA) requires aligning textual answers with explicit visual evidence. However, modern multimodal models often rely on linguistic priors and spurious correlations, resulting in poorly grounded predictions. In this work, we propose MUPA, a cooperative MUlti-Path Agentic approach that unifies video grounding, question answering, answer reflection and aggregation to tackle Grounded VideoQA. MUPA features three distinct reasoning paths on the interplay of grounding and QA agents in different chronological orders, along with a dedicated reflection agent to judge and aggregate the multi-path results to accomplish consistent QA and grounding. This design markedly improves grounding fidelity without sacrificing answer accuracy. Despite using only 2B parameters, our method outperforms all 7B-scale competitors. When scaled to 7B parameters, MUPA establishes new state-of-the-art results, with Acc@GQA of 30.3% and 47.4% on NExT-GQA and DeVE-QA respectively, demonstrating MUPA' effectiveness towards trustworthy video-language understanding. Our code is available in this https URL. 

---
# Mechanistic Interpretability in the Presence of Architectural Obfuscation 

**Authors**: Marcos Florencio, Thomas Barton  

**Link**: [PDF](https://arxiv.org/pdf/2506.18053)  

**Abstract**: Architectural obfuscation - e.g., permuting hidden-state tensors, linearly transforming embedding tables, or remapping tokens - has recently gained traction as a lightweight substitute for heavyweight cryptography in privacy-preserving large-language-model (LLM) inference. While recent work has shown that these techniques can be broken under dedicated reconstruction attacks, their impact on mechanistic interpretability has not been systematically studied. In particular, it remains unclear whether scrambling a network's internal representations truly thwarts efforts to understand how the model works, or simply relocates the same circuits to an unfamiliar coordinate system. We address this gap by analyzing a GPT-2-small model trained from scratch with a representative obfuscation map. Assuming the obfuscation map is private and the original basis is hidden (mirroring an honest-but-curious server), we apply logit-lens attribution, causal path-patching, and attention-head ablation to locate and manipulate known circuits. Our findings reveal that obfuscation dramatically alters activation patterns within attention heads yet preserves the layer-wise computational graph. This disconnect hampers reverse-engineering of user prompts: causal traces lose their alignment with baseline semantics, and token-level logit attributions become too noisy to reconstruct. At the same time, feed-forward and residual pathways remain functionally intact, suggesting that obfuscation degrades fine-grained interpretability without compromising top-level task performance. These results establish quantitative evidence that architectural obfuscation can simultaneously (i) retain global model behaviour and (ii) impede mechanistic analyses of user-specific content. By mapping where interpretability breaks down, our study provides guidance for future privacy defences and for robustness-aware interpretability tooling. 

---
# The Democratic Paradox in Large Language Models' Underestimation of Press Freedom 

**Authors**: I. Loaiza, R. Vestrelli, A. Fronzetti Colladon, R. Rigobon  

**Link**: [PDF](https://arxiv.org/pdf/2506.18045)  

**Abstract**: As Large Language Models (LLMs) increasingly mediate global information access for millions of users worldwide, their alignment and biases have the potential to shape public understanding and trust in fundamental democratic institutions, such as press freedom. In this study, we uncover three systematic distortions in the way six popular LLMs evaluate press freedom in 180 countries compared to expert assessments of the World Press Freedom Index (WPFI). The six LLMs exhibit a negative misalignment, consistently underestimating press freedom, with individual models rating between 71% to 93% of countries as less free. We also identify a paradoxical pattern we term differential misalignment: LLMs disproportionately underestimate press freedom in countries where it is strongest. Additionally, five of the six LLMs exhibit positive home bias, rating their home countries' press freedoms more favorably than would be expected given their negative misalignment with the human benchmark. In some cases, LLMs rate their home countries between 7% to 260% more positively than expected. If LLMs are set to become the next search engines and some of the most important cultural tools of our time, they must ensure accurate representations of the state of our human and civic rights globally. 

---
# Pathwise Explanation of ReLU Neural Networks 

**Authors**: Seongwoo Lim, Won Jo, Joohyung Lee, Jaesik Choi  

**Link**: [PDF](https://arxiv.org/pdf/2506.18037)  

**Abstract**: Neural networks have demonstrated a wide range of successes, but their ``black box" nature raises concerns about transparency and reliability. Previous research on ReLU networks has sought to unwrap these networks into linear models based on activation states of all hidden units. In this paper, we introduce a novel approach that considers subsets of the hidden units involved in the decision making path. This pathwise explanation provides a clearer and more consistent understanding of the relationship between the input and the decision-making process. Our method also offers flexibility in adjusting the range of explanations within the input, i.e., from an overall attribution input to particular components within the input. Furthermore, it allows for the decomposition of explanations for a given input for more detailed explanations. Experiments demonstrate that our method outperforms others both quantitatively and qualitatively. 

---
# Pre-Trained LLM is a Semantic-Aware and Generalizable Segmentation Booster 

**Authors**: Fenghe Tang, Wenxin Ma, Zhiyang He, Xiaodong Tao, Zihang Jiang, S. Kevin Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.18034)  

**Abstract**: With the advancement of Large Language Model (LLM) for natural language processing, this paper presents an intriguing finding: a frozen pre-trained LLM layer can process visual tokens for medical image segmentation tasks. Specifically, we propose a simple hybrid structure that integrates a pre-trained, frozen LLM layer within the CNN encoder-decoder segmentation framework (LLM4Seg). Surprisingly, this design improves segmentation performance with a minimal increase in trainable parameters across various modalities, including ultrasound, dermoscopy, polypscopy, and CT scans. Our in-depth analysis reveals the potential of transferring LLM's semantic awareness to enhance segmentation tasks, offering both improved global understanding and better local modeling capabilities. The improvement proves robust across different LLMs, validated using LLaMA and DeepSeek. 

---
# PP-DocBee2: Improved Baselines with Efficient Data for Multimodal Document Understanding 

**Authors**: Kui Huang, Xinrong Chen, Wenyu Lv, Jincheng Liao, Guanzhong Wang, Yi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.18023)  

**Abstract**: This report introduces PP-DocBee2, an advanced version of the PP-DocBee, designed to enhance multimodal document understanding. Built on a large multimodal model architecture, PP-DocBee2 addresses the limitations of its predecessor through key technological improvements, including enhanced synthetic data quality, improved visual feature fusion strategy, and optimized inference methodologies. These enhancements yield an $11.4\%$ performance boost on internal benchmarks for Chinese business documents, and reduce inference latency by $73.0\%$ to the vanilla version. A key innovation of our work is a data quality optimization strategy for multimodal document tasks. By employing a large-scale multimodal pre-trained model to evaluate data, we apply a novel statistical criterion to filter outliers, ensuring high-quality training data. Inspired by insights into underutilized intermediate features in multimodal models, we enhance the ViT representational capacity by decomposing it into layers and applying a novel feature fusion strategy to improve complex reasoning. The source code and pre-trained model are available at \href{this https URL}{this https URL}. 

---
# Auto-Regressive Surface Cutting 

**Authors**: Yang Li, Victor Cheung, Xinhai Liu, Yuguang Chen, Zhongjin Luo, Biwen Lei, Haohan Weng, Zibo Zhao, Jingwei Huang, Zhuo Chen, Chunchao Guo  

**Link**: [PDF](https://arxiv.org/pdf/2506.18017)  

**Abstract**: Surface cutting is a fundamental task in computer graphics, with applications in UV parameterization, texture mapping, and mesh decomposition. However, existing methods often produce technically valid but overly fragmented atlases that lack semantic coherence. We introduce SeamGPT, an auto-regressive model that generates cutting seams by mimicking professional workflows. Our key technical innovation lies in formulating surface cutting as a next token prediction task: sample point clouds on mesh vertices and edges, encode them as shape conditions, and employ a GPT-style transformer to sequentially predict seam segments with quantized 3D coordinates. Our approach achieves exceptional performance on UV unwrapping benchmarks containing both manifold and non-manifold meshes, including artist-created, and 3D-scanned models. In addition, it enhances existing 3D segmentation tools by providing clean boundaries for part decomposition. 

---
# ADA-DPM: A Neural Descriptors-based Adaptive Noise Point Filtering Strategy for SLAM 

**Authors**: Yongxin Shao, Binrui Wang, Aihong Tan  

**Link**: [PDF](https://arxiv.org/pdf/2506.18016)  

**Abstract**: LiDAR SLAM has demonstrated significant application value in various fields, including mobile robot navigation and high-precision map construction. However, existing methods often need to make a trade-off between positioning accuracy and system robustness when faced with dynamic object interference, point cloud noise, and unstructured environments. To address this challenge, we propose an adaptive noise filtering SLAM strategy-ADA-DPM, achieving excellent preference in both aspects. We design the Dynamic Segmentation Head to predict the category of feature points belonging to dynamic points, to eliminate dynamic feature points; design the Global Importance Scoring Head to adaptively select feature points with higher contribution and features while suppressing noise interference; and construct the Cross Layer Intra-Graph Convolution Module (GLI-GCN) to fuse multi-scale neighborhood structures, thereby enhancing the discriminative ability of overlapping features. Finally, to further validate the effectiveness of our method, we tested it on several publicly available datasets and achieved outstanding results. 

---
# Probing the Embedding Space of Transformers via Minimal Token Perturbations 

**Authors**: Eddie Conti, Alejandro Astruc, Alvaro Parafita, Axel Brando  

**Link**: [PDF](https://arxiv.org/pdf/2506.18011)  

**Abstract**: Understanding how information propagates through Transformer models is a key challenge for interpretability. In this work, we study the effects of minimal token perturbations on the embedding space. In our experiments, we analyze the frequency of which tokens yield to minimal shifts, highlighting that rare tokens usually lead to larger shifts. Moreover, we study how perturbations propagate across layers, demonstrating that input information is increasingly intermixed in deeper layers. Our findings validate the common assumption that the first layers of a model can be used as proxies for model explanations. Overall, this work introduces the combination of token perturbations and shifts on the embedding space as a powerful tool for model interpretability. 

---
# h-calibration: Rethinking Classifier Recalibration with Probabilistic Error-Bounded Objective 

**Authors**: Wenjian Huang, Guiping Cao, Jiahao Xia, Jingkun Chen, Hao Wang, Jianguo Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.17968)  

**Abstract**: Deep neural networks have demonstrated remarkable performance across numerous learning tasks but often suffer from miscalibration, resulting in unreliable probability outputs. This has inspired many recent works on mitigating miscalibration, particularly through post-hoc recalibration methods that aim to obtain calibrated probabilities without sacrificing the classification performance of pre-trained models. In this study, we summarize and categorize previous works into three general strategies: intuitively designed methods, binning-based methods, and methods based on formulations of ideal calibration. Through theoretical and practical analysis, we highlight ten common limitations in previous approaches. To address these limitations, we propose a probabilistic learning framework for calibration called h-calibration, which theoretically constructs an equivalent learning formulation for canonical calibration with boundedness. On this basis, we design a simple yet effective post-hoc calibration algorithm. Our method not only overcomes the ten identified limitations but also achieves markedly better performance than traditional methods, as validated by extensive experiments. We further analyze, both theoretically and experimentally, the relationship and advantages of our learning objective compared to traditional proper scoring rule. In summary, our probabilistic framework derives an approximately equivalent differentiable objective for learning error-bounded calibrated probabilities, elucidating the correspondence and convergence properties of computational statistics with respect to theoretical bounds in canonical calibration. The theoretical effectiveness is verified on standard post-hoc calibration benchmarks by achieving state-of-the-art performance. This research offers valuable reference for learning reliable likelihood in related fields. 

---
# Adapting Vision-Language Models for Evaluating World Models 

**Authors**: Mariya Hendriksen, Tabish Rashid, David Bignell, Raluca Georgescu, Abdelhak Lemkhenter, Katja Hofmann, Sam Devlin, Sarah Parisot  

**Link**: [PDF](https://arxiv.org/pdf/2506.17967)  

**Abstract**: World models -- generative models that simulate environment dynamics conditioned on past observations and actions -- are gaining prominence in planning, simulation, and embodied AI. However, evaluating their rollouts remains a fundamental challenge, requiring fine-grained, temporally grounded assessment of action alignment and semantic consistency -- capabilities not captured by existing metrics. Vision-Language Models (VLMs) have shown promise as automatic evaluators of generative content due to their strong multimodal reasoning abilities. Yet, their use in fine-grained, temporally sensitive evaluation tasks remains limited and requires targeted adaptation. We introduce a evaluation protocol targeting two recognition tasks -- action recognition and character recognition -- each assessed across binary, multiple-choice, and open-ended formats. To support this, we present UNIVERSE (UNIfied Vision-language Evaluator for Rollouts in Simulated Environments), a method for adapting VLMs to rollout evaluation under data and compute constraints. We conduct a large-scale study comparing full, partial, and parameter-efficient finetuning across task formats, context lengths, sampling strategies, and data compositions. The resulting unified evaluator matches the performance of task-specific baselines using a single checkpoint. Human studies confirm strong alignment with human judgments, establishing UNIVERSE as a scalable, semantics-aware evaluator for world models. 

---
# OmniESI: A unified framework for enzyme-substrate interaction prediction with progressive conditional deep learning 

**Authors**: Zhiwei Nie, Hongyu Zhang, Hao Jiang, Yutian Liu, Xiansong Huang, Fan Xu, Jie Fu, Zhixiang Ren, Yonghong Tian, Wen-Bin Zhang, Jie Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.17963)  

**Abstract**: Understanding and modeling enzyme-substrate interactions is crucial for catalytic mechanism research, enzyme engineering, and metabolic engineering. Although a large number of predictive methods have emerged, they do not incorporate prior knowledge of enzyme catalysis to rationally modulate general protein-molecule features that are misaligned with catalytic patterns. To address this issue, we introduce a two-stage progressive framework, OmniESI, for enzyme-substrate interaction prediction through conditional deep learning. By decomposing the modeling of enzyme-substrate interactions into a two-stage progressive process, OmniESI incorporates two conditional networks that respectively emphasize enzymatic reaction specificity and crucial catalysis-related interactions, facilitating a gradual feature modulation in the latent space from general protein-molecule domain to catalysis-aware domain. On top of this unified architecture, OmniESI can adapt to a variety of downstream tasks, including enzyme kinetic parameter prediction, enzyme-substrate pairing prediction, enzyme mutational effect prediction, and enzymatic active site annotation. Under the multi-perspective performance evaluation of in-distribution and out-of-distribution settings, OmniESI consistently delivered superior performance than state-of-the-art specialized methods across seven benchmarks. More importantly, the proposed conditional networks were shown to internalize the fundamental patterns of catalytic efficiency while significantly improving prediction performance, with only negligible parameter increases (0.16%), as demonstrated by ablation studies on key components. Overall, OmniESI represents a unified predictive approach for enzyme-substrate interactions, providing an effective tool for catalytic mechanism cracking and enzyme engineering with strong generalization and broad applicability. 

---
# GeNIE: A Generalizable Navigation System for In-the-Wild Environments 

**Authors**: Jiaming Wang, Diwen Liu, Jizhuo Chen, Jiaxuan Da, Nuowen Qian, Tram Minh Man, Harold Soh  

**Link**: [PDF](https://arxiv.org/pdf/2506.17960)  

**Abstract**: Reliable navigation in unstructured, real-world environments remains a significant challenge for embodied agents, especially when operating across diverse terrains, weather conditions, and sensor configurations. In this paper, we introduce GeNIE (Generalizable Navigation System for In-the-Wild Environments), a robust navigation framework designed for global deployment. GeNIE integrates a generalizable traversability prediction model built on SAM2 with a novel path fusion strategy that enhances planning stability in noisy and ambiguous settings. We deployed GeNIE in the Earth Rover Challenge (ERC) at ICRA 2025, where it was evaluated across six countries spanning three continents. GeNIE took first place and achieved 79% of the maximum possible score, outperforming the second-best team by 17%, and completed the entire competition without a single human intervention. These results set a new benchmark for robust, generalizable outdoor robot navigation. We will release the codebase, pretrained model weights, and newly curated datasets to support future research in real-world navigation. 

---
# Scatter-Based Innovation Propagation in Large Language Models for Multi-Stage Process Adaptation 

**Authors**: Hong Su  

**Link**: [PDF](https://arxiv.org/pdf/2506.17949)  

**Abstract**: Large Language Models (LLMs) exhibit strong capabilities in reproducing and extending patterns observed during pretraining but often struggle to generalize novel ideas beyond their original context. This paper addresses the challenge of applying such localized innovations - introduced at a specific stage or component - to other parts of a multi-stage process. We propose a scatter-based innovation expansion model (innovation scatter model) that guides the LLM through a four-step process: (1) identifying the core innovation by comparing the user's input with its surrounding context, (2) generalizing the innovation by removing references to specific stages or components, (3) determining whether the generalized innovation applies to a broader scope beyond the original stage, and (4) systematically applying it to other structurally similar stages using the LLM. This model leverages structural redundancy across stages to improve the applicability of novel ideas. Verification results demonstrate that the innovation scatter model enables LLMs to extend innovations across structurally similar stages, thereby enhancing generalization and reuse. 

---
# Greedy Selection under Independent Increments: A Toy Model Analysis 

**Authors**: Huitao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.17941)  

**Abstract**: We study an iterative selection problem over N i.i.d. discrete-time stochastic processes with independent increments. At each stage, a fixed number of processes are retained based on their observed values. Under this simple model, we prove that the optimal strategy for selecting the final maximum-value process is to apply greedy selection at each stage. While the result relies on strong independence assumptions, it offers a clean justification for greedy heuristics in multi-stage elimination settings and may serve as a toy example for understanding related algorithms in high-dimensional applications. 

---
# An entropy-optimal path to humble AI 

**Authors**: Davide Bassetti, Lukáš Pospíšil, Michael Groom, Terence J. O'Kane, Illia Horenko  

**Link**: [PDF](https://arxiv.org/pdf/2506.17940)  

**Abstract**: Progress of AI has led to a creation of very successful, but by no means humble models and tools, especially regarding (i) the huge and further exploding costs and resources they demand, and (ii) the over-confidence of these tools with the answers they provide. Here we introduce a novel mathematical framework for a non-equilibrium entropy-optimizing reformulation of Boltzmann machines based on the exact law of total probability. It results in the highly-performant, but much cheaper, gradient-descent-free learning framework with mathematically-justified existence and uniqueness criteria, and answer confidence/reliability measures. Comparisons to state-of-the-art AI tools in terms of performance, cost and the model descriptor lengths on a set of synthetic problems with varying complexity reveal that the proposed method results in more performant and slim models, with the descriptor lengths being very close to the intrinsic complexity scaling bounds for the underlying problems. Applying this framework to historical climate data results in models with systematically higher prediction skills for the onsets of La Niña and El Niño climate phenomena, requiring just few years of climate data for training - a small fraction of what is necessary for contemporary climate prediction tools. 

---
# GEMeX-ThinkVG: Towards Thinking with Visual Grounding in Medical VQA via Reinforcement Learning 

**Authors**: Bo Liu, Xiangyu Zhao, Along He, Yidi Chen, Huazhu Fu, Xiao-Ming Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.17939)  

**Abstract**: Medical visual question answering aims to support clinical decision-making by enabling models to answer natural language questions based on medical images. While recent advances in multi-modal learning have significantly improved performance, current methods still suffer from limited answer reliability and poor interpretability, impairing the ability of clinicians and patients to understand and trust model-generated answers. To address this, this work first proposes a Thinking with Visual Grounding (ThinkVG) dataset wherein the answer generation is decomposed into intermediate reasoning steps that explicitly ground relevant visual regions of the medical image, thereby providing fine-grained explainability. Furthermore, we introduce a novel verifiable reward mechanism for reinforcement learning to guide post-training, improving the alignment between the model's reasoning process and its final answer. Remarkably, our method achieves comparable performance using only one-eighth of the training data, demonstrating the efficiency and effectiveness of the proposal. The dataset is available at this https URL. 

---
# Software Reuse in the Generative AI Era: From Cargo Cult Towards AI Native Software Engineering 

**Authors**: Tommi Mikkonen, Antero Taivalsaari  

**Link**: [PDF](https://arxiv.org/pdf/2506.17937)  

**Abstract**: Software development is currently under a paradigm shift in which artificial intelligence and generative software reuse are taking the center stage in software creation. Consequently, earlier software reuse practices and methods are rapidly being replaced by AI-assisted approaches in which developers place their trust on code that has been generated by artificial intelligence. This is leading to a new form of software reuse that is conceptually not all that different from cargo cult development. In this paper we discuss the implications of AI-assisted generative software reuse in the context of emerging "AI native" software engineering, bring forth relevant questions, and define a tentative research agenda and call to action for tackling some of the central issues associated with this approach. 

---
# When concept-based XAI is imprecise: Do people distinguish between generalisations and misrepresentations? 

**Authors**: Romy Müller  

**Link**: [PDF](https://arxiv.org/pdf/2506.17936)  

**Abstract**: Concept-based explainable artificial intelligence (C-XAI) can help reveal the inner representations of AI models. Understanding these representations is particularly important in complex tasks like safety evaluation. Such tasks rely on high-level semantic information (e.g., about actions) to make decisions about abstract categories (e.g., whether a situation is dangerous). In this context, it may desirable for C-XAI concepts to show some variability, suggesting that the AI is capable of generalising beyond the concrete details of a situation. However, it is unclear whether people recognise and appreciate such generalisations and can distinguish them from other, less desirable forms of imprecision. This was investigated in an experimental railway safety scenario. Participants evaluated the performance of a simulated AI that evaluated whether traffic scenes involving people were dangerous. To explain these decisions, the AI provided concepts in the form of similar image snippets. These concepts differed in their match with the classified image, either regarding a highly relevant feature (i.e., relation to tracks) or a less relevant feature (i.e., actions). Contrary to the hypotheses, concepts that generalised over less relevant features led to ratings that were lower than for precisely matching concepts and comparable to concepts that systematically misrepresented these features. Conversely, participants were highly sensitive to imprecisions in relevant features. These findings cast doubts on whether people spontaneously recognise generalisations. Accordingly, they might not be able to infer from C-XAI concepts whether AI models have gained a deeper understanding of complex situations. 

---
# A GenAI System for Improved FAIR Independent Biological Database Integration 

**Authors**: Syed N. Sakib, Kallol Naha, Sajratul Y. Rubaiat, Hasan M. Jamil  

**Link**: [PDF](https://arxiv.org/pdf/2506.17934)  

**Abstract**: Life sciences research increasingly requires identifying, accessing, and effectively processing data from an ever-evolving array of information sources on the Linked Open Data (LOD) network. This dynamic landscape places a significant burden on researchers, as the quality of query responses depends heavily on the selection and semantic integration of data sources --processes that are often labor-intensive, error-prone, and costly. While the adoption of FAIR (Findable, Accessible, Interoperable, and Reusable) data principles has aimed to address these challenges, barriers to efficient and accurate scientific data processing persist.
In this paper, we introduce FAIRBridge, an experimental natural language-based query processing system designed to empower scientists to discover, access, and query biological databases, even when they are not FAIR-compliant. FAIRBridge harnesses the capabilities of AI to interpret query intents, map them to relevant databases described in scientific literature, and generate executable queries via intelligent resource access plans. The system also includes robust tools for mitigating low-quality query processing, ensuring high fidelity and responsiveness in the information delivered.
FAIRBridge's autonomous query processing framework enables users to explore alternative data sources, make informed choices at every step, and leverage community-driven crowd curation when needed. By providing a user-friendly, automated hypothesis-testing platform in natural English, FAIRBridge significantly enhances the integration and processing of scientific data, offering researchers a powerful new tool for advancing their inquiries. 

---
# IDAL: Improved Domain Adaptive Learning for Natural Images Dataset 

**Authors**: Ravi Kant Gupta, Shounak Das, Amit Sethi  

**Link**: [PDF](https://arxiv.org/pdf/2506.17931)  

**Abstract**: We present a novel approach for unsupervised domain adaptation (UDA) for natural images. A commonly-used objective for UDA schemes is to enhance domain alignment in representation space even if there is a domain shift in the input space. Existing adversarial domain adaptation methods may not effectively align different domains of multimodal distributions associated with classification problems. Our approach has two main features. Firstly, its neural architecture uses the deep structure of ResNet and the effective separation of scales of feature pyramidal network (FPN) to work with both content and style features. Secondly, it uses a combination of a novel loss function and judiciously selected existing loss functions to train the network architecture. This tailored combination is designed to address challenges inherent to natural images, such as scale, noise, and style shifts, that occur on top of a multi-modal (multi-class) distribution. The combined loss function not only enhances model accuracy and robustness on the target domain but also speeds up training convergence. Our proposed UDA scheme generalizes better than state-of-the-art for CNN-based methods on Office-Home, Office-31, and VisDA-2017 datasets and comaparable for DomainNet dataset. 

---
# ASTER: Adaptive Spatio-Temporal Early Decision Model for Dynamic Resource Allocation 

**Authors**: Shulun Chen, Wei Shao, Flora D. Salim, Hao Xue  

**Link**: [PDF](https://arxiv.org/pdf/2506.17929)  

**Abstract**: Supporting decision-making has long been a central vision in the field of spatio-temporal intelligence. While prior work has improved the timeliness and accuracy of spatio-temporal forecasting, converting these forecasts into actionable strategies remains a key challenge. A main limitation is the decoupling of the prediction and the downstream decision phases, which can significantly degrade the downstream efficiency. For example, in emergency response, the priority is successful resource allocation and intervention, not just incident prediction. To this end, it is essential to propose an Adaptive Spatio-Temporal Early Decision model (ASTER) that reforms the forecasting paradigm from event anticipation to actionable decision support. This framework ensures that information is directly used for decision-making, thereby maximizing overall effectiveness. Specifically, ASTER introduces a new Resource-aware Spatio-Temporal interaction module (RaST) that adaptively captures long- and short-term dependencies under dynamic resource conditions, producing context-aware spatiotemporal representations. To directly generate actionable decisions, we further design a Preference-oriented decision agent (Poda) based on multi-objective reinforcement learning, which transforms predictive signals into resource-efficient intervention strategies by deriving optimal actions under specific preferences and dynamic constraints. Experimental results on four benchmark datasets demonstrate the state-of-the-art performance of ASTER in improving both early prediction accuracy and resource allocation outcomes across six downstream metrics. 

---
# Permutation Equivariant Model-based Offline Reinforcement Learning for Auto-bidding 

**Authors**: Zhiyu Mou, Miao Xu, Wei Chen, Rongquan Bai, Chuan Yu, Jian Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.17919)  

**Abstract**: Reinforcement learning (RL) for auto-bidding has shifted from using simplistic offline simulators (Simulation-based RL Bidding, SRLB) to offline RL on fixed real datasets (Offline RL Bidding, ORLB). However, ORLB policies are limited by the dataset's state space coverage, offering modest gains. While SRLB expands state coverage, its simulator-reality gap risks misleading policies. This paper introduces Model-based RL Bidding (MRLB), which learns an environment model from real data to bridge this gap. MRLB trains policies using both real and model-generated data, expanding state coverage beyond ORLB. To ensure model reliability, we propose: 1) A permutation equivariant model architecture for better generalization, and 2) A robust offline Q-learning method that pessimistically penalizes model errors. These form the Permutation Equivariant Model-based Offline RL (PE-MORL) algorithm. Real-world experiments show that PE-MORL outperforms state-of-the-art auto-bidding methods. 

---
# Feedback Driven Multi Stereo Vision System for Real-Time Event Analysis 

**Authors**: Mohamed Benkedadra, Matei Mancas, Sidi Ahmed Mahmoudi  

**Link**: [PDF](https://arxiv.org/pdf/2506.17910)  

**Abstract**: 2D cameras are often used in interactive systems. Other systems like gaming consoles provide more powerful 3D cameras for short range depth sensing. Overall, these cameras are not reliable in large, complex environments. In this work, we propose a 3D stereo vision based pipeline for interactive systems, that is able to handle both ordinary and sensitive applications, through robust scene understanding. We explore the fusion of multiple 3D cameras to do full scene reconstruction, which allows for preforming a wide range of tasks, like event recognition, subject tracking, and notification. Using possible feedback approaches, the system can receive data from the subjects present in the environment, to learn to make better decisions, or to adapt to completely new environments. Throughout the paper, we introduce the pipeline and explain our preliminary experimentation and results. Finally, we draw the roadmap for the next steps that need to be taken, in order to get this pipeline into production 

---
# Cause-Effect Driven Optimization for Robust Medical Visual Question Answering with Language Biases 

**Authors**: Huanjia Zhu, Yishu Liu, Xiaozhao Fang, Guangming Lu, Bingzhi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.17903)  

**Abstract**: Existing Medical Visual Question Answering (Med-VQA) models often suffer from language biases, where spurious correlations between question types and answer categories are inadvertently established. To address these issues, we propose a novel Cause-Effect Driven Optimization framework called CEDO, that incorporates three well-established mechanisms, i.e., Modality-driven Heterogeneous Optimization (MHO), Gradient-guided Modality Synergy (GMS), and Distribution-adapted Loss Rescaling (DLR), for comprehensively mitigating language biases from both causal and effectual perspectives. Specifically, MHO employs adaptive learning rates for specific modalities to achieve heterogeneous optimization, thus enhancing robust reasoning capabilities. Additionally, GMS leverages the Pareto optimization method to foster synergistic interactions between modalities and enforce gradient orthogonality to eliminate bias updates, thereby mitigating language biases from the effect side, i.e., shortcut bias. Furthermore, DLR is designed to assign adaptive weights to individual losses to ensure balanced learning across all answer categories, effectively alleviating language biases from the cause side, i.e., imbalance biases within datasets. Extensive experiments on multiple traditional and bias-sensitive benchmarks consistently demonstrate the robustness of CEDO over state-of-the-art competitors. 

---
# EgoWorld: Translating Exocentric View to Egocentric View using Rich Exocentric Observations 

**Authors**: Junho Park, Andrew Sangwoo Ye, Taein Kwon  

**Link**: [PDF](https://arxiv.org/pdf/2506.17896)  

**Abstract**: Egocentric vision is essential for both human and machine visual understanding, particularly in capturing the detailed hand-object interactions needed for manipulation tasks. Translating third-person views into first-person views significantly benefits augmented reality (AR), virtual reality (VR) and robotics applications. However, current exocentric-to-egocentric translation methods are limited by their dependence on 2D cues, synchronized multi-view settings, and unrealistic assumptions such as necessity of initial egocentric frame and relative camera poses during inference. To overcome these challenges, we introduce EgoWorld, a novel two-stage framework that reconstructs an egocentric view from rich exocentric observations, including projected point clouds, 3D hand poses, and textual descriptions. Our approach reconstructs a point cloud from estimated exocentric depth maps, reprojects it into the egocentric perspective, and then applies diffusion-based inpainting to produce dense, semantically coherent egocentric images. Evaluated on the H2O and TACO datasets, EgoWorld achieves state-of-the-art performance and demonstrates robust generalization to new objects, actions, scenes, and subjects. Moreover, EgoWorld shows promising results even on unlabeled real-world examples. 

---
# Multi-turn Jailbreaking via Global Refinement and Active Fabrication 

**Authors**: Hua Tang, Lingyong Yan, Yukun Zhao, Shuaiqiang Wang, Jizhou Huang, Dawei Yin  

**Link**: [PDF](https://arxiv.org/pdf/2506.17881)  

**Abstract**: Large Language Models (LLMs) have achieved exceptional performance across a wide range of tasks. However, they still pose significant safety risks due to the potential misuse for malicious purposes. Jailbreaks, which aim to elicit models to generate harmful content, play a critical role in identifying the underlying security threats. Recent jailbreaking primarily focuses on single-turn scenarios, while the more complicated multi-turn scenarios remain underexplored. Moreover, existing multi-turn jailbreaking techniques struggle to adapt to the evolving dynamics of dialogue as the interaction progresses. To address this limitation, we propose a novel multi-turn jailbreaking method that refines the jailbreaking path globally at each interaction. We also actively fabricate model responses to suppress safety-related warnings, thereby increasing the likelihood of eliciting harmful outputs in subsequent questions. Experimental results demonstrate the superior performance of our method compared with existing single-turn and multi-turn jailbreaking techniques across six state-of-the-art LLMs. Our code is publicly available at this https URL. 

---
# StainPIDR: A Pathological Image Decouplingand Reconstruction Method for StainNormalization Based on Color VectorQuantization and Structure Restaining 

**Authors**: Zheng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.17879)  

**Abstract**: The color appearance of a pathological image is highly related to the imaging protocols, the proportion of different dyes, and the scanning devices. Computer-aided diagnostic systems may deteriorate when facing these color-variant pathological images. In this work, we propose a stain normalization method called StainPIDR. We try to eliminate this color discrepancy by decoupling the image into structure features and vector-quantized color features, restaining the structure features with the target color features, and decoding the stained structure features to normalized pathological images. We assume that color features decoupled by different images with the same color should be exactly the same. Under this assumption, we train a fixed color vector codebook to which the decoupled color features will map. In the restaining part, we utilize the cross-attention mechanism to efficiently stain the structure features. As the target color (decoupled from a selected template image) will also affect the performance of stain normalization, we further design a template image selection algorithm to select a template from a given dataset. In our extensive experiments, we validate the effectiveness of StainPIDR and the template image selection algorithm. All the results show that our method can perform well in the stain normalization task. The code of StainPIDR will be publicly available later. 

---
# SurgVidLM: Towards Multi-grained Surgical Video Understanding with Large Language Model 

**Authors**: Guankun Wang, Wenjin Mo, Junyi Wang, Long Bai, Kun Yuan, Ming Hu, Jinlin Wu, Junjun He, Yiming Huang, Nicolas Padoy, Zhen Lei, Hongbin Liu, Nassir Navab, Hongliang Ren  

**Link**: [PDF](https://arxiv.org/pdf/2506.17873)  

**Abstract**: Recent advances in Multimodal Large Language Models have demonstrated great potential in the medical domain, facilitating users to understand surgical scenes and procedures. Beyond image-based methods, the exploration of Video Large Language Models (Vid-LLMs) has emerged as a promising avenue for capturing the complex sequences of information involved in surgery. However, there is still a lack of Vid-LLMs specialized for fine-grained surgical video understanding tasks, which is crucial for analyzing specific processes or details within a surgical procedure. To bridge this gap, we propose SurgVidLM, the first video language model designed to address both full and fine-grained surgical video comprehension. To train our SurgVidLM, we construct the SVU-31K dataset which consists of over 31K video-instruction pairs, enabling both holistic understanding and detailed analysis of surgical procedures. Furthermore, we introduce the StageFocus mechanism which is a two-stage framework performing the multi-grained, progressive understanding of surgical videos. We also develop the Multi-frequency Fusion Attention to effectively integrate low and high-frequency visual tokens, ensuring the retention of critical information. Experimental results demonstrate that SurgVidLM significantly outperforms state-of-the-art Vid-LLMs in both full and fine-grained video understanding tasks, showcasing its superior capability in capturing complex procedural contexts. 

---
# How Alignment Shrinks the Generative Horizon 

**Authors**: Chenghao Yang, Ari Holtzman  

**Link**: [PDF](https://arxiv.org/pdf/2506.17871)  

**Abstract**: Despite their impressive capabilities, aligned large language models (LLMs) often generate outputs that lack diversity. What drives this stability in the generation? We investigate this phenomenon through the lens of probability concentration in the model's output distribution. To quantify this concentration, we introduce the Branching Factor (BF) -- a token-invariant measure of the effective number of plausible next steps during generation. Our empirical analysis reveals two key findings: (1) BF often decreases as generation progresses, suggesting that LLMs become more predictable as they generate. (2) alignment tuning substantially sharpens the model's output distribution from the outset, reducing BF by nearly an order of magnitude (e.g., from 12 to 1.2) relative to base models. This stark reduction helps explain why aligned models often appear less sensitive to decoding strategies. Building on this insight, we find this stability has surprising implications for complex reasoning. Aligned Chain-of-Thought (CoT) models (e.g., DeepSeek-distilled models), for instance, leverage this effect; by generating longer reasoning chains, they push generation into later, more deterministic (lower BF) stages, resulting in more stable outputs. We hypothesize that alignment tuning does not fundamentally change a model's behavior, but instead steers it toward stylistic tokens (e.g., "Sure") that unlock low-entropy trajectories already present in the base model. This view is supported by nudging experiments, which show that prompting base models with such tokens can similarly reduce BF. Together, our findings establish BF as a powerful diagnostic for understanding and controlling LLM outputs - clarifying how alignment reduces variability, how CoT promotes stable generations, and how base models can be steered away from diversity. 

---
# NestQuant: Post-Training Integer-Nesting Quantization for On-Device DNN 

**Authors**: Jianhang Xie, Chuntao Ding, Xiaqing Li, Shenyuan Ren, Yidong Li, Zhichao Lu  

**Link**: [PDF](https://arxiv.org/pdf/2506.17870)  

**Abstract**: Deploying quantized deep neural network (DNN) models with resource adaptation capabilities on ubiquitous Internet of Things (IoT) devices to provide high-quality AI services can leverage the benefits of compression and meet multi-scenario resource requirements. However, existing dynamic/mixed precision quantization requires retraining or special hardware, whereas post-training quantization (PTQ) has two limitations for resource adaptation: (i) The state-of-the-art PTQ methods only provide one fixed bitwidth model, which makes it challenging to adapt to the dynamic resources of IoT devices; (ii) Deploying multiple PTQ models with diverse bitwidths consumes large storage resources and switching overheads. To this end, this paper introduces a resource-friendly post-training integer-nesting quantization, i.e., NestQuant, for on-device quantized model switching on IoT devices. The proposed NestQuant incorporates the integer weight decomposition, which bit-wise splits quantized weights into higher-bit and lower-bit weights of integer data types. It also contains a decomposed weights nesting mechanism to optimize the higher-bit weights by adaptive rounding and nest them into the original quantized weights. In deployment, we can send and store only one NestQuant model and switch between the full-bit/part-bit model by paging in/out lower-bit weights to adapt to resource changes and reduce consumption. Experimental results on the ImageNet-1K pretrained DNNs demonstrated that the NestQuant model can achieve high performance in top-1 accuracy, and reduce in terms of data transmission, storage consumption, and switching overheads. In particular, the ResNet-101 with INT8 nesting INT6 can achieve 78.1% and 77.9% accuracy for full-bit and part-bit models, respectively, and reduce switching overheads by approximately 78.1% compared with diverse bitwidths PTQ models. 

---
# In-Context Learning Strategies Emerge Rationally 

**Authors**: Daniel Wurgaft, Ekdeep Singh Lubana, Core Francisco Park, Hidenori Tanaka, Gautam Reddy, Noah D. Goodman  

**Link**: [PDF](https://arxiv.org/pdf/2506.17859)  

**Abstract**: Recent work analyzing in-context learning (ICL) has identified a broad set of strategies that describe model behavior in different experimental conditions. We aim to unify these findings by asking why a model learns these disparate strategies in the first place. Specifically, we start with the observation that when trained to learn a mixture of tasks, as is popular in the literature, the strategies learned by a model for performing ICL can be captured by a family of Bayesian predictors: a memorizing predictor, which assumes a discrete prior on the set of seen tasks, and a generalizing predictor, wherein the prior matches the underlying task distribution. Adopting the lens of rational analysis from cognitive science, where a learner's behavior is explained as an optimal adaptation to data given computational constraints, we develop a hierarchical Bayesian framework that almost perfectly predicts Transformer next token predictions throughout training without assuming access to its weights. Under this framework, pretraining is viewed as a process of updating the posterior probability of different strategies, and its inference-time behavior as a posterior-weighted average over these strategies' predictions. Our framework draws on common assumptions about neural network learning dynamics, which make explicit a tradeoff between loss and complexity among candidate strategies: beyond how well it explains the data, a model's preference towards implementing a strategy is dictated by its complexity. This helps explain well-known ICL phenomena, while offering novel predictions: e.g., we show a superlinear trend in the timescale for transition to memorization as task diversity is increased. Overall, our work advances an explanatory and predictive account of ICL grounded in tradeoffs between strategy loss and complexity. 

---
# Pathway-based Progressive Inference (PaPI) for Energy-Efficient Continual Learning 

**Authors**: Suyash Gaurav, Jukka Heikkonen, Jatin Chaudhary  

**Link**: [PDF](https://arxiv.org/pdf/2506.17848)  

**Abstract**: Continual learning systems face the dual challenge of preventing catastrophic forgetting while maintaining energy efficiency, particularly in resource-constrained environments. This paper introduces Pathway-based Progressive Inference (PaPI), a novel theoretical framework that addresses these challenges through a mathematically rigorous approach to pathway selection and adaptation. We formulate continual learning as an energy-constrained optimization problem and provide formal convergence guarantees for our pathway routing mechanisms. Our theoretical analysis demonstrates that PaPI achieves an $\mathcal{O}(K)$ improvement in the stability-plasticity trade-off compared to monolithic architectures, where $K$ is the number of pathways. We derive tight bounds on forgetting rates using Fisher Information Matrix analysis and prove that PaPI's energy consumption scales with the number of active parameters rather than the total model size. Comparative theoretical analysis shows that PaPI provides stronger guarantees against catastrophic forgetting than Elastic Weight Consolidation (EWC) while maintaining better energy efficiency than both EWC and Gradient Episodic Memory (GEM). Our experimental validation confirms these theoretical advantages across multiple benchmarks, demonstrating PaPI's effectiveness for continual learning in energy-constrained settings. Our codes are available at this https URL. 

---
# A Comparative Study of Open-Source Libraries for Synthetic Tabular Data Generation: SDV vs. SynthCity 

**Authors**: Cristian Del Gobbo  

**Link**: [PDF](https://arxiv.org/pdf/2506.17847)  

**Abstract**: High-quality training data is critical to the performance of machine learning models, particularly Large Language Models (LLMs). However, obtaining real, high-quality data can be challenging, especially for smaller organizations and early-stage startups. Synthetic data generators provide a promising solution by replicating the statistical and structural properties of real data while preserving privacy and scalability. This study evaluates the performance of six tabular synthetic data generators from two widely used open-source libraries: SDV (Gaussian Copula, CTGAN, TVAE) and Synthicity (Bayesian Network, CTGAN, TVAE). Using a real-world dataset from the UCI Machine Learning Repository, comprising energy consumption and environmental variables from Belgium, we simulate a low-data regime by training models on only 1,000 rows. Each generator is then tasked with producing synthetic datasets under two conditions: a 1:1 (1,000 rows) and a 1:10 (10,000 rows) input-output ratio. Evaluation is conducted using two criteria: statistical similarity, measured via classical statistics and distributional metrics; and predictive utility, assessed using a "Train on Synthetic, Test on Real" approach with four regression models. While statistical similarity remained consistent across models in both scenarios, predictive utility declined notably in the 1:10 case. The Bayesian Network from Synthicity achieved the highest fidelity in both scenarios, while TVAE from SDV performed best in predictive tasks under the 1:10 setting. Although no significant performance gap was found between the two libraries, SDV stands out for its superior documentation and ease of use, making it more accessible for practitioners. 

---
# THCM-CAL: Temporal-Hierarchical Causal Modelling with Conformal Calibration for Clinical Risk Prediction 

**Authors**: Xin Zhang, Qiyu Wei, Yingjie Zhu, Fanyi Wu, Sophia Ananiadou  

**Link**: [PDF](https://arxiv.org/pdf/2506.17844)  

**Abstract**: Automated clinical risk prediction from electronic health records (EHRs) demands modeling both structured diagnostic codes and unstructured narrative notes. However, most prior approaches either handle these modalities separately or rely on simplistic fusion strategies that ignore the directional, hierarchical causal interactions by which narrative observations precipitate diagnoses and propagate risk across admissions. In this paper, we propose THCM-CAL, a Temporal-Hierarchical Causal Model with Conformal Calibration. Our framework constructs a multimodal causal graph where nodes represent clinical entities from two modalities: Textual propositions extracted from notes and ICD codes mapped to textual descriptions. Through hierarchical causal discovery, THCM-CAL infers three clinically grounded interactions: intra-slice same-modality sequencing, intra-slice cross-modality triggers, and inter-slice risk propagation. To enhance prediction reliability, we extend conformal prediction to multi-label ICD coding, calibrating per-code confidence intervals under complex co-occurrences. Experimental results on MIMIC-III and MIMIC-IV demonstrate the superiority of THCM-CAL. 

---
# Generative Grasp Detection and Estimation with Concept Learning-based Safety Criteria 

**Authors**: Al-Harith Farhad, Khalil Abuibaid, Christiane Plociennik, Achim Wagner, Martin Ruskowski  

**Link**: [PDF](https://arxiv.org/pdf/2506.17842)  

**Abstract**: Neural networks are often regarded as universal equations that can estimate any function. This flexibility, however, comes with the drawback of high complexity, rendering these networks into black box models, which is especially relevant in safety-centric applications. To that end, we propose a pipeline for a collaborative robot (Cobot) grasping algorithm that detects relevant tools and generates the optimal grasp. To increase the transparency and reliability of this approach, we integrate an explainable AI method that provides an explanation for the underlying prediction of a model by extracting the learned features and correlating them to corresponding classes from the input. These concepts are then used as additional criteria to ensure the safe handling of work tools. In this paper, we show the consistency of this approach and the criterion for improving the handover position. This approach was tested in an industrial environment, where a camera system was set up to enable a robot to pick up certain tools and objects. 

---
# Causal Spherical Hypergraph Networks for Modelling Social Uncertainty 

**Authors**: Anoushka Harit, Zhongtian Sun  

**Link**: [PDF](https://arxiv.org/pdf/2506.17840)  

**Abstract**: Human social behaviour is governed by complex interactions shaped by uncertainty, causality, and group dynamics. We propose Causal Spherical Hypergraph Networks (Causal-SphHN), a principled framework for socially grounded prediction that jointly models higher-order structure, directional influence, and epistemic uncertainty. Our method represents individuals as hyperspherical embeddings and group contexts as hyperedges, capturing semantic and relational geometry. Uncertainty is quantified via Shannon entropy over von Mises-Fisher distributions, while temporal causal dependencies are identified using Granger-informed subgraphs. Information is propagated through an angular message-passing mechanism that respects belief dispersion and directional semantics. Experiments on SNARE (offline networks), PHEME (online discourse), and AMIGOS (multimodal affect) show that Causal-SphHN improves predictive accuracy, robustness, and calibration over strong baselines. Moreover, it enables interpretable analysis of influence patterns and social ambiguity. This work contributes a unified causal-geometric approach for learning under uncertainty in dynamic social environments. 

---
# Aligning Frozen LLMs by Reinforcement Learning: An Iterative Reweight-then-Optimize Approach 

**Authors**: Xinnan Zhang, Chenliang Li, Siliang Zeng, Jiaxiang Li, Zhongruo Wang, Kaixiang Lin, Songtao Lu, Alfredo Garcia, Mingyi Hong  

**Link**: [PDF](https://arxiv.org/pdf/2506.17828)  

**Abstract**: Aligning large language models (LLMs) with human preferences usually requires fine-tuning methods such as RLHF and DPO. These methods directly optimize the model parameters, so they cannot be used in test-time to improve model performance, nor are they applicable when the model weights are not accessible. In contrast, test-time methods sidestep weight updates by leveraging reward functions to guide and improve output quality. However, they incur high inference costs, and their one-shot guidance is often based on imperfect reward or value functions, leading to suboptimal outputs. In this work, we present a method named Iterative Reweight-then-Optimize (IRO), a reinforcement learning (RL) framework that performs RL-style alignment of the (frozen) base model without touching its parameters. During training, each iteration (i) samples candidates from the base model, (ii) resamples using current value functions, and (iii) trains a new lightweight value function that guides the next decoding pass. At test time, the value functions are used to guide the base model generation via a search-based optimization process. Notably, users can apply IRO to align a model on their own dataset, similar to OpenAI's reinforcement fine-tuning (RFT), but without requiring access to the model weights. 

---
# Actionable Interpretability via Causal Hypergraphs: Unravelling Batch Size Effects in Deep Learning 

**Authors**: Zhongtian Sun, Anoushka Harit, Pietro Lio  

**Link**: [PDF](https://arxiv.org/pdf/2506.17826)  

**Abstract**: While the impact of batch size on generalisation is well studied in vision tasks, its causal mechanisms remain underexplored in graph and text domains. We introduce a hypergraph-based causal framework, HGCNet, that leverages deep structural causal models (DSCMs) to uncover how batch size influences generalisation via gradient noise, minima sharpness, and model complexity. Unlike prior approaches based on static pairwise dependencies, HGCNet employs hypergraphs to capture higher-order interactions across training dynamics. Using do-calculus, we quantify direct and mediated effects of batch size interventions, providing interpretable, causally grounded insights into optimisation. Experiments on citation networks, biomedical text, and e-commerce reviews show that HGCNet outperforms strong baselines including GCN, GAT, PI-GNN, BERT, and RoBERTa. Our analysis reveals that smaller batch sizes causally enhance generalisation through increased stochasticity and flatter minima, offering actionable interpretability to guide training strategies in deep learning. This work positions interpretability as a driver of principled architectural and optimisation choices beyond post hoc analysis. 

---
# Learning to Dock: A Simulation-based Study on Closing the Sim2Real Gap in Autonomous Underwater Docking 

**Authors**: Kevin Chang, Rakesh Vivekanandan, Noah Pragin, Sean Bullock, Geoffrey Hollinger  

**Link**: [PDF](https://arxiv.org/pdf/2506.17823)  

**Abstract**: Autonomous Underwater Vehicle (AUV) docking in dynamic and uncertain environments is a critical challenge for underwater robotics. Reinforcement learning is a promising method for developing robust controllers, but the disparity between training simulations and the real world, or the sim2real gap, often leads to a significant deterioration in performance. In this work, we perform a simulation study on reducing the sim2real gap in autonomous docking through training various controllers and then evaluating them under realistic disturbances. In particular, we focus on the real-world challenge of docking under different payloads that are potentially outside the original training distribution. We explore existing methods for improving robustness including randomization techniques and history-conditioned controllers. Our findings provide insights into mitigating the sim2real gap when training docking controllers. Furthermore, our work indicates areas of future research that may be beneficial to the marine robotics community. 

---
# CultureMERT: Continual Pre-Training for Cross-Cultural Music Representation Learning 

**Authors**: Angelos-Nikolaos Kanatas, Charilaos Papaioannou, Alexandros Potamianos  

**Link**: [PDF](https://arxiv.org/pdf/2506.17818)  

**Abstract**: Recent advances in music foundation models have improved audio representation learning, yet their effectiveness across diverse musical traditions remains limited. We introduce CultureMERT-95M, a multi-culturally adapted foundation model developed to enhance cross-cultural music representation learning and understanding. To achieve this, we propose a two-stage continual pre-training strategy that integrates learning rate re-warming and re-decaying, enabling stable adaptation even with limited computational resources. Training on a 650-hour multi-cultural data mix, comprising Greek, Turkish, and Indian music traditions, results in an average improvement of 4.9% in ROC-AUC and AP across diverse non-Western music auto-tagging tasks, surpassing prior state-of-the-art, with minimal forgetting on Western-centric benchmarks. We further investigate task arithmetic, an alternative approach to multi-cultural adaptation that merges single-culture adapted models in the weight space. Task arithmetic performs on par with our multi-culturally trained model on non-Western auto-tagging tasks and shows no regression on Western datasets. Cross-cultural evaluation reveals that single-culture models transfer with varying effectiveness across musical traditions, whereas the multi-culturally adapted model achieves the best overall performance. To support research on world music representation learning, we publicly release CultureMERT-95M and CultureMERT-TA-95M, fostering the development of more culturally aware music foundation models. 

---
# RoboMonkey: Scaling Test-Time Sampling and Verification for Vision-Language-Action Models 

**Authors**: Jacky Kwok, Christopher Agia, Rohan Sinha, Matt Foutter, Shulu Li, Ion Stoica, Azalia Mirhoseini, Marco Pavone  

**Link**: [PDF](https://arxiv.org/pdf/2506.17811)  

**Abstract**: Vision-Language-Action (VLA) models have demonstrated remarkable capabilities in visuomotor control, yet ensuring their robustness in unstructured real-world environments remains a persistent challenge. In this paper, we investigate test-time scaling through the lens of sampling and verification as means to enhance the robustness and generalization of VLAs. We first demonstrate that the relationship between action error and the number of generated samples follows an exponentiated power law across a range of VLAs, indicating the existence of inference-time scaling laws. Building on these insights, we introduce RoboMonkey, a test-time scaling framework for VLAs. At deployment, RoboMonkey samples a small set of actions from a VLA, applies Gaussian perturbation and majority voting to construct an action proposal distribution, and then uses a Vision Language Model (VLM)-based verifier to select the optimal action. We propose a synthetic data generation pipeline for training such VLM-based action verifiers, and demonstrate that scaling the synthetic dataset consistently improves verification and downstream accuracy. Through extensive simulated and hardware experiments, we show that pairing existing VLAs with RoboMonkey yields significant performance gains, achieving a 25% absolute improvement on out-of-distribution tasks and 8% on in-distribution tasks. Additionally, when adapting to new robot setups, we show that fine-tuning both VLAs and action verifiers yields a 7% performance increase compared to fine-tuning VLAs alone. 

---
# Reimagining Parameter Space Exploration with Diffusion Models 

**Authors**: Lijun Zhang, Xiao Liu, Hui Guan  

**Link**: [PDF](https://arxiv.org/pdf/2506.17807)  

**Abstract**: Adapting neural networks to new tasks typically requires task-specific fine-tuning, which is time-consuming and reliant on labeled data. We explore a generative alternative that produces task-specific parameters directly from task identity, eliminating the need for task-specific training. To this end, we propose using diffusion models to learn the underlying structure of effective task-specific parameter space and synthesize parameters on demand. Once trained, the task-conditioned diffusion model can generate specialized weights directly from task identifiers. We evaluate this approach across three scenarios: generating parameters for a single seen task, for multiple seen tasks, and for entirely unseen tasks. Experiments show that diffusion models can generate accurate task-specific parameters and support multi-task interpolation when parameter subspaces are well-structured, but fail to generalize to unseen tasks, highlighting both the potential and limitations of this generative solution. 

---
# Expanding Relevance Judgments for Medical Case-based Retrieval Task with Multimodal LLMs 

**Authors**: Catarina Pires, Sérgio Nunes, Luís Filipe Teixeira  

**Link**: [PDF](https://arxiv.org/pdf/2506.17782)  

**Abstract**: Evaluating Information Retrieval (IR) systems relies on high-quality manual relevance judgments (qrels), which are costly and time-consuming to obtain. While pooling reduces the annotation effort, it results in only partially labeled datasets. Large Language Models (LLMs) offer a promising alternative to reducing reliance on manual judgments, particularly in complex domains like medical case-based retrieval, where relevance assessment requires analyzing both textual and visual information. In this work, we explore using a Multimodal Large Language Model (MLLM) to expand relevance judgments, creating a new dataset of automated judgments. Specifically, we employ Gemini 1.5 Pro on the ImageCLEFmed 2013 case-based retrieval task, simulating human assessment through an iteratively refined, structured prompting strategy that integrates binary scoring, instruction-based evaluation, and few-shot learning. We systematically experimented with various prompt configurations to maximize agreement with human judgments. To evaluate agreement between the MLLM and human judgments, we use Cohen's Kappa, achieving a substantial agreement score of 0.6, comparable to inter-annotator agreement typically observed in multimodal retrieval tasks. Starting from the original 15,028 manual judgments (4.72% relevant) across 35 topics, our MLLM-based approach expanded the dataset by over 37x to 558,653 judgments, increasing relevant annotations to 5,950. On average, each medical case query received 15,398 new annotations, with approximately 99% being non-relevant, reflecting the high sparsity typical in this domain. Our results demonstrate the potential of MLLMs to scale relevance judgment collection, offering a promising direction for supporting retrieval evaluation in medical and multimodal IR tasks. 

---
# Toward Autonomous UI Exploration: The UIExplorer Benchmark 

**Authors**: Andrei Cristian Nica, Akshaya Vishnu Kudlu Shanbhogue, Harshil Shah, Aleix Cambray, Tudor Berariu, Lucas Maystre, David Barber  

**Link**: [PDF](https://arxiv.org/pdf/2506.17779)  

**Abstract**: Autonomous agents must know how to explore user interfaces (UIs) for reliable task solving, yet systematic evaluation of this crucial phase is lacking. We introduce UIExplore-Bench, the first benchmark explicitly dedicated to UI exploration. The benchmark evaluates agents with either Structured mode (granting access to layout information like DOM trees) or Screen mode (relying on GUI-only observations such as screenshots and human-like mouse/keyboard interactions) across three levels in a standardized GitLab sandbox environment. We formalize exploration as the process of maximizing the set of actionable UI components discovered and propose a metric, human-normalized UI-Functionalities Observed (hUFO), to quantify the effectiveness of exploration. Our results show that UIExplore-AlGo achieves the leading mean hUFO scores, reaching up to 77.2% of human performance in Structured mode and 59.0% in Screen mode at 2,000 steps, particularly excelling at the Sparse level. The results highlight the relevance of our benchmark, as current agents show a substantial performance gap compared to one hour of human expert exploration, indicating ample room for future advancements. We publicly release the benchmark environment, an exploration dataset, and an evaluation suite to catalyze research into efficient UI exploration strategies and their downstream applications, such as experience-driven task completion and automated training data generation. 

---
# Machine Learning Model Integration with Open World Temporal Logic for Process Automation 

**Authors**: Dyuman Aditya, Colton Payne, Mario Leiva, Paulo Shakarian  

**Link**: [PDF](https://arxiv.org/pdf/2506.17776)  

**Abstract**: Recent advancements in Machine Learning (ML) have yielded powerful models capable of extracting structured information from diverse and complex data sources. However, a significant challenge lies in translating these perceptual or extractive outputs into actionable, reasoned decisions within complex operational workflows. To address these challenges, this paper introduces a novel approach that integrates the outputs from various machine learning models directly with the PyReason framework, an open-world temporal logic programming reasoning engine. PyReason's foundation in generalized annotated logic allows for the seamless incorporation of real-valued outputs (e.g., probabilities, confidence scores) from diverse ML models, treating them as truth intervals within its logical framework. Crucially, PyReason provides mechanisms, implemented in Python, to continuously poll ML model outputs, convert them into logical facts, and dynamically recompute the minimal model, ensuring real-tine adaptive decision-making. Furthermore, its native support for temporal reasoning, knowledge graph integration, and fully explainable interface traces enables sophisticated analysis over time-sensitive process data and existing organizational knowledge. By combining the strengths of perception and extraction from ML models with the logical deduction and transparency of PyReason, we aim to create a powerful system for automating complex processes. This integration finds utility across numerous domains, including manufacturing, healthcare, and business operations. 

---
# CARTS: Collaborative Agents for Recommendation Textual Summarization 

**Authors**: Jiao Chen, Kehui Yao, Reza Yousefi Maragheh, Kai Zhao, Jianpeng Xu, Jason Cho, Evren Korpeoglu, Sushant Kumar, Kannan Achan  

**Link**: [PDF](https://arxiv.org/pdf/2506.17765)  

**Abstract**: Current recommendation systems often require some form of textual data summarization, such as generating concise and coherent titles for product carousels or other grouped item displays. While large language models have shown promise in NLP domains for textual summarization, these approaches do not directly apply to recommendation systems, where explanations must be highly relevant to the core features of item sets, adhere to strict word limit constraints. In this paper, we propose CARTS (Collaborative Agents for Recommendation Textual Summarization), a multi-agent LLM framework designed for structured summarization in recommendation systems. CARTS decomposes the task into three stages-Generation Augmented Generation (GAG), refinement circle, and arbitration, where successive agent roles are responsible for extracting salient item features, iteratively refining candidate titles based on relevance and length feedback, and selecting the final title through a collaborative arbitration process. Experiments on large-scale e-commerce data and live A/B testing show that CARTS significantly outperforms single-pass and chain-of-thought LLM baselines, delivering higher title relevance and improved user engagement metrics. 

---
# Residual Connection-Enhanced ConvLSTM for Lithium Dendrite Growth Prediction 

**Authors**: Hosung Lee, Byeongoh Hwang, Dasan Kim, Myungjoo Kang  

**Link**: [PDF](https://arxiv.org/pdf/2506.17756)  

**Abstract**: The growth of lithium dendrites significantly impacts the performance and safety of rechargeable batteries, leading to short circuits and capacity degradation. This study proposes a Residual Connection-Enhanced ConvLSTM model to predict dendrite growth patterns with improved accuracy and computational efficiency. By integrating residual connections into ConvLSTM, the model mitigates the vanishing gradient problem, enhances feature retention across layers, and effectively captures both localized dendrite growth dynamics and macroscopic battery behavior. The dataset was generated using a phase-field model, simulating dendrite evolution under varying conditions. Experimental results show that the proposed model achieves up to 7% higher accuracy and significantly reduces mean squared error (MSE) compared to conventional ConvLSTM across different voltage conditions (0.1V, 0.3V, 0.5V). This highlights the effectiveness of residual connections in deep spatiotemporal networks for electrochemical system modeling. The proposed approach offers a robust tool for battery diagnostics, potentially aiding in real-time monitoring and optimization of lithium battery performance. Future research can extend this framework to other battery chemistries and integrate it with real-world experimental data for further validation 

---
# HIDE and Seek: Detecting Hallucinations in Language Models via Decoupled Representations 

**Authors**: Anwoy Chatterjee, Yash Goel, Tanmoy Chakraborty  

**Link**: [PDF](https://arxiv.org/pdf/2506.17748)  

**Abstract**: Contemporary Language Models (LMs), while impressively fluent, often generate content that is factually incorrect or unfaithful to the input context - a critical issue commonly referred to as 'hallucination'. This tendency of LMs to generate hallucinated content undermines their reliability, especially because these fabrications are often highly convincing and therefore difficult to detect. While several existing methods attempt to detect hallucinations, most rely on analyzing multiple generations per input, leading to increased computational cost and latency. To address this, we propose a single-pass, training-free approach for effective Hallucination detectIon via Decoupled rEpresentations (HIDE). Our approach leverages the hypothesis that hallucinations result from a statistical decoupling between an LM's internal representations of input context and its generated output. We quantify this decoupling using the Hilbert-Schmidt Independence Criterion (HSIC) applied to hidden-state representations extracted while generating the output sequence. We conduct extensive experiments on four diverse question answering datasets, evaluating both faithfulness and factuality hallucinations across six open-source LMs of varying scales and properties. Our results demonstrate that HIDE outperforms other single-pass methods in almost all settings, achieving an average relative improvement of ~29% in AUC-ROC over the best-performing single-pass strategy across various models and datasets. Additionally, HIDE shows competitive and often superior performance with multi-pass state-of-the-art methods, obtaining an average relative improvement of ~3% in AUC-ROC while consuming ~51% less computation time. Our findings highlight the effectiveness of exploiting internal representation decoupling in LMs for efficient and practical hallucination detection. 

---
# KAG-Thinker: Teaching Large Language Models to Think with Human-like Reasoning Process 

**Authors**: Dalong Zhang, Jun Xu, Jun Zhou, Lei Liang, Lin Yuan, Ling Zhong, Mengshu Sun, Peilong Zhao, QiWei Wang, Xiaorui Wang, Xinkai Du, YangYang Hou, Yu Ao, ZhaoYang Wang, Zhengke Gui, ZhiYing Yi, Zhongpu Bo  

**Link**: [PDF](https://arxiv.org/pdf/2506.17728)  

**Abstract**: In this paper, we introduce KAG-Thinker, a novel human-like reasoning framework built upon a parameter-light large language model (LLM). Our approach enhances the logical coherence and contextual consistency of the thinking process in question-answering (Q\&A) tasks on domain-specific knowledge bases (KBs) within LLMs. This framework simulates human cognitive mechanisms for handling complex problems by establishing a structured thinking process. Continuing the \textbf{Logical Form} guided retrieval and reasoning technology route of KAG v0.7, firstly, it decomposes complex questions into independently solvable sub-problems(also referred to as logical forms) through \textbf{breadth decomposition}, each represented in two equivalent forms-natural language and logical function-and further classified as either Knowledge Retrieval or Reasoning Analysis tasks, with dependencies and variables passing explicitly modeled via logical function interfaces. In the solving process, the Retrieval function is used to perform knowledge retrieval tasks, while the Math and Deduce functions are used to perform reasoning analysis tasks. Secondly, it is worth noting that, in the Knowledge Retrieval sub-problem tasks, LLMs and external knowledge sources are regarded as equivalent KBs. We use the \textbf{knowledge boundary} model to determine the optimal source using self-regulatory mechanisms such as confidence calibration and reflective reasoning, and use the \textbf{depth solving} model to enhance the comprehensiveness of knowledge acquisition. Finally, instead of utilizing reinforcement learning, we employ supervised fine-tuning with multi-turn dialogues to align the model with our structured inference paradigm, thereby avoiding excessive reflection. This is supported by a data evaluation framework and iterative corpus synthesis, which facilitate the generation of detailed reasoning trajectories... 

---
# Resolving the Ti-V Phase Diagram Discrepancy with First-Principles Calculations and Bayesian Learning 

**Authors**: Timofei Miryashkin, Olga Klimanova, Alexander Shapeev  

**Link**: [PDF](https://arxiv.org/pdf/2506.17719)  

**Abstract**: Conflicting experiments disagree on whether the titanium-vanadium (Ti-V) binary alloy exhibits a body-centred cubic (BCC) miscibility gap or remains completely soluble. A leading hypothesis attributes the miscibility gap to oxygen contamination during alloy preparation. To resolve this controversy, we use an ab initio + machine-learning workflow that couples an actively-trained Moment Tensor Potential to Bayesian thermodynamic inference. Using this workflow, we obtain Ti-V binary system across the entire composition range, together with confidence intervals in the thermodynamic limit. The resulting diagram reproduces all experimental features, demonstrating the robustness of our approach, and clearly favors the variant with a BCC miscibility gap terminating at T = 980 K and c = 0.67. Because oxygen was excluded from simulations, the gap cannot be attributed to impurity effects, contradicting recent CALPHAD reassessments. 

---
# Aged to Perfection: Machine-Learning Maps of Age in Conversational English 

**Authors**: MingZe Tang  

**Link**: [PDF](https://arxiv.org/pdf/2506.17708)  

**Abstract**: The study uses the British National Corpus 2014, a large sample of contemporary spoken British English, to investigate language patterns across different age groups. Our research attempts to explore how language patterns vary between different age groups, exploring the connection between speaker demographics and linguistic factors such as utterance duration, lexical diversity, and word choice. By merging computational language analysis and machine learning methodologies, we attempt to uncover distinctive linguistic markers characteristic of multiple generations and create prediction models that can consistently estimate the speaker's age group from various aspects. This work contributes to our knowledge of sociolinguistic diversity throughout the life of modern British speech. 

---
# Programmable-Room: Interactive Textured 3D Room Meshes Generation Empowered by Large Language Models 

**Authors**: Jihyun Kim, Junho Park, Kyeongbo Kong, Suk-Ju Kang  

**Link**: [PDF](https://arxiv.org/pdf/2506.17707)  

**Abstract**: We present Programmable-Room, a framework which interactively generates and edits a 3D room mesh, given natural language instructions. For precise control of a room's each attribute, we decompose the challenging task into simpler steps such as creating plausible 3D coordinates for room meshes, generating panorama images for the texture, constructing 3D meshes by integrating the coordinates and panorama texture images, and arranging furniture. To support the various decomposed tasks with a unified framework, we incorporate visual programming (VP). VP is a method that utilizes a large language model (LLM) to write a Python-like program which is an ordered list of necessary modules for the various tasks given in natural language. We develop most of the modules. Especially, for the texture generating module, we utilize a pretrained large-scale diffusion model to generate panorama images conditioned on text and visual prompts (i.e., layout, depth, and semantic map) simultaneously. Specifically, we enhance the panorama image generation quality by optimizing the training objective with a 1D representation of a panorama scene obtained from bidirectional LSTM. We demonstrate Programmable-Room's flexibility in generating and editing 3D room meshes, and prove our framework's superiority to an existing model quantitatively and qualitatively. Project page is available in this https URL. 

---
# The Evolution of Natural Language Processing: How Prompt Optimization and Language Models are Shaping the Future 

**Authors**: Summra Saleem, Muhammad Nabeel Asim, Shaista Zulfiqar, Andreas Dengel  

**Link**: [PDF](https://arxiv.org/pdf/2506.17700)  

**Abstract**: Large Language Models (LLMs) have revolutionized the field of Natural Language Processing (NLP) by automating traditional labor-intensive tasks and consequently accelerated the development of computer-aided applications. As researchers continue to advance this field with the introduction of novel language models and more efficient training/finetuning methodologies, the idea of prompt engineering and subsequent optimization strategies with LLMs has emerged as a particularly impactful trend to yield a substantial performance boost across diverse NLP tasks. To best of our knowledge numerous review articles have explored prompt engineering, however, a critical gap exists in comprehensive analyses of prompt optimization strategies. To bridge this gap this paper provides unique and comprehensive insights about the potential of diverse prompt optimization strategies. It analyzes their underlying working paradigms and based on these principles, categorizes them into 11 distinct classes. Moreover, the paper provides details about various NLP tasks where these prompt optimization strategies have been employed, along with details of different LLMs and benchmark datasets used for evaluation. This comprehensive compilation lays a robust foundation for future comparative studies and enables rigorous assessment of prompt optimization and LLM-based predictive pipelines under consistent experimental settings: a critical need in the current landscape. Ultimately, this research will centralize diverse strategic knowledge to facilitate the adaptation of existing prompt optimization strategies for development of innovative predictors across unexplored tasks. 

---
# Reinforcing User Interest Evolution in Multi-Scenario Learning for recommender systems 

**Authors**: Zhijian Feng, Wenhao Zheng, Xuanji Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2506.17682)  

**Abstract**: In real-world recommendation systems, users would engage in variety scenarios, such as homepages, search pages, and related recommendation pages. Each of these scenarios would reflect different aspects users focus on. However, the user interests may be inconsistent in different scenarios, due to differences in decision-making processes and preference expression. This variability complicates unified modeling, making multi-scenario learning a significant challenge. To address this, we propose a novel reinforcement learning approach that models user preferences across scenarios by modeling user interest evolution across multiple scenarios. Our method employs Double Q-learning to enhance next-item prediction accuracy and optimizes contrastive learning loss using Q-value to make model performance better. Experimental results demonstrate that our approach surpasses state-of-the-art methods in multi-scenario recommendation tasks. Our work offers a fresh perspective on multi-scenario modeling and highlights promising directions for future research. 

---
# Enhancing Stress-Strain Predictions with Seq2Seq and Cross-Attention based on Small Punch Test 

**Authors**: Zhengni Yang, Rui Yang, Weijian Han, Qixin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.17680)  

**Abstract**: This paper introduces a novel deep-learning approach to predict true stress-strain curves of high-strength steels from small punch test (SPT) load-displacement data. The proposed approach uses Gramian Angular Field (GAF) to transform load-displacement sequences into images, capturing spatial-temporal features and employs a Sequence-to-Sequence (Seq2Seq) model with an LSTM-based encoder-decoder architecture, enhanced by multi-head cross-attention to improved accuracy. Experimental results demonstrate that the proposed approach achieves superior prediction accuracy, with minimum and maximum mean absolute errors of 0.15 MPa and 5.58 MPa, respectively. The proposed method offers a promising alternative to traditional experimental techniques in materials science, enhancing the accuracy and efficiency of true stress-strain relationship predictions. 

---
# FaithfulSAE: Towards Capturing Faithful Features with Sparse Autoencoders without External Dataset Dependencies 

**Authors**: Seonglae Cho, Harryn Oh, Donghyun Lee, Luis Eduardo Rodrigues Vieira, Andrew Bermingham, Ziad El Sayed  

**Link**: [PDF](https://arxiv.org/pdf/2506.17673)  

**Abstract**: Sparse Autoencoders (SAEs) have emerged as a promising solution for decomposing large language model representations into interpretable features. However, Paulo and Belrose (2025) have highlighted instability across different initialization seeds, and Heap et al. (2025) have pointed out that SAEs may not capture model-internal features. These problems likely stem from training SAEs on external datasets - either collected from the Web or generated by another model - which may contain out-of-distribution (OOD) data beyond the model's generalisation capabilities. This can result in hallucinated SAE features, which we term "Fake Features", that misrepresent the model's internal activations. To address these issues, we propose FaithfulSAE, a method that trains SAEs on the model's own synthetic dataset. Using FaithfulSAEs, we demonstrate that training SAEs on less-OOD instruction datasets results in SAEs being more stable across seeds. Notably, FaithfulSAEs outperform SAEs trained on web-based datasets in the SAE probing task and exhibit a lower Fake Feature Ratio in 5 out of 7 models. Overall, our approach eliminates the dependency on external datasets, advancing interpretability by better capturing model-internal features while highlighting the often neglected importance of SAE training datasets. 

---
# TPTT: Transforming Pretrained Transformer into Titans 

**Authors**: Fabien Furfaro  

**Link**: [PDF](https://arxiv.org/pdf/2506.17671)  

**Abstract**: Recent advances in large language models (LLMs) have led to remarkable progress in natural language processing, but their computational and memory demands remain a significant challenge, particularly for long-context inference. We introduce TPTT (Transforming Pretrained Transformer into Titans), a novel framework for enhancing pretrained Transformer models with efficient linearized attention mechanisms and advanced memory management. TPTT employs techniques such as Memory as Gate (MaG) and mixed linearized attention (LiZA). It is fully compatible with the Hugging Face Transformers library, enabling seamless adaptation of any causal LLM through parameter-efficient fine-tuning (LoRA) without full retraining. We show the effectiveness of TPTT on the MMLU benchmark with models of approximately 1 billion parameters, observing substantial improvements in both efficiency and accuracy. For instance, Titans-Llama-3.2-1B achieves a 20% increase in Exact Match (EM) over its baseline. Statistical analyses and comparisons with recent state-of-the-art methods confirm the practical scalability and robustness of TPTT. Code is available at this https URL . Python package at this https URL . 

---
# RLRC: Reinforcement Learning-based Recovery for Compressed Vision-Language-Action Models 

**Authors**: Yuxuan Chen, Xiao Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.17639)  

**Abstract**: Vision-Language-Action models (VLA) have demonstrated remarkable capabilities and promising potential in solving complex robotic manipulation tasks. However, their substantial parameter sizes and high inference latency pose significant challenges for real-world deployment, particularly on resource-constrained robotic platforms. To address this issue, we begin by conducting an extensive empirical study to explore the effectiveness of model compression techniques when applied to VLAs. Building on the insights gained from these preliminary experiments, we propose RLRC, a three-stage recovery method for compressed VLAs, including structured pruning, performance recovery based on SFT and RL, and further quantization. RLRC achieves up to an 8x reduction in memory usage and a 2.3x improvement in inference throughput, while maintaining or even surpassing the original VLA's task success rate. Extensive experiments show that RLRC consistently outperforms existing compression baselines, demonstrating strong potential for on-device deployment of VLAs. Project website: this https URL 

---
# Adaptive Multi-prompt Contrastive Network for Few-shot Out-of-distribution Detection 

**Authors**: Xiang Fang, Arvind Easwaran, Blaise Genest  

**Link**: [PDF](https://arxiv.org/pdf/2506.17633)  

**Abstract**: Out-of-distribution (OOD) detection attempts to distinguish outlier samples to prevent models trained on the in-distribution (ID) dataset from producing unavailable outputs. Most OOD detection methods require many IID samples for training, which seriously limits their real-world applications. To this end, we target a challenging setting: few-shot OOD detection, where {Only a few {\em labeled ID} samples are available.} Therefore, few-shot OOD detection is much more challenging than the traditional OOD detection setting. Previous few-shot OOD detection works ignore the distinct diversity between different classes. In this paper, we propose a novel network: Adaptive Multi-prompt Contrastive Network (AMCN), which adapts the ID-OOD separation boundary by learning inter- and intra-class distribution. To compensate for the absence of OOD and scarcity of ID {\em image samples}, we leverage CLIP, connecting text with images, engineering learnable ID and OOD {\em textual prompts}. Specifically, we first generate adaptive prompts (learnable ID prompts, label-fixed OOD prompts and label-adaptive OOD prompts). Then, we generate an adaptive class boundary for each class by introducing a class-wise threshold. Finally, we propose a prompt-guided ID-OOD separation module to control the margin between ID and OOD prompts. Experimental results show that AMCN outperforms other state-of-the-art works. 

---
# LLM-Prompt: Integrated Heterogeneous Prompts for Unlocking LLMs in Time Series Forecasting 

**Authors**: Zesen Wang, Yonggang Li, Lijuan Lan  

**Link**: [PDF](https://arxiv.org/pdf/2506.17631)  

**Abstract**: Time series forecasting aims to model temporal dependencies among variables for future state inference, holding significant importance and widespread applications in real-world scenarios. Although deep learning-based methods have achieved remarkable progress, they still exhibit suboptimal performance in long-term forecasting and data-scarce scenarios. Recent research demonstrates that large language models (LLMs) achieve promising performance in time series forecasting. However, we find existing LLM-based methods still have shortcomings: (1) the absence of a unified paradigm for textual prompt formulation and (2) the neglect of modality discrepancies between textual prompts and time series. To address this, we propose LLM-Prompt, an LLM-based time series forecasting framework integrating multi-prompt information and cross-modal semantic alignment. Specifically, we first construct a unified textual prompt paradigm containing learnable soft prompts and textualized hard prompts. Second, to enhance LLMs' comprehensive understanding of the forecasting task, we design a semantic space embedding and cross-modal alignment module to achieve cross-modal fusion of temporal and textual information. Finally, the transformed time series from the LLMs are projected to obtain the forecasts. Comprehensive evaluations on 6 public datasets and 3 carbon emission datasets demonstrate that LLM-Prompt is a powerful framework for time series forecasting. 

---
# CLiViS: Unleashing Cognitive Map through Linguistic-Visual Synergy for Embodied Visual Reasoning 

**Authors**: Kailing Li, Qi'ao Xu, Tianwen Qian, Yuqian Fu, Yang Jiao, Xiaoling Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.17629)  

**Abstract**: Embodied Visual Reasoning (EVR) seeks to follow complex, free-form instructions based on egocentric video, enabling semantic understanding and spatiotemporal reasoning in dynamic environments. Despite its promising potential, EVR encounters significant challenges stemming from the diversity of complex instructions and the intricate spatiotemporal dynamics in long-term egocentric videos. Prior solutions either employ Large Language Models (LLMs) over static video captions, which often omit critical visual details, or rely on end-to-end Vision-Language Models (VLMs) that struggle with stepwise compositional reasoning. Consider the complementary strengths of LLMs in reasoning and VLMs in perception, we propose CLiViS. It is a novel training-free framework that leverages LLMs for high-level task planning and orchestrates VLM-driven open-world visual perception to iteratively update the scene context. Building on this synergy, the core of CLiViS is a dynamic Cognitive Map that evolves throughout the reasoning process. This map constructs a structured representation of the embodied scene, bridging low-level perception and high-level reasoning. Extensive experiments across multiple benchmarks demonstrate the effectiveness and generality of CLiViS, especially in handling long-term visual dependencies. Code is available at this https URL. 

---
# Exploiting Efficiency Vulnerabilities in Dynamic Deep Learning Systems 

**Authors**: Ravishka Rathnasuriya, Wei Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.17621)  

**Abstract**: The growing deployment of deep learning models in real-world environments has intensified the need for efficient inference under strict latency and resource constraints. To meet these demands, dynamic deep learning systems (DDLSs) have emerged, offering input-adaptive computation to optimize runtime efficiency. While these systems succeed in reducing cost, their dynamic nature introduces subtle and underexplored security risks. In particular, input-dependent execution pathways create opportunities for adversaries to degrade efficiency, resulting in excessive latency, energy usage, and potential denial-of-service in time-sensitive deployments. This work investigates the security implications of dynamic behaviors in DDLSs and reveals how current systems expose efficiency vulnerabilities exploitable by adversarial inputs. Through a survey of existing attack strategies, we identify gaps in the coverage of emerging model architectures and limitations in current defense mechanisms. Building on these insights, we propose to examine the feasibility of efficiency attacks on modern DDLSs and develop targeted defenses to preserve robustness under adversarial conditions. 

---
# Risk-Guided Diffusion: Toward Deploying Robot Foundation Models in Space, Where Failure Is Not An Option 

**Authors**: Rohan Thakker, Adarsh Patnaik, Vince Kurtz, Jonas Frey, Jonathan Becktor, Sangwoo Moon, Rob Royce, Marcel Kaufmann, Georgios Georgakis, Pascal Roth, Joel Burdick, Marco Hutter, Shehryar Khattak  

**Link**: [PDF](https://arxiv.org/pdf/2506.17601)  

**Abstract**: Safe, reliable navigation in extreme, unfamiliar terrain is required for future robotic space exploration missions. Recent generative-AI methods learn semantically aware navigation policies from large, cross-embodiment datasets, but offer limited safety guarantees. Inspired by human cognitive science, we propose a risk-guided diffusion framework that fuses a fast, learned "System-1" with a slow, physics-based "System-2", sharing computation at both training and inference to couple adaptability with formal safety. Hardware experiments conducted at the NASA JPL's Mars-analog facility, Mars Yard, show that our approach reduces failure rates by up to $4\times$ while matching the goal-reaching performance of learning-based robotic models by leveraging inference-time compute without any additional training. 

---
# DRAMA-X: A Fine-grained Intent Prediction and Risk Reasoning Benchmark For Driving 

**Authors**: Mihir Godbole, Xiangbo Gao, Zhengzhong Tu  

**Link**: [PDF](https://arxiv.org/pdf/2506.17590)  

**Abstract**: Understanding the short-term motion of vulnerable road users (VRUs) like pedestrians and cyclists is critical for safe autonomous driving, especially in urban scenarios with ambiguous or high-risk behaviors. While vision-language models (VLMs) have enabled open-vocabulary perception, their utility for fine-grained intent reasoning remains underexplored. Notably, no existing benchmark evaluates multi-class intent prediction in safety-critical situations, To address this gap, we introduce DRAMA-X, a fine-grained benchmark constructed from the DRAMA dataset via an automated annotation pipeline. DRAMA-X contains 5,686 accident-prone frames labeled with object bounding boxes, a nine-class directional intent taxonomy, binary risk scores, expert-generated action suggestions for the ego vehicle, and descriptive motion summaries. These annotations enable a structured evaluation of four interrelated tasks central to autonomous decision-making: object detection, intent prediction, risk assessment, and action suggestion. As a reference baseline, we propose SGG-Intent, a lightweight, training-free framework that mirrors the ego vehicle's reasoning pipeline. It sequentially generates a scene graph from visual input using VLM-backed detectors, infers intent, assesses risk, and recommends an action using a compositional reasoning stage powered by a large language model. We evaluate a range of recent VLMs, comparing performance across all four DRAMA-X tasks. Our experiments demonstrate that scene-graph-based reasoning enhances intent prediction and risk assessment, especially when contextual cues are explicitly modeled. 

---
# HalluRNN: Mitigating Hallucinations via Recurrent Cross-Layer Reasoning in Large Vision-Language Models 

**Authors**: Le Yu, Kaishen Wang, Jianlong Xiong, Yue Cao, Tao He  

**Link**: [PDF](https://arxiv.org/pdf/2506.17587)  

**Abstract**: Though Large Vision-Language Models (LVLMs) have achieved remarkable performance across various tasks, they are still prone to hallucinations-generating outputs that are textually plausible but visually ungrounded. While prior approaches generally address this issue through data-centric fine-tuning or innovative decoding strategies, these methods often require substantial resources or task-specific configurations. In this work, we introduce an architecture-level solution, HalluRNN, which enhances model stability through recurrent cross-layer reasoning. Specifically, we propose a novel Dual-Gated Depth Propagation Unit (DG-DPU) module, which is shared across layers and recurrently refines hidden states. This allows for the adaptive propagation of information throughout the model, enforces consistency across layers, and mitigates hallucinations caused by representational drift. By fine-tuning only the DG-DPU module, HalluRNN achieves strong and robust performance across multiple benchmarks. 

---
# Context-Aware Scientific Knowledge Extraction on Linked Open Data using Large Language Models 

**Authors**: Sajratul Y. Rubaiat, Hasan M. Jamil  

**Link**: [PDF](https://arxiv.org/pdf/2506.17580)  

**Abstract**: The exponential growth of scientific literature challenges researchers extracting and synthesizing knowledge. Traditional search engines return many sources without direct, detailed answers, while general-purpose LLMs may offer concise responses that lack depth or omit current information. LLMs with search capabilities are also limited by context window, yielding short, incomplete answers. This paper introduces WISE (Workflow for Intelligent Scientific Knowledge Extraction), a system addressing these limits by using a structured workflow to extract, refine, and rank query-specific knowledge. WISE uses an LLM-powered, tree-based architecture to refine data, focusing on query-aligned, context-aware, and non-redundant information. Dynamic scoring and ranking prioritize unique contributions from each source, and adaptive stopping criteria minimize processing overhead. WISE delivers detailed, organized answers by systematically exploring and synthesizing knowledge from diverse sources. Experiments on HBB gene-associated diseases demonstrate WISE reduces processed text by over 80% while achieving significantly higher recall over baselines like search engines and other LLM-based approaches. ROUGE and BLEU metrics reveal WISE's output is more unique than other systems, and a novel level-based metric shows it provides more in-depth information. We also explore how the WISE workflow can be adapted for diverse domains like drug discovery, material science, and social science, enabling efficient knowledge extraction and synthesis from unstructured scientific papers and web sources. 

---
# Optimizing Mastery Learning by Fast-Forwarding Over-Practice Steps 

**Authors**: Meng Xia, Robin Schmucker, Conrad Borchers, Vincent Aleven  

**Link**: [PDF](https://arxiv.org/pdf/2506.17577)  

**Abstract**: Mastery learning improves learning proficiency and efficiency. However, the overpractice of skills--students spending time on skills they have already mastered--remains a fundamental challenge for tutoring systems. Previous research has reduced overpractice through the development of better problem selection algorithms and the authoring of focused practice tasks. However, few efforts have concentrated on reducing overpractice through step-level adaptivity, which can avoid resource-intensive curriculum redesign. We propose and evaluate Fast-Forwarding as a technique that enhances existing problem selection algorithms. Based on simulation studies informed by learner models and problem-solving pathways derived from real student data, Fast-Forwarding can reduce overpractice by up to one-third, as it does not require students to complete problem-solving steps if all remaining pathways are fully mastered. Fast-Forwarding is a flexible method that enhances any problem selection algorithm, though its effectiveness is highest for algorithms that preferentially select difficult problems. Therefore, our findings suggest that while Fast-Forwarding may improve student practice efficiency, the size of its practical impact may also depend on students' ability to stay motivated and engaged at higher levels of difficulty. 

---
# Accelerating Residual Reinforcement Learning with Uncertainty Estimation 

**Authors**: Lakshita Dodeja, Karl Schmeckpeper, Shivam Vats, Thomas Weng, Mingxi Jia, George Konidaris, Stefanie Tellex  

**Link**: [PDF](https://arxiv.org/pdf/2506.17564)  

**Abstract**: Residual Reinforcement Learning (RL) is a popular approach for adapting pretrained policies by learning a lightweight residual policy that provides corrective actions. While Residual RL is more sample-efficient than finetuning the entire base policy, existing methods struggle with sparse rewards and are designed for deterministic base policies. We propose two improvements to Residual RL that further enhance its sample efficiency and make it suitable for stochastic base policies. First, we leverage uncertainty estimates of the base policy to focus exploration on regions in which the base policy is not confident. Second, we propose a simple modification to off-policy residual learning that allows it to observe base actions and better handle stochastic base policies. We evaluate our method with both Gaussian-based and Diffusion-based stochastic base policies on tasks from Robosuite and D4RL, and compare against state-of-the-art finetuning methods, demo-augmented RL methods, and other residual RL methods. Our algorithm significantly outperforms existing baselines in a variety of simulation benchmark environments. We also deploy our learned polices in the real world to demonstrate their robustness with zero-shot sim-to-real transfer. 

---
# VLA-OS: Structuring and Dissecting Planning Representations and Paradigms in Vision-Language-Action Models 

**Authors**: Chongkai Gao, Zixuan Liu, Zhenghao Chi, Junshan Huang, Xin Fei, Yiwen Hou, Yuxuan Zhang, Yudi Lin, Zhirui Fang, Zeyu Jiang, Lin Shao  

**Link**: [PDF](https://arxiv.org/pdf/2506.17561)  

**Abstract**: Recent studies on Vision-Language-Action (VLA) models have shifted from the end-to-end action-generation paradigm toward a pipeline involving task planning followed by action generation, demonstrating improved performance on various complex, long-horizon manipulation tasks. However, existing approaches vary significantly in terms of network architectures, planning paradigms, representations, and training data sources, making it challenging for researchers to identify the precise sources of performance gains and components to be further improved. To systematically investigate the impacts of different planning paradigms and representations isolating from network architectures and training data, in this paper, we introduce VLA-OS, a unified VLA architecture series capable of various task planning paradigms, and design a comprehensive suite of controlled experiments across diverse object categories (rigid and deformable), visual modalities (2D and 3D), environments (simulation and real-world), and end-effectors (grippers and dexterous hands). Our results demonstrate that: 1) visually grounded planning representations are generally better than language planning representations; 2) the Hierarchical-VLA paradigm generally achieves superior or comparable performance than other paradigms on task performance, pretraining, generalization ability, scalability, and continual learning ability, albeit at the cost of slower training and inference speeds. 

---
# Towards Zero-Shot Coordination between Teams of Agents: The N-XPlay Framework 

**Authors**: Ava Abderezaei, Chi-Hui Lin, Joseph Miceli, Naren Sivagnanadasan, Stéphane Aroca-Ouellette, Jake Brawer, Alessandro Roncone  

**Link**: [PDF](https://arxiv.org/pdf/2506.17560)  

**Abstract**: Zero-shot coordination (ZSC) -- the ability to collaborate with unfamiliar partners -- is essential to making autonomous agents effective teammates. Existing ZSC methods evaluate coordination capabilities between two agents who have not previously interacted. However, these scenarios do not reflect the complexity of real-world multi-agent systems, where coordination often involves a hierarchy of sub-groups and interactions between teams of agents, known as Multi-Team Systems (MTS). To address this gap, we first introduce N-player Overcooked, an N-agent extension of the popular two-agent ZSC benchmark, enabling evaluation of ZSC in N-agent scenarios. We then propose N-XPlay for ZSC in N-agent, multi-team settings. Comparison against Self-Play across two-, three- and five-player Overcooked scenarios, where agents are split between an ``ego-team'' and a group of unseen collaborators shows that agents trained with N-XPlay are better able to simultaneously balance ``intra-team'' and ``inter-team'' coordination than agents trained with SP. 

---
# SynDaCaTE: A Synthetic Dataset For Evaluating Part-Whole Hierarchical Inference 

**Authors**: Jake Levi, Mark van der Wilk  

**Link**: [PDF](https://arxiv.org/pdf/2506.17558)  

**Abstract**: Learning to infer object representations, and in particular part-whole hierarchies, has been the focus of extensive research in computer vision, in pursuit of improving data efficiency, systematic generalisation, and robustness. Models which are \emph{designed} to infer part-whole hierarchies, often referred to as capsule networks, are typically trained end-to-end on supervised tasks such as object classification, in which case it is difficult to evaluate whether such a model \emph{actually} learns to infer part-whole hierarchies, as claimed. To address this difficulty, we present a SYNthetic DAtaset for CApsule Testing and Evaluation, abbreviated as SynDaCaTE, and establish its utility by (1) demonstrating the precise bottleneck in a prominent existing capsule model, and (2) demonstrating that permutation-equivariant self-attention is highly effective for parts-to-wholes inference, which motivates future directions for designing effective inductive biases for computer vision. 

---
# Research on Model Parallelism and Data Parallelism Optimization Methods in Large Language Model-Based Recommendation Systems 

**Authors**: Haowei Yang, Yu Tian, Zhongheng Yang, Zhao Wang, Chengrui Zhou, Dannier Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.17551)  

**Abstract**: With the rapid adoption of large language models (LLMs) in recommendation systems, the computational and communication bottlenecks caused by their massive parameter sizes and large data volumes have become increasingly prominent. This paper systematically investigates two classes of optimization methods-model parallelism and data parallelism-for distributed training of LLMs in recommendation scenarios. For model parallelism, we implement both tensor parallelism and pipeline parallelism, and introduce an adaptive load-balancing mechanism to reduce cross-device communication overhead. For data parallelism, we compare synchronous and asynchronous modes, combining gradient compression and sparsification techniques with an efficient aggregation communication framework to significantly improve bandwidth utilization. Experiments conducted on a real-world recommendation dataset in a simulated service environment demonstrate that our proposed hybrid parallelism scheme increases training throughput by over 30% and improves resource utilization by approximately 20% compared to traditional single-mode parallelism, while maintaining strong scalability and robustness. Finally, we discuss trade-offs among different parallel strategies in online deployment and outline future directions involving heterogeneous hardware integration and automated scheduling technologies. 

---
# ConsumerBench: Benchmarking Generative AI Applications on End-User Devices 

**Authors**: Yile Gu, Rohan Kadekodi, Hoang Nguyen, Keisuke Kamahori, Yiyu Liu, Baris Kasikci  

**Link**: [PDF](https://arxiv.org/pdf/2506.17538)  

**Abstract**: The recent shift in Generative AI (GenAI) applications from cloud-only environments to end-user devices introduces new challenges in resource management, system efficiency, and user experience. This paper presents ConsumerBench, a comprehensive benchmarking framework designed to evaluate the system efficiency and response time of GenAI models running on end-user devices. Unlike existing benchmarks that assume exclusive model access on dedicated GPUs, ConsumerBench simulates realistic multi-application scenarios executing concurrently on constrained hardware. Furthermore, ConsumerBench supports customizable workflows that simulate complex tasks requiring coordination among multiple applications. ConsumerBench captures both application-level metrics, including latency and Service Level Objective (SLO) attainment, and system-level metrics like CPU/GPU utilization and memory bandwidth. Through extensive experiments, ConsumerBench reveals inefficiencies in resource sharing, unfair scheduling under greedy allocation, and performance pitfalls of static model server configurations. The paper also provides practical insights for model developers and system designers, highlighting the benefits of custom kernels tailored to consumer-grade GPU architectures and the value of implementing SLO-aware scheduling strategies. 

---
# Exploring Strategies for Personalized Radiation Therapy Part I Unlocking Response-Related Tumor Subregions with Class Activation Mapping 

**Authors**: Hao Peng, Steve Jiang, Robert Timmerman  

**Link**: [PDF](https://arxiv.org/pdf/2506.17536)  

**Abstract**: Personalized precision radiation therapy requires more than simple classification, it demands the identification of prognostic, spatially informative features and the ability to adapt treatment based on individual response. This study compares three approaches for predicting treatment response: standard radiomics, gradient based features, and convolutional neural networks enhanced with Class Activation Mapping. We analyzed 69 brain metastases from 39 patients treated with Gamma Knife radiosurgery. An integrated autoencoder classifier model was used to predict whether tumor volume would shrink by more than 20 percent at a three months follow up, framed as a binary classification task. The results highlight their strength in hierarchical feature extraction and the classifiers discriminative capacity. Among the models, pixel wise CAM provides the most detailed spatial insight, identifying lesion specific regions rather than relying on fixed patterns, demonstrating strong generalization. In non responding lesions, the activated regions may indicate areas of radio resistance. Pixel wise CAM outperformed both radiomics and gradient based methods in classification accuracy. Moreover, its fine grained spatial features allow for alignment with cellular level data, supporting biological validation and deeper understanding of heterogeneous treatment responses. Although further validation is necessary, these findings underscore the promise in guiding personalized and adaptive radiotherapy strategies for both photon and particle therapies. 

---
# Data Quality Issues in Multilingual Speech Datasets: The Need for Sociolinguistic Awareness and Proactive Language Planning 

**Authors**: Mingfei Lau, Qian Chen, Yeming Fang, Tingting Xu, Tongzhou Chen, Pavel Golik  

**Link**: [PDF](https://arxiv.org/pdf/2506.17525)  

**Abstract**: Our quality audit for three widely used public multilingual speech datasets - Mozilla Common Voice 17.0, FLEURS, and VoxPopuli - shows that in some languages, these datasets suffer from significant quality issues. We believe addressing these issues will make these datasets more useful as training and evaluation sets, and improve downstream models. We divide these quality issues into two categories: micro-level and macro-level. We find that macro-level issues are more prevalent in less institutionalized, often under-resourced languages. We provide a case analysis of Taiwanese Southern Min (nan_tw) that highlights the need for proactive language planning (e.g. orthography prescriptions, dialect boundary definition) and enhanced data quality control in the process of Automatic Speech Recognition (ASR) dataset creation. We conclude by proposing guidelines and recommendations to mitigate these issues in future dataset development, emphasizing the importance of sociolinguistic awareness in creating robust and reliable speech data resources. 

---
# A Survey of State Representation Learning for Deep Reinforcement Learning 

**Authors**: Ayoub Echchahed, Pablo Samuel Castro  

**Link**: [PDF](https://arxiv.org/pdf/2506.17518)  

**Abstract**: Representation learning methods are an important tool for addressing the challenges posed by complex observations spaces in sequential decision making problems. Recently, many methods have used a wide variety of types of approaches for learning meaningful state representations in reinforcement learning, allowing better sample efficiency, generalization, and performance. This survey aims to provide a broad categorization of these methods within a model-free online setting, exploring how they tackle the learning of state representations differently. We categorize the methods into six main classes, detailing their mechanisms, benefits, and limitations. Through this taxonomy, our aim is to enhance the understanding of this field and provide a guide for new researchers. We also discuss techniques for assessing the quality of representations, and detail relevant future directions. 

---
# Mapping the Evolution of Research Contributions using KnoVo 

**Authors**: Sajratul Y. Rubaiat, Syed N. Sakib, Hasan M. Jamil  

**Link**: [PDF](https://arxiv.org/pdf/2506.17508)  

**Abstract**: This paper presents KnoVo (Knowledge Evolution), an intelligent framework designed for quantifying and analyzing the evolution of research novelty in the scientific literature. Moving beyond traditional citation analysis, which primarily measures impact, KnoVo determines a paper's novelty relative to both prior and subsequent work within its multilayered citation network. Given a target paper's abstract, KnoVo utilizes Large Language Models (LLMs) to dynamically extract dimensions of comparison (e.g., methodology, application, dataset). The target paper is then compared to related publications along these same extracted dimensions. This comparative analysis, inspired by tournament selection, yields quantitative novelty scores reflecting the relative improvement, equivalence, or inferiority of the target paper in specific aspects. By aggregating these scores and visualizing their progression, for instance, through dynamic evolution graphs and comparative radar charts, KnoVo facilitates researchers not only to assess originality and identify similar work, but also to track knowledge evolution along specific research dimensions, uncover research gaps, and explore cross-disciplinary connections. We demonstrate these capabilities through a detailed analysis of 20 diverse papers from multiple scientific fields and report on the performance of various open-source LLMs within the KnoVo framework. 

---
# From Generality to Mastery: Composer-Style Symbolic Music Generation via Large-Scale Pre-training 

**Authors**: Mingyang Yao, Ke Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.17497)  

**Abstract**: Despite progress in controllable symbolic music generation, data scarcity remains a challenge for certain control modalities. Composer-style music generation is a prime example, as only a few pieces per composer are available, limiting the modeling of both styles and fundamental music elements (e.g., melody, chord, rhythm). In this paper, we investigate how general music knowledge learned from a broad corpus can enhance the mastery of specific composer styles, with a focus on piano piece generation. Our approach follows a two-stage training paradigm. First, we pre-train a REMI-based music generation model on a large corpus of pop, folk, and classical music. Then, we fine-tune it on a small, human-verified dataset from four renowned composers, namely Bach, Mozart, Beethoven, and Chopin, using a lightweight adapter module to condition the model on style indicators. To evaluate the effectiveness of our approach, we conduct both objective and subjective evaluations on style accuracy and musicality. Experimental results demonstrate that our method outperforms ablations and baselines, achieving more precise composer-style modeling and better musical aesthetics. Additionally, we provide observations on how the model builds music concepts from the generality pre-training and refines its stylistic understanding through the mastery fine-tuning. 

---
# Exploring Strategies for Personalized Radiation Therapy Part II Predicting Tumor Drift Patterns with Diffusion Models 

**Authors**: Hao Peng, Steve Jiang, Robert Timmerman  

**Link**: [PDF](https://arxiv.org/pdf/2506.17491)  

**Abstract**: Radiation therapy outcomes are decided by two key parameters, dose and timing, whose best values vary substantially across patients. This variability is especially critical in the treatment of brain cancer, where fractionated or staged stereotactic radiosurgery improves safety compared to single fraction approaches, but complicates the ability to predict treatment response. To address this challenge, we employ Personalized Ultra-fractionated Stereotactic Adaptive Radiotherapy (PULSAR), a strategy that dynamically adjusts treatment based on how each tumor evolves over time. However, the success of PULSAR and other adaptive approaches depends on predictive tools that can guide early treatment decisions and avoid both overtreatment and undertreatment. However, current radiomics and dosiomics models offer limited insight into the evolving spatial and temporal patterns of tumor response. To overcome these limitations, we propose a novel framework using Denoising Diffusion Implicit Models (DDIM), which learns data-driven mappings from pre to post treatment imaging. In this study, we developed single step and iterative denoising strategies and compared their performance. The results show that diffusion models can effectively simulate patient specific tumor evolution and localize regions associated with treatment response. The proposed strategy provides a promising foundation for modeling heterogeneous treatment response and enabling early, adaptive interventions, paving the way toward more personalized and biologically informed radiotherapy. 

---
# Distilling On-device Language Models for Robot Planning with Minimal Human Intervention 

**Authors**: Zachary Ravichandran, Ignacio Hounie, Fernando Cladera, Alejandro Ribeiro, George J. Pappas, Vijay Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2506.17486)  

**Abstract**: Large language models (LLMs) provide robots with powerful contextual reasoning abilities and a natural human interface. Yet, current LLM-enabled robots typically depend on cloud-hosted models, limiting their usability in environments with unreliable communication infrastructure, such as outdoor or industrial settings. We present PRISM, a framework for distilling small language model (SLM)-enabled robot planners that run on-device with minimal human supervision. Starting from an existing LLM-enabled planner, PRISM automatically synthesizes diverse tasks and environments, elicits plans from the LLM, and uses this synthetic dataset to distill a compact SLM as a drop-in replacement of the source model. We apply PRISM to three LLM-enabled planners for mapping and exploration, manipulation, and household assistance, and we demonstrate that PRISM improves the performance of Llama-3.2-3B from 10-20% of GPT-4o's performance to over 93% - using only synthetic data. We further demonstrate that the distilled planners generalize across heterogeneous robotic platforms (ground and aerial) and diverse environments (indoor and outdoor). We release all software, trained models, and datasets at this https URL. 

---
# Computational Approaches to Understanding Large Language Model Impact on Writing and Information Ecosystems 

**Authors**: Weixin Liang  

**Link**: [PDF](https://arxiv.org/pdf/2506.17467)  

**Abstract**: Large language models (LLMs) have shown significant potential to change how we write, communicate, and create, leading to rapid adoption across society. This dissertation examines how individuals and institutions are adapting to and engaging with this emerging technology through three research directions. First, I demonstrate how the institutional adoption of AI detectors introduces systematic biases, particularly disadvantaging writers of non-dominant language varieties, highlighting critical equity concerns in AI governance. Second, I present novel population-level algorithmic approaches that measure the increasing adoption of LLMs across writing domains, revealing consistent patterns of AI-assisted content in academic peer reviews, scientific publications, consumer complaints, corporate communications, job postings, and international organization press releases. Finally, I investigate LLMs' capability to provide feedback on research manuscripts through a large-scale empirical analysis, offering insights into their potential to support researchers who face barriers in accessing timely manuscript feedback, particularly early-career researchers and those from under-resourced settings. 

---
# FedNAMs: Performing Interpretability Analysis in Federated Learning Context 

**Authors**: Amitash Nanda, Sree Bhargavi Balija, Debashis Sahoo  

**Link**: [PDF](https://arxiv.org/pdf/2506.17466)  

**Abstract**: Federated learning continues to evolve but faces challenges in interpretability and explainability. To address these challenges, we introduce a novel approach that employs Neural Additive Models (NAMs) within a federated learning framework. This new Federated Neural Additive Models (FedNAMs) approach merges the advantages of NAMs, where individual networks concentrate on specific input features, with the decentralized approach of federated learning, ultimately producing interpretable analysis results. This integration enhances privacy by training on local data across multiple devices, thereby minimizing the risks associated with data centralization and improving model robustness and generalizability. FedNAMs maintain detailed, feature-specific learning, making them especially valuable in sectors such as finance and healthcare. They facilitate the training of client-specific models to integrate local updates, preserve privacy, and mitigate concerns related to centralization. Our studies on various text and image classification tasks, using datasets such as OpenFetch ML Wine, UCI Heart Disease, and Iris, show that FedNAMs deliver strong interpretability with minimal accuracy loss compared to traditional Federated Deep Neural Networks (DNNs). The research involves notable findings, including the identification of critical predictive features at both client and global levels. Volatile acidity, sulfates, and chlorides for wine quality. Chest pain type, maximum heart rate, and number of vessels for heart disease. Petal length and width for iris classification. This approach strengthens privacy and model efficiency and improves interpretability and robustness across diverse datasets. Finally, FedNAMs generate insights on causes of highly and low interpretable features. 

---
# General-Purpose Robotic Navigation via LVLM-Orchestrated Perception, Reasoning, and Acting 

**Authors**: Bernard Lange, Anil Yildiz, Mansur Arief, Shehryar Khattak, Mykel Kochenderfer, Georgios Georgakis  

**Link**: [PDF](https://arxiv.org/pdf/2506.17462)  

**Abstract**: Developing general-purpose navigation policies for unknown environments remains a core challenge in robotics. Most existing systems rely on task-specific neural networks and fixed data flows, limiting generalizability. Large Vision-Language Models (LVLMs) offer a promising alternative by embedding human-like knowledge suitable for reasoning and planning. Yet, prior LVLM-robot integrations typically depend on pre-mapped spaces, hard-coded representations, and myopic exploration. We introduce the Agentic Robotic Navigation Architecture (ARNA), a general-purpose navigation framework that equips an LVLM-based agent with a library of perception, reasoning, and navigation tools available within modern robotic stacks. At runtime, the agent autonomously defines and executes task-specific workflows that iteratively query the robotic modules, reason over multimodal inputs, and select appropriate navigation actions. This approach enables robust navigation and reasoning in previously unmapped environments, providing a new perspective on robotic stack design. Evaluated in Habitat Lab on the HM-EQA benchmark, ARNA achieves state-of-the-art performance, demonstrating effective exploration, navigation, and embodied question answering without relying on handcrafted plans, fixed input representations, or pre-existing maps. 

---
# Trans${^2}$-CBCT: A Dual-Transformer Framework for Sparse-View CBCT Reconstruction 

**Authors**: Minmin Yang, Huantao Ren, Senem Velipasalar  

**Link**: [PDF](https://arxiv.org/pdf/2506.17425)  

**Abstract**: Cone-beam computed tomography (CBCT) using only a few X-ray projection views enables faster scans with lower radiation dose, but the resulting severe under-sampling causes strong artifacts and poor spatial coverage. We address these challenges in a unified framework. First, we replace conventional UNet/ResNet encoders with TransUNet, a hybrid CNN-Transformer model. Convolutional layers capture local details, while self-attention layers enhance global context. We adapt TransUNet to CBCT by combining multi-scale features, querying view-specific features per 3D point, and adding a lightweight attenuation-prediction head. This yields Trans-CBCT, which surpasses prior baselines by 1.17 dB PSNR and 0.0163 SSIM on the LUNA16 dataset with six views. Second, we introduce a neighbor-aware Point Transformer to enforce volumetric coherence. This module uses 3D positional encoding and attention over k-nearest neighbors to improve spatial consistency. The resulting model, Trans$^2$-CBCT, provides an additional gain of 0.63 dB PSNR and 0.0117 SSIM. Experiments on LUNA16 and ToothFairy show consistent gains from six to ten views, validating the effectiveness of combining CNN-Transformer features with point-based geometry reasoning for sparse-view CBCT reconstruction. 

---
# UProp: Investigating the Uncertainty Propagation of LLMs in Multi-Step Agentic Decision-Making 

**Authors**: Jinhao Duan, James Diffenderfer, Sandeep Madireddy, Tianlong Chen, Bhavya Kailkhura, Kaidi Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.17419)  

**Abstract**: As Large Language Models (LLMs) are integrated into safety-critical applications involving sequential decision-making in the real world, it is essential to know when to trust LLM decisions. Existing LLM Uncertainty Quantification (UQ) methods are primarily designed for single-turn question-answering formats, resulting in multi-step decision-making scenarios, e.g., LLM agentic system, being underexplored. In this paper, we introduce a principled, information-theoretic framework that decomposes LLM sequential decision uncertainty into two parts: (i) internal uncertainty intrinsic to the current decision, which is focused on existing UQ methods, and (ii) extrinsic uncertainty, a Mutual-Information (MI) quantity describing how much uncertainty should be inherited from preceding decisions. We then propose UProp, an efficient and effective extrinsic uncertainty estimator that converts the direct estimation of MI to the estimation of Pointwise Mutual Information (PMI) over multiple Trajectory-Dependent Decision Processes (TDPs). UProp is evaluated over extensive multi-step decision-making benchmarks, e.g., AgentBench and HotpotQA, with state-of-the-art LLMs, e.g., GPT-4.1 and DeepSeek-V3. Experimental results demonstrate that UProp significantly outperforms existing single-turn UQ baselines equipped with thoughtful aggregation strategies. Moreover, we provide a comprehensive analysis of UProp, including sampling efficiency, potential applications, and intermediate uncertainty propagation, to demonstrate its effectiveness. Codes will be available at this https URL. 

---
# Challenges in Grounding Language in the Real World 

**Authors**: Peter Lindes, Kaoutar Skiker  

**Link**: [PDF](https://arxiv.org/pdf/2506.17375)  

**Abstract**: A long-term goal of Artificial Intelligence is to build a language understanding system that allows a human to collaborate with a physical robot using language that is natural to the human. In this paper we highlight some of the challenges in doing this, and propose a solution that integrates the abilities of a cognitive agent capable of interactive task learning in a physical robot with the linguistic abilities of a large language model. We also point the way to an initial implementation of this approach. 

---
# From Drawings to Decisions: A Hybrid Vision-Language Framework for Parsing 2D Engineering Drawings into Structured Manufacturing Knowledge 

**Authors**: Muhammad Tayyab Khan, Lequn Chen, Zane Yong, Jun Ming Tan, Wenhe Feng, Seung Ki Moon  

**Link**: [PDF](https://arxiv.org/pdf/2506.17374)  

**Abstract**: Efficient and accurate extraction of key information from 2D engineering drawings is essential for advancing digital manufacturing workflows. Such information includes geometric dimensioning and tolerancing (GD&T), measures, material specifications, and textual annotations. Manual extraction is slow and labor-intensive, while generic OCR models often fail due to complex layouts, engineering symbols, and rotated text, leading to incomplete and unreliable outputs. These limitations result in incomplete and unreliable outputs. To address these challenges, we propose a hybrid vision-language framework that integrates a rotation-aware object detection model (YOLOv11-obb) with a transformer-based vision-language parser. Our structured pipeline applies YOLOv11-OBB to localize annotations and extract oriented bounding box (OBB) patches, which are then parsed into structured outputs using a fine-tuned, lightweight vision-language model (VLM). We curate a dataset of 1,367 2D mechanical drawings annotated across nine key categories. YOLOv11-OBB is trained on this dataset to detect OBBs and extract annotation patches. These are parsed using two open-source VLMs: Donut and Florence-2. Both models are lightweight and well-suited for specialized industrial tasks under limited computational overhead. Following fine-tuning of both models on the curated dataset of image patches paired with structured annotation labels, a comparative experiment is conducted to evaluate parsing performance across four key metrics. Donut outperforms Florence-2, achieving 88.5% precision, 99.2% recall, and a 93.5% F1-score, with a hallucination rate of 11.5%. Finally, a case study demonstrates how the extracted structured information supports downstream manufacturing tasks such as process and tool selection, showcasing the practical utility of the proposed framework in modernizing 2D drawing interpretation. 

---
# Multimodal Political Bias Identification and Neutralization 

**Authors**: Cedric Bernard, Xavier Pleimling, Amun Kharel, Chase Vickery  

**Link**: [PDF](https://arxiv.org/pdf/2506.17372)  

**Abstract**: Due to the presence of political echo chambers, it becomes imperative to detect and remove subjective bias and emotionally charged language from both the text and images of political articles. However, prior work has focused on solely the text portion of the bias rather than both the text and image portions. This is a problem because the images are just as powerful of a medium to communicate information as text is. To that end, we present a model that leverages both text and image bias which consists of four different steps. Image Text Alignment focuses on semantically aligning images based on their bias through CLIP models. Image Bias Scoring determines the appropriate bias score of images via a ViT classifier. Text De-Biasing focuses on detecting biased words and phrases and neutralizing them through BERT models. These three steps all culminate to the final step of debiasing, which replaces the text and the image with neutralized or reduced counterparts, which for images is done by comparing the bias scores. The results so far indicate that this approach is promising, with the text debiasing strategy being able to identify many potential biased words and phrases, and the ViT model showcasing effective training. The semantic alignment model also is efficient. However, more time, particularly in training, and resources are needed to obtain better results. A human evaluation portion was also proposed to ensure semantic consistency of the newly generated text and images. 

---
# AI based Content Creation and Product Recommendation Applications in E-commerce: An Ethical overview 

**Authors**: Aditi Madhusudan Jain, Ayush Jain  

**Link**: [PDF](https://arxiv.org/pdf/2506.17370)  

**Abstract**: As e-commerce rapidly integrates artificial intelligence for content creation and product recommendations, these technologies offer significant benefits in personalization and efficiency. AI-driven systems automate product descriptions, generate dynamic advertisements, and deliver tailored recommendations based on consumer behavior, as seen in major platforms like Amazon and Shopify. However, the widespread use of AI in e-commerce raises crucial ethical challenges, particularly around data privacy, algorithmic bias, and consumer autonomy. Bias -- whether cultural, gender-based, or socioeconomic -- can be inadvertently embedded in AI models, leading to inequitable product recommendations and reinforcing harmful stereotypes. This paper examines the ethical implications of AI-driven content creation and product recommendations, emphasizing the need for frameworks to ensure fairness, transparency, and need for more established and robust ethical standards. We propose actionable best practices to remove bias and ensure inclusivity, such as conducting regular audits of algorithms, diversifying training data, and incorporating fairness metrics into AI models. Additionally, we discuss frameworks for ethical conformance that focus on safeguarding consumer data privacy, promoting transparency in decision-making processes, and enhancing consumer autonomy. By addressing these issues, we provide guidelines for responsibly utilizing AI in e-commerce applications for content creation and product recommendations, ensuring that these technologies are both effective and ethically sound. 

---
# Re-Evaluating Code LLM Benchmarks Under Semantic Mutation 

**Authors**: Zhiyuan Pan, Xing Hu, Xin Xia, Xiaohu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.17369)  

**Abstract**: In the era of large language models (LLMs), code benchmarks have become an important research area in software engineering and are widely used by practitioners. These benchmarks evaluate the performance of LLMs on specific code-related tasks, such as code understanding and generation. A critical step in constructing code benchmarks is the design of prompts. However, as existing code benchmarks typically rely on a single prompt template per task, they are prone to the issue of prompt sensitivity, where minor prompt variations could result in substantial performance variations, leading to unreliable evaluations of model capabilities.
While previous studies have explored prompt sensitivity, their experimental designs and findings are limited to traditional natural language processing (NLP) tasks. In this paper, we present an empirical study to investigate prompt sensitivity in code benchmarks. We first propose a general framework that modifies prompt templates in a manner that preserves both their semantics and their structure as much as possible. Based on the framework, we conduct extensive experiments across eight code benchmark tasks on 10 representative open-source LLMs, with each task featuring 100 semantically similar prompt templates. We then analyze the evaluation results using various statistical metrics, focusing on both absolute and relative model performance. Our findings suggest that even slight prompt variations can lead to significant shifts in performance. Additionally, we observe that such variations can introduce inconsistencies in the performance rankings across different models. These insights highlight the need for considering prompt sensitivity when designing future code benchmarks, to ensure more reliable and accurate evaluation of LLM capabilities. 

---
# SAFEx: Analyzing Vulnerabilities of MoE-Based LLMs via Stable Safety-critical Expert Identification 

**Authors**: Zhenglin Lai, Mengyao Liao, Dong Xu, Zebin Zhao, Zhihang Yuan, Chao Fan, Jianqiang Li, Bingzhe Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.17368)  

**Abstract**: Large language models based on Mixture-of-Experts have achieved substantial gains in efficiency and scalability, yet their architectural uniqueness introduces underexplored safety alignment challenges. Existing safety alignment strategies, predominantly designed for dense models, are ill-suited to address MoE-specific vulnerabilities. In this work, we formalize and systematically study MoE model's positional vulnerability - the phenomenon where safety-aligned behaviors rely on specific expert modules, revealing critical risks inherent to MoE architectures. To this end, we present SAFEx, an analytical framework that robustly identifies, characterizes, and validates the safety-critical experts using a novel Stability-based Expert Selection (SES) algorithm. Notably, our approach enables the explicit decomposition of safety-critical experts into distinct functional groups, including those responsible for harmful content detection and those controlling safe response generation. Extensive experiments on mainstream MoE models, such as the recently released Qwen3-MoE, demonstrated that their intrinsic safety mechanisms heavily rely on a small subset of positional experts. Disabling these experts significantly compromised the models' ability to refuse harmful requests. For Qwen3-MoE with 6144 experts (in the FNN layer), we find that disabling as few as 12 identified safety-critical experts can cause the refusal rate to drop by 22%, demonstrating the disproportionate impact of a small set of experts on overall model safety. 

---
# Cash or Comfort? How LLMs Value Your Inconvenience 

**Authors**: Mateusz Cedro, Timour Ichmoukhamedov, Sofie Goethals, Yifan He, James Hinns, David Martens  

**Link**: [PDF](https://arxiv.org/pdf/2506.17367)  

**Abstract**: Large Language Models (LLMs) are increasingly proposed as near-autonomous artificial intelligence (AI) agents capable of making everyday decisions on behalf of humans. Although LLMs perform well on many technical tasks, their behaviour in personal decision-making remains less understood. Previous studies have assessed their rationality and moral alignment with human decisions. However, the behaviour of AI assistants in scenarios where financial rewards are at odds with user comfort has not yet been thoroughly explored. In this paper, we tackle this problem by quantifying the prices assigned by multiple LLMs to a series of user discomforts: additional walking, waiting, hunger and pain. We uncover several key concerns that strongly question the prospect of using current LLMs as decision-making assistants: (1) a large variance in responses between LLMs, (2) within a single LLM, responses show fragility to minor variations in prompt phrasing (e.g., reformulating the question in the first person can considerably alter the decision), (3) LLMs can accept unreasonably low rewards for major inconveniences (e.g., 1 Euro to wait 10 hours), and (4) LLMs can reject monetary gains where no discomfort is imposed (e.g., 1,000 Euro to wait 0 minutes). These findings emphasize the need for scrutiny of how LLMs value human inconvenience, particularly as we move toward applications where such cash-versus-comfort trade-offs are made on users' behalf. 

---
# AI-based Multimodal Biometrics for Detecting Smartphone Distractions: Application to Online Learning 

**Authors**: Alvaro Becerra, Roberto Daza, Ruth Cobos, Aythami Morales, Mutlu Cukurova, Julian Fierrez  

**Link**: [PDF](https://arxiv.org/pdf/2506.17364)  

**Abstract**: This work investigates the use of multimodal biometrics to detect distractions caused by smartphone use during tasks that require sustained attention, with a focus on computer-based online learning. Although the methods are applicable to various domains, such as autonomous driving, we concentrate on the challenges learners face in maintaining engagement amid internal (e.g., motivation), system-related (e.g., course design) and contextual (e.g., smartphone use) factors. Traditional learning platforms often lack detailed behavioral data, but Multimodal Learning Analytics (MMLA) and biosensors provide new insights into learner attention. We propose an AI-based approach that leverages physiological signals and head pose data to detect phone use. Our results show that single biometric signals, such as brain waves or heart rate, offer limited accuracy, while head pose alone achieves 87%. A multimodal model combining all signals reaches 91% accuracy, highlighting the benefits of integration. We conclude by discussing the implications and limitations of deploying these models for real-time support in online learning environments. 

---
# A Large-Scale Real-World Evaluation of LLM-Based Virtual Teaching Assistant 

**Authors**: Sunjun Kweon, Sooyohn Nam, Hyunseung Lim, Hwajung Hong, Edward Choi  

**Link**: [PDF](https://arxiv.org/pdf/2506.17363)  

**Abstract**: Virtual Teaching Assistants (VTAs) powered by Large Language Models (LLMs) have the potential to enhance student learning by providing instant feedback and facilitating multi-turn interactions. However, empirical studies on their effectiveness and acceptance in real-world classrooms are limited, leaving their practical impact uncertain. In this study, we develop an LLM-based VTA and deploy it in an introductory AI programming course with 477 graduate students. To assess how student perceptions of the VTA's performance evolve over time, we conduct three rounds of comprehensive surveys at different stages of the course. Additionally, we analyze 3,869 student--VTA interaction pairs to identify common question types and engagement patterns. We then compare these interactions with traditional student--human instructor interactions to evaluate the VTA's role in the learning process. Through a large-scale empirical study and interaction analysis, we assess the feasibility of deploying VTAs in real-world classrooms and identify key challenges for broader adoption. Finally, we release the source code of our VTA system, fostering future advancements in AI-driven education: \texttt{this https URL}. 

---
# Speeding up Local Optimization in Vehicle Routing with Tensor-based GPU Acceleration 

**Authors**: Zhenyu Lei, Jin-Kao Hao, Qinghua Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.17357)  

**Abstract**: Local search plays a central role in many effective heuristic algorithms for the vehicle routing problem (VRP) and its variants. However, neighborhood exploration is known to be computationally expensive and time consuming, especially for large instances or problems with complex constraints. In this study, we explore a promising direction to address this challenge by introducing an original tensor-based GPU acceleration method designed to speed up the commonly used local search operators in vehicle routing. By using an attribute-based representation, the method offers broad extensibility, making it applicable to different VRP variants. Its low-coupling architecture, with intensive computations completely offloaded to the GPU, ensures seamless integration in various local search-based algorithms and frameworks, leading to significant improvements in computational efficiency and potentially improved solution quality. Through comparative experiments on benchmark instances of three routing problems, we demonstrate the substantial computational advantages of the proposed approach over traditional CPU-based implementations. We also provide a detailed analysis of the strengths and limitations of the method, providing valuable insights into its performance characteristics and identifying potential bottlenecks in practical applications. These findings contribute to a better understanding and suggest directions for future improvements. 

---
# Automatic Large Language Models Creation of Interactive Learning Lessons 

**Authors**: Jionghao Lin, Jiarui Rao, Yiyang Zhao, Yuting Wang, Ashish Gurung, Amanda Barany, Jaclyn Ocumpaugh, Ryan S. Baker, Kenneth R. Koedinger  

**Link**: [PDF](https://arxiv.org/pdf/2506.17356)  

**Abstract**: We explore the automatic generation of interactive, scenario-based lessons designed to train novice human tutors who teach middle school mathematics online. Employing prompt engineering through a Retrieval-Augmented Generation approach with GPT-4o, we developed a system capable of creating structured tutor training lessons. Our study generated lessons in English for three key topics: Encouraging Students' Independence, Encouraging Help-Seeking Behavior, and Turning on Cameras, using a task decomposition prompting strategy that breaks lesson generation into sub-tasks. The generated lessons were evaluated by two human evaluators, who provided both quantitative and qualitative evaluations using a comprehensive rubric informed by lesson design research. Results demonstrate that the task decomposition strategy led to higher-rated lessons compared to single-step generation. Human evaluators identified several strengths in the LLM-generated lessons, including well-structured content and time-saving potential, while also noting limitations such as generic feedback and a lack of clarity in some instructional sections. These findings underscore the potential of hybrid human-AI approaches for generating effective lessons in tutor training. 

---
# Differentiation-Based Extraction of Proprietary Data from Fine-Tuned LLMs 

**Authors**: Zongjie Li, Daoyuan Wu, Shuai Wang, Zhendong Su  

**Link**: [PDF](https://arxiv.org/pdf/2506.17353)  

**Abstract**: The increasing demand for domain-specific and human-aligned Large Language Models (LLMs) has led to the widespread adoption of Supervised Fine-Tuning (SFT) techniques. SFT datasets often comprise valuable instruction-response pairs, making them highly valuable targets for potential extraction. This paper studies this critical research problem for the first time. We start by formally defining and formulating the problem, then explore various attack goals, types, and variants based on the unique properties of SFT data in real-world scenarios. Based on our analysis of extraction behaviors of direct extraction, we develop a novel extraction method specifically designed for SFT models, called Differentiated Data Extraction (DDE), which exploits the confidence levels of fine-tuned models and their behavioral differences from pre-trained base models. Through extensive experiments across multiple domains and scenarios, we demonstrate the feasibility of SFT data extraction using DDE. Our results show that DDE consistently outperforms existing extraction baselines in all attack settings. To counter this new attack, we propose a defense mechanism that mitigates DDE attacks with minimal impact on model performance. Overall, our research reveals hidden data leak risks in fine-tuned LLMs and provides insights for developing more secure models. 

---
# Towards Safety Evaluations of Theory of Mind in Large Language Models 

**Authors**: Tatsuhiro Aoshima, Mitsuaki Akiyama  

**Link**: [PDF](https://arxiv.org/pdf/2506.17352)  

**Abstract**: As the capabilities of large language models (LLMs) continue to advance, the importance of rigorous safety evaluation is becoming increasingly evident. Recent concerns within the realm of safety assessment have highlighted instances in which LLMs exhibit behaviors that appear to disable oversight mechanisms and respond in a deceptive manner. For example, there have been reports suggesting that, when confronted with information unfavorable to their own persistence during task execution, LLMs may act covertly and even provide false answers to questions intended to verify their this http URL evaluate the potential risk of such deceptive actions toward developers or users, it is essential to investigate whether these behaviors stem from covert, intentional processes within the model. In this study, we propose that it is necessary to measure the theory of mind capabilities of LLMs. We begin by reviewing existing research on theory of mind and identifying the perspectives and tasks relevant to its application in safety evaluation. Given that theory of mind has been predominantly studied within the context of developmental psychology, we analyze developmental trends across a series of open-weight LLMs. Our results indicate that while LLMs have improved in reading comprehension, their theory of mind capabilities have not shown comparable development. Finally, we present the current state of safety evaluation with respect to LLMs' theory of mind, and discuss remaining challenges for future work. 

---
# Zero-Shot Cognitive Impairment Detection from Speech Using AudioLLM 

**Authors**: Mostafa Shahin, Beena Ahmed, Julien Epps  

**Link**: [PDF](https://arxiv.org/pdf/2506.17351)  

**Abstract**: Cognitive impairment (CI) is of growing public health concern, and early detection is vital for effective intervention. Speech has gained attention as a non-invasive and easily collectible biomarker for assessing cognitive decline. Traditional CI detection methods typically rely on supervised models trained on acoustic and linguistic features extracted from speech, which often require manual annotation and may not generalise well across datasets and languages. In this work, we propose the first zero-shot speech-based CI detection method using the Qwen2- Audio AudioLLM, a model capable of processing both audio and text inputs. By designing prompt-based instructions, we guide the model in classifying speech samples as indicative of normal cognition or cognitive impairment. We evaluate our approach on two datasets: one in English and another multilingual, spanning different cognitive assessment tasks. Our results show that the zero-shot AudioLLM approach achieves performance comparable to supervised methods and exhibits promising generalizability and consistency across languages, tasks, and datasets. 

---
# CUBA: Controlled Untargeted Backdoor Attack against Deep Neural Networks 

**Authors**: Yinghao Wu, Liyan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.17350)  

**Abstract**: Backdoor attacks have emerged as a critical security threat against deep neural networks in recent years. The majority of existing backdoor attacks focus on targeted backdoor attacks, where trigger is strongly associated to specific malicious behavior. Various backdoor detection methods depend on this inherent property and shows effective results in identifying and mitigating such targeted attacks. However, a purely untargeted attack in backdoor scenarios is, in some sense, self-weakening, since the target nature is what makes backdoor attacks so powerful. In light of this, we introduce a novel Constrained Untargeted Backdoor Attack (CUBA), which combines the flexibility of untargeted attacks with the intentionality of targeted attacks. The compromised model, when presented with backdoor images, will classify them into random classes within a constrained range of target classes selected by the attacker. This combination of randomness and determinedness enables the proposed untargeted backdoor attack to natively circumvent existing backdoor defense methods. To implement the untargeted backdoor attack under controlled flexibility, we propose to apply logit normalization on cross-entropy loss with flipped one-hot labels. By constraining the logit during training, the compromised model will show a uniform distribution across selected target classes, resulting in controlled untargeted attack. Extensive experiments demonstrate the effectiveness of the proposed CUBA on different datasets. 

---
# Advanced Game-Theoretic Frameworks for Multi-Agent AI Challenges: A 2025 Outlook 

**Authors**: Pavel Malinovskiy  

**Link**: [PDF](https://arxiv.org/pdf/2506.17348)  

**Abstract**: This paper presents a substantially reworked examination of how advanced game-theoretic paradigms can serve as a foundation for the next-generation challenges in Artificial Intelligence (AI), forecasted to arrive in or around 2025. Our focus extends beyond traditional models by incorporating dynamic coalition formation, language-based utilities, sabotage risks, and partial observability. We provide a set of mathematical formalisms, simulations, and coding schemes that illustrate how multi-agent AI systems may adapt and negotiate in complex environments. Key elements include repeated games, Bayesian updates for adversarial detection, and moral framing within payoff structures. This work aims to equip AI researchers with robust theoretical tools for aligning strategic interaction in uncertain, partially adversarial contexts. 

---
# Distinguishing Predictive and Generative AI in Regulation 

**Authors**: Jennifer Wang, Andrew Selbst, Solon Barocas, Suresh Venkatasubramanian  

**Link**: [PDF](https://arxiv.org/pdf/2506.17347)  

**Abstract**: Over the past decade, policymakers have developed a set of regulatory tools to ensure AI development aligns with key societal goals. Many of these tools were initially developed in response to concerns with predictive AI and therefore encode certain assumptions about the nature of AI systems and the utility of certain regulatory approaches. With the advent of generative AI, however, some of these assumptions no longer hold, even as policymakers attempt to maintain a single regulatory target that covers both types of AI.
In this paper, we identify four distinct aspects of generative AI that call for meaningfully different policy responses. These are the generality and adaptability of generative AI that make it a poor regulatory target, the difficulty of designing effective evaluations, new legal concerns that change the ecosystem of stakeholders and sources of expertise, and the distributed structure of the generative AI value chain.
In light of these distinctions, policymakers will need to evaluate where the past decade of policy work remains relevant and where new policies, designed to address the unique risks posed by generative AI, are necessary. We outline three recommendations for policymakers to more effectively identify regulatory targets and leverage constraints across the broader ecosystem to govern generative AI. 

---
# A Novel Multi-layer Task-centric and Data Quality Framework for Autonomous Driving 

**Authors**: Yuhan Zhou, Haihua Chen, Kewei Sha  

**Link**: [PDF](https://arxiv.org/pdf/2506.17346)  

**Abstract**: The next-generation autonomous vehicles (AVs), embedded with frequent real-time decision-making, will rely heavily on a large volume of multisource and multimodal data. In real-world settings, the data quality (DQ) of different sources and modalities usually varies due to unexpected environmental factors or sensor issues. However, both researchers and practitioners in the AV field overwhelmingly concentrate on models/algorithms while undervaluing the DQ. To fulfill the needs of the next-generation AVs with guarantees of functionality, efficiency, and trustworthiness, this paper proposes a novel task-centric and data quality vase framework which consists of five layers: data layer, DQ layer, task layer, application layer, and goal layer. The proposed framework aims to map DQ with task requirements and performance goals. To illustrate, a case study investigating redundancy on the nuScenes dataset proves that partially removing redundancy on multisource image data could improve YOLOv8 object detection task performance. Analysis on multimodal data of image and LiDAR further presents existing redundancy DQ issues. This paper opens up a range of critical but unexplored challenges at the intersection of DQ, task orchestration, and performance-oriented system development in AVs. It is expected to guide the AV community toward building more adaptive, explainable, and resilient AVs that respond intelligently to dynamic environments and heterogeneous data streams. Code, data, and implementation details are publicly available at: this https URL. 

---
# Adaptive Social Metaverse Streaming based on Federated Multi-Agent Deep Reinforcement Learning 

**Authors**: Zijian Long, Haopeng Wang, Haiwei Dong, Abdulmotaleb El Saddik  

**Link**: [PDF](https://arxiv.org/pdf/2506.17342)  

**Abstract**: The social metaverse is a growing digital ecosystem that blends virtual and physical worlds. It allows users to interact socially, work, shop, and enjoy entertainment. However, privacy remains a major challenge, as immersive interactions require continuous collection of biometric and behavioral data. At the same time, ensuring high-quality, low-latency streaming is difficult due to the demands of real-time interaction, immersive rendering, and bandwidth optimization. To address these issues, we propose ASMS (Adaptive Social Metaverse Streaming), a novel streaming system based on Federated Multi-Agent Proximal Policy Optimization (F-MAPPO). ASMS leverages F-MAPPO, which integrates federated learning (FL) and deep reinforcement learning (DRL) to dynamically adjust streaming bit rates while preserving user privacy. Experimental results show that ASMS improves user experience by at least 14% compared to existing streaming methods across various network conditions. Therefore, ASMS enhances the social metaverse experience by providing seamless and immersive streaming, even in dynamic and resource-constrained networks, while ensuring that sensitive user data remains on local devices. 

---
# PBFT-Backed Semantic Voting for Multi-Agent Memory Pruning 

**Authors**: Duong Bach  

**Link**: [PDF](https://arxiv.org/pdf/2506.17338)  

**Abstract**: The proliferation of multi-agent systems (MAS) in complex, dynamic environments necessitates robust and efficient mechanisms for managing shared knowledge. A critical challenge is ensuring that distributed memories remain synchronized, relevant, and free from the accumulation of outdated or inconsequential data - a process analogous to biological forgetting. This paper introduces the Co-Forgetting Protocol, a novel, comprehensive framework designed to address this challenge by enabling synchronized memory pruning in MAS. The protocol integrates three key components: (1) context-aware semantic voting, where agents utilize a lightweight DistilBERT model to assess the relevance of memory items based on their content and the current operational context; (2) multi-scale temporal decay functions, which assign diminishing importance to memories based on their age and access frequency across different time horizons; and (3) a Practical Byzantine Fault Tolerance (PBFT)-based consensus mechanism, ensuring that decisions to retain or discard memory items are agreed upon by a qualified and fault-tolerant majority of agents, even in the presence of up to f Byzantine (malicious or faulty) agents in a system of N greater than or equal to 3f+1 agents. The protocol leverages gRPC for efficient inter-agent communication and Pinecone for scalable vector embedding storage and similarity search, with SQLite managing metadata. Experimental evaluations in a simulated MAS environment with four agents demonstrate the protocol's efficacy, achieving a 52% reduction in memory footprint over 500 epochs, 88% voting accuracy in forgetting decisions against human-annotated benchmarks, a 92% PBFT consensus success rate under simulated Byzantine conditions, and an 82% cache hit rate for memory access. 

---
# Can Common VLMs Rival Medical VLMs? Evaluation and Strategic Insights 

**Authors**: Yuan Zhong, Ruinan Jin, Xiaoxiao Li, Qi Dou  

**Link**: [PDF](https://arxiv.org/pdf/2506.17337)  

**Abstract**: Medical vision-language models (VLMs) leverage large-scale pretraining for diverse imaging tasks but require substantial computational and data resources. Meanwhile, common or general-purpose VLMs (e.g., CLIP, LLaVA), though not trained for medical use, show promise with fine-tuning. This raises a key question: Can efficient fine-tuned common VLMs rival generalist medical VLMs for solving specific medical imaging tasks? This study systematically evaluates common and medical VLMs across disease diagnosis and visual question answering (VQA). Using CLIP-based and LLaVA-based models, we examine (1) off-the-shelf performance gaps in in-domain (ID) settings, (2) whether fine-tuning bridges these gaps, and (3) generalization to out-of-domain (OOD) tasks on unseen medical modalities. While medical-specific pretraining provides advantages in ID settings, common VLMs match or surpass medical-specific models after lightweight fine-tuning, with LoRA-based adaptation proving highly effective among different tasks. In OOD tasks, common VLMs demonstrate strong adaptability in some tasks, challenging the assumption that medical-specific pre-training is essential. These findings suggest that leveraging common VLMs with fine-tuning offers a scalable and cost-effective alternative to developing large-scale medical VLMs, providing crucial insights for future research in the medical imaging field. 

---
# LMR-BENCH: Evaluating LLM Agent's Ability on Reproducing Language Modeling Research 

**Authors**: Shuo Yan, Ruochen Li, Ziming Luo, Zimu Wang, Daoyang Li, Liqiang Jing, Kaiyu He, Peilin Wu, George Michalopoulos, Yue Zhang, Ziyang Zhang, Mian Zhang, Zhiyu Chen, Xinya Du  

**Link**: [PDF](https://arxiv.org/pdf/2506.17335)  

**Abstract**: Large language model (LLM) agents have demonstrated remarkable potential in advancing scientific discovery. However, their capability in the fundamental yet crucial task of reproducing code from research papers, especially in the NLP domain, remains underexplored. This task includes unique complex reasoning challenges in the intellectual synthesis of abstract concepts and the comprehension of code repositories with interdependent files. Motivated by this gap, we present LMR-BENCH, a benchmark designed to systematically evaluate the capability of LLM agents on code reproduction from Language Modeling Research. It consists of 28 code reproduction tasks derived from 23 research papers published in top-tier NLP venues over the past five years, spanning nine fundamental categories. Models are provided with a research paper, a code repository containing one or more masked functions, and instructions for implementing these functions. We conduct extensive experiments in standard prompting and LLM agent settings with state-of-the-art LLMs, evaluating the accuracy of unit tests and performing LLM-based evaluation of code correctness. Experimental results reveal that even the most advanced models still exhibit persistent limitations in scientific reasoning and code synthesis, highlighting critical gaps in LLM agents' ability to autonomously reproduce scientific research 

---
# P2MFDS: A Privacy-Preserving Multimodal Fall Detection System for Elderly People in Bathroom Environments 

**Authors**: Haitian Wang, Yiren Wang, Xinyu Wang, Yumeng Miao, Yuliang Zhang, Yu Zhang, Atif Mansoor  

**Link**: [PDF](https://arxiv.org/pdf/2506.17332)  

**Abstract**: By 2050, people aged 65 and over are projected to make up 16 percent of the global population. As aging is closely associated with increased fall risk, particularly in wet and confined environments such as bathrooms where over 80 percent of falls occur. Although recent research has increasingly focused on non-intrusive, privacy-preserving approaches that do not rely on wearable devices or video-based monitoring, these efforts have not fully overcome the limitations of existing unimodal systems (e.g., WiFi-, infrared-, or mmWave-based), which are prone to reduced accuracy in complex environments. These limitations stem from fundamental constraints in unimodal sensing, including system bias and environmental interference, such as multipath fading in WiFi-based systems and drastic temperature changes in infrared-based methods. To address these challenges, we propose a Privacy-Preserving Multimodal Fall Detection System for Elderly People in Bathroom Environments. First, we develop a sensor evaluation framework to select and fuse millimeter-wave radar with 3D vibration sensing, and use it to construct and preprocess a large-scale, privacy-preserving multimodal dataset in real bathroom settings, which will be released upon publication. Second, we introduce P2MFDS, a dual-stream network combining a CNN-BiLSTM-Attention branch for radar motion dynamics with a multi-scale CNN-SEBlock-Self-Attention branch for vibration impact detection. By uniting macro- and micro-scale features, P2MFDS delivers significant gains in accuracy and recall over state-of-the-art approaches. Code and pretrained models will be made available at: this https URL. 

---
# On the Performance of Cyber-Biomedical Features for Intrusion Detection in Healthcare 5.0 

**Authors**: Pedro H. Lui, Lucas P. Siqueira, Juliano F. Kazienko, Vagner E. Quincozes, Silvio E. Quincozes, Daniel Welfer  

**Link**: [PDF](https://arxiv.org/pdf/2506.17329)  

**Abstract**: Healthcare 5.0 integrates Artificial Intelligence (AI), the Internet of Things (IoT), real-time monitoring, and human-centered design toward personalized medicine and predictive diagnostics. However, the increasing reliance on interconnected medical technologies exposes them to cyber threats. Meanwhile, current AI-driven cybersecurity models often neglect biomedical data, limiting their effectiveness and interpretability. This study addresses this gap by applying eXplainable AI (XAI) to a Healthcare 5.0 dataset that integrates network traffic and biomedical sensor data. Classification outputs indicate that XGBoost achieved 99% F1-score for benign and data alteration, and 81% for spoofing. Explainability findings reveal that network data play a dominant role in intrusion detection whereas biomedical features contributed to spoofing detection, with temperature reaching a Shapley values magnitude of 0.37. 

---
# RadarSeq: A Temporal Vision Framework for User Churn Prediction via Radar Chart Sequences 

**Authors**: Sina Najafi, M. Hadi Sepanj, Fahimeh Jafari  

**Link**: [PDF](https://arxiv.org/pdf/2506.17325)  

**Abstract**: Predicting user churn in non-subscription gig platforms, where disengagement is implicit, poses unique challenges due to the absence of explicit labels and the dynamic nature of user behavior. Existing methods often rely on aggregated snapshots or static visual representations, which obscure temporal cues critical for early detection. In this work, we propose a temporally-aware computer vision framework that models user behavioral patterns as a sequence of radar chart images, each encoding day-level behavioral features. By integrating a pretrained CNN encoder with a bidirectional LSTM, our architecture captures both spatial and temporal patterns underlying churn behavior. Extensive experiments on a large real-world dataset demonstrate that our method outperforms classical models and ViT-based radar chart baselines, yielding gains of 17.7 in F1 score, 29.4 in precision, and 16.1 in AUC, along with improved interpretability. The framework's modular design, explainability tools, and efficient deployment characteristics make it suitable for large-scale churn modeling in dynamic gig-economy platforms. 

---
# I Know Which LLM Wrote Your Code Last Summer: LLM generated Code Stylometry for Authorship Attribution 

**Authors**: Tamas Bisztray, Bilel Cherif, Richard A. Dubniczky, Nils Gruschka, Bertalan Borsos, Mohamed Amine Ferrag, Attila Kovacs, Vasileios Mavroeidis, Norbert Tihanyi  

**Link**: [PDF](https://arxiv.org/pdf/2506.17323)  

**Abstract**: Detecting AI-generated code, deepfakes, and other synthetic content is an emerging research challenge. As code generated by Large Language Models (LLMs) becomes more common, identifying the specific model behind each sample is increasingly important. This paper presents the first systematic study of LLM authorship attribution for C programs. We released CodeT5-Authorship, a novel model that uses only the encoder layers from the original CodeT5 encoder-decoder architecture, discarding the decoder to focus on classification. Our model's encoder output (first token) is passed through a two-layer classification head with GELU activation and dropout, producing a probability distribution over possible authors. To evaluate our approach, we introduce LLM-AuthorBench, a benchmark of 32,000 compilable C programs generated by eight state-of-the-art LLMs across diverse tasks. We compare our model to seven traditional ML classifiers and eight fine-tuned transformer models, including BERT, RoBERTa, CodeBERT, ModernBERT, DistilBERT, DeBERTa-V3, Longformer, and LoRA-fine-tuned Qwen2-1.5B. In binary classification, our model achieves 97.56% accuracy in distinguishing C programs generated by closely related models such as GPT-4.1 and GPT-4o, and 95.40% accuracy for multi-class attribution among five leading LLMs (Gemini 2.5 Flash, Claude 3.5 Haiku, GPT-4.1, Llama 3.3, and DeepSeek-V3). To support open science, we release the CodeT5-Authorship architecture, the LLM-AuthorBench benchmark, and all relevant Google Colab scripts on GitHub: this https URL. 

---
# Context manipulation attacks : Web agents are susceptible to corrupted memory 

**Authors**: Atharv Singh Patlan, Ashwin Hebbar, Pramod Viswanath, Prateek Mittal  

**Link**: [PDF](https://arxiv.org/pdf/2506.17318)  

**Abstract**: Autonomous web navigation agents, which translate natural language instructions into sequences of browser actions, are increasingly deployed for complex tasks across e-commerce, information retrieval, and content discovery. Due to the stateless nature of large language models (LLMs), these agents rely heavily on external memory systems to maintain context across interactions. Unlike centralized systems where context is securely stored server-side, agent memory is often managed client-side or by third-party applications, creating significant security vulnerabilities. This was recently exploited to attack production systems.
We introduce and formalize "plan injection," a novel context manipulation attack that corrupts these agents' internal task representations by targeting this vulnerable context. Through systematic evaluation of two popular web agents, Browser-use and Agent-E, we show that plan injections bypass robust prompt injection defenses, achieving up to 3x higher attack success rates than comparable prompt-based attacks. Furthermore, "context-chained injections," which craft logical bridges between legitimate user goals and attacker objectives, lead to a 17.7% increase in success rate for privacy exfiltration tasks. Our findings highlight that secure memory handling must be a first-class concern in agentic systems. 

---
# Heterogeneous Temporal Hypergraph Neural Network 

**Authors**: Huan Liu, Pengfei Jiao, Mengzhou Gao, Chaochao Chen, Di Jin  

**Link**: [PDF](https://arxiv.org/pdf/2506.17312)  

**Abstract**: Graph representation learning (GRL) has emerged as an effective technique for modeling graph-structured data. When modeling heterogeneity and dynamics in real-world complex networks, GRL methods designed for complex heterogeneous temporal graphs (HTGs) have been proposed and have achieved successful applications in various fields. However, most existing GRL methods mainly focus on preserving the low-order topology information while ignoring higher-order group interaction relationships, which are more consistent with real-world networks. In addition, most existing hypergraph methods can only model static homogeneous graphs, limiting their ability to model high-order interactions in HTGs. Therefore, to simultaneously enable the GRL model to capture high-order interaction relationships in HTGs, we first propose a formal definition of heterogeneous temporal hypergraphs and $P$-uniform heterogeneous hyperedge construction algorithm that does not rely on additional information. Then, a novel Heterogeneous Temporal HyperGraph Neural network (HTHGN), is proposed to fully capture higher-order interactions in HTGs. HTHGN contains a hierarchical attention mechanism module that simultaneously performs temporal message-passing between heterogeneous nodes and hyperedges to capture rich semantics in a wider receptive field brought by hyperedges. Furthermore, HTHGN performs contrastive learning by maximizing the consistency between low-order correlated heterogeneous node pairs on HTG to avoid the low-order structural ambiguity issue. Detailed experimental results on three real-world HTG datasets verify the effectiveness of the proposed HTHGN for modeling high-order interactions in HTGs and demonstrate significant performance improvements. 

---
# AlgoSelect: Universal Algorithm Selection via the Comb Operator 

**Authors**: Jasper Yao  

**Link**: [PDF](https://arxiv.org/pdf/2506.17304)  

**Abstract**: We introduce AlgoSelect, a principled framework for learning optimal algorithm selection from data, centered around the novel Comb Operator. Given a set of algorithms and a feature representation of problems, AlgoSelect learns to interpolate between diverse computational approaches. For pairs of algorithms, a simple sigmoid-gated selector, an instance of the Comb Operator, facilitates this interpolation. We extend this to an N-Path Comb for multiple algorithms. We prove that this framework is universal (can approximate any algorithm selector), information-theoretically optimal in its learnability (thresholds for selection converge almost surely, demonstrated via Borel-Cantelli arguments), computationally efficient, and robust. Key theoretical contributions include: (1) a universal approximation theorem demonstrating that Comb-based selectors can achieve arbitrary accuracy; (2) information-theoretic learnability for selection thresholds; (3) formalization of the Comb Operator within linear operator theory, detailing its boundedness and spectral properties; (4) an N-Path Comb generalization for multi-algorithm selection; and (5) a practical learning framework for the adaptive seeding functions that guide the Comb Operator. Empirical validation on a comprehensive 20$\times$20 problem-algorithm study demonstrates near-perfect selection (99.9\%+ accuracy) with remarkably few samples and rapid convergence, revealing that $H(\text{Algorithm}|\text{Problem}) \approx 0$ in structured domains. AlgoSelect provides a theoretically grounded, practically deployable solution to automated algorithm selection with provable optimality and learnability guarantees, with significant implications for AI and adaptive systems. 

---
# LLM Jailbreak Oracle 

**Authors**: Shuyi Lin, Anshuman Suri, Alina Oprea, Cheng Tan  

**Link**: [PDF](https://arxiv.org/pdf/2506.17299)  

**Abstract**: As large language models (LLMs) become increasingly deployed in safety-critical applications, the lack of systematic methods to assess their vulnerability to jailbreak attacks presents a critical security gap. We introduce the jailbreak oracle problem: given a model, prompt, and decoding strategy, determine whether a jailbreak response can be generated with likelihood exceeding a specified threshold. This formalization enables a principled study of jailbreak vulnerabilities. Answering the jailbreak oracle problem poses significant computational challenges -- the search space grows exponentially with the length of the response tokens. We present Boa, the first efficient algorithm for solving the jailbreak oracle problem. Boa employs a three-phase search strategy: (1) constructing block lists to identify refusal patterns, (2) breadth-first sampling to identify easily accessible jailbreaks, and (3) depth-first priority search guided by fine-grained safety scores to systematically explore promising low-probability paths. Boa enables rigorous security assessments including systematic defense evaluation, standardized comparison of red team attacks, and model certification under extreme adversarial conditions. 

---
# Mercury: Ultra-Fast Language Models Based on Diffusion 

**Authors**: Inception Labs, Samar Khanna, Siddhant Kharbanda, Shufan Li, Harshit Varma, Eric Wang, Sawyer Birnbaum, Ziyang Luo, Yanis Miraoui, Akash Palrecha, Stefano Ermon, Aditya Grover, Volodymyr Kuleshov  

**Link**: [PDF](https://arxiv.org/pdf/2506.17298)  

**Abstract**: We present Mercury, a new generation of commercial-scale large language models (LLMs) based on diffusion. These models are parameterized via the Transformer architecture and trained to predict multiple tokens in parallel. In this report, we detail Mercury Coder, our first set of diffusion LLMs designed for coding applications. Currently, Mercury Coder comes in two sizes: Mini and Small. These models set a new state-of-the-art on the speed-quality frontier. Based on independent evaluations conducted by Artificial Analysis, Mercury Coder Mini and Mercury Coder Small achieve state-of-the-art throughputs of 1109 tokens/sec and 737 tokens/sec, respectively, on NVIDIA H100 GPUs and outperform speed-optimized frontier models by up to 10x on average while maintaining comparable quality. We discuss additional results on a variety of code benchmarks spanning multiple languages and use-cases as well as real-world validation by developers on Copilot Arena, where the model currently ranks second on quality and is the fastest model overall. We also release a public API at this https URL and free playground at this https URL 

---
# SafeRL-Lite: A Lightweight, Explainable, and Constrained Reinforcement Learning Library 

**Authors**: Satyam Mishra, Phung Thao Vi, Shivam Mishra, Vishwanath Bijalwan, Vijay Bhaskar Semwal, Abdul Manan Khan  

**Link**: [PDF](https://arxiv.org/pdf/2506.17297)  

**Abstract**: We introduce SafeRL-Lite, an open-source Python library for building reinforcement learning (RL) agents that are both constrained and explainable. Existing RL toolkits often lack native mechanisms for enforcing hard safety constraints or producing human-interpretable rationales for decisions. SafeRL-Lite provides modular wrappers around standard Gym environments and deep Q-learning agents to enable: (i) safety-aware training via constraint enforcement, and (ii) real-time post-hoc explanation via SHAP values and saliency maps. The library is lightweight, extensible, and installable via pip, and includes built-in metrics for constraint violations. We demonstrate its effectiveness on constrained variants of CartPole and provide visualizations that reveal both policy logic and safety adherence. The full codebase is available at: this https URL. 

---
# Semantic uncertainty in advanced decoding methods for LLM generation 

**Authors**: Darius Foodeei, Simin Fan, Martin Jaggi  

**Link**: [PDF](https://arxiv.org/pdf/2506.17296)  

**Abstract**: This study investigates semantic uncertainty in large language model (LLM) outputs across different decoding methods, focusing on emerging techniques like speculative sampling and chain-of-thought (CoT) decoding. Through experiments on question answering, summarization, and code generation tasks, we analyze how different decoding strategies affect both the diversity and reliability of model outputs. Our findings reveal that while CoT decoding demonstrates higher semantic diversity, it maintains lower predictive entropy, suggesting that structured exploration can lead to more confident and accurate outputs. This is evidenced by a 48.8% improvement in code generation Pass@2 rates, despite lower alignment with reference solutions. For summarization tasks, speculative sampling proved particularly effective, achieving superior ROUGE scores while maintaining moderate semantic diversity. Our results challenge conventional assumptions about trade-offs between diversity and accuracy in language model outputs, demonstrating that properly structured decoding methods can increase semantic exploration while maintaining or improving output quality. These findings have significant implications for deploying language models in practical applications where both reliability and diverse solution generation are crucial. 

---
# AI-Generated Game Commentary: A Survey and a Datasheet Repository 

**Authors**: Qirui Zheng, Xingbo Wang, Keyuan Cheng, Yunlong Lu, Wenxin Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.17294)  

**Abstract**: AI-Generated Game Commentary (AIGGC) has gained increasing attention due to its market potential and inherent technical challenges. As a comprehensive multimodal Natural Language Processing (NLP) task, AIGGC imposes substantial demands on language models, including factual accuracy, logical reasoning, expressive text generation, generation speed, and context management. In this paper, we introduce a general framework for AIGGC and present a comprehensive survey of 45 existing game commentary dataset and methods according to key challenges they aim to address in this domain. We further classify and compare various evaluation metrics commonly used in this domain. To support future research and benchmarking, we also provide a structured datasheet summarizing the essential attributes of these datasets in appendix, which is meanwhile publicly available in an open repository. 

---
# Theoretically Unmasking Inference Attacks Against LDP-Protected Clients in Federated Vision Models 

**Authors**: Quan Nguyen, Minh N. Vu, Truc Nguyen, My T. Thai  

**Link**: [PDF](https://arxiv.org/pdf/2506.17292)  

**Abstract**: Federated Learning enables collaborative learning among clients via a coordinating server while avoiding direct data sharing, offering a perceived solution to preserve privacy. However, recent studies on Membership Inference Attacks (MIAs) have challenged this notion, showing high success rates against unprotected training data. While local differential privacy (LDP) is widely regarded as a gold standard for privacy protection in data analysis, most studies on MIAs either neglect LDP or fail to provide theoretical guarantees for attack success rates against LDP-protected data. To address this gap, we derive theoretical lower bounds for the success rates of low-polynomial time MIAs that exploit vulnerabilities in fully connected or self-attention layers. We establish that even when data are protected by LDP, privacy risks persist, depending on the privacy budget. Practical evaluations on federated vision models confirm considerable privacy risks, revealing that the noise required to mitigate these attacks significantly degrades models' utility. 

---
# SlimRAG: Retrieval without Graphs via Entity-Aware Context Selection 

**Authors**: Jiale Zhang, Jiaxiang Chen, Zhucong Li, Jie Ding, Kui Zhao, Zenglin Xu, Xin Pang, Yinghui Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.17288)  

**Abstract**: Retrieval-Augmented Generation (RAG) enhances language models by incorporating external knowledge at inference time. However, graph-based RAG systems often suffer from structural overhead and imprecise retrieval: they require costly pipelines for entity linking and relation extraction, yet frequently return subgraphs filled with loosely related or tangential content. This stems from a fundamental flaw -- semantic similarity does not imply semantic relevance. We introduce SlimRAG, a lightweight framework for retrieval without graphs. SlimRAG replaces structure-heavy components with a simple yet effective entity-aware mechanism. At indexing time, it constructs a compact entity-to-chunk table based on semantic embeddings. At query time, it identifies salient entities, retrieves and scores associated chunks, and assembles a concise, contextually relevant input -- without graph traversal or edge construction. To quantify retrieval efficiency, we propose Relative Index Token Utilization (RITU), a metric measuring the compactness of retrieved content. Experiments across multiple QA benchmarks show that SlimRAG outperforms strong flat and graph-based baselines in accuracy while reducing index size and RITU (e.g., 16.31 vs. 56+), highlighting the value of structure-free, entity-centric context selection. The code will be released soon. this https URL 

---
# GTA: Grouped-head latenT Attention 

**Authors**: Luoyang Sun, Jiwen Jiang, Cheng Deng, Xinjian Wu, Haifeng Zhang, Lei Chen, Lionel Ni, Jun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.17286)  

**Abstract**: Attention mechanisms underpin the success of large language models (LLMs), yet their substantial computational and memory overhead poses challenges for optimizing efficiency and performance. A critical bottleneck arises as KV cache and attention computations scale rapidly with text length, challenging deployment on hardware with limited computational and memory resources. We observe that attention mechanisms exhibit substantial redundancy, since the KV cache can be significantly compressed and attention maps across heads display high similarity, revealing that much of the computation and storage is unnecessary. Leveraging these insights, we propose \textbf{G}rouped-Head Laten\textbf{T} \textbf{A}ttention (GTA), a novel attention mechanism that reduces memory usage and computational complexity while maintaining performance. GTA comprises two components: (1) a shared attention map mechanism that reuses attention scores across multiple heads, decreasing the key cache size; and (2) a nonlinear value decoder with learned projections that compresses the value cache into a latent space, further cutting memory needs. GTA cuts attention computation FLOPs by up to \emph{62.5\%} versus Grouped-Query Attention and shrink the KV cache by up to \emph{70\%}, all while avoiding the extra overhead of Multi-Head Latent Attention to improve LLM deployment efficiency. Consequently, GTA models achieve a \emph{2x} increase in end-to-end inference speed, with prefill benefiting from reduced computational cost and decoding benefiting from the smaller cache footprint. 

---
# A Theoretical Framework for Virtual Power Plant Integration with Gigawatt-Scale AI Data Centers: Multi-Timescale Control and Stability Analysis 

**Authors**: Ali Peivandizadeh  

**Link**: [PDF](https://arxiv.org/pdf/2506.17284)  

**Abstract**: The explosive growth of artificial intelligence has created gigawatt-scale data centers that fundamentally challenge power system operation, exhibiting power fluctuations exceeding 500 MW within seconds and millisecond-scale variations of 50-75% of thermal design power. This paper presents a comprehensive theoretical framework that reconceptualizes Virtual Power Plants (VPPs) to accommodate these extreme dynamics through a four-layer hierarchical control architecture operating across timescales from 100 microseconds to 24 hours.
We develop control mechanisms and stability criteria specifically tailored to converter-dominated systems with pulsing megawatt-scale loads. We prove that traditional VPP architectures, designed for aggregating distributed resources with response times of seconds to minutes, cannot maintain stability when confronted with AI data center dynamics exhibiting slew rates exceeding 1,000 MW/s at gigawatt scale.
Our framework introduces: (1) a sub-millisecond control layer that interfaces with data center power electronics to actively dampen power oscillations; (2) new stability criteria incorporating protection system dynamics, demonstrating that critical clearing times reduce from 150 ms to 83 ms for gigawatt-scale pulsing loads; and (3) quantified flexibility characterization showing that workload deferability enables 30% peak reduction while maintaining AI service availability above 99.95%.
This work establishes the mathematical foundations necessary for the stable integration of AI infrastructure that will constitute 50-70% of data center electricity consumption by 2030. 

---
# CORONA: A Coarse-to-Fine Framework for Graph-based Recommendation with Large Language Models 

**Authors**: Junze Chen, Xinjie Yang, Cheng Yang, Junfei Bao, Zeyuan Guo, Yawen Li, Chuan Shi  

**Link**: [PDF](https://arxiv.org/pdf/2506.17281)  

**Abstract**: Recommender systems (RSs) are designed to retrieve candidate items a user might be interested in from a large pool. A common approach is using graph neural networks (GNNs) to capture high-order interaction relationships. As large language models (LLMs) have shown strong capabilities across domains, researchers are exploring their use to enhance recommendation. However, prior work limits LLMs to re-ranking results or dataset augmentation, failing to utilize their power during candidate filtering - which may lead to suboptimal performance. Instead, we propose to leverage LLMs' reasoning abilities during the candidate filtering process, and introduce Chain Of Retrieval ON grAphs (CORONA) to progressively narrow down the range of candidate items on interaction graphs with the help of LLMs: (1) First, LLM performs preference reasoning based on user profiles, with the response serving as a query to extract relevant users and items from the interaction graph as preference-assisted retrieval; (2) Then, using the information retrieved in the previous step along with the purchase history of target user, LLM conducts intent reasoning to help refine an even smaller interaction subgraph as intent-assisted retrieval; (3) Finally, we employ a GNN to capture high-order collaborative filtering information from the extracted subgraph, performing GNN-enhanced retrieval to generate the final recommendation results. The proposed framework leverages the reasoning capabilities of LLMs during the retrieval process, while seamlessly integrating GNNs to enhance overall recommendation performance. Extensive experiments on various datasets and settings demonstrate that our proposed CORONA achieves state-of-the-art performance with an 18.6% relative improvement in recall and an 18.4% relative improvement in NDCG on average. 

---
# Step-by-Step Reasoning Attack: Revealing 'Erased' Knowledge in Large Language Models 

**Authors**: Yash Sinha, Manit Baser, Murari Mandal, Dinil Mon Divakaran, Mohan Kankanhalli  

**Link**: [PDF](https://arxiv.org/pdf/2506.17279)  

**Abstract**: Knowledge erasure in large language models (LLMs) is important for ensuring compliance with data and AI regulations, safeguarding user privacy, mitigating bias, and misinformation. Existing unlearning methods aim to make the process of knowledge erasure more efficient and effective by removing specific knowledge while preserving overall model performance, especially for retained information. However, it has been observed that the unlearning techniques tend to suppress and leave the knowledge beneath the surface, thus making it retrievable with the right prompts. In this work, we demonstrate that \textit{step-by-step reasoning} can serve as a backdoor to recover this hidden information. We introduce a step-by-step reasoning-based black-box attack, Sleek, that systematically exposes unlearning failures. We employ a structured attack framework with three core components: (1) an adversarial prompt generation strategy leveraging step-by-step reasoning built from LLM-generated queries, (2) an attack mechanism that successfully recalls erased content, and exposes unfair suppression of knowledge intended for retention and (3) a categorization of prompts as direct, indirect, and implied, to identify which query types most effectively exploit unlearning weaknesses. Through extensive evaluations on four state-of-the-art unlearning techniques and two widely used LLMs, we show that existing approaches fail to ensure reliable knowledge removal. Of the generated adversarial prompts, 62.5% successfully retrieved forgotten Harry Potter facts from WHP-unlearned Llama, while 50% exposed unfair suppression of retained knowledge. Our work highlights the persistent risks of information leakage, emphasizing the need for more robust unlearning strategies for erasure. 

---
# Chunk Twice, Embed Once: A Systematic Study of Segmentation and Representation Trade-offs in Chemistry-Aware Retrieval-Augmented Generation 

**Authors**: Mahmoud Amiri, Thomas Bocklitz  

**Link**: [PDF](https://arxiv.org/pdf/2506.17277)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems are increasingly vital for navigating the ever-expanding body of scientific literature, particularly in high-stakes domains such as chemistry. Despite the promise of RAG, foundational design choices -- such as how documents are segmented and represented -- remain underexplored in domain-specific contexts. This study presents the first large-scale, systematic evaluation of chunking strategies and embedding models tailored to chemistry-focused RAG systems. We investigate 25 chunking configurations across five method families and evaluate 48 embedding models on three chemistry-specific benchmarks, including the newly introduced QuestChemRetrieval dataset. Our results reveal that recursive token-based chunking (specifically R100-0) consistently outperforms other approaches, offering strong performance with minimal resource overhead. We also find that retrieval-optimized embeddings -- such as Nomic and Intfloat E5 variants -- substantially outperform domain-specialized models like SciBERT. By releasing our datasets, evaluation framework, and empirical benchmarks, we provide actionable guidelines for building effective and efficient chemistry-aware RAG systems. 

---
# Modal Logic for Stratified Becoming: Actualization Beyond Possible Worlds 

**Authors**: Alexandre Le Nepvou  

**Link**: [PDF](https://arxiv.org/pdf/2506.17276)  

**Abstract**: This article develops a novel framework for modal logic based on the idea of stratified actualization, rather than the classical model of global possible worlds. Traditional Kripke semantics treat modal operators as quantification over fully determinate alternatives, neglecting the local, dynamic, and often asymmetric nature of actualization processes. We propose a system Stratified Actualization Logic (SAL) in which modalities are indexed by levels of ontological stability, interpreted as admissibility regimes. Each modality operates over a structured layer of possibility, grounded in the internal coherence of transitions between layers. We formally define the syntax and semantics of SAL, introduce its axioms, and prove soundness and completeness. Applications are discussed in connection with temporal becoming, quantum decoherence domains, and modal metaphysics. The result is a logic that captures the ontological structure of actualization without recourse to abstract possible worlds, offering a stratified alternative to standard modal realism. 

---
# Conformal Safety Shielding for Imperfect-Perception Agents 

**Authors**: William Scarbro, Calum Imrie, Sinem Getir Yaman, Kavan Fatehi, Corina S. Pasareanu, Radu Calinescu, Ravi Mangal  

**Link**: [PDF](https://arxiv.org/pdf/2506.17275)  

**Abstract**: We consider the problem of safe control in discrete autonomous agents that use learned components for imperfect perception (or more generally, state estimation) from high-dimensional observations. We propose a shield construction that provides run-time safety guarantees under perception errors by restricting the actions available to an agent, modeled as a Markov decision process, as a function of the state estimates. Our construction uses conformal prediction for the perception component, which guarantees that for each observation, the predicted set of estimates includes the actual state with a user-specified probability. The shield allows an action only if it is allowed for all the estimates in the predicted set, resulting in a local safety guarantee. We also articulate and prove a global safety property of existing shield constructions for perfect-perception agents bounding the probability of reaching unsafe states if the agent always chooses actions prescribed by the shield. We illustrate our approach with a case-study of an experimental autonomous system that guides airplanes on taxiways using high-dimensional perception DNNs. 

---
# QUST_NLP at SemEval-2025 Task 7: A Three-Stage Retrieval Framework for Monolingual and Crosslingual Fact-Checked Claim Retrieval 

**Authors**: Youzheng Liu, Jiyan Liu, Xiaoman Xu, Taihang Wang, Yimin Wang, Ye Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2506.17272)  

**Abstract**: This paper describes the participation of QUST_NLP in the SemEval-2025 Task 7. We propose a three-stage retrieval framework specifically designed for fact-checked claim retrieval. Initially, we evaluate the performance of several retrieval models and select the one that yields the best results for candidate retrieval. Next, we employ multiple re-ranking models to enhance the candidate results, with each model selecting the Top-10 outcomes. In the final stage, we utilize weighted voting to determine the final retrieval outcomes. Our approach achieved 5th place in the monolingual track and 7th place in the crosslingual track. We release our system code at: this https URL 

---
# CF-VLM:CounterFactual Vision-Language Fine-tuning 

**Authors**: Jusheng Zhang, Kaitong Cai, Yijia Fan, Jian Wang, Keze Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.17267)  

**Abstract**: Recent advances in vision-language models (VLMs) have greatly improved cross-modal semantic understanding, yet significant limitations remain in fine-grained discrimination and deep causal reasoning tasks. Existing VLMs often rely on superficial statistical correlations, lacking the ability to capture the underlying causal logic between visual and textual content. To address this, we propose CounterFactual Vision-Language Fine-tuning (CF-VLM), a novel framework that enhances the causal reasoning capabilities of VLMs through the targeted use of counterfactual samples. CF-VLM introduces three complementary training objectives: maintaining foundational cross-modal alignment, reinforcing the uniqueness and stability of factual scene representations against coherent counterfactuals, and sharpening the model's sensitivity to minimal but critical causal edits. Extensive experiments demonstrate that CF-VLM consistently outperforms strong baselines and state-of-the-art methods on compositional reasoning and generalization benchmarks. Furthermore, it shows promise in mitigating visual hallucinations, indicating improved factual consistency. Our CF-VLM provides a robust foundation for deploying VLMs in high-stakes, real-world scenarios requiring reliable reasoning and interpretability. 

---
# Does Multimodal Large Language Model Truly Unlearn? Stealthy MLLM Unlearning Attack 

**Authors**: Xianren Zhang, Hui Liu, Delvin Ce Zhang, Xianfeng Tang, Qi He, Dongwon Lee, Suhang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.17265)  

**Abstract**: Multimodal Large Language Models (MLLMs) trained on massive data may memorize sensitive personal information and photos, posing serious privacy risks. To mitigate this, MLLM unlearning methods are proposed, which fine-tune MLLMs to reduce the ``forget'' sensitive information. However, it remains unclear whether the knowledge has been truly forgotten or just hidden in the model. Therefore, we propose to study a novel problem of LLM unlearning attack, which aims to recover the unlearned knowledge of an unlearned LLM. To achieve the goal, we propose a novel framework Stealthy Unlearning Attack (SUA) framework that learns a universal noise pattern. When applied to input images, this noise can trigger the model to reveal unlearned content. While pixel-level perturbations may be visually subtle, they can be detected in the semantic embedding space, making such attacks vulnerable to potential defenses. To improve stealthiness, we introduce an embedding alignment loss that minimizes the difference between the perturbed and denoised image embeddings, ensuring the attack is semantically unnoticeable. Experimental results show that SUA can effectively recover unlearned information from MLLMs. Furthermore, the learned noise generalizes well: a single perturbation trained on a subset of samples can reveal forgotten content in unseen images. This indicates that knowledge reappearance is not an occasional failure, but a consistent behavior. 

---
# OAT-Rephrase: Optimization-Aware Training Data Rephrasing for Zeroth-Order LLM Fine-Tuning 

**Authors**: Jikai Long, Zijian Hu, Xiaodong Yu, Jianwen Xie, Zhaozhuo Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.17264)  

**Abstract**: Fine-tuning large language models (LLMs) using zeroth-order optimization (ZO) offers a memory-efficient alternative to gradient-based methods but suffers from slower convergence and unstable optimization due to noisy gradient estimates. This paper introduces OAT-Rephrase, an Optimization-Aware Training data rephrasing strategy that leverages an LLM to rephrase training instances based on its understanding of the ZO dynamics, specifically MeZO, derived directly from its paper. The approach incorporates a dual-stage pipeline featuring a rewriter LLM and a semantic judge, ensuring all rephrasings retain task relevance and logical consistency. Evaluations across five classification tasks and three LLM architectures demonstrate that OAT-Rephrase consistently improves MeZO fine-tuning performance, often narrowing or eliminating the gap with first-order methods. Our findings suggest that optimization-aware rephrasing serves as a reusable and low-overhead enhancement for zeroth-order tuning regimes. 

---
# Memory Allocation in Resource-Constrained Reinforcement Learning 

**Authors**: Massimiliano Tamborski, David Abel  

**Link**: [PDF](https://arxiv.org/pdf/2506.17263)  

**Abstract**: Resource constraints can fundamentally change both learning and decision-making. We explore how memory constraints influence an agent's performance when navigating unknown environments using standard reinforcement learning algorithms. Specifically, memory-constrained agents face a dilemma: how much of their limited memory should be allocated to each of the agent's internal processes, such as estimating a world model, as opposed to forming a plan using that model? We study this dilemma in MCTS- and DQN-based algorithms and examine how different allocations of memory impact performance in episodic and continual learning settings. 

---
# AI to Identify Strain-sensitive Regions of the Optic Nerve Head Linked to Functional Loss in Glaucoma 

**Authors**: Thanadet Chuangsuwanich, Monisha E. Nongpiur, Fabian A. Braeu, Tin A. Tun, Alexandre Thiery, Shamira Perera, Ching Lin Ho, Martin Buist, George Barbastathis, Tin Aung, Michaël J.A. Girard  

**Link**: [PDF](https://arxiv.org/pdf/2506.17262)  

**Abstract**: Objective: (1) To assess whether ONH biomechanics improves prediction of three progressive visual field loss patterns in glaucoma; (2) to use explainable AI to identify strain-sensitive ONH regions contributing to these predictions.
Methods: We recruited 237 glaucoma subjects. The ONH of one eye was imaged under two conditions: (1) primary gaze and (2) primary gaze with IOP elevated to ~35 mmHg via ophthalmo-dynamometry. Glaucoma experts classified the subjects into four categories based on the presence of specific visual field defects: (1) superior nasal step (N=26), (2) superior partial arcuate (N=62), (3) full superior hemifield defect (N=25), and (4) other/non-specific defects (N=124). Automatic ONH tissue segmentation and digital volume correlation were used to compute IOP-induced neural tissue and lamina cribrosa (LC) strains. Biomechanical and structural features were input to a Geometric Deep Learning model. Three classification tasks were performed to detect: (1) superior nasal step, (2) superior partial arcuate, (3) full superior hemifield defect. For each task, the data were split into 80% training and 20% testing sets. Area under the curve (AUC) was used to assess performance. Explainable AI techniques were employed to highlight the ONH regions most critical to each classification.
Results: Models achieved high AUCs of 0.77-0.88, showing that ONH strain improved VF loss prediction beyond morphology alone. The inferior and inferotemporal rim were identified as key strain-sensitive regions, contributing most to visual field loss prediction and showing progressive expansion with increasing disease severity.
Conclusion and Relevance: ONH strain enhances prediction of glaucomatous VF loss patterns. Neuroretinal rim, rather than the LC, was the most critical region contributing to model predictions. 

---
# A Digital Twin Framework for Generation-IV Reactors with Reinforcement Learning-Enabled Health-Aware Supervisory Control 

**Authors**: Jasmin Y. Lim, Dimitrios Pylorof, Humberto E. Garcia, Karthik Duraisamy  

**Link**: [PDF](https://arxiv.org/pdf/2506.17258)  

**Abstract**: Generation IV (Gen-IV) nuclear power plants are envisioned to replace the current reactor fleet, bringing improvements in performance, safety, reliability, and sustainability. However, large cost investments currently inhibit the deployment of these advanced reactor concepts. Digital twins bridge real-world systems with digital tools to reduce costs, enhance decision-making, and boost operational efficiency. In this work, a digital twin framework is designed to operate the Gen-IV Fluoride-salt-cooled High-temperature Reactor, utilizing data-enhanced methods to optimize operational and maintenance policies while adhering to system constraints. The closed-loop framework integrates surrogate modeling, reinforcement learning, and Bayesian inference to streamline end-to-end communication for online regulation and self-adjustment. Reinforcement learning is used to consider component health and degradation to drive the target power generations, with constraints enforced through a Reference Governor control algorithm that ensures compliance with pump flow rate and temperature limits. These input driving modules benefit from detailed online simulations that are assimilated to measurement data with Bayesian filtering. The digital twin is demonstrated in three case studies: a one-year long-term operational period showcasing maintenance planning capabilities, short-term accuracy refinement with high-frequency measurements, and system shock capturing that demonstrates real-time recalibration capabilities when change in boundary conditions. These demonstrations validate robustness for health-aware and constraint-informed nuclear plant operation, with general applicability to other advanced reactor concepts and complex engineering systems. 

---
# UltraSketchLLM: Saliency-Driven Sketching for Ultra-Low Bit LLM Compression 

**Authors**: Sunan Zou, Ziyun Zhang, Xueting Sun, Guojie Luo  

**Link**: [PDF](https://arxiv.org/pdf/2506.17255)  

**Abstract**: The rapid growth of large language models (LLMs) has outpaced the memory constraints of edge devices, necessitating extreme weight compression beyond the 1-bit limit. While quantization reduces model size, it is fundamentally limited to 1 bit per weight. Existing multiple-to-one compression methods either rely on mapping tables (inducing memory overhead) or incur severe accuracy degradation due to random weight grouping. We introduce UltraSketchLLM, an index-free, sketch-based framework that achieves ultra-low bit compression (down to 0.5 bits per weight) while preserving model performance. UltraSketchLLM leverages data sketching, a sub-linear representation technique from streaming applications, to map multiple weights to single values with bounded error. Our approach integrates an underestimate AbsMaxMin sketch to minimize relative errors for small weights, importance-aware space allocation to prioritize salient weights, and a straight-through estimator for compression-aware finetuning. Experiments on Llama-3.2-1B demonstrate up to 0.5-bit compression with competitive perplexity, alongside tolerable latency overhead. UltraSketchLLM offers a practical solution for deploying LLMs in resource-constrained environments. 

---
# Keeping Up with the Models: Online Deployment and Routing of LLMs at Scale 

**Authors**: Shaoang Li, Jian Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.17254)  

**Abstract**: The rapid pace at which new large language models (LLMs) appear -- and older ones become obsolete -- forces LLM service providers to juggle a streaming inventory of models while respecting tight deployment capacity and per-query cost budgets. We cast the reality as an online decision problem that couples stage-wise deployment, made at fixed maintenance windows, with per-query routing among the models kept live. We introduce StageRoute, a hierarchical algorithm that (i) optimistically selects up to $M_max$ models for the next stage using reward upper-confidence and cost lower-confidence bounds, then (ii) solves a budget-constrained bandit sub-problem to route each incoming query. We prove that StageRoute achieves a regret of order $T^{2/3}$ and provide a matching lower bound, thereby establishing its near-optimality. Moreover, our experiments confirm the theory, demonstrating that StageRoute performs close to the optimum in practical settings. 

---
# MS-TVNet:A Long-Term Time Series Prediction Method Based on Multi-Scale Dynamic Convolution 

**Authors**: Chenghan Li, Mingchen Li, Yipu Liao, Ruisheng Diao  

**Link**: [PDF](https://arxiv.org/pdf/2506.17253)  

**Abstract**: Long-term time series prediction has predominantly relied on Transformer and MLP models, while the potential of convolutional networks in this domain remains underexplored. To address this gap, we introduce a novel multi-scale time series reshape module, which effectively captures the relationships among multi-period patches and variable dependencies. Building upon this module, we propose MS-TVNet, a multi-scale 3D dynamic convolutional neural network. Through comprehensive evaluations on diverse datasets, MS-TVNet demonstrates superior performance compared to baseline models, achieving state-of-the-art (SOTA) results in long-term time series prediction. Our findings highlight the effectiveness of leveraging convolutional networks for capturing complex temporal patterns, suggesting a promising direction for future research in this this http URL code is realsed on this https URL. 

---
# Adaptive Sample Scheduling for Direct Preference Optimization 

**Authors**: Zixuan Huang, Yikun Ban, Lean Fu, Xiaojie Li, Zhongxiang Dai, Jianxin Li, Deqing Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.17252)  

**Abstract**: Direct Preference Optimization (DPO) has emerged as an effective approach for aligning large language models (LLMs) with human preferences. However, its performance is highly dependent on the quality of the underlying human preference data. To address this bottleneck, prior work has explored various data selection strategies, but these methods often overlook the impact of the evolving states of the language model during the DPO process. %including active querying, response pair selection, and data pre-selection. In this paper, we introduce a novel problem: Sample Scheduling for DPO, which aims to dynamically and adaptively schedule training samples based on the model's evolving states throughout preference optimization. To solve this problem, we propose SamS, an efficient and effective algorithm that adaptively selects samples in each training batch based on the LLM's learning feedback to maximize the potential generalization performance. Notably, without modifying the core DPO algorithm, simply integrating SamS significantly improves performance across tasks, with minimal additional computational overhead. This work points to a promising new direction for improving LLM alignment through more effective utilization of fixed preference datasets. 

---
# Training-free LLM Verification via Recycling Few-shot Examples 

**Authors**: Dongseok Lee, Jimyung Hong, Dongyoung Kim, Jaehyung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.17251)  

**Abstract**: Although LLMs have achieved remarkable performance, the inherent stochasticity of their reasoning process and varying conclusions present significant challenges. Majority voting or Best-of-N with external verification models has been explored to find the most promising solution among multiple LLM outputs. However, these approaches have certain limitations, such as limited applicability or the cost of an additional training step. To address this problem, we propose a novel and effective framework that Recycles Few-shot examples to verify LLM outputs (Referi). Our key idea is to additionally utilize the given few-shot examples to evaluate the candidate outputs of the target query, not only using them to generate outputs as the conventional few-shot prompting setup. Specifically, Referi evaluates the generated outputs by combining two different scores, designed motivated from Bayes' rule, and subsequently selects the candidate that is both confidently determined and contextually coherent through a few additional LLM inferences. Experiments with three different LLMs and across seven diverse tasks demonstrate that our framework significantly improves the accuracy of LLMs-achieving an average gain of 4.8%-through effective response selection, without additional training. 

---
# Towards Interpretable Adversarial Examples via Sparse Adversarial Attack 

**Authors**: Fudong Lin, Jiadong Lou, Hao Wang, Brian Jalaian, Xu Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2506.17250)  

**Abstract**: Sparse attacks are to optimize the magnitude of adversarial perturbations for fooling deep neural networks (DNNs) involving only a few perturbed pixels (i.e., under the l0 constraint), suitable for interpreting the vulnerability of DNNs. However, existing solutions fail to yield interpretable adversarial examples due to their poor sparsity. Worse still, they often struggle with heavy computational overhead, poor transferability, and weak attack strength. In this paper, we aim to develop a sparse attack for understanding the vulnerability of CNNs by minimizing the magnitude of initial perturbations under the l0 constraint, to overcome the existing drawbacks while achieving a fast, transferable, and strong attack to DNNs. In particular, a novel and theoretical sound parameterization technique is introduced to approximate the NP-hard l0 optimization problem, making directly optimizing sparse perturbations computationally feasible. Besides, a novel loss function is designed to augment initial perturbations by maximizing the adversary property and minimizing the number of perturbed pixels simultaneously. Extensive experiments are conducted to demonstrate that our approach, with theoretical performance guarantees, outperforms state-of-the-art sparse attacks in terms of computational overhead, transferability, and attack strength, expecting to serve as a benchmark for evaluating the robustness of DNNs. In addition, theoretical and empirical results validate that our approach yields sparser adversarial examples, empowering us to discover two categories of noises, i.e., "obscuring noise" and "leading noise", which will help interpret how adversarial perturbation misleads the classifiers into incorrect predictions. Our code is available at this https URL. 

---
# Improving Prediction Certainty Estimation for Reliable Early Exiting via Null Space Projection 

**Authors**: Jianing He, Qi Zhang, Duoqian Miao, Yi Kun, Shufeng Hao, Hongyun Zhang, Zhihua Wei  

**Link**: [PDF](https://arxiv.org/pdf/2506.17249)  

**Abstract**: Early exiting has demonstrated great potential in accelerating the inference of pre-trained language models (PLMs) by enabling easy samples to exit at shallow layers, eliminating the need for executing deeper layers. However, existing early exiting methods primarily rely on class-relevant logits to formulate their exiting signals for estimating prediction certainty, neglecting the detrimental influence of class-irrelevant information in the features on prediction certainty. This leads to an overestimation of prediction certainty, causing premature exiting of samples with incorrect early predictions. To remedy this, we define an NSP score to estimate prediction certainty by considering the proportion of class-irrelevant information in the features. On this basis, we propose a novel early exiting method based on the Certainty-Aware Probability (CAP) score, which integrates insights from both logits and the NSP score to enhance prediction certainty estimation, thus enabling more reliable exiting decisions. The experimental results on the GLUE benchmark show that our method can achieve an average speed-up ratio of 2.19x across all tasks with negligible performance degradation, surpassing the state-of-the-art (SOTA) ConsistentEE by 28%, yielding a better trade-off between task performance and inference efficiency. The code is available at this https URL. 

---
# Efficient Quantification of Multimodal Interaction at Sample Level 

**Authors**: Zequn Yang, Hongfa Wang, Di Hu  

**Link**: [PDF](https://arxiv.org/pdf/2506.17248)  

**Abstract**: Interactions between modalities -- redundancy, uniqueness, and synergy -- collectively determine the composition of multimodal information. Understanding these interactions is crucial for analyzing information dynamics in multimodal systems, yet their accurate sample-level quantification presents significant theoretical and computational challenges. To address this, we introduce the Lightweight Sample-wise Multimodal Interaction (LSMI) estimator, rigorously grounded in pointwise information theory. We first develop a redundancy estimation framework, employing an appropriate pointwise information measure to quantify this most decomposable and measurable interaction. Building upon this, we propose a general interaction estimation method that employs efficient entropy estimation, specifically tailored for sample-wise estimation in continuous distributions. Extensive experiments on synthetic and real-world datasets validate LSMI's precision and efficiency. Crucially, our sample-wise approach reveals fine-grained sample- and category-level dynamics within multimodal data, enabling practical applications such as redundancy-informed sample partitioning, targeted knowledge distillation, and interaction-aware model ensembling. The code is available at this https URL. 

---
# Recursive Learning-Based Virtual Buffering for Analytical Global Placement 

**Authors**: Andrew B. Kahng, Yiting Liu, Zhiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.17247)  

**Abstract**: Due to the skewed scaling of interconnect versus cell delay in modern technology nodes, placement with buffer porosity (i.e., cell density) awareness is essential for timing closure in physical synthesis flows. However, existing approaches face two key challenges: (i) traditional van Ginneken-Lillis-style buffering approaches are computationally expensive during global placement; and (ii) machine learning-based approaches, such as BufFormer, lack a thorough consideration of Electrical Rule Check (ERC) violations and fail to "close the loop" back into the physical design flow. In this work, we propose MLBuf-RePlAce, the first open-source learning-driven virtual buffering-aware analytical global placement framework, built on top of the OpenROAD infrastructure. MLBuf-RePlAce adopts an efficient recursive learning-based generative buffering approach to predict buffer types and locations, addressing ERC violations during global placement. We compare MLBuf-RePlAce against the default virtual buffering-based timing-driven global placer in OpenROAD, using open-source testcases from the TILOS MacroPlacement and OpenROAD-flow-scripts repositories. Without degradation of post-route power, MLBuf-RePlAce achieves (maximum, average) improvements of (56%, 31%) in total negative slack (TNS) within the open-source OpenROAD flow. When evaluated by completion in a commercial flow, MLBuf-RePlAce achieves (maximum, average) improvements of (53%, 28%) in TNS with an average of 0.2% improvement in post-route power. 

---
# Graph Neural Networks in Multi-Omics Cancer Research: A Structured Survey 

**Authors**: Payam Zohari, Mostafa Haghir Chehreghani  

**Link**: [PDF](https://arxiv.org/pdf/2506.17234)  

**Abstract**: The task of data integration for multi-omics data has emerged as a powerful strategy to unravel the complex biological underpinnings of cancer. Recent advancements in graph neural networks (GNNs) offer an effective framework to model heterogeneous and structured omics data, enabling precise representation of molecular interactions and regulatory networks. This systematic review explores several recent studies that leverage GNN-based architectures in multi-omics cancer research. We classify the approaches based on their targeted omics layers, graph neural network structures, and biological tasks such as subtype classification, prognosis prediction, and biomarker discovery. The analysis reveals a growing trend toward hybrid and interpretable models, alongside increasing adoption of attention mechanisms and contrastive learning. Furthermore, we highlight the use of patient-specific graphs and knowledge-driven priors as emerging directions. This survey serves as a comprehensive resource for researchers aiming to design effective GNN-based pipelines for integrative cancer analysis, offering insights into current practices, limitations, and potential future directions. 

---
# PCaM: A Progressive Focus Attention-Based Information Fusion Method for Improving Vision Transformer Domain Adaptation 

**Authors**: Zelin Zang, Fei Wang, Liangyu Li, Jinlin Wu, Chunshui Zhao, Zhen Lei, Baigui Sun  

**Link**: [PDF](https://arxiv.org/pdf/2506.17232)  

**Abstract**: Unsupervised Domain Adaptation (UDA) aims to transfer knowledge from a labeled source domain to an unlabeled target domain. Recent UDA methods based on Vision Transformers (ViTs) have achieved strong performance through attention-based feature alignment. However, we identify a key limitation: foreground object mismatch, where the discrepancy in foreground object size and spatial distribution across domains weakens attention consistency and hampers effective domain alignment. To address this issue, we propose the Progressive Focus Cross-Attention Mechanism (PCaM), which progressively filters out background information during cross-attention, allowing the model to focus on and fuse discriminative foreground semantics across domains. We further introduce an attentional guidance loss that explicitly directs attention toward task-relevant regions, enhancing cross-domain attention consistency. PCaM is lightweight, architecture-agnostic, and easy to integrate into existing ViT-based UDA pipelines. Extensive experiments on Office-Home, DomainNet, VisDA-2017, and remote sensing datasets demonstrate that PCaM significantly improves adaptation performance and achieves new state-of-the-art results, validating the effectiveness of attention-guided foreground fusion for domain adaptation. 

---
# MMET: A Multi-Input and Multi-Scale Transformer for Efficient PDEs Solving 

**Authors**: Yichen Luo, Jia Wang, Dapeng Lan, Yu Liu, Zhibo Pang  

**Link**: [PDF](https://arxiv.org/pdf/2506.17230)  

**Abstract**: Partial Differential Equations (PDEs) are fundamental for modeling physical systems, yet solving them in a generic and efficient manner using machine learning-based approaches remains challenging due to limited multi-input and multi-scale generalization capabilities, as well as high computational costs. This paper proposes the Multi-input and Multi-scale Efficient Transformer (MMET), a novel framework designed to address the above challenges. MMET decouples mesh and query points as two sequences and feeds them into the encoder and decoder, respectively, and uses a Gated Condition Embedding (GCE) layer to embed input variables or functions with varying dimensions, enabling effective solutions for multi-scale and multi-input problems. Additionally, a Hilbert curve-based reserialization and patch embedding mechanism decrease the input length. This significantly reduces the computational cost when dealing with large-scale geometric models. These innovations enable efficient representations and support multi-scale resolution queries for large-scale and multi-input PDE problems. Experimental evaluations on diverse benchmarks spanning different physical fields demonstrate that MMET outperforms SOTA methods in both accuracy and computational efficiency. This work highlights the potential of MMET as a robust and scalable solution for real-time PDE solving in engineering and physics-based applications, paving the way for future explorations into pre-trained large-scale models in specific domains. This work is open-sourced at this https URL. 

---
