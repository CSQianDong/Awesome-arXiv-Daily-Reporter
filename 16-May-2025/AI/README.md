# Towards a Deeper Understanding of Reasoning Capabilities in Large Language Models 

**Authors**: Annie Wong, Thomas Bäck, Aske Plaat, Niki van Stein, Anna V. Kononova  

**Link**: [PDF](https://arxiv.org/pdf/2505.10543)  

**Abstract**: While large language models demonstrate impressive performance on static benchmarks, the true potential of large language models as self-learning and reasoning agents in dynamic environments remains unclear. This study systematically evaluates the efficacy of self-reflection, heuristic mutation, and planning as prompting techniques to test the adaptive capabilities of agents. We conduct experiments with various open-source language models in dynamic environments and find that larger models generally outperform smaller ones, but that strategic prompting can close this performance gap. Second, a too-long prompt can negatively impact smaller models on basic reactive tasks, while larger models show more robust behaviour. Third, advanced prompting techniques primarily benefit smaller models on complex games, but offer less improvement for already high-performing large language models. Yet, we find that advanced reasoning methods yield highly variable outcomes: while capable of significantly improving performance when reasoning and decision-making align, they also introduce instability and can lead to big performance drops. Compared to human performance, our findings reveal little evidence of true emergent reasoning. Instead, large language model performance exhibits persistent limitations in crucial areas such as planning, reasoning, and spatial coordination, suggesting that current-generation large language models still suffer fundamental shortcomings that may not be fully overcome through self-reflective prompting alone. Reasoning is a multi-faceted task, and while reasoning methods like Chain of thought improves multi-step reasoning on math word problems, our findings using dynamic benchmarks highlight important shortcomings in general reasoning capabilities, indicating a need to move beyond static benchmarks to capture the complexity of reasoning. 

---
# AI Agents vs. Agentic AI: A Conceptual Taxonomy, Applications and Challenge 

**Authors**: Ranjan Sapkota, Konstantinos I. Roumeliotis, Manoj Karkee  

**Link**: [PDF](https://arxiv.org/pdf/2505.10468)  

**Abstract**: This study critically distinguishes between AI Agents and Agentic AI, offering a structured conceptual taxonomy, application mapping, and challenge analysis to clarify their divergent design philosophies and capabilities. We begin by outlining the search strategy and foundational definitions, characterizing AI Agents as modular systems driven by Large Language Models (LLMs) and Large Image Models (LIMs) for narrow, task-specific automation. Generative AI is positioned as a precursor, with AI Agents advancing through tool integration, prompt engineering, and reasoning enhancements. In contrast, Agentic AI systems represent a paradigmatic shift marked by multi-agent collaboration, dynamic task decomposition, persistent memory, and orchestrated autonomy. Through a sequential evaluation of architectural evolution, operational mechanisms, interaction styles, and autonomy levels, we present a comparative analysis across both paradigms. Application domains such as customer support, scheduling, and data summarization are contrasted with Agentic AI deployments in research automation, robotic coordination, and medical decision support. We further examine unique challenges in each paradigm including hallucination, brittleness, emergent behavior, and coordination failure and propose targeted solutions such as ReAct loops, RAG, orchestration layers, and causal modeling. This work aims to provide a definitive roadmap for developing robust, scalable, and explainable AI agent and Agentic AI-driven systems. >AI Agents, Agent-driven, Vision-Language-Models, Agentic AI Decision Support System, Agentic-AI Applications 

---
# Evaluating Model Explanations without Ground Truth 

**Authors**: Kaivalya Rawal, Zihao Fu, Eoin Delaney, Chris Russell  

**Link**: [PDF](https://arxiv.org/pdf/2505.10399)  

**Abstract**: There can be many competing and contradictory explanations for a single model prediction, making it difficult to select which one to use. Current explanation evaluation frameworks measure quality by comparing against ideal "ground-truth" explanations, or by verifying model sensitivity to important inputs. We outline the limitations of these approaches, and propose three desirable principles to ground the future development of explanation evaluation strategies for local feature importance explanations. We propose a ground-truth Agnostic eXplanation Evaluation framework (AXE) for evaluating and comparing model explanations that satisfies these principles. Unlike prior approaches, AXE does not require access to ideal ground-truth explanations for comparison, or rely on model sensitivity - providing an independent measure of explanation quality. We verify AXE by comparing with baselines, and show how it can be used to detect explanation fairwashing. Our code is available at this https URL. 

---
# Plasticity as the Mirror of Empowerment 

**Authors**: David Abel, Michael Bowling, André Barreto, Will Dabney, Shi Dong, Steven Hansen, Anna Harutyunyan, Khimya Khetarpal, Clare Lyle, Razvan Pascanu, Georgios Piliouras, Doina Precup, Jonathan Richens, Mark Rowland, Tom Schaul, Satinder Singh  

**Link**: [PDF](https://arxiv.org/pdf/2505.10361)  

**Abstract**: Agents are minimally entities that are influenced by their past observations and act to influence future observations. This latter capacity is captured by empowerment, which has served as a vital framing concept across artificial intelligence and cognitive science. This former capacity, however, is equally foundational: In what ways, and to what extent, can an agent be influenced by what it observes? In this paper, we ground this concept in a universal agent-centric measure that we refer to as plasticity, and reveal a fundamental connection to empowerment. Following a set of desiderata on a suitable definition, we define plasticity using a new information-theoretic quantity we call the generalized directed information. We show that this new quantity strictly generalizes the directed information introduced by Massey (1990) while preserving all of its desirable properties. Our first finding is that plasticity is the mirror of empowerment: The agent's plasticity is identical to the empowerment of the environment, and vice versa. Our second finding establishes a tension between the plasticity and empowerment of an agent, suggesting that agent design needs to be mindful of both characteristics. We explore the implications of these findings, and suggest that plasticity, empowerment, and their relationship are essential to understanding agency. 

---
# A Comparative Study of SMT and MILP for the Nurse Rostering Problem 

**Authors**: Alvin Combrink, Stephie Do, Kristofer Bengtsson, Sabino Francesco Roselli, Martin Fabian  

**Link**: [PDF](https://arxiv.org/pdf/2505.10328)  

**Abstract**: The effects of personnel scheduling on the quality of care and working conditions for healthcare personnel have been thoroughly documented. However, the ever-present demand and large variation of constraints make healthcare scheduling particularly challenging. This problem has been studied for decades, with limited research aimed at applying Satisfiability Modulo Theories (SMT). SMT has gained momentum within the formal verification community in the last decades, leading to the advancement of SMT solvers that have been shown to outperform standard mathematical programming techniques.
In this work, we propose generic constraint formulations that can model a wide range of real-world scheduling constraints. Then, the generic constraints are formulated as SMT and MILP problems and used to compare the respective state-of-the-art solvers, Z3 and Gurobi, on academic and real-world inspired rostering problems. Experimental results show how each solver excels for certain types of problems; the MILP solver generally performs better when the problem is highly constrained or infeasible, while the SMT solver performs better otherwise. On real-world inspired problems containing a more varied set of shifts and personnel, the SMT solver excels. Additionally, it was noted during experimentation that the SMT solver was more sensitive to the way the generic constraints were formulated, requiring careful consideration and experimentation to achieve better performance. We conclude that SMT-based methods present a promising avenue for future research within the domain of personnel scheduling. 

---
# Empirically evaluating commonsense intelligence in large language models with large-scale human judgments 

**Authors**: Tuan Dung Nguyen, Duncan J. Watts, Mark E. Whiting  

**Link**: [PDF](https://arxiv.org/pdf/2505.10309)  

**Abstract**: Commonsense intelligence in machines is often assessed by static benchmarks that compare a model's output against human-prescribed correct labels. An important, albeit implicit, assumption of these labels is that they accurately capture what any human would think, effectively treating human common sense as homogeneous. However, recent empirical work has shown that humans vary enormously in what they consider commonsensical; thus what appears self-evident to one benchmark designer may not be so to another. Here, we propose a novel method for evaluating common sense in artificial intelligence (AI), specifically in large language models (LLMs), that incorporates empirically observed heterogeneity among humans by measuring the correspondence between a model's judgment and that of a human population. We first find that, when treated as independent survey respondents, most LLMs remain below the human median in their individual commonsense competence. Second, when used as simulators of a hypothetical population, LLMs correlate with real humans only modestly in the extent to which they agree on the same set of statements. In both cases, smaller, open-weight models are surprisingly more competitive than larger, proprietary frontier models. Our evaluation framework, which ties commonsense intelligence to its cultural basis, contributes to the growing call for adapting AI models to human collectivities that possess different, often incompatible, social stocks of knowledge. 

---
# MASS: Multi-Agent Simulation Scaling for Portfolio Construction 

**Authors**: Taian Guo, Haiyang Shen, Jinsheng Huang, Zhengyang Mao, Junyu Luo, Zhuoru Chen, Xuhui Liu, Bingyu Xia, Luchen Liu, Yun Ma, Ming Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.10278)  

**Abstract**: LLM-based multi-agent has gained significant attention for their potential in simulation and enhancing performance. However, existing works are limited to pure simulations or are constrained by predefined workflows, restricting their applicability and effectiveness. In this paper, we introduce the Multi-Agent Scaling Simulation (MASS) for portfolio construction. MASS achieves stable and continuous excess returns by progressively increasing the number of agents for large-scale simulations to gain a superior understanding of the market and optimizing agent distribution end-to-end through a reverse optimization process, rather than relying on a fixed workflow. We demonstrate its superiority through performance experiments, ablation studies, backtesting experiments, experiments on updated data and stock pools, scaling experiments, parameter sensitivity experiments, and visualization experiments, conducted in comparison with 6 state-of-the-art baselines on 3 challenging A-share stock pools. We expect the paradigm established by MASS to expand to other tasks with similar characteristics. The implementation of MASS has been open-sourced at this https URL. 

---
# A User Study Evaluating Argumentative Explanations in Diagnostic Decision Support 

**Authors**: Felix Liedeker, Olivia Sanchez-Graillet, Moana Seidler, Christian Brandt, Jörg Wellmer, Philipp Cimiano  

**Link**: [PDF](https://arxiv.org/pdf/2505.10188)  

**Abstract**: As the field of healthcare increasingly adopts artificial intelligence, it becomes important to understand which types of explanations increase transparency and empower users to develop confidence and trust in the predictions made by machine learning (ML) systems. In shared decision-making scenarios where doctors cooperate with ML systems to reach an appropriate decision, establishing mutual trust is crucial. In this paper, we explore different approaches to generating explanations in eXplainable AI (XAI) and make their underlying arguments explicit so that they can be evaluated by medical experts. In particular, we present the findings of a user study conducted with physicians to investigate their perceptions of various types of AI-generated explanations in the context of diagnostic decision support. The study aims to identify the most effective and useful explanations that enhance the diagnostic process. In the study, medical doctors filled out a survey to assess different types of explanations. Further, an interview was carried out post-survey to gain qualitative insights on the requirements of explanations incorporated in diagnostic decision support. Overall, the insights gained from this study contribute to understanding the types of explanations that are most effective. 

---
# From Text to Network: Constructing a Knowledge Graph of Taiwan-Based China Studies Using Generative AI 

**Authors**: Hsuan-Lei Shao  

**Link**: [PDF](https://arxiv.org/pdf/2505.10093)  

**Abstract**: Taiwanese China Studies (CS) has developed into a rich, interdisciplinary research field shaped by the unique geopolitical position and long standing academic engagement with Mainland China. This study responds to the growing need to systematically revisit and reorganize decades of Taiwan based CS scholarship by proposing an AI assisted approach that transforms unstructured academic texts into structured, interactive knowledge representations. We apply generative AI (GAI) techniques and large language models (LLMs) to extract and standardize entity relation triples from 1,367 peer reviewed CS articles published between 1996 and 2019. These triples are then visualized through a lightweight this http URL based system, forming the foundation of a domain specific knowledge graph and vector database for the field. This infrastructure allows users to explore conceptual nodes and semantic relationships across the corpus, revealing previously uncharted intellectual trajectories, thematic clusters, and research gaps. By decomposing textual content into graph structured knowledge units, our system enables a paradigm shift from linear text consumption to network based knowledge navigation. In doing so, it enhances scholarly access to CS literature while offering a scalable, data driven alternative to traditional ontology construction. This work not only demonstrates how generative AI can augment area studies and digital humanities but also highlights its potential to support a reimagined scholarly infrastructure for regional knowledge systems. 

---
# Leveraging Graph Retrieval-Augmented Generation to Support Learners' Understanding of Knowledge Concepts in MOOCs 

**Authors**: Mohamed Abdelmagied, Mohamed Amine Chatti, Shoeb Joarder, Qurat Ul Ain, Rawaa Alatrash  

**Link**: [PDF](https://arxiv.org/pdf/2505.10074)  

**Abstract**: Massive Open Online Courses (MOOCs) lack direct interaction between learners and instructors, making it challenging for learners to understand new knowledge concepts. Recently, learners have increasingly used Large Language Models (LLMs) to support them in acquiring new knowledge. However, LLMs are prone to hallucinations which limits their reliability. Retrieval-Augmented Generation (RAG) addresses this issue by retrieving relevant documents before generating a response. However, the application of RAG across different MOOCs is limited by unstructured learning material. Furthermore, current RAG systems do not actively guide learners toward their learning needs. To address these challenges, we propose a Graph RAG pipeline that leverages Educational Knowledge Graphs (EduKGs) and Personal Knowledge Graphs (PKGs) to guide learners to understand knowledge concepts in the MOOC platform CourseMapper. Specifically, we implement (1) a PKG-based Question Generation method to recommend personalized questions for learners in context, and (2) an EduKG-based Question Answering method that leverages the relationships between knowledge concepts in the EduKG to answer learner selected questions. To evaluate both methods, we conducted a study with 3 expert instructors on 3 different MOOCs in the MOOC platform CourseMapper. The results of the evaluation show the potential of Graph RAG to empower learners to understand new knowledge concepts in a personalized learning experience. 

---
# The First MPDD Challenge: Multimodal Personality-aware Depression Detection 

**Authors**: Changzeng Fu, Zelin Fu, Xinhe Kuang, Jiacheng Dong, Qi Zhang, Kaifeng Su, Yikai Su, Wenbo Shi, Junfeng Yao, Yuliang Zhao, Shiqi Zhao, Jiadong Wang, Siyang Song, Chaoran Liu, Yuichiro Yoshikawa, Björn Schuller, Hiroshi Ishiguro  

**Link**: [PDF](https://arxiv.org/pdf/2505.10034)  

**Abstract**: Depression is a widespread mental health issue affecting diverse age groups, with notable prevalence among college students and the elderly. However, existing datasets and detection methods primarily focus on young adults, neglecting the broader age spectrum and individual differences that influence depression manifestation. Current approaches often establish a direct mapping between multimodal data and depression indicators, failing to capture the complexity and diversity of depression across individuals. This challenge includes two tracks based on age-specific subsets: Track 1 uses the MPDD-Elderly dataset for detecting depression in older adults, and Track 2 uses the MPDD-Young dataset for detecting depression in younger participants. The Multimodal Personality-aware Depression Detection (MPDD) Challenge aims to address this gap by incorporating multimodal data alongside individual difference factors. We provide a baseline model that fuses audio and video modalities with individual difference information to detect depression manifestations in diverse populations. This challenge aims to promote the development of more personalized and accurate de pression detection methods, advancing mental health research and fostering inclusive detection systems. More details are available on the official challenge website: this https URL. 

---
# Pre-Act: Multi-Step Planning and Reasoning Improves Acting in LLM Agents 

**Authors**: Mrinal Rawat, Ambuje Gupta, Rushil Goomer, Alessandro Di Bari, Neha Gupta, Roberto Pieraccini  

**Link**: [PDF](https://arxiv.org/pdf/2505.09970)  

**Abstract**: The ReAct (Reasoning + Action) capability in large language models (LLMs) has become the foundation of modern agentic systems. Recent LLMs, such as DeepSeek-R1 and OpenAI o1/o3, exemplify this by emphasizing reasoning through the generation of ample intermediate tokens, which help build a strong premise before producing the final output tokens. In this paper, we introduce Pre-Act, a novel approach that enhances the agent's performance by creating a multi-step execution plan along with the detailed reasoning for the given user input. This plan incrementally incorporates previous steps and tool outputs, refining itself after each step execution until the final response is obtained. Our approach is applicable to both conversational and non-conversational agents. To measure the performance of task-oriented agents comprehensively, we propose a two-level evaluation framework: (1) turn level and (2) end-to-end. Our turn-level evaluation, averaged across five models, shows that our approach, Pre-Act, outperforms ReAct by 70% in Action Recall on the Almita dataset. While this approach is effective for larger models, smaller models crucial for practical applications, where latency and cost are key constraints, often struggle with complex reasoning tasks required for agentic systems. To address this limitation, we fine-tune relatively small models such as Llama 3.1 (8B & 70B) using the proposed Pre-Act approach. Our experiments show that the fine-tuned 70B model outperforms GPT-4, achieving a 69.5% improvement in action accuracy (turn-level) and a 28% improvement in goal completion rate (end-to-end) on the Almita (out-of-domain) dataset. 

---
# Demystifying AI Agents: The Final Generation of Intelligence 

**Authors**: Kevin J McNamara, Rhea Pritham Marpu  

**Link**: [PDF](https://arxiv.org/pdf/2505.09932)  

**Abstract**: The trajectory of artificial intelligence (AI) has been one of relentless acceleration, evolving from rudimentary rule-based systems to sophisticated, autonomous agents capable of complex reasoning and interaction. This whitepaper chronicles this remarkable journey, charting the key technological milestones--advancements in prompting, training methodologies, hardware capabilities, and architectural innovations--that have converged to create the AI agents of today. We argue that these agents, exemplified by systems like OpenAI's ChatGPT with plugins and xAI's Grok, represent a culminating phase in AI development, potentially constituting the "final generation" of intelligence as we currently conceive it. We explore the capabilities and underlying technologies of these agents, grounded in practical examples, while also examining the profound societal implications and the unprecedented pace of progress that suggests intelligence is now doubling approximately every six months. The paper concludes by underscoring the critical need for wisdom and foresight in navigating the opportunities and challenges presented by this powerful new era of intelligence. 

---
# "There Is No Such Thing as a Dumb Question," But There Are Good Ones 

**Authors**: Minjung Shin, Donghyun Kim, Jeh-Kwang Ryu  

**Link**: [PDF](https://arxiv.org/pdf/2505.09923)  

**Abstract**: Questioning has become increasingly crucial for both humans and artificial intelligence, yet there remains limited research comprehensively assessing question quality. In response, this study defines good questions and presents a systematic evaluation framework. We propose two key evaluation dimensions: appropriateness (sociolinguistic competence in context) and effectiveness (strategic competence in goal achievement). Based on these foundational dimensions, a rubric-based scoring system was developed. By incorporating dynamic contextual variables, our evaluation framework achieves structure and flexibility through semi-adaptive criteria. The methodology was validated using the CAUS and SQUARE datasets, demonstrating the ability of the framework to access both well-formed and problematic questions while adapting to varied contexts. As we establish a flexible and comprehensive framework for question evaluation, this study takes a significant step toward integrating questioning behavior with structured analytical methods grounded in the intrinsic nature of questioning. 

---
# Offline Reinforcement Learning for Microgrid Voltage Regulation 

**Authors**: Shan Yang, Yongli Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2505.09920)  

**Abstract**: This paper presents a study on using different offline reinforcement learning algorithms for microgrid voltage regulation with solar power penetration. When environment interaction is unviable due to technical or safety reasons, the proposed approach can still obtain an applicable model through offline-style training on a previously collected dataset, lowering the negative impact of lacking online environment interactions. Experiment results on the IEEE 33-bus system demonstrate the feasibility and effectiveness of the proposed approach on different offline datasets, including the one with merely low-quality experience. 

---
# A Multimodal Multi-Agent Framework for Radiology Report Generation 

**Authors**: Ziruo Yi, Ting Xiao, Mark V. Albert  

**Link**: [PDF](https://arxiv.org/pdf/2505.09787)  

**Abstract**: Radiology report generation (RRG) aims to automatically produce diagnostic reports from medical images, with the potential to enhance clinical workflows and reduce radiologists' workload. While recent approaches leveraging multimodal large language models (MLLMs) and retrieval-augmented generation (RAG) have achieved strong results, they continue to face challenges such as factual inconsistency, hallucination, and cross-modal misalignment. We propose a multimodal multi-agent framework for RRG that aligns with the stepwise clinical reasoning workflow, where task-specific agents handle retrieval, draft generation, visual analysis, refinement, and synthesis. Experimental results demonstrate that our approach outperforms a strong baseline in both automatic metrics and LLM-based evaluations, producing more accurate, structured, and interpretable reports. This work highlights the potential of clinically aligned multi-agent frameworks to support explainable and trustworthy clinical AI applications. 

---
# Explainability Through Human-Centric Design for XAI in Lung Cancer Detection 

**Authors**: Amy Rafferty, Rishi Ramaesh, Ajitha Rajan  

**Link**: [PDF](https://arxiv.org/pdf/2505.09755)  

**Abstract**: Deep learning models have shown promise in lung pathology detection from chest X-rays, but widespread clinical adoption remains limited due to opaque model decision-making. In prior work, we introduced ClinicXAI, a human-centric, expert-guided concept bottleneck model (CBM) designed for interpretable lung cancer diagnosis. We now extend that approach and present XpertXAI, a generalizable expert-driven model that preserves human-interpretable clinical concepts while scaling to detect multiple lung pathologies. Using a high-performing InceptionV3-based classifier and a public dataset of chest X-rays with radiology reports, we compare XpertXAI against leading post-hoc explainability methods and an unsupervised CBM, XCBs. We assess explanations through comparison with expert radiologist annotations and medical ground truth. Although XpertXAI is trained for multiple pathologies, our expert validation focuses on lung cancer. We find that existing techniques frequently fail to produce clinically meaningful explanations, omitting key diagnostic features and disagreeing with radiologist judgments. XpertXAI not only outperforms these baselines in predictive accuracy but also delivers concept-level explanations that better align with expert reasoning. While our focus remains on explainability in lung cancer detection, this work illustrates how human-centric model design can be effectively extended to broader diagnostic contexts - offering a scalable path toward clinically meaningful explainable AI in medical diagnostics. 

---
# General Dynamic Goal Recognition 

**Authors**: Osher Elhadad, Reuth Mirsky  

**Link**: [PDF](https://arxiv.org/pdf/2505.09737)  

**Abstract**: Understanding an agent's intent through its behavior is essential in human-robot interaction, interactive AI systems, and multi-agent collaborations. This task, known as Goal Recognition (GR), poses significant challenges in dynamic environments where goals are numerous and constantly evolving. Traditional GR methods, designed for a predefined set of goals, often struggle to adapt to these dynamic scenarios. To address this limitation, we introduce the General Dynamic GR problem - a broader definition of GR - aimed at enabling real-time GR systems and fostering further research in this area. Expanding on this foundation, this paper employs a model-free goal-conditioned RL approach to enable fast adaptation for GR across various changing tasks. 

---
# Feature Relevancy, Necessity and Usefulness: Complexity and Algorithms 

**Authors**: Tomás Capdevielle, Santiago Cifuentes  

**Link**: [PDF](https://arxiv.org/pdf/2505.09640)  

**Abstract**: Given a classification model and a prediction for some input, there are heuristic strategies for ranking features according to their importance in regard to the prediction. One common approach to this task is rooted in propositional logic and the notion of \textit{sufficient reason}. Through this concept, the categories of relevant and necessary features were proposed in order to identify the crucial aspects of the input. This paper improves the existing techniques and algorithms for deciding which are the relevant and/or necessary features, showing in particular that necessity can be detected efficiently in complex models such as neural networks. We also generalize the notion of relevancy and study associated problems. Moreover, we present a new global notion (i.e. that intends to explain whether a feature is important for the behavior of the model in general, not depending on a particular input) of \textit{usefulness} and prove that it is related to relevancy and necessity. Furthermore, we develop efficient algorithms for detecting it in decision trees and other more complex models, and experiment on three datasets to analyze its practical utility. 

---
# Study and improvement of search algorithms in two-players perfect information games 

**Authors**: Quentin Cohen-Solal  

**Link**: [PDF](https://arxiv.org/pdf/2505.09639)  

**Abstract**: Games, in their mathematical sense, are everywhere (game industries, economics, defense, education, chemistry, biology, ...).Search algorithms in games are artificial intelligence methods for playing such games. Unfortunately, there is no study on these algorithms that evaluates the generality of their performance. We propose to address this gap in the case of two-player zero-sum games with perfect information. Furthermore, we propose a new search algorithm and we show that, for a short search time, it outperforms all studied algorithms on all games in this large experiment and that, for a medium search time, it outperforms all studied algorithms on 17 of the 22 studied games. 

---
# Neural Thermodynamic Laws for Large Language Model Training 

**Authors**: Ziming Liu, Yizhou Liu, Jeff Gore, Max Tegmark  

**Link**: [PDF](https://arxiv.org/pdf/2505.10559)  

**Abstract**: Beyond neural scaling laws, little is known about the laws underlying large language models (LLMs). We introduce Neural Thermodynamic Laws (NTL) -- a new framework that offers fresh insights into LLM training dynamics. On the theoretical side, we demonstrate that key thermodynamic quantities (e.g., temperature, entropy, heat capacity, thermal conduction) and classical thermodynamic principles (e.g., the three laws of thermodynamics and the equipartition theorem) naturally emerge under river-valley loss landscape assumptions. On the practical side, this scientific perspective yields intuitive guidelines for designing learning rate schedules. 

---
# MathCoder-VL: Bridging Vision and Code for Enhanced Multimodal Mathematical Reasoning 

**Authors**: Ke Wang, Junting Pan, Linda Wei, Aojun Zhou, Weikang Shi, Zimu Lu, Han Xiao, Yunqiao Yang, Houxing Ren, Mingjie Zhan, Hongsheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.10557)  

**Abstract**: Natural language image-caption datasets, widely used for training Large Multimodal Models, mainly focus on natural scenarios and overlook the intricate details of mathematical figures that are critical for problem-solving, hindering the advancement of current LMMs in multimodal mathematical reasoning. To this end, we propose leveraging code as supervision for cross-modal alignment, since code inherently encodes all information needed to generate corresponding figures, establishing a precise connection between the two modalities. Specifically, we co-develop our image-to-code model and dataset with model-in-the-loop approach, resulting in an image-to-code model, FigCodifier and ImgCode-8.6M dataset, the largest image-code dataset to date. Furthermore, we utilize FigCodifier to synthesize novel mathematical figures and then construct MM-MathInstruct-3M, a high-quality multimodal math instruction fine-tuning dataset. Finally, we present MathCoder-VL, trained with ImgCode-8.6M for cross-modal alignment and subsequently fine-tuned on MM-MathInstruct-3M for multimodal math problem solving. Our model achieves a new open-source SOTA across all six metrics. Notably, it surpasses GPT-4o and Claude 3.5 Sonnet in the geometry problem-solving subset of MathVista, achieving improvements of 8.9% and 9.2%. The dataset and models will be released at this https URL. 

---
# Does Feasibility Matter? Understanding the Impact of Feasibility on Synthetic Training Data 

**Authors**: Yiwen Liu, Jessica Bader, Jae Myung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2505.10551)  

**Abstract**: With the development of photorealistic diffusion models, models trained in part or fully on synthetic data achieve progressively better results. However, diffusion models still routinely generate images that would not exist in reality, such as a dog floating above the ground or with unrealistic texture artifacts. We define the concept of feasibility as whether attributes in a synthetic image could realistically exist in the real-world domain; synthetic images containing attributes that violate this criterion are considered infeasible. Intuitively, infeasible images are typically considered out-of-distribution; thus, training on such images is expected to hinder a model's ability to generalize to real-world data, and they should therefore be excluded from the training set whenever possible. However, does feasibility really matter? In this paper, we investigate whether enforcing feasibility is necessary when generating synthetic training data for CLIP-based classifiers, focusing on three target attributes: background, color, and texture. We introduce VariReal, a pipeline that minimally edits a given source image to include feasible or infeasible attributes given by the textual prompt generated by a large language model. Our experiments show that feasibility minimally affects LoRA-fine-tuned CLIP performance, with mostly less than 0.3% difference in top-1 accuracy across three fine-grained datasets. Also, the attribute matters on whether the feasible/infeasible images adversarially influence the classification performance. Finally, mixing feasible and infeasible images in training datasets does not significantly impact performance compared to using purely feasible or infeasible datasets. 

---
# Real-Time Out-of-Distribution Failure Prevention via Multi-Modal Reasoning 

**Authors**: Milan Ganai, Rohan Sinha, Christopher Agia, Daniel Morton, Marco Pavone  

**Link**: [PDF](https://arxiv.org/pdf/2505.10547)  

**Abstract**: Foundation models can provide robust high-level reasoning on appropriate safety interventions in hazardous scenarios beyond a robot's training data, i.e. out-of-distribution (OOD) failures. However, due to the high inference latency of Large Vision and Language Models, current methods rely on manually defined intervention policies to enact fallbacks, thereby lacking the ability to plan generalizable, semantically safe motions. To overcome these challenges we present FORTRESS, a framework that generates and reasons about semantically safe fallback strategies in real time to prevent OOD failures. At a low frequency in nominal operations, FORTRESS uses multi-modal reasoners to identify goals and anticipate failure modes. When a runtime monitor triggers a fallback response, FORTRESS rapidly synthesizes plans to fallback goals while inferring and avoiding semantically unsafe regions in real time. By bridging open-world, multi-modal reasoning with dynamics-aware planning, we eliminate the need for hard-coded fallbacks and human safety interventions. FORTRESS outperforms on-the-fly prompting of slow reasoning models in safety classification accuracy on synthetic benchmarks and real-world ANYmal robot data, and further improves system safety and planning success in simulation and on quadrotor hardware for urban navigation. 

---
# LibIQ: Toward Real-Time Spectrum Classification in O-RAN dApps 

**Authors**: Filippo Olimpieri, Noemi Giustini, Andrea Lacava, Salvatore D'Oro, Tommaso Melodia, Francesca Cuomo  

**Link**: [PDF](https://arxiv.org/pdf/2505.10537)  

**Abstract**: The O-RAN architecture is transforming cellular networks by adopting RAN softwarization and disaggregation concepts to enable data-driven monitoring and control of the network. Such management is enabled by RICs, which facilitate near-real-time and non-real-time network control through xApps and rApps. However, they face limitations, including latency overhead in data exchange between the RAN and RIC, restricting real-time monitoring, and the inability to access user plain data due to privacy and security constraints, hindering use cases like beamforming and spectrum classification. In this paper, we leverage the dApps concept to enable real-time RF spectrum classification with LibIQ, a novel library for RF signals that facilitates efficient spectrum monitoring and signal classification by providing functionalities to read I/Q samples as time-series, create datasets and visualize time-series data through plots and spectrograms. Thanks to LibIQ, I/Q samples can be efficiently processed to detect external RF signals, which are subsequently classified using a CNN inside the library. To achieve accurate spectrum analysis, we created an extensive dataset of time-series-based I/Q samples, representing distinct signal types captured using a custom dApp running on a 5G deployment over the Colosseum network emulator and an OTA testbed. We evaluate our model by deploying LibIQ in heterogeneous scenarios with varying center frequencies, time windows, and external RF signals. In real-time analysis, the model classifies the processed I/Q samples, achieving an average accuracy of approximately 97.8\% in identifying signal types across all scenarios. We pledge to release both LibIQ and the dataset created as a publicly available framework upon acceptance. 

---
# Knowledge capture, adaptation and composition (KCAC): A framework for cross-task curriculum learning in robotic manipulation 

**Authors**: Xinrui Wang, Yan Jin  

**Link**: [PDF](https://arxiv.org/pdf/2505.10522)  

**Abstract**: Reinforcement learning (RL) has demonstrated remarkable potential in robotic manipulation but faces challenges in sample inefficiency and lack of interpretability, limiting its applicability in real world scenarios. Enabling the agent to gain a deeper understanding and adapt more efficiently to diverse working scenarios is crucial, and strategic knowledge utilization is a key factor in this process. This paper proposes a Knowledge Capture, Adaptation, and Composition (KCAC) framework to systematically integrate knowledge transfer into RL through cross-task curriculum learning. KCAC is evaluated using a two block stacking task in the CausalWorld benchmark, a complex robotic manipulation environment. To our knowledge, existing RL approaches fail to solve this task effectively, reflecting deficiencies in knowledge capture. In this work, we redesign the benchmark reward function by removing rigid constraints and strict ordering, allowing the agent to maximize total rewards concurrently and enabling flexible task completion. Furthermore, we define two self-designed sub-tasks and implement a structured cross-task curriculum to facilitate efficient learning. As a result, our KCAC approach achieves a 40 percent reduction in training time while improving task success rates by 10 percent compared to traditional RL methods. Through extensive evaluation, we identify key curriculum design parameters subtask selection, transition timing, and learning rate that optimize learning efficiency and provide conceptual guidance for curriculum based RL frameworks. This work offers valuable insights into curriculum design in RL and robotic learning. 

---
# Multi-Token Prediction Needs Registers 

**Authors**: Anastasios Gerontopoulos, Spyros Gidaris, Nikos Komodakis  

**Link**: [PDF](https://arxiv.org/pdf/2505.10518)  

**Abstract**: Multi-token prediction has emerged as a promising objective for improving language model pretraining, but its benefits have not consistently generalized to other settings such as fine-tuning. In this paper, we propose MuToR, a simple and effective approach to multi-token prediction that interleaves learnable register tokens into the input sequence, each tasked with predicting future targets. Compared to existing methods, MuToR offers several key advantages: it introduces only a negligible number of additional parameters, requires no architectural changes--ensuring compatibility with off-the-shelf pretrained language models--and remains aligned with the next-token pretraining objective, making it especially well-suited for supervised fine-tuning. Moreover, it naturally supports scalable prediction horizons. We demonstrate the effectiveness and versatility of MuToR across a range of use cases, including supervised fine-tuning, parameter-efficient fine-tuning (PEFT), and pretraining, on challenging generative tasks in both language and vision domains. Our code will be available at: this https URL. 

---
# PnPXAI: A Universal XAI Framework Providing Automatic Explanations Across Diverse Modalities and Models 

**Authors**: Seongun Kim, Sol A Kim, Geonhyeong Kim, Enver Menadjiev, Chanwoo Lee, Seongwook Chung, Nari Kim, Jaesik Choi  

**Link**: [PDF](https://arxiv.org/pdf/2505.10515)  

**Abstract**: Recently, post hoc explanation methods have emerged to enhance model transparency by attributing model outputs to input features. However, these methods face challenges due to their specificity to certain neural network architectures and data modalities. Existing explainable artificial intelligence (XAI) frameworks have attempted to address these challenges but suffer from several limitations. These include limited flexibility to diverse model architectures and data modalities due to hard-coded implementations, a restricted number of supported XAI methods because of the requirements for layer-specific operations of attribution methods, and sub-optimal recommendations of explanations due to the lack of evaluation and optimization phases. Consequently, these limitations impede the adoption of XAI technology in real-world applications, making it difficult for practitioners to select the optimal explanation method for their domain. To address these limitations, we introduce \textbf{PnPXAI}, a universal XAI framework that supports diverse data modalities and neural network models in a Plug-and-Play (PnP) manner. PnPXAI automatically detects model architectures, recommends applicable explanation methods, and optimizes hyperparameters for optimal explanations. We validate the framework's effectiveness through user surveys and showcase its versatility across various domains, including medicine and finance. 

---
# UniEval: Unified Holistic Evaluation for Unified Multimodal Understanding and Generation 

**Authors**: Yi Li, Haonan Wang, Qixiang Zhang, Boyu Xiao, Chenchang Hu, Hualiang Wang, Xiaomeng Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.10483)  

**Abstract**: The emergence of unified multimodal understanding and generation models is rapidly attracting attention because of their ability to enhance instruction-following capabilities while minimizing model redundancy. However, there is a lack of a unified evaluation framework for these models, which would enable an elegant, simplified, and overall evaluation. Current models conduct evaluations on multiple task-specific benchmarks, but there are significant limitations, such as the lack of overall results, errors from extra evaluation models, reliance on extensive labeled images, benchmarks that lack diversity, and metrics with limited capacity for instruction-following evaluation. To tackle these challenges, we introduce UniEval, the first evaluation framework designed for unified multimodal models without extra models, images, or annotations. This facilitates a simplified and unified evaluation process. The UniEval framework contains a holistic benchmark, UniBench (supports both unified and visual generation models), along with the corresponding UniScore metric. UniBench includes 81 fine-grained tags contributing to high diversity. Experimental results indicate that UniBench is more challenging than existing benchmarks, and UniScore aligns closely with human evaluations, surpassing current metrics. Moreover, we extensively evaluated SoTA unified and visual generation models, uncovering new insights into Univeral's unique values. 

---
# Fine-tuning Diffusion Policies with Backpropagation Through Diffusion Timesteps 

**Authors**: Ningyuan Yang, Jiaxuan Gao, Feng Gao, Yi Wu, Chao Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.10482)  

**Abstract**: Diffusion policies, widely adopted in decision-making scenarios such as robotics, gaming and autonomous driving, are capable of learning diverse skills from demonstration data due to their high representation power. However, the sub-optimal and limited coverage of demonstration data could lead to diffusion policies that generate sub-optimal trajectories and even catastrophic failures. While reinforcement learning (RL)-based fine-tuning has emerged as a promising solution to address these limitations, existing approaches struggle to effectively adapt Proximal Policy Optimization (PPO) to diffusion models. This challenge stems from the computational intractability of action likelihood estimation during the denoising process, which leads to complicated optimization objectives. In our experiments starting from randomly initialized policies, we find that online tuning of Diffusion Policies demonstrates much lower sample efficiency compared to directly applying PPO on MLP policies (MLP+PPO). To address these challenges, we introduce NCDPO, a novel framework that reformulates Diffusion Policy as a noise-conditioned deterministic policy. By treating each denoising step as a differentiable transformation conditioned on pre-sampled noise, NCDPO enables tractable likelihood evaluation and gradient backpropagation through all diffusion timesteps. Our experiments demonstrate that NCDPO achieves sample efficiency comparable to MLP+PPO when training from scratch, outperforming existing methods in both sample efficiency and final performance across diverse benchmarks, including continuous robot control and multi-agent game scenarios. Furthermore, our experimental results show that our method is robust to the number denoising timesteps in the Diffusion Policy. 

---
# Superposition Yields Robust Neural Scaling 

**Authors**: Yizhou liu, Ziming Liu, Jeff Gore  

**Link**: [PDF](https://arxiv.org/pdf/2505.10465)  

**Abstract**: The success of today's large language models (LLMs) depends on the observation that larger models perform better. However, the origin of this neural scaling law -- the finding that loss decreases as a power law with model size -- remains unclear. Starting from two empirical principles -- that LLMs represent more things than the model dimensions (widths) they have (i.e., representations are superposed), and that words or concepts in language occur with varying frequencies -- we constructed a toy model to study the loss scaling with model size. We found that when superposition is weak, meaning only the most frequent features are represented without interference, the scaling of loss with model size depends on the underlying feature frequency; if feature frequencies follow a power law, so does the loss. In contrast, under strong superposition, where all features are represented but overlap with each other, the loss becomes inversely proportional to the model dimension across a wide range of feature frequency distributions. This robust scaling behavior is explained geometrically: when many more vectors are packed into a lower dimensional space, the interference (squared overlaps) between vectors scales inversely with that dimension. We then analyzed four families of open-sourced LLMs and found that they exhibit strong superposition and quantitatively match the predictions of our toy model. The Chinchilla scaling law turned out to also agree with our results. We conclude that representation superposition is an important mechanism underlying the observed neural scaling laws. We anticipate that these insights will inspire new training strategies and model architectures to achieve better performance with less computation and fewer parameters. 

---
# SEAL: Searching Expandable Architectures for Incremental Learning 

**Authors**: Matteo Gambella, Vicente Javier Castro Solar, Manuel Roveri  

**Link**: [PDF](https://arxiv.org/pdf/2505.10457)  

**Abstract**: Incremental learning is a machine learning paradigm where a model learns from a sequential stream of tasks. This setting poses a key challenge: balancing plasticity (learning new tasks) and stability (preserving past knowledge). Neural Architecture Search (NAS), a branch of AutoML, automates the design of the architecture of Deep Neural Networks and has shown success in static settings. However, existing NAS-based approaches to incremental learning often rely on expanding the model at every task, making them impractical in resource-constrained environments. In this work, we introduce SEAL, a NAS-based framework tailored for data-incremental learning, a scenario where disjoint data samples arrive sequentially and are not stored for future access. SEAL adapts the model structure dynamically by expanding it only when necessary, based on a capacity estimation metric. Stability is preserved through cross-distillation training after each expansion step. The NAS component jointly searches for both the architecture and the optimal expansion policy. Experiments across multiple benchmarks demonstrate that SEAL effectively reduces forgetting and enhances accuracy while maintaining a lower model size compared to prior methods. These results highlight the promise of combining NAS and selective expansion for efficient, adaptive learning in incremental scenarios. 

---
# Vision language models have difficulty recognizing virtual objects 

**Authors**: Tyler Tran, Sangeet Khemlani, J.G. Trafton  

**Link**: [PDF](https://arxiv.org/pdf/2505.10453)  

**Abstract**: Vision language models (VLMs) are AI systems paired with both language and vision encoders to process multimodal input. They are capable of performing complex semantic tasks such as automatic captioning, but it remains an open question about how well they comprehend the visuospatial properties of scenes depicted in the images they process. We argue that descriptions of virtual objects -- objects that are not visually represented in an image -- can help test scene comprehension in these AI systems. For example, an image that depicts a person standing under a tree can be paired with the following prompt: imagine that a kite is stuck in the tree. VLMs that comprehend the scene should update their representations and reason sensibly about the spatial relations between all three objects. We describe systematic evaluations of state-of-the-art VLMs and show that their ability to process virtual objects is inadequate. 

---
# Are Large Language Models Robust in Understanding Code Against Semantics-Preserving Mutations? 

**Authors**: Pedro Orvalho, Marta Kwiatkowska  

**Link**: [PDF](https://arxiv.org/pdf/2505.10443)  

**Abstract**: Understanding the reasoning and robustness of Large Language Models (LLMs) is critical for their reliable use in programming tasks. While recent studies have assessed LLMs' ability to predict program outputs, most focus solely on the accuracy of those predictions, without evaluating the reasoning behind them. Moreover, it has been observed on mathematical reasoning tasks that LLMs can arrive at correct answers through flawed logic, raising concerns about similar issues in code understanding.
In this work, we evaluate whether state-of-the-art LLMs with up to 8B parameters can reason about Python programs or are simply guessing. We apply five semantics-preserving code mutations: renaming variables, mirroring comparison expressions, swapping if-else branches, converting for loops to while, and loop unrolling. These mutations maintain program semantics while altering its syntax. We evaluated six LLMs and performed a human expert analysis using LiveCodeBench to assess whether the correct predictions are based on sound reasoning. We also evaluated prediction stability across different code mutations on LiveCodeBench and CruxEval. Our findings show that some LLMs, such as Llama3.2, produce correct predictions based on flawed reasoning in up to 61% of cases. Furthermore, LLMs often change predictions in response to our code mutations, indicating limited robustness in their semantic understanding. 

---
# IN-RIL: Interleaved Reinforcement and Imitation Learning for Policy Fine-Tuning 

**Authors**: Dechen Gao, Hang Wang, Hanchu Zhou, Nejib Ammar, Shatadal Mishra, Ahmadreza Moradipari, Iman Soltani, Junshan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.10442)  

**Abstract**: Imitation learning (IL) and reinforcement learning (RL) each offer distinct advantages for robotics policy learning: IL provides stable learning from demonstrations, and RL promotes generalization through exploration. While existing robot learning approaches using IL-based pre-training followed by RL-based fine-tuning are promising, this two-step learning paradigm often suffers from instability and poor sample efficiency during the RL fine-tuning phase. In this work, we introduce IN-RIL, INterleaved Reinforcement learning and Imitation Learning, for policy fine-tuning, which periodically injects IL updates after multiple RL updates and hence can benefit from the stability of IL and the guidance of expert data for more efficient exploration throughout the entire fine-tuning process. Since IL and RL involve different optimization objectives, we develop gradient separation mechanisms to prevent destructive interference during \ABBR fine-tuning, by separating possibly conflicting gradient updates in orthogonal subspaces. Furthermore, we conduct rigorous analysis, and our findings shed light on why interleaving IL with RL stabilizes learning and improves sample-efficiency. Extensive experiments on 14 robot manipulation and locomotion tasks across 3 benchmarks, including FurnitureBench, OpenAI Gym, and Robomimic, demonstrate that \ABBR can significantly improve sample efficiency and mitigate performance collapse during online finetuning in both long- and short-horizon tasks with either sparse or dense rewards. IN-RIL, as a general plug-in compatible with various state-of-the-art RL algorithms, can significantly improve RL fine-tuning, e.g., from 12\% to 88\% with 6.3x improvement in the success rate on Robomimic Transport. Project page: this https URL. 

---
# PIF: Anomaly detection via preference embedding 

**Authors**: Filippo Leveni, Luca Magri, Giacomo Boracchi, Cesare Alippi  

**Link**: [PDF](https://arxiv.org/pdf/2505.10441)  

**Abstract**: We address the problem of detecting anomalies with respect to structured patterns. To this end, we conceive a novel anomaly detection method called PIF, that combines the advantages of adaptive isolation methods with the flexibility of preference embedding. Specifically, we propose to embed the data in a high dimensional space where an efficient tree-based method, PI-Forest, is employed to compute an anomaly score. Experiments on synthetic and real datasets demonstrate that PIF favorably compares with state-of-the-art anomaly detection techniques, and confirm that PI-Forest is better at measuring arbitrary distances and isolate points in the preference space. 

---
# Learned Lightweight Smartphone ISP with Unpaired Data 

**Authors**: Andrei Arhire, Radu Timofte  

**Link**: [PDF](https://arxiv.org/pdf/2505.10420)  

**Abstract**: The Image Signal Processor (ISP) is a fundamental component in modern smartphone cameras responsible for conversion of RAW sensor image data to RGB images with a strong focus on perceptual quality. Recent work highlights the potential of deep learning approaches and their ability to capture details with a quality increasingly close to that of professional cameras. A difficult and costly step when developing a learned ISP is the acquisition of pixel-wise aligned paired data that maps the raw captured by a smartphone camera sensor to high-quality reference images. In this work, we address this challenge by proposing a novel training method for a learnable ISP that eliminates the need for direct correspondences between raw images and ground-truth data with matching content. Our unpaired approach employs a multi-term loss function guided by adversarial training with multiple discriminators processing feature maps from pre-trained networks to maintain content structure while learning color and texture characteristics from the target RGB dataset. Using lightweight neural network architectures suitable for mobile devices as backbones, we evaluated our method on the Zurich RAW to RGB and Fujifilm UltraISP datasets. Compared to paired training methods, our unpaired learning strategy shows strong potential and achieves high fidelity across multiple evaluation metrics. The code and pre-trained models are available at this https URL . 

---
# Visual Fidelity Index for Generative Semantic Communications with Critical Information Embedding 

**Authors**: Jianhao Huang, Qunsong Zeng, Kaibin Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.10405)  

**Abstract**: Generative semantic communication (Gen-SemCom) with large artificial intelligence (AI) model promises a transformative paradigm for 6G networks, which reduces communication costs by transmitting low-dimensional prompts rather than raw data. However, purely prompt-driven generation loses fine-grained visual details. Additionally, there is a lack of systematic metrics to evaluate the performance of Gen-SemCom systems. To address these issues, we develop a hybrid Gen-SemCom system with a critical information embedding (CIE) framework, where both text prompts and semantically critical features are extracted for transmissions. First, a novel approach of semantic filtering is proposed to select and transmit the semantically critical features of images relevant to semantic label. By integrating the text prompt and critical features, the receiver reconstructs high-fidelity images using a diffusion-based generative model. Next, we propose the generative visual information fidelity (GVIF) metric to evaluate the visual quality of the generated image. By characterizing the statistical models of image features, the GVIF metric quantifies the mutual information between the distorted features and their original counterparts. By maximizing the GVIF metric, we design a channel-adaptive Gen-SemCom system that adaptively control the volume of features and compression rate according to the channel state. Experimental results validate the GVIF metric's sensitivity to visual fidelity, correlating with both the PSNR and critical information volume. In addition, the optimized system achieves superior performance over benchmarking schemes in terms of higher PSNR and lower FID scores. 

---
# Rethinking Repetition Problems of LLMs in Code Generation 

**Authors**: Yihong Dong, Yuchen Liu, Xue Jiang, Zhi Jin, Ge Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.10402)  

**Abstract**: With the advent of neural language models, the performance of code generation has been significantly boosted. However, the problem of repetitions during the generation process continues to linger. Previous work has primarily focused on content repetition, which is merely a fraction of the broader repetition problem in code generation. A more prevalent and challenging problem is structural repetition. In structural repetition, the repeated code appears in various patterns but possesses a fixed structure, which can be inherently reflected in grammar. In this paper, we formally define structural repetition and propose an efficient decoding approach called RPG, which stands for Repetition Penalization based on Grammar, to alleviate the repetition problems in code generation for LLMs. Specifically, RPG first leverages grammar rules to identify repetition problems during code generation, and then strategically decays the likelihood of critical tokens that contribute to repetitions, thereby mitigating them in code generation. To facilitate this study, we construct a new dataset CodeRepetEval to comprehensively evaluate approaches for mitigating the repetition problems in code generation. Extensive experimental results demonstrate that RPG substantially outperforms the best-performing baselines on CodeRepetEval dataset as well as HumanEval and MBPP benchmarks, effectively reducing repetitions and enhancing the quality of generated code. 

---
# Inconsistency Handling in DatalogMTL 

**Authors**: Meghyn Bienvenu, Camille Bourgaux, Atefe Khodadaditaghanaki  

**Link**: [PDF](https://arxiv.org/pdf/2505.10394)  

**Abstract**: In this paper, we explore the issue of inconsistency handling in DatalogMTL, an extension of Datalog with metric temporal operators. Since facts are associated with time intervals, there are different manners to restore consistency when they contradict the rules, such as removing facts or modifying their time intervals. Our first contribution is the definition of relevant notions of conflicts (minimal explanations for inconsistency) and repairs (possible ways of restoring consistency) for this setting and the study of the properties of these notions and the associated inconsistency-tolerant semantics. Our second contribution is a data complexity analysis of the tasks of generating a single conflict / repair and query entailment under repair-based semantics. 

---
# Uncovering Magnetic Phases with Synthetic Data and Physics-Informed Training 

**Authors**: Agustin Medina, Marcelo Arlego, Carlos A. Lamas  

**Link**: [PDF](https://arxiv.org/pdf/2505.10393)  

**Abstract**: We investigate the efficient learning of magnetic phases using artificial neural networks trained on synthetic data, combining computational simplicity with physics-informed strategies. Focusing on the diluted Ising model, which lacks an exact analytical solution, we explore two complementary approaches: a supervised classification using simple dense neural networks, and an unsupervised detection of phase transitions using convolutional autoencoders trained solely on idealized spin configurations.
To enhance model performance, we incorporate two key forms of physics-informed guidance. First, we exploit architectural biases which preferentially amplify features related to symmetry breaking. Second, we include training configurations that explicitly break $\mathbb{Z}_2$ symmetry, reinforcing the network's ability to detect ordered phases. These mechanisms, acting in tandem, increase the network's sensitivity to phase structure even in the absence of explicit labels. We validate the machine learning predictions through comparison with direct numerical estimates of critical temperatures and percolation thresholds.
Our results show that synthetic, structured, and computationally efficient training schemes can reveal physically meaningful phase boundaries, even in complex systems. This framework offers a low-cost and robust alternative to conventional methods, with potential applications in broader condensed matter and statistical physics contexts. 

---
# Schreier-Coset Graph Propagation 

**Authors**: Aryan Mishra, Lizhen Lin  

**Link**: [PDF](https://arxiv.org/pdf/2505.10392)  

**Abstract**: Graph Neural Networks (GNNs) offer a principled framework for learning over graph-structured data, yet their expressive capacity is often hindered by over-squashing, wherein information from distant nodes is compressed into fixed-size vectors. Existing solutions, including graph rewiring and bottleneck-resistant architectures such as Cayley and expander graphs, avoid this problem but introduce scalability bottlenecks. In particular, the Cayley graphs constructed over $SL(2,\mathbb{Z}_n)$ exhibit strong theoretical properties, yet suffer from cubic node growth $O(n^3)$, leading to high memory usage. To address this, this work introduces Schrier-Coset Graph Propagation (SCGP), a group-theoretic augmentation method that enriches node features through Schreier-coset embeddings without altering the input graph topology. SCGP embeds bottleneck-free connectivity patterns into a compact feature space, improving long-range message passing while maintaining computational efficiency. Empirical evaluations across standard node and graph classification benchmarks demonstrate that SCGP achieves performance comparable to, or exceeding, expander graph and rewired GNN baselines. Furthermore, SCGP exhibits particular advantages in processing hierarchical and modular graph structures, offering reduced inference latency, improved scalability, and a low memory footprint, making it suitable for real-time and resource-constrained applications. 

---
# Multi-Agent Path Finding For Large Agents Is Intractable 

**Authors**: Artem Agafonov, Konstantin Yakovlev  

**Link**: [PDF](https://arxiv.org/pdf/2505.10387)  

**Abstract**: The multi-agent path finding (MAPF) problem asks to find a set of paths on a graph such that when synchronously following these paths the agents never encounter a conflict. In the most widespread MAPF formulation, the so-called Classical MAPF, the agents sizes are neglected and two types of conflicts are considered: occupying the same vertex or using the same edge at the same time step. Meanwhile in numerous practical applications, e.g. in robotics, taking into account the agents' sizes is vital to ensure that the MAPF solutions can be safely executed. Introducing large agents yields an additional type of conflict arising when one agent follows an edge and its body overlaps with the body of another agent that is actually not using this same edge (e.g. staying still at some distinct vertex of the graph). Until now it was not clear how harder the problem gets when such conflicts are to be considered while planning. Specifically, it was known that Classical MAPF problem on an undirected graph can be solved in polynomial time, however no complete polynomial-time algorithm was presented to solve MAPF with large agents. In this paper we, for the first time, establish that the latter problem is NP-hard and, thus, if P!=NP no polynomial algorithm for it can, unfortunately, be presented. Our proof is based on the prevalent in the field technique of reducing the seminal 3SAT problem (which is known to be an NP-complete problem) to the problem at hand. In particular, for an arbitrary 3SAT formula we procedurally construct a dedicated graph with specific start and goal vertices and show that the given 3SAT formula is satisfiable iff the corresponding path finding instance has a solution. 

---
# Are Sparse Autoencoders Useful for Java Function Bug Detection? 

**Authors**: Rui Melo, Claudia Mamede, Andre Catarino, Rui Abreu, Henrique Lopes Cardoso  

**Link**: [PDF](https://arxiv.org/pdf/2505.10375)  

**Abstract**: Software vulnerabilities such as buffer overflows and SQL injections are a major source of security breaches. Traditional methods for vulnerability detection remain essential but are limited by high false positive rates, scalability issues, and reliance on manual effort. These constraints have driven interest in AI-based approaches to automated vulnerability detection and secure code generation. While Large Language Models (LLMs) have opened new avenues for classification tasks, their complexity and opacity pose challenges for interpretability and deployment. Sparse Autoencoder offer a promising solution to this problem. We explore whether SAEs can serve as a lightweight, interpretable alternative for bug detection in Java functions. We evaluate the effectiveness of SAEs when applied to representations from GPT-2 Small and Gemma 2B, examining their capacity to highlight buggy behaviour without fine-tuning the underlying LLMs. We found that SAE-derived features enable bug detection with an F1 score of up to 89%, consistently outperforming fine-tuned transformer encoder baselines. Our work provides the first empirical evidence that SAEs can be used to detect software bugs directly from the internal representations of pretrained LLMs, without any fine-tuning or task-specific supervision. 

---
# ILIF: Temporal Inhibitory Leaky Integrate-and-Fire Neuron for Overactivation in Spiking Neural Networks 

**Authors**: Kai Sun, Peibo Duan, Levin Kuhlmann, Beilun Wang, Bin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.10371)  

**Abstract**: The Spiking Neural Network (SNN) has drawn increasing attention for its energy-efficient, event-driven processing and biological plausibility. To train SNNs via backpropagation, surrogate gradients are used to approximate the non-differentiable spike function, but they only maintain nonzero derivatives within a narrow range of membrane potentials near the firing threshold, referred to as the surrogate gradient support width gamma. We identify a major challenge, termed the dilemma of gamma: a relatively large gamma leads to overactivation, characterized by excessive neuron firing, which in turn increases energy consumption, whereas a small gamma causes vanishing gradients and weakens temporal dependencies. To address this, we propose a temporal Inhibitory Leaky Integrate-and-Fire (ILIF) neuron model, inspired by biological inhibitory mechanisms. This model incorporates interconnected inhibitory units for membrane potential and current, effectively mitigating overactivation while preserving gradient propagation. Theoretical analysis demonstrates ILIF effectiveness in overcoming the gamma dilemma, and extensive experiments on multiple datasets show that ILIF improves energy efficiency by reducing firing rates, stabilizes training, and enhances accuracy. The code is available at this http URL. 

---
# FactsR: A Safer Method for Producing High Quality Healthcare Documentation 

**Authors**: Victor Petrén Bach Hansen, Lasse Krogsbøll, Jonas Lyngsø, Mathias Baltzersen, Andreas Motzfeldt, Kevin Pelgrims, Lars Maaløe  

**Link**: [PDF](https://arxiv.org/pdf/2505.10360)  

**Abstract**: There are now a multitude of AI-scribing solutions for healthcare promising the utilization of large language models for ambient documentation. However, these AI scribes still rely on one-shot, or few-shot prompts for generating notes after the consultation has ended, employing little to no reasoning. This risks long notes with an increase in hallucinations, misrepresentation of the intent of the clinician, and reliance on the proofreading of the clinician to catch errors. A dangerous combination for patient safety if vigilance is compromised by workload and fatigue. In this paper, we introduce a method for extracting salient clinical information in real-time alongside the healthcare consultation, denoted Facts, and use that information recursively to generate the final note. The FactsR method results in more accurate and concise notes by placing the clinician-in-the-loop of note generation, while opening up new use cases within real-time decision support. 

---
# SpikeVideoFormer: An Efficient Spike-Driven Video Transformer with Hamming Attention and $\mathcal{O}(T)$ Complexity 

**Authors**: Shihao Zou, Qingfeng Li, Wei Ji, Jingjing Li, Yongkui Yang, Guoqi Li, Chao Dong  

**Link**: [PDF](https://arxiv.org/pdf/2505.10352)  

**Abstract**: Spiking Neural Networks (SNNs) have shown competitive performance to Artificial Neural Networks (ANNs) in various vision tasks, while offering superior energy efficiency. However, existing SNN-based Transformers primarily focus on single-image tasks, emphasizing spatial features while not effectively leveraging SNNs' efficiency in video-based vision tasks. In this paper, we introduce SpikeVideoFormer, an efficient spike-driven video Transformer, featuring linear temporal complexity $\mathcal{O}(T)$. Specifically, we design a spike-driven Hamming attention (SDHA) which provides a theoretically guided adaptation from traditional real-valued attention to spike-driven attention. Building on SDHA, we further analyze various spike-driven space-time attention designs and identify an optimal scheme that delivers appealing performance for video tasks, while maintaining only linear temporal complexity. The generalization ability and efficiency of our model are demonstrated across diverse downstream video tasks, including classification, human pose tracking, and semantic segmentation. Empirical results show our method achieves state-of-the-art (SOTA) performance compared to existing SNN approaches, with over 15\% improvement on the latter two tasks. Additionally, it matches the performance of recent ANN-based methods while offering significant efficiency gains, achieving $\times 16$, $\times 10$ and $\times 5$ improvements on the three tasks. this https URL 

---
# Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning 

**Authors**: Gabriel S. Gama, Valdir Grassi Jr  

**Link**: [PDF](https://arxiv.org/pdf/2505.10347)  

**Abstract**: Specialized Multi-Task Optimizers (SMTOs) balance task learning in Multi-Task Learning by addressing issues like conflicting gradients and differing gradient norms, which hinder equal-weighted task training. However, recent critiques suggest that equally weighted tasks can achieve competitive results compared to SMTOs, arguing that previous SMTO results were influenced by poor hyperparameter optimization and lack of regularization. In this work, we evaluate these claims through an extensive empirical evaluation of SMTOs, including some of the latest methods, on more complex multi-task problems to clarify this behavior. Our findings indicate that SMTOs perform well compared to uniform loss and that fixed weights can achieve competitive performance compared to SMTOs. Furthermore, we demonstrate why uniform loss perform similarly to SMTOs in some instances. The code will be made publicly available. 

---
# Emergence of Structure in Ensembles of Random Neural Networks 

**Authors**: Luca Muscarnera, Luigi Loreti, Giovanni Todeschini, Alessio Fumagalli, Francesco Regazzoni  

**Link**: [PDF](https://arxiv.org/pdf/2505.10331)  

**Abstract**: Randomness is ubiquitous in many applications across data science and machine learning. Remarkably, systems composed of random components often display emergent global behaviors that appear deterministic, manifesting a transition from microscopic disorder to macroscopic organization. In this work, we introduce a theoretical model for studying the emergence of collective behaviors in ensembles of random classifiers. We argue that, if the ensemble is weighted through the Gibbs measure defined by adopting the classification loss as an energy, then there exists a finite temperature parameter for the distribution such that the classification is optimal, with respect to the loss (or the energy). Interestingly, for the case in which samples are generated by a Gaussian distribution and labels are constructed by employing a teacher perceptron, we analytically prove and numerically confirm that such optimal temperature does not depend neither on the teacher classifier (which is, by construction of the learning problem, unknown), nor on the number of random classifiers, highlighting the universal nature of the observed behavior. Experiments on the MNIST dataset underline the relevance of this phenomenon in high-quality, noiseless, datasets. Finally, a physical analogy allows us to shed light on the self-organizing nature of the studied phenomenon. 

---
# Efficient Adaptation of Reinforcement Learning Agents to Sudden Environmental Change 

**Authors**: Jonathan Clifford Balloch  

**Link**: [PDF](https://arxiv.org/pdf/2505.10330)  

**Abstract**: Real-world autonomous decision-making systems, from robots to recommendation engines, must operate in environments that change over time. While deep reinforcement learning (RL) has shown an impressive ability to learn optimal policies in stationary environments, most methods are data intensive and assume a world that does not change between training and test time. As a result, conventional RL methods struggle to adapt when conditions change. This poses a fundamental challenge: how can RL agents efficiently adapt their behavior when encountering novel environmental changes during deployment without catastrophically forgetting useful prior knowledge? This dissertation demonstrates that efficient online adaptation requires two key capabilities: (1) prioritized exploration and sampling strategies that help identify and learn from relevant experiences, and (2) selective preservation of prior knowledge through structured representations that can be updated without disruption to reusable components. 

---
# AutoPentest: Enhancing Vulnerability Management With Autonomous LLM Agents 

**Authors**: Julius Henke  

**Link**: [PDF](https://arxiv.org/pdf/2505.10321)  

**Abstract**: A recent area of increasing research is the use of Large Language Models (LLMs) in penetration testing, which promises to reduce costs and thus allow for higher frequency. We conduct a review of related work, identifying best practices and common evaluation issues. We then present AutoPentest, an application for performing black-box penetration tests with a high degree of autonomy. AutoPentest is based on the LLM GPT-4o from OpenAI and the LLM agent framework LangChain. It can perform complex multi-step tasks, augmented by external tools and knowledge bases. We conduct a study on three capture-the-flag style Hack The Box (HTB) machines, comparing our implementation AutoPentest with the baseline approach of manually using the ChatGPT-4o user interface. Both approaches are able to complete 15-25 % of the subtasks on the HTB machines, with AutoPentest slightly outperforming ChatGPT. We measure a total cost of \$96.20 US when using AutoPentest across all experiments, while a one-month subscription to ChatGPT Plus costs \$20. The results show that further implementation efforts and the use of more powerful LLMs released in the future are likely to make this a viable part of vulnerability management. 

---
# J1: Incentivizing Thinking in LLM-as-a-Judge via Reinforcement Learning 

**Authors**: Chenxi Whitehouse, Tianlu Wang, Ping Yu, Xian Li, Jason Weston, Ilia Kulikov, Swarnadeep Saha  

**Link**: [PDF](https://arxiv.org/pdf/2505.10320)  

**Abstract**: The progress of AI is bottlenecked by the quality of evaluation, and powerful LLM-as-a-Judge models have proved to be a core solution. Improved judgment ability is enabled by stronger chain-of-thought reasoning, motivating the need to find the best recipes for training such models to think. In this work we introduce J1, a reinforcement learning approach to training such models. Our method converts both verifiable and non-verifiable prompts to judgment tasks with verifiable rewards that incentivize thinking and mitigate judgment bias. In particular, our approach outperforms all other existing 8B or 70B models when trained at those sizes, including models distilled from DeepSeek-R1. J1 also outperforms o1-mini, and even R1 on some benchmarks, despite training a smaller model. We provide analysis and ablations comparing Pairwise-J1 vs Pointwise-J1 models, offline vs online training recipes, reward strategies, seed prompts, and variations in thought length and content. We find that our models make better judgments by learning to outline evaluation criteria, comparing against self-generated reference answers, and re-evaluating the correctness of model responses. 

---
# Private Transformer Inference in MLaaS: A Survey 

**Authors**: Yang Li, Xinyu Zhou, Yitong Wang, Liangxin Qian, Jun Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.10315)  

**Abstract**: Transformer models have revolutionized AI, powering applications like content generation and sentiment analysis. However, their deployment in Machine Learning as a Service (MLaaS) raises significant privacy concerns, primarily due to the centralized processing of sensitive user data. Private Transformer Inference (PTI) offers a solution by utilizing cryptographic techniques such as secure multi-party computation and homomorphic encryption, enabling inference while preserving both user data and model privacy. This paper reviews recent PTI advancements, highlighting state-of-the-art solutions and challenges. We also introduce a structured taxonomy and evaluation framework for PTI, focusing on balancing resource efficiency with privacy and bridging the gap between high-performance inference and data privacy. 

---
# AI LEGO: Scaffolding Cross-Functional Collaboration in Industrial Responsible AI Practices during Early Design Stages 

**Authors**: Muzhe Wu, Yanzhi Zhao, Shuyi Han, Michael Xieyang Liu, Hong Shen  

**Link**: [PDF](https://arxiv.org/pdf/2505.10300)  

**Abstract**: Responsible AI (RAI) efforts increasingly emphasize the importance of addressing potential harms early in the AI development lifecycle through social-technical lenses. However, in cross-functional industry teams, this work is often stalled by a persistent knowledge handoff challenge: the difficulty of transferring high-level, early-stage technical design rationales from technical experts to non-technical or user-facing roles for ethical evaluation and harm identification. Through literature review and a co-design study with 8 practitioners, we unpack how this challenge manifests -- technical design choices are rarely handed off in ways that support meaningful engagement by non-technical roles; collaborative workflows lack shared, visual structures to support mutual understanding; and non-technical practitioners are left without scaffolds for systematic harm evaluation. Existing tools like JIRA or Google Docs, while useful for product tracking, are ill-suited for supporting joint harm identification across roles, often requiring significant extra effort to align understanding. To address this, we developed AI LEGO, a web-based prototype that supports cross-functional AI practitioners in effectively facilitating knowledge handoff and identifying harmful design choices in the early design stages. Technical roles use interactive blocks to draft development plans, while non-technical roles engage with those blocks through stage-specific checklists and LLM-driven persona simulations to surface potential harms. In a study with 18 cross-functional practitioners, AI LEGO increased the volume and likelihood of harms identified compared to baseline worksheets. Participants found that its modular structure and persona prompts made harm identification more accessible, fostering clearer and more collaborative RAI practices in early design. 

---
# Defending the Edge: Representative-Attention for Mitigating Backdoor Attacks in Federated Learning 

**Authors**: Chibueze Peace Obioma, Youcheng Sun, Mustafa A. Mustafa  

**Link**: [PDF](https://arxiv.org/pdf/2505.10297)  

**Abstract**: Federated learning (FL) enhances privacy and reduces communication cost for resource-constrained edge clients by supporting distributed model training at the edge. However, the heterogeneous nature of such devices produces diverse, non-independent, and identically distributed (non-IID) data, making the detection of backdoor attacks more challenging. In this paper, we propose a novel federated representative-attention-based defense mechanism, named FeRA, that leverages cross-client attention over internal feature representations to distinguish benign from malicious clients. FeRA computes an anomaly score based on representation reconstruction errors, effectively identifying clients whose internal activations significantly deviate from the group consensus. Our evaluation demonstrates FeRA's robustness across various FL scenarios, including challenging non-IID data distributions typical of edge devices. Experimental results show that it effectively reduces backdoor attack success rates while maintaining high accuracy on the main task. The method is model-agnostic, attack-agnostic, and does not require labeled reference data, making it well suited to heterogeneous and resource-limited edge deployments. 

---
# AttentionGuard: Transformer-based Misbehavior Detection for Secure Vehicular Platoons 

**Authors**: Hexu Li, Konstantinos Kalogiannis, Ahmed Mohamed Hussain, Panos Papadimitratos  

**Link**: [PDF](https://arxiv.org/pdf/2505.10273)  

**Abstract**: Vehicle platooning, with vehicles traveling in close formation coordinated through Vehicle-to-Everything (V2X) communications, offers significant benefits in fuel efficiency and road utilization. However, it is vulnerable to sophisticated falsification attacks by authenticated insiders that can destabilize the formation and potentially cause catastrophic collisions. This paper addresses this challenge: misbehavior detection in vehicle platooning systems. We present AttentionGuard, a transformer-based framework for misbehavior detection that leverages the self-attention mechanism to identify anomalous patterns in mobility data. Our proposal employs a multi-head transformer-encoder to process sequential kinematic information, enabling effective differentiation between normal mobility patterns and falsification attacks across diverse platooning scenarios, including steady-state (no-maneuver) operation, join, and exit maneuvers. Our evaluation uses an extensive simulation dataset featuring various attack vectors (constant, gradual, and combined falsifications) and operational parameters (controller types, vehicle speeds, and attacker positions). Experimental results demonstrate that AttentionGuard achieves up to 0.95 F1-score in attack detection, with robust performance maintained during complex maneuvers. Notably, our system performs effectively with minimal latency (100ms decision intervals), making it suitable for real-time transportation safety applications. Comparative analysis reveals superior detection capabilities and establishes the transformer-encoder as a promising approach for securing Cooperative Intelligent Transport Systems (C-ITS) against sophisticated insider threats. 

---
# Cutting Through Privacy: A Hyperplane-Based Data Reconstruction Attack in Federated Learning 

**Authors**: Francesco Diana, André Nusser, Chuan Xu, Giovanni Neglia  

**Link**: [PDF](https://arxiv.org/pdf/2505.10264)  

**Abstract**: Federated Learning (FL) enables collaborative training of machine learning models across distributed clients without sharing raw data, ostensibly preserving data privacy. Nevertheless, recent studies have revealed critical vulnerabilities in FL, showing that a malicious central server can manipulate model updates to reconstruct clients' private training data. Existing data reconstruction attacks have important limitations: they often rely on assumptions about the clients' data distribution or their efficiency significantly degrades when batch sizes exceed just a few tens of samples.
In this work, we introduce a novel data reconstruction attack that overcomes these limitations. Our method leverages a new geometric perspective on fully connected layers to craft malicious model parameters, enabling the perfect recovery of arbitrarily large data batches in classification tasks without any prior knowledge of clients' data. Through extensive experiments on both image and tabular datasets, we demonstrate that our attack outperforms existing methods and achieves perfect reconstruction of data batches two orders of magnitude larger than the state of the art. 

---
# The Evolving Landscape of Generative Large Language Models and Traditional Natural Language Processing in Medicine 

**Authors**: Rui Yang, Huitao Li, Matthew Yu Heng Wong, Yuhe Ke, Xin Li, Kunyu Yu, Jingchi Liao, Jonathan Chong Kai Liew, Sabarinath Vinod Nair, Jasmine Chiat Ling Ong, Irene Li, Douglas Teodoro, Chuan Hong, Daniel Shu Wei Ting, Nan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.10261)  

**Abstract**: Natural language processing (NLP) has been traditionally applied to medicine, and generative large language models (LLMs) have become prominent recently. However, the differences between them across different medical tasks remain underexplored. We analyzed 19,123 studies, finding that generative LLMs demonstrate advantages in open-ended tasks, while traditional NLP dominates in information extraction and analysis tasks. As these technologies advance, ethical use of them is essential to ensure their potential in medical applications. 

---
# Comparing LLM Text Annotation Skills: A Study on Human Rights Violations in Social Media Data 

**Authors**: Poli Apollinaire Nemkova, Solomon Ubani, Mark V. Albert  

**Link**: [PDF](https://arxiv.org/pdf/2505.10260)  

**Abstract**: In the era of increasingly sophisticated natural language processing (NLP) systems, large language models (LLMs) have demonstrated remarkable potential for diverse applications, including tasks requiring nuanced textual understanding and contextual reasoning. This study investigates the capabilities of multiple state-of-the-art LLMs - GPT-3.5, GPT-4, LLAMA3, Mistral 7B, and Claude-2 - for zero-shot and few-shot annotation of a complex textual dataset comprising social media posts in Russian and Ukrainian. Specifically, the focus is on the binary classification task of identifying references to human rights violations within the dataset.
To evaluate the effectiveness of these models, their annotations are compared against a gold standard set of human double-annotated labels across 1000 samples. The analysis includes assessing annotation performance under different prompting conditions, with prompts provided in both English and Russian. Additionally, the study explores the unique patterns of errors and disagreements exhibited by each model, offering insights into their strengths, limitations, and cross-linguistic adaptability.
By juxtaposing LLM outputs with human annotations, this research contributes to understanding the reliability and applicability of LLMs for sensitive, domain-specific tasks in multilingual contexts. It also sheds light on how language models handle inherently subjective and context-dependent judgments, a critical consideration for their deployment in real-world scenarios. 

---
# On the Interplay of Human-AI Alignment,Fairness, and Performance Trade-offs in Medical Imaging 

**Authors**: Haozhe Luo, Ziyu Zhou, Zixin Shu, Aurélie Pahud de Mortanges, Robert Berke, Mauricio Reyes  

**Link**: [PDF](https://arxiv.org/pdf/2505.10231)  

**Abstract**: Deep neural networks excel in medical imaging but remain prone to biases, leading to fairness gaps across demographic groups. We provide the first systematic exploration of Human-AI alignment and fairness in this domain. Our results show that incorporating human insights consistently reduces fairness gaps and enhances out-of-domain generalization, though excessive alignment can introduce performance trade-offs, emphasizing the need for calibrated strategies. These findings highlight Human-AI alignment as a promising approach for developing fair, robust, and generalizable medical AI systems, striking a balance between expert guidance and automated efficiency. Our code is available at this https URL. 

---
# Do LLMs Memorize Recommendation Datasets? A Preliminary Study on MovieLens-1M 

**Authors**: Dario Di Palma, Felice Antonio Merra, Maurizio Sfilio, Vito Walter Anelli, Fedelucio Narducci, Tommaso Di Noia  

**Link**: [PDF](https://arxiv.org/pdf/2505.10212)  

**Abstract**: Large Language Models (LLMs) have become increasingly central to recommendation scenarios due to their remarkable natural language understanding and generation capabilities. Although significant research has explored the use of LLMs for various recommendation tasks, little effort has been dedicated to verifying whether they have memorized public recommendation dataset as part of their training data. This is undesirable because memorization reduces the generalizability of research findings, as benchmarking on memorized datasets does not guarantee generalization to unseen datasets. Furthermore, memorization can amplify biases, for example, some popular items may be recommended more frequently than others.
In this work, we investigate whether LLMs have memorized public recommendation datasets. Specifically, we examine two model families (GPT and Llama) across multiple sizes, focusing on one of the most widely used dataset in recommender systems: MovieLens-1M. First, we define dataset memorization as the extent to which item attributes, user profiles, and user-item interactions can be retrieved by prompting the LLMs. Second, we analyze the impact of memorization on recommendation performance. Lastly, we examine whether memorization varies across model families and model sizes. Our results reveal that all models exhibit some degree of memorization of MovieLens-1M, and that recommendation performance is related to the extent of memorization. We have made all the code publicly available at: this https URL 

---
# A Fine-Grained Complexity View on Propositional Abduction -- Algorithms and Lower Bounds 

**Authors**: Victor Lagerkvist, Mohamed Maizia, Johannes Schmidt  

**Link**: [PDF](https://arxiv.org/pdf/2505.10201)  

**Abstract**: The Boolean satisfiability problem (SAT) is a well-known example of monotonic reasoning, of intense practical interest due to fast solvers, complemented by rigorous fine-grained complexity results. However, for non-monotonic reasoning, e.g., abductive reasoning, comparably little is known outside classic complexity theory. In this paper we take a first step of bridging the gap between monotonic and non-monotonic reasoning by analyzing the complexity of intractable abduction problems under the seemingly overlooked but natural parameter n: the number of variables in the knowledge base. We obtain several positive results for $\Sigma^P_2$- as well as NP- and coNP-complete fragments, which implies the first example of beating exhaustive search for a $\Sigma^P_2$-complete problem (to the best of our knowledge). We complement this with lower bounds and for many fragments rule out improvements under the (strong) exponential-time hypothesis. 

---
# Advancing Community Detection with Graph Convolutional Neural Networks: Bridging Topological and Attributive Cohesion 

**Authors**: Anjali de Silva, Gang Chen, Hui Ma, Seyed Mohammad Nekooei, Xingquan Zuo  

**Link**: [PDF](https://arxiv.org/pdf/2505.10197)  

**Abstract**: Community detection, a vital technology for real-world applications, uncovers cohesive node groups (communities) by leveraging both topological and attribute similarities in social networks. However, existing Graph Convolutional Networks (GCNs) trained to maximize modularity often converge to suboptimal solutions. Additionally, directly using human-labeled communities for training can undermine topological cohesiveness by grouping disconnected nodes based solely on node attributes. We address these issues by proposing a novel Topological and Attributive Similarity-based Community detection (TAS-Com) method. TAS-Com introduces a novel loss function that exploits the highly effective and scalable Leiden algorithm to detect community structures with global optimal modularity. Leiden is further utilized to refine human-labeled communities to ensure connectivity within each community, enabling TAS-Com to detect community structures with desirable trade-offs between modularity and compliance with human labels. Experimental results on multiple benchmark networks confirm that TAS-Com can significantly outperform several state-of-the-art algorithms. 

---
# LanTu: Dynamics-Enhanced Deep Learning for Eddy-Resolving Ocean Forecasting 

**Authors**: Qingyu Zheng, Qi Shao, Guijun Han, Wei Li, Hong Li, Xuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.10191)  

**Abstract**: Mesoscale eddies dominate the spatiotemporal multiscale variability of the ocean, and their impact on the energy cascade of the global ocean cannot be ignored. Eddy-resolving ocean forecasting is providing more reliable protection for fisheries and navigational safety, but also presents significant scientific challenges and high computational costs for traditional numerical models. Artificial intelligence (AI)-based weather and ocean forecasting systems are becoming powerful tools that balance forecast performance with computational efficiency. However, the complex multiscale features in the ocean dynamical system make AI models still face many challenges in mesoscale eddy forecasting (especially regional modelling). Here, we develop LanTu, a regional eddy-resolving ocean forecasting system based on dynamics-enhanced deep learning. We incorporate cross-scale interactions into LanTu and construct multiscale physical constraint for optimising LanTu guided by knowledge of eddy dynamics in order to improve the forecasting skill of LanTu for mesoscale evolution. The results show that LanTu outperforms the existing advanced operational numerical ocean forecasting system (NOFS) and AI-based ocean forecasting system (AI-OFS) in temperature, salinity, sea level anomaly and current prediction, with a lead time of more than 10 days. Our study highlights that dynamics-enhanced deep learning (LanTu) can be a powerful paradigm for eddy-resolving ocean forecasting. 

---
# The CoT Encyclopedia: Analyzing, Predicting, and Controlling how a Reasoning Model will Think 

**Authors**: Seongyun Lee, Seungone Kim, Minju Seo, Yongrae Jo, Dongyoung Go, Hyeonbin Hwang, Jinho Park, Xiang Yue, Sean Welleck, Graham Neubig, Moontae Lee, Minjoon Seo  

**Link**: [PDF](https://arxiv.org/pdf/2505.10185)  

**Abstract**: Long chain-of-thought (CoT) is an essential ingredient in effective usage of modern large language models, but our understanding of the reasoning strategies underlying these capabilities remains limited. While some prior works have attempted to categorize CoTs using predefined strategy types, such approaches are constrained by human intuition and fail to capture the full diversity of model behaviors. In this work, we introduce the CoT Encyclopedia, a bottom-up framework for analyzing and steering model reasoning. Our method automatically extracts diverse reasoning criteria from model-generated CoTs, embeds them into a semantic space, clusters them into representative categories, and derives contrastive rubrics to interpret reasoning behavior. Human evaluations show that this framework produces more interpretable and comprehensive analyses than existing methods. Moreover, we demonstrate that this understanding enables performance gains: we can predict which strategy a model is likely to use and guide it toward more effective alternatives. Finally, we provide practical insights, such as that training data format (e.g., free-form vs. multiple-choice) has a far greater impact on reasoning behavior than data domain, underscoring the importance of format-aware model design. 

---
# KAITIAN: A Unified Communication Framework for Enabling Efficient Collaboration Across Heterogeneous Accelerators in Embodied AI Systems 

**Authors**: Jieke Lin, Wanyu Wang, Longxiang Yin, Yinhe Han  

**Link**: [PDF](https://arxiv.org/pdf/2505.10183)  

**Abstract**: Embodied Artificial Intelligence (AI) systems, such as autonomous robots and intelligent vehicles, are increasingly reliant on diverse heterogeneous accelerators (e.g., GPGPUs, NPUs, FPGAs) to meet stringent real-time processing and energy-efficiency demands. However, the proliferation of vendor-specific proprietary communication libraries creates significant interoperability barriers, hindering seamless collaboration between different accelerator types and leading to suboptimal resource utilization and performance bottlenecks in distributed AI workloads. This paper introduces KAITIAN, a novel distributed communication framework designed to bridge this gap. KAITIAN provides a unified abstraction layer that intelligently integrates vendor-optimized communication libraries for intra-group efficiency with general-purpose communication protocols for inter-group interoperability. Crucially, it incorporates a load-adaptive scheduling mechanism that dynamically balances computational tasks across heterogeneous devices based on their real-time performance characteristics. Implemented as an extension to PyTorch and rigorously evaluated on a testbed featuring NVIDIA GPUs and Cambricon MLUs, KAITIAN demonstrates significant improvements in resource utilization and scalability for distributed training tasks. Experimental results show that KAITIAN can accelerate training time by up to 42% compared to baseline homogeneous systems, while incurring minimal communication overhead (2.8--4.3%) and maintaining model accuracy. KAITIAN paves the way for more flexible and powerful heterogeneous computing in complex embodied AI applications. 

---
# Does Scaling Law Apply in Time Series Forecasting? 

**Authors**: Zeyan Li, Libing Chen, Yin Tang  

**Link**: [PDF](https://arxiv.org/pdf/2505.10172)  

**Abstract**: Rapid expansion of model size has emerged as a key challenge in time series forecasting. From early Transformer with tens of megabytes to recent architectures like TimesNet with thousands of megabytes, performance gains have often come at the cost of exponentially increasing parameter counts. But is this scaling truly necessary? To question the applicability of the scaling law in time series forecasting, we propose Alinear, an ultra-lightweight forecasting model that achieves competitive performance using only k-level parameters. We introduce a horizon-aware adaptive decomposition mechanism that dynamically rebalances component emphasis across different forecast lengths, alongside a progressive frequency attenuation strategy that achieves stable prediction in various forecasting horizons without incurring the computational overhead of attention mechanisms. Extensive experiments on seven benchmark datasets demonstrate that Alinear consistently outperforms large-scale models while using less than 1% of their parameters, maintaining strong accuracy across both short and ultra-long forecasting horizons. Moreover, to more fairly evaluate model efficiency, we propose a new parameter-aware evaluation metric that highlights the superiority of ALinear under constrained model budgets. Our analysis reveals that the relative importance of trend and seasonal components varies depending on data characteristics rather than following a fixed pattern, validating the necessity of our adaptive design. This work challenges the prevailing belief that larger models are inherently better and suggests a paradigm shift toward more efficient time series modeling. 

---
# Modeling Saliency Dataset Bias 

**Authors**: Matthias Kümmerer, Harneet Khanuja, Matthias Bethge  

**Link**: [PDF](https://arxiv.org/pdf/2505.10169)  

**Abstract**: Recent advances in image-based saliency prediction are approaching gold standard performance levels on existing benchmarks. Despite this success, we show that predicting fixations across multiple saliency datasets remains challenging due to dataset bias. We find a significant performance drop (around 40%) when models trained on one dataset are applied to another. Surprisingly, increasing dataset diversity does not resolve this inter-dataset gap, with close to 60% attributed to dataset-specific biases. To address this remaining generalization gap, we propose a novel architecture extending a mostly dataset-agnostic encoder-decoder structure with fewer than 20 dataset-specific parameters that govern interpretable mechanisms such as multi-scale structure, center bias, and fixation spread. Adapting only these parameters to new data accounts for more than 75% of the generalization gap, with a large fraction of the improvement achieved with as few as 50 samples. Our model sets a new state-of-the-art on all three datasets of the MIT/Tuebingen Saliency Benchmark (MIT300, CAT2000, and COCO-Freeview), even when purely generalizing from unrelated datasets, but with a substantial boost when adapting to the respective training datasets. The model also provides valuable insights into spatial saliency properties, revealing complex multi-scale effects that combine both absolute and relative sizes. 

---
# QuXAI: Explainers for Hybrid Quantum Machine Learning Models 

**Authors**: Saikat Barua, Mostafizur Rahman, Shehenaz Khaled, Md Jafor Sadek, Rafiul Islam, Shahnewaz Siddique  

**Link**: [PDF](https://arxiv.org/pdf/2505.10167)  

**Abstract**: The emergence of hybrid quantum-classical machine learning (HQML) models opens new horizons of computational intelligence but their fundamental complexity frequently leads to black box behavior that undermines transparency and reliability in their application. Although XAI for quantum systems still in its infancy, a major research gap is evident in robust global and local explainability approaches that are designed for HQML architectures that employ quantized feature encoding followed by classical learning. The gap is the focus of this work, which introduces QuXAI, an framework based upon Q-MEDLEY, an explainer for explaining feature importance in these hybrid systems. Our model entails the creation of HQML models incorporating quantum feature maps, the use of Q-MEDLEY, which combines feature based inferences, preserving the quantum transformation stage and visualizing the resulting attributions. Our result shows that Q-MEDLEY delineates influential classical aspects in HQML models, as well as separates their noise, and competes well against established XAI techniques in classical validation settings. Ablation studies more significantly expose the virtues of the composite structure used in Q-MEDLEY. The implications of this work are critically important, as it provides a route to improve the interpretability and reliability of HQML models, thus promoting greater confidence and being able to engage in safer and more responsible use of quantum-enhanced AI technology. 

---
# Large Wireless Localization Model (LWLM): A Foundation Model for Positioning in 6G Networks 

**Authors**: Guangjin Pan, Kaixuan Huang, Hui Chen, Shunqing Zhang, Christian Häger, Henk Wymeersch  

**Link**: [PDF](https://arxiv.org/pdf/2505.10134)  

**Abstract**: Accurate and robust localization is a critical enabler for emerging 5G and 6G applications, including autonomous driving, extended reality (XR), and smart manufacturing. While data-driven approaches have shown promise, most existing models require large amounts of labeled data and struggle to generalize across deployment scenarios and wireless configurations. To address these limitations, we propose a foundation-model-based solution tailored for wireless localization. We first analyze how different self-supervised learning (SSL) tasks acquire general-purpose and task-specific semantic features based on information bottleneck (IB) theory. Building on this foundation, we design a pretraining methodology for the proposed Large Wireless Localization Model (LWLM). Specifically, we propose an SSL framework that jointly optimizes three complementary objectives: (i) spatial-frequency masked channel modeling (SF-MCM), (ii) domain-transformation invariance (DTI), and (iii) position-invariant contrastive learning (PICL). These objectives jointly capture the underlying semantics of wireless channel from multiple perspectives. We further design lightweight decoders for key downstream tasks, including time-of-arrival (ToA) estimation, angle-of-arrival (AoA) estimation, single base station (BS) localization, and multiple BS localization. Comprehensive experimental results confirm that LWLM consistently surpasses both model-based and supervised learning baselines across all localization tasks. In particular, LWLM achieves 26.0%--87.5% improvement over transformer models without pretraining, and exhibits strong generalization under label-limited fine-tuning and unseen BS configurations, confirming its potential as a foundation model for wireless localization. 

---
# Robust Federated Learning on Edge Devices with Domain Heterogeneity 

**Authors**: Huy Q. Le, Latif U. Khan, Choong Seon Hong  

**Link**: [PDF](https://arxiv.org/pdf/2505.10128)  

**Abstract**: Federated Learning (FL) allows collaborative training while ensuring data privacy across distributed edge devices, making it a popular solution for privacy-sensitive applications. However, FL faces significant challenges due to statistical heterogeneity, particularly domain heterogeneity, which impedes the global mode's convergence. In this study, we introduce a new framework to address this challenge by improving the generalization ability of the FL global model under domain heterogeneity, using prototype augmentation. Specifically, we introduce FedAPC (Federated Augmented Prototype Contrastive Learning), a prototype-based FL framework designed to enhance feature diversity and model robustness. FedAPC leverages prototypes derived from the mean features of augmented data to capture richer representations. By aligning local features with global prototypes, we enable the model to learn meaningful semantic features while reducing overfitting to any specific domain. Experimental results on the Office-10 and Digits datasets illustrate that our framework outperforms SOTA baselines, demonstrating superior performance. 

---
# All You Need Is Synthetic Task Augmentation 

**Authors**: Guillaume Godin  

**Link**: [PDF](https://arxiv.org/pdf/2505.10120)  

**Abstract**: Injecting rule-based models like Random Forests into differentiable neural network frameworks remains an open challenge in machine learning. Recent advancements have demonstrated that pretrained models can generate efficient molecular embeddings. However, these approaches often require extensive pretraining and additional techniques, such as incorporating posterior probabilities, to boost performance. In our study, we propose a novel strategy that jointly trains a single Graph Transformer neural network on both sparse multitask molecular property experimental targets and synthetic targets derived from XGBoost models trained on Osmordred molecular descriptors. These synthetic tasks serve as independent auxiliary tasks. Our results show consistent and significant performance improvement across all 19 molecular property prediction tasks. For 16 out of 19 targets, the multitask Graph Transformer outperforms the XGBoost single-task learner. This demonstrates that synthetic task augmentation is an effective method for enhancing neural model performance in multitask molecular property prediction without the need for feature injection or pretraining. 

---
# EmbodiedMAE: A Unified 3D Multi-Modal Representation for Robot Manipulation 

**Authors**: Zibin Dong, Fei Ni, Yifu Yuan, Yinchuan Li, Jianye Hao  

**Link**: [PDF](https://arxiv.org/pdf/2505.10105)  

**Abstract**: We present EmbodiedMAE, a unified 3D multi-modal representation for robot manipulation. Current approaches suffer from significant domain gaps between training datasets and robot manipulation tasks, while also lacking model architectures that can effectively incorporate 3D information. To overcome these limitations, we enhance the DROID dataset with high-quality depth maps and point clouds, constructing DROID-3D as a valuable supplement for 3D embodied vision research. Then we develop EmbodiedMAE, a multi-modal masked autoencoder that simultaneously learns representations across RGB, depth, and point cloud modalities through stochastic masking and cross-modal fusion. Trained on DROID-3D, EmbodiedMAE consistently outperforms state-of-the-art vision foundation models (VFMs) in both training efficiency and final performance across 70 simulation tasks and 20 real-world robot manipulation tasks on two robot platforms. The model exhibits strong scaling behavior with size and promotes effective policy learning from 3D inputs. Experimental results establish EmbodiedMAE as a reliable unified 3D multi-modal VFM for embodied AI systems, particularly in precise tabletop manipulation settings where spatial perception is critical. 

---
# LAV: Audio-Driven Dynamic Visual Generation with Neural Compression and StyleGAN2 

**Authors**: Jongmin Jung, Dasaem Jeong  

**Link**: [PDF](https://arxiv.org/pdf/2505.10101)  

**Abstract**: This paper introduces LAV (Latent Audio-Visual), a system that integrates EnCodec's neural audio compression with StyleGAN2's generative capabilities to produce visually dynamic outputs driven by pre-recorded audio. Unlike previous works that rely on explicit feature mappings, LAV uses EnCodec embeddings as latent representations, directly transformed into StyleGAN2's style latent space via randomly initialized linear mapping. This approach preserves semantic richness in the transformation, enabling nuanced and semantically coherent audio-visual translations. The framework demonstrates the potential of using pretrained audio compression models for artistic and computational applications. 

---
# Multi-Robot Task Allocation for Homogeneous Tasks with Collision Avoidance via Spatial Clustering 

**Authors**: Rathin Chandra Shit, Sharmila Subudhi  

**Link**: [PDF](https://arxiv.org/pdf/2505.10073)  

**Abstract**: In this paper, a novel framework is presented that achieves a combined solution based on Multi-Robot Task Allocation (MRTA) and collision avoidance with respect to homogeneous measurement tasks taking place in industrial environments. The spatial clustering we propose offers to simultaneously solve the task allocation problem and deal with collision risks by cutting the workspace into distinguishable operational zones for each robot. To divide task sites and to schedule robot routes within corresponding clusters, we use K-means clustering and the 2-Opt algorithm. The presented framework shows satisfactory performance, where up to 93\% time reduction (1.24s against 17.62s) with a solution quality improvement of up to 7\% compared to the best performing method is demonstrated. Our method also completely eliminates collision points that persist in comparative methods in a most significant sense. Theoretical analysis agrees with the claim that spatial partitioning unifies the apparently disjoint tasks allocation and collision avoidance problems under conditions of many identical tasks to be distributed over sparse geographical areas. Ultimately, the findings in this work are of substantial importance for real world applications where both computational efficiency and operation free from collisions is of paramount importance. 

---
# Dark LLMs: The Growing Threat of Unaligned AI Models 

**Authors**: Michael Fire, Yitzhak Elbazis, Adi Wasenstein, Lior Rokach  

**Link**: [PDF](https://arxiv.org/pdf/2505.10066)  

**Abstract**: Large Language Models (LLMs) rapidly reshape modern life, advancing fields from healthcare to education and beyond. However, alongside their remarkable capabilities lies a significant threat: the susceptibility of these models to jailbreaking. The fundamental vulnerability of LLMs to jailbreak attacks stems from the very data they learn from. As long as this training data includes unfiltered, problematic, or 'dark' content, the models can inherently learn undesirable patterns or weaknesses that allow users to circumvent their intended safety controls. Our research identifies the growing threat posed by dark LLMs models deliberately designed without ethical guardrails or modified through jailbreak techniques. In our research, we uncovered a universal jailbreak attack that effectively compromises multiple state-of-the-art models, enabling them to answer almost any question and produce harmful outputs upon request. The main idea of our attack was published online over seven months ago. However, many of the tested LLMs were still vulnerable to this attack. Despite our responsible disclosure efforts, responses from major LLM providers were often inadequate, highlighting a concerning gap in industry practices regarding AI safety. As model training becomes more accessible and cheaper, and as open-source LLMs proliferate, the risk of widespread misuse escalates. Without decisive intervention, LLMs may continue democratizing access to dangerous knowledge, posing greater risks than anticipated. 

---
# PsOCR: Benchmarking Large Multimodal Models for Optical Character Recognition in Low-resource Pashto Language 

**Authors**: Ijazul Haq, Yingjie Zhang, Irfan Ali Khan  

**Link**: [PDF](https://arxiv.org/pdf/2505.10055)  

**Abstract**: This paper evaluates the performance of Large Multimodal Models (LMMs) on Optical Character Recognition (OCR) in the low-resource Pashto language. Natural Language Processing (NLP) in Pashto faces several challenges due to the cursive nature of its script and a scarcity of structured datasets. To address this, we developed a synthetic Pashto OCR dataset, PsOCR, consisting of one million images annotated with bounding boxes at word, line, and document levels, suitable for training and evaluating models based on different architectures, including Convolutional Neural Networks (CNNs) and Transformers. PsOCR covers variations across 1,000 unique font families, colors, image sizes, and layouts. A benchmark subset of 10K images was selected to evaluate the performance of several LMMs, including seven open-source models: DeepSeek's Janus, InternVL, MiniCPM, Florence, and Qwen (3B and 7B), and four closed-source models: GPT-4o, Gemini, Claude, and Grok. Experimental results demonstrate that Gemini achieves the best performance among all models, whereas among open-source models, Qwen-7B stands out. This work provides an insightful assessment of the capabilities and limitations of current LMMs for OCR tasks in Pashto and establishes a foundation for further research not only in Pashto OCR but also for other similar scripts such as Arabic, Persian, and Urdu. PsOCR is available at this https URL. 

---
# Financial Fraud Detection Using Explainable AI and Stacking Ensemble Methods 

**Authors**: Fahad Almalki, Mehedi Masud  

**Link**: [PDF](https://arxiv.org/pdf/2505.10050)  

**Abstract**: Traditional machine learning models often prioritize predictive accuracy, often at the expense of model transparency and interpretability. The lack of transparency makes it difficult for organizations to comply with regulatory requirements and gain stakeholders trust. In this research, we propose a fraud detection framework that combines a stacking ensemble of well-known gradient boosting models: XGBoost, LightGBM, and CatBoost. In addition, explainable artificial intelligence (XAI) techniques are used to enhance the transparency and interpretability of the model's decisions. We used SHAP (SHapley Additive Explanations) for feature selection to identify the most important features. Further efforts were made to explain the model's predictions using Local Interpretable Model-Agnostic Explanation (LIME), Partial Dependence Plots (PDP), and Permutation Feature Importance (PFI). The IEEE-CIS Fraud Detection dataset, which includes more than 590,000 real transaction records, was used to evaluate the proposed model. The model achieved a high performance with an accuracy of 99% and an AUC-ROC score of 0.99, outperforming several recent related approaches. These results indicate that combining high prediction accuracy with transparent interpretability is possible and could lead to a more ethical and trustworthy solution in financial fraud detection. 

---
# Boosting Text-to-Chart Retrieval through Training with Synthesized Semantic Insights 

**Authors**: Yifan Wu, Lutao Yan, Yizhang Zhu, Yinan Mei, Jiannan Wang, Nan Tang, Yuyu Luo  

**Link**: [PDF](https://arxiv.org/pdf/2505.10043)  

**Abstract**: Charts are crucial for data analysis and this http URL-to-chart retrieval systems have become increasingly important for Business Intelligence (BI), where users need to find relevant charts that match their analytical needs. These needs can be categorized into precise queries that are well-specified and fuzzy queries that are more exploratory -- both require understanding the semantics and context of the charts. However, existing text-to-chart retrieval solutions often fail to capture the semantic content and contextual information of charts, primarily due to the lack of comprehensive metadata (or semantic insights). To address this limitation, we propose a training data development pipeline that automatically synthesizes hierarchical semantic insights for charts, covering visual patterns (visual-oriented), statistical properties (statistics-oriented), and practical applications (task-oriented), which produces 207,498 semantic insights for 69,166 charts. Based on these, we train a CLIP-based model named ChartFinder to learn better representations of charts for text-to-chart retrieval. Our method leverages rich semantic insights during the training phase to develop a model that understands both visual and semantic aspects of this http URL evaluate text-to-chart retrieval performance, we curate the first benchmark, CRBench, for this task with 21,862 charts and 326 text queries from real-world BI applications, with ground-truth labels verified by the crowd this http URL show that ChartFinder significantly outperforms existing methods in text-to-chart retrieval tasks across various settings. For precise queries, ChartFinder achieves up to 66.9% NDCG@10, which is 11.58% higher than state-of-the-art models. In fuzzy query tasks, our method also demonstrates consistent improvements, with an average increase of 5% across nearly all metrics. 

---
# Optimal normalization in quantum-classical hybrid models for anti-cancer drug response prediction 

**Authors**: Takafumi Ito, Lysenko Artem, Tatsuhiko Tsunoda  

**Link**: [PDF](https://arxiv.org/pdf/2505.10037)  

**Abstract**: Quantum-classical Hybrid Machine Learning (QHML) models are recognized for their robust performance and high generalization ability even for relatively small datasets. These qualities offer unique advantages for anti-cancer drug response prediction, where the number of available samples is typically small. However, such hybrid models appear to be very sensitive to the data encoding used at the interface of a neural network and a quantum circuit, with suboptimal choices leading to stability issues. To address this problem, we propose a novel strategy that uses a normalization function based on a moderated gradient version of the $\tanh$. This method transforms the outputs of the neural networks without concentrating them at the extreme value ranges. Our idea was evaluated on a dataset of gene expression and drug response measurements for various cancer cell lines, where we compared the prediction performance of a classical deep learning model and several QHML models. These results confirmed that QHML performed better than the classical models when data was optimally normalized. This study opens up new possibilities for biomedical data analysis using quantum computers. 

---
# ORL-LDM: Offline Reinforcement Learning Guided Latent Diffusion Model Super-Resolution Reconstruction 

**Authors**: Shijie Lyu  

**Link**: [PDF](https://arxiv.org/pdf/2505.10027)  

**Abstract**: With the rapid advancement of remote sensing technology, super-resolution image reconstruction is of great research and practical significance. Existing deep learning methods have made progress but still face limitations in handling complex scenes and preserving image details. This paper proposes a reinforcement learning-based latent diffusion model (LDM) fine-tuning method for remote sensing image super-resolution. The method constructs a reinforcement learning environment with states, actions, and rewards, optimizing decision objectives through proximal policy optimization (PPO) during the reverse denoising process of the LDM model. Experiments on the RESISC45 dataset show significant improvements over the baseline model in PSNR, SSIM, and LPIPS, with PSNR increasing by 3-4dB, SSIM improving by 0.08-0.11, and LPIPS reducing by 0.06-0.10, particularly in structured and complex natural scenes. The results demonstrate the method's effectiveness in enhancing super-resolution quality and adaptability across scenes. 

---
# Application of YOLOv8 in monocular downward multiple Car Target detection 

**Authors**: Shijie Lyu  

**Link**: [PDF](https://arxiv.org/pdf/2505.10016)  

**Abstract**: Autonomous driving technology is progressively transforming traditional car driving methods, marking a significant milestone in modern transportation. Object detection serves as a cornerstone of autonomous systems, playing a vital role in enhancing driving safety, enabling autonomous functionality, improving traffic efficiency, and facilitating effective emergency responses. However, current technologies such as radar for environmental perception, cameras for road perception, and vehicle sensor networks face notable challenges, including high costs, vulnerability to weather and lighting conditions, and limited this http URL address these limitations, this paper presents an improved autonomous target detection network based on YOLOv8. By integrating structural reparameterization technology, a bidirectional pyramid structure network model, and a novel detection pipeline into the YOLOv8 framework, the proposed approach achieves highly efficient and precise detection of multi-scale, small, and remote objects. Experimental results demonstrate that the enhanced model can effectively detect both large and small objects with a detection accuracy of 65%, showcasing significant advancements over traditional this http URL improved model holds substantial potential for real-world applications and is well-suited for autonomous driving competitions, such as the Formula Student Autonomous China (FSAC), particularly excelling in scenarios involving single-target and small-object detection. 

---
# Quantum Computing and AI: Perspectives on Advanced Automation in Science and Engineering 

**Authors**: Tadashi Kadowaki  

**Link**: [PDF](https://arxiv.org/pdf/2505.10012)  

**Abstract**: Recent advances in artificial intelligence (AI) and quantum computing are accelerating automation in scientific and engineering processes, fundamentally reshaping research methodologies. This perspective highlights parallels between scientific automation and established Computer-Aided Engineering (CAE) practices, introducing Quantum CAE as a framework that leverages quantum algorithms for simulation, optimization, and machine learning within engineering design. Practical implementations of Quantum CAE are illustrated through case studies for combinatorial optimization problems. Further discussions include advancements toward higher automation levels, highlighting the critical role of specialized AI agents proficient in quantum algorithm design. The integration of quantum computing with AI raises significant questions about the collaborative dynamics among human scientists and engineers, AI systems, and quantum computational resources, underscoring a transformative future for automated discovery and innovation. 

---
# AI Greenferencing: Routing AI Inferencing to Green Modular Data Centers with Heron 

**Authors**: Tella Rajashekhar Reddy, Palak, Rohan Gandhi, Anjaly Parayil, Chaojie Zhang, Mike Shepperd, Liangcheng Yu, Jayashree Mohan, Srinivasan Iyengar, Shivkumar Kalyanaraman, Debopam Bhattacherjee  

**Link**: [PDF](https://arxiv.org/pdf/2505.09989)  

**Abstract**: AI power demand is growing unprecedentedly thanks to the high power density of AI compute and the emerging inferencing workload. On the supply side, abundant wind power is waiting for grid access in interconnection queues. In this light, this paper argues bringing AI workload to modular compute clusters co-located in wind farms. Our deployment right-sizing strategy makes it economically viable to deploy more than 6 million high-end GPUs today that could consume cheap, green power at its source. We built Heron, a cross-site software router, that could efficiently leverage the complementarity of power generation across wind farms by routing AI inferencing workload around power drops. Using 1-week ofcoding and conversation production traces from Azure and (real) variable wind power traces, we show how Heron improves aggregate goodput of AI compute by up to 80% compared to the state-of-the-art. 

---
# Analysing Safety Risks in LLMs Fine-Tuned with Pseudo-Malicious Cyber Security Data 

**Authors**: Adel ElZemity, Budi Arief, Shujun Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.09974)  

**Abstract**: The integration of large language models (LLMs) into cyber security applications presents significant opportunities, such as enhancing threat analysis and malware detection, but can also introduce critical risks and safety concerns, including personal data leakage and automated generation of new malware. We present a systematic evaluation of safety risks in fine-tuned LLMs for cyber security applications. Using the OWASP Top 10 for LLM Applications framework, we assessed seven open-source LLMs: Phi 3 Mini 3.8B, Mistral 7B, Qwen 2.5 7B, Llama 3 8B, Llama 3.1 8B, Gemma 2 9B, and Llama 2 70B. Our evaluation shows that fine-tuning reduces safety resilience across all tested LLMs (e.g., the safety score of Llama 3.1 8B against prompt injection drops from 0.95 to 0.15). We propose and evaluate a safety alignment approach that carefully rewords instruction-response pairs to include explicit safety precautions and ethical considerations. This approach demonstrates that it is possible to maintain or even improve model safety while preserving technical utility, offering a practical path forward for developing safer fine-tuning methodologies. This work offers a systematic evaluation for safety risks in LLMs, enabling safer adoption of generative AI in sensitive domains, and contributing towards the development of secure, trustworthy, and ethically aligned LLMs. 

---
# A Comprehensive Machine Learning Framework for Heart Disease Prediction: Performance Evaluation and Future Perspectives 

**Authors**: Ali Azimi Lamir, Shiva Razzagzadeh, Zeynab Rezaei  

**Link**: [PDF](https://arxiv.org/pdf/2505.09969)  

**Abstract**: This study presents a machine learning-based framework for heart disease prediction using the heart-disease dataset, comprising 303 samples with 14 features. The methodology involves data preprocessing, model training, and evaluation using three classifiers: Logistic Regression, K-Nearest Neighbors (KNN), and Random Forest. Hyperparameter tuning with GridSearchCV and RandomizedSearchCV was employed to enhance model performance. The Random Forest classifier outperformed other models, achieving an accuracy of 91% and an F1-score of 0.89. Evaluation metrics, including precision, recall, and confusion matrix, revealed balanced performance across classes. The proposed model demonstrates strong potential for aiding clinical decision-making by effectively predicting heart disease. Limitations such as dataset size and generalizability underscore the need for future studies using larger and more diverse datasets. This work highlights the utility of machine learning in healthcare, offering insights for further advancements in predictive diagnostics. 

---
# TransPL: VQ-Code Transition Matrices for Pseudo-Labeling of Time Series Unsupervised Domain Adaptation 

**Authors**: Jaeho Kim, Seulki Lee  

**Link**: [PDF](https://arxiv.org/pdf/2505.09955)  

**Abstract**: Unsupervised domain adaptation (UDA) for time series data remains a critical challenge in deep learning, with traditional pseudo-labeling strategies failing to capture temporal patterns and channel-wise shifts between domains, producing sub-optimal pseudo-labels. As such, we introduce TransPL, a novel approach that addresses these limitations by modeling the joint distribution $P(\mathbf{X}, y)$ of the source domain through code transition matrices, where the codes are derived from vector quantization (VQ) of time series patches. Our method constructs class- and channel-wise code transition matrices from the source domain and employs Bayes' rule for target domain adaptation, generating pseudo-labels based on channel-wise weighted class-conditional likelihoods. TransPL offers three key advantages: explicit modeling of temporal transitions and channel-wise shifts between different domains, versatility towards different UDA scenarios (e.g., weakly-supervised UDA), and explainable pseudo-label generation. We validate TransPL's effectiveness through extensive analysis on four time series UDA benchmarks and confirm that it consistently outperforms state-of-the-art pseudo-labeling methods by a strong margin (6.1% accuracy improvement, 4.9% F1 improvement), while providing interpretable insights into the domain adaptation process through its learned code transition matrices. 

---
# Task-Core Memory Management and Consolidation for Long-term Continual Learning 

**Authors**: Tianyu Huai, Jie Zhou, Yuxuan Cai, Qin Chen, Wen Wu, Xingjiao Wu, Xipeng Qiu, Liang He  

**Link**: [PDF](https://arxiv.org/pdf/2505.09952)  

**Abstract**: In this paper, we focus on a long-term continual learning (CL) task, where a model learns sequentially from a stream of vast tasks over time, acquiring new knowledge while retaining previously learned information in a manner akin to human learning. Unlike traditional CL settings, long-term CL involves handling a significantly larger number of tasks, which exacerbates the issue of catastrophic forgetting. Our work seeks to address two critical questions: 1) How do existing CL methods perform in the context of long-term CL? and 2) How can we mitigate the catastrophic forgetting that arises from prolonged sequential updates? To tackle these challenges, we propose a novel framework inspired by human memory mechanisms for long-term continual learning (Long-CL). Specifically, we introduce a task-core memory management strategy to efficiently index crucial memories and adaptively update them as learning progresses. Additionally, we develop a long-term memory consolidation mechanism that selectively retains hard and discriminative samples, ensuring robust knowledge retention. To facilitate research in this area, we construct and release two multi-modal and textual benchmarks, MMLongCL-Bench and TextLongCL-Bench, providing a valuable resource for evaluating long-term CL approaches. Experimental results show that Long-CL outperforms the previous state-of-the-art by 7.4\% and 6.5\% AP on the two benchmarks, respectively, demonstrating the effectiveness of our approach. 

---
# Personalizing Large Language Models using Retrieval Augmented Generation and Knowledge Graph 

**Authors**: Deeksha Prahlad, Chanhee Lee, Dongha Kim, Hokeun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2505.09945)  

**Abstract**: The advent of large language models (LLMs) has allowed numerous applications, including the generation of queried responses, to be leveraged in chatbots and other conversational assistants. Being trained on a plethora of data, LLMs often undergo high levels of over-fitting, resulting in the generation of extra and incorrect data, thus causing hallucinations in output generation. One of the root causes of such problems is the lack of timely, factual, and personalized information fed to the LLM. In this paper, we propose an approach to address these problems by introducing retrieval augmented generation (RAG) using knowledge graphs (KGs) to assist the LLM in personalized response generation tailored to the users. KGs have the advantage of storing continuously updated factual information in a structured way. While our KGs can be used for a variety of frequently updated personal data, such as calendar, contact, and location data, we focus on calendar data in this paper. Our experimental results show that our approach works significantly better in understanding personal information and generating accurate responses compared to the baseline LLMs using personal data as text inputs, with a moderate reduction in response time. 

---
# VRU-CIPI: Crossing Intention Prediction at Intersections for Improving Vulnerable Road Users Safety 

**Authors**: Ahmed S. Abdelrahman, Mohamed Abdel-Aty, Quoc Dai Tran  

**Link**: [PDF](https://arxiv.org/pdf/2505.09935)  

**Abstract**: Understanding and predicting human behavior in-thewild, particularly at urban intersections, remains crucial for enhancing interaction safety between road users. Among the most critical behaviors are crossing intentions of Vulnerable Road Users (VRUs), where misinterpretation may result in dangerous conflicts with oncoming vehicles. In this work, we propose the VRU-CIPI framework with a sequential attention-based model designed to predict VRU crossing intentions at intersections. VRU-CIPI employs Gated Recurrent Unit (GRU) to capture temporal dynamics in VRU movements, combined with a multi-head Transformer self-attention mechanism to encode contextual and spatial dependencies critical for predicting crossing direction. Evaluated on UCF-VRU dataset, our proposed achieves state-of-the-art performance with an accuracy of 96.45% and achieving real-time inference speed reaching 33 frames per second. Furthermore, by integrating with Infrastructure-to-Vehicles (I2V) communication, our approach can proactively enhance intersection safety through timely activation of crossing signals and providing early warnings to connected vehicles, ensuring smoother and safer interactions for all road users. 

---
# AdaptCLIP: Adapting CLIP for Universal Visual Anomaly Detection 

**Authors**: Bin-Bin Gao, Yue Zhu, Jiangtao Yan, Yuezhi Cai, Weixi Zhang, Meng Wang, Jun Liu, Yong Liu, Lei Wang, Chengjie Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.09926)  

**Abstract**: Universal visual anomaly detection aims to identify anomalies from novel or unseen vision domains without additional fine-tuning, which is critical in open scenarios. Recent studies have demonstrated that pre-trained vision-language models like CLIP exhibit strong generalization with just zero or a few normal images. However, existing methods struggle with designing prompt templates, complex token interactions, or requiring additional fine-tuning, resulting in limited flexibility. In this work, we present a simple yet effective method called AdaptCLIP based on two key insights. First, adaptive visual and textual representations should be learned alternately rather than jointly. Second, comparative learning between query and normal image prompt should incorporate both contextual and aligned residual features, rather than relying solely on residual features. AdaptCLIP treats CLIP models as a foundational service, adding only three simple adapters, visual adapter, textual adapter, and prompt-query adapter, at its input or output ends. AdaptCLIP supports zero-/few-shot generalization across domains and possesses a training-free manner on target domains once trained on a base dataset. AdaptCLIP achieves state-of-the-art performance on 12 anomaly detection benchmarks from industrial and medical domains, significantly outperforming existing competitive methods. We will make the code and model of AdaptCLIP available at this https URL. 

---
# Reinforced Interactive Continual Learning via Real-time Noisy Human Feedback 

**Authors**: Yutao Yang, Jie Zhou, Junsong Li, Qianjun Pan, Bihao Zhan, Qin Chen, Xipeng Qiu, Liang He  

**Link**: [PDF](https://arxiv.org/pdf/2505.09925)  

**Abstract**: This paper introduces an interactive continual learning paradigm where AI models dynamically learn new skills from real-time human feedback while retaining prior knowledge. This paradigm distinctively addresses two major limitations of traditional continual learning: (1) dynamic model updates using streaming, real-time human-annotated data, rather than static datasets with fixed labels, and (2) the assumption of clean labels, by explicitly handling the noisy feedback common in real-world interactions. To tackle these problems, we propose RiCL, a Reinforced interactive Continual Learning framework leveraging Large Language Models (LLMs) to learn new skills effectively from dynamic feedback. RiCL incorporates three key components: a temporal consistency-aware purifier to automatically discern clean from noisy samples in data streams; an interaction-aware direct preference optimization strategy to align model behavior with human intent by reconciling AI-generated and human-provided feedback; and a noise-resistant contrastive learning module that captures robust representations by exploiting inherent data relationships, thus avoiding reliance on potentially unreliable labels. Extensive experiments on two benchmark datasets (FewRel and TACRED), contaminated with realistic noise patterns, demonstrate that our RiCL approach substantially outperforms existing combinations of state-of-the-art online continual learning and noisy-label learning methods. 

---
# Avocado Price Prediction Using a Hybrid Deep Learning Model: TCN-MLP-Attention Architecture 

**Authors**: Linwei Zhang, LuFeng, Ruijia Liang  

**Link**: [PDF](https://arxiv.org/pdf/2505.09907)  

**Abstract**: With the growing demand for healthy foods, agricultural product price forecasting has become increasingly important. Hass avocados, as a high-value crop, exhibit complex price fluctuations influenced by factors such as seasonality, region, and weather. Traditional prediction models often struggle with highly nonlinear and dynamic data. To address this, we propose a hybrid deep learning model, TCN-MLP-Attention Architecture, combining Temporal Convolutional Networks (TCN) for sequential feature extraction, Multi-Layer Perceptrons (MLP) for nonlinear interactions, and an Attention mechanism for dynamic feature weighting. The dataset used covers over 50,000 records of Hass avocado sales across the U.S. from 2015 to 2018, including variables such as sales volume, average price, time, region, weather, and variety type, collected from point-of-sale systems and the Hass Avocado Board. After systematic preprocessing, including missing value imputation and feature normalization, the proposed model was trained and evaluated. Experimental results demonstrate that the TCN-MLP-Attention model achieves excellent predictive performance, with an RMSE of 1.23 and an MSE of 1.51, outperforming traditional methods. This research provides a scalable and effective approach for time series forecasting in agricultural markets and offers valuable insights for intelligent supply chain management and price strategy optimization. 

---
# Comparing Exploration-Exploitation Strategies of LLMs and Humans: Insights from Standard Multi-armed Bandit Tasks 

**Authors**: Ziyuan Zhang, Darcy Wang, Ningyuan Chen, Rodrigo Mansur, Vahid Sarhangian  

**Link**: [PDF](https://arxiv.org/pdf/2505.09901)  

**Abstract**: Large language models (LLMs) are increasingly used to simulate or automate human behavior in complex sequential decision-making tasks. A natural question is then whether LLMs exhibit similar decision-making behavior to humans, and can achieve comparable (or superior) performance. In this work, we focus on the exploration-exploitation (E&E) tradeoff, a fundamental aspect of dynamic decision-making under uncertainty. We employ canonical multi-armed bandit (MAB) tasks introduced in the cognitive science and psychiatry literature to conduct a comparative study of the E&E strategies of LLMs, humans, and MAB algorithms. We use interpretable choice models to capture the E&E strategies of the agents and investigate how explicit reasoning, through both prompting strategies and reasoning-enhanced models, shapes LLM decision-making. We find that reasoning shifts LLMs toward more human-like behavior, characterized by a mix of random and directed exploration. In simple stationary tasks, reasoning-enabled LLMs exhibit similar levels of random and directed exploration compared to humans. However, in more complex, non-stationary environments, LLMs struggle to match human adaptability, particularly in effective directed exploration, despite achieving similar regret in certain scenarios. Our findings highlight both the promise and limits of LLMs as simulators of human behavior and tools for automated decision-making and point to potential areas of improvements. 

---
# Which Demographic Features Are Relevant for Individual Fairness Evaluation of U.S. Recidivism Risk Assessment Tools? 

**Authors**: Tin Trung Nguyen, Jiannan Xu, Phuong-Anh Nguyen-Le, Jonathan Lazar, Donald Braman, Hal Daumé III, Zubin Jelveh  

**Link**: [PDF](https://arxiv.org/pdf/2505.09868)  

**Abstract**: Despite its U.S. constitutional foundation, the technical ``individual fairness'' criterion has not been operationalized in state or federal statutes/regulations. We conduct a human subjects experiment to address this gap, evaluating which demographic features are relevant for individual fairness evaluation of recidivism risk assessment (RRA) tools. Our analyses conclude that the individual similarity function should consider age and sex, but it should ignore race. 

---
# LiDDA: Data Driven Attribution at LinkedIn 

**Authors**: John Bencina, Erkut Aykutlug, Yue Chen, Zerui Zhang, Stephanie Sorenson, Shao Tang, Changshuai Wei  

**Link**: [PDF](https://arxiv.org/pdf/2505.09861)  

**Abstract**: Data Driven Attribution, which assigns conversion credits to marketing interactions based on causal patterns learned from data, is the foundation of modern marketing intelligence and vital to any marketing businesses and advertising platform. In this paper, we introduce a unified transformer-based attribution approach that can handle member-level data, aggregate-level data, and integration of external macro factors. We detail the large scale implementation of the approach at LinkedIn, showcasing significant impact. We also share learning and insights that are broadly applicable to the marketing and ad tech fields. 

---
# Predictability Shapes Adaptation: An Evolutionary Perspective on Modes of Learning in Transformers 

**Authors**: Alexander Y. Ku, Thomas L. Griffiths, Stephanie C.Y. Chan  

**Link**: [PDF](https://arxiv.org/pdf/2505.09855)  

**Abstract**: Transformer models learn in two distinct modes: in-weights learning (IWL), encoding knowledge into model weights, and in-context learning (ICL), adapting flexibly to context without weight modification. To better understand the interplay between these learning modes, we draw inspiration from evolutionary biology's analogous adaptive strategies: genetic encoding (akin to IWL, adapting over generations and fixed within an individual's lifetime) and phenotypic plasticity (akin to ICL, enabling flexible behavioral responses to environmental cues). In evolutionary biology, environmental predictability dictates the balance between these strategies: stability favors genetic encoding, while reliable predictive cues promote phenotypic plasticity. We experimentally operationalize these dimensions of predictability and systematically investigate their influence on the ICL/IWL balance in Transformers. Using regression and classification tasks, we show that high environmental stability decisively favors IWL, as predicted, with a sharp transition at maximal stability. Conversely, high cue reliability enhances ICL efficacy, particularly when stability is low. Furthermore, learning dynamics reveal task-contingent temporal evolution: while a canonical ICL-to-IWL shift occurs in some settings (e.g., classification with many classes), we demonstrate that scenarios with easier IWL (e.g., fewer classes) or slower ICL acquisition (e.g., regression) can exhibit an initial IWL phase later yielding to ICL dominance. These findings support a relative-cost hypothesis for explaining these learning mode transitions, establishing predictability as a critical factor governing adaptive strategies in Transformers, and offering novel insights for understanding ICL and guiding training methodologies. 

---
# Do Large Language Models Know Conflict? Investigating Parametric vs. Non-Parametric Knowledge of LLMs for Conflict Forecasting 

**Authors**: Apollinaire Poli Nemkova, Sarath Chandra Lingareddy, Sagnik Ray Choudhury, Mark V. Albert  

**Link**: [PDF](https://arxiv.org/pdf/2505.09852)  

**Abstract**: Large Language Models (LLMs) have shown impressive performance across natural language tasks, but their ability to forecast violent conflict remains underexplored. We investigate whether LLMs possess meaningful parametric knowledge-encoded in their pretrained weights-to predict conflict escalation and fatalities without external data. This is critical for early warning systems, humanitarian planning, and policy-making. We compare this parametric knowledge with non-parametric capabilities, where LLMs access structured and unstructured context from conflict datasets (e.g., ACLED, GDELT) and recent news reports via Retrieval-Augmented Generation (RAG). Incorporating external information could enhance model performance by providing up-to-date context otherwise missing from pretrained weights. Our two-part evaluation framework spans 2020-2024 across conflict-prone regions in the Horn of Africa and the Middle East. In the parametric setting, LLMs predict conflict trends and fatalities relying only on pretrained knowledge. In the non-parametric setting, models receive summaries of recent conflict events, indicators, and geopolitical developments. We compare predicted conflict trend labels (e.g., Escalate, Stable Conflict, De-escalate, Peace) and fatalities against historical data. Our findings highlight the strengths and limitations of LLMs for conflict forecasting and the benefits of augmenting them with structured external knowledge. 

---
# Causal Predictive Optimization and Generation for Business AI 

**Authors**: Liyang Zhao, Olurotimi Seton, Himadeep Reddy Reddivari, Suvendu Jena, Shadow Zhao, Rachit Kumar, Changshuai Wei  

**Link**: [PDF](https://arxiv.org/pdf/2505.09847)  

**Abstract**: The sales process involves sales functions converting leads or opportunities to customers and selling more products to existing customers. The optimization of the sales process thus is key to success of any B2B business. In this work, we introduce a principled approach to sales optimization and business AI, namely the Causal Predictive Optimization and Generation, which includes three layers: 1) prediction layer with causal ML 2) optimization layer with constraint optimization and contextual bandit 3) serving layer with Generative AI and feedback-loop for system enhancement. We detail the implementation and deployment of the system in LinkedIn, showcasing significant wins over legacy systems and sharing learning and insight broadly applicable to this field. 

---
# Evaluating Large Language Models for the Generation of Unit Tests with Equivalence Partitions and Boundary Values 

**Authors**: Martín Rodríguez, Gustavo Rossi, Alejandro Fernandez  

**Link**: [PDF](https://arxiv.org/pdf/2505.09830)  

**Abstract**: The design and implementation of unit tests is a complex task many programmers neglect. This research evaluates the potential of Large Language Models (LLMs) in automatically generating test cases, comparing them with manual tests. An optimized prompt was developed, that integrates code and requirements, covering critical cases such as equivalence partitions and boundary values. The strengths and weaknesses of LLMs versus trained programmers were compared through quantitative metrics and manual qualitative analysis. The results show that the effectiveness of LLMs depends on well-designed prompts, robust implementation, and precise requirements. Although flexible and promising, LLMs still require human supervision. This work highlights the importance of manual qualitative analysis as an essential complement to automation in unit test evaluation. 

---
# $XX^{t}$ Can Be Faster 

**Authors**: Dmitry Rybin, Yushun Zhang, Zhi-Quan Luo  

**Link**: [PDF](https://arxiv.org/pdf/2505.09814)  

**Abstract**: We present a new algorithm RXTX that computes product of matrix by its transpose $XX^{t}$. RXTX uses $5\%$ less multiplications and additions than State-of-the-Art and achieves accelerations even for small sizes of matrix $X$. The algorithm was discovered by combining Machine Learning-based search methods with Combinatorial Optimization. 

---
# Exploring the generalization of LLM truth directions on conversational formats 

**Authors**: Timour Ichmoukhamedov, David Martens  

**Link**: [PDF](https://arxiv.org/pdf/2505.09807)  

**Abstract**: Several recent works argue that LLMs have a universal truth direction where true and false statements are linearly separable in the activation space of the model. It has been demonstrated that linear probes trained on a single hidden state of the model already generalize across a range of topics and might even be used for lie detection in LLM conversations. In this work we explore how this truth direction generalizes between various conversational formats. We find good generalization between short conversations that end on a lie, but poor generalization to longer formats where the lie appears earlier in the input prompt. We propose a solution that significantly improves this type of generalization by adding a fixed key phrase at the end of each conversation. Our results highlight the challenges towards reliable LLM lie detectors that generalize to new settings. 

---
# Contextual Phenotyping of Pediatric Sepsis Cohort Using Large Language Models 

**Authors**: Aditya Nagori, Ayush Gautam, Matthew O. Wiens, Vuong Nguyen, Nathan Kenya Mugisha, Jerome Kabakyenga, Niranjan Kissoon, John Mark Ansermino, Rishikesan Kamaleswaran  

**Link**: [PDF](https://arxiv.org/pdf/2505.09805)  

**Abstract**: Clustering patient subgroups is essential for personalized care and efficient resource use. Traditional clustering methods struggle with high-dimensional, heterogeneous healthcare data and lack contextual understanding. This study evaluates Large Language Model (LLM) based clustering against classical methods using a pediatric sepsis dataset from a low-income country (LIC), containing 2,686 records with 28 numerical and 119 categorical variables. Patient records were serialized into text with and without a clustering objective. Embeddings were generated using quantized LLAMA 3.1 8B, DeepSeek-R1-Distill-Llama-8B with low-rank adaptation(LoRA), and Stella-En-400M-V5 models. K-means clustering was applied to these embeddings. Classical comparisons included K-Medoids clustering on UMAP and FAMD-reduced mixed data. Silhouette scores and statistical tests evaluated cluster quality and distinctiveness. Stella-En-400M-V5 achieved the highest Silhouette Score (0.86). LLAMA 3.1 8B with the clustering objective performed better with higher number of clusters, identifying subgroups with distinct nutritional, clinical, and socioeconomic profiles. LLM-based methods outperformed classical techniques by capturing richer context and prioritizing key features. These results highlight potential of LLMs for contextual phenotyping and informed decision-making in resource-limited settings. 

---
# Virtual Dosimetrists: A Radiotherapy Training "Flight Simulator" 

**Authors**: Skylar S. Gay, Tucker Netherton, Barbara Marquez, Raymond Mumme, Mary Gronberg, Brent Parker, Chelsea Pinnix, Sanjay Shete, Carlos Cardenas, Laurence Court  

**Link**: [PDF](https://arxiv.org/pdf/2505.09796)  

**Abstract**: Effective education in radiotherapy plan quality review requires a robust, regularly updated set of examples and the flexibility to demonstrate multiple possible planning approaches and their consequences. However, the current clinic-based paradigm does not support these needs. To address this, we have developed 'Virtual Dosimetrist' models that can both generate training examples of suboptimal treatment plans and then allow trainees to improve the plan quality through simple natural language prompts, as if communicating with a dosimetrist. The dose generation and modification process is accurate, rapid, and requires only modest resources. This work is the first to combine dose distribution prediction with natural language processing; providing a robust pipeline for both generating suboptimal training plans and allowing trainees to practice their critical plan review and improvement skills that addresses the challenges of the current clinic-based paradigm. 

---
# Automated Detection of Clinical Entities in Lung and Breast Cancer Reports Using NLP Techniques 

**Authors**: J. Moreno-Casanova, J.M. Auñón, A. Mártinez-Pérez, M.E. Pérez-Martínez, M.E. Gas-López  

**Link**: [PDF](https://arxiv.org/pdf/2505.09794)  

**Abstract**: Research projects, including those focused on cancer, rely on the manual extraction of information from clinical reports. This process is time-consuming and prone to errors, limiting the efficiency of data-driven approaches in healthcare. To address these challenges, Natural Language Processing (NLP) offers an alternative for automating the extraction of relevant data from electronic health records (EHRs). In this study, we focus on lung and breast cancer due to their high incidence and the significant impact they have on public health. Early detection and effective data management in both types of cancer are crucial for improving patient outcomes. To enhance the accuracy and efficiency of data extraction, we utilized GMV's NLP tool uQuery, which excels at identifying relevant entities in clinical texts and converting them into standardized formats such as SNOMED and OMOP. uQuery not only detects and classifies entities but also associates them with contextual information, including negated entities, temporal aspects, and patient-related details. In this work, we explore the use of NLP techniques, specifically Named Entity Recognition (NER), to automatically identify and extract key clinical information from EHRs related to these two cancers. A dataset from Health Research Institute Hospital La Fe (IIS La Fe), comprising 200 annotated breast cancer and 400 lung cancer reports, was used, with eight clinical entities manually labeled using the Doccano platform. To perform NER, we fine-tuned the bsc-bio-ehr-en3 model, a RoBERTa-based biomedical linguistic model pre-trained in Spanish. Fine-tuning was performed using the Transformers architecture, enabling accurate recognition of clinical entities in these cancer types. Our results demonstrate strong overall performance, particularly in identifying entities like MET and PAT, although challenges remain with less frequent entities like EVOL. 

---
# On the Well-Posedness of Green's Function Reconstruction via the Kirchhoff-Helmholtz Equation for One-Speed Neutron Diffusion 

**Authors**: Roberto Ponciroli  

**Link**: [PDF](https://arxiv.org/pdf/2505.09766)  

**Abstract**: This work presents a methodology for reconstructing the spatial distribution of the neutron flux in a nuclear reactor, leveraging real-time measurements obtained from ex-core detectors. The Kirchhoff-Helmholtz (K-H) equation inherently defines the problem of estimating a scalar field within a domain based on boundary data, making it a natural mathematical framework for this task. The main challenge lies in deriving the Green's function specific to the domain and the neutron diffusion process. While analytical solutions for Green's functions exist for simplified geometries, their derivation of complex, heterogeneous domains-such as a nuclear reactor-requires a numerical approach. The objective of this work is to demonstrate the well-posedness of the data-driven Green's function approximation by formulating and solving the K-H equation as an inverse problem. After establishing the symmetry properties that the Green's function must satisfy, the K-H equation is derived from the one-speed neutron diffusion model. This is followed by a comprehensive description of the procedure for interpreting sensor readings and implementing the neutron flux reconstruction algorithm. Finally, the existence and uniqueness of the Green's function inferred from the sampled data are demonstrated, ensuring the reliability of the proposed method and its predictions. 

---
# Trustless Autonomy: Understanding Motivations, Benefits and Governance Dilemma in Self-Sovereign Decentralized AI Agents 

**Authors**: Botao Amber Hu, Yuhan Liu, Helena Rong  

**Link**: [PDF](https://arxiv.org/pdf/2505.09757)  

**Abstract**: The recent trend of self-sovereign Decentralized AI Agents (DeAgents) combines Large Language Model (LLM)-based AI agents with decentralization technologies such as blockchain smart contracts and trusted execution environments (TEEs). These tamper-resistant trustless substrates allow agents to achieve self-sovereignty through ownership of cryptowallet private keys and control of digital assets and social media accounts. DeAgent eliminates centralized control and reduces human intervention, addressing key trust concerns inherent in centralized AI systems. However, given ongoing challenges in LLM reliability such as hallucinations, this creates paradoxical tension between trustlessness and unreliable autonomy. This study addresses this empirical research gap through interviews with DeAgents stakeholders-experts, founders, and developers-to examine their motivations, benefits, and governance dilemmas. The findings will guide future DeAgents system and protocol design and inform discussions about governance in sociotechnical AI systems in the future agentic web. 

---
# Healthy Distrust in AI systems 

**Authors**: Benjamin Paaßen, Suzana Alpsancar, Tobias Matzner, Ingrid Scharlau  

**Link**: [PDF](https://arxiv.org/pdf/2505.09747)  

**Abstract**: Under the slogan of trustworthy AI, much of contemporary AI research is focused on designing AI systems and usage practices that inspire human trust and, thus, enhance adoption of AI systems. However, a person affected by an AI system may not be convinced by AI system design alone -- neither should they, if the AI system is embedded in a social context that gives good reason to believe that it is used in tension with a person's interest. In such cases, distrust in the system may be justified and necessary to build meaningful trust in the first place. We propose the term "healthy distrust" to describe such a justified, careful stance towards certain AI usage practices. We investigate prior notions of trust and distrust in computer science, sociology, history, psychology, and philosophy, outline a remaining gap that healthy distrust might fill and conceptualize healthy distrust as a crucial part for AI usage that respects human autonomy. 

---
# A Generative Neural Annealer for Black-Box Combinatorial Optimization 

**Authors**: Yuan-Hang Zhang, Massimiliano Di Ventra  

**Link**: [PDF](https://arxiv.org/pdf/2505.09742)  

**Abstract**: We propose a generative, end-to-end solver for black-box combinatorial optimization that emphasizes both sample efficiency and solution quality on NP problems. Drawing inspiration from annealing-based algorithms, we treat the black-box objective as an energy function and train a neural network to model the associated Boltzmann distribution. By conditioning on temperature, the network captures a continuum of distributions--from near-uniform at high temperatures to sharply peaked around global optima at low temperatures--thereby learning the structure of the energy landscape and facilitating global optimization. When queries are expensive, the temperature-dependent distributions naturally enable data augmentation and improve sample efficiency. When queries are cheap but the problem remains hard, the model learns implicit variable interactions, effectively "opening" the black box. We validate our approach on challenging combinatorial tasks under both limited and unlimited query budgets, showing competitive performance against state-of-the-art black-box optimizers. 

---
# Achieving Tokenizer Flexibility in Language Models through Heuristic Adaptation and Supertoken Learning 

**Authors**: Shaurya Sharthak, Vinayak Pahalwan, Adithya Kamath, Adarsh Shirawalmath  

**Link**: [PDF](https://arxiv.org/pdf/2505.09738)  

**Abstract**: Pretrained language models (LLMs) are often constrained by their fixed tokenization schemes, leading to inefficiencies and performance limitations, particularly for multilingual or specialized applications. This tokenizer lock-in presents significant challenges. standard methods to overcome this often require prohibitive computational resources. Although tokenizer replacement with heuristic initialization aims to reduce this burden, existing methods often require exhaustive residual fine-tuning and still may not fully preserve semantic nuances or adequately address the underlying compression inefficiencies. Our framework introduces two innovations: first, Tokenadapt, a model-agnostic tokenizer transplantation method, and second, novel pre-tokenization learning for multi-word Supertokens to enhance compression and reduce fragmentation. Tokenadapt initializes new unique token embeddings via a hybrid heuristic that combines two methods: a local estimate based on subword decomposition using the old tokenizer, and a global estimate utilizing the top-k semantically similar tokens from the original vocabulary. This methodology aims to preserve semantics while significantly minimizing retraining requirements. Empirical investigations validate both contributions: the transplantation heuristic successfully initializes unique tokens, markedly outperforming conventional baselines and sophisticated methods including Transtokenizer and ReTok, while our Supertokens achieve notable compression gains. Our zero-shot perplexity results demonstrate that the TokenAdapt hybrid initialization consistently yields lower perplexity ratios compared to both ReTok and TransTokenizer baselines across different base models and newly trained target tokenizers. TokenAdapt typically reduced the overall perplexity ratio significantly compared to ReTok, yielding at least a 2-fold improvement in these aggregate scores. 

---
# Robust Federated Learning with Confidence-Weighted Filtering and GAN-Based Completion under Noisy and Incomplete Data 

**Authors**: Alpaslan Gokcen, Ali Boyaci  

**Link**: [PDF](https://arxiv.org/pdf/2505.09733)  

**Abstract**: Federated learning (FL) presents an effective solution for collaborative model training while maintaining data privacy across decentralized client datasets. However, data quality issues such as noisy labels, missing classes, and imbalanced distributions significantly challenge its effectiveness. This study proposes a federated learning methodology that systematically addresses data quality issues, including noise, class imbalance, and missing labels. The proposed approach systematically enhances data integrity through adaptive noise cleaning, collaborative conditional GAN-based synthetic data generation, and robust federated model training. Experimental evaluations conducted on benchmark datasets (MNIST and Fashion-MNIST) demonstrate significant improvements in federated model performance, particularly macro-F1 Score, under varying noise and class imbalance conditions. Additionally, the proposed framework carefully balances computational feasibility and substantial performance gains, ensuring practicality for resource constrained edge devices while rigorously maintaining data privacy. Our results indicate that this method effectively mitigates common data quality challenges, providing a robust, scalable, and privacy compliant solution suitable for diverse real-world federated learning scenarios. 

---
# An AI-Powered Research Assistant in the Lab: A Practical Guide for Text Analysis Through Iterative Collaboration with LLMs 

**Authors**: Gino Carmona-Díaz, William Jiménez-Leal, María Alejandra Grisales, Chandra Sripada, Santiago Amaya, Michael Inzlicht, Juan Pablo Bermúdez  

**Link**: [PDF](https://arxiv.org/pdf/2505.09724)  

**Abstract**: Analyzing texts such as open-ended responses, headlines, or social media posts is a time- and labor-intensive process highly susceptible to bias. LLMs are promising tools for text analysis, using either a predefined (top-down) or a data-driven (bottom-up) taxonomy, without sacrificing quality. Here we present a step-by-step tutorial to efficiently develop, test, and apply taxonomies for analyzing unstructured data through an iterative and collaborative process between researchers and LLMs. Using personal goals provided by participants as an example, we demonstrate how to write prompts to review datasets and generate a taxonomy of life domains, evaluate and refine the taxonomy through prompt and direct modifications, test the taxonomy and assess intercoder agreements, and apply the taxonomy to categorize an entire dataset with high intercoder reliability. We discuss the possibilities and limitations of using LLMs for text analysis. 

---
# Out-of-distribution generalisation is hard: evidence from ARC-like tasks 

**Authors**: George Dimitriadis. Spyridon Samothrakis  

**Link**: [PDF](https://arxiv.org/pdf/2505.09716)  

**Abstract**: Out-of-distribution (OOD) generalisation is considered a hallmark of human and animal intelligence. To achieve OOD through composition, a system must discover the environment-invariant properties of experienced input-output mappings and transfer them to novel inputs. This can be realised if an intelligent system can identify appropriate, task-invariant, and composable input features, as well as the composition methods, thus allowing it to act based not on the interpolation between learnt data points but on the task-invariant composition of those features. We propose that in order to confirm that an algorithm does indeed learn compositional structures from data, it is not enough to just test on an OOD setup, but one also needs to confirm that the features identified are indeed compositional. We showcase this by exploring two tasks with clearly defined OOD metrics that are not OOD solvable by three commonly used neural networks: a Multi-Layer Perceptron (MLP), a Convolutional Neural Network (CNN), and a Transformer. In addition, we develop two novel network architectures imbued with biases that allow them to be successful in OOD scenarios. We show that even with correct biases and almost perfect OOD performance, an algorithm can still fail to learn the correct features for compositional generalisation. 

---
# Energy-Efficient Federated Learning for AIoT using Clustering Methods 

**Authors**: Roberto Pereira, Fernanda Famá, Charalampos Kalalas, Paolo Dini  

**Link**: [PDF](https://arxiv.org/pdf/2505.09704)  

**Abstract**: While substantial research has been devoted to optimizing model performance, convergence rates, and communication efficiency, the energy implications of federated learning (FL) within Artificial Intelligence of Things (AIoT) scenarios are often overlooked in the existing literature. This study examines the energy consumed during the FL process, focusing on three main energy-intensive processes: pre-processing, communication, and local learning, all contributing to the overall energy footprint. We rely on the observation that device/client selection is crucial for speeding up the convergence of model training in a distributed AIoT setting and propose two clustering-informed methods. These clustering solutions are designed to group AIoT devices with similar label distributions, resulting in clusters composed of nearly heterogeneous devices. Hence, our methods alleviate the heterogeneity often encountered in real-world distributed learning applications. Throughout extensive numerical experimentation, we demonstrate that our clustering strategies typically achieve high convergence rates while maintaining low energy consumption when compared to other recent approaches available in the literature. 

---
# ManipBench: Benchmarking Vision-Language Models for Low-Level Robot Manipulation 

**Authors**: Enyu Zhao, Vedant Raval, Hejia Zhang, Jiageng Mao, Zeyu Shangguan, Stefanos Nikolaidis, Yue Wang, Daniel Seita  

**Link**: [PDF](https://arxiv.org/pdf/2505.09698)  

**Abstract**: Vision-Language Models (VLMs) have revolutionized artificial intelligence and robotics due to their commonsense reasoning capabilities. In robotic manipulation, VLMs are used primarily as high-level planners, but recent work has also studied their lower-level reasoning ability, which refers to making decisions about precise robot movements. However, the community currently lacks a clear and common benchmark that can evaluate how well VLMs can aid low-level reasoning in robotics. Consequently, we propose a novel benchmark, ManipBench, to evaluate the low-level robot manipulation reasoning capabilities of VLMs across various dimensions, including how well they understand object-object interactions and deformable object manipulation. We extensively test 33 representative VLMs across 10 model families on our benchmark, including variants to test different model sizes. Our evaluation shows that the performance of VLMs significantly varies across tasks, and there is a strong correlation between this performance and trends in our real-world manipulation tasks. It also shows that there remains a significant gap between these models and human-level understanding. See our website at: this https URL. 

---
# System Prompt Optimization with Meta-Learning 

**Authors**: Yumin Choi, Jinheon Baek, Sung Ju Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2505.09666)  

**Abstract**: Large Language Models (LLMs) have shown remarkable capabilities, with optimizing their input prompts playing a pivotal role in maximizing their performance. However, while LLM prompts consist of both the task-agnostic system prompts and task-specific user prompts, existing work on prompt optimization has focused on user prompts specific to individual queries or tasks, and largely overlooked the system prompt that is, once optimized, applicable across different tasks and domains. Motivated by this, we introduce the novel problem of bilevel system prompt optimization, whose objective is to design system prompts that are robust to diverse user prompts and transferable to unseen tasks. To tackle this problem, we then propose a meta-learning framework, which meta-learns the system prompt by optimizing it over various user prompts across multiple datasets, while simultaneously updating the user prompts in an iterative manner to ensure synergy between them. We conduct experiments on 14 unseen datasets spanning 5 different domains, on which we show that our approach produces system prompts that generalize effectively to diverse user prompts. Also, our findings reveal that the optimized system prompt enables rapid adaptation even to unseen tasks, requiring fewer optimization steps for test-time user prompts while achieving improved performance. 

---
# Introducing voice timbre attribute detection 

**Authors**: Jinghao He, Zhengyan Sheng, Liping Chen, Kong Aik Lee, Zhen-Hua Ling  

**Link**: [PDF](https://arxiv.org/pdf/2505.09661)  

**Abstract**: This paper focuses on explaining the timbre conveyed by speech signals and introduces a task termed voice timbre attribute detection (vTAD). In this task, voice timbre is explained with a set of sensory attributes describing its human perception. A pair of speech utterances is processed, and their intensity is compared in a designated timbre descriptor. Moreover, a framework is proposed, which is built upon the speaker embeddings extracted from the speech utterances. The investigation is conducted on the VCTK-RVA dataset. Experimental examinations on the ECAPA-TDNN and FACodec speaker encoders demonstrated that: 1) the ECAPA-TDNN speaker encoder was more capable in the seen scenario, where the testing speakers were included in the training set; 2) the FACodec speaker encoder was superior in the unseen scenario, where the testing speakers were not part of the training, indicating enhanced generalization capability. The VCTK-RVA dataset and open-source code are available on the website this https URL. 

---
# Differentiable Quantum Architecture Search in Quantum-Enhanced Neural Network Parameter Generation 

**Authors**: Samuel Yen-Chi Chen, Chen-Yu Liu, Kuan-Cheng Chen, Wei-Jia Huang, Yen-Jui Chang, Wei-Hao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.09653)  

**Abstract**: The rapid advancements in quantum computing (QC) and machine learning (ML) have led to the emergence of quantum machine learning (QML), which integrates the strengths of both fields. Among QML approaches, variational quantum circuits (VQCs), also known as quantum neural networks (QNNs), have shown promise both empirically and theoretically. However, their broader adoption is hindered by reliance on quantum hardware during inference. Hardware imperfections and limited access to quantum devices pose practical challenges. To address this, the Quantum-Train (QT) framework leverages the exponential scaling of quantum amplitudes to generate classical neural network parameters, enabling inference without quantum hardware and achieving significant parameter compression. Yet, designing effective quantum circuit architectures for such quantum-enhanced neural programmers remains non-trivial and often requires expertise in quantum information science. In this paper, we propose an automated solution using differentiable optimization. Our method jointly optimizes both conventional circuit parameters and architectural parameters in an end-to-end manner via automatic differentiation. We evaluate the proposed framework on classification, time-series prediction, and reinforcement learning tasks. Simulation results show that our method matches or outperforms manually designed QNN architectures. This work offers a scalable and automated pathway for designing QNNs that can generate classical neural network parameters across diverse applications. 

---
# Unlocking Location Intelligence: A Survey from Deep Learning to The LLM Era 

**Authors**: Xixuan Hao, Yutian Jiang, Xingchen Zou, Jiabo Liu, Yifang Yin, Yuxuan Liang  

**Link**: [PDF](https://arxiv.org/pdf/2505.09651)  

**Abstract**: Location Intelligence (LI), the science of transforming location-centric geospatial data into actionable knowledge, has become a cornerstone of modern spatial decision-making. The rapid evolution of Geospatial Representation Learning is fundamentally reshaping LI development through two successive technological revolutions: the deep learning breakthrough and the emerging large language model (LLM) paradigm. While deep neural networks (DNNs) have demonstrated remarkable success in automated feature extraction from structured geospatial data (e.g., satellite imagery, GPS trajectories), the recent integration of LLMs introduces transformative capabilities for cross-modal geospatial reasoning and unstructured geo-textual data processing. This survey presents a comprehensive review of geospatial representation learning across both technological eras, organizing them into a structured taxonomy based on the complete pipeline comprising: (1) data perspective, (2) methodological perspective and (3) application perspective. We also highlight current advancements, discuss existing limitations, and propose potential future research directions in the LLM era. This work offers a thorough exploration of the field and providing a roadmap for further innovation in LI. The summary of the up-to-date paper list can be found in this https URL and will undergo continuous updates. 

---
# Temporal Interception and Present Reconstruction: A Cognitive-Signal Model for Human and AI Decision Making 

**Authors**: Carmel Mary Esther A  

**Link**: [PDF](https://arxiv.org/pdf/2505.09646)  

**Abstract**: This paper proposes a novel theoretical model to explain how the human mind and artificial intelligence can approach real-time awareness by reducing perceptual delays. By investigating cosmic signal delay, neurological reaction times, and the ancient cognitive state of stillness, we explore how one may shift from reactive perception to a conscious interface with the near future. This paper introduces both a physical and cognitive model for perceiving the present not as a linear timestamp, but as an interference zone where early-arriving cosmic signals and reactive human delays intersect. We propose experimental approaches to test these ideas using human neural observation and neuro-receptive extensions. Finally, we propose a mathematical framework to guide the evolution of AI systems toward temporally efficient, ethically sound, and internally conscious decision-making processes 

---
# Neurophysiologically Realistic Environment for Comparing Adaptive Deep Brain Stimulation Algorithms in Parkinson Disease 

**Authors**: Ekaterina Kuzmina, Dmitrii Kriukov, Mikhail Lebedev, Dmitry V. Dylov  

**Link**: [PDF](https://arxiv.org/pdf/2505.09624)  

**Abstract**: Adaptive deep brain stimulation (aDBS) has emerged as a promising treatment for Parkinson disease (PD). In aDBS, a surgically placed electrode sends dynamically altered stimuli to the brain based on neurophysiological feedback: an invasive gadget that limits the amount of data one could collect for optimizing the control offline. As a consequence, a plethora of synthetic models of PD and those of the control algorithms have been proposed. Herein, we introduce the first neurophysiologically realistic benchmark for comparing said models. Specifically, our methodology covers not only conventional basal ganglia circuit dynamics and pathological oscillations, but also captures 15 previously dismissed physiological attributes, such as signal instabilities and noise, neural drift, electrode conductance changes and individual variability - all modeled as spatially distributed and temporally registered features via beta-band activity in the brain and a feedback. Furthermore, we purposely built our framework as a structured environment for training and evaluating deep reinforcement learning (RL) algorithms, opening new possibilities for optimizing aDBS control strategies and inviting the machine learning community to contribute to the emerging field of intelligent neurostimulation interfaces. 

---
# Predictive Models for Chronic Heart Failure 

**Authors**: Pietro Cassieri, Aiman Faiz, Anna Maria De Roberto, Claudio Pascarelli, Gianvito Mitrano, Gianluca Fimiani, Marina Garofano, Christiancarmine Esposito, Genoveffa Tortora, Alessia Bramanti, Giuseppe Scanniello  

**Link**: [PDF](https://arxiv.org/pdf/2505.09619)  

**Abstract**: The management of chronic Heart Failure (HF) presents significant challenges in modern healthcare, requiring continuous monitoring, early detection of exacerbations, and personalized treatment strategies. In this paper, we present a predictive model founded on Machine Learning (ML) techniques to identify patients at HF risk. This model is an ensemble learning approach, a modified stacking technique, that uses two specialized models leveraging clinical and echocardiographic features and then a meta-model to combine the predictions of these two models. We initially assess the model on a real dataset and the obtained results suggest that it performs well in the stratification of patients at HR risk. Specifically, we obtained high sensitivity (95\%), ensuring that nearly all high-risk patients are identified. As for accuracy, we obtained 84\%, which can be considered moderate in some ML contexts. However, it is acceptable given our priority of identifying patients at risk of HF because they will be asked to participate in the telemonitoring program of the PrediHealth research project on which some of the authors of this paper are working. The initial findings also suggest that ML-based risk stratification models can serve as valuable decision-support tools not only in the PrediHealth project but also for healthcare professionals, aiding in early intervention and personalized patient management. To have a better understanding of the value and of potentiality of our predictive model, we also contrasted its results with those obtained by using three baseline models. The preliminary results indicate that our predictive model outperforms these baselines that flatly consider features, \ie not grouping them in clinical and echocardiographic features. 

---
# SpecWav-Attack: Leveraging Spectrogram Resizing and Wav2Vec 2.0 for Attacking Anonymized Speech 

**Authors**: Yuqi Li, Yuanzhong Zheng, Zhongtian Guo, Yaoxuan Wang, Jianjun Yin, Haojun Fei  

**Link**: [PDF](https://arxiv.org/pdf/2505.09616)  

**Abstract**: This paper presents SpecWav-Attack, an adversarial model for detecting speakers in anonymized speech. It leverages Wav2Vec2 for feature extraction and incorporates spectrogram resizing and incremental training for improved performance. Evaluated on librispeech-dev and librispeech-test, SpecWav-Attack outperforms conventional attacks, revealing vulnerabilities in anonymized speech systems and emphasizing the need for stronger defenses, benchmarked against the ICASSP 2025 Attacker Challenge. 

---
# Online Isolation Forest 

**Authors**: Filippo Leveni, Guilherme Weigert Cassales, Bernhard Pfahringer, Albert Bifet, Giacomo Boracchi  

**Link**: [PDF](https://arxiv.org/pdf/2505.09593)  

**Abstract**: The anomaly detection literature is abundant with offline methods, which require repeated access to data in memory, and impose impractical assumptions when applied to a streaming context. Existing online anomaly detection methods also generally fail to address these constraints, resorting to periodic retraining to adapt to the online context. We propose Online-iForest, a novel method explicitly designed for streaming conditions that seamlessly tracks the data generating process as it evolves over time. Experimental validation on real-world datasets demonstrated that Online-iForest is on par with online alternatives and closely rivals state-of-the-art offline anomaly detection techniques that undergo periodic retraining. Notably, Online-iForest consistently outperforms all competitors in terms of efficiency, making it a promising solution in applications where fast identification of anomalies is of primary importance such as cybersecurity, fraud and fault detection. 

---
# AI and Generative AI Transforming Disaster Management: A Survey of Damage Assessment and Response Techniques 

**Authors**: Aman Raj, Lakshit Arora, Sanjay Surendranath Girija, Shashank Kapoor, Dipen Pradhan, Ankit Shetgaonkar  

**Link**: [PDF](https://arxiv.org/pdf/2505.08202)  

**Abstract**: Natural disasters, including earthquakes, wildfires and cyclones, bear a huge risk on human lives as well as infrastructure assets. An effective response to disaster depends on the ability to rapidly and efficiently assess the intensity of damage. Artificial Intelligence (AI) and Generative Artificial Intelligence (GenAI) presents a breakthrough solution, capable of combining knowledge from multiple types and sources of data, simulating realistic scenarios of disaster, and identifying emerging trends at a speed previously unimaginable. In this paper, we present a comprehensive review on the prospects of AI and GenAI in damage assessment for various natural disasters, highlighting both its strengths and limitations. We talk about its application to multimodal data such as text, image, video, and audio, and also cover major issues of data privacy, security, and ethical use of the technology during crises. The paper also recognizes the threat of Generative AI misuse, in the form of dissemination of misinformation and for adversarial attacks. Finally, we outline avenues of future research, emphasizing the need for secure, reliable, and ethical Generative AI systems for disaster management in general. We believe that this work represents the first comprehensive survey of Gen-AI techniques being used in the field of Disaster Assessment and Response. 

---
# Adversarial Attacks in Multimodal Systems: A Practitioner's Survey 

**Authors**: Shashank Kapoor, Sanjay Surendranath Girija, Lakshit Arora, Dipen Pradhan, Ankit Shetgaonkar, Aman Raj  

**Link**: [PDF](https://arxiv.org/pdf/2505.03084)  

**Abstract**: The introduction of multimodal models is a huge step forward in Artificial Intelligence. A single model is trained to understand multiple modalities: text, image, video, and audio. Open-source multimodal models have made these breakthroughs more accessible. However, considering the vast landscape of adversarial attacks across these modalities, these models also inherit vulnerabilities of all the modalities, and ultimately, the adversarial threat amplifies. While broad research is available on possible attacks within or across these modalities, a practitioner-focused view that outlines attack types remains absent in the multimodal world. As more Machine Learning Practitioners adopt, fine-tune, and deploy open-source models in real-world applications, it's crucial that they can view the threat landscape and take the preventive actions necessary. This paper addresses the gap by surveying adversarial attacks targeting all four modalities: text, image, video, and audio. This survey provides a view of the adversarial attack landscape and presents how multimodal adversarial threats have evolved. To the best of our knowledge, this survey is the first comprehensive summarization of the threat landscape in the multimodal world. 

---
# Change Detection in Multivariate data streams: Online Analysis with Kernel-QuantTree 

**Authors**: Michelangelo Olmo Nogara Notarianni, Filippo Leveni, Diego Stucchi, Luca Frittoli, Giacomo Boracchi  

**Link**: [PDF](https://arxiv.org/pdf/2410.13778)  

**Abstract**: We present Kernel-QuantTree Exponentially Weighted Moving Average (KQT-EWMA), a non-parametric change-detection algorithm that combines the Kernel-QuantTree (KQT) histogram and the EWMA statistic to monitor multivariate data streams online. The resulting monitoring scheme is very flexible, since histograms can be used to model any stationary distribution, and practical, since the distribution of test statistics does not depend on the distribution of datastream in stationary conditions (non-parametric monitoring). KQT-EWMA enables controlling false alarms by operating at a pre-determined Average Run Length ($ARL_0$), which measures the expected number of stationary samples to be monitored before triggering a false alarm. The latter peculiarity is in contrast with most non-parametric change-detection tests, which rarely can control the $ARL_0$ a priori. Our experiments on synthetic and real-world datasets demonstrate that KQT-EWMA can control $ARL_0$ while achieving detection delays comparable to or lower than state-of-the-art methods designed to work in the same conditions. 

---
# UOD: Universal One-shot Detection of Anatomical Landmarks 

**Authors**: Heqin Zhu, Quan Quan, Qingsong Yao, Zaiyi Liu, S. Kevin Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2306.07615)  

**Abstract**: One-shot medical landmark detection gains much attention and achieves great success for its label-efficient training process. However, existing one-shot learning methods are highly specialized in a single domain and suffer domain preference heavily in the situation of multi-domain unlabeled data. Moreover, one-shot learning is not robust that it faces performance drop when annotating a sub-optimal image. To tackle these issues, we resort to developing a domain-adaptive one-shot landmark detection framework for handling multi-domain medical images, named Universal One-shot Detection (UOD). UOD consists of two stages and two corresponding universal models which are designed as combinations of domain-specific modules and domain-shared modules. In the first stage, a domain-adaptive convolution model is self-supervised learned to generate pseudo landmark labels. In the second stage, we design a domain-adaptive transformer to eliminate domain preference and build the global context for multi-domain data. Even though only one annotated sample from each domain is available for training, the domain-shared modules help UOD aggregate all one-shot samples to detect more robust and accurate landmarks. We investigated both qualitatively and quantitatively the proposed UOD on three widely-used public X-ray datasets in different anatomical domains (i.e., head, hand, chest) and obtained state-of-the-art performances in each domain. The code is available at this https URL. 

---
