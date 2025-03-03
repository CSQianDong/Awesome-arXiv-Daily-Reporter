# Contextualizing biological perturbation experiments through language 

**Authors**: Menghua Wu, Russell Littman, Jacob Levine, Lin Qiu, Tommaso Biancalani, David Richmond, Jan-Christian Huetter  

**Link**: [PDF](https://arxiv.org/pdf/2502.21290)  

**Abstract**: High-content perturbation experiments allow scientists to probe biomolecular systems at unprecedented resolution, but experimental and analysis costs pose significant barriers to widespread adoption. Machine learning has the potential to guide efficient exploration of the perturbation space and extract novel insights from these data. However, current approaches neglect the semantic richness of the relevant biology, and their objectives are misaligned with downstream biological analyses. In this paper, we hypothesize that large language models (LLMs) present a natural medium for representing complex biological relationships and rationalizing experimental outcomes. We propose PerturbQA, a benchmark for structured reasoning over perturbation experiments. Unlike current benchmarks that primarily interrogate existing knowledge, PerturbQA is inspired by open problems in perturbation modeling: prediction of differential expression and change of direction for unseen perturbations, and gene set enrichment. We evaluate state-of-the-art machine learning and statistical approaches for modeling perturbations, as well as standard LLM reasoning strategies, and we find that current methods perform poorly on PerturbQA. As a proof of feasibility, we introduce Summer (SUMMarize, retrievE, and answeR, a simple, domain-informed LLM framework that matches or exceeds the current state-of-the-art. Our code and data are publicly available at this https URL. 

---
# Modeling Human Beliefs about AI Behavior for Scalable Oversight 

**Authors**: Leon Lang, Patrick Forré  

**Link**: [PDF](https://arxiv.org/pdf/2502.21262)  

**Abstract**: Contemporary work in AI alignment often relies on human feedback to teach AI systems human preferences and values. Yet as AI systems grow more capable, human feedback becomes increasingly unreliable. This raises the problem of scalable oversight: How can we supervise AI systems that exceed human capabilities? In this work, we propose to model the human evaluator's beliefs about the AI system's behavior to better interpret the human's feedback. We formalize human belief models and theoretically analyze their role in inferring human values. We then characterize the remaining ambiguity in this inference and conditions for which the ambiguity disappears. To mitigate reliance on exact belief models, we then introduce the relaxation of human belief model covering. Finally, we propose using foundation models to construct covering belief models, providing a new potential approach to scalable oversight. 

---
# Towards Developing Ethical Reasoners: Integrating Probabilistic Reasoning and Decision-Making for Complex AI Systems 

**Authors**: Nijesh Upreti, Jessica Ciupa, Vaishak Belle  

**Link**: [PDF](https://arxiv.org/pdf/2502.21250)  

**Abstract**: A computational ethics framework is essential for AI and autonomous systems operating in complex, real-world environments. Existing approaches often lack the adaptability needed to integrate ethical principles into dynamic and ambiguous contexts, limiting their effectiveness across diverse scenarios. To address these challenges, we outline the necessary ingredients for building a holistic, meta-level framework that combines intermediate representations, probabilistic reasoning, and knowledge representation. The specifications therein emphasize scalability, supporting ethical reasoning at both individual decision-making levels and within the collective dynamics of multi-agent systems. By integrating theoretical principles with contextual factors, it facilitates structured and context-aware decision-making, ensuring alignment with overarching ethical standards. We further explore proposed theorems outlining how ethical reasoners should operate, offering a foundation for practical implementation. These constructs aim to support the development of robust and ethically reliable AI systems capable of navigating the complexities of real-world moral decision-making scenarios. 

---
# Transforming Tuberculosis Care: Optimizing Large Language Models For Enhanced Clinician-Patient Communication 

**Authors**: Daniil Filienko, Mahek Nizar, Javier Roberti, Denise Galdamez, Haroon Jakher, Sarah Iribarren, Weichao Yuwen, Martine De Cock  

**Link**: [PDF](https://arxiv.org/pdf/2502.21236)  

**Abstract**: Tuberculosis (TB) is the leading cause of death from an infectious disease globally, with the highest burden in low- and middle-income countries. In these regions, limited healthcare access and high patient-to-provider ratios impede effective patient support, communication, and treatment completion. To bridge this gap, we propose integrating a specialized Large Language Model into an efficacious digital adherence technology to augment interactive communication with treatment supporters. This AI-powered approach, operating within a human-in-the-loop framework, aims to enhance patient engagement and improve TB treatment outcomes. 

---
# An Algebraic Framework for Hierarchical Probabilistic Abstraction 

**Authors**: Nijesh Upreti, Vaishak Belle  

**Link**: [PDF](https://arxiv.org/pdf/2502.21216)  

**Abstract**: Abstraction is essential for reducing the complexity of systems across diverse fields, yet designing effective abstraction methodology for probabilistic models is inherently challenging due to stochastic behaviors and uncertainties. Current approaches often distill detailed probabilistic data into higher-level summaries to support tractable and interpretable analyses, though they typically struggle to fully represent the relational and probabilistic hierarchies through single-layered abstractions. We introduce a hierarchical probabilistic abstraction framework aimed at addressing these challenges by extending a measure-theoretic foundation for hierarchical abstraction. The framework enables modular problem-solving via layered mappings, facilitating both detailed layer-specific analysis and a cohesive system-wide understanding. This approach bridges high-level conceptualization with low-level perceptual data, enhancing interpretability and allowing layered analysis. Our framework provides a robust foundation for abstraction analysis across AI subfields, particularly in aligning System 1 and System 2 thinking, thereby supporting the development of diverse abstraction methodologies. 

---
# ARIES: Autonomous Reasoning with LLMs on Interactive Thought Graph Environments 

**Authors**: Pedro Gimenes, Zeyu Cao, Jeffrey Wong, Yiren Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2502.21208)  

**Abstract**: Recent research has shown that LLM performance on reasoning tasks can be enhanced by scaling test-time compute. One promising approach, particularly with decomposable problems, involves arranging intermediate solutions as a graph on which transformations are performed to explore the solution space. However, prior works rely on pre-determined, task-specific transformation schedules which are subject to a set of searched hyperparameters. In this work, we view thought graph transformations as actions in a Markov decision process, and implement policy agents to drive effective action policies for the underlying reasoning LLM agent. In particular, we investigate the ability for another LLM to act as a policy agent on thought graph environments and introduce ARIES, a multi-agent architecture for reasoning with LLMs. In ARIES, reasoning LLM agents solve decomposed subproblems, while policy LLM agents maintain visibility of the thought graph states, and dynamically adapt the problem-solving strategy. Through extensive experiments, we observe that using off-the-shelf LLMs as policy agents with no supervised fine-tuning (SFT) can yield up to $29\%$ higher accuracy on HumanEval relative to static transformation schedules, as well as reducing inference costs by $35\%$ and avoid any search requirements. We also conduct a thorough analysis of observed failure modes, highlighting that limitations on LLM sizes and the depth of problem decomposition can be seen as challenges to scaling LLM-guided reasoning. 

---
# A Survey of Link Prediction in Temporal Networks 

**Authors**: Jiafeng Xiong, Ahmad Zareie, Rizos Sakellariou  

**Link**: [PDF](https://arxiv.org/pdf/2502.21185)  

**Abstract**: Temporal networks have gained significant prominence in the past decade for modelling dynamic interactions within complex systems. A key challenge in this domain is Temporal Link Prediction (TLP), which aims to forecast future connections by analysing historical network structures across various applications including social network analysis. While existing surveys have addressed specific aspects of TLP, they typically lack a comprehensive framework that distinguishes between representation and inference methods. This survey bridges this gap by introducing a novel taxonomy that explicitly examines representation and inference from existing methods, providing a novel classification of approaches for TLP. We analyse how different representation techniques capture temporal and structural dynamics, examining their compatibility with various inference methods for both transductive and inductive prediction tasks. Our taxonomy not only clarifies the methodological landscape but also reveals promising unexplored combinations of existing techniques. This taxonomy provides a systematic foundation for emerging challenges in TLP, including model explainability and scalable architectures for complex temporal networks. 

---
# Multimodal Dreaming: A Global Workspace Approach to World Model-Based Reinforcement Learning 

**Authors**: Léopold Maytié, Roland Bertin Johannet, Rufin VanRullen  

**Link**: [PDF](https://arxiv.org/pdf/2502.21142)  

**Abstract**: Humans leverage rich internal models of the world to reason about the future, imagine counterfactuals, and adapt flexibly to new situations. In Reinforcement Learning (RL), world models aim to capture how the environment evolves in response to the agent's actions, facilitating planning and generalization. However, typical world models directly operate on the environment variables (e.g. pixels, physical attributes), which can make their training slow and cumbersome; instead, it may be advantageous to rely on high-level latent dimensions that capture relevant multimodal variables. Global Workspace (GW) Theory offers a cognitive framework for multimodal integration and information broadcasting in the brain, and recent studies have begun to introduce efficient deep learning implementations of GW. Here, we evaluate the capabilities of an RL system combining GW with a world model. We compare our GW-Dreamer with various versions of the standard PPO and the original Dreamer algorithms. We show that performing the dreaming process (i.e., mental simulation) inside the GW latent space allows for training with fewer environment steps. As an additional emergent property, the resulting model (but not its comparison baselines) displays strong robustness to the absence of one of its observation modalities (images or simulation attributes). We conclude that the combination of GW with World Models holds great potential for improving decision-making in RL agents. 

---
# Optimizing Large Language Models for ESG Activity Detection in Financial Texts 

**Authors**: Mattia Birti, Francesco Osborne, Andrea Maurino  

**Link**: [PDF](https://arxiv.org/pdf/2502.21112)  

**Abstract**: The integration of Environmental, Social, and Governance (ESG) factors into corporate decision-making is a fundamental aspect of sustainable finance. However, ensuring that business practices align with evolving regulatory frameworks remains a persistent challenge. AI-driven solutions for automatically assessing the alignment of sustainability reports and non-financial disclosures with specific ESG activities could greatly support this process. Yet, this task remains complex due to the limitations of general-purpose Large Language Models (LLMs) in domain-specific contexts and the scarcity of structured, high-quality datasets. In this paper, we investigate the ability of current-generation LLMs to identify text related to environmental activities. Furthermore, we demonstrate that their performance can be significantly enhanced through fine-tuning on a combination of original and synthetically generated data. To this end, we introduce ESG-Activities, a benchmark dataset containing 1,325 labelled text segments classified according to the EU ESG taxonomy. Our experimental results show that fine-tuning on ESG-Activities significantly enhances classification accuracy, with open models such as Llama 7B and Gemma 7B outperforming large proprietary solutions in specific configurations. These findings have important implications for financial analysts, policymakers, and AI researchers seeking to enhance ESG transparency and compliance through advanced natural language processing techniques. 

---
# Re-evaluating Theory of Mind evaluation in large language models 

**Authors**: Jennifer Hu, Felix Sosa, Tomer Ullman  

**Link**: [PDF](https://arxiv.org/pdf/2502.21098)  

**Abstract**: The question of whether large language models (LLMs) possess Theory of Mind (ToM) -- often defined as the ability to reason about others' mental states -- has sparked significant scientific and public interest. However, the evidence as to whether LLMs possess ToM is mixed, and the recent growth in evaluations has not resulted in a convergence. Here, we take inspiration from cognitive science to re-evaluate the state of ToM evaluation in LLMs. We argue that a major reason for the disagreement on whether LLMs have ToM is a lack of clarity on whether models should be expected to match human behaviors, or the computations underlying those behaviors. We also highlight ways in which current evaluations may be deviating from "pure" measurements of ToM abilities, which also contributes to the confusion. We conclude by discussing several directions for future research, including the relationship between ToM and pragmatic communication, which could advance our understanding of artificial systems as well as human cognition. 

---
# An LLM-based Delphi Study to Predict GenAI Evolution 

**Authors**: Francesco Bertolotti, Luca Mari  

**Link**: [PDF](https://arxiv.org/pdf/2502.21092)  

**Abstract**: Predicting the future trajectory of complex and rapidly evolving systems remains a significant challenge, particularly in domains where data is scarce or unreliable. This study introduces a novel approach to qualitative forecasting by leveraging Large Language Models to conduct Delphi studies. The methodology was applied to explore the future evolution of Generative Artificial Intelligence, revealing insights into key factors such as geopolitical tensions, economic disparities, regulatory frameworks, and ethical considerations. The results highlight how LLM-based Delphi studies can facilitate structured scenario analysis, capturing diverse perspectives while mitigating issues such as respondent fatigue. However, limitations emerge in terms of knowledge cutoffs, inherent biases, and sensitivity to initial conditions. While the approach provides an innovative means for structured foresight, this method could be also considered as a novel form of reasoning. further research is needed to refine its ability to manage heterogeneity, improve reliability, and integrate external data sources. 

---
# Are foundation models useful feature extractors for electroencephalography analysis? 

**Authors**: Özgün Turgut, Felix S. Bott, Markus Ploner, Daniel Rueckert  

**Link**: [PDF](https://arxiv.org/pdf/2502.21086)  

**Abstract**: The success of foundation models in natural language processing and computer vision has motivated similar approaches for general time series analysis. While these models are effective for a variety of tasks, their applicability in medical domains with limited data remains largely unexplored. To address this, we investigate the effectiveness of foundation models in medical time series analysis involving electroencephalography (EEG). Through extensive experiments on tasks such as age prediction, seizure detection, and the classification of clinically relevant EEG events, we compare their diagnostic accuracy with that of specialised EEG models. Our analysis shows that foundation models extract meaningful EEG features, outperform specialised models even without domain adaptation, and localise task-specific biomarkers. Moreover, we demonstrate that diagnostic accuracy is substantially influenced by architectural choices such as context length. Overall, our study reveals that foundation models with general time series understanding eliminate the dependency on large domain-specific datasets, making them valuable tools for clinical practice. 

---
# Merging Clinical Knowledge into Large Language Models for Medical Research and Applications: A Survey 

**Authors**: Qiyuan Li, Haijiang Liu, Caicai Guo, Deyu Chen, Meng Wang, Feng Gao, Jinguang Gu  

**Link**: [PDF](https://arxiv.org/pdf/2502.20988)  

**Abstract**: Clinical knowledge is the collection of information learned from studies on the causes, prognosis, diagnosis, and treatment of diseases. This type of knowledge can improve curing performances, and promote physical health. With the emergence of large language models (LLMs), medical artificial intelligence (medical AI), which aims to apply academic medical AI systems to real-world medical scenarios, has entered a new age of development, resulting in excellent works such as DoctorGPT and Pangu-Drug from academic and industrial researches. However, the field lacks a comprehensive compendium and comparison of building medical AI systems from academia and industry. Therefore, this survey focuses on the building paradigms of medical AI systems including the use of clinical databases, datasets, training pipelines, integrating medical knowledge graphs, system applications, and evaluation systems. We hope that this survey can help relevant practical researchers understand the current performance of academic models in various fields of healthcare, as well as the potential problems and future directions for implementing these scientific achievements. 

---
# A Pilot Empirical Study on When and How to Use Knowledge Graphs as Retrieval Augmented Generation 

**Authors**: Xujie Yuan, Yongxu Liu, Shimin Di, Shiwen Wu, Libin Zheng, Rui Meng, Xiaofang Zhou, Lei Chen, Jian Yin  

**Link**: [PDF](https://arxiv.org/pdf/2502.20854)  

**Abstract**: The integration of Knowledge Graphs (KGs) into the Retrieval Augmented Generation (RAG) framework has attracted significant interest, with early studies showing promise in mitigating hallucinations and improving model accuracy. However, a systematic understanding and comparative analysis of the rapidly emerging KG-RAG methods are still lacking. This paper seeks to lay the foundation for systematically answering the question of when and how to use KG-RAG by analyzing their performance in various application scenarios associated with different technical configurations. After outlining the mind map using KG-RAG framework and summarizing its popular pipeline, we conduct a pilot empirical study of KG-RAG works to reimplement and evaluate 6 KG-RAG methods across 7 datasets in diverse scenarios, analyzing the impact of 9 KG-RAG configurations in combination with 17 LLMs. Our results underscore the critical role of appropriate application conditions and optimal configurations of KG-RAG components. 

---
# MV-MATH: Evaluating Multimodal Math Reasoning in Multi-Visual Contexts 

**Authors**: Peijie Wang, Zhongzhi Li, Fei Yin, Dekang Ran, Chenglin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.20808)  

**Abstract**: Multimodal Large Language Models (MLLMs) have shown promising capabilities in mathematical reasoning within visual contexts across various datasets. However, most existing multimodal math benchmarks are limited to single-visual contexts, which diverges from the multi-visual scenarios commonly encountered in real-world mathematical applications. To address this gap, we introduce MV-MATH: a meticulously curated dataset of 2,009 high-quality mathematical problems. Each problem integrates multiple images interleaved with text, derived from authentic K-12 scenarios, and enriched with detailed annotations. MV-MATH includes multiple-choice, free-form, and multi-step questions, covering 11 subject areas across 3 difficulty levels, and serves as a comprehensive and rigorous benchmark for assessing MLLMs' mathematical reasoning in multi-visual contexts. Through extensive experimentation, we observe that MLLMs encounter substantial challenges in multi-visual math tasks, with a considerable performance gap relative to human capabilities on MV-MATH. Furthermore, we analyze the performance and error patterns of various models, providing insights into MLLMs' mathematical reasoning capabilities within multi-visual settings. 

---
# MedHallTune: An Instruction-Tuning Benchmark for Mitigating Medical Hallucination in Vision-Language Models 

**Authors**: Qiao Yan, Yuchen Yuan, Xiaowei Hu, Yihan Wang, Jiaqi Xu, Jinpeng Li, Chi-Wing Fu, Pheng-Ann Heng  

**Link**: [PDF](https://arxiv.org/pdf/2502.20780)  

**Abstract**: The increasing use of vision-language models (VLMs) in healthcare applications presents great challenges related to hallucinations, in which the models may generate seemingly plausible results that are in fact incorrect. Such hallucinations can jeopardize clinical decision making, potentially harming the diagnosis and treatments. In this work, we propose MedHallTune, a large-scale benchmark designed specifically to evaluate and mitigate hallucinations in medical VLMs. Comprising over 100,000 images and 1,000,000 instruction pairs, MedHallTune includes both hallucination and non-hallucination samples, each with ground-truth annotations. We conduct a comprehensive evaluation of current medical and general VLMs using MedHallTune, assessing their performance across key metrics, including clinical accuracy, relevance, detail level, and risk level. The experimental results show that fine-tuning with MedHallTune successfully improves the ability of several existing models to manage hallucinations and boost their zero-shot performance on downstream visual-question-answering (VQA) tasks, making them more reliable for practical medical applications. Our work contributes to the development of more trustworthy VLMs. Codes and dataset will be available at \href{this https URL}{MedHallTune}. 

---
# Damper-B-PINN: Damper Characteristics-Based Bayesian Physics-Informed Neural Network for Vehicle State Estimation 

**Authors**: Tianyi Zeng, Tianyi Wang, Junfeng Jiao, Xinbo Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.20772)  

**Abstract**: State estimation for Multi-Input Multi-Output (MIMO) systems with noise, such as vehicle chassis systems, presents a significant challenge due to the imperfect and complex relationship between inputs and outputs. To solve this problem, we design a Damper characteristics-based Bayesian Physics-Informed Neural Network (Damper-B-PINN). First, we introduce a neuron forward process inspired by the mechanical properties of dampers, which limits abrupt jumps in neuron values between epochs while maintaining search capability. Additionally, we apply an optimized Bayesian dropout layer to the MIMO system to enhance robustness against noise and prevent non-convergence issues. Physical information is incorporated into the loss function to serve as a physical prior for the neural network. The effectiveness of our Damper-B-PINN architecture is then validated across ten datasets and fourteen vehicle types, demonstrating superior accuracy, computational efficiency, and convergence in vehicle state estimation (i.e., dynamic wheel load) compared to other state-of-the-art benchmarks. 

---
# Acquiring Grounded Representations of Words with Situated Interactive Instruction 

**Authors**: Shiwali Mohan, Aaron H. Mininger, James R. Kirk, John E. Laird  

**Link**: [PDF](https://arxiv.org/pdf/2502.20754)  

**Abstract**: We present an approach for acquiring grounded representations of words from mixed-initiative, situated interactions with a human instructor. The work focuses on the acquisition of diverse types of knowledge including perceptual, semantic, and procedural knowledge along with learning grounded meanings. Interactive learning allows the agent to control its learning by requesting instructions about unknown concepts, making learning efficient. Our approach has been instantiated in Soar and has been evaluated on a table-top robotic arm capable of manipulating small objects. 

---
# DeepSolution: Boosting Complex Engineering Solution Design via Tree-based Exploration and Bi-point Thinking 

**Authors**: Zhuoqun Li, Haiyang Yu, Xuanang Chen, Hongyu Lin, Yaojie Lu, Fei Huang, Xianpei Han, Yongbin Li, Le Sun  

**Link**: [PDF](https://arxiv.org/pdf/2502.20730)  

**Abstract**: Designing solutions for complex engineering challenges is crucial in human production activities. However, previous research in the retrieval-augmented generation (RAG) field has not sufficiently addressed tasks related to the design of complex engineering solutions. To fill this gap, we introduce a new benchmark, SolutionBench, to evaluate a system's ability to generate complete and feasible solutions for engineering problems with multiple complex constraints. To further advance the design of complex engineering solutions, we propose a novel system, SolutionRAG, that leverages the tree-based exploration and bi-point thinking mechanism to generate reliable solutions. Extensive experimental results demonstrate that SolutionRAG achieves state-of-the-art (SOTA) performance on the SolutionBench, highlighting its potential to enhance the automation and reliability of complex engineering solution design in real-world applications. 

---
# Fuzzy Speculative Decoding for a Tunable Accuracy-Runtime Tradeoff 

**Authors**: Maximilian Holsman, Yukun Huang, Bhuwan Dhingra  

**Link**: [PDF](https://arxiv.org/pdf/2502.20704)  

**Abstract**: Speculative Decoding (SD) enforces strict distributional equivalence to the target model, limiting potential speed ups as distributions of near-equivalence achieve comparable outcomes in many cases. Furthermore, enforcing distributional equivalence means that users are unable to trade deviations from the target model distribution for further inference speed gains. To address these limitations, we introduce Fuzzy Speculative Decoding (FSD) - a decoding algorithm that generalizes SD by accepting candidate tokens purely based on the divergences between the target and draft model distributions. By allowing for controlled divergence from the target model, FSD enables users to flexibly trade generation quality for inference speed. Across several benchmarks, our method is able to achieve significant runtime improvements of over 5 tokens per second faster than SD at only an approximate 2% absolute reduction in benchmark accuracy. In many cases, FSD is even able to match SD benchmark accuracy at over 2 tokens per second faster, demonstrating that distributional equivalence is not necessary to maintain target model performance. 

---
# Why Trust in AI May Be Inevitable 

**Authors**: Nghi Truong, Phanish Puranam, Ilia Testlin  

**Link**: [PDF](https://arxiv.org/pdf/2502.20701)  

**Abstract**: In human-AI interactions, explanation is widely seen as necessary for enabling trust in AI systems. We argue that trust, however, may be a pre-requisite because explanation is sometimes impossible. We derive this result from a formalization of explanation as a search process through knowledge networks, where explainers must find paths between shared concepts and the concept to be explained, within finite time. Our model reveals that explanation can fail even under theoretically ideal conditions - when actors are rational, honest, motivated, can communicate perfectly, and possess overlapping knowledge. This is because successful explanation requires not just the existence of shared knowledge but also finding the connection path within time constraints, and it can therefore be rational to cease attempts at explanation before the shared knowledge is discovered. This result has important implications for human-AI interaction: as AI systems, particularly Large Language Models, become more sophisticated and able to generate superficially compelling but spurious explanations, humans may default to trust rather than demand genuine explanations. This creates risks of both misplaced trust and imperfect knowledge integration. 

---
# ProAI: Proactive Multi-Agent Conversational AI with Structured Knowledge Base for Psychiatric Diagnosis 

**Authors**: Yuqi Wu, Guangya Wan, Jingjing Li, Shengming Zhao, Lingfeng Ma, Tianyi Ye, Ion Pop, Yanbo Zhang, Jie Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.20689)  

**Abstract**: Most LLM-driven conversational AI systems operate reactively, responding to user prompts without guiding the interaction. Most LLM-driven conversational AI systems operate reactively, responding to user prompts without guiding the interaction. However, many real-world applications-such as psychiatric diagnosis, consulting, and interviews-require AI to take a proactive role, asking the right questions and steering conversations toward specific objectives. Using mental health differential diagnosis as an application context, we introduce ProAI, a goal-oriented, proactive conversational AI framework. ProAI integrates structured knowledge-guided memory, multi-agent proactive reasoning, and a multi-faceted evaluation strategy, enabling LLMs to engage in clinician-style diagnostic reasoning rather than simple response generation. Through simulated patient interactions, user experience assessment, and professional clinical validation, we demonstrate that ProAI achieves up to 83.3% accuracy in mental disorder differential diagnosis while maintaining professional and empathetic interaction standards. These results highlight the potential for more reliable, adaptive, and goal-driven AI diagnostic assistants, advancing LLMs beyond reactive dialogue systems. 

---
# Automatic database description generation for Text-to-SQL 

**Authors**: Yingqi Gao, Zhiling Luo  

**Link**: [PDF](https://arxiv.org/pdf/2502.20657)  

**Abstract**: In the context of the Text-to-SQL task, table and column descriptions are crucial for bridging the gap between natural language and database schema. This report proposes a method for automatically generating effective database descriptions when explicit descriptions are unavailable. The proposed method employs a dual-process approach: a coarse-to-fine process, followed by a fine-to-coarse process. The coarse-to-fine approach leverages the inherent knowledge of LLM to guide the understanding process from databases to tables and finally to columns. This approach provides a holistic understanding of the database structure and ensures contextual alignment. Conversely, the fine-to-coarse approach starts at the column level, offering a more accurate and nuanced understanding when stepping back to the table level. Experimental results on the Bird benchmark indicate that using descriptions generated by the proposed improves SQL generation accuracy by 0.93\% compared to not using descriptions, and achieves 37\% of human-level performance. The source code is publicly available at this https URL. 

---
# PersonaBench: Evaluating AI Models on Understanding Personal Information through Accessing (Synthetic) Private User Data 

**Authors**: Juntao Tan, Liangwei Yang, Zuxin Liu, Zhiwei Liu, Rithesh Murthy, Tulika Manoj Awalgaonkar, Jianguo Zhang, Weiran Yao, Ming Zhu, Shirley Kokane, Silvio Savarese, Huan Wang, Caiming Xiong, Shelby Heinecke  

**Link**: [PDF](https://arxiv.org/pdf/2502.20616)  

**Abstract**: Personalization is critical in AI assistants, particularly in the context of private AI models that work with individual users. A key scenario in this domain involves enabling AI models to access and interpret a user's private data (e.g., conversation history, user-AI interactions, app usage) to understand personal details such as biographical information, preferences, and social connections. However, due to the sensitive nature of such data, there are no publicly available datasets that allow us to assess an AI model's ability to understand users through direct access to personal information.
To address this gap, we introduce a synthetic data generation pipeline that creates diverse, realistic user profiles and private documents simulating human activities. Leveraging this synthetic data, we present PersonaBench, a benchmark designed to evaluate AI models' performance in understanding personal information derived from simulated private user data.
We evaluate Retrieval-Augmented Generation (RAG) pipelines using questions directly related to a user's personal information, supported by the relevant private documents provided to the models. Our results reveal that current retrieval-augmented AI models struggle to answer private questions by extracting personal information from user documents, highlighting the need for improved methodologies to enhance personalization capabilities in AI. 

---
# NutriGen: Personalized Meal Plan Generator Leveraging Large Language Models to Enhance Dietary and Nutritional Adherence 

**Authors**: Saman Khamesian, Asiful Arefeen, Stephanie M. Carpenter, Hassan Ghasemzadeh  

**Link**: [PDF](https://arxiv.org/pdf/2502.20601)  

**Abstract**: Maintaining a balanced diet is essential for overall health, yet many individuals struggle with meal planning due to nutritional complexity, time constraints, and lack of dietary knowledge. Personalized food recommendations can help address these challenges by tailoring meal plans to individual preferences, habits, and dietary restrictions. However, existing dietary recommendation systems often lack adaptability, fail to consider real-world constraints such as food ingredient availability, and require extensive user input, making them impractical for sustainable and scalable daily use. To address these limitations, we introduce NutriGen, a framework based on large language models (LLM) designed to generate personalized meal plans that align with user-defined dietary preferences and constraints. By building a personalized nutrition database and leveraging prompt engineering, our approach enables LLMs to incorporate reliable nutritional references like the USDA nutrition database while maintaining flexibility and ease-of-use. We demonstrate that LLMs have strong potential in generating accurate and user-friendly food recommendations, addressing key limitations in existing dietary recommendation systems by providing structured, practical, and scalable meal plans. Our evaluation shows that Llama 3.1 8B and GPT-3.5 Turbo achieve the lowest percentage errors of 1.55\% and 3.68\%, respectively, producing meal plans that closely align with user-defined caloric targets while minimizing deviation and improving precision. Additionally, we compared the performance of DeepSeek V3 against several established models to evaluate its potential in personalized nutrition planning. 

---
# On Benchmarking Human-Like Intelligence in Machines 

**Authors**: Lance Ying, Katherine M. Collins, Lionel Wong, Ilia Sucholutsky, Ryan Liu, Adrian Weller, Tianmin Shu, Thomas L. Griffiths, Joshua B. Tenenbaum  

**Link**: [PDF](https://arxiv.org/pdf/2502.20502)  

**Abstract**: Recent benchmark studies have claimed that AI has approached or even surpassed human-level performances on various cognitive tasks. However, this position paper argues that current AI evaluation paradigms are insufficient for assessing human-like cognitive capabilities. We identify a set of key shortcomings: a lack of human-validated labels, inadequate representation of human response variability and uncertainty, and reliance on simplified and ecologically-invalid tasks. We support our claims by conducting a human evaluation study on ten existing AI benchmarks, suggesting significant biases and flaws in task and label designs. To address these limitations, we propose five concrete recommendations for developing future benchmarks that will enable more rigorous and meaningful evaluations of human-like cognitive capacities in AI with various implications for such AI applications. 

---
# R-ParVI: Particle-based variational inference through lens of rewards 

**Authors**: Yongchao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2502.20482)  

**Abstract**: A reward-guided, gradient-free ParVI method, \textit{R-ParVI}, is proposed for sampling partially known densities (e.g. up to a constant). R-ParVI formulates the sampling problem as particle flow driven by rewards: particles are drawn from a prior distribution, navigate through parameter space with movements determined by a reward mechanism blending assessments from the target density, with the steady state particle configuration approximating the target geometry. Particle-environment interactions are simulated by stochastic perturbations and the reward mechanism, which drive particles towards high density regions while maintaining diversity (e.g. preventing from collapsing into clusters). R-ParVI offers fast, flexible, scalable and stochastic sampling and inference for a class of probabilistic models such as those encountered in Bayesian inference and generative modelling. 

---
# Large Language Model Strategic Reasoning Evaluation through Behavioral Game Theory 

**Authors**: Jingru Jia, Zehua Yuan, Junhao Pan, Paul E. McNamara, Deming Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.20432)  

**Abstract**: Strategic decision-making involves interactive reasoning where agents adapt their choices in response to others, yet existing evaluations of large language models (LLMs) often emphasize Nash Equilibrium (NE) approximation, overlooking the mechanisms driving their strategic choices. To bridge this gap, we introduce an evaluation framework grounded in behavioral game theory, disentangling reasoning capability from contextual effects. Testing 22 state-of-the-art LLMs, we find that GPT-o3-mini, GPT-o1, and DeepSeek-R1 dominate most games yet also demonstrate that the model scale alone does not determine performance. In terms of prompting enhancement, Chain-of-Thought (CoT) prompting is not universally effective, as it increases strategic reasoning only for models at certain levels while providing limited gains elsewhere. Additionally, we investigate the impact of encoded demographic features on the models, observing that certain assignments impact the decision-making pattern. For instance, GPT-4o shows stronger strategic reasoning with female traits than males, while Gemma assigns higher reasoning levels to heterosexual identities compared to other sexual orientations, indicating inherent biases. These findings underscore the need for ethical standards and contextual alignment to balance improved reasoning with fairness. 

---
# Beyond transparency: computational reliabilism as an externalist epistemology of algorithms 

**Authors**: Juan Manuel Durán  

**Link**: [PDF](https://arxiv.org/pdf/2502.20402)  

**Abstract**: This chapter is interested in the epistemology of algorithms. As I intend to approach the topic, this is an issue about epistemic justification. Current approaches to justification emphasize the transparency of algorithms, which entails elucidating their internal mechanisms -- such as functions and variables -- and demonstrating how (or that) these produce outputs. Thus, the mode of justification through transparency is contingent on what can be shown about the algorithm and, in this sense, is internal to the algorithm. In contrast, I advocate for an externalist epistemology of algorithms that I term computational reliabilism (CR). While I have previously introduced and examined CR in the field of computer simulations ([42, 53, 4]), this chapter extends this reliabilist epistemology to encompass a broader spectrum of algorithms utilized in various scientific disciplines, with a particular emphasis on machine learning applications. At its core, CR posits that an algorithm's output is justified if it is produced by a reliable algorithm. A reliable algorithm is one that has been specified, coded, used, and maintained utilizing reliability indicators. These reliability indicators stem from formal methods, algorithmic metrics, expert competencies, cultures of research, and other scientific endeavors. The primary aim of this chapter is to delineate the foundations of CR, explicate its operational mechanisms, and outline its potential as an externalist epistemology of algorithms. 

---
# FANformer: Improving Large Language Models Through Effective Periodicity Modeling 

**Authors**: Yihong Dong, Ge Li, Xue Jiang, Yongding Tao, Kechi Zhang, Hao Zhu, Huanyu Liu, Jiazheng Ding, Jia Li, Jinliang Deng, Hong Mei  

**Link**: [PDF](https://arxiv.org/pdf/2502.21309)  

**Abstract**: Periodicity, as one of the most important basic characteristics, lays the foundation for facilitating structured knowledge acquisition and systematic cognitive processes within human learning paradigms. However, the potential flaws of periodicity modeling in Transformer affect the learning efficiency and establishment of underlying principles from data for large language models (LLMs) built upon it. In this paper, we demonstrate that integrating effective periodicity modeling can improve the learning efficiency and performance of LLMs. We introduce FANformer, which integrates Fourier Analysis Network (FAN) into attention mechanism to achieve efficient periodicity modeling, by modifying the feature projection process of attention mechanism. Extensive experimental results on language modeling show that FANformer consistently outperforms Transformer when scaling up model size and training tokens, underscoring its superior learning efficiency. To further validate the effectiveness of FANformer, we pretrain a FANformer-1B on 1 trillion tokens. FANformer-1B exhibits marked improvements on downstream tasks compared to open-source LLMs with similar model parameters or training tokens. The results position FANformer as an effective and promising architecture for advancing LLMs. 

---
# Clustering Context in Off-Policy Evaluation 

**Authors**: Daniel Guzman-Olivares, Philipp Schmidt, Jacek Golebiowski, Artur Bekasov  

**Link**: [PDF](https://arxiv.org/pdf/2502.21304)  

**Abstract**: Off-policy evaluation can leverage logged data to estimate the effectiveness of new policies in e-commerce, search engines, media streaming services, or automatic diagnostic tools in healthcare. However, the performance of baseline off-policy estimators like IPS deteriorates when the logging policy significantly differs from the evaluation policy. Recent work proposes sharing information across similar actions to mitigate this problem. In this work, we propose an alternative estimator that shares information across similar contexts using clustering. We study the theoretical properties of the proposed estimator, characterizing its bias and variance under different conditions. We also compare the performance of the proposed estimator and existing approaches in various synthetic problems, as well as a real-world recommendation dataset. Our experimental results confirm that clustering contexts improves estimation accuracy, especially in deficient information settings. 

---
# L-Lipschitz Gershgorin ResNet Network 

**Authors**: Marius F. R. Juston, William R. Norris, Dustin Nottage, Ahmet Soylemezoglu  

**Link**: [PDF](https://arxiv.org/pdf/2502.21279)  

**Abstract**: Deep residual networks (ResNets) have demonstrated outstanding success in computer vision tasks, attributed to their ability to maintain gradient flow through deep architectures. Simultaneously, controlling the Lipschitz bound in neural networks has emerged as an essential area of research for enhancing adversarial robustness and network certifiability. This paper uses a rigorous approach to design $\mathcal{L}$-Lipschitz deep residual networks using a Linear Matrix Inequality (LMI) framework. The ResNet architecture was reformulated as a pseudo-tri-diagonal LMI with off-diagonal elements and derived closed-form constraints on network parameters to ensure $\mathcal{L}$-Lipschitz continuity. To address the lack of explicit eigenvalue computations for such matrix structures, the Gershgorin circle theorem was employed to approximate eigenvalue locations, guaranteeing the LMI's negative semi-definiteness. Our contributions include a provable parameterization methodology for constructing Lipschitz-constrained networks and a compositional framework for managing recursive systems within hierarchical architectures. These findings enable robust network designs applicable to adversarial robustness, certified training, and control systems. However, a limitation was identified in the Gershgorin-based approximations, which over-constrain the system, suppressing non-linear dynamics and diminishing the network's expressive capacity. 

---
# BAnG: Bidirectional Anchored Generation for Conditional RNA Design 

**Authors**: Roman Klypa, Alberto Bietti, Sergei Grudinin  

**Link**: [PDF](https://arxiv.org/pdf/2502.21274)  

**Abstract**: Designing RNA molecules that interact with specific proteins is a critical challenge in experimental and computational biology. Existing computational approaches require a substantial amount of experimentally determined RNA sequences for each specific protein or a detailed knowledge of RNA structure, restricting their utility in practice. To address this limitation, we develop RNA-BAnG, a deep learning-based model designed to generate RNA sequences for protein interactions without these requirements. Central to our approach is a novel generative method, Bidirectional Anchored Generation (BAnG), which leverages the observation that protein-binding RNA sequences often contain functional binding motifs embedded within broader sequence contexts. We first validate our method on generic synthetic tasks involving similar localized motifs to those appearing in RNAs, demonstrating its benefits over existing generative approaches. We then evaluate our model on biological sequences, showing its effectiveness for conditional RNA sequence design given a binding protein. 

---
# Adaptive Keyframe Sampling for Long Video Understanding 

**Authors**: Xi Tang, Jihao Qiu, Lingxi Xie, Yunjie Tian, Jianbin Jiao, Qixiang Ye  

**Link**: [PDF](https://arxiv.org/pdf/2502.21271)  

**Abstract**: Multimodal large language models (MLLMs) have enabled open-world visual understanding by injecting visual input as extra tokens into large language models (LLMs) as contexts. However, when the visual input changes from a single image to a long video, the above paradigm encounters difficulty because the vast amount of video tokens has significantly exceeded the maximal capacity of MLLMs. Therefore, existing video-based MLLMs are mostly established upon sampling a small portion of tokens from input data, which can cause key information to be lost and thus produce incorrect answers. This paper presents a simple yet effective algorithm named Adaptive Keyframe Sampling (AKS). It inserts a plug-and-play module known as keyframe selection, which aims to maximize the useful information with a fixed number of video tokens. We formulate keyframe selection as an optimization involving (1) the relevance between the keyframes and the prompt, and (2) the coverage of the keyframes over the video, and present an adaptive algorithm to approximate the best solution. Experiments on two long video understanding benchmarks validate that Adaptive Keyframe Sampling improves video QA accuracy (beyond strong baselines) upon selecting informative keyframes. Our study reveals the importance of information pre-filtering in video-based MLLMs. Code is available at this https URL. 

---
# ReaLJam: Real-Time Human-AI Music Jamming with Reinforcement Learning-Tuned Transformers 

**Authors**: Alexander Scarlatos, Yusong Wu, Ian Simon, Adam Roberts, Tim Cooijmans, Natasha Jaques, Cassie Tarakajian, Cheng-Zhi Anna Huang  

**Link**: [PDF](https://arxiv.org/pdf/2502.21267)  

**Abstract**: Recent advances in generative artificial intelligence (AI) have created models capable of high-quality musical content generation. However, little consideration is given to how to use these models for real-time or cooperative jamming musical applications because of crucial required features: low latency, the ability to communicate planned actions, and the ability to adapt to user input in real-time. To support these needs, we introduce ReaLJam, an interface and protocol for live musical jamming sessions between a human and a Transformer-based AI agent trained with reinforcement learning. We enable real-time interactions using the concept of anticipation, where the agent continually predicts how the performance will unfold and visually conveys its plan to the user. We conduct a user study where experienced musicians jam in real-time with the agent through ReaLJam. Our results demonstrate that ReaLJam enables enjoyable and musically interesting sessions, and we uncover important takeaways for future work. 

---
# Supporting the development of Machine Learning for fundamental science in a federated Cloud with the AI_INFN platform 

**Authors**: Lucio Anderlini, Matteo Barbetti, Giulio Bianchini, Diego Ciangottini, Stefano Dal Pra, Diego Michelotto, Carmelo Pellegrino, Rosa Petrini, Alessandro Pascolini, Daniele Spiga  

**Link**: [PDF](https://arxiv.org/pdf/2502.21266)  

**Abstract**: Machine Learning (ML) is driving a revolution in the way scientists design, develop, and deploy data-intensive software. However, the adoption of ML presents new challenges for the computing infrastructure, particularly in terms of provisioning and orchestrating access to hardware accelerators for development, testing, and production. The INFN-funded project AI_INFN ("Artificial Intelligence at INFN") aims at fostering the adoption of ML techniques within INFN use cases by providing support on multiple aspects, including the provision of AI-tailored computing resources. It leverages cloud-native solutions in the context of INFN Cloud, to share hardware accelerators as effectively as possible, ensuring the diversity of the Institute's research activities is not compromised. In this contribution, we provide an update on the commissioning of a Kubernetes platform designed to ease the development of GPU-powered data analysis workflows and their scalability on heterogeneous, distributed computing resources, possibly federated as Virtual Kubelets with the interLink provider. 

---
# Foundation Models -- A Panacea for Artificial Intelligence in Pathology? 

**Authors**: Nita Mulliqi, Anders Blilie, Xiaoyi Ji, Kelvin Szolnoky, Henrik Olsson, Sol Erika Boman, Matteo Titus, Geraldine Martinez Gonzalez, Julia Anna Mielcarz, Masi Valkonen, Einar Gudlaugsson, Svein R. Kjosavik, José Asenjo, Marcello Gambacorta, Paolo Libretti, Marcin Braun, Radzislaw Kordek, Roman Łowicki, Kristina Hotakainen, Päivi Väre, Bodil Ginnerup Pedersen, Karina Dalsgaard Sørensen, Benedicte Parm Ulhøi, Pekka Ruusuvuori, Brett Delahunt, Hemamali Samaratunga, Toyonori Tsuzuki, Emilius A.M. Janssen, Lars Egevad, Martin Eklund, Kimmo Kartasalo  

**Link**: [PDF](https://arxiv.org/pdf/2502.21264)  

**Abstract**: The role of artificial intelligence (AI) in pathology has evolved from aiding diagnostics to uncovering predictive morphological patterns in whole slide images (WSIs). Recently, foundation models (FMs) leveraging self-supervised pre-training have been widely advocated as a universal solution for diverse downstream tasks. However, open questions remain about their clinical applicability and generalization advantages over end-to-end learning using task-specific (TS) models. Here, we focused on AI with clinical-grade performance for prostate cancer diagnosis and Gleason grading. We present the largest validation of AI for this task, using over 100,000 core needle biopsies from 7,342 patients across 15 sites in 11 countries. We compared two FMs with a fully end-to-end TS model in a multiple instance learning framework. Our findings challenge assumptions that FMs universally outperform TS models. While FMs demonstrated utility in data-scarce scenarios, their performance converged with - and was in some cases surpassed by - TS models when sufficient labeled training data were available. Notably, extensive task-specific training markedly reduced clinically significant misgrading, misdiagnosis of challenging morphologies, and variability across different WSI scanners. Additionally, FMs used up to 35 times more energy than the TS model, raising concerns about their sustainability. Our results underscore that while FMs offer clear advantages for rapid prototyping and research, their role as a universal solution for clinically applicable medical AI remains uncertain. For high-stakes clinical applications, rigorous validation and consideration of task-specific training remain critically important. We advocate for integrating the strengths of FMs and end-to-end learning to achieve robust and resource-efficient AI pathology solutions fit for clinical use. 

---
# RuCCoD: Towards Automated ICD Coding in Russian 

**Authors**: Aleksandr Nesterov, Andrey Sakhovskiy, Ivan Sviridov, Airat Valiev, Vladimir Makharev, Petr Anokhin, Galina Zubkova, Elena Tutubalina  

**Link**: [PDF](https://arxiv.org/pdf/2502.21263)  

**Abstract**: This study investigates the feasibility of automating clinical coding in Russian, a language with limited biomedical resources. We present a new dataset for ICD coding, which includes diagnosis fields from electronic health records (EHRs) annotated with over 10,000 entities and more than 1,500 unique ICD codes. This dataset serves as a benchmark for several state-of-the-art models, including BERT, LLaMA with LoRA, and RAG, with additional experiments examining transfer learning across domains (from PubMed abstracts to medical diagnosis) and terminologies (from UMLS concepts to ICD codes). We then apply the best-performing model to label an in-house EHR dataset containing patient histories from 2017 to 2021. Our experiments, conducted on a carefully curated test set, demonstrate that training with the automated predicted codes leads to a significant improvement in accuracy compared to manually annotated data from physicians. We believe our findings offer valuable insights into the potential for automating clinical coding in resource-limited languages like Russian, which could enhance clinical efficiency and data accuracy in these contexts. 

---
# ByteScale: Efficient Scaling of LLM Training with a 2048K Context Length on More Than 12,000 GPUs 

**Authors**: Hao Ge, Junda Feng, Qi Huang, Fangcheng Fu, Xiaonan Nie, Lei Zuo, Haibin Lin, Bin Cui, Xin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.21231)  

**Abstract**: Scaling long-context ability is essential for Large Language Models (LLMs). To amortize the memory consumption across multiple devices in long-context training, inter-data partitioning (a.k.a. Data Parallelism) and intra-data partitioning (a.k.a. Context Parallelism) are commonly used. Current training frameworks predominantly treat the two techniques as orthogonal, and establish static communication groups to organize the devices as a static mesh (e.g., a 2D mesh). However, the sequences for LLM training typically vary in lengths, no matter for texts, multi-modalities or reinforcement learning. The mismatch between data heterogeneity and static mesh causes redundant communication and imbalanced computation, degrading the training efficiency.
In this work, we introduce ByteScale, an efficient, flexible, and scalable LLM training framework for large-scale mixed training of long and short sequences. The core of ByteScale is a novel parallelism strategy, namely Hybrid Data Parallelism (HDP), which unifies the inter- and intra-data partitioning with a dynamic mesh design. In particular, we build a communication optimizer, which eliminates the redundant communication for short sequences by data-aware sharding and dynamic communication, and further compresses the communication cost for long sequences by selective offloading. Besides, we also develop a balance scheduler to mitigate the imbalanced computation by parallelism-aware data assignment. We evaluate ByteScale with the model sizes ranging from 7B to 141B, context lengths from 256K to 2048K, on a production cluster with more than 12,000 GPUs. Experiment results show that ByteScale outperforms the state-of-the-art training system by up to 7.89x. 

---
# ECLeKTic: a Novel Challenge Set for Evaluation of Cross-Lingual Knowledge Transfer 

**Authors**: Omer Goldman, Uri Shaham, Dan Malkin, Sivan Eiger, Avinatan Hassidim, Yossi Matias, Joshua Maynez, Adi Mayrav Gilady, Jason Riesa, Shruti Rijhwani, Laura Rimell, Idan Szpektor, Reut Tsarfaty, Matan Eyal  

**Link**: [PDF](https://arxiv.org/pdf/2502.21228)  

**Abstract**: To achieve equitable performance across languages, multilingual large language models (LLMs) must be able to abstract knowledge beyond the language in which it was acquired. However, the current literature lacks reliable ways to measure LLMs' capability of cross-lingual knowledge transfer. To that end, we present ECLeKTic, a multilingual closed-book QA (CBQA) dataset that Evaluates Cross-Lingual Knowledge Transfer in a simple, black-box manner. We detected information with uneven coverage across languages by controlling for presence and absence of Wikipedia articles in 12 languages. We generated knowledge-seeking questions in a source language, for which the answer appears in a relevant Wikipedia article and translated them to all other 11 languages, for which the respective Wikipedias lack equivalent articles. Assuming that Wikipedia reflects the prominent knowledge in the LLM's training data, to solve ECLeKTic's CBQA task the model is required to transfer knowledge between languages. Experimenting with 8 LLMs, we show that SOTA models struggle to effectively share knowledge across, languages even if they can predict the answer well for queries in the same language the knowledge was acquired in. 

---
# XAIxArts Manifesto: Explainable AI for the Arts 

**Authors**: Nick Bryan-Kinns, Shuoyang Jasper Zheng, Francisco Castro, Makayla Lewis, Jia-Rey Chang, Gabriel Vigliensoni, Terence Broad, Michael Clemens, Elizabeth Wilson  

**Link**: [PDF](https://arxiv.org/pdf/2502.21220)  

**Abstract**: Explainable AI (XAI) is concerned with how to make AI models more understandable to people. To date these explanations have predominantly been technocentric - mechanistic or productivity oriented. This paper introduces the Explainable AI for the Arts (XAIxArts) manifesto to provoke new ways of thinking about explainability and AI beyond technocentric discourses. Manifestos offer a means to communicate ideas, amplify unheard voices, and foster reflection on practice. To supports the co-creation and revision of the XAIxArts manifesto we combine a World Café style discussion format with a living manifesto to question four core themes: 1) Empowerment, Inclusion, and Fairness; 2) Valuing Artistic Practice; 3) Hacking and Glitches; and 4) Openness. Through our interactive living manifesto experience we invite participants to actively engage in shaping this XIAxArts vision within the CHI community and beyond. 

---
# Transformers Learn to Implement Multi-step Gradient Descent with Chain of Thought 

**Authors**: Jianhao Huang, Zixuan Wang, Jason D. Lee  

**Link**: [PDF](https://arxiv.org/pdf/2502.21212)  

**Abstract**: Chain of Thought (CoT) prompting has been shown to significantly improve the performance of large language models (LLMs), particularly in arithmetic and reasoning tasks, by instructing the model to produce intermediate reasoning steps. Despite the remarkable empirical success of CoT and its theoretical advantages in enhancing expressivity, the mechanisms underlying CoT training remain largely unexplored. In this paper, we study the training dynamics of transformers over a CoT objective on an in-context weight prediction task for linear regression. We prove that while a one-layer linear transformer without CoT can only implement a single step of gradient descent (GD) and fails to recover the ground-truth weight vector, a transformer with CoT prompting can learn to perform multi-step GD autoregressively, achieving near-exact recovery. Furthermore, we show that the trained transformer effectively generalizes on the unseen data. With our technique, we also show that looped transformers significantly improve final performance compared to transformers without looping in the in-context learning of linear regression. Empirically, we demonstrate that CoT prompting yields substantial performance improvements. 

---
# The PanAf-FGBG Dataset: Understanding the Impact of Backgrounds in Wildlife Behaviour Recognition 

**Authors**: Otto Brookes, Maksim Kukushkin, Majid Mirmehdi, Colleen Stephens, Paula Dieguez, Thurston C. Hicks, Sorrel Jones, Kevin Lee, Maureen S. McCarthy, Amelia Meier, Emmanuelle Normand, Erin G. Wessling, Roman M.Wittig, Kevin Langergraber, Klaus Zuberbühler, Lukas Boesch, Thomas Schmid, Mimi Arandjelovic, Hjalmar Kühl, Tilo Burghardt  

**Link**: [PDF](https://arxiv.org/pdf/2502.21201)  

**Abstract**: Computer vision analysis of camera trap video footage is essential for wildlife conservation, as captured behaviours offer some of the earliest indicators of changes in population health. Recently, several high-impact animal behaviour datasets and methods have been introduced to encourage their use; however, the role of behaviour-correlated background information and its significant effect on out-of-distribution generalisation remain unexplored. In response, we present the PanAf-FGBG dataset, featuring 20 hours of wild chimpanzee behaviours, recorded at over 350 individual camera locations. Uniquely, it pairs every video with a chimpanzee (referred to as a foreground video) with a corresponding background video (with no chimpanzee) from the same camera location. We present two views of the dataset: one with overlapping camera locations and one with disjoint locations. This setup enables, for the first time, direct evaluation of in-distribution and out-of-distribution conditions, and for the impact of backgrounds on behaviour recognition models to be quantified. All clips come with rich behavioural annotations and metadata including unique camera IDs and detailed textual scene descriptions. Additionally, we establish several baselines and present a highly effective latent-space normalisation technique that boosts out-of-distribution performance by +5.42% mAP for convolutional and +3.75% mAP for transformer-based models. Finally, we provide an in-depth analysis on the role of backgrounds in out-of-distribution behaviour recognition, including the so far unexplored impact of background durations (i.e., the count of background frames within foreground videos). 

---
# AMPLE: Event-Driven Accelerator for Mixed-Precision Inference of Graph Neural Networks 

**Authors**: Pedro Gimenes, Yiren Zhao, George Constantinides  

**Link**: [PDF](https://arxiv.org/pdf/2502.21196)  

**Abstract**: Graph Neural Networks (GNNs) have recently gained attention due to their performance on non-Euclidean data. The use of custom hardware architectures proves particularly beneficial for GNNs due to their irregular memory access patterns, resulting from the sparse structure of graphs. However, existing FPGA accelerators are limited by their double buffering mechanism, which doesn't account for the irregular node distribution in typical graph datasets. To address this, we introduce \textbf{AMPLE} (Accelerated Message Passing Logic Engine), an FPGA accelerator leveraging a new event-driven programming flow. We develop a mixed-arithmetic architecture, enabling GNN inference to be quantized at a node-level granularity. Finally, prefetcher for data and instructions is implemented to optimize off-chip memory access and maximize node parallelism. Evaluation on citation and social media graph datasets ranging from $2$K to $700$K nodes showed a mean speedup of $243\times$ and $7.2\times$ against CPU and GPU counterparts, respectively. 

---
# Scalable Decision-Making in Stochastic Environments through Learned Temporal Abstraction 

**Authors**: Baiting Luo, Ava Pettet, Aron Laszka, Abhishek Dubey, Ayan Mukhopadhyay  

**Link**: [PDF](https://arxiv.org/pdf/2502.21186)  

**Abstract**: Sequential decision-making in high-dimensional continuous action spaces, particularly in stochastic environments, faces significant computational challenges. We explore this challenge in the traditional offline RL setting, where an agent must learn how to make decisions based on data collected through a stochastic behavior policy. We present \textit{Latent Macro Action Planner} (L-MAP), which addresses this challenge by learning a set of temporally extended macro-actions through a state-conditional Vector Quantized Variational Autoencoder (VQ-VAE), effectively reducing action dimensionality. L-MAP employs a (separate) learned prior model that acts as a latent transition model and allows efficient sampling of plausible actions. During planning, our approach accounts for stochasticity in both the environment and the behavior policy by using Monte Carlo tree search (MCTS). In offline RL settings, including stochastic continuous control tasks, L-MAP efficiently searches over discrete latent actions to yield high expected returns. Empirical results demonstrate that L-MAP maintains low decision latency despite increased action dimensionality. Notably, across tasks ranging from continuous control with inherently stochastic dynamics to high-dimensional robotic hand manipulation, L-MAP significantly outperforms existing model-based methods and performs on-par with strong model-free actor-critic baselines, highlighting the effectiveness of the proposed approach in planning in complex and stochastic environments with high-dimensional action spaces. 

---
# Predicting clinical outcomes from patient care pathways represented with temporal knowledge graphs 

**Authors**: Jong Ho Jhee, Alberto Megina, Pacôme Constant Dit Beaufils, Matilde Karakachoff, Richard Redon, Alban Gaignard, Adrien Coulet  

**Link**: [PDF](https://arxiv.org/pdf/2502.21138)  

**Abstract**: Background: With the increasing availability of healthcare data, predictive modeling finds many applications in the biomedical domain, such as the evaluation of the level of risk for various conditions, which in turn can guide clinical decision making. However, it is unclear how knowledge graph data representations and their embedding, which are competitive in some settings, could be of interest in biomedical predictive modeling. Method: We simulated synthetic but realistic data of patients with intracranial aneurysm and experimented on the task of predicting their clinical outcome. We compared the performance of various classification approaches on tabular data versus a graph-based representation of the same data. Next, we investigated how the adopted schema for representing first individual data and second temporal data impacts predictive performances. Results: Our study illustrates that in our case, a graph representation and Graph Convolutional Network (GCN) embeddings reach the best performance for a predictive task from observational data. We emphasize the importance of the adopted schema and of the consideration of literal values in the representation of individual data. Our study also moderates the relative impact of various time encoding on GCN performance. 

---
# Dynamically Local-Enhancement Planner for Large-Scale Autonomous Driving 

**Authors**: Nanshan Deng, Weitao Zhou, Bo Zhang, Junze Wen, Kun Jiang, Zhong Cao, Diange Yang  

**Link**: [PDF](https://arxiv.org/pdf/2502.21134)  

**Abstract**: Current autonomous vehicles operate primarily within limited regions, but there is increasing demand for broader applications. However, as models scale, their limited capacity becomes a significant challenge for adapting to novel scenarios. It is increasingly difficult to improve models for new situations using a single monolithic model. To address this issue, we introduce the concept of dynamically enhancing a basic driving planner with local driving data, without permanently modifying the planner itself. This approach, termed the Dynamically Local-Enhancement (DLE) Planner, aims to improve the scalability of autonomous driving systems without significantly expanding the planner's size. Our approach introduces a position-varying Markov Decision Process formulation coupled with a graph neural network that extracts region-specific driving features from local observation data. The learned features describe the local behavior of the surrounding objects, which is then leveraged to enhance a basic reinforcement learning-based policy. We evaluated our approach in multiple scenarios and compared it with a one-for-all driving model. The results show that our method outperforms the baseline policy in both safety (collision rate) and average reward, while maintaining a lighter scale. This approach has the potential to benefit large-scale autonomous vehicles without the need for largely expanding on-device driving models. 

---
# Einleitung [Introduction] 

**Authors**: Vincent C. Müller  

**Link**: [PDF](https://arxiv.org/pdf/2502.21131)  

**Abstract**: Hilary Putnam's biography and philosophical development reflect the history of Anglo-Saxon philosophy over the last 40 years. Putnam has influenced this history significantly for almost as long. In this introduction, the main aim is to present the context in which Putnam stands and from which his philosophical contributions can be understood. In the context of a sketch of Putnam's philosophical development, a preliminary historical classification of his work will also be attempted, even if this is not the place for a comprehensive critique or presentation: The introduction must remain at a fairly elementary level and of course cannot replace a reading of the texts. Since Putnam's work is certainly part of a rapprochement between 'analytic' and 'continental' philosophy, the introduction to the texts translated here should finally make clear what Putnam has to offer non-analytically oriented readers.
Hilary Putnams Biographie und philosophische Entwicklung spiegeln die Geschichte der angelsächsischen Philosophie in den letzten 40 Jahren. Beinahe ebenso lange hat Putnam diese Geschichte wesentlich beeinflußt. In der vorliegenden Einleitung soll vor allem der Kontext dargestellt werden, in dem Putnam steht und aus dem heraus verständlich wird, was er philosophisch zu sagen hat. Im Rahmen einer Skizze von Putnams philosophischer Entwicklung soll zudem eine vorläufige philosophiehistorische Einordnung versucht werden, auch wenn hier nicht der Ort für eine umfassende Kritik oder Darstellung sein kann: Die Einleitung muß auf recht elementarem Niveau bleiben und kann eine Lektüre der Texte natürlich nicht ersetzen. Da Putnams Werk sicherlich Teil einer Annäherung von 'analytischer' und 'kontinentaler' Philosophie ist, soll bei der Einführung in die hier übersetzten Texte schließlich deutlich werden, was Putnam nicht analytisch orientierten Lesern zu bieten hat. 

---
# Causality Is Key to Understand and Balance Multiple Goals in Trustworthy ML and Foundation Models 

**Authors**: Ruta Binkyte, Ivaxi Sheth, Zhijing Jin, Muhammad Havaei, Bernhardt Schölkopf, Mario Fritz  

**Link**: [PDF](https://arxiv.org/pdf/2502.21123)  

**Abstract**: Ensuring trustworthiness in machine learning (ML) systems is crucial as they become increasingly embedded in high-stakes domains. This paper advocates for the integration of causal methods into machine learning to navigate the trade-offs among key principles of trustworthy ML, including fairness, privacy, robustness, accuracy, and explainability. While these objectives should ideally be satisfied simultaneously, they are often addressed in isolation, leading to conflicts and suboptimal solutions. Drawing on existing applications of causality in ML that successfully align goals such as fairness and accuracy or privacy and robustness, this paper argues that a causal approach is essential for balancing multiple competing objectives in both trustworthy ML and foundation models. Beyond highlighting these trade-offs, we examine how causality can be practically integrated into ML and foundation models, offering solutions to enhance their reliability and interpretability. Finally, we discuss the challenges, limitations, and opportunities in adopting causal frameworks, paving the way for more accountable and ethically sound AI systems. 

---
# AuthSim: Towards Authentic and Effective Safety-critical Scenario Generation for Autonomous Driving Tests 

**Authors**: Yukuan Yang, Xucheng Lu, Zhili Zhang, Zepeng Wu, Guoqi Li, Lingzhong Meng, Yunzhi Xue  

**Link**: [PDF](https://arxiv.org/pdf/2502.21100)  

**Abstract**: Generating adversarial safety-critical scenarios is a pivotal method for testing autonomous driving systems, as it identifies potential weaknesses and enhances system robustness and reliability. However, existing approaches predominantly emphasize unrestricted collision scenarios, prompting non-player character (NPC) vehicles to attack the ego vehicle indiscriminately. These works overlook these scenarios' authenticity, rationality, and relevance, resulting in numerous extreme, contrived, and largely unrealistic collision events involving aggressive NPC vehicles. To rectify this issue, we propose a three-layer relative safety region model, which partitions the area based on danger levels and increases the likelihood of NPC vehicles entering relative boundary regions. This model directs NPC vehicles to engage in adversarial actions within relatively safe boundary regions, thereby augmenting the scenarios' authenticity. We introduce AuthSim, a comprehensive platform for generating authentic and effective safety-critical scenarios by integrating the three-layer relative safety region model with reinforcement learning. To our knowledge, this is the first attempt to address the authenticity and effectiveness of autonomous driving system test scenarios comprehensively. Extensive experiments demonstrate that AuthSim outperforms existing methods in generating effective safety-critical scenarios. Notably, AuthSim achieves a 5.25% improvement in average cut-in distance and a 27.12% enhancement in average collision interval time, while maintaining higher efficiency in generating effective safety-critical scenarios compared to existing methods. This underscores its significant advantage in producing authentic scenarios over current methodologies. 

---
# PASemiQA: Plan-Assisted Agent for Question Answering on Semi-Structured Data with Text and Relational Information 

**Authors**: Hansi Yang, Qi Zhang, Wei Jiang, Jianguo Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.21087)  

**Abstract**: Large language models (LLMs) have shown impressive abilities in answering questions across various domains, but they often encounter hallucination issues on questions that require professional and up-to-date knowledge. To address this limitation, retrieval-augmented generation (RAG) techniques have been proposed, which retrieve relevant information from external sources to inform their responses. However, existing RAG methods typically focus on a single type of external data, such as vectorized text database or knowledge graphs, and cannot well handle real-world questions on semi-structured data containing both text and relational information. To bridge this gap, we introduce PASemiQA, a novel approach that jointly leverages text and relational information in semi-structured data to answer questions. PASemiQA first generates a plan to identify relevant text and relational information to answer the question in semi-structured data, and then uses an LLM agent to traverse the semi-structured data and extract necessary information. Our empirical results demonstrate the effectiveness of PASemiQA across different semi-structured datasets from various domains, showcasing its potential to improve the accuracy and reliability of question answering systems on semi-structured data. 

---
# Enhancing deep neural networks through complex-valued representations and Kuramoto synchronization dynamics 

**Authors**: Sabine Muzellec, Andrea Alamia, Thomas Serre, Rufin VanRullen  

**Link**: [PDF](https://arxiv.org/pdf/2502.21077)  

**Abstract**: Neural synchrony is hypothesized to play a crucial role in how the brain organizes visual scenes into structured representations, enabling the robust encoding of multiple objects within a scene. However, current deep learning models often struggle with object binding, limiting their ability to represent multiple objects effectively. Inspired by neuroscience, we investigate whether synchrony-based mechanisms can enhance object encoding in artificial models trained for visual categorization. Specifically, we combine complex-valued representations with Kuramoto dynamics to promote phase alignment, facilitating the grouping of features belonging to the same object. We evaluate two architectures employing synchrony: a feedforward model and a recurrent model with feedback connections to refine phase synchronization using top-down information. Both models outperform their real-valued counterparts and complex-valued models without Kuramoto synchronization on tasks involving multi-object images, such as overlapping handwritten digits, noisy inputs, and out-of-distribution transformations. Our findings highlight the potential of synchrony-driven mechanisms to enhance deep learning models, improving their performance, robustness, and generalization in complex visual categorization tasks. 

---
# FC-Attack: Jailbreaking Large Vision-Language Models via Auto-Generated Flowcharts 

**Authors**: Ziyi Zhang, Zhen Sun, Zongmin Zhang, Jihui Guo, Xinlei He  

**Link**: [PDF](https://arxiv.org/pdf/2502.21059)  

**Abstract**: Large Vision-Language Models (LVLMs) have become powerful and widely adopted in some practical applications. However, recent research has revealed their vulnerability to multimodal jailbreak attacks, whereby the model can be induced to generate harmful content, leading to safety risks. Although most LVLMs have undergone safety alignment, recent research shows that the visual modality is still vulnerable to jailbreak attacks. In our work, we discover that by using flowcharts with partially harmful information, LVLMs can be induced to provide additional harmful details. Based on this, we propose a jailbreak attack method based on auto-generated flowcharts, FC-Attack. Specifically, FC-Attack first fine-tunes a pre-trained LLM to create a step-description generator based on benign datasets. The generator is then used to produce step descriptions corresponding to a harmful query, which are transformed into flowcharts in 3 different shapes (vertical, horizontal, and S-shaped) as visual prompts. These flowcharts are then combined with a benign textual prompt to execute a jailbreak attack on LVLMs. Our evaluations using the Advbench dataset show that FC-Attack achieves over 90% attack success rates on Gemini-1.5, Llaval-Next, Qwen2-VL, and InternVL-2.5 models, outperforming existing LVLM jailbreak methods. Additionally, we investigate factors affecting the attack performance, including the number of steps and the font styles in the flowcharts. Our evaluation shows that FC-Attack can improve the jailbreak performance from 4% to 28% in Claude-3.5 by changing the font style. To mitigate the attack, we explore several defenses and find that AdaShield can largely reduce the jailbreak performance but with the cost of utility drop. 

---
# Robust Deterministic Policy Gradient for Disturbance Attenuation and Its Application to Quadrotor Control 

**Authors**: Taeho Lee, Donghwan Lee  

**Link**: [PDF](https://arxiv.org/pdf/2502.21057)  

**Abstract**: Practical control systems pose significant challenges in identifying optimal control policies due to uncertainties in the system model and external disturbances. While $H_\infty$ control techniques are commonly used to design robust controllers that mitigate the effects of disturbances, these methods often require complex and computationally intensive calculations. To address this issue, this paper proposes a reinforcement learning algorithm called Robust Deterministic Policy Gradient (RDPG), which formulates the $H_\infty$ control problem as a two-player zero-sum dynamic game. In this formulation, one player (the user) aims to minimize the cost, while the other player (the adversary) seeks to maximize it. We then employ deterministic policy gradient (DPG) and its deep reinforcement learning counterpart to train a robust control policy with effective disturbance attenuation. In particular, for practical implementation, we introduce an algorithm called robust deep deterministic policy gradient (RDDPG), which employs a deep neural network architecture and integrates techniques from the twin-delayed deep deterministic policy gradient (TD3) to enhance stability and learning efficiency. To evaluate the proposed algorithm, we implement it on an unmanned aerial vehicle (UAV) tasked with following a predefined path in a disturbance-prone environment. The experimental results demonstrate that the proposed method outperforms other control approaches in terms of robustness against disturbances, enabling precise real-time tracking of moving targets even under severe disturbance conditions. 

---
# Synthesizing Individualized Aging Brains in Health and Disease with Generative Models and Parallel Transport 

**Authors**: Jingru Fu, Yuqi Zheng, Neel Dey, Daniel Ferreira, Rodrigo Moreno  

**Link**: [PDF](https://arxiv.org/pdf/2502.21049)  

**Abstract**: Simulating prospective magnetic resonance imaging (MRI) scans from a given individual brain image is challenging, as it requires accounting for canonical changes in aging and/or disease progression while also considering the individual brain's current status and unique characteristics. While current deep generative models can produce high-resolution anatomically accurate templates for population-wide studies, their ability to predict future aging trajectories for individuals remains limited, particularly in capturing subject-specific neuroanatomical variations over time. In this study, we introduce Individualized Brain Synthesis (InBrainSyn), a framework for synthesizing high-resolution subject-specific longitudinal MRI scans that simulate neurodegeneration in both Alzheimer's disease (AD) and normal aging. InBrainSyn uses a parallel transport algorithm to adapt the population-level aging trajectories learned by a generative deep template network, enabling individualized aging synthesis. As InBrainSyn uses diffeomorphic transformations to simulate aging, the synthesized images are topologically consistent with the original anatomy by design. We evaluated InBrainSyn both quantitatively and qualitatively on AD and healthy control cohorts from the Open Access Series of Imaging Studies - version 3 dataset. Experimentally, InBrainSyn can also model neuroanatomical transitions between normal aging and AD. An evaluation of an external set supports its generalizability. Overall, with only a single baseline scan, InBrainSyn synthesizes realistic 3D spatiotemporal T1w MRI scans, producing personalized longitudinal aging trajectories. The code for InBrainSyn is available at: this https URL. 

---
# Fast Adversarial Training against Sparse Attacks Requires Loss Smoothing 

**Authors**: Xuyang Zhong, Yixiao Huang, Chen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.21041)  

**Abstract**: This paper studies fast adversarial training against sparse adversarial perturbations bounded by $l_0$ norm. We demonstrate the challenges of employing $1$-step attacks on $l_0$ bounded perturbations for fast adversarial training, including degraded performance and the occurrence of catastrophic overfitting (CO). We highlight that CO in $l_0$ adversarial training is caused by sub-optimal perturbation locations of $1$-step attack. Theoretical and empirical analyses reveal that the loss landscape of $l_0$ adversarial training is more craggy compared to its $l_\infty$, $l_2$ and $l_1$ counterparts. Moreover, we corroborate that the craggy loss landscape can aggravate CO. To address these issues, we propose Fast-LS-$l_0$ that incorporates soft labels and the trade-off loss function to smooth the adversarial loss landscape. Extensive experiments demonstrate our method can overcome the challenge of catastrophic overfitting, achieve state-of-the-art performance, and narrow down the performance gap between $1$-step and multi-step adversarial training against sparse attacks. 

---
# Reward Learning from Multiple Feedback Types 

**Authors**: Yannick Metz, András Geiszl, Raphaël Baur, Mennatallah El-Assady  

**Link**: [PDF](https://arxiv.org/pdf/2502.21038)  

**Abstract**: Learning rewards from preference feedback has become an important tool in the alignment of agentic models. Preference-based feedback, often implemented as a binary comparison between multiple completions, is an established method to acquire large-scale human feedback. However, human feedback in other contexts is often much more diverse. Such diverse feedback can better support the goals of a human annotator, and the simultaneous use of multiple sources might be mutually informative for the learning process or carry type-dependent biases for the reward learning process. Despite these potential benefits, learning from different feedback types has yet to be explored extensively. In this paper, we bridge this gap by enabling experimentation and evaluating multi-type feedback in a broad set of environments. We present a process to generate high-quality simulated feedback of six different types. Then, we implement reward models and downstream RL training for all six feedback types. Based on the simulated feedback, we investigate the use of types of feedback across ten RL environments and compare them to pure preference-based baselines. We show empirically that diverse types of feedback can be utilized and lead to strong reward modeling performance. This work is the first strong indicator of the potential of multi-type feedback for RLHF. 

---
# Synthesizing Tabular Data Using Selectivity Enhanced Generative Adversarial Networks 

**Authors**: Youran Zhou, Jianzhong Qi  

**Link**: [PDF](https://arxiv.org/pdf/2502.21034)  

**Abstract**: As E-commerce platforms face surging transactions during major shopping events like Black Friday, stress testing with synthesized data is crucial for resource planning. Most recent studies use Generative Adversarial Networks (GANs) to generate tabular data while ensuring privacy and machine learning utility. However, these methods overlook the computational demands of processing GAN-generated data, making them unsuitable for E-commerce stress testing.
This thesis introduces a novel GAN-based approach incorporating query selectivity constraints, a key factor in database transaction processing. We integrate a pre-trained deep neural network to maintain selectivity consistency between real and synthetic data. Our method, tested on five real-world datasets, outperforms three state-of-the-art GANs and a VAE model, improving selectivity estimation accuracy by up to 20pct and machine learning utility by up to 6 pct. 

---
# Beyond Words: A Latent Memory Approach to Internal Reasoning in LLMs 

**Authors**: José I. Orlicki  

**Link**: [PDF](https://arxiv.org/pdf/2502.21030)  

**Abstract**: Recent advances in large language models (LLMs) have popularized the chain-of-thought (CoT) paradigm, in which models produce explicit reasoning steps in natural language. Although this approach improves interpretability and facilitates external auditing, it may not represent the most computationally efficient method for internal reasoning. In contrast, human cognition relies on implicit mental representations that recall past sensory and episodic information without requiring complete verbalization. In this paper, we propose a framework that integrates implicit mental representations into the internal reasoning processes of LLMs. Preliminary experiments indicate that incorporating an Implicit Memory Module (IMM) into a simple GPT model yields a reduction of between 35% and 57% in final training loss compared to a regular GPT baseline. The addition of an explicit interpretability channel (e.g., a chain-of-thought decoder) is straightforward to implement within this approach. We outline theoretical foundations, propose technical mechanisms to scale the memory module, and discuss how these ideas may lead to more efficient and robust reasoning, with optional future extensions for explicit auditability. 

---
# Measuring and identifying factors of individuals' trust in Large Language Models 

**Authors**: Edoardo Sebastiano De Duro, Giuseppe Alessandro Veltri, Hudson Golino, Massimo Stella  

**Link**: [PDF](https://arxiv.org/pdf/2502.21028)  

**Abstract**: Large Language Models (LLMs) can engage in human-looking conversational exchanges. Although conversations can elicit trust between users and LLMs, scarce empirical research has examined trust formation in human-LLM contexts, beyond LLMs' trustworthiness or human trust in AI in general. Here, we introduce the Trust-In-LLMs Index (TILLMI) as a new framework to measure individuals' trust in LLMs, extending McAllister's cognitive and affective trust dimensions to LLM-human interactions. We developed TILLMI as a psychometric scale, prototyped with a novel protocol we called LLM-simulated validity. The LLM-based scale was then validated in a sample of 1,000 US respondents. Exploratory Factor Analysis identified a two-factor structure. Two items were then removed due to redundancy, yielding a final 6-item scale with a 2-factor structure. Confirmatory Factor Analysis on a separate subsample showed strong model fit ($CFI = .995$, $TLI = .991$, $RMSEA = .046$, $p_{X^2} > .05$). Convergent validity analysis revealed that trust in LLMs correlated positively with openness to experience, extraversion, and cognitive flexibility, but negatively with neuroticism. Based on these findings, we interpreted TILLMI's factors as "closeness with LLMs" (affective dimension) and "reliance on LLMs" (cognitive dimension). Younger males exhibited higher closeness with- and reliance on LLMs compared to older women. Individuals with no direct experience with LLMs exhibited lower levels of trust compared to LLMs' users. These findings offer a novel empirical foundation for measuring trust in AI-driven verbal communication, informing responsible design, and fostering balanced human-AI collaboration. 

---
# LesionLocator: Zero-Shot Universal Tumor Segmentation and Tracking in 3D Whole-Body Imaging 

**Authors**: Maximilian Rokuss, Yannick Kirchhoff, Seval Akbal, Balint Kovacs, Saikat Roy, Constantin Ulrich, Tassilo Wald, Lukas T. Rotkopf, Heinz-Peter Schlemmer, Klaus Maier-Hein  

**Link**: [PDF](https://arxiv.org/pdf/2502.20985)  

**Abstract**: In this work, we present LesionLocator, a framework for zero-shot longitudinal lesion tracking and segmentation in 3D medical imaging, establishing the first end-to-end model capable of 4D tracking with dense spatial prompts. Our model leverages an extensive dataset of 23,262 annotated medical scans, as well as synthesized longitudinal data across diverse lesion types. The diversity and scale of our dataset significantly enhances model generalizability to real-world medical imaging challenges and addresses key limitations in longitudinal data availability. LesionLocator outperforms all existing promptable models in lesion segmentation by nearly 10 dice points, reaching human-level performance, and achieves state-of-the-art results in lesion tracking, with superior lesion retrieval and segmentation accuracy. LesionLocator not only sets a new benchmark in universal promptable lesion segmentation and automated longitudinal lesion tracking but also provides the first open-access solution of its kind, releasing our synthetic 4D dataset and model to the community, empowering future advancements in medical imaging. Code is available at: this http URL 

---
# UoR-NCL at SemEval-2025 Task 1: Using Generative LLMs and CLIP Models for Multilingual Multimodal Idiomaticity Representation 

**Authors**: Thanet Markchom, Tong Wu, Liting Huang, Huizhi Liang  

**Link**: [PDF](https://arxiv.org/pdf/2502.20984)  

**Abstract**: SemEval-2025 Task 1 focuses on ranking images based on their alignment with a given nominal compound that may carry idiomatic meaning in both English and Brazilian Portuguese. To address this challenge, this work uses generative large language models (LLMs) and multilingual CLIP models to enhance idiomatic compound representations. LLMs generate idiomatic meanings for potentially idiomatic compounds, enriching their semantic interpretation. These meanings are then encoded using multilingual CLIP models, serving as representations for image ranking. Contrastive learning and data augmentation techniques are applied to fine-tune these embeddings for improved performance. Experimental results show that multimodal representations extracted through this method outperformed those based solely on the original nominal compounds. The fine-tuning approach shows promising outcomes but is less effective than using embeddings without fine-tuning. The source code used in this paper is available at this https URL. 

---
# Improving Open-world Continual Learning under the Constraints of Scarce Labeled Data 

**Authors**: Yujie Li, Xiangkun Wang, Xin Yang, Marcello Bonsangue, Junbo Zhang, Tianrui Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.20974)  

**Abstract**: Open-world continual learning (OWCL) adapts to sequential tasks with open samples, learning knowledge incrementally while preventing forgetting. However, existing OWCL still requires a large amount of labeled data for training, which is often impractical in real-world applications. Given that new categories/entities typically come with limited annotations and are in small quantities, a more realistic situation is OWCL with scarce labeled data, i.e., few-shot training samples. Hence, this paper investigates the problem of open-world few-shot continual learning (OFCL), challenging in (i) learning unbounded tasks without forgetting previous knowledge and avoiding overfitting, (ii) constructing compact decision boundaries for open detection with limited labeled data, and (iii) transferring knowledge about knowns and unknowns and even update the unknowns to knowns once the labels of open samples are learned. In response, we propose a novel OFCL framework that integrates three key components: (1) an instance-wise token augmentation (ITA) that represents and enriches sample representations with additional knowledge, (2) a margin-based open boundary (MOB) that supports open detection with new tasks emerge over time, and (3) an adaptive knowledge space (AKS) that endows unknowns with knowledge for the updating from unknowns to knowns. Finally, extensive experiments show the proposed OFCL framework outperforms all baselines remarkably with practical importance and reproducibility. The source code is released at this https URL. 

---
# Fine-Grained Retrieval-Augmented Generation for Visual Question Answering 

**Authors**: Zhengxuan Zhang, Yin Wu, Yuyu Luo, Nan Tang  

**Link**: [PDF](https://arxiv.org/pdf/2502.20964)  

**Abstract**: Visual Question Answering (VQA) focuses on providing answers to natural language questions by utilizing information from images. Although cutting-edge multimodal large language models (MLLMs) such as GPT-4o achieve strong performance on VQA tasks, they frequently fall short in accessing domain-specific or the latest knowledge. To mitigate this issue, retrieval-augmented generation (RAG) leveraging external knowledge bases (KBs), referred to as KB-VQA, emerges as a promising approach. Nevertheless, conventional unimodal retrieval techniques, which translate images into textual descriptions, often result in the loss of critical visual details. This study presents fine-grained knowledge units, which merge textual snippets with entity images stored in vector databases. Furthermore, we introduce a knowledge unit retrieval-augmented generation framework (KU-RAG) that integrates fine-grained retrieval with MLLMs. The proposed KU-RAG framework ensures precise retrieval of relevant knowledge and enhances reasoning capabilities through a knowledge correction chain. Experimental findings demonstrate that our approach significantly boosts the performance of leading KB-VQA methods, achieving improvements of up to 10%. 

---
# Retrieval Augmented Generation for Topic Modeling in Organizational Research: An Introduction with Empirical Demonstration 

**Authors**: Gerion Spielberger, Florian Artinger, Jochen Reb, Rudolf Kerschreiter  

**Link**: [PDF](https://arxiv.org/pdf/2502.20963)  

**Abstract**: Analyzing textual data is the cornerstone of qualitative research. While traditional methods such as grounded theory and content analysis are widely used, they are labor-intensive and time-consuming. Topic modeling offers an automated complement. Yet, existing approaches, including LLM-based topic modeling, still struggle with issues such as high data preprocessing requirements, interpretability, and reliability. This paper introduces Agentic Retrieval-Augmented Generation (Agentic RAG) as a method for topic modeling with LLMs. It integrates three key components: (1) retrieval, enabling automatized access to external data beyond an LLM's pre-trained knowledge; (2) generation, leveraging LLM capabilities for text synthesis; and (3) agent-driven learning, iteratively refining retrieval and query formulation processes. To empirically validate Agentic RAG for topic modeling, we reanalyze a Twitter/X dataset, previously examined by Mu et al. (2024a). Our findings demonstrate that the approach is more efficient, interpretable and at the same time achieves higher reliability and validity in comparison to the standard machine learning approach but also in comparison to LLM prompting for topic modeling. These results highlight Agentic RAG's ability to generate semantically relevant and reproducible topics, positioning it as a robust, scalable, and transparent alternative for AI-driven qualitative research in leadership, managerial, and organizational research. 

---
# Concealed Adversarial attacks on neural networks for sequential data 

**Authors**: Petr Sokerin, Dmitry Anikin, Sofia Krehova, Alexey Zaytsev  

**Link**: [PDF](https://arxiv.org/pdf/2502.20948)  

**Abstract**: The emergence of deep learning led to the broad usage of neural networks in the time series domain for various applications, including finance and medicine. While powerful, these models are prone to adversarial attacks: a benign targeted perturbation of input data leads to significant changes in a classifier's output. However, formally small attacks in the time series domain become easily detected by the human eye or a simple detector model.
We develop a concealed adversarial attack for different time-series models: it provides more realistic perturbations, being hard to detect by a human or model discriminator. To achieve this goal, the proposed adversarial attack maximizes an aggregation of a classifier and a trained discriminator loss. To make the attack stronger, we also propose a training procedure for a discriminator that provides broader coverage of possible attacks. Extensive benchmarking on six UCR time series datasets across four diverse architectures - including recurrent, convolutional, state-space, and transformer-based models - demonstrates the superiority of our attack for a concealability-efficiency trade-off. Our findings highlight the growing challenge of designing robust time series models, emphasizing the need for improved defenses against realistic and effective attacks. 

---
# Generative Uncertainty in Diffusion Models 

**Authors**: Metod Jazbec, Eliot Wong-Toi, Guoxuan Xia, Dan Zhang, Eric Nalisnick, Stephan Mandt  

**Link**: [PDF](https://arxiv.org/pdf/2502.20946)  

**Abstract**: Diffusion models have recently driven significant breakthroughs in generative modeling. While state-of-the-art models produce high-quality samples on average, individual samples can still be low quality. Detecting such samples without human inspection remains a challenging task. To address this, we propose a Bayesian framework for estimating generative uncertainty of synthetic samples. We outline how to make Bayesian inference practical for large, modern generative models and introduce a new semantic likelihood (evaluated in the latent space of a feature extractor) to address the challenges posed by high-dimensional sample spaces. Through our experiments, we demonstrate that the proposed generative uncertainty effectively identifies poor-quality samples and significantly outperforms existing uncertainty-based methods. Notably, our Bayesian framework can be applied post-hoc to any pretrained diffusion or flow matching model (via the Laplace approximation), and we propose simple yet effective techniques to minimize its computational overhead during sampling. 

---
# A Deep User Interface for Exploring LLaMa 

**Authors**: Divya Perumal, Swaroop Panda  

**Link**: [PDF](https://arxiv.org/pdf/2502.20938)  

**Abstract**: The growing popularity and widespread adoption of large language models (LLMs) necessitates the development of tools that enhance the effectiveness of user interactions with these models. Understanding the structures and functions of these models poses a significant challenge for users. Visual analytics-driven tools enables users to explore and compare, facilitating better decision-making. This paper presents a visual analytics-driven tool equipped with interactive controls for key hyperparameters, including top-p, frequency and presence penalty, enabling users to explore, examine and compare the outputs of LLMs. In a user study, we assessed the tool's effectiveness, which received favorable feedback for its visual design, with particular commendation for the interface layout and ease of navigation. Additionally, the feedback provided valuable insights for enhancing the effectiveness of Human-LLM interaction tools. 

---
# WebFAQ: A Multilingual Collection of Natural Q&A Datasets for Dense Retrieval 

**Authors**: Michael Dinzinger, Laura Caspari, Kanishka Ghosh Dastidar, Jelena Mitrović, Michael Granitzer  

**Link**: [PDF](https://arxiv.org/pdf/2502.20936)  

**Abstract**: We present WebFAQ, a large-scale collection of open-domain question answering datasets derived from FAQ-style this http URL annotations. In total, the data collection consists of 96 million natural question-answer (QA) pairs across 75 languages, including 47 million (49%) non-English samples. WebFAQ further serves as the foundation for 20 monolingual retrieval benchmarks with a total size of 11.2 million QA pairs (5.9 million non-English). These datasets are carefully curated through refined filtering and near-duplicate detection, yielding high-quality resources for training and evaluating multilingual dense retrieval models. To empirically confirm WebFAQ's efficacy, we use the collected QAs to fine-tune an in-domain pretrained XLM-RoBERTa model. Through this process of dataset-specific fine-tuning, the model achieves significant retrieval performance gains, which generalize - beyond WebFAQ - to other multilingual retrieval benchmarks evaluated in zero-shot setting. Last but not least, we utilize WebFAQ to construct a set of QA-aligned bilingual corpora spanning over 1000 language pairs using state-of-the-art bitext mining and automated LLM-assessed translation evaluation. Due to our advanced, automated method of bitext dataset generation, the resulting bilingual corpora demonstrate higher translation quality compared to similar datasets. WebFAQ and all associated resources are publicly available on GitHub and HuggingFace. 

---
# Less is More? Revisiting the Importance of Frame Rate in Real-Time Zero-Shot Surgical Video Segmentation 

**Authors**: Utku Ozbulak, Seyed Amir Mousavi, Francesca Tozzi, Nikdokht Rashidian, Wouter Willaert, Wesley De Neve, Joris Vankerschaver  

**Link**: [PDF](https://arxiv.org/pdf/2502.20934)  

**Abstract**: Real-time video segmentation is a promising feature for AI-assisted surgery, providing intraoperative guidance by identifying surgical tools and anatomical structures. However, deploying state-of-the-art segmentation models, such as SAM2, in real-time settings is computationally demanding, which makes it essential to balance frame rate and segmentation performance. In this study, we investigate the impact of frame rate on zero-shot surgical video segmentation, evaluating SAM2's effectiveness across multiple frame sampling rates for cholecystectomy procedures. Surprisingly, our findings indicate that in conventional evaluation settings, frame rates as low as a single frame per second can outperform 25 FPS, as fewer frames smooth out segmentation inconsistencies. However, when assessed in a real-time streaming scenario, higher frame rates yield superior temporal coherence and stability, particularly for dynamic objects such as surgical graspers. Finally, we investigate human perception of real-time surgical video segmentation among professionals who work closely with such data and find that respondents consistently prefer high FPS segmentation mask overlays, reinforcing the importance of real-time evaluation in AI-assisted surgery. 

---
# Everything, Everywhere, All at Once: Is Mechanistic Interpretability Identifiable? 

**Authors**: Maxime Méloux, Silviu Maniu, François Portet, Maxime Peyrard  

**Link**: [PDF](https://arxiv.org/pdf/2502.20914)  

**Abstract**: As AI systems are used in high-stakes applications, ensuring interpretability is crucial. Mechanistic Interpretability (MI) aims to reverse-engineer neural networks by extracting human-understandable algorithms to explain their behavior. This work examines a key question: for a given behavior, and under MI's criteria, does a unique explanation exist? Drawing on identifiability in statistics, where parameters are uniquely inferred under specific assumptions, we explore the identifiability of MI explanations.
We identify two main MI strategies: (1) "where-then-what," which isolates a circuit replicating model behavior before interpreting it, and (2) "what-then-where," which starts with candidate algorithms and searches for neural activation subspaces implementing them, using causal alignment.
We test both strategies on Boolean functions and small multi-layer perceptrons, fully enumerating candidate explanations. Our experiments reveal systematic non-identifiability: multiple circuits can replicate behavior, a circuit can have multiple interpretations, several algorithms can align with the network, and one algorithm can align with different subspaces.
Is uniqueness necessary? A pragmatic approach may require only predictive and manipulability standards. If uniqueness is essential for understanding, stricter criteria may be needed. We also reference the inner interpretability framework, which validates explanations through multiple criteria. This work contributes to defining explanation standards in AI. 

---
# DexGraspVLA: A Vision-Language-Action Framework Towards General Dexterous Grasping 

**Authors**: Yifan Zhong, Xuchuan Huang, Ruochong Li, Ceyao Zhang, Yitao Liang, Yaodong Yang, Yuanpei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.20900)  

**Abstract**: Dexterous grasping remains a fundamental yet challenging problem in robotics. A general-purpose robot must be capable of grasping diverse objects in arbitrary scenarios. However, existing research typically relies on specific assumptions, such as single-object settings or limited environments, leading to constrained generalization. Our solution is DexGraspVLA, a hierarchical framework that utilizes a pre-trained Vision-Language model as the high-level task planner and learns a diffusion-based policy as the low-level Action controller. The key insight lies in iteratively transforming diverse language and visual inputs into domain-invariant representations, where imitation learning can be effectively applied due to the alleviation of domain shift. Thus, it enables robust generalization across a wide range of real-world scenarios. Notably, our method achieves a 90+% success rate under thousands of unseen object, lighting, and background combinations in a ``zero-shot'' environment. Empirical analysis further confirms the consistency of internal model behavior across environmental variations, thereby validating our design and explaining its generalization performance. We hope our work can be a step forward in achieving general dexterous grasping. Our demo and code can be found at this https URL. 

---
# A Fused Gromov-Wasserstein Approach to Subgraph Contrastive Learning 

**Authors**: Amadou S. Sangare, Nicolas Dunou, Jhony H. Giraldo, Fragkiskos D. Malliaros  

**Link**: [PDF](https://arxiv.org/pdf/2502.20885)  

**Abstract**: Self-supervised learning has become a key method for training deep learning models when labeled data is scarce or unavailable. While graph machine learning holds great promise across various domains, the design of effective pretext tasks for self-supervised graph representation learning remains challenging. Contrastive learning, a popular approach in graph self-supervised learning, leverages positive and negative pairs to compute a contrastive loss function. However, current graph contrastive learning methods often struggle to fully use structural patterns and node similarities. To address these issues, we present a new method called Fused Gromov Wasserstein Subgraph Contrastive Learning (FOSSIL). Our model integrates node-level and subgraph-level contrastive learning, seamlessly combining a standard node-level contrastive loss with the Fused Gromov-Wasserstein distance. This combination helps our method capture both node features and graph structure together. Importantly, our approach works well with both homophilic and heterophilic graphs and can dynamically create views for generating positive and negative pairs. Through extensive experiments on benchmark graph datasets, we show that FOSSIL outperforms or achieves competitive performance compared to current state-of-the-art methods. 

---
# Oscillation-Reduced MXFP4 Training for Vision Transformers 

**Authors**: Yuxiang Chen, Haocheng Xi, Jun Zhu, Jianfei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.20853)  

**Abstract**: Pre-training Transformers in FP4 precision is becoming a promising approach to gain substantial speedup, but it comes with a considerable loss of accuracy. Microscaling (MX) data format provides a fine-grained per-group quantization method to improve the representation ability of the FP4 format and is supported by the next-generation Blackwell GPU architecture. However, training with MXFP4 data format still results in significant degradation and there is a lack of systematic research on the reason.
In this work, we propose a novel training method TetraJet for a more accurate FP4 training. We comprehensively evaluate all of the quantizers involved in the training, and identify the weight oscillation problem in the forward pass as the main source of the degradation in MXFP4 training. Therefore, we introduce two novel methods, EMA Quantizer (Q-EMA) and Adaptive Ramping Optimizer (Q-Ramping), to resolve the oscillation problem. Extensive experiments on Vision Transformers demonstrate that TetraJet consistently outperforms the existing 4-bit training methods, and Q-EMA & Q-Ramping can provide additional enhancement by effectively reducing oscillation. We decreased the accuracy degradation by more than $50\%$ compared to the baseline, and can even achieve competitive performance compared to full precision training. The codes are available at this https URL 

---
# Reinforcement Learning with Curriculum-inspired Adaptive Direct Policy Guidance for Truck Dispatching 

**Authors**: Shi Meng, Bin Tian, Xiaotong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.20845)  

**Abstract**: Efficient truck dispatching via Reinforcement Learning (RL) in open-pit mining is often hindered by reliance on complex reward engineering and value-based methods. This paper introduces Curriculum-inspired Adaptive Direct Policy Guidance, a novel curriculum learning strategy for policy-based RL to address these issues. We adapt Proximal Policy Optimization (PPO) for mine dispatching's uneven decision intervals using time deltas in Temporal Difference and Generalized Advantage Estimation, and employ a Shortest Processing Time teacher policy for guided exploration via policy regularization and adaptive guidance. Evaluations in OpenMines demonstrate our approach yields a 10% performance gain and faster convergence over standard PPO across sparse and dense reward settings, showcasing improved robustness to reward design. This direct policy guidance method provides a general and effective curriculum learning technique for RL-based truck dispatching, enabling future work on advanced architectures. 

---
# Neuro-Symbolic Learning for Galois Groups: Unveiling Probabilistic Trends in Polynomials 

**Authors**: Elira Shaska, Tony Shaska  

**Link**: [PDF](https://arxiv.org/pdf/2502.20844)  

**Abstract**: This paper presents a neurosymbolic approach to classifying Galois groups of polynomials, integrating classical Galois theory with machine learning to address challenges in algebraic computation. By combining neural networks with symbolic reasoning we develop a model that outperforms purely numerical methods in accuracy and interpretability. Focusing on sextic polynomials with height $\leq 6$, we analyze a database of 53,972 irreducible examples, uncovering novel distributional trends, such as the 20 sextic polynomials with Galois group $C_6$ spanning just seven invariant-defined equivalence classes. These findings offer the first empirical insights into Galois group probabilities under height constraints and lay the groundwork for exploring solvability by radicals. Demonstrating AI's potential to reveal patterns beyond traditional symbolic techniques, this work paves the way for future research in computational algebra, with implications for probabilistic conjectures and higher degree classifications. 

---
# Hierarchical and Modular Network on Non-prehensile Manipulation in General Environments 

**Authors**: Yoonyoung Cho, Junhyek Han, Jisu Han, Beomjoon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2502.20843)  

**Abstract**: For robots to operate in general environments like households, they must be able to perform non-prehensile manipulation actions such as toppling and rolling to manipulate ungraspable objects. However, prior works on non-prehensile manipulation cannot yet generalize across environments with diverse geometries. The main challenge lies in adapting to varying environmental constraints: within a cabinet, the robot must avoid walls and ceilings; to lift objects to the top of a step, the robot must account for the step's pose and extent. While deep reinforcement learning (RL) has demonstrated impressive success in non-prehensile manipulation, accounting for such variability presents a challenge for the generalist policy, as it must learn diverse strategies for each new combination of constraints. To address this, we propose a modular and reconfigurable architecture that adaptively reconfigures network modules based on task requirements. To capture the geometric variability in environments, we extend the contact-based object representation (CORN) to environment geometries, and propose a procedural algorithm for generating diverse environments to train our agent. Taken together, the resulting policy can zero-shot transfer to novel real-world environments and objects despite training entirely within a simulator. We additionally release a simulation-based benchmark featuring nine digital twins of real-world scenes with 353 objects to facilitate non-prehensile manipulation research in realistic domains. 

---
# Weakly Supervised Multiple Instance Learning for Whale Call Detection and Localization in Long-Duration Passive Acoustic Monitoring 

**Authors**: Ragib Amin Nihal, Benjamin Yen, Runwu Shi, Kazuhiro Nakadai  

**Link**: [PDF](https://arxiv.org/pdf/2502.20838)  

**Abstract**: Marine ecosystem monitoring via Passive Acoustic Monitoring (PAM) generates vast data, but deep learning often requires precise annotations and short segments. We introduce DSMIL-LocNet, a Multiple Instance Learning framework for whale call detection and localization using only bag-level labels. Our dual-stream model processes 2-30 minute audio segments, leveraging spectral and temporal features with attention-based instance selection. Tests on Antarctic whale data show longer contexts improve classification (F1: 0.8-0.9) while medium instances ensure localization precision (0.65-0.70). This suggests MIL can enhance scalable marine monitoring. Code: this https URL 

---
# LADs: Leveraging LLMs for AI-Driven DevOps 

**Authors**: Ahmad Faraz Khan, Azal Ahmad Khan, Anas Mohamed, Haider Ali, Suchithra Moolinti, Sabaat Haroon, Usman Tahir, Mattia Fazzini, Ali R. Butt, Ali Anwar  

**Link**: [PDF](https://arxiv.org/pdf/2502.20825)  

**Abstract**: Automating cloud configuration and deployment remains a critical challenge due to evolving infrastructures, heterogeneous hardware, and fluctuating workloads. Existing solutions lack adaptability and require extensive manual tuning, leading to inefficiencies and misconfigurations. We introduce LADs, the first LLM-driven framework designed to tackle these challenges by ensuring robustness, adaptability, and efficiency in automated cloud management. Instead of merely applying existing techniques, LADs provides a principled approach to configuration optimization through in-depth analysis of what optimization works under which conditions. By leveraging Retrieval-Augmented Generation, Few-Shot Learning, Chain-of-Thought, and Feedback-Based Prompt Chaining, LADs generates accurate configurations and learns from deployment failures to iteratively refine system settings. Our findings reveal key insights into the trade-offs between performance, cost, and scalability, helping practitioners determine the right strategies for different deployment scenarios. For instance, we demonstrate how prompt chaining-based adaptive feedback loops enhance fault tolerance in multi-tenant environments and how structured log analysis with example shots improves configuration accuracy. Through extensive evaluations, LADs reduces manual effort, optimizes resource utilization, and improves system reliability. By open-sourcing LADs, we aim to drive further innovation in AI-powered DevOps automation. 

---
# Multimodal Learning for Just-In-Time Software Defect Prediction in Autonomous Driving Systems 

**Authors**: Faisal Mohammad, Duksan Ryu  

**Link**: [PDF](https://arxiv.org/pdf/2502.20806)  

**Abstract**: In recent years, the rise of autonomous driving technologies has highlighted the critical importance of reliable software for ensuring safety and performance. This paper proposes a novel approach for just-in-time software defect prediction (JIT-SDP) in autonomous driving software systems using multimodal learning. The proposed model leverages the multimodal transformers in which the pre-trained transformers and a combining module deal with the multiple data modalities of the software system datasets such as code features, change metrics, and contextual information. The key point for adapting multimodal learning is to utilize the attention mechanism between the different data modalities such as text, numerical, and categorical. In the combining module, the output of a transformer model on text data and tabular features containing categorical and numerical data are combined to produce the predictions using the fully connected layers. Experiments conducted on three open-source autonomous driving system software projects collected from the GitHub repository (Apollo, Carla, and Donkeycar) demonstrate that the proposed approach significantly outperforms state-of-the-art deep learning and machine learning models regarding evaluation metrics. Our findings highlight the potential of multimodal learning to enhance the reliability and safety of autonomous driving software through improved defect prediction. 

---
# Characteristics Analysis of Autonomous Vehicle Pre-crash Scenarios 

**Authors**: Yixuan Li, Xuesong Wang, Tianyi Wang, Qian Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.20789)  

**Abstract**: To date, hundreds of crashes have occurred in open road testing of automated vehicles (AVs), highlighting the need for improving AV reliability and safety. Pre-crash scenario typology classifies crashes based on vehicle dynamics and kinematics features. Building on this, characteristics analysis can identify similar features under comparable crashes, offering a more effective reflection of general crash patterns and providing more targeted recommendations for enhancing AV performance. However, current studies primarily concentrated on crashes among conventional human-driven vehicles, leaving a gap in research dedicated to in-depth AV crash analyses. In this paper, we analyzed the latest California AV collision reports and used the newly revised pre-crash scenario typology to identify pre-crash scenarios. We proposed a set of mapping rules for automatically extracting these AV pre-crash scenarios, successfully identifying 24 types with a 98.1% accuracy rate, and obtaining two key scenarios of AV crashes (i.e., rear-end scenarios and intersection scenarios) through detailed analysis. Association analyses of rear-end scenarios showed that the significant environmental influencing factors were traffic control type, location type, light, etc. For intersection scenarios prone to severe crashes with detailed descriptions, we employed causal analyses to obtain the significant causal factors: habitual violations and expectations of certain behavior. Optimization recommendations were then formulated, addressing both governmental oversight and AV manufacturers' potential improvements. The findings of this paper could guide government authorities to develop related regulations, help manufacturers design AV test scenarios, and identify potential shortcomings in control algorithms specific to various real-world scenarios, thereby optimizing AV systems effectively. 

---
# Flattening Supply Chains: When do Technology Improvements lead to Disintermediation? 

**Authors**: S. Nageeb Ali, Nicole Immorlica, Meena Jagadeesan, Brendan Lucier  

**Link**: [PDF](https://arxiv.org/pdf/2502.20783)  

**Abstract**: In the digital economy, technological innovations make it cheaper to produce high-quality content. For example, generative AI tools reduce costs for creators who develop content to be distributed online, but can also reduce production costs for the users who consume that content. These innovations can thus lead to disintermediation, since consumers may choose to use these technologies directly, bypassing intermediaries. To investigate when technological improvements lead to disintermediation, we study a game with an intermediary, suppliers of a production technology, and consumers. First, we show disintermediation occurs whenever production costs are too high or too low. We then investigate the consequences of disintermediation for welfare and content quality at equilibrium. While the intermediary is welfare-improving, the intermediary extracts all gains to social welfare and its presence can raise or lower content quality. We further analyze how disintermediation is affected by the level of competition between suppliers and the intermediary's fee structure. More broadly, our results take a step towards assessing how production technology innovations affect the survival of intermediaries and impact the digital economy. 

---
# Triple Phase Transitions: Understanding the Learning Dynamics of Large Language Models from a Neuroscience Perspective 

**Authors**: Yuko Nakagi, Keigo Tada, Sota Yoshino, Shinji Nishimoto, Yu Takagi  

**Link**: [PDF](https://arxiv.org/pdf/2502.20779)  

**Abstract**: Large language models (LLMs) often exhibit abrupt emergent behavior, whereby new abilities arise at certain points during their training. This phenomenon, commonly referred to as a ''phase transition'', remains poorly understood. In this study, we conduct an integrative analysis of such phase transitions by examining three interconnected perspectives: the similarity between LLMs and the human brain, the internal states of LLMs, and downstream task performance. We propose a novel interpretation for the learning dynamics of LLMs that vary in both training data and architecture, revealing that three phase transitions commonly emerge across these models during training: (1) alignment with the entire brain surges as LLMs begin adhering to task instructions Brain Alignment and Instruction Following, (2) unexpectedly, LLMs diverge from the brain during a period in which downstream task accuracy temporarily stagnates Brain Detachment and Stagnation, and (3) alignment with the brain reoccurs as LLMs become capable of solving the downstream tasks Brain Realignment and Consolidation. These findings illuminate the underlying mechanisms of phase transitions in LLMs, while opening new avenues for interdisciplinary research bridging AI and neuroscience. 

---
# Collective Reasoning Among LLMs A Framework for Answer Validation Without Ground Truth 

**Authors**: Seyed Pouyan Mousavi Davoudi, Alireza Shafiee Fard, Alireza Amiri-Margavi  

**Link**: [PDF](https://arxiv.org/pdf/2502.20758)  

**Abstract**: We present a collaborative framework where multiple large language models, namely GPT-4-0125-preview, Meta-LLaMA-3-70B-Instruct, Claude-3-Opus, and Gemini-1.5-Flash, work together to generate and respond to complex PhD-level probability questions in the absence of definitive ground truth. This study explores how inter-model consensus enhances response reliability and serves as a proxy for assessing the quality of generated questions. To quantify agreement and consistency, we employ statistical methods including chi-square tests, Fleiss' Kappa, and confidence interval analysis, measuring both response precision and question clarity. Our findings highlight that Claude and Gemini generate well-structured and less ambiguous questions, leading to higher inter-model agreement. This is reflected in their narrower confidence intervals and stronger alignment with answering models. Conversely, LLaMA demonstrates increased variability and lower reliability in question formulation, as indicated by broader confidence intervals and reduced consensus rates. These results suggest that multi-model collaboration not only enhances the reliability of responses but also provides a valuable framework for assessing and improving question quality in the absence of explicit ground truth. This research offers meaningful insights into optimizing AI-driven reasoning through collaborative large-language model interactions. 

---
# Teach-to-Reason with Scoring: Self-Explainable Rationale-Driven Multi-Trait Essay Scoring 

**Authors**: Heejin Do, Sangwon Ryu, Gary Geunbae Lee  

**Link**: [PDF](https://arxiv.org/pdf/2502.20748)  

**Abstract**: Multi-trait automated essay scoring (AES) systems provide a fine-grained evaluation of an essay's diverse aspects. While they excel in scoring, prior systems fail to explain why specific trait scores are assigned. This lack of transparency leaves instructors and learners unconvinced of the AES outputs, hindering their practical use. To address this, we propose a self-explainable Rationale-Driven Multi-trait automated Essay scoring (RaDME) framework. RaDME leverages the reasoning capabilities of large language models (LLMs) by distilling them into a smaller yet effective scorer. This more manageable student model is optimized to sequentially generate a trait score followed by the corresponding rationale, thereby inherently learning to select a more justifiable score by considering the subsequent rationale during training. Our findings indicate that while LLMs underperform in direct AES tasks, they excel in rationale generation when provided with precise numerical scores. Thus, RaDME integrates the superior reasoning capacities of LLMs into the robust scoring accuracy of an optimized smaller model. Extensive experiments demonstrate that RaDME achieves both accurate and adequate reasoning while supporting high-quality multi-trait scoring, significantly enhancing the transparency of AES. 

---
# Structured Preference Optimization for Vision-Language Long-Horizon Task Planning 

**Authors**: Xiwen Liang, Min Lin, Weiqi Ruan, Rongtao Xu, Yuecheng Liu, Jiaqi Chen, Bingqian Lin, Yuzheng Zhuang, Xiaodan Liang  

**Link**: [PDF](https://arxiv.org/pdf/2502.20742)  

**Abstract**: Existing methods for vision-language task planning excel in short-horizon tasks but often fall short in complex, long-horizon planning within dynamic environments. These challenges primarily arise from the difficulty of effectively training models to produce high-quality reasoning processes for long-horizon tasks. To address this, we propose Structured Preference Optimization (SPO), which aims to enhance reasoning and action selection in long-horizon task planning through structured preference evaluation and optimized training strategies. Specifically, SPO introduces: 1) Preference-Based Scoring and Optimization, which systematically evaluates reasoning chains based on task relevance, visual grounding, and historical consistency; and 2) Curriculum-Guided Training, where the model progressively adapts from simple to complex tasks, improving its generalization ability in long-horizon scenarios and enhancing reasoning robustness. To advance research in vision-language long-horizon task planning, we introduce ExtendaBench, a comprehensive benchmark covering 1,509 tasks across VirtualHome and Habitat 2.0, categorized into ultra-short, short, medium, and long tasks. Experimental results demonstrate that SPO significantly improves reasoning quality and final decision accuracy, outperforming prior methods on long-horizon tasks and underscoring the effectiveness of preference-driven optimization in vision-language task planning. Specifically, SPO achieves a +5.98% GCR and +4.68% SR improvement in VirtualHome and a +3.30% GCR and +2.11% SR improvement in Habitat over the best-performing baselines. 

---
# NeuroMorse: A Temporally Structured Dataset For Neuromorphic Computing 

**Authors**: Ben Walters, Yeshwanth Bethi, Taylor Kergan, Binh Nguyen, Amirali Amirsoleimani, Jason K. Eshraghian, Saeed Afshar, Mostafa Rahimi Azghadi  

**Link**: [PDF](https://arxiv.org/pdf/2502.20729)  

**Abstract**: Neuromorphic engineering aims to advance computing by mimicking the brain's efficient processing, where data is encoded as asynchronous temporal events. This eliminates the need for a synchronisation clock and minimises power consumption when no data is present. However, many benchmarks for neuromorphic algorithms primarily focus on spatial features, neglecting the temporal dynamics that are inherent to most sequence-based tasks. This gap may lead to evaluations that fail to fully capture the unique strengths and characteristics of neuromorphic systems. In this paper, we present NeuroMorse, a temporally structured dataset designed for benchmarking neuromorphic learning systems. NeuroMorse converts the top 50 words in the English language into temporal Morse code spike sequences. Despite using only two input spike channels for Morse dots and dashes, complex information is encoded through temporal patterns in the data. The proposed benchmark contains feature hierarchy at multiple temporal scales that test the capacity of neuromorphic algorithms to decompose input patterns into spatial and temporal hierarchies. We demonstrate that our training set is challenging to categorise using a linear classifier and that identifying keywords in the test set is difficult using conventional methods. The NeuroMorse dataset is available at Zenodo, with our accompanying code on GitHub at this https URL. 

---
# SPD: Sync-Point Drop for efficient tensor parallelism of Large Language Models 

**Authors**: Han-Byul Kim, Duc Hoang, Arnav Kundu, Mohammad Samragh, Minsik Cho  

**Link**: [PDF](https://arxiv.org/pdf/2502.20727)  

**Abstract**: With the rapid expansion in the scale of large language models (LLMs), enabling efficient distributed inference across multiple computing units has become increasingly critical. However, communication overheads from popular distributed inference techniques such as Tensor Parallelism pose a significant challenge to achieve scalability and low latency. Therefore, we introduce a novel optimization technique, Sync-Point Drop (SPD), to reduce communication overheads in tensor parallelism by selectively dropping synchronization on attention outputs. In detail, we first propose a block design that allows execution to proceed without communication through SPD. Second, we apply different SPD strategies to attention blocks based on their sensitivity to the model accuracy. The proposed methods effectively alleviate communication bottlenecks while minimizing accuracy degradation during LLM inference, offering a scalable solution for diverse distributed environments: SPD offered about 20% overall inference latency reduction with < 1% accuracy regression for LLaMA2-70B inference over 8 GPUs. 

---
# Generating Clinically Realistic EHR Data via a Hierarchy- and Semantics-Guided Transformer 

**Authors**: Guanglin Zhou, Sebastiano Barbieri  

**Link**: [PDF](https://arxiv.org/pdf/2502.20719)  

**Abstract**: Generating realistic synthetic electronic health records (EHRs) holds tremendous promise for accelerating healthcare research, facilitating AI model development and enhancing patient privacy. However, existing generative methods typically treat EHRs as flat sequences of discrete medical codes. This approach overlooks two critical aspects: the inherent hierarchical organization of clinical coding systems and the rich semantic context provided by code descriptions. Consequently, synthetic patient sequences often lack high clinical fidelity and have limited utility in downstream clinical tasks. In this paper, we propose the Hierarchy- and Semantics-Guided Transformer (HiSGT), a novel framework that leverages both hierarchical and semantic information for the generative process. HiSGT constructs a hierarchical graph to encode parent-child and sibling relationships among clinical codes and employs a graph neural network to derive hierarchy-aware embeddings. These are then fused with semantic embeddings extracted from a pre-trained clinical language model (e.g., ClinicalBERT), enabling the Transformer-based generator to more accurately model the nuanced clinical patterns inherent in real EHRs. Extensive experiments on the MIMIC-III and MIMIC-IV datasets demonstrate that HiSGT significantly improves the statistical alignment of synthetic data with real patient records, as well as supports robust downstream applications such as chronic disease classification. By addressing the limitations of conventional raw code-based generative models, HiSGT represents a significant step toward clinically high-fidelity synthetic data generation and a general paradigm suitable for interpretable medical code representation, offering valuable applications in data augmentation and privacy-preserving healthcare analytics. 

---
# WorldModelBench: Judging Video Generation Models As World Models 

**Authors**: Dacheng Li, Yunhao Fang, Yukang Chen, Shuo Yang, Shiyi Cao, Justin Wong, Michael Luo, Xiaolong Wang, Hongxu Yin, Joseph E. Gonzalez, Ion Stoica, Song Han, Yao Lu  

**Link**: [PDF](https://arxiv.org/pdf/2502.20694)  

**Abstract**: Video generation models have rapidly progressed, positioning themselves as video world models capable of supporting decision-making applications like robotics and autonomous driving. However, current benchmarks fail to rigorously evaluate these claims, focusing only on general video quality, ignoring important factors to world models such as physics adherence. To bridge this gap, we propose WorldModelBench, a benchmark designed to evaluate the world modeling capabilities of video generation models in application-driven domains. WorldModelBench offers two key advantages: (1) Against to nuanced world modeling violations: By incorporating instruction-following and physics-adherence dimensions, WorldModelBench detects subtle violations, such as irregular changes in object size that breach the mass conservation law - issues overlooked by prior benchmarks. (2) Aligned with large-scale human preferences: We crowd-source 67K human labels to accurately measure 14 frontier models. Using our high-quality human labels, we further fine-tune an accurate judger to automate the evaluation procedure, achieving 8.6% higher average accuracy in predicting world modeling violations than GPT-4o with 2B parameters. In addition, we demonstrate that training to align human annotations by maximizing the rewards from the judger noticeably improve the world modeling capability. The website is available at this https URL. 

---
# Unleashing the Potential of Two-Tower Models: Diffusion-Based Cross-Interaction for Large-Scale Matching 

**Authors**: Yihan Wang, Fei Xiong, Zhexin Han, Qi Song, Kaiqiao Zhan, Ben Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.20687)  

**Abstract**: Two-tower models are widely adopted in the industrial-scale matching stage across a broad range of application domains, such as content recommendations, advertisement systems, and search engines. This model efficiently handles large-scale candidate item screening by separating user and item representations. However, the decoupling network also leads to a neglect of potential information interaction between the user and item representations. Current state-of-the-art (SOTA) approaches include adding a shallow fully connected layer(i.e., COLD), which is limited by performance and can only be used in the ranking stage. For performance considerations, another approach attempts to capture historical positive interaction information from the other tower by regarding them as the input features(i.e., DAT). Later research showed that the gains achieved by this method are still limited because of lacking the guidance on the next user intent. To address the aforementioned challenges, we propose a "cross-interaction decoupling architecture" within our matching paradigm. This user-tower architecture leverages a diffusion module to reconstruct the next positive intention representation and employs a mixed-attention module to facilitate comprehensive cross-interaction. During the next positive intention generation, we further enhance the accuracy of its reconstruction by explicitly extracting the temporal drift within user behavior sequences. Experiments on two real-world datasets and one industrial dataset demonstrate that our method outperforms the SOTA two-tower models significantly, and our diffusion approach outperforms other generative models in reconstructing item representations. 

---
# JAM: Controllable and Responsible Text Generation via Causal Reasoning and Latent Vector Manipulation 

**Authors**: Yingbing Huang, Deming Chen, Abhishek K. Umrawal  

**Link**: [PDF](https://arxiv.org/pdf/2502.20684)  

**Abstract**: While large language models (LLMs) have made significant strides in generating coherent and contextually relevant text, they often function as opaque black boxes, trained on vast unlabeled datasets with statistical objectives, lacking an interpretable framework for responsible control. In this paper, we introduce JAM (Just A Move), a novel framework that interprets and controls text generation by integrating cause-effect analysis within the latent space of LLMs. Based on our observations, we uncover the inherent causality in LLM generation, which is critical for producing responsible and realistic outputs. Moreover, we explore latent vectors as fundamental components in LLM architectures, aiming to understand and manipulate them for more effective and efficient controllable text generation. We evaluate our framework using a range of tools, including the HHH criteria, toxicity reduction benchmarks, and GPT-4 alignment measures. Our results show that JAM achieves up to a 22% improvement over previous Controllable Text Generation (CTG) methods across multiple quantitative metrics and human-centric evaluations. Furthermore, JAM demonstrates greater computational efficiency compared to other CTG methods. These results highlight the effectiveness and efficiency of JAM for responsible and realistic text generation, paving the way for more interpretable and controllable models. 

---
# Fine-tuning BERT with Bidirectional LSTM for Fine-grained Movie Reviews Sentiment Analysis 

**Authors**: Gibson Nkhata, Susan Gauch, Usman Anjum, Justin Zhan  

**Link**: [PDF](https://arxiv.org/pdf/2502.20682)  

**Abstract**: Sentiment Analysis (SA) is instrumental in understanding peoples viewpoints facilitating social media monitoring recognizing products and brands and gauging customer satisfaction. Consequently SA has evolved into an active research domain within Natural Language Processing (NLP). Many approaches outlined in the literature devise intricate frameworks aimed at achieving high accuracy, focusing exclusively on either binary sentiment classification or fine-grained sentiment classification. In this paper our objective is to fine-tune the pre-trained BERT model with Bidirectional LSTM (BiLSTM) to enhance both binary and fine-grained SA specifically for movie reviews. Our approach involves conducting sentiment classification for each review followed by computing the overall sentiment polarity across all reviews. We present our findings on binary classification as well as fine-grained classification utilizing benchmark datasets. Additionally we implement and assess two accuracy improvement techniques Synthetic Minority Oversampling Technique (SMOTE) and NLP Augmenter (NLPAUG) to bolster the models generalization in fine-grained sentiment classification. Finally a heuristic algorithm is employed to calculate the overall polarity of predicted reviews from the BERT+BiLSTM output vector. Our approach performs comparably with state-of-the-art (SOTA) techniques in both classifications. For instance in binary classification we achieve 97.67% accuracy surpassing the leading SOTA model NB-weighted-BON+dv-cosine by 0.27% on the renowned IMDb dataset. Conversely for five-class classification on SST-5 while the top SOTA model RoBERTa+large+Self-explaining attains 55.5% accuracy our model achieves 59.48% accuracy surpassing the BERT-large baseline by 3.6%. 

---
# Disentangling Feature Structure: A Mathematically Provable Two-Stage Training Dynamics in Transformers 

**Authors**: Zixuan Gong, Jiaye Teng, Yong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.20681)  

**Abstract**: Transformers may exhibit two-stage training dynamics during the real-world training process. For instance, when training GPT-2 on the Counterfact dataset, the answers progress from syntactically incorrect to syntactically correct to semantically correct. However, existing theoretical analyses hardly account for this two-stage phenomenon. In this paper, we theoretically demonstrate how such two-stage training dynamics occur in transformers. Specifically, we analyze the dynamics of transformers using feature learning techniques under in-context learning regimes, based on a disentangled two-type feature structure. Such disentanglement of feature structure is general in practice, e.g., natural languages contain syntax and semantics, and proteins contain primary and secondary structures. To our best known, this is the first rigorous result regarding a two-stage optimization process in transformers. Additionally, a corollary indicates that such a two-stage process is closely related to the spectral properties of the attention weights, which accords well with empirical findings. 

---
# OpenEarthSensing: Large-Scale Fine-Grained Benchmark for Open-World Remote Sensing 

**Authors**: Xiang Xiang, Zhuo Xu, Yao Deng, Qinhao Zhou, Yifan Liang, Ke Chen, Qingfang Zheng, Yaowei Wang, Xilin Chen, Wen Gao  

**Link**: [PDF](https://arxiv.org/pdf/2502.20668)  

**Abstract**: In open-world remote sensing, deployed models must continuously adapt to a steady influx of new data, which often exhibits various shifts compared to what the model encountered during the training phase. To effectively handle the new data, models are required to detect semantic shifts, adapt to covariate shifts, and continuously update themselves. These challenges give rise to a variety of open-world tasks. However, existing open-world remote sensing studies typically train and test within a single dataset to simulate open-world conditions. Currently, there is a lack of large-scale benchmarks capable of evaluating multiple open-world tasks. In this paper, we introduce OpenEarthSensing, a large-scale fine-grained benchmark for open-world remote sensing. OpenEarthSensing includes 189 scene and objects categories, covering the vast majority of potential semantic shifts that may occur in the real world. Additionally, OpenEarthSensing encompasses five data domains with significant covariate shifts, including two RGB satellite domians, one RGB aerial domian, one MS RGB domian, and one infrared domian. The various domains provide a more comprehensive testbed for evaluating the generalization performance of open-world models. We conduct the baseline evaluation of current mainstream open-world tasks and methods on OpenEarthSensing, demonstrating that it serves as a challenging benchmark for open-world remote sensing. 

---
# Advancing AI-Powered Medical Image Synthesis: Insights from MedVQA-GI Challenge Using CLIP, Fine-Tuned Stable Diffusion, and Dream-Booth + LoRA 

**Authors**: Ojonugwa Oluwafemi Ejiga Peter, Md Mahmudur Rahman, Fahmi Khalifa  

**Link**: [PDF](https://arxiv.org/pdf/2502.20667)  

**Abstract**: The MEDVQA-GI challenge addresses the integration of AI-driven text-to-image generative models in medical diagnostics, aiming to enhance diagnostic capabilities through synthetic image generation. Existing methods primarily focus on static image analysis and lack the dynamic generation of medical imagery from textual descriptions. This study intends to partially close this gap by introducing a novel approach based on fine-tuned generative models to generate dynamic, scalable, and precise images from textual descriptions. Particularly, our system integrates fine-tuned Stable Diffusion and DreamBooth models, as well as Low-Rank Adaptation (LORA), to generate high-fidelity medical images. The problem is around two sub-tasks namely: image synthesis (IS) and optimal prompt production (OPG). The former creates medical images via verbal prompts, whereas the latter provides prompts that produce high-quality images in specified categories. The study emphasizes the limitations of traditional medical image generation methods, such as hand sketching, constrained datasets, static procedures, and generic models. Our evaluation measures showed that Stable Diffusion surpasses CLIP and DreamBooth + LORA in terms of producing high-quality, diversified images. Specifically, Stable Diffusion had the lowest Fréchet Inception Distance (FID) scores (0.099 for single center, 0.064 for multi-center, and 0.067 for combined), indicating higher image quality. Furthermore, it had the highest average Inception Score (2.327 across all datasets), indicating exceptional diversity and quality. This advances the field of AI-powered medical diagnosis. Future research will concentrate on model refining, dataset augmentation, and ethical considerations for efficiently implementing these advances into clinical practice 

---
# Dataset Distillation with Neural Characteristic Function: A Minmax Perspective 

**Authors**: Shaobo Wang, Yicun Yang, Zhiyuan Liu, Chenghao Sun, Xuming Hu, Conghui He, Linfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.20653)  

**Abstract**: Dataset distillation has emerged as a powerful approach for reducing data requirements in deep learning. Among various methods, distribution matching-based approaches stand out for their balance of computational efficiency and strong performance. However, existing distance metrics used in distribution matching often fail to accurately capture distributional differences, leading to unreliable measures of discrepancy. In this paper, we reformulate dataset distillation as a minmax optimization problem and introduce Neural Characteristic Function Discrepancy (NCFD), a comprehensive and theoretically grounded metric for measuring distributional differences. NCFD leverages the Characteristic Function (CF) to encapsulate full distributional information, employing a neural network to optimize the sampling strategy for the CF's frequency arguments, thereby maximizing the discrepancy to enhance distance estimation. Simultaneously, we minimize the difference between real and synthetic data under this optimized NCFD measure. Our approach, termed Neural Characteristic Function Matching (\mymethod{}), inherently aligns the phase and amplitude of neural features in the complex plane for both real and synthetic data, achieving a balance between realism and diversity in synthetic samples. Experiments demonstrate that our method achieves significant performance gains over state-of-the-art methods on both low- and high-resolution datasets. Notably, we achieve a 20.5\% accuracy boost on ImageSquawk. Our method also reduces GPU memory usage by over 300$\times$ and achieves 20$\times$ faster processing speeds compared to state-of-the-art methods. To the best of our knowledge, this is the first work to achieve lossless compression of CIFAR-100 on a single NVIDIA 2080 Ti GPU using only 2.3 GB of memory. 

---
# Consistency Evaluation of News Article Summaries Generated by Large (and Small) Language Models 

**Authors**: Colleen Gilhuly, Haleh Shahzad  

**Link**: [PDF](https://arxiv.org/pdf/2502.20647)  

**Abstract**: Text summarizing is a critical Natural Language Processing (NLP) task with applications ranging from information retrieval to content generation. Large Language Models (LLMs) have shown remarkable promise in generating fluent abstractive summaries but they can produce hallucinated details not grounded in the source text. Regardless of the method of generating a summary, high quality automated evaluations remain an open area of investigation. This paper embarks on an exploration of text summarization with a diverse set of techniques, including TextRank, BART, Mistral-7B-Instruct, and OpenAI GPT-3.5-Turbo. The generated summaries are evaluated using traditional metrics such as the Recall-Oriented Understudy for Gisting Evaluation (ROUGE) Score and Bidirectional Encoder Representations from Transformers (BERT) Score, as well as LLM-powered evaluation methods that directly assess a generated summary's consistency with the source text. We introduce a meta evaluation score which directly assesses the performance of the LLM evaluation system (prompt + model). We find that that all summarization models produce consistent summaries when tested on the XL-Sum dataset, exceeding the consistency of the reference summaries. 

---
# FedConv: A Learning-on-Model Paradigm for Heterogeneous Federated Clients 

**Authors**: Leming Shen, Qiang Yang, Kaiyan Cui, Yuanqing Zheng, Xiao-Yong Wei, Jianwei Liu, Jinsong Han  

**Link**: [PDF](https://arxiv.org/pdf/2502.20639)  

**Abstract**: Federated Learning (FL) facilitates collaborative training of a shared global model without exposing clients' private data. In practical FL systems, clients (e.g., edge servers, smartphones, and wearables) typically have disparate system resources. Conventional FL, however, adopts a one-size-fits-all solution, where a homogeneous large global model is transmitted to and trained on each client, resulting in an overwhelming workload for less capable clients and starvation for other clients. To address this issue, we propose FedConv, a client-friendly FL framework, which minimizes the computation and memory burden on resource-constrained clients by providing heterogeneous customized sub-models. FedConv features a novel learning-on-model paradigm that learns the parameters of the heterogeneous sub-models via convolutional compression. Unlike traditional compression methods, the compressed models in FedConv can be directly trained on clients without decompression. To aggregate the heterogeneous sub-models, we propose transposed convolutional dilation to convert them back to large models with a unified size while retaining personalized information from clients. The compression and dilation processes, transparent to clients, are optimized on the server leveraging a small public dataset. Extensive experiments on six datasets demonstrate that FedConv outperforms state-of-the-art FL systems in terms of model accuracy (by more than 35% on average), computation and communication overhead (with 33% and 25% reduction, respectively). 

---
# A Compact Model for Large-Scale Time Series Forecasting 

**Authors**: Chin-Chia Michael Yeh, Xiran Fan, Zhimeng Jiang, Yujie Fan, Huiyuan Chen, Uday Singh Saini, Vivian Lai, Xin Dai, Junpeng Wang, Zhongfang Zhuang, Liang Wang, Yan Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2502.20634)  

**Abstract**: Spatio-temporal data, which commonly arise in real-world applications such as traffic monitoring, financial transactions, and ride-share demands, represent a special category of multivariate time series. They exhibit two distinct characteristics: high dimensionality and commensurability across spatial locations. These attributes call for computationally efficient modeling approaches and facilitate the use of univariate forecasting models in a channel-independent fashion. SparseTSF, a recently introduced competitive univariate forecasting model, harnesses periodicity to achieve compactness by concentrating on cross-period dynamics, thereby extending the Pareto frontier with respect to model size and predictive performance. Nonetheless, it underperforms on spatio-temporal data due to an inadequate capture of intra-period temporal dependencies. To address this shortcoming, we propose UltraSTF, which integrates a cross-period forecasting module with an ultra-compact shape bank component. Our model effectively detects recurring patterns in time series through the attention mechanism of the shape bank component, thereby strengthening its ability to learn intra-period dynamics. UltraSTF achieves state-of-the-art performance on the LargeST benchmark while employing fewer than 0.2% of the parameters required by the second-best approaches, thus further extending the Pareto frontier of existing methods. 

---
# Lattice Protein Folding with Variational Annealing 

**Authors**: Shoummo Ahsan Khandoker, Estelle M. Inack, Mohamed Hibat-Allah  

**Link**: [PDF](https://arxiv.org/pdf/2502.20632)  

**Abstract**: Understanding the principles of protein folding is a cornerstone of computational biology, with implications for drug design, bioengineering, and the understanding of fundamental biological processes. Lattice protein folding models offer a simplified yet powerful framework for studying the complexities of protein folding, enabling the exploration of energetically optimal folds under constrained conditions. However, finding these optimal folds is a computationally challenging combinatorial optimization problem. In this work, we introduce a novel upper-bound training scheme that employs masking to identify the lowest-energy folds in two-dimensional Hydrophobic-Polar (HP) lattice protein folding. By leveraging Dilated Recurrent Neural Networks (RNNs) integrated with an annealing process driven by temperature-like fluctuations, our method accurately predicts optimal folds for benchmark systems of up to 60 beads. Our approach also effectively masks invalid folds from being sampled without compromising the autoregressive sampling properties of RNNs. This scheme is generalizable to three spatial dimensions and can be extended to lattice protein models with larger alphabets. Our findings emphasize the potential of advanced machine learning techniques in tackling complex protein folding problems and a broader class of constrained combinatorial optimization challenges. 

---
# Subtask-Aware Visual Reward Learning from Segmented Demonstrations 

**Authors**: Changyeon Kim, Minho Heo, Doohyun Lee, Jinwoo Shin, Honglak Lee, Joseph J. Lim, Kimin Lee  

**Link**: [PDF](https://arxiv.org/pdf/2502.20630)  

**Abstract**: Reinforcement Learning (RL) agents have demonstrated their potential across various robotic tasks. However, they still heavily rely on human-engineered reward functions, requiring extensive trial-and-error and access to target behavior information, often unavailable in real-world settings. This paper introduces REDS: REward learning from Demonstration with Segmentations, a novel reward learning framework that leverages action-free videos with minimal supervision. Specifically, REDS employs video demonstrations segmented into subtasks from diverse sources and treats these segments as ground-truth rewards. We train a dense reward function conditioned on video segments and their corresponding subtasks to ensure alignment with ground-truth reward signals by minimizing the Equivalent-Policy Invariant Comparison distance. Additionally, we employ contrastive learning objectives to align video representations with subtasks, ensuring precise subtask inference during online interactions. Our experiments show that REDS significantly outperforms baseline methods on complex robotic manipulation tasks in Meta-World and more challenging real-world tasks, such as furniture assembly in FurnitureBench, with minimal human intervention. Moreover, REDS facilitates generalization to unseen tasks and robot embodiments, highlighting its potential for scalable deployment in diverse environments. 

---
# Continuous Adversarial Text Representation Learning for Affective Recognition 

**Authors**: Seungah Son, Andrez Saurez, Dongsoo Har  

**Link**: [PDF](https://arxiv.org/pdf/2502.20613)  

**Abstract**: While pre-trained language models excel at semantic understanding, they often struggle to capture nuanced affective information critical for affective recognition tasks. To address these limitations, we propose a novel framework for enhancing emotion-aware embeddings in transformer-based models. Our approach introduces a continuous valence-arousal labeling system to guide contrastive learning, which captures subtle and multi-dimensional emotional nuances more effectively. Furthermore, we employ a dynamic token perturbation mechanism, using gradient-based saliency to focus on sentiment-relevant tokens, improving model sensitivity to emotional cues. The experimental results demonstrate that the proposed framework outperforms existing methods, achieving up to 15.5% improvement in the emotion classification benchmark, highlighting the importance of employing continuous labels. This improvement demonstrates that the proposed framework is effective in affective representation learning and enables precise and contextually relevant emotional understanding. 

---
# Leveraging Large Language Models for Building Interpretable Rule-Based Data-to-Text Systems 

**Authors**: Jędrzej Warczyński, Mateusz Lango, Ondrej Dusek  

**Link**: [PDF](https://arxiv.org/pdf/2502.20609)  

**Abstract**: We introduce a simple approach that uses a large language model (LLM) to automatically implement a fully interpretable rule-based data-to-text system in pure Python. Experimental evaluation on the WebNLG dataset showed that such a constructed system produces text of better quality (according to the BLEU and BLEURT metrics) than the same LLM prompted to directly produce outputs, and produces fewer hallucinations than a BART language model fine-tuned on the same data. Furthermore, at runtime, the approach generates text in a fraction of the processing time required by neural approaches, using only a single CPU 

---
# Exploring the Impact of Temperature Scaling in Softmax for Classification and Adversarial Robustness 

**Authors**: Hao Xuan, Bokai Yang, Xingyu Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.20604)  

**Abstract**: The softmax function is a fundamental component in deep learning. This study delves into the often-overlooked parameter within the softmax function, known as "temperature," providing novel insights into the practical and theoretical aspects of temperature scaling for image classification. Our empirical studies, adopting convolutional neural networks and transformers on multiple benchmark datasets, reveal that moderate temperatures generally introduce better overall performance. Through extensive experiments and rigorous theoretical analysis, we explore the role of temperature scaling in model training and unveil that temperature not only influences learning step size but also shapes the model's optimization direction. Moreover, for the first time, we discover a surprising benefit of elevated temperatures: enhanced model robustness against common corruption, natural perturbation, and non-targeted adversarial attacks like Projected Gradient Descent. We extend our discoveries to adversarial training, demonstrating that, compared to the standard softmax function with the default temperature value, higher temperatures have the potential to enhance adversarial training. The insights of this work open new avenues for improving model performance and security in deep learning applications. 

---
# Scalable Coordinated Learning for H2M/R Applications over Optical Access Networks (Invited) 

**Authors**: Sourav Mondal, Elaine Wong  

**Link**: [PDF](https://arxiv.org/pdf/2502.20598)  

**Abstract**: One of the primary research interests adhering to next-generation fiber-wireless access networks is human-to-machine/robot (H2M/R) collaborative communications facilitating Industry 5.0. This paper discusses scalable H2M/R communications across large geographical distances that also allow rapid onboarding of new machines/robots as $\sim72\%$ training time is saved through global-local coordinated learning. 

---
# LLMs Have Rhythm: Fingerprinting Large Language Models Using Inter-Token Times and Network Traffic Analysis 

**Authors**: Saeif Alhazbi, Ahmed Mohamed Hussain, Gabriele Oligeri, Panos Papadimitratos  

**Link**: [PDF](https://arxiv.org/pdf/2502.20589)  

**Abstract**: As Large Language Models (LLMs) become increasingly integrated into many technological ecosystems across various domains and industries, identifying which model is deployed or being interacted with is critical for the security and trustworthiness of the systems. Current verification methods typically rely on analyzing the generated output to determine the source model. However, these techniques are susceptible to adversarial attacks, operate in a post-hoc manner, and may require access to the model weights to inject a verifiable fingerprint. In this paper, we propose a novel passive and non-invasive fingerprinting technique that operates in real-time and remains effective even under encrypted network traffic conditions. Our method leverages the intrinsic autoregressive generation nature of language models, which generate text one token at a time based on all previously generated tokens, creating a unique temporal pattern like a rhythm or heartbeat that persists even when the output is streamed over a network. We find that measuring the Inter-Token Times (ITTs)-time intervals between consecutive tokens-can identify different language models with high accuracy. We develop a Deep Learning (DL) pipeline to capture these timing patterns using network traffic analysis and evaluate it on 16 Small Language Models (SLMs) and 10 proprietary LLMs across different deployment scenarios, including local host machine (GPU/CPU), Local Area Network (LAN), Remote Network, and Virtual Private Network (VPN). The experimental results confirm that our proposed technique is effective and maintains high accuracy even when tested in different network conditions. This work opens a new avenue for model identification in real-world scenarios and contributes to more secure and trustworthy language model deployment. 

---
# LiteASR: Efficient Automatic Speech Recognition with Low-Rank Approximation 

**Authors**: Keisuke Kamahori, Jungo Kasai, Noriyuki Kojima, Baris Kasikci  

**Link**: [PDF](https://arxiv.org/pdf/2502.20583)  

**Abstract**: Modern automatic speech recognition (ASR) models, such as OpenAI's Whisper, rely on deep encoder-decoder architectures, and their encoders are a critical bottleneck for efficient deployment due to high computational intensity. We introduce LiteASR, a low-rank compression scheme for ASR encoders that significantly reduces inference costs while maintaining transcription accuracy. Our approach leverages the strong low-rank properties observed in intermediate activations: by applying principal component analysis (PCA) with a small calibration dataset, we approximate linear transformations with a chain of low-rank matrix multiplications, and further optimize self-attention to work in the reduced dimension. Evaluation results show that our method can compress Whisper large-v3's encoder size by over 50%, matching Whisper medium's size with better transcription accuracy, thereby establishing a new Pareto-optimal frontier of efficiency and performance. The code of LiteASR is available at this https URL. 

---
# Interpreting CLIP with Hierarchical Sparse Autoencoders 

**Authors**: Vladimir Zaigrajew, Hubert Baniecki, Przemyslaw Biecek  

**Link**: [PDF](https://arxiv.org/pdf/2502.20578)  

**Abstract**: Sparse autoencoders (SAEs) are useful for detecting and steering interpretable features in neural networks, with particular potential for understanding complex multimodal representations. Given their ability to uncover interpretable features, SAEs are particularly valuable for analyzing large-scale vision-language models (e.g., CLIP and SigLIP), which are fundamental building blocks in modern systems yet remain challenging to interpret and control. However, current SAE methods are limited by optimizing both reconstruction quality and sparsity simultaneously, as they rely on either activation suppression or rigid sparsity constraints. To this end, we introduce Matryoshka SAE (MSAE), a new architecture that learns hierarchical representations at multiple granularities simultaneously, enabling a direct optimization of both metrics without compromise. MSAE establishes a new state-of-the-art Pareto frontier between reconstruction quality and sparsity for CLIP, achieving 0.99 cosine similarity and less than 0.1 fraction of variance unexplained while maintaining ~80% sparsity. Finally, we demonstrate the utility of MSAE as a tool for interpreting and controlling CLIP by extracting over 120 semantic concepts from its representation to perform concept-based similarity search and bias analysis in downstream tasks like CelebA. 

---
# PFformer: A Position-Free Transformer Variant for Extreme-Adaptive Multivariate Time Series Forecasting 

**Authors**: Yanhong Li, David C. Anastasiu  

**Link**: [PDF](https://arxiv.org/pdf/2502.20571)  

**Abstract**: Multivariate time series (MTS) forecasting is vital in fields like weather, energy, and finance. However, despite deep learning advancements, traditional Transformer-based models often diminish the effect of crucial inter-variable relationships by singular token embedding and struggle to effectively capture complex dependencies among variables, especially in datasets with rare or extreme events. These events create significant imbalances and lead to high skewness, complicating accurate prediction efforts. This study introduces PFformer, a position-free Transformer-based model designed for single-target MTS forecasting, specifically for challenging datasets characterized by extreme variability. PFformer integrates two novel embedding strategies: Enhanced Feature-based Embedding (EFE) and Auto-Encoder-based Embedding (AEE). EFE effectively encodes inter-variable dependencies by mapping related sequence subsets to high-dimensional spaces without positional constraints, enhancing the encoder's functionality. PFformer shows superior forecasting accuracy without the traditional limitations of positional encoding in MTS modeling. We evaluated PFformer across four challenging datasets, focusing on two key forecasting scenarios: long sequence prediction for 3 days ahead and rolling predictions every four hours to reflect real-time decision-making processes in water management. PFformer demonstrated remarkable improvements, from 20% to 60%, compared with state-of-the-art models. 

---
# DPZV: Resource Efficient ZO Optimization For Differentially Private VFL 

**Authors**: Jianing Zhang, Evan Chen, Chaoyue Liu, Christopher G. Brinton  

**Link**: [PDF](https://arxiv.org/pdf/2502.20565)  

**Abstract**: Vertical Federated Learning (VFL) enables collaborative model training across feature-partitioned data, yet faces significant privacy risks and inefficiencies when scaling to large models. We propose DPZV, a memory-efficient Zeroth-Order(ZO) optimization framework that integrates differential privacy (DP) with vertical federated learning, addressing three critical challenges: (1) privacy vulnerabilities from gradient leakage, (2) high computation/communication costs of first-order methods, and (3) excessive memory footprint in conventional zeroth-order approaches. Our framework eliminates backpropagation through two-point gradient estimation, reducing client memory usage by 90\% compared to first-order counterparts while enabling asynchronous communication. By strategically injecting Gaussian noise on the server, DPZV achieves rigorous $(\epsilon, \delta)$-DP guarantees without third-party trust assumptions. Theoretical analysis establishes a convergence rate matching centralized case under non-convex objectives. Extensive experiments on image and NLP benchmarks demonstrate that DPZV outperforms all baselines in accuracy while providing strong privacy assurances ($\epsilon \leq 10$) and requiring far fewer computation resources, establishing new state-of-the-art privacy-utility tradeoffs for resource-constrained VFL deployments. 

---
# $Q\sharp$: Provably Optimal Distributional RL for LLM Post-Training 

**Authors**: Jin Peng Zhou, Kaiwen Wang, Jonathan Chang, Zhaolin Gao, Nathan Kallus, Kilian Q. Weinberger, Kianté Brantley, Wen Sun  

**Link**: [PDF](https://arxiv.org/pdf/2502.20548)  

**Abstract**: Reinforcement learning (RL) post-training is crucial for LLM alignment and reasoning, but existing policy-based methods, such as PPO and DPO, can fall short of fixing shortcuts inherited from pre-training. In this work, we introduce $Q\sharp$, a value-based algorithm for KL-regularized RL that guides the reference policy using the optimal regularized $Q$ function. We propose to learn the optimal $Q$ function using distributional RL on an aggregated online dataset. Unlike prior value-based baselines that guide the model using unregularized $Q$-values, our method is theoretically principled and provably learns the optimal policy for the KL-regularized RL problem. Empirically, $Q\sharp$ outperforms prior baselines in math reasoning benchmarks while maintaining a smaller KL divergence to the reference policy. Theoretically, we establish a reduction from KL-regularized RL to no-regret online learning, providing the first bounds for deterministic MDPs under only realizability. Thanks to distributional RL, our bounds are also variance-dependent and converge faster when the reference policy has small variance. In sum, our results highlight $Q\sharp$ as an effective approach for post-training LLMs, offering both improved performance and theoretical guarantees. The code can be found at this https URL. 

---
# Revisiting Kernel Attention with Correlated Gaussian Process Representation 

**Authors**: Long Minh Bui, Tho Tran Huu, Duy Dinh, Tan Minh Nguyen, Trong Nghia Hoang  

**Link**: [PDF](https://arxiv.org/pdf/2502.20525)  

**Abstract**: Transformers have increasingly become the de facto method to model sequential data with state-of-the-art performance. Due to its widespread use, being able to estimate and calibrate its modeling uncertainty is important to understand and design robust transformer models. To achieve this, previous works have used Gaussian processes (GPs) to perform uncertainty calibration for the attention units of transformers and attained notable successes. However, such approaches have to confine the transformers to the space of symmetric attention to ensure the necessary symmetric requirement of their GP's kernel specification, which reduces the representation capacity of the model. To mitigate this restriction, we propose the Correlated Gaussian Process Transformer (CGPT), a new class of transformers whose self-attention units are modeled as cross-covariance between two correlated GPs (CGPs). This allows asymmetries in attention and can enhance the representation capacity of GP-based transformers. We also derive a sparse approximation for CGP to make it scale better. Our empirical studies show that both CGP-based and sparse CGP-based transformers achieve better performance than state-of-the-art GP-based transformers on a variety of benchmark tasks. The code for our experiments is available at this https URL. 

---
# Personas Evolved: Designing Ethical LLM-Based Conversational Agent Personalities 

**Authors**: Smit Desai, Mateusz Dubiel, Nima Zargham, Thomas Mildner, Laura Spillner  

**Link**: [PDF](https://arxiv.org/pdf/2502.20513)  

**Abstract**: The emergence of Large Language Models (LLMs) has revolutionized Conversational User Interfaces (CUIs), enabling more dynamic, context-aware, and human-like interactions across diverse domains, from social sciences to healthcare. However, the rapid adoption of LLM-based personas raises critical ethical and practical concerns, including bias, manipulation, and unforeseen social consequences. Unlike traditional CUIs, where personas are carefully designed with clear intent, LLM-based personas generate responses dynamically from vast datasets, making their behavior less predictable and harder to govern. This workshop aims to bridge the gap between CUI and broader AI communities by fostering a cross-disciplinary dialogue on the responsible design and evaluation of LLM-based personas. Bringing together researchers, designers, and practitioners, we will explore best practices, develop ethical guidelines, and promote frameworks that ensure transparency, inclusivity, and user-centered interactions. By addressing these challenges collaboratively, we seek to shape the future of LLM-driven CUIs in ways that align with societal values and expectations. 

---
# TripCraft: A Benchmark for Spatio-Temporally Fine Grained Travel Planning 

**Authors**: Soumyabrata Chaudhuri, Pranav Purkar, Ritwik Raghav, Shubhojit Mallick, Manish Gupta, Abhik Jana, Shreya Ghosh  

**Link**: [PDF](https://arxiv.org/pdf/2502.20508)  

**Abstract**: Recent advancements in probing Large Language Models (LLMs) have explored their latent potential as personalized travel planning agents, yet existing benchmarks remain limited in real world applicability. Existing datasets, such as TravelPlanner and TravelPlanner+, suffer from semi synthetic data reliance, spatial inconsistencies, and a lack of key travel constraints, making them inadequate for practical itinerary generation. To address these gaps, we introduce TripCraft, a spatiotemporally coherent travel planning dataset that integrates real world constraints, including public transit schedules, event availability, diverse attraction categories, and user personas for enhanced personalization. To evaluate LLM generated plans beyond existing binary validation methods, we propose five continuous evaluation metrics, namely Temporal Meal Score, Temporal Attraction Score, Spatial Score, Ordering Score, and Persona Score which assess itinerary quality across multiple dimensions. Our parameter informed setting significantly enhances meal scheduling, improving the Temporal Meal Score from 61% to 80% in a 7 day scenario. TripCraft establishes a new benchmark for LLM driven personalized travel planning, offering a more realistic, constraint aware framework for itinerary generation. Dataset and Codebase will be made publicly available upon acceptance. 

---
# A Thousand Words or An Image: Studying the Influence of Persona Modality in Multimodal LLMs 

**Authors**: Julius Broomfield, Kartik Sharma, Srijan Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2502.20504)  

**Abstract**: Large language models (LLMs) have recently demonstrated remarkable advancements in embodying diverse personas, enhancing their effectiveness as conversational agents and virtual assistants. Consequently, LLMs have made significant strides in processing and integrating multimodal information. However, even though human personas can be expressed in both text and image, the extent to which the modality of a persona impacts the embodiment by the LLM remains largely unexplored. In this paper, we investigate how do different modalities influence the expressiveness of personas in multimodal LLMs. To this end, we create a novel modality-parallel dataset of 40 diverse personas varying in age, gender, occupation, and location. This consists of four modalities to equivalently represent a persona: image-only, text-only, a combination of image and small text, and typographical images, where text is visually stylized to convey persona-related attributes. We then create a systematic evaluation framework with 60 questions and corresponding metrics to assess how well LLMs embody each persona across its attributes and scenarios. Comprehensive experiments on $5$ multimodal LLMs show that personas represented by detailed text show more linguistic habits, while typographical images often show more consistency with the persona. Our results reveal that LLMs often overlook persona-specific details conveyed through images, highlighting underlying limitations and paving the way for future research to bridge this gap. We release the data and code at this https URL . 

---
# Unified Kernel-Segregated Transpose Convolution Operation 

**Authors**: Vijay Srinivas Tida, Md Imran Hossen, Liqun Shan, Sai Venkatesh Chilukoti, Sonya Hsu, Xiali Hei  

**Link**: [PDF](https://arxiv.org/pdf/2502.20493)  

**Abstract**: The optimization of the transpose convolution layer for deep learning applications is achieved with the kernel segregation mechanism. However, kernel segregation has disadvantages, such as computing extra elements to obtain the output feature map with odd dimensions while launching a thread. To mitigate this problem, we introduce a unified kernel segregation approach that limits the usage of memory and computational resources by employing one unified kernel to execute four sub-kernels. The findings reveal that the suggested approach achieves an average computational speedup of 2.03x (3.89x) when tested on specific datasets with an RTX 2070 GPU (Intel Xeon CPU). The ablation study shows an average computational speedup of 3.5x when evaluating the transpose convolution layers from well-known Generative Adversarial Networks (GANs). The implementation of the proposed method for the transpose convolution layers in the EB-GAN model demonstrates significant memory savings of up to 35 MB. 

---
# EgoNormia: Benchmarking Physical Social Norm Understanding 

**Authors**: MohammadHossein Rezaei, Yicheng Fu, Phil Cuvin, Caleb Ziems, Yanzhe Zhang, Hao Zhu, Diyi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2502.20490)  

**Abstract**: Human activity is moderated by norms. When performing actions in the real world, humans not only follow norms, but also consider the trade-off between different norms However, machines are often trained without explicit supervision on norm understanding and reasoning, especially when the norms are grounded in a physical and social context. To improve and evaluate the normative reasoning capability of vision-language models (VLMs), we present EgoNormia $\|\epsilon\|$, consisting of 1,853 ego-centric videos of human interactions, each of which has two related questions evaluating both the prediction and justification of normative actions. The normative actions encompass seven categories: safety, privacy, proxemics, politeness, cooperation, coordination/proactivity, and communication/legibility. To compile this dataset at scale, we propose a novel pipeline leveraging video sampling, automatic answer generation, filtering, and human validation. Our work demonstrates that current state-of-the-art vision-language models lack robust norm understanding, scoring a maximum of 45% on EgoNormia (versus a human bench of 92%). Our analysis of performance in each dimension highlights the significant risks of safety, privacy, and the lack of collaboration and communication capability when applied to real-world agents. We additionally show that through a retrieval-based generation method, it is possible to use EgoNomia to enhance normative reasoning in VLMs. 

---
# Promote, Suppress, Iterate: How Language Models Answer One-to-Many Factual Queries 

**Authors**: Tianyi Lorena Yan, Robin Jia  

**Link**: [PDF](https://arxiv.org/pdf/2502.20475)  

**Abstract**: To answer one-to-many factual queries (e.g., listing cities of a country), a language model (LM) must simultaneously recall knowledge and avoid repeating previous answers. How are these two subtasks implemented and integrated internally? Across multiple datasets and models, we identify a promote-then-suppress mechanism: the model first recalls all answers, and then suppresses previously generated ones. Specifically, LMs use both the subject and previous answer tokens to perform knowledge recall, with attention propagating subject information and MLPs promoting the answers. Then, attention attends to and suppresses previous answer tokens, while MLPs amplify the suppression signal. Our mechanism is corroborated by extensive experimental evidence: in addition to using early decoding and causal tracing, we analyze how components use different tokens by introducing both \emph{Token Lens}, which decodes aggregated attention updates from specified tokens, and a knockout method that analyzes changes in MLP outputs after removing attention to specified tokens. Overall, we provide new insights into how LMs' internal components interact with different input tokens to support complex factual recall. Code is available at this https URL. 

---
# Will AI replace Software Engineers? Hold your Breath 

**Authors**: Abhik Roychoudhury, Andreas Zeller  

**Link**: [PDF](https://arxiv.org/pdf/2502.20429)  

**Abstract**: Artificial Intelligence (AI) technology such as Large Language Models (LLMs) have become extremely popular in creating code. This has led to the conjecture that future software jobs will be exclusively conducted by LLMs, and the software industry will cease to exist. But software engineering is much more than producing code -- notably, \emph{maintaining} large software and keeping it reliable is a major part of software engineering, which LLMs are not yet capable of. 

---
# DeePen: Penetration Testing for Audio Deepfake Detection 

**Authors**: Nicolas Müller, Piotr Kawa, Adriana Stan, Thien-Phuc Doan, Souhwan Jung, Wei Herng Choong, Philip Sperl, Konstantin Böttinger  

**Link**: [PDF](https://arxiv.org/pdf/2502.20427)  

**Abstract**: Deepfakes - manipulated or forged audio and video media - pose significant security risks to individuals, organizations, and society at large. To address these challenges, machine learning-based classifiers are commonly employed to detect deepfake content. In this paper, we assess the robustness of such classifiers through a systematic penetration testing methodology, which we introduce as DeePen. Our approach operates without prior knowledge of or access to the target deepfake detection models. Instead, it leverages a set of carefully selected signal processing modifications - referred to as attacks - to evaluate model vulnerabilities. Using DeePen, we analyze both real-world production systems and publicly available academic model checkpoints, demonstrating that all tested systems exhibit weaknesses and can be reliably deceived by simple manipulations such as time-stretching or echo addition. Furthermore, our findings reveal that while some attacks can be mitigated by retraining detection systems with knowledge of the specific attack, others remain persistently effective. We release all associated code. 

---
# Among Them: A game-based framework for assessing persuasion capabilities of LLMs 

**Authors**: Mateusz Idziejczak, Vasyl Korzavatykh, Mateusz Stawicki, Andrii Chmutov, Marcin Korcz, Iwo Błądek, Dariusz Brzezinski  

**Link**: [PDF](https://arxiv.org/pdf/2502.20426)  

**Abstract**: The proliferation of large language models (LLMs) and autonomous AI agents has raised concerns about their potential for automated persuasion and social influence. While existing research has explored isolated instances of LLM-based manipulation, systematic evaluations of persuasion capabilities across different models remain limited. In this paper, we present an Among Us-inspired game framework for assessing LLM deception skills in a controlled environment. The proposed framework makes it possible to compare LLM models by game statistics, as well as quantify in-game manipulation according to 25 persuasion strategies from social psychology and rhetoric. Experiments between 8 popular language models of different types and sizes demonstrate that all tested models exhibit persuasive capabilities, successfully employing 22 of the 25 anticipated techniques. We also find that larger models do not provide any persuasion advantage over smaller models and that longer model outputs are negatively correlated with the number of games won. Our study provides insights into the deception capabilities of LLMs, as well as tools and data for fostering future research on the topic. 

---
# Efficient Risk-sensitive Planning via Entropic Risk Measures 

**Authors**: Alexandre Marthe, Samuel Bounan, Aurélien Garivier, Claire Vernade  

**Link**: [PDF](https://arxiv.org/pdf/2502.20423)  

**Abstract**: Risk-sensitive planning aims to identify policies maximizing some tail-focused metrics in Markov Decision Processes (MDPs). Such an optimization task can be very costly for the most widely used and interpretable metrics such as threshold probabilities or (Conditional) Values at Risk. Indeed, previous work showed that only Entropic Risk Measures (EntRM) can be efficiently optimized  through dynamic programming, leaving a hard-to-interpret parameter to choose.     We show that the computation of the full set of optimal policies for EntRM across parameter values leads to tight approximations for the metrics of interest. We prove that this optimality front can be computed effectively thanks to a novel structural analysis and smoothness properties of entropic risks.     Empirical results demonstrate that our approach achieves strong performance in a variety of decision-making scenarios. 

---
# SEKI: Self-Evolution and Knowledge Inspiration based Neural Architecture Search via Large Language Models 

**Authors**: Zicheng Cai, Yaohua Tang, Yutao Lai, Hua Wang, Zhi Chen, Hao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.20422)  

**Abstract**: We introduce SEKI, a novel large language model (LLM)-based neural architecture search (NAS) method. Inspired by the chain-of-thought (CoT) paradigm in modern LLMs, SEKI operates in two key stages: self-evolution and knowledge distillation. In the self-evolution stage, LLMs initially lack sufficient reference examples, so we implement an iterative refinement mechanism that enhances architectures based on performance feedback. Over time, this process accumulates a repository of high-performance architectures. In the knowledge distillation stage, LLMs analyze common patterns among these architectures to generate new, optimized designs. Combining these two stages, SEKI greatly leverages the capacity of LLMs on NAS and without requiring any domain-specific data. Experimental results show that SEKI achieves state-of-the-art (SOTA) performance across various datasets and search spaces while requiring only 0.05 GPU-days, outperforming existing methods in both efficiency and accuracy. Furthermore, SEKI demonstrates strong generalization capabilities, achieving SOTA-competitive results across multiple tasks. 

---
# Backpropagation-free Spiking Neural Networks with the Forward-Forward Algorithm 

**Authors**: Mohammadnavid Ghader, Saeed Reza Kheradpisheh, Bahar Farahani, Mahmood Fazlali  

**Link**: [PDF](https://arxiv.org/pdf/2502.20411)  

**Abstract**: Spiking Neural Networks (SNNs) offer a biologically inspired computational paradigm that emulates neuronal activity through discrete spike-based processing. Despite their advantages, training SNNs with traditional backpropagation (BP) remains challenging due to computational inefficiencies and a lack of biological plausibility. This study explores the Forward-Forward (FF) algorithm as an alternative learning framework for SNNs. Unlike backpropagation, which relies on forward and backward passes, the FF algorithm employs two forward passes, enabling localized learning, enhanced computational efficiency, and improved compatibility with neuromorphic hardware. We introduce an FF-based SNN training framework and evaluate its performance across both non-spiking (MNIST, Fashion-MNIST, CIFAR-10) and spiking (Neuro-MNIST, SHD) datasets. Experimental results demonstrate that our model surpasses existing FF-based SNNs by over 5% on MNIST and Fashion-MNIST while achieving accuracy comparable to state-of-the-art backpropagation-trained SNNs. On more complex tasks such as CIFAR-10 and SHD, our approach outperforms other SNN models by up to 6% and remains competitive with leading backpropagation-trained SNNs. These findings highlight the FF algorithm's potential to advance SNN training methodologies and neuromorphic computing by addressing key limitations of backpropagation. 

---
# Brain-Inspired Exploration of Functional Networks and Key Neurons in Large Language Models 

**Authors**: Yiheng Liu, Xiaohui Gao, Haiyang Sun, Bao Ge, Tianming Liu, Junwei Han, Xintao Hu  

**Link**: [PDF](https://arxiv.org/pdf/2502.20408)  

**Abstract**: In recent years, the rapid advancement of large language models (LLMs) in natural language processing has sparked significant interest among researchers to understand their mechanisms and functional characteristics. Although existing studies have attempted to explain LLM functionalities by identifying and interpreting specific neurons, these efforts mostly focus on individual neuron contributions, neglecting the fact that human brain functions are realized through intricate interaction networks. Inspired by cognitive neuroscience research on functional brain networks (FBNs), this study introduces a novel approach to investigate whether similar functional networks exist within LLMs. We use methods similar to those in the field of functional neuroimaging analysis to locate and identify functional networks in LLM. Experimental results show that, similar to the human brain, LLMs contain functional networks that frequently recur during operation. Further analysis shows that these functional networks are crucial for LLM performance. Masking key functional networks significantly impairs the model's performance, while retaining just a subset of these networks is adequate to maintain effective operation. This research provides novel insights into the interpretation of LLMs and the lightweighting of LLMs for certain downstream tasks. Code is available at this https URL. 

---
# Pause-Tuning for Long-Context Comprehension: A Lightweight Approach to LLM Attention Recalibration 

**Authors**: James Begin, Namit Agrawal, Eshan Singh, Yicheng Fu, Sean O'Brien, Vasu Sharma, Kevin Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2502.20405)  

**Abstract**: LLMs have demonstrated remarkable proficiency in understanding tasks but continue to struggle with long-context comprehension, particularly with content located in the middle of extensive inputs. This limitation, known as the Lost-in-the-Middle (LITM) problem, hinders models from fully processing and utilizing information across lengthy contexts. To address this issue, we introduce pause-tuning, a technique that redistributes attention to enhance comprehension of long-context inputs. Our approach involves fine-tuning language models on datasets with artificially inserted pause tokens, which serve to segment the input into smaller, more manageable parts. We evaluate pause-tuning against alternative approaches using the Needle-in-a-Haystack benchmark, where models must retrieve information embedded within contexts of up to 128K tokens. Experimental results demonstrate significant performance gains, with the LLaMA 3.2 3B Instruct model and the LLaMA 3.1 8B Instruct model improving by 10.61% and 3.57% respectively on average, suggesting that pause-tuning successfully enhances attention redistribution and improves long-context retention. The code and data are available at this https URL. 

---
# Adversarial Robustness of Partitioned Quantum Classifiers 

**Authors**: Pouya Kananian, Hans-Arno Jacobsen  

**Link**: [PDF](https://arxiv.org/pdf/2502.20403)  

**Abstract**: Adversarial robustness in quantum classifiers is a critical area of study, providing insights into their performance compared to classical models and uncovering potential advantages inherent to quantum machine learning. In the NISQ era of quantum computing, circuit cutting is a notable technique for simulating circuits that exceed the qubit limitations of current devices, enabling the distribution of a quantum circuit's execution across multiple quantum processing units through classical communication. We examine how partitioning quantum classifiers through circuit cutting increase their susceptibility to adversarial attacks, establishing a link between attacking the state preparation channels in wire cutting and implementing adversarial gates within intermediate layers of a quantum classifier. We then proceed to study the latter problem from both a theoretical and experimental perspective. 

---
