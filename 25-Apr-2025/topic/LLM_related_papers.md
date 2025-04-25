# DataScout: Automatic Data Fact Retrieval for Statement Augmentation with an LLM-Based Agent 

**Authors**: Chuer Chen, Yuqi Liu, Danqing Shi, Shixiong Cao, Nan Cao  

**Link**: [PDF](https://arxiv.org/pdf/2504.17334)  

**Abstract**: A data story typically integrates data facts from multiple perspectives and stances to construct a comprehensive and objective narrative. However, retrieving these facts demands time for data search and challenges the creator's analytical skills. In this work, we introduce DataScout, an interactive system that automatically performs reasoning and stance-based data facts retrieval to augment the user's statement. Particularly, DataScout leverages an LLM-based agent to construct a retrieval tree, enabling collaborative control of its expansion between users and the agent. The interface visualizes the retrieval tree as a mind map that eases users to intuitively steer the retrieval direction and effectively engage in reasoning and analysis. We evaluate the proposed system through case studies and in-depth expert interviews. Our evaluation demonstrates that DataScout can effectively retrieve multifaceted data facts from different stances, helping users verify their statements and enhance the credibility of their stories. 

---
# Does Knowledge Distillation Matter for Large Language Model based Bundle Generation? 

**Authors**: Kaidong Feng, Zhu Sun, Jie Yang, Hui Fang, Xinghua Qu, Wenyuan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.17220)  

**Abstract**: LLMs are increasingly explored for bundle generation, thanks to their reasoning capabilities and knowledge. However, deploying large-scale LLMs introduces significant efficiency challenges, primarily high computational costs during fine-tuning and inference due to their massive parameterization. Knowledge distillation (KD) offers a promising solution, transferring expertise from large teacher models to compact student models. This study systematically investigates knowledge distillation approaches for bundle generation, aiming to minimize computational demands while preserving performance. We explore three critical research questions: (1) how does the format of KD impact bundle generation performance? (2) to what extent does the quantity of distilled knowledge influence performance? and (3) how do different ways of utilizing the distilled knowledge affect performance? We propose a comprehensive KD framework that (i) progressively extracts knowledge (patterns, rules, deep thoughts); (ii) captures varying quantities of distilled knowledge through different strategies; and (iii) exploits complementary LLM adaptation techniques (in-context learning, supervised fine-tuning, combination) to leverage distilled knowledge in small student models for domain-specific adaptation and enhanced efficiency. Extensive experiments provide valuable insights into how knowledge format, quantity, and utilization methodologies collectively shape LLM-based bundle generation performance, exhibiting KD's significant potential for more efficient yet effective LLM-based bundle generation. 

---
# Adaptive Orchestration of Modular Generative Information Access Systems 

**Authors**: Mohanna Hoveyda, Harrie Oosterhuis, Arjen P. de Vries, Maarten de Rijke, Faegheh Hasibi  

**Link**: [PDF](https://arxiv.org/pdf/2504.17454)  

**Abstract**: Advancements in large language models (LLMs) have driven the emergence of complex new systems to provide access to information, that we will collectively refer to as modular generative information access (GenIA) systems. They integrate a broad and evolving range of specialized components, including LLMs, retrieval models, and a heterogeneous set of sources and tools. While modularity offers flexibility, it also raises critical challenges: How can we systematically characterize the space of possible modules and their interactions? How can we automate and optimize interactions among these heterogeneous components? And, how do we enable this modular system to dynamically adapt to varying user query requirements and evolving module capabilities? In this perspective paper, we argue that the architecture of future modular generative information access systems will not just assemble powerful components, but enable a self-organizing system through real-time adaptive orchestration -- where components' interactions are dynamically configured for each user input, maximizing information relevance while minimizing computational overhead. We give provisional answers to the questions raised above with a roadmap that depicts the key principles and methods for designing such an adaptive modular system. We identify pressing challenges, and propose avenues for addressing them in the years ahead. This perspective urges the IR community to rethink modular system designs for developing adaptive, self-optimizing, and future-ready architectures that evolve alongside their rapidly advancing underlying technologies. 

---
# Beyond Whole Dialogue Modeling: Contextual Disentanglement for Conversational Recommendation 

**Authors**: Guojia An, Jie Zou, Jiwei Wei, Chaoning Zhang, Fuming Sun, Yang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.17427)  

**Abstract**: Conversational recommender systems aim to provide personalized recommendations by analyzing and utilizing contextual information related to dialogue. However, existing methods typically model the dialogue context as a whole, neglecting the inherent complexity and entanglement within the dialogue. Specifically, a dialogue comprises both focus information and background information, which mutually influence each other. Current methods tend to model these two types of information mixedly, leading to misinterpretation of users' actual needs, thereby lowering the accuracy of recommendations. To address this issue, this paper proposes a novel model to introduce contextual disentanglement for improving conversational recommender systems, named DisenCRS. The proposed model DisenCRS employs a dual disentanglement framework, including self-supervised contrastive disentanglement and counterfactual inference disentanglement, to effectively distinguish focus information and background information from the dialogue context under unsupervised conditions. Moreover, we design an adaptive prompt learning module to automatically select the most suitable prompt based on the specific dialogue context, fully leveraging the power of large language models. Experimental results on two widely used public datasets demonstrate that DisenCRS significantly outperforms existing conversational recommendation models, achieving superior performance on both item recommendation and response generation tasks. 

---
# Assessing the Capability of Large Language Models for Domain-Specific Ontology Generation 

**Authors**: Anna Sofia Lippolis, Mohammad Javad Saeedizade, Robin Keskisarkka, Aldo Gangemi, Eva Blomqvist, Andrea Giovanni Nuzzolese  

**Link**: [PDF](https://arxiv.org/pdf/2504.17402)  

**Abstract**: Large Language Models (LLMs) have shown significant potential for ontology engineering. However, it is still unclear to what extent they are applicable to the task of domain-specific ontology generation. In this study, we explore the application of LLMs for automated ontology generation and evaluate their performance across different domains. Specifically, we investigate the generalizability of two state-of-the-art LLMs, DeepSeek and o1-preview, both equipped with reasoning capabilities, by generating ontologies from a set of competency questions (CQs) and related user stories. Our experimental setup comprises six distinct domains carried out in existing ontology engineering projects and a total of 95 curated CQs designed to test the models' reasoning for ontology engineering. Our findings show that with both LLMs, the performance of the experiments is remarkably consistent across all domains, indicating that these methods are capable of generalizing ontology generation tasks irrespective of the domain. These results highlight the potential of LLM-based approaches in achieving scalable and domain-agnostic ontology construction and lay the groundwork for further research into enhancing automated reasoning and knowledge representation techniques. 

---
# Leveraging LLMs as Meta-Judges: A Multi-Agent Framework for Evaluating LLM Judgments 

**Authors**: Yuran Li, Jama Hussein Mohamud, Chongren Sun, Di Wu, Benoit Boulet  

**Link**: [PDF](https://arxiv.org/pdf/2504.17087)  

**Abstract**: Large language models (LLMs) are being widely applied across various fields, but as tasks become more complex, evaluating their responses is increasingly challenging. Compared to human evaluators, the use of LLMs to support performance evaluation offers a more efficient alternative. However, most studies focus mainly on aligning LLMs' judgments with human preferences, overlooking the existence of biases and mistakes in human judgment. Furthermore, how to select suitable LLM judgments given multiple potential LLM responses remains underexplored. To address these two aforementioned issues, we propose a three-stage meta-judge selection pipeline: 1) developing a comprehensive rubric with GPT-4 and human experts, 2) using three advanced LLM agents to score judgments, and 3) applying a threshold to filter out low-scoring judgments. Compared to methods using a single LLM as both judge and meta-judge, our pipeline introduces multi-agent collaboration and a more comprehensive rubric. Experimental results on the JudgeBench dataset show about 15.55\% improvement compared to raw judgments and about 8.37\% improvement over the single-agent baseline. Our work demonstrates the potential of LLMs as meta-judges and lays the foundation for future research on constructing preference datasets for LLM-as-a-judge reinforcement learning. 

---
# Auditing the Ethical Logic of Generative AI Models 

**Authors**: W. Russell Neuman, Chad Coleman, Ali Dasdan, Safinah Ali, Manan Shah  

**Link**: [PDF](https://arxiv.org/pdf/2504.17544)  

**Abstract**: As generative AI models become increasingly integrated into high-stakes domains, the need for robust methods to evaluate their ethical reasoning becomes increasingly important. This paper introduces a five-dimensional audit model -- assessing Analytic Quality, Breadth of Ethical Considerations, Depth of Explanation, Consistency, and Decisiveness -- to evaluate the ethical logic of leading large language models (LLMs). Drawing on traditions from applied ethics and higher-order thinking, we present a multi-battery prompt approach, including novel ethical dilemmas, to probe the models' reasoning across diverse contexts. We benchmark seven major LLMs finding that while models generally converge on ethical decisions, they vary in explanatory rigor and moral prioritization. Chain-of-Thought prompting and reasoning-optimized models significantly enhance performance on our audit metrics. This study introduces a scalable methodology for ethical benchmarking of AI systems and highlights the potential for AI to complement human moral reasoning in complex decision-making contexts. 

---
# Neural Theorem Proving: Generating and Structuring Proofs for Formal Verification 

**Authors**: Balaji Rao, William Eiers, Carlo Lipizzi  

**Link**: [PDF](https://arxiv.org/pdf/2504.17017)  

**Abstract**: Formally verifying properties of software code has been a highly desirable task, especially with the emergence of LLM-generated code. In the same vein, they provide an interesting avenue for the exploration of formal verification and mechanistic interpretability. Since the introduction of code-specific models, despite their successes in generating code in Lean4 and Isabelle, the task of generalized theorem proving still remains far from being fully solved and will be a benchmark for reasoning capability in LLMs. In this work, we introduce a framework that generates whole proofs in a formal language to be used within systems that utilize the power of built-in tactics and off-the-shelf automated theorem provers. Our framework includes 3 components: generating natural language statements of the code to be verified, an LLM that generates formal proofs for the given statement, and a module employing heuristics for building the final proof. To train the LLM, we employ a 2-stage fine-tuning process, where we first use SFT-based training to enable the model to generate syntactically correct Isabelle code and then RL-based training that encourages the model to generate proofs verified by a theorem prover. We validate our framework using the miniF2F-test benchmark and the Isabelle proof assistant and design a use case to verify the correctness of the AWS S3 bucket access policy code. We also curate a dataset based on the FVEL\textsubscript{\textnormal{ER}} dataset for future training tasks. 

---
# A Desideratum for Conversational Agents: Capabilities, Challenges, and Future Directions 

**Authors**: Emre Can Acikgoz, Cheng Qian, Hongru Wang, Vardhan Dongre, Xiusi Chen, Heng Ji, Dilek Hakkani-Tür, Gokhan Tur  

**Link**: [PDF](https://arxiv.org/pdf/2504.16939)  

**Abstract**: Recent advances in Large Language Models (LLMs) have propelled conversational AI from traditional dialogue systems into sophisticated agents capable of autonomous actions, contextual awareness, and multi-turn interactions with users. Yet, fundamental questions about their capabilities, limitations, and paths forward remain open. This survey paper presents a desideratum for next-generation Conversational Agents - what has been achieved, what challenges persist, and what must be done for more scalable systems that approach human-level intelligence. To that end, we systematically analyze LLM-driven Conversational Agents by organizing their capabilities into three primary dimensions: (i) Reasoning - logical, systematic thinking inspired by human intelligence for decision making, (ii) Monitor - encompassing self-awareness and user interaction monitoring, and (iii) Control - focusing on tool utilization and policy following. Building upon this, we introduce a novel taxonomy by classifying recent work on Conversational Agents around our proposed desideratum. We identify critical research gaps and outline key directions, including realistic evaluations, long-term multi-turn reasoning skills, self-evolution capabilities, collaborative and multi-agent task completion, personalization, and proactivity. This work aims to provide a structured foundation, highlight existing limitations, and offer insights into potential future research directions for Conversational Agents, ultimately advancing progress toward Artificial General Intelligence (AGI). We maintain a curated repository of papers at: this https URL. 

---
# Comprehend, Divide, and Conquer: Feature Subspace Exploration via Multi-Agent Hierarchical Reinforcement Learning 

**Authors**: Weiliang Zhang, Xiaohan Huang, Yi Du, Ziyue Qiao, Qingqing Long, Zhen Meng, Yuanchun Zhou, Meng Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2504.17356)  

**Abstract**: Feature selection aims to preprocess the target dataset, find an optimal and most streamlined feature subset, and enhance the downstream machine learning task. Among filter, wrapper, and embedded-based approaches, the reinforcement learning (RL)-based subspace exploration strategy provides a novel objective optimization-directed perspective and promising performance. Nevertheless, even with improved performance, current reinforcement learning approaches face challenges similar to conventional methods when dealing with complex datasets. These challenges stem from the inefficient paradigm of using one agent per feature and the inherent complexities present in the datasets. This observation motivates us to investigate and address the above issue and propose a novel approach, namely HRLFS. Our methodology initially employs a Large Language Model (LLM)-based hybrid state extractor to capture each feature's mathematical and semantic characteristics. Based on this information, features are clustered, facilitating the construction of hierarchical agents for each cluster and sub-cluster. Extensive experiments demonstrate the efficiency, scalability, and robustness of our approach. Compared to contemporary or the one-feature-one-agent RL-based approaches, HRLFS improves the downstream ML performance with iterative feature subspace exploration while accelerating total run time by reducing the number of agents involved. 

---
# AI-Enhanced Business Process Automation: A Case Study in the Insurance Domain Using Object-Centric Process Mining 

**Authors**: Shahrzad Khayatbashi, Viktor Sjölind, Anders Granåker, Amin Jalali  

**Link**: [PDF](https://arxiv.org/pdf/2504.17295)  

**Abstract**: Recent advancements in Artificial Intelligence (AI), particularly Large Language Models (LLMs), have enhanced organizations' ability to reengineer business processes by automating knowledge-intensive tasks. This automation drives digital transformation, often through gradual transitions that improve process efficiency and effectiveness. To fully assess the impact of such automation, a data-driven analysis approach is needed - one that examines how traditional and AI-enhanced process variants coexist during this transition. Object-Centric Process Mining (OCPM) has emerged as a valuable method that enables such analysis, yet real-world case studies are still needed to demonstrate its applicability. This paper presents a case study from the insurance sector, where an LLM was deployed in production to automate the identification of claim parts, a task previously performed manually and identified as a bottleneck for scalability. To evaluate this transformation, we apply OCPM to assess the impact of AI-driven automation on process scalability. Our findings indicate that while LLMs significantly enhance operational capacity, they also introduce new process dynamics that require further refinement. This study also demonstrates the practical application of OCPM in a real-world setting, highlighting its advantages and limitations. 

---
# Multilingual Performance Biases of Large Language Models in Education 

**Authors**: Vansh Gupta, Sankalan Pal Chowdhury, Vilém Zouhar, Donya Rooein, Mrinmaya Sachan  

**Link**: [PDF](https://arxiv.org/pdf/2504.17720)  

**Abstract**: Large language models (LLMs) are increasingly being adopted in educational settings. These applications expand beyond English, though current LLMs remain primarily English-centric. In this work, we ascertain if their use in education settings in non-English languages is warranted. We evaluated the performance of popular LLMs on four educational tasks: identifying student misconceptions, providing targeted feedback, interactive tutoring, and grading translations in six languages (Hindi, Arabic, Farsi, Telugu, Ukrainian, Czech) in addition to English. We find that the performance on these tasks somewhat corresponds to the amount of language represented in training data, with lower-resource languages having poorer task performance. Although the models perform reasonably well in most languages, the frequent performance drop from English is significant. Thus, we recommend that practitioners first verify that the LLM works well in the target language for their educational task before deployment. 

---
# Towards Machine-Generated Code for the Resolution of User Intentions 

**Authors**: Justus Flerlage, Ilja Behnke, Odej Kao  

**Link**: [PDF](https://arxiv.org/pdf/2504.17531)  

**Abstract**: The growing capabilities of Artificial Intelligence (AI), particularly Large Language Models (LLMs), prompt a reassessment of the interaction mechanisms between users and their devices. Currently, users are required to use a set of high-level applications to achieve their desired results. However, the advent of AI may signal a shift in this regard, as its capabilities have generated novel prospects for user-provided intent resolution through the deployment of model-generated code, which is tantamount to the generation of workflows comprising a multitude of interdependent steps. This development represents a significant progression in the realm of hybrid workflows, where human and artificial intelligence collaborate to address user intentions, with the former responsible for defining these intentions and the latter for implementing the solutions to address them. In this paper, we investigate the feasibility of generating and executing workflows through code generation that results from prompting an LLM with a concrete user intention, such as \emph{Please send my car title to my insurance company}, and a simplified application programming interface for a GUI-less operating system. We provide in-depth analysis and comparison of various user intentions, the resulting code, and its execution. The findings demonstrate a general feasibility of our approach and that the employed LLM, GPT-4o-mini, exhibits remarkable proficiency in the generation of code-oriented workflows in accordance with provided user intentions. 

---
# HalluLens: LLM Hallucination Benchmark 

**Authors**: Yejin Bang, Ziwei Ji, Alan Schelten, Anthony Hartshorn, Tara Fowler, Cheng Zhang, Nicola Cancedda, Pascale Fung  

**Link**: [PDF](https://arxiv.org/pdf/2504.17550)  

**Abstract**: Large language models (LLMs) often generate responses that deviate from user input or training data, a phenomenon known as "hallucination." These hallucinations undermine user trust and hinder the adoption of generative AI systems. Addressing hallucinations is essential for the advancement of LLMs. This paper introduces a comprehensive hallucination benchmark, incorporating both new extrinsic and existing intrinsic evaluation tasks, built upon clear taxonomy of hallucination. A major challenge in benchmarking hallucinations is the lack of a unified framework due to inconsistent definitions and categorizations. We disentangle LLM hallucination from "factuality," proposing a clear taxonomy that distinguishes between extrinsic and intrinsic hallucinations, to promote consistency and facilitate research. Extrinsic hallucinations, where the generated content is not consistent with the training data, are increasingly important as LLMs evolve. Our benchmark includes dynamic test set generation to mitigate data leakage and ensure robustness against such leakage. We also analyze existing benchmarks, highlighting their limitations and saturation. The work aims to: (1) establish a clear taxonomy of hallucinations, (2) introduce new extrinsic hallucination tasks, with data that can be dynamically regenerated to prevent saturation by leakage, (3) provide a comprehensive analysis of existing benchmarks, distinguishing them from factuality evaluations. 

---
# INSIGHT: Bridging the Student-Teacher Gap in Times of Large Language Models 

**Authors**: Jarne Thys, Sebe Vanbrabant, Davy Vanacken, Gustavo Rovelo Ruiz  

**Link**: [PDF](https://arxiv.org/pdf/2504.17677)  

**Abstract**: The rise of AI, especially Large Language Models, presents challenges and opportunities to integrate such technology into the classroom. AI has the potential to revolutionize education by helping teaching staff with various tasks, such as personalizing their teaching methods, but it also raises concerns, for example, about the degradation of student-teacher interactions and user privacy. This paper introduces INSIGHT, a proof of concept to combine various AI tools to assist teaching staff and students in the process of solving exercises. INSIGHT has a modular design that allows it to be integrated into various higher education courses. We analyze students' questions to an LLM by extracting keywords, which we use to dynamically build an FAQ from students' questions and provide new insights for the teaching staff to use for more personalized face-to-face support. Future work could build upon INSIGHT by using the collected data to provide adaptive learning and adjust content based on student progress and learning styles to offer a more interactive and inclusive learning experience. 

---
# Towards Leveraging Large Language Model Summaries for Topic Modeling in Source Code 

**Authors**: Michele Carissimi, Martina Saletta, Claudio Ferretti  

**Link**: [PDF](https://arxiv.org/pdf/2504.17426)  

**Abstract**: Understanding source code is a topic of great interest in the software engineering community, since it can help programmers in various tasks such as software maintenance and reuse. Recent advances in large language models (LLMs) have demonstrated remarkable program comprehension capabilities, while transformer-based topic modeling techniques offer effective ways to extract semantic information from text. This paper proposes and explores a novel approach that combines these strengths to automatically identify meaningful topics in a corpus of Python programs. Our method consists in applying topic modeling on the descriptions obtained by asking an LLM to summarize the code. To assess the internal consistency of the extracted topics, we compare them against topics inferred from function names alone, and those derived from existing docstrings. Experimental results suggest that leveraging LLM-generated summaries provides interpretable and semantically rich representation of code structure. The promising results suggest that our approach can be fruitfully applied in various software engineering tasks such as automatic documentation and tagging, code search, software reorganization and knowledge discovery in large repositories. 

---
# Towards Harnessing the Collaborative Power of Large and Small Models for Domain Tasks 

**Authors**: Yang Liu, Bingjie Yan, Tianyuan Zou, Jianqing Zhang, Zixuan Gu, Jianbing Ding, Xidong Wang, Jingyi Li, Xiaozhou Ye, Ye Ouyang, Qiang Yang, Ya-Qin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.17421)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable capabilities, but they require vast amounts of data and computational resources. In contrast, smaller models (SMs), while less powerful, can be more efficient and tailored to specific domains. In this position paper, we argue that taking a collaborative approach, where large and small models work synergistically, can accelerate the adaptation of LLMs to private domains and unlock new potential in AI. We explore various strategies for model collaboration and identify potential challenges and opportunities. Building upon this, we advocate for industry-driven research that prioritizes multi-objective benchmarks on real-world private datasets and applications. 

---
# Exploring Context-aware and LLM-driven Locomotion for Immersive Virtual Reality 

**Authors**: Süleyman Özdel, Kadir Burak Buldu, Enkelejda Kasneci, Efe Bozkir  

**Link**: [PDF](https://arxiv.org/pdf/2504.17331)  

**Abstract**: Locomotion plays a crucial role in shaping the user experience within virtual reality environments. In particular, hands-free locomotion offers a valuable alternative by supporting accessibility and freeing users from reliance on handheld controllers. To this end, traditional speech-based methods often depend on rigid command sets, limiting the naturalness and flexibility of interaction. In this study, we propose a novel locomotion technique powered by large language models (LLMs), which allows users to navigate virtual environments using natural language with contextual awareness. We evaluate three locomotion methods: controller-based teleportation, voice-based steering, and our language model-driven approach. Our evaluation measures include eye-tracking data analysis, including explainable machine learning through SHAP analysis as well as standardized questionnaires for usability, presence, cybersickness, and cognitive load to examine user attention and engagement. Our findings indicate that the LLM-driven locomotion possesses comparable usability, presence, and cybersickness scores to established methods like teleportation, demonstrating its novel potential as a comfortable, natural language-based, hands-free alternative. In addition, it enhances user attention within the virtual environment, suggesting greater engagement. Complementary to these findings, SHAP analysis revealed that fixation, saccade, and pupil-related features vary across techniques, indicating distinct patterns of visual attention and cognitive processing. Overall, we state that our method can facilitate hands-free locomotion in virtual spaces, especially in supporting accessibility. 

---
# DIMT25@ICDAR2025: HW-TSC's End-to-End Document Image Machine Translation System Leveraging Large Vision-Language Model 

**Authors**: Zhanglin Wu, Tengfei Song, Ning Xie, Weidong Zhang, Pengfei Li, Shuang Wu, Chong Li, Junhao Zhu, Hao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.17315)  

**Abstract**: This paper presents the technical solution proposed by Huawei Translation Service Center (HW-TSC) for the "End-to-End Document Image Machine Translation for Complex Layouts" competition at the 19th International Conference on Document Analysis and Recognition (DIMT25@ICDAR2025). Leveraging state-of-the-art open-source large vision-language model (LVLM), we introduce a training framework that combines multi-task learning with perceptual chain-of-thought to develop a comprehensive end-to-end document translation system. During the inference phase, we apply minimum Bayesian decoding and post-processing strategies to further enhance the system's translation capabilities. Our solution uniquely addresses both OCR-based and OCR-free document image translation tasks within a unified framework. This paper systematically details the training methods, inference strategies, LVLM base models, training data, experimental setups, and results, demonstrating an effective approach to document image machine translation. 

---
# FLUKE: A Linguistically-Driven and Task-Agnostic Framework for Robustness Evaluation 

**Authors**: Yulia Otmakhova, Hung Thinh Truong, Rahmad Mahendra, Zenan Zhai, Rongxin Zhu, Daniel Beck, Jey Han Lau  

**Link**: [PDF](https://arxiv.org/pdf/2504.17311)  

**Abstract**: We present FLUKE (Framework for LingUistically-driven and tasK-agnostic robustness Evaluation), a task-agnostic framework for assessing model robustness through systematic minimal variations of test data. FLUKE introduces controlled variations across linguistic levels - from orthography to dialect and style varieties - and leverages large language models (LLMs) with human validation to generate modifications. We demonstrate FLUKE's utility by evaluating both fine-tuned models and LLMs across four diverse NLP tasks, and reveal that (1) the impact of linguistic variations is highly task-dependent, with some tests being critical for certain tasks but irrelevant for others; (2) while LLMs have better overall robustness compared to fine-tuned models, they still exhibit significant brittleness to certain linguistic variations; (3) all models show substantial vulnerability to negation modifications across most tasks. These findings highlight the importance of systematic robustness testing for understanding model behaviors. 

---
# Combining GCN Structural Learning with LLM Chemical Knowledge for or Enhanced Virtual Screening 

**Authors**: Radia Berreziga, Mohammed Brahimi, Khairedine Kraim, Hamid Azzoune  

**Link**: [PDF](https://arxiv.org/pdf/2504.17497)  

**Abstract**: Virtual screening plays a critical role in modern drug discovery by enabling the identification of promising candidate molecules for experimental validation. Traditional machine learning methods such as support vector machines (SVM) and XGBoost rely on predefined molecular representations, often leading to information loss and potential bias. In contrast, deep learning approaches-particularly Graph Convolutional Networks (GCNs)-offer a more expressive and unbiased alternative by operating directly on molecular graphs. Meanwhile, Large Language Models (LLMs) have recently demonstrated state-of-the-art performance in drug design, thanks to their capacity to capture complex chemical patterns from large-scale data via attention mechanisms.
In this paper, we propose a hybrid architecture that integrates GCNs with LLM-derived embeddings to combine localized structural learning with global chemical knowledge. The LLM embeddings can be precomputed and stored in a molecular feature library, removing the need to rerun the LLM during training or inference and thus maintaining computational efficiency. We found that concatenating the LLM embeddings after each GCN layer-rather than only at the final layer-significantly improves performance, enabling deeper integration of global context throughout the network. The resulting model achieves superior results, with an F1-score of (88.8%), outperforming standalone GCN (87.9%), XGBoost (85.5%), and SVM (85.4%) baselines. 

---
# Automatically Generating Rules of Malicious Software Packages via Large Language Model 

**Authors**: XiangRui Zhang, HaoYu Chen, Yongzhong He, Wenjia Niu, Qiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.17198)  

**Abstract**: Today's security tools predominantly rely on predefined rules crafted by experts, making them poorly adapted to the emergence of software supply chain attacks. To tackle this limitation, we propose a novel tool, RuleLLM, which leverages large language models (LLMs) to automate rule generation for OSS ecosystems. RuleLLM extracts metadata and code snippets from malware as its input, producing YARA and Semgrep rules that can be directly deployed in software development. Specifically, the rule generation task involves three subtasks: crafting rules, refining rules, and aligning rules. To validate RuleLLM's effectiveness, we implemented a prototype system and conducted experiments on the dataset of 1,633 malicious packages. The results are promising that RuleLLM generated 763 rules (452 YARA and 311 Semgrep) with a precision of 85.2\% and a recall of 91.8\%, outperforming state-of-the-art (SOTA) tools and scored-based approaches. We further analyzed generated rules and proposed a rule taxonomy: 11 categories and 38 subcategories. 

---
# Robo-Troj: Attacking LLM-based Task Planners 

**Authors**: Mohaiminul Al Nahian, Zainab Altaweel, David Reitano, Sabbir Ahmed, Saumitra Lohokare, Shiqi Zhang, Adnan Siraj Rakin  

**Link**: [PDF](https://arxiv.org/pdf/2504.17070)  

**Abstract**: Robots need task planning methods to achieve goals that require more than individual actions. Recently, large language models (LLMs) have demonstrated impressive performance in task planning. LLMs can generate a step-by-step solution using a description of actions and the goal. Despite the successes in LLM-based task planning, there is limited research studying the security aspects of those systems. In this paper, we develop Robo-Troj, the first multi-trigger backdoor attack for LLM-based task planners, which is the main contribution of this work. As a multi-trigger attack, Robo-Troj is trained to accommodate the diversity of robot application domains. For instance, one can use unique trigger words, e.g., "herical", to activate a specific malicious behavior, e.g., cutting hand on a kitchen robot. In addition, we develop an optimization method for selecting the trigger words that are most effective. Through demonstrating the vulnerability of LLM-based planners, we aim to promote the development of secured robot systems. 

---
# (Im)possibility of Automated Hallucination Detection in Large Language Models 

**Authors**: Amin Karbasi, Omar Montasser, John Sous, Grigoris Velegkas  

**Link**: [PDF](https://arxiv.org/pdf/2504.17004)  

**Abstract**: Is automated hallucination detection possible? In this work, we introduce a theoretical framework to analyze the feasibility of automatically detecting hallucinations produced by large language models (LLMs). Inspired by the classical Gold-Angluin framework for language identification and its recent adaptation to language generation by Kleinberg and Mullainathan, we investigate whether an algorithm, trained on examples drawn from an unknown target language $K$ (selected from a countable collection) and given access to an LLM, can reliably determine whether the LLM's outputs are correct or constitute hallucinations.
First, we establish an equivalence between hallucination detection and the classical task of language identification. We prove that any hallucination detection method can be converted into a language identification method, and conversely, algorithms solving language identification can be adapted for hallucination detection. Given the inherent difficulty of language identification, this implies that hallucination detection is fundamentally impossible for most language collections if the detector is trained using only correct examples from the target language.
Second, we show that the use of expert-labeled feedback, i.e., training the detector with both positive examples (correct statements) and negative examples (explicitly labeled incorrect statements), dramatically changes this conclusion. Under this enriched training regime, automated hallucination detection becomes possible for all countable language collections.
These results highlight the essential role of expert-labeled examples in training hallucination detectors and provide theoretical support for feedback-based methods, such as reinforcement learning with human feedback (RLHF), which have proven critical for reliable LLM deployment. 

---
# MIRAGE: A Metric-Intensive Benchmark for Retrieval-Augmented Generation Evaluation 

**Authors**: Chanhee Park, Hyeonseok Moon, Chanjun Park, Heuiseok Lim  

**Link**: [PDF](https://arxiv.org/pdf/2504.17137)  

**Abstract**: Retrieval-Augmented Generation (RAG) has gained prominence as an effective method for enhancing the generative capabilities of Large Language Models (LLMs) through the incorporation of external knowledge. However, the evaluation of RAG systems remains a challenge, due to the intricate interplay between retrieval and generation components. This limitation has resulted in a scarcity of benchmarks that facilitate a detailed, component-specific assessment. In this work, we present MIRAGE, a Question Answering dataset specifically designed for RAG evaluation. MIRAGE consists of 7,560 curated instances mapped to a retrieval pool of 37,800 entries, enabling an efficient and precise evaluation of both retrieval and generation tasks. We also introduce novel evaluation metrics aimed at measuring RAG adaptability, encompassing dimensions such as noise vulnerability, context acceptability, context insensitivity, and context misinterpretation. Through comprehensive experiments across various retriever-LLM configurations, we provide new insights into the optimal alignment of model pairs and the nuanced dynamics within RAG systems. The dataset and evaluation code are publicly available, allowing for seamless integration and customization in diverse research settings\footnote{The MIRAGE code and data are available at this https URL. 

---
# Evaluating Grounded Reasoning by Code-Assisted Large Language Models for Mathematics 

**Authors**: Zena Al-Khalili, Nick Howell, Dietrich Klakow  

**Link**: [PDF](https://arxiv.org/pdf/2504.17665)  

**Abstract**: Assisting LLMs with code generation improved their performance on mathematical reasoning tasks. However, the evaluation of code-assisted LLMs is generally restricted to execution correctness, lacking a rigorous evaluation of their generated programs. In this work, we bridge this gap by conducting an in-depth analysis of code-assisted LLMs' generated programs in response to math reasoning tasks. Our evaluation focuses on the extent to which LLMs ground their programs to math rules, and how that affects their end performance. For this purpose, we assess the generations of five different LLMs, on two different math datasets, both manually and automatically. Our results reveal that the distribution of grounding depends on LLMs' capabilities and the difficulty of math problems. Furthermore, mathematical grounding is more effective for closed-source models, while open-source models fail to employ math rules in their solutions correctly. On MATH500, the percentage of grounded programs decreased to half, while the ungrounded generations doubled in comparison to ASDiv grade-school problems. Our work highlights the need for in-depth evaluation beyond execution accuracy metrics, toward a better understanding of code-assisted LLMs' capabilities and limits in the math domain. 

---
# DeepDistill: Enhancing LLM Reasoning Capabilities via Large-Scale Difficulty-Graded Data Training 

**Authors**: Xiaoyu Tian, Sitong Zhao, Haotian Wang, Shuaiting Chen, Yiping Peng, Yunjie Ji, Han Zhao, Xiangang Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.17565)  

**Abstract**: Although large language models (LLMs) have recently achieved remarkable performance on various complex reasoning benchmarks, the academic community still lacks an in-depth understanding of base model training processes and data quality. To address this, we construct a large-scale, difficulty-graded reasoning dataset containing approximately 3.34 million unique queries of varying difficulty levels and about 40 million distilled responses generated by multiple models over several passes. Leveraging pass rate and Coefficient of Variation (CV), we precisely select the most valuable training data to enhance reasoning capability. Notably, we observe a training pattern shift, indicating that reasoning-focused training based on base models requires higher learning rates for effective training. Using this carefully selected data, we significantly improve the reasoning capabilities of the base model, achieving a pass rate of 79.2\% on the AIME2024 mathematical reasoning benchmark. This result surpasses most current distilled models and closely approaches state-of-the-art performance. We provide detailed descriptions of our data processing, difficulty assessment, and training methodology, and have publicly released all datasets and methods to promote rapid progress in open-source long-reasoning LLMs. The dataset is available at: this https URL 

---
# Creating Targeted, Interpretable Topic Models with LLM-Generated Text Augmentation 

**Authors**: Anna Lieb, Maneesh Arora, Eni Mustafaraj  

**Link**: [PDF](https://arxiv.org/pdf/2504.17445)  

**Abstract**: Unsupervised machine learning techniques, such as topic modeling and clustering, are often used to identify latent patterns in unstructured text data in fields such as political science and sociology. These methods overcome common concerns about reproducibility and costliness involved in the labor-intensive process of human qualitative analysis. However, two major limitations of topic models are their interpretability and their practicality for answering targeted, domain-specific social science research questions. In this work, we investigate opportunities for using LLM-generated text augmentation to improve the usefulness of topic modeling output. We use a political science case study to evaluate our results in a domain-specific application, and find that topic modeling using GPT-4 augmentations creates highly interpretable categories that can be used to investigate domain-specific research questions with minimal human guidance. 

---
# LiveLongBench: Tackling Long-Context Understanding for Spoken Texts from Live Streams 

**Authors**: Yongxuan Wu, Runyu Chen, Peiyu Liu, Hongjin Qian  

**Link**: [PDF](https://arxiv.org/pdf/2504.17366)  

**Abstract**: Long-context understanding poses significant challenges in natural language processing, particularly for real-world dialogues characterized by speech-based elements, high redundancy, and uneven information density. Although large language models (LLMs) achieve impressive results on existing benchmarks, these datasets fail to reflect the complexities of such texts, limiting their applicability to practical scenarios. To bridge this gap, we construct the first spoken long-text dataset, derived from live streams, designed to reflect the redundancy-rich and conversational nature of real-world scenarios. We construct tasks in three categories: retrieval-dependent, reasoning-dependent, and hybrid. We then evaluate both popular LLMs and specialized methods to assess their ability to understand long-contexts in these tasks. Our results show that current methods exhibit strong task-specific preferences and perform poorly on highly redundant inputs, with no single method consistently outperforming others. We propose a new baseline that better handles redundancy in spoken text and achieves strong performance across tasks. Our findings highlight key limitations of current methods and suggest future directions for improving long-context understanding. Finally, our benchmark fills a gap in evaluating long-context spoken language understanding and provides a practical foundation for developing real-world e-commerce systems. The code and benchmark are available at this https URL. 

---
# Conversational Assistants to support Heart Failure Patients: comparing a Neurosymbolic Architecture with ChatGPT 

**Authors**: Anuja Tayal, Devika Salunke, Barbara Di Eugenio, Paula Allen-Meares, Eulalia Puig Abril, Olga Garcia, Carolyn Dickens, Andrew Boyd  

**Link**: [PDF](https://arxiv.org/pdf/2504.17753)  

**Abstract**: Conversational assistants are becoming more and more popular, including in healthcare, partly because of the availability and capabilities of Large Language Models. There is a need for controlled, probing evaluations with real stakeholders which can highlight advantages and disadvantages of more traditional architectures and those based on generative AI. We present a within-group user study to compare two versions of a conversational assistant that allows heart failure patients to ask about salt content in food. One version of the system was developed in-house with a neurosymbolic architecture, and one is based on ChatGPT. The evaluation shows that the in-house system is more accurate, completes more tasks and is less verbose than the one based on ChatGPT; on the other hand, the one based on ChatGPT makes fewer speech errors and requires fewer clarifications to complete the task. Patients show no preference for one over the other. 

---
# A RAG-Based Multi-Agent LLM System for Natural Hazard Resilience and Adaptation 

**Authors**: Yangxinyu Xie, Bowen Jiang, Tanwi Mallick, Joshua David Bergerson, John K. Hutchison, Duane R. Verner, Jordan Branham, M. Ross Alexander, Robert B. Ross, Yan Feng, Leslie-Anne Levy, Weijie Su, Camillo J. Taylor  

**Link**: [PDF](https://arxiv.org/pdf/2504.17200)  

**Abstract**: Large language models (LLMs) are a transformational capability at the frontier of artificial intelligence and machine learning that can support decision-makers in addressing pressing societal challenges such as extreme natural hazard events. As generalized models, LLMs often struggle to provide context-specific information, particularly in areas requiring specialized knowledge. In this work we propose a retrieval-augmented generation (RAG)-based multi-agent LLM system to support analysis and decision-making in the context of natural hazards and extreme weather events. As a proof of concept, we present WildfireGPT, a specialized system focused on wildfire hazards. The architecture employs a user-centered, multi-agent design to deliver tailored risk insights across diverse stakeholder groups. By integrating natural hazard and extreme weather projection data, observational datasets, and scientific literature through an RAG framework, the system ensures both the accuracy and contextual relevance of the information it provides. Evaluation across ten expert-led case studies demonstrates that WildfireGPT significantly outperforms existing LLM-based solutions for decision support. 

---
# PatientDx: Merging Large Language Models for Protecting Data-Privacy in Healthcare 

**Authors**: Jose G. Moreno, Jesus Lovon, M'Rick Robin-Charlet, Christine Damase-Michel, Lynda Tamine  

**Link**: [PDF](https://arxiv.org/pdf/2504.17360)  

**Abstract**: Fine-tuning of Large Language Models (LLMs) has become the default practice for improving model performance on a given task. However, performance improvement comes at the cost of training on vast amounts of annotated data which could be sensitive leading to significant data privacy concerns. In particular, the healthcare domain is one of the most sensitive domains exposed to data privacy issues. In this paper, we present PatientDx, a framework of model merging that allows the design of effective LLMs for health-predictive tasks without requiring fine-tuning nor adaptation on patient data. Our proposal is based on recently proposed techniques known as merging of LLMs and aims to optimize a building block merging strategy. PatientDx uses a pivotal model adapted to numerical reasoning and tunes hyperparameters on examples based on a performance metric but without training of the LLM on these data. Experiments using the mortality tasks of the MIMIC-IV dataset show improvements up to 7% in terms of AUROC when compared to initial models. Additionally, we confirm that when compared to fine-tuned models, our proposal is less prone to data leak problems without hurting performance. Finally, we qualitatively show the capabilities of our proposal through a case study. Our best model is publicly available at this https URL Jgmorenof/mistral\_merged\_0\_4. 

---
# Bridging Cognition and Emotion: Empathy-Driven Multimodal Misinformation Detection 

**Authors**: Zihan Wang, Lu Yuan, Zhengxuan Zhang, Qing Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2504.17332)  

**Abstract**: In the digital era, social media has become a major conduit for information dissemination, yet it also facilitates the rapid spread of misinformation. Traditional misinformation detection methods primarily focus on surface-level features, overlooking the crucial roles of human empathy in the propagation process. To address this gap, we propose the Dual-Aspect Empathy Framework (DAE), which integrates cognitive and emotional empathy to analyze misinformation from both the creator and reader perspectives. By examining creators' cognitive strategies and emotional appeals, as well as simulating readers' cognitive judgments and emotional responses using Large Language Models (LLMs), DAE offers a more comprehensive and human-centric approach to misinformation detection. Moreover, we further introduce an empathy-aware filtering mechanism to enhance response authenticity and diversity. Experimental results on benchmark datasets demonstrate that DAE outperforms existing methods, providing a novel paradigm for multimodal misinformation detection. 

---
# Paper2Code: Automating Code Generation from Scientific Papers in Machine Learning 

**Authors**: Minju Seo, Jinheon Baek, Seongyun Lee, Sung Ju Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2504.17192)  

**Abstract**: Despite the rapid growth of machine learning research, corresponding code implementations are often unavailable, making it slow and labor-intensive for researchers to reproduce results and build upon prior work. In the meantime, recent Large Language Models (LLMs) excel at understanding scientific documents and generating high-quality code. Inspired by this, we introduce PaperCoder, a multi-agent LLM framework that transforms machine learning papers into functional code repositories. PaperCoder operates in three stages: planning, where it constructs a high-level roadmap, designs the system architecture with diagrams, identifies file dependencies, and generates configuration files; analysis, which focuses on interpreting implementation-specific details; and generation, where modular, dependency-aware code is produced. Moreover, each phase is instantiated through a set of specialized agents designed to collaborate effectively across the pipeline. We then evaluate PaperCoder on generating code implementations from machine learning papers based on both model-based and human evaluations, specifically from the original paper authors, with author-released repositories as ground truth if available. Our results demonstrate the effectiveness of PaperCoder in creating high-quality, faithful implementations. Furthermore, it consistently shows strengths in the recently released PaperBench benchmark, surpassing strong baselines by substantial margins. 

---
# Optimizing LLMs for Italian: Reducing Token Fertility and Enhancing Efficiency Through Vocabulary Adaptation 

**Authors**: Luca Moroni, Giovanni Puccetti, Pere-Lluis Huguet Cabot, Andrei Stefan Bejgu, Edoardo Barba, Alessio Miaschi, Felice Dell'Orletta, Andrea Esuli, Roberto Navigli  

**Link**: [PDF](https://arxiv.org/pdf/2504.17025)  

**Abstract**: The number of pretrained Large Language Models (LLMs) is increasing steadily, though the majority are designed predominantly for the English language. While state-of-the-art LLMs can handle other languages, due to language contamination or some degree of multilingual pretraining data, they are not optimized for non-English languages, leading to inefficient encoding (high token "fertility") and slower inference speed. In this work, we thoroughly compare a variety of vocabulary adaptation techniques for optimizing English LLMs for the Italian language, and put forward Semantic Alignment Vocabulary Adaptation (SAVA), a novel method that leverages neural mapping for vocabulary substitution. SAVA achieves competitive performance across multiple downstream tasks, enhancing grounded alignment strategies. We adapt two LLMs: Mistral-7b-v0.1, reducing token fertility by 25\%, and Llama-3.1-8B, optimizing the vocabulary and reducing the number of parameters by 1 billion. We show that, following the adaptation of the vocabulary, these models can recover their performance with a relatively limited stage of continual training on the target language. Finally, we test the capabilities of the adapted models on various multi-choice and generative tasks. 

---
# Crisp: Cognitive Restructuring of Negative Thoughts through Multi-turn Supportive Dialogues 

**Authors**: Jinfeng Zhou, Yuxuan Chen, Jianing Yin, Yongkang Huang, Yihan Shi, Xikun Zhang, Libiao Peng, Rongsheng Zhang, Tangjie Lv, Zhipeng Hu, Hongning Wang, Minlie Huang  

**Link**: [PDF](https://arxiv.org/pdf/2504.17238)  

**Abstract**: Cognitive Restructuring (CR) is a psychotherapeutic process aimed at identifying and restructuring an individual's negative thoughts, arising from mental health challenges, into more helpful and positive ones via multi-turn dialogues. Clinician shortage and stigma urge the development of human-LLM interactive psychotherapy for CR. Yet, existing efforts implement CR via simple text rewriting, fixed-pattern dialogues, or a one-shot CR workflow, failing to align with the psychotherapeutic process for effective CR. To address this gap, we propose CRDial, a novel framework for CR, which creates multi-turn dialogues with specifically designed identification and restructuring stages of negative thoughts, integrates sentence-level supportive conversation strategies, and adopts a multi-channel loop mechanism to enable iterative CR. With CRDial, we distill Crisp, a large-scale and high-quality bilingual dialogue dataset, from LLM. We then train Crispers, Crisp-based conversational LLMs for CR, at 7B and 14B scales. Extensive human studies show the superiority of Crispers in pointwise, pairwise, and intervention evaluations. 

---
# Steering the CensorShip: Uncovering Representation Vectors for LLM "Thought" Control 

**Authors**: Hannah Cyberey, David Evans  

**Link**: [PDF](https://arxiv.org/pdf/2504.17130)  

**Abstract**: Large language models (LLMs) have transformed the way we access information. These models are often tuned to refuse to comply with requests that are considered harmful and to produce responses that better align with the preferences of those who control the models. To understand how this "censorship" works. We use representation engineering techniques to study open-weights safety-tuned models. We present a method for finding a refusal--compliance vector that detects and controls the level of censorship in model outputs. We also analyze recent reasoning LLMs, distilled from DeepSeek-R1, and uncover an additional dimension of censorship through "thought suppression". We show a similar approach can be used to find a vector that suppresses the model's reasoning process, allowing us to remove censorship by applying the negative multiples of this vector 

---
# TimeSoccer: An End-to-End Multimodal Large Language Model for Soccer Commentary Generation 

**Authors**: Ling You, Wenxuan Huang, Xinni Xie, Xiangyi Wei, Bangyan Li, Shaohui Lin, Yang Li, Changbo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.17365)  

**Abstract**: Soccer is a globally popular sporting event, typically characterized by long matches and distinctive highlight moments. Recent advances in Multimodal Large Language Models (MLLMs) offer promising capabilities in temporal grounding and video understanding, soccer commentary generation often requires precise temporal localization and semantically rich descriptions over long-form video. However, existing soccer MLLMs often rely on the temporal a priori for caption generation, so they cannot process the soccer video end-to-end. While some traditional approaches follow a two-step paradigm that is complex and fails to capture the global context to achieve suboptimal performance. To solve the above issues, we present TimeSoccer, the first end-to-end soccer MLLM for Single-anchor Dense Video Captioning (SDVC) in full-match soccer videos. TimeSoccer jointly predicts timestamps and generates captions in a single pass, enabling global context modeling across 45-minute matches. To support long video understanding of soccer matches, we introduce MoFA-Select, a training-free, motion-aware frame compression module that adaptively selects representative frames via a coarse-to-fine strategy, and incorporates complementary training paradigms to strengthen the model's ability to handle long temporal sequences. Extensive experiments demonstrate that our TimeSoccer achieves State-of-The-Art (SoTA) performance on the SDVC task in an end-to-end form, generating high-quality commentary with accurate temporal alignment and strong semantic relevance. 

---
# Do Words Reflect Beliefs? Evaluating Belief Depth in Large Language Models 

**Authors**: Shariar Kabir, Kevin Esterling, Yue Dong  

**Link**: [PDF](https://arxiv.org/pdf/2504.17052)  

**Abstract**: Large Language Models (LLMs) are increasingly shaping political discourse, yet their responses often display inconsistency when subjected to scrutiny. While prior research has primarily categorized LLM outputs as left- or right-leaning to assess their political stances, a critical question remains: Do these responses reflect genuine internal beliefs or merely surface-level alignment with training data? To address this, we propose a novel framework for evaluating belief depth by analyzing (1) argumentative consistency and (2) uncertainty quantification. We evaluate 12 LLMs on 19 economic policies from the Political Compass Test, challenging their belief stability with both supportive and opposing arguments. Our analysis reveals that LLMs exhibit topic-specific belief stability rather than a uniform ideological stance. Notably, up to 95% of left-leaning models' responses and 89% of right-leaning models' responses remain consistent under the challenge, enabling semantic entropy to achieve high accuracy (AUROC=0.78), effectively distinguishing between surface-level alignment from genuine belief. These findings call into question the assumption that LLMs maintain stable, human-like political ideologies, emphasizing the importance of conducting topic-specific reliability assessments for real-world applications. 

---
