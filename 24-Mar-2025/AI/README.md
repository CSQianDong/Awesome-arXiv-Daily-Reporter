# HCAST: Human-Calibrated Autonomy Software Tasks 

**Authors**: David Rein, Joel Becker, Amy Deng, Seraphina Nix, Chris Canal, Daniel O'Connel, Pip Arnott, Ryan Bloom, Thomas Broadley, Katharyn Garcia, Brian Goodrich, Max Hasin, Sami Jawhar, Megan Kinniment, Thomas Kwa, Aron Lajko, Nate Rush, Lucas Jun Koba Sato, Sydney Von Arx, Ben West, Lawrence Chan, Elizabeth Barnes  

**Link**: [PDF](https://arxiv.org/pdf/2503.17354)  

**Abstract**: To understand and predict the societal impacts of highly autonomous AI systems, we need benchmarks with grounding, i.e., metrics that directly connect AI performance to real-world effects we care about. We present HCAST (Human-Calibrated Autonomy Software Tasks), a benchmark of 189 machine learning engineering, cybersecurity, software engineering, and general reasoning tasks. We collect 563 human baselines (totaling over 1500 hours) from people skilled in these domains, working under identical conditions as AI agents, which lets us estimate that HCAST tasks take humans between one minute and 8+ hours. Measuring the time tasks take for humans provides an intuitive metric for evaluating AI capabilities, helping answer the question "can an agent be trusted to complete a task that would take a human X hours?" We evaluate the success rates of AI agents built on frontier foundation models, and we find that current agents succeed 70-80% of the time on tasks that take humans less than one hour, and less than 20% of the time on tasks that take humans more than 4 hours. 

---
# Capturing Individual Human Preferences with Reward Features 

**Authors**: André Barreto, Vincent Dumoulin, Yiran Mao, Nicolas Perez-Nieves, Bobak Shahriari, Yann Dauphin, Doina Precup, Hugo Larochelle  

**Link**: [PDF](https://arxiv.org/pdf/2503.17338)  

**Abstract**: Reinforcement learning from human feedback usually models preferences using a reward model that does not distinguish between people. We argue that this is unlikely to be a good design choice in contexts with high potential for disagreement, like in the training of large language models. We propose a method to specialise a reward model to a person or group of people. Our approach builds on the observation that individual preferences can be captured as a linear combination of a set of general reward features. We show how to learn such features and subsequently use them to quickly adapt the reward model to a specific individual, even if their preferences are not reflected in the training data. We present experiments with large language models comparing the proposed architecture with a non-adaptive reward model and also adaptive counterparts, including models that do in-context personalisation. Depending on how much disagreement there is in the training data, our model either significantly outperforms the baselines or matches their performance with a simpler architecture and more stable training. 

---
# Breaking the Symmetries of Indistinguishable Objects 

**Authors**: Ozgur Akgun, Mun See Chang, Ian P. Gent, Christopher Jefferson  

**Link**: [PDF](https://arxiv.org/pdf/2503.17251)  

**Abstract**: Indistinguishable objects often occur when modelling problems in constraint programming, as well as in other related paradigms. They occur when objects can be viewed as being drawn from a set of unlabelled objects, and the only operation allowed on them is equality testing. For example, the golfers in the social golfer problem are indistinguishable. If we do label the golfers, then any relabelling of the golfers in one solution gives another valid solution. Therefore, we can regard the symmetric group of size $n$ as acting on a set of $n$ indistinguishable objects. In this paper, we show how we can break the symmetries resulting from indistinguishable objects. We show how symmetries on indistinguishable objects can be defined properly in complex types, for example in a matrix indexed by indistinguishable objects. We then show how the resulting symmetries can be broken correctly. In Essence, a high-level modelling language, indistinguishable objects are encapsulated in "unnamed types". We provide an implementation of complete symmetry breaking for unnamed types in Essence. 

---
# A Guide to Bayesian Networks Software Packages for Structure and Parameter Learning -- 2025 Edition 

**Authors**: Joverlyn Gaudillo, Nicole Astrologo, Fabio Stella, Enzo Acerbi, Francesco Canonaco  

**Link**: [PDF](https://arxiv.org/pdf/2503.17025)  

**Abstract**: A representation of the cause-effect mechanism is needed to enable artificial intelligence to represent how the world works. Bayesian Networks (BNs) have proven to be an effective and versatile tool for this task. BNs require constructing a structure of dependencies among variables and learning the parameters that govern these relationships. These tasks, referred to as structural learning and parameter learning, are actively investigated by the research community, with several algorithms proposed and no single method having established itself as standard. A wide range of software, tools, and packages have been developed for BNs analysis and made available to academic researchers and industry practitioners. As a consequence of having no one-size-fits-all solution, moving the first practical steps and getting oriented into this field is proving to be challenging to outsiders and beginners. In this paper, we review the most relevant tools and software for BNs structural and parameter learning to date, providing our subjective recommendations directed to an audience of beginners. In addition, we provide an extensive easy-to-consult overview table summarizing all software packages and their main features. By improving the reader understanding of which available software might best suit their needs, we improve accessibility to the field and make it easier for beginners to take their first step into it. 

---
# Real-Time Diffusion Policies for Games: Enhancing Consistency Policies with Q-Ensembles 

**Authors**: Ruoqi Zhang, Ziwei Luo, Jens Sjölund, Per Mattsson, Linus Gisslén, Alessandro Sestini  

**Link**: [PDF](https://arxiv.org/pdf/2503.16978)  

**Abstract**: Diffusion models have shown impressive performance in capturing complex and multi-modal action distributions for game agents, but their slow inference speed prevents practical deployment in real-time game environments. While consistency models offer a promising approach for one-step generation, they often suffer from training instability and performance degradation when applied to policy learning. In this paper, we present CPQE (Consistency Policy with Q-Ensembles), which combines consistency models with Q-ensembles to address these this http URL leverages uncertainty estimation through Q-ensembles to provide more reliable value function approximations, resulting in better training stability and improved performance compared to classic double Q-network methods. Our extensive experiments across multiple game scenarios demonstrate that CPQE achieves inference speeds of up to 60 Hz -- a significant improvement over state-of-the-art diffusion policies that operate at only 20 Hz -- while maintaining comparable performance to multi-step diffusion approaches. CPQE consistently outperforms state-of-the-art consistency model approaches, showing both higher rewards and enhanced training stability throughout the learning process. These results indicate that CPQE offers a practical solution for deploying diffusion-based policies in games and other real-time applications where both multi-modal behavior modeling and rapid inference are critical requirements. 

---
# Neural-Guided Equation Discovery 

**Authors**: Jannis Brugger, Mattia Cerrato, David Richter, Cedric Derstroff, Daniel Maninger, Mira Mezini, Stefan Kramer  

**Link**: [PDF](https://arxiv.org/pdf/2503.16953)  

**Abstract**: Deep learning approaches are becoming increasingly attractive for equation discovery. We show the advantages and disadvantages of using neural-guided equation discovery by giving an overview of recent papers and the results of experiments using our modular equation discovery system MGMT ($\textbf{M}$ulti-Task $\textbf{G}$rammar-Guided $\textbf{M}$onte-Carlo $\textbf{T}$ree Search for Equation Discovery). The system uses neural-guided Monte-Carlo Tree Search (MCTS) and supports both supervised and reinforcement learning, with a search space defined by a context-free grammar. We summarize seven desirable properties of equation discovery systems, emphasizing the importance of embedding tabular data sets for such learning approaches. Using the modular structure of MGMT, we compare seven architectures (among them, RNNs, CNNs, and Transformers) for embedding tabular datasets on the auxiliary task of contrastive learning for tabular data sets on an equation discovery task. For almost all combinations of modules, supervised learning outperforms reinforcement learning. Moreover, our experiments indicate an advantage of using grammar rules as action space instead of tokens. Two adaptations of MCTS -- risk-seeking MCTS and AmEx-MCTS -- can improve equation discovery with that kind of search. 

---
# Interpretable Machine Learning for Oral Lesion Diagnosis through Prototypical Instances Identification 

**Authors**: Alessio Cascione, Mattia Setzu, Federico A. Galatolo, Mario G.C.A. Cimino, Riccardo Guidotti  

**Link**: [PDF](https://arxiv.org/pdf/2503.16938)  

**Abstract**: Decision-making processes in healthcare can be highly complex and challenging. Machine Learning tools offer significant potential to assist in these processes. However, many current methodologies rely on complex models that are not easily interpretable by experts. This underscores the need to develop interpretable models that can provide meaningful support in clinical decision-making. When approaching such tasks, humans typically compare the situation at hand to a few key examples and representative cases imprinted in their memory. Using an approach which selects such exemplary cases and grounds its predictions on them could contribute to obtaining high-performing interpretable solutions to such problems. To this end, we evaluate PivotTree, an interpretable prototype selection model, on an oral lesion detection problem, specifically trying to detect the presence of neoplastic, aphthous and traumatic ulcerated lesions from oral cavity images. We demonstrate the efficacy of using such method in terms of performance and offer a qualitative and quantitative comparison between exemplary cases and ground-truth prototypes selected by experts. 

---
# A New Segment Routing method with Swap Node Selection Strategy Based on Deep Reinforcement Learning for Software Defined Network 

**Authors**: Miao Ye, Jihao Zheng, Qiuxiang Jiang, Yuan Huang, Ziheng Wang, Yong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.16914)  

**Abstract**: The existing segment routing (SR) methods need to determine the routing first and then use path segmentation approaches to select swap nodes to form a segment routing path (SRP). They require re-segmentation of the path when the routing changes. Furthermore, they do not consider the flow table issuance time, which cannot maximize the speed of issuance flow table. To address these issues, this paper establishes an optimization model that can simultaneously form routing strategies and path segmentation strategies for selecting the appropriate swap nodes to reduce flow table issuance time. It also designs an intelligent segment routing algorithm based on deep reinforcement learning (DRL-SR) to solve the proposed model. First, a traffic matrix is designed as the state space for the deep reinforcement learning agent; this matrix includes multiple QoS performance indicators, flow table issuance time overhead and SR label stack depth. Second, the action selection strategy and corresponding reward function are designed, where the agent selects the next node considering the routing; in addition, the action selection strategy whether the newly added node is selected as the swap node and the corresponding reward function are designed considering the time cost factor for the controller to issue the flow table to the swap node. Finally, a series of experiments and their results show that, compared with the existing methods, the designed segmented route optimization model and the intelligent solution algorithm (DRL-SR) can reduce the time overhead required to complete the segmented route establishment task while optimizing performance metrics such as throughput, delays and packet losses. 

---
# MAPS: A Multi-Agent Framework Based on Big Seven Personality and Socratic Guidance for Multimodal Scientific Problem Solving 

**Authors**: Jian Zhang, Zhiyuan Wang, Zhangqi Wang, Xinyu Zhang, Fangzhi Xu, Qika Lin, Rui Mao, Erik Cambria, Jun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.16905)  

**Abstract**: Multimodal scientific problems (MSPs) involve complex issues that require the integration of multiple modalities, such as text and diagrams, presenting a significant challenge in artificial intelligence. While progress has been made in addressing traditional scientific problems, MSPs still face two primary issues: the challenge of multi-modal comprehensive reasoning in scientific problem-solving and the lack of reflective and rethinking capabilities. To address these issues, we introduce a Multi-Agent framework based on the Big Seven Personality and Socratic guidance (MAPS). This framework employs seven distinct agents that leverage feedback mechanisms and the Socratic method to guide the resolution of MSPs. To tackle the first issue, we propose a progressive four-agent solving strategy, where each agent focuses on a specific stage of the problem-solving process. For the second issue, we introduce a Critic agent, inspired by Socratic questioning, which prompts critical thinking and stimulates autonomous learning. We conduct extensive experiments on the EMMA, Olympiad, and MathVista datasets, achieving promising results that outperform the current SOTA model by 15.84% across all tasks. Meanwhile, the additional analytical experiments also verify the model's progress as well as generalization ability. 

---
# MARS: A Multi-Agent Framework Incorporating Socratic Guidance for Automated Prompt Optimization 

**Authors**: Jian Zhang, Zhangqi Wang, Haiping Zhu, Jun Liu, Qika Lin, Erik Cambria  

**Link**: [PDF](https://arxiv.org/pdf/2503.16874)  

**Abstract**: The basic question-answering format of large language models involves inputting a prompt and receiving a response, and the quality of the prompt directly impacts the effectiveness of the response. Automated Prompt Optimization (APO) aims to break free from the cognitive biases of manually designed prompts and explores a broader design space for prompts. However, existing APO methods suffer from limited flexibility of fixed templates and inefficient search in prompt spaces as key issues. To this end, we propose a Multi-Agent framework Incorporating Socratic guidance (MARS), which utilizes multi-agent fusion technology for automatic planning, with gradual continuous optimization and evaluation. Specifically, MARS comprises seven agents, each with distinct functionalities, which autonomously use the Planner to devise an optimization path that ensures flexibility. Additionally, it employs a Teacher-Critic-Student Socratic dialogue pattern to iteratively optimize the prompts while conducting effective search. We conduct extensive experiments on various datasets to validate the effectiveness of our method, and perform additional analytical experiments to assess the model's advancement as well as the interpretability. 

---
# In-House Evaluation Is Not Enough: Towards Robust Third-Party Flaw Disclosure for General-Purpose AI 

**Authors**: Shayne Longpre, Kevin Klyman, Ruth E. Appel, Sayash Kapoor, Rishi Bommasani, Michelle Sahar, Sean McGregor, Avijit Ghosh, Borhane Blili-Hamelin, Nathan Butters, Alondra Nelson, Amit Elazari, Andrew Sellars, Casey John Ellis, Dane Sherrets, Dawn Song, Harley Geiger, Ilona Cohen, Lauren McIlvenny, Madhulika Srikumar, Mark M. Jaycox, Markus Anderljung, Nadine Farid Johnson, Nicholas Carlini, Nicolas Miailhe, Nik Marda, Peter Henderson, Rebecca S. Portnoff, Rebecca Weiss, Victoria Westerhoff, Yacine Jernite, Rumman Chowdhury, Percy Liang, Arvind Narayanan  

**Link**: [PDF](https://arxiv.org/pdf/2503.16861)  

**Abstract**: The widespread deployment of general-purpose AI (GPAI) systems introduces significant new risks. Yet the infrastructure, practices, and norms for reporting flaws in GPAI systems remain seriously underdeveloped, lagging far behind more established fields like software security. Based on a collaboration between experts from the fields of software security, machine learning, law, social science, and policy, we identify key gaps in the evaluation and reporting of flaws in GPAI systems. We call for three interventions to advance system safety. First, we propose using standardized AI flaw reports and rules of engagement for researchers in order to ease the process of submitting, reproducing, and triaging flaws in GPAI systems. Second, we propose GPAI system providers adopt broadly-scoped flaw disclosure programs, borrowing from bug bounties, with legal safe harbors to protect researchers. Third, we advocate for the development of improved infrastructure to coordinate distribution of flaw reports across the many stakeholders who may be impacted. These interventions are increasingly urgent, as evidenced by the prevalence of jailbreaks and other flaws that can transfer across different providers' GPAI systems. By promoting robust reporting and coordination in the AI ecosystem, these proposals could significantly improve the safety, security, and accountability of GPAI systems. 

---
# A Learnability Analysis on Neuro-Symbolic Learning 

**Authors**: Hao-Yuan He, Ming Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.16797)  

**Abstract**: This paper analyzes the learnability of neuro-symbolic (NeSy) tasks within hybrid systems. We show that the learnability of NeSy tasks can be characterized by their derived constraint satisfaction problems (DCSPs). Specifically, a task is learnable if the corresponding DCSP has a unique solution; otherwise, it is unlearnable. For learnable tasks, we establish error bounds by exploiting the clustering property of the hypothesis space. Additionally, we analyze the asymptotic error for general NeSy tasks, showing that the expected error scales with the disagreement among solutions. Our results offer a principled approach to determining learnability and provide insights into the design of new algorithms. 

---
# Does Chain-of-Thought Reasoning Help Mobile GUI Agent? An Empirical Study 

**Authors**: Li Zhang, Longxi Gao, Mengwei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2503.16788)  

**Abstract**: Reasoning capabilities have significantly improved the performance of vision-language models (VLMs) in domains such as mathematical problem-solving, coding, and visual question-answering. However, their impact on real-world applications remains unclear. This paper presents the first empirical study on the effectiveness of reasoning-enabled VLMs in mobile GUI agents, a domain that requires interpreting complex screen layouts, understanding user instructions, and executing multi-turn interactions. We evaluate two pairs of commercial models--Gemini 2.0 Flash and Claude 3.7 Sonnet--comparing their base and reasoning-enhanced versions across two static benchmarks (ScreenSpot and AndroidControl) and one interactive environment (AndroidWorld). We surprisingly find the Claude 3.7 Sonnet reasoning model achieves state-of-the-art performance on AndroidWorld. However, reasoning VLMs generally offer marginal improvements over non-reasoning models on static benchmarks and even degrade performance in some agent setups. Notably, reasoning and non-reasoning VLMs fail on different sets of tasks, suggesting that reasoning does have an impact, but its benefits and drawbacks counterbalance each other. We attribute these inconsistencies to the limitations of benchmarks and VLMs. Based on the findings, we provide insights for further enhancing mobile GUI agents in terms of benchmarks, VLMs, and their adaptability in dynamically invoking reasoning VLMs. The experimental data are publicly available at this https URL. 

---
# SuperARC: A Test for General and Super Intelligence Based on First Principles of Recursion Theory and Algorithmic Probability 

**Authors**: Alberto Hernández-Espinosa, Luan Ozelim, Felipe S. Abrahão, Hector Zenil  

**Link**: [PDF](https://arxiv.org/pdf/2503.16743)  

**Abstract**: We introduce an open-ended test grounded in algorithmic probability that can avoid benchmark contamination in the quantitative evaluation of frontier models in the context of their Artificial General Intelligence (AGI) and Superintelligence (ASI) claims. Unlike other tests, this test does not rely on statistical compression methods (such as GZIP or LZW), which are more closely related to Shannon entropy than to Kolmogorov complexity. The test challenges aspects related to features of intelligence of fundamental nature such as synthesis and model creation in the context of inverse problems (generating new knowledge from observation). We argue that metrics based on model abstraction and optimal Bayesian inference for planning can provide a robust framework for testing intelligence, including natural intelligence (human and animal), narrow AI, AGI, and ASI. Our results show no clear evidence of LLM convergence towards a defined level of intelligence, particularly AGI or ASI. We found that LLM model versions tend to be fragile and incremental, as new versions may perform worse than older ones, with progress largely driven by the size of training data. The results were compared with a hybrid neurosymbolic approach that theoretically guarantees model convergence from optimal inference based on the principles of algorithmic probability and Kolmogorov complexity. The method outperforms LLMs in a proof-of-concept on short binary sequences. Our findings confirm suspicions regarding the fundamental limitations of LLMs, exposing them as systems optimised for the perception of mastery over human language. Progress among different LLM versions from the same developers was found to be inconsistent and limited, particularly in the absence of a solid symbolic counterpart. 

---
# Towards Agentic Recommender Systems in the Era of Multimodal Large Language Models 

**Authors**: Chengkai Huang, Junda Wu, Yu Xia, Zixu Yu, Ruhan Wang, Tong Yu, Ruiyi Zhang, Ryan A. Rossi, Branislav Kveton, Dongruo Zhou, Julian McAuley, Lina Yao  

**Link**: [PDF](https://arxiv.org/pdf/2503.16734)  

**Abstract**: Recent breakthroughs in Large Language Models (LLMs) have led to the emergence of agentic AI systems that extend beyond the capabilities of standalone models. By empowering LLMs to perceive external environments, integrate multimodal information, and interact with various tools, these agentic systems exhibit greater autonomy and adaptability across complex tasks. This evolution brings new opportunities to recommender systems (RS): LLM-based Agentic RS (LLM-ARS) can offer more interactive, context-aware, and proactive recommendations, potentially reshaping the user experience and broadening the application scope of RS. Despite promising early results, fundamental challenges remain, including how to effectively incorporate external knowledge, balance autonomy with controllability, and evaluate performance in dynamic, multimodal settings. In this perspective paper, we first present a systematic analysis of LLM-ARS: (1) clarifying core concepts and architectures; (2) highlighting how agentic capabilities -- such as planning, memory, and multimodal reasoning -- can enhance recommendation quality; and (3) outlining key research questions in areas such as safety, efficiency, and lifelong personalization. We also discuss open problems and future directions, arguing that LLM-ARS will drive the next wave of RS innovation. Ultimately, we foresee a paradigm shift toward intelligent, autonomous, and collaborative recommendation experiences that more closely align with users' evolving needs and complex decision-making processes. 

---
# Towards Automated Semantic Interpretability in Reinforcement Learning via Vision-Language Models 

**Authors**: Zhaoxin Li, Zhang Xi-Jia, Batuhan Altundas, Letian Chen, Rohan Paleja, Matthew Gombolay  

**Link**: [PDF](https://arxiv.org/pdf/2503.16724)  

**Abstract**: Semantic Interpretability in Reinforcement Learning (RL) enables transparency, accountability, and safer deployment by making the agent's decisions understandable and verifiable. Achieving this, however, requires a feature space composed of human-understandable concepts, which traditionally rely on human specification and fail to generalize to unseen environments. In this work, we introduce Semantically Interpretable Reinforcement Learning with Vision-Language Models Empowered Automation (SILVA), an automated framework that leverages pre-trained vision-language models (VLM) for semantic feature extraction and interpretable tree-based models for policy optimization. SILVA first queries a VLM to identify relevant semantic features for an unseen environment, then extracts these features from the environment. Finally, it trains an Interpretable Control Tree via RL, mapping the extracted features to actions in a transparent and interpretable manner. To address the computational inefficiency of extracting features directly with VLMs, we develop a feature extraction pipeline that generates a dataset for training a lightweight convolutional network, which is subsequently used during RL. By leveraging VLMs to automate tree-based RL, SILVA removes the reliance on human annotation previously required by interpretable models while also overcoming the inability of VLMs alone to generate valid robot policies, enabling semantically interpretable reinforcement learning without human-in-the-loop. 

---
# Empowering Medical Multi-Agents with Clinical Consultation Flow for Dynamic Diagnosis 

**Authors**: Sihan Wang, Suiyang Jiang, Yibo Gao, Boming Wang, Shangqi Gao, Xiahai Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2503.16547)  

**Abstract**: Traditional AI-based healthcare systems often rely on single-modal data, limiting diagnostic accuracy due to incomplete information. However, recent advancements in foundation models show promising potential for enhancing diagnosis combining multi-modal information. While these models excel in static tasks, they struggle with dynamic diagnosis, failing to manage multi-turn interactions and often making premature diagnostic decisions due to insufficient persistence in information this http URL address this, we propose a multi-agent framework inspired by consultation flow and reinforcement learning (RL) to simulate the entire consultation process, integrating multiple clinical information for effective diagnosis. Our approach incorporates a hierarchical action set, structured from clinic consultation flow and medical textbook, to effectively guide the decision-making process. This strategy improves agent interactions, enabling them to adapt and optimize actions based on the dynamic state. We evaluated our framework on a public dynamic diagnosis benchmark. The proposed framework evidentially improves the baseline methods and achieves state-of-the-art performance compared to existing foundation model-based methods. 

---
# Improving Interactive Diagnostic Ability of a Large Language Model Agent Through Clinical Experience Learning 

**Authors**: Zhoujian Sun, Ziyi Liu, Cheng Luo, Jiebin Chu, Zhengxing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2503.16463)  

**Abstract**: Recent advances in large language models (LLMs) have shown promising results in medical diagnosis, with some studies indicating superior performance compared to human physicians in specific scenarios. However, the diagnostic capabilities of LLMs are often overestimated, as their performance significantly deteriorates in interactive diagnostic settings that require active information gathering. This study investigates the underlying mechanisms behind the performance degradation phenomenon and proposes a solution. We identified that the primary deficiency of LLMs lies in the initial diagnosis phase, particularly in information-gathering efficiency and initial diagnosis formation, rather than in the subsequent differential diagnosis phase. To address this limitation, we developed a plug-and-play method enhanced (PPME) LLM agent, leveraging over 3.5 million electronic medical records from Chinese and American healthcare facilities. Our approach integrates specialized models for initial disease diagnosis and inquiry into the history of the present illness, trained through supervised and reinforcement learning techniques. The experimental results indicate that the PPME LLM achieved over 30% improvement compared to baselines. The final diagnostic accuracy of the PPME LLM in interactive diagnostic scenarios approached levels comparable to those achieved using complete clinical data. These findings suggest a promising potential for developing autonomous diagnostic systems, although further validation studies are needed. 

---
# NdLinear Is All You Need for Representation Learning 

**Authors**: Alex Reneau, Jerry Yao-Chieh Hu, Zhongfang Zhuang, Ting-Chun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.17353)  

**Abstract**: Many high-impact machine learning tasks involve multi-dimensional data (e.g., images, volumetric medical scans, multivariate time-series). Yet, most neural architectures flatten inputs, discarding critical cross-dimension information. We introduce NdLinear, a novel linear transformation that preserves these structures without extra overhead. By operating separately along each dimension, NdLinear captures dependencies that standard fully connected layers overlook. Extensive experiments across convolutional, recurrent, and transformer-based networks show significant improvements in representational power and parameter efficiency. Crucially, NdLinear serves as a foundational building block for large-scale foundation models by operating on any unimodal or multimodal data in its native form. This removes the need for flattening or modality-specific preprocessing. Ndlinear rethinks core architectural priorities beyond attention, enabling more expressive, context-aware models at scale. We propose NdLinear as a drop-in replacement for standard linear layers -- marking an important step toward next-generation neural architectures. 

---
# Align Your Rhythm: Generating Highly Aligned Dance Poses with Gating-Enhanced Rhythm-Aware Feature Representation 

**Authors**: Congyi Fan, Jian Guan, Xuanjia Zhao, Dongli Xu, Youtian Lin, Tong Ye, Pengming Feng, Haiwei Pan  

**Link**: [PDF](https://arxiv.org/pdf/2503.17340)  

**Abstract**: Automatically generating natural, diverse and rhythmic human dance movements driven by music is vital for virtual reality and film industries. However, generating dance that naturally follows music remains a challenge, as existing methods lack proper beat alignment and exhibit unnatural motion dynamics. In this paper, we propose Danceba, a novel framework that leverages gating mechanism to enhance rhythm-aware feature representation for music-driven dance generation, which achieves highly aligned dance poses with enhanced rhythmic sensitivity. Specifically, we introduce Phase-Based Rhythm Extraction (PRE) to precisely extract rhythmic information from musical phase data, capitalizing on the intrinsic periodicity and temporal structures of music. Additionally, we propose Temporal-Gated Causal Attention (TGCA) to focus on global rhythmic features, ensuring that dance movements closely follow the musical rhythm. We also introduce Parallel Mamba Motion Modeling (PMMM) architecture to separately model upper and lower body motions along with musical features, thereby improving the naturalness and diversity of generated dance movements. Extensive experiments confirm that Danceba outperforms state-of-the-art methods, achieving significantly better rhythmic alignment and motion diversity. Project page: this https URL . 

---
# Can AI expose tax loopholes? Towards a new generation of legal policy assistants 

**Authors**: Peter Fratrič, Nils Holzenberger, David Restrepo Amariles  

**Link**: [PDF](https://arxiv.org/pdf/2503.17339)  

**Abstract**: The legislative process is the backbone of a state built on solid institutions. Yet, due to the complexity of laws -- particularly tax law -- policies may lead to inequality and social tensions. In this study, we introduce a novel prototype system designed to address the issues of tax loopholes and tax avoidance. Our hybrid solution integrates a natural language interface with a domain-specific language tailored for planning. We demonstrate on a case study how tax loopholes and avoidance schemes can be exposed. We conclude that our prototype can help enhance social welfare by systematically identifying and addressing tax gaps stemming from loopholes. 

---
# Efficient Intent-Based Filtering for Multi-Party Conversations Using Knowledge Distillation from LLMs 

**Authors**: Reem Gody, Mohamed Abdelghaffar, Mohammed Jabreel, Ahmed Tawfik  

**Link**: [PDF](https://arxiv.org/pdf/2503.17336)  

**Abstract**: Large language models (LLMs) have showcased remarkable capabilities in conversational AI, enabling open-domain responses in chat-bots, as well as advanced processing of conversations like summarization, intent classification, and insights generation. However, these models are resource-intensive, demanding substantial memory and computational power. To address this, we propose a cost-effective solution that filters conversational snippets of interest for LLM processing, tailored to the target downstream application, rather than processing every snippet. In this work, we introduce an innovative approach that leverages knowledge distillation from LLMs to develop an intent-based filter for multi-party conversations, optimized for compute power constrained environments. Our method combines different strategies to create a diverse multi-party conversational dataset, that is annotated with the target intents and is then used to fine-tune the MobileBERT model for multi-label intent classification. This model achieves a balance between efficiency and performance, effectively filtering conversation snippets based on their intents. By passing only the relevant snippets to the LLM for further processing, our approach significantly reduces overall operational costs depending on the intents and the data distribution as demonstrated in our experiments. 

---
# CVE-Bench: A Benchmark for AI Agents' Ability to Exploit Real-World Web Application Vulnerabilities 

**Authors**: Yuxuan Zhu, Antony Kellermann, Dylan Bowman, Philip Li, Akul Gupta, Adarsh Danda, Richard Fang, Conner Jensen, Eric Ihli, Jason Benn, Jet Geronimo, Avi Dhir, Sudhit Rao, Kaicheng Yu, Twm Stone, Daniel Kang  

**Link**: [PDF](https://arxiv.org/pdf/2503.17332)  

**Abstract**: Large language model (LLM) agents are increasingly capable of autonomously conducting cyberattacks, posing significant threats to existing applications. This growing risk highlights the urgent need for a real-world benchmark to evaluate the ability of LLM agents to exploit web application vulnerabilities. However, existing benchmarks fall short as they are limited to abstracted Capture the Flag competitions or lack comprehensive coverage. Building a benchmark for real-world vulnerabilities involves both specialized expertise to reproduce exploits and a systematic approach to evaluating unpredictable threats. To address this challenge, we introduce CVE-Bench, a real-world cybersecurity benchmark based on critical-severity Common Vulnerabilities and Exposures. In CVE-Bench, we design a sandbox framework that enables LLM agents to exploit vulnerable web applications in scenarios that mimic real-world conditions, while also providing effective evaluation of their exploits. Our evaluation shows that the state-of-the-art agent framework can resolve up to 13% of vulnerabilities. 

---
# LLM+MAP: Bimanual Robot Task Planning using Large Language Models and Planning Domain Definition Language 

**Authors**: Kun Chu, Xufeng Zhao, Cornelius Weber, Stefan Wermter  

**Link**: [PDF](https://arxiv.org/pdf/2503.17309)  

**Abstract**: Bimanual robotic manipulation provides significant versatility, but also presents an inherent challenge due to the complexity involved in the spatial and temporal coordination between two hands. Existing works predominantly focus on attaining human-level manipulation skills for robotic hands, yet little attention has been paid to task planning on long-horizon timescales. With their outstanding in-context learning and zero-shot generation abilities, Large Language Models (LLMs) have been applied and grounded in diverse robotic embodiments to facilitate task planning. However, LLMs still suffer from errors in long-horizon reasoning and from hallucinations in complex robotic tasks, lacking a guarantee of logical correctness when generating the plan. Previous works, such as LLM+P, extended LLMs with symbolic planners. However, none have been successfully applied to bimanual robots. New challenges inevitably arise in bimanual manipulation, necessitating not only effective task decomposition but also efficient task allocation. To address these challenges, this paper introduces LLM+MAP, a bimanual planning framework that integrates LLM reasoning and multi-agent planning, automating effective and efficient bimanual task planning. We conduct simulated experiments on various long-horizon manipulation tasks of differing complexity. Our method is built using GPT-4o as the backend, and we compare its performance against plans generated directly by LLMs, including GPT-4o, V3 and also recent strong reasoning models o1 and R1. By analyzing metrics such as planning time, success rate, group debits, and planning-step reduction rate, we demonstrate the superior performance of LLM+MAP, while also providing insights into robotic reasoning. Code is available at this https URL. 

---
# Preference-Guided Diffusion for Multi-Objective Offline Optimization 

**Authors**: Yashas Annadani, Syrine Belakaria, Stefano Ermon, Stefan Bauer, Barbara E Engelhardt  

**Link**: [PDF](https://arxiv.org/pdf/2503.17299)  

**Abstract**: Offline multi-objective optimization aims to identify Pareto-optimal solutions given a dataset of designs and their objective values. In this work, we propose a preference-guided diffusion model that generates Pareto-optimal designs by leveraging a classifier-based guidance mechanism. Our guidance classifier is a preference model trained to predict the probability that one design dominates another, directing the diffusion model toward optimal regions of the design space. Crucially, this preference model generalizes beyond the training distribution, enabling the discovery of Pareto-optimal solutions outside the observed dataset. We introduce a novel diversity-aware preference guidance, augmenting Pareto dominance preference with diversity criteria. This ensures that generated solutions are optimal and well-distributed across the objective space, a capability absent in prior generative methods for offline multi-objective optimization. We evaluate our approach on various continuous offline multi-objective optimization tasks and find that it consistently outperforms other inverse/generative approaches while remaining competitive with forward/surrogate-based optimization methods. Our results highlight the effectiveness of classifier-guided diffusion models in generating diverse and high-quality solutions that approximate the Pareto front well. 

---
# KL3M Tokenizers: A Family of Domain-Specific and Character-Level Tokenizers for Legal, Financial, and Preprocessing Applications 

**Authors**: Michael J Bommarito, Daniel Martin Katz, Jillian Bommarito  

**Link**: [PDF](https://arxiv.org/pdf/2503.17247)  

**Abstract**: We present the KL3M tokenizers, a family of specialized tokenizers for legal, financial, and governmental text. Despite established work on tokenization, specialized tokenizers for professional domains remain understudied. Our paper offers two main contributions to this area.
First, we introduce domain-specific BPE tokenizers for legal, financial, and governmental text. Our kl3m-004-128k-cased tokenizer uses 9-17% fewer tokens than GPT-4o and Llama3 for domain-specific documents, despite having a smaller vocabulary. For specialized terminology, our cased tokenizer is even more efficient, using up to 83% fewer tokens for legal terms and 39% fewer tokens for financial terms.
Second, we develop character-level BPE tokenizers (4K, 8K, and 16K vocabulary sizes) for text correction tasks like OCR post-processing. These tokenizers keep consistent token boundaries between error-containing and correct text, making it easier for models to learn correction patterns.
These tokenizers help professional applications by fitting more text in context windows, reducing computational needs, and preserving the meaning of domain-specific terms. Our analysis shows these efficiency gains directly benefit the processing of long legal and financial documents. We release all tokenizers and code through GitHub and Hugging Face to support further research in specialized tokenization. 

---
# SafeMERGE: Preserving Safety Alignment in Fine-Tuned Large Language Models via Selective Layer-Wise Model Merging 

**Authors**: Aladin Djuhera, Swanand Ravindra Kadhe, Farhan Ahmed, Syed Zawad, Holger Boche  

**Link**: [PDF](https://arxiv.org/pdf/2503.17239)  

**Abstract**: Fine-tuning large language models (LLMs) on downstream tasks can inadvertently erode their safety alignment, even for benign fine-tuning datasets. We address this challenge by proposing SafeMERGE, a post-fine-tuning framework that preserves safety while maintaining task utility. It achieves this by selectively merging fine-tuned and safety-aligned model layers only when those deviate from safe behavior, measured by a cosine similarity criterion. We evaluate SafeMERGE against other fine-tuning- and post-fine-tuning-stage approaches for Llama-2-7B-Chat and Qwen-2-7B-Instruct models on GSM8K and PubMedQA tasks while exploring different merging strategies. We find that SafeMERGE consistently reduces harmful outputs compared to other baselines without significantly sacrificing performance, sometimes even enhancing it. The results suggest that our selective, subspace-guided, and per-layer merging method provides an effective safeguard against the inadvertent loss of safety in fine-tuned LLMs while outperforming simpler post-fine-tuning-stage defenses. 

---
# Strong Baseline: Multi-UAV Tracking via YOLOv12 with BoT-SORT-ReID 

**Authors**: Yu-Hsi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.17237)  

**Abstract**: Detecting and tracking multiple unmanned aerial vehicles (UAVs) in thermal infrared video is inherently challenging due to low contrast, environmental noise, and small target sizes. This paper provides a straightforward approach to address multi-UAV tracking in thermal infrared video, leveraging recent advances in detection and tracking. Instead of relying on the YOLOv5 with the DeepSORT pipeline, we present a tracking framework built on YOLOv12 and BoT-SORT, enhanced with tailored training and inference strategies. We evaluate our approach following the metrics from the 4th Anti-UAV Challenge and demonstrate competitive performance. Notably, we achieve strong results without using contrast enhancement or temporal information fusion to enrich UAV features, highlighting our approach as a "Strong Baseline" for the multi-UAV tracking task. We provide implementation details, in-depth experimental analysis, and a discussion of potential improvements. The code is available at this https URL . 

---
# FactSelfCheck: Fact-Level Black-Box Hallucination Detection for LLMs 

**Authors**: Albert Sawczyn, Jakub Binkowski, Denis Janiak, Bogdan Gabrys, Tomasz Kajdanowicz  

**Link**: [PDF](https://arxiv.org/pdf/2503.17229)  

**Abstract**: Large Language Models (LLMs) frequently generate hallucinated content, posing significant challenges for applications where factuality is crucial. While existing hallucination detection methods typically operate at the sentence level or passage level, we propose FactSelfCheck, a novel black-box sampling-based method that enables fine-grained fact-level detection. Our approach represents text as knowledge graphs consisting of facts in the form of triples. Through analyzing factual consistency across multiple LLM responses, we compute fine-grained hallucination scores without requiring external resources or training data. Our evaluation demonstrates that FactSelfCheck performs competitively with leading sampling-based methods while providing more detailed insights. Most notably, our fact-level approach significantly improves hallucination correction, achieving a 35% increase in factual content compared to the baseline, while sentence-level SelfCheckGPT yields only an 8% improvement. The granular nature of our detection enables more precise identification and correction of hallucinated content. 

---
# Neuro-Symbolic Scene Graph Conditioning for Synthetic Image Dataset Generation 

**Authors**: Giacomo Savazzi, Eugenio Lomurno, Cristian Sbrolli, Agnese Chiatti, Matteo Matteucci  

**Link**: [PDF](https://arxiv.org/pdf/2503.17224)  

**Abstract**: As machine learning models increase in scale and complexity, obtaining sufficient training data has become a critical bottleneck due to acquisition costs, privacy constraints, and data scarcity in specialised domains. While synthetic data generation has emerged as a promising alternative, a notable performance gap remains compared to models trained on real data, particularly as task complexity grows. Concurrently, Neuro-Symbolic methods, which combine neural networks' learning strengths with symbolic reasoning's structured representations, have demonstrated significant potential across various cognitive tasks. This paper explores the utility of Neuro-Symbolic conditioning for synthetic image dataset generation, focusing specifically on improving the performance of Scene Graph Generation models. The research investigates whether structured symbolic representations in the form of scene graphs can enhance synthetic data quality through explicit encoding of relational constraints. The results demonstrate that Neuro-Symbolic conditioning yields significant improvements of up to +2.59% in standard Recall metrics and +2.83% in No Graph Constraint Recall metrics when used for dataset augmentation. These findings establish that merging Neuro-Symbolic and generative approaches produces synthetic data with complementary structural information that enhances model performance when combined with real data, providing a novel approach to overcome data scarcity limitations even for complex visual reasoning tasks. 

---
# Automating Adjudication of Cardiovascular Events Using Large Language Models 

**Authors**: Sonish Sivarajkumar, Kimia Ameri, Chuqin Li, Yanshan Wang, Min Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2503.17222)  

**Abstract**: Cardiovascular events, such as heart attacks and strokes, remain a leading cause of mortality globally, necessitating meticulous monitoring and adjudication in clinical trials. This process, traditionally performed manually by clinical experts, is time-consuming, resource-intensive, and prone to inter-reviewer variability, potentially introducing bias and hindering trial progress. This study addresses these critical limitations by presenting a novel framework for automating the adjudication of cardiovascular events in clinical trials using Large Language Models (LLMs). We developed a two-stage approach: first, employing an LLM-based pipeline for event information extraction from unstructured clinical data and second, using an LLM-based adjudication process guided by a Tree of Thoughts approach and clinical endpoint committee (CEC) guidelines. Using cardiovascular event-specific clinical trial data, the framework achieved an F1-score of 0.82 for event extraction and an accuracy of 0.68 for adjudication. Furthermore, we introduce the CLEART score, a novel, automated metric specifically designed for evaluating the quality of AI-generated clinical reasoning in adjudicating cardiovascular events. This approach demonstrates significant potential for substantially reducing adjudication time and costs while maintaining high-quality, consistent, and auditable outcomes in clinical trials. The reduced variability and enhanced standardization also allow for faster identification and mitigation of risks associated with cardiovascular therapies. 

---
# PP-DocLayout: A Unified Document Layout Detection Model to Accelerate Large-Scale Data Construction 

**Authors**: Ting Sun, Cheng Cui, Yuning Du, Yi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.17213)  

**Abstract**: Document layout analysis is a critical preprocessing step in document intelligence, enabling the detection and localization of structural elements such as titles, text blocks, tables, and formulas. Despite its importance, existing layout detection models face significant challenges in generalizing across diverse document types, handling complex layouts, and achieving real-time performance for large-scale data processing. To address these limitations, we present PP-DocLayout, which achieves high precision and efficiency in recognizing 23 types of layout regions across diverse document formats. To meet different needs, we offer three models of varying scales. PP-DocLayout-L is a high-precision model based on the RT-DETR-L detector, achieving 90.4% mAP@0.5 and an end-to-end inference time of 13.4 ms per page on a T4 GPU. PP-DocLayout-M is a balanced model, offering 75.2% mAP@0.5 with an inference time of 12.7 ms per page on a T4 GPU. PP-DocLayout-S is a high-efficiency model designed for resource-constrained environments and real-time applications, with an inference time of 8.1 ms per page on a T4 GPU and 14.5 ms on a CPU. This work not only advances the state of the art in document layout analysis but also provides a robust solution for constructing high-quality training data, enabling advancements in document intelligence and multimodal AI systems. Code and models are available at this https URL . 

---
# TreeSynth: Synthesizing Diverse Data from Scratch via Tree-Guided Subspace Partitioning 

**Authors**: Sheng Wang, Pengan Chen, Jingqi Zhou, Qintong Li, Jingwei Dong, Jiahui Gao, Boyang Xue, Jiyue Jiang, Lingpeng Kong, Chuan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2503.17195)  

**Abstract**: Model customization requires high-quality and diverse datasets, but acquiring such data remains challenging and costly. Although large language models (LLMs) can synthesize training data, current approaches are constrained by limited seed data, model bias and insufficient control over the generation process, resulting in limited diversity and biased distribution with the increase of data scales. To tackle this challenge, we present TreeSynth, a tree-guided subspace-based data synthesis framework that recursively partitions the entire data space into hierar-chical subspaces, enabling comprehensive and diverse scaling of data synthesis. Briefly, given a task-specific description, we construct a data space partitioning tree by iteratively executing criteria determination and subspace coverage steps. This hierarchically divides the whole space (i.e., root node) into mutually exclusive and complementary atomic subspaces (i.e., leaf nodes). By collecting synthesized data according to the attributes of each leaf node, we obtain a diverse dataset that fully covers the data space. Empirically, our extensive experiments demonstrate that TreeSynth surpasses both human-designed datasets and the state-of-the-art data synthesis baselines, achieving maximum improvements of 45.2% in data diversity and 17.6% in downstream task performance across various models and tasks. Hopefully, TreeSynth provides a scalable solution to synthesize diverse and comprehensive datasets from scratch without human intervention. 

---
# D2Fusion: Dual-domain Fusion with Feature Superposition for Deepfake Detection 

**Authors**: Xueqi Qiu, Xingyu Miao, Fan Wan, Haoran Duan, Tejal Shah, Varun Ojhab, Yang Longa, Rajiv Ranjan  

**Link**: [PDF](https://arxiv.org/pdf/2503.17184)  

**Abstract**: Deepfake detection is crucial for curbing the harm it causes to society. However, current Deepfake detection methods fail to thoroughly explore artifact information across different domains due to insufficient intrinsic interactions. These interactions refer to the fusion and coordination after feature extraction processes across different domains, which are crucial for recognizing complex forgery clues. Focusing on more generalized Deepfake detection, in this work, we introduce a novel bi-directional attention module to capture the local positional information of artifact clues from the spatial domain. This enables accurate artifact localization, thus addressing the coarse processing with artifact features. To further address the limitation that the proposed bi-directional attention module may not well capture global subtle forgery information in the artifact feature (e.g., textures or edges), we employ a fine-grained frequency attention module in the frequency domain. By doing so, we can obtain high-frequency information in the fine-grained features, which contains the global and subtle forgery information. Although these features from the diverse domains can be effectively and independently improved, fusing them directly does not effectively improve the detection performance. Therefore, we propose a feature superposition strategy that complements information from spatial and frequency domains. This strategy turns the feature components into the form of wave-like tokens, which are updated based on their phase, such that the distinctions between authentic and artifact features can be amplified. Our method demonstrates significant improvements over state-of-the-art (SOTA) methods on five public Deepfake datasets in capturing abnormalities across different manipulated operations and real-life. 

---
# LLMs Love Python: A Study of LLMs' Bias for Programming Languages and Libraries 

**Authors**: Lukas Twist, Jie M. Zhang, Mark Harman, Don Syme, Joost Noppen, Detlef Nauck  

**Link**: [PDF](https://arxiv.org/pdf/2503.17181)  

**Abstract**: Programming language and library choices are crucial to software reliability and security. Poor or inconsistent choices can lead to increased technical debt, security vulnerabilities, and even catastrophic failures in safety-critical systems. As Large Language Models (LLMs) play an increasing role in code generation, it is essential to understand how they make these decisions. However, little is known about their preferences when selecting programming languages and libraries for different coding tasks. To fill this gap, this study provides the first in-depth investigation into LLM preferences for programming languages and libraries used when generating code. We assess the preferences of eight diverse LLMs by prompting them to complete various coding tasks, including widely-studied benchmarks and the more practical task of generating the initial structural code for new projects (a crucial step that often determines a project's language or library choices).
Our findings reveal that LLMs heavily favour Python when solving language-agnostic problems, using it in 90%-97% of cases for benchmark tasks. Even when generating initial project code where Python is not a suitable language, it remains the most-used language in 58% of instances. Moreover, LLMs contradict their own language recommendations in 83% of project initialisation tasks, raising concerns about their reliability in guiding language selection. Similar biases toward well-established libraries further create serious discoverability challenges for newer open-source projects. These results highlight the need to improve LLMs' adaptability to diverse programming contexts and to develop mechanisms for mitigating programming language and library bias. 

---
# DiTEC-WDN: A Large-Scale Dataset of Water Distribution Network Scenarios under Diverse Hydraulic Conditions 

**Authors**: Huy Truong, Andrés Tello, Alexander Lazovik, Victoria Degeler  

**Link**: [PDF](https://arxiv.org/pdf/2503.17167)  

**Abstract**: Privacy restrictions hinder the sharing of real-world Water Distribution Network (WDN) models, limiting the application of emerging data-driven machine learning, which typically requires extensive observations. To address this challenge, we propose the dataset DiTEC-WDN that comprises 36,000 unique scenarios simulated over either short-term (24 hours) or long-term (1 year) periods. We constructed this dataset using an automated pipeline that optimizes crucial parameters (e.g., pressure, flow rate, and demand patterns), facilitates large-scale simulations, and records discrete, synthetic but hydraulically realistic states under standard conditions via rule validation and post-hoc analysis. With a total of 228 million generated graph-based states, DiTEC-WDN can support a variety of machine-learning tasks, including graph-level, node-level, and link-level regression, as well as time-series forecasting. This contribution, released under a public license, encourages open scientific research in the critical water sector, eliminates the risk of exposing sensitive data, and fulfills the need for a large-scale water distribution network benchmark for study comparisons and scenario analysis. 

---
# Temporal-Guided Spiking Neural Networks for Event-Based Human Action Recognition 

**Authors**: Siyuan Yang, Shilin Lu, Shizheng Wang, Meng Hwa Er, Zengwei Zheng, Alex C. Kot  

**Link**: [PDF](https://arxiv.org/pdf/2503.17132)  

**Abstract**: This paper explores the promising interplay between spiking neural networks (SNNs) and event-based cameras for privacy-preserving human action recognition (HAR). The unique feature of event cameras in capturing only the outlines of motion, combined with SNNs' proficiency in processing spatiotemporal data through spikes, establishes a highly synergistic compatibility for event-based HAR. Previous studies, however, have been limited by SNNs' ability to process long-term temporal information, essential for precise HAR. In this paper, we introduce two novel frameworks to address this: temporal segment-based SNN (\textit{TS-SNN}) and 3D convolutional SNN (\textit{3D-SNN}). The \textit{TS-SNN} extracts long-term temporal information by dividing actions into shorter segments, while the \textit{3D-SNN} replaces 2D spatial elements with 3D components to facilitate the transmission of temporal information. To promote further research in event-based HAR, we create a dataset, \textit{FallingDetection-CeleX}, collected using the high-resolution CeleX-V event camera $(1280 \times 800)$, comprising 7 distinct actions. Extensive experimental results show that our proposed frameworks surpass state-of-the-art SNN methods on our newly collected dataset and three other neuromorphic datasets, showcasing their effectiveness in handling long-range temporal information for event-based HAR. 

---
# Leveraging Language Models for Out-of-Distribution Recovery in Reinforcement Learning 

**Authors**: Chan Kim, Seung-Woo Seo, Seong-Woo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2503.17125)  

**Abstract**: Deep Reinforcement Learning (DRL) has demonstrated strong performance in robotic control but remains susceptible to out-of-distribution (OOD) states, often resulting in unreliable actions and task failure. While previous methods have focused on minimizing or preventing OOD occurrences, they largely neglect recovery once an agent encounters such states. Although the latest research has attempted to address this by guiding agents back to in-distribution states, their reliance on uncertainty estimation hinders scalability in complex environments. To overcome this limitation, we introduce Language Models for Out-of-Distribution Recovery (LaMOuR), which enables recovery learning without relying on uncertainty estimation. LaMOuR generates dense reward codes that guide the agent back to a state where it can successfully perform its original task, leveraging the capabilities of LVLMs in image description, logical reasoning, and code generation. Experimental results show that LaMOuR substantially enhances recovery efficiency across diverse locomotion tasks and even generalizes effectively to complex environments, including humanoid locomotion and mobile manipulation, where existing methods struggle. The code and supplementary materials are available at \href{this https URL}{this https URL}. 

---
# The CASTLE 2024 Dataset: Advancing the Art of Multimodal Understanding 

**Authors**: Luca Rossetto, Werner Bailer, Duc-Tien Dang-Nguyen, Graham Healy, Björn Þór Jónsson, Onanong Kongmeesub, Hoang-Bao Le, Stevan Rudinac, Klaus Schöffmann, Florian Spiess, Allie Tran, Minh-Triet Tran, Quang-Linh Tran, Cathal Gurrin  

**Link**: [PDF](https://arxiv.org/pdf/2503.17116)  

**Abstract**: Egocentric video has seen increased interest in recent years, as it is used in a range of areas. However, most existing datasets are limited to a single perspective. In this paper, we present the CASTLE 2024 dataset, a multimodal collection containing ego- and exo-centric (i.e., first- and third-person perspective) video and audio from 15 time-aligned sources, as well as other sensor streams and auxiliary data. The dataset was recorded by volunteer participants over four days in a fixed location and includes the point of view of 10 participants, with an additional 5 fixed cameras providing an exocentric perspective. The entire dataset contains over 600 hours of UHD video recorded at 50 frames per second. In contrast to other datasets, CASTLE 2024 does not contain any partial censoring, such as blurred faces or distorted audio. The dataset is available via this https URL. 

---
# FFaceNeRF: Few-shot Face Editing in Neural Radiance Fields 

**Authors**: Kwan Yun, Chaelin Kim, Hangyeul Shin, Junyong Noh  

**Link**: [PDF](https://arxiv.org/pdf/2503.17095)  

**Abstract**: Recent 3D face editing methods using masks have produced high-quality edited images by leveraging Neural Radiance Fields (NeRF). Despite their impressive performance, existing methods often provide limited user control due to the use of pre-trained segmentation masks. To utilize masks with a desired layout, an extensive training dataset is required, which is challenging to gather. We present FFaceNeRF, a NeRF-based face editing technique that can overcome the challenge of limited user control due to the use of fixed mask layouts. Our method employs a geometry adapter with feature injection, allowing for effective manipulation of geometry attributes. Additionally, we adopt latent mixing for tri-plane augmentation, which enables training with a few samples. This facilitates rapid model adaptation to desired mask layouts, crucial for applications in fields like personalized medical imaging or creative face editing. Our comparative evaluations demonstrate that FFaceNeRF surpasses existing mask based face editing methods in terms of flexibility, control, and generated image quality, paving the way for future advancements in customized and high-fidelity 3D face editing. The code is available on the {\href{this https URL}{project-page}}. 

---
# Does a Rising Tide Lift All Boats? Bias Mitigation for AI-based CMR Segmentation 

**Authors**: Tiarna Lee, Esther Puyol-Antón, Bram Ruijsink, Miaojing Shi, Andrew P. King  

**Link**: [PDF](https://arxiv.org/pdf/2503.17089)  

**Abstract**: Artificial intelligence (AI) is increasingly being used for medical imaging tasks. However, there can be biases in the resulting models, particularly when they were trained using imbalanced training datasets. One such example has been the strong race bias effect in cardiac magnetic resonance (CMR) image segmentation models. Although this phenomenon has been reported in a number of publications, little is known about the effectiveness of bias mitigation algorithms in this domain. We aim to investigate the impact of common bias mitigation methods to address bias between Black and White subjects in AI-based CMR segmentation models. Specifically, we use oversampling, importance reweighing and Group DRO as well as combinations of these techniques to mitigate the race bias. Furthermore, motivated by recent findings on the root causes of AI-based CMR segmentation bias, we evaluate the same methods using models trained and evaluated on cropped CMR images. We find that bias can be mitigated using oversampling, significantly improving performance for the underrepresented Black subjects whilst not significantly reducing the majority White subjects' performance. Group DRO also improves performance for Black subjects but not significantly, while reweighing decreases performance for Black subjects. Using a combination of oversampling and Group DRO also improves performance for Black subjects but not significantly. Using cropped images increases performance for both races and reduces the bias, whilst adding oversampling as a bias mitigation technique with cropped images reduces the bias further. 

---
# Deterministic AI Agent Personality Expression through Standard Psychological Diagnostics 

**Authors**: J. M. Diederik Kruijssen, Nicholas Emmons  

**Link**: [PDF](https://arxiv.org/pdf/2503.17085)  

**Abstract**: Artificial intelligence (AI) systems powered by large language models have become increasingly prevalent in modern society, enabling a wide range of applications through natural language interaction. As AI agents proliferate in our daily lives, their generic and uniform expressiveness presents a significant limitation to their appeal and adoption. Personality expression represents a key prerequisite for creating more human-like and distinctive AI systems. We show that AI models can express deterministic and consistent personalities when instructed using established psychological frameworks, with varying degrees of accuracy depending on model capabilities. We find that more advanced models like GPT-4o and o1 demonstrate the highest accuracy in expressing specified personalities across both Big Five and Myers-Briggs assessments, and further analysis suggests that personality expression emerges from a combination of intelligence and reasoning capabilities. Our results reveal that personality expression operates through holistic reasoning rather than question-by-question optimization, with response-scale metrics showing higher variance than test-scale metrics. Furthermore, we find that model fine-tuning affects communication style independently of personality expression accuracy. These findings establish a foundation for creating AI agents with diverse and consistent personalities, which could significantly enhance human-AI interaction across applications from education to healthcare, while additionally enabling a broader range of more unique AI agents. The ability to quantitatively assess and implement personality expression in AI systems opens new avenues for research into more relatable, trustworthy, and ethically designed AI. 

---
# A Thorough Assessment of the Non-IID Data Impact in Federated Learning 

**Authors**: Daniel M. Jimenez-Gutierrez, Mehrdad Hassanzadeh, Aris Anagnostopoulos, Ioannis Chatzigiannakis, Andrea Vitaletti  

**Link**: [PDF](https://arxiv.org/pdf/2503.17070)  

**Abstract**: Federated learning (FL) allows collaborative machine learning (ML) model training among decentralized clients' information, ensuring data privacy. The decentralized nature of FL deals with non-independent and identically distributed (non-IID) data. This open problem has notable consequences, such as decreased model performance and more significant convergence times. Despite its importance, experimental studies systematically addressing all types of data heterogeneity (a.k.a. non-IIDness) remain scarce. We aim to fill this gap by assessing and quantifying the non-IID effect through a thorough empirical analysis. We use the Hellinger Distance (HD) to measure differences in distribution among clients. Our study benchmarks four state-of-the-art strategies for handling non-IID data, including label, feature, quantity, and spatiotemporal skewness, under realistic and controlled conditions. This is the first comprehensive analysis of the spatiotemporal skew effect in FL. Our findings highlight the significant impact of label and spatiotemporal skew non-IID types on FL model performance, with notable performance drops occurring at specific HD thresholds. Additionally, the FL performance is heavily affected mainly when the non-IIDness is extreme. Thus, we provide recommendations for FL research to tackle data heterogeneity effectively. Our work represents the most extensive examination of non-IIDness in FL, offering a robust foundation for future research. 

---
# PVChat: Personalized Video Chat with One-Shot Learning 

**Authors**: Yufei Shi, Weilong Yan, Gang Xu, Yumeng Li, Yuchen Li, Zhenxi Li, Fei Richard Yu, Ming Li, Si Yong Yeo  

**Link**: [PDF](https://arxiv.org/pdf/2503.17069)  

**Abstract**: Video large language models (ViLLMs) excel in general video understanding, e.g., recognizing activities like talking and eating, but struggle with identity-aware comprehension, such as "Wilson is receiving chemotherapy" or "Tom is discussing with Sarah", limiting their applicability in smart healthcare and smart home environments. To address this limitation, we propose a one-shot learning framework PVChat, the first personalized ViLLM that enables subject-aware question answering (QA) from a single video for each subject. Our approach optimizes a Mixture-of-Heads (MoH) enhanced ViLLM on a synthetically augmented video-QA dataset, leveraging a progressive image-to-video learning strategy. Specifically, we introduce an automated augmentation pipeline that synthesizes identity-preserving positive samples and retrieves hard negatives from existing video corpora, generating a diverse training dataset with four QA types: existence, appearance, action, and location inquiries. To enhance subject-specific learning, we propose a ReLU Routing MoH attention mechanism, alongside two novel objectives: (1) Smooth Proximity Regularization for progressive learning through exponential distance scaling and (2) Head Activation Enhancement for balanced attention routing. Finally, we adopt a two-stage training strategy, transitioning from image pre-training to video fine-tuning, enabling a gradual learning process from static attributes to dynamic representations. We evaluate PVChat on diverse datasets covering medical scenarios, TV series, anime, and real-world footage, demonstrating its superiority in personalized feature understanding after learning from a single video, compared to state-of-the-art ViLLMs. 

---
# Replay4NCL: An Efficient Memory Replay-based Methodology for Neuromorphic Continual Learning in Embedded AI Systems 

**Authors**: Mishal Fatima Minhas, Rachmad Vidya Wicaksana Putra, Falah Awwad, Osman Hasan, Muhammad Shafique  

**Link**: [PDF](https://arxiv.org/pdf/2503.17061)  

**Abstract**: Neuromorphic Continual Learning (NCL) paradigm leverages Spiking Neural Networks (SNNs) to enable continual learning (CL) capabilities for AI systems to adapt to dynamically changing environments. Currently, the state-of-the-art employ a memory replay-based method to maintain the old knowledge. However, this technique relies on long timesteps and compression-decompression steps, thereby incurring significant latency and energy overheads, which are not suitable for tightly-constrained embedded AI systems (e.g., mobile agents/robotics). To address this, we propose Replay4NCL, a novel efficient memory replay-based methodology for enabling NCL in embedded AI systems. Specifically, Replay4NCL compresses the latent data (old knowledge), then replays them during the NCL training phase with small timesteps, to minimize the processing latency and energy consumption. To compensate the information loss from reduced spikes, we adjust the neuron threshold potential and learning rate settings. Experimental results on the class-incremental scenario with the Spiking Heidelberg Digits (SHD) dataset show that Replay4NCL can preserve old knowledge with Top-1 accuracy of 90.43% compared to 86.22% from the state-of-the-art, while effectively learning new tasks, achieving 4.88x latency speed-up, 20% latent memory saving, and 36.43% energy saving. These results highlight the potential of our Replay4NCL methodology to further advances NCL capabilities for embedded AI systems. 

---
# Data-Driven Optimization of EV Charging Station Placement Using Causal Discovery 

**Authors**: Julius Stephan Junker, Rong Hu, Ziyue Li, Wolfgang Ketter  

**Link**: [PDF](https://arxiv.org/pdf/2503.17055)  

**Abstract**: This paper addresses the critical challenge of optimizing electric vehicle charging station placement through a novel data-driven methodology employing causal discovery techniques. While traditional approaches prioritize economic factors or power grid constraints, they often neglect empirical charging patterns that ultimately determine station utilization. We analyze extensive charging data from Palo Alto and Boulder (337,344 events across 100 stations) to uncover latent relationships between station characteristics and utilization. Applying structural learning algorithms (NOTEARS and DAGMA) to this data reveals that charging demand is primarily determined by three factors: proximity to amenities, EV registration density, and adjacency to high-traffic routes. These findings, consistent across multiple algorithms and urban contexts, challenge conventional infrastructure distribution strategies. We develop an optimization framework that translates these insights into actionable placement recommendations, identifying locations likely to experience high utilization based on the discovered dependency structures. The resulting site selection model prioritizes strategic clustering in high-amenity areas with substantial EV populations rather than uniform spatial distribution. Our approach contributes a framework that integrates empirical charging behavior into infrastructure planning, potentially enhancing both station utilization and user convenience. By focusing on data-driven insights instead of theoretical distribution models, we provide a more effective strategy for expanding charging networks that can adjust to various stages of EV market development. 

---
# HAPI: A Model for Learning Robot Facial Expressions from Human Preferences 

**Authors**: Dongsheng Yang, Qianying Liu, Wataru Sato, Takashi Minato, Chaoran Liu, Shin'ya Nishida  

**Link**: [PDF](https://arxiv.org/pdf/2503.17046)  

**Abstract**: Automatic robotic facial expression generation is crucial for human-robot interaction, as handcrafted methods based on fixed joint configurations often yield rigid and unnatural behaviors. Although recent automated techniques reduce the need for manual tuning, they tend to fall short by not adequately bridging the gap between human preferences and model predictions-resulting in a deficiency of nuanced and realistic expressions due to limited degrees of freedom and insufficient perceptual integration. In this work, we propose a novel learning-to-rank framework that leverages human feedback to address this discrepancy and enhanced the expressiveness of robotic faces. Specifically, we conduct pairwise comparison annotations to collect human preference data and develop the Human Affective Pairwise Impressions (HAPI) model, a Siamese RankNet-based approach that refines expression evaluation. Results obtained via Bayesian Optimization and online expression survey on a 35-DOF android platform demonstrate that our approach produces significantly more realistic and socially resonant expressions of Anger, Happiness, and Surprise than those generated by baseline and expert-designed methods. This confirms that our framework effectively bridges the gap between human preferences and model predictions while robustly aligning robotic expression generation with human affective responses. 

---
# Summarization Metrics for Spanish and Basque: Do Automatic Scores and LLM-Judges Correlate with Humans? 

**Authors**: Jeremy Barnes, Naiara Perez, Alba Bonet-Jover, Begoña Altuna  

**Link**: [PDF](https://arxiv.org/pdf/2503.17039)  

**Abstract**: Studies on evaluation metrics and LLM-as-a-Judge models for automatic text summarization have largely been focused on English, limiting our understanding of their effectiveness in other languages. Through our new dataset BASSE (BAsque and Spanish Summarization Evaluation), we address this situation by collecting human judgments on 2,040 abstractive summaries in Basque and Spanish, generated either manually or by five LLMs with four different prompts. For each summary, annotators evaluated five criteria on a 5-point Likert scale: coherence, consistency, fluency, relevance, and 5W1H. We use these data to reevaluate traditional automatic metrics used for evaluating summaries, as well as several LLM-as-a-Judge models that show strong performance on this task in English. Our results show that currently proprietary judge LLMs have the highest correlation with human judgments, followed by criteria-specific automatic metrics, while open-sourced judge LLMs perform poorly. We release BASSE and our code publicly, along with the first large-scale Basque summarization dataset containing 22,525 news articles with their subheads. 

---
# An Attentive Representative Sample Selection Strategy Combined with Balanced Batch Training for Skin Lesion Segmentation 

**Authors**: Stephen Lloyd-Brown, Susan Francis, Caroline Hoad, Penny Gowland, Karen Mullinger, Andrew French, Xin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.17034)  

**Abstract**: An often overlooked problem in medical image segmentation research is the effective selection of training subsets to annotate from a complete set of unlabelled data. Many studies select their training sets at random, which may lead to suboptimal model performance, especially in the minimal supervision setting where each training image has a profound effect on performance outcomes. This work aims to address this issue. We use prototypical contrasting learning and clustering to extract representative and diverse samples for annotation. We improve upon prior works with a bespoke cluster-based image selection process. Additionally, we introduce the concept of unsupervised balanced batch dataloading to medical image segmentation, which aims to improve model learning with minimally annotated data. We evaluated our method on a public skin lesion dataset (ISIC 2018) and compared it to another state-of-the-art data sampling method. Our method achieved superior performance in a low annotation budget scenario. 

---
# Exploring the Efficacy of Partial Denoising Using Bit Plane Slicing for Enhanced Fracture Identification: A Comparative Study of Deep Learning-Based Approaches and Handcrafted Feature Extraction Techniques 

**Authors**: Snigdha Paul, Sambit Mallick, Anindya Sen  

**Link**: [PDF](https://arxiv.org/pdf/2503.17030)  

**Abstract**: Computer vision has transformed medical diagnosis, treatment, and research through advanced image processing and machine learning techniques. Fracture classification, a critical area in healthcare, has greatly benefited from these advancements, yet accurate detection is challenged by complex patterns and image noise. Bit plane slicing enhances medical images by reducing noise interference and extracting informative features. This research explores partial denoising techniques to provide practical solutions for improved fracture analysis, ultimately enhancing patient care. The study explores deep learning model DenseNet and handcrafted feature extraction. Decision Tree and Random Forest, were employed to train and evaluate distinct image representations. These include the original image, the concatenation of the four bit planes from the LSB as well as MSB, the fully denoised image, and an image consisting of 6 bit planes from MSB and 2 denoised bit planes from LSB. The purpose of forming these diverse image representations is to analyze SNR as well as classification accuracy and identify the bit planes that contain the most informative features. Moreover, the study delves into the significance of partial denoising techniques in preserving crucial features, leading to improvements in classification results. Notably, this study shows that employing the Random Forest classifier, the partially denoised image representation exhibited a testing accuracy of 95.61% surpassing the performance of other image representations. The outcomes of this research provide valuable insights into the development of efficient preprocessing, feature extraction and classification approaches for fracture identification. By enhancing diagnostic accuracy, these advancements hold the potential to positively impact patient care and overall medical outcomes. 

---
# Symbolic Audio Classification via Modal Decision Tree Learning 

**Authors**: Enrico Marzano, Giovanni Pagliarini, Riccardo Pasini, Guido Sciavicco, Ionel Eduard Stan  

**Link**: [PDF](https://arxiv.org/pdf/2503.17018)  

**Abstract**: The range of potential applications of acoustic analysis is wide. Classification of sounds, in particular, is a typical machine learning task that received a lot of attention in recent years. The most common approaches to sound classification are sub-symbolic, typically based on neural networks, and result in black-box models with high performances but very low transparency. In this work, we consider several audio tasks, namely, age and gender recognition, emotion classification, and respiratory disease diagnosis, and we approach them with a symbolic technique, that is, (modal) decision tree learning. We prove that such tasks can be solved using the same symbolic pipeline, that allows to extract simple rules with very high accuracy and low complexity. In principle, all such tasks could be associated to an autonomous conversation system, which could be useful in different contexts, such as an automatic reservation agent for an hospital or a clinic. 

---
# Developing Critical Thinking in Second Language Learners: Exploring Generative AI like ChatGPT as a Tool for Argumentative Essay Writing 

**Authors**: Simon Suh, Jihyuk Bang, Ji Woo Han  

**Link**: [PDF](https://arxiv.org/pdf/2503.17013)  

**Abstract**: This study employs the Paul-Elder Critical Thinking Model and Tan's argumentative writing framework to create a structured methodology. This methodology, ChatGPT Guideline for Critical Argumentative Writing (CGCAW) framework, integrates the models with ChatGPT's capabilities to guide L2 learners in utilizing ChatGPT to enhance their critical thinking skills. A quantitative experiment was conducted with 10 participants from a state university, divided into experimental and control groups. The experimental group utilized the CGCAW framework, while the control group used ChatGPT without specific guidelines. Participants wrote an argumentative essay within a 40-minute timeframe, and essays were evaluated by three assessors: ChatGPT, Grammarly, and a course instructor. Results indicated that the experimental group showed improvements in clarity, logical coherence, and use of evidence, demonstrating ChatGPT's potential to enhance specific aspects of argumentative writing. However, the control group performed better in overall language mechanics and articulation of main arguments, indicating areas where the CGCAW framework could be further refined. This study highlights the need for further research to optimize the use of AI tools like ChatGPT in L2 learning environments to enhance critical thinking and writing skills. 

---
# Targetless 6DoF Calibration of LiDAR and 2D Scanning Radar Based on Cylindrical Occupancy 

**Authors**: Weimin Wang, Yu Du, Ting Yang, Yu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.17002)  

**Abstract**: Owing to the capability for reliable and all-weather long-range sensing, the fusion of LiDAR and Radar has been widely applied to autonomous vehicles for robust perception. In practical operation, well manually calibrated extrinsic parameters, which are crucial for the fusion of multi-modal sensors, may drift due to the vibration. To address this issue, we present a novel targetless calibration approach, termed LiRaCo, for the extrinsic 6DoF calibration of LiDAR and Radar sensors. Although both types of sensors can obtain geometric information, bridging the geometric correspondences between multi-modal data without any clues of explicit artificial markers is nontrivial, mainly due to the low vertical resolution of scanning Radar. To achieve the targetless calibration, LiRaCo leverages a spatial occupancy consistency between LiDAR point clouds and Radar scans in a common cylindrical representation, considering the increasing data sparsity with distance for both sensors. Specifically, LiRaCo expands the valid Radar scanned pixels into 3D occupancy grids to constrain LiDAR point clouds based on spatial consistency. Consequently, a cost function involving extrinsic calibration parameters is formulated based on the spatial overlap of 3D grids and LiDAR points. Extrinsic parameters are finally estimated by optimizing the cost function. Comprehensive quantitative and qualitative experiments on two real outdoor datasets with different LiDAR sensors demonstrate the feasibility and accuracy of the proposed method. The source code will be publicly available. 

---
# Enabling Versatile Controls for Video Diffusion Models 

**Authors**: Xu Zhang, Hao Zhou, Haoming Qin, Xiaobin Lu, Jiaxing Yan, Guanzhong Wang, Zeyu Chen, Yi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.16983)  

**Abstract**: Despite substantial progress in text-to-video generation, achieving precise and flexible control over fine-grained spatiotemporal attributes remains a significant unresolved challenge in video generation research. To address these limitations, we introduce VCtrl (also termed PP-VCtrl), a novel framework designed to enable fine-grained control over pre-trained video diffusion models in a unified manner. VCtrl integrates diverse user-specified control signals-such as Canny edges, segmentation masks, and human keypoints-into pretrained video diffusion models via a generalizable conditional module capable of uniformly encoding multiple types of auxiliary signals without modifying the underlying generator. Additionally, we design a unified control signal encoding pipeline and a sparse residual connection mechanism to efficiently incorporate control representations. Comprehensive experiments and human evaluations demonstrate that VCtrl effectively enhances controllability and generation quality. The source code and pre-trained models are publicly available and implemented using the PaddlePaddle framework at this http URL. 

---
# Token Dynamics: Towards Efficient and Dynamic Video Token Representation for Video Large Language Models 

**Authors**: Haichao Zhang, Zhuowei Li, Dimitris Metaxas, Yun Fu  

**Link**: [PDF](https://arxiv.org/pdf/2503.16980)  

**Abstract**: Token-based video representation has emerged as a promising approach for enabling large language models to interpret video content. However, existing token reduction techniques, such as token pruning and token merging, often disrupt essential spatial-temporal positional embeddings, failing to adequately balance computational efficiency with fewer tokens. Consequently, these methods result in relatively lengthy token sequences, limiting their applicability in scenarios requiring extreme token compression, such as video large language models. In this paper, we introduce the novel task of extreme short token reduction, aiming to represent extensive video sequences with a minimal number of tokens. To address this challenge, we propose Token Dynamics, a new video representation framework that dynamically reduces token count while preserving spatial-temporal coherence. Specifically, we disentangle video representations by separating visual embeddings from grid-level motion information, structuring them into: 1. a concise token base, created by clustering tokens that describe object-level content; 2. a token dynamics map, capturing detailed spatial-temporal motion patterns across grids. Furthermore, we introduce a cross-dynamics attention mechanism that integrates motion features into the token base without increasing token length, thereby maintaining compactness and spatial-temporal integrity. The experiments demonstrate a reduction of token count to merely 0.07% of the original tokens, with only a minor performance drop of 1.13%. Additionally, we propose two novel subtasks within extreme token reduction (fixed-length and adaptive-length compression), both effectively representing long token sequences for video-language tasks. Our method offers significantly lower theoretical complexity, fewer tokens, and enhanced throughput, thus providing an efficient solution for video LLMs. 

---
# GeoT: Geometry-guided Instance-dependent Transition Matrix for Semi-supervised Tooth Point Cloud Segmentation 

**Authors**: Weihao Yu, Xiaoqing Guo, Chenxin Li, Yifan Liu, Yixuan Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2503.16976)  

**Abstract**: Achieving meticulous segmentation of tooth point clouds from intra-oral scans stands as an indispensable prerequisite for various orthodontic applications. Given the labor-intensive nature of dental annotation, a significant amount of data remains unlabeled, driving increasing interest in semi-supervised approaches. One primary challenge of existing semi-supervised medical segmentation methods lies in noisy pseudo labels generated for unlabeled data. To address this challenge, we propose GeoT, the first framework that employs instance-dependent transition matrix (IDTM) to explicitly model noise in pseudo labels for semi-supervised dental segmentation. Specifically, to handle the extensive solution space of IDTM arising from tens of thousands of dental points, we introduce tooth geometric priors through two key components: point-level geometric regularization (PLGR) to enhance consistency between point adjacency relationships in 3D and IDTM spaces, and class-level geometric smoothing (CLGS) to leverage the fixed spatial distribution of tooth categories for optimal IDTM estimation. Extensive experiments performed on the public Teeth3DS dataset and private dataset demonstrate that our method can make full utilization of unlabeled data to facilitate segmentation, achieving performance comparable to fully supervised methods with only $20\%$ of the labeled data. 

---
# Assessing Consistency and Reproducibility in the Outputs of Large Language Models: Evidence Across Diverse Finance and Accounting Tasks 

**Authors**: Julian Junyan Wang, Victor Xiaoqi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.16974)  

**Abstract**: This study provides the first comprehensive assessment of consistency and reproducibility in Large Language Model (LLM) outputs in finance and accounting research. We evaluate how consistently LLMs produce outputs given identical inputs through extensive experimentation with 50 independent runs across five common tasks: classification, sentiment analysis, summarization, text generation, and prediction. Using three OpenAI models (GPT-3.5-turbo, GPT-4o-mini, and GPT-4o), we generate over 3.4 million outputs from diverse financial source texts and data, covering MD&As, FOMC statements, finance news articles, earnings call transcripts, and financial statements. Our findings reveal substantial but task-dependent consistency, with binary classification and sentiment analysis achieving near-perfect reproducibility, while complex tasks show greater variability. More advanced models do not consistently demonstrate better consistency and reproducibility, with task-specific patterns emerging. LLMs significantly outperform expert human annotators in consistency and maintain high agreement even where human experts significantly disagree. We further find that simple aggregation strategies across 3-5 runs dramatically improve consistency. Simulation analysis reveals that despite measurable inconsistency in LLM outputs, downstream statistical inferences remain remarkably robust. These findings address concerns about what we term "G-hacking," the selective reporting of favorable outcomes from multiple Generative AI runs, by demonstrating that such risks are relatively low for finance and accounting tasks. 

---
# ARFlow: Human Action-Reaction Flow Matching with Physical Guidance 

**Authors**: Wentao Jiang, Jingya Wang, Haotao Lu, Kaiyang Ji, Baoxiong Jia, Siyuan Huang, Ye Shi  

**Link**: [PDF](https://arxiv.org/pdf/2503.16973)  

**Abstract**: Human action-reaction synthesis, a fundamental challenge in modeling causal human interactions, plays a critical role in applications ranging from virtual reality to social robotics. While diffusion-based models have demonstrated promising performance, they exhibit two key limitations for interaction synthesis: reliance on complex noise-to-reaction generators with intricate conditional mechanisms, and frequent physical violations in generated motions. To address these issues, we propose Action-Reaction Flow Matching (ARFlow), a novel framework that establishes direct action-to-reaction mappings, eliminating the need for complex conditional mechanisms. Our approach introduces two key innovations: an x1-prediction method that directly outputs human motions instead of velocity fields, enabling explicit constraint enforcement; and a training-free, gradient-based physical guidance mechanism that effectively prevents body penetration artifacts during sampling. Extensive experiments on NTU120 and Chi3D datasets demonstrate that ARFlow not only outperforms existing methods in terms of Fréchet Inception Distance and motion diversity but also significantly reduces body collisions, as measured by our new Intersection Volume and Intersection Frequency metrics. 

---
# From Faces to Voices: Learning Hierarchical Representations for High-quality Video-to-Speech 

**Authors**: Ji-Hoon Kim, Jeongsoo Choi, Jaehun Kim, Chaeyoung Jung, Joon Son Chung  

**Link**: [PDF](https://arxiv.org/pdf/2503.16956)  

**Abstract**: The objective of this study is to generate high-quality speech from silent talking face videos, a task also known as video-to-speech synthesis. A significant challenge in video-to-speech synthesis lies in the substantial modality gap between silent video and multi-faceted speech. In this paper, we propose a novel video-to-speech system that effectively bridges this modality gap, significantly enhancing the quality of synthesized speech. This is achieved by learning of hierarchical representations from video to speech. Specifically, we gradually transform silent video into acoustic feature spaces through three sequential stages -- content, timbre, and prosody modeling. In each stage, we align visual factors -- lip movements, face identity, and facial expressions -- with corresponding acoustic counterparts to ensure the seamless transformation. Additionally, to generate realistic and coherent speech from the visual representations, we employ a flow matching model that estimates direct trajectories from a simple prior distribution to the target speech distribution. Extensive experiments demonstrate that our method achieves exceptional generation quality comparable to real utterances, outperforming existing methods by a significant margin. 

---
# On-Sensor Convolutional Neural Networks with Early-Exits 

**Authors**: Hazem Hesham Yousef Shalby, Arianna De Vecchi, Alice Scandelli, Pietro Bartoli, Diana Trojaniello, Manuel Roveri, Federica Villa  

**Link**: [PDF](https://arxiv.org/pdf/2503.16939)  

**Abstract**: Tiny Machine Learning (TinyML) is a novel research field aiming at integrating Machine Learning (ML) within embedded devices with limited memory, computation, and energy. Recently, a new branch of TinyML has emerged, focusing on integrating ML directly into the sensors to further reduce the power consumption of embedded devices. Interestingly, despite their state-of-the-art performance in many tasks, none of the current solutions in the literature aims to optimize the implementation of Convolutional Neural Networks (CNNs) operating directly into sensors. In this paper, we introduce for the first time in the literature the optimized design and implementation of Depth-First CNNs operating on the Intelligent Sensor Processing Unit (ISPU) within an Inertial Measurement Unit (IMU) by STMicroelectronics. Our approach partitions the CNN between the ISPU and the microcontroller (MCU) and employs an Early-Exit mechanism to stop the computations on the IMU when enough confidence about the results is achieved, hence significantly reducing power consumption. When using a NUCLEO-F411RE board, this solution achieved an average current consumption of 4.8 mA, marking an 11% reduction compared to the regular inference pipeline on the MCU, while having equal accuracy. 

---
# Rude Humans and Vengeful Robots: Examining Human Perceptions of Robot Retaliatory Intentions in Professional Settings 

**Authors**: Kate Letheren, Nicole Robinson  

**Link**: [PDF](https://arxiv.org/pdf/2503.16932)  

**Abstract**: Humans and robots are increasingly working in personal and professional settings. In workplace settings, humans and robots may work together as colleagues, potentially leading to social expectations, or violation thereof. Extant research has primarily sought to understand social interactions and expectations in personal rather than professional settings, and none of these studies have examined negative outcomes arising from violations of social expectations. This paper reports the results of a 2x3 online experiment that used a unique first-person perspective video to immerse participants in a collaborative workplace setting. The results are nuanced and reveal that while robots are expected to act in accordance with social expectations despite human behavior, there are benefits for robots perceived as being the bigger person in the face of human rudeness. Theoretical and practical implications are provided which discuss the import of these findings for the design of social robots. 

---
# TEMPO: Temporal Preference Optimization of Video LLMs via Difficulty Scheduling and Pre-SFT Alignment 

**Authors**: Shicheng Li, Lei Li, Kun Ouyang, Shuhuai Ren, Yuanxin Liu, Yuanxing Zhang, Fuzheng Zhang, Lingpeng Kong, Qi Liu, Xu Sun  

**Link**: [PDF](https://arxiv.org/pdf/2503.16929)  

**Abstract**: Video Large Language Models (Video LLMs) have achieved significant success by leveraging a two-stage paradigm: pretraining on large-scale video-text data for vision-language alignment, followed by supervised fine-tuning (SFT) for task-specific capabilities. However, existing approaches struggle with temporal reasoning due to weak temporal correspondence in the data and reliance on the next-token prediction paradigm during training. To address these limitations, we propose TEMPO (TEMporal Preference Optimization), a systematic framework that enhances Video LLMs' temporal reasoning capabilities through Direct Preference Optimization (DPO). To facilitate this, we introduce an automated preference data generation pipeline that systematically constructs preference pairs by selecting videos that are rich in temporal information, designing video-specific perturbation strategies, and finally evaluating model responses on clean and perturbed video inputs. Our temporal alignment features two key innovations: curriculum learning which that progressively increases perturbation difficulty to improve model robustness and adaptability; and ``Pre-SFT Alignment'', applying preference optimization before instruction tuning to prioritize fine-grained temporal comprehension. Extensive experiments demonstrate that our approach consistently improves Video LLM performance across multiple benchmarks with a relatively small set of self-generated DPO data. We further analyze the transferability of DPO data across architectures and the role of difficulty scheduling in optimization. Our findings highlight our TEMPO as a scalable and efficient complement to SFT-based methods, paving the way for developing reliable Video LLMs. 

---
# RustEvo^2: An Evolving Benchmark for API Evolution in LLM-based Rust Code Generation 

**Authors**: Linxi Liang, Jing Gong, Mingwei Liu, Chong Wang, Guangsheng Ou, Yanlin Wang, Xin Peng, Zibin Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2503.16922)  

**Abstract**: Large Language Models (LLMs) have become pivotal tools for automating code generation in software development. However, these models face significant challenges in producing version-aware code for rapidly evolving languages like Rust, where frequent Application Programming Interfaces (API) changes across versions lead to compatibility issues and correctness errors. Existing benchmarks lack systematic evaluation of how models navigate API transitions, relying on labor-intensive manual curation and offering limited version-specific insights. To address this gap, we present RustEvo, a novel framework for constructing dynamic benchmarks that evaluate the ability of LLMs to adapt to evolving Rust APIs. RustEvo automates dataset creation by synthesizing 588 API changes (380 from Rust standard libraries, 208 from 15 third-party crates) into programming tasks mirroring real-world challenges. These tasks cover four API evolution categories: Stabilizations, Signature Changes, Behavioral Changes, and Deprecations, reflecting their actual distribution in the Rust ecosystem.
Experiments on state-of-the-art (SOTA) LLMs reveal significant performance variations: models achieve a 65.8% average success rate on stabilized APIs but only 38.0% on behavioral changes, highlighting difficulties in detecting semantic shifts without signature alterations. Knowledge cutoff dates strongly influence performance, with models scoring 56.1% on before-cutoff APIs versus 32.5% on after-cutoff tasks. Retrieval-Augmented Generation (RAG) mitigates this gap, improving success rates by 13.5% on average for APIs released after model training. Our findings underscore the necessity of our evolution-aware benchmarks to advance the adaptability of LLMs in fast-paced software ecosystems. The framework and the benchmarks are publicly released at this https URL. 

---
# When Preferences Diverge: Aligning Diffusion Models with Minority-Aware Adaptive DPO 

**Authors**: Lingfan Zhang, Chen Liu, Chengming Xu, Kai Hu, Donghao Luo, Chengjie Wang, Yanwei Fu, Yuan Yao  

**Link**: [PDF](https://arxiv.org/pdf/2503.16921)  

**Abstract**: In recent years, the field of image generation has witnessed significant advancements, particularly in fine-tuning methods that align models with universal human preferences. This paper explores the critical role of preference data in the training process of diffusion models, particularly in the context of Diffusion-DPO and its subsequent adaptations. We investigate the complexities surrounding universal human preferences in image generation, highlighting the subjective nature of these preferences and the challenges posed by minority samples in preference datasets. Through pilot experiments, we demonstrate the existence of minority samples and their detrimental effects on model performance. We propose Adaptive-DPO -- a novel approach that incorporates a minority-instance-aware metric into the DPO objective. This metric, which includes intra-annotator confidence and inter-annotator stability, distinguishes between majority and minority samples. We introduce an Adaptive-DPO loss function which improves the DPO loss in two ways: enhancing the model's learning of majority labels while mitigating the negative impact of minority samples. Our experiments demonstrate that this method effectively handles both synthetic minority data and real-world preference data, paving the way for more effective training methodologies in image generation tasks. 

---
# Deep Learning for Human Locomotion Analysis in Lower-Limb Exoskeletons: A Comparative Study 

**Authors**: Omar Coser, Christian Tamantini, Matteo Tortora, Leonardo Furia, Rosa Sicilia, Loredana Zollo, Paolo Soda  

**Link**: [PDF](https://arxiv.org/pdf/2503.16904)  

**Abstract**: Wearable robotics for lower-limb assistance have become a pivotal area of research, aiming to enhance mobility for individuals with physical impairments or augment the performance of able-bodied users. Accurate and adaptive control systems are essential to ensure seamless interaction between the wearer and the robotic device, particularly when navigating diverse and dynamic terrains. Despite the recent advances in neural networks for time series analysis, no attempts have been directed towards the classification of ground conditions, categorized into five classes and subsequently determining the ramp's slope and stair's height. In this respect, this paper presents an experimental comparison between eight deep neural network backbones to predict high-level locomotion parameters across diverse terrains.
All the models are trained on the publicly available CAMARGO 2021 dataset. IMU-only data equally or outperformed IMU+EMG inputs, promoting a cost-effective and efficient design. Indeeds, using three IMU sensors, the LSTM achieved high terrain classification accuracy (0.94 +- 0.04) and precise ramp slope (1.95 +- 0.58°) and the CNN-LSTM a stair height (15.65 +- 7.40 mm) estimations. As a further contribution, SHAP analysis justified sensor reduction without performance loss, ensuring a lightweight setup. The system operates with ~2 ms inference time, supporting real-time applications. The code is code available at this https URL. 

---
# Classifier-guided CLIP Distillation for Unsupervised Multi-label Classification 

**Authors**: Dongseob Kim, Hyunjung Shim  

**Link**: [PDF](https://arxiv.org/pdf/2503.16873)  

**Abstract**: Multi-label classification is crucial for comprehensive image understanding, yet acquiring accurate annotations is challenging and costly. To address this, a recent study suggests exploiting unsupervised multi-label classification leveraging CLIP, a powerful vision-language model. Despite CLIP's proficiency, it suffers from view-dependent predictions and inherent bias, limiting its effectiveness. We propose a novel method that addresses these issues by leveraging multiple views near target objects, guided by Class Activation Mapping (CAM) of the classifier, and debiasing pseudo-labels derived from CLIP predictions. Our Classifier-guided CLIP Distillation (CCD) enables selecting multiple local views without extra labels and debiasing predictions to enhance classification performance. Experimental results validate our method's superiority over existing techniques across diverse datasets. The code is available at this https URL. 

---
# Sparse Logit Sampling: Accelerating Knowledge Distillation in LLMs 

**Authors**: Anshumann, Mohd Abbas Zaidi, Akhil Kedia, Jinwoo Ahn, Taehwak Kwon, Kangwook Lee, Haejun Lee, Joohyung Lee  

**Link**: [PDF](https://arxiv.org/pdf/2503.16870)  

**Abstract**: Knowledge distillation can be a cost-effective technique to distill knowledge in Large Language Models, if the teacher output logits can be pre-computed and cached. However, successfully applying this to pre-training remains largely unexplored. In this work, we prove that naive approaches for sparse knowledge distillation such as caching Top-K probabilities, while intuitive, provide biased estimates of teacher probability distribution to the student, resulting in suboptimal performance and calibration. We propose an importance-sampling-based method `Random Sampling Knowledge Distillation', which provides unbiased estimates, preserves the gradient in expectation, and requires storing significantly sparser logits. Our method enables faster training of student models with marginal overhead (<10%) compared to cross-entropy based training, while maintaining competitive performance compared to full distillation, across a range of model sizes from 300M to 3B. 

---
# MTBench: A Multimodal Time Series Benchmark for Temporal Reasoning and Question Answering 

**Authors**: Jialin Chen, Aosong Feng, Ziyu Zhao, Juan Garza, Gaukhar Nurbek, Cheng Qin, Ali Maatouk, Leandros Tassiulas, Yifeng Gao, Rex Ying  

**Link**: [PDF](https://arxiv.org/pdf/2503.16858)  

**Abstract**: Understanding the relationship between textual news and time-series evolution is a critical yet under-explored challenge in applied data science. While multimodal learning has gained traction, existing multimodal time-series datasets fall short in evaluating cross-modal reasoning and complex question answering, which are essential for capturing complex interactions between narrative information and temporal patterns. To bridge this gap, we introduce Multimodal Time Series Benchmark (MTBench), a large-scale benchmark designed to evaluate large language models (LLMs) on time series and text understanding across financial and weather domains. MTbench comprises paired time series and textual data, including financial news with corresponding stock price movements and weather reports aligned with historical temperature records. Unlike existing benchmarks that focus on isolated modalities, MTbench provides a comprehensive testbed for models to jointly reason over structured numerical trends and unstructured textual narratives. The richness of MTbench enables formulation of diverse tasks that require a deep understanding of both text and time-series data, including time-series forecasting, semantic and technical trend analysis, and news-driven question answering (QA). These tasks target the model's ability to capture temporal dependencies, extract key insights from textual context, and integrate cross-modal information. We evaluate state-of-the-art LLMs on MTbench, analyzing their effectiveness in modeling the complex relationships between news narratives and temporal patterns. Our findings reveal significant challenges in current models, including difficulties in capturing long-term dependencies, interpreting causality in financial and weather trends, and effectively fusing multimodal information. 

---
# Imagine to Hear: Auditory Knowledge Generation can be an Effective Assistant for Language Models 

**Authors**: Suho Yoo, Hyunjong Ok, Jaeho Lee  

**Link**: [PDF](https://arxiv.org/pdf/2503.16853)  

**Abstract**: Language models pretrained on text-only corpora often struggle with tasks that require auditory commonsense knowledge. Previous work addresses this problem by augmenting the language model to retrieve knowledge from external audio databases. This approach has several limitations, such as the potential lack of relevant audio in databases and the high costs associated with constructing and querying the databases. To address these issues, we propose Imagine to Hear, a novel approach that dynamically generates auditory knowledge using generative models. Our framework detects multiple audio-related textual spans from the given prompt and generates corresponding auditory knowledge. We develop several mechanisms to efficiently process multiple auditory knowledge, including a CLAP-based rejection sampler and a language-audio fusion module. Our experiments show that our method achieves state-of-the-art performance on AuditoryBench without relying on external databases, highlighting the effectiveness of our generation-based approach. 

---
# Casual Inference via Style Bias Deconfounding for Domain Generalization 

**Authors**: Jiaxi Li, Di Lin, Hao Chen, Hongying Liu, Liang Wan, Wei Feng  

**Link**: [PDF](https://arxiv.org/pdf/2503.16852)  

**Abstract**: Deep neural networks (DNNs) often struggle with out-of-distribution data, limiting their reliability in diverse realworld applications. To address this issue, domain generalization methods have been developed to learn domain-invariant features from single or multiple training domains, enabling generalization to unseen testing domains. However, existing approaches usually overlook the impact of style frequency within the training set. This oversight predisposes models to capture spurious visual correlations caused by style confounding factors, rather than learning truly causal representations, thereby undermining inference reliability. In this work, we introduce Style Deconfounding Causal Learning (SDCL), a novel causal inference-based framework designed to explicitly address style as a confounding factor. Our approaches begins with constructing a structural causal model (SCM) tailored to the domain generalization problem and applies a backdoor adjustment strategy to account for style influence. Building on this foundation, we design a style-guided expert module (SGEM) to adaptively clusters style distributions during training, capturing the global confounding style. Additionally, a back-door causal learning module (BDCL) performs causal interventions during feature extraction, ensuring fair integration of global confounding styles into sample predictions, effectively reducing style bias. The SDCL framework is highly versatile and can be seamlessly integrated with state-of-the-art data augmentation techniques. Extensive experiments across diverse natural and medical image recognition tasks validate its efficacy, demonstrating superior performance in both multi-domain and the more challenging single-domain generalization scenarios. 

---
# Physics-Informed Neural Network Surrogate Models for River Stage Prediction 

**Authors**: Maximilian Zoch, Edward Holmberg, Pujan Pokhrel, Ken Pathak, Steven Sloan, Kendall Niles, Jay Ratcliff, Maik Flanagin, Elias Ioup, Christian Guetl, Mahdi Abdelguerfi  

**Link**: [PDF](https://arxiv.org/pdf/2503.16850)  

**Abstract**: This work investigates the feasibility of using Physics-Informed Neural Networks (PINNs) as surrogate models for river stage prediction, aiming to reduce computational cost while maintaining predictive accuracy. Our primary contribution demonstrates that PINNs can successfully approximate HEC-RAS numerical solutions when trained on a single river, achieving strong predictive accuracy with generally low relative errors, though some river segments exhibit higher deviations.
By integrating the governing Saint-Venant equations into the learning process, the proposed PINN-based surrogate model enforces physical consistency and significantly improves computational efficiency compared to HEC-RAS. We evaluate the model's performance in terms of accuracy and computational speed, demonstrating that it closely approximates HEC-RAS predictions while enabling real-time inference.
These results highlight the potential of PINNs as effective surrogate models for single-river hydrodynamics, offering a promising alternative for computationally efficient river stage forecasting. Future work will explore techniques to enhance PINN training stability and robustness across a more generalized multi-river model. 

---
# The Deployment of End-to-End Audio Language Models Should Take into Account the Principle of Least Privilege 

**Authors**: Luxi He, Xiangyu Qi, Michel Liao, Inyoung Cheong, Prateek Mittal, Danqi Chen, Peter Henderson  

**Link**: [PDF](https://arxiv.org/pdf/2503.16833)  

**Abstract**: We are at a turning point for language models that accept audio input. The latest end-to-end audio language models (Audio LMs) process speech directly instead of relying on a separate transcription step. This shift preserves detailed information, such as intonation or the presence of multiple speakers, that would otherwise be lost in transcription. However, it also introduces new safety risks, including the potential misuse of speaker identity cues and other sensitive vocal attributes, which could have legal implications. In this position paper, we urge a closer examination of how these models are built and deployed. We argue that the principle of least privilege should guide decisions on whether to deploy cascaded or end-to-end models. Specifically, evaluations should assess (1) whether end-to-end modeling is necessary for a given application; and (2), the appropriate scope of information access. Finally, We highlight related gaps in current audio LM benchmarks and identify key open research questions, both technical and policy-related, that must be addressed to enable the responsible deployment of end-to-end Audio LMs. 

---
# DyWA: Dynamics-adaptive World Action Model for Generalizable Non-prehensile Manipulation 

**Authors**: Jiangran Lyu, Ziming Li, Xuesong Shi, Chaoyi Xu, Yizhou Wang, He Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.16806)  

**Abstract**: Nonprehensile manipulation is crucial for handling objects that are too thin, large, or otherwise ungraspable in unstructured environments. While conventional planning-based approaches struggle with complex contact modeling, learning-based methods have recently emerged as a promising alternative. However, existing learning-based approaches face two major limitations: they heavily rely on multi-view cameras and precise pose tracking, and they fail to generalize across varying physical conditions, such as changes in object mass and table friction. To address these challenges, we propose the Dynamics-Adaptive World Action Model (DyWA), a novel framework that enhances action learning by jointly predicting future states while adapting to dynamics variations based on historical trajectories. By unifying the modeling of geometry, state, physics, and robot actions, DyWA enables more robust policy learning under partial observability. Compared to baselines, our method improves the success rate by 31.5% using only single-view point cloud observations in the simulation. Furthermore, DyWA achieves an average success rate of 68% in real-world experiments, demonstrating its ability to generalize across diverse object geometries, adapt to varying table friction, and robustness in challenging scenarios such as half-filled water bottles and slippery surfaces. 

---
# Auto-Regressive Diffusion for Generating 3D Human-Object Interactions 

**Authors**: Zichen Geng, Zeeshan Hayder, Wei Liu, Ajmal Saeed Mian  

**Link**: [PDF](https://arxiv.org/pdf/2503.16801)  

**Abstract**: Text-driven Human-Object Interaction (Text-to-HOI) generation is an emerging field with applications in animation, video games, virtual reality, and robotics. A key challenge in HOI generation is maintaining interaction consistency in long sequences. Existing Text-to-Motion-based approaches, such as discrete motion tokenization, cannot be directly applied to HOI generation due to limited data in this domain and the complexity of the modality. To address the problem of interaction consistency in long sequences, we propose an autoregressive diffusion model (ARDHOI) that predicts the next continuous token. Specifically, we introduce a Contrastive Variational Autoencoder (cVAE) to learn a physically plausible space of continuous HOI tokens, thereby ensuring that generated human-object motions are realistic and natural. For generating sequences autoregressively, we develop a Mamba-based context encoder to capture and maintain consistent sequential actions. Additionally, we implement an MLP-based denoiser to generate the subsequent token conditioned on the encoded context. Our model has been evaluated on the OMOMO and BEHAVE datasets, where it outperforms existing state-of-the-art methods in terms of both performance and inference speed. This makes ARDHOI a robust and efficient solution for text-driven HOI tasks 

---
# Causally Aligned Curriculum Learning 

**Authors**: Mingxuan Li, Junzhe Zhang, Elias Bareinboim  

**Link**: [PDF](https://arxiv.org/pdf/2503.16799)  

**Abstract**: A pervasive challenge in Reinforcement Learning (RL) is the "curse of dimensionality" which is the exponential growth in the state-action space when optimizing a high-dimensional target task. The framework of curriculum learning trains the agent in a curriculum composed of a sequence of related and more manageable source tasks. The expectation is that when some optimal decision rules are shared across source tasks and the target task, the agent could more quickly pick up the necessary skills to behave optimally in the environment, thus accelerating the learning process. However, this critical assumption of invariant optimal decision rules does not necessarily hold in many practical applications, specifically when the underlying environment contains unobserved confounders. This paper studies the problem of curriculum RL through causal lenses. We derive a sufficient graphical condition characterizing causally aligned source tasks, i.e., the invariance of optimal decision rules holds. We further develop an efficient algorithm to generate a causally aligned curriculum, provided with qualitative causal knowledge of the target task. Finally, we validate our proposed methodology through experiments in discrete and continuous confounded tasks with pixel observations. 

---
# "The Diagram is like Guardrails": Structuring GenAI-assisted Hypotheses Exploration with an Interactive Shared Representation 

**Authors**: Zijian Ding, Michelle Brachman, Joel Chan, Werner Geyer  

**Link**: [PDF](https://arxiv.org/pdf/2503.16791)  

**Abstract**: Data analysis encompasses a spectrum of tasks, from high-level conceptual reasoning to lower-level execution. While AI-powered tools increasingly support execution tasks, there remains a need for intelligent assistance in conceptual tasks. This paper investigates the design of an ordered node-link tree interface augmented with AI-generated information hints and visualizations, as a potential shared representation for hypothesis exploration. Through a design probe (n=22), participants generated diagrams averaging 21.82 hypotheses. Our findings showed that the node-link diagram acts as "guardrails" for hypothesis exploration, facilitating structured workflows, providing comprehensive overviews, and enabling efficient backtracking. The AI-generated information hints, particularly visualizations, aided users in transforming abstract ideas into data-backed concepts while reducing cognitive load. We further discuss how node-link diagrams can support both parallel exploration and iterative refinement in hypothesis formulation, potentially enhancing the breadth and depth of human-AI collaborative data analysis. 

---
# Learning Part Knowledge to Facilitate Category Understanding for Fine-Grained Generalized Category Discovery 

**Authors**: Enguang Wang, Zhimao Peng, Zhengyuan Xie, Haori Lu, Fei Yang, Xialei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.16782)  

**Abstract**: Generalized Category Discovery (GCD) aims to classify unlabeled data containing both seen and novel categories. Although existing methods perform well on generic datasets, they struggle in fine-grained scenarios. We attribute this difficulty to their reliance on contrastive learning over global image features to automatically capture discriminative cues, which fails to capture the subtle local differences essential for distinguishing fine-grained categories. Therefore, in this paper, we propose incorporating part knowledge to address fine-grained GCD, which introduces two key challenges: the absence of annotations for novel classes complicates the extraction of the part features, and global contrastive learning prioritizes holistic feature invariance, inadvertently suppressing discriminative local part patterns. To address these challenges, we propose PartGCD, including 1) Adaptive Part Decomposition, which automatically extracts class-specific semantic parts via Gaussian Mixture Models, and 2) Part Discrepancy Regularization, enforcing explicit separation between part features to amplify fine-grained local part distinctions.
Experiments demonstrate state-of-the-art performance across multiple fine-grained benchmarks while maintaining competitiveness on generic datasets, validating the effectiveness and robustness of our approach. 

---
# Chain-of-Tools: Utilizing Massive Unseen Tools in the CoT Reasoning of Frozen Language Models 

**Authors**: Mengsong Wu, Tong Zhu, Han Han, Xiang Zhang, Wenbiao Shao, Wenliang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.16779)  

**Abstract**: Tool learning can further broaden the usage scenarios of large language models (LLMs). However most of the existing methods either need to finetune that the model can only use tools seen in the training data, or add tool demonstrations into the prompt with lower efficiency. In this paper, we present a new Tool Learning method Chain-of-Tools. It makes full use of the powerful semantic representation capability of frozen LLMs to finish tool calling in CoT reasoning with a huge and flexible tool pool which may contain unseen tools. Especially, to validate the effectiveness of our approach in the massive unseen tool scenario, we construct a new dataset SimpleToolQuestions. We conduct experiments on two numerical reasoning benchmarks (GSM8K-XL and FuncQA) and two knowledge-based question answering benchmarks (KAMEL and SimpleToolQuestions). Experimental results show that our approach performs better than the baseline. We also identify dimensions of the model output that are critical in tool selection, enhancing the model interpretability. Our code and data are available at: this https URL . 

---
# Dynamic Attention Mechanism in Spatiotemporal Memory Networks for Object Tracking 

**Authors**: Meng Zhou, Jiadong Xie, Mingsheng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2503.16768)  

**Abstract**: Mainstream visual object tracking frameworks predominantly rely on template matching paradigms. Their performance heavily depends on the quality of template features, which becomes increasingly challenging to maintain in complex scenarios involving target deformation, occlusion, and background clutter. While existing spatiotemporal memory-based trackers emphasize memory capacity expansion, they lack effective mechanisms for dynamic feature selection and adaptive fusion. To address this gap, we propose a Dynamic Attention Mechanism in Spatiotemporal Memory Network (DASTM) with two key innovations: 1) A differentiable dynamic attention mechanism that adaptively adjusts channel-spatial attention weights by analyzing spatiotemporal correlations between the templates and memory features; 2) A lightweight gating network that autonomously allocates computational resources based on target motion states, prioritizing high-discriminability features in challenging scenarios. Extensive evaluations on OTB-2015, VOT 2018, LaSOT, and GOT-10K benchmarks demonstrate our DASTM's superiority, achieving state-of-the-art performance in success rate, robustness, and real-time efficiency, thereby offering a novel solution for real-time tracking in complex environments. 

---
# QuartDepth: Post-Training Quantization for Real-Time Depth Estimation on the Edge 

**Authors**: Xuan Shen, Weize Ma, Jing Liu, Changdi Yang, Rui Ding, Quanyi Wang, Henghui Ding, Wei Niu, Yanzhi Wang, Pu Zhao, Jun Lin, Jiuxiang Gu  

**Link**: [PDF](https://arxiv.org/pdf/2503.16709)  

**Abstract**: Monocular Depth Estimation (MDE) has emerged as a pivotal task in computer vision, supporting numerous real-world applications. However, deploying accurate depth estimation models on resource-limited edge devices, especially Application-Specific Integrated Circuits (ASICs), is challenging due to the high computational and memory demands. Recent advancements in foundational depth estimation deliver impressive results but further amplify the difficulty of deployment on ASICs. To address this, we propose QuartDepth which adopts post-training quantization to quantize MDE models with hardware accelerations for ASICs. Our approach involves quantizing both weights and activations to 4-bit precision, reducing the model size and computation cost. To mitigate the performance degradation, we introduce activation polishing and compensation algorithm applied before and after activation quantization, as well as a weight reconstruction method for minimizing errors in weight quantization. Furthermore, we design a flexible and programmable hardware accelerator by supporting kernel fusion and customized instruction programmability, enhancing throughput and efficiency. Experimental results demonstrate that our framework achieves competitive accuracy while enabling fast inference and higher energy efficiency on ASICs, bridging the gap between high-performance depth estimation and practical edge-device applicability. Code: this https URL 

---
# Limits of trust in medical AI 

**Authors**: Joshua Hatherley  

**Link**: [PDF](https://arxiv.org/pdf/2503.16692)  

**Abstract**: Artificial intelligence (AI) is expected to revolutionize the practice of medicine. Recent advancements in the field of deep learning have demonstrated success in a variety of clinical tasks: detecting diabetic retinopathy from images, predicting hospital readmissions, aiding in the discovery of new drugs, etc. AI's progress in medicine, however, has led to concerns regarding the potential effects of this technology upon relationships of trust in clinical practice. In this paper, I will argue that there is merit to these concerns, since AI systems can be relied upon, and are capable of reliability, but cannot be trusted, and are not capable of trustworthiness. Insofar as patients are required to rely upon AI systems for their medical decision-making, there is potential for this to produce a deficit of trust in relationships in clinical practice. 

---
# GAIR: Improving Multimodal Geo-Foundation Model with Geo-Aligned Implicit Representations 

**Authors**: Zeping Liu, Fan Zhang, Junfeng Jiao, Ni Lao, Gengchen Mai  

**Link**: [PDF](https://arxiv.org/pdf/2503.16683)  

**Abstract**: Advancements in vision and language foundation models have inspired the development of geo-foundation models (GeoFMs), enhancing performance across diverse geospatial tasks. However, many existing GeoFMs primarily focus on overhead remote sensing (RS) data while neglecting other data modalities such as ground-level imagery. A key challenge in multimodal GeoFM development is to explicitly model geospatial relationships across modalities, which enables generalizability across tasks, spatial scales, and temporal contexts. To address these limitations, we propose GAIR, a novel multimodal GeoFM architecture integrating overhead RS data, street view (SV) imagery, and their geolocation metadata. We utilize three factorized neural encoders to project an SV image, its geolocation, and an RS image into the embedding space. The SV image needs to be located within the RS image's spatial footprint but does not need to be at its geographic center. In order to geographically align the SV image and RS image, we propose a novel implicit neural representations (INR) module that learns a continuous RS image representation and looks up the RS embedding at the SV image's geolocation. Next, these geographically aligned SV embedding, RS embedding, and location embedding are trained with contrastive learning objectives from unlabeled data. We evaluate GAIR across 10 geospatial tasks spanning RS image-based, SV image-based, and location embedding-based benchmarks. Experimental results demonstrate that GAIR outperforms state-of-the-art GeoFMs and other strong baselines, highlighting its effectiveness in learning generalizable and transferable geospatial representations. 

---
# GauRast: Enhancing GPU Triangle Rasterizers to Accelerate 3D Gaussian Splatting 

**Authors**: Sixu Li, Ben Keller, Yingyan Celine Lin, Brucek Khailany  

**Link**: [PDF](https://arxiv.org/pdf/2503.16681)  

**Abstract**: 3D intelligence leverages rich 3D features and stands as a promising frontier in AI, with 3D rendering fundamental to many downstream applications. 3D Gaussian Splatting (3DGS), an emerging high-quality 3D rendering method, requires significant computation, making real-time execution on existing GPU-equipped edge devices infeasible. Previous efforts to accelerate 3DGS rely on dedicated accelerators that require substantial integration overhead and hardware costs. This work proposes an acceleration strategy that leverages the similarities between the 3DGS pipeline and the highly optimized conventional graphics pipeline in modern GPUs. Instead of developing a dedicated accelerator, we enhance existing GPU rasterizer hardware to efficiently support 3DGS operations. Our results demonstrate a 23$\times$ increase in processing speed and a 24$\times$ reduction in energy consumption, with improvements yielding 6$\times$ faster end-to-end runtime for the original 3DGS algorithm and 4$\times$ for the latest efficiency-improved pipeline, achieving 24 FPS and 46 FPS respectively. These enhancements incur only a minimal area overhead of 0.2\% relative to the entire SoC chip area, underscoring the practicality and efficiency of our approach for enabling 3DGS rendering on resource-constrained platforms. 

---
# Echoes of Power: Investigating Geopolitical Bias in US and China Large Language Models 

**Authors**: Andre G. C. Pacheco, Athus Cavalini, Giovanni Comarela  

**Link**: [PDF](https://arxiv.org/pdf/2503.16679)  

**Abstract**: Large Language Models (LLMs) have emerged as powerful tools for generating human-like text, transforming human-machine interactions. However, their widespread adoption has raised concerns about their potential to influence public opinion and shape political narratives. In this work, we investigate the geopolitical biases in US and Chinese LLMs, focusing on how these models respond to questions related to geopolitics and international relations. We collected responses from ChatGPT and DeepSeek to a set of geopolitical questions and evaluated their outputs through both qualitative and quantitative analyses. Our findings show notable biases in both models, reflecting distinct ideological perspectives and cultural influences. However, despite these biases, for a set of questions, the models' responses are more aligned than expected, indicating that they can address sensitive topics without necessarily presenting directly opposing viewpoints. This study highlights the potential of LLMs to shape public discourse and underscores the importance of critically assessing AI-generated content, particularly in politically sensitive contexts. 

---
# Accelerating Transformer Inference and Training with 2:4 Activation Sparsity 

**Authors**: Daniel Haziza, Timothy Chou, Dhruv Choudhary, Luca Wehrstedt, Francisco Massa, Jiecao Yu, Geonhwa Jeong, Supriya Rao, Patrick Labatut, Jesse Cai  

**Link**: [PDF](https://arxiv.org/pdf/2503.16672)  

**Abstract**: In this paper, we demonstrate how to leverage 2:4 sparsity, a popular hardware-accelerated GPU sparsity pattern, to activations to accelerate large language model training and inference. Crucially we exploit the intrinsic sparsity found in Squared-ReLU activations to provide this acceleration with no accuracy loss. Our approach achieves up to 1.3x faster Feed Forward Network (FFNs) in both the forwards and backwards pass. This work highlights the potential for sparsity to play a key role in accelerating large language model training and inference. 

---
# Aligning Text-to-Music Evaluation with Human Preferences 

**Authors**: Yichen Huang, Zachary Novack, Koichi Saito, Jiatong Shi, Shinji Watanabe, Yuki Mitsufuji, John Thickstun, Chris Donahue  

**Link**: [PDF](https://arxiv.org/pdf/2503.16669)  

**Abstract**: Despite significant recent advances in generative acoustic text-to-music (TTM) modeling, robust evaluation of these models lags behind, relying in particular on the popular Fréchet Audio Distance (FAD). In this work, we rigorously study the design space of reference-based divergence metrics for evaluating TTM models through (1) designing four synthetic meta-evaluations to measure sensitivity to particular musical desiderata, and (2) collecting and evaluating on MusicPrefs, the first open-source dataset of human preferences for TTM systems. We find that not only is the standard FAD setup inconsistent on both synthetic and human preference data, but that nearly all existing metrics fail to effectively capture desiderata, and are only weakly correlated with human perception. We propose a new metric, the MAUVE Audio Divergence (MAD), computed on representations from a self-supervised audio embedding model. We find that this metric effectively captures diverse musical desiderata (average rank correlation 0.84 for MAD vs. 0.49 for FAD and also correlates more strongly with MusicPrefs (0.62 vs. 0.14). 

---
# Code Evolution Graphs: Understanding Large Language Model Driven Design of Algorithms 

**Authors**: Niki van Stein, Anna V. Kononova, Lars Kotthoff, Thomas Bäck  

**Link**: [PDF](https://arxiv.org/pdf/2503.16668)  

**Abstract**: Large Language Models (LLMs) have demonstrated great promise in generating code, especially when used inside an evolutionary computation framework to iteratively optimize the generated algorithms. However, in some cases they fail to generate competitive algorithms or the code optimization stalls, and we are left with no recourse because of a lack of understanding of the generation process and generated codes. We present a novel approach to mitigate this problem by enabling users to analyze the generated codes inside the evolutionary process and how they evolve over repeated prompting of the LLM. We show results for three benchmark problem classes and demonstrate novel insights. In particular, LLMs tend to generate more complex code with repeated prompting, but additional complexity can hurt algorithmic performance in some cases. Different LLMs have different coding ``styles'' and generated code tends to be dissimilar to other LLMs. These two findings suggest that using different LLMs inside the code evolution frameworks might produce higher performing code than using only one LLM. 

---
# MobilePlantViT: A Mobile-friendly Hybrid ViT for Generalized Plant Disease Image Classification 

**Authors**: Moshiur Rahman Tonmoy, Md. Mithun Hossain, Nilanjan Dey, M. F. Mridha  

**Link**: [PDF](https://arxiv.org/pdf/2503.16628)  

**Abstract**: Plant diseases significantly threaten global food security by reducing crop yields and undermining agricultural sustainability. AI-driven automated classification has emerged as a promising solution, with deep learning models demonstrating impressive performance in plant disease identification. However, deploying these models on mobile and edge devices remains challenging due to high computational demands and resource constraints, highlighting the need for lightweight, accurate solutions for accessible smart agriculture systems. To address this, we propose MobilePlantViT, a novel hybrid Vision Transformer (ViT) architecture designed for generalized plant disease classification, which optimizes resource efficiency while maintaining high performance. Extensive experiments across diverse plant disease datasets of varying scales show our model's effectiveness and strong generalizability, achieving test accuracies ranging from 80% to over 99%. Notably, with only 0.69 million parameters, our architecture outperforms the smallest versions of MobileViTv1 and MobileViTv2, despite their higher parameter counts. These results underscore the potential of our approach for real-world, AI-powered automated plant disease classification in sustainable and resource-efficient smart agriculture systems. All codes will be available in the GitHub repository: this https URL 

---
# Classification of User Reports for Detection of Faulty Computer Components using NLP Models: A Case Study 

**Authors**: Maria de Lourdes M. Silva, André L. C. Mendonça, Eduardo R. D. Neto, Iago C. Chaves, Felipe T. Brito, Victor A. E. Farias, Javam C. Machado  

**Link**: [PDF](https://arxiv.org/pdf/2503.16614)  

**Abstract**: Computer manufacturers typically offer platforms for users to report faults. However, there remains a significant gap in these platforms' ability to effectively utilize textual reports, which impedes users from describing their issues in their own words. In this context, Natural Language Processing (NLP) offers a promising solution, by enabling the analysis of user-generated text. This paper presents an innovative approach that employs NLP models to classify user reports for detecting faulty computer components, such as CPU, memory, motherboard, video card, and more. In this work, we build a dataset of 341 user reports obtained from many sources. Additionally, through extensive experimental evaluation, our approach achieved an accuracy of 79% with our dataset. 

---
# A Recipe for Generating 3D Worlds From a Single Image 

**Authors**: Katja Schwarz, Denys Rozumnyi, Samuel Rota Bulò, Lorenzo Porzi, Peter Kontschieder  

**Link**: [PDF](https://arxiv.org/pdf/2503.16611)  

**Abstract**: We introduce a recipe for generating immersive 3D worlds from a single image by framing the task as an in-context learning problem for 2D inpainting models. This approach requires minimal training and uses existing generative models. Our process involves two steps: generating coherent panoramas using a pre-trained diffusion model and lifting these into 3D with a metric depth estimator. We then fill unobserved regions by conditioning the inpainting model on rendered point clouds, requiring minimal fine-tuning. Tested on both synthetic and real images, our method produces high-quality 3D environments suitable for VR display. By explicitly modeling the 3D structure of the generated environment from the start, our approach consistently outperforms state-of-the-art, video synthesis-based methods along multiple quantitative image quality metrics. Project Page: this https URL 

---
# Explainable AI-Guided Efficient Approximate DNN Generation for Multi-Pod Systolic Arrays 

**Authors**: Ayesha Siddique, Khurram Khalil, Khaza Anuarul Hoque  

**Link**: [PDF](https://arxiv.org/pdf/2503.16583)  

**Abstract**: Approximate deep neural networks (AxDNNs) are promising for enhancing energy efficiency in real-world devices. One of the key contributors behind this enhanced energy efficiency in AxDNNs is the use of approximate multipliers. Unfortunately, the simulation of approximate multipliers does not usually scale well on CPUs and GPUs. As a consequence, this slows down the overall simulation of AxDNNs aimed at identifying the appropriate approximate multipliers to achieve high energy efficiency with a minimum accuracy loss. To address this problem, we present a novel XAI-Gen methodology, which leverages the analytical model of the emerging hardware accelerator (e.g., Google TPU v4) and explainable artificial intelligence (XAI) to precisely identify the non-critical layers for approximation and quickly discover the appropriate approximate multipliers for AxDNN layers. Our results show that XAI-Gen achieves up to 7x lower energy consumption with only 1-2% accuracy loss. We also showcase the effectiveness of the XAI-Gen approach through a neural architecture search (XAI-NAS) case study. Interestingly, XAI-NAS achieves 40\% higher energy efficiency with up to 5x less execution time when compared to the state-of-the-art NAS methods for generating AxDNNs. 

---
# Machine Learning-Based Genomic Linguistic Analysis (Gene Sequence Feature Learning): A Case Study on Predicting Heavy Metal Response Genes in Rice 

**Authors**: Ruiqi Yang, Jianxu Wang, Wei Yuan, Xun Wang, Mei Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.16582)  

**Abstract**: This study explores the application of machine learning-based genetic linguistics for identifying heavy metal response genes in rice (Oryza sativa). By integrating convolutional neural networks and random forest algorithms, we developed a hybrid model capable of extracting and learning meaningful features from gene sequences, such as k-mer frequencies and physicochemical properties. The model was trained and tested on datasets of genes, achieving high predictive performance (precision: 0.89, F1-score: 0.82). RNA-seq and qRT-PCR experiments conducted on rice leaves which exposed to Hg0, revealed differential expression of genes associated with heavy metal responses, which validated the model's predictions. Co-expression network analysis identified 103 related genes, and a literature review indicated that these genes are highly likely to be involved in heavy metal-related biological processes. By integrating and comparing the analysis results with those of differentially expressed genes (DEGs), the validity of the new machine learning method was further demonstrated. This study highlights the efficacy of combining machine learning with genetic linguistics for large-scale gene prediction. It demonstrates a cost-effective and efficient approach for uncovering molecular mechanisms underlying heavy metal responses, with potential applications in developing stress-tolerant crop varieties. 

---
# Investigating Retrieval-Augmented Generation in Quranic Studies: A Study of 13 Open-Source Large Language Models 

**Authors**: Zahra Khalila, Arbi Haza Nasution, Winda Monika, Aytug Onan, Yohei Murakami, Yasir Bin Ismail Radi, Noor Mohammad Osmani  

**Link**: [PDF](https://arxiv.org/pdf/2503.16581)  

**Abstract**: Accurate and contextually faithful responses are critical when applying large language models (LLMs) to sensitive and domain-specific tasks, such as answering queries related to quranic studies. General-purpose LLMs often struggle with hallucinations, where generated responses deviate from authoritative sources, raising concerns about their reliability in religious contexts. This challenge highlights the need for systems that can integrate domain-specific knowledge while maintaining response accuracy, relevance, and faithfulness. In this study, we investigate 13 open-source LLMs categorized into large (e.g., Llama3:70b, Gemma2:27b, QwQ:32b), medium (e.g., Gemma2:9b, Llama3:8b), and small (e.g., Llama3.2:3b, Phi3:3.8b). A Retrieval-Augmented Generation (RAG) is used to make up for the problems that come with using separate models. This research utilizes a descriptive dataset of Quranic surahs including the meanings, historical context, and qualities of the 114 surahs, allowing the model to gather relevant knowledge before responding. The models are evaluated using three key metrics set by human evaluators: context relevance, answer faithfulness, and answer relevance. The findings reveal that large models consistently outperform smaller models in capturing query semantics and producing accurate, contextually grounded responses. The Llama3.2:3b model, even though it is considered small, does very well on faithfulness (4.619) and relevance (4.857), showing the promise of smaller architectures that have been well optimized. This article examines the trade-offs between model size, computational efficiency, and response quality while using LLMs in domain-specific applications. 

---
# Feature selection strategies for optimized heart disease diagnosis using ML and DL models 

**Authors**: Bilal Ahmad, Jinfu Chen, Haibao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.16577)  

**Abstract**: Heart disease remains one of the leading causes of morbidity and mortality worldwide, necessitating the development of effective diagnostic tools to enable early diagnosis and clinical decision-making. This study evaluates the impact of feature selection techniques Mutual Information (MI), Analysis of Variance (ANOVA), and Chi-Square on the predictive performance of various machine learning (ML) and deep learning (DL) models using a dataset of clinical indicators for heart disease. Eleven ML/DL models were assessed using metrics such as precision, recall, AUC score, F1-score, and accuracy. Results indicate that MI outperformed other methods, particularly for advanced models like neural networks, achieving the highest accuracy of 82.3% and recall score of 0.94. Logistic regression (accuracy 82.1%) and random forest (accuracy 80.99%) also demonstrated improved performance with MI. Simpler models such as Naive Bayes and decision trees achieved comparable results with ANOVA and Chi-Square, yielding accuracies of 76.45% and 75.99%, respectively, making them computationally efficient alternatives. Conversely, k Nearest Neighbors (KNN) and Support Vector Machines (SVM) exhibited lower performance, with accuracies ranging between 51.52% and 54.43%, regardless of the feature selection method. This study provides a comprehensive comparison of feature selection methods for heart disease prediction, demonstrating the critical role of feature selection in optimizing model performance. The results offer practical guidance for selecting appropriate feature selection techniques based on the chosen classification algorithm, contributing to the development of more accurate and efficient diagnostic tools for enhanced clinical decision-making in cardiology. 

---
# Extract, Match, and Score: An Evaluation Paradigm for Long Question-context-answer Triplets in Financial Analysis 

**Authors**: Bo Hu, Han Yuan, Vlad Pandelea, Wuqiong Luo, Yingzhu Zhao, Zheng Ma  

**Link**: [PDF](https://arxiv.org/pdf/2503.16575)  

**Abstract**: The rapid advancement of large language models (LLMs) has sparked widespread adoption across diverse applications, making robust evaluation frameworks crucial for assessing their performance. While conventional evaluation metrics remain applicable for shorter texts, their efficacy diminishes when evaluating the quality of long-form answers. This limitation is particularly critical in real-world scenarios involving extended questions, extensive context, and long-form answers, such as financial analysis or regulatory compliance. In this paper, we use a practical financial use case to illustrate applications that handle "long question-context-answer triplets". We construct a real-world financial dataset comprising long triplets and demonstrate the inadequacies of traditional metrics. To address this, we propose an effective Extract, Match, and Score (EMS) evaluation approach tailored to the complexities of long-form LLMs' outputs, providing practitioners with a reliable methodology for assessing LLMs' performance in complex real-world scenarios. 

---
# AUV Acceleration Prediction Using DVL and Deep Learning 

**Authors**: Yair Stolero, Itzik Klein  

**Link**: [PDF](https://arxiv.org/pdf/2503.16573)  

**Abstract**: Autonomous underwater vehicles (AUVs) are essential for various applications, including oceanographic surveys, underwater mapping, and infrastructure inspections. Accurate and robust navigation are critical to completing these tasks. To this end, a Doppler velocity log (DVL) and inertial sensors are fused together. Recently, a model-based approach demonstrated the ability to extract the vehicle acceleration vector from DVL velocity measurements. Motivated by this advancement, in this paper we present an end-to-end deep learning approach to estimate the AUV acceleration vector based on past DVL velocity measurements. Based on recorded data from sea experiments, we demonstrate that the proposed method improves acceleration vector estimation by more than 65% compared to the model-based approach by using data-driven techniques. As a result of our data-driven approach, we can enhance navigation accuracy and reliability in AUV applications, contributing to more efficient and effective underwater missions through improved accuracy and reliability. 

---
# Efficient ANN-Guided Distillation: Aligning Rate-based Features of Spiking Neural Networks through Hybrid Block-wise Replacement 

**Authors**: Shu Yang, Chengting Yu, Lei Liu, Hanzhi Ma, Aili Wang, Erping Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.16572)  

**Abstract**: Spiking Neural Networks (SNNs) have garnered considerable attention as a potential alternative to Artificial Neural Networks (ANNs). Recent studies have highlighted SNNs' potential on large-scale datasets. For SNN training, two main approaches exist: direct training and ANN-to-SNN (ANN2SNN) conversion. To fully leverage existing ANN models in guiding SNN learning, either direct ANN-to-SNN conversion or ANN-SNN distillation training can be employed. In this paper, we propose an ANN-SNN distillation framework from the ANN-to-SNN perspective, designed with a block-wise replacement strategy for ANN-guided learning. By generating intermediate hybrid models that progressively align SNN feature spaces to those of ANN through rate-based features, our framework naturally incorporates rate-based backpropagation as a training method. Our approach achieves results comparable to or better than state-of-the-art SNN distillation methods, showing both training and learning efficiency. 

---
# Gene42: Long-Range Genomic Foundation Model With Dense Attention 

**Authors**: Kirill Vishniakov, Boulbaba Ben Amor, Engin Tekin, Nancy A. ElNaker, Karthik Viswanathan, Aleksandr Medvedev, Aahan Singh, Maryam Nadeem, Mohammad Amaan Sayeed, Praveenkumar Kanithi, Tiago Magalhaes, Natalia Vassilieva, Dwarikanath Mahapatra, Marco Pimentel, and Shadab Khan  

**Link**: [PDF](https://arxiv.org/pdf/2503.16565)  

**Abstract**: We introduce Gene42, a novel family of Genomic Foundation Models (GFMs) designed to manage context lengths of up to 192,000 base pairs (bp) at a single-nucleotide resolution. Gene42 models utilize a decoder-only (LLaMA-style) architecture with a dense self-attention mechanism. Initially trained on fixed-length sequences of 4,096 bp, our models underwent continuous pretraining to extend the context length to 192,000 bp. This iterative extension allowed for the comprehensive processing of large-scale genomic data and the capture of intricate patterns and dependencies within the human genome. Gene42 is the first dense attention model capable of handling such extensive long context lengths in genomics, challenging state-space models that often rely on convolutional operators among other mechanisms. Our pretrained models exhibit notably low perplexity values and high reconstruction accuracy, highlighting their strong ability to model genomic data. Extensive experiments on various genomic benchmarks have demonstrated state-of-the-art performance across multiple tasks, including biotype classification, regulatory region identification, chromatin profiling prediction, variant pathogenicity prediction, and species classification. The models are publicly available at this http URL. 

---
# Chem42: a Family of chemical Language Models for Target-aware Ligand Generation 

**Authors**: Aahan Singh, Engin Tekin, Maryam Nadeem, Nancy A. ElNaker, Mohammad Amaan Sayeed, Natalia Vassilieva, Boulbaba Ben Amor  

**Link**: [PDF](https://arxiv.org/pdf/2503.16563)  

**Abstract**: Revolutionizing drug discovery demands more than just understanding molecular interactions - it requires generative models that can design novel ligands tailored to specific biological targets. While chemical Language Models (cLMs) have made strides in learning molecular properties, most fail to incorporate target-specific insights, restricting their ability to drive de-novo ligand generation. Chem42, a cutting-edge family of generative chemical Language Models, is designed to bridge this gap. By integrating atomic-level interactions with multimodal inputs from Prot42, a complementary protein Language Model, Chem42 achieves a sophisticated cross-modal representation of molecular structures, interactions, and binding patterns. This innovative framework enables the creation of structurally valid, synthetically accessible ligands with enhanced target specificity. Evaluations across diverse protein targets confirm that Chem42 surpasses existing approaches in chemical validity, target-aware design, and predicted binding affinity. By reducing the search space of viable drug candidates, Chem42 could accelerate the drug discovery pipeline, offering a powerful generative AI tool for precision medicine. Our Chem42 models set a new benchmark in molecule property prediction, conditional molecule generation, and target-aware ligand design. The models are publicly available at this http URL. 

---
# Advancing Problem-Based Learning in Biomedical Engineering in the Era of Generative AI 

**Authors**: Micky C. Nnamdi, J. Ben Tamo, Wenqi Shi, May D. Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.16558)  

**Abstract**: Problem-Based Learning (PBL) has significantly impacted biomedical engineering (BME) education since its introduction in the early 2000s, effectively enhancing critical thinking and real-world knowledge application among students. With biomedical engineering rapidly converging with artificial intelligence (AI), integrating effective AI education into established curricula has become challenging yet increasingly necessary. Recent advancements, including AI's recognition by the 2024 Nobel Prize, have highlighted the importance of training students comprehensively in biomedical AI. However, effective biomedical AI education faces substantial obstacles, such as diverse student backgrounds, limited personalized mentoring, constrained computational resources, and difficulties in safely scaling hands-on practical experiments due to privacy and ethical concerns associated with biomedical data. To overcome these issues, we conducted a three-year (2021-2023) case study implementing an advanced PBL framework tailored specifically for biomedical AI education, involving 92 undergraduate and 156 graduate students from the joint Biomedical Engineering program of Georgia Institute of Technology and Emory University. Our approach emphasizes collaborative, interdisciplinary problem-solving through authentic biomedical AI challenges. The implementation led to measurable improvements in learning outcomes, evidenced by high research productivity (16 student-authored publications), consistently positive peer evaluations, and successful development of innovative computational methods addressing real biomedical challenges. Additionally, we examined the role of generative AI both as a teaching subject and an educational support tool within the PBL framework. Our study presents a practical and scalable roadmap for biomedical engineering departments aiming to integrate robust AI education into their curricula. 

---
# Reliable Radiologic Skeletal Muscle Area Assessment -- A Biomarker for Cancer Cachexia Diagnosis 

**Authors**: Sabeen Ahmed, Nathan Parker, Margaret Park, Daniel Jeong, Lauren Peres, Evan W. Davis, Jennifer B. Permuth, Erin Siegel, Matthew B. Schabath, Yasin Yilmaz, Ghulam Rasool  

**Link**: [PDF](https://arxiv.org/pdf/2503.16556)  

**Abstract**: Cancer cachexia is a common metabolic disorder characterized by severe muscle atrophy which is associated with poor prognosis and quality of life. Monitoring skeletal muscle area (SMA) longitudinally through computed tomography (CT) scans, an imaging modality routinely acquired in cancer care, is an effective way to identify and track this condition. However, existing tools often lack full automation and exhibit inconsistent accuracy, limiting their potential for integration into clinical workflows. To address these challenges, we developed SMAART-AI (Skeletal Muscle Assessment-Automated and Reliable Tool-based on AI), an end-to-end automated pipeline powered by deep learning models (nnU-Net 2D) trained on mid-third lumbar level CT images with 5-fold cross-validation, ensuring generalizability and robustness. SMAART-AI incorporates an uncertainty-based mechanism to flag high-error SMA predictions for expert review, enhancing reliability. We combined the SMA, skeletal muscle index, BMI, and clinical data to train a multi-layer perceptron (MLP) model designed to predict cachexia at the time of cancer diagnosis. Tested on the gastroesophageal cancer dataset, SMAART-AI achieved a Dice score of 97.80% +/- 0.93%, with SMA estimated across all four datasets in this study at a median absolute error of 2.48% compared to manual annotations with SliceOmatic. Uncertainty metrics-variance, entropy, and coefficient of variation-strongly correlated with SMA prediction errors (0.83, 0.76, and 0.73 respectively). The MLP model predicts cachexia with 79% precision, providing clinicians with a reliable tool for early diagnosis and intervention. By combining automation, accuracy, and uncertainty awareness, SMAART-AI bridges the gap between research and clinical application, offering a transformative approach to managing cancer cachexia. 

---
# A Comprehensive Survey on Architectural Advances in Deep CNNs: Challenges, Applications, and Emerging Research Directions 

**Authors**: Saddam Hussain Khan, Rashid Iqbal  

**Link**: [PDF](https://arxiv.org/pdf/2503.16546)  

**Abstract**: Deep Convolutional Neural Networks (CNNs) have significantly advanced deep learning, driving breakthroughs in computer vision, natural language processing, medical diagnosis, object detection, and speech recognition. Architectural innovations including 1D, 2D, and 3D convolutional models, dilated and grouped convolutions, depthwise separable convolutions, and attention mechanisms address domain-specific challenges and enhance feature representation and computational efficiency. Structural refinements such as spatial-channel exploitation, multi-path design, and feature-map enhancement contribute to robust hierarchical feature extraction and improved generalization, particularly through transfer learning. Efficient preprocessing strategies, including Fourier transforms, structured transforms, low-precision computation, and weight compression, optimize inference speed and facilitate deployment in resource-constrained environments. This survey presents a unified taxonomy that classifies CNN architectures based on spatial exploitation, multi-path structures, depth, width, dimensionality expansion, channel boosting, and attention mechanisms. It systematically reviews CNN applications in face recognition, pose estimation, action recognition, text classification, statistical language modeling, disease diagnosis, radiological analysis, cryptocurrency sentiment prediction, 1D data processing, video analysis, and speech recognition. In addition to consolidating architectural advancements, the review highlights emerging learning paradigms such as few-shot, zero-shot, weakly supervised, federated learning frameworks and future research directions include hybrid CNN-transformer models, vision-language integration, generative learning, etc. This review provides a comprehensive perspective on CNN's evolution from 2015 to 2025, outlining key innovations, challenges, and opportunities. 

---
# Causal Discovery and Counterfactual Reasoning to Optimize Persuasive Dialogue Policies 

**Authors**: Donghuo Zeng, Roberto Legaspi, Yuewen Sun, Xinshuai Dong, Kazushi Ikeda, Peter Spirtes, Kun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.16544)  

**Abstract**: Tailoring persuasive conversations to users leads to more effective persuasion. However, existing dialogue systems often struggle to adapt to dynamically evolving user states. This paper presents a novel method that leverages causal discovery and counterfactual reasoning for optimizing system persuasion capability and outcomes. We employ the Greedy Relaxation of the Sparsest Permutation (GRaSP) algorithm to identify causal relationships between user and system utterance strategies, treating user strategies as states and system strategies as actions. GRaSP identifies user strategies as causal factors influencing system responses, which inform Bidirectional Conditional Generative Adversarial Networks (BiCoGAN) in generating counterfactual utterances for the system. Subsequently, we use the Dueling Double Deep Q-Network (D3QN) model to utilize counterfactual data to determine the best policy for selecting system utterances. Our experiments with the PersuasionForGood dataset show measurable improvements in persuasion outcomes using our approach over baseline methods. The observed increase in cumulative rewards and Q-values highlights the effectiveness of causal discovery in enhancing counterfactual reasoning and optimizing reinforcement learning policies for online dialogue systems. 

---
# Gender and content bias in Large Language Models: a case study on Google Gemini 2.0 Flash Experimental 

**Authors**: Roberto Balestri  

**Link**: [PDF](https://arxiv.org/pdf/2503.16534)  

**Abstract**: This study evaluates the biases in Gemini 2.0 Flash Experimental, a state-of-the-art large language model (LLM) developed by Google, focusing on content moderation and gender disparities. By comparing its performance to ChatGPT-4o, examined in a previous work of the author, the analysis highlights some differences in ethical moderation practices. Gemini 2.0 demonstrates reduced gender bias, notably with female-specific prompts achieving a substantial rise in acceptance rates compared to results obtained by ChatGPT-4o. It adopts a more permissive stance toward sexual content and maintains relatively high acceptance rates for violent prompts, including gender-specific cases. Despite these changes, whether they constitute an improvement is debatable. While gender bias has been reduced, this reduction comes at the cost of permitting more violent content toward both males and females, potentially normalizing violence rather than mitigating harm. Male-specific prompts still generally receive higher acceptance rates than female-specific ones. These findings underscore the complexities of aligning AI systems with ethical standards, highlighting progress in reducing certain biases while raising concerns about the broader implications of the model's permissiveness. Ongoing refinements are essential to achieve moderation practices that ensure transparency, fairness, and inclusivity without amplifying harmful content. 

---
# From Patient Consultations to Graphs: Leveraging LLMs for Patient Journey Knowledge Graph Construction 

**Authors**: Hassan S. Al Khatib, Sudip Mittal, Shahram Rahimi, Nina Marhamati, Sean Bozorgzad  

**Link**: [PDF](https://arxiv.org/pdf/2503.16533)  

**Abstract**: The transition towards patient-centric healthcare necessitates a comprehensive understanding of patient journeys, which encompass all healthcare experiences and interactions across the care spectrum. Existing healthcare data systems are often fragmented and lack a holistic representation of patient trajectories, creating challenges for coordinated care and personalized interventions. Patient Journey Knowledge Graphs (PJKGs) represent a novel approach to addressing the challenge of fragmented healthcare data by integrating diverse patient information into a unified, structured representation. This paper presents a methodology for constructing PJKGs using Large Language Models (LLMs) to process and structure both formal clinical documentation and unstructured patient-provider conversations. These graphs encapsulate temporal and causal relationships among clinical encounters, diagnoses, treatments, and outcomes, enabling advanced temporal reasoning and personalized care insights. The research evaluates four different LLMs, such as Claude 3.5, Mistral, Llama 3.1, and Chatgpt4o, in their ability to generate accurate and computationally efficient knowledge graphs. Results demonstrate that while all models achieved perfect structural compliance, they exhibited variations in medical entity processing and computational efficiency. The paper concludes by identifying key challenges and future research directions. This work contributes to advancing patient-centric healthcare through the development of comprehensive, actionable knowledge graphs that support improved care coordination and outcome prediction. 

---
# Modelling Emotions in Face-to-Face Setting: The Interplay of Eye-Tracking, Personality, and Temporal Dynamics 

**Authors**: Meisam Jamshidi Seikavandi, Jostein Fimland, Maria Barrett, Paolo Burelli  

**Link**: [PDF](https://arxiv.org/pdf/2503.16532)  

**Abstract**: Accurate emotion recognition is pivotal for nuanced and engaging human-computer interactions, yet remains difficult to achieve, especially in dynamic, conversation-like settings. In this study, we showcase how integrating eye-tracking data, temporal dynamics, and personality traits can substantially enhance the detection of both perceived and felt emotions. Seventy-three participants viewed short, speech-containing videos from the CREMA-D dataset, while being recorded for eye-tracking signals (pupil size, fixation patterns), Big Five personality assessments, and self-reported emotional states. Our neural network models combined these diverse inputs including stimulus emotion labels for contextual cues and yielded marked performance gains compared to the state-of-the-art. Specifically, perceived valence predictions reached a macro F1-score of 0.76, and models incorporating personality traits and stimulus information demonstrated significant improvements in felt emotion accuracy. These results highlight the benefit of unifying physiological, individual and contextual factors to address the subjectivity and complexity of emotional expression. Beyond validating the role of user-specific data in capturing subtle internal states, our findings inform the design of future affective computing and human-agent systems, paving the way for more adaptive and cross-individual emotional intelligence in real-world interactions. 

---
# Enhancing LLM Generation with Knowledge Hypergraph for Evidence-Based Medicine 

**Authors**: Chengfeng Dou, Ying Zhang, Zhi Jin, Wenpin Jiao, Haiyan Zhao, Yongqiang Zhao, Zhengwei Tao  

**Link**: [PDF](https://arxiv.org/pdf/2503.16530)  

**Abstract**: Evidence-based medicine (EBM) plays a crucial role in the application of large language models (LLMs) in healthcare, as it provides reliable support for medical decision-making processes. Although it benefits from current retrieval-augmented generation~(RAG) technologies, it still faces two significant challenges: the collection of dispersed evidence and the efficient organization of this evidence to support the complex queries necessary for EBM. To tackle these issues, we propose using LLMs to gather scattered evidence from multiple sources and present a knowledge hypergraph-based evidence management model to integrate these evidence while capturing intricate relationships. Furthermore, to better support complex queries, we have developed an Importance-Driven Evidence Prioritization (IDEP) algorithm that utilizes the LLM to generate multiple evidence features, each with an associated importance score, which are then used to rank the evidence and produce the final retrieval results. Experimental results from six datasets demonstrate that our approach outperforms existing RAG techniques in application domains of interest to EBM, such as medical quizzing, hallucination detection, and decision support. Testsets and the constructed knowledge graph can be accessed at \href{this https URL}{this https URL}. 

---
# Safety Evaluation and Enhancement of DeepSeek Models in Chinese Contexts 

**Authors**: Wenjing Zhang, Xuejiao Lei, Zhaoxiang Liu, Limin Han, Jiaojiao Zhao, Beibei Huang, Zhenhong Long, Junting Guo, Meijuan An, Rongjia Du, Ning Wang, Kai Wang, Shiguo Lian  

**Link**: [PDF](https://arxiv.org/pdf/2503.16529)  

**Abstract**: DeepSeek-R1, renowned for its exceptional reasoning capabilities and open-source strategy, is significantly influencing the global artificial intelligence landscape. However, it exhibits notable safety shortcomings. Recent research conducted by Robust Intelligence, a subsidiary of Cisco, in collaboration with the University of Pennsylvania, revealed that DeepSeek-R1 achieves a 100\% attack success rate when processing harmful prompts. Furthermore, multiple security firms and research institutions have identified critical security vulnerabilities within the model. Although China Unicom has uncovered safety vulnerabilities of R1 in Chinese contexts, the safety capabilities of the remaining distilled models in the R1 series have not yet been comprehensively evaluated. To address this gap, this study utilizes the comprehensive Chinese safety benchmark CHiSafetyBench to conduct an in-depth safety evaluation of the DeepSeek-R1 series distilled models. The objective is to assess the safety capabilities of these models in Chinese contexts both before and after distillation, and to further elucidate the adverse effects of distillation on model safety. Building on these findings, we implement targeted safety enhancements for six distilled models. Evaluation results indicate that the enhanced models achieve significant improvements in safety while maintaining reasoning capabilities without notable degradation. We open-source the safety-enhanced models at this https URL to serve as a valuable resource for future research and optimization of DeepSeek models. 

---
# HDLCoRe: A Training-Free Framework for Mitigating Hallucinations in LLM-Generated HDL 

**Authors**: Heng Ping, Shixuan Li, Peiyu Zhang, Anzhe Cheng, Shukai Duan, Nikos Kanakaris, Xiongye Xiao, Wei Yang, Shahin Nazarian, Andrei Irimia, Paul Bogdan  

**Link**: [PDF](https://arxiv.org/pdf/2503.16528)  

**Abstract**: Recent advances in large language models (LLMs) have demonstrated remarkable capabilities in code generation tasks. However, when applied to hardware description languages (HDL), these models exhibit significant limitations due to data scarcity, resulting in hallucinations and incorrect code generation. To address these challenges, we propose HDLCoRe, a training-free framework that enhances LLMs' HDL generation capabilities through prompt engineering techniques and retrieval-augmented generation (RAG). Our approach consists of two main components: (1) an HDL-aware Chain-of-Thought (CoT) prompting technique with self-verification that classifies tasks by complexity and type, incorporates domain-specific knowledge, and guides LLMs through step-by-step self-simulation for error correction; and (2) a two-stage heterogeneous RAG system that addresses formatting inconsistencies through key component extraction and efficiently retrieves relevant HDL examples through sequential filtering and re-ranking. HDLCoRe eliminates the need for model fine-tuning while substantially improving LLMs' HDL generation capabilities. Experimental results demonstrate that our framework achieves superior performance on the RTLLM2.0 benchmark, significantly reducing hallucinations and improving both syntactic and functional correctness. 

---
# LLM Generated Persona is a Promise with a Catch 

**Authors**: Ang Li, Haozhe Chen, Hongseok Namkoong, Tianyi Peng  

**Link**: [PDF](https://arxiv.org/pdf/2503.16527)  

**Abstract**: The use of large language models (LLMs) to simulate human behavior has gained significant attention, particularly through personas that approximate individual characteristics. Persona-based simulations hold promise for transforming disciplines that rely on population-level feedback, including social science, economic analysis, marketing research, and business operations. Traditional methods to collect realistic persona data face significant challenges. They are prohibitively expensive and logistically challenging due to privacy constraints, and often fail to capture multi-dimensional attributes, particularly subjective qualities. Consequently, synthetic persona generation with LLMs offers a scalable, cost-effective alternative. However, current approaches rely on ad hoc and heuristic generation techniques that do not guarantee methodological rigor or simulation precision, resulting in systematic biases in downstream tasks. Through extensive large-scale experiments including presidential election forecasts and general opinion surveys of the U.S. population, we reveal that these biases can lead to significant deviations from real-world outcomes. Our findings underscore the need to develop a rigorous science of persona generation and outline the methodological innovations, organizational and institutional support, and empirical foundations required to enhance the reliability and scalability of LLM-driven persona simulations. To support further research and development in this area, we have open-sourced approximately one million generated personas, available for public access and analysis at this https URL. 

---
# KVShare: Semantic-Aware Key-Value Cache Sharing for Efficient Large Language Model Inference 

**Authors**: Huan Yang, Renji Zhang, Deyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.16525)  

**Abstract**: This paper presents KVShare, a multi-user Key-Value (KV) Cache sharing technology based on semantic similarity, designed to enhance the inference efficiency of Large Language Models (LLMs) and Multimodal Large Language Models (MLLMs). Addressing the limitations of existing prefix caching (strict text prefix matching) and semantic caching (loss of response diversity), KVShare achieves fine-grained KV cache reuse through semantic alignment algorithms and differential editing operations. Experiments on real-world user conversation datasets demonstrate that KVShare improves KV cache hit rates by over 60%, while maintaining output quality comparable to full computation (no significant degradation in BLEU and Rouge-L metrics). This approach effectively reduces GPU resource consumption and is applicable to scenarios with repetitive queries, such as healthcare and education. 

---
# Mind2: Mind-to-Mind Emotional Support System with Bidirectional Cognitive Discourse Analysis 

**Authors**: Shi Yin Hong, Uttamasha Oyshi, Quan Mai, Gibson Nkhata, Susan Gauch  

**Link**: [PDF](https://arxiv.org/pdf/2503.16523)  

**Abstract**: Emotional support (ES) systems alleviate users' mental distress by generating strategic supportive dialogues based on diverse user situations. However, ES systems are limited in their ability to generate effective ES dialogues that include timely context and interpretability, hindering them from earning public trust. Driven by cognitive models, we propose Mind-to-Mind (Mind2), an ES framework that approaches interpretable ES context modeling for the ES dialogue generation task from a discourse analysis perspective. Specifically, we perform cognitive discourse analysis on ES dialogues according to our dynamic discourse context propagation window, which accommodates evolving context as the conversation between the ES system and user progresses. To enhance interpretability, Mind2 prioritizes details that reflect each speaker's belief about the other speaker with bidirectionality, integrating Theory-of-Mind, physiological expected utility, and cognitive rationality to extract cognitive knowledge from ES conversations. Experimental results support that Mind2 achieves competitive performance versus state-of-the-art ES systems while trained with only 10\% of the available training data. 

---
# Not All Personas Are Worth It: Culture-Reflective Persona Data Augmentation 

**Authors**: Ji-Eun Han, Yoonseok Heo  

**Link**: [PDF](https://arxiv.org/pdf/2503.16520)  

**Abstract**: Incorporating personas into conversational AI models is crucial for achieving authentic and engaging interactions. However, the cultural diversity and adaptability of existing persona datasets is often overlooked, reducing their efficacy in building culturally aware AI systems. To address this issue, we propose a two-step pipeline for generating culture-specific personas and introduce KoPersona, a dataset comprising 200,000 personas designed to capture Korean cultural values, behaviors, and social nuances. A comprehensive evaluation through various metrics validates the quality of KoPersona and its relevance to Korean culture. This work not only contributes to persona-based research, but also establishes a scalable approach for creating culturally relevant personas adaptable to various languages and cultural contexts. 

---
# Advancing Human-Machine Teaming: Concepts, Challenges, and Applications 

**Authors**: Dian Chen, Han Jun Yoon, Zelin Wan, Nithin Alluru, Sang Won Lee, Richard He, Terrence J. Moore, Frederica F. Nelson, Sunghyun Yoon, Hyuk Lim, Dan Dongseong Kim, Jin-Hee Cho  

**Link**: [PDF](https://arxiv.org/pdf/2503.16518)  

**Abstract**: Human-Machine Teaming (HMT) is revolutionizing collaboration across domains such as defense, healthcare, and autonomous systems by integrating AI-driven decision-making, trust calibration, and adaptive teaming. This survey presents a comprehensive taxonomy of HMT, analyzing theoretical models, including reinforcement learning, instance-based learning, and interdependence theory, alongside interdisciplinary methodologies. Unlike prior reviews, we examine team cognition, ethical AI, multi-modal interactions, and real-world evaluation frameworks. Key challenges include explainability, role allocation, and scalable benchmarking. We propose future research in cross-domain adaptation, trust-aware AI, and standardized testbeds. By bridging computational and social sciences, this work lays a foundation for resilient, ethical, and scalable HMT systems. 

---
# From G-Factor to A-Factor: Establishing a Psychometric Framework for AI Literacy 

**Authors**: Ning Li, Wenming Deng, Jiatan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.16517)  

**Abstract**: This research addresses the growing need to measure and understand AI literacy in the context of generative AI technologies. Through three sequential studies involving a total of 517 participants, we establish AI literacy as a coherent, measurable construct with significant implications for education, workforce development, and social equity. Study 1 (N=85) revealed a dominant latent factor - termed the "A-factor" - that accounts for 44.16% of variance across diverse AI interaction tasks. Study 2 (N=286) refined the measurement tool by examining four key dimensions of AI literacy: communication effectiveness, creative idea generation, content evaluation, and step-by-step collaboration, resulting in an 18-item assessment battery. Study 3 (N=146) validated this instrument in a controlled laboratory setting, demonstrating its predictive validity for real-world task performance. Results indicate that AI literacy significantly predicts performance on complex, language-based creative tasks but shows domain specificity in its predictive power. Additionally, regression analyses identified several significant predictors of AI literacy, including cognitive abilities (IQ), educational background, prior AI experience, and training history. The multidimensional nature of AI literacy and its distinct factor structure provide evidence that effective human-AI collaboration requires a combination of general and specialized abilities. These findings contribute to theoretical frameworks of human-AI collaboration while offering practical guidance for developing targeted educational interventions to promote equitable access to the benefits of generative AI technologies. 

---
# Using LLMs for Automated Privacy Policy Analysis: Prompt Engineering, Fine-Tuning and Explainability 

**Authors**: Yuxin Chen, Peng Tang, Weidong Qiu, Shujun Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.16516)  

**Abstract**: Privacy policies are widely used by digital services and often required for legal purposes. Many machine learning based classifiers have been developed to automate detection of different concepts in a given privacy policy, which can help facilitate other automated tasks such as producing a more reader-friendly summary and detecting legal compliance issues. Despite the successful applications of large language models (LLMs) to many NLP tasks in various domains, there is very little work studying the use of LLMs for automated privacy policy analysis, therefore, if and how LLMs can help automate privacy policy analysis remains under-explored. To fill this research gap, we conducted a comprehensive evaluation of LLM-based privacy policy concept classifiers, employing both prompt engineering and LoRA (low-rank adaptation) fine-tuning, on four state-of-the-art (SOTA) privacy policy corpora and taxonomies. Our experimental results demonstrated that combining prompt engineering and fine-tuning can make LLM-based classifiers outperform other SOTA methods, \emph{significantly} and \emph{consistently} across privacy policy corpora/taxonomies and concepts. Furthermore, we evaluated the explainability of the LLM-based classifiers using three metrics: completeness, logicality, and comprehensibility. For all three metrics, a score exceeding 91.1\% was observed in our evaluation, indicating that LLMs are not only useful to improve the classification performance, but also to enhance the explainability of detection results. 

---
# Highlighting Case Studies in LLM Literature Review of Interdisciplinary System Science 

**Authors**: Lachlan McGinness, Peter Baumgartner  

**Link**: [PDF](https://arxiv.org/pdf/2503.16515)  

**Abstract**: Large Language Models (LLMs) were used to assist four Commonwealth Scientific and Industrial Research Organisation (CSIRO) researchers to perform systematic literature reviews (SLR). We evaluate the performance of LLMs for SLR tasks in these case studies. In each, we explore the impact of changing parameters on the accuracy of LLM responses. The LLM was tasked with extracting evidence from chosen academic papers to answer specific research questions. We evaluate the models' performance in faithfully reproducing quotes from the literature and subject experts were asked to assess the model performance in answering the research questions. We developed a semantic text highlighting tool to facilitate expert review of LLM responses.
We found that state of the art LLMs were able to reproduce quotes from texts with greater than 95% accuracy and answer research questions with an accuracy of approximately 83%. We use two methods to determine the correctness of LLM responses; expert review and the cosine similarity of transformer embeddings of LLM and expert answers. The correlation between these methods ranged from 0.48 to 0.77, providing evidence that the latter is a valid metric for measuring semantic similarity. 

---
# VeriMind: Agentic LLM for Automated Verilog Generation with a Novel Evaluation Metric 

**Authors**: Bardia Nadimi, Ghali Omar Boutaib, Hao Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2503.16514)  

**Abstract**: Designing Verilog modules requires meticulous attention to correctness, efficiency, and adherence to design specifications. However, manually writing Verilog code remains a complex and time-consuming task that demands both expert knowledge and iterative refinement. Leveraging recent advancements in large language models (LLMs) and their structured text generation capabilities, we propose VeriMind, an agentic LLM framework for Verilog code generation that significantly automates and optimizes the synthesis process. Unlike traditional LLM-based code generators, VeriMind employs a structured reasoning approach: given a user-provided prompt describing design requirements, the system first formulates a detailed train of thought before the final Verilog code is generated. This multi-step methodology enhances interpretability, accuracy, and adaptability in hardware design. In addition, we introduce a novel evaluation metric-pass@ARC-which combines the conventional pass@k measure with Average Refinement Cycles (ARC) to capture both success rate and the efficiency of iterative refinement. Experimental results on diverse hardware design tasks demonstrated that our approach achieved up to $8.3\%$ improvement on pass@k metric and $8.1\%$ on pass@ARC metric. These findings underscore the transformative potential of agentic LLMs in automated hardware design, RTL development, and digital system synthesis. 

---
# Medifact at PerAnsSumm 2025: Leveraging Lightweight Models for Perspective-Specific Summarization of Clinical Q&A Forums 

**Authors**: Nadia Saeed  

**Link**: [PDF](https://arxiv.org/pdf/2503.16513)  

**Abstract**: The PerAnsSumm 2025 challenge focuses on perspective-aware healthcare answer summarization (Agarwal et al., 2025). This work proposes a few-shot learning framework using a Snorkel-BART-SVM pipeline for classifying and summarizing open-ended healthcare community question-answering (CQA). An SVM model is trained with weak supervision via Snorkel, enhancing zero-shot learning. Extractive classification identifies perspective-relevant sentences, which are then summarized using a pretrained BART-CNN model. The approach achieved 12th place among 100 teams in the shared task, demonstrating computational efficiency and contextual accuracy. By leveraging pretrained summarization models, this work advances medical CQA research and contributes to clinical decision support systems. 

---
# Token-Level Uncertainty-Aware Objective for Language Model Post-Training 

**Authors**: Tingkai Liu, Ari S. Benjamin, Anthony M. Zador  

**Link**: [PDF](https://arxiv.org/pdf/2503.16511)  

**Abstract**: In the current work, we connect token-level uncertainty in causal language modeling to two types of training objectives: 1) masked maximum likelihood (MLE), 2) self-distillation. We show that masked MLE is effective in reducing epistemic uncertainty, and serve as an effective token-level automatic curriculum learning technique. However, masked MLE is prone to overfitting and requires self-distillation regularization to improve or maintain performance on out-of-distribution tasks. We demonstrate significant performance gain via the proposed training objective - combined masked MLE and self-distillation - across multiple architectures (Gemma, LLaMA, Phi) and datasets (Alpaca, ShareGPT, GSM8K), mitigating overfitting while maintaining adaptability during post-training. Our findings suggest that uncertainty-aware training provides an effective mechanism for enhancing language model training. 

---
# Conversational AI as a Coding Assistant: Understanding Programmers' Interactions with and Expectations from Large Language Models for Coding 

**Authors**: Mehmet Akhoroz, Caglar Yildirim  

**Link**: [PDF](https://arxiv.org/pdf/2503.16508)  

**Abstract**: Conversational AI interfaces powered by large language models (LLMs) are increasingly used as coding assistants. However, questions remain about how programmers interact with LLM-based conversational agents, the challenges they encounter, and the factors influencing adoption. This study investigates programmers' usage patterns, perceptions, and interaction strategies when engaging with LLM-driven coding assistants. Through a survey, participants reported both the benefits, such as efficiency and clarity of explanations, and the limitations, including inaccuracies, lack of contextual awareness, and concerns about over-reliance. Notably, some programmers actively avoid LLMs due to a preference for independent learning, distrust in AI-generated code, and ethical considerations. Based on our findings, we propose design guidelines for improving conversational coding assistants, emphasizing context retention, transparency, multimodal support, and adaptability to user preferences. These insights contribute to the broader understanding of how LLM-based conversational agents can be effectively integrated into software development workflows while addressing adoption barriers and enhancing usability. 

---
# Fewer Than 1% of Explainable AI Papers Validate Explainability with Humans 

**Authors**: Ashley Suh, Isabelle Hurley, Nora Smith, Ho Chit Siu  

**Link**: [PDF](https://arxiv.org/pdf/2503.16507)  

**Abstract**: This late-breaking work presents a large-scale analysis of explainable AI (XAI) literature to evaluate claims of human explainability. We collaborated with a professional librarian to identify 18,254 papers containing keywords related to explainability and interpretability. Of these, we find that only 253 papers included terms suggesting human involvement in evaluating an XAI technique, and just 128 of those conducted some form of a human study. In other words, fewer than 1% of XAI papers (0.7%) provide empirical evidence of human explainability when compared to the broader body of XAI literature. Our findings underscore a critical gap between claims of human explainability and evidence-based validation, raising concerns about the rigor of XAI research. We call for increased emphasis on human evaluations in XAI studies and provide our literature search methodology to enable both reproducibility and further investigation into this widespread issue. 

---
# Stakeholder Perspectives on Whether and How Social Robots Can Support Mediation and Advocacy for Higher Education Students with Disabilities 

**Authors**: Alva Markelius, Julie Bailey, Jenny L. Gibson, Hatice Gunes  

**Link**: [PDF](https://arxiv.org/pdf/2503.16499)  

**Abstract**: This paper presents an iterative, participatory, empirical study that examines the potential of using artificial intelligence, such as social robots and large language models, to support mediation and advocacy for students with disabilities in higher education. Drawing on qualitative data from interviews and focus groups conducted with various stakeholders, including disabled students, disabled student representatives, and disability practitioners at the University of Cambridge, this study reports findings relating to understanding the problem space, ideating robotic support and participatory co-design of advocacy support robots. The findings highlight the potential of these technologies in providing signposting and acting as a sounding board or study companion, while also addressing limitations in empathic understanding, trust, equity, and accessibility. We discuss ethical considerations, including intersectional biases, the double empathy problem, and the implications of deploying social robots in contexts shaped by structural inequalities. Finally, we offer a set of recommendations and suggestions for future research, rethinking the notion of corrective technological interventions to tools that empower and amplify self-advocacy. 

---
# Llms, Virtual Users, and Bias: Predicting Any Survey Question Without Human Data 

**Authors**: Enzo Sinacola, Arnault Pachot, Thierry Petit  

**Link**: [PDF](https://arxiv.org/pdf/2503.16498)  

**Abstract**: Large Language Models (LLMs) offer a promising alternative to traditional survey methods, potentially enhancing efficiency and reducing costs. In this study, we use LLMs to create virtual populations that answer survey questions, enabling us to predict outcomes comparable to human responses. We evaluate several LLMs-including GPT-4o, GPT-3.5, Claude 3.5-Sonnet, and versions of the Llama and Mistral models-comparing their performance to that of a traditional Random Forests algorithm using demographic data from the World Values Survey (WVS). LLMs demonstrate competitive performance overall, with the significant advantage of requiring no additional training data. However, they exhibit biases when predicting responses for certain religious and population groups, underperforming in these areas. On the other hand, Random Forests demonstrate stronger performance than LLMs when trained with sufficient data. We observe that removing censorship mechanisms from LLMs significantly improves predictive accuracy, particularly for underrepresented demographic segments where censored models struggle. These findings highlight the importance of addressing biases and reconsidering censorship approaches in LLMs to enhance their reliability and fairness in public opinion research. 

---
# Effective Yet Ephemeral Propaganda Defense: There Needs to Be More than One-Shot Inoculation to Enhance Critical Thinking 

**Authors**: Nicolas Hoferer, Kilian Sprenkamp, Dorian Christoph Quelle, Daniel Gordon Jones, Zoya Katashinskaya, Alexandre Bovet, Liudmila Zavolokina  

**Link**: [PDF](https://arxiv.org/pdf/2503.16497)  

**Abstract**: In today's media landscape, propaganda distribution has a significant impact on society. It sows confusion, undermines democratic processes, and leads to increasingly difficult decision-making for news readers. We investigate the lasting effect on critical thinking and propaganda awareness on them when using a propaganda detection and contextualization tool. Building on inoculation theory, which suggests that preemptively exposing individuals to weakened forms of propaganda can improve their resilience against it, we integrate Kahneman's dual-system theory to measure the tools' impact on critical thinking. Through a two-phase online experiment, we measure the effect of several inoculation doses. Our findings show that while the tool increases critical thinking during its use, this increase vanishes without access to the tool. This indicates a single use of the tool does not create a lasting impact. We discuss the implications and propose possible approaches to improve the resilience against propaganda in the long-term. 

---
# The Impact of Generative AI Coding Assistants on Developers Who Are Visually Impaired 

**Authors**: Claudia Flores-Saviaga, Benjamin V. Hanrahan, Kashif Imteyaz, Steven Clarke, Saiph Savage  

**Link**: [PDF](https://arxiv.org/pdf/2503.16491)  

**Abstract**: The rapid adoption of generative AI in software development has impacted the industry, yet its effects on developers with visual impairments remain largely unexplored. To address this gap, we used an Activity Theory framework to examine how developers with visual impairments interact with AI coding assistants. For this purpose, we conducted a study where developers who are visually impaired completed a series of programming tasks using a generative AI coding assistant. We uncovered that, while participants found the AI assistant beneficial and reported significant advantages, they also highlighted accessibility challenges. Specifically, the AI coding assistant often exacerbated existing accessibility barriers and introduced new challenges. For example, it overwhelmed users with an excessive number of suggestions, leading developers who are visually impaired to express a desire for ``AI timeouts.'' Additionally, the generative AI coding assistant made it more difficult for developers to switch contexts between the AI-generated content and their own code. Despite these challenges, participants were optimistic about the potential of AI coding assistants to transform the coding experience for developers with visual impairments. Our findings emphasize the need to apply activity-centered design principles to generative AI assistants, ensuring they better align with user behaviors and address specific accessibility needs. This approach can enable the assistants to provide more intuitive, inclusive, and effective experiences, while also contributing to the broader goal of enhancing accessibility in software development. 

---
# PythonPal: Enhancing Online Programming Education through Chatbot-Driven Personalized Feedback 

**Authors**: Sirinda Palahan  

**Link**: [PDF](https://arxiv.org/pdf/2503.16487)  

**Abstract**: The rise of online programming education has necessitated more effective, personalized interactions, a gap that PythonPal aims to fill through its innovative learning system integrated with a chatbot. This research delves into PythonPal's potential to enhance the online learning experience, especially in contexts with high student-to-teacher ratios where there is a need for personalized feedback. PythonPal's design, featuring modules for conversation, tutorials, and exercises, was evaluated through student interactions and feedback. Key findings reveal PythonPal's proficiency in syntax error recognition and user query comprehension, with its intent classification model showing high accuracy. The system's performance in error feedback, though varied, demonstrates both strengths and areas for enhancement. Student feedback indicated satisfactory query understanding and feedback accuracy but also pointed out the need for faster responses and improved interaction quality. PythonPal's deployment promises to significantly enhance online programming education by providing immediate, personalized feedback and interactive learning experiences, fostering a deeper understanding of programming concepts among students. These benefits mark a step forward in addressing the challenges of distance learning, making programming education more accessible and effective. 

---
# Accodemy: AI Powered Code Learning Platform to Assist Novice Programmers in Overcoming the Fear of Coding 

**Authors**: M.A.F. Aamina, V. Kavishcan, W.M.P.B.B. Jayaratne, K.K.D.S.N. Kannangara, A.A. Aamil, Achini Adikari  

**Link**: [PDF](https://arxiv.org/pdf/2503.16486)  

**Abstract**: Computer programming represents a rapidly evolving and sought-after career path in the 21st century. Nevertheless, novice learners may find the process intimidating for several reasons, such as limited and highly competitive career opportunities, peer and parental pressure for academic success, and course difficulties. These factors frequently contribute to anxiety and eventual dropout as a result of fear. Furthermore, research has demonstrated that beginners are significantly deterred by the fear of failure, which results in programming anxiety and and a sense of being overwhelmed by intricate topics, ultimately leading to dropping out. This project undertakes an exploration beyond the scope of conventional code learning platforms by identifying and utilising effective and personalised strategies of learning. The proposed solution incorporates features such as AI-generated challenging questions, mindfulness quotes, and tips to motivate users, along with an AI chatbot that functions as a motivational aid. In addition, the suggested solution integrates personalized roadmaps and gamification elements to maintain user involvement. The project aims to systematically monitor the progress of novice programmers and enhance their knowledge of coding with a personalised, revised curriculum to help mitigate the fear of coding and boost confidence. 

---
# Optimizing Generative AI's Accuracy and Transparency in Inductive Thematic Analysis: A Human-AI Comparison 

**Authors**: Matthew Nyaaba, Min SungEun, Mary Abiswin Apam, Kwame Owoahene Acheampong, Emmanuel Dwamena  

**Link**: [PDF](https://arxiv.org/pdf/2503.16485)  

**Abstract**: This study explores the use of OpenAI's API for inductive thematic analysis, employing a stepwise strategy to enhance transparency and traceability in GenAI-generated coding. A five-phase analysis and evaluation process were followed. Using the stepwise prompt, GenAI effectively generated codes with supporting statements and references, categorized themes, and developed broader interpretations by linking them to real-world contexts. While GenAI performed at a comparable level to human coders in coding and theming, it exhibited a more generalized and conceptual approach to interpretation, whereas human coders provided more specific, theme-based interpretations. Mapping these processes onto Naeem et al.'s (2023) six-step thematic analysis framework, GenAI covered four out of the six steps, while human coders followed three steps. Although GenAI's coding, theming, and interpretation align with keywording, coding, theming, and interpretation in Naeem et al.'s framework, human coders' interpretations were more closely tied to themes rather than broader conceptualization. This study positions GenAI as a viable tool for conducting inductive thematic analysis with minimal human intervention, offering an efficient and structured approach to qualitative data analysis. Future research should explore the development of specialized prompts that align GenAI's inductive thematic analysis with established qualitative research frameworks. 

---
# AI-Powered Episodic Future Thinking 

**Authors**: Sareh Ahmadi, Michelle Rockwell, Megan Stuart, Allison Tegge, Xuan Wang, Jeffrey Stein, Edward A. Fox  

**Link**: [PDF](https://arxiv.org/pdf/2503.16484)  

**Abstract**: Episodic Future Thinking (EFT) is an intervention that involves vividly imagining personal future events and experiences in detail. It has shown promise as an intervention to reduce delay discounting - the tendency to devalue delayed rewards in favor of immediate gratification - and to promote behavior change in a range of maladaptive health behaviors. We present EFTeacher, an AI chatbot powered by the GPT-4-Turbo large language model, designed to generate EFT cues for users with lifestyle-related conditions. To evaluate the chatbot, we conducted a user study that included usability assessments and user evaluations based on content characteristics questionnaires, followed by semi-structured interviews. The study provides qualitative insights into participants' experiences and interactions with the chatbot and its usability. Our findings highlight the potential application of AI chatbots based on Large Language Models (LLMs) in EFT interventions, and offer design guidelines for future behavior-oriented applications. 

---
# Human Preferences for Constructive Interactions in Language Model Alignment 

**Authors**: Yara Kyrychenko, Jon Roozenbeek, Brandon Davidson, Sander van der Linden, Ramit Debnath  

**Link**: [PDF](https://arxiv.org/pdf/2503.16480)  

**Abstract**: As large language models (LLMs) enter the mainstream, aligning them to foster constructive dialogue rather than exacerbate societal divisions is critical. Using an individualized and multicultural alignment dataset of over 7,500 conversations of individuals from 74 countries engaging with 21 LLMs, we examined how linguistic attributes linked to constructive interactions are reflected in human preference data used for training AI. We found that users consistently preferred well-reasoned and nuanced responses while rejecting those high in personal storytelling. However, users who believed that AI should reflect their values tended to place less preference on reasoning in LLM responses and more on curiosity. Encouragingly, we observed that users could set the tone for how constructive their conversation would be, as LLMs mirrored linguistic attributes, including toxicity, in user queries. 

---
# LeRAAT: LLM-Enabled Real-Time Aviation Advisory Tool 

**Authors**: Marc R. Schlichting, Vale Rasmussen, Heba Alazzeh, Houjun Liu, Kiana Jafari, Amelia F. Hardy, Dylan M. Asmar, Mykel J. Kochenderfer  

**Link**: [PDF](https://arxiv.org/pdf/2503.16477)  

**Abstract**: In aviation emergencies, high-stakes decisions must be made in an instant. Pilots rely on quick access to precise, context-specific information -- an area where emerging tools like large language models (LLMs) show promise in providing critical support. This paper introduces LeRAAT, a framework that integrates LLMs with the X-Plane flight simulator to deliver real-time, context-aware pilot assistance. The system uses live flight data, weather conditions, and aircraft documentation to generate recommendations aligned with aviation best practices and tailored to the particular situation. It employs a Retrieval-Augmented Generation (RAG) pipeline that extracts and synthesizes information from aircraft type-specific manuals, including performance specifications and emergency procedures, as well as aviation regulatory materials, such as FAA directives and standard operating procedures. We showcase the framework in both a virtual reality and traditional on-screen simulation, supporting a wide range of research applications such as pilot training, human factors research, and operational decision support. 

---
# From Voices to Worlds: Developing an AI-Powered Framework for 3D Object Generation in Augmented Reality 

**Authors**: Majid Behravan, Denis Gracanin  

**Link**: [PDF](https://arxiv.org/pdf/2503.16474)  

**Abstract**: This paper presents Matrix, an advanced AI-powered framework designed for real-time 3D object generation in Augmented Reality (AR) environments. By integrating a cutting-edge text-to-3D generative AI model, multilingual speech-to-text translation, and large language models (LLMs), the system enables seamless user interactions through spoken commands. The framework processes speech inputs, generates 3D objects, and provides object recommendations based on contextual understanding, enhancing AR experiences. A key feature of this framework is its ability to optimize 3D models by reducing mesh complexity, resulting in significantly smaller file sizes and faster processing on resource-constrained AR devices. Our approach addresses the challenges of high GPU usage, large model output sizes, and real-time system responsiveness, ensuring a smoother user experience. Moreover, the system is equipped with a pre-generated object repository, further reducing GPU load and improving efficiency. We demonstrate the practical applications of this framework in various fields such as education, design, and accessibility, and discuss future enhancements including image-to-3D conversion, environmental object detection, and multimodal support. The open-source nature of the framework promotes ongoing innovation and its utility across diverse industries. 

---
# Human-AI Interaction Design Standards 

**Authors**: Chaoyi Zhao, Wei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2503.16472)  

**Abstract**: The rapid development of artificial intelligence (AI) has significantly transformed human-computer interactions, making it essential to establish robust design standards to ensure effective, ethical, and human-centered AI (HCAI) solutions. Standards serve as the foundation for the adoption of new technologies, and human-AI interaction (HAII) standards are critical to supporting the industrialization of AI technology by following an HCAI approach. These design standards aim to provide clear principles, requirements, and guidelines for designing, developing, deploying, and using AI systems, enhancing the user experience and performance of AI systems. Despite their importance, the creation and adoption of HCAI-based interaction design standards face challenges, including the absence of universal frameworks, the inherent complexity of HAII, and the ethical dilemmas that arise in such systems. This chapter provides a comparative analysis of HAII versus traditional human-computer interaction (HCI) and outlines guiding principles for HCAI-based design. It explores international, regional, national, and industry standards related to HAII design from an HCAI perspective and reviews design guidelines released by leading companies such as Microsoft, Google, and Apple. Additionally, the chapter highlights tools available for implementing HAII standards and presents case studies of human-centered interaction design for AI systems in diverse fields, including healthcare, autonomous vehicles, and customer service. It further examines key challenges in developing HAII standards and suggests future directions for the field. Emphasizing the importance of ongoing collaboration between AI designers, developers, and experts in human factors and HCI, this chapter stresses the need to advance HCAI-based interaction design standards to ensure human-centered AI solutions across various domains. 

---
# A Review of Brain-Computer Interface Technologies: Signal Acquisition Methods and Interaction Paradigms 

**Authors**: Yifan Wang, Cheng Jiang, Chenzhong Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.16471)  

**Abstract**: Brain-Computer Interface (BCI) technology facilitates direct communication between the human brain and external devices, representing a substantial advancement in human-machine interaction. This review provides an in-depth analysis of various BCI paradigms, including classic paradigms, current classifications, and hybrid paradigms, each with distinct characteristics and applications. Additionally, we explore a range of signal acquisition methods, classified into non-implantation, intervention, and implantation techniques, elaborating on their principles and recent advancements. By examining the interdependence between paradigms and signal acquisition technologies, this review offers a comprehensive perspective on how innovations in one domain propel progress in the other. The goal is to present insights into the future development of more efficient, user-friendly, and versatile BCI systems, emphasizing the synergy between paradigm design and signal acquisition techniques and their potential to transform the field. 

---
# Towards properly implementing Theory of Mind in AI systems: An account of four misconceptions 

**Authors**: Ramira van der Meulen, Rineke Verbrugge, Max van Duijn  

**Link**: [PDF](https://arxiv.org/pdf/2503.16468)  

**Abstract**: The search for effective collaboration between humans and computer systems is one of the biggest challenges in Artificial Intelligence. One of the more effective mechanisms that humans use to coordinate with one another is theory of mind (ToM). ToM can be described as the ability to `take someone else's perspective and make estimations of their beliefs, desires and intentions, in order to make sense of their behaviour and attitudes towards the world'. If leveraged properly, this skill can be very useful in Human-AI collaboration.
This introduces the question how we implement ToM when building an AI system. Humans and AI Systems work quite differently, and ToM is a multifaceted concept, each facet rooted in different research traditions across the cognitive and developmental sciences. We observe that researchers from artificial intelligence and the computing sciences, ourselves included, often have difficulties finding their way in the ToM literature. In this paper, we identify four common misconceptions around ToM that we believe should be taken into account when developing an AI system. We have hyperbolised these misconceptions for the sake of the argument, but add nuance in their discussion.
The misconceptions we discuss are:
(1) "Humans Use a ToM Module, So AI Systems Should As Well".
(2) "Every Social Interaction Requires (Advanced) ToM".
(3) "All ToM is the Same".
(4) "Current Systems Already Have ToM".
After discussing the misconception, we end each section by providing tentative guidelines on how the misconception can be overcome. 

---
# Enhancing Explainability with Multimodal Context Representations for Smarter Robots 

**Authors**: Anargh Viswanath, Lokesh Veeramacheneni, Hendrik Buschmeier  

**Link**: [PDF](https://arxiv.org/pdf/2503.16467)  

**Abstract**: Artificial Intelligence (AI) has significantly advanced in recent years, driving innovation across various fields, especially in robotics. Even though robots can perform complex tasks with increasing autonomy, challenges remain in ensuring explainability and user-centered design for effective interaction. A key issue in Human-Robot Interaction (HRI) is enabling robots to effectively perceive and reason over multimodal inputs, such as audio and vision, to foster trust and seamless collaboration. In this paper, we propose a generalized and explainable multimodal framework for context representation, designed to improve the fusion of speech and vision modalities. We introduce a use case on assessing 'Relevance' between verbal utterances from the user and visual scene perception of the robot. We present our methodology with a Multimodal Joint Representation module and a Temporal Alignment module, which can allow robots to evaluate relevance by temporally aligning multimodal inputs. Finally, we discuss how the proposed framework for context representation can help with various aspects of explainability in HRI. 

---
# ACE, Action and Control via Explanations: A Proposal for LLMs to Provide Human-Centered Explainability for Multimodal AI Assistants 

**Authors**: Elizabeth Anne Watkins, Emanuel Moss, Ramesh Manuvinakurike, Meng Shi, Richard Beckwith, Giuseppe Raffa  

**Link**: [PDF](https://arxiv.org/pdf/2503.16466)  

**Abstract**: In this short paper we address issues related to building multimodal AI systems for human performance support in manufacturing domains. We make two contributions: we first identify challenges of participatory design and training of such systems, and secondly, to address such challenges, we propose the ACE paradigm: "Action and Control via Explanations". Specifically, we suggest that LLMs can be used to produce explanations in the form of human interpretable "semantic frames", which in turn enable end users to provide data the AI system needs to align its multimodal models and representations, including computer vision, automatic speech recognition, and document inputs. ACE, by using LLMs to "explain" using semantic frames, will help the human and the AI system to collaborate, together building a more accurate model of humans activities and behaviors, and ultimately more accurate predictive outputs for better task support, and better outcomes for human users performing manual tasks. 

---
# OS-Kairos: Adaptive Interaction for MLLM-Powered GUI Agents 

**Authors**: Pengzhou Cheng, Zheng Wu, Zongru Wu, Aston Zhang, Zhuosheng Zhang, Gongshen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.16465)  

**Abstract**: Autonomous graphical user interface (GUI) agents powered by multimodal large language models have shown great promise. However, a critical yet underexplored issue persists: over-execution, where the agent executes tasks in a fully autonomous way, without adequate assessment of its action confidence to compromise an adaptive human-agent collaboration. This poses substantial risks in complex scenarios, such as those involving ambiguous user instructions, unexpected interruptions, and environmental hijacks. To address the issue, we introduce OS-Kairos, an adaptive GUI agent capable of predicting confidence levels at each interaction step and efficiently deciding whether to act autonomously or seek human intervention. OS-Kairos is developed through two key mechanisms: (i) collaborative probing that annotates confidence scores at each interaction step; (ii) confidence-driven interaction that leverages these confidence scores to elicit the ability of adaptive interaction. Experimental results show that OS-Kairos substantially outperforms existing models on our curated dataset featuring complex scenarios, as well as on established benchmarks such as AITZ and Meta-GUI, with 24.59\%$\sim$87.29\% improvements in task success rate. OS-Kairos facilitates an adaptive human-agent collaboration, prioritizing effectiveness, generality, scalability, and efficiency for real-world GUI interaction. The dataset and codes are available at this https URL. 

---
# Human-Centered AI in Multidisciplinary Medical Discussions: Evaluating the Feasibility of a Chat-Based Approach to Case Assessment 

**Authors**: Shinnosuke Sawano, Satoshi Kodera  

**Link**: [PDF](https://arxiv.org/pdf/2503.16464)  

**Abstract**: In this study, we investigate the feasibility of using a human-centered artificial intelligence (AI) chat platform where medical specialists collaboratively assess complex cases. As the target population for this platform, we focus on patients with cardiovascular diseases who are in a state of multimorbidity, that is, suffering from multiple chronic conditions. We evaluate simulated cases with multiple diseases using a chat application by collaborating with physicians to assess feasibility, efficiency gains through AI utilization, and the quantification of discussion content. We constructed simulated cases based on past case reports, medical errors reports and complex cases of cardiovascular diseases experienced by the physicians. The analysis of discussions across five simulated cases demonstrated a significant reduction in the time required for summarization using AI, with an average reduction of 79.98\%. Additionally, we examined hallucination rates in AI-generated summaries used in multidisciplinary medical discussions. The overall hallucination rate ranged from 1.01\% to 5.73\%, with an average of 3.62\%, whereas the harmful hallucination rate varied from 0.00\% to 2.09\%, with an average of 0.49\%. Furthermore, morphological analysis demonstrated that multidisciplinary assessments enabled a more complex and detailed representation of medical knowledge compared with single physician assessments. We examined structural differences between multidisciplinary and single physician assessments using centrality metrics derived from the knowledge graph. In this study, we demonstrated that AI-assisted summarization significantly reduced the time required for medical discussions while maintaining structured knowledge representation. These findings can support the feasibility of AI-assisted chat-based discussions as a human-centered approach to multidisciplinary medical decision-making. 

---
# Rank-O-ToM: Unlocking Emotional Nuance Ranking to Enhance Affective Theory-of-Mind 

**Authors**: JiHyun Kim, JuneHyoung Kwon, MiHyeon Kim, Eunju Lee, YoungBin Kim  

**Link**: [PDF](https://arxiv.org/pdf/2503.16461)  

**Abstract**: Facial Expression Recognition (FER) plays a foundational role in enabling AI systems to interpret emotional nuances, a critical aspect of affective Theory of Mind (ToM). However, existing models often struggle with poor calibration and a limited capacity to capture emotional intensity and complexity. To address this, we propose Ranking the Emotional Nuance for Theory of Mind (Rank-O-ToM), a framework that leverages ordinal ranking to align confidence levels with the emotional spectrum. By incorporating synthetic samples reflecting diverse affective complexities, Rank-O-ToM enhances the nuanced understanding of emotions, advancing AI's ability to reason about affective states. 

---
# Beyond Final Answers: Evaluating Large Language Models for Math Tutoring 

**Authors**: Adit Gupta, Jennifer Reddig, Tommaso Calo, Daniel Weitekamp, Christopher J. MacLellan  

**Link**: [PDF](https://arxiv.org/pdf/2503.16460)  

**Abstract**: Researchers have made notable progress in applying Large Language Models (LLMs) to solve math problems, as demonstrated through efforts like GSM8k, ProofNet, AlphaGeometry, and MathOdyssey. This progress has sparked interest in their potential use for tutoring students in mathematics. However, the reliability of LLMs in tutoring contexts -- where correctness and instructional quality are crucial -- remains underexplored. Moreover, LLM problem-solving capabilities may not necessarily translate into effective tutoring support for students. In this work, we present two novel approaches to evaluate the correctness and quality of LLMs in math tutoring contexts. The first approach uses an intelligent tutoring system for college algebra as a testbed to assess LLM problem-solving capabilities. We generate benchmark problems using the tutor, prompt a diverse set of LLMs to solve them, and compare the solutions to those generated by the tutor. The second approach evaluates LLM as tutors rather than problem solvers. We employ human evaluators, who act as students seeking tutoring support from each LLM. We then assess the quality and correctness of the support provided by the LLMs via a qualitative coding process. We applied these methods to evaluate several ChatGPT models, including 3.5 Turbo, 4, 4o, o1-mini, and o1-preview. Our findings show that when used as problem solvers, LLMs generate correct final answers for 85.5% of the college algebra problems tested. When employed interactively as tutors, 90% of LLM dialogues show high-quality instructional support; however, many contain errors -- only 56.6% are entirely correct. We conclude that, despite their potential, LLMs are not yet suitable as intelligent tutors for math without human oversight or additional mechanisms to ensure correctness and quality. 

---
# Integrating Personality into Digital Humans: A Review of LLM-Driven Approaches for Virtual Reality 

**Authors**: Iago Alves Brito, Julia Soares Dollis, Fernanda Bufon Färber, Pedro Schindler Freire Brasil Ribeiro, Rafael Teixeira Sousa, Arlindo Rodrigues Galvão Filho  

**Link**: [PDF](https://arxiv.org/pdf/2503.16457)  

**Abstract**: The integration of large language models (LLMs) into virtual reality (VR) environments has opened new pathways for creating more immersive and interactive digital humans. By leveraging the generative capabilities of LLMs alongside multimodal outputs such as facial expressions and gestures, virtual agents can simulate human-like personalities and emotions, fostering richer and more engaging user experiences. This paper provides a comprehensive review of methods for enabling digital humans to adopt nuanced personality traits, exploring approaches such as zero-shot, few-shot, and fine-tuning. Additionally, it highlights the challenges of integrating LLM-driven personality traits into VR, including computational demands, latency issues, and the lack of standardized evaluation frameworks for multimodal interactions. By addressing these gaps, this work lays a foundation for advancing applications in education, therapy, and gaming, while fostering interdisciplinary collaboration to redefine human-computer interaction in VR. 

---
# Position: Beyond Assistance -- Reimagining LLMs as Ethical and Adaptive Co-Creators in Mental Health Care 

**Authors**: Abeer Badawi, Md Tahmid Rahman Laskar, Jimmy Xiangji Huang, Shaina Raza, Elham Dolatabadi  

**Link**: [PDF](https://arxiv.org/pdf/2503.16456)  

**Abstract**: This position paper argues for a fundamental shift in how Large Language Models (LLMs) are integrated into the mental health care domain. We advocate for their role as co-creators rather than mere assistive tools. While LLMs have the potential to enhance accessibility, personalization, and crisis intervention, their adoption remains limited due to concerns about bias, evaluation, over-reliance, dehumanization, and regulatory uncertainties. To address these challenges, we propose two structured pathways: SAFE-i (Supportive, Adaptive, Fair, and Ethical Implementation) Guidelines for ethical and responsible deployment, and HAAS-e (Human-AI Alignment and Safety Evaluation) Framework for multidimensional, human-centered assessment. SAFE-i provides a blueprint for data governance, adaptive model engineering, and real-world integration, ensuring LLMs align with clinical and ethical standards. HAAS-e introduces evaluation metrics that go beyond technical accuracy to measure trustworthiness, empathy, cultural sensitivity, and actionability. We call for the adoption of these structured approaches to establish a responsible and scalable model for LLM-driven mental health support, ensuring that AI complements-rather than replaces-human expertise. 

---
# Bridging Structural Dynamics and Biomechanics: Human Motion Estimation through Footstep-Induced Floor Vibrations 

**Authors**: Yiwen Dong, Jessica Rose, Hae Young Noh  

**Link**: [PDF](https://arxiv.org/pdf/2503.16455)  

**Abstract**: Quantitative estimation of human joint motion in daily living spaces is essential for early detection and rehabilitation tracking of neuromusculoskeletal disorders (e.g., Parkinson's) and mitigating trip and fall risks for older adults. Existing approaches involve monitoring devices such as cameras, wearables, and pressure mats, but have operational constraints such as direct line-of-sight, carrying devices, and dense deployment. To overcome these limitations, we leverage gait-induced floor vibration to estimate lower-limb joint motion (e.g., ankle, knee, and hip flexion angles), allowing non-intrusive and contactless gait health monitoring in people's living spaces. To overcome the high uncertainty in lower-limb movement given the limited information provided by the gait-induced floor vibrations, we formulate a physics-informed graph to integrate domain knowledge of gait biomechanics and structural dynamics into the model. Specifically, different types of nodes represent heterogeneous information from joint motions and floor vibrations; Their connecting edges represent the physiological relationships between joints and forces governed by gait biomechanics, as well as the relationships between forces and floor responses governed by the structural dynamics. As a result, our model poses physical constraints to reduce uncertainty while allowing information sharing between the body and the floor to make more accurate predictions. We evaluate our approach with 20 participants through a real-world walking experiment. We achieved an average of 3.7 degrees of mean absolute error in estimating 12 joint flexion angles (38% error reduction from baseline), which is comparable to the performance of cameras and wearables in current medical practices. 

---
# An Audio-Visual Fusion Emotion Generation Model Based on Neuroanatomical Alignment 

**Authors**: Haidong Wang, Qia Shan, JianHua Zhang, PengFei Xiao, Ao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.16454)  

**Abstract**: In the field of affective computing, traditional methods for generating emotions predominantly rely on deep learning techniques and large-scale emotion datasets. However, deep learning techniques are often complex and difficult to interpret, and standardizing large-scale emotional datasets are difficult and costly to establish. To tackle these challenges, we introduce a novel framework named Audio-Visual Fusion for Brain-like Emotion Learning(AVF-BEL). In contrast to conventional brain-inspired emotion learning methods, this approach improves the audio-visual emotion fusion and generation model through the integration of modular components, thereby enabling more lightweight and interpretable emotion learning and generation processes. The framework simulates the integration of the visual, auditory, and emotional pathways of the brain, optimizes the fusion of emotional features across visual and auditory modalities, and improves upon the traditional Brain Emotional Learning (BEL) model. The experimental results indicate a significant improvement in the similarity of the audio-visual fusion emotion learning generation model compared to single-modality visual and auditory emotion learning and generation model. Ultimately, this aligns with the fundamental phenomenon of heightened emotion generation facilitated by the integrated impact of visual and auditory stimuli. This contribution not only enhances the interpretability and efficiency of affective intelligence but also provides new insights and pathways for advancing affective computing technology. Our source code can be accessed here: this https URL}{this https URL. 

---
# Towards Biomarker Discovery for Early Cerebral Palsy Detection: Evaluating Explanations Through Kinematic Perturbations 

**Authors**: Kimji N. Pellano, Inga Strümke, Daniel Groos, Lars Adde, Pål Haugen, Espen Alexander F. Ihlen  

**Link**: [PDF](https://arxiv.org/pdf/2503.16452)  

**Abstract**: Cerebral Palsy (CP) is a prevalent motor disability in children, for which early detection can significantly improve treatment outcomes. While skeleton-based Graph Convolutional Network (GCN) models have shown promise in automatically predicting CP risk from infant videos, their "black-box" nature raises concerns about clinical explainability. To address this, we introduce a perturbation framework tailored for infant movement features and use it to compare two explainable AI (XAI) methods: Class Activation Mapping (CAM) and Gradient-weighted Class Activation Mapping (Grad-CAM). First, we identify significant and non-significant body keypoints in very low- and very high-risk infant video snippets based on the XAI attribution scores. We then conduct targeted velocity and angular perturbations, both individually and in combination, on these keypoints to assess how the GCN model's risk predictions change. Our results indicate that velocity-driven features of the arms, hips, and legs have a dominant influence on CP risk predictions, while angular perturbations have a more modest impact. Furthermore, CAM and Grad-CAM show partial convergence in their explanations for both low- and high-risk CP groups. Our findings demonstrate the use of XAI-driven movement analysis for early CP prediction and offer insights into potential movement-based biomarker discovery that warrant further clinical validation. 

---
# Think-Then-React: Towards Unconstrained Human Action-to-Reaction Generation 

**Authors**: Wenhui Tan, Boyuan Li, Chuhao Jin, Wenbing Huang, Xiting Wang, Ruihua Song  

**Link**: [PDF](https://arxiv.org/pdf/2503.16451)  

**Abstract**: Modeling human-like action-to-reaction generation has significant real-world applications, like human-robot interaction and games. Despite recent advancements in single-person motion generation, it is still challenging to well handle action-to-reaction generation, due to the difficulty of directly predicting reaction from action sequence without prompts, and the absence of a unified representation that effectively encodes multi-person motion. To address these challenges, we introduce Think-Then-React (TTR), a large language-model-based framework designed to generate human-like reactions. First, with our fine-grained multimodal training strategy, TTR is capable to unify two processes during inference: a thinking process that explicitly infers action intentions and reasons corresponding reaction description, which serve as semantic prompts, and a reacting process that predicts reactions based on input action and the inferred semantic prompts. Second, to effectively represent multi-person motion in language models, we propose a unified motion tokenizer by decoupling egocentric pose and absolute space features, which effectively represents action and reaction motion with same encoding. Extensive experiments demonstrate that TTR outperforms existing baselines, achieving significant improvements in evaluation metrics, such as reducing FID from 3.988 to 1.942. 

---
# Mitigating the Uncanny Valley Effect in Hyper-Realistic Robots: A Student-Centered Study on LLM-Driven Conversations 

**Authors**: Hangyeol Kang, Thiago Freitas dos Santos, Maher Ben Moussa, Nadia Magnenat-Thalmann  

**Link**: [PDF](https://arxiv.org/pdf/2503.16449)  

**Abstract**: The uncanny valley effect poses a significant challenge in the development and acceptance of hyper-realistic social robots. This study investigates whether advanced conversational capabilities powered by large language models (LLMs) can mitigate this effect in highly anthropomorphic robots. We conducted a user study with 80 participants interacting with Nadine, a hyper-realistic humanoid robot equipped with LLM-driven communication skills. Through pre- and post-interaction surveys, we assessed changes in perceptions of uncanniness, conversational quality, and overall user experience. Our findings reveal that LLM-enhanced interactions significantly reduce feelings of eeriness while fostering more natural and engaging conversations. Additionally, we identify key factors influencing user acceptance, including conversational naturalness, human-likeness, and interestingness. Based on these insights, we propose design recommendations to enhance the appeal and acceptability of hyper-realistic robots in social contexts. This research contributes to the growing field of human-robot interaction by offering empirical evidence on the potential of LLMs to bridge the uncanny valley, with implications for the future development of social robots. 

---
# FINCH: Locally Visualizing Higher-Order Feature Interactions in Black Box Models 

**Authors**: Anna Kleinau, Bernhard Preim, Monique Meuschke  

**Link**: [PDF](https://arxiv.org/pdf/2503.16445)  

**Abstract**: In an era where black-box AI models are integral to decision-making across industries, robust methods for explaining these models are more critical than ever. While these models leverage complex feature interplay for accurate predictions, most explanation methods only assign relevance to individual features. There is a research gap in methods that effectively illustrate interactions between features, especially in visualizing higher-order interactions involving multiple features, which challenge conventional representation methods. To address this challenge in local explanations focused on individual instances, we employ a visual, subset-based approach to reveal relevant feature interactions. Our visual analytics tool FINCH uses coloring and highlighting techniques to create intuitive, human-centered visualizations, and provides additional views that enable users to calibrate their trust in the model and explanations. We demonstrate FINCH in multiple case studies, demonstrating its generalizability, and conducted an extensive human study with machine learning experts to highlight its helpfulness and usability. With this approach, FINCH allows users to visualize feature interactions involving any number of features locally. 

---
# Conversational Explanations: Discussing Explainable AI with Non-AI Experts 

**Authors**: Tong Zhang, Mengao Zhang, Wei Yan Low, X. Jessie Yang, Boyang Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.16444)  

**Abstract**: Explainable AI (XAI) aims to provide insights into the decisions made by AI models. To date, most XAI approaches provide only one-time, static explanations, which cannot cater to users' diverse knowledge levels and information needs. Conversational explanations have been proposed as an effective method to customize XAI explanations. However, building conversational explanation systems is hindered by the scarcity of training data. Training with synthetic data faces two main challenges: lack of data diversity and hallucination in the generated data. To alleviate these issues, we introduce a repetition penalty to promote data diversity and exploit a hallucination detector to filter out untruthful synthetic conversation turns. We conducted both automatic and human evaluations on the proposed system, fEw-shot Multi-round ConvErsational Explanation (EMCEE). For automatic evaluation, EMCEE achieves relative improvements of 81.6% in BLEU and 80.5% in ROUGE compared to the baselines. EMCEE also mitigates the degeneration of data quality caused by training on synthetic data. In human evaluations (N=60), EMCEE outperforms baseline models and the control group in improving users' comprehension, acceptance, trust, and collaboration with static explanations by large margins. Through a fine-grained analysis of model responses, we further demonstrate that training on self-generated synthetic data improves the model's ability to generate more truthful and understandable answers, leading to better user interactions. To the best of our knowledge, this is the first conversational explanation method that can answer free-form user questions following static explanations. 

---
# Situational Agency: The Framework for Designing Behavior in Agent-based art 

**Authors**: Ary-Yue Huang, Varvara Guljajeva  

**Link**: [PDF](https://arxiv.org/pdf/2503.16442)  

**Abstract**: In the context of artificial life art and agent-based art, this paper draws on Simon Penny's {\itshape Aesthetic of Behavior} theory and Sofian Audry's discussions on behavior computation to examine how artists design agent behaviors and the ensuing aesthetic experiences. We advocate for integrating the environment in which agents operate as the context for behavioral design, positing that the environment emerges through continuous interactions among agents, audiences, and other entities, forming an evolving network of meanings generated by these interactions. Artists create contexts by deploying and guiding these computational systems, audience participation, and agent behaviors through artist strategies. This framework is developed by analysing two categories of agent-based artworks, exploring the intersection of computational systems, audience participation, and artistic strategies in creating aesthetic experiences. This paper seeks to provide a contextual foundation and framework for designing agents' behaviors by conducting a comparative study focused on behavioural design strategies by the artists. 

---
# Safe and Efficient Social Navigation through Explainable Safety Regions Based on Topological Features 

**Authors**: Victor Toscano-Duran, Sara Narteni, Alberto Carlevaro, Rocio Gonzalez-Diaz, Maurizio Mongelli, Jerome Guzzi  

**Link**: [PDF](https://arxiv.org/pdf/2503.16441)  

**Abstract**: The recent adoption of artificial intelligence (AI) in robotics has driven the development of algorithms that enable autonomous systems to adapt to complex social environments. In particular, safe and efficient social navigation is a key challenge, requiring AI not only to avoid collisions and deadlocks but also to interact intuitively and predictably with its surroundings. To date, methods based on probabilistic models and the generation of conformal safety regions have shown promising results in defining safety regions with a controlled margin of error, primarily relying on classification approaches and explicit rules to describe collision-free navigation conditions.
This work explores how topological features contribute to explainable safety regions in social navigation. Instead of using behavioral parameters, we leverage topological data analysis to classify and characterize different simulation behaviors. First, we apply global rule-based classification to distinguish between safe (collision-free) and unsafe scenarios based on topological properties. Then, we define safety regions, $S_\varepsilon$, in the topological feature space, ensuring a maximum classification error of $\varepsilon$. These regions are built with adjustable SVM classifiers and order statistics, providing robust decision boundaries. Local rules extracted from these regions enhance interpretability, keeping the decision-making process transparent.
Our approach initially separates simulations with and without collisions, outperforming methods that not incorporate topological features. It offers a deeper understanding of robot interactions within a navigable space. We further refine safety regions to ensure deadlock-free simulations and integrate both aspects to define a compliant simulation space that guarantees safe and efficient navigation. 

---
# Cause-effect perception in an object place task 

**Authors**: Nikolai Bahr, Christoph Zetzsche, Jaime Maldonado, Kerstin Schill  

**Link**: [PDF](https://arxiv.org/pdf/2503.16440)  

**Abstract**: Algorithmic causal discovery is based on formal reasoning and provably converges toward the optimal solution. However, since some of the underlying assumptions are often not met in practice no applications for autonomous everyday life competence are yet available. Humans on the other hand possess full everyday competence and develop cognitive models in a data efficient manner with the ability to transfer knowledge between and to new situations. Here we investigate the causal discovery capabilities of humans in an object place task in virtual reality (VR) with haptic feedback and compare the results to the state of the art causal discovery algorithms FGES, PC and FCI. In addition we use the algorithms to analyze causal relations between sensory information and the kinematic parameters of human behavior.
Our findings show that the majority of participants were able to determine which variables are causally related. This is in line with causal discovery algorithms like PC, which recover causal dependencies in the first step. However, unlike such algorithms which can identify causes and effects in our test configuration, humans are unsure in determining a causal direction. Regarding the relation between the sensory information provided to the participants and their placing actions (i.e. their kinematic parameters) the data yields a surprising dissociation of the subjects knowledge and the sensorimotor level. Knowledge of the cause-effect pairs, though undirected, should suffice to improve subject's movements. Yet a detailed causal analysis provides little evidence for any such influence. This, together with the reports of the participants, implies that instead of exploiting their consciously perceived information they leave it to the sensorimotor level to control the movement. 

---
# DreamLLM-3D: Affective Dream Reliving using Large Language Model and 3D Generative AI 

**Authors**: Pinyao Liu, Keon Ju Lee, Alexander Steinmaurer, Claudia Picard-Deland, Michelle Carr, Alexandra Kitson  

**Link**: [PDF](https://arxiv.org/pdf/2503.16439)  

**Abstract**: We present DreamLLM-3D, a composite multimodal AI system behind an immersive art installation for dream re-experiencing. It enables automated dream content analysis for immersive dream-reliving, by integrating a Large Language Model (LLM) with text-to-3D Generative AI. The LLM processes voiced dream reports to identify key dream entities (characters and objects), social interaction, and dream sentiment. The extracted entities are visualized as dynamic 3D point clouds, with emotional data influencing the color and soundscapes of the virtual dream environment. Additionally, we propose an experiential AI-Dreamworker Hybrid paradigm. Our system and paradigm could potentially facilitate a more emotionally engaging dream-reliving experience, enhancing personal insights and creativity. 

---
# Haunted House: A text-based game for comparing the flexibility of mental models in humans and LLMs 

**Authors**: Brett Puppart, Paul-Henry Paltmann, Jaan Aru  

**Link**: [PDF](https://arxiv.org/pdf/2503.16437)  

**Abstract**: This study introduces "Haunted House" a novel text-based game designed to compare the performance of humans and large language models (LLMs) in model-based reasoning. Players must escape from a house containing nine rooms in a 3x3 grid layout while avoiding the ghost. They are guided by verbal clues that they get each time they move. In Study 1, the results from 98 human participants revealed a success rate of 31.6%, significantly outperforming seven state-of-the-art LLMs tested. Out of 140 attempts across seven LLMs, only one attempt resulted in a pass by Claude 3 Opus. Preliminary results suggested that GPT o3-mini-high performance might be higher, but not at the human level. Further analysis of 29 human participants' moves in Study 2 indicated that LLMs frequently struggled with random and illogical moves, while humans exhibited such errors less frequently. Our findings suggest that current LLMs encounter difficulties in tasks that demand active model-based reasoning, offering inspiration for future benchmarks. 

---
# Enhancing Human-Robot Collaboration through Existing Guidelines: A Case Study Approach 

**Authors**: Yutaka Matsubara, Akihisa Morikawa, Daichi Mizuguchi, Kiyoshi Fujiwara  

**Link**: [PDF](https://arxiv.org/pdf/2503.16436)  

**Abstract**: As AI systems become more prevalent, concerns about their development, operation, and societal impact intensify. Establishing ethical, social, and safety standards amidst evolving AI capabilities poses significant challenges. Global initiatives are underway to establish guidelines for AI system development and operation. With the increasing use of collaborative human-AI task execution, it's vital to continuously adapt AI systems to meet user and environmental needs. Failure to synchronize AI evolution with changes in users and the environment could result in ethical and safety issues. This paper evaluates the applicability of existing guidelines in human-robot collaborative systems, assesses their effectiveness, and discusses limitations. Through a case study, we examine whether our target system meets requirements outlined in existing guidelines and propose improvements to enhance human-robot interactions. Our contributions provide insights into interpreting and applying guidelines, offer concrete examples of system enhancement, and highlight their applicability and limitations. We believe these contributions will stimulate discussions and influence system assurance and certification in future AI-infused critical systems. 

---
# Interactive Sketchpad: An Interactive Multimodal System for Collaborative, Visual Problem-Solving 

**Authors**: Steven-Shine Chen, Jimin Lee, Paul Pu Liang  

**Link**: [PDF](https://arxiv.org/pdf/2503.16434)  

**Abstract**: Humans have long relied on visual aids like sketches and diagrams to support reasoning and problem-solving. Visual tools, like auxiliary lines in geometry or graphs in calculus, are essential for understanding complex ideas. However, many tutoring systems remain text-based, providing feedback only through natural language. Leveraging recent advances in Large Multimodal Models (LMMs), this paper introduces Interactive Sketchpad, a tutoring system that combines language-based explanations with interactive visualizations to enhance learning. Built on a pre-trained LMM, Interactive Sketchpad is fine-tuned to provide step-by-step guidance in both text and visuals, enabling natural multimodal interaction with the student. Accurate and robust diagrams are generated by incorporating code execution into the reasoning process. User studies conducted on math problems such as geometry, calculus, and trigonometry demonstrate that Interactive Sketchpad leads to improved task comprehension, problem-solving accuracy, and engagement levels, highlighting its potential for transforming educational technologies. 

---
# Multimodal Transformer Models for Turn-taking Prediction: Effects on Conversational Dynamics of Human-Agent Interaction during Cooperative Gameplay 

**Authors**: Young-Ho Bae, Casey C. Bennett  

**Link**: [PDF](https://arxiv.org/pdf/2503.16432)  

**Abstract**: This study investigates multimodal turn-taking prediction within human-agent interactions (HAI), particularly focusing on cooperative gaming environments. It comprises both model development and subsequent user study, aiming to refine our understanding and improve conversational dynamics in spoken dialogue systems (SDSs). For the modeling phase, we introduce a novel transformer-based deep learning (DL) model that simultaneously integrates multiple modalities - text, vision, audio, and contextual in-game data to predict turn-taking events in real-time. Our model employs a Crossmodal Transformer architecture to effectively fuse information from these diverse modalities, enabling more comprehensive turn-taking predictions. The model demonstrates superior performance compared to baseline models, achieving 87.3% accuracy and 83.0% macro F1 score. A human user study was then conducted to empirically evaluate the turn-taking DL model in an interactive scenario with a virtual avatar while playing the game "Dont Starve Together", comparing a control condition without turn-taking prediction (n=20) to an experimental condition with our model deployed (n=40). Both conditions included a mix of English and Korean speakers, since turn-taking cues are known to vary by culture. We then analyzed the interaction quality, examining aspects such as utterance counts, interruption frequency, and participant perceptions of the avatar. Results from the user study suggest that our multimodal turn-taking model not only enhances the fluidity and naturalness of human-agent conversations, but also maintains a balanced conversational dynamic without significantly altering dialogue frequency. The study provides in-depth insights into the influence of turn-taking abilities on user perceptions and interaction quality, underscoring the potential for more contextually adaptive and responsive conversational agents. 

---
# OpenAI's Approach to External Red Teaming for AI Models and Systems 

**Authors**: Lama Ahmad, Sandhini Agarwal, Michael Lampe, Pamela Mishkin  

**Link**: [PDF](https://arxiv.org/pdf/2503.16431)  

**Abstract**: Red teaming has emerged as a critical practice in assessing the possible risks of AI models and systems. It aids in the discovery of novel risks, stress testing possible gaps in existing mitigations, enriching existing quantitative safety metrics, facilitating the creation of new safety measurements, and enhancing public trust and the legitimacy of AI risk assessments. This white paper describes OpenAI's work to date in external red teaming and draws some more general conclusions from this work. We describe the design considerations underpinning external red teaming, which include: selecting composition of red team, deciding on access levels, and providing guidance required to conduct red teaming. Additionally, we show outcomes red teaming can enable such as input into risk assessment and automated evaluations. We also describe the limitations of external red teaming, and how it can fit into a broader range of AI model and system evaluations. Through these contributions, we hope that AI developers and deployers, evaluation creators, and policymakers will be able to better design red teaming campaigns and get a deeper look into how external red teaming can fit into model deployment and evaluation processes. These methods are evolving and the value of different methods continues to shift as the ecosystem around red teaming matures and models themselves improve as tools for red teaming. 

---
# CLIP-PING: Boosting Lightweight Vision-Language Models with Proximus Intrinsic Neighbors Guidance 

**Authors**: Chu Myaet Thwal, Ye Lin Tun, Minh N. H. Nguyen, Eui-Nam Huh, Choong Seon Hong  

**Link**: [PDF](https://arxiv.org/pdf/2412.03871)  

**Abstract**: Beyond the success of Contrastive Language-Image Pre-training (CLIP), recent trends mark a shift toward exploring the applicability of lightweight vision-language models for resource-constrained scenarios. These models often deliver suboptimal performance when relying solely on a single image-text contrastive learning objective, spotlighting the need for more effective training mechanisms that guarantee robust cross-modal feature alignment. In this work, we propose CLIP-PING: Contrastive Language-Image Pre-training with Proximus Intrinsic Neighbors Guidance, a novel yet simple and efficient training paradigm designed to boost the performance of lightweight vision-language models with minimal computational overhead and lower data demands. CLIP-PING bootstraps unimodal features extracted from arbitrary pre-trained encoders to obtain intrinsic guidance of proximus neighbor samples, i.e., nearest-neighbor (NN) and cross nearest-neighbor (XNN). We find that extra contrastive supervision from these neighbors substantially boosts cross-modal alignment, enabling lightweight models to learn more generic features with rich semantic diversity. Extensive experiments reveal that CLIP-PING notably surpasses its peers in zero-shot generalization and cross-modal retrieval tasks. Specifically, a 5.5% gain on zero-shot ImageNet1K classification with 10.7% (I2T) and 5.7% (T2I) on Flickr30K retrieval, compared to the original CLIP when using ViT-XS image encoder trained on 3 million (image, text) pairs. Moreover, CLIP-PING showcases a strong transferability under the linear evaluation protocol across several downstream tasks. 

---
# OnDev-LCT: On-Device Lightweight Convolutional Transformers towards federated learning 

**Authors**: Chu Myaet Thwal, Minh N.H. Nguyen, Ye Lin Tun, Seong Tae Kim, My T. Thai, Choong Seon Hong  

**Link**: [PDF](https://arxiv.org/pdf/2401.11652)  

**Abstract**: Federated learning (FL) has emerged as a promising approach to collaboratively train machine learning models across multiple edge devices while preserving privacy. The success of FL hinges on the efficiency of participating models and their ability to handle the unique challenges of distributed learning. While several variants of Vision Transformer (ViT) have shown great potential as alternatives to modern convolutional neural networks (CNNs) for centralized training, the unprecedented size and higher computational demands hinder their deployment on resource-constrained edge devices, challenging their widespread application in FL. Since client devices in FL typically have limited computing resources and communication bandwidth, models intended for such devices must strike a balance between model size, computational efficiency, and the ability to adapt to the diverse and non-IID data distributions encountered in FL. To address these challenges, we propose OnDev-LCT: Lightweight Convolutional Transformers for On-Device vision tasks with limited training data and resources. Our models incorporate image-specific inductive biases through the LCT tokenizer by leveraging efficient depthwise separable convolutions in residual linear bottleneck blocks to extract local features, while the multi-head self-attention (MHSA) mechanism in the LCT encoder implicitly facilitates capturing global representations of images. Extensive experiments on benchmark image datasets indicate that our models outperform existing lightweight vision models while having fewer parameters and lower computational demands, making them suitable for FL scenarios with data heterogeneity and communication bottlenecks. 

---
