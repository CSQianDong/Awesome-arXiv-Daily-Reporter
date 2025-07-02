# Enhancing LLM Agent Safety via Causal Influence Prompting 

**Authors**: Dongyoon Hahm, Woogyeol Jin, June Suk Choi, Sungsoo Ahn, Kimin Lee  

**Link**: [PDF](https://arxiv.org/pdf/2507.00979)  

**Abstract**: As autonomous agents powered by large language models (LLMs) continue to demonstrate potential across various assistive tasks, ensuring their safe and reliable behavior is crucial for preventing unintended consequences. In this work, we introduce CIP, a novel technique that leverages causal influence diagrams (CIDs) to identify and mitigate risks arising from agent decision-making. CIDs provide a structured representation of cause-and-effect relationships, enabling agents to anticipate harmful outcomes and make safer decisions. Our approach consists of three key steps: (1) initializing a CID based on task specifications to outline the decision-making process, (2) guiding agent interactions with the environment using the CID, and (3) iteratively refining the CID based on observed behaviors and outcomes. Experimental results demonstrate that our method effectively enhances safety in both code execution and mobile device control tasks. 

---
# Thinking Beyond Tokens: From Brain-Inspired Intelligence to Cognitive Foundations for Artificial General Intelligence and its Societal Impact 

**Authors**: Rizwan Qureshi, Ranjan Sapkota, Abbas Shah, Amgad Muneer, Anas Zafar, Ashmal Vayani, Maged Shoman, Abdelrahman B. M. Eldaly, Kai Zhang, Ferhat Sadak, Shaina Raza, Xinqi Fan, Ravid Shwartz-Ziv, Hong Yan, Vinjia Jain, Aman Chadha, Manoj Karkee, Jia Wu, Philip Torr, Seyedali Mirjalili  

**Link**: [PDF](https://arxiv.org/pdf/2507.00951)  

**Abstract**: Can machines truly think, reason and act in domains like humans? This enduring question continues to shape the pursuit of Artificial General Intelligence (AGI). Despite the growing capabilities of models such as GPT-4.5, DeepSeek, Claude 3.5 Sonnet, Phi-4, and Grok 3, which exhibit multimodal fluency and partial reasoning, these systems remain fundamentally limited by their reliance on token-level prediction and lack of grounded agency. This paper offers a cross-disciplinary synthesis of AGI development, spanning artificial intelligence, cognitive neuroscience, psychology, generative models, and agent-based systems. We analyze the architectural and cognitive foundations of general intelligence, highlighting the role of modular reasoning, persistent memory, and multi-agent coordination. In particular, we emphasize the rise of Agentic RAG frameworks that combine retrieval, planning, and dynamic tool use to enable more adaptive behavior. We discuss generalization strategies, including information compression, test-time adaptation, and training-free methods, as critical pathways toward flexible, domain-agnostic intelligence. Vision-Language Models (VLMs) are reexamined not just as perception modules but as evolving interfaces for embodied understanding and collaborative task completion. We also argue that true intelligence arises not from scale alone but from the integration of memory and reasoning: an orchestration of modular, interactive, and self-improving components where compression enables adaptive behavior. Drawing on advances in neurosymbolic systems, reinforcement learning, and cognitive scaffolding, we explore how recent architectures begin to bridge the gap between statistical learning and goal-directed cognition. Finally, we identify key scientific, technical, and ethical challenges on the path to AGI. 

---
# SafeMobile: Chain-level Jailbreak Detection and Automated Evaluation for Multimodal Mobile Agents 

**Authors**: Siyuan Liang, Tianmeng Fang, Zhe Liu, Aishan Liu, Yan Xiao, Jinyuan He, Ee-Chien Chang, Xiaochun Cao  

**Link**: [PDF](https://arxiv.org/pdf/2507.00841)  

**Abstract**: With the wide application of multimodal foundation models in intelligent agent systems, scenarios such as mobile device control, intelligent assistant interaction, and multimodal task execution are gradually relying on such large model-driven agents. However, the related systems are also increasingly exposed to potential jailbreak risks. Attackers may induce the agents to bypass the original behavioral constraints through specific inputs, and then trigger certain risky and sensitive operations, such as modifying settings, executing unauthorized commands, or impersonating user identities, which brings new challenges to system security. Existing security measures for intelligent agents still have limitations when facing complex interactions, especially in detecting potentially risky behaviors across multiple rounds of conversations or sequences of tasks. In addition, an efficient and consistent automated methodology to assist in assessing and determining the impact of such risks is currently lacking. This work explores the security issues surrounding mobile multimodal agents, attempts to construct a risk discrimination mechanism by incorporating behavioral sequence information, and designs an automated assisted assessment scheme based on a large language model. Through preliminary validation in several representative high-risk tasks, the results show that the method can improve the recognition of risky behaviors to some extent and assist in reducing the probability of agents being jailbroken. We hope that this study can provide some valuable references for the security risk modeling and protection of multimodal intelligent agent systems. 

---
# A Robust Algorithm for Non-IID Machine Learning Problems with Convergence Analysis 

**Authors**: Qing Xu, Xiaohua Xuan  

**Link**: [PDF](https://arxiv.org/pdf/2507.00810)  

**Abstract**: In this paper, we propose an improved numerical algorithm for solving minimax problems based on nonsmooth optimization, quadratic programming and iterative process. We also provide a rigorous proof of convergence for our algorithm under some mild assumptions, such as gradient continuity and boundedness. Such an algorithm can be widely applied in various fields such as robust optimization, imbalanced learning, etc. 

---
# Can Large Language Models Develop Strategic Reasoning? Post-training Insights from Learning Chess 

**Authors**: Dongyoon Hwang, Hojoon Lee, Jaegul Choo, Dongmin Park, Jongho Park  

**Link**: [PDF](https://arxiv.org/pdf/2507.00726)  

**Abstract**: While reinforcement learning (RL) for large language models (LLMs) has shown promise in mathematical reasoning, strategic reasoning for LLMs using RL remains largely unexplored. We investigate whether LLMs can develop strategic reasoning capabilities through RL in chess. To this end, we leverage a chess-pretrained action-value network to provide dense reward on the LLM's output move quality, which can be seen as a form of knowledge distillation. Our experiments show that our distillation-based dense rewards often outperform sparse binary rewards. However, surprisingly, all models plateau far below expert levels. We provide SFT and RL ablations on chess reasoning training and find evidence that this limitation stems from a deficit in the pretrained models' internal understanding of chess--a deficit which RL alone may not be able to fully overcome. 

---
# Advancing Local Search in SMT-NRA with MCSAT Integration 

**Authors**: Tianyi Ding, Haokun Li, Xinpeng Ni, Bican Xia, Tianqi Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2507.00557)  

**Abstract**: In this paper, we advance local search for Satisfiability Modulo the Theory of Nonlinear Real Arithmetic (SMT-NRA for short). First, we introduce a two-dimensional cell-jump move, called \emph{$2d$-cell-jump}, generalizing the key operation, cell-jump, of the local search method for SMT-NRA. Then, we propose an extended local search framework, named \emph{$2d$-LS} (following the local search framework, LS, for SMT-NRA), integrating the model constructing satisfiability calculus (MCSAT) framework to improve search efficiency. To further improve the efficiency of MCSAT, we implement a recently proposed technique called \emph{sample-cell projection operator} for MCSAT, which is well suited for CDCL-style search in the real domain and helps guide the search away from conflicting states. Finally, we design a hybrid framework for SMT-NRA combining MCSAT, $2d$-LS and OpenCAD, to improve search efficiency through information exchange. The experimental results demonstrate improvements in local search performance, highlighting the effectiveness of the proposed methods. 

---
# Does Math Reasoning Improve General LLM Capabilities? Understanding Transferability of LLM Reasoning 

**Authors**: Maggie Huan, Yuetai Li, Tuney Zheng, Xiaoyu Xu, Seungone Kim, Minxin Du, Radha Poovendran, Graham Neubig, Xiang Yue  

**Link**: [PDF](https://arxiv.org/pdf/2507.00432)  

**Abstract**: Math reasoning has become the poster child of progress in large language models (LLMs), with new models rapidly surpassing human-level performance on benchmarks like MATH and AIME. But as math leaderboards improve week by week, it is worth asking: do these gains reflect broader problem-solving ability or just narrow overfitting? To answer this question, we evaluate over 20 open-weight reasoning-tuned models across a broad suite of tasks, including math, scientific QA, agent planning, coding, and standard instruction-following. We surprisingly find that most models that succeed in math fail to transfer their gains to other domains. To rigorously study this phenomenon, we conduct controlled experiments on Qwen3-14B models using math-only data but different tuning methods. We find that reinforcement learning (RL)-tuned models generalize well across domains, while supervised fine-tuning (SFT)-tuned models often forget general capabilities. Latent-space representation and token-space distribution shift analyses reveal that SFT induces substantial representation and output drift, while RL preserves general-domain structure. Our results suggest a need to rethink standard post-training recipes, particularly the reliance on SFT-distilled data for advancing reasoning models. 

---
# ASTRO: Teaching Language Models to Reason by Reflecting and Backtracking In-Context 

**Authors**: Joongwon Kim, Anirudh Goyal, Liang Tan, Hannaneh Hajishirzi, Srinivasan Iyer, Tianlu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.00417)  

**Abstract**: We introduce ASTRO, the "Autoregressive Search-Taught Reasoner", a framework for training language models to reason like search algorithms, explicitly leveraging self-reflection, backtracking, and exploration in their outputs. Recently, training large language models (LLMs) via reinforcement learning (RL) has led to the advent of reasoning models with greatly enhanced reasoning capabilities. Open-source replications of reasoning models, while successful, build upon models that already exhibit strong reasoning capabilities along with search behavior observed even before RL. As a result, it is yet unclear how to boost the reasoning capabilities of other non-reasoner models including Llama 3. ASTRO teaches such models to internalize structured search behavior through a synthetic dataset derived from Monte Carlo Tree Search (MCTS) over mathematical problem-solving trajectories. By converting search traces into natural language chain-of-thoughts that capture both successes and recoveries from failure, ASTRO bootstraps models with a rich prior for exploration during RL. We finetune our models on these search-derived traces and further improve performance via RL with verifiable rewards. We apply ASTRO to the Llama 3 family of models and achieve absolute performance gains of 16.0% on MATH-500, 26.9% on AMC 2023, and 20.0% on AIME 2024, especially improving upon challenging problems that require iterative correction. Our results demonstrate that search-inspired training offers a principled way to instill robust reasoning capabilities into open LLMs. 

---
# Learning for routing: A guided review of recent developments and future directions 

**Authors**: Fangting Zhou, Attila Lischka, Balazs Kulcsar, Jiaming Wu, Morteza Haghir Chehreghani, Gilbert Laporte  

**Link**: [PDF](https://arxiv.org/pdf/2507.00218)  

**Abstract**: This paper reviews the current progress in applying machine learning (ML) tools to solve NP-hard combinatorial optimization problems, with a focus on routing problems such as the traveling salesman problem (TSP) and the vehicle routing problem (VRP). Due to the inherent complexity of these problems, exact algorithms often require excessive computational time to find optimal solutions, while heuristics can only provide approximate solutions without guaranteeing optimality. With the recent success of machine learning models, there is a growing trend in proposing and implementing diverse ML techniques to enhance the resolution of these challenging routing problems. We propose a taxonomy categorizing ML-based routing methods into construction-based and improvement-based approaches, highlighting their applicability to various problem characteristics. This review aims to integrate traditional OR methods with state-of-the-art ML techniques, providing a structured framework to guide future research and address emerging VRP variants. 

---
# Holistic Artificial Intelligence in Medicine; improved performance and explainability 

**Authors**: Periklis Petridis, Georgios Margaritis, Vasiliki Stoumpou, Dimitris Bertsimas  

**Link**: [PDF](https://arxiv.org/pdf/2507.00205)  

**Abstract**: With the increasing interest in deploying Artificial Intelligence in medicine, we previously introduced HAIM (Holistic AI in Medicine), a framework that fuses multimodal data to solve downstream clinical tasks. However, HAIM uses data in a task-agnostic manner and lacks explainability. To address these limitations, we introduce xHAIM (Explainable HAIM), a novel framework leveraging Generative AI to enhance both prediction and explainability through four structured steps: (1) automatically identifying task-relevant patient data across modalities, (2) generating comprehensive patient summaries, (3) using these summaries for improved predictive modeling, and (4) providing clinical explanations by linking predictions to patient-specific medical knowledge. Evaluated on the HAIM-MIMIC-MM dataset, xHAIM improves average AUC from 79.9% to 90.3% across chest pathology and operative tasks. Importantly, xHAIM transforms AI from a black-box predictor into an explainable decision support system, enabling clinicians to interactively trace predictions back to relevant patient data, bridging AI advancements with clinical utility. 

---
# ChatGPT produces more "lazy" thinkers: Evidence of cognitive engagement decline 

**Authors**: Georgios P. Georgiou  

**Link**: [PDF](https://arxiv.org/pdf/2507.00181)  

**Abstract**: Despite the increasing use of large language models (LLMs) in education, concerns have emerged about their potential to reduce deep thinking and active learning. This study investigates the impact of generative artificial intelligence (AI) tools, specifically ChatGPT, on the cognitive engagement of students during academic writing tasks. The study employed an experimental design with participants randomly assigned to either an AI-assisted (ChatGPT) or a non-assisted (control) condition. Participants completed a structured argumentative writing task followed by a cognitive engagement scale (CES), the CES-AI, developed to assess mental effort, attention, deep processing, and strategic thinking. The results revealed significantly lower cognitive engagement scores in the ChatGPT group compared to the control group. These findings suggest that AI assistance may lead to cognitive offloading. The study contributes to the growing body of literature on the psychological implications of AI in education and raises important questions about the integration of such tools into academic practice. It calls for pedagogical strategies that promote active, reflective engagement with AI-generated content to avoid compromising self-regulated learning and deep cognitive involvement of students. 

---
# BlackBoxToBlueprint: Extracting Interpretable Logic from Legacy Systems using Reinforcement Learning and Counterfactual Analysis 

**Authors**: Vidhi Rathore  

**Link**: [PDF](https://arxiv.org/pdf/2507.00180)  

**Abstract**: Modernizing legacy software systems is a critical but challenging task, often hampered by a lack of documentation and understanding of the original system's intricate decision logic. Traditional approaches like behavioral cloning merely replicate input-output behavior without capturing the underlying intent. This paper proposes a novel pipeline to automatically extract interpretable decision logic from legacy systems treated as black boxes. The approach uses a Reinforcement Learning (RL) agent to explore the input space and identify critical decision boundaries by rewarding actions that cause meaningful changes in the system's output. These counterfactual state transitions, where the output changes, are collected and clustered using K-Means. Decision trees are then trained on these clusters to extract human-readable rules that approximate the system's decision logic near the identified boundaries. I demonstrated the pipeline's effectiveness on three dummy legacy systems with varying complexity, including threshold-based, combined-conditional, and non-linear range logic. Results show that the RL agent successfully focuses exploration on relevant boundary regions, and the extracted rules accurately reflect the core logic of the underlying dummy systems, providing a promising foundation for generating specifications and test cases during legacy migration. 

---
# Thinking About Thinking: SAGE-nano's Inverse Reasoning for Self-Aware Language Models 

**Authors**: Basab Jha, Firoj Paudel, Ujjwal Puri, Zhang Yuting, Choi Donghyuk, Wang Junhao  

**Link**: [PDF](https://arxiv.org/pdf/2507.00092)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities at solving complex reasoning tasks with Chain-of-Thought (CoT) prompting, but their decision-making processes remain somewhat blackbox. We introduce textbfinverse reasoning, a novel paradigm enabling LLMs to decompose and explain their own reasoning chains post-hoc. Our approach, used in SAGE-nano, a 4-billion-parameter reasoning model, employs a metacognitive structure that reflects back via attention processes to identify major decision points and generate explanations of reasoning choices. While typical CoT approaches are directed towards forward reasoning generation, inverse reasoning provides insight into why specific reasoning chains were selected over others. Through thorough testing of logical reasoning puzzles, math problems and ethical dilemmas from AQUA-RAT, CommonsenseQA, and customized benchmarks, we demonstrate that SAGE-nano is at the cutting edge both on reasoning accuracy (74.6% on AQUA-RAT) and explanation quality (92.1% human preference score) for its task, and offers performance almost on par with models like Claude-3.5 Sonnet or GPT-4o. Our contributions are: (i) the first rigorous framework for LLM self-reflection via inverse reasoning, (ii) a novel metalearning framework to reverse the attention flow, (iii) comprehensive evaluation frameworks for reasoning transparency, and (iv) evidence that increasing reasoning using inverse reasoning improves interpretability along with reasoning performance. Our work creates new avenues for transparent AI systems and closes significant gaps in AI safety, education, and scientific discovery. 

---
# VoyagerVision: Investigating the Role of Multi-modal Information for Open-ended Learning Systems 

**Authors**: Ethan Smyth, Alessandro Suglia  

**Link**: [PDF](https://arxiv.org/pdf/2507.00079)  

**Abstract**: Open-endedness is an active field of research in the pursuit of capable Artificial General Intelligence (AGI), allowing models to pursue tasks of their own choosing. Simultaneously, recent advancements in Large Language Models (LLMs) such as GPT-4o [9] have allowed such models to be capable of interpreting image inputs. Implementations such as OMNI-EPIC [4] have made use of such features, providing an LLM with pixel data of an agent's POV to parse the environment and allow it to solve tasks. This paper proposes that providing these visual inputs to a model gives it greater ability to interpret spatial environments, and as such, can increase the number of tasks it can successfully perform, extending its open-ended potential. To this aim, this paper proposes VoyagerVision -- a multi-modal model capable of creating structures within Minecraft using screenshots as a form of visual feedback, building on the foundation of Voyager. VoyagerVision was capable of creating an average of 2.75 unique structures within fifty iterations of the system, as Voyager was incapable of this, it is an extension in an entirely new direction. Additionally, in a set of building unit tests VoyagerVision was successful in half of all attempts in flat worlds, with most failures arising in more complex structures. Project website is available at this https URL 

---
# Enhancing Reasoning Capabilities in SLMs with Reward Guided Dataset Distillation 

**Authors**: Shreyansh Padarha  

**Link**: [PDF](https://arxiv.org/pdf/2507.00054)  

**Abstract**: The push to compress and impart the proficiency of Large Language Models (LLMs) into more deployable and efficient Small Language Models (SLMs) has benefited from improvements in knowledge distillation (KD) techniques. These techniques allow a smaller student model to learn from a more capable and larger teacher model's responses. However, distillation often revolves around the student model merely copying the teacher's in-distribution responses, limiting its generalisability. This limitation is amplified on reasoning tasks and can be computationally expensive. In this study, we propose AdvDistill, a reward-guided dataset distillation framework. We utilise multiple generations (responses) from a teacher for each prompt and assign rewards based on rule-based verifiers. These varying and normally distributed rewards serve as weights when training student models. Our methods and their subsequent behavioural analysis demonstrate a significant improvement in student model performance for mathematical and complex reasoning tasks, showcasing the efficacy and benefits of incorporating a rewarding mechanism in dataset distillation processes. 

---
# SEZ-HARN: Self-Explainable Zero-shot Human Activity Recognition Network 

**Authors**: Devin Y. De Silva, Sandareka Wickramanayake, Dulani Meedeniya, Sanka Rasnayaka  

**Link**: [PDF](https://arxiv.org/pdf/2507.00050)  

**Abstract**: Human Activity Recognition (HAR), which uses data from Inertial Measurement Unit (IMU) sensors, has many practical applications in healthcare and assisted living environments. However, its use in real-world scenarios has been limited by the lack of comprehensive IMU-based HAR datasets that cover a wide range of activities and the lack of transparency in existing HAR models. Zero-shot HAR (ZS-HAR) overcomes the data limitations, but current models struggle to explain their decisions, making them less transparent. This paper introduces a novel IMU-based ZS-HAR model called the Self-Explainable Zero-shot Human Activity Recognition Network (SEZ-HARN). It can recognize activities not encountered during training and provide skeleton videos to explain its decision-making process. We evaluate the effectiveness of the proposed SEZ-HARN on four benchmark datasets PAMAP2, DaLiAc, HTD-MHAD and MHealth and compare its performance against three state-of-the-art black-box ZS-HAR models. The experiment results demonstrate that SEZ-HARN produces realistic and understandable explanations while achieving competitive Zero-shot recognition accuracy. SEZ-HARN achieves a Zero-shot prediction accuracy within 3\% of the best-performing black-box model on PAMAP2 while maintaining comparable performance on the other three datasets. 

---
# A collaborative digital twin built on FAIR data and compute infrastructure 

**Authors**: Thomas M. Deucher, Juan C. Verduzco, Michael Titus, Alejandro Strachan  

**Link**: [PDF](https://arxiv.org/pdf/2507.00048)  

**Abstract**: The integration of machine learning with automated experimentation in self-driving laboratories (SDL) offers a powerful approach to accelerate discovery and optimization tasks in science and engineering applications. When supported by findable, accessible, interoperable, and reusable (FAIR) data infrastructure, SDLs with overlapping interests can collaborate more effectively. This work presents a distributed SDL implementation built on nanoHUB services for online simulation and FAIR data management. In this framework, geographically dispersed collaborators conducting independent optimization tasks contribute raw experimental data to a shared central database. These researchers can then benefit from analysis tools and machine learning models that automatically update as additional data become available. New data points are submitted through a simple web interface and automatically processed using a nanoHUB Sim2L, which extracts derived quantities and indexes all inputs and outputs in a FAIR data repository called ResultsDB. A separate nanoHUB workflow enables sequential optimization using active learning, where researchers define the optimization objective, and machine learning models are trained on-the-fly with all existing data, guiding the selection of future experiments. Inspired by the concept of ``frugal twin", the optimization task seeks to find the optimal recipe to combine food dyes to achieve the desired target color. With easily accessible and inexpensive materials, researchers and students can set up their own experiments, share data with collaborators, and explore the combination of FAIR data, predictive ML models, and sequential optimization. The tools introduced are generally applicable and can easily be extended to other optimization problems. 

---
# TalentMine: LLM-Based Extraction and Question-Answering from Multimodal Talent Tables 

**Authors**: Varun Mannam, Fang Wang, Chaochun Liu, Xin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.00041)  

**Abstract**: In talent management systems, critical information often resides in complex tabular formats, presenting significant retrieval challenges for conventional language models. These challenges are pronounced when processing Talent documentation that requires precise interpretation of tabular relationships for accurate information retrieval and downstream decision-making. Current table extraction methods struggle with semantic understanding, resulting in poor performance when integrated into retrieval-augmented chat applications. This paper identifies a key bottleneck - while structural table information can be extracted, the semantic relationships between tabular elements are lost, causing downstream query failures. To address this, we introduce TalentMine, a novel LLM-enhanced framework that transforms extracted tables into semantically enriched representations. Unlike conventional approaches relying on CSV or text linearization, our method employs specialized multimodal reasoning to preserve both structural and semantic dimensions of tabular data. Experimental evaluation across employee benefits document collections demonstrates TalentMine's superior performance, achieving 100% accuracy in query answering tasks compared to 0% for standard AWS Textract extraction and 40% for AWS Textract Visual Q&A capabilities. Our comparative analysis also reveals that the Claude v3 Haiku model achieves optimal performance for talent management applications. The key contributions of this work include (1) a systematic analysis of semantic information loss in current table extraction pipelines, (2) a novel LLM-based method for semantically enriched table representation, (3) an efficient integration framework for retrieval-augmented systems as end-to-end systems, and (4) comprehensive benchmarks on talent analytics tasks showing substantial improvements across multiple categories. 

---
# DiMo-GUI: Advancing Test-time Scaling in GUI Grounding via Modality-Aware Visual Reasoning 

**Authors**: Hang Wu, Hongkai Chen, Yujun Cai, Chang Liu, Qingwen Ye, Ming-Hsuan Yang, Yiwei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.00008)  

**Abstract**: Grounding natural language queries in graphical user interfaces (GUIs) poses unique challenges due to the diversity of visual elements, spatial clutter, and the ambiguity of language. In this paper, we introduce DiMo-GUI, a training-free framework for GUI grounding that leverages two core strategies: dynamic visual grounding and modality-aware optimization. Instead of treating the GUI as a monolithic image, our method splits the input into textual elements and iconic elements, allowing the model to reason over each modality independently using general-purpose vision-language models. When predictions are ambiguous or incorrect, DiMo-GUI dynamically focuses attention by generating candidate focal regions centered on the model's initial predictions and incrementally zooms into subregions to refine the grounding result. This hierarchical refinement process helps disambiguate visually crowded layouts without the need for additional training or annotations. We evaluate our approach on standard GUI grounding benchmarks and demonstrate consistent improvements over baseline inference pipelines, highlighting the effectiveness of combining modality separation with region-focused reasoning. 

---
# GLM-4.1V-Thinking: Towards Versatile Multimodal Reasoning with Scalable Reinforcement Learning 

**Authors**: Wenyi Hong, Wenmeng Yu, Xiaotao Gu, Guo Wang, Guobing Gan, Haomiao Tang, Jiale Cheng, Ji Qi, Junhui Ji, Lihang Pan, Shuaiqi Duan, Weihan Wang, Yan Wang, Yean Cheng, Zehai He, Zhe Su, Zhen Yang, Ziyang Pan, Aohan Zeng, Baoxu Wang, Boyan Shi, Changyu Pang, Chenhui Zhang, Da Yin, Fan Yang, Guoqing Chen, Jiazheng Xu, Jiali Chen, Jing Chen, Jinhao Chen, Jinghao Lin, Jinjiang Wang, Junjie Chen, Leqi Lei, Leyi Pan, Mingzhi Zhang, Qinkai Zheng, Sheng Yang, Shi Zhong, Shiyu Huang, Shuyuan Zhao, Siyan Xue, Shangqin Tu, Shengbiao Meng, Tianshu Zhang, Tianwei Luo, Tianxiang Hao, Tianle Gong, Wenkai Li, Wei Jia, Xin Lyu, Xuancheng Huang, Yanling Wang, Yadong Xue, Yanfeng Wang, Yifan An, Yifan Du, Yiming Shi, Yiheng Huang, Yilin Niu, Yuan Wang, Yuanchang Yue, Yuchen Li, Yutao Zhang, Yuxuan Zhang, Zhanxiao Du, Zhenyu Hou, Zhao Xue, Zhengxiao Du, Zihan Wang, Peng Zhang, Debing Liu, Bin Xu, Juanzi Li, Minlie Huang, Yuxiao Dong, Jie Tang  

**Link**: [PDF](https://arxiv.org/pdf/2507.01006)  

**Abstract**: We present GLM-4.1V-Thinking, a vision-language model (VLM) designed to advance general-purpose multimodal reasoning. In this report, we share our key findings in the development of the reasoning-centric training framework. We first develop a capable vision foundation model with significant potential through large-scale pre-training, which arguably sets the upper bound for the final performance. Reinforcement Learning with Curriculum Sampling (RLCS) then unlocks the full potential of the model, leading to comprehensive capability enhancement across a diverse range of tasks, including STEM problem solving, video understanding, content recognition, coding, grounding, GUI-based agents, and long document understanding, among others. To facilitate research in this field, we open-source GLM-4.1V-9B-Thinking, which achieves state-of-the-art performance among models of comparable size. In a comprehensive evaluation across 28 public benchmarks, our model outperforms Qwen2.5-VL-7B on nearly all tasks and achieves comparable or even superior performance on 18 benchmarks relative to the significantly larger Qwen2.5-VL-72B. Notably, GLM-4.1V-9B-Thinking also demonstrates competitive or superior performance compared to closed-source models such as GPT-4o on challenging tasks including long document understanding and STEM reasoning, further underscoring its strong capabilities. Code, models and more information are released at this https URL. 

---
# Description of the Training Process of Neural Networks via Ergodic Theorem : Ghost nodes 

**Authors**: Eun-Ji Park, Sangwon Yun  

**Link**: [PDF](https://arxiv.org/pdf/2507.01003)  

**Abstract**: Recent studies have proposed interpreting the training process from an ergodic perspective. Building on this foundation we present a unified framework for understanding and accelerating the training of deep neural networks via stochastic gradient descent. By analyzing the geometric landscape of the objective function we introduce a practical diagnostic, the running estimate of the largest Lyapunov exponent, which provably distinguishes genuine convergence toward stable minimizers from mere statistical stabilization near saddle points. We then propose a ghost category extension for standard classifiers that adds auxiliary ghost output nodes so the model gains extra descent directions that open a lateral corridor around narrow loss barriers and enable the optimizer to bypass poor basins during the early training phase. We show that this extension strictly reduces approximation error and that after sufficient convergence the ghost dimensions collapse and the extended model's invariant law coincides with that of the original and there exists a path in the enlarged parameter space along which the total loss does not increase while the original loss decreases by an arbitrary margin. Taken together these results provide a principled architecture level intervention that accelerates early stage trainability while preserving asymptotic behavior. 

---
# SciArena: An Open Evaluation Platform for Foundation Models in Scientific Literature Tasks 

**Authors**: Yilun Zhao, Kaiyan Zhang, Tiansheng Hu, Sihong Wu, Ronan Le Bras, Taira Anderson, Jonathan Bragg, Joseph Chee Chang, Jesse Dodge, Matt Latzke, Yixin Liu, Charles McGrady, Xiangru Tang, Zihang Wang, Chen Zhao, Hannaneh Hajishirzi, Doug Downey, Arman Cohan  

**Link**: [PDF](https://arxiv.org/pdf/2507.01001)  

**Abstract**: We present SciArena, an open and collaborative platform for evaluating foundation models on scientific literature tasks. Unlike traditional benchmarks for scientific literature understanding and synthesis, SciArena engages the research community directly, following the Chatbot Arena evaluation approach of community voting on model comparisons. By leveraging collective intelligence, SciArena offers a community-driven evaluation of model performance on open-ended scientific tasks that demand literature-grounded, long-form responses. The platform currently supports 23 open-source and proprietary foundation models and has collected over 13,000 votes from trusted researchers across diverse scientific domains. We analyze the data collected so far and confirm that the submitted questions are diverse, aligned with real-world literature needs, and that participating researchers demonstrate strong self-consistency and inter-annotator agreement in their evaluations. We discuss the results and insights based on the model ranking leaderboard. To further promote research in building model-based automated evaluation systems for literature tasks, we release SciArena-Eval, a meta-evaluation benchmark based on our collected preference data. The benchmark measures the accuracy of models in judging answer quality by comparing their pairwise assessments with human votes. Our experiments highlight the benchmark's challenges and emphasize the need for more reliable automated evaluation methods. 

---
# Robotic Manipulation by Imitating Generated Videos Without Physical Demonstrations 

**Authors**: Shivansh Patel, Shraddhaa Mohan, Hanlin Mai, Unnat Jain, Svetlana Lazebnik, Yunzhu Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.00990)  

**Abstract**: This work introduces Robots Imitating Generated Videos (RIGVid), a system that enables robots to perform complex manipulation tasks--such as pouring, wiping, and mixing--purely by imitating AI-generated videos, without requiring any physical demonstrations or robot-specific training. Given a language command and an initial scene image, a video diffusion model generates potential demonstration videos, and a vision-language model (VLM) automatically filters out results that do not follow the command. A 6D pose tracker then extracts object trajectories from the video, and the trajectories are retargeted to the robot in an embodiment-agnostic fashion. Through extensive real-world evaluations, we show that filtered generated videos are as effective as real demonstrations, and that performance improves with generation quality. We also show that relying on generated videos outperforms more compact alternatives such as keypoint prediction using VLMs, and that strong 6D pose tracking outperforms other ways to extract trajectories, such as dense feature point tracking. These findings suggest that videos produced by a state-of-the-art off-the-shelf model can offer an effective source of supervision for robotic manipulation. 

---
# Reasoning as an Adaptive Defense for Safety 

**Authors**: Taeyoun Kim, Fahim Tajwar, Aditi Raghunathan, Aviral Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2507.00971)  

**Abstract**: Reasoning methods that adaptively allocate test-time compute have advanced LLM performance on easy to verify domains such as math and code. In this work, we study how to utilize this approach to train models that exhibit a degree of robustness to safety vulnerabilities, and show that doing so can provide benefits. We build a recipe called $\textit{TARS}$ (Training Adaptive Reasoners for Safety), a reinforcement learning (RL) approach that trains models to reason about safety using chain-of-thought traces and a reward signal that balances safety with task completion. To build TARS, we identify three critical design choices: (1) a "lightweight" warmstart SFT stage, (2) a mix of harmful, harmless, and ambiguous prompts to prevent shortcut behaviors such as too many refusals, and (3) a reward function to prevent degeneration of reasoning capabilities during training. Models trained with TARS exhibit adaptive behaviors by spending more compute on ambiguous queries, leading to better safety-refusal trade-offs. They also internally learn to better distinguish between safe and unsafe prompts and attain greater robustness to both white-box (e.g., GCG) and black-box attacks (e.g., PAIR). Overall, our work provides an effective, open recipe for training LLMs against jailbreaks and harmful requests by reasoning per prompt. 

---
# Surgical Neural Radiance Fields from One Image 

**Authors**: Alberto Neri, Maximilan Fehrentz, Veronica Penza, Leonardo S. Mattos, Nazim Haouchine  

**Link**: [PDF](https://arxiv.org/pdf/2507.00969)  

**Abstract**: Purpose: Neural Radiance Fields (NeRF) offer exceptional capabilities for 3D reconstruction and view synthesis, yet their reliance on extensive multi-view data limits their application in surgical intraoperative settings where only limited data is available. In particular, collecting such extensive data intraoperatively is impractical due to time constraints. This work addresses this challenge by leveraging a single intraoperative image and preoperative data to train NeRF efficiently for surgical scenarios.
Methods: We leverage preoperative MRI data to define the set of camera viewpoints and images needed for robust and unobstructed training. Intraoperatively, the appearance of the surgical image is transferred to the pre-constructed training set through neural style transfer, specifically combining WTC2 and STROTSS to prevent over-stylization. This process enables the creation of a dataset for instant and fast single-image NeRF training.
Results: The method is evaluated with four clinical neurosurgical cases. Quantitative comparisons to NeRF models trained on real surgical microscope images demonstrate strong synthesis agreement, with similarity metrics indicating high reconstruction fidelity and stylistic alignment. When compared with ground truth, our method demonstrates high structural similarity, confirming good reconstruction quality and texture preservation.
Conclusion: Our approach demonstrates the feasibility of single-image NeRF training in surgical settings, overcoming the limitations of traditional multi-view methods. 

---
# MambAttention: Mamba with Multi-Head Attention for Generalizable Single-Channel Speech Enhancement 

**Authors**: Nikolai Lund Kühne, Jesper Jensen, Jan Østergaard, Zheng-Hua Tan  

**Link**: [PDF](https://arxiv.org/pdf/2507.00966)  

**Abstract**: With the advent of new sequence models like Mamba and xLSTM, several studies have shown that these models match or outperform state-of-the-art models in single-channel speech enhancement, automatic speech recognition, and self-supervised audio representation learning. However, prior research has demonstrated that sequence models like LSTM and Mamba tend to overfit to the training set. To address this issue, previous works have shown that adding self-attention to LSTMs substantially improves generalization performance for single-channel speech enhancement. Nevertheless, neither the concept of hybrid Mamba and time-frequency attention models nor their generalization performance have been explored for speech enhancement. In this paper, we propose a novel hybrid architecture, MambAttention, which combines Mamba and shared time- and frequency-multi-head attention modules for generalizable single-channel speech enhancement. To train our model, we introduce VoiceBank+Demand Extended (VB-DemandEx), a dataset inspired by VoiceBank+Demand but with more challenging noise types and lower signal-to-noise ratios. Trained on VB-DemandEx, our proposed MambAttention model significantly outperforms existing state-of-the-art LSTM-, xLSTM-, Mamba-, and Conformer-based systems of similar complexity across all reported metrics on two out-of-domain datasets: DNS 2020 and EARS-WHAM_v2, while matching their performance on the in-domain dataset VB-DemandEx. Ablation studies highlight the role of weight sharing between the time- and frequency-multi-head attention modules for generalization performance. Finally, we explore integrating the shared time- and frequency-multi-head attention modules with LSTM and xLSTM, which yields a notable performance improvement on the out-of-domain datasets. However, our MambAttention model remains superior on both out-of-domain datasets across all reported evaluation metrics. 

---
# From Sentences to Sequences: Rethinking Languages in Biological System 

**Authors**: Ke Liu, Shuanke Shen, Hao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.00953)  

**Abstract**: The paradigm of large language models in natural language processing (NLP) has also shown promise in modeling biological languages, including proteins, RNA, and DNA. Both the auto-regressive generation paradigm and evaluation metrics have been transferred from NLP to biological sequence modeling. However, the intrinsic structural correlations in natural and biological languages differ fundamentally. Therefore, we revisit the notion of language in biological systems to better understand how NLP successes can be effectively translated to biological domains. By treating the 3D structure of biomolecules as the semantic content of a sentence and accounting for the strong correlations between residues or bases, we highlight the importance of structural evaluation and demonstrate the applicability of the auto-regressive paradigm in biological language modeling. Code can be found at \href{this https URL}{this http URL} 

---
# WebArXiv: Evaluating Multimodal Agents on Time-Invariant arXiv Tasks 

**Authors**: Zihao Sun, Meng Fang, Ling Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.00938)  

**Abstract**: Recent progress in large language models (LLMs) has enabled the development of autonomous web agents capable of navigating and interacting with real websites. However, evaluating such agents remains challenging due to the instability and inconsistency of existing benchmarks, which often rely on dynamic content or oversimplified simulations. In this work, we introduce WebArXiv, a static and time-invariant benchmark comprising 275 web-based tasks grounded in the arXiv platform. WebArXiv ensures reproducible and reliable evaluation by anchoring tasks in fixed web snapshots with deterministic ground truths and standardized action trajectories. Through behavioral analysis, we identify a common failure mode, Rigid History Reflection, where agents over-rely on fixed interaction histories. To address this, we propose a lightweight dynamic reflection mechanism that allows agents to selectively retrieve relevant past steps during decision-making. We evaluate ten state-of-the-art web agents on WebArXiv. Results demonstrate clear performance differences across agents and validate the effectiveness of our proposed reflection strategy. 

---
# Large Language Model Powered Intelligent Urban Agents: Concepts, Capabilities, and Applications 

**Authors**: Jindong Han, Yansong Ning, Zirui Yuan, Hang Ni, Fan Liu, Tengfei Lyu, Hao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.00914)  

**Abstract**: The long-standing vision of intelligent cities is to create efficient, livable, and sustainable urban environments using big data and artificial intelligence technologies. Recently, the advent of Large Language Models (LLMs) has opened new ways toward realizing this vision. With powerful semantic understanding and reasoning capabilities, LLMs can be deployed as intelligent agents capable of autonomously solving complex problems across domains. In this article, we focus on Urban LLM Agents, which are LLM-powered agents that are semi-embodied within the hybrid cyber-physical-social space of cities and used for system-level urban decision-making. First, we introduce the concept of urban LLM agents, discussing their unique capabilities and features. Second, we survey the current research landscape from the perspective of agent workflows, encompassing urban sensing, memory management, reasoning, execution, and learning. Third, we categorize the application domains of urban LLM agents into five groups: urban planning, transportation, environment, public safety, and urban society, presenting representative works in each group. Finally, we discuss trustworthiness and evaluation issues that are critical for real-world deployment, and identify several open problems for future research. This survey aims to establish a foundation for the emerging field of urban LLM agents and to provide a roadmap for advancing the intersection of LLMs and urban intelligence. A curated list of relevant papers and open-source resources is maintained and continuously updated at this https URL. 

---
# Turning AI Data Centers into Grid-Interactive Assets: Results from a Field Demonstration in Phoenix, Arizona 

**Authors**: Philip Colangelo, Ayse K. Coskun, Jack Megrue, Ciaran Roberts, Shayan Sengupta, Varun Sivaram, Ethan Tiao, Aroon Vijaykar, Chris Williams, Daniel C. Wilson, Zack MacFarland, Daniel Dreiling, Nathan Morey, Anuja Ratnayake, Baskar Vairamohan  

**Link**: [PDF](https://arxiv.org/pdf/2507.00909)  

**Abstract**: Artificial intelligence (AI) is fueling exponential electricity demand growth, threatening grid reliability, raising prices for communities paying for new energy infrastructure, and stunting AI innovation as data centers wait for interconnection to constrained grids. This paper presents the first field demonstration, in collaboration with major corporate partners, of a software-only approach--Emerald Conductor--that transforms AI data centers into flexible grid resources that can efficiently and immediately harness existing power systems without massive infrastructure buildout. Conducted at a 256-GPU cluster running representative AI workloads within a commercial, hyperscale cloud data center in Phoenix, Arizona, the trial achieved a 25% reduction in cluster power usage for three hours during peak grid events while maintaining AI quality of service (QoS) guarantees. By orchestrating AI workloads based on real-time grid signals without hardware modifications or energy storage, this platform reimagines data centers as grid-interactive assets that enhance grid reliability, advance affordability, and accelerate AI's development. 

---
# The Age of Sensorial Zero Trust: Why We Can No Longer Trust Our Senses 

**Authors**: Fabio Correa Xavier  

**Link**: [PDF](https://arxiv.org/pdf/2507.00907)  

**Abstract**: In a world where deepfakes and cloned voices are emerging as sophisticated attack vectors, organizations require a new security mindset: Sensorial Zero Trust [9]. This article presents a scientific analysis of the need to systematically doubt information perceived through the senses, establishing rigorous verification protocols to mitigate the risks of fraud based on generative artificial intelligence. Key concepts, such as Out-of-Band verification, Vision-Language Models (VLMs) as forensic collaborators, cryptographic provenance, and human training, are integrated into a framework that extends Zero Trust principles to human sensory information. The approach is grounded in empirical findings and academic research, emphasizing that in an era of AI-generated realities, even our eyes and ears can no longer be implicitly trusted without verification. Leaders are called to foster a culture of methodological skepticism to protect organizational integrity in this new threat landscape. 

---
# Deep learning-based segmentation of T1 and T2 cardiac MRI maps for automated disease detection 

**Authors**: Andreea Bianca Popescu, Andreas Seitz, Heiko Mahrholdt, Jens Wetzl, Athira Jacob, Lucian Mihai Itu, Constantin Suciu, Teodora Chitiboi  

**Link**: [PDF](https://arxiv.org/pdf/2507.00903)  

**Abstract**: Objectives Parametric tissue mapping enables quantitative cardiac tissue characterization but is limited by inter-observer variability during manual delineation. Traditional approaches relying on average relaxation values and single cutoffs may oversimplify myocardial complexity. This study evaluates whether deep learning (DL) can achieve segmentation accuracy comparable to inter-observer variability, explores the utility of statistical features beyond mean T1/T2 values, and assesses whether machine learning (ML) combining multiple features enhances disease detection. Materials & Methods T1 and T2 maps were manually segmented. The test subset was independently annotated by two observers, and inter-observer variability was assessed. A DL model was trained to segment left ventricle blood pool and myocardium. Average (A), lower quartile (LQ), median (M), and upper quartile (UQ) were computed for the myocardial pixels and employed in classification by applying cutoffs or in ML. Dice similarity coefficient (DICE) and mean absolute percentage error evaluated segmentation performance. Bland-Altman plots assessed inter-user and model-observer agreement. Receiver operating characteristic analysis determined optimal cutoffs. Pearson correlation compared features from model and manual segmentations. F1-score, precision, and recall evaluated classification performance. Wilcoxon test assessed differences between classification methods, with p < 0.05 considered statistically significant. Results 144 subjects were split into training (100), validation (15) and evaluation (29) subsets. Segmentation model achieved a DICE of 85.4%, surpassing inter-observer agreement. Random forest applied to all features increased F1-score (92.7%, p < 0.001). Conclusion DL facilitates segmentation of T1/ T2 maps. Combining multiple features with ML improves disease detection. 

---
# Constellation as a Service: Tailored Connectivity Management in Direct-Satellite-to-Device Networks 

**Authors**: Feng Wang, Shengyu Zhang, Een-Kee Hong, Tony Q.S. Quek  

**Link**: [PDF](https://arxiv.org/pdf/2507.00902)  

**Abstract**: Direct-satellite-to-device (DS2D) communication is emerging as a promising solution for global mobile service extension, leveraging the deployment of satellite constellations. However, the challenge of managing DS2D connectivity for multi-constellations becomes outstanding, including high interference and frequent handovers caused by multi-coverage overlap and rapid satellite movement. Moreover, existing approaches primarily operate within single-constellation shell, which inherently limits the ability to exploit the vast potential of multi-constellation connectivity provision, resulting in suboptimal DS2D service performances. To address these challenges, this article proposes a Constellation as a Service (CaaS) framework, which treats the entire multi-constellation infrastructure as a shared resource pool and dynamically forms optimal sub-constellations (SCs) for each DS2D service region. The formation of each SC integrates satellites from various orbits to provide tailored connectivity based on user demands, guided by two innovative strategies: predictive satellite beamforming using generative artificial intelligence (GenAI) and pre-configured handover path for efficient satellite access and mobility management. Simulation results demonstrate that CaaS significantly improves satellite service rates while reducing handover overhead, making it an efficient and continuable solution for managing DS2D connectivity in multi-constellation environments. 

---
# MemeCMD: An Automatically Generated Chinese Multi-turn Dialogue Dataset with Contextually Retrieved Memes 

**Authors**: Yuheng Wang, Xianhe Tang, Pufeng Huang  

**Link**: [PDF](https://arxiv.org/pdf/2507.00891)  

**Abstract**: Memes are widely used in online social interactions, providing vivid, intuitive, and often humorous means to express intentions and emotions. Existing dialogue datasets are predominantly limited to either manually annotated or pure-text conversations, lacking the expressiveness and contextual nuance that multimodal interactions this http URL address these challenges, we introduce MemeCMD, an automatically generated Chinese Multi-turn Dialogue dataset with contextually retrieved memes. Our dataset combines a large-scale, MLLM-annotated meme library with dialogues auto-generated by dual agents across diverse scenarios. We introduce a retrieval framework and adaptive threshold to ensure contextually relevant, naturally spaced meme usage. Experiments demonstrate the effectiveness of our approach in generating contextually appropriate and diverse meme-incorporated dialogues, offering a scalable and privacy-preserving resource for advancing multimodal conversational AI. 

---
# NN-Former: Rethinking Graph Structure in Neural Architecture Representation 

**Authors**: Ruihan Xu, Haokui Zhang, Yaowei Wang, Wei Zeng, Shiliang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.00880)  

**Abstract**: The growing use of deep learning necessitates efficient network design and deployment, making neural predictors vital for estimating attributes such as accuracy and latency. Recently, Graph Neural Networks (GNNs) and transformers have shown promising performance in representing neural architectures. However, each of both methods has its disadvantages. GNNs lack the capabilities to represent complicated features, while transformers face poor generalization when the depth of architecture grows. To mitigate the above issues, we rethink neural architecture topology and show that sibling nodes are pivotal while overlooked in previous research. We thus propose a novel predictor leveraging the strengths of GNNs and transformers to learn the enhanced topology. We introduce a novel token mixer that considers siblings, and a new channel mixer named bidirectional graph isomorphism feed-forward network. Our approach consistently achieves promising performance in both accuracy and latency prediction, providing valuable insights for learning Directed Acyclic Graph (DAG) topology. The code is available at this https URL. 

---
# Stylometry recognizes human and LLM-generated texts in short samples 

**Authors**: Karol Przystalski, Jan K. Argasiński, Iwona Grabska-Gradzińska, Jeremi K. Ochab  

**Link**: [PDF](https://arxiv.org/pdf/2507.00838)  

**Abstract**: The paper explores stylometry as a method to distinguish between texts created by Large Language Models (LLMs) and humans, addressing issues of model attribution, intellectual property, and ethical AI use. Stylometry has been used extensively to characterise the style and attribute authorship of texts. By applying it to LLM-generated texts, we identify their emergent writing patterns. The paper involves creating a benchmark dataset based on Wikipedia, with (a) human-written term summaries, (b) texts generated purely by LLMs (GPT-3.5/4, LLaMa 2/3, Orca, and Falcon), (c) processed through multiple text summarisation methods (T5, BART, Gensim, and Sumy), and (d) rephrasing methods (Dipper, T5). The 10-sentence long texts were classified by tree-based models (decision trees and LightGBM) using human-designed (StyloMetrix) and n-gram-based (our own pipeline) stylometric features that encode lexical, grammatical, syntactic, and punctuation patterns. The cross-validated results reached a performance of up to .87 Matthews correlation coefficient in the multiclass scenario with 7 classes, and accuracy between .79 and 1. in binary classification, with the particular example of Wikipedia and GPT-4 reaching up to .98 accuracy on a balanced dataset. Shapley Additive Explanations pinpointed features characteristic of the encyclopaedic text type, individual overused words, as well as a greater grammatical standardisation of LLMs with respect to human-written texts. These results show -- crucially, in the context of the increasingly sophisticated LLMs -- that it is possible to distinguish machine- from human-generated texts at least for a well-defined text type. 

---
# HumanoidGen: Data Generation for Bimanual Dexterous Manipulation via LLM Reasoning 

**Authors**: Zhi Jing, Siyuan Yang, Jicong Ao, Ting Xiao, Yugang Jiang, Chenjia Bai  

**Link**: [PDF](https://arxiv.org/pdf/2507.00833)  

**Abstract**: For robotic manipulation, existing robotics datasets and simulation benchmarks predominantly cater to robot-arm platforms. However, for humanoid robots equipped with dual arms and dexterous hands, simulation tasks and high-quality demonstrations are notably lacking. Bimanual dexterous manipulation is inherently more complex, as it requires coordinated arm movements and hand operations, making autonomous data collection challenging. This paper presents HumanoidGen, an automated task creation and demonstration collection framework that leverages atomic dexterous operations and LLM reasoning to generate relational constraints. Specifically, we provide spatial annotations for both assets and dexterous hands based on the atomic operations, and perform an LLM planner to generate a chain of actionable spatial constraints for arm movements based on object affordances and scenes. To further improve planning ability, we employ a variant of Monte Carlo tree search to enhance LLM reasoning for long-horizon tasks and insufficient annotation. In experiments, we create a novel benchmark with augmented scenarios to evaluate the quality of the collected data. The results show that the performance of the 2D and 3D diffusion policies can scale with the generated dataset. Project page is this https URL. 

---
# Automated anatomy-based post-processing reduces false positives and improved interpretability of deep learning intracranial aneurysm detection 

**Authors**: Jisoo Kim, Chu-Hsuan Lin, Alberto Ceballos-Arroyo, Ping Liu, Huaizu Jiang, Shrikanth Yadav, Qi Wan, Lei Qin, Geoffrey S Young  

**Link**: [PDF](https://arxiv.org/pdf/2507.00832)  

**Abstract**: Introduction: Deep learning (DL) models can help detect intracranial aneurysms on CTA, but high false positive (FP) rates remain a barrier to clinical translation, despite improvement in model architectures and strategies like detection threshold tuning. We employed an automated, anatomy-based, heuristic-learning hybrid artery-vein segmentation post-processing method to further reduce FPs. Methods: Two DL models, CPM-Net and a deformable 3D convolutional neural network-transformer hybrid (3D-CNN-TR), were trained with 1,186 open-source CTAs (1,373 annotated aneurysms), and evaluated with 143 held-out private CTAs (218 annotated aneurysms). Brain, artery, vein, and cavernous venous sinus (CVS) segmentation masks were applied to remove possible FPs in the DL outputs that overlapped with: (1) brain mask; (2) vein mask; (3) vein more than artery masks; (4) brain plus vein mask; (5) brain plus vein more than artery masks. Results: CPM-Net yielded 139 true-positives (TP); 79 false-negative (FN); 126 FP. 3D-CNN-TR yielded 179 TP; 39 FN; 182 FP. FPs were commonly extracranial (CPM-Net 27.3%; 3D-CNN-TR 42.3%), venous (CPM-Net 56.3%; 3D-CNN-TR 29.1%), arterial (CPM-Net 11.9%; 3D-CNN-TR 53.3%), and non-vascular (CPM-Net 25.4%; 3D-CNN-TR 9.3%) structures. Method 5 performed best, reducing CPM-Net FP by 70.6% (89/126) and 3D-CNN-TR FP by 51.6% (94/182), without reducing TP, lowering the FP/case rate from 0.88 to 0.26 for CPM-NET, and from 1.27 to 0.62 for the 3D-CNN-TR. Conclusion: Anatomy-based, interpretable post-processing can improve DL-based aneurysm detection model performance. More broadly, automated, domain-informed, hybrid heuristic-learning processing holds promise for improving the performance and clinical acceptance of aneurysm detection models. 

---
# CAVALRY-V: A Large-Scale Generator Framework for Adversarial Attacks on Video MLLMs 

**Authors**: Jiaming Zhang, Rui Hu, Qing Guo, Wei Yang Bryan Lim  

**Link**: [PDF](https://arxiv.org/pdf/2507.00817)  

**Abstract**: Video Multimodal Large Language Models (V-MLLMs) have shown impressive capabilities in temporal reasoning and cross-modal understanding, yet their vulnerability to adversarial attacks remains underexplored due to unique challenges: complex cross-modal reasoning mechanisms, temporal dependencies, and computational constraints. We present CAVALRY-V (Cross-modal Language-Vision Adversarial Yielding for Videos), a novel framework that directly targets the critical interface between visual perception and language generation in V-MLLMs. Our approach introduces two key innovations: (1) a dual-objective semantic-visual loss function that simultaneously disrupts the model's text generation logits and visual representations to undermine cross-modal integration, and (2) a computationally efficient two-stage generator framework that combines large-scale pre-training for cross-model transferability with specialized fine-tuning for spatiotemporal coherence. Empirical evaluation on comprehensive video understanding benchmarks demonstrates that CAVALRY-V significantly outperforms existing attack methods, achieving 22.8% average improvement over the best baseline attacks on both commercial systems (GPT-4.1, Gemini 2.0) and open-source models (QwenVL-2.5, InternVL-2.5, Llava-Video, Aria, MiniCPM-o-2.6). Our framework achieves flexibility through implicit temporal coherence modeling rather than explicit regularization, enabling significant performance improvements even on image understanding (34.4% average gain). This capability demonstrates CAVALRY-V's potential as a foundational approach for adversarial research across multimodal systems. 

---
# PI-WAN: A Physics-Informed Wind-Adaptive Network for Quadrotor Dynamics Prediction in Unknown Environments 

**Authors**: Mengyun Wang, Bo Wang, Yifeng Niu, Chang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.00816)  

**Abstract**: Accurate dynamics modeling is essential for quadrotors to achieve precise trajectory tracking in various applications. Traditional physical knowledge-driven modeling methods face substantial limitations in unknown environments characterized by variable payloads, wind disturbances, and external perturbations. On the other hand, data-driven modeling methods suffer from poor generalization when handling out-of-distribution (OoD) data, restricting their effectiveness in unknown scenarios. To address these challenges, we introduce the Physics-Informed Wind-Adaptive Network (PI-WAN), which combines knowledge-driven and data-driven modeling methods by embedding physical constraints directly into the training process for robust quadrotor dynamics learning. Specifically, PI-WAN employs a Temporal Convolutional Network (TCN) architecture that efficiently captures temporal dependencies from historical flight data, while a physics-informed loss function applies physical principles to improve model generalization and robustness across previously unseen conditions. By incorporating real-time prediction results into a model predictive control (MPC) framework, we achieve improvements in closed-loop tracking performance. Comprehensive simulations and real-world flight experiments demonstrate that our approach outperforms baseline methods in terms of prediction accuracy, tracking precision, and robustness to unknown environments. 

---
# Many LLMs Are More Utilitarian Than One 

**Authors**: Anita Keshmirian, Razan Baltaji, Babak Hemmatian, Hadi Asghari, Lav R. Varshney  

**Link**: [PDF](https://arxiv.org/pdf/2507.00814)  

**Abstract**: Moral judgment is integral to large language model (LLM) alignment and social reasoning. As multi-agent systems gain prominence, it becomes crucial to understand how LLMs function collectively during collaboration, compared to individual agents. In human moral judgment, group deliberation leads to a utilitarian boost: a tendency to endorse norm violations that maximize benefits for the greatest number of people despite harms. We study whether a similar dynamic emerges in multi-agent LLM systems. We tested six models on well-established sets of moral dilemmas across two conditions: (1) Solo, where models reasoned independently, and (2) Group, where they engaged in multi-turn discussions in pairs or triads. In personal moral dilemmas, where agents must decide to directly harm one individual to maximize the utility for others, all models found moral violations to be more acceptable when part of a group than individually, similar to human experiments. Some models endorsed actions that maximized overall well-being, even if they benefited strangers over familiar individuals. Others became more willing to violate moral norms in groups. However, while human groups show a similar action bias, the mechanism for their utilitarian boost differs from LLMs. Whereas the human shift comes from heightened sensitivity to decision outcomes, LLM groups show either reduced norm sensitivity or enhanced impartiality. This suggests that while the surface behavior of LLM collectives mimics human group reasoning, the underlying drivers differ. We discuss the implications for AI alignment, multi-agent design, and artificial moral reasoning. 

---
# LD-RPS: Zero-Shot Unified Image Restoration via Latent Diffusion Recurrent Posterior Sampling 

**Authors**: Huaqiu Li, Yong Wang, Tongwen Huang, Hailang Huang, Haoqian Wang, Xiangxiang Chu  

**Link**: [PDF](https://arxiv.org/pdf/2507.00790)  

**Abstract**: Unified image restoration is a significantly challenging task in low-level vision. Existing methods either make tailored designs for specific tasks, limiting their generalizability across various types of degradation, or rely on training with paired datasets, thereby suffering from closed-set constraints. To address these issues, we propose a novel, dataset-free, and unified approach through recurrent posterior sampling utilizing a pretrained latent diffusion model. Our method incorporates the multimodal understanding model to provide sematic priors for the generative model under a task-blind condition. Furthermore, it utilizes a lightweight module to align the degraded input with the generated preference of the diffusion model, and employs recurrent refinement for posterior sampling. Extensive experiments demonstrate that our method outperforms state-of-the-art methods, validating its effectiveness and robustness. Our code and data will be available at this https URL. 

---
# Echoes of AI: Investigating the Downstream Effects of AI Assistants on Software Maintainability 

**Authors**: Markus Borg, Dave Hewett, Nadim Hagatulah, Noric Couderc, Emma Söderberg, Donald Graham, Uttam Kini, Dave Farley  

**Link**: [PDF](https://arxiv.org/pdf/2507.00788)  

**Abstract**: [Context] AI assistants, like GitHub Copilot and Cursor, are transforming software engineering. While several studies highlight productivity improvements, their impact on maintainability requires further investigation. [Objective] This study investigates whether co-development with AI assistants affects software maintainability, specifically how easily other developers can evolve the resulting source code. [Method] We conducted a two-phase controlled experiment involving 151 participants, 95% of whom were professional developers. In Phase 1, participants added a new feature to a Java web application, with or without AI assistance. In Phase 2, a randomized controlled trial, new participants evolved these solutions without AI assistance. [Results] AI-assisted development in Phase 1 led to a modest speedup in subsequent evolution and slightly higher average CodeHealth. Although neither difference was significant overall, the increase in CodeHealth was statistically significant when habitual AI users completed Phase 1. For Phase 1, we also observed a significant effect that corroborates previous productivity findings: using an AI assistant yielded a 30.7% median decrease in task completion time. Moreover, for habitual AI users, the mean speedup was 55.9%. [Conclusions] Our study adds to the growing evidence that AI assistants can effectively accelerate development. Moreover, we did not observe warning signs of degraded code-level maintainability. We recommend that future research focus on risks such as code bloat from excessive code generation and the build-up of cognitive debt as developers invest less mental effort during implementation. 

---
# LitBench: A Benchmark and Dataset for Reliable Evaluation of Creative Writing 

**Authors**: Daniel Fein, Sebastian Russo, Violet Xiang, Kabir Jolly, Rafael Rafailov, Nick Haber  

**Link**: [PDF](https://arxiv.org/pdf/2507.00769)  

**Abstract**: Evaluating creative writing generated by large language models (LLMs) remains challenging because open-ended narratives lack ground truths. Without performant automated evaluation methods, off-the-shelf (OTS) language models are employed as zero-shot judges, yet their reliability is unclear in this context. In pursuit of robust evaluation for creative writing, we introduce LitBench, the first standardized benchmark and paired dataset for creative writing verification, comprising a held-out test set of 2,480 debiased, human-labeled story comparisons drawn from Reddit and a 43,827-pair training corpus of human preference labels. Using LitBench, we (i) benchmark zero-shot LLM judges, (ii) train Bradley Terry and generative reward models, and (iii) conduct an online human study to validate reward model rankings on newly LLM-generated stories. Our benchmark identifies Claude-3.7-Sonnet as the strongest off-the-shelf judge, reaching 73% agreement with human preferences; among trained reward models, Bradley-Terry and Generative reward models both attain an accuracy of 78%, outperforming all off-the-shelf judges. An online human study further confirms that our trained reward models consistently align with human preferences in novel LLM-generated stories. We release LitBench and reward models at this https URL, providing a vetted resource for reliable, automated evaluation and optimization of creative writing systems. 

---
# LearnAFE: Circuit-Algorithm Co-design Framework for Learnable Audio Analog Front-End 

**Authors**: Jinhai Hu, Zhongyi Zhang, Cong Sheng Leow, Wang Ling Goh, Yuan Gao  

**Link**: [PDF](https://arxiv.org/pdf/2507.00755)  

**Abstract**: This paper presents a circuit-algorithm co-design framework for learnable analog front-end (AFE) in audio signal classification. Designing AFE and backend classifiers separately is a common practice but non-ideal, as shown in this paper. Instead, this paper proposes a joint optimization of the backend classifier with the AFE's transfer function to achieve system-level optimum. More specifically, the transfer function parameters of an analog bandpass filter (BPF) bank are tuned in a signal-to-noise ratio (SNR)-aware training loop for the classifier. Using a co-design loss function LBPF, this work shows superior optimization of both the filter bank and the classifier. Implemented in open-source SKY130 130nm CMOS process, the optimized design achieved 90.5%-94.2% accuracy for 10-keyword classification task across a wide range of input signal SNR from 5 dB to 20 dB, with only 22k classifier parameters. Compared to conventional approach, the proposed audio AFE achieves 8.7% and 12.9% reduction in power and capacitor area respectively. 

---
# Holmes: Towards Effective and Harmless Model Ownership Verification to Personalized Large Vision Models via Decoupling Common Features 

**Authors**: Linghui Zhu, Yiming Li, Haiqin Weng, Yan Liu, Tianwei Zhang, Shu-Tao Xia, Zhi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.00724)  

**Abstract**: Large vision models achieve remarkable performance in various downstream tasks, primarily by personalizing pre-trained models through fine-tuning with private and valuable local data, which makes the personalized model a valuable intellectual property for its owner. Similar to the era of traditional DNNs, model stealing attacks also pose significant risks to these personalized models. However, in this paper, we reveal that most existing defense methods (developed for traditional DNNs), typically designed for models trained from scratch, either introduce additional security risks, are prone to misjudgment, or are even ineffective for fine-tuned models. To alleviate these problems, this paper proposes a harmless model ownership verification method for personalized models by decoupling similar common features. In general, our method consists of three main stages. In the first stage, we create shadow models that retain common features of the victim model while disrupting dataset-specific features. We represent the dataset-specific features of the victim model by the output differences between the shadow and victim models. After that, a meta-classifier is trained to identify stolen models by determining whether suspicious models contain the dataset-specific features of the victim. In the third stage, we conduct model ownership verification by hypothesis test to mitigate randomness and enhance robustness. Extensive experiments on benchmark datasets verify the effectiveness of the proposed method in detecting different types of model stealing simultaneously. 

---
# TopoStreamer: Temporal Lane Segment Topology Reasoning in Autonomous Driving 

**Authors**: Yiming Yang, Yueru Luo, Bingkun He, Hongbin Lin, Suzhong Fu, Chao Yan, Kun Tang, Xinrui Yan, Chao Zheng, Shuguang Cui, Zhen Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.00709)  

**Abstract**: Lane segment topology reasoning constructs a comprehensive road network by capturing the topological relationships between lane segments and their semantic types. This enables end-to-end autonomous driving systems to perform road-dependent maneuvers such as turning and lane changing. However, the limitations in consistent positional embedding and temporal multiple attribute learning in existing methods hinder accurate roadnet reconstruction. To address these issues, we propose TopoStreamer, an end-to-end temporal perception model for lane segment topology reasoning. Specifically, TopoStreamer introduces three key improvements: streaming attribute constraints, dynamic lane boundary positional encoding, and lane segment denoising. The streaming attribute constraints enforce temporal consistency in both centerline and boundary coordinates, along with their classifications. Meanwhile, dynamic lane boundary positional encoding enhances the learning of up-to-date positional information within queries, while lane segment denoising helps capture diverse lane segment patterns, ultimately improving model performance. Additionally, we assess the accuracy of existing models using a lane boundary classification metric, which serves as a crucial measure for lane-changing scenarios in autonomous driving. On the OpenLane-V2 dataset, TopoStreamer demonstrates significant improvements over state-of-the-art methods, achieving substantial performance gains of +3.4% mAP in lane segment perception and +2.1% OLS in centerline perception tasks. 

---
# Audio-3DVG: Unified Audio - Point Cloud Fusion for 3D Visual Grounding 

**Authors**: Duc Cao-Dinh, Khai Le-Duc, Anh Dao, Bach Phan Tat, Chris Ngo, Duy M. H. Nguyen, Nguyen X. Khanh, Thanh Nguyen-Tang  

**Link**: [PDF](https://arxiv.org/pdf/2507.00669)  

**Abstract**: 3D Visual Grounding (3DVG) involves localizing target objects in 3D point clouds based on natural language. While prior work has made strides using textual descriptions, leveraging spoken language-known as Audio-based 3D Visual Grounding-remains underexplored and challenging. Motivated by advances in automatic speech recognition (ASR) and speech representation learning, we propose Audio-3DVG, a simple yet effective framework that integrates audio and spatial information for enhanced grounding. Rather than treating speech as a monolithic input, we decompose the task into two complementary components. First, we introduce Object Mention Detection, a multi-label classification task that explicitly identifies which objects are referred to in the audio, enabling more structured audio-scene reasoning. Second, we propose an Audio-Guided Attention module that captures interactions between candidate objects and relational speech cues, improving target discrimination in cluttered scenes. To support benchmarking, we synthesize audio descriptions for standard 3DVG datasets, including ScanRefer, Sr3D, and Nr3D. Experimental results demonstrate that Audio-3DVG not only achieves new state-of-the-art performance in audio-based grounding, but also competes with text-based methods-highlighting the promise of integrating spoken language into 3D vision tasks. 

---
# SAFER: Probing Safety in Reward Models with Sparse Autoencoder 

**Authors**: Sihang Li, Wei Shi, Ziyuan Xie, Tao Liang, Guojun Ma, Xiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.00665)  

**Abstract**: Reinforcement learning from human feedback (RLHF) is a key paradigm for aligning large language models (LLMs) with human values, yet the reward models at its core remain largely opaque. In this work, we present sparse Autoencoder For Enhanced Reward model (\textbf{SAFER}), a novel framework for interpreting and improving reward models through mechanistic analysis. Leveraging Sparse Autoencoders (SAEs), we uncover human-interpretable features in reward model activations, enabling insight into safety-relevant decision-making. We apply SAFER to safety-oriented preference datasets and quantify the salience of individual features by activation differences between chosen and rejected responses. Using these feature-level signals, we design targeted data poisoning and denoising strategies. Experiments show that SAFER can precisely degrade or enhance safety alignment with minimal data modification, without sacrificing general chat performance. Our approach contributes to interpreting, auditing and refining reward models in high-stakes LLM alignment tasks. Our codes are available at this https URL. \textit{This paper discusses topics related to large language model safety and may include discussions or examples that highlight potential risks or unsafe outcomes.} 

---
# MTCNet: Motion and Topology Consistency Guided Learning for Mitral Valve Segmentationin 4D Ultrasound 

**Authors**: Rusi Chen, Yuanting Yang, Jiezhi Yao, Hongning Song, Ji Zhang, Yongsong Zhou, Yuhao Huang, Ronghao Yang, Dan Jia, Yuhan Zhang, Xing Tao, Haoran Dou, Qing Zhou, Xin Yang, Dong Ni  

**Link**: [PDF](https://arxiv.org/pdf/2507.00660)  

**Abstract**: Mitral regurgitation is one of the most prevalent cardiac disorders. Four-dimensional (4D) ultrasound has emerged as the primary imaging modality for assessing dynamic valvular morphology. However, 4D mitral valve (MV) analysis remains challenging due to limited phase annotations, severe motion artifacts, and poor imaging quality. Yet, the absence of inter-phase dependency in existing methods hinders 4D MV analysis. To bridge this gap, we propose a Motion-Topology guided consistency network (MTCNet) for accurate 4D MV ultrasound segmentation in semi-supervised learning (SSL). MTCNet requires only sparse end-diastolic and end-systolic annotations. First, we design a cross-phase motion-guided consistency learning strategy, utilizing a bi-directional attention memory bank to propagate spatio-temporal features. This enables MTCNet to achieve excellent performance both per- and inter-phase. Second, we devise a novel topology-guided correlation regularization that explores physical prior knowledge to maintain anatomically plausible. Therefore, MTCNet can effectively leverage structural correspondence between labeled and unlabeled phases. Extensive evaluations on the first largest 4D MV dataset, with 1408 phases from 160 patients, show that MTCNet performs superior cross-phase consistency compared to other advanced methods (Dice: 87.30%, HD: 1.75mm). Both the code and the dataset are available at this https URL. 

---
# Generative Exaggeration in LLM Social Agents: Consistency, Bias, and Toxicity 

**Authors**: Jacopo Nudo, Mario Edoardo Pandolfo, Edoardo Loru, Mattia Samory, Matteo Cinelli, Walter Quattrociocchi  

**Link**: [PDF](https://arxiv.org/pdf/2507.00657)  

**Abstract**: We investigate how Large Language Models (LLMs) behave when simulating political discourse on social media. Leveraging 21 million interactions on X during the 2024 U.S. presidential election, we construct LLM agents based on 1,186 real users, prompting them to reply to politically salient tweets under controlled conditions. Agents are initialized either with minimal ideological cues (Zero Shot) or recent tweet history (Few Shot), allowing one-to-one comparisons with human replies. We evaluate three model families (Gemini, Mistral, and DeepSeek) across linguistic style, ideological consistency, and toxicity. We find that richer contextualization improves internal consistency but also amplifies polarization, stylized signals, and harmful language. We observe an emergent distortion that we call "generation exaggeration": a systematic amplification of salient traits beyond empirical baselines. Our analysis shows that LLMs do not emulate users, they reconstruct them. Their outputs, indeed, reflect internal optimization dynamics more than observed behavior, introducing structural biases that compromise their reliability as social proxies. This challenges their use in content moderation, deliberative simulations, and policy modeling. 

---
# Cognitive Load-Aware Inference: A Neuro-Symbolic Framework for Optimizing the Token Economy of Large Language Models 

**Authors**: Yilun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.00653)  

**Abstract**: The escalating computational costs of Large Language Model (LLM) inference have become a critical barrier to their widespread and sustainable deployment. While existing optimization strategies are effective, they are predominantly based on statistical heuristics or architectural modifications, lacking a guiding cognitive theory to manage the inference process itself. This paper aims to bridge this gap by introducing a novel paradigm: the Cognitive Load-Aware Inference (CLAI) framework, which operationalizes principles from Cognitive Load Theory (CLT) and neuroscience for LLM inference. We formalize the concepts of Intrinsic Cognitive Load, Extraneous Cognitive Load, and Germane Cognitive Load into quantifiable LLM metrics ($ICL_{LLM}$, $ECL_{LLM}$, and $GCL_{LLM}$), thereby reframing the inference process as a cognitive economics optimization problem: based on the intrinsic complexity of a problem ($ICL_{LLM}$), minimize wasteful computation ($ECL_{LLM}$), and strategically allocate the token budget to productive reasoning ($GCL_{LLM}$). We propose two implementation paths: CLAI-Prompt, a zero-shot method that guides a base LLM through cognitive control steps via a structured meta-prompt, and CLAI-Tune, a fine-tuned model that internalizes these principles for spontaneous cognitive economy. Across a range of benchmarks in complex reasoning, long-context question answering, and code generation, our methods achieve significant reductions in token consumption (up to 45\%) without sacrificing accuracy. Furthermore, CLAI-Tune exhibits an emergent ability to autonomously decompose difficult problems, a key characteristic of human expert cognition. This work demonstrates that by emulating the brain's resource management strategies, we can build more efficient, robust, and capable artificial intelligence systems. 

---
# Horus: A Protocol for Trustless Delegation Under Uncertainty 

**Authors**: David Shi, Kevin Joo  

**Link**: [PDF](https://arxiv.org/pdf/2507.00631)  

**Abstract**: Correctness is an emergent property of systems where exposing error is cheaper than committing it. In dynamic, low-trust environments, autonomous AI agents benefit from delegating work to sub-agents, yet correctness cannot be assured through upfront specification or centralized oversight. We propose a protocol that enforces correctness through collateralized claims in a recursive verification game. Tasks are published as intents, and solvers compete to fulfill them. Selected solvers carry out tasks under risk, with correctness checked post hoc by verifiers. Any challenger can challenge a result by staking against it to trigger the verification process. Incorrect agents are slashed and correct opposition is rewarded, with an escalation path that penalizes erroneous verifiers themselves. When incentives are aligned across solvers, challengers, and verifiers, falsification conditions make correctness the Nash equilibrium. 

---
# Physics-Informed Neural ODEs for Temporal Dynamics Modeling in Cardiac T1 Mapping 

**Authors**: Nuno Capitão, Yi Zhang, Yidong Zhao, Qian Tao  

**Link**: [PDF](https://arxiv.org/pdf/2507.00613)  

**Abstract**: Spin-lattice relaxation time ($T_1$) is an important biomarker in cardiac parametric mapping for characterizing myocardial tissue and diagnosing cardiomyopathies. Conventional Modified Look-Locker Inversion Recovery (MOLLI) acquires 11 breath-hold baseline images with interleaved rest periods to ensure mapping accuracy. However, prolonged scanning can be challenging for patients with poor breathholds, often leading to motion artifacts that degrade image quality. In addition, $T_1$ mapping requires voxel-wise nonlinear fitting to a signal recovery model involving an iterative estimation process. Recent studies have proposed deep-learning approaches for rapid $T_1$ mapping using shortened sequences to reduce acquisition time for patient comfort. Nevertheless, existing methods overlook important physics constraints, limiting interpretability and generalization. In this work, we present an accelerated, end-to-end $T_1$ mapping framework leveraging Physics-Informed Neural Ordinary Differential Equations (ODEs) to model temporal dynamics and address these challenges. Our method achieves high-accuracy $T_1$ estimation from a sparse subset of baseline images and ensures efficient null index estimation at test time. Specifically, we develop a continuous-time LSTM-ODE model to enable selective Look-Locker (LL) data acquisition with arbitrary time lags. Experimental results show superior performance in $T_1$ estimation for both native and post-contrast sequences and demonstrate the strong benefit of our physics-based formulation over direct data-driven $T_1$ priors. 

---
# Residual Reward Models for Preference-based Reinforcement Learning 

**Authors**: Chenyang Cao, Miguel Rogel-García, Mohamed Nabail, Xueqian Wang, Nicholas Rhinehart  

**Link**: [PDF](https://arxiv.org/pdf/2507.00611)  

**Abstract**: Preference-based Reinforcement Learning (PbRL) provides a way to learn high-performance policies in environments where the reward signal is hard to specify, avoiding heuristic and time-consuming reward design. However, PbRL can suffer from slow convergence speed since it requires training in a reward model. Prior work has proposed learning a reward model from demonstrations and fine-tuning it using preferences. However, when the model is a neural network, using different loss functions for pre-training and fine-tuning can pose challenges to reliable optimization. In this paper, we propose a method to effectively leverage prior knowledge with a Residual Reward Model (RRM). An RRM assumes that the true reward of the environment can be split into a sum of two parts: a prior reward and a learned reward. The prior reward is a term available before training, for example, a user's ``best guess'' reward function, or a reward function learned from inverse reinforcement learning (IRL), and the learned reward is trained with preferences. We introduce state-based and image-based versions of RRM and evaluate them on several tasks in the Meta-World environment suite. Experimental results show that our method substantially improves the performance of a common PbRL method. Our method achieves performance improvements for a variety of different types of prior rewards, including proxy rewards, a reward obtained from IRL, and even a negated version of the proxy reward. We also conduct experiments with a Franka Panda to show that our method leads to superior performance on a real robot. It significantly accelerates policy learning for different tasks, achieving success in fewer steps than the baseline. The videos are presented at this https URL. 

---
# Mixture of Reasonings: Teach Large Language Models to Reason with Adaptive Strategies 

**Authors**: Tao Xiong, Xavier Hu, Wenyan Fan, Shengyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.00606)  

**Abstract**: Large language models (LLMs) excel in complex tasks through advanced prompting techniques like Chain-of-Thought (CoT) and Tree-of-Thought (ToT), but their reliance on manually crafted, task-specific prompts limits adaptability and efficiency. We introduce Mixture of Reasoning (MoR), a training framework that embeds diverse reasoning strategies into LLMs for autonomous, task-adaptive reasoning without external prompt engineering. MoR has two phases: Thought Generation, creating reasoning chain templates with models like GPT-4o, and SFT Dataset Construction, pairing templates with benchmark datasets for supervised this http URL experiments show that MoR significantly enhances performance, with MoR150 achieving 0.730 (2.2% improvement) using CoT prompting and 0.734 (13.5% improvement) compared to baselines. MoR eliminates the need for task-specific prompts, offering a generalizable solution for robust reasoning across diverse tasks. 

---
# High-resolution spatial memory requires grid-cell-like neural codes 

**Authors**: Madison Cotteret, Christopher J. Kymn, Hugh Greatorex, Martin Ziegler, Elisabetta Chicca, Friedrich T. Sommer  

**Link**: [PDF](https://arxiv.org/pdf/2507.00598)  

**Abstract**: Continuous attractor networks (CANs) are widely used to model how the brain temporarily retains continuous behavioural variables via persistent recurrent activity, such as an animal's position in an environment. However, this memory mechanism is very sensitive to even small imperfections, such as noise or heterogeneity, which are both common in biological systems. Previous work has shown that discretising the continuum into a finite set of discrete attractor states provides robustness to these imperfections, but necessarily reduces the resolution of the represented variable, creating a dilemma between stability and resolution. We show that this stability-resolution dilemma is most severe for CANs using unimodal bump-like codes, as in traditional models. To overcome this, we investigate sparse binary distributed codes based on random feature embeddings, in which neurons have spatially-periodic receptive fields. We demonstrate theoretically and with simulations that such grid-cell-like codes enable CANs to achieve both high stability and high resolution simultaneously. The model extends to embedding arbitrary nonlinear manifolds into a CAN, such as spheres or tori, and generalises linear path integration to integration along freely-programmable on-manifold vector fields. Together, this work provides a theory of how the brain could robustly represent continuous variables with high resolution and perform flexible computations over task-relevant manifolds. 

---
# Quantum Circuit Structure Optimization for Quantum Reinforcement Learning 

**Authors**: Seok Bin Son, Joongheon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2507.00589)  

**Abstract**: Reinforcement learning (RL) enables agents to learn optimal policies through environmental interaction. However, RL suffers from reduced learning efficiency due to the curse of dimensionality in high-dimensional spaces. Quantum reinforcement learning (QRL) addresses this issue by leveraging superposition and entanglement in quantum computing, allowing efficient handling of high-dimensional problems with fewer resources. QRL combines quantum neural networks (QNNs) with RL, where the parameterized quantum circuit (PQC) acts as the core computational module. The PQC performs linear and nonlinear transformations through gate operations, similar to hidden layers in classical neural networks. Previous QRL studies, however, have used fixed PQC structures based on empirical intuition without verifying their optimality. This paper proposes a QRL-NAS algorithm that integrates quantum neural architecture search (QNAS) to optimize PQC structures within QRL. Experiments demonstrate that QRL-NAS achieves higher rewards than QRL with fixed circuits, validating its effectiveness and practical utility. 

---
# AI-Generated Video Detection via Perceptual Straightening 

**Authors**: Christian Internò, Robert Geirhos, Markus Olhofer, Sunny Liu, Barbara Hammer, David Klindt  

**Link**: [PDF](https://arxiv.org/pdf/2507.00583)  

**Abstract**: The rapid advancement of generative AI enables highly realistic synthetic videos, posing significant challenges for content authentication and raising urgent concerns about misuse. Existing detection methods often struggle with generalization and capturing subtle temporal inconsistencies. We propose ReStraV(Representation Straightening Video), a novel approach to distinguish natural from AI-generated videos. Inspired by the "perceptual straightening" hypothesis -- which suggests real-world video trajectories become more straight in neural representation domain -- we analyze deviations from this expected geometric property. Using a pre-trained self-supervised vision transformer (DINOv2), we quantify the temporal curvature and stepwise distance in the model's representation domain. We aggregate statistics of these measures for each video and train a classifier. Our analysis shows that AI-generated videos exhibit significantly different curvature and distance patterns compared to real videos. A lightweight classifier achieves state-of-the-art detection performance (e.g., 97.17% accuracy and 98.63% AUROC on the VidProM benchmark), substantially outperforming existing image- and video-based methods. ReStraV is computationally efficient, it is offering a low-cost and effective detection solution. This work provides new insights into using neural representation geometry for AI-generated video detection. 

---
# TUM-MiKaNi at SemEval-2025 Task 3: Towards Multilingual and Knowledge-Aware Non-factual Hallucination Identification 

**Authors**: Miriam Anschütz, Ekaterina Gikalo, Niklas Herbster, Georg Groh  

**Link**: [PDF](https://arxiv.org/pdf/2507.00579)  

**Abstract**: Hallucinations are one of the major problems of LLMs, hindering their trustworthiness and deployment to wider use cases. However, most of the research on hallucinations focuses on English data, neglecting the multilingual nature of LLMs. This paper describes our submission to the SemEval-2025 Task-3 - Mu-SHROOM, the Multilingual Shared-task on Hallucinations and Related Observable Overgeneration Mistakes. We propose a two-part pipeline that combines retrieval-based fact verification against Wikipedia with a BERT-based system fine-tuned to identify common hallucination patterns. Our system achieves competitive results across all languages, reaching top-10 results in eight languages, including English. Moreover, it supports multiple languages beyond the fourteen covered by the shared task. This multilingual hallucination identifier can help to improve LLM outputs and their usefulness in the future. 

---
# BadViM: Backdoor Attack against Vision Mamba 

**Authors**: Yinghao Wu, Liyan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.00577)  

**Abstract**: Vision State Space Models (SSMs), particularly architectures like Vision Mamba (ViM), have emerged as promising alternatives to Vision Transformers (ViTs). However, the security implications of this novel architecture, especially their vulnerability to backdoor attacks, remain critically underexplored. Backdoor attacks aim to embed hidden triggers into victim models, causing the model to misclassify inputs containing these triggers while maintaining normal behavior on clean inputs. This paper investigates the susceptibility of ViM to backdoor attacks by introducing BadViM, a novel backdoor attack framework specifically designed for Vision Mamba. The proposed BadViM leverages a Resonant Frequency Trigger (RFT) that exploits the frequency sensitivity patterns of the victim model to create stealthy, distributed triggers. To maximize attack efficacy, we propose a Hidden State Alignment loss that strategically manipulates the internal representations of model by aligning the hidden states of backdoor images with those of target classes. Extensive experimental results demonstrate that BadViM achieves superior attack success rates while maintaining clean data accuracy. Meanwhile, BadViM exhibits remarkable resilience against common defensive measures, including PatchDrop, PatchShuffle and JPEG compression, which typically neutralize normal backdoor attacks. 

---
# Inverse Design in Nanophotonics via Representation Learning 

**Authors**: Reza Marzban, Ali Adibi, Raphael Pestourie  

**Link**: [PDF](https://arxiv.org/pdf/2507.00546)  

**Abstract**: Inverse design in nanophotonics, the computational discovery of structures achieving targeted electromagnetic (EM) responses, has become a key tool for recent optical advances. Traditional intuition-driven or iterative optimization methods struggle with the inherently high-dimensional, non-convex design spaces and the substantial computational demands of EM simulations. Recently, machine learning (ML) has emerged to address these bottlenecks effectively. This review frames ML-enhanced inverse design methodologies through the lens of representation learning, classifying them into two categories: output-side and input-side approaches. Output-side methods use ML to learn a representation in the solution space to create a differentiable solver that accelerates optimization. Conversely, input-side techniques employ ML to learn compact, latent-space representations of feasible device geometries, enabling efficient global exploration through generative models. Each strategy presents unique trade-offs in data requirements, generalization capacity, and novel design discovery potentials. Hybrid frameworks that combine physics-based optimization with data-driven representations help escape poor local optima, improve scalability, and facilitate knowledge transfer. We conclude by highlighting open challenges and opportunities, emphasizing complexity management, geometry-independent representations, integration of fabrication constraints, and advancements in multiphysics co-designs. 

---
# Not All Attention Heads Are What You Need: Refining CLIP's Image Representation with Attention Ablation 

**Authors**: Feng Lin, Marco Chen, Haokui Zhang, Xiaotian Yu, Guangming Lu, Rong Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2507.00537)  

**Abstract**: This paper studies the role of attention heads in CLIP's image encoder. While CLIP has exhibited robust performance across diverse applications, we hypothesize that certain attention heads negatively affect final representations and that ablating them can improve performance in downstream tasks. To capitalize on this insight, we propose a simple yet effective method, called Attention Ablation Technique (AAT), to suppress the contribution of specific heads by manipulating attention weights. By integrating two alternative strategies tailored for different application scenarios, AAT systematically identifies and ablates detrimental attention heads to enhance representation quality. Experiments demonstrate that AAT consistently improves downstream task performance across various domains, boosting recall rate by up to 11.1% on CLIP-family models for cross-modal retrieval. The results highlight the potential of AAT to effectively refine large-scale vision-language models with virtually no increase in inference cost. 

---
# Rethinking Group Recommender Systems in the Era of Generative AI: From One-Shot Recommendations to Agentic Group Decision Support 

**Authors**: Dietmar Jannach, Amra Delić, Francesco Ricci, Markus Zanker  

**Link**: [PDF](https://arxiv.org/pdf/2507.00535)  

**Abstract**: More than twenty-five years ago, first ideas were developed on how to design a system that can provide recommendations to groups of users instead of individual users. Since then, a rich variety of algorithmic proposals were published, e.g., on how to acquire individual preferences, how to aggregate them, and how to generate recommendations for groups of users. However, despite the rich literature on the topic, barely any examples of real-world group recommender systems can be found. This lets us question common assumptions in academic research, in particular regarding communication processes in a group and how recommendation-supported decisions are made. In this essay, we argue that these common assumptions and corresponding system designs often may not match the needs or expectations of users. We thus call for a reorientation in this research area, leveraging the capabilities of modern Generative AI assistants like ChatGPT. Specifically, as one promising future direction, we envision group recommender systems to be systems where human group members interact in a chat and an AI-based group recommendation agent assists the decision-making process in an agentic way. Ultimately, this shall lead to a more natural group decision-making environment and finally to wider adoption of group recommendation systems in practice. 

---
# Box-QAymo: Box-Referring VQA Dataset for Autonomous Driving 

**Authors**: Djamahl Etchegaray, Yuxia Fu, Zi Huang, Yadan Luo  

**Link**: [PDF](https://arxiv.org/pdf/2507.00525)  

**Abstract**: Interpretable communication is essential for safe and trustworthy autonomous driving, yet current vision-language models (VLMs) often operate under idealized assumptions and struggle to capture user intent in real-world scenarios. Existing driving-oriented VQA datasets are limited to full-scene descriptions or waypoint prediction, preventing the assessment of whether VLMs can respond to localized user-driven queries. We introduce Box-QAymo, a box-referring dataset and benchmark designed to both evaluate and finetune VLMs on spatial and temporal reasoning over user-specified objects. Users express intent by drawing bounding boxes, offering a fast and intuitive interface for focused queries in complex scenes. Specifically, we propose a hierarchical evaluation protocol that begins with binary sanity-check questions to assess basic model capacities, and progresses to (1) attribute prediction for box-referred objects, (2) motion understanding of target instances, and (3) spatiotemporal motion reasoning over inter-object dynamics across frames. To support this, we crowd-sourced fine-grained object classes and visual attributes that reflect the complexity drivers encounter, and extract object trajectories to construct temporally grounded QA pairs. Rigorous quality control through negative sampling, temporal consistency checks, and difficulty-aware balancing guarantee dataset robustness and diversity. Our comprehensive evaluation reveals significant limitations in current VLMs when queried about perception questions, highlighting the gap in achieving real-world performance. This work provides a foundation for developing more robust and interpretable autonomous driving systems that can communicate effectively with users under real-world conditions. Project page and dataset are available at this https URL. 

---
# Customer Service Representative's Perception of the AI Assistant in an Organization's Call Center 

**Authors**: Kai Qin, Kexin Du, Yimeng Chen, Yueyan Liu, Jie Cai, Zhiqiang Nie, Nan Gao, Guohui Wei, Shengzhu Wang, Chun Yu  

**Link**: [PDF](https://arxiv.org/pdf/2507.00513)  

**Abstract**: The integration of various AI tools creates a complex socio-technical environment where employee-customer interactions form the core of work practices. This study investigates how customer service representatives (CSRs) at the power grid service customer service call center perceive AI assistance in their interactions with customers. Through a field visit and semi-structured interviews with 13 CSRs, we found that AI can alleviate some traditional burdens during the call (e.g., typing and memorizing) but also introduces new burdens (e.g., earning, compliance, psychological burdens). This research contributes to a more nuanced understanding of AI integration in organizational settings and highlights the efforts and burdens undertaken by CSRs to adapt to the updated system. 

---
# TeamCMU at Touché: Adversarial Co-Evolution for Advertisement Integration and Detection in Conversational Search 

**Authors**: To Eun Kim, João Coelho, Gbemileke Onilude, Jai Singh  

**Link**: [PDF](https://arxiv.org/pdf/2507.00509)  

**Abstract**: As conversational search engines increasingly adopt generation-based paradigms powered by Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG), the integration of advertisements into generated responses presents both commercial opportunities and challenges for user experience. Unlike traditional search, where advertisements are clearly delineated, generative systems blur the boundary between informational content and promotional material, raising concerns around transparency and trust. In this work, we propose a modular pipeline for advertisement management in RAG-based conversational systems, consisting of an ad-rewriter for seamless ad integration and a robust ad-classifier for detection. We leverage synthetic data to train high-performing classifiers, which are then used to guide two complementary ad-integration strategies: supervised fine-tuning of the ad-rewriter and a best-of-N sampling approach that selects the least detectable ad-integrated response among multiple candidates. Our evaluation focuses on two core questions: the effectiveness of ad classifiers in detecting diverse ad integration strategies, and the training methods that best support coherent, minimally intrusive ad insertion. Experimental results show that our ad-classifier, trained on synthetic advertisement data inspired by marketing strategies and enhanced through curriculum learning, achieves robust detection performance. Additionally, we demonstrate that classifier-guided optimization, through both fine-tuning and best-of-N sampling, significantly improves ad stealth, enabling more seamless integration. These findings contribute an adversarial co-evolution framework for developing more sophisticated ad-aware generative search systems and robust ad classifiers. 

---
# Visual Anagrams Reveal Hidden Differences in Holistic Shape Processing Across Vision Models 

**Authors**: Fenil R. Doshi, Thomas Fel, Talia Konkle, George Alvarez  

**Link**: [PDF](https://arxiv.org/pdf/2507.00493)  

**Abstract**: Humans are able to recognize objects based on both local texture cues and the configuration of object parts, yet contemporary vision models primarily harvest local texture cues, yielding brittle, non-compositional features. Work on shape-vs-texture bias has pitted shape and texture representations in opposition, measuring shape relative to texture, ignoring the possibility that models (and humans) can simultaneously rely on both types of cues, and obscuring the absolute quality of both types of representation. We therefore recast shape evaluation as a matter of absolute configural competence, operationalized by the Configural Shape Score (CSS), which (i) measures the ability to recognize both images in Object-Anagram pairs that preserve local texture while permuting global part arrangement to depict different object categories. Across 86 convolutional, transformer, and hybrid models, CSS (ii) uncovers a broad spectrum of configural sensitivity with fully self-supervised and language-aligned transformers -- exemplified by DINOv2, SigLIP2 and EVA-CLIP -- occupying the top end of the CSS spectrum. Mechanistic probes reveal that (iii) high-CSS networks depend on long-range interactions: radius-controlled attention masks abolish performance showing a distinctive U-shaped integration profile, and representational-similarity analyses expose a mid-depth transition from local to global coding. A BagNet control remains at chance (iv), ruling out "border-hacking" strategies. Finally, (v) we show that configural shape score also predicts other shape-dependent evals. Overall, we propose that the path toward truly robust, generalizable, and human-like vision systems may not lie in forcing an artificial choice between shape and texture, but rather in architectural and learning frameworks that seamlessly integrate both local-texture and global configural shape. 

---
# Twill: Scheduling Compound AI Systems on Heterogeneous Mobile Edge Platforms 

**Authors**: Zain Taufique, Aman Vyas, Antonio Miele, Pasi Liljeberg, Anil Kanduri  

**Link**: [PDF](https://arxiv.org/pdf/2507.00491)  

**Abstract**: Compound AI (cAI) systems chain multiple AI models to solve complex problems. cAI systems are typically composed of deep neural networks (DNNs), transformers, and large language models (LLMs), exhibiting a high degree of computational diversity and dynamic workload variation. Deploying cAI services on mobile edge platforms poses a significant challenge in scheduling concurrent DNN-transformer inference tasks, which arrive dynamically in an unknown sequence. Existing mobile edge AI inference strategies manage multi-DNN or transformer-only workloads, relying on design-time profiling, and cannot handle concurrent inference of DNNs and transformers required by cAI systems. In this work, we address the challenge of scheduling cAI systems on heterogeneous mobile edge platforms. We present Twill, a run-time framework to handle concurrent inference requests of cAI workloads through task affinity-aware cluster mapping and migration, priority-aware task freezing/unfreezing, and DVFS, while minimizing inference latency within power budgets. We implement and deploy our Twill framework on the Nvidia Jetson Orin NX platform. We evaluate Twill against state-of-the-art edge AI inference techniques over contemporary DNNs and LLMs, reducing inference latency by 54% on average, while honoring power budgets. 

---
# PNAct: Crafting Backdoor Attacks in Safe Reinforcement Learning 

**Authors**: Weiran Guo, Guanjun Liu, Ziyuan Zhou, Ling Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.00485)  

**Abstract**: Reinforcement Learning (RL) is widely used in tasks where agents interact with an environment to maximize rewards. Building on this foundation, Safe Reinforcement Learning (Safe RL) incorporates a cost metric alongside the reward metric, ensuring that agents adhere to safety constraints during decision-making. In this paper, we identify that Safe RL is vulnerable to backdoor attacks, which can manipulate agents into performing unsafe actions. First, we introduce the relevant concepts and evaluation metrics for backdoor attacks in Safe RL. It is the first attack framework in the Safe RL field that involves both Positive and Negative Action sample (PNAct) is to implant backdoors, where positive action samples provide reference actions and negative action samples indicate actions to be avoided. We theoretically point out the properties of PNAct and design an attack algorithm. Finally, we conduct experiments to evaluate the effectiveness of our proposed backdoor attack framework, evaluating it with the established metrics. This paper highlights the potential risks associated with Safe RL and underscores the feasibility of such attacks. Our code and supplementary material are available at this https URL. 

---
# Physics-Aware Style Transfer for Adaptive Holographic Reconstruction 

**Authors**: Chanseok Lee, Fakhriyya Mammadova, Jiseong Barg, Mooseok Jang  

**Link**: [PDF](https://arxiv.org/pdf/2507.00482)  

**Abstract**: Inline holographic imaging presents an ill-posed inverse problem of reconstructing objects' complex amplitude from recorded diffraction patterns. Although recent deep learning approaches have shown promise over classical phase retrieval algorithms, they often require high-quality ground truth datasets of complex amplitude maps to achieve a statistical inverse mapping operation between the two domains. Here, we present a physics-aware style transfer approach that interprets the object-to-sensor distance as an implicit style within diffraction patterns. Using the style domain as the intermediate domain to construct cyclic image translation, we show that the inverse mapping operation can be learned in an adaptive manner only with datasets composed of intensity measurements. We further demonstrate its biomedical applicability by reconstructing the morphology of dynamically flowing red blood cells, highlighting its potential for real-time, label-free imaging. As a framework that leverages physical cues inherently embedded in measurements, the presented method offers a practical learning strategy for imaging applications where ground truth is difficult or impossible to obtain. 

---
# Diversity Conscious Refined Random Forest 

**Authors**: Sijan Bhattarai, Saurav Bhandari, Girija Bhusal, Saroj Shakya, Tapendra Pandey  

**Link**: [PDF](https://arxiv.org/pdf/2507.00467)  

**Abstract**: Random Forest (RF) is a widely used ensemble learning technique known for its robust classification performance across diverse domains. However, it often relies on hundreds of trees and all input features, leading to high inference cost and model redundancy. In this work, our goal is to grow trees dynamically only on informative features and then enforce maximal diversity by clustering and retaining uncorrelated trees. Therefore, we propose a Refined Random Forest Classifier that iteratively refines itself by first removing the least informative features and then analytically determines how many new trees should be grown, followed by correlation-based clustering to remove redundant trees. The classification accuracy of our model was compared against the standard RF on the same number of trees. Experiments on 8 multiple benchmark datasets, including binary and multiclass datasets, demonstrate that the proposed model achieves improved accuracy compared to standard RF. 

---
# Novel Complex-Valued Hopfield Neural Networks with Phase and Magnitude Quantization 

**Authors**: Garimella Ramamurthy, Marcos Eduardo Valle, Tata Jagannadha Swamy  

**Link**: [PDF](https://arxiv.org/pdf/2507.00461)  

**Abstract**: This research paper introduces two novel complex-valued Hopfield neural networks (CvHNNs) that incorporate phase and magnitude quantization. The first CvHNN employs a ceiling-type activation function that operates on the rectangular coordinate representation of the complex net contribution. The second CvHNN similarly incorporates phase and magnitude quantization but utilizes a ceiling-type activation function based on the polar coordinate representation of the complex net contribution. The proposed CvHNNs, with their phase and magnitude quantization, significantly increase the number of states compared to existing models in the literature, thereby expanding the range of potential applications for CvHNNs. 

---
# Process-aware and high-fidelity microstructure generation using stable diffusion 

**Authors**: Hoang Cuong Phan, Minh Tien Tran, Chihun Lee, Hoheok Kim, Sehyok Oh, Dong-Kyu Kim, Ho Won Lee  

**Link**: [PDF](https://arxiv.org/pdf/2507.00459)  

**Abstract**: Synthesizing realistic microstructure images conditioned on processing parameters is crucial for understanding process-structure relationships in materials design. However, this task remains challenging due to limited training micrographs and the continuous nature of processing variables. To overcome these challenges, we present a novel process-aware generative modeling approach based on Stable Diffusion 3.5 Large (SD3.5-Large), a state-of-the-art text-to-image diffusion model adapted for microstructure generation. Our method introduces numeric-aware embeddings that encode continuous variables (annealing temperature, time, and magnification) directly into the model's conditioning, enabling controlled image generation under specified process conditions and capturing process-driven microstructural variations. To address data scarcity and computational constraints, we fine-tune only a small fraction of the model's weights via DreamBooth and Low-Rank Adaptation (LoRA), efficiently transferring the pre-trained model to the materials domain. We validate realism using a semantic segmentation model based on a fine-tuned U-Net with a VGG16 encoder on 24 labeled micrographs. It achieves 97.1% accuracy and 85.7% mean IoU, outperforming previous methods. Quantitative analyses using physical descriptors and spatial statistics show strong agreement between synthetic and real microstructures. Specifically, two-point correlation and lineal-path errors remain below 2.1% and 0.6%, respectively. Our method represents the first adaptation of SD3.5-Large for process-aware microstructure generation, offering a scalable approach for data-driven materials design. 

---
# ATSTrack: Enhancing Visual-Language Tracking by Aligning Temporal and Spatial Scales 

**Authors**: Yihao Zhen, Qiang Wang, Yu Qiao, Liangqiong Qu, Huijie Fan  

**Link**: [PDF](https://arxiv.org/pdf/2507.00454)  

**Abstract**: A main challenge of Visual-Language Tracking (VLT) is the misalignment between visual inputs and language descriptions caused by target movement. Previous trackers have explored many effective feature modification methods to preserve more aligned features. However, an important yet unexplored factor ultimately hinders their capability, which is the inherent differences in the temporal and spatial scale of information between visual and language inputs. To address this issue, we propose a novel visual-language tracker that enhances the effect of feature modification by \textbf{A}ligning \textbf{T}emporal and \textbf{S}patial scale of different input components, named as \textbf{ATSTrack}. Specifically, we decompose each language description into phrases with different attributes based on their temporal and spatial correspondence with visual inputs, and modify their features in a fine-grained manner. Moreover, we introduce a Visual-Language token that comprises modified linguistic information from the previous frame to guide the model to extract visual features that are more relevant to language description, thereby reducing the impact caused by the differences in spatial scale. Experimental results show that our proposed ATSTrack achieves performance comparable to existing methods. Our code will be released. 

---
# Best Agent Identification for General Game Playing 

**Authors**: Matthew Stephenson, Alex Newcombe, Eric Piette, Dennis Soemers  

**Link**: [PDF](https://arxiv.org/pdf/2507.00451)  

**Abstract**: We present an efficient and generalised procedure to accurately identify the best performing algorithm for each sub-task in a multi-problem domain. Our approach treats this as a set of best arm identification problems for multi-armed bandits, where each bandit corresponds to a specific task and each arm corresponds to a specific algorithm or agent. We propose an optimistic selection process based on the Wilson score interval (Optimistic-WS) that ranks each arm across all bandits in terms of their potential regret reduction. We evaluate the performance of Optimistic-WS on two of the most popular general game domains, the General Video Game AI (GVGAI) framework and the Ludii general game playing system, with the goal of identifying the highest performing agent for each game within a limited number of trials. Compared to previous best arm identification algorithms for multi-armed bandits, our results demonstrate a substantial performance improvement in terms of average simple regret. This novel approach can be used to significantly improve the quality and accuracy of agent evaluation procedures for general game frameworks, as well as other multi-task domains with high algorithm runtimes. 

---
# Iterative Distillation for Reward-Guided Fine-Tuning of Diffusion Models in Biomolecular Design 

**Authors**: Xingyu Su, Xiner Li, Masatoshi Uehara, Sunwoo Kim, Yulai Zhao, Gabriele Scalia, Ehsan Hajiramezanali, Tommaso Biancalani, Degui Zhi, Shuiwang Ji  

**Link**: [PDF](https://arxiv.org/pdf/2507.00445)  

**Abstract**: We address the problem of fine-tuning diffusion models for reward-guided generation in biomolecular design. While diffusion models have proven highly effective in modeling complex, high-dimensional data distributions, real-world applications often demand more than high-fidelity generation, requiring optimization with respect to potentially non-differentiable reward functions such as physics-based simulation or rewards based on scientific knowledge. Although RL methods have been explored to fine-tune diffusion models for such objectives, they often suffer from instability, low sample efficiency, and mode collapse due to their on-policy nature. In this work, we propose an iterative distillation-based fine-tuning framework that enables diffusion models to optimize for arbitrary reward functions. Our method casts the problem as policy distillation: it collects off-policy data during the roll-in phase, simulates reward-based soft-optimal policies during roll-out, and updates the model by minimizing the KL divergence between the simulated soft-optimal policy and the current model policy. Our off-policy formulation, combined with KL divergence minimization, enhances training stability and sample efficiency compared to existing RL-based methods. Empirical results demonstrate the effectiveness and superior reward optimization of our approach across diverse tasks in protein, small molecule, and regulatory DNA design. 

---
# Novel Pigeon-inspired 3D Obstacle Detection and Avoidance Maneuver for Multi-UAV Systems 

**Authors**: Reza Ahmadvand, Sarah Safura Sharif, Yaser Mike Banad  

**Link**: [PDF](https://arxiv.org/pdf/2507.00443)  

**Abstract**: Recent advances in multi-agent systems manipulation have demonstrated a rising demand for the implementation of multi-UAV systems in urban areas, which are always subjected to the presence of static and dynamic obstacles. Inspired by the collective behavior of tilapia fish and pigeons, the focus of the presented research is on the introduction of a nature-inspired collision-free formation control for a multi-UAV system, considering the obstacle avoidance maneuvers. The developed framework in this study utilizes a semi-distributed control approach, in which, based on a probabilistic Lloyd's algorithm, a centralized guidance algorithm works for optimal positioning of the UAVs, while a distributed control approach has been used for the intervehicle collision and obstacle avoidance. Further, the presented framework has been extended to the 3D space with a novel definition of 3D maneuvers. Finally, the presented framework has been applied to multi-UAV systems in 2D and 3D scenarios, and the obtained results demonstrated the validity of the presented method in dynamic environments with stationary and moving obstacles. 

---
# A Recipe for Causal Graph Regression: Confounding Effects Revisited 

**Authors**: Yujia Yin, Tianyi Qu, Zihao Wang, Yifan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.00440)  

**Abstract**: Through recognizing causal subgraphs, causal graph learning (CGL) has risen to be a promising approach for improving the generalizability of graph neural networks under out-of-distribution (OOD) scenarios. However, the empirical successes of CGL techniques are mostly exemplified in classification settings, while regression tasks, a more challenging setting in graph learning, are overlooked. We thus devote this work to tackling causal graph regression (CGR); to this end we reshape the processing of confounding effects in existing CGL studies, which mainly deal with classification. Specifically, we reflect on the predictive power of confounders in graph-level regression, and generalize classification-specific causal intervention techniques to regression through a lens of contrastive learning. Extensive experiments on graph OOD benchmarks validate the efficacy of our proposals for CGR. The model implementation and the code are provided on this https URL. 

---
# RoboEval: Where Robotic Manipulation Meets Structured and Scalable Evaluation 

**Authors**: Yi Ru Wang, Carter Ung, Grant Tannert, Jiafei Duan, Josephine Li, Amy Le, Rishabh Oswal, Markus Grotz, Wilbert Pumacay, Yuquan Deng, Ranjay Krishna, Dieter Fox, Siddhartha Srinivasa  

**Link**: [PDF](https://arxiv.org/pdf/2507.00435)  

**Abstract**: We present RoboEval, a simulation benchmark and structured evaluation framework designed to reveal the limitations of current bimanual manipulation policies. While prior benchmarks report only binary task success, we show that such metrics often conceal critical weaknesses in policy behavior -- such as poor coordination, slipping during grasping, or asymmetric arm usage. RoboEval introduces a suite of tiered, semantically grounded tasks decomposed into skill-specific stages, with variations that systematically challenge spatial, physical, and coordination capabilities. Tasks are paired with fine-grained diagnostic metrics and 3000+ human demonstrations to support imitation learning. Our experiments reveal that policies with similar success rates diverge in how tasks are executed -- some struggle with alignment, others with temporally consistent bimanual control. We find that behavioral metrics correlate with success in over half of task-metric pairs, and remain informative even when binary success saturates. By pinpointing when and how policies fail, RoboEval enables a deeper, more actionable understanding of robotic manipulation -- and highlights the need for evaluation tools that go beyond success alone. 

---
# Geological Everything Model 3D: A Promptable Foundation Model for Unified and Zero-Shot Subsurface Understanding 

**Authors**: Yimin Dou, Xinming Wu, Nathan L Bangs, Harpreet Singh Sethi, Jintao Li, Hang Gao, Zhixiang Guo  

**Link**: [PDF](https://arxiv.org/pdf/2507.00419)  

**Abstract**: Understanding Earth's subsurface is critical for energy transition, natural hazard mitigation, and planetary science. Yet subsurface analysis remains fragmented, with separate models required for structural interpretation, stratigraphic analysis, geobody segmentation, and property modeling-each tightly coupled to specific data distributions and task formulations. We introduce the Geological Everything Model 3D (GEM), a unified generative architecture that reformulates all these tasks as prompt-conditioned inference along latent structural frameworks derived from subsurface imaging. This formulation moves beyond task-specific models by enabling a shared inference mechanism, where GEM propagates human-provided prompts-such as well logs, masks, or structural sketches-along inferred structural frameworks to produce geologically coherent outputs. Through this mechanism, GEM achieves zero-shot generalization across tasks with heterogeneous prompt types, without retraining for new tasks or data sources. This capability emerges from a two-stage training process that combines self-supervised representation learning on large-scale field seismic data with adversarial fine-tuning using mixed prompts and labels across diverse subsurface tasks. GEM demonstrates broad applicability across surveys and tasks, including Martian radar stratigraphy analysis, structural interpretation in subduction zones, full seismic stratigraphic interpretation, geobody delineation, and property modeling. By bridging expert knowledge with generative reasoning in a structurally aware manner, GEM lays the foundation for scalable, human-in-the-loop geophysical AI-transitioning from fragmented pipelines to a vertically integrated, promptable reasoning system. Project page: this https URL 

---
# Serving LLMs in HPC Clusters: A Comparative Study of Qualcomm Cloud AI 100 Ultra and High-Performance GPUs 

**Authors**: Mohammad Firas Sada, John J. Graham, Elham E Khoda, Mahidhar Tatineni, Dmitry Mishin, Rajesh K. Gupta, Rick Wagner, Larry Smarr, Thomas A. DeFanti, Frank Würthwein  

**Link**: [PDF](https://arxiv.org/pdf/2507.00418)  

**Abstract**: This study presents a benchmarking analysis of the Qualcomm Cloud AI 100 Ultra (QAic) accelerator for large language model (LLM) inference, evaluating its energy efficiency (throughput per watt) and performance against leading NVIDIA (A100, H200) and AMD (MI300A) GPUs within the National Research Platform (NRP) ecosystem. A total of 15 open-source LLMs, ranging from 117 million to 90 billion parameters, are served using the vLLM framework. The QAic inference cards appears to be energy efficient and performs well in the energy efficiency metric in most cases. The findings offer insights into the potential of the Qualcomm Cloud AI 100 Ultra for high-performance computing (HPC) applications within the National Research Platform (NRP). 

---
# Augmenting Molecular Graphs with Geometries via Machine Learning Interatomic Potentials 

**Authors**: Cong Fu, Yuchao Lin, Zachary Krueger, Haiyang Yu, Maho Nakata, Jianwen Xie, Emine Kucukbenli, Xiaofeng Qian, Shuiwang Ji  

**Link**: [PDF](https://arxiv.org/pdf/2507.00407)  

**Abstract**: Accurate molecular property predictions require 3D geometries, which are typically obtained using expensive methods such as density functional theory (DFT). Here, we attempt to obtain molecular geometries by relying solely on machine learning interatomic potential (MLIP) models. To this end, we first curate a large-scale molecular relaxation dataset comprising 3.5 million molecules and 300 million snapshots. Then MLIP foundation models are trained with supervised learning to predict energy and forces given 3D molecular structures. Once trained, we show that the foundation models can be used in different ways to obtain geometries either explicitly or implicitly. First, it can be used to obtain low-energy 3D geometries via geometry optimization, providing relaxed 3D geometries for downstream molecular property predictions. To mitigate potential biases and enhance downstream predictions, we introduce geometry fine-tuning based on the relaxed 3D geometries. Second, the foundation models can be directly fine-tuned for property prediction when ground truth 3D geometries are available. Our results demonstrate that MLIP foundation models trained on relaxation data can provide valuable molecular geometries that benefit property predictions. 

---
# iPanda: An Intelligent Protocol Testing and Debugging Agent for Conformance Testing 

**Authors**: Xikai Sun, Fan Dang, Kebin Liu, Xin Miao, Zihao Yang, Haimo Lu, Yawen Zheng, Yunhao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.00378)  

**Abstract**: Conformance testing is essential for ensuring that protocol implementations comply with their specifications. However, traditional testing approaches involve manually creating numerous test cases and scripts, making the process labor-intensive and inefficient. Recently, Large Language Models (LLMs) have demonstrated impressive text comprehension and code generation abilities, providing promising opportunities for automation. In this paper, we propose iPanda, the first end-to-end framework that leverages LLMs to automate protocol conformance testing. Given a protocol specification document and its implementation, iPanda first employs a keyword-based method to automatically generate comprehensive test cases. Then, it utilizes a code-based retrieval-augmented generation approach to effectively interpret the implementation and produce executable test code. To further enhance code quality, iPanda incorporates an iterative self-correction mechanism to refine generated test scripts interactively. Finally, by executing and analyzing the generated tests, iPanda systematically verifies compliance between implementations and protocol specifications. Comprehensive experiments on various protocols show that iPanda significantly outperforms pure LLM-based approaches, improving the success rate (Pass@1) of test-code generation by factors ranging from 4.675 times to 10.751 times. 

---
# Data-Driven Exploration for a Class of Continuous-Time Linear--Quadratic Reinforcement Learning Problems 

**Authors**: Yilie Huang, Xun Yu Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2507.00358)  

**Abstract**: We study reinforcement learning (RL) for the same class of continuous-time stochastic linear--quadratic (LQ) control problems as in \cite{huang2024sublinear}, where volatilities depend on both states and controls while states are scalar-valued and running control rewards are absent. We propose a model-free, data-driven exploration mechanism that adaptively adjusts entropy regularization by the critic and policy variance by the actor. Unlike the constant or deterministic exploration schedules employed in \cite{huang2024sublinear}, which require extensive tuning for implementations and ignore learning progresses during iterations, our adaptive exploratory approach boosts learning efficiency with minimal tuning. Despite its flexibility, our method achieves a sublinear regret bound that matches the best-known model-free results for this class of LQ problems, which were previously derived only with fixed exploration schedules. Numerical experiments demonstrate that adaptive explorations accelerate convergence and improve regret performance compared to the non-adaptive model-free and model-based counterparts. 

---
# CGEarthEye:A High-Resolution Remote Sensing Vision Foundation Model Based on the Jilin-1 Satellite Constellation 

**Authors**: Zhiwei Yi, Xin Cheng, Jingyu Ma, Ruifei Zhu, Junwei Tian, Yuanxiu Zhou, Xinge Zhao, Hongzhe Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.00356)  

**Abstract**: Deep learning methods have significantly advanced the development of intelligent rinterpretation in remote sensing (RS), with foundational model research based on large-scale pre-training paradigms rapidly reshaping various domains of Earth Observation (EO). However, compared to the open accessibility and high spatiotemporal coverage of medium-resolution data, the limited acquisition channels for ultra-high-resolution optical RS imagery have constrained the progress of high-resolution remote sensing vision foundation models (RSVFM). As the world's largest sub-meter-level commercial RS satellite constellation, the Jilin-1 constellation possesses abundant sub-meter-level image resources. This study proposes CGEarthEye, a RSVFM framework specifically designed for Jilin-1 satellite characteristics, comprising five backbones with different parameter scales with totaling 2.1 billion parameters. To enhance the representational capacity of the foundation model, we developed JLSSD, the first 15-million-scale multi-temporal self-supervised learning (SSL) dataset featuring global coverage with quarterly temporal sampling within a single year, constructed through multi-level representation clustering and sampling strategies. The framework integrates seasonal contrast, augmentation-based contrast, and masked patch token contrastive strategies for pre-training. Comprehensive evaluations across 10 benchmark datasets covering four typical RS tasks demonstrate that the CGEarthEye consistently achieves state-of-the-art (SOTA) performance. Further analysis reveals CGEarthEye's superior characteristics in feature visualization, model convergence, parameter efficiency, and practical mapping applications. This study anticipates that the exceptional representation capabilities of CGEarthEye will facilitate broader and more efficient applications of Jilin-1 data in traditional EO application. 

---
# An AST-guided LLM Approach for SVRF Code Synthesis 

**Authors**: Abanoub E. Abdelmalak, Mohamed A. Elsayed, David Abercrombie, Ilhami Torunoglu  

**Link**: [PDF](https://arxiv.org/pdf/2507.00352)  

**Abstract**: Standard Verification Rule Format (SVRF) is essential for semiconductor applications like Design Rule Check (DRC), Layout Versus Schematic (LVS), and Optical Proximity Correction (OPC) and it faces challenges as advancing nodes create complex design rules that renders traditional SVRF development ineffective and highlight an expertise gap. This paper introduces a novel methodology integrating Abstract Syntax Tree (AST) embedding and Retrieval-Augmented Generation (RAG) for enhanced SVRF code synthesis, ensuring semantic accuracy and error minimization through structural validation with domain-specific insights for precise code generation.
We evaluate different T5-based models and propose an innovative SVRF-specific scoring framework that complements standard metrics like BLEU and ROUGE-L. In our approach, AST provides rigorous structural validation, while RAG infuses relevant domain knowledge, effectively enhancing the code generation workflow.
Testing on a comprehensive benchmark of 740 DRC rule implementations, our methodology demonstrates up to a 40\% improvement in code generation accuracy compared to basic text-based fine-tuning process. This fusion of industry expertise with advanced coding strategies not only optimizes SVRF development under limited dataset constraints but also creates a more intuitive and efficient coding environment. Consequently, users can rapidly iterate through design cycles, reduce manual error correction, and significantly improve overall productivity. 

---
# VTS-Guided AI Interaction Workflow for Business Insights 

**Authors**: Sun Ding, Ude Enebeli, Atilhan, Manay, Ryan Pua, Kamal Kotak  

**Link**: [PDF](https://arxiv.org/pdf/2507.00347)  

**Abstract**: Modern firms face a flood of dense, unstructured reports. Turning these documents into usable insights takes heavy effort and is far from agile when quick answers are needed. VTS-AI tackles this gap. It integrates Visual Thinking Strategies, which emphasize evidence-based observation, linking, and thinking, into AI agents, so the agents can extract business insights from unstructured text, tables, and images at scale. The system works in three tiers (micro, meso, macro). It tags issues, links them to source pages, and rolls them into clear action levers stored in a searchable YAML file. In tests on an 18-page business report, VTS-AI matched the speed of a one-shot ChatGPT prompt yet produced richer findings: page locations, verbatim excerpts, severity scores, and causal links. Analysts can accept or adjust these outputs in the same IDE, keeping human judgment in the loop. Early results show VTS-AI spots the direction of key metrics and flags where deeper number-crunching is needed. Next steps include mapping narrative tags to financial ratios, adding finance-tuned language models through a Model-Context Protocol, and building a Risk & Safety Layer to stress-test models and secure data. These upgrades aim to make VTS-AI a production-ready, audit-friendly tool for rapid business analysis. 

---
# Training for X-Ray Vision: Amodal Segmentation, Amodal Content Completion, and View-Invariant Object Representation from Multi-Camera Video 

**Authors**: Alexander Moore, Amar Saini, Kylie Cancilla, Doug Poland, Carmen Carrano  

**Link**: [PDF](https://arxiv.org/pdf/2507.00339)  

**Abstract**: Amodal segmentation and amodal content completion require using object priors to estimate occluded masks and features of objects in complex scenes. Until now, no data has provided an additional dimension for object context: the possibility of multiple cameras sharing a view of a scene. We introduce MOVi-MC-AC: Multiple Object Video with Multi-Cameras and Amodal Content, the largest amodal segmentation and first amodal content dataset to date. Cluttered scenes of generic household objects are simulated in multi-camera video. MOVi-MC-AC contributes to the growing literature of object detection, tracking, and segmentation by including two new contributions to the deep learning for computer vision world. Multiple Camera (MC) settings where objects can be identified and tracked between various unique camera perspectives are rare in both synthetic and real-world video. We introduce a new complexity to synthetic video by providing consistent object ids for detections and segmentations between both frames and multiple cameras each with unique features and motion patterns on a single scene. Amodal Content (AC) is a reconstructive task in which models predict the appearance of target objects through occlusions. In the amodal segmentation literature, some datasets have been released with amodal detection, tracking, and segmentation labels. While other methods rely on slow cut-and-paste schemes to generate amodal content pseudo-labels, they do not account for natural occlusions present in the modal masks. MOVi-MC-AC provides labels for ~5.8 million object instances, setting a new maximum in the amodal dataset literature, along with being the first to provide ground-truth amodal content. The full dataset is available at this https URL , 

---
# Failure by Interference: Language Models Make Balanced Parentheses Errors When Faulty Mechanisms Overshadow Sound Ones 

**Authors**: Daking Rai, Samuel Miller, Kevin Moran, Ziyu Yao  

**Link**: [PDF](https://arxiv.org/pdf/2507.00322)  

**Abstract**: Despite remarkable advances in coding capabilities, language models (LMs) still struggle with simple syntactic tasks such as generating balanced parentheses. In this study, we investigate the underlying mechanisms behind the persistence of these errors across LMs of varying sizes (124M-7B) to both understand and mitigate the errors. Our study reveals that LMs rely on a number of components (attention heads and FF neurons) that independently make their own predictions. While some components reliably promote correct answers across a generalized range of inputs (i.e., implementing "sound mechanisms''), others are less reliable and introduce noise by promoting incorrect tokens (i.e., implementing "faulty mechanisms''). Errors occur when the faulty mechanisms overshadow the sound ones and dominantly affect the predictions. Motivated by this insight, we introduce RASteer, a steering method to systematically identify and increase the contribution of reliable components for improving model performance. RASteer substantially improves performance on balanced parentheses tasks, boosting accuracy of some models from $0$% to around $100$% without impairing the models' general coding ability. We further demonstrate its broader applicability in arithmetic reasoning tasks, achieving performance gains of up to around $20$%. 

---
# Open-ended Scientific Discovery via Bayesian Surprise 

**Authors**: Dhruv Agarwal, Bodhisattwa Prasad Majumder, Reece Adamson, Megha Chakravorty, Satvika Reddy Gavireddy, Aditya Parashar, Harshit Surana, Bhavana Dalvi Mishra, Andrew McCallum, Ashish Sabharwal, Peter Clark  

**Link**: [PDF](https://arxiv.org/pdf/2507.00310)  

**Abstract**: The promise of autonomous scientific discovery (ASD) hinges not only on answering questions, but also on knowing which questions to ask. Most recent works in ASD explore the use of large language models (LLMs) in goal-driven settings, relying on human-specified research questions to guide hypothesis generation. However, scientific discovery may be accelerated further by allowing the AI system to drive exploration by its own criteria. The few existing approaches in open-ended ASD select hypotheses based on diversity heuristics or subjective proxies for human interestingness, but the former struggles to meaningfully navigate the typically vast hypothesis space, and the latter suffers from imprecise definitions. This paper presents AutoDS -- a method for open-ended ASD that instead drives scientific exploration using Bayesian surprise. Here, we quantify the epistemic shift from the LLM's prior beliefs about a hypothesis to its posterior beliefs after gathering experimental results. To efficiently explore the space of nested hypotheses, our method employs a Monte Carlo tree search (MCTS) strategy with progressive widening using surprisal as the reward function. We evaluate AutoDS in the setting of data-driven discovery across 21 real-world datasets spanning domains such as biology, economics, finance, and behavioral science. Our results demonstrate that under a fixed budget, AutoDS substantially outperforms competitors by producing 5--29\% more discoveries deemed surprising by the LLM. Our human evaluation further finds that two-thirds of AutoDS discoveries are surprising to the domain experts, suggesting this is an important step forward towards building open-ended ASD systems. 

---
# Natural language processing for African languages 

**Authors**: David Ifeoluwa Adelani  

**Link**: [PDF](https://arxiv.org/pdf/2507.00297)  

**Abstract**: Recent advances in word embeddings and language models use large-scale, unlabelled data and self-supervised learning to boost NLP performance. Multilingual models, often trained on web-sourced data like Wikipedia, face challenges: few low-resource languages are included, their data is often noisy, and lack of labeled datasets makes it hard to evaluate performance outside high-resource languages like English. In this dissertation, we focus on languages spoken in Sub-Saharan Africa where all the indigenous languages in this region can be regarded as low-resourced in terms of the availability of labelled data for NLP tasks and unlabelled data found on the web. We analyse the noise in the publicly available corpora, and curate a high-quality corpus, demonstrating that the quality of semantic representations learned in word embeddings does not only depend on the amount of data but on the quality of pre-training data. We demonstrate empirically the limitations of word embeddings, and the opportunities the multilingual pre-trained language model (PLM) offers especially for languages unseen during pre-training and low-resource scenarios. We further study how to adapt and specialize multilingual PLMs to unseen African languages using a small amount of monolingual texts. To address the under-representation of the African languages in NLP research, we developed large scale human-annotated labelled datasets for 21 African languages in two impactful NLP tasks: named entity recognition and machine translation. We conduct an extensive empirical evaluation using state-of-the-art methods across supervised, weakly-supervised, and transfer learning settings. 

---
# Reducing Variability of Multiple Instance Learning Methods for Digital Pathology 

**Authors**: Ali Mammadov, Loïc Le Folgoc, Guillaume Hocquet, Pietro Gori  

**Link**: [PDF](https://arxiv.org/pdf/2507.00292)  

**Abstract**: Digital pathology has revolutionized the field by enabling the digitization of tissue samples into whole slide images (WSIs). However, the high resolution and large size of WSIs present significant challenges when it comes to applying Deep Learning models. As a solution, WSIs are often divided into smaller patches with a global label (\textit{i.e., diagnostic}) per slide, instead of a (too) costly pixel-wise annotation. By treating each slide as a bag of patches, Multiple Instance Learning (MIL) methods have emerged as a suitable solution for WSI classification. A major drawback of MIL methods is their high variability in performance across different runs, which can reach up to 10-15 AUC points on the test set, making it difficult to compare different MIL methods reliably. This variability mainly comes from three factors: i) weight initialization, ii) batch (shuffling) ordering, iii) and learning rate. To address that, we introduce a Multi-Fidelity, Model Fusion strategy for MIL methods. We first train multiple models for a few epochs and average the most stable and promising ones based on validation scores. This approach can be applied to any existing MIL model to reduce performance variability. It also simplifies hyperparameter tuning and improves reproducibility while maintaining computational efficiency. We extensively validate our approach on WSI classification tasks using 2 different datasets, 3 initialization strategies and 5 MIL methods, for a total of more than 2000 experiments. 

---
# Reconfiguring Digital Accountability: AI-Powered Innovations and Transnational Governance in a Postnational Accounting Context 

**Authors**: Claire Li, David Freeborn  

**Link**: [PDF](https://arxiv.org/pdf/2507.00288)  

**Abstract**: This study explores how AI-powered digital innovations are reshaping organisational accountability in a transnational governance context. As AI systems increasingly mediate decision-making in domains such as auditing and financial reporting, traditional mechanisms of accountability, based on control, transparency, and auditability, are being destabilised. We integrate the Technology Acceptance Model (TAM), Actor-Network Theory (ANT), and institutional theory to examine how organisations adopt AI technologies in response to regulatory, ethical, and cultural pressures that transcend national boundaries. We argue that accountability is co-constructed within global socio-technical networks, shaped not only by user perceptions but also by governance logics and normative expectations. Extending TAM, we incorporate compliance and legitimacy as key factors in perceived usefulness and usability. Drawing on ANT, we reconceptualise accountability as a relational and emergent property of networked assemblages. We propose two organisational strategies including internal governance reconfiguration and external actor-network engagement to foster responsible, legitimate, and globally accepted AI adoption in the accounting domain. 

---
# Self-Supervised Multiview Xray Matching 

**Authors**: Mohamad Dabboussi, Malo Huard, Yann Gousseau, Pietro Gori  

**Link**: [PDF](https://arxiv.org/pdf/2507.00287)  

**Abstract**: Accurate interpretation of multi-view radiographs is crucial for diagnosing fractures, muscular injuries, and other anomalies. While significant advances have been made in AI-based analysis of single images, current methods often struggle to establish robust correspondences between different X-ray views, an essential capability for precise clinical evaluations. In this work, we present a novel self-supervised pipeline that eliminates the need for manual annotation by automatically generating a many-to-many correspondence matrix between synthetic X-ray views. This is achieved using digitally reconstructed radiographs (DRR), which are automatically derived from unannotated CT volumes. Our approach incorporates a transformer-based training phase to accurately predict correspondences across two or more X-ray views. Furthermore, we demonstrate that learning correspondences among synthetic X-ray views can be leveraged as a pretraining strategy to enhance automatic multi-view fracture detection on real data. Extensive evaluations on both synthetic and real X-ray datasets show that incorporating correspondences improves performance in multi-view fracture classification. 

---
# Visual Privacy Management with Generative AI for Blind and Low-Vision People 

**Authors**: Tanusree Sharma, Yu-Yun Tseng, Lotus Zhang, Ayae Ide, Kelly Avery Mack, Leah Findlater, Danna Gurari, Yang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.00286)  

**Abstract**: Blind and low vision (BLV) individuals use Generative AI (GenAI) tools to interpret and manage visual content in their daily lives. While such tools can enhance the accessibility of visual content and so enable greater user independence, they also introduce complex challenges around visual privacy. In this paper, we investigate the current practices and future design preferences of blind and low vision individuals through an interview study with 21 participants. Our findings reveal a range of current practices with GenAI that balance privacy, efficiency, and emotional agency, with users accounting for privacy risks across six key scenarios, such as self-presentation, indoor/outdoor spatial privacy, social sharing, and handling professional content. Our findings reveal design preferences, including on-device processing, zero-retention guarantees, sensitive content redaction, privacy-aware appearance indicators, and multimodal tactile mirrored interaction methods. We conclude with actionable design recommendations to support user-centered visual privacy through GenAI, expanding the notion of privacy and responsible handling of others data. 

---
# Double Q-learning for Value-based Deep Reinforcement Learning, Revisited 

**Authors**: Prabhat Nagarajan, Martha White, Marlos C. Machado  

**Link**: [PDF](https://arxiv.org/pdf/2507.00275)  

**Abstract**: Overestimation is pervasive in reinforcement learning (RL), including in Q-learning, which forms the algorithmic basis for many value-based deep RL algorithms. Double Q-learning is an algorithm introduced to address Q-learning's overestimation by training two Q-functions and using both to de-correlate action-selection and action-evaluation in bootstrap targets. Shortly after Q-learning was adapted to deep RL in the form of deep Q-networks (DQN), Double Q-learning was adapted to deep RL in the form of Double DQN. However, Double DQN only loosely adapts Double Q-learning, forgoing the training of two different Q-functions that bootstrap off one another. In this paper, we study algorithms that adapt this core idea of Double Q-learning for value-based deep RL. We term such algorithms Deep Double Q-learning (DDQL). Our aim is to understand whether DDQL exhibits less overestimation than Double DQN and whether performant instantiations of DDQL exist. We answer both questions affirmatively, demonstrating that DDQL reduces overestimation and outperforms Double DQN in aggregate across 57 Atari 2600 games, without requiring additional hyperparameters. We also study several aspects of DDQL, including its network architecture, replay ratio, and minibatch sampling strategy. 

---
# Feature Integration Spaces: Joint Training Reveals Dual Encoding in Neural Network Representations 

**Authors**: Omar Claflin  

**Link**: [PDF](https://arxiv.org/pdf/2507.00269)  

**Abstract**: Current sparse autoencoder (SAE) approaches to neural network interpretability assume that activations can be decomposed through linear superposition into sparse, interpretable features. Despite high reconstruction fidelity, SAEs consistently fail to eliminate polysemanticity and exhibit pathological behavioral errors. We propose that neural networks encode information in two complementary spaces compressed into the same substrate: feature identity and feature integration. To test this dual encoding hypothesis, we develop sequential and joint-training architectures to capture identity and integration patterns simultaneously. Joint training achieves 41.3% reconstruction improvement and 51.6% reduction in KL divergence errors. This architecture spontaneously develops bimodal feature organization: low squared norm features contributing to integration pathways and the rest contributing directly to the residual. Small nonlinear components (3% of parameters) achieve 16.5% standalone improvements, demonstrating parameter-efficient capture of computational relationships crucial for behavior. Additionally, intervention experiments using 2x2 factorial stimulus designs demonstrated that integration features exhibit selective sensitivity to experimental manipulations and produce systematic behavioral effects on model outputs, including significant interaction effects across semantic dimensions. This work provides systematic evidence for (1) dual encoding in neural representations, (2) meaningful nonlinearly encoded feature interactions, and (3) introduces an architectural paradigm shift from post-hoc feature analysis to integrated computational design, establishing foundations for next-generation SAEs. 

---
# Control-Optimized Deep Reinforcement Learning for Artificially Intelligent Autonomous Systems 

**Authors**: Oren Fivel, Matan Rudman, Kobi Cohen  

**Link**: [PDF](https://arxiv.org/pdf/2507.00268)  

**Abstract**: Deep reinforcement learning (DRL) has become a powerful tool for complex decision-making in machine learning and AI. However, traditional methods often assume perfect action execution, overlooking the uncertainties and deviations between an agent's selected actions and the actual system response. In real-world applications, such as robotics, mechatronics, and communication networks, execution mismatches arising from system dynamics, hardware constraints, and latency can significantly degrade performance. This work advances AI by developing a novel control-optimized DRL framework that explicitly models and compensates for action execution mismatches, a challenge largely overlooked in existing methods. Our approach establishes a structured two-stage process: determining the desired action and selecting the appropriate control signal to ensure proper execution. It trains the agent while accounting for action mismatches and controller corrections. By incorporating these factors into the training process, the AI agent optimizes the desired action with respect to both the actual control signal and the intended outcome, explicitly considering execution errors. This approach enhances robustness, ensuring that decision-making remains effective under real-world uncertainties. Our approach offers a substantial advancement for engineering practice by bridging the gap between idealized learning and real-world implementation. It equips intelligent agents operating in engineering environments with the ability to anticipate and adjust for actuation errors and system disturbances during training. We evaluate the framework in five widely used open-source mechanical simulation environments we restructured and developed to reflect real-world operating conditions, showcasing its robustness against uncertainties and offering a highly practical and efficient solution for control-oriented applications. 

---
# Impact of Fine-Tuning Methods on Memorization in Large Language Models 

**Authors**: Jie Hou, Chuxiong Wu, Lannan Luo, Qiang Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2507.00258)  

**Abstract**: As the capabilities of pre-trained large language models (LLMs) continue to advance, the "pre-train and fine-tune" paradigm has become increasingly mainstream, leading to the development of various fine-tuning methods. However, the privacy risks arising from memorization during fine-tuning have received relatively little attention. To address this gap, we categorize popular fine-tuning approaches and assess their impact on memorization through the lens of membership inference attacks (MIAs). Our results show that, compared to parameter-based fine-tuning, prompt-based fine-tuning achieves competitive performance while exhibiting lower vulnerability to MIAs. Furthermore, prompt-based methods maintain low memorization regardless of model scale. These findings suggest that parameter-based fine-tuning is more prone to leaking private information, whereas prompt-based fine-tuning serves as a more privacy-preserving option. 

---
# Gym4ReaL: A Suite for Benchmarking Real-World Reinforcement Learning 

**Authors**: Davide Salaorni, Vincenzo De Paola, Samuele Delpero, Giovanni Dispoto, Paolo Bonetti, Alessio Russo, Giuseppe Calcagno, Francesco Trovò, Matteo Papini, Alberto Maria Metelli, Marco Mussi, Marcello Restelli  

**Link**: [PDF](https://arxiv.org/pdf/2507.00257)  

**Abstract**: In recent years, \emph{Reinforcement Learning} (RL) has made remarkable progress, achieving superhuman performance in a wide range of simulated environments. As research moves toward deploying RL in real-world applications, the field faces a new set of challenges inherent to real-world settings, such as large state-action spaces, non-stationarity, and partial observability. Despite their importance, these challenges are often underexplored in current benchmarks, which tend to focus on idealized, fully observable, and stationary environments, often neglecting to incorporate real-world complexities explicitly. In this paper, we introduce \texttt{Gym4ReaL}, a comprehensive suite of realistic environments designed to support the development and evaluation of RL algorithms that can operate in real-world scenarios. The suite includes a diverse set of tasks that expose algorithms to a variety of practical challenges. Our experimental results show that, in these settings, standard RL algorithms confirm their competitiveness against rule-based benchmarks, motivating the development of new methods to fully exploit the potential of RL to tackle the complexities of real-world tasks. 

---
# Developing Lightweight DNN Models With Limited Data For Real-Time Sign Language Recognition 

**Authors**: Nikita Nikitin, Eugene Fomin  

**Link**: [PDF](https://arxiv.org/pdf/2507.00248)  

**Abstract**: We present a novel framework for real-time sign language recognition using lightweight DNNs trained on limited data. Our system addresses key challenges in sign language recognition, including data scarcity, high computational costs, and discrepancies in frame rates between training and inference environments. By encoding sign language specific parameters, such as handshape, palm orientation, movement, and location into vectorized inputs, and leveraging MediaPipe for landmark extraction, we achieve highly separable input data representations. Our DNN architecture, optimized for sub 10MB deployment, enables accurate classification of 343 signs with less than 10ms latency on edge devices. The data annotation platform 'slait data' facilitates structured labeling and vector extraction. Our model achieved 92% accuracy in isolated sign recognition and has been integrated into the 'slait ai' web application, where it demonstrates stable inference. 

---
# Linearly Decoding Refused Knowledge in Aligned Language Models 

**Authors**: Aryan Shrivastava, Ari Holtzman  

**Link**: [PDF](https://arxiv.org/pdf/2507.00239)  

**Abstract**: Most commonly used language models (LMs) are instruction-tuned and aligned using a combination of fine-tuning and reinforcement learning, causing them to refuse users requests deemed harmful by the model. However, jailbreak prompts can often bypass these refusal mechanisms and elicit harmful responses. In this work, we study the extent to which information accessed via jailbreak prompts is decodable using linear probes trained on LM hidden states. We show that a great deal of initially refused information is linearly decodable. For example, across models, the response of a jailbroken LM for the average IQ of a country can be predicted by a linear probe with Pearson correlations exceeding $0.8$. Surprisingly, we find that probes trained on base models (which do not refuse) sometimes transfer to their instruction-tuned versions and are capable of revealing information that jailbreaks decode generatively, suggesting that the internal representations of many refused properties persist from base LMs through instruction-tuning. Importantly, we show that this information is not merely "leftover" in instruction-tuned models, but is actively used by them: we find that probe-predicted values correlate with LM generated pairwise comparisons, indicating that the information decoded by our probes align with suppressed generative behavior that may be expressed more subtly in other downstream tasks. Overall, our results suggest that instruction-tuning does not wholly eliminate or even relocate harmful information in representation space-they merely suppress its direct expression, leaving it both linearly accessible and indirectly influential in downstream behavior. 

---
# Interpretable AI for Time-Series: Multi-Model Heatmap Fusion with Global Attention and NLP-Generated Explanations 

**Authors**: Jiztom Kavalakkatt Francis, Matthew J Darr  

**Link**: [PDF](https://arxiv.org/pdf/2507.00234)  

**Abstract**: In this paper, we present a novel framework for enhancing model interpretability by integrating heatmaps produced separately by ResNet and a restructured 2D Transformer with globally weighted input saliency. We address the critical problem of spatial-temporal misalignment in existing interpretability methods, where convolutional networks fail to capture global context and Transformers lack localized precision - a limitation that impedes actionable insights in safety-critical domains like healthcare and industrial monitoring. Our method merges gradient-weighted activation maps (ResNet) and Transformer attention rollout into a unified visualization, achieving full spatial-temporal alignment while preserving real-time performance. Empirical evaluations on clinical (ECG arrhythmia detection) and industrial (energy consumption prediction) datasets demonstrate significant improvements: the hybrid framework achieves 94.1% accuracy (F1 0.93) on the PhysioNet dataset and reduces regression error to RMSE = 0.28 kWh (R2 = 0.95) on the UCI Energy Appliance dataset-outperforming standalone ResNet, Transformer, and InceptionTime baselines by 3.8-12.4%. An NLP module translates fused heatmaps into domain-specific narratives (e.g., "Elevated ST-segment between 2-4 seconds suggests myocardial ischemia"), validated via BLEU-4 (0.586) and ROUGE-L (0.650) scores. By formalizing interpretability as causal fidelity and spatial-temporal alignment, our approach bridges the gap between technical outputs and stakeholder understanding, offering a scalable solution for transparent, time-aware decision-making. 

---
# A High-Fidelity Speech Super Resolution Network using a Complex Global Attention Module with Spectro-Temporal Loss 

**Authors**: Tarikul Islam Tamiti, Biraj Joshi, Rida Hasan, Rashedul Hasan, Taieba Athay, Nursad Mamun, Anomadarshi Barua  

**Link**: [PDF](https://arxiv.org/pdf/2507.00229)  

**Abstract**: Speech super-resolution (SSR) enhances low-resolution speech by increasing the sampling rate. While most SSR methods focus on magnitude reconstruction, recent research highlights the importance of phase reconstruction for improved perceptual quality. Therefore, we introduce CTFT-Net, a Complex Time-Frequency Transformation Network that reconstructs both magnitude and phase in complex domains for improved SSR tasks. It incorporates a complex global attention block to model inter-phoneme and inter-frequency dependencies and a complex conformer to capture long-range and local features, improving frequency reconstruction and noise robustness. CTFT-Net employs time-domain and multi-resolution frequency-domain loss functions for better generalization. Experiments show CTFT-Net outperforms state-of-the-art models (NU-Wave, WSRGlow, NVSR, AERO) on the VCTK dataset, particularly for extreme upsampling (2 kHz to 48 kHz), reconstructing high frequencies effectively without noisy artifacts. 

---
# Investigating Stochastic Methods for Prosody Modeling in Speech Synthesis 

**Authors**: Paul Mayer, Florian Lux, Alejandro Pérez-González-de-Martos, Angelina Elizarova, Lindsey Vanderlyn, Dirk Väth, Ngoc Thang Vu  

**Link**: [PDF](https://arxiv.org/pdf/2507.00227)  

**Abstract**: While generative methods have progressed rapidly in recent years, generating expressive prosody for an utterance remains a challenging task in text-to-speech synthesis. This is particularly true for systems that model prosody explicitly through parameters such as pitch, energy, and duration, which is commonly done for the sake of interpretability and controllability. In this work, we investigate the effectiveness of stochastic methods for this task, including Normalizing Flows, Conditional Flow Matching, and Rectified Flows. We compare these methods to a traditional deterministic baseline, as well as to real human realizations. Our extensive subjective and objective evaluations demonstrate that stochastic methods produce natural prosody on par with human speakers by capturing the variability inherent in human speech. Further, they open up additional controllability options by allowing the sampling temperature to be tuned. 

---
# Discovering the underlying analytic structure within Standard Model constants using artificial intelligence 

**Authors**: S. V. Chekanov, H. Kjellerstrand  

**Link**: [PDF](https://arxiv.org/pdf/2507.00225)  

**Abstract**: This paper presents a search for underlying analytic structures among the fundamental parameters of the Standard Model (SM) using symbolic regression and genetic programming. We identify the simplest analytic relationships connecting pairs of these constants and report several notable observations based on about a thousand expressions with relative precision better than 1%. These results may serve as valuable inputs for model builders and artificial intelligence methods aimed at uncovering hidden patterns among the SM constants, or potentially used as building blocks for a deeper underlying law that connects all parameters of the SM through a small set of fundamental constants. 

---
# Two-Stage Reasoning-Infused Learning: Improving Classification with LLM-Generated Reasoning 

**Authors**: Mads Henrichsen, Rasmus Krebs  

**Link**: [PDF](https://arxiv.org/pdf/2507.00214)  

**Abstract**: Standard classification models often map inputs directly to labels without explicit reasoning, potentially limiting their performance, robustness, and interpretability. This paper introduces a novel two-stage approach to enhance text classification by leveraging Large Language Model (LLM)-generated reasonings. In the first stage, we fine-tune a Llama-3.2-1B-Instruct model (henceforth Llama-R-Gen) on a general-purpose reasoning dataset (syvai/reasoning-gen) to generate textual reasoning (R) given a question and its answer. In the second stage, this generally trained Llama-R-Gen is used offline to create an augmented training dataset for a downstream generative model. This downstream model, based on Llama-3.2-1B-Instruct, takes only the input text (Q) and is trained to output the generated reasoning (R) immediately followed by the predicted emotion (A). We demonstrate this methodology on the dair-ai/emotion dataset for emotion classification. Our experiments show that the generative model trained to output reasoning and the emotion (Classifier Q->RA) achieves a significant improvement of 8.7 percentage points in accuracy (for emotion prediction) compared to a baseline generative model trained solely to output the emotion (Classifier Q->A), highlighting the strong generalization capabilities of the reasoning generation and the benefit of explicit reasoning training. This work underscores the potential of LLM-generated reasonings for creating richer training datasets, thereby improving the performance of diverse downstream NLP tasks and providing explicit explanations. 

---
# SurgiSR4K: A High-Resolution Endoscopic Video Dataset for Robotic-Assisted Minimally Invasive Procedures 

**Authors**: Fengyi Jiang, Xiaorui Zhang, Lingbo Jin, Ruixing Liang, Yuxin Chen, Adi Chola Venkatesh, Jason Culman, Tiantian Wu, Lirong Shao, Wenqing Sun, Cong Gao, Hallie McNamara, Jingpei Lu, Omid Mohareri  

**Link**: [PDF](https://arxiv.org/pdf/2507.00209)  

**Abstract**: High-resolution imaging is crucial for enhancing visual clarity and enabling precise computer-assisted guidance in minimally invasive surgery (MIS). Despite the increasing adoption of 4K endoscopic systems, there remains a significant gap in publicly available native 4K datasets tailored specifically for robotic-assisted MIS. We introduce SurgiSR4K, the first publicly accessible surgical imaging and video dataset captured at a native 4K resolution, representing realistic conditions of robotic-assisted procedures. SurgiSR4K comprises diverse visual scenarios including specular reflections, tool occlusions, bleeding, and soft tissue deformations, meticulously designed to reflect common challenges faced during laparoscopic and robotic surgeries. This dataset opens up possibilities for a broad range of computer vision tasks that might benefit from high resolution data, such as super resolution (SR), smoke removal, surgical instrument detection, 3D tissue reconstruction, monocular depth estimation, instance segmentation, novel view synthesis, and vision-language model (VLM) development. SurgiSR4K provides a robust foundation for advancing research in high-resolution surgical imaging and fosters the development of intelligent imaging technologies aimed at enhancing performance, safety, and usability in image-guided robotic surgeries. 

---
# What Makes Local Updates Effective: The Role of Data Heterogeneity and Smoothness 

**Authors**: Kumar Kshitij Patel  

**Link**: [PDF](https://arxiv.org/pdf/2507.00195)  

**Abstract**: This thesis contributes to the theoretical understanding of local update algorithms, especially Local SGD, in distributed and federated optimization under realistic models of data heterogeneity. A central focus is on the bounded second-order heterogeneity assumption, which is shown to be both necessary and sufficient for local updates to outperform centralized or mini-batch methods in convex and non-convex settings. The thesis establishes tight upper and lower bounds in several regimes for various local update algorithms and characterizes the min-max complexity of multiple problem classes. At its core is a fine-grained consensus-error-based analysis framework that yields sharper finite-time convergence bounds under third-order smoothness and relaxed heterogeneity assumptions. The thesis also extends to online federated learning, providing fundamental regret bounds under both first-order and bandit feedback. Together, these results clarify when and why local updates offer provable advantages, and the thesis serves as a self-contained guide for analyzing Local SGD in heterogeneous environments. 

---
# Beyond Sensor Data: Foundation Models of Behavioral Data from Wearables Improve Health Predictions 

**Authors**: Eray Erturk, Fahad Kamran, Salar Abbaspourazad, Sean Jewell, Harsh Sharma, Yujie Li, Sinead Williamson, Nicholas J Foti, Joseph Futoma  

**Link**: [PDF](https://arxiv.org/pdf/2507.00191)  

**Abstract**: Wearable devices record physiological and behavioral signals that can improve health predictions. While foundation models are increasingly used for such predictions, they have been primarily applied to low-level sensor data, despite behavioral data often being more informative due to their alignment with physiologically relevant timescales and quantities. We develop foundation models of such behavioral signals using over 2.5B hours of wearable data from 162K individuals, systematically optimizing architectures and tokenization strategies for this unique dataset. Evaluated on 57 health-related tasks, our model shows strong performance across diverse real-world applications including individual-level classification and time-varying health state prediction. The model excels in behavior-driven tasks like sleep prediction, and improves further when combined with representations of raw sensor data. These results underscore the importance of tailoring foundation model design to wearables and demonstrate the potential to enable new health applications. 

---
# Multimodal, Multi-Disease Medical Imaging Foundation Model (MerMED-FM) 

**Authors**: Yang Zhou, Chrystie Wan Ning Quek, Jun Zhou, Yan Wang, Yang Bai, Yuhe Ke, Jie Yao, Laura Gutierrez, Zhen Ling Teo, Darren Shu Jeng Ting, Brian T. Soetikno, Christopher S. Nielsen, Tobias Elze, Zengxiang Li, Linh Le Dinh, Lionel Tim-Ee Cheng, Tran Nguyen Tuan Anh, Chee Leong Cheng, Tien Yin Wong, Nan Liu, Iain Beehuat Tan, Tony Kiat Hon Lim, Rick Siow Mong Goh, Yong Liu, Daniel Shu Wei Ting  

**Link**: [PDF](https://arxiv.org/pdf/2507.00185)  

**Abstract**: Current artificial intelligence models for medical imaging are predominantly single modality and single disease. Attempts to create multimodal and multi-disease models have resulted in inconsistent clinical accuracy. Furthermore, training these models typically requires large, labour-intensive, well-labelled datasets. We developed MerMED-FM, a state-of-the-art multimodal, multi-specialty foundation model trained using self-supervised learning and a memory module. MerMED-FM was trained on 3.3 million medical images from over ten specialties and seven modalities, including computed tomography (CT), chest X-rays (CXR), ultrasound (US), pathology patches, color fundus photography (CFP), optical coherence tomography (OCT) and dermatology images. MerMED-FM was evaluated across multiple diseases and compared against existing foundational models. Strong performance was achieved across all modalities, with AUROCs of 0.988 (OCT); 0.982 (pathology); 0.951 (US); 0.943 (CT); 0.931 (skin); 0.894 (CFP); 0.858 (CXR). MerMED-FM has the potential to be a highly adaptable, versatile, cross-specialty foundation model that enables robust medical imaging interpretation across diverse medical disciplines. 

---
# Text-to-Level Diffusion Models With Various Text Encoders for Super Mario Bros 

**Authors**: Jacob Schrum, Olivia Kilday, Emilio Salas, Bess Hagan, Reid Williams  

**Link**: [PDF](https://arxiv.org/pdf/2507.00184)  

**Abstract**: Recent research shows how diffusion models can unconditionally generate tile-based game levels, but use of diffusion models for text-to-level generation is underexplored. There are practical considerations for creating a usable model: caption/level pairs are needed, as is a text embedding model, and a way of generating entire playable levels, rather than individual scenes. We present strategies to automatically assign descriptive captions to an existing level dataset, and train diffusion models using both pretrained text encoders and simple transformer models trained from scratch. Captions are automatically assigned to generated levels so that the degree of overlap between input and output captions can be compared. We also assess the diversity and playability of the resulting levels. Results are compared with an unconditional diffusion model and a generative adversarial network, as well as the text-to-level approaches Five-Dollar Model and MarioGPT. Notably, the best diffusion model uses a simple transformer model for text embedding, and takes less time to train than diffusion models employing more complex text encoders, indicating that reliance on larger language models is not necessary. We also present a GUI allowing designers to construct long levels from model-generated scenes. 

---
# Designing an Adaptive Storytelling Platform to Promote Civic Education in Politically Polarized Learning Environments 

**Authors**: Christopher M. Wegemer, Edward Halim, Jeff Burke  

**Link**: [PDF](https://arxiv.org/pdf/2507.00161)  

**Abstract**: Political polarization undermines democratic civic education by exacerbating identity-based resistance to opposing viewpoints. Emerging AI technologies offer new opportunities to advance interventions that reduce polarization and promote political open-mindedness. We examined novel design strategies that leverage adaptive and emotionally-responsive civic narratives that may sustain students' emotional engagement in stories, and in turn, promote perspective-taking toward members of political out-groups. Drawing on theories from political psychology and narratology, we investigate how affective computing techniques can support three storytelling mechanisms: transportation into a story world, identification with characters, and interaction with the storyteller. Using a design-based research (DBR) approach, we iteratively developed and refined an AI-mediated Digital Civic Storytelling (AI-DCS) platform. Our prototype integrates facial emotion recognition and attention tracking to assess users' affective and attentional states in real time. Narrative content is organized around pre-structured story outlines, with beat-by-beat language adaptation implemented via GPT-4, personalizing linguistic tone to sustain students' emotional engagement in stories that center political perspectives different from their own. Our work offers a foundation for AI-supported, emotionally-sensitive strategies that address affective polarization while preserving learner autonomy. We conclude with implications for civic education interventions, algorithmic literacy, and HCI challenges associated with AI dialogue management and affect-adaptive learning environments. 

---
# AI-Hybrid TRNG: Kernel-Based Deep Learning for Near-Uniform Entropy Harvesting from Physical Noise 

**Authors**: Hasan Yiğit  

**Link**: [PDF](https://arxiv.org/pdf/2507.00145)  

**Abstract**: AI-Hybrid TRNG is a deep-learning framework that extracts near-uniform entropy directly from physical noise, eliminating the need for bulky quantum devices or expensive laboratory-grade RF receivers. Instead, it relies on a low-cost, thumb-sized RF front end, plus CPU-timing jitter, for training, and then emits 32-bit high-entropy streams without any quantization step.
Unlike deterministic or trained artificial intelligence random number generators (RNGs), our dynamic inner-outer network couples adaptive natural sources and reseeding, yielding truly unpredictable and autonomous sequences. Generated numbers pass the NIST SP 800-22 battery better than a CPU-based method. It also passes nineteen bespoke statistical tests for both bit- and integer-level analysis. All results satisfy cryptographic standards, while forward and backward prediction experiments reveal no exploitable biases. The model's footprint is below 0.5 MB, making it deployable on MCUs and FPGA soft cores, as well as suitable for other resource-constrained platforms.
By detaching randomness quality from dedicated hardware, AI-Hybrid TRNG broadens the reach of high-integrity random number generators across secure systems, cryptographic protocols, embedded and edge devices, stochastic simulations, and server applications that need randomness. 

---
# Teaching Programming in the Age of Generative AI: Insights from Literature, Pedagogical Proposals, and Student Perspectives 

**Authors**: Clemente Rubio-Manzano, Jazna Meza, Rodolfo Fernandez-Santibanez, Christian Vidal-Castro  

**Link**: [PDF](https://arxiv.org/pdf/2507.00108)  

**Abstract**: Computer programming is undergoing a true transformation driven by powerful new tools for automatic source code generation based on large language models. This transformation is also manifesting in introductory programming courses at universities around the world, generating an in-depth debate about how programming content should be taught, learned, and assessed in the context of generative artificial intelligence.
This article aims, on the one hand, to review the most relevant studies on this issue, highlighting the advantages and disadvantages identified in the specialized literature. On the other hand, it proposes enriching teaching and learning methodologies by focusing on code comprehension and execution rather than on mere coding or program functionality. In particular, it advocates for the use of visual representations of code and visual simulations of its execution as effective tools for teaching, learning, and assessing programming, thus fostering a deeper understanding among students.
Finally, the opinions of students who took the object-oriented programming course are presented to provide preliminary context supporting the incorporation of visual simulations in Java (or other languages) as part of the training process. 

---
# Towards transparent and data-driven fault detection in manufacturing: A case study on univariate, discrete time series 

**Authors**: Bernd Hofmann, Patrick Bruendl, Huong Giang Nguyen, Joerg Franke  

**Link**: [PDF](https://arxiv.org/pdf/2507.00102)  

**Abstract**: Ensuring consistent product quality in modern manufacturing is crucial, particularly in safety-critical applications. Conventional quality control approaches, reliant on manually defined thresholds and features, lack adaptability to the complexity and variability inherent in production data and necessitate extensive domain expertise. Conversely, data-driven methods, such as machine learning, demonstrate high detection performance but typically function as black-box models, thereby limiting their acceptance in industrial environments where interpretability is paramount. This paper introduces a methodology for industrial fault detection, which is both data-driven and transparent. The approach integrates a supervised machine learning model for multi-class fault classification, Shapley Additive Explanations for post-hoc interpretability, and a do-main-specific visualisation technique that maps model explanations to operator-interpretable features. Furthermore, the study proposes an evaluation methodology that assesses model explanations through quantitative perturbation analysis and evaluates visualisations by qualitative expert assessment. The approach was applied to the crimping process, a safety-critical joining technique, using a dataset of univariate, discrete time series. The system achieves a fault detection accuracy of 95.9 %, and both quantitative selectivity analysis and qualitative expert evaluations confirmed the relevance and inter-pretability of the generated explanations. This human-centric approach is designed to enhance trust and interpretability in data-driven fault detection, thereby contributing to applied system design in industrial quality control. 

---
# AI-Governed Agent Architecture for Web-Trustworthy Tokenization of Alternative Assets 

**Authors**: Ailiya Borjigin, Wei Zhou, Cong He  

**Link**: [PDF](https://arxiv.org/pdf/2507.00096)  

**Abstract**: Alternative Assets tokenization is transforming non-traditional financial instruments are represented and traded on the web. However, ensuring trustworthiness in web-based tokenized ecosystems poses significant challenges, from verifying off-chain asset data to enforcing regulatory compliance. This paper proposes an AI-governed agent architecture that integrates intelligent agents with blockchain to achieve web-trustworthy tokenization of alternative assets. In the proposed architecture, autonomous agents orchestrate the tokenization process (asset verification, valuation, compliance checking, and lifecycle management), while an AI-driven governance layer monitors agent behavior and enforces trust through adaptive policies and cryptoeconomic incentives. We demonstrate that this approach enhances transparency, security, and compliance in asset tokenization, addressing key concerns around data authenticity and fraud. A case study on tokenizing real estate assets illustrates how the architecture mitigates risks (e.g., fraudulent listings and money laundering) through real-time AI anomaly detection and on-chain enforcement. Our evaluation and analysis suggest that combining AI governance with multi-agent systems and blockchain can significantly bolster trust in tokenized asset ecosystems. This work offers a novel framework for trustworthy asset tokenization on the web and provides insights for practitioners aiming to deploy secure, compliant tokenization platforms. 

---
# Efficient Conformance Checking of Rich Data-Aware Declare Specifications (Extended) 

**Authors**: Jacobo Casas-Ramos, Sarah Winkler, Alessandro Gianola, Marco Montali, Manuel Mucientes, Manuel Lama  

**Link**: [PDF](https://arxiv.org/pdf/2507.00094)  

**Abstract**: Despite growing interest in process analysis and mining for data-aware specifications, alignment-based conformance checking for declarative process models has focused on pure control-flow specifications, or mild data-aware extensions limited to numerical data and variable-to-constant comparisons. This is not surprising: finding alignments is computationally hard, even more so in the presence of data dependencies. In this paper, we challenge this problem in the case where the reference model is captured using data-aware Declare with general data types and data conditions. We show that, unexpectedly, it is possible to compute data-aware optimal alignments in this rich setting, enjoying at once efficiency and expressiveness. This is achieved by carefully combining the two best-known approaches to deal with control flow and data dependencies when computing alignments, namely A* search and SMT solving. Specifically, we introduce a novel algorithmic technique that efficiently explores the search space, generating descendant states through the application of repair actions aiming at incrementally resolving constraint violations. We prove the correctness of our algorithm and experimentally show its efficiency. The evaluation witnesses that our approach matches or surpasses the performance of the state of the art while also supporting significantly more expressive data dependencies, showcasing its potential to support real-world applications. 

---
# $σ$-Maximal Ancestral Graphs 

**Authors**: Binghua Yao, Joris M. Mooij  

**Link**: [PDF](https://arxiv.org/pdf/2507.00093)  

**Abstract**: Maximal Ancestral Graphs (MAGs) provide an abstract representation of Directed Acyclic Graphs (DAGs) with latent (selection) variables. These graphical objects encode information about ancestral relations and d-separations of the DAGs they represent. This abstract representation has been used amongst others to prove the soundness and completeness of the FCI algorithm for causal discovery, and to derive a do-calculus for its output. One significant inherent limitation of MAGs is that they rule out the possibility of cyclic causal relationships. In this work, we address that limitation. We introduce and study a class of graphical objects that we coin ''$\sigma$-Maximal Ancestral Graphs'' (''$\sigma$-MAGs''). We show how these graphs provide an abstract representation of (possibly cyclic) Directed Graphs (DGs) with latent (selection) variables, analogously to how MAGs represent DAGs. We study the properties of these objects and provide a characterization of their Markov equivalence classes. 

---
# Generating Heterogeneous Multi-dimensional Data : A Comparative Study 

**Authors**: Corbeau Michael, Claeys Emmanuelle, Serrurier Mathieu, Zaraté Pascale  

**Link**: [PDF](https://arxiv.org/pdf/2507.00090)  

**Abstract**: Allocation of personnel and material resources is highly sensible in the case of firefighter interventions. This allocation relies on simulations to experiment with various scenarios. The main objective of this allocation is the global optimization of the firefighters response. Data generation is then mandatory to study various scenarios In this study, we propose to compare different data generation methods. Methods such as Random Sampling, Tabular Variational Autoencoders, standard Generative Adversarial Networks, Conditional Tabular Generative Adversarial Networks and Diffusion Probabilistic Models are examined to ascertain their efficacy in capturing the intricacies of firefighter interventions. Traditional evaluation metrics often fall short in capturing the nuanced requirements of synthetic datasets for real-world scenarios. To address this gap, an evaluation of synthetic data quality is conducted using a combination of domain-specific metrics tailored to the firefighting domain and standard measures such as the Wasserstein distance. Domain-specific metrics include response time distribution, spatial-temporal distribution of interventions, and accidents representation. These metrics are designed to assess data variability, the preservation of fine and complex correlations and anomalies such as event with a very low occurrence, the conformity with the initial statistical distribution and the operational relevance of the synthetic data. The distribution has the particularity of being highly unbalanced, none of the variables following a Gaussian distribution, adding complexity to the data generation process. 

---
# How large language models judge and influence human cooperation 

**Authors**: Alexandre S. Pires, Laurens Samson, Sennay Ghebreab, Fernando P. Santos  

**Link**: [PDF](https://arxiv.org/pdf/2507.00088)  

**Abstract**: Humans increasingly rely on large language models (LLMs) to support decisions in social settings. Previous work suggests that such tools shape people's moral and political judgements. However, the long-term implications of LLM-based social decision-making remain unknown. How will human cooperation be affected when the assessment of social interactions relies on language models? This is a pressing question, as human cooperation is often driven by indirect reciprocity, reputations, and the capacity to judge interactions of others. Here, we assess how state-of-the-art LLMs judge cooperative actions. We provide 21 different LLMs with an extensive set of examples where individuals cooperate -- or refuse cooperating -- in a range of social contexts, and ask how these interactions should be judged. Furthermore, through an evolutionary game-theoretical model, we evaluate cooperation dynamics in populations where the extracted LLM-driven judgements prevail, assessing the long-term impact of LLMs on human prosociality. We observe a remarkable agreement in evaluating cooperation against good opponents. On the other hand, we notice within- and between-model variance when judging cooperation with ill-reputed individuals. We show that the differences revealed between models can significantly impact the prevalence of cooperation. Finally, we test prompts to steer LLM norms, showing that such interventions can shape LLM judgements, particularly through goal-oriented prompts. Our research connects LLM-based advices and long-term social dynamics, and highlights the need to carefully align LLM norms in order to preserve human cooperation. 

---
# pUniFind: a unified large pre-trained deep learning model pushing the limit of mass spectra interpretation 

**Authors**: Jiale Zhao, Pengzhi Mao, Kaifei Wang, Yiming Li, Yaping Peng, Ranfei Chen, Shuqi Lu, Xiaohong Ji, Jiaxiang Ding, Xin Zhang, Yucheng Liao, Weinan E, Weijie Zhang, Han Wen, Hao Chi  

**Link**: [PDF](https://arxiv.org/pdf/2507.00087)  

**Abstract**: Deep learning has advanced mass spectrometry data interpretation, yet most models remain feature extractors rather than unified scoring frameworks. We present pUniFind, the first large-scale multimodal pre-trained model in proteomics that integrates end-to-end peptide-spectrum scoring with open, zero-shot de novo sequencing. Trained on over 100 million open search-derived spectra, pUniFind aligns spectral and peptide modalities via cross modality prediction and outperforms traditional engines across diverse datasets, particularly achieving a 42.6 percent increase in the number of identified peptides in immunopeptidomics. Supporting over 1,300 modifications, pUniFind identifies 60 percent more PSMs than existing de novo methods despite a 300-fold larger search space. A deep learning based quality control module further recovers 38.5 percent additional peptides including 1,891 mapped to the genome but absent from reference proteomes while preserving full fragment ion coverage. These results establish a unified, scalable deep learning framework for proteomic analysis, offering improved sensitivity, modification coverage, and interpretability. 

---
# A Joint Topology-Data Fusion Graph Network for Robust Traffic Speed Prediction with Data Anomalism 

**Authors**: Ruiyuan Jiang, Dongyao Jia, Eng Gee Lim, Pengfei Fan, Yuli Zhang, Shangbo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.00085)  

**Abstract**: Accurate traffic prediction is essential for Intelligent Transportation Systems (ITS), yet current methods struggle with the inherent complexity and non-linearity of traffic dynamics, making it difficult to integrate spatial and temporal characteristics. Furthermore, existing approaches use static techniques to address non-stationary and anomalous historical data, which limits adaptability and undermines data smoothing. To overcome these challenges, we propose the Graph Fusion Enhanced Network (GFEN), an innovative framework for network-level traffic speed prediction. GFEN introduces a novel topological spatiotemporal graph fusion technique that meticulously extracts and merges spatial and temporal correlations from both data distribution and network topology using trainable methods, enabling the modeling of multi-scale spatiotemporal features. Additionally, GFEN employs a hybrid methodology combining a k-th order difference-based mathematical framework with an attention-based deep learning structure to adaptively smooth historical observations and dynamically mitigate data anomalies and non-stationarity. Extensive experiments demonstrate that GFEN surpasses state-of-the-art methods by approximately 6.3% in prediction accuracy and exhibits convergence rates nearly twice as fast as recent hybrid models, confirming its superior performance and potential to significantly enhance traffic prediction system efficiency. 

---
# Strategic Counterfactual Modeling of Deep-Target Airstrike Systems via Intervention-Aware Spatio-Causal Graph Networks 

**Authors**: Wei Meng  

**Link**: [PDF](https://arxiv.org/pdf/2507.00083)  

**Abstract**: This study addresses the lack of structured causal modeling between tactical strike behavior and strategic delay in current strategic-level simulations, particularly the structural bottlenecks in capturing intermediate variables within the "resilience - nodal suppression - negotiation window" chain. We propose the Intervention-Aware Spatio-Temporal Graph Neural Network (IA-STGNN), a novel framework that closes the causal loop from tactical input to strategic delay output. The model integrates graph attention mechanisms, counterfactual simulation units, and spatial intervention node reconstruction to enable dynamic simulations of strike configurations and synchronization strategies. Training data are generated from a multi-physics simulation platform (GEANT4 + COMSOL) under NIST SP 800-160 standards, ensuring structural traceability and policy-level validation. Experimental results demonstrate that IA-STGNN significantly outperforms baseline models (ST-GNN, GCN-LSTM, XGBoost), achieving a 12.8 percent reduction in MAE and 18.4 percent increase in Top-5 percent accuracy, while improving causal path consistency and intervention stability. IA-STGNN enables interpretable prediction of strategic delay and supports applications such as nuclear deterrence simulation, diplomatic window assessment, and multi-strategy optimization, providing a structured and transparent AI decision-support mechanism for high-level policy modeling. 

---
# Federated Learning-Enabled Hybrid Language Models for Communication-Efficient Token Transmission 

**Authors**: Faranaksadat Solat, Joohyung Lee, Mohamed Seif, Dusit Niyato, H. Vincent Poor  

**Link**: [PDF](https://arxiv.org/pdf/2507.00082)  

**Abstract**: Hybrid Language Models (HLMs) combine the low-latency efficiency of Small Language Models (SLMs) on edge devices with the high accuracy of Large Language Models (LLMs) on centralized servers. Unlike traditional end-to-end LLM inference, HLMs reduce latency and communication by invoking LLMs only when local SLM predictions are uncertain, i.e., when token-level confidence is low or entropy is high. However, ambiguous or low-confidence predictions still require frequent offloading to the LLM, leading to significant communication overhead in bandwidth-constrained settings. To address this, we propose FedHLM, a communication-efficient HLM framework that integrates uncertainty-aware inference with Federated Learning (FL). FedHLM's key innovation lies in collaboratively learning token-level uncertainty thresholds that govern when LLM assistance is needed. Rather than using static or manually tuned thresholds, FedHLM employs FL to optimize these thresholds in a privacy-preserving, distributed manner. Additionally, it leverages embedding-based token representations for Peer-to-Peer (P2P) resolution, enabling clients to reuse tokens inferred by semantically similar peers without engaging the LLM. We further introduce hierarchical model aggregation: edge servers refine local routing policies through client updates, while cross-cluster coordination aligns global decision boundaries. This layered design captures recurring uncertainty patterns, reducing redundant LLM queries. Experiments on large-scale news classification tasks show that FedHLM reduces LLM transmissions by over 95 percent with negligible accuracy loss, making it well-suited for scalable and efficient edge-AI applications. 

---
# State and Memory is All You Need for Robust and Reliable AI Agents 

**Authors**: Matthew Muhoberac, Atharva Parikh, Nirvi Vakharia, Saniya Virani, Aco Radujevic, Savannah Wood, Meghav Verma, Dimitri Metaxotos, Jeyaraman Soundararajan, Thierry Masquelin, Alexander G. Godfrey, Sean Gardner, Dobrila Rudnicki, Sam Michael, Gaurav Chopra  

**Link**: [PDF](https://arxiv.org/pdf/2507.00081)  

**Abstract**: Large language models (LLMs) have enabled powerful advances in natural language understanding and generation. Yet their application to complex, real-world scientific workflows remain limited by challenges in memory, planning, and tool integration. Here, we introduce SciBORG (Scientific Bespoke Artificial Intelligence Agents Optimized for Research Goals), a modular agentic framework that allows LLM-based agents to autonomously plan, reason, and achieve robust and reliable domain-specific task execution. Agents are constructed dynamically from source code documentation and augmented with finite-state automata (FSA) memory, enabling persistent state tracking and context-aware decision-making. This approach eliminates the need for manual prompt engineering and allows for robust, scalable deployment across diverse applications via maintaining context across extended workflows and to recover from tool or execution failures. We validate SciBORG through integration with both physical and virtual hardware, such as microwave synthesizers for executing user-specified reactions, with context-aware decision making and demonstrate its use in autonomous multi-step bioassay retrieval from the PubChem database utilizing multi-step planning, reasoning, agent-to-agent communication and coordination for execution of exploratory tasks. Systematic benchmarking shows that SciBORG agents achieve reliable execution, adaptive planning, and interpretable state transitions. Our results show that memory and state awareness are critical enablers of agentic planning and reliability, offering a generalizable foundation for deploying AI agents in complex environments. 

---
# The language of time: a language model perspective on time-series foundation models 

**Authors**: Yi Xie, Yun Xiong, Zejian Shi, Hao Niu, Zhengfu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.00078)  

**Abstract**: With the rise of large language models, the paradigm of training foundation models with massive parameter counts on vast datasets has been adopted in multiple domains to achieve remarkable success. Time series foundation models represent a significant extension of this paradigm, demonstrating exceptional expressive power, generalization, and cross-domain transferability. However, this gives rise to a fundamental paradox: time series data reflect distinct dynamical systems, making cross-domain transfer intuitively implausible, yet this is contradicted by the models' empirical success. To resolve this paradox, this paper investigates, from both theoretical and experimental perspectives, the representation learning mechanisms and generalization capabilities of patch-based time series foundation models. We argue that such models are not merely applying a new architecture but are fundamentally generalizing the representation paradigm of language models by extending deterministic vector-based representations to latent probabilistic distributional forms. Our theoretical analysis supports this framework by demonstrating that continuous time-series patches can be faithfully quantized into a discrete vocabulary whose key statistical properties are highly consistent with those of natural language. This generalization allows time series models to inherit the robust representation and transfer abilities of large language models, thereby explaining their superior performance in temporal tasks. Ultimately, our work provides a rigorous theoretical cornerstone for understanding, evaluating, and improving the safety and reliability of large-scale time series foundation models. 

---
# Theoretical Modeling of LLM Self-Improvement Training Dynamics Through Solver-Verifier Gap 

**Authors**: Yifan Sun, Yushan Liang, Zhen Zhang, Jiaye Teng  

**Link**: [PDF](https://arxiv.org/pdf/2507.00075)  

**Abstract**: Self-improvement is among the most prominent techniques within the realm of large language models (LLM), aiming to enhance the LLM performance without relying on external data. Despite its significance, generally how LLM performances evolve during the self-improvement process remains underexplored. In this paper, we theoretically model the training dynamics of self-improvement via the concept of solver-verifier gap. This is inspired by the conjecture that the performance enhancement of self-improvement stems from the gap between LLM's solver capability and verifier capability. Based on the theoretical framework, we further introduce how to predict the ultimate power of self-improvement using only information from the first few training epochs. We empirically validate the effectiveness of the theoretical model on various LLMs and datasets. Beyond self-improvement, we extend our analysis to investigate how external data influences these dynamics within the framework. Notably, we find that under limited external data regimes, such external data can be utilized at any stage without significantly affecting final performances, which accords with the empirical observations. 

---
# An efficient plant disease detection using transfer learning approach 

**Authors**: Bosubabu Sambana, Hillary Sunday Nnadi, Mohd Anas Wajid, Nwosu Ogochukwu Fidelia, Claudia Camacho-Zuñiga, Henry Dozie Ajuzie, Edeh Michael Onyema  

**Link**: [PDF](https://arxiv.org/pdf/2507.00070)  

**Abstract**: Plant diseases pose significant challenges to farmers and the agricultural sector at large. However, early detection of plant diseases is crucial to mitigating their effects and preventing widespread damage, as outbreaks can severely impact the productivity and quality of crops. With advancements in technology, there are increasing opportunities for automating the monitoring and detection of disease outbreaks in plants. This study proposed a system designed to identify and monitor plant diseases using a transfer learning approach. Specifically, the study utilizes YOLOv7 and YOLOv8, two state-ofthe-art models in the field of object detection. By fine-tuning these models on a dataset of plant leaf images, the system is able to accurately detect the presence of Bacteria, Fungi and Viral diseases such as Powdery Mildew, Angular Leaf Spot, Early blight and Tomato mosaic virus. The model's performance was evaluated using several metrics, including mean Average Precision (mAP), F1-score, Precision, and Recall, yielding values of 91.05, 89.40, 91.22, and 87.66, respectively. The result demonstrates the superior effectiveness and efficiency of YOLOv8 compared to other object detection methods, highlighting its potential for use in modern agricultural practices. The approach provides a scalable, automated solution for early any plant disease detection, contributing to enhanced crop yield, reduced reliance on manual monitoring, and supporting sustainable agricultural practices. 

---
# MANTA: Cross-Modal Semantic Alignment and Information-Theoretic Optimization for Long-form Multimodal Understanding 

**Authors**: Ziqi Zhong, Daniel Tang  

**Link**: [PDF](https://arxiv.org/pdf/2507.00068)  

**Abstract**: While multi-modal learning has advanced significantly, current approaches often treat modalities separately, creating inconsistencies in representation and reasoning. We introduce MANTA (Multi-modal Abstraction and Normalization via Textual Alignment), a theoretically-grounded framework that unifies visual and auditory inputs into a structured textual space for seamless processing with large language models. MANTA addresses four key challenges: (1) semantic alignment across modalities with information-theoretic optimization, (2) adaptive temporal synchronization for varying information densities, (3) hierarchical content representation for multi-scale understanding, and (4) context-aware retrieval of sparse information from long sequences. We formalize our approach within a rigorous mathematical framework, proving its optimality for context selection under token constraints. Extensive experiments on the challenging task of Long Video Question Answering show that MANTA improves state-of-the-art models by up to 22.6% in overall accuracy, with particularly significant gains (27.3%) on videos exceeding 30 minutes. Additionally, we demonstrate MANTA's superiority on temporal reasoning tasks (23.8% improvement) and cross-modal understanding (25.1% improvement). Our framework introduces novel density estimation techniques for redundancy minimization while preserving rare signals, establishing new foundations for unifying multimodal representations through structured text. 

---
# InSight-R: A Framework for Risk-informed Human Failure Event Identification and Interface-Induced Risk Assessment Driven by AutoGraph 

**Authors**: Xingyu Xiao, Jiejuan Tong, Peng Chen, Jun Sun, Zhe Sui, Jingang Liang, Hongru Zhao, Jun Zhao, Haitao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.00066)  

**Abstract**: Human reliability remains a critical concern in safety-critical domains such as nuclear power, where operational failures are often linked to human error. While conventional human reliability analysis (HRA) methods have been widely adopted, they rely heavily on expert judgment for identifying human failure events (HFEs) and assigning performance influencing factors (PIFs). This reliance introduces challenges related to reproducibility, subjectivity, and limited integration of interface-level data. In particular, current approaches lack the capacity to rigorously assess how human-machine interface design contributes to operator performance variability and error susceptibility. To address these limitations, this study proposes a framework for risk-informed human failure event identification and interface-induced risk assessment driven by AutoGraph (InSight-R). By linking empirical behavioral data to the interface-embedded knowledge graph (IE-KG) constructed by the automated graph-based execution framework (AutoGraph), the InSight-R framework enables automated HFE identification based on both error-prone and time-deviated operational paths. Furthermore, we discuss the relationship between designer-user conflicts and human error. The results demonstrate that InSight-R not only enhances the objectivity and interpretability of HFE identification but also provides a scalable pathway toward dynamic, real-time human reliability assessment in digitalized control environments. This framework offers actionable insights for interface design optimization and contributes to the advancement of mechanism-driven HRA methodologies. 

---
# Smooth-Distill: A Self-distillation Framework for Multitask Learning with Wearable Sensor Data 

**Authors**: Hoang-Dieu Vu, Duc-Nghia Tran, Quang-Tu Pham, Hieu H. Pham, Nicolas Vuillerme, Duc-Tan Tran  

**Link**: [PDF](https://arxiv.org/pdf/2507.00061)  

**Abstract**: This paper introduces Smooth-Distill, a novel self-distillation framework designed to simultaneously perform human activity recognition (HAR) and sensor placement detection using wearable sensor data. The proposed approach utilizes a unified CNN-based architecture, MTL-net, which processes accelerometer data and branches into two outputs for each respective task. Unlike conventional distillation methods that require separate teacher and student models, the proposed framework utilizes a smoothed, historical version of the model itself as the teacher, significantly reducing training computational overhead while maintaining performance benefits. To support this research, we developed a comprehensive accelerometer-based dataset capturing 12 distinct sleep postures across three different wearing positions, complementing two existing public datasets (MHealth and WISDM). Experimental results show that Smooth-Distill consistently outperforms alternative approaches across different evaluation scenarios, achieving notable improvements in both human activity recognition and device placement detection tasks. This method demonstrates enhanced stability in convergence patterns during training and exhibits reduced overfitting compared to traditional multitask learning baselines. This framework contributes to the practical implementation of knowledge distillation in human activity recognition systems, offering an effective solution for multitask learning with accelerometer data that balances accuracy and training efficiency. More broadly, it reduces the computational cost of model training, which is critical for scenarios requiring frequent model updates or training on resource-constrained platforms. The code and model are available at this https URL\_distill. 

---
# Estimating Correctness Without Oracles in LLM-Based Code Generation 

**Authors**: Thomas Valentin, Ardi Madadi, Gaetano Sapia, Marcel Böhme  

**Link**: [PDF](https://arxiv.org/pdf/2507.00057)  

**Abstract**: Generating code from natural language specifications is one of the most successful applications of Large Language Models (LLMs). Yet, they hallucinate: LLMs produce outputs that may be grammatically correct but are factually incorrect. Without an existing, correct implementation (i.e., an oracle), can we quantify how likely the generated program is correct?
In this paper, we propose a measure of incorrectness, called incoherence, that can be estimated efficiently in the absence of an oracle and provides a lower bound on the error, i.e., the probability that the LLM-generated program for that specification is incorrect. Our experiments demonstrate an extraordinary effectiveness. For the average code generation task, our incoherence-based methodology can automatically identify about two-thirds of incorrect programs without reports of false positives. In fact, an oracle-based evaluation of LLMs can be reliably replaced by an incoherence-based evaluation. In particular, we find a very strong agreement between the ranking of LLMs by the number of programs deemed correct via an oracle (pass@1) and the ranking of LLMs by the number of programs deemed correct via our incoherence. 

---
# VSF-Med:A Vulnerability Scoring Framework for Medical Vision-Language Models 

**Authors**: Binesh Sadanandan, Vahid Behzadan  

**Link**: [PDF](https://arxiv.org/pdf/2507.00052)  

**Abstract**: Vision Language Models (VLMs) hold great promise for streamlining labour-intensive medical imaging workflows, yet systematic security evaluations in clinical settings remain scarce. We introduce VSF--Med, an end-to-end vulnerability-scoring framework for medical VLMs that unites three novel components: (i) a rich library of sophisticated text-prompt attack templates targeting emerging threat vectors; (ii) imperceptible visual perturbations calibrated by structural similarity (SSIM) thresholds to preserve clinical realism; and (iii) an eight-dimensional rubric evaluated by two independent judge LLMs, whose raw scores are consolidated via z-score normalization to yield a 0--32 composite risk metric. Built entirely on publicly available datasets and accompanied by open-source code, VSF--Med synthesizes over 30,000 adversarial variants from 5,000 radiology images and enables reproducible benchmarking of any medical VLM with a single command. Our consolidated analysis reports mean z-score shifts of $0.90\sigma$ for persistence-of-attack-effects, $0.74\sigma$ for prompt-injection effectiveness, and $0.63\sigma$ for safety-bypass success across state-of-the-art VLMs. Notably, Llama-3.2-11B-Vision-Instruct exhibits a peak vulnerability increase of $1.29\sigma$ for persistence-of-attack-effects, while GPT-4o shows increases of $0.69\sigma$ for that same vector and $0.28\sigma$ for prompt-injection attacks. 

---
# CaughtCheating: Is Your MLLM a Good Cheating Detective? Exploring the Boundary of Visual Perception and Reasoning 

**Authors**: Ming Li, Chenguang Wang, Yijun Liang, Xiyao Wang, Yuhang Zhou, Xiyang Wu, Yuqing Zhang, Ruiyi Zhang, Tianyi Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2507.00045)  

**Abstract**: Recent agentic Multi-Modal Large Language Models (MLLMs) such as GPT-o3 have achieved near-ceiling scores on various existing benchmarks, motivating a demand for more challenging test tasks. These MLLMs have been reported to excel in a few expert-level tasks for humans, e.g., GeoGuesser, reflecting their potential as a detective who can notice minuscule cues in an image and weave them into coherent, situational explanations, leading to a reliable answer. But can they match the performance of excellent human detectives? To answer this question, we investigate some hard scenarios where GPT-o3 can still handle, and find a common scenario where o3's performance drops to nearly zero, which we name CaughtCheating. It is inspired by the social media requests that ask others to detect suspicious clues from photos shared by the poster's partner. We conduct extensive experiments and analysis to understand why existing MLLMs lack sufficient capability to solve this kind of task. CaughtCheating provides a class of challenging visual perception and reasoning tasks with great value and practical usage. Success in these tasks paves the way for MLLMs to acquire human-level detective perception and reasoning capabilities. 

---
# HistoART: Histopathology Artifact Detection and Reporting Tool 

**Authors**: Seyed Kahaki, Alexander R. Webber, Ghada Zamzmi, Adarsh Subbaswamy, Rucha Deshpande, Aldo Badano  

**Link**: [PDF](https://arxiv.org/pdf/2507.00044)  

**Abstract**: In modern cancer diagnostics, Whole Slide Imaging (WSI) is widely used to digitize tissue specimens for detailed, high-resolution examination; however, other diagnostic approaches, such as liquid biopsy and molecular testing, are also utilized based on the cancer type and clinical context. While WSI has revolutionized digital histopathology by enabling automated, precise analysis, it remains vulnerable to artifacts introduced during slide preparation and scanning. These artifacts can compromise downstream image analysis. To address this challenge, we propose and compare three robust artifact detection approaches for WSIs: (1) a foundation model-based approach (FMA) using a fine-tuned Unified Neural Image (UNI) architecture, (2) a deep learning approach (DLA) built on a ResNet50 backbone, and (3) a knowledge-based approach (KBA) leveraging handcrafted features from texture, color, and frequency-based metrics. The methods target six common artifact types: tissue folds, out-of-focus regions, air bubbles, tissue damage, marker traces, and blood contamination. Evaluations were conducted on 50,000+ image patches from diverse scanners (Hamamatsu, Philips, Leica Aperio AT2) across multiple sites. The FMA achieved the highest patch-wise AUROC of 0.995 (95% CI [0.994, 0.995]), outperforming the ResNet50-based method (AUROC: 0.977, 95% CI [0.977, 0.978]) and the KBA (AUROC: 0.940, 95% CI [0.933, 0.946]). To translate detection into actionable insights, we developed a quality report scorecard that quantifies high-quality patches and visualizes artifact distributions. 

---
# MR-CLIP: Efficient Metadata-Guided Learning of MRI Contrast Representations 

**Authors**: Mehmet Yigit Avci, Pedro Borges, Paul Wright, Mehmet Yigitsoy, Sebastien Ourselin, Jorge Cardoso  

**Link**: [PDF](https://arxiv.org/pdf/2507.00043)  

**Abstract**: Accurate interpretation of Magnetic Resonance Imaging scans in clinical systems is based on a precise understanding of image contrast. This contrast is primarily governed by acquisition parameters, such as echo time and repetition time, which are stored in the DICOM metadata. To simplify contrast identification, broad labels such as T1-weighted or T2-weighted are commonly used, but these offer only a coarse approximation of the underlying acquisition settings. In many real-world datasets, such labels are entirely missing, leaving raw acquisition parameters as the only indicators of contrast. Adding to this challenge, the available metadata is often incomplete, noisy, or inconsistent. The lack of reliable and standardized metadata complicates tasks such as image interpretation, retrieval, and integration into clinical workflows. Furthermore, robust contrast-aware representations are essential to enable more advanced clinical applications, such as achieving modality-invariant representations and data harmonization. To address these challenges, we propose MR-CLIP, a multimodal contrastive learning framework that aligns MR images with their DICOM metadata to learn contrast-aware representations, without relying on manual labels. Trained on a diverse clinical dataset that spans various scanners and protocols, MR-CLIP captures contrast variations across acquisitions and within scans, enabling anatomy-invariant representations. We demonstrate its effectiveness in cross-modal retrieval and contrast classification, highlighting its scalability and potential for further clinical applications. The code and weights are publicly available at this https URL. 

---
# Catastrophic Forgetting Mitigation via Discrepancy-Weighted Experience Replay 

**Authors**: Xinrun Xu, Jianwen Yang, Qiuhong Zhang, Zhanbiao Lian, Zhiming Ding, Shan Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2507.00042)  

**Abstract**: Continually adapting edge models in cloud-edge collaborative object detection for traffic monitoring suffers from catastrophic forgetting, where models lose previously learned knowledge when adapting to new data distributions. This is especially problematic in dynamic traffic environments characterised by periodic variations (e.g., day/night, peak hours), where past knowledge remains valuable. Existing approaches like experience replay and visual prompts offer some mitigation, but struggle to effectively prioritize and leverage historical data for optimal knowledge retention and adaptation. Specifically, simply storing and replaying all historical data can be inefficient, while treating all historical experiences as equally important overlooks their varying relevance to the current domain. This paper proposes ER-EMU, an edge model update algorithm based on adaptive experience replay, to address these limitations. ER-EMU utilizes a limited-size experience buffer managed using a First-In-First-Out (FIFO) principle, and a novel Domain Distance Metric-based Experience Selection (DDM-ES) algorithm. DDM-ES employs the multi-kernel maximum mean discrepancy (MK-MMD) to quantify the dissimilarity between target domains, prioritizing the selection of historical data that is most dissimilar to the current target domain. This ensures training diversity and facilitates the retention of knowledge from a wider range of past experiences, while also preventing overfitting to the new domain. The experience buffer is also updated using a simple random sampling strategy to maintain a balanced representation of previous domains. Experiments on the Bellevue traffic video dataset, involving repeated day/night cycles, demonstrate that ER-EMU consistently improves the performance of several state-of-the-art cloud-edge collaborative object detection frameworks. 

---
# Pattern-Based Graph Classification: Comparison of Quality Measures and Importance of Preprocessing 

**Authors**: Lucas Potin, Rosa Figueiredo, Vincent Labatut, Christine Largeron  

**Link**: [PDF](https://arxiv.org/pdf/2507.00039)  

**Abstract**: Graph classification aims to categorize graphs based on their structural and attribute features, with applications in diverse fields such as social network analysis and bioinformatics. Among the methods proposed to solve this task, those relying on patterns (i.e. subgraphs) provide good explainability, as the patterns used for classification can be directly interpreted. To identify meaningful patterns, a standard approach is to use a quality measure, i.e. a function that evaluates the discriminative power of each pattern. However, the literature provides tens of such measures, making it difficult to select the most appropriate for a given application. Only a handful of surveys try to provide some insight by comparing these measures, and none of them specifically focuses on graphs. This typically results in the systematic use of the most widespread measures, without thorough evaluation. To address this issue, we present a comparative analysis of 38 quality measures from the literature. We characterize them theoretically, based on four mathematical properties. We leverage publicly available datasets to constitute a benchmark, and propose a method to elaborate a gold standard ranking of the patterns. We exploit these resources to perform an empirical comparison of the measures, both in terms of pattern ranking and classification performance. Moreover, we propose a clustering-based preprocessing step, which groups patterns appearing in the same graphs to enhance classification performance. Our experimental results demonstrate the effectiveness of this step, reducing the number of patterns to be processed while achieving comparable performance. Additionally, we show that some popular measures widely used in the literature are not associated with the best results. 

---
# Quality over Quantity: An Effective Large-Scale Data Reduction Strategy Based on Pointwise V-Information 

**Authors**: Fei Chen, Wenchi Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2507.00038)  

**Abstract**: Data reduction plays a vital role in data-centric AI by identifying the most informative instance within large-scale datasets to enhance model training efficiency. The core challenge lies in how to select the optimal instances-rather than the entire datasets-to improve data quality and training efficiency. In this paper, we propose an effective data reduction strategy based on Pointwise V-information(PVI). First, we quantify instance difficulty using PVI and filter out low-difficulty instances enabling a static approach. Experiments demonstrate that removing 10%-30% of the data preserves the classifier performance with only a 0.0001% to 0.76% loss in this http URL, we use a progressive learning approach to training the classifiers on instances sorted by ascending PVI, accelerating convergence and achieving a 0.8% accuracy gain over conventional training. Our results suggest that with the effective data reduction strategy, training a classifier on the selected optimal subset could enhance the model performance and boost training efficiency. Moreover, we have transferred the PVI framework, which previously applied only to English datasets, to diverse Chinese NLP tasks and base models, leading to valuable insights for cross-lingual data reduction and faster training. The codes are released at this https URL. 

---
# Model Fusion via Neuron Interpolation 

**Authors**: Phoomraphee Luenam, Andreas Spanopoulos, Amit Sant, Thomas Hofmann, Sotiris Anagnostidis, Sidak Pal Singh  

**Link**: [PDF](https://arxiv.org/pdf/2507.00037)  

**Abstract**: Model fusion aims to combine the knowledge of multiple models by creating one representative model that captures the strengths of all of its parents. However, this process is non-trivial due to differences in internal representations, which can stem from permutation invariance, random initialization, or differently distributed training data. We present a novel, neuron-centric family of model fusion algorithms designed to integrate multiple trained neural networks into a single network effectively regardless of training data distribution. Our algorithms group intermediate neurons of parent models to create target representations that the fused model approximates with its corresponding sub-network. Unlike prior approaches, our approach incorporates neuron attribution scores into the fusion process. Furthermore, our algorithms can generalize to arbitrary layer types. Experimental results on various benchmark datasets demonstrate that our algorithms consistently outperform previous fusion techniques, particularly in zero-shot and non-IID fusion scenarios. The code is available at this https URL. 

---
# Moment Sampling in Video LLMs for Long-Form Video QA 

**Authors**: Mustafa Chasmai, Gauri Jagatap, Gouthaman KV, Grant Van Horn, Subhransu Maji, Andrea Fanelli  

**Link**: [PDF](https://arxiv.org/pdf/2507.00033)  

**Abstract**: Recent advancements in video large language models (Video LLMs) have significantly advanced the field of video question answering (VideoQA). While existing methods perform well on short videos, they often struggle with long-range reasoning in longer videos. To scale Video LLMs for longer video content, frame sub-sampling (selecting frames at regular intervals) is commonly used. However, this approach is suboptimal, often leading to the loss of crucial frames or the inclusion of redundant information from multiple similar frames. Missing key frames impairs the model's ability to answer questions accurately, while redundant frames lead the model to focus on irrelevant video segments and increase computational resource consumption. In this paper, we investigate the use of a general-purpose text-to-video moment retrieval model to guide the frame sampling process. We propose "moment sampling", a novel, model-agnostic approach that enables the model to select the most relevant frames according to the context of the question. Specifically, we employ a lightweight moment retrieval model to prioritize frame selection. By focusing on the frames most pertinent to the given question, our method enhances long-form VideoQA performance in Video LLMs. Through extensive experiments on four long-form VideoQA datasets, using four state-of-the-art Video LLMs, we demonstrate the effectiveness of the proposed approach. 

---
# Ken Utilization Layer: Hebbian Replay Within a Student's Ken for Adaptive Knowledge Tracing 

**Authors**: Grey Kuling, Marinka Zitnik  

**Link**: [PDF](https://arxiv.org/pdf/2507.00032)  

**Abstract**: We introduce KUL-KT, a biologically inspired architecture for knowledge tracing (KT), combining Hebbian memory encoding with gradient-based consolidation in a scalable, input-agnostic framework. KUL-KT adapts the principle of memory consolidation in neural systems, to student modeling by introducing two key innovations: (i) a time-decaying Hebbian memory update that enables graceful forgetting, and (ii) a novel Loss-aligned Internal Target (LIT) method to compute an ideal internal state, allowing continual learning without backpropagation through time. The architecture consists of a fast Hebbian memory that captures each learner interaction via a single associative update, and a slower linear network that consolidates recalled samples through gradient descent. This design enables few-shot personalization and natural forgetting without storing raw data or relying on large cohort training. Operating entirely in embedding space, KUL-KT supports both structured (tabular) and unstructured (short-answer) inputs. Empirically, KUL-KT outperforms strong baselines on ten public KT benchmarks in rank-sensitive metrics such as nDCG and Recall@10. In a classroom deployment, KUL-KT personalized quizzes from short-answer data, leading to improved learner-perceived helpfulness and reduced difficulty (p < 0.05). Ablation studies confirm that Hebbian decay and LIT are critical for continual adaptation. Compared to a strong graph-based KT model, KUL-KT trains 1.75x faster and uses 99.01\% less memory. These results position KUL-KT as a biologically grounded, memory-efficient, and input-flexible framework for personalized learning at scale. 

---
# Adaptive Action Duration with Contextual Bandits for Deep Reinforcement Learning in Dynamic Environments 

**Authors**: Abhishek Verma, Nallarasan V, Balaraman Ravindran  

**Link**: [PDF](https://arxiv.org/pdf/2507.00030)  

**Abstract**: Deep Reinforcement Learning (DRL) has achieved remarkable success in complex sequential decision-making tasks, such as playing Atari 2600 games and mastering board games. A critical yet underexplored aspect of DRL is the temporal scale of action execution. We propose a novel paradigm that integrates contextual bandits with DRL to adaptively select action durations, enhancing policy flexibility and computational efficiency. Our approach augments a Deep Q-Network (DQN) with a contextual bandit module that learns to choose optimal action repetition rates based on state contexts. Experiments on Atari 2600 games demonstrate significant performance improvements over static duration baselines, highlighting the efficacy of adaptive temporal abstractions in DRL. This paradigm offers a scalable solution for real-time applications like gaming and robotics, where dynamic action durations are critical. 

---
# LoRA-Mixer: Coordinate Modular LoRA Experts Through Serial Attention Routing 

**Authors**: Wenbing Li, Zikai Song, Hang Zhou, Yunyao Zhang, Junqing Yu, Wei Yang  

**Link**: [PDF](https://arxiv.org/pdf/2507.00029)  

**Abstract**: Recent efforts to combine low-rank adaptation (LoRA) with mixture-of-experts (MoE) for adapting large language models (LLMs) to multiple tasks still exhibit prevailing limitations: they either swap entire attention/feed-forward layers for switch experts or bolt on parallel expert branches, diluting parameter efficiency and task fidelity. We propose the LoRA-Mixer, a modular and lightweight MoE framework that integrates LoRA experts. Our core innovation lies in replacing the projection matrices of the attention module's input/output linear layers with dynamically routed, task-specific LoRA experts. This design ensures seamless compatibility with diverse foundation models, including transformers and state space models (SSMs), by leveraging their inherent linear projection structures. The framework supports two operational paradigms: (1) joint optimization of LoRA experts and routing mechanisms via a novel hard-soft routing strategy, or (2) direct deployment of pre-trained, frozen LoRA modules sourced from external repositories. To enable robust router training with limited data while ensuring stable routing decisions and maximizing expert reuse, we introduce an adaptive Specialization Balance Loss (SBL) that jointly optimizes expert balance and task-specific alignment. Extensive experiments on seven benchmark datasets, including MedQA, CoLA, SST-2, GSM8K, ARC-E, ARC-C, and HumanEval, demonstrate the effectiveness of LoRA-Mixer. On datasets such as GSM8K, HumanEval, and MedQA, LoRA-Mixer achieves significant improvements of 7.61%, 4.88%, and 3.08% over the base models, respectively. Compared with state-of-the-art methods, LoRA-Mixer achieves additional improvements of 1.09%, 1.45%, and 1.68%, respectively, using only 48% of the parameters, demonstrating its efficiency and strong performance. 

---
# HiT-JEPA: A Hierarchical Self-supervised Trajectory Embedding Framework for Similarity Computation 

**Authors**: Lihuan Li, Hao Xue, Shuang Ao, Yang Song, Flora Salim  

**Link**: [PDF](https://arxiv.org/pdf/2507.00028)  

**Abstract**: The representation of urban trajectory data plays a critical role in effectively analyzing spatial movement patterns. Despite considerable progress, the challenge of designing trajectory representations that can capture diverse and complementary information remains an open research problem. Existing methods struggle in incorporating trajectory fine-grained details and high-level summary in a single model, limiting their ability to attend to both long-term dependencies while preserving local nuances. To address this, we propose HiT-JEPA (Hierarchical Interactions of Trajectory Semantics via a Joint Embedding Predictive Architecture), a unified framework for learning multi-scale urban trajectory representations across semantic abstraction levels. HiT-JEPA adopts a three-layer hierarchy that progressively captures point-level fine-grained details, intermediate patterns, and high-level trajectory abstractions, enabling the model to integrate both local dynamics and global semantics in one coherent structure. Extensive experiments on multiple real-world datasets for trajectory similarity computation show that HiT-JEPA's hierarchical design yields richer, multi-scale representations. Code is available at: this https URL. 

---
# ROSE: Toward Reality-Oriented Safety Evaluation of Large Language Models 

**Authors**: Jiale Ding, Xiang Zheng, Cong Wang, Wei-Bin Lee, Xingjun Ma, Yu-Gang Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2507.00026)  

**Abstract**: As Large Language Models (LLMs) are increasingly deployed as black-box components in real-world applications, evaluating their safety-especially under adversarial prompting-has become critical. Arguably, effective safety evaluations should be adaptive, evolving with LLM capabilities, and also cover a broad spectrum of harmful topics and real-world scenarios to fully expose potential vulnerabilities. Existing manual safety benchmarks, built on handcrafted adversarial prompts, are limited by their static nature and the intensive labor required to update them, making it difficult to keep pace with rapidly advancing LLMs. In contrast, automated adversarial prompt generation offers a promising path toward adaptive evaluation. However, current methods often suffer from insufficient adversarial topic coverage (topic-level diversity) and weak alignment with real-world contexts. These shortcomings stem from the exploration-exploitation dilemma in black-box optimization and a lack of real-world contextualization, resulting in adversarial prompts that are both topically narrow and scenario-repetitive. To address these issues, we propose Reality-Oriented Safety Evaluation (ROSE), a novel framework that uses multi-objective reinforcement learning to fine-tune an adversarial LLM for generating topically diverse and contextually rich adversarial prompts. Experiments show that ROSE outperforms existing methods in uncovering safety vulnerabilities in state-of-the-art LLMs, with notable improvements in integrated evaluation metrics. We hope ROSE represents a step toward more practical and reality-oriented safety evaluation of LLMs. WARNING: This paper contains examples of potentially harmful text. 

---
# Generalizing to New Dynamical Systems via Frequency Domain Adaptation 

**Authors**: Tiexin Qin, Hong Yan, Haoliang Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.00025)  

**Abstract**: Learning the underlying dynamics from data with deep neural networks has shown remarkable potential in modeling various complex physical dynamics. However, current approaches are constrained in their ability to make reliable predictions in a specific domain and struggle with generalizing to unseen systems that are governed by the same general dynamics but differ in environmental characteristics. In this work, we formulate a parameter-efficient method, Fourier Neural Simulator for Dynamical Adaptation (FNSDA), that can readily generalize to new dynamics via adaptation in the Fourier space. Specifically, FNSDA identifies the shareable dynamics based on the known environments using an automatic partition in Fourier modes and learns to adjust the modes specific for each new environment by conditioning on low-dimensional latent systematic parameters for efficient generalization. We evaluate our approach on four representative families of dynamic systems, and the results show that FNSDA can achieve superior or competitive generalization performance compared to existing methods with a significantly reduced parameter cost. Our code is available at this https URL. 

---
# AIMatDesign: Knowledge-Augmented Reinforcement Learning for Inverse Materials Design under Data Scarcity 

**Authors**: Yeyong Yu, Xilei Bian, Jie Xiong, Xing Wu, Quan Qian  

**Link**: [PDF](https://arxiv.org/pdf/2507.00024)  

**Abstract**: With the growing demand for novel materials, machine learning-driven inverse design methods face significant challenges in reconciling the high-dimensional materials composition space with limited experimental data. Existing approaches suffer from two major limitations: (I) machine learning models often lack reliability in high-dimensional spaces, leading to prediction biases during the design process; (II) these models fail to effectively incorporate domain expert knowledge, limiting their capacity to support knowledge-guided inverse design. To address these challenges, we introduce AIMatDesign, a reinforcement learning framework that addresses these limitations by augmenting experimental data using difference-based algorithms to build a trusted experience pool, accelerating model convergence. To enhance model reliability, an automated refinement strategy guided by large language models (LLMs) dynamically corrects prediction inconsistencies, reinforcing alignment between reward signals and state value functions. Additionally, a knowledge-based reward function leverages expert domain rules to improve stability and efficiency during training. Our experiments demonstrate that AIMatDesign significantly surpasses traditional machine learning and reinforcement learning methods in discovery efficiency, convergence speed, and success rates. Among the numerous candidates proposed by AIMatDesign, experimental synthesis of representative Zr-based alloys yielded a top-performing BMG with 1.7GPa yield strength and 10.2\% elongation, closely matching predictions. Moreover, the framework accurately captured the trend of yield strength variation with composition, demonstrating its reliability and potential for closed-loop materials discovery. 

---
# GLU Attention Improve Transformer 

**Authors**: Zehao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.00022)  

**Abstract**: Gated Linear Units (GLU) have shown great potential in enhancing neural network performance. In this paper, I introduce a novel attention mechanism called GLU Attention, which introduces nonlinearity into the values of Attention. My experiments demonstrate that GLU Attention improves both model performance and convergence speed across text and vision modalities with zero additional parameters and negligible computational costs. GLU Attention is lightweight and can seamlessly integrate with other technologies, such as Flash Attention, Rotary Position Embedding (RoPE), and various Multi-Head Attention (MHA) variants such as Grouped-Query Attention (GQA). This project is open-sourced at github. 

---
# Quantum Inspired Encoding Strategies for Machine Learning Models: Proposing and Evaluating Instance Level, Global Discrete, and Class Conditional Representations 

**Authors**: Minati Rath, Hema Date  

**Link**: [PDF](https://arxiv.org/pdf/2507.00019)  

**Abstract**: In this study, we propose, evaluate and compare three quantum inspired data encoding strategies, Instance Level Strategy (ILS), Global Discrete Strategy (GDS) and Class Conditional Value Strategy (CCVS), for transforming classical data into quantum data for use in pure classical machine learning models. The primary objective is to reduce high encoding time while ensuring correct encoding values and analyzing their impact on classification performance. The Instance Level Strategy treats each row of dataset independently; mimics local quantum states. Global Discrete Value Based encoding strategy maps all unique feature values across the full dataset to quantum states uniformly. In contrast, the Class conditional Value based encoding strategy encodes unique values separately for each class, preserving class dependent information.
We apply these encoding strategies to a classification task and assess their impact on en-coding efficiency, correctness, model accuracy, and computational cost. By analyzing the trade offs between encoding time, precision, and predictive performance, this study provides insights into optimizing quantum inspired data transformations for classical machine learning workflows. 

---
# Implicit Reward as the Bridge: A Unified View of SFT and DPO Connections 

**Authors**: Bo Wang, Qinyuan Cheng, Runyu Peng, Rong Bao, Peiji Li, Qipeng Guo, Linyang Li, Zhiyuan Zeng, Yunhua Zhou, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2507.00018)  

**Abstract**: Post-training processes are essential phases in grounding pre-trained language models to real-world tasks, with learning from demonstrations or preference signals playing a crucial role in this adaptation. We present a unified theoretical framework bridging Supervised Fine-Tuning (SFT) and preference learning in Large Language Model (LLM) post-training. Through rigorous mathematical derivation, we demonstrate that both SFT and preference learning methods like Direct Preference Optimization (DPO) operate within the same optimal policy-reward subspace, with SFT representing a special case of implicit reward learning. Our analysis reveals a critical limitation in conventional SFT: the KL divergence term in distribution matching becomes constant with respect to the policy during optimization, failing to constrain model updates. To address this, we propose a simple yet effective learning rate reduction approach that yields significant performance improvements (up to \textbf{25\%} relative gain and \textbf{6\%} absolute win rate increase in instruction following tasks. Additionally, we derive alternative SFT objectives from various f-divergence functions that preserve the KL term during optimization, further enhancing post-DPO model performance. Finally, we extend the theoretical relationship between LLM logits and Q-functions from preference learning to the SFT context, providing mathematical derivations and experimental validation. 

---
# Gradient-based Fine-Tuning through Pre-trained Model Regularization 

**Authors**: Xuanbo Liu, Liu Liu, Fuxiang Wu, Fusheng Hao, Xianglong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.00016)  

**Abstract**: Large pre-trained models have demonstrated extensive applications across various fields. However, fine-tuning these models for specific downstream tasks demands significant computational resources and storage. One fine-tuning method, gradient-based parameter selection (GPS), focuses on fine-tuning only the parameters with high gradients in each neuron, thereby reducing the number of training parameters. Nevertheless, this approach increases computational resource requirements and storage demands. In this paper, we propose an efficient gradient-based and regularized fine-tuning method (GRFT) that updates the rows or columns of the weight matrix. We theoretically demonstrate that the rows or columns with the highest sum of squared gradients are optimal for updating. This strategy effectively reduces storage overhead and improves the efficiency of parameter selection. Additionally, we incorporate regularization to enhance knowledge transfer from the pre-trained model. GRFT achieves state-of-the-art performance, surpassing existing methods such as GPS, Adapter Tuning, and LoRA. Notably, GRFT requires updating only 1.22% and 0.30% of the total parameters on FGVC and VTAB datasets, respectively, demonstrating its high efficiency and effectiveness. The source code will be released soon. 

---
# Vision Transformer with Adversarial Indicator Token against Adversarial Attacks in Radio Signal Classifications 

**Authors**: Lu Zhang, Sangarapillai Lambotharan, Gan Zheng, Guisheng Liao, Xuekang Liu, Fabio Roli, Carsten Maple  

**Link**: [PDF](https://arxiv.org/pdf/2507.00015)  

**Abstract**: The remarkable success of transformers across various fields such as natural language processing and computer vision has paved the way for their applications in automatic modulation classification, a critical component in the communication systems of Internet of Things (IoT) devices. However, it has been observed that transformer-based classification of radio signals is susceptible to subtle yet sophisticated adversarial attacks. To address this issue, we have developed a defensive strategy for transformer-based modulation classification systems to counter such adversarial attacks. In this paper, we propose a novel vision transformer (ViT) architecture by introducing a new concept known as adversarial indicator (AdvI) token to detect adversarial attacks. To the best of our knowledge, this is the first work to propose an AdvI token in ViT to defend against adversarial attacks. Integrating an adversarial training method with a detection mechanism using AdvI token, we combine a training time defense and running time defense in a unified neural network model, which reduces architectural complexity of the system compared to detecting adversarial perturbations using separate models. We investigate into the operational principles of our method by examining the attention mechanism. We show the proposed AdvI token acts as a crucial element within the ViT, influencing attention weights and thereby highlighting regions or features in the input data that are potentially suspicious or anomalous. Through experimental results, we demonstrate that our approach surpasses several competitive methods in handling white-box attack scenarios, including those utilizing the fast gradient method, projected gradient descent attacks and basic iterative method. 

---
# SWE-Bench-CL: Continual Learning for Coding Agents 

**Authors**: Thomas Joshi, Shayan Chowdhury, Fatih Uysal  

**Link**: [PDF](https://arxiv.org/pdf/2507.00014)  

**Abstract**: Large Language Models (LLMs) have achieved impressive results on static code-generation benchmarks, but real-world software development unfolds as a continuous stream of evolving issues, fixes, and feature requests. We introduce SWE-Bench-CL, a novel continual learning benchmark built on the human-verified SWE-Bench Verified dataset introduced by OpenAI and Princeton-NLP in 2024. By organizing GitHub issues into chronologically ordered sequences that reflect natural repository evolution, SWE-Bench-CL enables direct evaluation of an agent's ability to accumulate experience, transfer knowledge across tasks, and resist catastrophic forgetting. We complement the dataset with (i) a preliminary analysis of inter-task structural similarity and contextual sensitivity, (ii) an interactive LangGraph-based evaluation framework augmented with a FAISS-backed semantic memory module, and (iii) a suite of specialized continual learning metrics -- including average accuracy, forgetting, forward/backward transfer, tool-use efficiency, and a generalized Composite Continual Learning Score and CL-F-beta score -- to capture the stability-plasticity trade-off. We outline a rigorous experimental protocol comparing memory-enabled and memory-disabled agents across diverse Python repositories. All code and data are publicly available at this https URL, providing the community with a reproducible platform for developing more adaptive and robust AI agents in software engineering. 

---
# ST-MTM: Masked Time Series Modeling with Seasonal-Trend Decomposition for Time Series Forecasting 

**Authors**: Hyunwoo Seo, Chiehyeon Lim  

**Link**: [PDF](https://arxiv.org/pdf/2507.00013)  

**Abstract**: Forecasting complex time series is an important yet challenging problem that involves various industrial applications. Recently, masked time-series modeling has been proposed to effectively model temporal dependencies for forecasting by reconstructing masked segments from unmasked ones. However, since the semantic information in time series is involved in intricate temporal variations generated by multiple time series components, simply masking a raw time series ignores the inherent semantic structure, which may cause MTM to learn spurious temporal patterns present in the raw data. To capture distinct temporal semantics, we show that masked modeling techniques should address entangled patterns through a decomposition approach. Specifically, we propose ST-MTM, a masked time-series modeling framework with seasonal-trend decomposition, which includes a novel masking method for the seasonal-trend components that incorporates different temporal variations from each component. ST-MTM uses a period masking strategy for seasonal components to produce multiple masked seasonal series based on inherent multi-periodicity and a sub-series masking strategy for trend components to mask temporal regions that share similar variations. The proposed masking method presents an effective pre-training task for learning intricate temporal variations and dependencies. Additionally, ST-MTM introduces a contrastive learning task to support masked modeling by enhancing contextual consistency among multiple masked seasonal representations. Experimental results show that our proposed ST-MTM achieves consistently superior forecasting performance compared to existing masked modeling, contrastive learning, and supervised forecasting methods. 

---
# Towards Undistillable Models by Minimizing Conditional Mutual Information 

**Authors**: Linfeng Ye, Shayan Mohajer Hamidi, En-hui Yang  

**Link**: [PDF](https://arxiv.org/pdf/2507.00012)  

**Abstract**: A deep neural network (DNN) is said to be undistillable if, when used as a black-box input-output teacher, it cannot be distilled through knowledge distillation (KD). In this case, the distilled student (referred to as the knockoff student) does not outperform a student trained independently with label smoothing (LS student) in terms of prediction accuracy. To protect intellectual property of DNNs, it is desirable to build undistillable DNNs. To this end, it is first observed that an undistillable DNN may have the trait that each cluster of its output probability distributions in response to all sample instances with the same label should be highly concentrated to the extent that each cluster corresponding to each label should ideally collapse into one probability distribution. Based on this observation and by measuring the concentration of each cluster in terms of conditional mutual information (CMI), a new training method called CMI minimized (CMIM) method is proposed, which trains a DNN by jointly minimizing the conventional cross entropy (CE) loss and the CMI values of all temperature scaled clusters across the entire temperature spectrum. The resulting CMIM model is shown, by extensive experiments, to be undistillable by all tested KD methods existing in the literature. That is, the knockoff students distilled by these KD methods from the CMIM model underperform the respective LS students. In addition, the CMIM model is also shown to performs better than the model trained with the CE loss alone in terms of their own prediction accuracy. 

---
# Novel RL approach for efficient Elevator Group Control Systems 

**Authors**: Nathan Vaartjes, Vincent Francois-Lavet  

**Link**: [PDF](https://arxiv.org/pdf/2507.00011)  

**Abstract**: Efficient elevator traffic management in large buildings is critical for minimizing passenger travel times and energy consumption. Because heuristic- or pattern-detection-based controllers struggle with the stochastic and combinatorial nature of dispatching, we model the six-elevator, fifteen-floor system at Vrije Universiteit Amsterdam as a Markov Decision Process and train an end-to-end Reinforcement Learning (RL) Elevator Group Control System (EGCS). Key innovations include a novel action space encoding to handle the combinatorial complexity of elevator dispatching, the introduction of infra-steps to model continuous passenger arrivals, and a tailored reward signal to improve learning efficiency. In addition, we explore various ways to adapt the discounting factor to the infra-step formulation. We investigate RL architectures based on Dueling Double Deep Q-learning, showing that the proposed RL-based EGCS adapts to fluctuating traffic patterns, learns from a highly stochastic environment, and thereby outperforms a traditional rule-based algorithm. 

---
# Integrating Universal Generative AI Platforms in Educational Labs to Foster Critical Thinking and Digital Literacy 

**Authors**: Vasiliy Znamenskiy, Rafael Niyazov, Joel Hernandez  

**Link**: [PDF](https://arxiv.org/pdf/2507.00007)  

**Abstract**: This paper presents a new educational framework for integrating generative artificial intelligence (GenAI) platforms such as ChatGPT, Claude, and Gemini into laboratory activities aimed at developing critical thinking and digital literacy among undergraduate students. Recognizing the limitations and risks of uncritical reliance on large language models (LLMs), the proposed pedagogical model reframes GenAI as a research subject and cognitive tool. Students formulate discipline-specific prompts and evaluate GenAI-generated responses in text, image, and video modalities. A pilot implementation in a general astronomy course for non-science majors demonstrated high levels of engagement and critical reflection, with many students continuing the activity after class and presenting results at a research symposium. The results highlight the importance of structured AI interactions in education and suggest that GenAI can improve learning outcomes when combined with reflective assessment methods. The study proposes a replicable model for interdisciplinary AI-integrated lab work, adaptable to scientific disciplines. See the guide to learning activities based on Generative-Ai platforms: this https URL 

---
# A Theory of Inference Compute Scaling: Reasoning through Directed Stochastic Skill Search 

**Authors**: Austin R. Ellis-Mohr, Anuj K. Nayak, Lav R. Varshney  

**Link**: [PDF](https://arxiv.org/pdf/2507.00004)  

**Abstract**: Large language models (LLMs) demand considerable computational, energy, and financial resources during both training and deployment. While scaling laws for training have guided much of the field's recent progress, inference costs now represent a significant and growing component of the overall resource burden, particularly for reasoning-focused models. Existing characterizations of compute-optimality that consider model size, dataset size, and inference tokens in isolation or in fixed combinations risk overlooking more efficient operating points. We introduce directed stochastic skill search (DS3), a general framework that represents inference as stochastic traversal over a learned skill graph. From a simplified yet expressive instantiation, we derive closed-form expressions for task success and compute cost across a wide range of inference strategies -- including chain-of-thought (CoT) and tree-of-thought (ToT) -- enabling comparative analysis as a function of task difficulty and model capability. To that end, we extend a prior first-principles tripartite graph framework of LLM training to incorporate inference, and separately bridge DS3 with empirical methods that characterize LLM scaling behavior. We theoretically recover empirically observed patterns, including: linear accuracy scaling with logarithmic compute; variation in preferred inference strategies as a function of task difficulty and model capability; emergent behavior elicited by reasoning even when performance plateaus under parameter scaling; and both best-of-N (BoN) and majority voting behavior captured within a unified analytical framework. By explicitly characterizing training-inference interdependencies, our framework deepens theoretical understanding and supports principled algorithmic design and resource allocation. 

---
# Deciding When Not to Decide: Indeterminacy-Aware Intrusion Detection with NeutroSENSE 

**Authors**: Eyhab Al-Masri  

**Link**: [PDF](https://arxiv.org/pdf/2507.00003)  

**Abstract**: This paper presents NeutroSENSE, a neutrosophic-enhanced ensemble framework for interpretable intrusion detection in IoT environments. By integrating Random Forest, XGBoost, and Logistic Regression with neutrosophic logic, the system decomposes prediction confidence into truth (T), falsity (F), and indeterminacy (I) components, enabling uncertainty quantification and abstention. Predictions with high indeterminacy are flagged for review using both global and adaptive, class-specific thresholds. Evaluated on the IoT-CAD dataset, NeutroSENSE achieved 97% accuracy, while demonstrating that misclassified samples exhibit significantly higher indeterminacy (I = 0.62) than correct ones (I = 0.24). The use of indeterminacy as a proxy for uncertainty enables informed abstention and targeted review-particularly valuable in edge deployments. Figures and tables validate the correlation between I-scores and error likelihood, supporting more trustworthy, human-in-the-loop AI decisions. This work shows that neutrosophic logic enhances both accuracy and explainability, providing a practical foundation for trust-aware AI in edge and fog-based IoT security systems. 

---
# Hypertokens: Holographic Associative Memory in Tokenized LLMs 

**Authors**: Christopher James Augeri  

**Link**: [PDF](https://arxiv.org/pdf/2507.00002)  

**Abstract**: Large Language Models (LLMs) exhibit remarkable capabilities but suffer from apparent precision loss, reframed here as information spreading. This reframing shifts the problem from computational precision to an information-theoretic communication issue. We address the K:V and V:K memory problem in LLMs by introducing HDRAM (Holographically Defined Random Access Memory), a symbolic memory framework treating transformer latent space as a spread-spectrum channel. Built upon hypertokens, structured symbolic codes integrating classical error-correcting codes (ECC), holographic computing, and quantum-inspired search, HDRAM recovers distributed information through principled despreading. These phase-coherent memory addresses enable efficient key-value operations and Grover-style search in latent space. By combining ECC grammar with compressed sensing and Krylov subspace alignment, HDRAM significantly improves associative retrieval without architectural changes, demonstrating how Classical-Holographic-Quantum-inspired (CHQ) principles can fortify transformer architectures. 

---
