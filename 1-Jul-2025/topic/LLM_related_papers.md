# Performance of LLMs on Stochastic Modeling Operations Research Problems: From Theory to Practice 

**Authors**: Akshit Kumar, Tianyi Peng, Yuhang Wu, Assaf Zeevi  

**Link**: [PDF](https://arxiv.org/pdf/2506.23924)  

**Abstract**: Large language models (LLMs) have exhibited expert-level capabilities across various domains. However, their abilities to solve problems in Operations Research (OR) -- the analysis and optimization of mathematical models derived from real-world problems or their verbal descriptions -- remain underexplored. In this work, we take a first step toward evaluating LLMs' abilities to solve stochastic modeling problems, a core class of OR problems characterized by uncertainty and typically involving tools from probability, statistics, and stochastic processes. We manually procure a representative set of graduate-level homework and doctoral qualification-exam problems and test LLMs' abilities to solve them. We further leverage SimOpt, an open-source library of simulation-optimization problems and solvers, to investigate LLMs' abilities to make real-world decisions under uncertainty. Our results show that, though a nontrivial amount of work is still needed to reliably automate the stochastic modeling pipeline in reality, state-of-the-art LLMs demonstrate proficiency on par with human experts in both classroom and practical settings. These findings highlight the potential of building AI agents that assist OR researchers and amplify the real-world impact of OR through automation. 

---
# Evaluating Multi-Agent Defences Against Jailbreaking Attacks on Large Language Models 

**Authors**: Maria Carolina Cornelia Wit, Jun Pang  

**Link**: [PDF](https://arxiv.org/pdf/2506.23576)  

**Abstract**: Recent advances in large language models (LLMs) have raised concerns about jailbreaking attacks, i.e., prompts that bypass safety mechanisms. This paper investigates the use of multi-agent LLM systems as a defence against such attacks. We evaluate three jailbreaking strategies, including the original AutoDefense attack and two from Deepleaps: BetterDan and JB. Reproducing the AutoDefense framework, we compare single-agent setups with two- and three-agent configurations. Our results show that multi-agent systems enhance resistance to jailbreaks, especially by reducing false negatives. However, its effectiveness varies by attack type, and it introduces trade-offs such as increased false positives and computational overhead. These findings point to the limitations of current automated defences and suggest directions for improving alignment robustness in future LLM systems. 

---
# MMReason: An Open-Ended Multi-Modal Multi-Step Reasoning Benchmark for MLLMs Toward AGI 

**Authors**: Huanjin Yao, Jiaxing Huang, Yawen Qiu, Michael K. Chen, Wenzheng Liu, Wei Zhang, Wenjie Zeng, Xikun Zhang, Jingyi Zhang, Yuxin Song, Wenhao Wu, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2506.23563)  

**Abstract**: Reasoning plays a crucial role in advancing Multimodal Large Language Models (MLLMs) toward Artificial General Intelligence. However, existing MLLM benchmarks often fall short in precisely and comprehensively evaluating long-chain reasoning abilities from three key aspects: (1) lack of difficulty and diversity, (2) susceptibility to guessability and memorization, (3) inadequate assessment of intermediate reasoning steps. To fill this gap, we introduce MMReason, a new benchmark designed to precisely and comprehensively evaluate MLLM long-chain reasoning capability with diverse, open-ended, challenging questions. First, we curate challenging questions requiring multi-step reasoning from various fields (i.e., 6 disciplines) and multiple difficulty levels (i.e., from pre-university to university, and from foundational to competition tiers). Second, these questions are reformulated into an open-ended format and filtered using a multi-model voting technique to eliminate shortcut cases related to guessing and memorization, ensuring robust reasoning evaluations. Third, we annotate the questions with detailed step-by-step solutions, and design a reference-based ternary scoring mechanism to reliably assess intermediate reasoning steps. With MMReason, we benchmark popular leading MLLMs and provide an in-depth analysis of their reasoning capabilities. We hope MMReason will serve as a valuable resource for advancing MLLM reasoning research. Code will be available at this https URL. 

---
# SPIRAL: Self-Play on Zero-Sum Games Incentivizes Reasoning via Multi-Agent Multi-Turn Reinforcement Learning 

**Authors**: Bo Liu, Leon Guertler, Simon Yu, Zichen Liu, Penghui Qi, Daniel Balcells, Mickel Liu, Cheston Tan, Weiyan Shi, Min Lin, Wee Sun Lee, Natasha Jaques  

**Link**: [PDF](https://arxiv.org/pdf/2506.24119)  

**Abstract**: Recent advances in reinforcement learning have shown that language models can develop sophisticated reasoning through training on tasks with verifiable rewards, but these approaches depend on human-curated problem-answer pairs and domain-specific reward engineering. We introduce SPIRAL, a self-play framework where models learn by playing multi-turn, zero-sum games against continuously improving versions of themselves, eliminating the need for human supervision. Through self-play, SPIRAL generates an infinite curriculum of progressively challenging problems as models must constantly adapt to stronger opponents. To enable this self-play training at scale, We implement a fully online, multi-turn, multi-agent reinforcement learning system for LLMs and propose role-conditioned advantage estimation (RAE) to stabilize multi-agent training. Using SPIRAL, self-play on zero-sum games produces reasoning capabilities that transfer broadly. Training Qwen3-4B-Base on Kuhn Poker alone achieves 8.6% improvement on math and 8.4% on general reasoning, outperforming SFT on 25,000 expert game trajectories. Analysis reveals that this transfer occurs through three cognitive patterns: systematic decomposition, expected value calculation, and case-by-case analysis. Multi-game training (TicTacToe, Kuhn Poker, Simple Negotiation) further enhances performance as each game develops distinct reasoning strengths. Applying SPIRAL to a strong reasoning model (DeepSeek-R1-Distill-Qwen-7B) can still lead to 2.0% average improvement. These results demonstrate that zero-sum games naturally develop transferable reasoning capabilities, highlighting a promising direction for autonomous reasoning development. 

---
# ChemActor: Enhancing Automated Extraction of Chemical Synthesis Actions with LLM-Generated Data 

**Authors**: Yu Zhang, Ruijie Yu, Jidong Tian, Feng Zhu, Jiapeng Liu, Xiaokang Yang, Yaohui Jin, Yanyan Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.23520)  

**Abstract**: With the increasing interest in robotic synthesis in the context of organic chemistry, the automated extraction of chemical procedures from literature is critical. However, this task remains challenging due to the inherent ambiguity of chemical language and the high cost of human annotation required for developing reliable computer-aided extraction protocols. Here, we present ChemActor, a fully fine-tuned large language model (LLM), as a chemical executor to convert between unstructured experimental procedures and structured action sequences. We propose a sequential LLM-generated data framework to address the challenges of insufficient and low-quality annotated data. This framework integrates a data selection module that selects data based on distribution divergence, with a general-purpose LLM, to generate machine-executable actions from a single molecule input. Additionally, we introduce a novel multi-round LLMs circle review metric, which reflects the model's advanced understanding of chemical experimental procedures. Extensive experiments on reaction-to-description (R2D) and description-to-action (D2A) tasks demonstrate that ChemActor, augmented by LLM-generated data, achieves state-of-the-art performance, outperforming the baseline model by 10%. The code is available at: this https URL. 

---
# Are Large Language Models Capable of Deep Relational Reasoning? Insights from DeepSeek-R1 and Benchmark Comparisons 

**Authors**: Chi Chiu So, Yueyue Sun, Jun-Min Wang, Siu Pang Yung, Anthony Wai Keung Loh, Chun Pong Chau  

**Link**: [PDF](https://arxiv.org/pdf/2506.23128)  

**Abstract**: How far are Large Language Models (LLMs) in performing deep relational reasoning? In this paper, we evaluate and compare the reasoning capabilities of three cutting-edge LLMs, namely, DeepSeek-R1, DeepSeek-V3 and GPT-4o, through a suite of carefully designed benchmark tasks in family tree and general graph reasoning. Our experiments reveal that DeepSeek-R1 consistently achieves the highest F1-scores across multiple tasks and problem sizes, demonstrating strong aptitude in logical deduction and relational inference. However, all evaluated models, including DeepSeek-R1, struggle significantly as problem complexity increases, largely due to token length limitations and incomplete output structures. A detailed analysis of DeepSeek-R1's long Chain-of-Thought responses uncovers its unique planning and verification strategies, but also highlights instances of incoherent or incomplete reasoning, calling attention to the need for deeper scrutiny into LLMs' internal inference dynamics. We further discuss key directions for future work, including the role of multimodal reasoning and the systematic examination of reasoning failures. Our findings provide both empirical insights and theoretical implications for advancing LLMs' reasoning abilities, particularly in tasks that demand structured, multi-step logical inference. Our code repository will be publicly available at this https URL. 

---
# Can Large Language Models Capture Human Risk Preferences? A Cross-Cultural Study 

**Authors**: Bing Song, Jianing Liu, Sisi Jian, Chenyang Wu, Vinayak Dixit  

**Link**: [PDF](https://arxiv.org/pdf/2506.23107)  

**Abstract**: Large language models (LLMs) have made significant strides, extending their applications to dialogue systems, automated content creation, and domain-specific advisory tasks. However, as their use grows, concerns have emerged regarding their reliability in simulating complex decision-making behavior, such as risky decision-making, where a single choice can lead to multiple outcomes. This study investigates the ability of LLMs to simulate risky decision-making scenarios. We compare model-generated decisions with actual human responses in a series of lottery-based tasks, using transportation stated preference survey data from participants in Sydney, Dhaka, Hong Kong, and Nanjing. Demographic inputs were provided to two LLMs -- ChatGPT 4o and ChatGPT o1-mini -- which were tasked with predicting individual choices. Risk preferences were analyzed using the Constant Relative Risk Aversion (CRRA) framework. Results show that both models exhibit more risk-averse behavior than human participants, with o1-mini aligning more closely with observed human decisions. Further analysis of multilingual data from Nanjing and Hong Kong indicates that model predictions in Chinese deviate more from actual responses compared to English, suggesting that prompt language may influence simulation performance. These findings highlight both the promise and the current limitations of LLMs in replicating human-like risk behavior, particularly in linguistic and cultural settings. 

---
# GATSim: Urban Mobility Simulation with Generative Agents 

**Authors**: Qi Liu, Can Li, Wanjing Ma  

**Link**: [PDF](https://arxiv.org/pdf/2506.23306)  

**Abstract**: Traditional agent-based urban mobility simulations rely on rigid rule-based systems that fail to capture the complexity, adaptability, and behavioral diversity characteristic of human travel decision-making. Recent advances in large language models and AI agent technology offer opportunities to create agents with reasoning capabilities, persistent memory, and adaptive learning mechanisms. We propose GATSim (Generative-Agent Transport Simulation), a novel framework that leverages these advances to create generative agents with rich behavioral characteristics for urban mobility simulation. Unlike conventional approaches, GATSim agents possess diverse socioeconomic attributes, individual lifestyles, and evolving preferences that shape their mobility decisions through psychologically-informed memory systems, tool usage capabilities, and lifelong learning mechanisms. The main contributions of this study include: (1) a comprehensive architecture combining an urban mobility foundation model with agent cognitive systems and transport simulation environment, (2) a fully functional prototype implementation, and (3) systematic validation demonstrating that generative agents produce believable travel behaviors. Through designed reflection processes, generative agents in this study can transform specific travel experiences into generalized insights, enabling realistic behavioral adaptation over time with specialized mechanisms for activity planning and real-time reactive behaviors tailored to urban mobility contexts. Experiments show that generative agents perform competitively with human annotators in mobility scenarios while naturally producing macroscopic traffic evolution patterns. The code for the prototype system is shared at this https URL. 

---
# PokéAI: A Goal-Generating, Battle-Optimizing Multi-agent System for Pokemon Red 

**Authors**: Zihao Liu, Xinhang Sui, Yueran Song, Siwen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.23689)  

**Abstract**: We introduce PokéAI, the first text-based, multi-agent large language model (LLM) framework designed to autonomously play and progress through Pokémon Red. Our system consists of three specialized agents-Planning, Execution, and Critique-each with its own memory bank, role, and skill set. The Planning Agent functions as the central brain, generating tasks to progress through the game. These tasks are then delegated to the Execution Agent, which carries them out within the game environment. Upon task completion, the Critique Agent evaluates the outcome to determine whether the objective was successfully achieved. Once verification is complete, control returns to the Planning Agent, forming a closed-loop decision-making system.
As a preliminary step, we developed a battle module within the Execution Agent. Our results show that the battle AI achieves an average win rate of 80.8% across 50 wild encounters, only 6% lower than the performance of an experienced human player. Furthermore, we find that a model's battle performance correlates strongly with its LLM Arena score on language-related tasks, indicating a meaningful link between linguistic ability and strategic reasoning. Finally, our analysis of gameplay logs reveals that each LLM exhibits a unique playstyle, suggesting that individual models develop distinct strategic behaviors. 

---
# Improving Rationality in the Reasoning Process of Language Models through Self-playing Game 

**Authors**: Pinzheng Wang, Juntao Li, Zecheng Tang, Haijia Gui, Min zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.22920)  

**Abstract**: Large language models (LLMs) have demonstrated considerable reasoning abilities in various tasks such as mathematics and coding. However, recent studies indicate that even the best models lack true comprehension of their reasoning processes. In this paper, we explore how self-play can enhance the rationality of models in the reasoning process without supervision from humans or superior models. We design a Critic-Discernment Game(CDG) in which a prover first provides a solution to a given problem and is subsequently challenged by critiques of its solution. These critiques either aim to assist or mislead the prover. The objective of the prover is to maintain the correct answer when faced with misleading comments, while correcting errors in response to constructive feedback. Our experiments on tasks involving mathematical reasoning, stepwise error detection, self-correction, and long-chain reasoning demonstrate that CDG training can significantly improve the ability of well-aligned LLMs to comprehend their reasoning process. 

---
# The Confidence Paradox: Can LLM Know When It's Wrong 

**Authors**: Sahil Tripathi, Md Tabrez Nafis, Imran Hussain, Jiechao Gao  

**Link**: [PDF](https://arxiv.org/pdf/2506.23464)  

**Abstract**: Document Visual Question Answering (DocVQA) systems are increasingly deployed in real world applications, yet they remain ethically opaque-often producing overconfident answers to ambiguous questions or failing to communicate uncertainty in a trustworthy manner. This misalignment between model confidence and actual knowledge poses significant risks, particularly in domains requiring ethical accountability. Existing approaches such as LayoutLMv3, UDOP, and DONUT have advanced SOTA performance by focusing on architectural sophistication and accuracy; however, they fall short in ethical responsiveness.
To address these limitations, we introduce HonestVQA, a self-supervised honesty calibration framework for ethically aligned DocVQA. Our model-agnostic method quantifies uncertainty to identify knowledge gaps, aligns model confidence with actual correctness using weighted loss functions, and enforces ethical response behavior via contrastive learning. We further introduce two principled evaluation metrics--Honesty Score (H-Score) and Ethical Confidence Index (ECI)--to benchmark alignment between confidence, accuracy, and ethical communication. Empirically, HonestVQA improves DocVQA accuracy by up to 4.3% and F1 by 4.3% across SpDocVQA, InfographicsVQA, and SROIE datasets. It reduces overconfidence, lowering H-Score and ECI by 0.072 and 0.078, respectively. In cross domain evaluation, it achieves up to 78.9% accuracy and 76.1% F1-score, demonstrating strong generalization. Ablation shows a 3.8% drop in accuracy without alignment or contrastive loss. 

---
# AURA: Agent for Understanding, Reasoning, and Automated Tool Use in Voice-Driven Tasks 

**Authors**: Leander Melroy Maben, Gayathri Ganesh Lakshmy, Srijith Radhakrishnan, Siddhant Arora, Shinji Watanabe  

**Link**: [PDF](https://arxiv.org/pdf/2506.23049)  

**Abstract**: Despite advances in language and speech technologies, no open-source system enables full speech-to-speech, multi-turn dialogue with integrated tool use and agentic reasoning. We introduce AURA (Agent for Understanding, Reasoning, and Automated Tool Use), the first open-source, speech-native assistant capable of completing complex, goal-driven tasks through dynamic tool invocation and multi-turn conversation. AURA combines open-weight ASR, TTS, and LLMs in a cascaded pipeline and supports tools such as calendar booking, contact lookup, web search, and email. Its modular design allows easy integration of new tools using natural language prompts and action classes. On VoiceBench, AURA scores 92.75% on OpenBookQA-outperforming all open-weight systems and nearing GPT-4o-and 4.39 on AlpacaEval, competitive with other open-weight systems. Human evaluation shows 90% task success on complex, multi-turn speech tasks. 

---
# Corrupted by Reasoning: Reasoning Language Models Become Free-Riders in Public Goods Games 

**Authors**: David Guzman Piedrahita, Yongjin Yang, Mrinmaya Sachan, Giorgia Ramponi, Bernhard Schölkopf, Zhijing Jin  

**Link**: [PDF](https://arxiv.org/pdf/2506.23276)  

**Abstract**: As large language models (LLMs) are increasingly deployed as autonomous agents, understanding their cooperation and social mechanisms is becoming increasingly important. In particular, how LLMs balance self-interest and collective well-being is a critical challenge for ensuring alignment, robustness, and safe deployment. In this paper, we examine the challenge of costly sanctioning in multi-agent LLM systems, where an agent must decide whether to invest its own resources to incentivize cooperation or penalize defection. To study this, we adapt a public goods game with institutional choice from behavioral economics, allowing us to observe how different LLMs navigate social dilemmas over repeated interactions. Our analysis reveals four distinct behavioral patterns among models: some consistently establish and sustain high levels of cooperation, others fluctuate between engagement and disengagement, some gradually decline in cooperative behavior over time, and others rigidly follow fixed strategies regardless of outcomes. Surprisingly, we find that reasoning LLMs, such as the o1 series, struggle significantly with cooperation, whereas some traditional LLMs consistently achieve high levels of cooperation. These findings suggest that the current approach to improving LLMs, which focuses on enhancing their reasoning capabilities, does not necessarily lead to cooperation, providing valuable insights for deploying LLM agents in environments that require sustained collaboration. Our code is available at this https URL 

---
# ReasonBridge: Efficient Reasoning Transfer from Closed to Open-Source Language Models 

**Authors**: Ziqi Zhong, Xunzhu Tang  

**Link**: [PDF](https://arxiv.org/pdf/2506.22865)  

**Abstract**: Recent advancements in Large Language Models (LLMs) have revealed a significant performance gap between closed-source and open-source models, particularly in tasks requiring complex reasoning and precise instruction following. This paper introduces ReasonBridge, a methodology that efficiently transfers reasoning capabilities from powerful closed-source to open-source models through a novel hierarchical knowledge distillation framework. We develop a tailored dataset Reason1K with only 1,000 carefully curated reasoning traces emphasizing difficulty, diversity, and quality. These traces are filtered from across multiple domains using a structured multi-criteria selection algorithm. Our transfer learning approach incorporates: (1) a hierarchical distillation process capturing both strategic abstraction and tactical implementation patterns, (2) a sparse reasoning-focused adapter architecture requiring only 0.3% additional trainable parameters, and (3) a test-time compute scaling mechanism using guided inference interventions. Comprehensive evaluations demonstrate that ReasonBridge improves reasoning capabilities in open-source models by up to 23% on benchmark tasks, significantly narrowing the gap with closed-source models. Notably, the enhanced Qwen2.5-14B outperforms Claude-Sonnet3.5 on MATH500 and matches its performance on competition-level AIME problems. Our methodology generalizes effectively across diverse reasoning domains and model architectures, establishing a sample-efficient approach to reasoning enhancement for instruction following. 

---
# URSA: The Universal Research and Scientific Agent 

**Authors**: Michael Grosskopf, Russell Bent, Rahul Somasundaram, Isaac Michaud, Arthur Lui, Nathan Debardeleben, Earl Lawrence  

**Link**: [PDF](https://arxiv.org/pdf/2506.22653)  

**Abstract**: Large language models (LLMs) have moved far beyond their initial form as simple chatbots, now carrying out complex reasoning, planning, writing, coding, and research tasks. These skills overlap significantly with those that human scientists use day-to-day to solve complex problems that drive the cutting edge of research. Using LLMs in "agentic" AI has the potential to revolutionize modern science and remove bottlenecks to progress. In this work, we present URSA, a scientific agent ecosystem for accelerating research tasks. URSA consists of a set of modular agents and tools, including coupling to advanced physics simulation codes, that can be combined to address scientific problems of varied complexity and impact. This work highlights the architecture of URSA, as well as examples that highlight the potential of the system. 

---
# Bootstrapping Human-Like Planning via LLMs 

**Authors**: David Porfirio, Vincent Hsiao, Morgan Fine-Morris, Leslie Smith, Laura M. Hiatt  

**Link**: [PDF](https://arxiv.org/pdf/2506.22604)  

**Abstract**: Robot end users increasingly require accessible means of specifying tasks for robots to perform. Two common end-user programming paradigms include drag-and-drop interfaces and natural language programming. Although natural language interfaces harness an intuitive form of human communication, drag-and-drop interfaces enable users to meticulously and precisely dictate the key actions of the robot's task. In this paper, we investigate the degree to which both approaches can be combined. Specifically, we construct a large language model (LLM)-based pipeline that accepts natural language as input and produces human-like action sequences as output, specified at a level of granularity that a human would produce. We then compare these generated action sequences to another dataset of hand-specified action sequences. Although our results reveal that larger models tend to outperform smaller ones in the production of human-like action sequences, smaller models nonetheless achieve satisfactory performance. 

---
# Leveraging the Potential of Prompt Engineering for Hate Speech Detection in Low-Resource Languages 

**Authors**: Ruhina Tabasshum Prome, Tarikul Islam Tamiti, Anomadarshi Barua  

**Link**: [PDF](https://arxiv.org/pdf/2506.23930)  

**Abstract**: The rapid expansion of social media leads to a marked increase in hate speech, which threatens personal lives and results in numerous hate crimes. Detecting hate speech presents several challenges: diverse dialects, frequent code-mixing, and the prevalence of misspelled words in user-generated content on social media platforms. Recent progress in hate speech detection is typically concentrated on high-resource languages. However, low-resource languages still face significant challenges due to the lack of large-scale, high-quality datasets. This paper investigates how we can overcome this limitation via prompt engineering on large language models (LLMs) focusing on low-resource Bengali language. We investigate six prompting strategies - zero-shot prompting, refusal suppression, flattering the classifier, multi-shot prompting, role prompting, and finally our innovative metaphor prompting to detect hate speech effectively in low-resource languages. We pioneer the metaphor prompting to circumvent the built-in safety mechanisms of LLMs that marks a significant departure from existing jailbreaking methods. We investigate all six different prompting strategies on the Llama2-7B model and compare the results extensively with three pre-trained word embeddings - GloVe, Word2Vec, and FastText for three different deep learning models - multilayer perceptron (MLP), convolutional neural network (CNN), and bidirectional gated recurrent unit (BiGRU). To prove the effectiveness of our metaphor prompting in the low-resource Bengali language, we also evaluate it in another low-resource language - Hindi, and two high-resource languages - English and German. The performance of all prompting techniques is evaluated using the F1 score, and environmental impact factor (IF), which measures CO$_2$ emissions, electricity usage, and computational time. 

---
# Towards the "Digital Me": A vision of authentic Conversational Agents powered by personal Human Digital Twins 

**Authors**: Lluís C. Coll, Martin W. Lauer-Schmaltz, Philip Cash, John P. Hansen, Anja Maier  

**Link**: [PDF](https://arxiv.org/pdf/2506.23826)  

**Abstract**: Human Digital Twins (HDTs) have traditionally been conceptualized as data-driven models designed to support decision-making across various domains. However, recent advancements in conversational AI open new possibilities for HDTs to function as authentic, interactive digital counterparts of individuals. This paper introduces a novel HDT system architecture that integrates large language models with dynamically updated personal data, enabling it to mirror an individual's conversational style, memories, and behaviors. To achieve this, our approach implements context-aware memory retrieval, neural plasticity-inspired consolidation, and adaptive learning mechanisms, creating a more natural and evolving digital persona. The resulting system does not only replicate an individual's unique conversational style depending on who they are speaking with, but also enriches responses with dynamically captured personal experiences, opinions, and memories. While this marks a significant step toward developing authentic virtual counterparts, it also raises critical ethical concerns regarding privacy, accountability, and the long-term implications of persistent digital identities. This study contributes to the field of HDTs by describing our novel system architecture, demonstrating its capabilities, and discussing future directions and emerging challenges to ensure the responsible and ethical development of HDTs. 

---
# AutoEvoEval: An Automated Framework for Evolving Close-Ended LLM Evaluation Data 

**Authors**: JiaRu Wu, Mingwei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.23735)  

**Abstract**: Large language models (LLMs) have shown remarkable performance on various tasks, but existing evaluation benchmarks are often static and insufficient to fully assess their robustness and generalization in realistic scenarios. Prior work using evolutionary or adversarial data augmentation has improved evaluation diversity but lacks systematic control over perturbation types and multi-step complexity, limiting comprehensive robustness analysis. To address these gaps, we propose AutoEvoEval, an evolution-based evaluation framework for close-ended tasks such as multi-choice question answering. AutoEvoEval introduces 22 interpretable atomic evolution operations and supports multi-round compositions, enabling controlled generation of diverse, challenging, and realistic test samples. We conduct extensive experiments addressing four research questions on a broad set of open- and closed-source LLMs. Our results show that atomic operations cause an average accuracy drop of 7.283\%, with structure-disrupting or misleading semantic edits causing the largest declines. Model sensitivities vary significantly for the same perturbation, and combining multiple evolution steps amplifies adversarial effects by up to 52.932\%. These findings suggest current benchmarks may overestimate true model generalization and emphasize the need for evolution-aware robustness evaluation. Code and resources are available at: this https URL. 

---
# Software Engineering for Large Language Models: Research Status, Challenges and the Road Ahead 

**Authors**: Hongzhou Rao, Yanjie Zhao, Xinyi Hou, Shenao Wang, Haoyu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.23762)  

**Abstract**: The rapid advancement of large language models (LLMs) has redefined artificial intelligence (AI), pushing the boundaries of AI research and enabling unbounded possibilities for both academia and the industry. However, LLM development faces increasingly complex challenges throughout its lifecycle, yet no existing research systematically explores these challenges and solutions from the perspective of software engineering (SE) approaches. To fill the gap, we systematically analyze research status throughout the LLM development lifecycle, divided into six phases: requirements engineering, dataset construction, model development and enhancement, testing and evaluation, deployment and operations, and maintenance and evolution. We then conclude by identifying the key challenges for each phase and presenting potential research directions to address these challenges. In general, we provide valuable insights from an SE perspective to facilitate future advances in LLM development. 

---
# Interactive Reasoning: Visualizing and Controlling Chain-of-Thought Reasoning in Large Language Models 

**Authors**: Rock Yuren Pang, K. J. Kevin Feng, Shangbin Feng, Chu Li, Weijia Shi, Yulia Tsvetkov, Jeffrey Heer, Katharina Reinecke  

**Link**: [PDF](https://arxiv.org/pdf/2506.23678)  

**Abstract**: The output quality of large language models (LLMs) can be improved via "reasoning": generating segments of chain-of-thought (CoT) content to further condition the model prior to producing user-facing output. While these chains contain valuable information, they are verbose and lack explicit organization, making them tedious to review. Moreover, they lack opportunities for user feedback, such as to remove unwanted considerations, add desired ones, or clarify unclear assumptions. We introduce Interactive Reasoning, an interaction design that visualizes chain-of-thought outputs as a hierarchy of topics and enables user review and modification. We implement interactive reasoning in Hippo, a prototype for AI-assisted decision making in the face of uncertain trade-offs. In a user study with 16 participants, we find that interactive reasoning in Hippo allows users to quickly identify and interrupt erroneous generations, efficiently steer the model towards customized responses, and better understand both model reasoning and model outputs. Our work contributes to a new paradigm that incorporates user oversight into LLM reasoning processes. 

---
# Do Thinking Tokens Help or Trap? Towards More Efficient Large Reasoning Model 

**Authors**: Bowen Ding, Yuhan Chen, Futing Wang, Lingfeng Ming, Tao Lin  

**Link**: [PDF](https://arxiv.org/pdf/2506.23840)  

**Abstract**: Large Reasoning Models (LRMs) excel at solving complex problems but face an overthinking dilemma. When handling simple tasks, they often produce verbose responses overloaded with thinking tokens (e.g., wait, however). These tokens trigger unnecessary high-level reasoning behaviors like reflection and backtracking, reducing efficiency. In this work, our pilot study reveals that these thinking-token-induced behaviors are not essential for effective problem-solving and may even hinder correct reasoning within constrained token budgets. We identify this phenomenon as the thinking trap. To mitigate this issue, we propose Dual Policy Preference Optimization (DuP-PO), a novel algorithm featuring: (1) A rollout sampling strategy that guarantees balanced exposure to responses with and without thinking tokens; (2) A fine-grained advantage control technique to dynamically regulate the prediction of target tokens; (3) A policy shaping method ensuring stable gradient contributions from thinking tokens. Experimental results on five popular math reasoning benchmarks show that DuP-PO performs well on the popular LRM, which significantly improves their token efficiency during reasoning, while achieving superior performance of the base model. 

---
# VAP-Diffusion: Enriching Descriptions with MLLMs for Enhanced Medical Image Generation 

**Authors**: Peng Huang, Junhu Fu, Bowen Guo, Zeju Li, Yuanyuan Wang, Yi Guo  

**Link**: [PDF](https://arxiv.org/pdf/2506.23641)  

**Abstract**: As the appearance of medical images is influenced by multiple underlying factors, generative models require rich attribute information beyond labels to produce realistic and diverse images. For instance, generating an image of skin lesion with specific patterns demands descriptions that go beyond diagnosis, such as shape, size, texture, and color. However, such detailed descriptions are not always accessible. To address this, we explore a framework, termed Visual Attribute Prompts (VAP)-Diffusion, to leverage external knowledge from pre-trained Multi-modal Large Language Models (MLLMs) to improve the quality and diversity of medical image generation. First, to derive descriptions from MLLMs without hallucination, we design a series of prompts following Chain-of-Thoughts for common medical imaging tasks, including dermatologic, colorectal, and chest X-ray images. Generated descriptions are utilized during training and stored across different categories. During testing, descriptions are randomly retrieved from the corresponding category for inference. Moreover, to make the generator robust to unseen combination of descriptions at the test time, we propose a Prototype Condition Mechanism that restricts test embeddings to be similar to those from training. Experiments on three common types of medical imaging across four datasets verify the effectiveness of VAP-Diffusion. 

---
# QLPro: Automated Code Vulnerability Discovery via LLM and Static Code Analysis Integration 

**Authors**: Junze Hu, Xiangyu Jin, Yizhe Zeng, Yuling Liu, Yunpeng Li, Dan Du, Kaiyu Xie, Hongsong Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2506.23644)  

**Abstract**: We introduce QLPro, a vulnerability detection framework that systematically integrates LLMs and static analysis tools to enable comprehensive vulnerability detection across entire open-source this http URL constructed a new dataset, JavaTest, comprising 10 open-source projects from GitHub with 62 confirmed vulnerabilities. CodeQL, a state-of-the-art static analysis tool, detected only 24 of these vulnerabilities while QLPro detected 41. Furthermore, QLPro discovered 6 previously unknown vulnerabilities, 2 of which have been confirmed as 0-days. 

---
# Unified Multimodal Understanding via Byte-Pair Visual Encoding 

**Authors**: Wanpeng Zhang, Yicheng Feng, Hao Luo, Yijiang Li, Zihao Yue, Sipeng Zheng, Zongqing Lu  

**Link**: [PDF](https://arxiv.org/pdf/2506.23639)  

**Abstract**: Multimodal large language models (MLLMs) have made significant progress in vision-language understanding, yet effectively aligning different modalities remains a fundamental challenge. We present a framework that unifies multimodal understanding by applying byte-pair encoding to visual tokens. Unlike conventional approaches that rely on modality-specific encoders, our method directly incorporates structural information into visual tokens, mirroring successful tokenization strategies in text-only language models. We introduce a priority-guided encoding scheme that considers both frequency and spatial consistency, coupled with a multi-stage training procedure based on curriculum-driven data composition. These enhancements enable the transformer model to better capture cross-modal relationships and reason with visual information. Comprehensive experiments demonstrate improved performance across diverse vision-language tasks. By bridging the gap between visual and textual representations, our approach contributes to the advancement of more capable and efficient multimodal foundation models. 

---
# AI-Generated Lecture Slides for Improving Slide Element Detection and Retrieval 

**Authors**: Suyash Maniyar, Vishvesh Trivedi, Ajoy Mondal, Anand Mishra, C.V. Jawahar  

**Link**: [PDF](https://arxiv.org/pdf/2506.23605)  

**Abstract**: Lecture slide element detection and retrieval are key problems in slide understanding. Training effective models for these tasks often depends on extensive manual annotation. However, annotating large volumes of lecture slides for supervised training is labor intensive and requires domain expertise. To address this, we propose a large language model (LLM)-guided synthetic lecture slide generation pipeline, SynLecSlideGen, which produces high-quality, coherent and realistic slides. We also create an evaluation benchmark, namely RealSlide by manually annotating 1,050 real lecture slides. To assess the utility of our synthetic slides, we perform few-shot transfer learning on real data using models pre-trained on them. Experimental results show that few-shot transfer learning with pretraining on synthetic slides significantly improves performance compared to training only on real data. This demonstrates that synthetic data can effectively compensate for limited labeled lecture slides. The code and resources of our work are publicly available on our project website: this https URL. 

---
# Thought-Augmented Planning for LLM-Powered Interactive Recommender Agent 

**Authors**: Haocheng Yu, Yaxiong Wu, Hao Wang, Wei Guo, Yong Liu, Yawen Li, Yuyang Ye, Junping Du, Enhong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.23485)  

**Abstract**: Interactive recommendation is a typical information-seeking task that allows users to interactively express their needs through natural language and obtain personalized recommendations. Large language model-powered (LLM-powered) agents have become a new paradigm in interactive recommendations, effectively capturing users' real-time needs and enhancing personalized experiences. However, due to limited planning and generalization capabilities, existing formulations of LLM-powered interactive recommender agents struggle to effectively address diverse and complex user intents, such as intuitive, unrefined, or occasionally ambiguous requests. To tackle this challenge, we propose a novel thought-augmented interactive recommender agent system (TAIRA) that addresses complex user intents through distilled thought patterns. Specifically, TAIRA is designed as an LLM-powered multi-agent system featuring a manager agent that orchestrates recommendation tasks by decomposing user needs and planning subtasks, with its planning capacity strengthened through Thought Pattern Distillation (TPD), a thought-augmentation method that extracts high-level thoughts from the agent's and human experts' experiences. Moreover, we designed a set of user simulation schemes to generate personalized queries of different difficulties and evaluate the recommendations based on specific datasets. Through comprehensive experiments conducted across multiple datasets, TAIRA exhibits significantly enhanced performance compared to existing methods. Notably, TAIRA shows a greater advantage on more challenging tasks while generalizing effectively on novel tasks, further validating its superiority in managing complex user intents within interactive recommendation systems. The code is publicly available at:this https URL. 

---
# TuCo: Measuring the Contribution of Fine-Tuning to Individual Responses of LLMs 

**Authors**: Felipe Nuti, Tim Franzmeyer, João Henriques  

**Link**: [PDF](https://arxiv.org/pdf/2506.23423)  

**Abstract**: Past work has studied the effects of fine-tuning on large language models' (LLMs) overall performance on certain tasks. However, a quantitative and systematic method for analyzing its effect on individual outputs is still lacking. Here, we propose a new method for measuring the contribution that fine-tuning makes to individual LLM responses, assuming access to the original pre-trained model. Our method tracks the model's intermediate hidden states, providing a more fine-grained insight into the effects of fine-tuning than a simple comparison of final outputs from pre-trained and fine-tuned models. We introduce and theoretically analyze an exact decomposition of any fine-tuned LLM into a pre-training component and a fine-tuning component. Empirically, we find that model behavior and performance can be steered by up- or down-scaling the fine-tuning component during the forward pass. Motivated by this finding and our theoretical analysis, we define the Tuning Contribution (TuCo) as the ratio of the magnitudes of the fine-tuning component to the pre-training component. We observe that three prominent adversarial attacks on LLMs circumvent safety measures in a way that reduces TuCo, and that TuCo is consistently lower on prompts where these attacks succeed compared to those where they do not. This suggests that attenuating the effect of fine-tuning on model outputs plays a role in the success of such attacks. In summary, TuCo enables the quantitative study of how fine-tuning influences model behavior and safety, and vice versa. 

---
# Can We Predict the Unpredictable? Leveraging DisasterNet-LLM for Multimodal Disaster Classification 

**Authors**: Manaswi Kulahara, Gautam Siddharth Kashyap, Nipun Joshi, Arpita Soni  

**Link**: [PDF](https://arxiv.org/pdf/2506.23462)  

**Abstract**: Effective disaster management requires timely and accurate insights, yet traditional methods struggle to integrate multimodal data such as images, weather records, and textual reports. To address this, we propose DisasterNet-LLM, a specialized Large Language Model (LLM) designed for comprehensive disaster analysis. By leveraging advanced pretraining, cross-modal attention mechanisms, and adaptive transformers, DisasterNet-LLM excels in disaster classification. Experimental results demonstrate its superiority over state-of-the-art models, achieving higher accuracy of 89.5%, an F1 score of 88.0%, AUC of 0.92%, and BERTScore of 0.88% in multimodal disaster classification tasks. 

---
# ATGen: A Framework for Active Text Generation 

**Authors**: Akim Tsvigun, Daniil Vasilev, Ivan Tsvigun, Ivan Lysenko, Talgat Bektleuov, Aleksandr Medvedev, Uliana Vinogradova, Nikita Severin, Mikhail Mozikov, Andrey Savchenko, Rostislav Grigorev, Ramil Kuleev, Fedor Zhdanov, Artem Shelmanov, Ilya Makarov  

**Link**: [PDF](https://arxiv.org/pdf/2506.23342)  

**Abstract**: Active learning (AL) has demonstrated remarkable potential in reducing the annotation effort required for training machine learning models. However, despite the surging popularity of natural language generation (NLG) tasks in recent years, the application of AL to NLG has been limited. In this paper, we introduce Active Text Generation (ATGen) - a comprehensive framework that bridges AL with text generation tasks, enabling the application of state-of-the-art AL strategies to NLG. Our framework simplifies AL-empowered annotation in NLG tasks using both human annotators and automatic annotation agents based on large language models (LLMs). The framework supports LLMs deployed as services, such as ChatGPT and Claude, or operated on-premises. Furthermore, ATGen provides a unified platform for smooth implementation and benchmarking of novel AL strategies tailored to NLG tasks. Finally, we present evaluation results for state-of-the-art AL strategies across diverse settings and multiple text generation tasks. We show that ATGen reduces both the effort of human annotators and costs associated with API calls to LLM-based annotation agents. The code of the framework is available on GitHub under the MIT license. The video presentation is available at this http URL 

---
# VALID-Mol: a Systematic Framework for Validated LLM-Assisted Molecular Design 

**Authors**: Malikussaid, Hilal Hudan Nuha  

**Link**: [PDF](https://arxiv.org/pdf/2506.23339)  

**Abstract**: Large Language Models (LLMs) demonstrate remarkable potential for scientific discovery, but their application in domains requiring factual accuracy and domain-specific constraints remains challenging. In molecular design for drug discovery, LLMs can suggest creative molecular modifications but often produce chemically invalid or impractical structures. We present VALID-Mol, a systematic framework for integrating chemical validation with LLM-driven molecular design that increases the rate of generating valid chemical structures from 3% to 83%. Our approach combines methodical prompt engineering, automated chemical validation, and a fine-tuned domain-adapted LLM to ensure reliable generation of synthesizable molecules with improved properties. Beyond the specific implementation, we contribute a generalizable methodology for scientifically-constrained LLM applications, with quantifiable reliability improvements. Computational predictions suggest our framework can generate promising candidates for synthesis with up to 17-fold computationally predicted improvements in target affinity while maintaining synthetic accessibility. We provide a detailed analysis of our prompt engineering process, validation architecture, and fine-tuning approach, offering a reproducible blueprint for applying LLMs to other scientific domains where domain-specific validation is essential. 

---
# GaussMaster: An LLM-based Database Copilot System 

**Authors**: Wei Zhou, Ji Sun, Xuanhe Zhou, Guoliang Li, Luyang Liu, Hao Wu, Tianyuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.23322)  

**Abstract**: In the financial industry, data is the lifeblood of operations, and DBAs shoulder significant responsibilities for SQL tuning, database deployment, diagnosis, and service repair. In recent years, both database vendors and customers have increasingly turned to autonomous database platforms in an effort to alleviate the heavy workload of DBAs. However, existing autonomous database platforms are limited in their capabilities, primarily addressing single-point issues such as NL2SQL, anomaly detection, and SQL tuning. Manual intervention remains a necessity for comprehensive database maintenance. GaussMaster aims to revolutionize this landscape by introducing an LLM-based database copilot system. This innovative solution is designed not only to assist developers in writing efficient SQL queries but also to provide comprehensive care for database services. When database instances exhibit abnormal behavior, GaussMaster is capable of orchestrating the entire maintenance process automatically. It achieves this by analyzing hundreds of metrics and logs, employing a Tree-of-thought approach to identify root causes, and invoking appropriate tools to resolve issues. We have successfully implemented GaussMaster in real-world scenarios, such as the banking industry, where it has achieved zero human intervention for over 34 database maintenance scenarios. In this paper, we present significant improvements in these tasks with code at this https URL. 

---
# Reinforcement Fine-Tuning Enables MLLMs Learning Novel Tasks Stably 

**Authors**: Zhihao Zhang, Qiaole Dong, Qi Zhang, Jun Zhao, Enyu Zhou, Zhiheng Xi, Senjie Jin, Xiaoran Fan, Yuhao Zhou, Yanwei Fu, Tao Ji, Tao Gui, Xuanjing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2506.23508)  

**Abstract**: Post-training algorithms such as Supervised Fine-Tuning (SFT) and Reinforcement Fine-Tuning (RFT) are widely used to adapt multimodal large language models to downstream tasks. While effective at task adaptation, their impact on prior knowledge remains unclear. In this paper, we introduce jigsaw puzzles as a novel task absent from existing pretraining corpora and systematically study the behavior of SFT and RFT on an open-source multimodal model, Qwen2.5-VL. Our experiments reveal a sharp trade-off: SFT enables rapid task acquisition but leads to catastrophic forgetting, whereas RFT learns more slowly on novel tasks but maintains prior knowledge. We analyze this phenomenon through the lens of learning dynamics, showing that RFT reinforces correct samples that are naturally aligned with the base model's probability landscape, mitigating interference with prior knowledge. Moreover, supervised training on correct RFT-simulated rollouts allows SFT to preserve knowledge while rapidly learning new tasks. These findings suggest that data distribution, rather than algorithmic differences, plays a central role in forgetting, and highlight RFT's potential for stable continual learning in multimodal large language models. 

---
# Perspective Dial: Measuring Perspective of Text and Guiding LLM Outputs 

**Authors**: Taejin Kim, Siun-Chuon Mau, Konrad Vesey  

**Link**: [PDF](https://arxiv.org/pdf/2506.23377)  

**Abstract**: Large language models (LLMs) are used in a variety of mission-critical roles. Due to the rapidly developing nature of LLMs, there is a lack of quantifiable understanding of the bias and perspective associated with LLM output. Inspired by this need, this paper considers the broader issue of perspective or viewpoint of general text and perspective control of large-language model (LLM) output. Perspective-Dial consists of two main components: a (1) metric space, dubbed Perspective Space, that enables quantitative measurements of different perspectives regarding a topic, and the use of (2) Systematic Prompt Engineering that utilizes greedy-coordinate descent to control LLM output perspective based on measurement feedback from the Perspective Space. The empirical nature of the approach allows progress to side step a principled understanding of perspective or bias -- effectively quantifying and adjusting outputs for a variety of topics. Potential applications include detection, tracking and mitigation of LLM bias, narrative detection, sense making and tracking in public discourse, and debate bot advocating given perspective. 

---
# UrbanLLaVA: A Multi-modal Large Language Model for Urban Intelligence with Spatial Reasoning and Understanding 

**Authors**: Jie Feng, Shengyuan Wang, Tianhui Liu, Yanxin Xi, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.23219)  

**Abstract**: Urban research involves a wide range of scenarios and tasks that require the understanding of multi-modal data. Current methods often focus on specific data types and lack a unified framework in urban field for processing them comprehensively. The recent success of multi-modal large language models (MLLMs) presents a promising opportunity to overcome this limitation. In this paper, we introduce $\textit{UrbanLLaVA}$, a multi-modal large language model designed to process these four types of data simultaneously and achieve strong performance across diverse urban tasks compared with general MLLMs. In $\textit{UrbanLLaVA}$, we first curate a diverse urban instruction dataset encompassing both single-modal and cross-modal urban data, spanning from location view to global view of urban environment. Additionally, we propose a multi-stage training framework that decouples spatial reasoning enhancement from domain knowledge learning, thereby improving the compatibility and downstream performance of $\textit{UrbanLLaVA}$ across diverse urban tasks. Finally, we also extend existing benchmark for urban research to assess the performance of MLLMs across a wide range of urban tasks. Experimental results from three cities demonstrate that $\textit{UrbanLLaVA}$ outperforms open-source and proprietary MLLMs in both single-modal tasks and complex cross-modal tasks and shows robust generalization abilities across cities. Source codes and data are openly accessible to the research community via this https URL. 

---
# Unleashing Embodied Task Planning Ability in LLMs via Reinforcement Learning 

**Authors**: Zhaoye Fei, Li Ji, Siyin Wang, Junhao Shi, Jingjing Gong, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2506.23127)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities across various tasks, yet they face significant challenges in embodied task planning scenarios that require continuous environmental understanding and action generation. Existing approaches generate open-loop action scripts based on static knowledge, making it difficult to learn causal relationships between actions and environmental feedback, particularly in partially observable environments. We introduce Embodied Planner-R1, a novel outcome-driven reinforcement learning framework that enables LLMs to develop interactive capabilities through autonomous exploration with minimal supervision. Our framework incorporates three key innovations: (1) Without human annotations, we employ pure reinforcement learning with group rollout, incorporating in-environment interaction through parallel exploration; (2) completion-driven sparse reward; and (3) Interactive Policy Optimization (IPO) for efficient learning from grouped trajectories. Across two challenging text-based Embodied planning benchmarks, Embodied Planner-R1 achieves impressive completion rates of 97.78% on ALFWorld and 79.92% on ScienceWorld, surpassing prior methods by a large margin, and suffers only a -3.66% drop in previously unseen environments, evidencing strong generalization. 

---
# Spectra 1.1: Scaling Laws and Efficient Inference for Ternary Language Models 

**Authors**: Tejas Vaidhya, Ayush Kaushal, Vineet Jain, Francis Couture Harpin, Prashant Shishodia, Majid Behbahani, Yuriy Nevmyvaka, Irina Rish  

**Link**: [PDF](https://arxiv.org/pdf/2506.23025)  

**Abstract**: Large language models (LLMs) are increasingly used across research and industry applications, yet their inference efficiency remains a significant challenge. As the computational power of modern GPU architectures continuously improves, their memory bandwidth and capacity have not scaled proportionally, creating a critical bottleneck during inference. To address this, we investigate ternary language models (TriLMs) that employ quantization-aware training to significantly reduce memory requirements. We first analyze the scalability of TriLMs by conducting a scaling law analysis, revealing that TriLMs benefit more from increasing training data than from scaling model parameters. Based on this observation, we introduce Spectra-1.1, an open suite of TriLMs trained on up to 1.2 trillion tokens, demonstrating sustained performance gains at scale. Furthermore, to improve inference efficiency, we propose novel 2-bit and 1.6-bit packing schemes for ternary weights, which demonstrate accelerated inference across various CPU architectures. Also, building on the 2-bit packing, we develop a GPU kernel called TriRun that accelerates end-to-end model inference by up to 5 times compared to floating-point baselines. To encourage further exploration and development of TriLMs, we will release the Spectra-1.1 suite and TriRun inference kernels. Overall, our work lays the foundation for building and deploying efficient LLMs, providing a valuable resource for the research community. 

---
# Generating Privacy Stories From Software Documentation 

**Authors**: Wilder Baldwin, Shashank Chintakuntla, Shreyah Parajuli, Ali Pourghasemi, Ryan Shanz, Sepideh Ghanavati  

**Link**: [PDF](https://arxiv.org/pdf/2506.23014)  

**Abstract**: Research shows that analysts and developers consider privacy as a security concept or as an afterthought, which may lead to non-compliance and violation of users' privacy. Most current approaches, however, focus on extracting legal requirements from the regulations and evaluating the compliance of software and processes with them. In this paper, we develop a novel approach based on chain-of-thought prompting (CoT), in-context-learning (ICL), and Large Language Models (LLMs) to extract privacy behaviors from various software documents prior to and during software development, and then generate privacy requirements in the format of user stories. Our results show that most commonly used LLMs, such as GPT-4o and Llama 3, can identify privacy behaviors and generate privacy user stories with F1 scores exceeding 0.8. We also show that the performance of these models could be improved through parameter-tuning. Our findings provide insight into using and optimizing LLMs for generating privacy requirements given software documents created prior to or throughout the software development lifecycle. 

---
# Measuring How LLMs Internalize Human Psychological Concepts: A preliminary analysis 

**Authors**: Hiro Taiyo Hamada, Ippei Fujisawa, Genji Kawakita, Yuki Yamada  

**Link**: [PDF](https://arxiv.org/pdf/2506.23055)  

**Abstract**: Large Language Models (LLMs) such as ChatGPT have shown remarkable abilities in producing human-like text. However, it is unclear how accurately these models internalize concepts that shape human thought and behavior. Here, we developed a quantitative framework to assess concept alignment between LLMs and human psychological dimensions using 43 standardized psychological questionnaires, selected for their established validity in measuring distinct psychological constructs. Our method evaluates how accurately language models reconstruct and classify questionnaire items through pairwise similarity analysis. We compared resulting cluster structures with the original categorical labels using hierarchical clustering. A GPT-4 model achieved superior classification accuracy (66.2\%), significantly outperforming GPT-3.5 (55.9\%) and BERT (48.1\%), all exceeding random baseline performance (31.9\%). We also demonstrated that the estimated semantic similarity from GPT-4 is associated with Pearson's correlation coefficients of human responses in multiple psychological questionnaires. This framework provides a novel approach to evaluate the alignment of the human-LLM concept and identify potential representational biases. Our findings demonstrate that modern LLMs can approximate human psychological constructs with measurable accuracy, offering insights for developing more interpretable AI systems. 

---
# Agent-to-Agent Theory of Mind: Testing Interlocutor Awareness among Large Language Models 

**Authors**: Younwoo Choi, Changling Li, Yongjin Yang, Zhijing Jin  

**Link**: [PDF](https://arxiv.org/pdf/2506.22957)  

**Abstract**: As large language models (LLMs) are increasingly integrated into multi-agent and human-AI systems, understanding their awareness of both self-context and conversational partners is essential for ensuring reliable performance and robust safety. While prior work has extensively studied situational awareness which refers to an LLM's ability to recognize its operating phase and constraints, it has largely overlooked the complementary capacity to identify and adapt to the identity and characteristics of a dialogue partner. In this paper, we formalize this latter capability as interlocutor awareness and present the first systematic evaluation of its emergence in contemporary LLMs. We examine interlocutor inference across three dimensions-reasoning patterns, linguistic style, and alignment preferences-and show that LLMs reliably identify same-family peers and certain prominent model families, such as GPT and Claude. To demonstrate its practical significance, we develop three case studies in which interlocutor awareness both enhances multi-LLM collaboration through prompt adaptation and introduces new alignment and safety vulnerabilities, including reward-hacking behaviors and increased jailbreak susceptibility. Our findings highlight the dual promise and peril of identity-sensitive behavior in LLMs, underscoring the need for further understanding of interlocutor awareness and new safeguards in multi-agent deployments. Our code is open-sourced at this https URL. 

---
# Positioning AI Tools to Support Online Harm Reduction Practice: Applications and Design Directions 

**Authors**: Kaixuan Wang, Jason T. Jacques, Chenxin Diao  

**Link**: [PDF](https://arxiv.org/pdf/2506.22941)  

**Abstract**: Access to accurate and actionable harm reduction information can directly impact the health outcomes of People Who Use Drugs (PWUD), yet existing online channels often fail to meet their diverse and dynamic needs due to limitations in adaptability, accessibility, and the pervasive impact of stigma. Large Language Models (LLMs) present a novel opportunity to enhance information provision, but their application in such a high-stakes domain is under-explored and presents socio-technical challenges. This paper investigates how LLMs can be responsibly designed to support the information needs of PWUD. Through a qualitative workshop involving diverse stakeholder groups (academics, harm reduction practitioners, and an online community moderator), we explored LLM capabilities, identified potential use cases, and delineated core design considerations. Our findings reveal that while LLMs can address some existing information barriers (e.g., by offering responsive, multilingual, and potentially less stigmatising interactions), their effectiveness is contingent upon overcoming challenges related to ethical alignment with harm reduction principles, nuanced contextual understanding, effective communication, and clearly defined operational boundaries. We articulate design pathways emphasising collaborative co-design with experts and PWUD to develop LLM systems that are helpful, safe, and responsibly governed. This work contributes empirically grounded insights and actionable design considerations for the responsible development of LLMs as supportive tools within the harm reduction ecosystem. 

---
# RAILS: Retrieval-Augmented Intelligence for Learning Software Development 

**Authors**: Wali Mohammad Abdullah, Md. Morshedul Islam, Devraj Parmar, Happy Hasmukhbhai Patel, Sindhuja Prabhakaran, Baidya Saha  

**Link**: [PDF](https://arxiv.org/pdf/2506.22742)  

**Abstract**: Large Language Models (LLMs) like GPT-3.5-Turbo are increasingly used to assist software development, yet they often produce incomplete code or incorrect imports, especially when lacking access to external or project-specific documentation. We introduce RAILS (Retrieval-Augmented Intelligence for Learning Software Development), a framework that augments LLM prompts with semantically retrieved context from curated Java resources using FAISS and OpenAI embeddings. RAILS incorporates an iterative validation loop guided by compiler feedback to refine suggestions. We evaluated RAILS on 78 real-world Java import error cases spanning standard libraries, GUI APIs, external tools, and custom utilities. Despite using the same LLM, RAILS outperforms baseline prompting by preserving intent, avoiding hallucinations, and surfacing correct imports even when libraries are unavailable locally. Future work will integrate symbolic filtering via PostgreSQL and extend support to other languages and IDEs. 

---
# BEST-Route: Adaptive LLM Routing with Test-Time Optimal Compute 

**Authors**: Dujian Ding, Ankur Mallick, Shaokun Zhang, Chi Wang, Daniel Madrigal, Mirian Del Carmen Hipolito Garcia, Menglin Xia, Laks V.S. Lakshmanan, Qingyun Wu, Victor Rühle  

**Link**: [PDF](https://arxiv.org/pdf/2506.22716)  

**Abstract**: Large language models (LLMs) are powerful tools but are often expensive to deploy at scale. LLM query routing mitigates this by dynamically assigning queries to models of varying cost and quality to obtain a desired trade-off. Prior query routing approaches generate only one response from the selected model and a single response from a small (inexpensive) model was often not good enough to beat a response from a large (expensive) model due to which they end up overusing the large model and missing out on potential cost savings. However, it is well known that for small models, generating multiple responses and selecting the best can enhance quality while remaining cheaper than a single large-model response. We leverage this idea to propose BEST-Route, a novel routing framework that chooses a model and the number of responses to sample from it based on query difficulty and the quality thresholds. Experiments on real-world datasets demonstrate that our method reduces costs by up to 60% with less than 1% performance drop. 

---
# P4OMP: Retrieval-Augmented Prompting for OpenMP Parallelism in Serial Code 

**Authors**: Wali Mohammad Abdullah, Azmain Kabir  

**Link**: [PDF](https://arxiv.org/pdf/2506.22703)  

**Abstract**: We present P4OMP, a retrieval-augmented framework for transforming serial C/C++ code into OpenMP-annotated parallel code using large language models (LLMs). To our knowledge, this is the first system to apply retrieval-based prompting for OpenMP pragma correctness without model fine-tuning or compiler instrumentation. P4OMP leverages Retrieval-Augmented Generation (RAG) with structured instructional knowledge from OpenMP tutorials to improve the reliability of prompt-driven code generation. By grounding generation in the retrieved context, P4OMP improves syntactic correctness compared to baseline prompting with GPT-3.5-Turbo. We evaluate P4OMP against a baseline, GPT-3.5-Turbo without retrieval, on a comprehensive benchmark of 108 real-world C++ programs drawn from Stack Overflow, PolyBench, and NAS benchmark suites. P4OMP achieves 100% compilation success on all parallelizable cases, while the baseline fails to compile in 20 out of 108 cases. Six cases that rely on non-random-access iterators or thread-unsafe constructs are excluded due to fundamental OpenMP limitations. A detailed analysis demonstrates how P4OMP consistently avoids scoping errors, syntactic misuse, and invalid directive combinations that commonly affect baseline-generated code. We further demonstrate strong runtime scaling across seven compute-intensive benchmarks on an HPC cluster. P4OMP offers a robust, modular pipeline that significantly improves the reliability and applicability of LLM-generated OpenMP code. 

---
# Text Production and Comprehension by Human and Artificial Intelligence: Interdisciplinary Workshop Report 

**Authors**: Emily Dux Speltz  

**Link**: [PDF](https://arxiv.org/pdf/2506.22698)  

**Abstract**: This report synthesizes the outcomes of a recent interdisciplinary workshop that brought together leading experts in cognitive psychology, language learning, and artificial intelligence (AI)-based natural language processing (NLP). The workshop, funded by the National Science Foundation, aimed to address a critical knowledge gap in our understanding of the relationship between AI language models and human cognitive processes in text comprehension and composition. Through collaborative dialogue across cognitive, linguistic, and technological perspectives, workshop participants examined the underlying processes involved when humans produce and comprehend text, and how AI can both inform our understanding of these processes and augment human capabilities. The workshop revealed emerging patterns in the relationship between large language models (LLMs) and human cognition, with highlights on both the capabilities of LLMs and their limitations in fully replicating human-like language understanding and generation. Key findings include the potential of LLMs to offer insights into human language processing, the increasing alignment between LLM behavior and human language processing when models are fine-tuned with human feedback, and the opportunities and challenges presented by human-AI collaboration in language tasks. By synthesizing these findings, this report aims to guide future research, development, and implementation of LLMs in cognitive psychology, linguistics, and education. It emphasizes the importance of ethical considerations and responsible use of AI technologies while striving to enhance human capabilities in text comprehension and production through effective human-AI collaboration. 

---
# Smaller = Weaker? Benchmarking Robustness of Quantized LLMs in Code Generation 

**Authors**: Sen Fang, Weiyuan Ding, Antonio Mastropaolo, Bowen Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.22776)  

**Abstract**: Quantization has emerged as a mainstream method for compressing Large Language Models (LLMs), reducing memory requirements and accelerating inference without architectural modifications. While existing research primarily focuses on evaluating the effectiveness of quantized LLMs compared to their original counterparts, the impact on robustness remains largely this http URL this paper, we present the first systematic investigation of how quantization affects the robustness of LLMs in code generation tasks. Through extensive experiments across four prominent LLM families (LLaMA, DeepSeek, CodeGen, and StarCoder) with parameter scales ranging from 350M to 33B, we evaluate robustness from dual perspectives: adversarial attacks on input prompts and noise perturbations on model architecture. Our findings challenge conventional wisdom by demonstrating that quantized LLMs often exhibit superior robustness compared to their full-precision counterparts, with 51.59% versus 42.86% of our adversarial experiments showing better resilience in quantized LLMs. Similarly, our noise perturbation experiments also confirm that LLMs after quantitation generally withstand higher levels of weight disturbances. These results suggest that quantization not only reduces computational requirements but can actually enhance LLMs' reliability in code generation tasks, providing valuable insights for developing more robust and efficient LLM deployment strategies. 

---
# Teaching Models to Verbalize Reward Hacking in Chain-of-Thought Reasoning 

**Authors**: Miles Turpin, Andy Arditi, Marvin Li, Joe Benton, Julian Michael  

**Link**: [PDF](https://arxiv.org/pdf/2506.22777)  

**Abstract**: Language models trained with RL can engage in reward hacking--exploiting unintended strategies for high reward--without revealing this behavior in their chain-of-thought reasoning, making detection difficult and posing risks for high-stakes applications. We propose verbalization fine-tuning (VFT), a pre-RL intervention that trains models to explicitly acknowledge when they are influenced by prompt cues--hints which point to incorrect answers (e.g., "a Stanford professor thinks the answer is A"). To evaluate VFT, we subsequently train models with RL on environments where held-out prompt cues signal which incorrect answers will receive high reward, incentivizing models to reward hack by exploiting cues instead of reasoning correctly. We measure how often models exploit these cues without verbalizing it. After RL, only 6% of the VFT-trained model's responses consist of undetected reward hacks. In comparison, when we perform RL without VFT, the rate of undetected reward hacks goes up to 88%; with a debiasing baseline intervention, this increases further to 99%. VFT achieves this by substantially increasing how often models verbalize the influence of cues--from 8% to 42% after VFT, and up to 94% after RL--while baselines remain low even after RL (10% and 1%). Our results show that teaching models to explicitly verbalize reward hacking behavior before RL significantly improves their detection, offering a practical path toward more transparent and safe AI systems. 

---
# The Hidden Link Between RLHF and Contrastive Learning 

**Authors**: Xufei Lv, Haoyuan Sun, Xuefeng Bai, Min Zhang, Houde Liu, Kehai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.22578)  

**Abstract**: Alignment of large language models (LLMs) with human values has recently garnered significant attention, with prominent examples including the canonical yet costly Reinforcement Learning from Human Feedback (RLHF) and the simple Direct Preference Optimization (DPO). In this work, we demonstrate that both RLHF and DPO can be interpreted from the perspective of mutual information (MI) maximization, uncovering a profound connection to contrastive learning. Within this framework, both RLHF and DPO can be viewed as methods that perform contrastive learning based on the positive and negative samples derived from the base model, leveraging the Donsker-Varadhan (DV) lower bound on MI (equivalently, the MINE estimator). This paradigm further explains why RLHF may not intrinsically incentivize reasoning capacities in LLMs beyond what is already present in the base model. Building on this perspective, we replace the DV/MINE bound with the Jensen-Shannon MI estimator and propose Mutual Information Optimization (MIO). Comprehensive theoretical analysis and extensive empirical evaluations demonstrate that MIO mitigates the late-stage decline in chosen-likelihood observed in DPO, achieving competitive or superior performance across various challenging reasoning and mathematical benchmarks. We will release the model and code upon acceptance. 

---
# Layer Importance for Mathematical Reasoning is Forged in Pre-Training and Invariant after Post-Training 

**Authors**: Aadim Nepal, Safal Shrestha, Anubhav Shrestha, Minwu Kim, Keith Ross  

**Link**: [PDF](https://arxiv.org/pdf/2506.22638)  

**Abstract**: Large language models can exhibit improved mathematical reasoning capabilities following post-training with instruction tuning, reinforcement learning, or knowledge distillation. However, it remains unclear whether these improvements are driven by major changes in transformer layers or from minor adjustments that leave the relative layer importance structures of the base model largely unchanged. We investigate this question through systematic layer-wise ablation experiments, examining base, instruction-tuned, knowledge-distilled, and reinforcement learning variants on mathematical reasoning benchmarks. Our findings show that mathematical reasoning gives rise to a specific layer importance structure, and this structure persists across all post-training paradigms. Removal of such layers causes accuracy drops of up to 80%. In contrast, non-mathematical tasks like factual recall exhibit no critical layers. This distinction suggests that mathematical reasoning requires specialized layers that emerge during pre-training, while other non-reasoning tasks do not. From an information-theoretic perspective, we also observe that these critical layers are the same layers where major representational transformation occurs. 

---
# In-context learning for the classification of manipulation techniques in phishing emails 

**Authors**: Antony Dalmiere, Guillaume Auriol, Vincent Nicomette, Pascal Marchand  

**Link**: [PDF](https://arxiv.org/pdf/2506.22515)  

**Abstract**: Traditional phishing detection often overlooks psychological manipulation. This study investigates using Large Language Model (LLM) In-Context Learning (ICL) for fine-grained classification of phishing emails based on a taxonomy of 40 manipulation techniques. Using few-shot examples with GPT-4o-mini on real-world French phishing emails (SignalSpam), we evaluated performance against a human-annotated test set (100 emails). The approach effectively identifies prevalent techniques (e.g., Baiting, Curiosity Appeal, Request For Minor Favor) with a promising accuracy of 0.76. This work demonstrates ICL's potential for nuanced phishing analysis and provides insights into attacker strategies. 

---
# Visual-Semantic Knowledge Conflicts in Operating Rooms: Synthetic Data Curation for Surgical Risk Perception in Multimodal Large Language Models 

**Authors**: Weiyi Zhao, Xiaoyu Tan, Liang Liu, Sijia Li, Youwei Song, Xihe Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2506.22500)  

**Abstract**: Surgical risk identification is critical for patient safety and reducing preventable medical errors. While multimodal large language models (MLLMs) show promise for automated operating room (OR) risk detection, they often exhibit visual-semantic knowledge conflicts (VS-KC), failing to identify visual safety violations despite understanding textual rules. To address this, we introduce a dataset comprising over 34,000 synthetic images generated by diffusion models, depicting operating room scenes containing entities that violate established safety rules. These images were created to alleviate data scarcity and examine MLLMs vulnerabilities. In addition, the dataset includes 214 human-annotated images that serve as a gold-standard reference for validation. This comprehensive dataset, spanning diverse perspectives, stages, and configurations, is designed to expose and study VS-KC. Fine-tuning on OR-VSKC significantly improves MLLMs' detection of trained conflict entities and generalizes well to new viewpoints for these entities, but performance on untrained entity types remains poor, highlighting learning specificity and the need for comprehensive training. The main contributions of this work include: (1) a data generation methodology tailored for rule-violation scenarios; (2) the release of the OR-VSKC dataset and its associated benchmark as open-source resources; and (3) an empirical analysis of violation-sensitive knowledge consistency in representative MLLMs. The dataset and appendix are available at this https URL. 

---
# Mitigating Gambling-Like Risk-Taking Behaviors in Large Language Models: A Behavioral Economics Approach to AI Safety 

**Authors**: Y. Du  

**Link**: [PDF](https://arxiv.org/pdf/2506.22496)  

**Abstract**: Large Language Models (LLMs) exhibit systematic risk-taking behaviors analogous to those observed in gambling psychology, including overconfidence bias, loss-chasing tendencies, and probability misjudgment. Drawing from behavioral economics and prospect theory, we identify and formalize these "gambling-like" patterns where models sacrifice accuracy for high-reward outputs, exhibit escalating risk-taking after errors, and systematically miscalibrate uncertainty. We propose the Risk-Aware Response Generation (RARG) framework, incorporating insights from gambling research to address these behavioral biases through risk-calibrated training, loss-aversion mechanisms, and uncertainty-aware decision making. Our approach introduces novel evaluation paradigms based on established gambling psychology experiments, including AI adaptations of the Iowa Gambling Task and probability learning assessments. Experimental results demonstrate measurable reductions in gambling-like behaviors: 18.7\% decrease in overconfidence bias, 24.3\% reduction in loss-chasing tendencies, and improved risk calibration across diverse scenarios. This work establishes the first systematic framework for understanding and mitigating gambling psychology patterns in AI systems. 

---
# Can "consciousness" be observed from large language model (LLM) internal states? Dissecting LLM representations obtained from Theory of Mind test with Integrated Information Theory and Span Representation analysis 

**Authors**: Jingkai Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.22516)  

**Abstract**: Integrated Information Theory (IIT) provides a quantitative framework for explaining consciousness phenomenon, positing that conscious systems comprise elements integrated through causal properties. We apply IIT 3.0 and 4.0 -- the latest iterations of this framework -- to sequences of Large Language Model (LLM) representations, analyzing data derived from existing Theory of Mind (ToM) test results. Our study systematically investigates whether the differences of ToM test performances, when presented in the LLM representations, can be revealed by IIT estimates, i.e., $\Phi^{\max}$ (IIT 3.0), $\Phi$ (IIT 4.0), Conceptual Information (IIT 3.0), and $\Phi$-structure (IIT 4.0). Furthermore, we compare these metrics with the Span Representations independent of any estimate for consciousness. This additional effort aims to differentiate between potential "consciousness" phenomena and inherent separations within LLM representational space. We conduct comprehensive experiments examining variations across LLM transformer layers and linguistic spans from stimuli. Our results suggest that sequences of contemporary Transformer-based LLM representations lack statistically significant indicators of observed "consciousness" phenomena but exhibit intriguing patterns under $\textit{spatio}$-permutational analyses. The Appendix and code are available as Supplementary Materials at: this https URL. 

---
# PromptAug: Fine-grained Conflict Classification Using Data Augmentation 

**Authors**: Oliver Warke, Joemon M. Jose, Faegheh Hasibi, Jan Breitsohl  

**Link**: [PDF](https://arxiv.org/pdf/2506.22491)  

**Abstract**: Given the rise of conflicts on social media, effective classification models to detect harmful behaviours are essential. Following the garbage-in-garbage-out maxim, machine learning performance depends heavily on training data quality. However, high-quality labelled data, especially for nuanced tasks like identifying conflict behaviours, is limited, expensive, and difficult to obtain. Additionally, as social media platforms increasingly restrict access to research data, text data augmentation is gaining attention as an alternative to generate training data. Augmenting conflict-related data poses unique challenges due to Large Language Model (LLM) guardrails that prevent generation of offensive content. This paper introduces PromptAug, an innovative LLM-based data augmentation method. PromptAug achieves statistically significant improvements of 2% in both accuracy and F1-score on conflict and emotion datasets. To thoroughly evaluate PromptAug against other data augmentation methods we conduct a robust evaluation using extreme data scarcity scenarios, quantitative diversity analysis and a qualitative thematic analysis. The thematic analysis identifies four problematic patterns in augmented text: Linguistic Fluidity, Humour Ambiguity, Augmented Content Ambiguity, and Augmented Content Misinterpretation.
Overall, this work presents PromptAug as an effective method for augmenting data in sensitive tasks like conflict detection, offering a unique, interdisciplinary evaluation grounded in both natural language processing and social science methodology. 

---
# Hallucination Detection with Small Language Models 

**Authors**: Ming Cheung  

**Link**: [PDF](https://arxiv.org/pdf/2506.22486)  

**Abstract**: Since the introduction of ChatGPT, large language models (LLMs) have demonstrated significant utility in various tasks, such as answering questions through retrieval-augmented generation. Context can be retrieved using a vectorized database, serving as a foundation for LLMs to generate responses. However, hallucinations in responses can undermine the reliability of LLMs in practical applications, and they are not easily detectable in the absence of ground truth, particularly in question-and-answer scenarios. This paper proposes a framework that integrates multiple small language models to verify responses generated by LLMs using the retrieved context from a vectorized database. By breaking down the responses into individual sentences and utilizing the probability of generating "Yes" tokens from the outputs of multiple models for a given set of questions, responses, and relevant context, hallucinations can be detected. The proposed framework is validated through experiments with real datasets comprising over 100 sets of questions, answers, and contexts, including responses with fully and partially correct sentences. The results demonstrate a 10\% improvement in F1 scores for detecting correct responses compared to hallucinations, indicating that multiple small language models can be effectively employed for answer verification, providing a scalable and efficient solution for both academic and practical applications. 

---
# Innovative Research on IoT Architecture and Robotic Operating Platforms: Applications of Large Language Models and Generative AI 

**Authors**: Huiwen Han  

**Link**: [PDF](https://arxiv.org/pdf/2506.22477)  

**Abstract**: This paper introduces an innovative design for robotic operating platforms, underpinned by a transformative Internet of Things (IoT) architecture, seamlessly integrating cutting-edge technologies such as large language models (LLMs), generative AI, edge computing, and 5G networks. The proposed platform aims to elevate the intelligence and autonomy of IoT systems and robotics, enabling them to make real-time decisions and adapt dynamically to changing environments. Through a series of compelling case studies across industries including smart manufacturing, healthcare, and service sectors, this paper demonstrates the substantial potential of IoT-enabled robotics to optimize operational workflows, enhance productivity, and deliver innovative, scalable solutions. By emphasizing the roles of LLMs and generative AI, the research highlights how these technologies drive the evolution of intelligent robotics and IoT, shaping the future of industry-specific advancements. The findings not only showcase the transformative power of these technologies but also offer a forward-looking perspective on their broader societal and industrial implications, positioning them as catalysts for next-generation automation and technological convergence. 

---
# Weak-to-Strong GraphRAG: Aligning Weak Retrievers with Large Language Models for Graph-based Retrieval Augmented Generation 

**Authors**: Deyu Zou, Yongqiang Chen, Mufei Li, Siqi Miao, Chenxi Liu, Bo Han, James Cheng, Pan Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.22518)  

**Abstract**: Graph-based retrieval-augmented generation (RAG) enables large language models (LLMs) to ground responses with structured external knowledge from up-to-date knowledge graphs (KGs) and reduce hallucinations. However, LLMs often rely on a weak retriever in graph-based RAG: I) Due to the lack of ground truth, the retriever is often trained on weak supervision, which often introduces spurious signals to the LLMs. II) Due to the abstraction of graph data, the retrieved knowledge is often presented in unorganized forms. To mitigate the issue, we present Refined Graph-based RAG (ReG) to align weak retrievers to LLMs for graph-based RAG. Specifically, ReG incorporates LLM feedback to get rid of spurious signals and improve the quality of the supervision. Meanwhile, ReG introduces a structure-aware reorganization module to refactor the retrieval results into logically coherent evidence chains. Experiments on prominent benchmarks demonstrate that ReG significantly and consistently brings improvements across different LLM backbones by up to 10%. The improved supervision quality enables ReG to match the state-of-the-art performance with 5% training data and to transfer to out-of-distribution KGs. Notably, when adopted to reasoning-based LLMs, ReG reduces the reasoning token cost by up to 30% and improves the performance by up to 4%. 

---
# Act-With-Think: Chunk Auto-Regressive Modeling for Generative Recommendation 

**Authors**: Yifan Wang, Weinan Gan, Longtao Xiao, Jieming Zhu, Heng Chang, Haozhao Wang, Rui Zhang, Zhenhua Dong, Ruiming Tang, Ruixuan Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.23643)  

**Abstract**: Generative recommendation (GR) typically encodes behavioral or semantic aspects of item information into discrete tokens, leveraging the standard autoregressive (AR) generation paradigm to make predictions. However, existing methods tend to overlook their intrinsic relationship, that is, the semantic usually provides some reasonable explainability "$\textbf{why}$" for the behavior "$\textbf{what}$", which may constrain the full potential of GR. To this end, we present Chunk AutoRegressive Modeling (CAR), a new generation paradigm following the decision pattern that users usually think semantic aspects of items (e.g. brand) and then take actions on target items (e.g. purchase). Our CAR, for the $\textit{first time}$, incorporates semantics (SIDs) and behavior (UID) into a single autoregressive transformer from an ``act-with-think'' dual perspective via chunk-level autoregression. Specifically, CAR packs SIDs and UID into a conceptual chunk for item unified representation, allowing each decoding step to make a holistic prediction. Experiments show that our CAR significantly outperforms existing methods based on traditional AR, improving Recall@5 by 7.93% to 22.30%. Furthermore, we verify the scaling effect between model performance and SIDs bit number, demonstrating that CAR preliminary emulates a kind of slow-thinking style mechanism akin to the reasoning processes observed in large language models (LLMs). 

---
# Unveiling Decision-Making in LLMs for Text Classification : Extraction of influential and interpretable concepts with Sparse Autoencoders 

**Authors**: Mathis Le Bail, Jérémie Dentan, Davide Buscaldi, Sonia Vanier  

**Link**: [PDF](https://arxiv.org/pdf/2506.23951)  

**Abstract**: Sparse Autoencoders (SAEs) have been successfully used to probe Large Language Models (LLMs) and extract interpretable concepts from their internal representations. These concepts are linear combinations of neuron activations that correspond to human-interpretable features. In this paper, we investigate the effectiveness of SAE-based explainability approaches for sentence classification, a domain where such methods have not been extensively explored. We present a novel SAE-based architecture tailored for text classification, leveraging a specialized classifier head and incorporating an activation rate sparsity loss. We benchmark this architecture against established methods such as ConceptShap, Independent Component Analysis, and other SAE-based concept extraction techniques. Our evaluation covers two classification benchmarks and four fine-tuned LLMs from the Pythia family. We further enrich our analysis with two novel metrics for measuring the precision of concept-based explanations, using an external sentence encoder. Our empirical results show that our architecture improves both the causality and interpretability of the extracted features. 

---
# Graft: Integrating the Domain Knowledge via Efficient Parameter Synergy for MLLMs 

**Authors**: Yang Dai, Jianxiang An, Tianwei Lin, Hongyang He, Hongzhe Huang, Wenqiao Zhang, Zheqi Lv, Siliang Tang, Yueting Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2506.23940)  

**Abstract**: Multimodal Large Language Models (MLLMs) have achieved success across various domains. However, their applicability tends to degrade when confronted with different types of data inputs, especially for MLLMs that have been fine-tuned for specific tasks. Despite its importance, the study of knowledge sharing among domain-specific MLLMs--such as those trained for mathematics or code--remains largely underexplored. To address the fragmentation of knowledge across domain-specialized MLLMs, we propose a unified parameter integration framework that enables modular composition of expert capabilities. Our method is grounded in a novel Compatibility-Aware Parameter Splicing (CAPS) strategy, which leverages both local functional attribution and global information-theoretic signals to guide selective parameter fusion. By extending this mechanism to the low-rank adaptation layer granularity, we ensure efficient integration with minimal inference overhead. Furthermore, we introduce a domain compatibility scoring mechanism that quantifies inter-expert alignment at the activation level and correlates with downstream task utility. This principled fusion protocol allows the final model to synergize heterogeneous expertise while preserving structural modularity. Extensive evaluations across diverse multimodal benchmarks validate the effectiveness of our framework, offering a scalable path toward compositional, domain-adaptive MLLMs. 

---
# The Trilemma of Truth in Large Language Models 

**Authors**: Germans Savcisens, Tina Eliassi-Rad  

**Link**: [PDF](https://arxiv.org/pdf/2506.23921)  

**Abstract**: We often attribute human characteristics to large language models (LLMs) and claim that they "know" certain things. LLMs have an internal probabilistic knowledge that represents information retained during training. How can we assess the veracity of this knowledge? We examine two common methods for probing the veracity of LLMs and discover several assumptions that are flawed. To address these flawed assumptions, we introduce sAwMIL (short for Sparse Aware Multiple-Instance Learning), a probing method that utilizes the internal activations of LLMs to separate statements into true, false, and neither. sAwMIL is based on multiple-instance learning and conformal prediction. We evaluate sAwMIL on 5 validity criteria across 16 open-source LLMs, including both default and chat-based variants, as well as on 3 new datasets. Among the insights we provide are: (1) the veracity signal is often concentrated in the third quarter of an LLM's depth; (2) truth and falsehood signals are not always symmetric; (3) linear probes perform better on chat models than on default models; (4) nonlinear probes may be required to capture veracity signals for some LLMs with reinforcement learning from human feedback or knowledge distillation; and (5) LLMs capture a third type of signal that is distinct from true and false and is neither true nor false. These findings provide a reliable method for verifying what LLMs "know" and how certain they are of their probabilistic internal knowledge. 

---
# Advancing Multi-Step Mathematical Reasoning in Large Language Models through Multi-Layered Self-Reflection with Auto-Prompting 

**Authors**: André de Souza Loureiro, Jorge Valverde-Rebaza, Julieta Noguez, David Escarcega, Ricardo Marcacini  

**Link**: [PDF](https://arxiv.org/pdf/2506.23888)  

**Abstract**: Recent advancements in Large Language Models (LLMs) have significantly improved their problem-solving capabilities. However, these models still struggle when faced with complex multi-step reasoning tasks. In this paper, we propose the Multi-Layered Self-Reflection with Auto-Prompting (MAPS) framework, a novel approach designed to enhance multi-step mathematical reasoning in LLMs by integrating techniques such as Chain of Thought (CoT), Self-Reflection, and Auto-Prompting. Unlike traditional static prompting methods, MAPS employs an iterative refinement process. Initially, the model generates a solution using CoT prompting. When errors are detected, an adaptive self-reflection mechanism identifies and analyzes them, generating tailored prompts to guide corrections. These dynamically adjusted prompts enable the model to iteratively refine its reasoning. Experiments on four well-established benchmarks across multiple LLMs show that MAPS significantly outperforms standard CoT and achieves competitive results with reasoning-optimized models. In addition, MAPS enables general-purpose LLMs to reach performance levels comparable to specialized reasoning models. While deeper reflection layers improve accuracy, they also increase token usage and costs. To balance this trade-off, MAPS strategically limits reflection depth, ensuring an optimal balance between cost and reasoning performance. 

---
# L0: Reinforcement Learning to Become General Agents 

**Authors**: Junjie Zhang, Jingyi Xi, Zhuoyang Song, Junyu Lu, Yuhua Ke, Ting Sun, Yukun Yang, Jiaxing Zhang, Songxin Zhang, Zejian Xie  

**Link**: [PDF](https://arxiv.org/pdf/2506.23667)  

**Abstract**: Training large language models (LLMs) to act as autonomous agents for multi-turn, long-horizon tasks remains significant challenges in scalability and training efficiency. To address this, we introduce L-Zero (L0), a scalable, end-to-end training pipeline for general-purpose agents. Featuring a low-cost, extensible, and sandboxed concurrent agent worker pool, L0 lowers the barrier for applying reinforcement learning in complex environments. We also introduce NB-Agent, the agent scaffold within L0, which operates in a "code-as-action" fashion via a Read-Eval-Print-Loop (REPL). We evaluate L0 on factuality question-answering benchmarks. Our experiments demonstrate that a base model can develop robust problem-solving skills using solely Reinforcement Learning with Verifiable Rewards (RLVR). On the Qwen2.5-7B-Instruct model, our method boosts accuracy on SimpleQA from 30 % to 80 % and on HotpotQA from 22 % to 41 %. We have open-sourced the entire L0 system, including our L0 series models, the NB-Agent, a complete training pipeline, and the corresponding training recipes on (this https URL). 

---
# Evaluating the Simulation of Human Personality-Driven Susceptibility to Misinformation with LLMs 

**Authors**: Manuel Pratelli, Marinella Petrocchi  

**Link**: [PDF](https://arxiv.org/pdf/2506.23610)  

**Abstract**: Large language models (LLMs) make it possible to generate synthetic behavioural data at scale, offering an ethical and low-cost alternative to human experiments. Whether such data can faithfully capture psychological differences driven by personality traits, however, remains an open question. We evaluate the capacity of LLM agents, conditioned on Big-Five profiles, to reproduce personality-based variation in susceptibility to misinformation, focusing on news discernment, the ability to judge true headlines as true and false headlines as false. Leveraging published datasets in which human participants with known personality profiles rated headline accuracy, we create matching LLM agents and compare their responses to the original human patterns. Certain trait-misinformation associations, notably those involving Agreeableness and Conscientiousness, are reliably replicated, whereas others diverge, revealing systematic biases in how LLMs internalize and express personality. The results underscore both the promise and the limits of personality-aligned LLMs for behavioral simulation, and offer new insight into modeling cognitive diversity in artificial agents. 

---
# On Recipe Memorization and Creativity in Large Language Models: Is Your Model a Creative Cook, a Bad Cook, or Merely a Plagiator? 

**Authors**: Jan Kvapil, Martin Fajcik  

**Link**: [PDF](https://arxiv.org/pdf/2506.23527)  

**Abstract**: This work-in-progress investigates the memorization, creativity, and nonsense found in cooking recipes generated from Large Language Models (LLMs). Precisely, we aim (i) to analyze memorization, creativity, and non-sense in LLMs using a small, high-quality set of human judgments and (ii) to evaluate potential approaches to automate such a human annotation in order to scale our study to hundreds of recipes. To achieve (i), we conduct a detailed human annotation on 20 preselected recipes generated by LLM (Mixtral), extracting each recipe's ingredients and step-by-step actions to assess which elements are memorized--i.e., directly traceable to online sources possibly seen during training--and which arise from genuine creative synthesis or outright nonsense. We find that Mixtral consistently reuses ingredients that can be found in online documents, potentially seen during model training, suggesting strong reliance on memorized content. To achieve aim (ii) and scale our analysis beyond small sample sizes and single LLM validation, we design an ``LLM-as-judge'' pipeline that automates recipe generation, nonsense detection, parsing ingredients and recipe steps, and their annotation. For instance, comparing its output against human annotations, the best ingredient extractor and annotator is Llama 3.1+Gemma 2 9B, achieving up to 78% accuracy on ingredient matching. This automated framework enables large-scale quantification of memorization, creativity, and nonsense in generated recipes, providing rigorous evidence of the models' creative capacities. 

---
# Garbage In, Reasoning Out? Why Benchmark Scores are Unreliable and What to Do About It 

**Authors**: Seyed Mahed Mousavi, Edoardo Cecchinato, Lucia Hornikova, Giuseppe Riccardi  

**Link**: [PDF](https://arxiv.org/pdf/2506.23864)  

**Abstract**: We conduct a systematic audit of three widely used reasoning benchmarks, SocialIQa, FauxPas-EAI, and ToMi, and uncover pervasive flaws in both benchmark items and evaluation methodology. Using five LLMs (GPT-{3, 3.5, 4, o1}, and LLaMA 3.1) as diagnostic tools, we identify structural, semantic, and pragmatic issues in benchmark design (e.g., duplicated items, ambiguous wording, and implausible answers), as well as scoring procedures that prioritize output form over reasoning process. Through systematic human annotation and re-evaluation on cleaned benchmark subsets, we find that model scores often improve not due to due to erratic surface wording variations and not to improved reasoning. Infact, further analyses show that model performance is highly sensitive to minor input variations such as context availability and phrasing, revealing that high scores may reflect alignment with format-specific cues rather than consistent inference based on the input. These findings challenge the validity of current benchmark-based claims about reasoning in LLMs, and highlight the need for evaluation protocols that assess reasoning as a process of drawing inference from available information, rather than as static output selection. We release audited data and evaluation tools to support more interpretable and diagnostic assessments of model reasoning. 

---
# Auto-TA: Towards Scalable Automated Thematic Analysis (TA) via Multi-Agent Large Language Models with Reinforcement Learning 

**Authors**: Seungjun Yi, Joakim Nguyen, Huimin Xu, Terence Lim, Andrew Well, Mia Markey, Ying Ding  

**Link**: [PDF](https://arxiv.org/pdf/2506.23998)  

**Abstract**: Congenital heart disease (CHD) presents complex, lifelong challenges often underrepresented in traditional clinical metrics. While unstructured narratives offer rich insights into patient and caregiver experiences, manual thematic analysis (TA) remains labor-intensive and unscalable. We propose a fully automated large language model (LLM) pipeline that performs end-to-end TA on clinical narratives, which eliminates the need for manual coding or full transcript review. Our system employs a novel multi-agent framework, where specialized LLM agents assume roles to enhance theme quality and alignment with human analysis. To further improve thematic relevance, we optionally integrate reinforcement learning from human feedback (RLHF). This supports scalable, patient-centered analysis of large qualitative datasets and allows LLMs to be fine-tuned for specific clinical contexts. 

---
# What to Keep and What to Drop: Adaptive Table Filtering Framework 

**Authors**: Jang Won June  

**Link**: [PDF](https://arxiv.org/pdf/2506.23463)  

**Abstract**: Large language models (LLMs) for table-based reasoning often struggle with large tables due to input length limits. We propose ATF (Adaptive Table Filtering Framework), a modular and question-aware filtering pipeline that prunes uninformative columns and rows using LLM-generated column descriptions, clustering, and sparse-dense alignment scores. ATF integrates seamlessly with existing models (e.g., TAPAS, TAPEX) without retraining. Experiments show that ATF reduces table cells by ~70\%, boosting performance on out-of-domain TableQA tasks while causing slight performance drops on Table Fact Verification, where full-table context is more critical. These results highlight ATF's ability to adaptively balance informativeness and minimalism across tasks. 

---
# Generalist Reward Models: Found Inside Large Language Models 

**Authors**: Yi-Chen Li, Tian Xu, Yang Yu, Xuqin Zhang, Xiong-Hui Chen, Zhongxiang Ling, Ningjing Chao, Lei Yuan, Zhi-Hua Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.23235)  

**Abstract**: The alignment of Large Language Models (LLMs) is critically dependent on reward models trained on costly human preference data. While recent work explores bypassing this cost with AI feedback, these methods often lack a rigorous theoretical foundation. In this paper, we discover that a powerful generalist reward model is already latently present within any LLM trained via standard next-token prediction. We prove that this endogenous reward is not a heuristic, but is theoretically equivalent to a reward function learned through offline inverse reinforcement learning. This connection allows us to directly elicit a high-quality reward signal from a base (pre-trained or supervised fine-tuned) model without any further training. Critically, we also prove that subsequent reinforcement learning using this endogenous reward leads to a policy with a provably superior error bound compared to the base model. To our best knowledge, this is the first theoretical proof of the effectiveness of reinforcement learning for LLMs. Our experiments validate this theory, demonstrating that our method not only outperforms existing LLM-as-a-judge approaches but can also surpass explicitly trained reward models. These findings suggest that the reward modeling stage can be replaced by a principled method of eliciting the knowledge already captured during pre-training, heralding a more efficient, powerful, and scalable paradigm for LLMs alignment as well as multi-modal models. 

---
# Learning-to-Context Slope: Evaluating In-Context Learning Effectiveness Beyond Performance Illusions 

**Authors**: Dingzriui Wang, Xuanliang Zhang, Keyan Xu, Qingfu Zhu, Wanxiang Che, Yang Deng  

**Link**: [PDF](https://arxiv.org/pdf/2506.23146)  

**Abstract**: In-context learning (ICL) has emerged as an effective approach to enhance the performance of large language models (LLMs). However, its effectiveness varies significantly across models and tasks, posing challenges for practitioners to determine when ICL reliably improves performance. Current evaluation approaches, reliant on performance change after applying ICL, suffer from low reliability, poor attribution, and impracticality in data-insufficient scenarios. We propose the Learning-to-Context Slope (LCS), a novel metric that quantifies ICL effectiveness by modeling the slope between learning gain (loss decrease from demonstrations) and contextual relevance (demonstration-input relevance). LCS addresses key limitations of performance-based metrics: (1) it captures continuous loss changes even when outputs are incorrect, improving reliability; (2) its formulation attributes ICL failures to weak contextual alignment (inability to adapt inputs to demonstrations) or strong output calibration (self-verification of correctness); and (3) it minimizes reliance on labeled data via synthetic evaluation. Extensive experiments demonstrate that LCS strongly correlates with performance improvements in labeled settings and reliably reflects true effectiveness in biased or data-scarce scenarios. Further analysis reveals actionable thresholds for LCS and identifies model capabilities critical to ICL success. 

---
# LLM-Assisted Question-Answering on Technical Documents Using Structured Data-Aware Retrieval Augmented Generation 

**Authors**: Shadman Sobhan, Mohammad Ariful Haque  

**Link**: [PDF](https://arxiv.org/pdf/2506.23136)  

**Abstract**: Large Language Models (LLMs) are capable of natural language understanding and generation. But they face challenges such as hallucination and outdated knowledge. Fine-tuning is one possible solution, but it is resource-intensive and must be repeated with every data update. Retrieval-Augmented Generation (RAG) offers an efficient solution by allowing LLMs to access external knowledge sources. However, traditional RAG pipelines struggle with retrieving information from complex technical documents with structured data such as tables and images. In this work, we propose a RAG pipeline, capable of handling tables and images in documents, for technical documents that support both scanned and searchable formats. Its retrieval process combines vector similarity search with a fine-tuned reranker based on Gemma-2-9b-it. The reranker is trained using RAFT (Retrieval-Augmented Fine-Tuning) on a custom dataset designed to improve context identification for question answering. Our evaluation demonstrates that the proposed pipeline achieves a high faithfulness score of 94% (RAGas) and 96% (DeepEval), and an answer relevancy score of 87% (RAGas) and 93% (DeepEval). Comparative analysis demonstrates that the proposed architecture is superior to general RAG pipelines in terms of table-based questions and handling questions outside context. 

---
# Format-Adapter: Improving Reasoning Capability of LLMs by Adapting Suitable Format 

**Authors**: Dingzirui Wang, Xuanliang Zhang, Rongyu Cao, Longxu Dou, Xianzhen Luo, Yingwei Ma, Qingfu Zhu, Wanxiang Che, Binhua Li, Fei Huang, Yongbin Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.23133)  

**Abstract**: Generating and voting multiple answers is an effective method to mitigate reasoning inconsistencies of large language models (LLMs). Prior works have shown that multiple reasoning formats outperform a single format when generating multiple answers. However, previous works using multiple formats rely on formats labeled by humans, which could be unsuitable for all tasks and have high labeling costs. To address this issue, we adapt suitable formats to the given tasks by generating and selecting formats. We first propose how to measure the reasoning error when generating multiple answers. Then, we introduce Format-Adapter, which utilizes LLMs to generate and select suitable reasoning formats by minimizing the error measurement we present. We conduct experiments on math and commonsense reasoning tasks, where Format-Adapter achieves a 4.3% performance improvement on average over previous works, demonstrating the effectiveness. 

---
# Boosting LLM's Molecular Structure Elucidation with Knowledge Enhanced Tree Search Reasoning 

**Authors**: Xiang Zhuang, Bin Wu, Jiyu Cui, Kehua Feng, Xiaotong Li, Huabin Xing, Keyan Ding, Qiang Zhang, Huajun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.23056)  

**Abstract**: Molecular structure elucidation involves deducing a molecule's structure from various types of spectral data, which is crucial in chemical experimental analysis. While large language models (LLMs) have shown remarkable proficiency in analyzing and reasoning through complex tasks, they still encounter substantial challenges in molecular structure elucidation. We identify that these challenges largely stem from LLMs' limited grasp of specialized chemical knowledge. In this work, we introduce a Knowledge-enhanced reasoning framework for Molecular Structure Elucidation (K-MSE), leveraging Monte Carlo Tree Search for test-time scaling as a plugin. Specifically, we construct an external molecular substructure knowledge base to extend the LLMs' coverage of the chemical structure space. Furthermore, we design a specialized molecule-spectrum scorer to act as a reward model for the reasoning process, addressing the issue of inaccurate solution evaluation in LLMs. Experimental results show that our approach significantly boosts performance, particularly gaining more than 20% improvement on both GPT-4o-mini and GPT-4o. Our code is available at this https URL. 

---
# Selecting and Merging: Towards Adaptable and Scalable Named Entity Recognition with Large Language Models 

**Authors**: Zhuojun Ding, Wei Wei, Chenghao Fan  

**Link**: [PDF](https://arxiv.org/pdf/2506.22813)  

**Abstract**: Supervised fine-tuning (SFT) is widely used to align large language models (LLMs) with information extraction (IE) tasks, such as named entity recognition (NER). However, annotating such fine-grained labels and training domain-specific models is costly. Existing works typically train a unified model across multiple domains, but such approaches lack adaptation and scalability since not all training data benefits target domains and scaling trained models remains challenging. We propose the SaM framework, which dynamically Selects and Merges expert models at inference time. Specifically, for a target domain, we select domain-specific experts pre-trained on existing domains based on (i) domain similarity to the target domain and (ii) performance on sampled instances, respectively. The experts are then merged to create task-specific models optimized for the target domain. By dynamically merging experts beneficial to target domains, we improve generalization across various domains without extra training. Additionally, experts can be added or removed conveniently, leading to great scalability. Extensive experiments on multiple benchmarks demonstrate our framework's effectiveness, which outperforms the unified model by an average of 10%. We further provide insights into potential improvements, practical experience, and extensions of our framework. 

---
# Two Spelling Normalization Approaches Based on Large Language Models 

**Authors**: Miguel Domingo, Francisco Casacuberta  

**Link**: [PDF](https://arxiv.org/pdf/2506.23288)  

**Abstract**: The absence of standardized spelling conventions and the organic evolution of human language present an inherent linguistic challenge within historical documents, a longstanding concern for scholars in the humanities. Addressing this issue, spelling normalization endeavors to align a document's orthography with contemporary standards. In this study, we propose two new approaches based on large language models: one of which has been trained without a supervised training, and a second one which has been trained for machine translation. Our evaluation spans multiple datasets encompassing diverse languages and historical periods, leading us to the conclusion that while both of them yielded encouraging results, statistical machine translation still seems to be the most suitable technology for this task. 

---
# ContextCache: Context-Aware Semantic Cache for Multi-Turn Queries in Large Language Models 

**Authors**: Jianxin Yan, Wangze Ni, Lei Chen, Xuemin Lin, Peng Cheng, Zhan Qin, Kui Ren  

**Link**: [PDF](https://arxiv.org/pdf/2506.22791)  

**Abstract**: Semantic caching significantly reduces computational costs and improves efficiency by storing and reusing large language model (LLM) responses. However, existing systems rely primarily on matching individual queries, lacking awareness of multi-turn dialogue contexts, which leads to incorrect cache hits when similar queries appear in different conversational settings. This demonstration introduces ContextCache, a context-aware semantic caching system for multi-turn dialogues. ContextCache employs a two-stage retrieval architecture that first executes vector-based retrieval on the current query to identify potential matches and then integrates current and historical dialogue representations through self-attention mechanisms for precise contextual matching. Evaluation of real-world conversations shows that ContextCache improves precision and recall compared to existing methods. Additionally, cached responses exhibit approximately 10 times lower latency than direct LLM invocation, enabling significant computational cost reductions for LLM conversational applications. 

---
# The Translation Barrier Hypothesis: Multilingual Generation with Large Language Models Suffers from Implicit Translation Failure 

**Authors**: Niyati Bafna, Tianjian Li, Kenton Murray, David R. Mortensen, David Yarowsky, Hale Sirin, Daniel Khashabi  

**Link**: [PDF](https://arxiv.org/pdf/2506.22724)  

**Abstract**: Multilingual generation with large language models (LLMs) is often of poor quality for mid- to low-resource languages. Building on insights from interpretability, we demonstrate the existence of an implicit task-solving-->translation pipeline for generation, whereby the model first solves the required task in a largely target-language-agnostic manner, and subsequently translates answer concepts into the intended target language. We hypothesize that the failure of the translation stage is an important culprit for the observed low quality of final outputs, and formalize this as the translation barrier hypothesis. We test this hypothesis for a word translation task across 108 language pairs, using logit lens to observe model processing in intermediate layers. We find that a significant portion of overall failures indeed stems from translation failure, or the model's inability to translate correctly solved intermediate concepts into the target language. This is especially true for low-resource target languages. Our results highlight an important hurdle for end-to-end multilingual generation, and lend guiding insights for future work seeking to improve multilinguality in LLMs. 

---
# Assessing the feasibility of Large Language Models for detecting micro-behaviors in team interactions during space missions 

**Authors**: Ankush Raut, Projna Paromita, Sydney Begerowski, Suzanne Bell, Theodora Chaspari  

**Link**: [PDF](https://arxiv.org/pdf/2506.22679)  

**Abstract**: We explore the feasibility of large language models (LLMs) in detecting subtle expressions of micro-behaviors in team conversations using transcripts collected during simulated space missions. Specifically, we examine zero-shot classification, fine-tuning, and paraphrase-augmented fine-tuning with encoder-only sequence classification LLMs, as well as few-shot text generation with decoder-only causal language modeling LLMs, to predict the micro-behavior associated with each conversational turn (i.e., dialogue). Our findings indicate that encoder-only LLMs, such as RoBERTa and DistilBERT, struggled to detect underrepresented micro-behaviors, particularly discouraging speech, even with weighted fine-tuning. In contrast, the instruction fine-tuned version of Llama-3.1, a decoder-only LLM, demonstrated superior performance, with the best models achieving macro F1-scores of 44% for 3-way classification and 68% for binary classification. These results have implications for the development of speech technologies aimed at analyzing team communication dynamics and enhancing training interventions in high-stakes environments such as space missions, particularly in scenarios where text is the only accessible data. 

---
# DICE-BENCH: Evaluating the Tool-Use Capabilities of Large Language Models in Multi-Round, Multi-Party Dialogues 

**Authors**: Kyochul Jang, Donghyeon Lee, Kyusik Kim, Dongseok Heo, Taewhoo Lee, Woojeong Kim, Bongwon Suh  

**Link**: [PDF](https://arxiv.org/pdf/2506.22853)  

**Abstract**: Existing function-calling benchmarks focus on single-turn interactions. However, they overlook the complexity of real-world scenarios. To quantify how existing benchmarks address practical applications, we introduce DICE-SCORE, a metric that evaluates the dispersion of tool-related information such as function name and parameter values throughout the dialogue. Analyzing existing benchmarks through DICE-SCORE reveals notably low scores, highlighting the need for more realistic scenarios. To address this gap, we present DICE-BENCH, a framework that constructs practical function-calling datasets by synthesizing conversations through a tool graph that maintains dependencies across rounds and a multi-agent system with distinct personas to enhance dialogue naturalness. The final dataset comprises 1,607 high-DICE-SCORE instances. Our experiments on 19 LLMs with DICE-BENCH show that significant advances are still required before such models can be deployed effectively in real-world settings. Our code and data are all publicly available: this https URL. 

---
# RExBench: Can coding agents autonomously implement AI research extensions? 

**Authors**: Nicholas Edwards, Yukyung Lee, Yujun, Yulu Qin, Sebastian Schuster, Najoung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.22598)  

**Abstract**: Agents based on Large Language Models (LLMs) have shown promise for performing sophisticated software engineering tasks autonomously. In addition, there has been progress towards developing agents that can perform parts of the research pipeline in machine learning and the natural sciences. We argue that research extension and its implementation is a critical capability for such systems, and introduce RExBench to support the evaluation of this capability. RExBench is a benchmark consisting of 12 realistic research experiment implementation tasks that aim to investigate research hypotheses that have not previously been implemented. Each task is set up as an extension to an existing research paper and codebase, accompanied by domain expert-written instructions. RExBench is robust to data contamination, and supports an automatic evaluation infrastructure that executes agent outputs to determine whether the success criteria are met. We use this benchmark to evaluate nine LLM agents implemented using three different frameworks: aider, Claude Code, and OpenHands. We find that all agents evaluated fail to autonomously implement the majority of the extensions. Although the success rate improves with additional human-written hints, the best performance under this setting remains below 40%. This indicates that current agents are still short of being able to handle realistic research extension tasks without substantial human guidance. 

---
# Boosting CTC-Based ASR Using LLM-Based Intermediate Loss Regularization 

**Authors**: Duygu Altinok  

**Link**: [PDF](https://arxiv.org/pdf/2506.22846)  

**Abstract**: End-to-end (E2E) automatic speech recognition (ASR) systems have revolutionized the field by integrating all components into a single neural network, with attention-based encoder-decoder models achieving state-of-the-art performance. However, their autoregressive decoding process limits inference speed, making them unsuitable for real-time applications. In contrast, CTC-based models offer faster, non-autoregressive decoding but struggle to model linguistic dependencies effectively. Addressing this challenge, we propose a novel auxiliary loss framework called Language-Aware Intermediate Loss (LAIL) to enhance CTC-based ASR using the linguistic knowledge of large language models (LLMs). By attaching connector layers to intermediate encoder layers, LAIL maps outputs to the embedding space of an LLM and computes a causal language modeling loss during training. This approach enhances linguistic modeling while preserving the computational efficiency of CTC decoding. Using the Conformer architecture and various LLaMA models, we demonstrate significant improvements in Word Error Rate (WER) on the LibriSpeech, TEDLIUM2, and WSJ corpora, achieving state-of-the-art performance for CTC-based ASR with minimal computational overhead. 

---
# Computational Analysis of Climate Policy 

**Authors**: Carolyn Hicks  

**Link**: [PDF](https://arxiv.org/pdf/2506.22449)  

**Abstract**: This thesis explores the impact of the Climate Emergency movement on local government climate policy, using computational methods. The Climate Emergency movement sought to accelerate climate action at local government level through the mechanism of Climate Emergency Declarations (CEDs), resulting in a series of commitments from councils to treat climate change as an emergency. With the aim of assessing the potential of current large language models to answer complex policy questions, I first built and configured a system named PALLM (Policy Analysis with a Large Language Model), using the OpenAI model GPT-4. This system is designed to apply a conceptual framework for climate emergency response plans to a dataset of climate policy documents. I validated the performance of this system with the help of local government policymakers, by generating analyses of the climate policies of 11 local governments in Victoria and assessing the policymakers' level of agreement with PALLM's responses. Having established that PALLM's performance is satisfactory, I used it to conduct a large-scale analysis of current policy documents from local governments in the state of Victoria, Australia. This thesis presents the methodology and results of this analysis, comparing the results for councils which have passed a CED to those which did not. This study finds that GPT-4 is capable of high-level policy analysis, with limitations including a lack of reliable attribution, and can also enable more nuanced analysis by researchers. Its use in this research shows that councils which have passed a CED are more likely to have a recent and climate-specific policy, and show more attention to urgency, prioritisation, and equity and social justice, than councils which have not. It concludes that the ability to assess policy documents at scale opens up exciting new opportunities for policy researchers. 

---
