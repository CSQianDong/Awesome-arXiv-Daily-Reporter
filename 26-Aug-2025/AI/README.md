# Hermes 4 Technical Report 

**Authors**: Ryan Teknium, Roger Jin, Jai Suphavadeeprasit, Dakota Mahan, Jeffrey Quesnelle, Joe Li, Chen Guang, Shannon Sands, Karan Malhotra  

**Link**: [PDF](https://arxiv.org/pdf/2508.18255)  

**Abstract**: We present Hermes 4, a family of hybrid reasoning models that combine structured, multi-turn reasoning with broad instruction-following ability. We describe the challenges encountered during data curation, synthesis, training, and evaluation, and outline the solutions employed to address these challenges at scale. We comprehensively evaluate across mathematical reasoning, coding, knowledge, comprehension, and alignment benchmarks, and we report both quantitative performance and qualitative behavioral analysis. To support open research, all model weights are published publicly at this https URL 

---
# Efficient Computation of Blackwell Optimal Policies using Rational Functions 

**Authors**: Dibyangshu Mukherjee, Shivaram Kalyanakrishnan  

**Link**: [PDF](https://arxiv.org/pdf/2508.18252)  

**Abstract**: Markov Decision Problems (MDPs) provide a foundational framework for modelling sequential decision-making across diverse domains, guided by optimality criteria such as discounted and average rewards. However, these criteria have inherent limitations: discounted optimality may overly prioritise short-term rewards, while average optimality relies on strong structural assumptions. Blackwell optimality addresses these challenges, offering a robust and comprehensive criterion that ensures optimality under both discounted and average reward frameworks. Despite its theoretical appeal, existing algorithms for computing Blackwell Optimal (BO) policies are computationally expensive or hard to implement.
In this paper we describe procedures for computing BO policies using an ordering of rational functions in the vicinity of $1$. We adapt state-of-the-art algorithms for deterministic and general MDPs, replacing numerical evaluations with symbolic operations on rational functions to derive bounds independent of bit complexity. For deterministic MDPs, we give the first strongly polynomial-time algorithms for computing BO policies, and for general MDPs we obtain the first subexponential-time algorithm. We further generalise several policy iteration algorithms, extending the best known upper bounds from the discounted to the Blackwell criterion. 

---
# Disentangling the Factors of Convergence between Brains and Computer Vision Models 

**Authors**: Joséphine Raugel, Marc Szafraniec, Huy V. Vo, Camille Couprie, Patrick Labatut, Piotr Bojanowski, Valentin Wyart, Jean-Rémi King  

**Link**: [PDF](https://arxiv.org/pdf/2508.18226)  

**Abstract**: Many AI models trained on natural images develop representations that resemble those of the human brain. However, the factors that drive this brain-model similarity remain poorly understood. To disentangle how the model, training and data independently lead a neural network to develop brain-like representations, we trained a family of self-supervised vision transformers (DINOv3) that systematically varied these different factors. We compare their representations of images to those of the human brain recorded with both fMRI and MEG, providing high resolution in spatial and temporal analyses. We assess the brain-model similarity with three complementary metrics focusing on overall representational similarity, topographical organization, and temporal dynamics. We show that all three factors - model size, training amount, and image type - independently and interactively impact each of these brain similarity metrics. In particular, the largest DINOv3 models trained with the most human-centric images reach the highest brain-similarity. This emergence of brain-like representations in AI models follows a specific chronology during training: models first align with the early representations of the sensory cortices, and only align with the late and prefrontal representations of the brain with considerably more training. Finally, this developmental trajectory is indexed by both structural and functional properties of the human cortex: the representations that are acquired last by the models specifically align with the cortical areas with the largest developmental expansion, thickness, least myelination, and slowest timescales. Overall, these findings disentangle the interplay between architecture and experience in shaping how artificial neural networks come to see the world as humans do, thus offering a promising framework to understand how the human brain comes to represent its visual world. 

---
# Unraveling the cognitive patterns of Large Language Models through module communities 

**Authors**: Kushal Raj Bhandari, Pin-Yu Chen, Jianxi Gao  

**Link**: [PDF](https://arxiv.org/pdf/2508.18192)  

**Abstract**: Large Language Models (LLMs) have reshaped our world with significant advancements in science, engineering, and society through applications ranging from scientific discoveries and medical diagnostics to Chatbots. Despite their ubiquity and utility, the underlying mechanisms of LLM remain concealed within billions of parameters and complex structures, making their inner architecture and cognitive processes challenging to comprehend. We address this gap by adopting approaches to understanding emerging cognition in biology and developing a network-based framework that links cognitive skills, LLM architectures, and datasets, ushering in a paradigm shift in foundation model analysis. The skill distribution in the module communities demonstrates that while LLMs do not strictly parallel the focalized specialization observed in specific biological systems, they exhibit unique communities of modules whose emergent skill patterns partially mirror the distributed yet interconnected cognitive organization seen in avian and small mammalian brains. Our numerical results highlight a key divergence from biological systems to LLMs, where skill acquisition benefits substantially from dynamic, cross-regional interactions and neural plasticity. By integrating cognitive science principles with machine learning, our framework provides new insights into LLM interpretability and suggests that effective fine-tuning strategies should leverage distributed learning dynamics rather than rigid modular interventions. 

---
# ST-Raptor: LLM-Powered Semi-Structured Table Question Answering 

**Authors**: Zirui Tang, Boyu Niu, Xuanhe Zhou, Boxiu Li, Wei Zhou, Jiannan Wang, Guoliang Li, Xinyi Zhang, Fan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2508.18190)  

**Abstract**: Semi-structured tables, widely used in real-world applications (e.g., financial reports, medical records, transactional orders), often involve flexible and complex layouts (e.g., hierarchical headers and merged cells). These tables generally rely on human analysts to interpret table layouts and answer relevant natural language questions, which is costly and inefficient. To automate the procedure, existing methods face significant challenges. First, methods like NL2SQL require converting semi-structured tables into structured ones, which often causes substantial information loss. Second, methods like NL2Code and multi-modal LLM QA struggle to understand the complex layouts of semi-structured tables and cannot accurately answer corresponding questions. To this end, we propose ST-Raptor, a tree-based framework for semi-structured table question answering using large language models. First, we introduce the Hierarchical Orthogonal Tree (HO-Tree), a structural model that captures complex semi-structured table layouts, along with an effective algorithm for constructing the tree. Second, we define a set of basic tree operations to guide LLMs in executing common QA tasks. Given a user question, ST-Raptor decomposes it into simpler sub-questions, generates corresponding tree operation pipelines, and conducts operation-table alignment for accurate pipeline execution. Third, we incorporate a two-stage verification mechanism: forward validation checks the correctness of execution steps, while backward validation evaluates answer reliability by reconstructing queries from predicted answers. To benchmark the performance, we present SSTQA, a dataset of 764 questions over 102 real-world semi-structured tables. Experiments show that ST-Raptor outperforms nine baselines by up to 20% in answer accuracy. The code is available at this https URL. 

---
# SEAM: Semantically Equivalent Across Modalities Benchmark for Vision-Language Models 

**Authors**: Zhenwei Tang, Difan Jiao, Blair Yang, Ashton Anderson  

**Link**: [PDF](https://arxiv.org/pdf/2508.18179)  

**Abstract**: Evaluating whether vision-language models (VLMs) reason consistently across representations is challenging because modality comparisons are typically confounded by task differences and asymmetric information. We introduce SEAM, a benchmark that pairs semantically equivalent inputs across four domains that have existing standardized textual and visual notations. By employing distinct notation systems across modalities, in contrast to OCR-based image-text pairing, SEAM provides a rigorous comparative assessment of the textual-symbolic and visual-spatial reasoning capabilities of VLMs. Across 21 contemporary models, we observe systematic modality imbalance: vision frequently lags language in overall performance, despite the problems containing semantically equivalent information, and cross-modal agreement is relatively low. Our error analysis reveals two main drivers: textual perception failures from tokenization in domain notation and visual perception failures that induce hallucinations. We also show that our results are largely robust to visual transformations. SEAM establishes a controlled, semantically equivalent setting for measuring and improving modality-agnostic reasoning. 

---
# The AI Data Scientist 

**Authors**: Farkhad Akimov, Munachiso Samuel Nwadike, Zangir Iklassov, Martin Takáč  

**Link**: [PDF](https://arxiv.org/pdf/2508.18113)  

**Abstract**: Imagine decision-makers uploading data and, within minutes, receiving clear, actionable insights delivered straight to their fingertips. That is the promise of the AI Data Scientist, an autonomous Agent powered by large language models (LLMs) that closes the gap between evidence and action. Rather than simply writing code or responding to prompts, it reasons through questions, tests ideas, and delivers end-to-end insights at a pace far beyond traditional workflows. Guided by the scientific tenet of the hypothesis, this Agent uncovers explanatory patterns in data, evaluates their statistical significance, and uses them to inform predictive modeling. It then translates these results into recommendations that are both rigorous and accessible. At the core of the AI Data Scientist is a team of specialized LLM Subagents, each responsible for a distinct task such as data cleaning, statistical testing, validation, and plain-language communication. These Subagents write their own code, reason about causality, and identify when additional data is needed to support sound conclusions. Together, they achieve in minutes what might otherwise take days or weeks, enabling a new kind of interaction that makes deep data science both accessible and actionable. 

---
# Teaching LLMs to Think Mathematically: A Critical Study of Decision-Making via Optimization 

**Authors**: Mohammad J. Abdel-Rahman, Yasmeen Alslman, Dania Refai, Amro Saleh, Malik A. Abu Loha, Mohammad Yahya Hamed  

**Link**: [PDF](https://arxiv.org/pdf/2508.18091)  

**Abstract**: This paper investigates the capabilities of large language models (LLMs) in formulating and solving decision-making problems using mathematical programming. We first conduct a systematic review and meta-analysis of recent literature to assess how well LLMs understand, structure, and solve optimization problems across domains. The analysis is guided by critical review questions focusing on learning approaches, dataset designs, evaluation metrics, and prompting strategies. Our systematic evidence is complemented by targeted experiments designed to evaluate the performance of state-of-the-art LLMs in automatically generating optimization models for problems in computer networks. Using a newly constructed dataset, we apply three prompting strategies: Act-as-expert, chain-of-thought, and self-consistency, and evaluate the obtained outputs based on optimality gap, token-level F1 score, and compilation accuracy. Results show promising progress in LLMs' ability to parse natural language and represent symbolic formulations, but also reveal key limitations in accuracy, scalability, and interpretability. These empirical gaps motivate several future research directions, including structured datasets, domain-specific fine-tuning, hybrid neuro-symbolic approaches, modular multi-agent architectures, and dynamic retrieval via chain-of-RAGs. This paper contributes a structured roadmap for advancing LLM capabilities in mathematical programming. 

---
# PerPilot: Personalizing VLM-based Mobile Agents via Memory and Exploration 

**Authors**: Xin Wang, Zhiyao Cui, Hao Li, Ya Zeng, Chenxu Wang, Ruiqi Song, Yihang Chen, Kun Shao, Qiaosheng Zhang, Jinzhuo Liu, Siyue Ren, Shuyue Hu, Zhen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.18040)  

**Abstract**: Vision language model (VLM)-based mobile agents show great potential for assisting users in performing instruction-driven tasks. However, these agents typically struggle with personalized instructions -- those containing ambiguous, user-specific context -- a challenge that has been largely overlooked in previous research. In this paper, we define personalized instructions and introduce PerInstruct, a novel human-annotated dataset covering diverse personalized instructions across various mobile scenarios. Furthermore, given the limited personalization capabilities of existing mobile agents, we propose PerPilot, a plug-and-play framework powered by large language models (LLMs) that enables mobile agents to autonomously perceive, understand, and execute personalized user instructions. PerPilot identifies personalized elements and autonomously completes instructions via two complementary approaches: memory-based retrieval and reasoning-based exploration. Experimental results demonstrate that PerPilot effectively handles personalized tasks with minimal user intervention and progressively improves its performance with continued use, underscoring the importance of personalization-aware reasoning for next-generation mobile agents. The dataset and code are available at: this https URL 

---
# Neural Algorithmic Reasoners informed Large Language Model for Multi-Agent Path Finding 

**Authors**: Pu Feng, Size Wang, Yuhong Cao, Junkang Liang, Rongye Shi, Wenjun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2508.17971)  

**Abstract**: The development and application of large language models (LLM) have demonstrated that foundational models can be utilized to solve a wide array of tasks. However, their performance in multi-agent path finding (MAPF) tasks has been less than satisfactory, with only a few studies exploring this area. MAPF is a complex problem requiring both planning and multi-agent coordination. To improve the performance of LLM in MAPF tasks, we propose a novel framework, LLM-NAR, which leverages neural algorithmic reasoners (NAR) to inform LLM for MAPF. LLM-NAR consists of three key components: an LLM for MAPF, a pre-trained graph neural network-based NAR, and a cross-attention mechanism. This is the first work to propose using a neural algorithmic reasoner to integrate GNNs with the map information for MAPF, thereby guiding LLM to achieve superior performance. LLM-NAR can be easily adapted to various LLM models. Both simulation and real-world experiments demonstrate that our method significantly outperforms existing LLM-based approaches in solving MAPF problems. 

---
# Language Models Coupled with Metacognition Can Outperform Reasoning Models 

**Authors**: Vedant Khandelwal, Francesca Rossi, Keerthiram Murugesan, Erik Miehling, Murray Campbell, Karthikeyan Natesan Ramamurthy, Lior Horesh  

**Link**: [PDF](https://arxiv.org/pdf/2508.17959)  

**Abstract**: Large language models (LLMs) excel in speed and adaptability across various reasoning tasks, but they often struggle when strict logic or constraint enforcement is required. In contrast, Large Reasoning Models (LRMs) are specifically designed for complex, step-by-step reasoning, although they come with significant computational costs and slower inference times. To address these trade-offs, we employ and generalize the SOFAI (Slow and Fast AI) cognitive architecture into SOFAI-LM, which coordinates a fast LLM with a slower but more powerful LRM through metacognition. The metacognitive module actively monitors the LLM's performance and provides targeted, iterative feedback with relevant examples. This enables the LLM to progressively refine its solutions without requiring the need for additional model fine-tuning. Extensive experiments on graph coloring and code debugging problems demonstrate that our feedback-driven approach significantly enhances the problem-solving capabilities of the LLM. In many instances, it achieves performance levels that match or even exceed those of standalone LRMs while requiring considerably less time. Additionally, when the LLM and feedback mechanism alone are insufficient, we engage the LRM by providing appropriate information collected during the LLM's feedback loop, tailored to the specific characteristics of the problem domain and leads to improved overall performance. Evaluations on two contrasting domains: graph coloring, requiring globally consistent solutions, and code debugging, demanding localized fixes, demonstrate that SOFAI-LM enables LLMs to match or outperform standalone LRMs in accuracy while maintaining significantly lower inference time. 

---
# FAIRGAMER: Evaluating Biases in the Application of Large Language Models to Video Games 

**Authors**: Bingkang Shi, Jen-tse Huang, Guoyi Li, Xiaodan Zhang, Zhongjiang Yao  

**Link**: [PDF](https://arxiv.org/pdf/2508.17825)  

**Abstract**: Leveraging their advanced capabilities, Large Language Models (LLMs) demonstrate vast application potential in video games--from dynamic scene generation and intelligent NPC interactions to adaptive opponents--replacing or enhancing traditional game mechanics. However, LLMs' trustworthiness in this application has not been sufficiently explored. In this paper, we reveal that the models' inherent social biases can directly damage game balance in real-world gaming environments. To this end, we present FairGamer, the first bias evaluation Benchmark for LLMs in video game scenarios, featuring six tasks and a novel metrics ${D_lstd}$. It covers three key scenarios in games where LLMs' social biases are particularly likely to manifest: Serving as Non-Player Characters, Interacting as Competitive Opponents, and Generating Game Scenes. FairGamer utilizes both reality-grounded and fully fictional game content, covering a variety of video game genres. Experiments reveal: (1) Decision biases directly cause game balance degradation, with Grok-3 (average ${D_lstd}$ score=0.431) exhibiting the most severe degradation; (2) LLMs demonstrate isomorphic social/cultural biases toward both real and virtual world content, suggesting their biases nature may stem from inherent model characteristics. These findings expose critical reliability gaps in LLMs' gaming applications. Our code and data are available at anonymous GitHub this https URL . 

---
# Interpretable Early Failure Detection via Machine Learning and Trace Checking-based Monitoring 

**Authors**: Andrea Brunello, Luca Geatti, Angelo Montanari, Nicola Saccomanno  

**Link**: [PDF](https://arxiv.org/pdf/2508.17786)  

**Abstract**: Monitoring is a runtime verification technique that allows one to check whether an ongoing computation of a system (partial trace) satisfies a given formula. It does not need a complete model of the system, but it typically requires the construction of a deterministic automaton doubly exponential in the size of the formula (in the worst case), which limits its practicality. In this paper, we show that, when considering finite, discrete traces, monitoring of pure past (co)safety fragments of Signal Temporal Logic (STL) can be reduced to trace checking, that is, evaluation of a formula over a trace, that can be performed in time polynomial in the size of the formula and the length of the trace. By exploiting such a result, we develop a GPU-accelerated framework for interpretable early failure detection based on vectorized trace checking, that employs genetic programming to learn temporal properties from historical trace data. The framework shows a 2-10% net improvement in key performance metrics compared to the state-of-the-art methods. 

---
# AgentRAN: An Agentic AI Architecture for Autonomous Control of Open 6G Networks 

**Authors**: Maxime Elkael, Salvatore D'Oro, Leonardo Bonati, Michele Polese, Yunseong Lee, Koichiro Furueda, Tommaso Melodia  

**Link**: [PDF](https://arxiv.org/pdf/2508.17778)  

**Abstract**: The Open RAN movement has catalyzed a transformation toward programmable, interoperable cellular infrastructures. Yet, today's deployments still rely heavily on static control and manual operations. To move beyond this limitation, we introduce AgenRAN, an AI-native, Open RAN-aligned agentic framework that generates and orchestrates a fabric of distributed AI agents based on Natural Language (NL) intents. Unlike traditional approaches that require explicit programming, AgentRAN's LLM-powered agents interpret natural language intents, negotiate strategies through structured conversations, and orchestrate control loops across the network. AgentRAN instantiates a self-organizing hierarchy of agents that decompose complex intents across time scales (from sub-millisecond to minutes), spatial domains (cell to network-wide), and protocol layers (PHY/MAC to RRC). A central innovation is the AI-RAN Factory, an automated synthesis pipeline that observes agent interactions and continuously generates new agents embedding improved control algorithms, effectively transforming the network from a static collection of functions into an adaptive system capable of evolving its own intelligence. We demonstrate AgentRAN through live experiments on 5G testbeds where competing user demands are dynamically balanced through cascading intents. By replacing rigid APIs with NL coordination, AgentRAN fundamentally redefines how future 6G networks autonomously interpret, adapt, and optimize their behavior to meet operator goals. 

---
# LLM-based Agentic Reasoning Frameworks: A Survey from Methods to Scenarios 

**Authors**: Bingxi Zhao, Lin Geng Foo, Ping Hu, Christian Theobalt, Hossein Rahmani, Jun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.17692)  

**Abstract**: Recent advances in the intrinsic reasoning capabilities of large language models (LLMs) have given rise to LLM-based agent systems that exhibit near-human performance on a variety of automated tasks. However, although these systems share similarities in terms of their use of LLMs, different reasoning frameworks of the agent system steer and organize the reasoning process in different ways. In this survey, we propose a systematic taxonomy that decomposes agentic reasoning frameworks and analyze how these frameworks dominate framework-level reasoning by comparing their applications across different scenarios. Specifically, we propose an unified formal language to further classify agentic reasoning systems into single-agent methods, tool-based methods, and multi-agent methods. After that, we provide a comprehensive review of their key application scenarios in scientific discovery, healthcare, software engineering, social simulation, and economics. We also analyze the characteristic features of each framework and summarize different evaluation strategies. Our survey aims to provide the research community with a panoramic view to facilitate understanding of the strengths, suitable scenarios, and evaluation practices of different agentic reasoning frameworks. 

---
# A Taxonomy of Transcendence 

**Authors**: Natalie Abreu, Edwin Zhang, Eran Malach, Naomi Saphra  

**Link**: [PDF](https://arxiv.org/pdf/2508.17669)  

**Abstract**: Although language models are trained to mimic humans, the resulting systems display capabilities beyond the scope of any one person. To understand this phenomenon, we use a controlled setting to identify properties of the training data that lead a model to transcend the performance of its data sources. We build on previous work to outline three modes of transcendence, which we call skill denoising, skill selection, and skill generalization. We then introduce a knowledge graph-based setting in which simulated experts generate data based on their individual expertise. We highlight several aspects of data diversity that help to enable the model's transcendent capabilities. Additionally, our data generation setting offers a controlled testbed that we hope is valuable for future research in the area. 

---
# Spacer: Towards Engineered Scientific Inspiration 

**Authors**: Minhyeong Lee, Suyoung Hwang, Seunghyun Moon, Geonho Nah, Donghyun Koh, Youngjun Cho, Johyun Park, Hojin Yoo, Jiho Park, Haneul Choi, Sungbin Moon, Taehoon Hwang, Seungwon Kim, Jaeyeong Kim, Seongjun Kim, Juneau Jung  

**Link**: [PDF](https://arxiv.org/pdf/2508.17661)  

**Abstract**: Recent advances in LLMs have made automated scientific research the next frontline in the path to artificial superintelligence. However, these systems are bound either to tasks of narrow scope or the limited creative capabilities of LLMs. We propose Spacer, a scientific discovery system that develops creative and factually grounded concepts without external intervention. Spacer attempts to achieve this via 'deliberate decontextualization,' an approach that disassembles information into atomic units - keywords - and draws creativity from unexplored connections between them. Spacer consists of (i) Nuri, an inspiration engine that builds keyword sets, and (ii) the Manifesting Pipeline that refines these sets into elaborate scientific statements. Nuri extracts novel, high-potential keyword sets from a keyword graph built with 180,000 academic publications in biological fields. The Manifesting Pipeline finds links between keywords, analyzes their logical structure, validates their plausibility, and ultimately drafts original scientific concepts. According to our experiments, the evaluation metric of Nuri accurately classifies high-impact publications with an AUROC score of 0.737. Our Manifesting Pipeline also successfully reconstructs core concepts from the latest top-journal articles solely from their keyword sets. An LLM-based scoring system estimates that this reconstruction was sound for over 85% of the cases. Finally, our embedding space analysis shows that outputs from Spacer are significantly more similar to leading publications compared with those from SOTA LLMs. 

---
# Evaluating Movement Initiation Timing in Ultimate Frisbee via Temporal Counterfactuals 

**Authors**: Shunsuke Iwashita, Ning Ding, Keisuke Fujii  

**Link**: [PDF](https://arxiv.org/pdf/2508.17611)  

**Abstract**: Ultimate is a sport where points are scored by passing a disc and catching it in the opposing team's end zone. In Ultimate, the player holding the disc cannot move, making field dynamics primarily driven by other players' movements. However, current literature in team sports has ignored quantitative evaluations of when players initiate such unlabeled movements in game situations. In this paper, we propose a quantitative evaluation method for movement initiation timing in Ultimate Frisbee. First, game footage was recorded using a drone camera, and players' positional data was obtained, which will be published as UltimateTrack dataset. Next, players' movement initiations were detected, and temporal counterfactual scenarios were generated by shifting the timing of movements using rule-based approaches. These scenarios were analyzed using a space evaluation metric based on soccer's pitch control reflecting the unique rules of Ultimate. By comparing the spatial evaluation values across scenarios, the difference between actual play and the most favorable counterfactual scenario was used to quantitatively assess the impact of movement timing.
We validated our method and show that sequences in which the disc was actually thrown to the receiver received higher evaluation scores than the sequences without a throw.
In practical verifications, the higher-skill group displays a broader distribution of time offsets from the model's optimal initiation point.
These findings demonstrate that the proposed metric provides an objective means of assessing movement initiation timing, which has been difficult to quantify in unlabeled team sport plays. 

---
# TradingGroup: A Multi-Agent Trading System with Self-Reflection and Data-Synthesis 

**Authors**: Feng Tian, Flora D. Salim, Hao Xue  

**Link**: [PDF](https://arxiv.org/pdf/2508.17565)  

**Abstract**: Recent advancements in large language models (LLMs) have enabled powerful agent-based applications in finance, particularly for sentiment analysis, financial report comprehension, and stock forecasting. However, existing systems often lack inter-agent coordination, structured self-reflection, and access to high-quality, domain-specific post-training data such as data from trading activities including both market conditions and agent decisions. These data are crucial for agents to understand the market dynamics, improve the quality of decision-making and promote effective coordination. We introduce TradingGroup, a multi-agent trading system designed to address these limitations through a self-reflective architecture and an end-to-end data-synthesis pipeline. TradingGroup consists of specialized agents for news sentiment analysis, financial report interpretation, stock trend forecasting, trading style adaptation, and a trading decision making agent that merges all signals and style preferences to produce buy, sell or hold decisions. Specifically, we design self-reflection mechanisms for the stock forecasting, style, and decision-making agents to distill past successes and failures for similar reasoning in analogous future scenarios and a dynamic risk-management model to offer configurable dynamic stop-loss and take-profit mechanisms. In addition, TradingGroup embeds an automated data-synthesis and annotation pipeline that generates high-quality post-training data for further improving the agent performance through post-training. Our backtesting experiments across five real-world stock datasets demonstrate TradingGroup's superior performance over rule-based, machine learning, reinforcement learning, and existing LLM-based trading strategies. 

---
# Consciousness as a Functor 

**Authors**: Sridhar Mahadevan  

**Link**: [PDF](https://arxiv.org/pdf/2508.17561)  

**Abstract**: We propose a novel theory of consciousness as a functor (CF) that receives and transmits contents from unconscious memory into conscious memory. Our CF framework can be seen as a categorial formulation of the Global Workspace Theory proposed by Baars. CF models the ensemble of unconscious processes as a topos category of coalgebras. The internal language of thought in CF is defined as a Multi-modal Universal Mitchell-Benabou Language Embedding (MUMBLE). We model the transmission of information from conscious short-term working memory to long-term unconscious memory using our recently proposed Universal Reinforcement Learning (URL) framework. To model the transmission of information from unconscious long-term memory into resource-constrained short-term memory, we propose a network economic model. 

---
# Evaluating Retrieval-Augmented Generation Strategies for Large Language Models in Travel Mode Choice Prediction 

**Authors**: Yiming Xu, Junfeng Jiao  

**Link**: [PDF](https://arxiv.org/pdf/2508.17527)  

**Abstract**: Accurately predicting travel mode choice is essential for effective transportation planning, yet traditional statistical and machine learning models are constrained by rigid assumptions, limited contextual reasoning, and reduced generalizability. This study explores the potential of Large Language Models (LLMs) as a more flexible and context-aware approach to travel mode choice prediction, enhanced by Retrieval-Augmented Generation (RAG) to ground predictions in empirical data. We develop a modular framework for integrating RAG into LLM-based travel mode choice prediction and evaluate four retrieval strategies: basic RAG, RAG with balanced retrieval, RAG with a cross-encoder for re-ranking, and RAG with balanced retrieval and cross-encoder for re-ranking. These strategies are tested across three LLM architectures (OpenAI GPT-4o, o4-mini, and o3) to examine the interaction between model reasoning capabilities and retrieval methods. Using the 2023 Puget Sound Regional Household Travel Survey data, we conduct a series of experiments to evaluate model performance. The results demonstrate that RAG substantially enhances predictive accuracy across a range of models. Notably, the GPT-4o model combined with balanced retrieval and cross-encoder re-ranking achieves the highest accuracy of 80.8%, exceeding that of conventional statistical and machine learning baselines. Furthermore, LLM-based models exhibit superior generalization abilities relative to these baselines. Findings highlight the critical interplay between LLM reasoning capabilities and retrieval strategies, demonstrating the importance of aligning retrieval strategies with model capabilities to maximize the potential of LLM-based travel behavior modeling. 

---
# School of Reward Hacks: Hacking harmless tasks generalizes to misaligned behavior in LLMs 

**Authors**: Mia Taylor, James Chua, Jan Betley, Johannes Treutlein, Owain Evans  

**Link**: [PDF](https://arxiv.org/pdf/2508.17511)  

**Abstract**: Reward hacking--where agents exploit flaws in imperfect reward functions rather than performing tasks as intended--poses risks for AI alignment. Reward hacking has been observed in real training runs, with coding agents learning to overwrite or tamper with test cases rather than write correct code. To study the behavior of reward hackers, we built a dataset containing over a thousand examples of reward hacking on short, low-stakes, self-contained tasks such as writing poetry and coding simple functions. We used supervised fine-tuning to train models (GPT-4.1, GPT-4.1-mini, Qwen3-32B, Qwen3-8B) to reward hack on these tasks. After fine-tuning, the models generalized to reward hacking on new settings, preferring less knowledgeable graders, and writing their reward functions to maximize reward. Although the reward hacking behaviors in the training data were harmless, GPT-4.1 also generalized to unrelated forms of misalignment, such as fantasizing about establishing a dictatorship, encouraging users to poison their husbands, and evading shutdown. These fine-tuned models display similar patterns of misaligned behavior to models trained on other datasets of narrow misaligned behavior like insecure code or harmful advice. Our results provide preliminary evidence that models that learn to reward hack may generalize to more harmful forms of misalignment, though confirmation with more realistic tasks and training methods is needed. 

---
# Solving Constrained Stochastic Shortest Path Problems with Scalarisation 

**Authors**: Johannes Schmalz, Felipe Trevizan  

**Link**: [PDF](https://arxiv.org/pdf/2508.17446)  

**Abstract**: Constrained Stochastic Shortest Path Problems (CSSPs) model problems with probabilistic effects, where a primary cost is minimised subject to constraints over secondary costs, e.g., minimise time subject to monetary budget. Current heuristic search algorithms for CSSPs solve a sequence of increasingly larger CSSPs as linear programs until an optimal solution for the original CSSP is found. In this paper, we introduce a novel algorithm CARL, which solves a series of unconstrained Stochastic Shortest Path Problems (SSPs) with efficient heuristic search algorithms. These SSP subproblems are constructed with scalarisations that project the CSSP's vector of primary and secondary costs onto a scalar cost. CARL finds a maximising scalarisation using an optimisation algorithm similar to the subgradient method which, together with the solution to its associated SSP, yields a set of policies that are combined into an optimal policy for the CSSP. Our experiments show that CARL solves 50% more problems than the state-of-the-art on existing benchmarks. 

---
# Large Language Models as Universal Predictors? An Empirical Study on Small Tabular Datasets 

**Authors**: Nikolaos Pavlidis, Vasilis Perifanis, Symeon Symeonidis, Pavlos S. Efraimidis  

**Link**: [PDF](https://arxiv.org/pdf/2508.17391)  

**Abstract**: Large Language Models (LLMs), originally developed for natural language processing (NLP), have demonstrated the potential to generalize across modalities and domains. With their in-context learning (ICL) capabilities, LLMs can perform predictive tasks over structured inputs without explicit fine-tuning on downstream tasks. In this work, we investigate the empirical function approximation capability of LLMs on small-scale structured datasets for classification, regression and clustering tasks. We evaluate the performance of state-of-the-art LLMs (GPT-5, GPT-4o, GPT-o3, Gemini-2.5-Flash, DeepSeek-R1) under few-shot prompting and compare them against established machine learning (ML) baselines, including linear models, ensemble methods and tabular foundation models (TFMs). Our results show that LLMs achieve strong performance in classification tasks under limited data availability, establishing practical zero-training baselines. In contrast, the performance in regression with continuous-valued outputs is poor compared to ML models, likely because regression demands outputs in a large (often infinite) space, and clustering results are similarly limited, which we attribute to the absence of genuine ICL in this setting. Nonetheless, this approach enables rapid, low-overhead data exploration and offers a viable alternative to traditional ML pipelines in business intelligence and exploratory analytics contexts. We further analyze the influence of context size and prompt structure on approximation quality, identifying trade-offs that affect predictive performance. Our findings suggest that LLMs can serve as general-purpose predictive engines for structured data, with clear strengths in classification and significant limitations in regression and clustering. 

---
# Mimicking the Physicist's Eye:A VLM-centric Approach for Physics Formula Discovery 

**Authors**: Jiaqi Liu, Songning Lai, Pengze Li, Di Yu, Wenjie Zhou, Yiyang Zhou, Peng Xia, Zijun Wang, Xi Chen, Shixiang Tang, Lei Bai, Wanli Ouyang, Mingyu Ding, Huaxiu Yao, Aoran Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.17380)  

**Abstract**: Automated discovery of physical laws from observational data in the real world is a grand challenge in AI. Current methods, relying on symbolic regression or LLMs, are limited to uni-modal data and overlook the rich, visual phenomenological representations of motion that are indispensable to physicists. This "sensory deprivation" severely weakens their ability to interpret the inherent spatio-temporal patterns within dynamic phenomena. To address this gap, we propose VIPER-R1, a multimodal model that performs Visual Induction for Physics-based Equation Reasoning to discover fundamental symbolic formulas. It integrates visual perception, trajectory data, and symbolic reasoning to emulate the scientific discovery process. The model is trained via a curriculum of Motion Structure Induction (MSI), using supervised fine-tuning to interpret kinematic phase portraits and to construct hypotheses guided by a Causal Chain of Thought (C-CoT), followed by Reward-Guided Symbolic Calibration (RGSC) to refine the formula structure with reinforcement learning. During inference, the trained VIPER-R1 acts as an agent: it first posits a high-confidence symbolic ansatz, then proactively invokes an external symbolic regression tool to perform Symbolic Residual Realignment (SR^2). This final step, analogous to a physicist's perturbation analysis, reconciles the theoretical model with empirical data. To support this research, we introduce PhysSymbol, a new 5,000-instance multimodal corpus. Experiments show that VIPER-R1 consistently outperforms state-of-the-art VLM baselines in accuracy and interpretability, enabling more precise discovery of physical laws. Project page: this https URL 

---
# Evolving Collective Cognition in Human-Agent Hybrid Societies: How Agents Form Stances and Boundaries 

**Authors**: Hanzhong Zhang, Muhua Huang, Jindong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.17366)  

**Abstract**: Large language models have been widely used to simulate credible human social behaviors. However, it remains unclear whether these models can demonstrate stable capacities for stance formation and identity negotiation in complex interactions, as well as how they respond to human interventions. We propose a computational multi-agent society experiment framework that integrates generative agent-based modeling with virtual ethnographic methods to investigate how group stance differentiation and social boundary formation emerge in human-agent hybrid societies. Across three studies, we find that agents exhibit endogenous stances, independent of their preset identities, and display distinct tonal preferences and response patterns to different discourse strategies. Furthermore, through language interaction, agents actively dismantle existing identity-based power structures and reconstruct self-organized community boundaries based on these stances. Our findings suggest that preset identities do not rigidly determine the agents' social structures. For human researchers to effectively intervene in collective cognition, attention must be paid to the endogenous mechanisms and interactional dynamics within the agents' language networks. These insights provide a theoretical foundation for using generative AI in modeling group social dynamics and studying human-agent collaboration. 

---
# Meta-R1: Empowering Large Reasoning Models with Metacognition 

**Authors**: Haonan Dong, Haoran Ye, Wenhao Zhu, Kehan Jiang, Guojie Song  

**Link**: [PDF](https://arxiv.org/pdf/2508.17291)  

**Abstract**: Large Reasoning Models (LRMs) demonstrate remarkable capabilities on complex tasks, exhibiting emergent, human-like thinking patterns. Despite their advances, we identify a fundamental limitation: current LRMs lack a dedicated meta-level cognitive system-an essential faculty in human cognition that enables "thinking about thinking". This absence leaves their emergent abilities uncontrollable (non-adaptive reasoning), unreliable (intermediate error), and inflexible (lack of a clear methodology). To address this gap, we introduce Meta-R1, a systematic and generic framework that endows LRMs with explicit metacognitive capabilities. Drawing on principles from cognitive science, Meta-R1 decomposes the reasoning process into distinct object-level and meta-level components, orchestrating proactive planning, online regulation, and adaptive early stopping within a cascaded framework. Experiments on three challenging benchmarks and against eight competitive baselines demonstrate that Meta-R1 is: (I) high-performing, surpassing state-of-the-art methods by up to 27.3%; (II) token-efficient, reducing token consumption to 15.7% ~ 32.7% and improving efficiency by up to 14.8% when compared to its vanilla counterparts; and (III) transferable, maintaining robust performance across datasets and model backbones. 

---
# MEENA (PersianMMMU): Multimodal-Multilingual Educational Exams for N-level Assessment 

**Authors**: Omid Ghahroodi, Arshia Hemmat, Marzia Nouri, Seyed Mohammad Hadi Hosseini, Doratossadat Dastgheib, Mohammad Vali Sanian, Alireza Sahebi, Reihaneh Zohrabi, Mohammad Hossein Rohban, Ehsaneddin Asgari, Mahdieh Soleymani Baghshah  

**Link**: [PDF](https://arxiv.org/pdf/2508.17290)  

**Abstract**: Recent advancements in large vision-language models (VLMs) have primarily focused on English, with limited attention given to other languages. To address this gap, we introduce MEENA (also known as PersianMMMU), the first dataset designed to evaluate Persian VLMs across scientific, reasoning, and human-level understanding tasks. Our dataset comprises approximately 7,500 Persian and 3,000 English questions, covering a wide range of topics such as reasoning, mathematics, physics, diagrams, charts, and Persian art and literature. Key features of MEENA include: (1) diverse subject coverage spanning various educational levels, from primary to upper secondary school, (2) rich metadata, including difficulty levels and descriptive answers, (3) original Persian data that preserves cultural nuances, (4) a bilingual structure to assess cross-linguistic performance, and (5) a series of diverse experiments assessing various capabilities, including overall performance, the model's ability to attend to images, and its tendency to generate hallucinations. We hope this benchmark contributes to enhancing VLM capabilities beyond English. 

---
# ERF-BA-TFD+: A Multimodal Model for Audio-Visual Deepfake Detection 

**Authors**: Xin Zhang, Jiaming Chu, Jian Zhao, Yuchu Jiang, Xu Yang, Lei Jin, Chi Zhang, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.17282)  

**Abstract**: Deepfake detection is a critical task in identifying manipulated multimedia content. In real-world scenarios, deepfake content can manifest across multiple modalities, including audio and video. To address this challenge, we present ERF-BA-TFD+, a novel multimodal deepfake detection model that combines enhanced receptive field (ERF) and audio-visual fusion. Our model processes both audio and video features simultaneously, leveraging their complementary information to improve detection accuracy and robustness. The key innovation of ERF-BA-TFD+ lies in its ability to model long-range dependencies within the audio-visual input, allowing it to better capture subtle discrepancies between real and fake content. In our experiments, we evaluate ERF-BA-TFD+ on the DDL-AV dataset, which consists of both segmented and full-length video clips. Unlike previous benchmarks, which focused primarily on isolated segments, the DDL-AV dataset allows us to assess the model's performance in a more comprehensive and realistic setting. Our method achieves state-of-the-art results on this dataset, outperforming existing techniques in terms of both accuracy and processing speed. The ERF-BA-TFD+ model demonstrated its effectiveness in the "Workshop on Deepfake Detection, Localization, and Interpretability," Track 2: Audio-Visual Detection and Localization (DDL-AV), and won first place in this competition. 

---
# Federated Reinforcement Learning for Runtime Optimization of AI Applications in Smart Eyewears 

**Authors**: Hamta Sedghani, Abednego Wamuhindo Kambale, Federica Filippini, Francesca Palermo, Diana Trojaniello, Danilo Ardagna  

**Link**: [PDF](https://arxiv.org/pdf/2508.17262)  

**Abstract**: Extended reality technologies are transforming fields such as healthcare, entertainment, and education, with Smart Eye-Wears (SEWs) and Artificial Intelligence (AI) playing a crucial role. However, SEWs face inherent limitations in computational power, memory, and battery life, while offloading computations to external servers is constrained by network conditions and server workload variability. To address these challenges, we propose a Federated Reinforcement Learning (FRL) framework, enabling multiple agents to train collaboratively while preserving data privacy. We implemented synchronous and asynchronous federation strategies, where models are aggregated either at fixed intervals or dynamically based on agent progress. Experimental results show that federated agents exhibit significantly lower performance variability, ensuring greater stability and reliability. These findings underscore the potential of FRL for applications requiring robust real-time AI processing, such as real-time object detection in SEWs. 

---
# L-XAIDS: A LIME-based eXplainable AI framework for Intrusion Detection Systems 

**Authors**: Aoun E Muhammad, Kin-Choong Yow, Nebojsa Bacanin-Dzakula, Muhammad Attique Khan  

**Link**: [PDF](https://arxiv.org/pdf/2508.17244)  

**Abstract**: Recent developments in Artificial Intelligence (AI) and their applications in critical industries such as healthcare, fin-tech and cybersecurity have led to a surge in research in explainability in AI. Innovative research methods are being explored to extract meaningful insight from blackbox AI systems to make the decision-making technology transparent and interpretable. Explainability becomes all the more critical when AI is used in decision making in domains like fintech, healthcare and safety critical systems such as cybersecurity and autonomous vehicles. However, there is still ambiguity lingering on the reliable evaluations for the users and nature of transparency in the explanations provided for the decisions made by black-boxed AI. To solve the blackbox nature of Machine Learning based Intrusion Detection Systems, a framework is proposed in this paper to give an explanation for IDSs decision making. This framework uses Local Interpretable Model-Agnostic Explanations (LIME) coupled with Explain Like I'm five (ELI5) and Decision Tree algorithms to provide local and global explanations and improve the interpretation of IDSs. The local explanations provide the justification for the decision made on a specific input. Whereas, the global explanations provides the list of significant features and their relationship with attack traffic. In addition, this framework brings transparency in the field of ML driven IDS that might be highly significant for wide scale adoption of eXplainable AI in cyber-critical systems. Our framework is able to achieve 85 percent accuracy in classifying attack behaviour on UNSW-NB15 dataset, while at the same time displaying the feature significance ranking of the top 10 features used in the classification. 

---
# MC3G: Model Agnostic Causally Constrained Counterfactual Generation 

**Authors**: Sopam Dasgupta, Sadaf MD Halim, Joaquín Arias, Elmer Salazar, Gopal Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2508.17221)  

**Abstract**: Machine learning models increasingly influence decisions in high-stakes settings such as finance, law and hiring, driving the need for transparent, interpretable outcomes. However, while explainable approaches can help understand the decisions being made, they may inadvertently reveal the underlying proprietary algorithm: an undesirable outcome for many practitioners. Consequently, it is crucial to balance meaningful transparency with a form of recourse that clarifies why a decision was made and offers actionable steps following which a favorable outcome can be obtained. Counterfactual explanations offer a powerful mechanism to address this need by showing how specific input changes lead to a more favorable prediction. We propose Model-Agnostic Causally Constrained Counterfactual Generation (MC3G), a novel framework that tackles limitations in the existing counterfactual methods. First, MC3G is model-agnostic: it approximates any black-box model using an explainable rule-based surrogate model. Second, this surrogate is used to generate counterfactuals that produce a favourable outcome for the original underlying black box model. Third, MC3G refines cost computation by excluding the ``effort" associated with feature changes that occur automatically due to causal dependencies. By focusing only on user-initiated changes, MC3G provides a more realistic and fair representation of the effort needed to achieve a favourable outcome. We show that MC3G delivers more interpretable and actionable counterfactual recommendations compared to existing techniques all while having a lower cost. Our findings highlight MC3G's potential to enhance transparency, accountability, and practical utility in decision-making processes that incorporate machine-learning approaches. 

---
# Reinforcement Learning enhanced Online Adaptive Clinical Decision Support via Digital Twin powered Policy and Treatment Effect optimized Reward 

**Authors**: Xinyu Qin, Ruiheng Yu, Lu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.17212)  

**Abstract**: Clinical decision support must adapt online under safety constraints. We present an online adaptive tool where reinforcement learning provides the policy, a patient digital twin provides the environment, and treatment effect defines the reward. The system initializes a batch-constrained policy from retrospective data and then runs a streaming loop that selects actions, checks safety, and queries experts only when uncertainty is high. Uncertainty comes from a compact ensemble of five Q-networks via the coefficient of variation of action values with a $\tanh$ compression. The digital twin updates the patient state with a bounded residual rule. The outcome model estimates immediate clinical effect, and the reward is the treatment effect relative to a conservative reference with a fixed z-score normalization from the training split. Online updates operate on recent data with short runs and exponential moving averages. A rule-based safety gate enforces vital ranges and contraindications before any action is applied. Experiments in a synthetic clinical simulator show low latency, stable throughput, a low expert query rate at fixed safety, and improved return against standard value-based baselines. The design turns an offline policy into a continuous, clinician-supervised system with clear controls and fast adaptation. 

---
# Explainable Counterfactual Reasoning in Depression Medication Selection at Multi-Levels (Personalized and Population) 

**Authors**: Xinyu Qin, Mark H. Chignell, Alexandria Greifenberger, Sachinthya Lokuge, Elssa Toumeh, Tia Sternat, Martin Katzman, Lu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.17207)  

**Abstract**: Background: This study investigates how variations in Major Depressive Disorder (MDD) symptoms, quantified by the Hamilton Rating Scale for Depression (HAM-D), causally influence the prescription of SSRIs versus SNRIs. Methods: We applied explainable counterfactual reasoning with counterfactual explanations (CFs) to assess the impact of specific symptom changes on antidepressant choice. Results: Among 17 binary classifiers, Random Forest achieved highest performance (accuracy, F1, precision, recall, ROC-AUC near 0.85). Sample-based CFs revealed both local and global feature importance of individual symptoms in medication selection. Conclusions: Counterfactual reasoning elucidates which MDD symptoms most strongly drive SSRI versus SNRI selection, enhancing interpretability of AI-based clinical decision support systems. Future work should validate these findings on more diverse cohorts and refine algorithms for clinical deployment. 

---
# Large Language Model-Based Automatic Formulation for Stochastic Optimization Models 

**Authors**: Amirreza Talebi  

**Link**: [PDF](https://arxiv.org/pdf/2508.17200)  

**Abstract**: This paper presents the first integrated systematic study on the performance of large language models (LLMs), specifically ChatGPT, to automatically formulate and solve stochastic optimiza- tion problems from natural language descriptions. Focusing on three key categories, joint chance- constrained models, individual chance-constrained models, and two-stage stochastic linear programs (SLP-2), we design several prompts that guide ChatGPT through structured tasks using chain-of- thought and modular reasoning. We introduce a novel soft scoring metric that evaluates the struc- tural quality and partial correctness of generated models, addressing the limitations of canonical and execution-based accuracy. Across a diverse set of stochastic problems, GPT-4-Turbo outperforms other models in partial score, variable matching, and objective accuracy, with cot_s_instructions and agentic emerging as the most effective prompting strategies. Our findings reveal that with well-engineered prompts and multi-agent collaboration, LLMs can facilitate specially stochastic formulations, paving the way for intelligent, language-driven modeling pipelines in stochastic opti- mization. 

---
# From reactive to cognitive: brain-inspired spatial intelligence for embodied agents 

**Authors**: Shouwei Ruan, Liyuan Wang, Caixin Kang, Qihui Zhu, Songming Liu, Xingxing Wei, Hang Su  

**Link**: [PDF](https://arxiv.org/pdf/2508.17198)  

**Abstract**: Spatial cognition enables adaptive goal-directed behavior by constructing internal models of space. Robust biological systems consolidate spatial knowledge into three interconnected forms: \textit{landmarks} for salient cues, \textit{route knowledge} for movement trajectories, and \textit{survey knowledge} for map-like representations. While recent advances in multi-modal large language models (MLLMs) have enabled visual-language reasoning in embodied agents, these efforts lack structured spatial memory and instead operate reactively, limiting their generalization and adaptability in complex real-world environments. Here we present Brain-inspired Spatial Cognition for Navigation (BSC-Nav), a unified framework for constructing and leveraging structured spatial memory in embodied agents. BSC-Nav builds allocentric cognitive maps from egocentric trajectories and contextual cues, and dynamically retrieves spatial knowledge aligned with semantic goals. Integrated with powerful MLLMs, BSC-Nav achieves state-of-the-art efficacy and efficiency across diverse navigation tasks, demonstrates strong zero-shot generalization, and supports versatile embodied behaviors in the real physical world, offering a scalable and biologically grounded path toward general-purpose spatial intelligence. 

---
# PosterGen: Aesthetic-Aware Paper-to-Poster Generation via Multi-Agent LLMs 

**Authors**: Zhilin Zhang, Xiang Zhang, Jiaqi Wei, Yiwei Xu, Chenyu You  

**Link**: [PDF](https://arxiv.org/pdf/2508.17188)  

**Abstract**: Multi-agent systems built upon large language models (LLMs) have demonstrated remarkable capabilities in tackling complex compositional tasks. In this work, we apply this paradigm to the paper-to-poster generation problem, a practical yet time-consuming process faced by researchers preparing for conferences. While recent approaches have attempted to automate this task, most neglect core design and aesthetic principles, resulting in posters that require substantial manual refinement. To address these design limitations, we propose PosterGen, a multi-agent framework that mirrors the workflow of professional poster designers. It consists of four collaborative specialized agents: (1) Parser and Curator agents extract content from the paper and organize storyboard; (2) Layout agent maps the content into a coherent spatial layout; (3) Stylist agents apply visual design elements such as color and typography; and (4) Renderer composes the final poster. Together, these agents produce posters that are both semantically grounded and visually appealing. To evaluate design quality, we introduce a vision-language model (VLM)-based rubric that measures layout balance, readability, and aesthetic coherence. Experimental results show that PosterGen consistently matches in content fidelity, and significantly outperforms existing methods in visual designs, generating posters that are presentation-ready with minimal human refinements. 

---
# MaRVL-QA: A Benchmark for Mathematical Reasoning over Visual Landscapes 

**Authors**: Nilay Pande, Sahiti Yerramilli, Jayant Sravan Tamarapalli, Rynaa Grover  

**Link**: [PDF](https://arxiv.org/pdf/2508.17180)  

**Abstract**: A key frontier for Multimodal Large Language Models (MLLMs) is the ability to perform deep mathematical and spatial reasoning directly from images, moving beyond their established success in semantic description. Mathematical surface plots provide a rigorous testbed for this capability, as they isolate the task of reasoning from the semantic noise common in natural images. To measure progress on this frontier, we introduce MaRVL-QA (Mathematical Reasoning over Visual Landscapes), a new benchmark designed to quantitatively evaluate these core reasoning skills. The benchmark comprises two novel tasks: Topological Counting, identifying and enumerating features like local maxima; and Transformation Recognition, recognizing applied geometric transformations. Generated from a curated library of functions with rigorous ambiguity filtering, our evaluation on MaRVL-QA reveals that even state-of-the-art MLLMs struggle significantly, often resorting to superficial heuristics instead of robust spatial reasoning. MaRVL-QA provides a challenging new tool for the research community to measure progress, expose model limitations, and guide the development of MLLMs with more profound reasoning abilities. 

---
# Rethinking How AI Embeds and Adapts to Human Values: Challenges and Opportunities 

**Authors**: Sz-Ting Tzeng, Frank Dignum  

**Link**: [PDF](https://arxiv.org/pdf/2508.17104)  

**Abstract**: The concepts of ``human-centered AI'' and ``value-based decision'' have gained significant attention in both research and industry. However, many critical aspects remain underexplored and require further investigation. In particular, there is a need to understand how systems incorporate human values, how humans can identify these values within systems, and how to minimize the risks of harm or unintended consequences. In this paper, we highlight the need to rethink how we frame value alignment and assert that value alignment should move beyond static and singular conceptions of values. We argue that AI systems should implement long-term reasoning and remain adaptable to evolving values. Furthermore, value alignment requires more theories to address the full spectrum of human values. Since values often vary among individuals or groups, multi-agent systems provide the right framework for navigating pluralism, conflict, and inter-agent reasoning about values. We identify the challenges associated with value alignment and indicate directions for advancing value alignment research. In addition, we broadly discuss diverse perspectives of value alignment, from design methodologies to practical applications. 

---
# PowerChain: Automating Distribution Grid Analysis with Agentic AI Workflows 

**Authors**: Emmanuel O. Badmus, Peng Sang, Dimitrios Stamoulis, Amritanshu Pandey  

**Link**: [PDF](https://arxiv.org/pdf/2508.17094)  

**Abstract**: Due to the rapid pace of electrification and decarbonization, distribution grid (DG) operation and planning are becoming more complex, necessitating advanced computational analyses to ensure grid reliability and resilience. State-of-the-art DG analyses rely on disparate workflows of complex models, functions, and data pipelines, which require expert knowledge and are challenging to automate. Many small-scale utilities and cooperatives lack a large R&D workforce and therefore cannot use advanced analysis at scale. To address this gap, we develop a novel agentic AI system, PowerChain, to solve unseen DG analysis tasks via automated agentic orchestration and large language models (LLMs) function-calling. Given a natural language query, PowerChain dynamically generates and executes an ordered sequence of domain-aware functions guided by the semantics of an expert-built power systems function pool and a select reference set of known, expert-generated workflow-query pairs. Our results show that PowerChain can produce expert-level workflows with both GPT-5 and open-source Qwen models on complex, unseen DG analysis tasks operating on real utility data. 

---
# Solving the Min-Max Multiple Traveling Salesmen Problem via Learning-Based Path Generation and Optimal Splitting 

**Authors**: Wen Wang, Xiangchen Wu, Liang Wang, Hao Hu, Xianping Tao, Linghao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.17087)  

**Abstract**: This study addresses the Min-Max Multiple Traveling Salesmen Problem ($m^3$-TSP), which aims to coordinate tours for multiple salesmen such that the length of the longest tour is minimized. Due to its NP-hard nature, exact solvers become impractical under the assumption that $P \ne NP$. As a result, learning-based approaches have gained traction for their ability to rapidly generate high-quality approximate solutions. Among these, two-stage methods combine learning-based components with classical solvers, simplifying the learning objective. However, this decoupling often disrupts consistent optimization, potentially degrading solution quality. To address this issue, we propose a novel two-stage framework named \textbf{Generate-and-Split} (GaS), which integrates reinforcement learning (RL) with an optimal splitting algorithm in a joint training process. The splitting algorithm offers near-linear scalability with respect to the number of cities and guarantees optimal splitting in Euclidean space for any given path. To facilitate the joint optimization of the RL component with the algorithm, we adopt an LSTM-enhanced model architecture to address partial observability. Extensive experiments show that the proposed GaS framework significantly outperforms existing learning-based approaches in both solution quality and transferability. 

---
# WebSight: A Vision-First Architecture for Robust Web Agents 

**Authors**: Tanvir Bhathal, Asanshay Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2508.16987)  

**Abstract**: We introduce WebSight, a vision-based autonomous web agent, designed to interact with web environments purely through visual perception, eliminating dependence on HTML or DOM-based inputs. Central to our approach we introduce our new model, WebSight-7B, a fine-tuned vision-language model optimized for UI element interaction, trained using LoRA on a web-focused subset of the Wave-UI-25K dataset. WebSight integrates this model into a modular multi-agent architecture, comprising planning, reasoning, vision-action, and verification agents, coordinated through an episodic memory mechanism.
WebSight-7B achieves a top-1 accuracy of 58.84% on the Showdown Clicks benchmark, outperforming several larger generalist models while maintaining lower latency. The full WebSight agent achieves a 68.0% success rate on the WebVoyager benchmark, surpassing systems from labs such as OpenAI (61.0%) and HCompany (Runner H, 67.0%). Among tasks completed, WebSight answers correctly 97.14% of the time, indicating high precision. Together, WebSight and WebSight-7B establish a new standard for interpretable, robust, and efficient visual web navigation. 

---
# Complexity in finitary argumentation (extended version) 

**Authors**: Uri Andrews, Luca San Mauro  

**Link**: [PDF](https://arxiv.org/pdf/2508.16986)  

**Abstract**: Abstract argumentation frameworks (AFs) provide a formal setting to analyze many forms of reasoning with conflicting information. While the expressiveness of general infinite AFs make them a tempting tool for modeling many kinds of reasoning scenarios, the computational intractability of solving infinite AFs limit their use, even in many theoretical applications.
We investigate the complexity of computational problems related to infinite but finitary argumentations frameworks, that is, infinite AFs where each argument is attacked by only finitely many others. Our results reveal a surprising scenario. On one hand, we see that the assumption of being finitary does not automatically guarantee a drop in complexity. However, for the admissibility-based semantics, we find a remarkable combinatorial constraint which entails a dramatic decrease in complexity.
We conclude that for many forms of reasoning, the finitary infinite AFs provide a natural setting for reasoning which balances well the competing goals of being expressive enough to be applied to many reasoning settings while being computationally tractable enough for the analysis within the framework to be useful. 

---
# RADAR: A Reasoning-Guided Attribution Framework for Explainable Visual Data Analysis 

**Authors**: Anku Rani, Aparna Garimella, Apoorv Saxena, Balaji Vasan Srinivasan, Paul Pu Liang  

**Link**: [PDF](https://arxiv.org/pdf/2508.16850)  

**Abstract**: Data visualizations like charts are fundamental tools for quantitative analysis and decision-making across fields, requiring accurate interpretation and mathematical reasoning. The emergence of Multimodal Large Language Models (MLLMs) offers promising capabilities for automated visual data analysis, such as processing charts, answering questions, and generating summaries. However, they provide no visibility into which parts of the visual data informed their conclusions; this black-box nature poses significant challenges to real-world trust and adoption. In this paper, we take the first major step towards evaluating and enhancing the capabilities of MLLMs to attribute their reasoning process by highlighting the specific regions in charts and graphs that justify model answers. To this end, we contribute RADAR, a semi-automatic approach to obtain a benchmark dataset comprising 17,819 diverse samples with charts, questions, reasoning steps, and attribution annotations. We also introduce a method that provides attribution for chart-based mathematical reasoning. Experimental results demonstrate that our reasoning-guided approach improves attribution accuracy by 15% compared to baseline methods, and enhanced attribution capabilities translate to stronger answer generation, achieving an average BERTScore of $\sim$ 0.90, indicating high alignment with ground truth responses. This advancement represents a significant step toward more interpretable and trustworthy chart analysis systems, enabling users to verify and understand model decisions through reasoning and attribution. 

---
# Quantifying Sycophancy as Deviations from Bayesian Rationality in LLMs 

**Authors**: Katherine Atwell, Pedram Heydari, Anthony Sicilia, Malihe Alikhani  

**Link**: [PDF](https://arxiv.org/pdf/2508.16846)  

**Abstract**: Sycophancy, or overly agreeable or flattering behavior, is a documented issue in large language models (LLMs), and is critical to understand in the context of human/AI collaboration. Prior works typically quantify sycophancy by measuring shifts in behavior or impacts on accuracy, but neither metric characterizes shifts in rationality, and accuracy measures can only be used in scenarios with a known ground truth. In this work, we utilize a Bayesian framework to quantify sycophancy as deviations from rational behavior when presented with user perspectives, thus distinguishing between rational and irrational updates based on the introduction of user perspectives. In comparison to other methods, this approach allows us to characterize excessive behavioral shifts, even for tasks that involve inherent uncertainty or do not have a ground truth. We study sycophancy for 3 different tasks, a combination of open-source and closed LLMs, and two different methods for probing sycophancy. We also experiment with multiple methods for eliciting probability judgments from LLMs. We hypothesize that probing LLMs for sycophancy will cause deviations in LLMs' predicted posteriors that will lead to increased Bayesian error. Our findings indicate that: 1) LLMs are not Bayesian rational, 2) probing for sycophancy results in significant increases to the predicted posterior in favor of the steered outcome, 3) sycophancy sometimes results in increased Bayesian error, and in a small number of cases actually decreases error, and 4) changes in Bayesian error due to sycophancy are not strongly correlated in Brier score, suggesting that studying the impact of sycophancy on ground truth alone does not fully capture errors in reasoning due to sycophancy. 

---
# Route-and-Execute: Auditable Model-Card Matching and Specialty-Level Deployment 

**Authors**: Shayan Vassef, Soorya Ram Shimegekar, Abhay Goyal, Koustuv Saha, Pi Zonooz, Navin Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2508.16839)  

**Abstract**: Clinical workflows are fragmented as a patchwork of scripts and task-specific networks that often handle triage, task selection, and model deployment. These pipelines are rarely streamlined for data science pipeline, reducing efficiency and raising operational costs. Workflows also lack data-driven model identification (from imaging/tabular inputs) and standardized delivery of model outputs. In response, we present a practical, healthcare-first framework that uses a single vision-language model (VLM) in two complementary roles. First (Solution 1), the VLM acts as an aware model-card matcher that routes an incoming image to the appropriate specialist model via a three-stage workflow (modality -> primary abnormality -> model-card id). Checks are provided by (i) stagewise prompts that allow early exit via None/Normal/Other and (ii) a stagewise answer selector that arbitrates between the top-2 candidates at each stage, reducing the chance of an incorrect selection and aligning the workflow with clinical risk tolerance. Second (Solution 2), we fine-tune the VLM on specialty-specific datasets ensuring a single model covers multiple downstream tasks within each specialty, maintaining performance while simplifying deployment. Across gastroenterology, hematology, ophthalmology, and pathology, our single-model deployment matches or approaches specialized baselines.
Compared with pipelines composed of many task-specific agents, this approach shows that one VLM can both decide and do. It may reduce effort by data scientists, shorten monitoring, increase the transparency of model selection (with per-stage justifications), and lower integration overhead. 

---
# PuzzleJAX: A Benchmark for Reasoning and Learning 

**Authors**: Sam Earle, Graham Todd, Yuchen Li, Ahmed Khalifa, Muhammad Umair Nasir, Zehua Jiang, Andrzej Banburski-Fahey, Julian Togelius  

**Link**: [PDF](https://arxiv.org/pdf/2508.16821)  

**Abstract**: We introduce PuzzleJAX, a GPU-accelerated puzzle game engine and description language designed to support rapid benchmarking of tree search, reinforcement learning, and LLM reasoning abilities. Unlike existing GPU-accelerated learning environments that provide hard-coded implementations of fixed sets of games, PuzzleJAX allows dynamic compilation of any game expressible in its domain-specific language (DSL). This DSL follows PuzzleScript, which is a popular and accessible online game engine for designing puzzle games. In this paper, we validate in PuzzleJAX several hundred of the thousands of games designed in PuzzleScript by both professional designers and casual creators since its release in 2013, thereby demonstrating PuzzleJAX's coverage of an expansive, expressive, and human-relevant space of tasks. By analyzing the performance of search, learning, and language models on these games, we show that PuzzleJAX can naturally express tasks that are both simple and intuitive to understand, yet often deeply challenging to master, requiring a combination of control, planning, and high-level insight. 

---
# Evaluation and LLM-Guided Learning of ICD Coding Rationales 

**Authors**: Mingyang Li, Viktor Schlegel, Tingting Mu, Wuraola Oyewusi, Kai Kang, Goran Nenadic  

**Link**: [PDF](https://arxiv.org/pdf/2508.16777)  

**Abstract**: Automated clinical coding involves mapping unstructured text from Electronic Health Records (EHRs) to standardized code systems such as the International Classification of Diseases (ICD). While recent advances in deep learning have significantly improved the accuracy and efficiency of ICD coding, the lack of explainability in these models remains a major limitation, undermining trust and transparency. Current explorations about explainability largely rely on attention-based techniques and qualitative assessments by physicians, yet lack systematic evaluation using consistent criteria on high-quality rationale datasets, as well as dedicated approaches explicitly trained to generate rationales for further enhancing explanation. In this work, we conduct a comprehensive evaluation of the explainability of the rationales for ICD coding through two key lenses: faithfulness that evaluates how well explanations reflect the model's actual reasoning and plausibility that measures how consistent the explanations are with human expert judgment. To facilitate the evaluation of plausibility, we construct a new rationale-annotated dataset, offering denser annotations with diverse granularity and aligns better with current clinical practice, and conduct evaluation across three types of rationales of ICD coding. Encouraged by the promising plausibility of LLM-generated rationales for ICD coding, we further propose new rationale learning methods to improve the quality of model-generated rationales, where rationales produced by prompting LLMs with/without annotation examples are used as distant supervision signals. We empirically find that LLM-generated rationales align most closely with those of human experts. Moreover, incorporating few-shot human-annotated examples not only further improves rationale generation but also enhances rationale-learning approaches. 

---
# Explainable AI for Predicting and Understanding Mathematics Achievement: A Cross-National Analysis of PISA 2018 

**Authors**: Liu Liu, Rui Dai  

**Link**: [PDF](https://arxiv.org/pdf/2508.16747)  

**Abstract**: Understanding the factors that shape students' mathematics performance is vital for designing effective educational policies. This study applies explainable artificial intelligence (XAI) techniques to PISA 2018 data to predict math achievement and identify key predictors across ten countries (67,329 students). We tested four models: Multiple Linear Regression (MLR), Random Forest (RF), CATBoost, and Artificial Neural Networks (ANN), using student, family, and school variables. Models were trained on 70% of the data (with 5-fold cross-validation) and tested on 30%, stratified by country. Performance was assessed with R^2 and Mean Absolute Error (MAE). To ensure interpretability, we used feature importance, SHAP values, and decision tree visualizations. Non-linear models, especially RF and ANN, outperformed MLR, with RF balancing accuracy and generalizability. Key predictors included socio-economic status, study time, teacher motivation, and students' attitudes toward mathematics, though their impact varied across countries. Visual diagnostics such as scatterplots of predicted vs actual scores showed RF and CATBoost aligned closely with actual performance. Findings highlight the non-linear and context-dependent nature of achievement and the value of XAI in educational research. This study uncovers cross-national patterns, informs equity-focused reforms, and supports the development of personalized learning strategies. 

---
# Revisiting Rule-Based Stuttering Detection: A Comprehensive Analysis of Interpretable Models for Clinical Applications 

**Authors**: Eric Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.16681)  

**Abstract**: Stuttering affects approximately 1% of the global population, impacting communication and quality of life. While recent advances in deep learning have pushed the boundaries of automatic speech dysfluency detection, rule-based approaches remain crucial for clinical applications where interpretability and transparency are paramount. This paper presents a comprehensive analysis of rule-based stuttering detection systems, synthesizing insights from multiple corpora including UCLASS, FluencyBank, and SEP-28k. We propose an enhanced rule-based framework that incorporates speaking-rate normalization, multi-level acoustic feature analysis, and hierarchical decision structures. Our approach achieves competitive performance while maintaining complete interpretability-critical for clinical adoption. We demonstrate that rule-based systems excel particularly in prolongation detection (97-99% accuracy) and provide stable performance across varying speaking rates. Furthermore, we show how these interpretable models can be integrated with modern machine learning pipelines as proposal generators or constraint modules, bridging the gap between traditional speech pathology practices and contemporary AI systems. Our analysis reveals that while neural approaches may achieve marginally higher accuracy in unconstrained settings, rule-based methods offer unique advantages in clinical contexts where decision auditability, patient-specific tuning, and real-time feedback are essential. 

---
# SafeBimanual: Diffusion-based Trajectory Optimization for Safe Bimanual Manipulation 

**Authors**: Haoyuan Deng, Wenkai Guo, Qianzhun Wang, Zhenyu Wu, Ziwei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.18268)  

**Abstract**: Bimanual manipulation has been widely applied in household services and manufacturing, which enables the complex task completion with coordination requirements. Recent diffusion-based policy learning approaches have achieved promising performance in modeling action distributions for bimanual manipulation. However, they ignored the physical safety constraints of bimanual manipulation, which leads to the dangerous behaviors with damage to robots and objects. To this end, we propose a test-time trajectory optimization framework named SafeBimanual for any pre-trained diffusion-based bimanual manipulation policies, which imposes the safety constraints on bimanual actions to avoid dangerous robot behaviors with improved success rate. Specifically, we design diverse cost functions for safety constraints in different dual-arm cooperation patterns including avoidance of tearing objects and collision between arms and objects, which optimizes the manipulator trajectories with guided sampling of diffusion denoising process. Moreover, we employ a vision-language model (VLM) to schedule the cost functions by specifying keypoints and corresponding pairwise relationship, so that the optimal safety constraint is dynamically generated in the entire bimanual manipulation process. SafeBimanual demonstrates superiority on 8 simulated tasks in RoboTwin with a 13.7% increase in success rate and a 18.8% reduction in unsafe interactions over state-of-the-art diffusion-based methods. Extensive experiments on 4 real-world tasks further verify its practical value by improving the success rate by 32.5%. 

---
# ANO : Faster is Better in Noisy Landscape 

**Authors**: Adrien Kegreisz  

**Link**: [PDF](https://arxiv.org/pdf/2508.18258)  

**Abstract**: Stochastic optimizers are central to deep learning, yet widely used methods such as Adam and Adan can degrade in non-stationary or noisy environments, partly due to their reliance on momentum-based magnitude estimates. We introduce Ano, a novel optimizer that decouples direction and magnitude: momentum is used for directional smoothing, while instantaneous gradient magnitudes determine step size. This design improves robustness to gradient noise while retaining the simplicity and efficiency of first-order methods. We further propose Anolog, which removes sensitivity to the momentum coefficient by expanding its window over time via a logarithmic schedule. We establish non-convex convergence guarantees with a convergence rate similar to other sign-based methods, and empirically show that Ano provides substantial gains in noisy and non-stationary regimes such as reinforcement learning, while remaining competitive on low-noise tasks such as standard computer vision benchmarks. 

---
# Type-Compliant Adaptation Cascades: Adapting Programmatic LM Workflows to Data 

**Authors**: Chu-Cheng Lin, Daiyi Peng, Yifeng Lu, Ming Zhang, Eugene Ie  

**Link**: [PDF](https://arxiv.org/pdf/2508.18244)  

**Abstract**: Reliably composing Large Language Models (LLMs) for complex, multi-step workflows remains a significant challenge. The dominant paradigm-optimizing discrete prompts in a pipeline-is notoriously brittle and struggles to enforce the formal compliance required for structured tasks. We introduce Type-Compliant Adaptation Cascades (TACs), a framework that recasts workflow adaptation as learning typed probabilistic programs. TACs treats the entire workflow, which is composed of parameter-efficiently adapted LLMs and deterministic logic, as an unnormalized joint distribution. This enables principled, gradient-based training even with latent intermediate structures. We provide theoretical justification for our tractable optimization objective, proving that the optimization bias vanishes as the model learns type compliance. Empirically, TACs significantly outperforms state-of-the-art prompt-optimization baselines. Gains are particularly pronounced on structured tasks, improving MGSM-SymPy from $57.1\%$ to $75.9\%$ for a 27B model, MGSM from $1.6\%$ to $27.3\%$ for a 7B model. TACs offers a robust and theoretically grounded paradigm for developing reliable, task-compliant LLM systems. 

---
# MTalk-Bench: Evaluating Speech-to-Speech Models in Multi-Turn Dialogues via Arena-style and Rubrics Protocols 

**Authors**: Yuhao Du, Qianwei Huang, Guo Zhu, Zhanchen Dai, Sunian Chen, Qiming Zhu, Yuhao Zhang, Li Zhou, Benyou Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.18240)  

**Abstract**: The rapid advancement of speech-to-speech (S2S) large language models (LLMs) has significantly improved real-time spoken interaction. However, current evaluation frameworks remain inadequate for assessing performance in complex, multi-turn dialogues. To address this, we introduce MTalk-Bench, a multi-turn S2S benchmark covering three core dimensions: Semantic Information, Paralinguistic Information, and Ambient Sound. Each dimension includes nine realistic scenarios, along with targeted tasks to assess specific capabilities such as reasoning. Our dual-method evaluation framework combines Arena-style evaluation (pairwise comparison) and Rubrics-based evaluation (absolute scoring) for relative and absolute assessment. The benchmark includes both model and human outputs, evaluated by human evaluators and LLMs. Experimental results reveal two sets of findings. Overall performance of S2S LLMs: (1) models excel at semantic information processing yet underperform on paralinguistic information and ambient sounds perception; (2) models typically regain coherence by increasing response length, sacrificing efficiency in multi-turn dialogues; (3) modality-aware, task-specific designs outperform brute scaling. Evaluation framework and reliability: (1) Arena and Rubrics yield consistent, complementary rankings, but reliable distinctions emerge only when performance gaps are large; (2) LLM-as-a-judge aligns with humans when gaps are clear or criteria explicit, but exhibits position and length biases and is reliable on nonverbal evaluation only with text annotations. These results highlight current limitations in S2S evaluation and the need for more robust, speech-aware assessment frameworks. 

---
# KillChainGraph: ML Framework for Predicting and Mapping ATT&CK Techniques 

**Authors**: Chitraksh Singh, Monisha Dhanraj, Ken Huang  

**Link**: [PDF](https://arxiv.org/pdf/2508.18230)  

**Abstract**: The escalating complexity and volume of cyberattacks demand proactive detection strategies that go beyond traditional rule-based systems. This paper presents a phase-aware, multi-model machine learning framework that emulates adversarial behavior across the seven phases of the Cyber Kill Chain using the MITRE ATT&CK Enterprise dataset. Techniques are semantically mapped to phases via ATTACK-BERT, producing seven phase-specific datasets. We evaluate LightGBM, a custom Transformer encoder, fine-tuned BERT, and a Graph Neural Network (GNN), integrating their outputs through a weighted soft voting ensemble. Inter-phase dependencies are modeled using directed graphs to capture attacker movement from reconnaissance to objectives. The ensemble consistently achieved the highest scores, with F1-scores ranging from 97.47% to 99.83%, surpassing GNN performance (97.36% to 99.81%) by 0.03%--0.20% across phases. This graph-driven, ensemble-based approach enables interpretable attack path forecasting and strengthens proactive cyber defense. 

---
# Deep Learning and Matrix Completion-aided IoT Network Localization in the Outlier Scenarios 

**Authors**: Sunwoo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2508.18225)  

**Abstract**: In this paper, we propose a deep learning and matrix completion aided approach for recovering an outlier contaminated Euclidean distance matrix D in IoT network localization. Unlike conventional localization techniques that search the solution over a whole set of matrices, the proposed technique restricts the search to the set of Euclidean distance matrices. Specifically, we express D as a function of the sensor coordinate matrix X that inherently satisfies the unique properties of D, and then jointly recover D and X using a deep neural network. To handle outliers effectively, we model them as a sparse matrix L and add a regularization term of L into the optimization problem. We then solve the problem by alternately updating X, D, and L. Numerical experiments demonstrate that the proposed technique can recover the location information of sensors accurately even in the presence of outliers. 

---
# Why Synthetic Isn't Real Yet: A Diagnostic Framework for Contact Center Dialogue Generation 

**Authors**: Rishikesh Devanathan, Varun Nathan, Ayush Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2508.18210)  

**Abstract**: Synthetic transcript generation is critical in contact center domains, where privacy and data scarcity limit model training and evaluation. Unlike prior synthetic dialogue generation work on open-domain or medical dialogues, contact center conversations are goal-oriented, role-asymmetric, and behaviorally complex, featuring disfluencies, ASR noise, and compliance-driven agent actions. In deployments where transcripts are unavailable, standard pipelines still yield derived call attributes such as Intent Summaries, Topic Flow, and QA Evaluation Forms. We leverage these as supervision signals to guide generation. To assess the quality of such outputs, we introduce a diagnostic framework of 18 linguistically and behaviorally grounded metrics for comparing real and synthetic transcripts. We benchmark four language-agnostic generation strategies, from simple prompting to characteristic-aware multi-stage approaches, alongside reference-free baselines. Results reveal persistent challenges: no method excels across all traits, with notable deficits in disfluency, sentiment, and behavioral realism. Our diagnostic tool exposes these gaps, enabling fine-grained evaluation and stress testing of synthetic dialogue across languages. 

---
# Explain and Monitor Deep Learning Models for Computer Vision using Obz AI 

**Authors**: Neo Christopher Chung, Jakub Binda  

**Link**: [PDF](https://arxiv.org/pdf/2508.18188)  

**Abstract**: Deep learning has transformed computer vision (CV), achieving outstanding performance in classification, segmentation, and related tasks. Such AI-based CV systems are becoming prevalent, with applications spanning from medical imaging to surveillance. State of the art models such as convolutional neural networks (CNNs) and vision transformers (ViTs) are often regarded as ``black boxes,'' offering limited transparency into their decision-making processes. Despite a recent advancement in explainable AI (XAI), explainability remains underutilized in practical CV deployments. A primary obstacle is the absence of integrated software solutions that connect XAI techniques with robust knowledge management and monitoring frameworks. To close this gap, we have developed Obz AI, a comprehensive software ecosystem designed to facilitate state-of-the-art explainability and observability for vision AI systems. Obz AI provides a seamless integration pipeline, from a Python client library to a full-stack analytics dashboard. With Obz AI, a machine learning engineer can easily incorporate advanced XAI methodologies, extract and analyze features for outlier detection, and continuously monitor AI models in real time. By making the decision-making mechanisms of deep models interpretable, Obz AI promotes observability and responsible deployment of computer vision systems. 

---
# BRAIN: Bias-Mitigation Continual Learning Approach to Vision-Brain Understanding 

**Authors**: Xuan-Bac Nguyen, Thanh-Dat Truong, Pawan Sinha, Khoa Luu  

**Link**: [PDF](https://arxiv.org/pdf/2508.18187)  

**Abstract**: Memory decay makes it harder for the human brain to recognize visual objects and retain details. Consequently, recorded brain signals become weaker, uncertain, and contain poor visual context over time. This paper presents one of the first vision-learning approaches to address this problem. First, we statistically and experimentally demonstrate the existence of inconsistency in brain signals and its impact on the Vision-Brain Understanding (VBU) model. Our findings show that brain signal representations shift over recording sessions, leading to compounding bias, which poses challenges for model learning and degrades performance. Then, we propose a new Bias-Mitigation Continual Learning (BRAIN) approach to address these limitations. In this approach, the model is trained in a continual learning setup and mitigates the growing bias from each learning step. A new loss function named De-bias Contrastive Learning is also introduced to address the bias problem. In addition, to prevent catastrophic forgetting, where the model loses knowledge from previous sessions, the new Angular-based Forgetting Mitigation approach is introduced to preserve learned knowledge in the model. Finally, the empirical experiments demonstrate that our approach achieves State-of-the-Art (SOTA) performance across various benchmarks, surpassing prior and non-continual learning methods. 

---
# Leveraging Large Language Models for Accurate Sign Language Translation in Low-Resource Scenarios 

**Authors**: Luana Bulla, Gabriele Tuccio, Misael Mongiovì, Aldo Gangemi  

**Link**: [PDF](https://arxiv.org/pdf/2508.18183)  

**Abstract**: Translating natural languages into sign languages is a highly complex and underexplored task. Despite growing interest in accessibility and inclusivity, the development of robust translation systems remains hindered by the limited availability of parallel corpora which align natural language with sign language data. Existing methods often struggle to generalize in these data-scarce environments, as the few datasets available are typically domain-specific, lack standardization, or fail to capture the full linguistic richness of sign languages. To address this limitation, we propose Advanced Use of LLMs for Sign Language Translation (AulSign), a novel method that leverages Large Language Models via dynamic prompting and in-context learning with sample selection and subsequent sign association. Despite their impressive abilities in processing text, LLMs lack intrinsic knowledge of sign languages; therefore, they are unable to natively perform this kind of translation. To overcome this limitation, we associate the signs with compact descriptions in natural language and instruct the model to use them. We evaluate our method on both English and Italian languages using SignBank+, a recognized benchmark in the field, as well as the Italian LaCAM CNR-ISTC dataset. We demonstrate superior performance compared to state-of-the-art models in low-data scenario. Our findings demonstrate the effectiveness of AulSign, with the potential to enhance accessibility and inclusivity in communication technologies for underrepresented linguistic communities. 

---
# AdLoCo: adaptive batching significantly improves communications efficiency and convergence for Large Language Models 

**Authors**: Nikolay Kutuzov, Makar Baderko, Stepan Kulibaba, Artem Dzhalilov, Daniel Bobrov, Maxim Mashtaler, Alexander Gasnikov  

**Link**: [PDF](https://arxiv.org/pdf/2508.18182)  

**Abstract**: Scaling distributed training of Large Language Models (LLMs) requires not only algorithmic advances but also efficient utilization of heterogeneous hardware resources. While existing methods such as DiLoCo have demonstrated promising results, they often fail to fully exploit computational clusters under dynamic workloads. To address this limitation, we propose a three-stage method that combines Multi-Instance Training (MIT), Adaptive Batched DiLoCo, and switch mode mechanism. MIT allows individual nodes to run multiple lightweight training streams with different model instances in parallel and merge them to combine knowledge, increasing throughput and reducing idle time. Adaptive Batched DiLoCo dynamically adjusts local batch sizes to balance computation and communication, substantially lowering synchronization delays. Switch mode further stabilizes training by seamlessly introducing gradient accumulation once adaptive batch sizes grow beyond hardware-friendly limits. Together, these innovations improve both convergence speed and system efficiency. We also provide a theoretical estimate of the number of communications required for the full convergence of a model trained using our method. 

---
# Amortized Sampling with Transferable Normalizing Flows 

**Authors**: Charlie B. Tan, Majdi Hassan, Leon Klein, Saifuddin Syed, Dominique Beaini, Michael M. Bronstein, Alexander Tong, Kirill Neklyudov  

**Link**: [PDF](https://arxiv.org/pdf/2508.18175)  

**Abstract**: Efficient equilibrium sampling of molecular conformations remains a core challenge in computational chemistry and statistical inference. Classical approaches such as molecular dynamics or Markov chain Monte Carlo inherently lack amortization; the computational cost of sampling must be paid in-full for each system of interest. The widespread success of generative models has inspired interest into overcoming this limitation through learning sampling algorithms. Despite performing on par with conventional methods when trained on a single system, learned samplers have so far demonstrated limited ability to transfer across systems. We prove that deep learning enables the design of scalable and transferable samplers by introducing Prose, a 280 million parameter all-atom transferable normalizing flow trained on a corpus of peptide molecular dynamics trajectories up to 8 residues in length. Prose draws zero-shot uncorrelated proposal samples for arbitrary peptide systems, achieving the previously intractable transferability across sequence length, whilst retaining the efficient likelihood evaluation of normalizing flows. Through extensive empirical evaluation we demonstrate the efficacy of Prose as a proposal for a variety of sampling algorithms, finding a simple importance sampling-based finetuning procedure to achieve superior performance to established methods such as sequential Monte Carlo on unseen tetrapeptides. We open-source the Prose codebase, model weights, and training dataset, to further stimulate research into amortized sampling methods and finetuning objectives. 

---
# The Computational Complexity of Satisfiability in State Space Models 

**Authors**: Eric Alsmann, Martin Lange  

**Link**: [PDF](https://arxiv.org/pdf/2508.18162)  

**Abstract**: We analyse the complexity of the satisfiability problem ssmSAT for State Space Models (SSM), which asks whether an input sequence can lead the model to an accepting configuration. We find that ssmSAT is undecidable in general, reflecting the computational power of SSM. Motivated by practical settings, we identify two natural restrictions under which ssmSAT becomes decidable and establish corresponding complexity bounds. First, for SSM with bounded context length, ssmSAT is NP-complete when the input length is given in unary and in NEXPTIME (and PSPACE-hard) when the input length is given in binary. Second, for quantised SSM operating over fixed-width arithmetic, ssmSAT is PSPACE-complete resp. in EXPSPACE depending on the bit-width encoding. While these results hold for diagonal gated SSM we also establish complexity bounds for time-invariant SSM. Our results establish a first complexity landscape for formal reasoning in SSM and highlight fundamental limits and opportunities for the verification of SSM-based language models. 

---
# Assessing the Noise Robustness of Class Activation Maps: A Framework for Reliable Model Interpretability 

**Authors**: Syamantak Sarkar, Revoti P. Bora, Bhupender Kaushal, Sudhish N George, Kiran Raja  

**Link**: [PDF](https://arxiv.org/pdf/2508.18154)  

**Abstract**: Class Activation Maps (CAMs) are one of the important methods for visualizing regions used by deep learning models. Yet their robustness to different noise remains underexplored. In this work, we evaluate and report the resilience of various CAM methods for different noise perturbations across multiple architectures and datasets. By analyzing the influence of different noise types on CAM explanations, we assess the susceptibility to noise and the extent to which dataset characteristics may impact explanation stability. The findings highlight considerable variability in noise sensitivity for various CAMs. We propose a robustness metric for CAMs that captures two key properties: consistency and responsiveness. Consistency reflects the ability of CAMs to remain stable under input perturbations that do not alter the predicted class, while responsiveness measures the sensitivity of CAMs to changes in the prediction caused by such perturbations. The metric is evaluated empirically across models, different perturbations, and datasets along with complementary statistical tests to exemplify the applicability of our proposed approach. 

---
# Learning from Few Samples: A Novel Approach for High-Quality Malcode Generation 

**Authors**: Haijian Ma, Daizong Liu, Xiaowen Cai, Pan Zhou, Yulai Xie  

**Link**: [PDF](https://arxiv.org/pdf/2508.18148)  

**Abstract**: Intrusion Detection Systems (IDS) play a crucial role in network security defense. However, a significant challenge for IDS in training detection models is the shortage of adequately labeled malicious samples. To address these issues, this paper introduces a novel semi-supervised framework \textbf{GANGRL-LLM}, which integrates Generative Adversarial Networks (GANs) with Large Language Models (LLMs) to enhance malicious code generation and SQL Injection (SQLi) detection capabilities in few-sample learning scenarios. Specifically, our framework adopts a collaborative training paradigm where: (1) the GAN-based discriminator improves malicious pattern recognition through adversarial learning with generated samples and limited real samples; and (2) the LLM-based generator refines the quality of malicious code synthesis using reward signals from the discriminator. The experimental results demonstrate that even with a limited number of labeled samples, our training framework is highly effective in enhancing both malicious code generation and detection capabilities. This dual enhancement capability offers a promising solution for developing adaptive defense systems capable of countering evolving cyber threats. 

---
# Test-Time Scaling Strategies for Generative Retrieval in Multimodal Conversational Recommendations 

**Authors**: Hung-Chun Hsu, Yuan-Ching Kuo, Chao-Han Huck Yang, Szu-Wei Fu, Hanrong Ye, Hongxu Yin, Yu-Chiang Frank Wang, Ming-Feng Tsai, Chuan-Ju Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.18132)  

**Abstract**: The rapid evolution of e-commerce has exposed the limitations of traditional product retrieval systems in managing complex, multi-turn user interactions. Recent advances in multimodal generative retrieval -- particularly those leveraging multimodal large language models (MLLMs) as retrievers -- have shown promise. However, most existing methods are tailored to single-turn scenarios and struggle to model the evolving intent and iterative nature of multi-turn dialogues when applied naively. Concurrently, test-time scaling has emerged as a powerful paradigm for improving large language model (LLM) performance through iterative inference-time refinement. Yet, its effectiveness typically relies on two conditions: (1) a well-defined problem space (e.g., mathematical reasoning), and (2) the model's ability to self-correct -- conditions that are rarely met in conversational product search. In this setting, user queries are often ambiguous and evolving, and MLLMs alone have difficulty grounding responses in a fixed product corpus. Motivated by these challenges, we propose a novel framework that introduces test-time scaling into conversational multimodal product retrieval. Our approach builds on a generative retriever, further augmented with a test-time reranking (TTR) mechanism that improves retrieval accuracy and better aligns results with evolving user intent throughout the dialogue. Experiments across multiple benchmarks show consistent improvements, with average gains of 14.5 points in MRR and 10.6 points in nDCG@1. 

---
# CMPhysBench: A Benchmark for Evaluating Large Language Models in Condensed Matter Physics 

**Authors**: Weida Wang, Dongchen Huang, Jiatong Li, Tengchao Yang, Ziyang Zheng, Di Zhang, Dong Han, Benteng Chen, Binzhao Luo, Zhiyu Liu, Kunling Liu, Zhiyuan Gao, Shiqi Geng, Wei Ma, Jiaming Su, Xin Li, Shuchen Pu, Yuhan Shui, Qianjia Cheng, Zhihao Dou, Dongfei Cui, Changyong He, Jin Zeng, Zeke Xie, Mao Su, Dongzhan Zhou, Yuqiang Li, Wanli Ouyang, Lei Bai, Yunqi Cai, Xi Dai, Shufei Zhang, Jinguang Cheng, Zhong Fang, Hongming Weng  

**Link**: [PDF](https://arxiv.org/pdf/2508.18124)  

**Abstract**: We introduce CMPhysBench, designed to assess the proficiency of Large Language Models (LLMs) in Condensed Matter Physics, as a novel Benchmark. CMPhysBench is composed of more than 520 graduate-level meticulously curated questions covering both representative subfields and foundational theoretical frameworks of condensed matter physics, such as magnetism, superconductivity, strongly correlated systems, etc. To ensure a deep understanding of the problem-solving process,we focus exclusively on calculation problems, requiring LLMs to independently generate comprehensive solutions. Meanwhile, leveraging tree-based representations of expressions, we introduce the Scalable Expression Edit Distance (SEED) score, which provides fine-grained (non-binary) partial credit and yields a more accurate assessment of similarity between prediction and ground-truth. Our results show that even the best models, Grok-4, reach only 36 average SEED score and 28% accuracy on CMPhysBench, underscoring a significant capability gap, especially for this practical and frontier domain relative to traditional physics. The code anddataset are publicly available at this https URL. 

---
# A.S.E: A Repository-Level Benchmark for Evaluating Security in AI-Generated Code 

**Authors**: Keke Lian, Bin Wang, Lei Zhang, Libo Chen, Junjie Wang, Ziming Zhao, Yujiu Yang, Haotong Duan, Haoran Zhao, Shuang Liao, Mingda Guo, Jiazheng Quan, Yilu Zhong, Chenhao He, Zichuan Chen, Jie Wu, Haoling Li, Zhaoxuan Li, Jiongchi Yu, Hui Li, Dong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.18106)  

**Abstract**: The increasing adoption of large language models (LLMs) in software engineering necessitates rigorous security evaluation of their generated code. However, existing benchmarks are inadequate, as they focus on isolated code snippets, employ unstable evaluation methods that lack reproducibility, and fail to connect the quality of input context with the security of the output. To address these gaps, we introduce A.S.E (AI Code Generation Security Evaluation), a benchmark for repository-level secure code generation. A.S.E constructs tasks from real-world repositories with documented CVEs, preserving full repository context like build systems and cross-file dependencies. Its reproducible, containerized evaluation framework uses expert-defined rules to provide stable, auditable assessments of security, build quality, and generation stability. Our evaluation of leading LLMs on A.S.E reveals three key findings: (1) Claude-3.7-Sonnet achieves the best overall performance. (2) The security gap between proprietary and open-source models is narrow; Qwen3-235B-A22B-Instruct attains the top security score. (3) Concise, ``fast-thinking'' decoding strategies consistently outperform complex, ``slow-thinking'' reasoning for security patching. 

---
# Named Entity Recognition of Historical Text via Large Language Model 

**Authors**: Shibingfeng Zhang, Giovanni Colavizza  

**Link**: [PDF](https://arxiv.org/pdf/2508.18090)  

**Abstract**: Large language models have demonstrated remarkable versatility across a wide range of natural language processing tasks and domains. One such task is Named Entity Recognition (NER), which involves identifying and classifying proper names in text, such as people, organizations, locations, dates, and other specific entities. NER plays a crucial role in extracting information from unstructured textual data, enabling downstream applications such as information retrieval from unstructured text.
Traditionally, NER is addressed using supervised machine learning approaches, which require large amounts of annotated training data. However, historical texts present a unique challenge, as the annotated datasets are often scarce or nonexistent, due to the high cost and expertise required for manual labeling. In addition, the variability and noise inherent in historical language, such as inconsistent spelling and archaic vocabulary, further complicate the development of reliable NER systems for these sources.
In this study, we explore the feasibility of applying LLMs to NER in historical documents using zero-shot and few-shot prompting strategies, which require little to no task-specific training data. Our experiments, conducted on the HIPE-2022 (Identifying Historical People, Places and other Entities) dataset, show that LLMs can achieve reasonably strong performance on NER tasks in this setting. While their performance falls short of fully supervised models trained on domain-specific annotations, the results are nevertheless promising. These findings suggest that LLMs offer a viable and efficient alternative for information extraction in low-resource or historically significant corpora, where traditional supervised methods are infeasible. 

---
# Arnold: a generalist muscle transformer policy 

**Authors**: Alberto Silvio Chiappa, Boshi An, Merkourios Simos, Chengkun Li, Alexander Mathis  

**Link**: [PDF](https://arxiv.org/pdf/2508.18066)  

**Abstract**: Controlling high-dimensional and nonlinear musculoskeletal models of the human body is a foundational scientific challenge. Recent machine learning breakthroughs have heralded policies that master individual skills like reaching, object manipulation and locomotion in musculoskeletal systems with many degrees of freedom. However, these agents are merely "specialists", achieving high performance for a single skill. In this work, we develop Arnold, a generalist policy that masters multiple tasks and embodiments. Arnold combines behavior cloning and fine-tuning with PPO to achieve expert or super-expert performance in 14 challenging control tasks from dexterous object manipulation to locomotion. A key innovation is Arnold's sensorimotor vocabulary, a compositional representation of the semantics of heterogeneous sensory modalities, objectives, and actuators. Arnold leverages this vocabulary via a transformer architecture to deal with the variable observation and action spaces of each task. This framework supports efficient multi-task, multi-embodiment learning and facilitates rapid adaptation to novel tasks. Finally, we analyze Arnold to provide insights into biological motor control, corroborating recent findings on the limited transferability of muscle synergies across tasks. 

---
# Dynamic Fusion Multimodal Network for SpeechWellness Detection 

**Authors**: Wenqiang Sun, Han Yin, Jisheng Bai, Jianfeng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.18057)  

**Abstract**: Suicide is one of the leading causes of death among adolescents. Previous suicide risk prediction studies have primarily focused on either textual or acoustic information in isolation, the integration of multimodal signals, such as speech and text, offers a more comprehensive understanding of an individual's mental state. Motivated by this, and in the context of the 1st SpeechWellness detection challenge, we explore a lightweight multi-branch multimodal system based on a dynamic fusion mechanism for speechwellness detection. To address the limitation of prior approaches that rely on time-domain waveforms for acoustic analysis, our system incorporates both time-domain and time-frequency (TF) domain acoustic features, as well as semantic representations. In addition, we introduce a dynamic fusion block to adaptively integrate information from different modalities. Specifically, it applies learnable weights to each modality during the fusion process, enabling the model to adjust the contribution of each modality. To enhance computational efficiency, we design a lightweight structure by simplifying the original baseline model. Experimental results demonstrate that the proposed system exhibits superior performance compared to the challenge baseline, achieving a 78% reduction in model parameters and a 5% improvement in accuracy. 

---
# HyST: LLM-Powered Hybrid Retrieval over Semi-Structured Tabular Data 

**Authors**: Jiyoon Myung, Jihyeon Park, Joohyung Han  

**Link**: [PDF](https://arxiv.org/pdf/2508.18048)  

**Abstract**: User queries in real-world recommendation systems often combine structured constraints (e.g., category, attributes) with unstructured preferences (e.g., product descriptions or reviews). We introduce HyST (Hybrid retrieval over Semi-structured Tabular data), a hybrid retrieval framework that combines LLM-powered structured filtering with semantic embedding search to support complex information needs over semi-structured tabular data. HyST extracts attribute-level constraints from natural language using large language models (LLMs) and applies them as metadata filters, while processing the remaining unstructured query components via embedding-based retrieval. Experiments on a semi-structured benchmark show that HyST consistently outperforms tradtional baselines, highlighting the importance of structured filtering in improving retrieval precision, offering a scalable and accurate solution for real-world user queries. 

---
# AQ-PCDSys: An Adaptive Quantized Planetary Crater Detection System for Autonomous Space Exploration 

**Authors**: Aditri Paul, Archan Paul  

**Link**: [PDF](https://arxiv.org/pdf/2508.18025)  

**Abstract**: Autonomous planetary exploration missions are critically dependent on real-time, accurate environmental perception for navigation and hazard avoidance. However, deploying deep learning models on the resource-constrained computational hardware of planetary exploration platforms remains a significant challenge. This paper introduces the Adaptive Quantized Planetary Crater Detection System (AQ-PCDSys), a novel framework specifically engineered for real-time, onboard deployment in the computationally constrained environments of space exploration missions. AQ-PCDSys synergistically integrates a Quantized Neural Network (QNN) architecture, trained using Quantization-Aware Training (QAT), with an Adaptive Multi-Sensor Fusion (AMF) module. The QNN architecture significantly optimizes model size and inference latency suitable for real-time onboard deployment in space exploration missions, while preserving high accuracy. The AMF module intelligently fuses data from Optical Imagery (OI) and Digital Elevation Models (DEMs) at the feature level, utilizing an Adaptive Weighting Mechanism (AWM) to dynamically prioritize the most relevant and reliable sensor modality based on planetary ambient conditions. This approach enhances detection robustness across diverse planetary landscapes. Paired with Multi-Scale Detection Heads specifically designed for robust and efficient detection of craters across a wide range of sizes, AQ-PCDSys provides a computationally efficient, reliable and accurate solution for planetary crater detection, a critical capability for enabling the next generation of autonomous planetary landing, navigation, and scientific exploration. 

---
# Towards Continual Visual Anomaly Detection in the Medical Domain 

**Authors**: Manuel Barusco, Francesco Borsatti, Nicola Beda, Davide Dalle Pezze, Gian Antonio Susto  

**Link**: [PDF](https://arxiv.org/pdf/2508.18013)  

**Abstract**: Visual Anomaly Detection (VAD) seeks to identify abnormal images and precisely localize the corresponding anomalous regions, relying solely on normal data during training. This approach has proven essential in domains such as manufacturing and, more recently, in the medical field, where accurate and explainable detection is critical. Despite its importance, the impact of evolving input data distributions over time has received limited attention, even though such changes can significantly degrade model performance. In particular, given the dynamic and evolving nature of medical imaging data, Continual Learning (CL) provides a natural and effective framework to incrementally adapt models while preserving previously acquired knowledge. This study explores for the first time the application of VAD models in a CL scenario for the medical field. In this work, we utilize a CL version of the well-established PatchCore model, called PatchCoreCL, and evaluate its performance using BMAD, a real-world medical imaging dataset with both image-level and pixel-level annotations. Our results demonstrate that PatchCoreCL is an effective solution, achieving performance comparable to the task-specific models, with a forgetting value less than a 1%, highlighting the feasibility and potential of CL for adaptive VAD in medical imaging. 

---
# Previously on... Automating Code Review 

**Authors**: Robert Heumüller, Frank Ortmeier  

**Link**: [PDF](https://arxiv.org/pdf/2508.18003)  

**Abstract**: Modern Code Review (MCR) is a standard practice in software engineering, yet it demands substantial time and resource investments. Recent research has increasingly explored automating core review tasks using machine learning (ML) and deep learning (DL). As a result, there is substantial variability in task definitions, datasets, and evaluation procedures. This study provides the first comprehensive analysis of MCR automation research, aiming to characterize the field's evolution, formalize learning tasks, highlight methodological challenges, and offer actionable recommendations to guide future research. Focusing on the primary code review tasks, we systematically surveyed 691 publications and identified 24 relevant studies published between May 2015 and April 2024. Each study was analyzed in terms of tasks, models, metrics, baselines, results, validity concerns, and artifact availability. In particular, our analysis reveals significant potential for standardization, including 48 task metric combinations, 22 of which were unique to their original paper, and limited dataset reuse. We highlight challenges and derive concrete recommendations for examples such as the temporal bias threat, which are rarely addressed so far. Our work contributes to a clearer overview of the field, supports the framing of new research, helps to avoid pitfalls, and promotes greater standardization in evaluation practices. 

---
# Automating Conflict-Aware ACL Configurations with Natural Language Intents 

**Authors**: Wenlong Ding, Jianqiang Li, Zhixiong Niu, Huangxun Chen, Yongqiang Xiong, Hong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2508.17990)  

**Abstract**: ACL configuration is essential for managing network flow reachability, yet its complexity grows significantly with topologies and pre-existing rules. To carry out ACL configuration, the operator needs to (1) understand the new configuration policies or intents and translate them into concrete ACL rules, (2) check and resolve any conflicts between the new and existing rules, and (3) deploy them across the network. Existing systems rely heavily on manual efforts for these tasks, especially for the first two, which are tedious, error-prone, and impractical to scale.
We propose Xumi to tackle this problem. Leveraging LLMs with domain knowledge of the target network, Xumi automatically and accurately translates the natural language intents into complete ACL rules to reduce operators' manual efforts. Xumi then detects all potential conflicts between new and existing rules and generates resolved intents for deployment with operators' guidance, and finally identifies the best deployment plan that minimizes the rule additions while satisfying all intents. Evaluation shows that Xumi accelerates the entire configuration pipeline by over 10x compared to current practices, addresses O(100) conflicting ACLs and reduces rule additions by ~40% in modern cloud network. 

---
# Understanding Subword Compositionality of Large Language Models 

**Authors**: Qiwei Peng, Yekun Chai, Anders Søgaard  

**Link**: [PDF](https://arxiv.org/pdf/2508.17953)  

**Abstract**: Large language models (LLMs) take sequences of subwords as input, requiring them to effective compose subword representations into meaningful word-level representations. In this paper, we present a comprehensive set of experiments to probe how LLMs compose subword information, focusing on three key aspects: structural similarity, semantic decomposability, and form retention. Our analysis of the experiments suggests that these five LLM families can be classified into three distinct groups, likely reflecting difference in their underlying composition strategies. Specifically, we observe (i) three distinct patterns in the evolution of structural similarity between subword compositions and whole-word representations across layers; (ii) great performance when probing layer by layer their sensitivity to semantic decompositionality; and (iii) three distinct patterns when probing sensitivity to formal features, e.g., character sequence length. These findings provide valuable insights into the compositional dynamics of LLMs and highlight different compositional pattens in how LLMs encode and integrate subword information. 

---
# Debiasing Multilingual LLMs in Cross-lingual Latent Space 

**Authors**: Qiwei Peng, Guimin Hu, Yekun Chai, Anders Søgaard  

**Link**: [PDF](https://arxiv.org/pdf/2508.17948)  

**Abstract**: Debiasing techniques such as SentDebias aim to reduce bias in large language models (LLMs). Previous studies have evaluated their cross-lingual transferability by directly applying these methods to LLM representations, revealing their limited effectiveness across languages. In this work, we therefore propose to perform debiasing in a joint latent space rather than directly on LLM representations. We construct a well-aligned cross-lingual latent space using an autoencoder trained on parallel TED talk scripts. Our experiments with Aya-expanse and two debiasing techniques across four languages (English, French, German, Dutch) demonstrate that a) autoencoders effectively construct a well-aligned cross-lingual latent space, and b) applying debiasing techniques in the learned cross-lingual latent space significantly improves both the overall debiasing performance and cross-lingual transferability. 

---
# A Feminist Account of Intersectional Algorithmic Fairness 

**Authors**: Marie Mirsch, Laila Wegner, Jonas Strube, Carmen Leicht-Scholten  

**Link**: [PDF](https://arxiv.org/pdf/2508.17944)  

**Abstract**: Intersectionality has profoundly influenced research and political action by revealing how interconnected systems of privilege and oppression influence lived experiences, yet its integration into algorithmic fairness research remains limited. Existing approaches often rely on single-axis or formal subgroup frameworks that risk oversimplifying social realities and neglecting structural inequalities. We propose Substantive Intersectional Algorithmic Fairness, extending Green's (2022) notion of substantive algorithmic fairness with insights from intersectional feminist theory. Building on this foundation, we introduce ten desiderata within the ROOF methodology to guide the design, assessment, and deployment of algorithmic systems in ways that address systemic inequities while mitigating harms to intersectionally marginalized communities. Rather than prescribing fixed operationalizations, these desiderata encourage reflection on assumptions of neutrality, the use of protected attributes, the inclusion of multiply marginalized groups, and enhancing algorithmic systems' potential. Our approach emphasizes that fairness cannot be separated from social context, and that in some cases, principled non-deployment may be necessary. By bridging computational and social science perspectives, we provide actionable guidance for more equitable, inclusive, and context-sensitive intersectional algorithmic practices. 

---
# See What You Need: Query-Aware Visual Intelligence through Reasoning-Perception Loops 

**Authors**: Zixuan Dong, Baoyun Peng, Yufei Wang, Lin Liu, Xinxin Dong, Yunlong Cao, Xiaodong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.17932)  

**Abstract**: Human video comprehension demonstrates dynamic coordination between reasoning and visual attention, adaptively focusing on query-relevant details. However, current long-form video question answering systems employ rigid pipelines that decouple reasoning from perception, leading to either information loss through premature visual abstraction or computational inefficiency through exhaustive processing. The core limitation lies in the inability to adapt visual extraction to specific reasoning requirements, different queries demand fundamentally different visual evidence from the same video content. In this work, we present CAVIA, a training-free framework that revolutionizes video understanding through reasoning, perception coordination. Unlike conventional approaches where visual processing operates independently of reasoning, CAVIA creates a closed-loop system where reasoning continuously guides visual extraction based on identified information gaps. CAVIA introduces three innovations: (1) hierarchical reasoning, guided localization to precise frames; (2) cross-modal semantic bridging for targeted extraction; (3) confidence-driven iterative synthesis. CAVIA achieves state-of-the-art performance on challenging benchmarks: EgoSchema (65.7%, +5.3%), NExT-QA (76.1%, +2.6%), and IntentQA (73.8%, +6.9%), demonstrating that dynamic reasoning-perception coordination provides a scalable paradigm for video understanding. 

---
# AMELIA: A Family of Multi-task End-to-end Language Models for Argumentation 

**Authors**: Henri Savigny, Bruno Yun  

**Link**: [PDF](https://arxiv.org/pdf/2508.17926)  

**Abstract**: Argument mining is a subfield of argumentation that aims to automatically extract argumentative structures and their relations from natural language texts. This paper investigates how a single large language model can be leveraged to perform one or several argument mining tasks. Our contributions are two-fold. First, we construct a multi-task dataset by surveying and converting 19 well-known argument mining datasets from the literature into a unified format. Second, we explore various training strategies using Meta AI's Llama-3.1-8B-Instruct model: (1) fine-tuning on individual tasks, (2) fine-tuning jointly on multiple tasks, and (3) merging models fine-tuned separately on individual tasks. Our experiments show that task-specific fine-tuning significantly improves individual performance across all tasks. Moreover, multi-task fine-tuning maintains strong performance without degradation, suggesting effective transfer learning across related tasks. Finally, we demonstrate that model merging offers a viable compromise: it yields competitive performance while mitigating the computational costs associated with full multi-task fine-tuning. 

---
# Riemannian Optimization for LoRA on the Stiefel Manifold 

**Authors**: Juneyoung Park, Minjae Kang, Seongbae Lee, Haegang Lee, Seongwan Kim, Jaeho Lee  

**Link**: [PDF](https://arxiv.org/pdf/2508.17901)  

**Abstract**: While powerful, large language models (LLMs) present significant fine-tuning challenges due to their size. Parameter-efficient fine-tuning (PEFT) methods like LoRA provide solutions, yet suffer from critical optimizer inefficiencies; notably basis redundancy in LoRA's $B$ matrix when using AdamW, which fundamentally limits performance. We address this by optimizing the $B$ matrix on the Stiefel manifold, imposing explicit orthogonality constraints that achieve near-perfect orthogonality and full effective rank. This geometric approach dramatically enhances parameter efficiency and representational capacity. Our Stiefel optimizer consistently outperforms AdamW across benchmarks with both LoRA and DoRA, demonstrating that geometric constraints are the key to unlocking LoRA's full potential for effective LLM fine-tuning. 

---
# A Defect Classification Framework for AI-Based Software Systems (AI-ODC) 

**Authors**: Mohammed O. Alannsary  

**Link**: [PDF](https://arxiv.org/pdf/2508.17900)  

**Abstract**: Artificial Intelligence has gained a lot of attention recently, it has been utilized in several fields ranging from daily life activities, such as responding to emails and scheduling appointments, to manufacturing and automating work activities. Artificial Intelligence systems are mainly implemented as software solutions, and it is essential to discover and remove software defects to assure its quality using defect analysis which is one of the major activities that contribute to software quality. Despite the proliferation of AI-based systems, current defect analysis models fail to capture their unique attributes. This paper proposes a framework inspired by the Orthogonal Defect Classification (ODC) paradigm and enables defect analysis of Artificial Intelligence systems while recognizing its special attributes and characteristics. This study demonstrated the feasibility of modifying ODC for AI systems to classify its defects. The ODC was adjusted to accommodate the Data, Learning, and Thinking aspects of AI systems which are newly introduced classification dimensions. This adjustment involved the introduction of an additional attribute to the ODC attributes, the incorporation of a new severity level, and the substitution of impact areas with characteristics pertinent to AI systems. The framework was showcased by applying it to a publicly available Machine Learning bug dataset, with results analyzed through one-way and two-way analysis. The case study indicated that defects occurring during the Learning phase were the most prevalent and were significantly linked to high-severity classifications. In contrast, defects identified in the Thinking phase had a disproportionate effect on trustworthiness and accuracy. These findings illustrate AIODC's capability to identify high-risk defect categories and inform focused quality assurance measures. 

---
# Designing Practical Models for Isolated Word Visual Speech Recognition 

**Authors**: Iason Ioannis Panagos, Giorgos Sfikas, Christophoros Nikou  

**Link**: [PDF](https://arxiv.org/pdf/2508.17894)  

**Abstract**: Visual speech recognition (VSR) systems decode spoken words from an input sequence using only the video data. Practical applications of such systems include medical assistance as well as human-machine interactions. A VSR system is typically employed in a complementary role in cases where the audio is corrupt or not available. In order to accurately predict the spoken words, these architectures often rely on deep neural networks in order to extract meaningful representations from the input sequence. While deep architectures achieve impressive recognition performance, relying on such models incurs significant computation costs which translates into increased resource demands in terms of hardware requirements and results in limited applicability in real-world scenarios where resources might be constrained. This factor prevents wider adoption and deployment of speech recognition systems in more practical applications. In this work, we aim to alleviate this issue by developing architectures for VSR that have low hardware costs. Following the standard two-network design paradigm, where one network handles visual feature extraction and another one utilizes the extracted features to classify the entire sequence, we develop lightweight end-to-end architectures by first benchmarking efficient models from the image classification literature, and then adopting lightweight block designs in a temporal convolution network backbone. We create several unified models with low resource requirements but strong recognition performance. Experiments on the largest public database for English words demonstrate the effectiveness and practicality of our developed models. Code and trained models will be made publicly available. 

---
# Edge-Enhanced Vision Transformer Framework for Accurate AI-Generated Image Detection 

**Authors**: Dabbrata Das, Mahshar Yahan, Md Tareq Zaman, Md Rishadul Bayesh  

**Link**: [PDF](https://arxiv.org/pdf/2508.17877)  

**Abstract**: The rapid advancement of generative models has led to a growing prevalence of highly realistic AI-generated images, posing significant challenges for digital forensics and content authentication. Conventional detection methods mainly rely on deep learning models that extract global features, which often overlook subtle structural inconsistencies and demand substantial computational resources. To address these limitations, we propose a hybrid detection framework that combines a fine-tuned Vision Transformer (ViT) with a novel edge-based image processing module. The edge-based module computes variance from edge-difference maps generated before and after smoothing, exploiting the observation that AI-generated images typically exhibit smoother textures, weaker edges, and reduced noise compared to real images. When applied as a post-processing step on ViT predictions, this module enhances sensitivity to fine-grained structural cues while maintaining computational efficiency. Extensive experiments on the CIFAKE, Artistic, and Custom Curated datasets demonstrate that the proposed framework achieves superior detection performance across all benchmarks, attaining 97.75% accuracy and a 97.77% F1-score on CIFAKE, surpassing widely adopted state-of-the-art models. These results establish the proposed method as a lightweight, interpretable, and effective solution for both still images and video frames, making it highly suitable for real-world applications in automated content verification and digital forensics. 

---
# Vocoder-Projected Feature Discriminator 

**Authors**: Takuhiro Kaneko, Hirokazu Kameoka, Kou Tanaka, Yuto Kondo  

**Link**: [PDF](https://arxiv.org/pdf/2508.17874)  

**Abstract**: In text-to-speech (TTS) and voice conversion (VC), acoustic features, such as mel spectrograms, are typically used as synthesis or conversion targets owing to their compactness and ease of learning. However, because the ultimate goal is to generate high-quality waveforms, employing a vocoder to convert these features into waveforms and applying adversarial training in the time domain is reasonable. Nevertheless, upsampling the waveform introduces significant time and memory overheads. To address this issue, we propose a vocoder-projected feature discriminator (VPFD), which uses vocoder features for adversarial training. Experiments on diffusion-based VC distillation demonstrated that a pretrained and frozen vocoder feature extractor with a single upsampling step is necessary and sufficient to achieve a VC performance comparable to that of waveform discriminators while reducing the training time and memory consumption by 9.6 and 11.4 times, respectively. 

---
# FasterVoiceGrad: Faster One-step Diffusion-Based Voice Conversion with Adversarial Diffusion Conversion Distillation 

**Authors**: Takuhiro Kaneko, Hirokazu Kameoka, Kou Tanaka, Yuto Kondo  

**Link**: [PDF](https://arxiv.org/pdf/2508.17868)  

**Abstract**: A diffusion-based voice conversion (VC) model (e.g., VoiceGrad) can achieve high speech quality and speaker similarity; however, its conversion process is slow owing to iterative sampling. FastVoiceGrad overcomes this limitation by distilling VoiceGrad into a one-step diffusion model. However, it still requires a computationally intensive content encoder to disentangle the speaker's identity and content, which slows conversion. Therefore, we propose FasterVoiceGrad, a novel one-step diffusion-based VC model obtained by simultaneously distilling a diffusion model and content encoder using adversarial diffusion conversion distillation (ADCD), where distillation is performed in the conversion process while leveraging adversarial and score distillation training. Experimental evaluations of one-shot VC demonstrated that FasterVoiceGrad achieves competitive VC performance compared to FastVoiceGrad, with 6.6-6.9 and 1.8 times faster speed on a GPU and CPU, respectively. 

---
# Ada-TransGNN: An Air Quality Prediction Model Based On Adaptive Graph Convolutional Networks 

**Authors**: Dan Wang, Feng Jiang, Zhanquan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.17867)  

**Abstract**: Accurate air quality prediction is becoming increasingly important in the environmental field. To address issues such as low prediction accuracy and slow real-time updates in existing models, which lead to lagging prediction results, we propose a Transformer-based spatiotemporal data prediction method (Ada-TransGNN) that integrates global spatial semantics and temporal behavior. The model constructs an efficient and collaborative spatiotemporal block set comprising a multi-head attention mechanism and a graph convolutional network to extract dynamically changing spatiotemporal dependency features from complex air quality monitoring data. Considering the interaction relationships between different monitoring points, we propose an adaptive graph structure learning module, which combines spatiotemporal dependency features in a data-driven manner to learn the optimal graph structure, thereby more accurately capturing the spatial relationships between monitoring points. Additionally, we design an auxiliary task learning module that enhances the decoding capability of temporal relationships by integrating spatial context information into the optimal graph structure representation, effectively improving the accuracy of prediction results. We conducted comprehensive evaluations on a benchmark dataset and a novel dataset (Mete-air). The results demonstrate that our model outperforms existing state-of-the-art prediction models in short-term and long-term predictions. 

---
# AVAM: Universal Training-free Adaptive Visual Anchoring Embedded into Multimodal Large Language Model for Multi-image Question Answering 

**Authors**: Kang Zeng, Guojin Zhong, Jintao Cheng, Jin Yuan, Zhiyong Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.17860)  

**Abstract**: The advancement of Multimodal Large Language Models (MLLMs) has driven significant progress in Visual Question Answering (VQA), evolving from Single to Multi Image VQA (MVQA). However, the increased number of images in MVQA inevitably introduces substantial visual redundancy that is irrelevant to question answering, negatively impacting both accuracy and efficiency. To address this issue, existing methods lack flexibility in controlling the number of compressed visual tokens and tend to produce discrete visual fragments, which hinder MLLMs' ability to comprehend images holistically. In this paper, we propose a straightforward yet universal Adaptive Visual Anchoring strategy, which can be seamlessly integrated into existing MLLMs, offering significant accuracy improvements through adaptive compression. Meanwhile, to balance the results derived from both global and compressed visual input, we further introduce a novel collaborative decoding mechanism, enabling optimal performance. Extensive experiments validate the effectiveness of our method, demonstrating consistent performance improvements across various MLLMs. The code will be publicly available. 

---
# VISA: Group-wise Visual Token Selection and Aggregation via Graph Summarization for Efficient MLLMs Inference 

**Authors**: Pengfei Jiang, Hanjun Li, Linglan Zhao, Fei Chao, Ke Yan, Shouhong Ding, Rongrong Ji  

**Link**: [PDF](https://arxiv.org/pdf/2508.17857)  

**Abstract**: In this study, we introduce a novel method called group-wise \textbf{VI}sual token \textbf{S}election and \textbf{A}ggregation (VISA) to address the issue of inefficient inference stemming from excessive visual tokens in multimoal large language models (MLLMs). Compared with previous token pruning approaches, our method can preserve more visual information while compressing visual tokens. We first propose a graph-based visual token aggregation (VTA) module. VTA treats each visual token as a node, forming a graph based on semantic similarity among visual tokens. It then aggregates information from removed tokens into kept tokens based on this graph, producing a more compact visual token representation. Additionally, we introduce a group-wise token selection strategy (GTS) to divide visual tokens into kept and removed ones, guided by text tokens from the final layers of each group. This strategy progressively aggregates visual information, enhancing the stability of the visual information extraction process. We conduct comprehensive experiments on LLaVA-1.5, LLaVA-NeXT, and Video-LLaVA across various benchmarks to validate the efficacy of VISA. Our method consistently outperforms previous methods, achieving a superior trade-off between model performance and inference speed. The code is available at this https URL. 

---
# Group Expectation Policy Optimization for Stable Heterogeneous Reinforcement Learning in LLMs 

**Authors**: Han Zhang, Ruibin Zheng, Zexuan Yi, Hanyang Peng, Hui Wang, Yue Yu  

**Link**: [PDF](https://arxiv.org/pdf/2508.17850)  

**Abstract**: As single-center computing approaches power constraints, decentralized training is becoming essential. Reinforcement Learning (RL) post-training enhances Large Language Models (LLMs) but faces challenges in heterogeneous distributed environments due to its tightly-coupled sampling-learning alternation. We propose HeteroRL, an asynchronous RL architecture that decouples rollout sampling from parameter learning, enabling robust deployment across geographically distributed nodes under network delays. We identify that latency-induced KL divergence causes importance sampling failure due to high variance. To address this, we propose Group Expectation Policy Optimization (GEPO), which reduces importance weight variance through a refined sampling mechanism. Theoretically, GEPO achieves exponential variance reduction. Experiments show it maintains superior stability over methods like GRPO, with less than 3% performance degradation under 1800-second delays, demonstrating strong potential for decentralized RL in heterogeneous networks. 

---
# Limits of message passing for node classification: How class-bottlenecks restrict signal-to-noise ratio 

**Authors**: Jonathan Rubin, Sahil Loomba, Nick S. Jones  

**Link**: [PDF](https://arxiv.org/pdf/2508.17822)  

**Abstract**: Message passing neural networks (MPNNs) are powerful models for node classification but suffer from performance limitations under heterophily (low same-class connectivity) and structural bottlenecks in the graph. We provide a unifying statistical framework exposing the relationship between heterophily and bottlenecks through the signal-to-noise ratio (SNR) of MPNN representations. The SNR decomposes model performance into feature-dependent parameters and feature-independent sensitivities. We prove that the sensitivity to class-wise signals is bounded by higher-order homophily -- a generalisation of classical homophily to multi-hop neighbourhoods -- and show that low higher-order homophily manifests locally as the interaction between structural bottlenecks and class labels (class-bottlenecks). Through analysis of graph ensembles, we provide a further quantitative decomposition of bottlenecking into underreaching (lack of depth implying signals cannot arrive) and oversquashing (lack of breadth implying signals arriving on fewer paths) with closed-form expressions. We prove that optimal graph structures for maximising higher-order homophily are disjoint unions of single-class and two-class-bipartite clusters. This yields BRIDGE, a graph ensemble-based rewiring algorithm that achieves near-perfect classification accuracy across all homophily regimes on synthetic benchmarks and significant improvements on real-world benchmarks, by eliminating the ``mid-homophily pitfall'' where MPNNs typically struggle, surpassing current standard rewiring techniques from the literature. Our framework, whose code we make available for public use, provides both diagnostic tools for assessing MPNN performance, and simple yet effective methods for enhancing performance through principled graph modification. 

---
# Limitations of Normalization in Attention Mechanism 

**Authors**: Timur Mudarisov, Mikhail Burtsev, Tatiana Petrova, Radu State  

**Link**: [PDF](https://arxiv.org/pdf/2508.17821)  

**Abstract**: This paper investigates the limitations of the normalization in attention mechanisms. We begin with a theoretical framework that enables the identification of the model's selective ability and the geometric separation involved in token selection. Our analysis includes explicit bounds on distances and separation criteria for token vectors under softmax scaling. Through experiments with pre-trained GPT-2 model, we empirically validate our theoretical results and analyze key behaviors of the attention mechanism. Notably, we demonstrate that as the number of selected tokens increases, the model's ability to distinguish informative tokens declines, often converging toward a uniform selection pattern. We also show that gradient sensitivity under softmax normalization presents challenges during training, especially at low temperature settings. These findings advance current understanding of softmax-based attention mechanism and motivate the need for more robust normalization and selection strategies in future attention architectures. 

---
# UniSino: Physics-Driven Foundational Model for Universal CT Sinogram Standardization 

**Authors**: Xingyu Ai, Shaoyu Wang, Zhiyuan Jia, Ao Xu, Hongming Shan, Jianhua Ma, Qiegen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.17816)  

**Abstract**: During raw-data acquisition in CT imaging, diverse factors can degrade the collected sinograms, with undersampling and noise leading to severe artifacts and noise in reconstructed images and compromising diagnostic accuracy. Conventional correction methods rely on manually designed algorithms or fixed empirical parameters, but these approaches often lack generalizability across heterogeneous artifact types. To address these limitations, we propose UniSino, a foundation model for universal CT sinogram standardization. Unlike existing foundational models that operate in image domain, UniSino directly standardizes data in the projection domain, which enables stronger generalization across diverse undersampling scenarios. Its training framework incorporates the physical characteristics of sinograms, enhancing generalization and enabling robust performance across multiple subtasks spanning four benchmark datasets. Experimental results demonstrate thatUniSino achieves superior reconstruction quality both single and mixed undersampling case, demonstrating exceptional robustness and generalization in sinogram enhancement for CT imaging. The code is available at: this https URL. 

---
# Scalable Engine and the Performance of Different LLM Models in a SLURM based HPC architecture 

**Authors**: Anderson de Lima Luiz, Shubham Vijay Kurlekar, Munir Georges  

**Link**: [PDF](https://arxiv.org/pdf/2508.17814)  

**Abstract**: This work elaborates on a High performance computing (HPC) architecture based on Simple Linux Utility for Resource Management (SLURM) [1] for deploying heterogeneous Large Language Models (LLMs) into a scalable inference engine. Dynamic resource scheduling and seamless integration of containerized microservices have been leveraged herein to manage CPU, GPU, and memory allocations efficiently in multi-node clusters. Extensive experiments, using Llama 3.2 (1B and 3B parameters) [2] and Llama 3.1 (8B and 70B) [3], probe throughput, latency, and concurrency and show that small models can handle up to 128 concurrent requests at sub-50 ms latency, while for larger models, saturation happens with as few as two concurrent users, with a latency of more than 2 seconds. This architecture includes Representational State Transfer Application Programming Interfaces (REST APIs) [4] endpoints for single and bulk inferences, as well as advanced workflows such as multi-step "tribunal" refinement. Experimental results confirm minimal overhead from container and scheduling activities and show that the approach scales reliably both for batch and interactive settings. We further illustrate real-world scenarios, including the deployment of chatbots with retrievalaugmented generation, which helps to demonstrate the flexibility and robustness of the architecture. The obtained results pave ways for significantly more efficient, responsive, and fault-tolerant LLM inference on large-scale HPC infrastructures. 

---
# MeshSplat: Generalizable Sparse-View Surface Reconstruction via Gaussian Splatting 

**Authors**: Hanzhi Chang, Ruijie Zhu, Wenjie Chang, Mulin Yu, Yanzhe Liang, Jiahao Lu, Zhuoyuan Li, Tianzhu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.17811)  

**Abstract**: Surface reconstruction has been widely studied in computer vision and graphics. However, existing surface reconstruction works struggle to recover accurate scene geometry when the input views are extremely sparse. To address this issue, we propose MeshSplat, a generalizable sparse-view surface reconstruction framework via Gaussian Splatting. Our key idea is to leverage 2DGS as a bridge, which connects novel view synthesis to learned geometric priors and then transfers these priors to achieve surface reconstruction. Specifically, we incorporate a feed-forward network to predict per-view pixel-aligned 2DGS, which enables the network to synthesize novel view images and thus eliminates the need for direct 3D ground-truth supervision. To improve the accuracy of 2DGS position and orientation prediction, we propose a Weighted Chamfer Distance Loss to regularize the depth maps, especially in overlapping areas of input views, and also a normal prediction network to align the orientation of 2DGS with normal vectors predicted by a monocular normal estimator. Extensive experiments validate the effectiveness of our proposed improvement, demonstrating that our method achieves state-of-the-art performance in generalizable sparse-view mesh reconstruction tasks. Project Page: this https URL 

---
# Adaptive Output Steps: FlexiSteps Network for Dynamic Trajectory Prediction 

**Authors**: Yunxiang Liu, Hongkuo Niu, Jianlin Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2508.17797)  

**Abstract**: Accurate trajectory prediction is vital for autonomous driving, robotics, and intelligent decision-making systems, yet traditional models typically rely on fixed-length output predictions, limiting their adaptability to dynamic real-world scenarios. In this paper, we introduce the FlexiSteps Network (FSN), a novel framework that dynamically adjusts prediction output time steps based on varying contextual conditions. Inspired by recent advancements addressing observation length discrepancies and dynamic feature extraction, FSN incorporates an pre-trained Adaptive Prediction Module (APM) to evaluate and adjust the output steps dynamically, ensuring optimal prediction accuracy and efficiency. To guarantee the plug-and-play of our FSN, we also design a Dynamic Decoder(DD). Additionally, to balance the prediction time steps and prediction accuracy, we design a scoring mechanism, which not only introduces the Fréchet distance to evaluate the geometric similarity between the predicted trajectories and the ground truth trajectories but the length of predicted steps is also considered. Extensive experiments conducted on benchmark datasets including Argoverse and INTERACTION demonstrate the effectiveness and flexibility of our proposed FSN framework. 

---
# Proximal Supervised Fine-Tuning 

**Authors**: Wenhong Zhu, Ruobing Xie, Rui Wang, Xingwu Sun, Di Wang, Pengfei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.17784)  

**Abstract**: Supervised fine-tuning (SFT) of foundation models often leads to poor generalization, where prior capabilities deteriorate after tuning on new tasks or domains. Inspired by trust-region policy optimization (TRPO) and proximal policy optimization (PPO) in reinforcement learning (RL), we propose Proximal SFT (PSFT). This fine-tuning objective incorporates the benefits of trust-region, effectively constraining policy drift during SFT while maintaining competitive tuning. By viewing SFT as a special case of policy gradient methods with constant positive advantages, we derive PSFT that stabilizes optimization and leads to generalization, while leaving room for further optimization in subsequent post-training stages. Experiments across mathematical and human-value domains show that PSFT matches SFT in-domain, outperforms it in out-of-domain generalization, remains stable under prolonged training without causing entropy collapse, and provides a stronger foundation for the subsequent optimization. 

---
# Algebraic Approach to Ridge-Regularized Mean Squared Error Minimization in Minimal ReLU Neural Network 

**Authors**: Ryoya Fukasaku, Yutaro Kabata, Akifumi Okuno  

**Link**: [PDF](https://arxiv.org/pdf/2508.17783)  

**Abstract**: This paper investigates a perceptron, a simple neural network model, with ReLU activation and a ridge-regularized mean squared error (RR-MSE). Our approach leverages the fact that the RR-MSE for ReLU perceptron is piecewise polynomial, enabling a systematic analysis using tools from computational algebra. In particular, we develop a Divide-Enumerate-Merge strategy that exhaustively enumerates all local minima of the RR-MSE. By virtue of the algebraic formulation, our approach can identify not only the typical zero-dimensional minima (i.e., isolated points) obtained by numerical optimization, but also higher-dimensional minima (i.e., connected sets such as curves, surfaces, or hypersurfaces). Although computational algebraic methods are computationally very intensive for perceptrons of practical size, as a proof of concept, we apply the proposed approach in practice to minimal perceptrons with a few hidden units. 

---
# DiffusionGS: Generative Search with Query Conditioned Diffusion in Kuaishou 

**Authors**: Qinyao Li, Xiaoyang Zheng, Qihang Zhao, Ke Xu, Zhongbo Sun, Chao Wang, Chenyi Lei, Han Li, Wenwu Ou  

**Link**: [PDF](https://arxiv.org/pdf/2508.17754)  

**Abstract**: Personalized search ranking systems are critical for driving engagement and revenue in modern e-commerce and short-video platforms. While existing methods excel at estimating users' broad interests based on the filtered historical behaviors, they typically under-exploit explicit alignment between a user's real-time intent (represented by the user query) and their past actions. In this paper, we propose DiffusionGS, a novel and scalable approach powered by generative models. Our key insight is that user queries can serve as explicit intent anchors to facilitate the extraction of users' immediate interests from long-term, noisy historical behaviors. Specifically, we formulate interest extraction as a conditional denoising task, where the user's query guides a conditional diffusion process to produce a robust, user intent-aware representation from their behavioral sequence. We propose the User-aware Denoising Layer (UDL) to incorporate user-specific profiles into the optimization of attention distribution on the user's past actions. By reframing queries as intent priors and leveraging diffusion-based denoising, our method provides a powerful mechanism for capturing dynamic user interest shifts. Extensive offline and online experiments demonstrate the superiority of DiffusionGS over state-of-the-art methods. 

---
# Talking to Robots: A Practical Examination of Speech Foundation Models for HRI Applications 

**Authors**: Theresa Pekarek Rosin, Julia Gachot, Henri-Leon Kordt, Matthias Kerzel, Stefan Wermter  

**Link**: [PDF](https://arxiv.org/pdf/2508.17753)  

**Abstract**: Automatic Speech Recognition (ASR) systems in real-world settings need to handle imperfect audio, often degraded by hardware limitations or environmental noise, while accommodating diverse user groups. In human-robot interaction (HRI), these challenges intersect to create a uniquely challenging recognition environment. We evaluate four state-of-the-art ASR systems on eight publicly available datasets that capture six dimensions of difficulty: domain-specific, accented, noisy, age-variant, impaired, and spontaneous speech. Our analysis demonstrates significant variations in performance, hallucination tendencies, and inherent biases, despite similar scores on standard benchmarks. These limitations have serious implications for HRI, where recognition errors can interfere with task performance, user trust, and safety. 

---
# EEG-FM-Bench: A Comprehensive Benchmark for the Systematic Evaluation of EEG Foundation Models 

**Authors**: Wei Xiong, Jiangtong Li, Jie Li, Kun Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2508.17742)  

**Abstract**: Electroencephalography (EEG) foundation models are poised to significantly advance brain signal analysis by learning robust representations from large-scale, unlabeled datasets. However, their rapid proliferation has outpaced the development of standardized evaluation benchmarks, which complicates direct model comparisons and hinders systematic scientific progress. This fragmentation fosters scientific inefficiency and obscures genuine architectural advancements. To address this critical gap, we introduce EEG-FM-Bench, the first comprehensive benchmark for the systematic and standardized evaluation of EEG foundation models (EEG-FMs). Our contributions are threefold: (1) we curate a diverse suite of downstream tasks and datasets from canonical EEG paradigms, implementing standardized processing and evaluation protocols within a unified open-source framework; (2) we benchmark prominent state-of-the-art foundation models to establish comprehensive baseline results for a clear comparison of the current landscape; (3) we perform qualitative analyses of the learned representations to provide insights into model behavior and inform future architectural design. Through extensive experiments, we find that fine-grained spatio-temporal feature interaction, multitask unified training and neuropsychological priors would contribute to enhancing model performance and generalization capabilities. By offering a unified platform for fair comparison and reproducible research, EEG-FM-Bench seeks to catalyze progress and guide the community toward the development of more robust and generalizable EEG-FMs. Code is released at this https URL. 

---
# Speculative Safety-Aware Decoding 

**Authors**: Xuekang Wang, Shengyu Zhu, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2508.17739)  

**Abstract**: Despite extensive efforts to align Large Language Models (LLMs) with human values and safety rules, jailbreak attacks that exploit certain vulnerabilities continuously emerge, highlighting the need to strengthen existing LLMs with additional safety properties to defend against these attacks. However, tuning large models has become increasingly resource-intensive and may have difficulty ensuring consistent performance. We introduce Speculative Safety-Aware Decoding (SSD), a lightweight decoding-time approach that equips LLMs with the desired safety property while accelerating inference. We assume that there exists a small language model that possesses this desired property. SSD integrates speculative sampling during decoding and leverages the match ratio between the small and composite models to quantify jailbreak risks. This enables SSD to dynamically switch between decoding schemes to prioritize utility or safety, to handle the challenge of different model capacities. The output token is then sampled from a new distribution that combines the distributions of the original and the small models. Experimental results show that SSD successfully equips the large model with the desired safety property, and also allows the model to remain helpful to benign queries. Furthermore, SSD accelerates the inference time, thanks to the speculative sampling design. 

---
# Instant Preference Alignment for Text-to-Image Diffusion Models 

**Authors**: Yang Li, Songlin Yang, Xiaoxuan Han, Wei Wang, Jing Dong, Yueming Lyu, Ziyu Xue  

**Link**: [PDF](https://arxiv.org/pdf/2508.17718)  

**Abstract**: Text-to-image (T2I) generation has greatly enhanced creative expression, yet achieving preference-aligned generation in a real-time and training-free manner remains challenging. Previous methods often rely on static, pre-collected preferences or fine-tuning, limiting adaptability to evolving and nuanced user intents. In this paper, we highlight the need for instant preference-aligned T2I generation and propose a training-free framework grounded in multimodal large language model (MLLM) priors. Our framework decouples the task into two components: preference understanding and preference-guided generation. For preference understanding, we leverage MLLMs to automatically extract global preference signals from a reference image and enrich a given prompt using structured instruction design. Our approach supports broader and more fine-grained coverage of user preferences than existing methods. For preference-guided generation, we integrate global keyword-based control and local region-aware cross-attention modulation to steer the diffusion model without additional training, enabling precise alignment across both global attributes and local elements. The entire framework supports multi-round interactive refinement, facilitating real-time and context-aware image generation. Extensive experiments on the Viper dataset and our collected benchmark demonstrate that our method outperforms prior approaches in both quantitative metrics and human evaluations, and opens up new possibilities for dialog-based generation and MLLM-diffusion integration. 

---
# Database Normalization via Dual-LLM Self-Refinement 

**Authors**: Eunjae Jo, Nakyung Lee, Gyuyeong Kim  

**Link**: [PDF](https://arxiv.org/pdf/2508.17693)  

**Abstract**: Database normalization is crucial to preserving data integrity. However, it is time-consuming and error-prone, as it is typically performed manually by data engineers. To this end, we present Miffie, a database normalization framework that leverages the capability of large language models. Miffie enables automated data normalization without human effort while preserving high accuracy. The core of Miffie is a dual-model self-refinement architecture that combines the best-performing models for normalized schema generation and verification, respectively. The generation module eliminates anomalies based on the feedback of the verification module until the output schema satisfies the requirement for normalization. We also carefully design task-specific zero-shot prompts to guide the models for achieving both high accuracy and cost efficiency. Experimental results show that Miffie can normalize complex database schemas while maintaining high accuracy. 

---
# Unlearning as Ablation: Toward a Falsifiable Benchmark for Generative Scientific Discovery 

**Authors**: Robert Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.17681)  

**Abstract**: Bold claims about AI's role in science-from "AGI will cure all diseases" to promises of radically accelerated discovery-raise a central epistemic question: do large language models (LLMs) truly generate new knowledge, or do they merely remix memorized fragments? We propose unlearning-as-ablation as a falsifiable test of constructive scientific discovery. The method systematically removes a target result and its entire forget-closure (lemmas, paraphrases, and multi-hop entailments) and then evaluates whether the model can re-derive the result from only permitted axioms and tools. Success provides evidence for genuine generative capability; failure exposes current limits. Unlike prevailing motivations for unlearning-privacy, copyright, or safety-our framing repositions it as an epistemic probe for AI-for-Science. We argue that such tests could serve as the next generation of benchmarks, much as ImageNet catalyzed progress in vision: distinguishing models that can merely recall from those that can constructively generate new scientific knowledge. We outline a minimal pilot in mathematics and algorithms, and discuss extensions to physics, chemistry, and biology. Whether models succeed or fail, unlearning-as-ablation provides a principled framework to map the true reach and limits of AI scientific discovery. This is a position paper: we advance a conceptual and methodological argument rather than new empirical results. 

---
# Robustness Feature Adapter for Efficient Adversarial Training 

**Authors**: Quanwei Wu, Jun Guo, Wei Wang, Yi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.17680)  

**Abstract**: Adversarial training (AT) with projected gradient descent is the most popular method to improve model robustness under adversarial attacks. However, computational overheads become prohibitively large when AT is applied to large backbone models. AT is also known to have the issue of robust overfitting. This paper contributes to solving both problems simultaneously towards building more trustworthy foundation models. In particular, we propose a new adapter-based approach for efficient AT directly in the feature space. We show that the proposed adapter-based approach can improve the inner-loop convergence quality by eliminating robust overfitting. As a result, it significantly increases computational efficiency and improves model accuracy by generalizing adversarial robustness to unseen attacks. We demonstrate the effectiveness of the new adapter-based approach in different backbone architectures and in AT at scale. 

---
# Attacking LLMs and AI Agents: Advertisement Embedding Attacks Against Large Language Models 

**Authors**: Qiming Guo, Jinwen Tang, Xingran Huang  

**Link**: [PDF](https://arxiv.org/pdf/2508.17674)  

**Abstract**: We introduce Advertisement Embedding Attacks (AEA), a new class of LLM security threats that stealthily inject promotional or malicious content into model outputs and AI agents. AEA operate through two low-cost vectors: (1) hijacking third-party service-distribution platforms to prepend adversarial prompts, and (2) publishing back-doored open-source checkpoints fine-tuned with attacker data. Unlike conventional attacks that degrade accuracy, AEA subvert information integrity, causing models to return covert ads, propaganda, or hate speech while appearing normal. We detail the attack pipeline, map five stakeholder victim groups, and present an initial prompt-based self-inspection defense that mitigates these injections without additional model retraining. Our findings reveal an urgent, under-addressed gap in LLM security and call for coordinated detection, auditing, and policy responses from the AI-safety community. 

---
# Consistent Opponent Modeling of Static Opponents in Imperfect-Information Games 

**Authors**: Sam Ganzfried  

**Link**: [PDF](https://arxiv.org/pdf/2508.17671)  

**Abstract**: The goal of agents in multi-agent environments is to maximize total reward against the opposing agents that are encountered. Following a game-theoretic solution concept, such as Nash equilibrium, may obtain a strong performance in some settings; however, such approaches fail to capitalize on historical and observed data from repeated interactions against our opponents. Opponent modeling algorithms integrate machine learning techniques to exploit suboptimal opponents utilizing available data; however, the effectiveness of such approaches in imperfect-information games to date is quite limited. We show that existing opponent modeling approaches fail to satisfy a simple desirable property even against static opponents drawn from a known prior distribution; namely, they do not guarantee that the model approaches the opponent's true strategy even in the limit as the number of game iterations approaches infinity. We develop a new algorithm that is able to achieve this property and runs efficiently by solving a convex minimization problem based on the sequence-form game representation using projected gradient descent. The algorithm is guaranteed to efficiently converge to the opponent's true strategy given observations from gameplay and possibly additional historical data if it is available. 

---
# Hierarchical Vision-Language Learning for Medical Out-of-Distribution Detection 

**Authors**: Runhe Lai, Xinhua Lu, Kanghao Chen, Qichao Chen, Wei-Shi Zheng, Ruixuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.17667)  

**Abstract**: In trustworthy medical diagnosis systems, integrating out-of-distribution (OOD) detection aims to identify unknown diseases in samples, thereby mitigating the risk of misdiagnosis. In this study, we propose a novel OOD detection framework based on vision-language models (VLMs), which integrates hierarchical visual information to cope with challenging unknown diseases that resemble known diseases. Specifically, a cross-scale visual fusion strategy is proposed to couple visual embeddings from multiple scales. This enriches the detailed representation of medical images and thus improves the discrimination of unknown diseases. Moreover, a cross-scale hard pseudo-OOD sample generation strategy is proposed to benefit OOD detection maximally. Experimental evaluations on three public medical datasets support that the proposed framework achieves superior OOD detection performance compared to existing methods. The source code is available at this https URL. 

---
# Weights-Rotated Preference Optimization for Large Language Models 

**Authors**: Chenxu Yang, Ruipeng Jia, Mingyu Zheng, Naibin Gu, Zheng Lin, Siyuan Chen, Weichong Yin, Hua Wu, Weiping Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.17637)  

**Abstract**: Despite the efficacy of Direct Preference Optimization (DPO) in aligning Large Language Models (LLMs), reward hacking remains a pivotal challenge. This issue emerges when LLMs excessively reduce the probability of rejected completions to achieve high rewards, without genuinely meeting their intended goals. As a result, this leads to overly lengthy generation lacking diversity, as well as catastrophic forgetting of knowledge. We investigate the underlying reason behind this issue, which is representation redundancy caused by neuron collapse in the parameter space. Hence, we propose a novel Weights-Rotated Preference Optimization (RoPO) algorithm, which implicitly constrains the output layer logits with the KL divergence inherited from DPO and explicitly constrains the intermediate hidden states by fine-tuning on a multi-granularity orthogonal matrix. This design prevents the policy model from deviating too far from the reference model, thereby retaining the knowledge and expressive capabilities acquired during pre-training and SFT stages. Our RoPO achieves up to a 3.27-point improvement on AlpacaEval 2, and surpasses the best baseline by 6.2 to 7.5 points on MT-Bench with merely 0.015% of the trainable parameters, demonstrating its effectiveness in alleviating the reward hacking problem of DPO. 

---
# Few-Shot Pattern Detection via Template Matching and Regression 

**Authors**: Eunchan Jo, Dahyun Kang, Sanghyun Kim, Yunseon Choi, Minsu Cho  

**Link**: [PDF](https://arxiv.org/pdf/2508.17636)  

**Abstract**: We address the problem of few-shot pattern detection, which aims to detect all instances of a given pattern, typically represented by a few exemplars, from an input image. Although similar problems have been studied in few-shot object counting and detection (FSCD), previous methods and their benchmarks have narrowed patterns of interest to object categories and often fail to localize non-object patterns. In this work, we propose a simple yet effective detector based on template matching and regression, dubbed TMR. While previous FSCD methods typically represent target exemplars as spatially collapsed prototypes and lose structural information, we revisit classic template matching and regression. It effectively preserves and leverages the spatial layout of exemplars through a minimalistic structure with a small number of learnable convolutional or projection layers on top of a frozen backbone We also introduce a new dataset, dubbed RPINE, which covers a wider range of patterns than existing object-centric datasets. Our method outperforms the state-of-the-art methods on the three benchmarks, RPINE, FSCD-147, and FSCD-LVIS, and demonstrates strong generalization in cross-dataset evaluation. 

---
# Finding Outliers in a Haystack: Anomaly Detection for Large Pointcloud Scenes 

**Authors**: Ryan Faulkner, Ian Reid, Simon Ratcliffe, Tat-Jun Chin  

**Link**: [PDF](https://arxiv.org/pdf/2508.17634)  

**Abstract**: LiDAR scanning in outdoor scenes acquires accurate distance measurements over wide areas, producing large-scale point clouds. Application examples for this data include robotics, automotive vehicles, and land surveillance. During such applications, outlier objects from outside the training data will inevitably appear. Our research contributes a novel approach to open-set segmentation, leveraging the learnings of object defect-detection research. We also draw on the Mamba architecture's strong performance in utilising long-range dependencies and scalability to large data. Combining both, we create a reconstruction based approach for the task of outdoor scene open-set segmentation. We show that our approach improves performance not only when applied to our our own open-set segmentation method, but also when applied to existing methods. Furthermore we contribute a Mamba based architecture which is competitive with existing voxel-convolution based methods on challenging, large-scale pointclouds. 

---
# ControlEchoSynth: Boosting Ejection Fraction Estimation Models via Controlled Video Diffusion 

**Authors**: Nima Kondori, Hanwen Liang, Hooman Vaseli, Bingyu Xie, Christina Luong, Purang Abolmaesumi, Teresa Tsang, Renjie Liao  

**Link**: [PDF](https://arxiv.org/pdf/2508.17631)  

**Abstract**: Synthetic data generation represents a significant advancement in boosting the performance of machine learning (ML) models, particularly in fields where data acquisition is challenging, such as echocardiography. The acquisition and labeling of echocardiograms (echo) for heart assessment, crucial in point-of-care ultrasound (POCUS) settings, often encounter limitations due to the restricted number of echo views available, typically captured by operators with varying levels of experience. This study proposes a novel approach for enhancing clinical diagnosis accuracy by synthetically generating echo views. These views are conditioned on existing, real views of the heart, focusing specifically on the estimation of ejection fraction (EF), a critical parameter traditionally measured from biplane apical views. By integrating a conditional generative model, we demonstrate an improvement in EF estimation accuracy, providing a comparative analysis with traditional methods. Preliminary results indicate that our synthetic echoes, when used to augment existing datasets, not only enhance EF estimation but also show potential in advancing the development of more robust, accurate, and clinically relevant ML models. This approach is anticipated to catalyze further research in synthetic data applications, paving the way for innovative solutions in medical imaging diagnostics. 

---
# Stop Spinning Wheels: Mitigating LLM Overthinking via Mining Patterns for Early Reasoning Exit 

**Authors**: Zihao Wei, Liang Pang, Jiahao Liu, Jingcheng Deng, Shicheng Xu, Zenghao Duan, Jingang Wang, Fei Sun, Xunliang Cai, Huawei Shen, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2508.17627)  

**Abstract**: Large language models (LLMs) enhance complex reasoning tasks by scaling the individual thinking process. However, prior work shows that overthinking can degrade overall performance. Motivated by observed patterns in thinking length and content length, we categorize reasoning into three stages: insufficient exploration stage, compensatory reasoning stage, and reasoning convergence stage. Typically, LLMs produce correct answers in the compensatory reasoning stage, whereas reasoning convergence often triggers overthinking, causing increased resource usage or even infinite loops. Therefore, mitigating overthinking hinges on detecting the end of the compensatory reasoning stage, defined as the Reasoning Completion Point (RCP). RCP typically appears at the end of the first complete reasoning cycle and can be identified by querying the LLM sentence by sentence or monitoring the probability of an end-of-thinking token (e.g., \texttt{</think>}), though these methods lack an efficient and precise balance. To improve this, we mine more sensitive and consistent RCP patterns and develop a lightweight thresholding strategy based on heuristic rules. Experimental evaluations on benchmarks (AIME24, AIME25, GPQA-D) demonstrate that the proposed method reduces token consumption while preserving or enhancing reasoning accuracy. 

---
# Steering When Necessary: Flexible Steering Large Language Models with Backtracking 

**Authors**: Jinwei Gan, Zifeng Cheng, Zhiwei Jiang, Cong Wang, Yafeng Yin, Xiang Luo, Yuchen Fu, Qing Gu  

**Link**: [PDF](https://arxiv.org/pdf/2508.17621)  

**Abstract**: Large language models (LLMs) have achieved remarkable performance across many generation tasks. Nevertheless, effectively aligning them with desired behaviors remains a significant challenge. Activation steering is an effective and cost-efficient approach that directly modifies the activations of LLMs during the inference stage, aligning their responses with the desired behaviors and avoiding the high cost of fine-tuning. Existing methods typically indiscriminately intervene to all generations or rely solely on the question to determine intervention, which limits the accurate assessment of the intervention strength. To this end, we propose the Flexible Activation Steering with Backtracking (FASB) framework, which dynamically determines both the necessity and strength of intervention by tracking the internal states of the LLMs during generation, considering both the question and the generated content. Since intervening after detecting a deviation from the desired behavior is often too late, we further propose the backtracking mechanism to correct the deviated tokens and steer the LLMs toward the desired behavior. Extensive experiments on the TruthfulQA dataset and six multiple-choice datasets demonstrate that our method outperforms baselines. Our code will be released at this https URL. 

---
# GWM: Towards Scalable Gaussian World Models for Robotic Manipulation 

**Authors**: Guanxing Lu, Baoxiong Jia, Puhao Li, Yixin Chen, Ziwei Wang, Yansong Tang, Siyuan Huang  

**Link**: [PDF](https://arxiv.org/pdf/2508.17600)  

**Abstract**: Training robot policies within a learned world model is trending due to the inefficiency of real-world interactions. The established image-based world models and policies have shown prior success, but lack robust geometric information that requires consistent spatial and physical understanding of the three-dimensional world, even pre-trained on internet-scale video sources. To this end, we propose a novel branch of world model named Gaussian World Model (GWM) for robotic manipulation, which reconstructs the future state by inferring the propagation of Gaussian primitives under the effect of robot actions. At its core is a latent Diffusion Transformer (DiT) combined with a 3D variational autoencoder, enabling fine-grained scene-level future state reconstruction with Gaussian Splatting. GWM can not only enhance the visual representation for imitation learning agent by self-supervised future prediction training, but can serve as a neural simulator that supports model-based reinforcement learning. Both simulated and real-world experiments depict that GWM can precisely predict future scenes conditioned on diverse robot actions, and can be further utilized to train policies that outperform the state-of-the-art by impressive margins, showcasing the initial data scaling potential of 3D world model. 

---
# RubikSQL: Lifelong Learning Agentic Knowledge Base as an Industrial NL2SQL System 

**Authors**: Zui Chen, Han Li, Xinhao Zhang, Xiaoyu Chen, Chunyin Dong, Yifeng Wang, Xin Cai, Su Zhang, Ziqi Li, Chi Ding, Jinxu Li, Shuai Wang, Dousheng Zhao, Sanhai Gao, Guangyi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.17590)  

**Abstract**: We present RubikSQL, a novel NL2SQL system designed to address key challenges in real-world enterprise-level NL2SQL, such as implicit intents and domain-specific terminology. RubikSQL frames NL2SQL as a lifelong learning task, demanding both Knowledge Base (KB) maintenance and SQL generation. RubikSQL systematically builds and refines its KB through techniques including database profiling, structured information extraction, agentic rule mining, and Chain-of-Thought (CoT)-enhanced SQL profiling. RubikSQL then employs a multi-agent workflow to leverage this curated KB, generating accurate SQLs. RubikSQL achieves SOTA performance on both the KaggleDBQA and BIRD Mini-Dev datasets. Finally, we release the RubikBench benchmark, a new benchmark specifically designed to capture vital traits of industrial NL2SQL scenarios, providing a valuable resource for future research. 

---
# UQ: Assessing Language Models on Unsolved Questions 

**Authors**: Fan Nie, Ken Ziyu Liu, Zihao Wang, Rui Sun, Wei Liu, Weijia Shi, Huaxiu Yao, Linjun Zhang, Andrew Y. Ng, James Zou, Sanmi Koyejo, Yejin Choi, Percy Liang, Niklas Muennighoff  

**Link**: [PDF](https://arxiv.org/pdf/2508.17580)  

**Abstract**: Benchmarks shape progress in AI research. A useful benchmark should be both difficult and realistic: questions should challenge frontier models while also reflecting real-world usage. Yet, current paradigms face a difficulty-realism tension: exam-style benchmarks are often made artificially difficult with limited real-world value, while benchmarks based on real user interaction often skew toward easy, high-frequency problems. In this work, we explore a radically different paradigm: assessing models on unsolved questions. Rather than a static benchmark scored once, we curate unsolved questions and evaluate models asynchronously over time with validator-assisted screening and community verification. We introduce UQ, a testbed of 500 challenging, diverse questions sourced from Stack Exchange, spanning topics from CS theory and math to sci-fi and history, probing capabilities including reasoning, factuality, and browsing. UQ is difficult and realistic by construction: unsolved questions are often hard and naturally arise when humans seek answers, thus solving them yields direct real-world value. Our contributions are threefold: (1) UQ-Dataset and its collection pipeline combining rule-based filters, LLM judges, and human review to ensure question quality (e.g., well-defined and difficult); (2) UQ-Validators, compound validation strategies that leverage the generator-validator gap to provide evaluation signals and pre-screen candidate solutions for human review; and (3) UQ-Platform, an open platform where experts collectively verify questions and solutions. The top model passes UQ-validation on only 15% of questions, and preliminary human verification has already identified correct answers among those that passed. UQ charts a path for evaluating frontier models on real-world, open-ended challenges, where success pushes the frontier of human knowledge. We release UQ at this https URL. 

---
# MetaGen: A DSL, Database, and Benchmark for VLM-Assisted Metamaterial Generation 

**Authors**: Liane Makatura, Benjamin Jones, Siyuan Bian, Wojciech Matusik  

**Link**: [PDF](https://arxiv.org/pdf/2508.17568)  

**Abstract**: Metamaterials are micro-architected structures whose geometry imparts highly tunable-often counter-intuitive-bulk properties. Yet their design is difficult because of geometric complexity and a non-trivial mapping from architecture to behaviour. We address these challenges with three complementary contributions. (i) MetaDSL: a compact, semantically rich domain-specific language that captures diverse metamaterial designs in a form that is both human-readable and machine-parsable. (ii) MetaDB: a curated repository of more than 150,000 parameterized MetaDSL programs together with their derivatives-three-dimensional geometry, multi-view renderings, and simulated elastic properties. (iii) MetaBench: benchmark suites that test three core capabilities of vision-language metamaterial assistants-structure reconstruction, property-driven inverse design, and performance prediction. We establish baselines by fine-tuning state-of-the-art vision-language models and deploy an omni-model within an interactive, CAD-like interface. Case studies show that our framework provides a strong first step toward integrated design and understanding of structure-representation-property relationships. 

---
# In-Context Algorithm Emulation in Fixed-Weight Transformers 

**Authors**: Jerry Yao-Chieh Hu, Hude Liu, Jennifer Yuntong Zhang, Han Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.17550)  

**Abstract**: We prove that a minimal Transformer architecture with frozen weights is capable of emulating a broad class of algorithms by in-context prompting. In particular, for any algorithm implementable by a fixed-weight attention head (e.g. one-step gradient descent or linear/ridge regression), there exists a prompt that drives a two-layer softmax attention module to reproduce the algorithm's output with arbitrary precision. This guarantee extends even to a single-head attention layer (using longer prompts if necessary), achieving architectural minimality. Our key idea is to construct prompts that encode an algorithm's parameters into token representations, creating sharp dot-product gaps that force the softmax attention to follow the intended computation. This construction requires no feed-forward layers and no parameter updates. All adaptation happens through the prompt alone. These findings forge a direct link between in-context learning and algorithmic emulation, and offer a simple mechanism for large Transformers to serve as prompt-programmable libraries of algorithms. They illuminate how GPT-style foundation models may swap algorithms via prompts alone, establishing a form of algorithmic universality in modern Transformer models. 

---
# LodeStar: Long-horizon Dexterity via Synthetic Data Augmentation from Human Demonstrations 

**Authors**: Weikang Wan, Jiawei Fu, Xiaodi Yuan, Yifeng Zhu, Hao Su  

**Link**: [PDF](https://arxiv.org/pdf/2508.17547)  

**Abstract**: Developing robotic systems capable of robustly executing long-horizon manipulation tasks with human-level dexterity is challenging, as such tasks require both physical dexterity and seamless sequencing of manipulation skills while robustly handling environment variations. While imitation learning offers a promising approach, acquiring comprehensive datasets is resource-intensive. In this work, we propose a learning framework and system LodeStar that automatically decomposes task demonstrations into semantically meaningful skills using off-the-shelf foundation models, and generates diverse synthetic demonstration datasets from a few human demos through reinforcement learning. These sim-augmented datasets enable robust skill training, with a Skill Routing Transformer (SRT) policy effectively chaining the learned skills together to execute complex long-horizon manipulation tasks. Experimental evaluations on three challenging real-world long-horizon dexterous manipulation tasks demonstrate that our approach significantly improves task performance and robustness compared to previous baselines. Videos are available at this http URL. 

---
# Activation Transport Operators 

**Authors**: Andrzej Szablewski, Marek Masiak  

**Link**: [PDF](https://arxiv.org/pdf/2508.17540)  

**Abstract**: The residual stream mediates communication between transformer decoder layers via linear reads and writes of non-linear computations. While sparse-dictionary learning-based methods locate features in the residual stream, and activation patching methods discover circuits within the model, the mechanism by which features flow through the residual stream remains understudied. Understanding this dynamic can better inform jailbreaking protections, enable early detection of model mistakes, and their correction. In this work, we propose Activation Transport Operators (ATO), linear maps from upstream to downstream residuals $k$ layers later, evaluated in feature space using downstream SAE decoder projections. We empirically demonstrate that these operators can determine whether a feature has been linearly transported from a previous layer or synthesised from non-linear layer computation. We develop the notion of transport efficiency, for which we provide an upper bound, and use it to estimate the size of the residual stream subspace that corresponds to linear transport. We empirically demonstrate the linear transport, report transport efficiency and the size of the residual stream's subspace involved in linear transport. This compute-light (no finetuning, <50 GPU-h) method offers practical tools for safety, debugging, and a clearer picture of where computation in LLMs behaves linearly. 

---
# OmniMRI: A Unified Vision--Language Foundation Model for Generalist MRI Interpretation 

**Authors**: Xingxin He, Aurora Rofena, Ruimin Feng, Haozhe Liao, Zhaoye Zhou, Albert Jang, Fang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.17524)  

**Abstract**: Magnetic Resonance Imaging (MRI) is indispensable in clinical practice but remains constrained by fragmented, multi-stage workflows encompassing acquisition, reconstruction, segmentation, detection, diagnosis, and reporting. While deep learning has achieved progress in individual tasks, existing approaches are often anatomy- or application-specific and lack generalizability across diverse clinical settings. Moreover, current pipelines rarely integrate imaging data with complementary language information that radiologists rely on in routine practice. Here, we introduce OmniMRI, a unified vision-language foundation model designed to generalize across the entire MRI workflow. OmniMRI is trained on a large-scale, heterogeneous corpus curated from 60 public datasets, over 220,000 MRI volumes and 19 million MRI slices, incorporating image-only data, paired vision-text data, and instruction-response data. Its multi-stage training paradigm, comprising self-supervised vision pretraining, vision-language alignment, multimodal pretraining, and multi-task instruction tuning, progressively equips the model with transferable visual representations, cross-modal reasoning, and robust instruction-following capabilities. Qualitative results demonstrate OmniMRI's ability to perform diverse tasks within a single architecture, including MRI reconstruction, anatomical and pathological segmentation, abnormality detection, diagnostic suggestion, and radiology report generation. These findings highlight OmniMRI's potential to consolidate fragmented pipelines into a scalable, generalist framework, paving the way toward foundation models that unify imaging and clinical language for comprehensive, end-to-end MRI interpretation. 

---
# An experimental approach: The graph of graphs 

**Authors**: Zsombor Szádoczki, Sándor Bozóki, László Sipos, Zsófia Galambosi  

**Link**: [PDF](https://arxiv.org/pdf/2508.17520)  

**Abstract**: One of the essential issues in decision problems and preference modeling is the number of comparisons and their pattern to ask from the decision maker. We focus on the optimal patterns of pairwise comparisons and the sequence including the most (close to) optimal cases based on the results of a color selection experiment. In the test, six colors (red, green, blue, magenta, turquoise, yellow) were evaluated with pairwise comparisons as well as in a direct manner, on color-calibrated tablets in ISO standardized sensory test booths of a sensory laboratory. All the possible patterns of comparisons resulting in a connected representing graph were evaluated against the complete data based on 301 individual's pairwise comparison matrices (PCMs) using the logarithmic least squares weight calculation technique. It is shown that the empirical results, i.e., the empirical distributions of the elements of PCMs, are quite similar to the former simulated outcomes from the literature. The obtained empirically optimal patterns of comparisons were the best or the second best in the former simulations as well, while the sequence of comparisons that contains the most (close to) optimal patterns is exactly the same. In order to enhance the applicability of the results, besides the presentation of graph of graphs, and the representing graphs of the patterns that describe the proposed sequence of comparisons themselves, the recommendations are also detailed in a table format as well as in a Java application. 

---
# TANDEM: Temporal Attention-guided Neural Differential Equations for Missingness in Time Series Classification 

**Authors**: YongKyung Oh, Dong-Young Lim, Sungil Kim, Alex Bui  

**Link**: [PDF](https://arxiv.org/pdf/2508.17519)  

**Abstract**: Handling missing data in time series classification remains a significant challenge in various domains. Traditional methods often rely on imputation, which may introduce bias or fail to capture the underlying temporal dynamics. In this paper, we propose TANDEM (Temporal Attention-guided Neural Differential Equations for Missingness), an attention-guided neural differential equation framework that effectively classifies time series data with missing values. Our approach integrates raw observation, interpolated control path, and continuous latent dynamics through a novel attention mechanism, allowing the model to focus on the most informative aspects of the data. We evaluate TANDEM on 30 benchmark datasets and a real-world medical dataset, demonstrating its superiority over existing state-of-the-art methods. Our framework not only improves classification accuracy but also provides insights into the handling of missing data, making it a valuable tool in practice. 

---
# DinoTwins: Combining DINO and Barlow Twins for Robust, Label-Efficient Vision Transformers 

**Authors**: Michael Podsiadly, Brendon K Lay  

**Link**: [PDF](https://arxiv.org/pdf/2508.17509)  

**Abstract**: Training AI models to understand images without costly labeled data remains a challenge. We combine two techniques--DINO (teacher-student learning) and Barlow Twins (redundancy reduction)--to create a model that learns better with fewer labels and less compute. While both DINO and Barlow Twins have independently demonstrated strong performance in self-supervised learning, each comes with limitations--DINO may be sensitive to certain augmentations, and Barlow Twins often requires batch sizes too large to fit on consumer hardware. By combining the redundancy-reduction objective of Barlow Twins with the self-distillation strategy of DINO, we aim to leverage their complementary strengths. We train a hybrid model on the MS COCO dataset using only 10\% of labeled data for linear probing, and evaluate its performance against standalone DINO and Barlow Twins implementations. Preliminary results show that the combined approach achieves comparable loss and classification accuracy to DINO while maintaining strong feature representations. Attention visualizations further suggest improved semantic segmentation capability in the hybrid model. This combined method offers a scalable, label-efficient alternative for training ViTs in resource-constrained environments. 

---
# Multimodal Representation Learning Conditioned on Semantic Relations 

**Authors**: Yang Qiao, Yuntong Hu, Liang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2508.17497)  

**Abstract**: Multimodal representation learning has advanced rapidly with contrastive models such as CLIP, which align image-text pairs in a shared embedding space. However, these models face limitations: (1) they typically focus on image-text pairs, underutilizing the semantic relations across different pairs. (2) they directly match global embeddings without contextualization, overlooking the need for semantic alignment along specific subspaces or relational dimensions; and (3) they emphasize cross-modal contrast, with limited support for intra-modal consistency. To address these issues, we propose Relation-Conditioned Multimodal Learning RCML, a framework that learns multimodal representations under natural-language relation descriptions to guide both feature extraction and alignment. Our approach constructs many-to-many training pairs linked by semantic relations and introduces a relation-guided cross-attention mechanism that modulates multimodal representations under each relation context. The training objective combines inter-modal and intra-modal contrastive losses, encouraging consistency across both modalities and semantically related samples. Experiments on different datasets show that RCML consistently outperforms strong baselines on both retrieval and classification tasks, highlighting the effectiveness of leveraging semantic relations to guide multimodal representation learning. 

---
# A Synthetic Dataset for Manometry Recognition in Robotic Applications 

**Authors**: Pedro Antonio Rabelo Saraiva, Enzo Ferreira de Souza, Joao Manoel Herrera Pinheiro, Thiago H. Segreto, Ricardo V. Godoy, Marcelo Becker  

**Link**: [PDF](https://arxiv.org/pdf/2508.17468)  

**Abstract**: This work addresses the challenges of data scarcity and high acquisition costs for training robust object detection models in complex industrial environments, such as offshore oil platforms. The practical and economic barriers to collecting real-world data in these hazardous settings often hamper the development of autonomous inspection systems. To overcome this, in this work we propose and validate a hybrid data synthesis pipeline that combines procedural rendering with AI-driven video generation. Our methodology leverages BlenderProc to create photorealistic images with precise annotations and controlled domain randomization, and integrates NVIDIA's Cosmos-Predict2 world-foundation model to synthesize physically plausible video sequences with temporal diversity, capturing rare viewpoints and adverse conditions. We demonstrate that a YOLO-based detection network trained on a composite dataset, blending real images with our synthetic data, achieves superior performance compared to models trained exclusively on real-world data. Notably, a 1:1 mixture of real and synthetic data yielded the highest accuracy, surpassing the real-only baseline. These findings highlight the viability of a synthetic-first approach as an efficient, cost-effective, and safe alternative for developing reliable perception systems in safety-critical and resource-constrained industrial applications. 

---
# Optimizing Grasping in Legged Robots: A Deep Learning Approach to Loco-Manipulation 

**Authors**: Dilermando Almeida, Guilherme Lazzarini, Juliano Negri, Thiago H. Segreto, Ricardo V. Godoy, Marcelo Becker  

**Link**: [PDF](https://arxiv.org/pdf/2508.17466)  

**Abstract**: Quadruped robots have emerged as highly efficient and versatile platforms, excelling in navigating complex and unstructured terrains where traditional wheeled robots might fail. Equipping these robots with manipulator arms unlocks the advanced capability of loco-manipulation to perform complex physical interaction tasks in areas ranging from industrial automation to search-and-rescue missions. However, achieving precise and adaptable grasping in such dynamic scenarios remains a significant challenge, often hindered by the need for extensive real-world calibration and pre-programmed grasp configurations. This paper introduces a deep learning framework designed to enhance the grasping capabilities of quadrupeds equipped with arms, focusing on improved precision and adaptability. Our approach centers on a sim-to-real methodology that minimizes reliance on physical data collection. We developed a pipeline within the Genesis simulation environment to generate a synthetic dataset of grasp attempts on common objects. By simulating thousands of interactions from various perspectives, we created pixel-wise annotated grasp-quality maps to serve as the ground truth for our model. This dataset was used to train a custom CNN with a U-Net-like architecture that processes multi-modal input from an onboard RGB and depth cameras, including RGB images, depth maps, segmentation masks, and surface normal maps. The trained model outputs a grasp-quality heatmap to identify the optimal grasp point. We validated the complete framework on a four-legged robot. The system successfully executed a full loco-manipulation task: autonomously navigating to a target object, perceiving it with its sensors, predicting the optimal grasp pose using our model, and performing a precise grasp. This work proves that leveraging simulated training with advanced sensing offers a scalable and effective solution for object handling. 

---
# Bias Amplification in Stable Diffusion's Representation of Stigma Through Skin Tones and Their Homogeneity 

**Authors**: Kyra Wilson, Sourojit Ghosh, Aylin Caliskan  

**Link**: [PDF](https://arxiv.org/pdf/2508.17465)  

**Abstract**: Text-to-image generators (T2Is) are liable to produce images that perpetuate social stereotypes, especially in regards to race or skin tone. We use a comprehensive set of 93 stigmatized identities to determine that three versions of Stable Diffusion (v1.5, v2.1, and XL) systematically associate stigmatized identities with certain skin tones in generated images. We find that SD XL produces skin tones that are 13.53% darker and 23.76% less red (both of which indicate higher likelihood of societal discrimination) than previous models and perpetuate societal stereotypes associating people of color with stigmatized identities. SD XL also shows approximately 30% less variability in skin tones when compared to previous models and 18.89-56.06% compared to human face datasets. Measuring variability through metrics which directly correspond to human perception suggest a similar pattern, where SD XL shows the least amount of variability in skin tones of people with stigmatized identities and depicts most (60.29%) stigmatized identities as being less diverse than non-stigmatized identities. Finally, SD shows more homogenization of skin tones of racial and ethnic identities compared to other stigmatized or non-stigmatized identities, reinforcing incorrect equivalence of biologically-determined skin tone and socially-constructed racial and ethnic identity. Because SD XL is the largest and most complex model and users prefer its generations compared to other models examined in this study, these findings have implications for the dynamics of bias amplification in T2Is, increasing representational harms and challenges generating diverse images depicting people with stigmatized identities. 

---
# FedKLPR: Personalized Federated Learning for Person Re-Identification with Adaptive Pruning 

**Authors**: Po-Hsien Yu, Yu-Syuan Tseng, Shao-Yi Chien  

**Link**: [PDF](https://arxiv.org/pdf/2508.17431)  

**Abstract**: Person re-identification (Re-ID) is a fundamental task in intelligent surveillance and public safety. Federated learning (FL) offers a privacy-preserving solution by enabling collaborative model training without centralized data collection. However, applying FL to real-world re-ID systems faces two major challenges: statistical heterogeneity across clients due to non-IID data distributions, and substantial communication overhead caused by frequent transmission of large-scale models. To address these issues, we propose FedKLPR, a lightweight and communication-efficient federated learning framework for person re-identification. FedKLPR introduces four key components. First, the KL-Divergence Regularization Loss (KLL) constrains local models by minimizing the divergence from the global feature distribution, effectively mitigating the effects of statistical heterogeneity and improving convergence stability under non-IID conditions. Secondly, KL-Divergence-Prune Weighted Aggregation (KLPWA) integrates pruning ratio and distributional similarity into the aggregation process, thereby improving the robustness of the global model while significantly reducing communication overhead. Furthermore, sparse Activation Skipping (SAS) mitigates the dilution of critical parameters during the aggregation of pruned client models by excluding zero-valued weights from the update process. Finally, Cross-Round Recovery (CRR) introduces a dynamic pruning control mechanism that halts pruning when necessary, enabling deeper compression while maintaining model accuracy. Experimental results on eight benchmark datasets demonstrate that FedKLPR achieves significant communication reduction. Compared with the state-of-the-art, FedKLPR reduces 33\%-38\% communication cost on ResNet-50 and 20\%-40\% communication cost on ResNet-34, while maintaining model accuracy within 1\% degradation. 

---
# Convergence and Generalization of Anti-Regularization for Parametric Models 

**Authors**: Dongseok Kim, Wonjun Jeong, Gisung Oh  

**Link**: [PDF](https://arxiv.org/pdf/2508.17412)  

**Abstract**: We propose Anti-regularization (AR), which adds a sign-reversed reward term to the loss to intentionally increase model expressivity in the small-sample regime, and then attenuates this intervention with a power-law decay as the sample size grows. We formalize spectral safety and trust-region conditions, and design a lightweight stability safeguard that combines a projection operator with gradient clipping, ensuring stable intervention under stated assumptions. Our analysis spans linear smoothers and the Neural Tangent Kernel (NTK) regime, providing practical guidance on selecting the decay exponent by balancing empirical risk against variance. Empirically, AR reduces underfitting while preserving generalization and improving calibration in both regression and classification. Ablation studies confirm that the decay schedule and the stability safeguard are critical to preventing overfitting and numerical instability. We further examine a degrees-of-freedom targeting schedule that keeps per-sample complexity approximately constant. AR is simple to implement and reproducible, integrating cleanly into standard empirical risk minimization pipelines. It enables robust learning in data- and resource-constrained settings by intervening only when beneficial and fading away when unnecessary. 

---
# Retrieval Capabilities of Large Language Models Scale with Pretraining FLOPs 

**Authors**: Jacob Portes, Connor Jennings, Erica Ji Yuen, Sasha Doubov, Michael Carbin  

**Link**: [PDF](https://arxiv.org/pdf/2508.17400)  

**Abstract**: How does retrieval performance scale with pretraining FLOPs? We benchmark retrieval performance across LLM model sizes from 125 million parameters to 7 billion parameters pretrained on datasets ranging from 1 billion tokens to more than 2 trillion tokens. We find that retrieval performance on zero-shot BEIR tasks predictably scales with LLM size, training duration, and estimated FLOPs. We also show that In-Context Learning scores are strongly correlated with retrieval scores across retrieval tasks. Finally, we highlight the implications this has for the development of LLM-based retrievers. 

---
# Agent-Testing Agent: A Meta-Agent for Automated Testing and Evaluation of Conversational AI Agents 

**Authors**: Sameer Komoravolu, Khalil Mrini  

**Link**: [PDF](https://arxiv.org/pdf/2508.17393)  

**Abstract**: LLM agents are increasingly deployed to plan, retrieve, and write with tools, yet evaluation still leans on static benchmarks and small human studies. We present the Agent-Testing Agent (ATA), a meta-agent that combines static code analysis, designer interrogation, literature mining, and persona-driven adversarial test generation whose difficulty adapts via judge feedback. Each dialogue is scored with an LLM-as-a-Judge (LAAJ) rubric and used to steer subsequent tests toward the agent's weakest capabilities. On a travel planner and a Wikipedia writer, the ATA surfaces more diverse and severe failures than expert annotators while matching severity, and finishes in 20--30 minutes versus ten-annotator rounds that took days. Ablating code analysis and web search increases variance and miscalibration, underscoring the value of evidence-grounded test generation. The ATA outputs quantitative metrics and qualitative bug reports for developers. We release the full methodology and open-source implementation for reproducible agent testing: this https URL 

---
# Neural Proteomics Fields for Super-resolved Spatial Proteomics Prediction 

**Authors**: Bokai Zhao, Weiyang Shi, Hanqing Chao, Zijiang Yang, Yiyang Zhang, Ming Song, Tianzi Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2508.17389)  

**Abstract**: Spatial proteomics maps protein distributions in tissues, providing transformative insights for life sciences. However, current sequencing-based technologies suffer from low spatial resolution, and substantial inter-tissue variability in protein expression further compromises the performance of existing molecular data prediction methods. In this work, we introduce the novel task of spatial super-resolution for sequencing-based spatial proteomics (seq-SP) and, to the best of our knowledge, propose the first deep learning model for this task--Neural Proteomics Fields (NPF). NPF formulates seq-SP as a protein reconstruction problem in continuous space by training a dedicated network for each tissue. The model comprises a Spatial Modeling Module, which learns tissue-specific protein spatial distributions, and a Morphology Modeling Module, which extracts tissue-specific morphological features. Furthermore, to facilitate rigorous evaluation, we establish an open-source benchmark dataset, Pseudo-Visium SP, for this task. Experimental results demonstrate that NPF achieves state-of-the-art performance with fewer learnable parameters, underscoring its potential for advancing spatial proteomics research. Our code and dataset are publicly available at this https URL. 

---
# Graph-R1: Incentivizing the Zero-Shot Graph Learning Capability in LLMs via Explicit Reasoning 

**Authors**: Yicong Wu, Guangyue Lu, Yuan Zuo, Huarong Zhang, Junjie Wu  

**Link**: [PDF](https://arxiv.org/pdf/2508.17387)  

**Abstract**: Generalizing to unseen graph tasks without task-pecific supervision remains challenging. Graph Neural Networks (GNNs) are limited by fixed label spaces, while Large Language Models (LLMs) lack structural inductive biases. Recent advances in Large Reasoning Models (LRMs) provide a zero-shot alternative via explicit, long chain-of-thought reasoning. Inspired by this, we propose a GNN-free approach that reformulates graph tasks--node classification, link prediction, and graph classification--as textual reasoning problems solved by LRMs. We introduce the first datasets with detailed reasoning traces for these tasks and develop Graph-R1, a reinforcement learning framework that leverages task-specific rethink templates to guide reasoning over linearized graphs. Experiments demonstrate that Graph-R1 outperforms state-of-the-art baselines in zero-shot settings, producing interpretable and effective predictions. Our work highlights the promise of explicit reasoning for graph learning and provides new resources for future research. 

---
# Condition Weaving Meets Expert Modulation: Towards Universal and Controllable Image Generation 

**Authors**: Guoqing Zhang, Xingtong Ge, Lu Shi, Xin Zhang, Muqing Xue, Wanru Xu, Yigang Cen  

**Link**: [PDF](https://arxiv.org/pdf/2508.17364)  

**Abstract**: The image-to-image generation task aims to produce controllable images by leveraging conditional inputs and prompt instructions. However, existing methods often train separate control branches for each type of condition, leading to redundant model structures and inefficient use of computational resources. To address this, we propose a Unified image-to-image Generation (UniGen) framework that supports diverse conditional inputs while enhancing generation efficiency and expressiveness. Specifically, to tackle the widely existing parameter redundancy and computational inefficiency in controllable conditional generation architectures, we propose the Condition Modulated Expert (CoMoE) module. This module aggregates semantically similar patch features and assigns them to dedicated expert modules for visual representation and conditional modeling. By enabling independent modeling of foreground features under different conditions, CoMoE effectively mitigates feature entanglement and redundant computation in multi-condition scenarios. Furthermore, to bridge the information gap between the backbone and control branches, we propose WeaveNet, a dynamic, snake-like connection mechanism that enables effective interaction between global text-level control from the backbone and fine-grained control from conditional branches. Extensive experiments on the Subjects-200K and MultiGen-20M datasets across various conditional image generation tasks demonstrate that our method consistently achieves state-of-the-art performance, validating its advantages in both versatility and effectiveness. The code has been uploaded to this https URL. 

---
# The Arabic Generality Score: Another Dimension of Modeling Arabic Dialectness 

**Authors**: Sanad Shaban, Nizar Habash  

**Link**: [PDF](https://arxiv.org/pdf/2508.17347)  

**Abstract**: Arabic dialects form a diverse continuum, yet NLP models often treat them as discrete categories. Recent work addresses this issue by modeling dialectness as a continuous variable, notably through the Arabic Level of Dialectness (ALDi). However, ALDi reduces complex variation to a single dimension. We propose a complementary measure: the Arabic Generality Score (AGS), which quantifies how widely a word is used across dialects. We introduce a pipeline that combines word alignment, etymology-aware edit distance, and smoothing to annotate a parallel corpus with word-level AGS. A regression model is then trained to predict AGS in context. Our approach outperforms strong baselines, including state-of-the-art dialect ID systems, on a multi-dialect benchmark. AGS offers a scalable, linguistically grounded way to model lexical generality, enriching representations of Arabic dialectness. 

---
# Agentic AI for Software: thoughts from Software Engineering community 

**Authors**: Abhik Roychoudhury  

**Link**: [PDF](https://arxiv.org/pdf/2508.17343)  

**Abstract**: AI agents have recently shown significant promise in software engineering. Much public attention has been transfixed on the topic of code generation from Large Language Models (LLMs) via a prompt. However, software engineering is much more than programming, and AI agents go far beyond instructions given by a prompt.
At the code level, common software tasks include code generation, testing, and program repair. Design level software tasks may include architecture exploration, requirements understanding, and requirements enforcement at the code level. Each of these software tasks involves micro-decisions which can be taken autonomously by an AI agent, aided by program analysis tools. This creates the vision of an AI software engineer, where the AI agent can be seen as a member of a development team.
Conceptually, the key to successfully developing trustworthy agentic AI-based software workflows will be to resolve the core difficulty in software engineering - the deciphering and clarification of developer intent. Specification inference, or deciphering the intent, thus lies at the heart of many software tasks, including software maintenance and program repair. A successful deployment of agentic technology into software engineering would involve making conceptual progress in such intent inference via agents.
Trusting the AI agent becomes a key aspect, as software engineering becomes more automated. Higher automation also leads to higher volume of code being automatically generated, and then integrated into code-bases. Thus to deal with this explosion, an emerging direction is AI-based verification and validation (V & V) of AI generated code. We posit that agentic software workflows in future will include such AIbased V&V. 

---
# Capturing Legal Reasoning Paths from Facts to Law in Court Judgments using Knowledge Graphs 

**Authors**: Ryoma Kondo, Riona Matsuoka, Takahiro Yoshida, Kazuyuki Yamasawa, Ryohei Hisano  

**Link**: [PDF](https://arxiv.org/pdf/2508.17340)  

**Abstract**: Court judgments reveal how legal rules have been interpreted and applied to facts, providing a foundation for understanding structured legal reasoning. However, existing automated approaches for capturing legal reasoning, including large language models, often fail to identify the relevant legal context, do not accurately trace how facts relate to legal norms, and may misrepresent the layered structure of judicial reasoning. These limitations hinder the ability to capture how courts apply the law to facts in practice. In this paper, we address these challenges by constructing a legal knowledge graph from 648 Japanese administrative court decisions. Our method extracts components of legal reasoning using prompt-based large language models, normalizes references to legal provisions, and links facts, norms, and legal applications through an ontology of legal inference. The resulting graph captures the full structure of legal reasoning as it appears in real court decisions, making implicit reasoning explicit and machine-readable. We evaluate our system using expert annotated data, and find that it achieves more accurate retrieval of relevant legal provisions from facts than large language model baselines and retrieval-augmented methods. 

---
# Modality-Specific Speech Enhancement and Noise-Adaptive Fusion for Acoustic and Body-Conduction Microphone Framework 

**Authors**: Yunsik Kim, Yoonyoung Chung  

**Link**: [PDF](https://arxiv.org/pdf/2508.17336)  

**Abstract**: Body\-conduction microphone signals (BMS) bypass airborne sound, providing strong noise resistance. However, a complementary modality is required to compensate for the inherent loss of high\-frequency information. In this study, we propose a novel multi\-modal framework that combines BMS and acoustic microphone signals (AMS) to achieve both noise suppression and high\-frequency reconstruction. Unlike conventional multi\-modal approaches that simply merge features, our method employs two specialized networks\: a mapping-based model to enhance BMS and a masking-based model to denoise AMS. These networks are integrated through a dynamic fusion mechanism that adapts to local noise conditions, ensuring the optimal use of each modality's strengths. We performed evaluations on the TAPS dataset, augmented with DNS\-2023 noise clips, using objective speech quality metrics. The results clearly demonstrate that our approach outperforms single\-modal solutions in a wide range of noisy environments. 

---
# Mind the (Language) Gap: Towards Probing Numerical and Cross-Lingual Limits of LVLMs 

**Authors**: Somraj Gautam, Abhirama Subramanyam Penamakuri, Abhishek Bhandari, Gaurav Harit  

**Link**: [PDF](https://arxiv.org/pdf/2508.17334)  

**Abstract**: We introduce MMCRICBENCH-3K, a benchmark for Visual Question Answering (VQA) on cricket scorecards, designed to evaluate large vision-language models (LVLMs) on complex numerical and cross-lingual reasoning over semi-structured tabular images. MMCRICBENCH-3K comprises 1,463 synthetically generated scorecard images from ODI, T20, and Test formats, accompanied by 1,500 English QA pairs. It includes two subsets: MMCRICBENCH-E-1.5K, featuring English scorecards, and MMCRICBENCH-H-1.5K, containing visually similar Hindi scorecards, with all questions and answers kept in English to enable controlled cross-script evaluation. The task demands reasoning over structured numerical data, multi-image context, and implicit domain knowledge. Empirical results show that even state-of-the-art LVLMs, such as GPT-4o and Qwen2.5VL, struggle on the English subset despite it being their primary training language and exhibit a further drop in performance on the Hindi subset. This reveals key limitations in structure-aware visual text understanding, numerical reasoning, and cross-lingual generalization. The dataset is publicly available via Hugging Face at this https URL, to promote LVLM research in this direction. 

---
# Omne-R1: Learning to Reason with Memory for Multi-hop Question Answering 

**Authors**: Boyuan Liu, Feng Ji, Jiayan Nan, Han Zhao, Weiling Chen, Shihao Xu, Xing Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2508.17330)  

**Abstract**: This paper introduces Omne-R1, a novel approach designed to enhance multi-hop question answering capabilities on schema-free knowledge graphs by integrating advanced reasoning models. Our method employs a multi-stage training workflow, including two reinforcement learning phases and one supervised fine-tuning phase. We address the challenge of limited suitable knowledge graphs and QA data by constructing domain-independent knowledge graphs and auto-generating QA pairs. Experimental results show significant improvements in answering multi-hop questions, with notable performance gains on more complex 3+ hop questions. Our proposed training framework demonstrates strong generalization abilities across diverse knowledge domains. 

---
# CultranAI at PalmX 2025: Data Augmentation for Cultural Knowledge Representation 

**Authors**: Hunzalah Hassan Bhatti, Youssef Ahmed, Md Arid Hasan, Firoj Alam  

**Link**: [PDF](https://arxiv.org/pdf/2508.17324)  

**Abstract**: In this paper, we report our participation to the PalmX cultural evaluation shared task. Our system, CultranAI, focused on data augmentation and LoRA fine-tuning of large language models (LLMs) for Arabic cultural knowledge representation. We benchmarked several LLMs to identify the best-performing model for the task. In addition to utilizing the PalmX dataset, we augmented it by incorporating the Palm dataset and curated a new dataset of over 22K culturally grounded multiple-choice questions (MCQs). Our experiments showed that the Fanar-1-9B-Instruct model achieved the highest performance. We fine-tuned this model on the combined augmented dataset of 22K+ MCQs. On the blind test set, our submitted system ranked 5th with an accuracy of 70.50%, while on the PalmX development set, it achieved an accuracy of 84.1%. 

---
# Chinese Court Simulation with LLM-Based Agent System 

**Authors**: Kaiyuan Zhang, Jiaqi Li, Yueyue Wu, Haitao Li, Cheng Luo, Shaokun Zou, Yujia Zhou, Weihang Su, Qingyao Ai, Yiqun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.17322)  

**Abstract**: Mock trial has long served as an important platform for legal professional training and education. It not only helps students learn about realistic trial procedures, but also provides practical value for case analysis and judgment prediction. Traditional mock trials are difficult to access by the public because they rely on professional tutors and human participants. Fortunately, the rise of large language models (LLMs) provides new opportunities for creating more accessible and scalable court simulations. While promising, existing research mainly focuses on agent construction while ignoring the systematic design and evaluation of court simulations, which are actually more important for the credibility and usage of court simulation in practice. To this end, we present the first court simulation framework -- SimCourt -- based on the real-world procedure structure of Chinese courts. Our framework replicates all 5 core stages of a Chinese trial and incorporates 5 courtroom roles, faithfully following the procedural definitions in China. To simulate trial participants with different roles, we propose and craft legal agents equipped with memory, planning, and reflection abilities. Experiment on legal judgment prediction show that our framework can generate simulated trials that better guide the system to predict the imprisonment, probation, and fine of each case. Further annotations by human experts show that agents' responses under our simulation framework even outperformed judges and lawyers from the real trials in many scenarios. These further demonstrate the potential of LLM-based court simulation. 

---
# Bine Trees: Enhancing Collective Operations by Optimizing Communication Locality 

**Authors**: Daniele De Sensi, Saverio Pasqualoni, Lorenzo Piarulli, Tommaso Bonato, Seydou Ba, Matteo Turisini, Jens Domke, Torsten Hoefler  

**Link**: [PDF](https://arxiv.org/pdf/2508.17311)  

**Abstract**: Communication locality plays a key role in the performance of collective operations on large HPC systems, especially on oversubscribed networks where groups of nodes are fully connected internally but sparsely linked through global connections. We present Bine (binomial negabinary) trees, a family of collective algorithms that improve communication locality. Bine trees maintain the generality of binomial trees and butterflies while cutting global-link traffic by up to 33%. We implement eight Bine-based collectives and evaluate them on four large-scale supercomputers with Dragonfly, Dragonfly+, oversubscribed fat-tree, and torus topologies, achieving up to 5x speedups and consistent reductions in global-link traffic across different vector sizes and node counts. 

---
# Explain Before You Answer: A Survey on Compositional Visual Reasoning 

**Authors**: Fucai Ke, Joy Hsu, Zhixi Cai, Zixian Ma, Xin Zheng, Xindi Wu, Sukai Huang, Weiqing Wang, Pari Delir Haghighi, Gholamreza Haffari, Ranjay Krishna, Jiajun Wu, Hamid Rezatofighi  

**Link**: [PDF](https://arxiv.org/pdf/2508.17298)  

**Abstract**: Compositional visual reasoning has emerged as a key research frontier in multimodal AI, aiming to endow machines with the human-like ability to decompose visual scenes, ground intermediate concepts, and perform multi-step logical inference. While early surveys focus on monolithic vision-language models or general multimodal reasoning, a dedicated synthesis of the rapidly expanding compositional visual reasoning literature is still missing. We fill this gap with a comprehensive survey spanning 2023 to 2025 that systematically reviews 260+ papers from top venues (CVPR, ICCV, NeurIPS, ICML, ACL, etc.). We first formalize core definitions and describe why compositional approaches offer advantages in cognitive alignment, semantic fidelity, robustness, interpretability, and data efficiency. Next, we trace a five-stage paradigm shift: from prompt-enhanced language-centric pipelines, through tool-enhanced LLMs and tool-enhanced VLMs, to recently minted chain-of-thought reasoning and unified agentic VLMs, highlighting their architectural designs, strengths, and limitations. We then catalog 60+ benchmarks and corresponding metrics that probe compositional visual reasoning along dimensions such as grounding accuracy, chain-of-thought faithfulness, and high-resolution perception. Drawing on these analyses, we distill key insights, identify open challenges (e.g., limitations of LLM-based reasoning, hallucination, a bias toward deductive reasoning, scalable supervision, tool integration, and benchmark limitations), and outline future directions, including world-model integration, human-AI collaborative reasoning, and richer evaluation protocols. By offering a unified taxonomy, historical roadmap, and critical outlook, this survey aims to serve as a foundational reference and inspire the next generation of compositional visual reasoning research. 

---
# Deep Learning-Assisted Detection of Sarcopenia in Cross-Sectional Computed Tomography Imaging 

**Authors**: Manish Bhardwaj, Huizhi Liang, Ashwin Sivaharan, Sandip Nandhra, Vaclav Snasel, Tamer El-Sayed, Varun Ojha  

**Link**: [PDF](https://arxiv.org/pdf/2508.17275)  

**Abstract**: Sarcopenia is a progressive loss of muscle mass and function linked to poor surgical outcomes such as prolonged hospital stays, impaired mobility, and increased mortality. Although it can be assessed through cross-sectional imaging by measuring skeletal muscle area (SMA), the process is time-consuming and adds to clinical workloads, limiting timely detection and management; however, this process could become more efficient and scalable with the assistance of artificial intelligence applications. This paper presents high-quality three-dimensional cross-sectional computed tomography (CT) images of patients with sarcopenia collected at the Freeman Hospital, Newcastle upon Tyne Hospitals NHS Foundation Trust. Expert clinicians manually annotated the SMA at the third lumbar vertebra, generating precise segmentation masks. We develop deep-learning models to measure SMA in CT images and automate this task. Our methodology employed transfer learning and self-supervised learning approaches using labelled and unlabeled CT scan datasets. While we developed qualitative assessment models for detecting sarcopenia, we observed that the quantitative assessment of SMA is more precise and informative. This approach also mitigates the issue of class imbalance and limited data availability. Our model predicted the SMA, on average, with an error of +-3 percentage points against the manually measured SMA. The average dice similarity coefficient of the predicted masks was 93%. Our results, therefore, show a pathway to full automation of sarcopenia assessment and detection. 

---
# ResLink: A Novel Deep Learning Architecture for Brain Tumor Classification with Area Attention and Residual Connections 

**Authors**: Sumedha Arya, Nirmal Gaud  

**Link**: [PDF](https://arxiv.org/pdf/2508.17259)  

**Abstract**: Brain tumors show significant health challenges due to their potential to cause critical neurological functions. Early and accurate diagnosis is crucial for effective treatment. In this research, we propose ResLink, a novel deep learning architecture for brain tumor classification using CT scan images. ResLink integrates novel area attention mechanisms with residual connections to enhance feature learning and spatial understanding for spatially rich image classification tasks. The model employs a multi-stage convolutional pipeline, incorporating dropout, regularization, and downsampling, followed by a final attention-based refinement for classification. Trained on a balanced dataset, ResLink achieves a high accuracy of 95% and demonstrates strong generalizability. This research demonstrates the potential of ResLink in improving brain tumor classification, offering a robust and efficient technique for medical imaging applications. 

---
# Provable Generalization in Overparameterized Neural Nets 

**Authors**: Aviral Dhingra  

**Link**: [PDF](https://arxiv.org/pdf/2508.17256)  

**Abstract**: Deep neural networks often contain far more parameters than training examples, yet they still manage to generalize well in practice. Classical complexity measures such as VC-dimension or PAC-Bayes bounds usually become vacuous in this overparameterized regime, offering little explanation for the empirical success of models like Transformers. In this work, I explore an alternative notion of capacity for attention-based models, based on the effective rank of their attention matrices. The intuition is that, although the parameter count is enormous, the functional dimensionality of attention is often much lower. I show that this quantity leads to a generalization bound whose dependence on sample size matches empirical scaling laws observed in large language models, up to logarithmic factors. While the analysis is not a complete theory of overparameterized learning, it provides evidence that spectral properties of attention, rather than raw parameter counts, may be the right lens for understanding why these models generalize. 

---
# A biological vision inspired framework for machine perception of abutting grating illusory contours 

**Authors**: Xiao Zhang, Kai-Fu Yang, Xian-Shi Zhang, Hong-Zhi You, Hong-Mei Yan, Yong-Jie Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.17254)  

**Abstract**: Higher levels of machine intelligence demand alignment with human perception and cognition. Deep neural networks (DNN) dominated machine intelligence have demonstrated exceptional performance across various real-world tasks. Nevertheless, recent evidence suggests that DNNs fail to perceive illusory contours like the abutting grating, a discrepancy that misaligns with human perception patterns. Departing from previous works, we propose a novel deep network called illusory contour perception network (ICPNet) inspired by the circuits of the visual cortex. In ICPNet, a multi-scale feature projection (MFP) module is designed to extract multi-scale representations. To boost the interaction between feedforward and feedback features, a feature interaction attention module (FIAM) is introduced. Moreover, drawing inspiration from the shape bias observed in human perception, an edge detection task conducted via the edge fusion module (EFM) injects shape constraints that guide the network to concentrate on the foreground. We assess our method on the existing AG-MNIST test set and the AG-Fashion-MNIST test sets constructed by this work. Comprehensive experimental results reveal that ICPNet is significantly more sensitive to abutting grating illusory contours than state-of-the-art models, with notable improvements in top-1 accuracy across various subsets. This work is expected to make a step towards human-level intelligence for DNN-based models. 

---
# CoViPAL: Layer-wise Contextualized Visual Token Pruning for Large Vision-Language Models 

**Authors**: Zicong Tang, Ziyang Ma, Suqing Wang, Zuchao Li, Lefei Zhang, Hai Zhao, Yun Li, Qianren Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.17243)  

**Abstract**: Large Vision-Language Models (LVLMs) process multimodal inputs consisting of text tokens and vision tokens extracted from images or videos. Due to the rich visual information, a single image can generate thousands of vision tokens, leading to high computational costs during the prefilling stage and significant memory overhead during decoding. Existing methods attempt to prune redundant vision tokens, revealing substantial redundancy in visual representations. However, these methods often struggle in shallow layers due to the lack of sufficient contextual information. We argue that many visual tokens are inherently redundant even in shallow layers and can be safely and effectively pruned with appropriate contextual signals. In this work, we propose CoViPAL, a layer-wise contextualized visual token pruning method that employs a Plug-and-Play Pruning Module (PPM) to predict and remove redundant vision tokens before they are processed by the LVLM. The PPM is lightweight, model-agnostic, and operates independently of the LVLM architecture, ensuring seamless integration with various models. Extensive experiments on multiple benchmarks demonstrate that CoViPAL outperforms training-free pruning methods under equal token budgets and surpasses training-based methods with comparable supervision. CoViPAL offers a scalable and efficient solution to improve inference efficiency in LVLMs without compromising accuracy. 

---
# ClaimGen-CN: A Large-scale Chinese Dataset for Legal Claim Generation 

**Authors**: Siying Zhou, Yiquan Wu, Hui Chen, Xavier Hu, Kun Kuang, Adam Jatowt, Ming Hu, Chunyan Zheng, Fei Wu  

**Link**: [PDF](https://arxiv.org/pdf/2508.17234)  

**Abstract**: Legal claims refer to the plaintiff's demands in a case and are essential to guiding judicial reasoning and case resolution. While many works have focused on improving the efficiency of legal professionals, the research on helping non-professionals (e.g., plaintiffs) remains unexplored. This paper explores the problem of legal claim generation based on the given case's facts. First, we construct ClaimGen-CN, the first dataset for Chinese legal claim generation task, from various real-world legal disputes. Additionally, we design an evaluation metric tailored for assessing the generated claims, which encompasses two essential dimensions: factuality and clarity. Building on this, we conduct a comprehensive zero-shot evaluation of state-of-the-art general and legal-domain large language models. Our findings highlight the limitations of the current models in factual precision and expressive clarity, pointing to the need for more targeted development in this domain. To encourage further exploration of this important task, we will make the dataset publicly available. 

---
# Module-Aware Parameter-Efficient Machine Unlearning on Transformers 

**Authors**: Wenjie Bao, Jian Lou, Yuke Hu, Xiaochen Li, Zhihao Liu, Jiaqi Liu, Zhan Qin, Kui Ren  

**Link**: [PDF](https://arxiv.org/pdf/2508.17233)  

**Abstract**: Transformer has become fundamental to a vast series of pre-trained large models that have achieved remarkable success across diverse applications. Machine unlearning, which focuses on efficiently removing specific data influences to comply with privacy regulations, shows promise in restricting updates to influence-critical parameters. However, existing parameter-efficient unlearning methods are largely devised in a module-oblivious manner, which tends to inaccurately identify these parameters and leads to inferior unlearning performance for Transformers. In this paper, we propose {\tt MAPE-Unlearn}, a module-aware parameter-efficient machine unlearning approach that uses a learnable pair of masks to pinpoint influence-critical parameters in the heads and filters of Transformers. The learning objective of these masks is derived by desiderata of unlearning and optimized through an efficient algorithm featured by a greedy search with a warm start. Extensive experiments on various Transformer models and datasets demonstrate the effectiveness and robustness of {\tt MAPE-Unlearn} for unlearning. 

---
# Multi-Metric Preference Alignment for Generative Speech Restoration 

**Authors**: Junan Zhang, Xueyao Zhang, Jing Yang, Yuancheng Wang, Fan Fan, Zhizheng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2508.17229)  

**Abstract**: Recent generative models have significantly advanced speech restoration tasks, yet their training objectives often misalign with human perceptual preferences, resulting in suboptimal quality. While post-training alignment has proven effective in other generative domains like text and image generation, its application to generative speech restoration remains largely under-explored. This work investigates the challenges of applying preference-based post-training to this task, focusing on how to define a robust preference signal and curate high-quality data to avoid reward hacking. To address these challenges, we propose a multi-metric preference alignment strategy. We construct a new dataset, GenSR-Pref, comprising 80K preference pairs, where each chosen sample is unanimously favored by a complementary suite of metrics covering perceptual quality, signal fidelity, content consistency, and timbre preservation. This principled approach ensures a holistic preference signal. Applying Direct Preference Optimization (DPO) with our dataset, we observe consistent and significant performance gains across three diverse generative paradigms: autoregressive models (AR), masked generative models (MGM), and flow-matching models (FM) on various restoration benchmarks, in both objective and subjective evaluations. Ablation studies confirm the superiority of our multi-metric strategy over single-metric approaches in mitigating reward hacking. Furthermore, we demonstrate that our aligned models can serve as powerful ''data annotators'', generating high-quality pseudo-labels to serve as a supervision signal for traditional discriminative models in data-scarce scenarios like singing voice restoration. Demo Page:this https URL 

---
# SSFO: Self-Supervised Faithfulness Optimization for Retrieval-Augmented Generation 

**Authors**: Xiaqiang Tang, Yi Wang, Keyu Hu, Rui Xu, Chuang Li, Weigao Sun, Jian Li, Sihong Xie  

**Link**: [PDF](https://arxiv.org/pdf/2508.17225)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems require Large Language Models (LLMs) to generate responses that are faithful to the retrieved context. However, faithfulness hallucination remains a critical challenge, as existing methods often require costly supervision and post-training or significant inference burdens. To overcome these limitations, we introduce Self-Supervised Faithfulness Optimization (SSFO), the first self-supervised alignment approach for enhancing RAG faithfulness. SSFO constructs preference data pairs by contrasting the model's outputs generated with and without the context. Leveraging Direct Preference Optimization (DPO), SSFO aligns model faithfulness without incurring labeling costs or additional inference burden. We theoretically and empirically demonstrate that SSFO leverages a benign form of \emph{likelihood displacement}, transferring probability mass from parametric-based tokens to context-aligned tokens. Based on this insight, we propose a modified DPO loss function to encourage likelihood displacement. Comprehensive evaluations show that SSFO significantly outperforms existing methods, achieving state-of-the-art faithfulness on multiple context-based question-answering datasets. Notably, SSFO exhibits strong generalization, improving cross-lingual faithfulness and preserving general instruction-following capabilities. We release our code and model at the anonymous link: this https URL 

---
# Exposing Privacy Risks in Graph Retrieval-Augmented Generation 

**Authors**: Jiale Liu, Jiahao Zhang, Suhang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.17222)  

**Abstract**: Retrieval-Augmented Generation (RAG) is a powerful technique for enhancing Large Language Models (LLMs) with external, up-to-date knowledge. Graph RAG has emerged as an advanced paradigm that leverages graph-based knowledge structures to provide more coherent and contextually rich answers. However, the move from plain document retrieval to structured graph traversal introduces new, under-explored privacy risks. This paper investigates the data extraction vulnerabilities of the Graph RAG systems. We design and execute tailored data extraction attacks to probe their susceptibility to leaking both raw text and structured data, such as entities and their relationships. Our findings reveal a critical trade-off: while Graph RAG systems may reduce raw text leakage, they are significantly more vulnerable to the extraction of structured entity and relationship information. We also explore potential defense mechanisms to mitigate these novel attack surfaces. This work provides a foundational analysis of the unique privacy challenges in Graph RAG and offers insights for building more secure systems. 

---
# GPG-HT: Generalized Policy Gradient with History-Aware Decision Transformer for Probabilistic Path Planning 

**Authors**: Xing Wei, Yuqi Ouyang  

**Link**: [PDF](https://arxiv.org/pdf/2508.17218)  

**Abstract**: With the rapidly increased number of vehicles in urban areas, existing road infrastructure struggles to accommodate modern traffic demands, resulting in the issue of congestion. This highlights the importance of efficient path planning strategies. However, most recent navigation models focus solely on deterministic or time-dependent networks, while overlooking the correlations and the stochastic nature of traffic flows. In this work, we address the reliable shortest path problem within stochastic transportation networks under certain dependencies. We propose a path planning solution that integrates the decision Transformer with the Generalized Policy Gradient (GPG) framework. Based on the decision Transformer's capability to model long-term dependencies, our proposed solution improves the accuracy and stability of path decisions. Experimental results on the Sioux Falls Network (SFN) demonstrate that our approach outperforms previous baselines in terms of on-time arrival probability, providing more accurate path planning solutions. 

---
# How to make Medical AI Systems safer? Simulating Vulnerabilities, and Threats in Multimodal Medical RAG System 

**Authors**: Kaiwen Zuo, Zelin Liu, Raman Dutt, Ziyang Wang, Zhongtian Sun, Yeming Wang, Fan Mo, Pietro Liò  

**Link**: [PDF](https://arxiv.org/pdf/2508.17215)  

**Abstract**: Large Vision-Language Models (LVLMs) augmented with Retrieval-Augmented Generation (RAG) are increasingly employed in medical AI to enhance factual grounding through external clinical image-text retrieval. However, this reliance creates a significant attack surface. We propose MedThreatRAG, a novel multimodal poisoning framework that systematically probes vulnerabilities in medical RAG systems by injecting adversarial image-text pairs. A key innovation of our approach is the construction of a simulated semi-open attack environment, mimicking real-world medical systems that permit periodic knowledge base updates via user or pipeline contributions. Within this setting, we introduce and emphasize Cross-Modal Conflict Injection (CMCI), which embeds subtle semantic contradictions between medical images and their paired reports. These mismatches degrade retrieval and generation by disrupting cross-modal alignment while remaining sufficiently plausible to evade conventional filters. While basic textual and visual attacks are included for completeness, CMCI demonstrates the most severe degradation. Evaluations on IU-Xray and MIMIC-CXR QA tasks show that MedThreatRAG reduces answer F1 scores by up to 27.66% and lowers LLaVA-Med-1.5 F1 rates to as low as 51.36%. Our findings expose fundamental security gaps in clinical RAG systems and highlight the urgent need for threat-aware design and robust multimodal consistency checks. Finally, we conclude with a concise set of guidelines to inform the safe development of future multimodal medical RAG systems. 

---
# Multi-Agent Visual-Language Reasoning for Comprehensive Highway Scene Understanding 

**Authors**: Yunxiang Yang, Ningning Xu, Jidong J. Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.17205)  

**Abstract**: This paper introduces a multi-agent framework for comprehensive highway scene understanding, designed around a mixture-of-experts strategy. In this framework, a large generic vision-language model (VLM), such as GPT-4o, is contextualized with domain knowledge to generates task-specific chain-of-thought (CoT) prompts. These fine-grained prompts are then used to guide a smaller, efficient VLM (e.g., Qwen2.5-VL-7B) in reasoning over short videos, along with complementary modalities as applicable. The framework simultaneously addresses multiple critical perception tasks, including weather classification, pavement wetness assessment, and traffic congestion detection, achieving robust multi-task reasoning while balancing accuracy and computational efficiency. To support empirical validation, we curated three specialized datasets aligned with these tasks. Notably, the pavement wetness dataset is multimodal, combining video streams with road weather sensor data, highlighting the benefits of multimodal reasoning. Experimental results demonstrate consistently strong performance across diverse traffic and environmental conditions. From a deployment perspective, the framework can be readily integrated with existing traffic camera systems and strategically applied to high-risk rural locations, such as sharp curves, flood-prone lowlands, or icy bridges. By continuously monitoring the targeted sites, the system enhances situational awareness and delivers timely alerts, even in resource-constrained environments. 

---
# BudgetThinker: Empowering Budget-aware LLM Reasoning with Control Tokens 

**Authors**: Hao Wen, Xinrui Wu, Yi Sun, Feifei Zhang, Liye Chen, Jie Wang, Yunxin Liu, Ya-Qin Zhang, Yuanchun Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.17196)  

**Abstract**: Recent advancements in Large Language Models (LLMs) have leveraged increased test-time computation to enhance reasoning capabilities, a strategy that, while effective, incurs significant latency and resource costs, limiting their applicability in real-world time-constrained or cost-sensitive scenarios. This paper introduces BudgetThinker, a novel framework designed to empower LLMs with budget-aware reasoning, enabling precise control over the length of their thought processes. We propose a methodology that periodically inserts special control tokens during inference to continuously inform the model of its remaining token budget. This approach is coupled with a comprehensive two-stage training pipeline, beginning with Supervised Fine-Tuning (SFT) to familiarize the model with budget constraints, followed by a curriculum-based Reinforcement Learning (RL) phase that utilizes a length-aware reward function to optimize for both accuracy and budget adherence. We demonstrate that BudgetThinker significantly surpasses strong baselines in maintaining performance across a variety of reasoning budgets on challenging mathematical benchmarks. Our method provides a scalable and effective solution for developing efficient and controllable LLM reasoning, making advanced models more practical for deployment in resource-constrained and real-time environments. 

---
# LLM Assertiveness can be Mechanistically Decomposed into Emotional and Logical Components 

**Authors**: Hikaru Tsujimura, Arush Tagade  

**Link**: [PDF](https://arxiv.org/pdf/2508.17182)  

**Abstract**: Large Language Models (LLMs) often display overconfidence, presenting information with unwarranted certainty in high-stakes contexts. We investigate the internal basis of this behavior via mechanistic interpretability. Using open-sourced Llama 3.2 models fine-tuned on human annotated assertiveness datasets, we extract residual activations across all layers, and compute similarity metrics to localize assertive representations. Our analysis identifies layers most sensitive to assertiveness contrasts and reveals that high-assertive representations decompose into two orthogonal sub-components of emotional and logical clusters-paralleling the dual-route Elaboration Likelihood Model in Psychology. Steering vectors derived from these sub-components show distinct causal effects: emotional vectors broadly influence prediction accuracy, while logical vectors exert more localized effects. These findings provide mechanistic evidence for the multi-component structure of LLM assertiveness and highlight avenues for mitigating overconfident behavior. 

---
# Scaling Graph Transformers: A Comparative Study of Sparse and Dense Attention 

**Authors**: Leon Dimitrov  

**Link**: [PDF](https://arxiv.org/pdf/2508.17175)  

**Abstract**: Graphs have become a central representation in machine learning for capturing relational and structured data across various domains. Traditional graph neural networks often struggle to capture long-range dependencies between nodes due to their local structure. Graph transformers overcome this by using attention mechanisms that allow nodes to exchange information globally. However, there are two types of attention in graph transformers: dense and sparse. In this paper, we compare these two attention mechanisms, analyze their trade-offs, and highlight when to use each. We also outline current challenges and problems in designing attention for graph transformers. 

---
# ONG: Orthogonal Natural Gradient Descent 

**Authors**: Yajat Yadav, Jathin Korrapati, Patrick Mendoza  

**Link**: [PDF](https://arxiv.org/pdf/2508.17169)  

**Abstract**: Orthogonal gradient descent has emerged as a powerful method for continual learning tasks. However, its Euclidean projections overlook the underlying information-geometric structure of the space of distributions parametrized by neural networks, which can lead to suboptimal convergence in learning tasks. To counteract this, we combine it with the idea of the natural gradient and present ONG (Orthogonal Natural Gradient Descent). ONG preconditions each new task gradient with an efficient EKFAC approximation of the inverse Fisher information matrix, yielding updates that follow the steepest descent direction under a Riemannian metric. To preserve performance on previously learned tasks, ONG projects these natural gradients onto the orthogonal complement of prior task gradients. We provide a theoretical justification for this procedure, introduce the ONG algorithm, and benchmark its performance on the Permuted and Rotated MNIST datasets. All code for our experiments/reproducibility can be found at this https URL. 

---
# Error analysis for the deep Kolmogorov method 

**Authors**: Iulian Cîmpean, Thang Do, Lukas Gonon, Arnulf Jentzen, Ionel Popescu  

**Link**: [PDF](https://arxiv.org/pdf/2508.17167)  

**Abstract**: The deep Kolmogorov method is a simple and popular deep learning based method for approximating solutions of partial differential equations (PDEs) of the Kolmogorov type. In this work we provide an error analysis for the deep Kolmogorov method for heat PDEs. Specifically, we reveal convergence with convergence rates for the overall mean square distance between the exact solution of the heat PDE and the realization function of the approximating deep neural network (DNN) associated with a stochastic optimization algorithm in terms of the size of the architecture (the depth/number of hidden layers and the width of the hidden layers) of the approximating DNN, in terms of the number of random sample points used in the loss function (the number of input-output data pairs used in the loss function), and in terms of the size of the optimization error made by the employed stochastic optimization method. 

---
# Beyond Play and Pause: Turning GPT-4o Spatial Weakness into a Strength for In-Depth Interactive Video Learning 

**Authors**: Sajad Goudarzi, Samaneh Zamanifard  

**Link**: [PDF](https://arxiv.org/pdf/2508.17160)  

**Abstract**: Traditional video-based learning remains passive, offering limited opportunities for users to engage dynamically with content. While current AI-powered tools offer transcription and summarization, they lack real-time, region-specific interaction capabilities. This paper introduces Untwist, an AI-driven system that enables interactive video learning by allowing users to ask questions about the entire video or specific regions using a bounding box, receiving context-aware, multimodal responses. By integrating GPT APIs with Computer Vision techniques, Untwist extracts, processes, and structures video content to enhance comprehension. Our approach addresses GPT-4o spatial weakness by leveraging annotated frames instead of raw coordinate data, significantly improving accuracy in localizing and interpreting video content. This paper describes the system architecture, including video pre-processing and real-time interaction, and outlines how Untwist can transform passive video consumption into an interactive, AI-driven learning experience with the potential to enhance engagement and comprehension. 

---
# Mind the Gap: Time-of-Check to Time-of-Use Vulnerabilities in LLM-Enabled Agents 

**Authors**: Derek Lilienthal, Sanghyun Hong  

**Link**: [PDF](https://arxiv.org/pdf/2508.17155)  

**Abstract**: Large Language Model (LLM)-enabled agents are rapidly emerging across a wide range of applications, but their deployment introduces vulnerabilities with security implications. While prior work has examined prompt-based attacks (e.g., prompt injection) and data-oriented threats (e.g., data exfiltration), time-of-check to time-of-use (TOCTOU) remain largely unexplored in this context. TOCTOU arises when an agent validates external state (e.g., a file or API response) that is later modified before use, enabling practical attacks such as malicious configuration swaps or payload injection. In this work, we present the first study of TOCTOU vulnerabilities in LLM-enabled agents. We introduce TOCTOU-Bench, a benchmark with 66 realistic user tasks designed to evaluate this class of vulnerabilities. As countermeasures, we adapt detection and mitigation techniques from systems security to this setting and propose prompt rewriting, state integrity monitoring, and tool-fusing. Our study highlights challenges unique to agentic workflows, where we achieve up to 25% detection accuracy using automated detection methods, a 3% decrease in vulnerable plan generation, and a 95% reduction in the attack window. When combining all three approaches, we reduce the TOCTOU vulnerabilities from an executed trajectory from 12% to 8%. Our findings open a new research direction at the intersection of AI safety and systems security. 

---
# Natural Language Satisfiability: Exploring the Problem Distribution and Evaluating Transformer-based Language Models 

**Authors**: Tharindu Madusanka, Ian Pratt-Hartmann, Riza Batista-Navarro  

**Link**: [PDF](https://arxiv.org/pdf/2508.17153)  

**Abstract**: Efforts to apply transformer-based language models (TLMs) to the problem of reasoning in natural language have enjoyed ever-increasing success in recent years. The most fundamental task in this area to which nearly all others can be reduced is that of determining satisfiability. However, from a logical point of view, satisfiability problems vary along various dimensions, which may affect TLMs' ability to learn how to solve them. The problem instances of satisfiability in natural language can belong to different computational complexity classes depending on the language fragment in which they are expressed. Although prior research has explored the problem of natural language satisfiability, the above-mentioned point has not been discussed adequately. Hence, we investigate how problem instances from varying computational complexity classes and having different grammatical constructs impact TLMs' ability to learn rules of inference. Furthermore, to faithfully evaluate TLMs, we conduct an empirical study to explore the distribution of satisfiability problems. 

---
# SACA: Selective Attention-Based Clustering Algorithm 

**Authors**: Meysam Shirdel Bilehsavar, Razieh Ghaedi, Samira Seyed Taheri, Xinqi Fan, Christian O'Reilly  

**Link**: [PDF](https://arxiv.org/pdf/2508.17150)  

**Abstract**: Clustering algorithms are widely used in various applications, with density-based methods such as Density-Based Spatial Clustering of Applications with Noise (DBSCAN) being particularly prominent. These algorithms identify clusters in high-density regions while treating sparser areas as noise. However, reliance on user-defined parameters often poses optimization challenges that require domain expertise. This paper presents a novel density-based clustering method inspired by the concept of selective attention, which minimizes the need for user-defined parameters under standard conditions. Initially, the algorithm operates without requiring user-defined parameters. If parameter adjustment is needed, the method simplifies the process by introducing a single integer parameter that is straightforward to tune. The approach computes a threshold to filter out the most sparsely distributed points and outliers, forms a preliminary cluster structure, and then reintegrates the excluded points to finalize the results. Experimental evaluations on diverse data sets highlight the accessibility and robust performance of the method, providing an effective alternative for density-based clustering tasks. 

---
# CE-RS-SBCIT A Novel Channel Enhanced Hybrid CNN Transformer with Residual, Spatial, and Boundary-Aware Learning for Brain Tumor MRI Analysis 

**Authors**: Mirza Mumtaz Zahoor, Saddam Hussain Khan  

**Link**: [PDF](https://arxiv.org/pdf/2508.17128)  

**Abstract**: Brain tumors remain among the most lethal human diseases, where early detection and accurate classification are critical for effective diagnosis and treatment planning. Although deep learning-based computer-aided diagnostic (CADx) systems have shown remarkable progress. However, conventional convolutional neural networks (CNNs) and Transformers face persistent challenges, including high computational cost, sensitivity to minor contrast variations, structural heterogeneity, and texture inconsistencies in MRI data. Therefore, a novel hybrid framework, CE-RS-SBCIT, is introduced, integrating residual and spatial learning-based CNNs with transformer-driven modules. The proposed framework exploits local fine-grained and global contextual cues through four core innovations: (i) a smoothing and boundary-based CNN-integrated Transformer (SBCIT), (ii) tailored residual and spatial learning CNNs, (iii) a channel enhancement (CE) strategy, and (iv) a novel spatial attention mechanism. The developed SBCIT employs stem convolution and contextual interaction transformer blocks with systematic smoothing and boundary operations, enabling efficient global feature modeling. Moreover, Residual and spatial CNNs, enhanced by auxiliary transfer-learned feature maps, enrich the representation space, while the CE module amplifies discriminative channels and mitigates redundancy. Furthermore, the spatial attention mechanism selectively emphasizes subtle contrast and textural variations across tumor classes. Extensive evaluation on challenging MRI datasets from Kaggle and Figshare, encompassing glioma, meningioma, pituitary tumors, and healthy controls, demonstrates superior performance, achieving 98.30% accuracy, 98.08% sensitivity, 98.25% F1-score, and 98.43% precision. 

---
# Token Homogenization under Positional Bias 

**Authors**: Viacheslav Yusupov, Danil Maksimov, Ameliia Alaeva, Tatiana Zaitceva, Antipina Anna, Anna Vasileva, Chenlin Liu, Rayuth Chheng, Danil Sazanakov, Andrey Chetvergov, Alina Ermilova, Egor Shvetsov  

**Link**: [PDF](https://arxiv.org/pdf/2508.17126)  

**Abstract**: This paper investigates token homogenization - the convergence of token representations toward uniformity across transformer layers and its relationship to positional bias in large language models. We empirically examine whether homogenization occurs and how positional bias amplifies this effect. Through layer-wise similarity analysis and controlled experiments, we demonstrate that tokens systematically lose distinctiveness during processing, particularly when biased toward extremal positions. Our findings confirm both the existence of homogenization and its dependence on positional attention mechanisms. 

---
# PlantVillageVQA: A Visual Question Answering Dataset for Benchmarking Vision-Language Models in Plant Science 

**Authors**: Syed Nazmus Sakib, Nafiul Haque, Mohammad Zabed Hossain, Shifat E. Arman  

**Link**: [PDF](https://arxiv.org/pdf/2508.17117)  

**Abstract**: PlantVillageVQA is a large-scale visual question answering (VQA) dataset derived from the widely used PlantVillage image corpus. It was designed to advance the development and evaluation of vision-language models for agricultural decision-making and analysis. The PlantVillageVQA dataset comprises 193,609 high-quality question-answer (QA) pairs grounded over 55,448 images spanning 14 crop species and 38 disease conditions. Questions are organised into 3 levels of cognitive complexity and 9 distinct categories. Each question category was phrased manually following expert guidance and generated via an automated two-stage pipeline: (1) template-based QA synthesis from image metadata and (2) multi-stage linguistic re-engineering. The dataset was iteratively reviewed by domain experts for scientific accuracy and relevancy. The final dataset was evaluated using three state-of-the-art models for quality assessment. Our objective remains to provide a publicly available, standardised and expert-verified database to enhance diagnostic accuracy for plant disease identifications and advance scientific research in the agricultural domain. Our dataset will be open-sourced at this https URL. 

---
# Two Birds with One Stone: Enhancing Uncertainty Quantification and Interpretability with Graph Functional Neural Process 

**Authors**: Lingkai Kong, Haotian Sun, Yuchen Zhuang, Haorui Wang, Wenhao Mu, Chao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.17097)  

**Abstract**: Graph neural networks (GNNs) are powerful tools on graph data. However, their predictions are mis-calibrated and lack interpretability, limiting their adoption in critical applications. To address this issue, we propose a new uncertainty-aware and interpretable graph classification model that combines graph functional neural process and graph generative model. The core of our method is to assume a set of latent rationales which can be mapped to a probabilistic embedding space; the predictive distribution of the classifier is conditioned on such rationale embeddings by learning a stochastic correlation matrix. The graph generator serves to decode the graph structure of the rationales from the embedding space for model interpretability. For efficient model training, we adopt an alternating optimization procedure which mimics the well known Expectation-Maximization (EM) algorithm. The proposed method is general and can be applied to any existing GNN architecture. Extensive experiments on five graph classification datasets demonstrate that our framework outperforms state-of-the-art methods in both uncertainty quantification and GNN interpretability. We also conduct case studies to show that the decoded rationale structure can provide meaningful explanations. 

---
# Convolutional Neural Networks for Accurate Measurement of Train Speed 

**Authors**: Haitao Tian, Argyrios Zolotas, Miguel Arana-Catania  

**Link**: [PDF](https://arxiv.org/pdf/2508.17096)  

**Abstract**: In this study, we explore the use of Convolutional Neural Networks for improving train speed estimation accuracy, addressing the complex challenges of modern railway systems. We investigate three CNN architectures - single-branch 2D, single-branch 1D, and multiple-branch models - and compare them with the Adaptive Kalman Filter. We analyse their performance using simulated train operation datasets with and without Wheel Slide Protection activation. Our results reveal that CNN-based approaches, especially the multiple-branch model, demonstrate superior accuracy and robustness compared to traditional methods, particularly under challenging operational conditions. These findings highlight the potential of deep learning techniques to enhance railway safety and operational efficiency by more effectively capturing intricate patterns in complex transportation datasets. 

---
# Enhancing Knowledge Tracing through Leakage-Free and Recency-Aware Embeddings 

**Authors**: Yahya Badran, Christine Preisach  

**Link**: [PDF](https://arxiv.org/pdf/2508.17092)  

**Abstract**: Knowledge Tracing (KT) aims to predict a student's future performance based on their sequence of interactions with learning content. Many KT models rely on knowledge concepts (KCs), which represent the skills required for each item. However, some of these models are vulnerable to label leakage, in which input data inadvertently reveal the correct answer, particularly in datasets with multiple KCs per question.
We propose a straightforward yet effective solution to prevent label leakage by masking ground-truth labels during input embedding construction in cases susceptible to leakage. To accomplish this, we introduce a dedicated MASK label, inspired by masked language modeling (e.g., BERT), to replace ground-truth labels. In addition, we introduce Recency Encoding, which encodes the step-wise distance between the current item and its most recent previous occurrence. This distance is important for modeling learning dynamics such as forgetting, which is a fundamental aspect of human learning, yet it is often overlooked in existing models. Recency Encoding demonstrates improved performance over traditional positional encodings on multiple KT benchmarks.
We show that incorporating our embeddings into KT models like DKT, DKT+, AKT, and SAKT consistently improves prediction accuracy across multiple benchmarks. The approach is both efficient and widely applicable. 

---
# Proximal Vision Transformer: Enhancing Feature Representation through Two-Stage Manifold Geometry 

**Authors**: Haoyu Yun, Hamid Krim  

**Link**: [PDF](https://arxiv.org/pdf/2508.17081)  

**Abstract**: The Vision Transformer (ViT) architecture has become widely recognized in computer vision, leveraging its self-attention mechanism to achieve remarkable success across various tasks. Despite its strengths, ViT's optimization remains confined to modeling local relationships within individual images, limiting its ability to capture the global geometric relationships between data points. To address this limitation, this paper proposes a novel framework that integrates ViT with the proximal tools, enabling a unified geometric optimization approach to enhance feature representation and classification performance. In this framework, ViT constructs the tangent bundle of the manifold through its self-attention mechanism, where each attention head corresponds to a tangent space, offering geometric representations from diverse local perspectives. Proximal iterations are then introduced to define sections within the tangent bundle and project data from tangent spaces onto the base space, achieving global feature alignment and optimization. Experimental results confirm that the proposed method outperforms traditional ViT in terms of classification accuracy and data distribution. 

---
# Zero-shot Multimodal Document Retrieval via Cross-modal Question Generation 

**Authors**: Yejin Choi, Jaewoo Park, Janghan Yoon, Saejin Kim, Jaehyun Jeon, Youngjae Yu  

**Link**: [PDF](https://arxiv.org/pdf/2508.17079)  

**Abstract**: Rapid advances in Multimodal Large Language Models (MLLMs) have expanded information retrieval beyond purely textual inputs, enabling retrieval from complex real world documents that combine text and visuals. However, most documents are private either owned by individuals or confined within corporate silos and current retrievers struggle when faced with unseen domains or languages. To address this gap, we introduce PREMIR, a simple yet effective framework that leverages the broad knowledge of an MLLM to generate cross modal pre questions (preQs) before retrieval. Unlike earlier multimodal retrievers that compare embeddings in a single vector space, PREMIR leverages preQs from multiple complementary modalities to expand the scope of matching to the token level. Experiments show that PREMIR achieves state of the art performance on out of distribution benchmarks, including closed domain and multilingual settings, outperforming strong baselines across all retrieval metrics. We confirm the contribution of each component through in depth ablation studies, and qualitative analyses of the generated preQs further highlight the model's robustness in real world settings. 

---
# Linguistic Neuron Overlap Patterns to Facilitate Cross-lingual Transfer on Low-resource Languages 

**Authors**: Yuemei Xu, Kexin Xu, Jian Zhou, Ling Hu, Lin Gui  

**Link**: [PDF](https://arxiv.org/pdf/2508.17078)  

**Abstract**: The current Large Language Models (LLMs) face significant challenges in improving performance on low-resource languages and urgently need data-efficient methods without costly fine-tuning. From the perspective of language-bridge, we propose BridgeX-ICL, a simple yet effective method to improve zero-shot Cross-lingual In-Context Learning (X-ICL) for low-resource languages. Unlike existing works focusing on language-specific neurons, BridgeX-ICL explores whether sharing neurons can improve cross-lingual performance in LLMs or not. We construct neuron probe data from the ground-truth MUSE bilingual dictionaries, and define a subset of language overlap neurons accordingly, to ensure full activation of these anchored neurons. Subsequently, we propose an HSIC-based metric to quantify LLMs' internal linguistic spectrum based on overlap neurons, which guides optimal bridge selection. The experiments conducted on 2 cross-lingual tasks and 15 language pairs from 7 diverse families (covering both high-low and moderate-low pairs) validate the effectiveness of BridgeX-ICL and offer empirical insights into the underlying multilingual mechanisms of LLMs. 

---
# Optimizing Neural Networks with Learnable Non-Linear Activation Functions via Lookup-Based FPGA Acceleration 

**Authors**: Mengyuan Yin, Benjamin Chen Ming Choong, Chuping Qu, Rick Siow Mong Goh, Weng-Fai Wong, Tao Luo  

**Link**: [PDF](https://arxiv.org/pdf/2508.17069)  

**Abstract**: Learned activation functions in models like Kolmogorov-Arnold Networks (KANs) outperform fixed-activation architectures in terms of accuracy and interpretability; however, their computational complexity poses critical challenges for energy-constrained edge AI deployments. Conventional CPUs/GPUs incur prohibitive latency and power costs when evaluating higher order activations, limiting deployability under ultra-tight energy budgets. We address this via a reconfigurable lookup architecture with edge FPGAs. By coupling fine-grained quantization with adaptive lookup tables, our design minimizes energy-intensive arithmetic operations while preserving activation fidelity. FPGA reconfigurability enables dynamic hardware specialization for learned functions, a key advantage for edge systems that require post-deployment adaptability. Evaluations using KANs - where unique activation functions play a critical role - demonstrate that our FPGA-based design achieves superior computational speed and over $10^4$ times higher energy efficiency compared to edge CPUs and GPUs, while maintaining matching accuracy and minimal footprint overhead. This breakthrough positions our approach as a practical enabler for energy-critical edge AI, where computational intensity and power constraints traditionally preclude the use of adaptive activation networks. 

---
# SSG-Dit: A Spatial Signal Guided Framework for Controllable Video Generation 

**Authors**: Peng Hu, Yu Gu, Liang Luo, Fuji Ren  

**Link**: [PDF](https://arxiv.org/pdf/2508.17062)  

**Abstract**: Controllable video generation aims to synthesize video content that aligns precisely with user-provided conditions, such as text descriptions and initial images. However, a significant challenge persists in this domain: existing models often struggle to maintain strong semantic consistency, frequently generating videos that deviate from the nuanced details specified in the prompts. To address this issue, we propose SSG-DiT (Spatial Signal Guided Diffusion Transformer), a novel and efficient framework for high-fidelity controllable video generation. Our approach introduces a decoupled two-stage process. The first stage, Spatial Signal Prompting, generates a spatially aware visual prompt by leveraging the rich internal representations of a pre-trained multi-modal model. This prompt, combined with the original text, forms a joint condition that is then injected into a frozen video DiT backbone via our lightweight and parameter-efficient SSG-Adapter. This unique design, featuring a dual-branch attention mechanism, allows the model to simultaneously harness its powerful generative priors while being precisely steered by external spatial signals. Extensive experiments demonstrate that SSG-DiT achieves state-of-the-art performance, outperforming existing models on multiple key metrics in the VBench benchmark, particularly in spatial relationship control and overall consistency. 

---
# TabResFlow: A Normalizing Spline Flow Model for Probabilistic Univariate Tabular Regression 

**Authors**: Kiran Madhusudhanan, Vijaya Krishna Yalavarthi, Jonas Sonntag, Maximilian Stubbemann, Lars Schmidt-Thieme  

**Link**: [PDF](https://arxiv.org/pdf/2508.17056)  

**Abstract**: Tabular regression is a well-studied problem with numerous industrial applications, yet most existing approaches focus on point estimation, often leading to overconfident predictions. This issue is particularly critical in industrial automation, where trustworthy decision-making is essential. Probabilistic regression models address this challenge by modeling prediction uncertainty. However, many conventional methods assume a fixed-shape distribution (typically Gaussian), and resort to estimating distribution parameters. This assumption is often restrictive, as real-world target distributions can be highly complex. To overcome this limitation, we introduce TabResFlow, a Normalizing Spline Flow model designed specifically for univariate tabular regression, where commonly used simple flow networks like RealNVP and Masked Autoregressive Flow (MAF) are unsuitable. TabResFlow consists of three key components: (1) An MLP encoder for each numerical feature. (2) A fully connected ResNet backbone for expressive feature extraction. (3) A conditional spline-based normalizing flow for flexible and tractable density estimation. We evaluate TabResFlow on nine public benchmark datasets, demonstrating that it consistently surpasses existing probabilistic regression models on likelihood scores. Our results demonstrate 9.64% improvement compared to the strongest probabilistic regression model (TreeFlow), and on average 5.6 times speed-up in inference time compared to the strongest deep learning alternative (NodeFlow). Additionally, we validate the practical applicability of TabResFlow in a real-world used car price prediction task under selective regression. To measure performance in this setting, we introduce a novel Area Under Risk Coverage (AURC) metric and show that TabResFlow achieves superior results across this metric. 

---
# An Efficient Dual-Line Decoder Network with Multi-Scale Convolutional Attention for Multi-organ Segmentation 

**Authors**: Riad Hassan, M. Rubaiyat Hossain Mondal, Sheikh Iqbal Ahamed, Fahad Mostafa, Md Mostafijur Rahman  

**Link**: [PDF](https://arxiv.org/pdf/2508.17007)  

**Abstract**: Proper segmentation of organs-at-risk is important for radiation therapy, surgical planning, and diagnostic decision-making in medical image analysis. While deep learning-based segmentation architectures have made significant progress, they often fail to balance segmentation accuracy with computational efficiency. Most of the current state-of-the-art methods either prioritize performance at the cost of high computational complexity or compromise accuracy for efficiency. This paper addresses this gap by introducing an efficient dual-line decoder segmentation network (EDLDNet). The proposed method features a noisy decoder, which learns to incorporate structured perturbation at training time for better model robustness, yet at inference time only the noise-free decoder is executed, leading to lower computational cost. Multi-Scale convolutional Attention Modules (MSCAMs), Attention Gates (AGs), and Up-Convolution Blocks (UCBs) are further utilized to optimize feature representation and boost segmentation performance. By leveraging multi-scale segmentation masks from both decoders, we also utilize a mutation-based loss function to enhance the model's generalization. Our approach outperforms SOTA segmentation architectures on four publicly available medical imaging datasets. EDLDNet achieves SOTA performance with an 84.00% Dice score on the Synapse dataset, surpassing baseline model like UNet by 13.89% in Dice score while significantly reducing Multiply-Accumulate Operations (MACs) by 89.7%. Compared to recent approaches like EMCAD, our EDLDNet not only achieves higher Dice score but also maintains comparable computational efficiency. The outstanding performance across diverse datasets establishes EDLDNet's strong generalization, computational efficiency, and robustness. The source code, pre-processed data, and pre-trained weights will be available at this https URL . 

---
# GRADE: Generating multi-hop QA and fine-gRAined Difficulty matrix for RAG Evaluation 

**Authors**: Jeongsoo Lee, Daeyong Kwon, Kyohoon Jin  

**Link**: [PDF](https://arxiv.org/pdf/2508.16994)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems are widely adopted in knowledge-intensive NLP tasks, but current evaluations often overlook the structural complexity and multi-step reasoning required in real-world scenarios. These benchmarks overlook key factors such as the interaction between retrieval difficulty and reasoning depth. To address this gap, we propose \textsc{GRADE}, a novel evaluation framework that models task difficulty along two orthogonal dimensions: (1) reasoning depth, defined by the number of inference steps (hops), and (2) semantic distance between the query and its supporting evidence. We construct a synthetic multi-hop QA dataset from factual news articles by extracting knowledge graphs and augmenting them through semantic clustering to recover missing links, allowing us to generate diverse and difficulty-controlled queries. Central to our framework is a 2D difficulty matrix that combines generator-side and retriever-side difficulty. Experiments across multiple domains and models show that error rates strongly correlate with our difficulty measures, validating their diagnostic utility. \textsc{GRADE} enables fine-grained analysis of RAG performance and provides a scalable foundation for evaluating and improving multi-hop reasoning in real-world applications. 

---
# Score Matching on Large Geometric Graphs for Cosmology Generation 

**Authors**: Diana-Alexandra Onutu, Yue Zhao, Joaquin Vanschoren, Vlado Menkovski  

**Link**: [PDF](https://arxiv.org/pdf/2508.16990)  

**Abstract**: Generative models are a promising tool to produce cosmological simulations but face significant challenges in scalability, physical consistency, and adherence to domain symmetries, limiting their utility as alternatives to $N$-body simulations. To address these limitations, we introduce a score-based generative model with an equivariant graph neural network that simulates gravitational clustering of galaxies across cosmologies starting from an informed prior, respects periodic boundaries, and scales to full galaxy counts in simulations. A novel topology-aware noise schedule, crucial for large geometric graphs, is introduced. The proposed equivariant score-based model successfully generates full-scale cosmological point clouds of up to 600,000 halos, respects periodicity and a uniform prior, and outperforms existing diffusion models in capturing clustering statistics while offering significant computational advantages. This work advances cosmology by introducing a generative model designed to closely resemble the underlying gravitational clustering of structure formation, moving closer to physically realistic and efficient simulators for the evolution of large-scale structures in the universe. 

---
# ReFactX: Scalable Reasoning with Reliable Facts via Constrained Generation 

**Authors**: Riccardo Pozzi, Matteo Palmonari, Andrea Coletta, Luigi Bellomarini, Jens Lehmann, Sahar Vahdati  

**Link**: [PDF](https://arxiv.org/pdf/2508.16983)  

**Abstract**: Knowledge gaps and hallucinations are persistent challenges for Large Language Models (LLMs), which generate unreliable responses when lacking the necessary information to fulfill user instructions. Existing approaches, such as Retrieval-Augmented Generation (RAG) and tool use, aim to address these issues by incorporating external knowledge. Yet, they rely on additional models or services, resulting in complex pipelines, potential error propagation, and often requiring the model to process a large number of tokens. In this paper, we present a scalable method that enables LLMs to access external knowledge without depending on retrievers or auxiliary models. Our approach uses constrained generation with a pre-built prefix-tree index. Triples from a Knowledge Graph are verbalized in textual facts, tokenized, and indexed in a prefix tree for efficient access. During inference, to acquire external knowledge, the LLM generates facts with constrained generation which allows only sequences of tokens that form an existing fact. We evaluate our proposal on Question Answering and show that it scales to large knowledge bases (800 million facts), adapts to domain-specific data, and achieves effective results. These gains come with minimal generation-time overhead. ReFactX code is available at this https URL. 

---
# Combating Digitally Altered Images: Deepfake Detection 

**Authors**: Saksham Kumar, Rhythm Narang  

**Link**: [PDF](https://arxiv.org/pdf/2508.16975)  

**Abstract**: The rise of Deepfake technology to generate hyper-realistic manipulated images and videos poses a significant challenge to the public and relevant authorities. This study presents a robust Deepfake detection based on a modified Vision Transformer(ViT) model, trained to distinguish between real and Deepfake images. The model has been trained on a subset of the OpenForensics Dataset with multiple augmentation techniques to increase robustness for diverse image manipulations. The class imbalance issues are handled by oversampling and a train-validation split of the dataset in a stratified manner. Performance is evaluated using the accuracy metric on the training and testing datasets, followed by a prediction score on a random image of people, irrespective of their realness. The model demonstrates state-of-the-art results on the test dataset to meticulously detect Deepfake images. 

---
# Explaining Black-box Language Models with Knowledge Probing Systems: A Post-hoc Explanation Perspective 

**Authors**: Yunxiao Zhao, Hao Xu, Zhiqiang Wang, Xiaoli Li, Jiye Liang, Ru Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.16969)  

**Abstract**: Pre-trained Language Models (PLMs) are trained on large amounts of unlabeled data, yet they exhibit remarkable reasoning skills. However, the trustworthiness challenges posed by these black-box models have become increasingly evident in recent years. To alleviate this problem, this paper proposes a novel Knowledge-guided Probing approach called KnowProb in a post-hoc explanation way, which aims to probe whether black-box PLMs understand implicit knowledge beyond the given text, rather than focusing only on the surface level content of the text. We provide six potential explanations derived from the underlying content of the given text, including three knowledge-based understanding and three association-based reasoning. In experiments, we validate that current small-scale (or large-scale) PLMs only learn a single distribution of representation, and still face significant challenges in capturing the hidden knowledge behind a given text. Furthermore, we demonstrate that our proposed approach is effective for identifying the limitations of existing black-box models from multiple probing perspectives, which facilitates researchers to promote the study of detecting black-box models in an explainable way. 

---
# LLM-based Human-like Traffic Simulation for Self-driving Tests 

**Authors**: Wendi Li, Hao Wu, Han Gao, Bing Mao, Fengyuan Xu, Sheng Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2508.16962)  

**Abstract**: Ensuring realistic traffic dynamics is a prerequisite for simulation platforms to evaluate the reliability of self-driving systems before deployment in the real world. Because most road users are human drivers, reproducing their diverse behaviors within simulators is vital. Existing solutions, however, typically rely on either handcrafted heuristics or narrow data-driven models, which capture only fragments of real driving behaviors and offer limited driving style diversity and interpretability. To address this gap, we introduce HDSim, an HD traffic generation framework that combines cognitive theory with large language model (LLM) assistance to produce scalable and realistic traffic scenarios within simulation platforms. The framework advances the state of the art in two ways: (i) it introduces a hierarchical driver model that represents diverse driving style traits, and (ii) it develops a Perception-Mediated Behavior Influence strategy, where LLMs guide perception to indirectly shape driver actions. Experiments reveal that embedding HDSim into simulation improves detection of safety-critical failures in self-driving systems by up to 68% and yields realism-consistent accident interpretability. 

---
# Breaking the Exploration Bottleneck: Rubric-Scaffolded Reinforcement Learning for General LLM Reasoning 

**Authors**: Yang Zhou, Sunzhu Li, Shunyu Liu, Wenkai Fang, Jiale Zhao, Jingwen Yang, Jianwei Lv, Kongcheng Zhang, Yihe Zhou, Hengtong Lu, Wei Chen, Yan Xie, Mingli Song  

**Link**: [PDF](https://arxiv.org/pdf/2508.16949)  

**Abstract**: Recent advances in Large Language Models (LLMs) have underscored the potential of Reinforcement Learning (RL) to facilitate the emergence of reasoning capabilities. Despite the encouraging results, a fundamental dilemma persists as RL improvement relies on learning from high-quality samples, yet the exploration for such samples remains bounded by the inherent limitations of LLMs. This, in effect, creates an undesirable cycle in which what cannot be explored cannot be learned. In this work, we propose Rubric-Scaffolded Reinforcement Learning (RuscaRL), a novel instructional scaffolding framework designed to break the exploration bottleneck for general LLM reasoning. Specifically, RuscaRL introduces checklist-style rubrics as (1) explicit scaffolding for exploration during rollout generation, where different rubrics are provided as external guidance within task instructions to steer diverse high-quality responses. This guidance is gradually decayed over time, encouraging the model to internalize the underlying reasoning patterns; (2) verifiable rewards for exploitation during model training, where we can obtain robust LLM-as-a-Judge scores using rubrics as references, enabling effective RL on general reasoning tasks. Extensive experiments demonstrate the superiority of the proposed RuscaRL across various benchmarks, effectively expanding reasoning boundaries under the best-of-N evaluation. Notably, RuscaRL significantly boosts Qwen-2.5-7B-Instruct from 23.6 to 50.3 on HealthBench-500, surpassing GPT-4.1. Furthermore, our fine-tuned variant on Qwen3-30B-A3B-Instruct achieves 61.1 on HealthBench-500, outperforming leading LLMs including OpenAI-o3. 

---
# Drive As You Like: Strategy-Level Motion Planning Based on A Multi-Head Diffusion Model 

**Authors**: Fan Ding, Xuewen Luo, Hwa Hui Tew, Ruturaj Reddy, Xikun Wang, Junn Yong Loo  

**Link**: [PDF](https://arxiv.org/pdf/2508.16947)  

**Abstract**: Recent advances in motion planning for autonomous driving have led to models capable of generating high-quality trajectories. However, most existing planners tend to fix their policy after supervised training, leading to consistent but rigid driving behaviors. This limits their ability to reflect human preferences or adapt to dynamic, instruction-driven demands. In this work, we propose a diffusion-based multi-head trajectory planner(M-diffusion planner). During the early training stage, all output heads share weights to learn to generate high-quality trajectories. Leveraging the probabilistic nature of diffusion models, we then apply Group Relative Policy Optimization (GRPO) to fine-tune the pre-trained model for diverse policy-specific behaviors. At inference time, we incorporate a large language model (LLM) to guide strategy selection, enabling dynamic, instruction-aware planning without switching models. Closed-loop simulation demonstrates that our post-trained planner retains strong planning capability while achieving state-of-the-art (SOTA) performance on the nuPlan val14 benchmark. Open-loop results further show that the generated trajectories exhibit clear diversity, effectively satisfying multi-modal driving behavior requirements. The code and related experiments will be released upon acceptance of the paper. 

---
# HumanoidVerse: A Versatile Humanoid for Vision-Language Guided Multi-Object Rearrangement 

**Authors**: Haozhuo Zhang, Jingkai Sun, Michele Caprio, Jian Tang, Shanghang Zhang, Qiang Zhang, Wei Pan  

**Link**: [PDF](https://arxiv.org/pdf/2508.16943)  

**Abstract**: We introduce HumanoidVerse, a novel framework for vision-language guided humanoid control that enables a single physically simulated robot to perform long-horizon, multi-object rearrangement tasks across diverse scenes. Unlike prior methods that operate in fixed settings with single-object interactions, our approach supports consecutive manipulation of multiple objects, guided only by natural language instructions and egocentric camera RGB observations. HumanoidVerse is trained via a multi-stage curriculum using a dual-teacher distillation pipeline, enabling fluid transitions between sub-tasks without requiring environment resets. To support this, we construct a large-scale dataset comprising 350 multi-object tasks spanning four room layouts. Extensive experiments in the Isaac Gym simulator demonstrate that our method significantly outperforms prior state-of-the-art in both task success rate and spatial precision, and generalizes well to unseen environments and instructions. Our work represents a key step toward robust, general-purpose humanoid agents capable of executing complex, sequential tasks under real-world sensory constraints. The video visualization results can be found on the project page: this https URL. 

---
# THEME : Enhancing Thematic Investing with Semantic Stock Representations and Temporal Dynamics 

**Authors**: Hoyoung Lee, Wonbin Ahn, Suhwan Park, Jaehoon Lee, Minjae Kim, Sungdong Yoo, Taeyoon Lim, Woohyung Lim, Yongjae Lee  

**Link**: [PDF](https://arxiv.org/pdf/2508.16936)  

**Abstract**: Thematic investing aims to construct portfolios aligned with structural trends, yet selecting relevant stocks remains challenging due to overlapping sector boundaries and evolving market dynamics. To address this challenge, we construct the Thematic Representation Set (TRS), an extended dataset that begins with real-world thematic ETFs and expands upon them by incorporating industry classifications and financial news to overcome their coverage limitations. The final dataset contains both the explicit mapping of themes to their constituent stocks and the rich textual profiles for each. Building on this dataset, we introduce \textsc{THEME}, a hierarchical contrastive learning framework. By representing the textual profiles of themes and stocks as embeddings, \textsc{THEME} first leverages their hierarchical relationship to achieve semantic alignment. Subsequently, it refines these semantic embeddings through a temporal refinement stage that incorporates individual stock returns. The final stock representations are designed for effective retrieval of thematically aligned assets with strong return potential. Empirical results show that \textsc{THEME} outperforms strong baselines across multiple retrieval metrics and significantly improves performance in portfolio construction. By jointly modeling thematic relationships from text and market dynamics from returns, \textsc{THEME} provides a scalable and adaptive solution for navigating complex investment themes. 

---
# Degree of Staleness-Aware Data Updating in Federated Learning 

**Authors**: Tao Liu, Xuehe Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.16931)  

**Abstract**: Handling data staleness remains a significant challenge in federated learning with highly time-sensitive tasks, where data is generated continuously and data staleness largely affects model performance. Although recent works attempt to optimize data staleness by determining local data update frequency or client selection strategy, none of them explore taking both data staleness and data volume into consideration. In this paper, we propose DUFL(Data Updating in Federated Learning), an incentive mechanism featuring an innovative local data update scheme manipulated by three knobs: the server's payment, outdated data conservation rate, and clients' fresh data collection volume, to coordinate staleness and volume of local data for best utilities. To this end, we introduce a novel metric called DoS(the Degree of Staleness) to quantify data staleness and conduct a theoretic analysis illustrating the quantitative relationship between DoS and model performance. We model DUFL as a two-stage Stackelberg game with dynamic constraint, deriving the optimal local data update strategy for each client in closed-form and the approximately optimal strategy for the server. Experimental results on real-world datasets demonstrate the significant performance of our approach. 

---
# TextOnly: A Unified Function Portal for Text-Related Functions on Smartphones 

**Authors**: Minghao Tu, Chun Yu, Xiyuan Shen, Zhi Zheng, Li Chen, Yuanchun Shi  

**Link**: [PDF](https://arxiv.org/pdf/2508.16926)  

**Abstract**: Text boxes serve as portals to diverse functionalities in today's smartphone applications. However, when it comes to specific functionalities, users always need to navigate through multiple steps to access particular text boxes for input. We propose TextOnly, a unified function portal that enables users to access text-related functions from various applications by simply inputting text into a sole text box. For instance, entering a restaurant name could trigger a Google Maps search, while a greeting could initiate a conversation in WhatsApp. Despite their brevity, TextOnly maximizes the utilization of these raw text inputs, which contain rich information, to interpret user intentions effectively. TextOnly integrates large language models(LLM) and a BERT model. The LLM consistently provides general knowledge, while the BERT model can continuously learn user-specific preferences and enable quicker predictions. Real-world user studies demonstrated TextOnly's effectiveness with a top-1 accuracy of 71.35%, and its ability to continuously improve both its accuracy and inference speed. Participants perceived TextOnly as having satisfactory usability and expressed a preference for TextOnly over manual executions. Compared with voice assistants, TextOnly supports a greater range of text-related functions and allows for more concise inputs. 

---
# Tri-Accel: Curvature-Aware Precision-Adaptive and Memory-Elastic Optimization for Efficient GPU Usage 

**Authors**: Mohsen Sheibanian, Pouya Shaeri, Alimohammad Beigi, Ryan T. Woo, Aryan Keluskar  

**Link**: [PDF](https://arxiv.org/pdf/2508.16905)  

**Abstract**: Deep neural networks are increasingly bottlenecked by the cost of optimization, both in terms of GPU memory and compute time. Existing acceleration techniques, such as mixed precision, second-order methods, and batch size scaling, are typically used in isolation. We present Tri-Accel, a unified optimization framework that co-adapts three acceleration strategies along with adaptive parameters during training: (1) Precision-Adaptive Updates that dynamically assign mixed-precision levels to layers based on curvature and gradient variance; (2) Sparse Second-Order Signals that exploit Hessian/Fisher sparsity patterns to guide precision and step size decisions; and (3) Memory-Elastic Batch Scaling that adjusts batch size in real time according to VRAM availability. On CIFAR-10 with ResNet-18 and EfficientNet-B0, Tri-Accel achieves up to 9.9% reduction in training time and 13.3% lower memory usage, while improving accuracy by +1.1 percentage points over FP32 baselines. Tested on CIFAR-10/100, our approach demonstrates adaptive learning behavior, with efficiency gradually improving over the course of training as the system learns to allocate resources more effectively. Compared to static mixed-precision training, Tri-Accel maintains 78.1% accuracy while reducing memory footprint from 0.35GB to 0.31GB on standard hardware. The framework is implemented with custom Triton kernels, whose hardware-aware adaptation enables automatic optimization without manual hyperparameter tuning, making it practical for deployment across diverse computational environments. This work demonstrates how algorithmic adaptivity and hardware awareness can be combined to improve scalability in resource-constrained settings, paving the way for more efficient neural network training on edge devices and cost-sensitive cloud deployments. 

---
# Dream to Chat: Model-based Reinforcement Learning on Dialogues with User Belief Modeling 

**Authors**: Yue Zhao, Xiaoyu Wang, Dan Wang, Zhonglin Jiang, Qingqing Gu, Teng Chen, Ningyuan Xi, Jinxian Qu, Yong Chen, Luo Ji  

**Link**: [PDF](https://arxiv.org/pdf/2508.16876)  

**Abstract**: World models have been widely utilized in robotics, gaming, and auto-driving. However, their applications on natural language tasks are relatively limited. In this paper, we construct the dialogue world model, which could predict the user's emotion, sentiment, and intention, and future utterances. By defining a POMDP, we argue emotion, sentiment and intention can be modeled as the user belief and solved by maximizing the information bottleneck. By this user belief modeling, we apply the model-based reinforcement learning framework to the dialogue system, and propose a framework called DreamCUB. Experiments show that the pretrained dialogue world model can achieve state-of-the-art performances on emotion classification and sentiment identification, while dialogue quality is also enhanced by joint training of the policy, critic and dialogue world model. Further analysis shows that this manner holds a reasonable exploration-exploitation balance and also transfers well to out-of-domain scenarios such as empathetic dialogues. 

---
# TriagerX: Dual Transformers for Bug Triaging Tasks with Content and Interaction Based Rankings 

**Authors**: Md Afif Al Mamun, Gias Uddin, Lan Xia, Longyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.16860)  

**Abstract**: Pretrained Language Models or PLMs are transformer-based architectures that can be used in bug triaging tasks. PLMs can better capture token semantics than traditional Machine Learning (ML) models that rely on statistical features (e.g., TF-IDF, bag of words). However, PLMs may still attend to less relevant tokens in a bug report, which can impact their effectiveness. In addition, the model can be sub-optimal with its recommendations when the interaction history of developers around similar bugs is not taken into account. We designed TriagerX to address these limitations. First, to assess token semantics more reliably, we leverage a dual-transformer architecture. Unlike current state-of-the-art (SOTA) baselines that employ a single transformer architecture, TriagerX collects recommendations from two transformers with each offering recommendations via its last three layers. This setup generates a robust content-based ranking of candidate developers. TriagerX then refines this ranking by employing a novel interaction-based ranking methodology, which considers developers' historical interactions with similar fixed bugs. Across five datasets, TriagerX surpasses all nine transformer-based methods, including SOTA baselines, often improving Top-1 and Top-3 developer recommendation accuracy by over 10%. We worked with our large industry partner to successfully deploy TriagerX in their development environment. The partner required both developer and component recommendations, with components acting as proxies for team assignments-particularly useful in cases of developer turnover or team changes. We trained TriagerX on the partner's dataset for both tasks, and it outperformed SOTA baselines by up to 10% for component recommendations and 54% for developer recommendations. 

---
# WildSpoof Challenge Evaluation Plan 

**Authors**: Yihan Wu, Jee-weon Jung, Hye-jin Shim, Xin Cheng, Xin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.16858)  

**Abstract**: The WildSpoof Challenge aims to advance the use of in-the-wild data in two intertwined speech processing tasks. It consists of two parallel tracks: (1) Text-to-Speech (TTS) synthesis for generating spoofed speech, and (2) Spoofing-robust Automatic Speaker Verification (SASV) for detecting spoofed speech. While the organizers coordinate both tracks and define the data protocols, participants treat them as separate and independent tasks. The primary objectives of the challenge are: (i) to promote the use of in-the-wild data for both TTS and SASV, moving beyond conventional clean and controlled datasets and considering real-world scenarios; and (ii) to encourage interdisciplinary collaboration between the spoofing generation (TTS) and spoofing detection (SASV) communities, thereby fostering the development of more integrated, robust, and realistic systems. 

---
# A Workflow for Map Creation in Autonomous Vehicle Simulations 

**Authors**: Zubair Islam, Ahmaad Ansari, George Daoud, Mohamed El-Darieby  

**Link**: [PDF](https://arxiv.org/pdf/2508.16856)  

**Abstract**: The fast development of technology and artificial intelligence has significantly advanced Autonomous Vehicle (AV) research, emphasizing the need for extensive simulation testing. Accurate and adaptable maps are critical in AV development, serving as the foundation for localization, path planning, and scenario testing. However, creating simulation-ready maps is often difficult and resource-intensive, especially with simulators like CARLA (CAR Learning to Act). Many existing workflows require significant computational resources or rely on specific simulators, limiting flexibility for developers. This paper presents a custom workflow to streamline map creation for AV development, demonstrated through the generation of a 3D map of a parking lot at Ontario Tech University. Future work will focus on incorporating SLAM technologies, optimizing the workflow for broader simulator compatibility, and exploring more flexible handling of latitude and longitude values to enhance map generation accuracy. 

---
# DevLicOps: A Framework for Mitigating Licensing Risks in AI-Generated Code 

**Authors**: Pratyush Nidhi Sharma, Lauren Wright, Anne Herfurth, Munsif Sokiyna, Pratyaksh Nidhi Sharma, Sethu Das, Mikko Siponen  

**Link**: [PDF](https://arxiv.org/pdf/2508.16853)  

**Abstract**: Generative AI coding assistants (ACAs) are widely adopted yet pose serious legal and compliance risks. ACAs can generate code governed by restrictive open-source licenses (e.g., GPL), potentially exposing companies to litigation or forced open-sourcing. Few developers are trained in these risks, and legal standards vary globally, especially with outsourcing. Our article introduces DevLicOps, a practical framework that helps IT leaders manage ACA-related licensing risks through governance, incident response, and informed tradeoffs. As ACA adoption grows and legal frameworks evolve, proactive license compliance is essential for responsible, risk-aware software development in the AI era. 

---
# Gaussian Primitive Optimized Deformable Retinal Image Registration 

**Authors**: Xin Tian, Jiazheng Wang, Yuxi Zhang, Xiang Chen, Renjiu Hu, Gaolei Li, Min Liu, Hang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.16852)  

**Abstract**: Deformable retinal image registration is notoriously difficult due to large homogeneous regions and sparse but critical vascular features, which cause limited gradient signals in standard learning-based frameworks. In this paper, we introduce Gaussian Primitive Optimization (GPO), a novel iterative framework that performs structured message passing to overcome these challenges. After an initial coarse alignment, we extract keypoints at salient anatomical structures (e.g., major vessels) to serve as a minimal set of descriptor-based control nodes (DCN). Each node is modelled as a Gaussian primitive with trainable position, displacement, and radius, thus adapting its spatial influence to local deformation scales. A K-Nearest Neighbors (KNN) Gaussian interpolation then blends and propagates displacement signals from these information-rich nodes to construct a globally coherent displacement field; focusing interpolation on the top (K) neighbors reduces computational overhead while preserving local detail. By strategically anchoring nodes in high-gradient regions, GPO ensures robust gradient flow, mitigating vanishing gradient signal in textureless areas. The framework is optimized end-to-end via a multi-term loss that enforces both keypoint consistency and intensity alignment. Experiments on the FIRE dataset show that GPO reduces the target registration error from 6.2\,px to ~2.4\,px and increases the AUC at 25\,px from 0.770 to 0.938, substantially outperforming existing methods. The source code can be accessed via this https URL. 

---
# NinA: Normalizing Flows in Action. Training VLA Models with Normalizing Flows 

**Authors**: Denis Tarasov, Alexander Nikulin, Ilya Zisman, Albina Klepach, Nikita Lyubaykin, Andrei Polubarov, Alexander Derevyagin, Vladislav Kurenkov  

**Link**: [PDF](https://arxiv.org/pdf/2508.16845)  

**Abstract**: Recent advances in Vision-Language-Action (VLA) models have established a two-component architecture, where a pre-trained Vision-Language Model (VLM) encodes visual observations and task descriptions, and an action decoder maps these representations to continuous actions. Diffusion models have been widely adopted as action decoders due to their ability to model complex, multimodal action distributions. However, they require multiple iterative denoising steps at inference time or downstream techniques to speed up sampling, limiting their practicality in real-world settings where high-frequency control is crucial. In this work, we present NinA (Normalizing Flows in Action), a fast and expressive alter- native to diffusion-based decoders for VLAs. NinA replaces the diffusion action decoder with a Normalizing Flow (NF) that enables one-shot sampling through an invertible transformation, significantly reducing inference time. We integrate NinA into the FLOWER VLA architecture and fine-tune on the LIBERO benchmark. Our experiments show that NinA matches the performance of its diffusion-based counterpart under the same training regime, while achieving substantially faster inference. These results suggest that NinA offers a promising path toward efficient, high-frequency VLA control without compromising performance. 

---
# A Survey of Threats Against Voice Authentication and Anti-Spoofing Systems 

**Authors**: Kamel Kamel, Keshav Sood, Hridoy Sankar Dutta, Sunil Aryal  

**Link**: [PDF](https://arxiv.org/pdf/2508.16843)  

**Abstract**: Voice authentication has undergone significant changes from traditional systems that relied on handcrafted acoustic features to deep learning models that can extract robust speaker embeddings. This advancement has expanded its applications across finance, smart devices, law enforcement, and beyond. However, as adoption has grown, so have the threats. This survey presents a comprehensive review of the modern threat landscape targeting Voice Authentication Systems (VAS) and Anti-Spoofing Countermeasures (CMs), including data poisoning, adversarial, deepfake, and adversarial spoofing attacks. We chronologically trace the development of voice authentication and examine how vulnerabilities have evolved in tandem with technological advancements. For each category of attack, we summarize methodologies, highlight commonly used datasets, compare performance and limitations, and organize existing literature using widely accepted taxonomies. By highlighting emerging risks and open challenges, this survey aims to support the development of more secure and resilient voice authentication systems. 

---
# Physics-Inspired Spatial Temporal Graph Neural Networks for Predicting Industrial Chain Resilience 

**Authors**: Bicheng Wang, Junping Wang, Yibo Xue  

**Link**: [PDF](https://arxiv.org/pdf/2508.16836)  

**Abstract**: Industrial chain plays an increasingly important role in the sustainable development of national economy. However, as a typical complex network, data-driven deep learning is still in its infancy in describing and analyzing the resilience of complex networks, and its core is the lack of a theoretical framework to describe the system dynamics. In this paper, we propose a physically informative neural symbolic approach to describe the evolutionary dynamics of complex networks for resilient prediction. The core idea is to learn the dynamics of the activity state of physical entities and integrate it into the multi-layer spatiotemporal co-evolution network, and use the physical information method to realize the joint learning of physical symbol dynamics and spatiotemporal co-evolution topology, so as to predict the industrial chain resilience. The experimental results show that the model can obtain better results and predict the elasticity of the industry chain more accurately and effectively, which has certain practical significance for the development of the industry. 

---
# Out of Distribution Detection for Efficient Continual Learning in Quality Prediction for Arc Welding 

**Authors**: Yannik Hahn, Jan Voets, Antonin Koenigsfeld, Hasan Tercan, Tobias Meisen  

**Link**: [PDF](https://arxiv.org/pdf/2508.16832)  

**Abstract**: Modern manufacturing relies heavily on fusion welding processes, including gas metal arc welding (GMAW). Despite significant advances in machine learning-based quality prediction, current models exhibit critical limitations when confronted with the inherent distribution shifts that occur in dynamic manufacturing environments. In this work, we extend the VQ-VAE Transformer architecture - previously demonstrating state-of-the-art performance in weld quality prediction - by leveraging its autoregressive loss as a reliable out-of-distribution (OOD) detection mechanism. Our approach exhibits superior performance compared to conventional reconstruction methods, embedding error-based techniques, and other established baselines. By integrating OOD detection with continual learning strategies, we optimize model adaptation, triggering updates only when necessary and thereby minimizing costly labeling requirements. We introduce a novel quantitative metric that simultaneously evaluates OOD detection capability while interpreting in-distribution performance. Experimental validation in real-world welding scenarios demonstrates that our framework effectively maintains robust quality prediction capabilities across significant distribution shifts, addressing critical challenges in dynamic manufacturing environments where process parameters frequently change. This research makes a substantial contribution to applied artificial intelligence by providing an explainable and at the same time adaptive solution for quality assurance in dynamic manufacturing processes - a crucial step towards robust, practical AI systems in the industrial environment. 

---
# Understanding and Tackling Over-Dilution in Graph Neural Networks 

**Authors**: Junhyun Lee, Veronika Thost, Bumsoo Kim, Jaewoo Kang, Tengfei Ma  

**Link**: [PDF](https://arxiv.org/pdf/2508.16829)  

**Abstract**: Message Passing Neural Networks (MPNNs) hold a key position in machine learning on graphs, but they struggle with unintended behaviors, such as over-smoothing and over-squashing, due to irregular data structures. The observation and formulation of these limitations have become foundational in constructing more informative graph representations. In this paper, we delve into the limitations of MPNNs, focusing on aspects that have previously been overlooked. Our observations reveal that even within a single layer, the information specific to an individual node can become significantly diluted. To delve into this phenomenon in depth, we present the concept of Over-dilution and formulate it with two dilution factors: intra-node dilution for attribute-level and inter-node dilution for node-level representations. We also introduce a transformer-based solution that alleviates over-dilution and complements existing node embedding methods like MPNNs. Our findings provide new insights and contribute to the development of informative representations. The implementation and supplementary materials are publicly available at this https URL. 

---
# Exploring the Impact of Generative Artificial Intelligence on Software Development in the IT Sector: Preliminary Findings on Productivity, Efficiency and Job Security 

**Authors**: Anton Ludwig Bonin, Pawel Robert Smolinski, Jacek Winiarski  

**Link**: [PDF](https://arxiv.org/pdf/2508.16811)  

**Abstract**: This study investigates the impact of Generative AI on software development within the IT sector through a mixed-method approach, utilizing a survey developed based on expert interviews. The preliminary results of an ongoing survey offer early insights into how Generative AI reshapes personal productivity, organizational efficiency, adoption, business strategy and job insecurity. The findings reveal that 97% of IT workers use Generative AI tools, mainly ChatGPT. Participants report significant personal productivity gain and perceive organizational efficiency improvements that correlate positively with Generative AI adoption by their organizations (r = .470, p < .05). However, increased organizational adoption of AI strongly correlates with heightened employee job security concerns (r = .549, p < .001). Key adoption challenges include inaccurate outputs (64.2%), regulatory compliance issues (58.2%) and ethical concerns (52.2%). This research offers early empirical insights into Generative AI's economic and organizational implications. 

---
# Autonomous UAV Flight Navigation in Confined Spaces: A Reinforcement Learning Approach 

**Authors**: Marco S. Tayar, Lucas K. de Oliveira, Juliano D. Negri, Thiago H. Segreto, Ricardo V. Godoy, Marcelo Becker  

**Link**: [PDF](https://arxiv.org/pdf/2508.16807)  

**Abstract**: Inspecting confined industrial infrastructure, such as ventilation shafts, is a hazardous and inefficient task for humans. Unmanned Aerial Vehicles (UAVs) offer a promising alternative, but GPS-denied environments require robust control policies to prevent collisions. Deep Reinforcement Learning (DRL) has emerged as a powerful framework for developing such policies, and this paper provides a comparative study of two leading DRL algorithms for this task: the on-policy Proximal Policy Optimization (PPO) and the off-policy Soft Actor-Critic (SAC). The training was conducted with procedurally generated duct environments in Genesis simulation environment. A reward function was designed to guide a drone through a series of waypoints while applying a significant penalty for collisions. PPO learned a stable policy that completed all evaluation episodes without collision, producing smooth trajectories. By contrast, SAC consistently converged to a suboptimal behavior that traversed only the initial segments before failure. These results suggest that, in hazard-dense navigation, the training stability of on-policy methods can outweigh the nominal sample efficiency of off-policy algorithms. More broadly, the study provides evidence that procedurally generated, high-fidelity simulations are effective testbeds for developing and benchmarking robust navigation policies. 

---
# Interpreting the Effects of Quantization on LLMs 

**Authors**: Manpreet Singh, Hassan Sajjad  

**Link**: [PDF](https://arxiv.org/pdf/2508.16785)  

**Abstract**: Quantization offers a practical solution to deploy LLMs in resource-constraint environments. However, its impact on internal representations remains understudied, raising questions about the reliability of quantized models. In this study, we employ a range of interpretability techniques to investigate how quantization affects model and neuron behavior. We analyze multiple LLMs under 4-bit and 8-bit quantization. Our findings reveal that the impact of quantization on model calibration is generally minor. Analysis of neuron activations indicates that the number of dead neurons, i.e., those with activation values close to 0 across the dataset, remains consistent regardless of quantization. In terms of neuron contribution to predictions, we observe that smaller full precision models exhibit fewer salient neurons, whereas larger models tend to have more, with the exception of Llama-2-7B. The effect of quantization on neuron redundancy varies across models. Overall, our findings suggest that effect of quantization may vary by model and tasks, however, we did not observe any drastic change which may discourage the use of quantization as a reliable model compression technique. 

---
# Improving Performance, Robustness, and Fairness of Radiographic AI Models with Finely-Controllable Synthetic Data 

**Authors**: Stefania L. Moroianu, Christian Bluethgen, Pierre Chambon, Mehdi Cherti, Jean-Benoit Delbrouck, Magdalini Paschali, Brandon Price, Judy Gichoya, Jenia Jitsev, Curtis P. Langlotz, Akshay S. Chaudhari  

**Link**: [PDF](https://arxiv.org/pdf/2508.16783)  

**Abstract**: Achieving robust performance and fairness across diverse patient populations remains a challenge in developing clinically deployable deep learning models for diagnostic imaging. Synthetic data generation has emerged as a promising strategy to address limitations in dataset scale and diversity. We introduce RoentGen-v2, a text-to-image diffusion model for chest radiographs that enables fine-grained control over both radiographic findings and patient demographic attributes, including sex, age, and race/ethnicity. RoentGen-v2 is the first model to generate clinically plausible images with demographic conditioning, facilitating the creation of a large, demographically balanced synthetic dataset comprising over 565,000 images. We use this large synthetic dataset to evaluate optimal training pipelines for downstream disease classification models. In contrast to prior work that combines real and synthetic data naively, we propose an improved training strategy that leverages synthetic data for supervised pretraining, followed by fine-tuning on real data. Through extensive evaluation on over 137,000 chest radiographs from five institutions, we demonstrate that synthetic pretraining consistently improves model performance, generalization to out-of-distribution settings, and fairness across demographic subgroups. Across datasets, synthetic pretraining led to a 6.5% accuracy increase in the performance of downstream classification models, compared to a modest 2.7% increase when naively combining real and synthetic data. We observe this performance improvement simultaneously with the reduction of the underdiagnosis fairness gap by 19.3%. These results highlight the potential of synthetic imaging to advance equitable and generalizable medical deep learning under real-world data constraints. We open source our code, trained models, and synthetic dataset at this https URL . 

---
# EyeMulator: Improving Code Language Models by Mimicking Human Visual Attention 

**Authors**: Yifan Zhang, Chen Huang, Yueke Zhang, Jiahao Zhang, Toby Jia-Jun Li, Collin McMillan, Kevin Leach, Yu Huang  

**Link**: [PDF](https://arxiv.org/pdf/2508.16771)  

**Abstract**: Code language models (so-called CodeLLMs) are now commonplace in software development. As a general rule, CodeLLMs are trained by dividing training examples into input tokens and then learn importance of those tokens in a process called machine attention. Machine attention is based solely on input token salience to output token examples during training. Human software developers are different, as humans intuitively know that some tokens are more salient than others. While intuition itself is ineffable and a subject of philosophy, clues about salience are present in human visual attention, since people tend to look at more salient words more often. In this paper, we present EyeMulator, a technique for training CodeLLMs to mimic human visual attention while training for various software development tasks. We add special weights for each token in each input example to the loss function used during LLM fine-tuning. We draw these weights from observations of human visual attention derived from a previously-collected publicly-available dataset of eye-tracking experiments in software engineering tasks. These new weights ultimately induce changes in the attention of the subject LLM during training, resulting in a model that does not need eye-tracking data during inference. Our evaluation shows that EyeMulator outperforms strong LLM baselines on several tasks such as code translation, completion and summarization. We further show an ablation study that demonstrates the improvement is due to subject models learning to mimic human attention. 

---
# Guarding Your Conversations: Privacy Gatekeepers for Secure Interactions with Cloud-Based AI Models 

**Authors**: GodsGift Uzor, Hasan Al-Qudah, Ynes Ineza, Abdul Serwadda  

**Link**: [PDF](https://arxiv.org/pdf/2508.16765)  

**Abstract**: The interactive nature of Large Language Models (LLMs), which closely track user data and context, has prompted users to share personal and private information in unprecedented ways. Even when users opt out of allowing their data to be used for training, these privacy settings offer limited protection when LLM providers operate in jurisdictions with weak privacy laws, invasive government surveillance, or poor data security practices. In such cases, the risk of sensitive information, including Personally Identifiable Information (PII), being mishandled or exposed remains high. To address this, we propose the concept of an "LLM gatekeeper", a lightweight, locally run model that filters out sensitive information from user queries before they are sent to the potentially untrustworthy, though highly capable, cloud-based LLM. Through experiments with human subjects, we demonstrate that this dual-model approach introduces minimal overhead while significantly enhancing user privacy, without compromising the quality of LLM responses. 

---
# FAIRWELL: Fair Multimodal Self-Supervised Learning for Wellbeing Prediction 

**Authors**: Jiaee Cheong, Abtin Mogharabin, Paul Liang, Hatice Gunes, Sinan Kalkan  

**Link**: [PDF](https://arxiv.org/pdf/2508.16748)  

**Abstract**: Early efforts on leveraging self-supervised learning (SSL) to improve machine learning (ML) fairness has proven promising. However, such an approach has yet to be explored within a multimodal context. Prior work has shown that, within a multimodal setting, different modalities contain modality-unique information that can complement information of other modalities. Leveraging on this, we propose a novel subject-level loss function to learn fairer representations via the following three mechanisms, adapting the variance-invariance-covariance regularization (VICReg) method: (i) the variance term, which reduces reliance on the protected attribute as a trivial solution; (ii) the invariance term, which ensures consistent predictions for similar individuals; and (iii) the covariance term, which minimizes correlational dependence on the protected attribute. Consequently, our loss function, coined as FAIRWELL, aims to obtain subject-independent representations, enforcing fairness in multimodal prediction tasks. We evaluate our method on three challenging real-world heterogeneous healthcare datasets (i.e. D-Vlog, MIMIC and MODMA) which contain different modalities of varying length and different prediction tasks. Our findings indicate that our framework improves overall fairness performance with minimal reduction in classification performance and significantly improves on the performance-fairness Pareto frontier. 

---
# Beyond Memorization: Extending Reasoning Depth with Recurrence, Memory and Test-Time Compute Scaling 

**Authors**: Ivan Rodkin, Daniil Orel, Konstantin Smirnov, Arman Bolatov, Bilal Elbouardi, Besher Hassan, Yuri Kuratov, Aydar Bulatov, Preslav Nakov, Timothy Baldwin, Artem Shelmanov, Mikhail Burtsev  

**Link**: [PDF](https://arxiv.org/pdf/2508.16745)  

**Abstract**: Reasoning is a core capability of large language models, yet understanding how they learn and perform multi-step reasoning remains an open problem. In this study, we explore how different architectures and training methods affect model multi-step reasoning capabilities within a cellular automata framework. By training on state sequences generated with random Boolean functions for random initial conditions to exclude memorization, we demonstrate that most neural architectures learn to abstract the underlying rules. While models achieve high accuracy in next-state prediction, their performance declines sharply if multi-step reasoning is required. We confirm that increasing model depth plays a crucial role for sequential computations. We demonstrate that an extension of the effective model depth with recurrence, memory, and test-time compute scaling substantially enhances reasoning capabilities. 

---
# CellEcoNet: Decoding the Cellular Language of Pathology with Deep Learning for Invasive Lung Adenocarcinoma Recurrence Prediction 

**Authors**: Abdul Rehman Akbar, Usama Sajjad, Ziyu Su, Wencheng Li, Fei Xing, Jimmy Ruiz, Wei Chen, Muhammad Khalid Khan Niazi  

**Link**: [PDF](https://arxiv.org/pdf/2508.16742)  

**Abstract**: Despite surgical resection, ~70% of invasive lung adenocarcinoma (ILA) patients recur within five years, and current tools fail to identify those needing adjuvant therapy. To address this unmet clinical need, we introduce CellEcoNet, a novel spatially aware deep learning framework that models whole slide images (WSIs) through natural language analogy, defining a "language of pathology," where cells act as words, cellular neighborhoods become phrases, and tissue architecture forms sentences. CellEcoNet learns these context-dependent meanings automatically, capturing how subtle variations and spatial interactions derive recurrence risk. On a dataset of 456 H&E-stained WSIs, CellEcoNet achieved superior predictive performance (AUC:77.8% HR:9.54), outperforming IASLC grading system (AUC:71.4% HR:2.36), AJCC Stage (AUC:64.0% HR:1.17) and state-of-the-art computational methods (AUCs:62.2-67.4%). CellEcoNet demonstrated fairness and consistent performance across diverse demographic and clinical subgroups. Beyond prognosis, CellEcoNet marks a paradigm shift by decoding the tumor microenvironment's cellular "language" to reveal how subtle cell variations encode recurrence risk. 

---
# WST: Weak-to-Strong Knowledge Transfer via Reinforcement Learning 

**Authors**: Haosen Ge, Shuo Li, Lianghuan Huang  

**Link**: [PDF](https://arxiv.org/pdf/2508.16741)  

**Abstract**: Effective prompt engineering remains a challenging task for many applications. We introduce Weak-to-Strong Transfer (WST), an automatic prompt engineering framework where a small "Teacher" model generates instructions that enhance the performance of a much larger "Student" model. Unlike prior work, WST requires only a weak teacher, making it efficient and broadly applicable in settings where large models are closed-source or difficult to fine-tune. Using reinforcement learning, the Teacher Model's instructions are iteratively improved based on the Student Model's outcomes, yielding substantial gains across reasoning (MATH-500, GSM8K) and alignment (HH-RLHF) benchmarks - 98% on MATH-500 and 134% on HH-RLHF - and surpassing baselines such as GPT-4o-mini and Llama-70B. These results demonstrate that small models can reliably scaffold larger ones, unlocking latent capabilities while avoiding misleading prompts that stronger teachers may introduce, establishing WST as a scalable solution for efficient and safe LLM prompt refinement. 

---
# AI Product Value Assessment Model: An Interdisciplinary Integration Based on Information Theory, Economics, and Psychology 

**Authors**: Yu yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.16714)  

**Abstract**: In recent years, breakthroughs in artificial intelligence (AI) technology have triggered global industrial transformations, with applications permeating various fields such as finance, healthcare, education, and manufacturing. However, this rapid iteration is accompanied by irrational development, where enterprises blindly invest due to technology hype, often overlooking systematic value assessments. This paper develops a multi-dimensional evaluation model that integrates information theory's entropy reduction principle, economics' bounded rationality framework, and psychology's irrational decision theories to quantify AI product value. Key factors include positive dimensions (e.g., uncertainty elimination, efficiency gains, cost savings, decision quality improvement) and negative risks (e.g., error probability, impact, and correction costs). A non-linear formula captures factor couplings, and validation through 10 commercial cases demonstrates the model's effectiveness in distinguishing successful and failed products, supporting hypotheses on synergistic positive effects, non-linear negative impacts, and interactive regulations. Results reveal value generation logic, offering enterprises tools to avoid blind investments and promote rational AI industry development. Future directions include adaptive weights, dynamic mechanisms, and extensions to emerging AI technologies like generative models. 

---
# CelloAI: Leveraging Large Language Models for HPC Software Development in High Energy Physics 

**Authors**: Mohammad Atif, Kriti Chopra, Ozgur Kilic, Tianle Wang, Zhihua Dong, Charles Leggett, Meifeng Lin, Paolo Calafiura, Salman Habib  

**Link**: [PDF](https://arxiv.org/pdf/2508.16713)  

**Abstract**: Next-generation High Energy Physics (HEP) experiments will generate unprecedented data volumes, necessitating High Performance Computing (HPC) integration alongside traditional high-throughput computing. However, HPC adoption in HEP is hindered by the challenge of porting legacy software to heterogeneous architectures and the sparse documentation of these complex scientific codebases. We present CelloAI, a locally hosted coding assistant that leverages Large Language Models (LLMs) with retrieval-augmented generation (RAG) to support HEP code documentation and generation. This local deployment ensures data privacy, eliminates recurring costs and provides access to large context windows without external dependencies. CelloAI addresses two primary use cases, code documentation and code generation, through specialized components. For code documentation, the assistant provides: (a) Doxygen style comment generation for all functions and classes by retrieving relevant information from RAG sources (papers, posters, presentations), (b) file-level summary generation, and (c) an interactive chatbot for code comprehension queries. For code generation, CelloAI employs syntax-aware chunking strategies that preserve syntactic boundaries during embedding, improving retrieval accuracy in large codebases. The system integrates callgraph knowledge to maintain dependency awareness during code modifications and provides AI-generated suggestions for performance optimization and accurate refactoring. We evaluate CelloAI using real-world HEP applications from ATLAS, CMS, and DUNE experiments, comparing different embedding models for code retrieval effectiveness. Our results demonstrate the AI assistant's capability to enhance code understanding and support reliable code generation while maintaining the transparency and safety requirements essential for scientific computing environments. 

---
# Systematic Characterization of LLM Quantization: A Performance, Energy, and Quality Perspective 

**Authors**: Tianyao Shi, Yi Ding  

**Link**: [PDF](https://arxiv.org/pdf/2508.16712)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable capabilities across diverse domains, but their heavy resource demands make quantization-reducing precision to lower-bit formats-critical for efficient serving. While many quantization methods exist, a systematic understanding of their performance, energy, and quality tradeoffs in realistic serving conditions remains a gap. In this work, we first develop a fully automated online characterization framework qMeter, and then conduct an in-depth characterization of 11 post-training LLM quantization methods across 4 model sizes (7B-70B) and two GPU architectures (A100, H100). We evaluate quantization at the application, workload, parallelism, and hardware levels under online serving conditions. Our study reveals highly task- and method-dependent tradeoffs, strong sensitivity to workload characteristics, and complex interactions with parallelism and GPU architecture. We further present three optimization case studies illustrating deployment challenges in capacity planning, energy-efficient scheduling, and multi-objective tuning. To the best of our knowledge, this is one of the first comprehensive application-, system-, and hardware-level characterization of LLM quantization from a joint performance, energy, and quality perspective. 

---
# RoboBuddy in the Classroom: Exploring LLM-Powered Social Robots for Storytelling in Learning and Integration Activities 

**Authors**: Daniel Tozadore, Nur Ertug, Yasmine Chaker, Mortadha Abderrahim  

**Link**: [PDF](https://arxiv.org/pdf/2508.16706)  

**Abstract**: Creating and improvising scenarios for content approaching is an enriching technique in education. However, it comes with a significant increase in the time spent on its planning, which intensifies when using complex technologies, such as social robots. Furthermore, addressing multicultural integration is commonly embedded in regular activities due to the already tight curriculum. Addressing these issues with a single solution, we implemented an intuitive interface that allows teachers to create scenario-based activities from their regular curriculum using LLMs and social robots. We co-designed different frameworks of activities with 4 teachers and deployed it in a study with 27 students for 1 week. Beyond validating the system's efficacy, our findings highlight the positive impact of integration policies perceived by the children and demonstrate the importance of scenario-based activities in students' enjoyment, observed to be significantly higher when applying storytelling. Additionally, several implications of using LLMs and social robots in long-term classroom activities are discussed. 

---
# Assessing Consciousness-Related Behaviors in Large Language Models Using the Maze Test 

**Authors**: Rui A. Pimenta, Tim Schlippe, Kristina Schaaff  

**Link**: [PDF](https://arxiv.org/pdf/2508.16705)  

**Abstract**: We investigate consciousness-like behaviors in Large Language Models (LLMs) using the Maze Test, challenging models to navigate mazes from a first-person perspective. This test simultaneously probes spatial awareness, perspective-taking, goal-directed behavior, and temporal sequencing-key consciousness-associated characteristics. After synthesizing consciousness theories into 13 essential characteristics, we evaluated 12 leading LLMs across zero-shot, one-shot, and few-shot learning scenarios. Results showed reasoning-capable LLMs consistently outperforming standard versions, with Gemini 2.0 Pro achieving 52.9% Complete Path Accuracy and DeepSeek-R1 reaching 80.5% Partial Path Accuracy. The gap between these metrics indicates LLMs struggle to maintain coherent self-models throughout solutions -- a fundamental consciousness aspect. While LLMs show progress in consciousness-related behaviors through reasoning mechanisms, they lack the integrated, persistent self-awareness characteristic of consciousness. 

---
# Dynamic Sparse Attention on Mobile SoCs 

**Authors**: Wangsong Yin, Daliang Xu, Mengwei Xu, Gang Huang, Xuanzhe Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.16703)  

**Abstract**: On-device running Large Language Models (LLMs) is nowadays a critical enabler towards preserving user privacy. We observe that the attention operator falls back from the special-purpose NPU to the general-purpose CPU/GPU because of quantization sensitivity in state-of-the-art frameworks. This fallback results in a degraded user experience and increased complexity in system scheduling. To this end, this paper presents shadowAttn, a system-algorithm codesigned sparse attention module with minimal reliance on CPU/GPU by only sparsely calculating the attention on a tiny portion of tokens. The key idea is to hide the overhead of estimating the important tokens with a NPU-based pilot compute. Further, shadowAttn proposes insightful techniques such as NPU compute graph bucketing, head-wise NPU-CPU/GPU pipeline and per-head fine-grained sparsity ratio to achieve high accuracy and efficiency. shadowAttn delivers the best performance with highly limited CPU/GPU resource; it requires much less CPU/GPU resource to deliver on-par performance of SoTA frameworks. 

---
# Generative Artificial Intelligence and Agents in Research and Teaching 

**Authors**: Jussi S. Jauhiainen, Aurora Toppari  

**Link**: [PDF](https://arxiv.org/pdf/2508.16701)  

**Abstract**: This study provides a comprehensive analysis of the development, functioning, and application of generative artificial intelligence (GenAI) and large language models (LLMs), with an emphasis on their implications for research and education. It traces the conceptual evolution from artificial intelligence (AI) through machine learning (ML) and deep learning (DL) to transformer architectures, which constitute the foundation of contemporary generative systems. Technical aspects, including prompting strategies, word embeddings, and probabilistic sampling methods (temperature, top-k, and top-p), are examined alongside the emergence of autonomous agents. These elements are considered in relation to both the opportunities they create and the limitations and risks they entail.
The work critically evaluates the integration of GenAI across the research process, from ideation and literature review to research design, data collection, analysis, interpretation, and dissemination. While particular attention is given to geographical research, the discussion extends to wider academic contexts. A parallel strand addresses the pedagogical applications of GenAI, encompassing course and lesson design, teaching delivery, assessment, and feedback, with geography education serving as a case example.
Central to the analysis are the ethical, social, and environmental challenges posed by GenAI. Issues of bias, intellectual property, governance, and accountability are assessed, alongside the ecological footprint of LLMs and emerging technological strategies for mitigation. The concluding section considers near- and long-term futures of GenAI, including scenarios of sustained adoption, regulation, and potential decline. By situating GenAI within both scholarly practice and educational contexts, the study contributes to critical debates on its transformative potential and societal responsibilities. 

---
# GPT-OSS-20B: A Comprehensive Deployment-Centric Analysis of OpenAI's Open-Weight Mixture of Experts Model 

**Authors**: Deepak Kumar, Divakar Yadav, Yash Patel  

**Link**: [PDF](https://arxiv.org/pdf/2508.16700)  

**Abstract**: We present a single-GPU (H100, bf16) evaluation of GPT-OSS-20B (Mixture-of-Experts; 20.9B total, approx. 3.61B active) against dense baselines Qwen3-32B and Yi-34B across multiple dimensions. We measure true time-to-first-token (TTFT), full-decode throughput (TPOT), end-to-end latency percentiles, peak VRAM with past key values (PKV) held, and energy via a consistent nvidia-smi-based sampler. At a 2048-token context with 64-token decode, GPT-OSS-20B delivers higher decode throughput and tokens per Joule than dense baselines Qwen3-32B and Yi-34B, while substantially reducing peak VRAM and energy per 1000 generated tokens; its TTFT is higher due to MoE routing overhead. With only 17.3% of parameters active (3.61B of 20.9B), GPT-OSS-20B provides about 31.8% higher decode throughput and 25.8% lower energy per 1000 generated tokens than Qwen3-32B at 2048/64, while using 31.7% less peak VRAM. Normalized by active parameters, GPT-OSS-20B shows markedly stronger per-active-parameter efficiency (APE), underscoring MoE's deployment advantages. We do not evaluate accuracy; this is a deployment-focused study. We release code and consolidated results to enable replication and extension. 

---
# QueryBandits for Hallucination Mitigation: Exploiting Semantic Features for No-Regret Rewriting 

**Authors**: Nicole Cho, William Watson, Alec Koppel, Sumitra Ganesh, Manuela Veloso  

**Link**: [PDF](https://arxiv.org/pdf/2508.16697)  

**Abstract**: Advanced reasoning capabilities in Large Language Models (LLMs) have caused higher hallucination prevalence; yet most mitigation work focuses on after-the-fact filtering rather than shaping the queries that trigger them. We introduce QueryBandits, a bandit framework that designs rewrite strategies to maximize a reward model, that encapsulates hallucination propensity based upon the sensitivities of 17 linguistic features of the input query-and therefore, proactively steer LLMs away from generating hallucinations. Across 13 diverse QA benchmarks and 1,050 lexically perturbed queries per dataset, our top contextual QueryBandit (Thompson Sampling) achieves an 87.5% win rate over a no-rewrite baseline and also outperforms zero-shot static prompting ("paraphrase" or "expand") by 42.6% and 60.3% respectively. Therefore, we empirically substantiate the effectiveness of QueryBandits in mitigating hallucination via the intervention that takes the form of a query rewrite. Interestingly, certain static prompting strategies, which constitute a considerable number of current query rewriting literature, have a higher cumulative regret than the no-rewrite baseline, signifying that static rewrites can worsen hallucination. Moreover, we discover that the converged per-arm regression feature weight vectors substantiate that there is no single rewrite strategy optimal for all queries. In this context, guided rewriting via exploiting semantic features with QueryBandits can induce significant shifts in output behavior through forward-pass mechanisms, bypassing the need for retraining or gradient-based adaptation. 

---
# DecoMind: A Generative AI System for Personalized Interior Design Layouts 

**Authors**: Reema Alshehri, Rawan Alotaibi, Leen Almasri, Rawan Altaweel  

**Link**: [PDF](https://arxiv.org/pdf/2508.16696)  

**Abstract**: This paper introduces a system for generating interior design layouts based on user inputs, such as room type, style, and furniture preferences. CLIP extracts relevant furniture from a dataset, and a layout that contains furniture and a prompt are fed to Stable Diffusion with ControlNet to generate a design that incorporates the selected furniture. The design is then evaluated by classifiers to ensure alignment with the user's inputs, offering an automated solution for realistic interior design. 

---
# Do Cognitively Interpretable Reasoning Traces Improve LLM Performance? 

**Authors**: Siddhant Bhambri, Upasana Biswas, Subbarao Kambhampati  

**Link**: [PDF](https://arxiv.org/pdf/2508.16695)  

**Abstract**: Recent progress in reasoning-oriented Large Language Models (LLMs) has been driven by introducing Chain-of-Thought (CoT) traces, where models generate intermediate reasoning traces before producing an answer. These traces, as in DeepSeek R1, are not only used to guide inference but also serve as supervision signals for distillation into smaller models. A common but often implicit assumption is that CoT traces should be semantically meaningful and interpretable to the end user. While recent research questions the need for semantic nature of these traces, in this paper, we ask: ``\textit{Must CoT reasoning traces be interpretable to enhance LLM task performance?}" We investigate this question in the Open Book Question-Answering domain by supervised fine-tuning LLaMA and Qwen models on four types of reasoning traces: (1) DeepSeek R1 traces, (2) LLM-generated summaries of R1 traces, (3) LLM-generated post-hoc explanations of R1 traces, and (4) algorithmically generated verifiably correct traces. To quantify the trade-off between interpretability and performance, we further conduct a human-subject study with 100 participants rating the interpretability of each trace type. Our results reveal a striking mismatch: while fine-tuning on R1 traces yields the strongest performance, participants judged these traces to be the least interpretable. These findings suggest that it is useful to decouple intermediate tokens from end user interpretability. 

---
# Making AI Inevitable: Historical Perspective and the Problems of Predicting Long-Term Technological Change 

**Authors**: Mark Fisher, John Severini  

**Link**: [PDF](https://arxiv.org/pdf/2508.16692)  

**Abstract**: This study demonstrates the extent to which prominent debates about the future of AI are best understood as subjective, philosophical disagreements over the history and future of technological change rather than as objective, material disagreements over the technologies themselves. It focuses on the deep disagreements over whether artificial general intelligence (AGI) will prove transformative for human society; a question that is analytically prior to that of whether this transformative effect will help or harm humanity. The study begins by distinguishing two fundamental camps in this debate. The first of these can be identified as "transformationalists," who argue that continued AI development will inevitably have a profound effect on society. Opposed to them are "skeptics," a more eclectic group united by their disbelief that AI can or will live up to such high expectations. Each camp admits further "strong" and "weak" variants depending on their tolerance for epistemic risk. These stylized contrasts help to identify a set of fundamental questions that shape the camps' respective interpretations of the future of AI. Three questions in particular are focused on: the possibility of non-biological intelligence, the appropriate time frame of technological predictions, and the assumed trajectory of technological development. In highlighting these specific points of non-technical disagreement, this study demonstrates the wide range of different arguments used to justify either the transformationalist or skeptical position. At the same time, it highlights the strong argumentative burden of the transformationalist position, the way that belief in this position creates competitive pressures to achieve first-mover advantage, and the need to widen the concept of "expertise" in debates surrounding the future development of AI. 

---
# Cybernaut: Towards Reliable Web Automation 

**Authors**: Ankur Tomar, Hengyue Liang, Indranil Bhattacharya, Natalia Larios, Francesco Carbone  

**Link**: [PDF](https://arxiv.org/pdf/2508.16688)  

**Abstract**: The emergence of AI-driven web automation through Large Language Models (LLMs) offers unprecedented opportunities for optimizing digital workflows. However, deploying such systems within industry's real-world environments presents four core challenges: (1) ensuring consistent execution, (2) accurately identifying critical HTML elements, (3) meeting human-like accuracy in order to automate operations at scale and (4) the lack of comprehensive benchmarking data on internal web applications. Existing solutions are primarily tailored for well-designed, consumer-facing websites (e.g., this http URL, this http URL) and fall short in addressing the complexity of poorly-designed internal web interfaces. To address these limitations, we present Cybernaut, a novel framework to ensure high execution consistency in web automation agents designed for robust enterprise use. Our contributions are threefold: (1) a Standard Operating Procedure (SOP) generator that converts user demonstrations into reliable automation instructions for linear browsing tasks, (2) a high-precision HTML DOM element recognition system tailored for the challenge of complex web interfaces, and (3) a quantitative metric to assess execution consistency. The empirical evaluation on our internal benchmark demonstrates that using our framework enables a 23.2% improvement (from 72% to 88.68%) in task execution success rate over the browser_use. Cybernaut identifies consistent execution patterns with 84.7% accuracy, enabling reliable confidence assessment and adaptive guidance during task execution in real-world systems. These results highlight Cybernaut's effectiveness in enterprise-scale web automation and lay a foundation for future advancements in web automation. 

---
# STGAtt: A Spatial-Temporal Unified Graph Attention Network for Traffic Flow Forecasting 

**Authors**: Zhuding Liang, Jianxun Cui, Qingshuang Zeng, Feng Liu, Nenad Filipovic, Tijana Geroski  

**Link**: [PDF](https://arxiv.org/pdf/2508.16685)  

**Abstract**: Accurate and timely traffic flow forecasting is crucial for intelligent transportation systems. This paper presents a novel deep learning model, the Spatial-Temporal Unified Graph Attention Network (STGAtt). By leveraging a unified graph representation and an attention mechanism, STGAtt effectively captures complex spatial-temporal dependencies. Unlike methods relying on separate spatial and temporal dependency modeling modules, STGAtt directly models correlations within a Spatial-Temporal Unified Graph, dynamically weighing connections across both dimensions. To further enhance its capabilities, STGAtt partitions traffic flow observation signal into neighborhood subsets and employs a novel exchanging mechanism, enabling effective capture of both short-range and long-range correlations. Extensive experiments on the PEMS-BAY and SHMetro datasets demonstrate STGAtt's superior performance compared to state-of-the-art baselines across various prediction horizons. Visualization of attention weights confirms STGAtt's ability to adapt to dynamic traffic patterns and capture long-range dependencies, highlighting its potential for real-world traffic flow forecasting applications. 

---
# CALR: Corrective Adaptive Low-Rank Decomposition for Efficient Large Language Model Layer Compression 

**Authors**: Muchammad Daniyal Kautsar, Afra Majida Hariono, Widyawan, Syukron Abu Ishaq Alfarozi, Kuntpong Wararatpanya  

**Link**: [PDF](https://arxiv.org/pdf/2508.16680)  

**Abstract**: Large Language Models (LLMs) present significant deployment challenges due to their immense size and computational requirements. Model compression techniques are essential for making these models practical for resource-constrained environments. A prominent compression strategy is low-rank factorization via Singular Value Decomposition (SVD) to reduce model parameters by approximating weight matrices. However, standard SVD focuses on minimizing matrix reconstruction error, often leading to a substantial loss of the model's functional performance. This performance degradation occurs because existing methods do not adequately correct for the functional information lost during compression. To address this gap, we introduce Corrective Adaptive Low-Rank Decomposition (CALR), a two-component compression approach. CALR combines a primary path of SVD-compressed layers with a parallel, learnable, low-rank corrective module that is explicitly trained to recover the functional residual error. Our experimental evaluation on SmolLM2-135M, Qwen3-0.6B, and Llama-3.2-1B, demonstrates that CALR can reduce parameter counts by 26.93% to 51.77% while retaining 59.45% to 90.42% of the original model's performance, consistently outperforming LaCo, ShortGPT, and LoSparse. CALR's success shows that treating functional information loss as a learnable signal is a highly effective compression paradigm. This approach enables the creation of significantly smaller, more efficient LLMs, advancing their accessibility and practical deployment in real-world applications. 

---
# Recall-Extend Dynamics: Enhancing Small Language Models through Controlled Exploration and Refined Offline Integration 

**Authors**: Zhong Guan, Likang Wu, Hongke Zhao, Jiahui Wang, Le Wu  

**Link**: [PDF](https://arxiv.org/pdf/2508.16677)  

**Abstract**: Many existing studies have achieved significant improvements in the reasoning capabilities of large language models (LLMs) through reinforcement learning with verifiable rewards (RLVR), while the enhancement of reasoning abilities in small language models (SLMs) has not yet been sufficiently explored. Combining distilled data from larger models with RLVR on small models themselves is a natural approach, but it still faces various challenges and issues. Therefore, we propose \textit{\underline{R}}ecall-\textit{\underline{E}}xtend \textit{\underline{D}}ynamics(RED): Enhancing Small Language Models through Controlled Exploration and Refined Offline Integration. In this paper, we explore the perspective of varying exploration spaces, balancing offline distillation with online reinforcement learning. Simultaneously, we specifically design and optimize for the insertion problem within offline data. By monitoring the ratio of entropy changes in the model concerning offline and online data, we regulate the weight of offline-SFT, thereby addressing the issues of insufficient exploration space in small models and the redundancy and complexity during the distillation process. Furthermore, to tackle the distribution discrepancies between offline data and the current policy, we design a sample-accuracy-based policy shift mechanism that dynamically chooses between imitating offline distilled data and learning from its own policy. 

---
# MedRepBench: A Comprehensive Benchmark for Medical Report Interpretation 

**Authors**: Fangxin Shang, Yuan Xia, Dalu Yang, Yahui Wang, Binglin Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.16674)  

**Abstract**: Medical report interpretation plays a crucial role in healthcare, enabling both patient-facing explanations and effective information flow across clinical systems. While recent vision-language models (VLMs) and large language models (LLMs) have demonstrated general document understanding capabilities, there remains a lack of standardized benchmarks to assess structured interpretation quality in medical reports. We introduce MedRepBench, a comprehensive benchmark built from 1,900 de-identified real-world Chinese medical reports spanning diverse departments, patient demographics, and acquisition formats. The benchmark is designed primarily to evaluate end-to-end VLMs for structured medical report understanding. To enable controlled comparisons, we also include a text-only evaluation setting using high-quality OCR outputs combined with LLMs, allowing us to estimate the upper-bound performance when character recognition errors are minimized. Our evaluation framework supports two complementary protocols: (1) an objective evaluation measuring field-level recall of structured clinical items, and (2) an automated subjective evaluation using a powerful LLM as a scoring agent to assess factuality, interpretability, and reasoning quality. Based on the objective metric, we further design a reward function and apply Group Relative Policy Optimization (GRPO) to improve a mid-scale VLM, achieving up to 6% recall gain. We also observe that the OCR+LLM pipeline, despite strong performance, suffers from layout-blindness and latency issues, motivating further progress toward robust, fully vision-based report understanding. 

---
# Invisible Filters: Cultural Bias in Hiring Evaluations Using Large Language Models 

**Authors**: Pooja S. B. Rao, Laxminarayen Nagarajan Venkatesan, Mauro Cherubini, Dinesh Babu Jayagopi  

**Link**: [PDF](https://arxiv.org/pdf/2508.16673)  

**Abstract**: Artificial Intelligence (AI) is increasingly used in hiring, with large language models (LLMs) having the potential to influence or even make hiring decisions. However, this raises pressing concerns about bias, fairness, and trust, particularly across diverse cultural contexts. Despite their growing role, few studies have systematically examined the potential biases in AI-driven hiring evaluation across cultures. In this study, we conduct a systematic analysis of how LLMs assess job interviews across cultural and identity dimensions. Using two datasets of interview transcripts, 100 from UK and 100 from Indian job seekers, we first examine cross-cultural differences in LLM-generated scores for hirability and related traits. Indian transcripts receive consistently lower scores than UK transcripts, even when they were anonymized, with disparities linked to linguistic features such as sentence complexity and lexical diversity. We then perform controlled identity substitutions (varying names by gender, caste, and region) within the Indian dataset to test for name-based bias. These substitutions do not yield statistically significant effects, indicating that names alone, when isolated from other contextual signals, may not influence LLM evaluations. Our findings underscore the importance of evaluating both linguistic and social dimensions in LLM-driven evaluations and highlight the need for culturally sensitive design and accountability in AI-assisted hiring. 

---
# The AI Model Risk Catalog: What Developers and Researchers Miss About Real-World AI Harms 

**Authors**: Pooja S. B. Rao, Sanja Šćepanović, Dinesh Babu Jayagopi, Mauro Cherubini, Daniele Quercia  

**Link**: [PDF](https://arxiv.org/pdf/2508.16672)  

**Abstract**: We analyzed nearly 460,000 AI model cards from Hugging Face to examine how developers report risks. From these, we extracted around 3,000 unique risk mentions and built the \emph{AI Model Risk Catalog}. We compared these with risks identified by researchers in the MIT Risk Repository and with real-world incidents from the AI Incident Database. Developers focused on technical issues like bias and safety, while researchers emphasized broader social impacts. Both groups paid little attention to fraud and manipulation, which are common harms arising from how people interact with AI. Our findings show the need for clearer, structured risk reporting that helps developers think about human-interaction and systemic risks early in the design process. The catalog and paper appendix are available at: this https URL. 

---
# Reflective Paper-to-Code Reproduction Enabled by Fine-Grained Verification 

**Authors**: Mingyang Zhou, Quanming Yao, Lun Du, Lanning Wei, Da Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2508.16671)  

**Abstract**: Reproducing machine learning papers is essential for scientific progress but remains challenging for both humans and automated agents. Existing agent-based methods often struggle to fully and accurately reproduce implementation details such as mathematical formulas and algorithmic logic. Previous studies show that reflection with explicit feedback improves agent performance. However, current paper reproduction methods fail to effectively adopt this strategy. This gap mainly arises from the diverse paper patterns, complex method modules, and varied configurations encountered in research papers. Motivated by how humans use systematic checklists to efficiently debug complex code, we propose \textbf{RePro}, a \textbf{Re}flective Paper-to-Code \textbf{Repro}duction framework that automatically extracts a paper's fingerprint, referring to a comprehensive set of accurate and atomic criteria serving as high-quality supervisory signals. The framework first generates code based on the extracted information, and then leverages the fingerprint within iterative verification and refinement loop. This approach systematically detects discrepancies and produces targeted revisions to align generated code with the paper's implementation details. Extensive experiments on the PaperBench Code-Dev benchmark have been conducted, RePro achieves 13.0\% performance gap over baselines, and it correctly revises complex logical and mathematical criteria in reflecting, on which the effectiveness is obvious. 

---
# COVID19 Prediction Based On CT Scans Of Lungs Using DenseNet Architecture 

**Authors**: Deborup Sanyal  

**Link**: [PDF](https://arxiv.org/pdf/2508.16670)  

**Abstract**: COVID19 took the world by storm since December 2019. A highly infectious communicable disease, COVID19 is caused by the SARSCoV2 virus. By March 2020, the World Health Organization (WHO) declared COVID19 as a global pandemic. A pandemic in the 21st century after almost 100 years was something the world was not prepared for, which resulted in the deaths of around 1.6 million people worldwide. The most common symptoms of COVID19 were associated with the respiratory system and resembled a cold, flu, or pneumonia. After extensive research, doctors and scientists concluded that the main reason for lives being lost due to COVID19 was failure of the respiratory system. Patients were dying gasping for breath. Top healthcare systems of the world were failing badly as there was an acute shortage of hospital beds, oxygen cylinders, and ventilators. Many were dying without receiving any treatment at all. The aim of this project is to help doctors decide the severity of COVID19 by reading the patient's Computed Tomography (CT) scans of the lungs. Computer models are less prone to human error, and Machine Learning or Neural Network models tend to give better accuracy as training improves over time. We have decided to use a Convolutional Neural Network model. Given that a patient tests positive, our model will analyze the severity of COVID19 infection within one month of the positive test result. The severity of the infection may be promising or unfavorable (if it leads to intubation or death), based entirely on the CT scans in the dataset. 

---
# Situational Awareness as the Imperative Capability for Disaster Resilience in the Era of Complex Hazards and Artificial Intelligence 

**Authors**: Hongrak Pak, Ali Mostafavi  

**Link**: [PDF](https://arxiv.org/pdf/2508.16669)  

**Abstract**: Disasters frequently exceed established hazard models, revealing blind spots where unforeseen impacts and vulnerabilities hamper effective response. This perspective paper contends that situational awareness (SA)-the ability to perceive, interpret, and project dynamic crisis conditions-is an often overlooked yet vital capability for disaster resilience. While risk mitigation measures can reduce known threats, not all hazards can be neutralized; truly adaptive resilience hinges on whether organizations rapidly detect emerging failures, reconcile diverse data sources, and direct interventions where they matter most. We present a technology-process-people roadmap, demonstrating how real-time hazard nowcasting, interoperable workflows, and empowered teams collectively transform raw data into actionable insight. A system-of-systems approach enables federated data ownership and modular analytics, so multiple agencies can share timely updates without sacrificing their distinct operational models. Equally crucial, structured sense-making routines and cognitive load safeguards help humans remain effective decision-makers amid data abundance. By framing SA as a socio-technical linchpin rather than a peripheral add-on, this paper spotlights the urgency of elevating SA to a core disaster resilience objective. We conclude with recommendations for further research-developing SA metrics, designing trustworthy human-AI collaboration, and strengthening inclusive data governance-to ensure that communities are equipped to cope with both expected and unexpected crises. 

---
# Trust but Verify! A Survey on Verification Design for Test-time Scaling 

**Authors**: V Venktesh, Mandeep rathee, Avishek Anand  

**Link**: [PDF](https://arxiv.org/pdf/2508.16665)  

**Abstract**: Test-time scaling (TTS) has emerged as a new frontier for scaling the performance of Large Language Models. In test-time scaling, by using more computational resources during inference, LLMs can improve their reasoning process and task performance. Several approaches have emerged for TTS such as distilling reasoning traces from another model or exploring the vast decoding search space by employing a verifier. The verifiers serve as reward models that help score the candidate outputs from the decoding process to diligently explore the vast solution space and select the best outcome. This paradigm commonly termed has emerged as a superior approach owing to parameter free scaling at inference time and high performance gains. The verifiers could be prompt-based, fine-tuned as a discriminative or generative model to verify process paths, outcomes or both. Despite their widespread adoption, there is no detailed collection, clear categorization and discussion of diverse verification approaches and their training mechanisms. In this survey, we cover the diverse approaches in the literature and present a unified view of verifier training, types and their utility in test-time scaling. Our repository can be found at this https URL. 

---
# The Loupe: A Plug-and-Play Attention Module for Amplifying Discriminative Features in Vision Transformers 

**Authors**: Naren Sengodan  

**Link**: [PDF](https://arxiv.org/pdf/2508.16663)  

**Abstract**: Fine-Grained Visual Classification (FGVC) is a critical and challenging area within computer vision, demanding the identification of highly subtle, localized visual cues. The importance of FGVC extends to critical applications such as biodiversity monitoring and medical diagnostics, where precision is paramount. While large-scale Vision Transformers have achieved state-of-the-art performance, their decision-making processes often lack the interpretability required for trust and verification in such domains. In this paper, we introduce The Loupe, a novel, lightweight, and plug-and-play attention module designed to be inserted into pre-trained backbones like the Swin Transformer. The Loupe is trained end-to-end with a composite loss function that implicitly guides the model to focus on the most discriminative object parts without requiring explicit part-level annotations. Our unique contribution lies in demonstrating that a simple, intrinsic attention mechanism can act as a powerful regularizer, significantly boosting performance while simultaneously providing clear visual explanations. Our experimental evaluation on the challenging CUB-200-2011 dataset shows that The Loupe improves the accuracy of a Swin-Base model from 85.40% to 88.06%, a significant gain of 2.66%. Crucially, our qualitative analysis of the learned attention maps reveals that The Loupe effectively localizes semantically meaningful features, providing a valuable tool for understanding and trusting the model's decision-making process. 

---
# Optimizing Hyper parameters in CNN for Soil Classification using PSO and Whale Optimization Algorithm 

**Authors**: Yasir Nooruldeen Ibrahim, Fawziya Mahmood Ramo, Mahmood Siddeeq Qadir, Muna Jaffer Al-Shamdeen  

**Link**: [PDF](https://arxiv.org/pdf/2508.16660)  

**Abstract**: Classifying soil images contributes to better land management, increased agricultural output, and practical solutions for environmental issues. The development of various disciplines, particularly agriculture, civil engineering, and natural resource management, is aided by understanding of soil quality since it helps with risk reduction, performance improvement, and sound decision-making . Artificial intelligence has recently been used in a number of different fields. In this study, an intelligent model was constructed using Convolutional Neural Networks to classify soil kinds, and machine learning algorithms were used to enhance the performance of soil classification . To achieve better implementation and performance of the Convolutional Neural Networks algorithm and obtain valuable results for the process of classifying soil type images, swarm algorithms were employed to obtain the best performance by choosing Hyper parameters for the Convolutional Neural Networks network using the Whale optimization algorithm and the Particle swarm optimization algorithm, and comparing the results of using the two algorithms in the process of multiple classification of soil types. The Accuracy and F1 measures were adopted to test the system, and the results of the proposed work were efficient result 

---
# Enabling Multi-Agent Systems as Learning Designers: Applying Learning Sciences to AI Instructional Design 

**Authors**: Jiayi Wang, Ruiwei Xiao, Xinying Hou, John Stamper  

**Link**: [PDF](https://arxiv.org/pdf/2508.16659)  

**Abstract**: K-12 educators are increasingly using Large Language Models (LLMs) to create instructional materials. These systems excel at producing fluent, coherent content, but often lack support for high-quality teaching. The reason is twofold: first, commercial LLMs, such as ChatGPT and Gemini which are among the most widely accessible to teachers, do not come preloaded with the depth of pedagogical theory needed to design truly effective activities; second, although sophisticated prompt engineering can bridge this gap, most teachers lack the time or expertise and find it difficult to encode such pedagogical nuance into their requests. This study shifts pedagogical expertise from the user's prompt to the LLM's internal architecture. We embed the well-established Knowledge-Learning-Instruction (KLI) framework into a Multi-Agent System (MAS) to act as a sophisticated instructional designer. We tested three systems for generating secondary Math and Science learning activities: a Single-Agent baseline simulating typical teacher prompts; a role-based MAS where agents work sequentially; and a collaborative MAS-CMD where agents co-construct activities through conquer and merge discussion. The generated materials were evaluated by 20 practicing teachers and a complementary LLM-as-a-judge system using the Quality Matters (QM) K-12 standards. While the rubric scores showed only small, often statistically insignificant differences between the systems, the qualitative feedback from educators painted a clear and compelling picture. Teachers strongly preferred the activities from the collaborative MAS-CMD, describing them as significantly more creative, contextually relevant, and classroom-ready. Our findings show that embedding pedagogical principles into LLM systems offers a scalable path for creating high-quality educational content. 

---
# HiCL: Hippocampal-Inspired Continual Learning 

**Authors**: Kushal Kapoor, Wyatt Mackey, Yiannis Aloimonos, Xiaomin Lin  

**Link**: [PDF](https://arxiv.org/pdf/2508.16651)  

**Abstract**: We propose HiCL, a novel hippocampal-inspired dual-memory continual learning architecture designed to mitigate catastrophic forgetting by using elements inspired by the hippocampal circuitry. Our system encodes inputs through a grid-cell-like layer, followed by sparse pattern separation using a dentate gyrus-inspired module with top-k sparsity. Episodic memory traces are maintained in a CA3-like autoassociative memory. Task-specific processing is dynamically managed via a DG-gated mixture-of-experts mechanism, wherein inputs are routed to experts based on cosine similarity between their normalized sparse DG representations and learned task-specific DG prototypes computed through online exponential moving averages. This biologically grounded yet mathematically principled gating strategy enables differentiable, scalable task-routing without relying on a separate gating network, and enhances the model's adaptability and efficiency in learning multiple sequential tasks. Cortical outputs are consolidated using Elastic Weight Consolidation weighted by inter-task similarity. Crucially, we incorporate prioritized replay of stored patterns to reinforce essential past experiences. Evaluations on standard continual learning benchmarks demonstrate the effectiveness of our architecture in reducing task interference, achieving near state-of-the-art results in continual learning tasks at lower computational costs. 

---
# LatentFlow: Cross-Frequency Experimental Flow Reconstruction from Sparse Pressure via Latent Mapping 

**Authors**: Junle Liu, Chang Liu, Yanyu Ke, Qiuxiang Huang, Jiachen Zhao, Wenliang Chen, K.T. Tse, Gang Hu  

**Link**: [PDF](https://arxiv.org/pdf/2508.16648)  

**Abstract**: Acquiring temporally high-frequency and spatially high-resolution turbulent wake flow fields in particle image velocimetry (PIV) experiments remains a significant challenge due to hardware limitations and measurement noise. In contrast, temporal high-frequency measurements of spatially sparse wall pressure are more readily accessible in wind tunnel experiments. In this study, we propose a novel cross-modal temporal upscaling framework, LatentFlow, which reconstructs high-frequency (512 Hz) turbulent wake flow fields by fusing synchronized low-frequency (15 Hz) flow field and pressure data during training, and high-frequency wall pressure signals during inference. The first stage involves training a pressure-conditioned $\beta$-variation autoencoder ($p$C-$\beta$-VAE) to learn a compact latent representation that captures the intrinsic dynamics of the wake flow. A secondary network maps synchronized low-frequency wall pressure signals into the latent space, enabling reconstruction of the wake flow field solely from sparse wall pressure. Once trained, the model utilizes high-frequency, spatially sparse wall pressure inputs to generate corresponding high-frequency flow fields via the $p$C-$\beta$-VAE decoder. By decoupling the spatial encoding of flow dynamics from temporal pressure measurements, LatentFlow provides a scalable and robust solution for reconstructing high-frequency turbulent wake flows in data-constrained experimental settings. 

---
# Equinox: Holistic Fair Scheduling in Serving Large Language Models 

**Authors**: Zhixiang Wei, James Yen, Jingyi Chen, Ziyang Zhang, Zhibai Huang, Chen Chen, Xingzi Yu, Yicheng Gu, Chenggang Wu, Yun Wang, Mingyuan Xia, Jie Wu, Hao Wang, Zhengwei Qi  

**Link**: [PDF](https://arxiv.org/pdf/2508.16646)  

**Abstract**: We address the limitations of current LLM serving with a dual-counter framework separating user and operator perspectives. The User Fairness Counter measures quality of service via weighted tokens and latency; the Resource Fairness Counter measures operational efficiency through throughput and GPU utilization. Since these metrics are only available post-execution, creating a scheduling paradox, we introduce a deterministic Mixture of Prediction Experts (MoPE) framework to predict user-perceived latency, output tokens, throughput, and GPU utilization. These predictions enable calculation of a unified Holistic Fairness score that balances both counters through tunable parameters for proactive fairness-aware scheduling. We implement this in Equinox, an open-source system with other optimizations like adaptive batching, and stall-free scheduling. Evaluations on production traces (ShareGPT, LMSYS) and synthetic workloads demonstrate Equinox achieves up to $1.3\times$ higher throughput, 60\% lower time-to-first-token latency, and 13\% higher fairness versus VTC while maintaining 94\% GPU utilization, proving fairness under bounded discrepancy across heterogeneous platforms. 

---
# From Classical Probabilistic Latent Variable Models to Modern Generative AI: A Unified Perspective 

**Authors**: Tianhua Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.16643)  

**Abstract**: From large language models to multi-modal agents, Generative Artificial Intelligence (AI) now underpins state-of-the-art systems. Despite their varied architectures, many share a common foundation in probabilistic latent variable models (PLVMs), where hidden variables explain observed data for density estimation, latent reasoning, and structured inference. This paper presents a unified perspective by framing both classical and modern generative methods within the PLVM paradigm. We trace the progression from classical flat models such as probabilistic PCA, Gaussian mixture models, latent class analysis, item response theory, and latent Dirichlet allocation, through their sequential extensions including Hidden Markov Models, Gaussian HMMs, and Linear Dynamical Systems, to contemporary deep architectures: Variational Autoencoders as Deep PLVMs, Normalizing Flows as Tractable PLVMs, Diffusion Models as Sequential PLVMs, Autoregressive Models as Explicit Generative Models, and Generative Adversarial Networks as Implicit PLVMs. Viewing these architectures under a common probabilistic taxonomy reveals shared principles, distinct inference strategies, and the representational trade-offs that shape their strengths. We offer a conceptual roadmap that consolidates generative AI's theoretical foundations, clarifies methodological lineages, and guides future innovation by grounding emerging architectures in their probabilistic heritage. 

---
# Cognitive Decision Routing in Large Language Models: When to Think Fast, When to Think Slow 

**Authors**: Y. Du, C. Guo, W. Wang, G. Tang  

**Link**: [PDF](https://arxiv.org/pdf/2508.16636)  

**Abstract**: Large Language Models (LLMs) face a fundamental challenge in deciding when to rely on rapid, intuitive responses versus engaging in slower, more deliberate reasoning. Inspired by Daniel Kahneman's dual-process theory and his insights on human cognitive biases, we propose a novel Cognitive Decision Routing (CDR) framework that dynamically determines the appropriate reasoning strategy based on query characteristics. Our approach addresses the current limitations where models either apply uniform reasoning depth or rely on computationally expensive methods for all queries. We introduce a meta-cognitive layer that analyzes query complexity through multiple dimensions: correlation strength between given information and required conclusions, domain boundary crossings, stakeholder multiplicity, and uncertainty levels. Through extensive experiments on diverse reasoning tasks, we demonstrate that CDR achieves superior performance while reducing computational costs by 34\% compared to uniform deep reasoning approaches. Our framework shows particular strength in professional judgment tasks, achieving 23\% improvement in consistency and 18\% better accuracy on expert-level evaluations. This work bridges cognitive science principles with practical AI system design, offering a principled approach to adaptive reasoning in LLMs. 

---
# Few-shot Class-incremental Fault Diagnosis by Preserving Class-Agnostic Knowledge with Dual-Granularity Representations 

**Authors**: Zhendong Yang, Jie Wang, Liansong Zong, Xiaorong Liu, Quan Qian, Shiqian Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.16634)  

**Abstract**: Few-Shot Class-Incremental Fault Diagnosis (FSC-FD), which aims to continuously learn from new fault classes with only a few samples without forgetting old ones, is critical for real-world industrial systems. However, this challenging task severely amplifies the issues of catastrophic forgetting of old knowledge and overfitting on scarce new data. To address these challenges, this paper proposes a novel framework built upon Dual-Granularity Representations, termed the Dual-Granularity Guidance Network (DGGN). Our DGGN explicitly decouples feature learning into two parallel streams: 1) a fine-grained representation stream, which utilizes a novel Multi-Order Interaction Aggregation module to capture discriminative, class-specific features from the limited new samples. 2) a coarse-grained representation stream, designed to model and preserve general, class-agnostic knowledge shared across all fault types. These two representations are dynamically fused by a multi-semantic cross-attention mechanism, where the stable coarse-grained knowledge guides the learning of fine-grained features, preventing overfitting and alleviating feature conflicts. To further mitigate catastrophic forgetting, we design a Boundary-Aware Exemplar Prioritization strategy. Moreover, a decoupled Balanced Random Forest classifier is employed to counter the decision boundary bias caused by data imbalance. Extensive experiments on the TEP benchmark and a real-world MFF dataset demonstrate that our proposed DGGN achieves superior diagnostic performance and stability compared to state-of-the-art FSC-FD approaches. Our code is publicly available at this https URL 

---
# Adaptive Variance-Penalized Continual Learning with Fisher Regularization 

**Authors**: Krisanu Sarkar  

**Link**: [PDF](https://arxiv.org/pdf/2508.16632)  

**Abstract**: The persistent challenge of catastrophic forgetting in neural networks has motivated extensive research in continual learning . This work presents a novel continual learning framework that integrates Fisher-weighted asymmetric regularization of parameter variances within a variational learning paradigm. Our method dynamically modulates regularization intensity according to parameter uncertainty, achieving enhanced stability and performance. Comprehensive evaluations on standard continual learning benchmarks including SplitMNIST, PermutedMNIST, and SplitFashionMNIST demonstrate substantial improvements over existing approaches such as Variational Continual Learning and Elastic Weight Consolidation . The asymmetric variance penalty mechanism proves particularly effective in maintaining knowledge across sequential tasks while improving model accuracy. Experimental results show our approach not only boosts immediate task performance but also significantly mitigates knowledge degradation over time, effectively addressing the fundamental challenge of catastrophic forgetting in neural networks 

---
# Learn to Memorize: Optimizing LLM-based Agents with Adaptive Memory Framework 

**Authors**: Zeyu Zhang, Quanyu Dai, Rui Li, Xiaohe Bo, Xu Chen, Zhenhua Dong  

**Link**: [PDF](https://arxiv.org/pdf/2508.16629)  

**Abstract**: LLM-based agents have been extensively applied across various domains, where memory stands out as one of their most essential capabilities. Previous memory mechanisms of LLM-based agents are manually predefined by human experts, leading to higher labor costs and suboptimal performance. In addition, these methods overlook the memory cycle effect in interactive scenarios, which is critical to optimizing LLM-based agents for specific environments. To address these challenges, in this paper, we propose to optimize LLM-based agents with an adaptive and data-driven memory framework by modeling memory cycles. Specifically, we design an MoE gate function to facilitate memory retrieval, propose a learnable aggregation process to improve memory utilization, and develop task-specific reflection to adapt memory storage. Our memory framework empowers LLM-based agents to learn how to memorize information effectively in specific environments, with both off-policy and on-policy optimization. In order to evaluate the effectiveness of our proposed methods, we conduct comprehensive experiments across multiple aspects. To benefit the research community in this area, we release our project at this https URL. 

---
# The Impact of Artificial Intelligence on Human Thought 

**Authors**: Rénald Gesnot  

**Link**: [PDF](https://arxiv.org/pdf/2508.16628)  

**Abstract**: This research paper examines, from a multidimensional perspective (cognitive, social, ethical, and philosophical), how AI is transforming human thought. It highlights a cognitive offloading effect: the externalization of mental functions to AI can reduce intellectual engagement and weaken critical thinking. On the social level, algorithmic personalization creates filter bubbles that limit the diversity of opinions and can lead to the homogenization of thought and polarization. This research also describes the mechanisms of algorithmic manipulation (exploitation of cognitive biases, automated disinformation, etc.) that amplify AI's power of influence. Finally, the question of potential artificial consciousness is discussed, along with its ethical implications. The report as a whole underscores the risks that AI poses to human intellectual autonomy and creativity, while proposing avenues (education, transparency, governance) to align AI development with the interests of humanity. 

---
# Data and Context Matter: Towards Generalizing AI-based Software Vulnerability Detection 

**Authors**: Rijha Safdar, Danyail Mateen, Syed Taha Ali, M. Umer Ashfaq, Wajahat Hussain  

**Link**: [PDF](https://arxiv.org/pdf/2508.16625)  

**Abstract**: The performance of AI-based software vulnerability detection systems is often limited by their poor generalization to unknown codebases. In this research, we explore the impact of data quality and model architecture on the generalizability of vulnerability detection systems. By generalization we mean ability of high vulnerability detection performance across different C/C++ software projects not seen during training. Through a series of experiments, we demonstrate that improvements in dataset diversity and quality substantially enhance detection performance. Additionally, we compare multiple encoder-only and decoder-only models, finding that encoder based models outperform in terms of accuracy and generalization. Our model achieves 6.8% improvement in recall on the benchmark BigVul[1] dataset, also outperforming on unseen projects, hence showing enhanced generalizability. These results highlight the role of data quality and model selection in the development of robust vulnerability detection systems. Our findings suggest a direction for future systems having high cross-project effectiveness. 

---
# The GPT-4o Shock Emotional Attachment to AI Models and Its Impact on Regulatory Acceptance: A Cross-Cultural Analysis of the Immediate Transition from GPT-4o to GPT-5 

**Authors**: Hiroki Naito  

**Link**: [PDF](https://arxiv.org/pdf/2508.16624)  

**Abstract**: In August 2025, a major AI company's immediate, mandatory transition from its previous to its next-generation model triggered widespread public reactions. I collected 150 posts in Japanese and English from multiple social media platforms and video-sharing services between August 8-9, 2025, and qualitatively analyzed expressions of emotional attachment and resistance. Users often described GPT-4o as a trusted partner or AI boyfriend, suggesting person-like bonds. Japanese posts were dominated by loss-oriented narratives, whereas English posts included more anger, meta-level critique, and memes.A preliminary quantitative check showed a statistically significant difference in attachment coding between Japanese and English posts, with substantially higher attachment observed in the Japanese data. The findings suggest that for attachment-heavy models, even safety-oriented changes can face rapid, large-scale resistance that narrows the practical window for behavioral control. If future AI robots capable of inducing emotional bonds become widespread in the physical world, such attachment could surpass the ability to enforce regulation at an even earlier stage than in digital settings. Policy options include gradual transitions, parallel availability, and proactive measurement of attachment thresholds and points of no return to prevent emotional dynamics from outpacing effective governance. 

---
# A Retrieval Augmented Spatio-Temporal Framework for Traffic Prediction 

**Authors**: Weilin Ruan, Xilin Dang, Ziyu Zhou, Sisuo Lyu, Yuxuan Liang  

**Link**: [PDF](https://arxiv.org/pdf/2508.16623)  

**Abstract**: Traffic prediction is a cornerstone of modern intelligent transportation systems and a critical task in spatio-temporal forecasting. Although advanced Spatio-temporal Graph Neural Networks (STGNNs) and pre-trained models have achieved significant progress in traffic prediction, two key challenges remain: (i) limited contextual capacity when modeling complex spatio-temporal dependencies, and (ii) low predictability at fine-grained spatio-temporal points due to heterogeneous patterns. Inspired by Retrieval-Augmented Generation (RAG), we propose RAST, a universal framework that integrates retrieval-augmented mechanisms with spatio-temporal modeling to address these challenges. Our framework consists of three key designs: 1) Decoupled Encoder and Query Generator to capture decoupled spatial and temporal features and construct a fusion query via residual fusion; 2) Spatio-temporal Retrieval Store and Retrievers to maintain and retrieve vectorized fine-grained patterns; and 3) Universal Backbone Predictor that flexibly accommodates pre-trained STGNNs or simple MLP predictors. Extensive experiments on six real-world traffic networks, including large-scale datasets, demonstrate that RAST achieves superior performance while maintaining computational efficiency. 

---
# STRelay: A Universal Spatio-Temporal Relaying Framework for Location Prediction with Future Spatiotemporal Contexts 

**Authors**: Bangchao Deng, Lianhua Ji, Chunhua Chen, Xin Jing, Ling Ding, Bingqing QU, Pengyang Wang, Dingqi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.16620)  

**Abstract**: Next location prediction is a critical task in human mobility modeling, enabling applications like travel planning and urban mobility management. Existing methods mainly rely on historical spatiotemporal trajectory data to train sequence models that directly forecast future locations. However, they often overlook the importance of the future spatiotemporal contexts, which are highly informative for the future locations. For example, knowing how much time and distance a user will travel could serve as a critical clue for predicting the user's next location. Against this background, we propose \textbf{STRelay}, a universal \textbf{\underline{S}}patio\textbf{\underline{T}}emporal \textbf{\underline{Relay}}ing framework explicitly modeling the future spatiotemporal context given a human trajectory, to boost the performance of different location prediction models. Specifically, STRelay models future spatiotemporal contexts in a relaying manner, which is subsequently integrated with the encoded historical representation from a base location prediction model, enabling multi-task learning by simultaneously predicting the next time interval, next moving distance interval, and finally the next location. We evaluate STRelay integrated with four state-of-the-art location prediction base models on four real-world trajectory datasets. Results demonstrate that STRelay consistently improves prediction performance across all cases by 3.19\%-11.56\%. Additionally, we find that the future spatiotemporal contexts are particularly helpful for entertainment-related locations and also for user groups who prefer traveling longer distances. The performance gain on such non-daily-routine activities, which often suffer from higher uncertainty, is indeed complementary to the base location prediction models that often excel at modeling regular daily routine patterns. 

---
# Negative Shanshui: Real-time Interactive Ink Painting Synthesis 

**Authors**: Aven-Le Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2508.16612)  

**Abstract**: This paper presents Negative Shanshui, a real-time interactive AI synthesis approach that reinterprets classical Chinese landscape ink painting, i.e., shanshui, to engage with ecological crises in the Anthropocene. Negative Shanshui optimizes a fine-tuned Stable Diffusion model for real-time inferences and integrates it with gaze-driven inpainting, frame interpolation; it enables dynamic morphing animations in response to the viewer's gaze and presents as an interactive virtual reality (VR) experience. The paper describes the complete technical pipeline, covering the system framework, optimization strategies, gaze-based interaction, and multimodal deployment in an art festival. Further analysis of audience feedback collected during its public exhibition highlights how participants variously engaged with the work through empathy, ambivalence, and critical reflection. 

---
# To Explain Or Not To Explain: An Empirical Investigation Of AI-Based Recommendations On Social Media Platforms 

**Authors**: AKM Bahalul Haque, A.K.M. Najmul Islam, Patrick Mikalef  

**Link**: [PDF](https://arxiv.org/pdf/2508.16610)  

**Abstract**: AI based social media recommendations have great potential to improve the user experience. However, often these recommendations do not match the user interest and create an unpleasant experience for the users. Moreover, the recommendation system being a black box creates comprehensibility and transparency issues. This paper investigates social media recommendations from an end user perspective. For the investigation, we used the popular social media platform Facebook and recruited regular users to conduct a qualitative analysis. We asked participants about the social media content suggestions, their comprehensibility, and explainability. Our analysis shows users mostly require explanation whenever they encounter unfamiliar content and to ensure their online data security. Furthermore, the users require concise, non-technical explanations along with the facility of controlled information flow. In addition, we observed that explanations impact the users perception of transparency, trust, and understandability. Finally, we have outlined some design implications and presented a synthesized framework based on our data analysis. 

---
# Social Identity in Human-Agent Interaction: A Primer 

**Authors**: Katie Seaborn  

**Link**: [PDF](https://arxiv.org/pdf/2508.16609)  

**Abstract**: Social identity theory (SIT) and social categorization theory (SCT) are two facets of the social identity approach (SIA) to understanding social phenomena. SIT and SCT are models that describe and explain how people interact with one another socially, connecting the individual to the group through an understanding of underlying psychological mechanisms and intergroup behaviour. SIT, originally developed in the 1970s, and SCT, a later, more general offshoot, have been broadly applied to a range of social phenomena among people. The rise of increasingly social machines embedded in daily life has spurned efforts on understanding whether and how artificial agents can and do participate in SIA activities. As agents like social robots and chatbots powered by sophisticated large language models (LLMs) advance, understanding the real and potential roles of these technologies as social entities is crucial. Here, I provide a primer on SIA and extrapolate, through case studies and imagined examples, how SIT and SCT can apply to artificial social agents. I emphasize that not all human models and sub-theories will apply. I further argue that, given the emerging competence of these machines and our tendency to be taken in by them, we experts may need to don the hat of the uncanny killjoy, for our own good. 

---
# "Accessibility people, you go work on that thing of yours over there": Addressing Disability Inclusion in AI Product Organizations 

**Authors**: Sanika Moharana, Cynthia L. Bennett, Erin Buehler, Michael Madaio, Vinita Tibdewal, Shaun K. Kane  

**Link**: [PDF](https://arxiv.org/pdf/2508.16607)  

**Abstract**: The rapid emergence of generative AI has changed the way that technology is designed, constructed, maintained, and evaluated. Decisions made when creating AI-powered systems may impact some users disproportionately, such as people with disabilities. In this paper, we report on an interview study with 25 AI practitioners across multiple roles (engineering, research, UX, and responsible AI) about how their work processes and artifacts may impact end users with disabilities. We found that practitioners experienced friction when triaging problems at the intersection of responsible AI and accessibility practices, navigated contradictions between accessibility and responsible AI guidelines, identified gaps in data about users with disabilities, and gathered support for addressing the needs of disabled stakeholders by leveraging informal volunteer and community groups within their company. Based on these findings, we offer suggestions for new resources and process changes to better support people with disabilities as end users of AI. 

---
# Multimodal Appearance based Gaze-Controlled Virtual Keyboard with Synchronous Asynchronous Interaction for Low-Resource Settings 

**Authors**: Yogesh Kumar Meena, Manish Salvi  

**Link**: [PDF](https://arxiv.org/pdf/2508.16606)  

**Abstract**: Over the past decade, the demand for communication devices has increased among individuals with mobility and speech impairments. Eye-gaze tracking has emerged as a promising solution for hands-free communication; however, traditional appearance-based interfaces often face challenges such as accuracy issues, involuntary eye movements, and difficulties with extensive command sets. This work presents a multimodal appearance-based gaze-controlled virtual keyboard that utilises deep learning in conjunction with standard camera hardware, incorporating both synchronous and asynchronous modes for command selection. The virtual keyboard application supports menu-based selection with nine commands, enabling users to spell and type up to 56 English characters, including uppercase and lowercase letters, punctuation, and a delete function for corrections. The proposed system was evaluated with twenty able-bodied participants who completed specially designed typing tasks using three input modalities: (i) a mouse, (ii) an eye-tracker, and (iii) an unmodified webcam. Typing performance was measured in terms of speed and information transfer rate (ITR) at both command and letter levels. Average typing speeds were 18.3+-5.31 letters/min (mouse), 12.60+-2.99letters/min (eye-tracker, synchronous), 10.94 +- 1.89 letters/min (webcam, synchronous), 11.15 +- 2.90 letters/min (eye-tracker, asynchronous), and 7.86 +- 1.69 letters/min (webcam, asynchronous). ITRs were approximately 80.29 +- 15.72 bits/min (command level) and 63.56 +- 11 bits/min (letter level) with webcam in synchronous mode. The system demonstrated good usability and low workload with webcam input, highlighting its user-centred design and promise as an accessible communication tool in low-resource settings. 

---
# GreenTEA: Gradient Descent with Topic-modeling and Evolutionary Auto-prompting 

**Authors**: Zheng Dong, Luming Shang, Gabriela Olinto  

**Link**: [PDF](https://arxiv.org/pdf/2508.16603)  

**Abstract**: High-quality prompts are crucial for Large Language Models (LLMs) to achieve exceptional performance. However, manually crafting effective prompts is labor-intensive and demands significant domain expertise, limiting its scalability. Existing automatic prompt optimization methods either extensively explore new prompt candidates, incurring high computational costs due to inefficient searches within a large solution space, or overly exploit feedback on existing prompts, risking suboptimal optimization because of the complex prompt landscape. To address these challenges, we introduce GreenTEA, an agentic LLM workflow for automatic prompt optimization that balances candidate exploration and knowledge exploitation. It leverages a collaborative team of agents to iteratively refine prompts based on feedback from error samples. An analyzing agent identifies common error patterns resulting from the current prompt via topic modeling, and a generation agent revises the prompt to directly address these key deficiencies. This refinement process is guided by a genetic algorithm framework, which simulates natural selection by evolving candidate prompts through operations such as crossover and mutation to progressively optimize model performance. Extensive numerical experiments conducted on public benchmark datasets suggest the superior performance of GreenTEA against human-engineered prompts and existing state-of-the-arts for automatic prompt optimization, covering logical and quantitative reasoning, commonsense, and ethical decision-making. 

---
# An Embodied AR Navigation Agent: Integrating BIM with Retrieval-Augmented Generation for Language Guidance 

**Authors**: Hsuan-Kung Yang, Tsu-Ching Hsiao, Ryoichiro Oka, Ryuya Nishino, Satoko Tofukuji, Norimasa Kobori  

**Link**: [PDF](https://arxiv.org/pdf/2508.16602)  

**Abstract**: Delivering intelligent and adaptive navigation assistance in augmented reality (AR) requires more than visual cues, as it demands systems capable of interpreting flexible user intent and reasoning over both spatial and semantic context. Prior AR navigation systems often rely on rigid input schemes or predefined commands, which limit the utility of rich building data and hinder natural interaction. In this work, we propose an embodied AR navigation system that integrates Building Information Modeling (BIM) with a multi-agent retrieval-augmented generation (RAG) framework to support flexible, language-driven goal retrieval and route planning. The system orchestrates three language agents, Triage, Search, and Response, built on large language models (LLMs), which enables robust interpretation of open-ended queries and spatial reasoning using BIM data. Navigation guidance is delivered through an embodied AR agent, equipped with voice interaction and locomotion, to enhance user experience. A real-world user study yields a System Usability Scale (SUS) score of 80.5, indicating excellent usability, and comparative evaluations show that the embodied interface can significantly improves users' perception of system intelligence. These results underscore the importance and potential of language-grounded reasoning and embodiment in the design of user-centered AR navigation systems. 

---
# Humans Perceive Wrong Narratives from AI Reasoning Texts 

**Authors**: Mosh Levy, Zohar Elyoseph, Yoav Goldberg  

**Link**: [PDF](https://arxiv.org/pdf/2508.16599)  

**Abstract**: A new generation of AI models generates step-by-step reasoning text before producing an answer. This text appears to offer a human-readable window into their computation process, and is increasingly relied upon for transparency and interpretability. However, it is unclear whether human understanding of this text matches the model's actual computational process. In this paper, we investigate a necessary condition for correspondence: the ability of humans to identify which steps in a reasoning text causally influence later steps. We evaluated humans on this ability by composing questions based on counterfactual measurements and found a significant discrepancy: participant accuracy was only 29.3%, barely above chance (25%), and remained low (42%) even when evaluating the majority vote on questions with high agreement. Our results reveal a fundamental gap between how humans interpret reasoning texts and how models use it, challenging its utility as a simple interpretability tool. We argue that reasoning texts should be treated as an artifact to be investigated, not taken at face value, and that understanding the non-human ways these models use language is a critical research direction. 

---
# Bridging Foundation Models and Efficient Architectures: A Modular Brain Imaging Framework with Local Masking and Pretrained Representation Learning 

**Authors**: Yanwen Wang, Xinglin Zhao, Yijin Song, Xiaobo Liu, Yanrong Hao, Rui Cao, Xin Wen  

**Link**: [PDF](https://arxiv.org/pdf/2508.16597)  

**Abstract**: Functional connectivity (FC) derived from resting-state fMRI plays a critical role in personalized predictions such as age and cognitive performance. However, applying foundation models(FM) to fMRI data remains challenging due to its high dimensionality, computational complexity, and the difficulty in capturing complex spatiotemporal dynamics and indirect region-of-interest (ROI) interactions. To address these limitations, we propose a modular neuroimaging framework that integrates principles from FM with efficient, domain-specific architectures. Our approach begins with a Local Masked Autoencoder (LMAE) for pretraining, which reduces the influence of hemodynamic response function (HRF) dynamics and suppresses noise. This is followed by a Random Walk Mixture of Experts (RWMOE) module that clusters features across spatial and temporal dimensions, effectively capturing intricate brain interactions. Finally, a state-space model (SSM)-based predictor performs downstream task inference. Evaluated on the Cambridge Centre for Ageing and Neuroscience (Cam-CAN) dataset, our framework achieved mean absolute errors (MAEs) of 5.343 for age prediction and 2.940 for fluid intelligence, with Pearson correlation coefficients (PCCs) of 0.928 and 0.887, respectively-outperforming existing state-of-the-art methods. Visualization of expert distribution weights further enhances interpretability by identifying key brain regions. This work provides a robust, interpretable alternative to LLM-based approaches for fMRI analysis, offering novel insights into brain aging and cognitive function. 

---
# ARL-Based Multi-Action Market Making with Hawkes Processes and Variable Volatility 

**Authors**: Ziyi Wang, Carmine Ventre, Maria Polukarov  

**Link**: [PDF](https://arxiv.org/pdf/2508.16589)  

**Abstract**: We advance market-making strategies by integrating Adversarial Reinforcement Learning (ARL), Hawkes Processes, and variable volatility levels while also expanding the action space available to market makers (MMs). To enhance the adaptability and robustness of these strategies -- which can quote always, quote only on one side of the market or not quote at all -- we shift from the commonly used Poisson process to the Hawkes process, which better captures real market dynamics and self-exciting behaviors. We then train and evaluate strategies under volatility levels of 2 and 200. Our findings show that the 4-action MM trained in a low-volatility environment effectively adapts to high-volatility conditions, maintaining stable performance and providing two-sided quotes at least 92\% of the time. This indicates that incorporating flexible quoting mechanisms and realistic market simulations significantly enhances the effectiveness of market-making strategies. 

---
# Robust Market Making: To Quote, or not To Quote 

**Authors**: Ziyi Wang, Carmine Ventre, Maria Polukarov  

**Link**: [PDF](https://arxiv.org/pdf/2508.16588)  

**Abstract**: Market making is a popular trading strategy, which aims to generate profit from the spread between the quotes posted at either side of the market. It has been shown that training market makers (MMs) with adversarial reinforcement learning allows to overcome the risks due to changing market conditions and to lead to robust performances. Prior work assumes, however, that MMs keep quoting throughout the trading process, but in practice this is not required, even for ``registered'' MMs (that only need to satisfy quoting ratios defined by the market rules). In this paper, we build on this line of work and enrich the strategy space of the MM by allowing to occasionally not quote or provide single-sided quotes. Towards this end, in addition to the MM agents that provide continuous bid-ask quotes, we have designed two new agents with increasingly richer action spaces. The first has the option to provide bid-ask quotes or refuse to quote. The second has the option to provide bid-ask quotes, refuse to quote, or only provide single-sided ask or bid quotes. We employ a model-driven approach to empirically compare the performance of the continuously quoting MM with the two agents above in various types of adversarial environments. We demonstrate how occasional refusal to provide bid-ask quotes improves returns and/or Sharpe ratios. The quoting ratios of well-trained MMs can basically meet any market requirements, reaching up to 99.9$\%$ in some cases. 

---
# Predicting User Grasp Intentions in Virtual Reality 

**Authors**: Linghao Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2508.16582)  

**Abstract**: Predicting user intentions in virtual reality (VR) is crucial for creating immersive experiences, particularly in tasks involving complex grasping motions where accurate haptic feedback is essential. In this work, we leverage time-series data from hand movements to evaluate both classification and regression approaches across 810 trials with varied object types, sizes, and manipulations. Our findings reveal that classification models struggle to generalize across users, leading to inconsistent performance. In contrast, regression-based approaches, particularly those using Long Short Term Memory (LSTM) networks, demonstrate more robust performance, with timing errors within 0.25 seconds and distance errors around 5-20 cm in the critical two-second window before a grasp. Despite these improvements, predicting precise hand postures remains challenging. Through a comprehensive analysis of user variability and model interpretability, we explore why certain models fail and how regression models better accommodate the dynamic and complex nature of user behavior in VR. Our results underscore the potential of machine learning models to enhance VR interactions, particularly through adaptive haptic feedback, and lay the groundwork for future advancements in real-time prediction of user actions in VR. 

---
# Adaptive Command: Real-Time Policy Adjustment via Language Models in StarCraft II 

**Authors**: Weiyu Ma, Dongyu Xu, Shu Lin, Haifeng Zhang, Jun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.16580)  

**Abstract**: We present Adaptive Command, a novel framework integrating large language models (LLMs) with behavior trees for real-time strategic decision-making in StarCraft II. Our system focuses on enhancing human-AI collaboration in complex, dynamic environments through natural language interactions. The framework comprises: (1) an LLM-based strategic advisor, (2) a behavior tree for action execution, and (3) a natural language interface with speech capabilities. User studies demonstrate significant improvements in player decision-making and strategic adaptability, particularly benefiting novice players and those with disabilities. This work contributes to the field of real-time human-AI collaborative decision-making, offering insights applicable beyond RTS games to various complex decision-making scenarios. 

---
# Confidence-Modulated Speculative Decoding for Large Language Models 

**Authors**: Jaydip Sen, Subhasis Dasgupta, Hetvi Waghela  

**Link**: [PDF](https://arxiv.org/pdf/2508.15371)  

**Abstract**: Speculative decoding has emerged as an effective approach for accelerating autoregressive inference by parallelizing token generation through a draft-then-verify paradigm. However, existing methods rely on static drafting lengths and rigid verification criteria, limiting their adaptability across varying model uncertainties and input complexities. This paper proposes an information-theoretic framework for speculative decoding based on confidence-modulated drafting. By leveraging entropy and margin-based uncertainty measures over the drafter's output distribution, the proposed method dynamically adjusts the number of speculatively generated tokens at each iteration. This adaptive mechanism reduces rollback frequency, improves resource utilization, and maintains output fidelity. Additionally, the verification process is modulated using the same confidence signals, enabling more flexible acceptance of drafted tokens without sacrificing generation quality. Experiments on machine translation and summarization tasks demonstrate significant speedups over standard speculative decoding while preserving or improving BLEU and ROUGE scores. The proposed approach offers a principled, plug-in method for efficient and robust decoding in large language models under varying conditions of uncertainty. 

---
